//! `.ttpf`: the title's texture page files (`data/ttdb/**`, `pack/*/data/ttdb/**`).
//!
//! Layout after five-star-recomp's format notes
//! (commit a19995d, GPL-3.0-or-later), written here from that description, no
//! code taken; the page entry as one u64, the stored size and the sentinel
//! from how the title builds its requests.
//!
//! ```text
//! 0x00  "RDS5"
//! 0x04  u32 n        page entries, the last a zero sentinel
//! 0x08  u32          address of the last page (mod 2^32)
//! 0x10  u32          ETPF offset / 0x100
//! 0x18  u64[n]       page entries, sorted by address and by file offset
//!         bits  0-23 file offset / 0x100
//!         bits 26-29 storage: bit 3 zlib, bits 0-2 BCPack mode (BC number)
//!         bits 30-51 address / 0x1000
//!         bits 52-63 zero padding after the stored bytes
//! ETPF  "ETPF", 634, count, section size, then 16-byte descriptors, then
//!       count u64 name hashes (sorted; hash i names descriptor i)
//!   d0  bits 0-13 width - 1, 14-27 height - 1, 28-31 mip field
//!   d1  bits 0-2 added to the mip field (= stored mips), 3-9 DXGI format,
//!       10-19 layers - 1 (array size, 6 on a cube, depth on a volume),
//!       20-23 kind (2 2D, 4 cube, 6 volume, 10 2D array), 24-31 layout mode
//!       (bits 0-2: 0 = 64 KiB tiles, 1-3 = 4 KiB tiles; bit 3 cube copies)
//!   o0  bits 0-20 address / 0x1000;  o1 unknown
//! ```
//!
//! A page's virtual size runs to the next entry's address, its stored bytes
//! to the next entry's file offset less the padding. A texture spans from its
//! address to the next texture's. A slice holds one mip chain smallest
//! first (the mip tail block, then the mips up to mip 0), as AMD's addrlib
//! places it: [`crate::texture::Layout`] with `SW_64KB_D` or `SW_4KB_S`.

use std::io::{Read, Seek, SeekFrom};

use crate::bcpack::{self, Tables};
use crate::texture::{self, Desc, Dimension, Layout, Swizzle};

pub type Res<T> = Result<T, String>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Page {
    /// Virtual address and size (the decoded size).
    pub addr: u64,
    pub size: u64,
    /// File offset and the exact stored byte count.
    pub offset: u64,
    pub stored: u64,
    /// Bits 26-29 of the entry: bit 3 zlib, bits 0-2 BCPack mode.
    pub storage: u8,
}

impl Page {
    pub fn zlib(&self) -> bool {
        self.storage & 8 != 0
    }

    /// The BCPack mode (BC number 1, 3-7), 0 for none.
    pub fn bcpack(&self) -> u8 {
        self.storage & 7
    }

    /// The DirectStorage option word the title puts in the request
    /// (its request builder): zlib in bit 0, BCPack mode x 2 in bits 1-3 when
    /// zlib is clear; zlib with a BCPack mode takes 0x20/0x60/0x40/0x80 for
    /// BC1/BC3/BC4/BC5 (other modes: none) in bits 4-7.
    pub fn request_options(&self) -> u64 {
        let s = u64::from(self.storage);
        let base = (s >> 3) | if s < 8 { 2 * s } else { 0 };
        base | match s {
            9 => 0x20,
            11 => 0x60,
            12 => 0x40,
            13 => 0x80,
            _ => 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Texture {
    pub hash: u64,
    pub width: u32,
    pub height: u32,
    /// Mip field + d1 bits 0-2: the mips stored (cube maps: + copies - 1).
    pub mips: u32,
    pub format: u32,
    pub layers: u32,
    pub kind: u8,
    pub mode: u8,
    pub addr: u64,
    pub span: u64,
    pub words: [u32; 4],
}

pub const KIND_2D: u8 = 2;
pub const KIND_CUBE: u8 = 4;
pub const KIND_VOLUME: u8 = 6;
pub const KIND_ARRAY: u8 = 10;

#[derive(Debug, Clone)]
pub struct Ttpf {
    /// Sorted by address.
    pub pages: Vec<Page>,
    /// In descriptor (hash) order.
    pub textures: Vec<Texture>,
    pub header_bytes: u64,
}

fn u32_at(b: &[u8], at: usize) -> Res<u32> {
    b.get(at..at + 4).map(|s| u32::from_le_bytes(s.try_into().expect("4 bytes"))).ok_or_else(|| "truncated".into())
}

fn read_at(f: &mut (impl Read + Seek), at: u64, len: usize) -> Res<Vec<u8>> {
    let mut v = vec![0; len];
    f.seek(SeekFrom::Start(at)).map_err(|e| e.to_string())?;
    f.read_exact(&mut v).map_err(|e| e.to_string())?;
    Ok(v)
}

impl Ttpf {
    /// Reads the page table and the ETPF section of an open file.
    pub fn read(f: &mut (impl Read + Seek), file_len: u64) -> Res<Ttpf> {
        let head = read_at(f, 0, 0x18)?;
        if &head[..4] != b"RDS5" {
            return Err("not RDS5".into());
        }
        let n = u32_at(&head, 4)? as usize;
        if n < 2 || 0x18 + 8 * n as u64 > file_len {
            return Err(format!("bad page count {n}"));
        }
        let table = read_at(f, 0x18, 8 * n)?;
        let entry = |i: usize| u64::from_le_bytes(table[8 * i..8 * i + 8].try_into().expect("8 bytes"));
        let mut pages = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            let (e, next) = (entry(i), entry(i + 1));
            let offset = (e & 0xff_ffff) << 8;
            let addr = ((e >> 30) & 0x3f_ffff) << 12;
            let pad = e >> 52;
            // The sentinel is zero in the file (the title fills it at load);
            // the last page then runs to the end of the file and takes 64 KiB
            // at most of address space (see `Ttpf::fix_last`).
            let (next_offset, next_addr) = if i + 1 == n - 1 {
                (file_len, addr + 0x1_0000)
            } else {
                ((next & 0xff_ffff) << 8, ((next >> 30) & 0x3f_ffff) << 12)
            };
            if next_addr < addr || next_offset < offset {
                return Err(format!("page {i}: table not sorted"));
            }
            pages.push(Page {
                addr,
                size: next_addr - addr,
                offset,
                stored: (next_offset - offset).saturating_sub(pad),
                storage: ((e >> 26) & 0xf) as u8,
            });
        }
        let etpf = u64::from(u32_at(&head, 0x10)?) << 8;
        let sect = read_at(f, etpf, 16)?;
        if &sect[..4] != b"ETPF" {
            return Err(format!("no ETPF at {etpf:#x}"));
        }
        let count = u32_at(&sect, 8)? as usize;
        let body = read_at(f, etpf + 16, 24 * count)?;
        let mut textures: Vec<Texture> = (0..count)
            .map(|i| {
                let w: [u32; 4] = std::array::from_fn(|k| u32_at(&body, 16 * i + 4 * k).expect("in bounds"));
                let hash = u64::from_le_bytes(body[16 * count + 8 * i..][..8].try_into().expect("8 bytes"));
                Texture {
                    hash,
                    width: (w[0] & 0x3fff) + 1,
                    height: ((w[0] >> 14) & 0x3fff) + 1,
                    mips: (w[0] >> 28) + (w[1] & 7),
                    format: (w[1] >> 3) & 0x7f,
                    layers: ((w[1] >> 10) & 0x3ff) + 1,
                    kind: ((w[1] >> 20) & 0xf) as u8,
                    mode: (w[1] >> 24) as u8,
                    addr: u64::from(w[2] & 0x1f_ffff) << 12,
                    span: 0,
                    words: w,
                }
            })
            .collect();
        let mut ttpf = Ttpf { pages, textures: vec![], header_bytes: etpf + 16 + 24 * count as u64 };
        ttpf.fix_last(f)?;
        let end = ttpf.pages.last().map_or(0, |p| p.addr + p.size);
        let mut order: Vec<usize> = (0..count).collect();
        order.sort_by_key(|&i| textures[i].addr);
        for k in 0..count {
            let next = order.get(k + 1).map_or(end, |&j| textures[j].addr);
            let t = &mut textures[order[k]];
            t.span = next.saturating_sub(t.addr);
        }
        ttpf.textures = textures;
        Ok(ttpf)
    }

    /// The last page's size is not in the file. A last entry at the end of
    /// the file with no bytes is an end marker (its address ends the
    /// previous page) and is dropped; otherwise a zlib page gives its size by
    /// inflating, any other takes 0x4000 (five-star-recomp's rule).
    fn fix_last(&mut self, f: &mut (impl Read + Seek)) -> Res<()> {
        if self.pages.last().is_some_and(|p| p.stored == 0) {
            self.pages.pop();
            return Ok(());
        }
        let Some(last) = self.pages.last().copied() else { return Ok(()) };
        let size = if last.zlib() {
            let src = read_at(f, last.offset, last.stored as usize)?;
            miniz_oxide::inflate::decompress_to_vec_zlib(&src).map_err(|e| format!("last page: {e:?}"))?.len() as u64
        } else {
            0x4000
        };
        self.pages.last_mut().expect("non-empty").size = size;
        Ok(())
    }

    /// Index of the page holding address `addr`.
    pub fn page_at(&self, addr: u64) -> Option<usize> {
        let i = self.pages.partition_point(|p| p.addr <= addr).checked_sub(1)?;
        (addr < self.pages[i].addr + self.pages[i].size).then_some(i)
    }

    /// Pages that a range of addresses touches.
    pub fn pages_of(&self, addr: u64, len: u64) -> std::ops::Range<usize> {
        let first = self.pages.partition_point(|p| p.addr + p.size <= addr);
        let end = self.pages.partition_point(|p| p.addr < addr + len);
        first..end.max(first)
    }

    /// Decodes one page to its virtual bytes, as a DirectStorage request of
    /// the title does (zlib first, then BCPack).
    pub fn decode_page(f: &mut (impl Read + Seek), page: &Page, tables: Option<&Tables>) -> Res<Vec<u8>> {
        let src = read_at(f, page.offset, page.stored as usize)?;
        let size = page.size as usize;
        let mid = if page.zlib() {
            let mut v = miniz_oxide::inflate::decompress_to_vec_zlib(&src).map_err(|e| format!("zlib: {e:?}"))?;
            if page.bcpack() == 0 {
                v.resize(size, 0);
            }
            v
        } else {
            src
        };
        match page.bcpack() {
            0 => {
                let mut v = mid;
                v.resize(size, 0);
                Ok(v)
            }
            bc => {
                let tables = tables.ok_or("BCPack page and no tables")?;
                let mut out = vec![0; size];
                bcpack::decode(bc, &mid, &mut out, tables).map_err(|e| format!("BCPack BC{bc}: {}", e.0))?;
                Ok(out)
            }
        }
    }

    /// The decoded bytes of `[addr, addr + len)`.
    pub fn read_span(&self, f: &mut (impl Read + Seek), addr: u64, len: u64, tables: Option<&Tables>) -> Res<Vec<u8>> {
        let mut out = vec![0; len as usize];
        for i in self.pages_of(addr, len) {
            let p = &self.pages[i];
            let bytes = Self::decode_page(f, p, tables)?;
            let from = p.addr.max(addr);
            let to = (p.addr + p.size).min(addr + len);
            out[(from - addr) as usize..(to - addr) as usize]
                .copy_from_slice(&bytes[(from - p.addr) as usize..(to - p.addr) as usize]);
        }
        Ok(out)
    }
}

/// The swizzle a layout mode selects (five-star-recomp's bit patterns are
/// addrlib's 64 KiB `D` and 4 KiB `S` ones; checked in the census).
pub fn swizzle(mode: u8) -> Swizzle {
    if mode & 7 == 0 { Swizzle::SW_64KB_D } else { Swizzle::SW_4KB_S }
}

/// One slice's mip chain of `t` (`mips` levels); volumes as one 3D chain.
pub fn chain(t: &Texture, mips: u32) -> Res<Layout> {
    let f = texture::format(t.format).ok_or_else(|| format!("format {} has no layout", t.format))?;
    let volume = t.kind == KIND_VOLUME;
    Layout::new(&Desc {
        dimension: if volume { Dimension::Tex3d } else { Dimension::Tex2d },
        width: t.width,
        height: t.height,
        depth: if volume { t.layers } else { 1 },
        mips: mips.max(1),
        element_bytes: f.bytes,
        block: f.block,
        samples: 1,
        swizzle: if volume { Swizzle::SW_4KB_S } else { swizzle(t.mode) },
    })
    .map_err(|e| format!("layout: {e:?}"))
}

/// A chain's storage units, smallest first: (offset, size) of the tail
/// block (if any), then of each mip below it up to mip 0.
pub fn units(l: &Layout) -> Vec<(u64, u64)> {
    let mut v = vec![];
    let tail = l.first_tail_mip as usize;
    if tail < l.mips.len() {
        v.push((0, l.alignment));
    }
    for m in (0..tail).rev() {
        let end = if m == 0 { l.slice_size } else { l.mips[m - 1].offset };
        v.push((l.mips[m].offset, end - l.mips[m].offset));
    }
    v
}

/// Where a texture's data lies in its span.
#[derive(Debug, Clone)]
pub struct Placement {
    pub chain: Layout,
    /// Copies of the chain (array slices, cube faces) and the bytes the
    /// layout gives; equal to the span when the layout is understood.
    pub slices: u32,
    pub size: u64,
}

/// The layout of `t` and the size it predicts, by kind: 2D one chain, 2D
/// array the units interleaved over the slices, cube the copies of
/// five-star-recomp's rule, volume one 3D chain.
pub fn place(t: &Texture) -> Res<Placement> {
    match t.kind {
        KIND_2D | KIND_ARRAY | KIND_VOLUME => {
            let chain = chain(t, t.mips)?;
            let slices = if t.kind == KIND_ARRAY { t.layers } else { 1 };
            // A volume's chain covers every depth slab.
            let size = if t.kind == KIND_VOLUME { chain.size } else { chain.slice_size * u64::from(slices) };
            Ok(Placement { chain, slices, size })
        }
        KIND_CUBE => {
            let copies = if t.mode & 8 != 0 { u32::from(t.mode & 7).max(1) } else { 1 };
            let mips = t.mips + 1 - copies;
            let chain = chain(t, mips)?;
            let u = units(&chain);
            let mut size = 0;
            for k in 0..copies as usize {
                let take = (u.len() + 1 + k).saturating_sub(copies as usize).min(u.len());
                let one: u64 = u[..take].iter().map(|x| x.1).sum::<u64>() * 6;
                size += if one > 0x2_0000 { one.next_multiple_of(0x1_0000) } else { one };
            }
            Ok(Placement { chain, slices: 6, size })
        }
        k => Err(format!("kind {k}")),
    }
}

/// Gathers slice `s` of a 2D array span (units interleaved over `slices`)
/// into one plain chain.
pub fn gather_slice(chain: &Layout, slices: u32, span: &[u8], s: u32) -> Vec<u8> {
    let mut out = vec![0; chain.slice_size as usize];
    for (off, size) in units(chain) {
        let from = (off * u64::from(slices) + u64::from(s) * size) as usize;
        out[off as usize..(off + size) as usize].copy_from_slice(&span[from..from + size as usize]);
    }
    out
}

/// Bytes of mip `m` (all slices given) in a linear, unpadded host layout.
pub fn linear_mip_bytes(t: &Texture, m: u32) -> u64 {
    let Some(f) = texture::format(t.format) else { return 0 };
    let w = u64::from((t.width >> m).max(1).div_ceil(f.block));
    let h = u64::from((t.height >> m).max(1).div_ceil(f.block));
    let d = if t.kind == KIND_VOLUME { u64::from((t.layers >> m).max(1)) } else { u64::from(t.layers) };
    w * h * d * u64::from(f.bytes)
}

/// A DDS file (DX10 header) of linear subresources: `mips` levels per slice,
/// slices one after the other.
pub fn dds(t: &Texture, mips: u32, slices: u32, data: &[u8]) -> Vec<u8> {
    let f = texture::format(t.format).expect("format with a layout");
    let pitch = linear_mip_bytes(&Texture { layers: 1, ..*t }, 0) as u32;
    let mut h = Vec::with_capacity(148 + data.len());
    let mut put = |v: u32| h.extend_from_slice(&v.to_le_bytes());
    put(0x2053_4444); // "DDS "
    put(124);
    put(0x1 | 0x2 | 0x4 | 0x1000 | 0x2_0000 | 0x8_0000); // caps, height, width, pixelformat, mipcount, linearsize
    put(t.height);
    put(t.width);
    put(if f.block == 4 { pitch } else { t.width * f.bytes });
    put(if t.kind == KIND_VOLUME { t.layers } else { 0 });
    put(mips);
    for _ in 0..11 {
        put(0);
    }
    put(32);
    put(0x4); // DDPF_FOURCC
    put(0x3031_5844); // "DX10"
    for _ in 0..5 {
        put(0);
    }
    put(0x1000 | if mips > 1 { 0x40_0008 } else { 0 });
    for _ in 0..4 {
        put(0);
    }
    put(t.format);
    put(if t.kind == KIND_VOLUME { 4 } else { 3 });
    put(if t.kind == KIND_CUBE { 4 } else { 0 });
    put(if t.kind == KIND_CUBE { slices / 6 } else { slices });
    put(0);
    h.extend_from_slice(data);
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(entries: &[u64], etpf: &[u8]) -> Vec<u8> {
        let mut v = b"RDS5".to_vec();
        v.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        v.extend_from_slice(&[0; 8]);
        v.extend_from_slice(&1u32.to_le_bytes()); // ETPF at 0x100
        v.extend_from_slice(&[0; 4]);
        for e in entries {
            v.extend_from_slice(&e.to_le_bytes());
        }
        v.resize(0x100, 0);
        v.extend_from_slice(etpf);
        v
    }

    #[test]
    fn page_entry_fields() {
        // Page at file 0x1000, address 0x5000, zlib (storage 8), 0x10 padding; next at 0x1200 / 0x7000.
        let e0 = 0x10u64 | (8 << 26) | (5 << 30) | (0x10 << 52);
        let e1 = 0x12u64 | (7 << 30);
        let mut etpf = b"ETPF".to_vec();
        for w in [634u32, 1, 40, (0xff) | (0xff << 14) | (2 << 28), (71 << 3) | 1 | (2 << 20), 5, 0] {
            etpf.extend_from_slice(&w.to_le_bytes());
        }
        etpf.extend_from_slice(&0x1234u64.to_le_bytes());
        let mut data = file(&[e0, e1, 0], &etpf);
        data.resize(0x1300, 0);
        let mut c = std::io::Cursor::new(&data);
        // The last page is stored (not zlib), so no inflate is attempted.
        let t = Ttpf::read(&mut c, data.len() as u64).unwrap();
        assert_eq!(t.pages[0], Page { addr: 0x5000, size: 0x2000, offset: 0x1000, stored: 0x1f0, storage: 8 });
        assert_eq!(t.pages[0].request_options(), 1);
        assert_eq!(t.pages[1].size, 0x4000);
        let x = &t.textures[0];
        assert_eq!((x.width, x.height, x.mips, x.format, x.kind, x.addr, x.hash), (256, 256, 3, 71, 2, 0x5000, 0x1234));
        assert_eq!(x.span, 0x4000 + 0x2000);
    }

    #[test]
    fn request_options_follow_the_title() {
        let p = |storage| Page { addr: 0, size: 0, offset: 0, stored: 0, storage };
        assert_eq!(p(0).request_options(), 0);
        assert_eq!(p(7).request_options(), 14);
        assert_eq!(p(8).request_options(), 1);
        assert_eq!(p(9).request_options(), 0x21);
        assert_eq!(p(13).request_options(), 0x81);
    }
}
