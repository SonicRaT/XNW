//! Texture packs (`.xtp`): the title's `.ttpf` textures converted offline to
//! the host's layout, so a streamed texture costs one positional read into an
//! upload buffer: no BCPack decode, no detile, no hashing.
//!
//! One pack per source `.ttpf`, named after its key (`data_ttdb_ui_ttdb_ui.xtp`).
//! A texture is found by (file id, texture index): the file id is
//! [`file_id`] of the path the title opens (`G:\.\data\ttdb\ui\ttdb_ui.ttpf`)
//! or of the host copy (`game\data\ttdb\ui\ttdb_ui.ttpf`), the same for
//! both; the texture index is the descriptor's index in `Ttpf::textures`
//! (= `origin::Origin::texture`).
//!
//! ```text
//! 0x000  "XTPK", u32 format version (1), u32 converter version, u32 mips dropped
//! 0x010  u64 file id, u64 source size, u64 source mtime (ns since 1970)
//! 0x028  u32 texture count, u32 record size (416), u64 index offset (0x1000), u64 data start
//! 0x040  u16 key length, key (UTF-8, "data/ttdb/ui/ttdb_ui.ttpf")
//! index  one record per source texture, in descriptor order (absent: flags 0)
//!   0  u32 flags (1 present), u32 DXGI format, u32 width, u32 height (mip 0, before any drop)
//!  16  u32 layers (array size; 6 on a cube; depth on a volume), u8 kind (ttpf::KIND_*),
//!      u8 mips (the source chain), u8 first stored mip (= mips dropped), u8 stored mips
//!  24  u64 block bytes (from the coarsest mip's offset to the end of mip `first`)
//!  32  16 x MipLoc {u64 offset, u32 bytes, u32 row pitch, u32 slice pitch, u32 rows},
//!      by source mip number; dropped mips are zero
//! data  per texture one block, stored mips coarsest first, so mips [a, b) are one
//!       contiguous read (`Entry::span`); the block starts 64 KiB aligned (4 KiB
//!       when under 64 KiB), each mip 512-aligned (D3D12 placement alignment),
//!       rows padded to 256 bytes (D3D12 row pitch), array slices and cube faces
//!       512-aligned (each its own subresource), volume depth slices packed.
//! ```
//!
//! Uncompressed in v1. Readers use positional reads (`seek_read`) on handles
//! opened once: no lock, any thread.

use std::collections::HashMap;
use std::io;
use std::ops::Range;
use std::os::windows::fs::FileExt;
use std::path::{Path, PathBuf};

use crate::ttpf::{self, KIND_ARRAY, KIND_CUBE, KIND_VOLUME, Texture, Ttpf};

/// Pack file format version.
pub const FORMAT: u32 = 1;
/// Converter version: raise when the converted bytes would change.
pub const CONVERTER: u32 = 2;
pub const MAX_MIPS: usize = 16;
pub const HEADER: u64 = 0x1000;
pub const RECORD: usize = 32 + 24 * MAX_MIPS;
pub const ROW_PITCH: u64 = 256;
pub const PLACEMENT: u64 = 512;
pub const EXTENSION: &str = "xtp";

/// The stable name of a `.ttpf` path: from `data/ttdb/` on, with `pack/<name>/`
/// in front when the path has it; lowercase, forward slashes, `.` parts
/// dropped. `None` if the path has no `data/ttdb` part.
pub fn file_key(path: &Path) -> Option<String> {
    let text = path.to_string_lossy().to_ascii_lowercase().replace('\\', "/");
    let parts: Vec<&str> = text.split('/').filter(|p| !p.is_empty() && *p != ".").collect();
    let at = parts.windows(2).rposition(|w| w == ["data", "ttdb"])?;
    let from = if at >= 2 && parts[at - 2] == "pack" { at - 2 } else { at };
    Some(parts[from..].join("/"))
}

/// FNV-1a 64 of [`file_key`]: the pack's file id.
pub fn file_id(path: &Path) -> Option<u64> {
    file_key(path).map(|k| fnv1a(k.as_bytes()))
}

pub fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}

/// The pack file name for a key.
pub fn pack_name(key: &str) -> String {
    let stem = key.strip_suffix(".ttpf").unwrap_or(key);
    format!("{}.{EXTENSION}", stem.replace('/', "_"))
}

/// Where one stored mip lies in the pack file.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MipLoc {
    /// Absolute file offset (512-aligned).
    pub offset: u64,
    /// Bytes of every slice of the mip, padding included.
    pub bytes: u32,
    /// Bytes per row of blocks (texels for uncompressed formats), a multiple of 256.
    pub row_pitch: u32,
    /// Bytes per array slice, cube face or volume depth slice.
    pub slice_pitch: u32,
    /// Rows per slice.
    pub rows: u32,
}

/// One texture in a pack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    /// Which pack of the [`PackDir`] holds it.
    pub pack: u32,
    pub texture: u32,
    pub format: u32,
    /// Of mip 0 of the source, also when it is dropped.
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub kind: u8,
    pub mips: u8,
    /// First stored mip (= mips dropped at conversion).
    pub first: u8,
    pub stored: u8,
    pub block: u64,
    /// By source mip number; dropped ones are zero.
    pub mip: [MipLoc; MAX_MIPS],
}

impl Entry {
    /// Source mips stored: `first .. mips`.
    pub fn stored_mips(&self) -> Range<u32> {
        u32::from(self.first)..u32::from(self.first) + u32::from(self.stored)
    }

    pub fn mip(&self, m: u32) -> Option<MipLoc> {
        self.stored_mips().contains(&m).then(|| self.mip[m as usize])
    }

    /// The file range holding source mips `mips` (clipped to the stored ones;
    /// coarsest first, so one contiguous range), or `None` if none is stored.
    pub fn span(&self, mips: Range<u32>) -> Option<(u64, u64)> {
        let s = self.stored_mips();
        let (a, b) = (mips.start.max(s.start), mips.end.min(s.end));
        if a >= b {
            return None;
        }
        let from = self.mip[b as usize - 1].offset;
        let last = self.mip[a as usize];
        Some((from, last.offset + u64::from(last.bytes) - from))
    }

    fn parse(pack: u32, texture: u32, r: &[u8]) -> Option<Entry> {
        let u32_at = |o: usize| u32::from_le_bytes(r[o..o + 4].try_into().expect("4 bytes"));
        if u32_at(0) & 1 == 0 {
            return None;
        }
        let mut mip = [MipLoc::default(); MAX_MIPS];
        for (k, m) in mip.iter_mut().enumerate() {
            let o = 32 + 24 * k;
            *m = MipLoc {
                offset: u64::from_le_bytes(r[o..o + 8].try_into().expect("8 bytes")),
                bytes: u32_at(o + 8),
                row_pitch: u32_at(o + 12),
                slice_pitch: u32_at(o + 16),
                rows: u32_at(o + 20),
            };
        }
        Some(Entry {
            pack,
            texture,
            format: u32_at(4),
            width: u32_at(8),
            height: u32_at(12),
            layers: u32_at(16),
            kind: r[20],
            mips: r[21],
            first: r[22],
            stored: r[23],
            block: u64::from_le_bytes(r[24..32].try_into().expect("8 bytes")),
            mip,
        })
    }

    fn record(&self) -> [u8; RECORD] {
        let mut r = [0u8; RECORD];
        let mut put = |o: usize, b: &[u8]| r[o..o + b.len()].copy_from_slice(b);
        put(0, &1u32.to_le_bytes());
        put(4, &self.format.to_le_bytes());
        put(8, &self.width.to_le_bytes());
        put(12, &self.height.to_le_bytes());
        put(16, &self.layers.to_le_bytes());
        put(20, &[self.kind, self.mips, self.first, self.stored]);
        put(24, &self.block.to_le_bytes());
        for (k, m) in self.mip.iter().enumerate() {
            let o = 32 + 24 * k;
            put(o, &m.offset.to_le_bytes());
            put(o + 8, &m.bytes.to_le_bytes());
            put(o + 12, &m.row_pitch.to_le_bytes());
            put(o + 16, &m.slice_pitch.to_le_bytes());
            put(o + 20, &m.rows.to_le_bytes());
        }
        r
    }
}

/// A pack file's header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    pub converter: u32,
    pub dropped: u32,
    pub file_id: u64,
    pub source_size: u64,
    pub source_mtime: u64,
    pub textures: u32,
    pub data_start: u64,
    pub key: String,
}

impl Header {
    pub fn parse(b: &[u8]) -> Option<Header> {
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().expect("4 bytes"));
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().expect("8 bytes"));
        if b.len() < HEADER as usize || &b[..4] != b"XTPK" || u32_at(4) != FORMAT || u32_at(0x2c) as usize != RECORD {
            return None;
        }
        let n = usize::from(u16::from_le_bytes([b[0x40], b[0x41]]));
        Some(Header {
            converter: u32_at(8),
            dropped: u32_at(12),
            file_id: u64_at(0x10),
            source_size: u64_at(0x18),
            source_mtime: u64_at(0x20),
            textures: u32_at(0x28),
            data_start: u64_at(0x38),
            key: String::from_utf8_lossy(b.get(0x42..0x42 + n)?).into_owned(),
        })
    }

    pub fn bytes(&self) -> Vec<u8> {
        let mut b = vec![0u8; HEADER as usize];
        b[..4].copy_from_slice(b"XTPK");
        b[4..8].copy_from_slice(&FORMAT.to_le_bytes());
        b[8..12].copy_from_slice(&self.converter.to_le_bytes());
        b[12..16].copy_from_slice(&self.dropped.to_le_bytes());
        b[0x10..0x18].copy_from_slice(&self.file_id.to_le_bytes());
        b[0x18..0x20].copy_from_slice(&self.source_size.to_le_bytes());
        b[0x20..0x28].copy_from_slice(&self.source_mtime.to_le_bytes());
        b[0x28..0x2c].copy_from_slice(&self.textures.to_le_bytes());
        b[0x2c..0x30].copy_from_slice(&(RECORD as u32).to_le_bytes());
        b[0x30..0x38].copy_from_slice(&HEADER.to_le_bytes());
        b[0x38..0x40].copy_from_slice(&self.data_start.to_le_bytes());
        let key = &self.key.as_bytes()[..self.key.len().min(0x1000 - 0x42)];
        b[0x40..0x42].copy_from_slice(&(key.len() as u16).to_le_bytes());
        b[0x42..0x42 + key.len()].copy_from_slice(key);
        b
    }

    /// Whether this pack was made from `source` as it is now.
    pub fn matches(&self, source: &Path) -> bool {
        source_stamp(source).is_ok_and(|(size, mtime)| size == self.source_size && mtime == self.source_mtime)
    }
}

/// (size, modification time in ns since 1970) of a source file.
pub fn source_stamp(path: &Path) -> io::Result<(u64, u64)> {
    let m = std::fs::metadata(path)?;
    let t = m.modified()?.duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    Ok((m.len(), t))
}

/// One open pack.
pub struct Pack {
    pub path: PathBuf,
    pub header: Header,
    file: std::fs::File,
    index: Vec<u8>,
}

impl Pack {
    pub fn open(path: &Path) -> io::Result<Pack> {
        let file = std::fs::File::open(path)?;
        let mut head = vec![0; HEADER as usize];
        read_exact_at(&file, &mut head, 0)?;
        let header = Header::parse(&head).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "not a texture pack"))?;
        let mut index = vec![0; header.textures as usize * RECORD];
        read_exact_at(&file, &mut index, HEADER)?;
        Ok(Pack { path: path.to_owned(), header, file, index })
    }

    pub fn entry(&self, pack: u32, texture: u32) -> Option<Entry> {
        let at = texture as usize * RECORD;
        Entry::parse(pack, texture, self.index.get(at..at + RECORD)?)
    }
}

fn read_exact_at(f: &std::fs::File, mut buf: &mut [u8], mut at: u64) -> io::Result<()> {
    while !buf.is_empty() {
        let n = f.seek_read(buf, at)?;
        if n == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        buf = &mut buf[n..];
        at += n as u64;
    }
    Ok(())
}

/// Every pack of a folder, by file id. Shared by reference across threads.
pub struct PackDir {
    pub packs: Vec<Pack>,
    by_id: HashMap<u64, u32>,
}

/// Opens every `.xtp` of `path` (a missing folder is an empty set; a file
/// that does not parse is skipped).
pub fn open_dir(path: &Path) -> io::Result<PackDir> {
    let mut packs = vec![];
    if path.is_dir() {
        for e in std::fs::read_dir(path)? {
            let p = e?.path();
            if p.extension().is_some_and(|x| x.eq_ignore_ascii_case(EXTENSION))
                && let Ok(pack) = Pack::open(&p)
            {
                packs.push(pack);
            }
        }
    }
    packs.sort_by(|a, b| a.path.cmp(&b.path));
    let by_id = packs.iter().enumerate().map(|(i, p)| (p.header.file_id, i as u32)).collect();
    Ok(PackDir { packs, by_id })
}

impl PackDir {
    /// The texture's entry, if converted.
    pub fn lookup(&self, file_id: u64, texture: u32) -> Option<Entry> {
        let pack = *self.by_id.get(&file_id)?;
        self.packs[pack as usize].entry(pack, texture)
    }

    /// Reads source mips `mips` (clipped to the stored ones) of `entry` into
    /// `out` from its start, coarsest first, exactly as in the file
    /// ([`Entry::span`]; mip m lands at `mip(m).offset - span.0`); returns
    /// the bytes read. `out` must hold the span.
    pub fn read_mips(&self, entry: &Entry, mips: Range<u32>, out: &mut [u8]) -> io::Result<usize> {
        let Some((from, len)) = entry.span(mips) else { return Ok(0) };
        let pack = self.packs.get(entry.pack as usize).ok_or(io::ErrorKind::NotFound)?;
        let out = out.get_mut(..len as usize).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "buffer smaller than the mips"))?;
        read_exact_at(&pack.file, out, from)?;
        Ok(len as usize)
    }

    /// Drops packs whose source under `game` (the folder holding `data/` and
    /// `pack/`) is missing or changed, or whose converter is older; the count dropped.
    pub fn drop_stale(&mut self, game: &Path) -> usize {
        let before = self.packs.len();
        self.packs.retain(|p| p.header.converter == CONVERTER && p.header.matches(&game.join(&p.header.key)));
        self.by_id = self.packs.iter().enumerate().map(|(i, p)| (p.header.file_id, i as u32)).collect();
        before - self.packs.len()
    }
}

// ---------------------------------------------------------------- converter

/// The shape of a texture in a pack: the mips stored and where each lands
/// relative to the block's start. No I/O.
#[derive(Debug, Clone)]
pub struct Plan {
    pub entry: Entry,
    /// Source slices in the console layout (array size, 6 faces, 1).
    pub slices: u32,
}

/// Source mips of `t` (a cube's copies are not mips).
pub fn source_mips(t: &Texture) -> u32 {
    if t.kind == KIND_CUBE {
        let copies = if t.mode & 8 != 0 { u32::from(t.mode & 7).max(1) } else { 1 };
        (t.mips + 1).saturating_sub(copies).max(1)
    } else {
        t.mips.max(1)
    }
}

/// Plans `t` with its `drop` finest mips left out (fewer if the chain is
/// shorter: one mip always stays). Offsets are relative to the block.
pub fn plan(t: &Texture, texture: u32, drop: u32) -> Option<Plan> {
    let f = crate::texture::format(t.format)?;
    let mips = source_mips(t);
    if mips as usize > MAX_MIPS {
        return None;
    }
    let first = drop.min(mips - 1);
    let volume = t.kind == KIND_VOLUME;
    let slices = if volume { 1 } else { t.layers.max(1) };
    let mut mip = [MipLoc::default(); MAX_MIPS];
    let mut at = 0u64;
    for m in (first..mips).rev() {
        let w = u64::from((t.width >> m).max(1).div_ceil(f.block));
        let rows = (t.height >> m).max(1).div_ceil(f.block);
        let row_pitch = (w * u64::from(f.bytes)).next_multiple_of(ROW_PITCH);
        let plane = row_pitch * u64::from(rows);
        let (count, slice_pitch) = if volume { (u64::from((t.layers >> m).max(1)), plane) } else { (u64::from(slices), plane.next_multiple_of(PLACEMENT)) };
        at = at.next_multiple_of(PLACEMENT);
        let bytes = slice_pitch * count;
        mip[m as usize] = MipLoc { offset: at, bytes: bytes as u32, row_pitch: row_pitch as u32, slice_pitch: slice_pitch as u32, rows };
        at += bytes;
    }
    Some(Plan {
        entry: Entry {
            pack: 0,
            texture,
            format: t.format,
            width: t.width,
            height: t.height,
            layers: t.layers,
            kind: t.kind,
            mips: mips as u8,
            first: first as u8,
            stored: (mips - first) as u8,
            block: at,
            mip,
        },
        slices: if t.kind == KIND_CUBE { 6 } else { slices },
    })
}

/// Block alignment for a block of `bytes`.
pub fn block_align(bytes: u64) -> u64 {
    if bytes >= 0x1_0000 { 0x1_0000 } else { 0x1000 }
}

/// The console layout's slices of `t` as one plain mip chain each, from its
/// decoded span (2D: the span; array: units gathered; cube: the last, full
/// copy, faces one after the other; volume: the span).
fn source_chains(t: &Texture, span: &[u8]) -> Result<(crate::texture::Layout, Vec<Vec<u8>>), String> {
    let p = ttpf::place(t)?;
    if p.size > t.span {
        return Err(format!("layout {:#x} longer than span {:#x}", p.size, t.span));
    }
    let chain = p.chain;
    let need = chain.size as usize;
    let fit = |mut v: Vec<u8>| {
        v.resize(v.len().max(need), 0);
        v
    };
    Ok(match t.kind {
        KIND_ARRAY => {
            let s = (0..p.slices).map(|s| fit(ttpf::gather_slice(&chain, p.slices, span, s))).collect();
            (chain, s)
        }
        KIND_CUBE => {
            // c copies, smallest first, each padded to 64 KiB above 128 KiB;
            // the last holds every unit of each face, face after face.
            let units = ttpf::units(&chain);
            let copies = (t.mips as usize + 1).saturating_sub(chain.mips.len()).max(1);
            let mut start = 0u64;
            for k in 0..copies - 1 {
                let take = (units.len() + 1 + k).saturating_sub(copies).min(units.len());
                let one: u64 = units[..take].iter().map(|u| u.1).sum::<u64>() * 6;
                start += if one > 0x2_0000 { one.next_multiple_of(0x1_0000) } else { one };
            }
            let (start, one) = (start as usize, chain.slice_size as usize);
            if start + 6 * one > span.len() {
                return Err("cube copy outside the span".into());
            }
            let s = (0..6).map(|f| fit(span[start + f * one..start + (f + 1) * one].to_vec())).collect();
            (chain, s)
        }
        _ => {
            let s = vec![fit(span.to_vec())];
            (chain, s)
        }
    })
}

/// Decodes and detiles texture `index` of `ttpf` into the block `plan`
/// describes (`plan.entry.block` bytes; offsets relative to the block).
pub fn convert(ttpf: &Ttpf, src: &mut std::fs::File, index: u32, plan: &Plan, tables: Option<&crate::bcpack::Tables>) -> Result<Vec<u8>, String> {
    let t = ttpf.textures.get(index as usize).ok_or("texture index")?;
    let span = ttpf.read_span(src, t.addr, t.span, tables)?;
    let (chain, slices) = source_chains(t, &span)?;
    let e = &plan.entry;
    let mut block = vec![0u8; e.block as usize];
    for m in e.stored_mips() {
        let loc = e.mip[m as usize];
        for (s, tiled) in slices.iter().enumerate() {
            let at = loc.offset as usize + s * loc.slice_pitch as usize;
            let len = if t.kind == KIND_VOLUME { loc.bytes } else { loc.slice_pitch } as usize;
            let out = &mut block[at..at + len];
            chain.detile(tiled, m, 0, out, loc.row_pitch as usize, loc.slice_pitch as usize).map_err(|e| format!("detile mip {m}: {e:?}"))?;
        }
    }
    Ok(block)
}

/// Writes a pack: create or resume, then add blocks from any thread.
pub struct Writer {
    pub path: PathBuf,
    file: std::fs::File,
    /// Next free byte, and the textures already present.
    end: std::sync::Mutex<u64>,
    present: Vec<bool>,
}

impl Writer {
    /// Opens `out/<pack name>` for `source`, resuming it when its header
    /// matches (source size and time, converter, drop), else starting over.
    pub fn open(out: &Path, source: &Path, textures: u32, dropped: u32) -> io::Result<Writer> {
        let key = file_key(source).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "no data/ttdb in the path"))?;
        let path = out.join(pack_name(&key));
        let (size, mtime) = source_stamp(source)?;
        let header = Header {
            converter: CONVERTER,
            dropped,
            file_id: fnv1a(key.as_bytes()),
            source_size: size,
            source_mtime: mtime,
            textures,
            data_start: (HEADER + textures as u64 * RECORD as u64).next_multiple_of(0x1_0000),
            key,
        };
        if let Ok(old) = Pack::open(&path)
            && old.header == header
        {
            let present = (0..textures).map(|i| old.entry(0, i).is_some()).collect();
            drop(old);
            let file = std::fs::OpenOptions::new().read(true).write(true).open(&path)?;
            let end = file.metadata()?.len().max(header.data_start);
            return Ok(Writer { path, file, end: std::sync::Mutex::new(end), present });
        }
        std::fs::create_dir_all(out)?;
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&path)?;
        write_all_at(&file, &header.bytes(), 0)?;
        file.set_len(header.data_start)?;
        Ok(Writer { path, file, end: std::sync::Mutex::new(header.data_start), present: vec![false; textures as usize] })
    }

    pub fn has(&self, texture: u32) -> bool {
        self.present.get(texture as usize).copied().unwrap_or(false)
    }

    /// Appends a block and then its record (so a crash leaves no record
    /// without data).
    pub fn add(&self, plan: &Plan, block: &[u8]) -> io::Result<()> {
        let at = {
            let mut end = self.end.lock().unwrap_or_else(|e| e.into_inner());
            let at = end.next_multiple_of(block_align(block.len() as u64));
            *end = at + block.len() as u64;
            at
        };
        write_all_at(&self.file, block, at)?;
        let mut e = plan.entry;
        for m in e.stored_mips() {
            e.mip[m as usize].offset += at;
        }
        write_all_at(&self.file, &e.record(), HEADER + u64::from(e.texture) * RECORD as u64)
    }
}

fn write_all_at(f: &std::fs::File, mut buf: &[u8], mut at: u64) -> io::Result<()> {
    while !buf.is_empty() {
        let n = f.seek_write(buf, at)?;
        buf = &buf[n..];
        at += n as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_agree_for_title_and_host_paths() {
        let title = file_key(Path::new(r"G:\.\data\ttdb\ui\ttdb_ui.ttpf")).unwrap();
        let host = file_key(Path::new(r"e:	itle\data\ttdb\UI\ttdb_ui.ttpf")).unwrap();
        assert_eq!(title, "data/ttdb/ui/ttdb_ui.ttpf");
        assert_eq!(title, host);
        let pack = file_key(Path::new(r"G:\pack\story000\data\ttdb\maps\ttdb_maps.ttpf")).unwrap();
        assert_eq!(pack, "pack/story000/data/ttdb/maps/ttdb_maps.ttpf");
        assert_eq!(file_id(Path::new(r"e:/title/pack/story000/data/ttdb/maps/ttdb_maps.ttpf")), Some(fnv1a(pack.as_bytes())));
        assert_eq!(pack_name(&pack), "pack_story000_data_ttdb_maps_ttdb_maps.xtp");
        assert!(file_key(Path::new(r"e:\x\y.ttpf")).is_none());
    }

    fn bc1(w: u32, h: u32, mips: u32) -> Texture {
        Texture { hash: 1, width: w, height: h, mips, format: 71, layers: 1, kind: ttpf::KIND_2D, mode: 0, addr: 0, span: 0, words: [0; 4] }
    }

    #[test]
    fn plan_is_coarsest_first_and_aligned() {
        let p = plan(&bc1(256, 128, 3), 7, 0).unwrap().entry;
        // Mip 2: 64x32 -> 16x8 blocks, 128 bytes a row -> pitch 256, 8 rows.
        assert_eq!(p.mip[2], MipLoc { offset: 0, bytes: 2048, row_pitch: 256, slice_pitch: 2048, rows: 8 });
        assert_eq!(p.mip[1].offset, 2048);
        assert_eq!(p.mip[0].offset, 2048 + 256 * 16);
        assert_eq!(p.block, p.mip[0].offset + 512 * 32);
        assert_eq!(p.span(0..3), Some((0, p.block)));
        assert_eq!(p.span(1..3), Some((0, 2048 + 4096)));
        let d = plan(&bc1(256, 128, 3), 7, 1).unwrap().entry;
        assert_eq!((d.first, d.stored, d.mip(0), d.span(0..1)), (1, 2, None, None));
        assert_eq!(plan(&bc1(256, 128, 1), 7, 2).unwrap().entry.stored, 1);
    }

    #[test]
    fn write_resume_and_read() {
        let dir = std::env::temp_dir().join(format!("xnw-texpack-{}", std::process::id()));
        let src_dir = dir.join("game").join("data").join("ttdb");
        std::fs::create_dir_all(&src_dir).unwrap();
        let src = src_dir.join("t.ttpf");
        std::fs::write(&src, b"RDS5 stand-in").unwrap();
        let out = dir.join("packs");
        let p = plan(&bc1(256, 128, 3), 1, 0).unwrap();
        let block: Vec<u8> = (0..p.entry.block).map(|i| i as u8).collect();
        {
            let w = Writer::open(&out, &src, 3, 0).unwrap();
            assert!(!w.has(1));
            w.add(&p, &block).unwrap();
        }
        let w = Writer::open(&out, &src, 3, 0).unwrap();
        assert!(w.has(1) && !w.has(0));
        drop(w);
        let mut packs = open_dir(&out).unwrap();
        assert_eq!(packs.drop_stale(&dir.join("game")), 0);
        let id = file_id(&src).unwrap();
        assert!(packs.lookup(id, 0).is_none());
        let e = packs.lookup(id, 1).unwrap();
        assert_eq!(e.mip[2].offset % 0x1000, 0);
        let mut buf = vec![0; e.block as usize];
        let n = packs.read_mips(&e, 0..3, &mut buf).unwrap();
        assert_eq!(&buf[..n], &block[..]);
        let (from, len) = e.span(1..3).unwrap();
        let n = packs.read_mips(&e, 1..3, &mut buf).unwrap();
        assert_eq!((n as u64, from), (len, e.mip[2].offset));
        std::fs::write(&src, b"RDS5 changed stand-in").unwrap();
        assert_eq!(packs.drop_stale(&dir.join("game")), 1);
        std::fs::remove_dir_all(&dir).ok();
    }
}
