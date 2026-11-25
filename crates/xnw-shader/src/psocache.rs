//! The title's pipeline library: the `PS 00 09` entry of `shaders.rpf`,
//! which the title loads as its `.d3d12PipelineCache` and creates pipelines
//! from through device slots 57 (graphics) and 67 (compute).
//!
//! Layout after five-star-recomp's pipeline-cache notes and
//! `src/FiveStar.Shaders/Cache/PipelineCache.cs`, `PipelineCacheHeader.cs`
//! (commit 0bc9cd731e363041637504dac75e22f53ce34c58, GPL-3.0-or-later),
//! cross-read with XSXEmu's pipeline-library notes (MIT);
//! written here from those descriptions. The graphics record stride
//! (0x70) and how a record's ids become the {pointer, size} ranges of the
//! slot-57 description are checked in the title (`title+0x248f680`,
//! `title+0x248feb0`, `title+0x248fee0`).
//!
//! ```text
//! header 0x80 | graphics 0x70 x G | compute 0x18 x C | metadata index 8 x M
//! | shader index 8 x S | metadata area | code area (256-byte aligned)
//! ```
//!
//! In the pack the library is a run of zlib streams, each padded with zero
//! bytes to the next, starting at the first stream whose output begins with
//! the magic and read until the header's total size.

use std::fmt;

pub const MAGIC: [u8; 4] = [b'P', b'S', 0, 9];
pub const HEADER: usize = 0x80;
pub const GRAPHICS_RECORD: usize = 0x70;
pub const COMPUTE_RECORD: usize = 0x18;
/// Shader slots of a graphics record: 0 pixel, 1 vertex, 2 mesh (NGG),
/// 3-5 never used, 6 the driver's fetch shader.
pub const SLOTS: usize = 7;
pub const SLOT_NAMES: [&str; SLOTS] = ["pixel", "vertex", "mesh", "slot3", "slot4", "slot5", "fetch"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LibraryError {
    NotFound,
    BadMagic,
    Truncated(&'static str),
    /// A zlib stream of the run does not inflate (archive offset).
    Inflate(usize),
}

impl fmt::Display for LibraryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LibraryError::NotFound => f.write_str("no pipeline library (PS 00 09) in the archive"),
            LibraryError::BadMagic => f.write_str("not a pipeline library (magic PS 00 09)"),
            LibraryError::Truncated(what) => write!(f, "pipeline library ends inside its {what}"),
            LibraryError::Inflate(at) => write!(f, "pipeline library stream at {at:#x} does not inflate"),
        }
    }
}

impl std::error::Error for LibraryError {}

fn u32_at(b: &[u8], at: usize) -> u32 {
    b.get(at..at + 4).map_or(0, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from(u32_at(b, at)) | u64::from(u32_at(b, at + 4)) << 32
}

fn text(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).into_owned()
}

/// The library header (0x80 bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    /// 0x04, 32 bytes of ASCII.
    pub id: String,
    /// 0x24, the driver version that compiled the code.
    pub driver: String,
    pub stamp: u64,
    pub graphics: usize,
    pub compute: usize,
    pub metadata: usize,
    pub shaders: usize,
    pub metadata_bytes: usize,
    pub code_bytes: usize,
}

impl Header {
    pub fn parse(b: &[u8]) -> Result<Header, LibraryError> {
        if b.len() < HEADER {
            return Err(LibraryError::Truncated("header"));
        }
        if b[..4] != MAGIC {
            return Err(LibraryError::BadMagic);
        }
        Ok(Header {
            id: text(&b[0x04..0x24]),
            driver: text(&b[0x24..0x48]),
            stamp: u64_at(b, 0x48),
            graphics: u32_at(b, 0x50) as usize,
            compute: u32_at(b, 0x54) as usize,
            metadata: u32_at(b, 0x58) as usize,
            shaders: u32_at(b, 0x5c) as usize,
            metadata_bytes: u32_at(b, 0x60) as usize,
            code_bytes: u32_at(b, 0x64) as usize,
        })
    }
    pub fn compute_at(&self) -> usize {
        HEADER + self.graphics * GRAPHICS_RECORD
    }
    pub fn metadata_index_at(&self) -> usize {
        self.compute_at() + self.compute * COMPUTE_RECORD
    }
    pub fn shader_index_at(&self) -> usize {
        self.metadata_index_at() + self.metadata * 8
    }
    pub fn metadata_at(&self) -> usize {
        self.shader_index_at() + self.shaders * 8
    }
    pub fn code_at(&self) -> usize {
        self.metadata_at() + self.metadata_bytes
    }
    pub fn total(&self) -> usize {
        self.code_at() + self.code_bytes
    }
}

/// One graphics pipeline record (0x70 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphicsRecord {
    pub hash: u64,
    /// Metadata id of the fixed-function state (context registers).
    pub state: u32,
    /// Metadata id of the outputs (formats) and vertex input layout.
    pub outputs: u32,
    /// 0x10: the shader hash the title keys root signatures by; 0x18 the
    /// fixed-function hash; 0x20/0x28/0x30 depth-stencil, rasterizer and
    /// blend hashes (five-star, inferred from the title's comparisons).
    pub key: u64,
    pub fixed_hash: u64,
    pub depth_hash: u64,
    pub raster_hash: u64,
    pub blend_hash: u64,
    /// Per slot `(shader id, metadata id)`, `u32::MAX` when empty.
    pub slots: [(u32, u32); SLOTS],
}

/// One compute pipeline record (0x18 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeRecord {
    pub hash: u64,
    /// Metadata id of the 28-byte compute state.
    pub state: u32,
    /// Metadata id of a 20-byte entry (a hash then 3, 0, 0, 0).
    pub extra: u32,
    pub shader: u32,
    pub shader_meta: u32,
}

/// The inflated library.
pub struct Library {
    pub header: Header,
    pub bytes: Vec<u8>,
}

impl Library {
    /// The library from its inflated bytes; checks that every table fits.
    pub fn new(bytes: Vec<u8>) -> Result<Library, LibraryError> {
        let header = Header::parse(&bytes)?;
        if bytes.len() < header.metadata_at() + header.metadata_bytes {
            return Err(LibraryError::Truncated("tables"));
        }
        Ok(Library { header, bytes })
    }

    /// Finds the library in an `RPF9` archive's bytes and inflates it.
    pub fn from_archive(data: &[u8]) -> Result<Library, LibraryError> {
        let at = find(data).ok_or(LibraryError::NotFound)?;
        Library::new(inflate_run(data, at)?)
    }

    pub fn graphics(&self, i: usize) -> Option<GraphicsRecord> {
        if i >= self.header.graphics {
            return None;
        }
        let r = &self.bytes[HEADER + i * GRAPHICS_RECORD..][..GRAPHICS_RECORD];
        let mut slots = [(u32::MAX, u32::MAX); SLOTS];
        for (s, slot) in slots.iter_mut().enumerate() {
            *slot = (u32_at(r, 0x38 + s * 8), u32_at(r, 0x3c + s * 8));
        }
        Some(GraphicsRecord {
            hash: u64_at(r, 0),
            state: u32_at(r, 0x08),
            outputs: u32_at(r, 0x0c),
            key: u64_at(r, 0x10),
            fixed_hash: u64_at(r, 0x18),
            depth_hash: u64_at(r, 0x20),
            raster_hash: u64_at(r, 0x28),
            blend_hash: u64_at(r, 0x30),
            slots,
        })
    }

    pub fn compute(&self, i: usize) -> Option<ComputeRecord> {
        if i >= self.header.compute {
            return None;
        }
        let r = &self.bytes[self.header.compute_at() + i * COMPUTE_RECORD..][..COMPUTE_RECORD];
        Some(ComputeRecord { hash: u64_at(r, 0), state: u32_at(r, 8), extra: u32_at(r, 0x0c), shader: u32_at(r, 0x10), shader_meta: u32_at(r, 0x14) })
    }

    /// Metadata entry `id` (`None` for `u32::MAX` or out of range), as the
    /// title's `title+0x248feb0` resolves it: offset and size from the index.
    pub fn metadata(&self, id: u32) -> Option<&[u8]> {
        self.range(self.header.metadata_index_at(), self.header.metadata, self.header.metadata_at(), id)
    }

    /// Code block of shader `id`.
    pub fn shader(&self, id: u32) -> Option<&[u8]> {
        self.range(self.header.shader_index_at(), self.header.shaders, self.header.code_at(), id)
    }

    fn range(&self, index: usize, count: usize, area: usize, id: u32) -> Option<&[u8]> {
        let id = id as usize;
        if id >= count {
            return None;
        }
        let offset = u32_at(&self.bytes, index + id * 8) as usize;
        let size = u32_at(&self.bytes, index + id * 8 + 4) as usize;
        self.bytes.get(area + offset..area + offset + size)
    }
}

/// The archive offset of the zlib stream that starts the library: on a
/// 4 KiB page after zero padding, inflating to the magic.
pub fn find(data: &[u8]) -> Option<usize> {
    const PAGE: usize = 0x1000;
    let mut page = PAGE;
    while page + 2 < data.len() {
        if data[page] == 0x78 && data[page - 16..page].iter().all(|&b| b == 0) && starts_with_magic(&data[page..]) {
            return Some(page);
        }
        page += PAGE;
    }
    None
}

fn starts_with_magic(data: &[u8]) -> bool {
    use miniz_oxide::inflate::stream::{InflateState, inflate};
    let mut state = InflateState::new_boxed(miniz_oxide::DataFormat::Zlib);
    let mut out = [0u8; 64];
    let r = inflate(&mut state, &data[..data.len().min(0x1000)], &mut out, miniz_oxide::MZFlush::None);
    r.status.is_ok() && r.bytes_written >= 4 && out[..4] == MAGIC
}

/// Inflates consecutive zlib streams from `at` until the header's total.
pub fn inflate_run(data: &[u8], mut at: usize) -> Result<Vec<u8>, LibraryError> {
    use miniz_oxide::inflate::stream::{InflateState, inflate};
    use miniz_oxide::{DataFormat, MZFlush, MZStatus};
    let mut out: Vec<u8> = Vec::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut total = usize::MAX;
    while out.len() < total {
        let skip = data.get(at..).and_then(|d| d.iter().position(|&b| b != 0)).ok_or(LibraryError::Truncated("zlib run"))?;
        at += skip;
        let begin = at;
        let mut state = InflateState::new_boxed(DataFormat::Zlib);
        loop {
            let r = inflate(&mut state, &data[at..], &mut buf, MZFlush::None);
            at += r.bytes_consumed;
            out.extend_from_slice(&buf[..r.bytes_written]);
            match r.status {
                Ok(MZStatus::StreamEnd) => break,
                Ok(_) if r.bytes_consumed > 0 || r.bytes_written > 0 => {}
                _ => return Err(LibraryError::Inflate(begin)),
            }
        }
        if total == usize::MAX && out.len() >= HEADER {
            let h = Header::parse(&out)?;
            total = h.total();
            out.reserve(total.saturating_sub(out.len()));
        }
    }
    out.truncate(total);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A library of one graphics and one compute record, built by hand.
    pub(crate) fn synthetic() -> Vec<u8> {
        let mut b = vec![0u8; HEADER];
        b[..4].copy_from_slice(&MAGIC);
        b[0x04..0x0e].copy_from_slice(b"0x65F41EDD");
        b[0x24..0x2f].copy_from_slice(b"930.048.020");
        let put = |b: &mut Vec<u8>, at: usize, v: u32| b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        put(&mut b, 0x50, 1);
        put(&mut b, 0x54, 1);
        put(&mut b, 0x58, 3);
        put(&mut b, 0x5c, 2);
        put(&mut b, 0x60, 12);
        put(&mut b, 0x64, 0x200);
        let mut g = vec![0u8; GRAPHICS_RECORD];
        g[0] = 0xaa;
        g[0x08] = 0; // state metadata 0
        g[0x0c] = 1; // outputs metadata 1
        for s in 0..SLOTS {
            let (sh, me) = if s == 0 { (0u32, 2u32) } else { (u32::MAX, u32::MAX) };
            g[0x38 + s * 8..0x3c + s * 8].copy_from_slice(&sh.to_le_bytes());
            g[0x3c + s * 8..0x40 + s * 8].copy_from_slice(&me.to_le_bytes());
        }
        b.extend_from_slice(&g);
        let mut c = vec![0u8; COMPUTE_RECORD];
        c[0x10] = 1;
        b.extend_from_slice(&c);
        for (o, s) in [(0u32, 4u32), (4, 4), (8, 4)] {
            b.extend_from_slice(&o.to_le_bytes());
            b.extend_from_slice(&s.to_le_bytes());
        }
        for (o, s) in [(0u32, 8u32), (0x100, 52)] {
            b.extend_from_slice(&o.to_le_bytes());
            b.extend_from_slice(&s.to_le_bytes());
        }
        b.extend_from_slice(&[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
        let code = b.len();
        b.resize(code + 0x200, 0);
        b[code] = 0x11;
        b[code + 0x100] = 0x22;
        b
    }

    #[test]
    fn reads_tables() {
        let lib = Library::new(synthetic()).unwrap();
        assert_eq!(lib.header.driver, "930.048.020");
        assert_eq!(lib.header.total(), lib.bytes.len());
        let g = lib.graphics(0).unwrap();
        assert_eq!(g.hash, 0xaa);
        assert_eq!(g.slots[0], (0, 2));
        assert_eq!(g.slots[1], (u32::MAX, u32::MAX));
        assert_eq!(lib.metadata(g.outputs), Some(&[2u8, 0, 0, 0][..]));
        assert_eq!(lib.metadata(u32::MAX), None);
        assert_eq!(lib.shader(0).unwrap()[0], 0x11);
        let c = lib.compute(0).unwrap();
        assert_eq!(lib.shader(c.shader).unwrap().len(), 52);
        assert_eq!(lib.shader(c.shader).unwrap()[0], 0x22);
    }

    #[test]
    fn finds_and_inflates_a_split_run() {
        let lib = synthetic();
        let (a, b) = lib.split_at(100);
        let mut data = vec![0u8; 0x1000];
        data.extend(miniz_oxide::deflate::compress_to_vec_zlib(a, 6));
        data.resize(0x2000, 0);
        data.extend(miniz_oxide::deflate::compress_to_vec_zlib(b, 6));
        data.resize(0x3000, 0);
        assert_eq!(find(&data), Some(0x1000));
        assert_eq!(inflate_run(&data, 0x1000).unwrap(), lib);
    }
}
