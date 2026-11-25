//! `RPF9`: the title's archive format.
//!
//! Layout after rpf9-research (`tools/rpf9.py`, commit
//! 144d345191bd37de3f6bd5e85df3e8ae8815552e, GPL-3.0); written here from that
//! description, no code taken:
//!
//! ```text
//! 0x00  "9FPR"      u32 0x52504639
//! 0x04  u32 w04     end of the encrypted table of contents [0x40, w04)
//! 0x08  u32 w08     a split inside it
//! 0x0c  u32 w0c     length of the encrypted trailer at [EOF - w0c, EOF)
//! 0x10  u64         archive id
//! 0x18  u32         0x00047a00 (constant)
//! 0x1c  u32         0x00000150 (constant)
//! 0x20  u32         flags / version
//! 0x24  u32         entry count
//! 0x28  u32         build or key tag
//! 0x2c  u32         0x00090f01 or 0x00090f02
//! 0x30  u32         16 (constant), then three zero u32
//! ```
//!
//! The table of contents and the trailer are encrypted and no key is known,
//! so entries cannot be listed from the index. They are recovered from the
//! payload instead: the first entry starts at the first non-zero byte at or
//! after `w04`; each further one starts on a 4 KiB boundary after at least 16
//! zero bytes of padding, with a recognisable signature (`SGD5`, a nested
//! `RPF9` header, or a zlib stream that inflates). The count found is checked
//! against the header's. An entry's size is the span to the next entry (or the
//! trailer) with trailing zero padding removed, so it can be short by any
//! zero bytes the entry itself ends with.

use std::fmt;

pub const MAGIC: [u8; 4] = *b"9FPR";
pub const HEADER_SIZE: usize = 0x40;
const PAGE: usize = 0x1000;
const PAD: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rpf9Error {
    Truncated,
    BadMagic,
    /// A field that is constant in every known archive has another value.
    BadConstant(&'static str),
    BadBounds(&'static str),
}

impl fmt::Display for Rpf9Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Rpf9Error::Truncated => f.write_str("RPF9: file shorter than its header"),
            Rpf9Error::BadMagic => f.write_str("RPF9: magic is not 9FPR"),
            Rpf9Error::BadConstant(what) => write!(f, "RPF9: unexpected {what}"),
            Rpf9Error::BadBounds(what) => write!(f, "RPF9: {what} outside the file"),
        }
    }
}

impl std::error::Error for Rpf9Error {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub toc_end: u32,
    pub toc_split: u32,
    pub trailer_len: u32,
    pub archive_id: u64,
    pub flags: u32,
    pub entry_count: u32,
    pub tag: u32,
    pub stride_flag: u32,
}

impl Header {
    pub fn parse(data: &[u8]) -> Result<Self, Rpf9Error> {
        let h = data.get(..HEADER_SIZE).ok_or(Rpf9Error::Truncated)?;
        if h[..4] != MAGIC {
            return Err(Rpf9Error::BadMagic);
        }
        let w = |i: usize| u32::from_le_bytes(h[4 * i..4 * i + 4].try_into().expect("4 bytes"));
        if w(6) != 0x0004_7A00 || w(7) != 0x150 {
            return Err(Rpf9Error::BadConstant("format constants at 0x18"));
        }
        if w(12) != 16 || w(13) != 0 || w(14) != 0 || w(15) != 0 {
            return Err(Rpf9Error::BadConstant("alignment field at 0x30"));
        }
        Ok(Header {
            toc_end: w(1),
            toc_split: w(2),
            trailer_len: w(3),
            archive_id: u64::from(w(4)) | u64::from(w(5)) << 32,
            flags: w(8),
            entry_count: w(9),
            tag: w(10),
            stride_flag: w(11),
        })
    }
}

/// What an entry starts with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EntryKind {
    Sgd5,
    Rpf9,
    Zlib,
    /// The first entry, whose start needs no signature.
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    pub offset: usize,
    pub size: usize,
    pub kind: EntryKind,
}

#[derive(Debug, Clone)]
pub struct Archive<'a> {
    pub data: &'a [u8],
    pub header: Header,
    pub entries: Vec<Entry>,
}

impl<'a> Archive<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, Rpf9Error> {
        let header = Header::parse(data)?;
        let toc_end = header.toc_end as usize;
        let trailer = header.trailer_len as usize;
        let payload_end = data
            .len()
            .checked_sub(trailer)
            .filter(|&e| e >= toc_end && toc_end >= HEADER_SIZE)
            .ok_or(Rpf9Error::BadBounds("table of contents or trailer"))?;
        let mut starts: Vec<(usize, EntryKind)> = Vec::new();
        if let Some(first) = data[toc_end..payload_end].iter().position(|&b| b != 0) {
            let at = toc_end + first;
            starts.push((at, kind_at(&data[at..payload_end]).unwrap_or(EntryKind::Other)));
            let mut page = (at + 1).next_multiple_of(PAGE);
            while page < payload_end {
                if data[page - PAD..page].iter().all(|&b| b == 0)
                    && let Some(kind) = kind_at(&data[page..payload_end])
                {
                    starts.push((page, kind));
                }
                page += PAGE;
            }
        }
        let entries = starts
            .iter()
            .enumerate()
            .map(|(i, &(offset, kind))| {
                let limit = starts.get(i + 1).map_or(payload_end, |s| s.0);
                let used = data[offset..limit].iter().rposition(|&b| b != 0).map_or(0, |p| p + 1);
                Entry {
                    offset,
                    size: used,
                    kind,
                }
            })
            .collect();
        Ok(Archive {
            data,
            header,
            entries,
        })
    }

    /// The bytes of entry `index`.
    pub fn read(&self, index: usize) -> Option<&'a [u8]> {
        let e = self.entries.get(index)?;
        self.data.get(e.offset..e.offset + e.size)
    }

    /// True when the entries found are as many as the header states.
    pub fn count_matches(&self) -> bool {
        self.entries.len() == self.header.entry_count as usize
    }
}

/// The signature an entry may start with.
fn kind_at(data: &[u8]) -> Option<EntryKind> {
    if data.starts_with(b"SGD5") {
        return Some(EntryKind::Sgd5);
    }
    if data.starts_with(&MAGIC) && Header::parse(data).is_ok() {
        return Some(EntryKind::Rpf9);
    }
    if is_zlib(data) {
        return Some(EntryKind::Zlib);
    }
    None
}

/// A zlib header whose stream inflates without error over its first bytes.
fn is_zlib(data: &[u8]) -> bool {
    if data.len() < 2 || data[0] != 0x78 || !(u16::from(data[0]) << 8 | u16::from(data[1])).is_multiple_of(31) {
        return false;
    }
    let window = &data[..data.len().min(0x4000)];
    let mut out = vec![0u8; 0x10000];
    let mut state = miniz_oxide::inflate::stream::InflateState::new(miniz_oxide::DataFormat::Zlib);
    let r = miniz_oxide::inflate::stream::inflate(
        &mut state,
        window,
        &mut out,
        miniz_oxide::MZFlush::None,
    );
    r.status.is_ok() && r.bytes_written > 0
}

/// What [`zlib_run`] found.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ZlibRun {
    pub streams: usize,
    pub inflated: u64,
    /// `DXBC` magics in the inflated bytes.
    pub dxbc: usize,
    /// Offset of the first stream that does not inflate.
    pub error: Option<usize>,
}

/// Inflates an entry made of consecutive zlib streams (each padded with zero
/// bytes to the next one), counting shader container magics inside.
pub fn zlib_run(data: &[u8]) -> ZlibRun {
    use miniz_oxide::inflate::stream::{InflateState, inflate};
    use miniz_oxide::{DataFormat, MZFlush, MZStatus};
    let mut run = ZlibRun::default();
    let mut out = vec![0u8; 1 << 20];
    // The tail of the previous output, for magics split across buffers.
    let mut carry = [0u8; 3];
    let mut at = 0;
    while let Some(skip) = data[at..].iter().position(|&b| b != 0) {
        at += skip;
        let begin = at;
        let mut state = InflateState::new_boxed(DataFormat::Zlib);
        loop {
            let r = inflate(&mut state, &data[at..], &mut out, MZFlush::None);
            at += r.bytes_consumed;
            let got = &out[..r.bytes_written];
            run.inflated += got.len() as u64;
            run.dxbc += got.windows(4).filter(|w| *w == b"DXBC").count();
            let mut seam = carry.to_vec();
            seam.extend_from_slice(&got[..got.len().min(3)]);
            run.dxbc += seam.windows(4).filter(|w| *w == b"DXBC").count();
            if got.len() >= 3 {
                carry.copy_from_slice(&got[got.len() - 3..]);
            }
            match r.status {
                Ok(MZStatus::StreamEnd) => break,
                Ok(_) if r.bytes_consumed > 0 || r.bytes_written > 0 => {}
                _ => {
                    run.error = Some(begin);
                    return run;
                }
            }
        }
        run.streams += 1;
    }
    run
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(toc_end: u32, trailer: u32, count: u32) -> Vec<u8> {
        let words = [
            0x5250_4639,
            toc_end,
            toc_end - 0x20,
            trailer,
            0x1234_5678,
            0x9ABC_DEF0,
            0x0004_7A00,
            0x150,
            0x0001_0005,
            count,
            0xDA72_573E,
            0x0009_0F01,
            16,
            0,
            0,
            0,
        ];
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// Header, TOC noise, three entries on pages, trailer noise.
    fn archive() -> Vec<u8> {
        let mut a = header(0x100, 0x30, 3);
        a.resize(0x100, 0xA5); // "encrypted" TOC
        a.resize(0x1000, 0);
        a.extend_from_slice(b"SGD5 first entry");
        a.resize(0x3000, 0);
        let z = miniz_oxide::deflate::compress_to_vec_zlib(&[7u8; 5000], 6);
        a.extend_from_slice(&z);
        a.resize(0x5000, 0);
        a.extend_from_slice(&header(0x40, 0x10, 0)); // nested archive
        a.extend_from_slice(&[1, 2, 3]);
        a.resize(a.len() + 7, 0);
        a.extend_from_slice(&[0x5A; 0x30]); // trailer
        a
    }

    #[test]
    fn lists_entries_by_signature() {
        let data = archive();
        let ar = Archive::parse(&data).unwrap();
        assert_eq!(ar.header.entry_count, 3);
        assert_eq!(ar.header.archive_id, 0x9ABC_DEF0_1234_5678);
        let found: Vec<_> = ar.entries.iter().map(|e| (e.offset, e.kind)).collect();
        assert_eq!(
            found,
            [(0x1000, EntryKind::Sgd5), (0x3000, EntryKind::Zlib), (0x5000, EntryKind::Rpf9)]
        );
        assert!(ar.count_matches());
        assert_eq!(ar.read(0).unwrap(), b"SGD5 first entry");
        assert_eq!(ar.read(2).unwrap().len(), HEADER_SIZE + 3);
        let z = ar.read(1).unwrap();
        assert_eq!(miniz_oxide::inflate::decompress_to_vec_zlib(z).unwrap(), vec![7u8; 5000]);
    }

    #[test]
    fn page_without_padding_or_signature_is_not_an_entry() {
        let mut data = archive();
        // Non-zero bytes right before 0x3000: no padding, so no new entry.
        data[0x2FFF] = 1;
        let ar = Archive::parse(&data).unwrap();
        assert_eq!(ar.entries.len(), 2);
        assert!(!ar.count_matches());
        // A padded page with no signature is no entry either.
        let mut data = archive();
        data[0x4000] = 0x11;
        assert_eq!(Archive::parse(&data).unwrap().entries.len(), 3);
    }

    #[test]
    fn inflates_a_run_of_streams() {
        let mut data = miniz_oxide::deflate::compress_to_vec_zlib(b"..DXBC..", 6);
        data.resize(data.len().next_multiple_of(16), 0);
        data.extend(miniz_oxide::deflate::compress_to_vec_zlib(&[3u8; 70000], 6));
        let run = zlib_run(&data);
        assert_eq!((run.streams, run.inflated, run.dxbc, run.error), (2, 70008, 1, None));
        data.extend_from_slice(&[0x78, 0xDA, 0xFF, 0xFF]);
        assert!(zlib_run(&data).error.is_some());
    }

    #[test]
    fn rejects_bad_headers() {
        assert_eq!(Header::parse(&[0; 16]), Err(Rpf9Error::Truncated));
        let mut h = header(0x100, 0, 1);
        h[0] = b'X';
        assert_eq!(Header::parse(&h), Err(Rpf9Error::BadMagic));
        let mut h = header(0x100, 0, 1);
        h[0x19] = 0;
        assert!(matches!(Header::parse(&h), Err(Rpf9Error::BadConstant(_))));
        // TOC end past the file.
        assert!(matches!(Archive::parse(&header(0x1000, 0, 1)), Err(Rpf9Error::BadBounds(_))));
    }
}
