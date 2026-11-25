//! Read-only PE32+ (x64) parsing over a byte slice. No dependencies, no
//! allocation beyond the returned tables. Used by `xnw inspect` and, later,
//! by the loader to plan a mapping before anything is executed.

use std::fmt;

#[derive(Debug)]
pub enum PeError {
    Truncated(&'static str),
    NotPe32Plus,
    BadRva(u32),
}

impl fmt::Display for PeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PeError::Truncated(what) => write!(f, "truncated image: {what}"),
            PeError::NotPe32Plus => write!(f, "not an x64 PE32+ image"),
            PeError::BadRva(rva) => write!(f, "rva {rva:#x} is not backed by file data"),
        }
    }
}

impl std::error::Error for PeError {}

type Result<T> = std::result::Result<T, PeError>;

/// Data directory indices.
pub mod dir {
    pub const EXPORT: usize = 0;
    pub const IMPORT: usize = 1;
    pub const EXCEPTION: usize = 3;
    pub const RELOC: usize = 5;
    pub const DEBUG: usize = 6;
    pub const TLS: usize = 9;
    pub const LOAD_CONFIG: usize = 10;
    pub const IAT: usize = 12;
    pub const DELAY_IMPORT: usize = 13;
}

#[derive(Debug, Clone)]
pub struct Section {
    pub name: String,
    pub rva: u32,
    pub virtual_size: u32,
    pub file_offset: u32,
    pub file_size: u32,
    pub flags: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Symbol {
    Name(String),
    Ordinal(u16),
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Symbol::Name(n) => f.write_str(n),
            Symbol::Ordinal(o) => write!(f, "#{o}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Import {
    pub symbol: Symbol,
    /// RVA of the IAT slot the loader fills.
    pub slot_rva: u32,
}

#[derive(Debug, Clone)]
pub struct ImportModule {
    pub dll: String,
    pub delay: bool,
    pub imports: Vec<Import>,
}

#[derive(Debug, Clone)]
pub struct Export {
    pub name: Option<String>,
    pub ordinal: u32,
    pub rva: u32,
}

#[derive(Debug, Clone)]
pub struct Tls {
    /// RVA of the TLS directory itself.
    pub directory_rva: u32,
    /// RVAs of the callbacks, in call order.
    pub callback_rvas: Vec<u32>,
    pub raw_start_va: u64,
    pub raw_end_va: u64,
    pub index_va: u64,
    pub callbacks_va: u64,
    pub zero_fill: u32,
}

#[derive(Debug)]
pub struct Image<'a> {
    pub data: &'a [u8],
    /// File offsets of the COFF file header and the optional header.
    pub file_header: usize,
    pub optional_header: usize,
    pub characteristics: u16,
    pub stack_reserve: u64,
    pub timestamp: u32,
    pub image_base: u64,
    pub entry_rva: u32,
    pub size_of_image: u32,
    pub subsystem: u16,
    pub dll_characteristics: u16,
    pub dirs: [(u32, u32); 16],
    pub sections: Vec<Section>,
}

fn u16_at(d: &[u8], o: usize) -> Result<u16> {
    d.get(o..o + 2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .ok_or(PeError::Truncated("u16"))
}

fn u32_at(d: &[u8], o: usize) -> Result<u32> {
    d.get(o..o + 4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .ok_or(PeError::Truncated("u32"))
}

fn u64_at(d: &[u8], o: usize) -> Result<u64> {
    d.get(o..o + 8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .ok_or(PeError::Truncated("u64"))
}

impl<'a> Image<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        if data.get(..2) != Some(b"MZ") {
            return Err(PeError::NotPe32Plus);
        }
        let pe = u32_at(data, 0x3c)? as usize;
        if data.get(pe..pe + 4) != Some(b"PE\0\0") || u16_at(data, pe + 4)? != 0x8664 {
            return Err(PeError::NotPe32Plus);
        }
        let section_count = u16_at(data, pe + 6)? as usize;
        let opt_size = u16_at(data, pe + 20)? as usize;
        let opt = pe + 24;
        if u16_at(data, opt)? != 0x20b {
            return Err(PeError::NotPe32Plus);
        }
        let mut dirs = [(0u32, 0u32); 16];
        let dir_count = (u32_at(data, opt + 108)? as usize).min(16);
        for (i, d) in dirs.iter_mut().enumerate().take(dir_count) {
            *d = (u32_at(data, opt + 112 + i * 8)?, u32_at(data, opt + 116 + i * 8)?);
        }
        let mut sections = Vec::with_capacity(section_count);
        for i in 0..section_count {
            let s = opt + opt_size + i * 40;
            let raw = data.get(s..s + 8).ok_or(PeError::Truncated("section header"))?;
            let end = raw.iter().position(|&b| b == 0).unwrap_or(8);
            sections.push(Section {
                name: String::from_utf8_lossy(&raw[..end]).into_owned(),
                virtual_size: u32_at(data, s + 8)?,
                rva: u32_at(data, s + 12)?,
                file_size: u32_at(data, s + 16)?,
                file_offset: u32_at(data, s + 20)?,
                flags: u32_at(data, s + 36)?,
            });
        }
        Ok(Image {
            data,
            file_header: pe + 4,
            optional_header: opt,
            characteristics: u16_at(data, pe + 22)?,
            stack_reserve: u64_at(data, opt + 72)?,
            timestamp: u32_at(data, pe + 8)?,
            entry_rva: u32_at(data, opt + 16)?,
            image_base: u64_at(data, opt + 24)?,
            size_of_image: u32_at(data, opt + 56)?,
            subsystem: u16_at(data, opt + 68)?,
            dll_characteristics: u16_at(data, opt + 70)?,
            dirs,
            sections,
        })
    }

    /// File offset backing `rva`, if any section holds it on disk.
    pub fn offset(&self, rva: u32) -> Result<usize> {
        self.sections
            .iter()
            .find(|s| rva >= s.rva && rva - s.rva < s.file_size)
            .map(|s| (rva - s.rva + s.file_offset) as usize)
            .ok_or(PeError::BadRva(rva))
    }

    pub fn va_to_rva(&self, va: u64) -> u32 {
        va.wrapping_sub(self.image_base) as u32
    }

    fn cstr(&self, rva: u32) -> Result<String> {
        let tail = &self.data[self.offset(rva)?..];
        let end = tail.iter().position(|&b| b == 0).ok_or(PeError::Truncated("string"))?;
        Ok(String::from_utf8_lossy(&tail[..end]).into_owned())
    }

    fn thunks(&self, lookup_rva: u32, iat_rva: u32) -> Result<Vec<Import>> {
        let mut out = Vec::new();
        let mut o = self.offset(lookup_rva)?;
        let mut slot_rva = iat_rva;
        loop {
            let v = u64_at(self.data, o)?;
            if v == 0 {
                return Ok(out);
            }
            let symbol = if v >> 63 != 0 {
                Symbol::Ordinal(v as u16)
            } else {
                Symbol::Name(self.cstr(v as u32 + 2)?)
            };
            out.push(Import { symbol, slot_rva });
            o += 8;
            slot_rva += 8;
        }
    }

    /// Normal and delay-load imports, in directory order.
    pub fn imports(&self) -> Result<Vec<ImportModule>> {
        let mut out = Vec::new();
        let (rva, _) = self.dirs[dir::IMPORT];
        if rva != 0 {
            let mut o = self.offset(rva)?;
            loop {
                let (lookup, name, iat) =
                    (u32_at(self.data, o)?, u32_at(self.data, o + 12)?, u32_at(self.data, o + 16)?);
                if name == 0 {
                    break;
                }
                out.push(ImportModule {
                    dll: self.cstr(name)?,
                    delay: false,
                    imports: self.thunks(if lookup != 0 { lookup } else { iat }, iat)?,
                });
                o += 20;
            }
        }
        let (rva, _) = self.dirs[dir::DELAY_IMPORT];
        if rva != 0 {
            let mut o = self.offset(rva)?;
            loop {
                let (name, iat, lookup) =
                    (u32_at(self.data, o + 4)?, u32_at(self.data, o + 12)?, u32_at(self.data, o + 16)?);
                if name == 0 {
                    break;
                }
                out.push(ImportModule {
                    dll: self.cstr(name)?,
                    delay: true,
                    imports: self.thunks(if lookup != 0 { lookup } else { iat }, iat)?,
                });
                o += 32;
            }
        }
        Ok(out)
    }

    pub fn exports(&self) -> Result<Vec<Export>> {
        let (rva, _) = self.dirs[dir::EXPORT];
        if rva == 0 {
            return Ok(Vec::new());
        }
        let o = self.offset(rva)?;
        let base = u32_at(self.data, o + 16)?;
        let (count, named) = (u32_at(self.data, o + 20)?, u32_at(self.data, o + 24)?);
        let funcs = self.offset(u32_at(self.data, o + 28)?)?;
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count {
            out.push(Export {
                name: None,
                ordinal: base + i,
                rva: u32_at(self.data, funcs + i as usize * 4)?,
            });
        }
        if named != 0 {
            let names = self.offset(u32_at(self.data, o + 32)?)?;
            let ords = self.offset(u32_at(self.data, o + 36)?)?;
            for i in 0..named as usize {
                let idx = u16_at(self.data, ords + i * 2)? as usize;
                if let Some(e) = out.get_mut(idx) {
                    e.name = Some(self.cstr(u32_at(self.data, names + i * 4)?)?);
                }
            }
        }
        Ok(out)
    }

    pub fn tls(&self) -> Result<Option<Tls>> {
        let (rva, _) = self.dirs[dir::TLS];
        if rva == 0 {
            return Ok(None);
        }
        let o = self.offset(rva)?;
        let callbacks_va = u64_at(self.data, o + 24)?;
        let mut callback_rvas = Vec::new();
        if callbacks_va != 0 {
            let mut c = self.offset(self.va_to_rva(callbacks_va))?;
            loop {
                let va = u64_at(self.data, c)?;
                if va == 0 {
                    break;
                }
                callback_rvas.push(self.va_to_rva(va));
                c += 8;
            }
        }
        Ok(Some(Tls {
            directory_rva: rva,
            callback_rvas,
            raw_start_va: u64_at(self.data, o)?,
            raw_end_va: u64_at(self.data, o + 8)?,
            index_va: u64_at(self.data, o + 16)?,
            callbacks_va: u64_at(self.data, o + 24)?,
            zero_fill: u32_at(self.data, o + 32)?,
        }))
    }

    /// Path of the PDB the image was linked with (CodeView RSDS record).
    pub fn pdb_path(&self) -> Result<Option<String>> {
        let (rva, size) = self.dirs[dir::DEBUG];
        if rva == 0 {
            return Ok(None);
        }
        let o = self.offset(rva)?;
        for i in 0..size as usize / 28 {
            let e = o + i * 28;
            if u32_at(self.data, e + 12)? == 2 {
                let raw = u32_at(self.data, e + 24)? as usize;
                if self.data.get(raw..raw + 4) == Some(b"RSDS") {
                    let tail = &self.data[raw + 24..];
                    let end = tail.iter().position(|&b| b == 0).unwrap_or(0);
                    return Ok(Some(String::from_utf8_lossy(&tail[..end]).into_owned()));
                }
            }
        }
        Ok(None)
    }

    pub fn is_dll(&self) -> bool {
        self.characteristics & 0x2000 != 0
    }

    /// File offset of data directory `index`'s `(rva, size)` pair.
    pub fn dir_offset(&self, index: usize) -> usize {
        self.optional_header + 112 + index * 8
    }

    /// Every `IMAGE_REL_BASED_DIR64` relocation: (file offset of its 16-bit
    /// entry, RVA it patches).
    pub fn dir64_relocations(&self) -> Vec<(usize, u32)> {
        let mut out = Vec::new();
        let (rva, size) = self.dirs[dir::RELOC];
        let Ok(start) = self.offset(rva) else { return out };
        let (mut at, end) = (start, start + size as usize);
        while at + 8 <= end {
            let (Ok(page), Ok(block)) = (u32_at(self.data, at), u32_at(self.data, at + 4)) else { break };
            let block = block as usize;
            if block < 8 || at + block > end {
                break;
            }
            for entry in (at + 8..at + block).step_by(2) {
                let Ok(e) = u16_at(self.data, entry) else { break };
                if e >> 12 == 10 {
                    out.push((entry, page + u32::from(e & 0xFFF)));
                }
            }
            at += block;
        }
        out
    }

    /// Number of `RUNTIME_FUNCTION` entries (functions with unwind data).
    pub fn function_count(&self) -> usize {
        self.dirs[dir::EXCEPTION].1 as usize / 12
    }
}
