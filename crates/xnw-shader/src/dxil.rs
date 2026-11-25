// Ported from VIctory crates/victory-shader/src/dxil.rs, commit d2a4bae1f49533ef5cb512fd1b1ba1513f2e1989, GPL-3.0-or-later.
// Changed: accepts the stripped form (`NOBC` in place of the bitcode).
//! The DXIL program inside a container's `DXIL` (or `STAT`) part.
//!
//! ```text
//! 0x00  program version u32   (kind << 16 | major << 4 | minor)
//! 0x04  size in dwords u32
//! 0x08  "DXIL"                <- bitcode offset is relative to here
//! 0x0C  dxil version u32
//! 0x10  bitcode offset u32    (16 in standard PC DXIL)
//! 0x14  bitcode size u32
//! ```
//!
//! The title ships without bitcode: in its place are the 4 bytes `NOBC`
//! ([`DxilProgram::stripped`]); stage and shader model are still in the
//! header, and the GFX10 code is in the `XDAT` blobs ([`crate::native`]).

use crate::ShaderError;
use crate::dxbc::u32_at;

const LLVM_BITCODE_MAGIC: [u8; 4] = [0x42, 0x43, 0xC0, 0xDE];
const NO_BITCODE: [u8; 4] = *b"NOBC";
#[cfg(test)]
const STANDARD_BITCODE_OFFSET: u32 = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderKind {
    Pixel,
    Vertex,
    Geometry,
    Hull,
    Domain,
    Compute,
    Library,
    RayGeneration,
    Intersection,
    AnyHit,
    ClosestHit,
    Miss,
    Callable,
    Mesh,
    Amplification,
    Node,
    Unknown(u16),
}

impl ShaderKind {
    fn from_raw(v: u16) -> Self {
        use ShaderKind::*;
        match v {
            0 => Pixel,
            1 => Vertex,
            2 => Geometry,
            3 => Hull,
            4 => Domain,
            5 => Compute,
            6 => Library,
            7 => RayGeneration,
            8 => Intersection,
            9 => AnyHit,
            10 => ClosestHit,
            11 => Miss,
            12 => Callable,
            13 => Mesh,
            14 => Amplification,
            15 => Node,
            other => Unknown(other),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DxilProgram<'a> {
    pub kind: ShaderKind,
    pub shader_model: (u8, u8),
    pub dxil_version: u32,
    /// LLVM bitcode (starts with `BC C0 DE`); empty when stripped.
    pub bitcode: &'a [u8],
    /// The bitcode was replaced by `NOBC`.
    pub stripped: bool,
    /// Bytes between the bitcode header and the bitcode. Empty on PC.
    pub prefix: &'a [u8],
    /// Bytes after the bitcode, up to the program size. Empty on PC.
    pub trailer: &'a [u8],
}

/// Xbox additions recognised around the bitcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct XboxExtras {
    /// The prefix starts with an `XDAT` table.
    pub xdat: bool,
    /// The trailer contains an `xbop` blob (compiler options).
    pub xbop: bool,
}

impl<'a> DxilProgram<'a> {
    pub fn parse(part: &'a [u8]) -> Result<Self, ShaderError> {
        let version = u32_at(part, 0)?;
        let size = (u32_at(part, 4)? as usize)
            .checked_mul(4)
            .filter(|&s| s <= part.len() && s >= 24)
            .ok_or(ShaderError::Malformed(
                "program size disagrees with part size",
            ))?;
        let part = &part[..size];
        if part.get(8..12) != Some(b"DXIL") {
            return Err(ShaderError::Malformed("missing DXIL magic"));
        }
        let dxil_version = u32_at(part, 12)?;
        let offset = u32_at(part, 16)?;
        let len = u32_at(part, 20)? as usize;
        let start = 8 + offset as usize;
        let bitcode = part
            .get(start..start + len)
            .ok_or(ShaderError::Malformed("bitcode out of bounds"))?;
        let stripped = bitcode == NO_BITCODE;
        if !stripped && bitcode.get(0..4) != Some(&LLVM_BITCODE_MAGIC) {
            return Err(ShaderError::Malformed("bitcode lacks LLVM magic"));
        }
        let prefix = part
            .get(24..start)
            .ok_or(ShaderError::Malformed("bitcode offset too small"))?;
        Ok(DxilProgram {
            kind: ShaderKind::from_raw((version >> 16) as u16),
            shader_model: (((version >> 4) & 0xF) as u8, (version & 0xF) as u8),
            dxil_version,
            bitcode: if stripped { &[] } else { bitcode },
            stripped,
            prefix,
            trailer: &part[start + len..],
        })
    }

    /// True for a program laid out exactly like PC DXIL.
    pub fn is_standard_layout(&self) -> bool {
        self.prefix.is_empty() && self.trailer.is_empty()
    }

    pub fn xbox_extras(&self) -> XboxExtras {
        XboxExtras {
            xdat: self.prefix.starts_with(b"XDAT"),
            xbop: self.trailer.windows(4).any(|w| w == b"xbop"),
        }
    }
}

/// Builds a standard PC-layout program around `bitcode` (test helper).
#[cfg(test)]
pub fn build_standard(kind: u16, sm: (u8, u8), dxil_version: u32, bitcode: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let version = (u32::from(kind) << 16) | (u32::from(sm.0) << 4) | u32::from(sm.1);
    let padded = bitcode.len().next_multiple_of(4);
    let size_dwords = (24 + padded) / 4;
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&(size_dwords as u32).to_le_bytes());
    out.extend_from_slice(b"DXIL");
    out.extend_from_slice(&dxil_version.to_le_bytes());
    out.extend_from_slice(&STANDARD_BITCODE_OFFSET.to_le_bytes());
    out.extend_from_slice(&(bitcode.len() as u32).to_le_bytes());
    out.extend_from_slice(bitcode);
    out.resize(24 + padded, 0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_bitcode() -> Vec<u8> {
        let mut b = LLVM_BITCODE_MAGIC.to_vec();
        b.extend_from_slice(&[0xAA; 12]);
        b
    }

    #[test]
    fn parses_standard_layout() {
        let bytes = build_standard(0, (6, 0), 0x100, &fake_bitcode());
        let p = DxilProgram::parse(&bytes).unwrap();
        assert_eq!(p.kind, ShaderKind::Pixel);
        assert_eq!(p.shader_model, (6, 0));
        assert!(p.is_standard_layout());
        assert_eq!(p.xbox_extras(), XboxExtras::default());
    }

    #[test]
    fn detects_xbox_layout() {
        // Xbox layout: XDAT table before the bitcode, xbop blob after it.
        let bitcode = fake_bitcode();
        let xdat = b"XDAT\x01\x00\x00\x00";
        let trailer = b"xbop\x02\x00\x00\x00";
        let mut bytes = Vec::new();
        let size = 24 + xdat.len() + bitcode.len() + trailer.len();
        bytes.extend_from_slice(&((5u32 << 16) | 0x65).to_le_bytes()); // cs_6_5
        bytes.extend_from_slice(&((size / 4) as u32).to_le_bytes());
        bytes.extend_from_slice(b"DXIL");
        bytes.extend_from_slice(&0x105u32.to_le_bytes());
        bytes.extend_from_slice(&(16 + xdat.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(bitcode.len() as u32).to_le_bytes());
        bytes.extend_from_slice(xdat);
        bytes.extend_from_slice(&bitcode);
        bytes.extend_from_slice(trailer);

        let p = DxilProgram::parse(&bytes).unwrap();
        assert_eq!(p.kind, ShaderKind::Compute);
        assert_eq!(p.shader_model, (6, 5));
        assert!(!p.is_standard_layout());
        assert_eq!(
            p.xbox_extras(),
            XboxExtras {
                xdat: true,
                xbop: true
            }
        );
        assert_eq!(p.bitcode, &bitcode[..]);
    }

    #[test]
    fn accepts_stripped_bitcode() {
        let bytes = build_standard(13, (6, 5), 0x106, b"NOBC");
        let p = DxilProgram::parse(&bytes).unwrap();
        assert_eq!((p.kind, p.shader_model, p.stripped), (ShaderKind::Mesh, (6, 5), true));
        assert!(p.bitcode.is_empty());
    }

    #[test]
    fn rejects_missing_bitcode_magic() {
        let bytes = build_standard(0, (6, 0), 0x100, &[0u8; 16]);
        assert!(DxilProgram::parse(&bytes).is_err());
    }
}
