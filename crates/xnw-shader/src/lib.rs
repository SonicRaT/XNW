//! Shader front end: the title's shader pack, read down to GFX10 instructions.
//!
//! Layers, outermost first:
//!
//! - [`rpf9`]: the `RPF9` archive. Its table of contents is encrypted, so the
//!   entries are found by their payload signatures at page boundaries.
//! - [`sgd5`]: the engine's shader file; one record per shader, deflate-packed.
//! - [`dxbc`]: the standard DXBC container inside each record.
//! - [`dxil`]: the `DXIL` part (stage, shader model, Xbox additions).
//! - [`native`]: the `XDAT` index in that part and the zlib-packed GFX10
//!   programs it points at (`XBCH` / `FSSM`).
//! - [`gfx10`]: the GFX10 (RDNA2) instruction decoder.
//!
//! Translation, in order: [`ir`] (one-to-one SSA over the decoded code),
//! [`structure`] (EXEC removal, per-lane booleans), [`resource`]
//! (descriptors traced to root parameters), [`io`] (stage inputs and
//! exports) and [`emit`] (DXIL written directly, LLVM 3.7 bitcode).
//!
//! Ported from VIctory (`crates/victory-shader`, `crates/victory-game/src/sgd5.rs`,
//! commit d2a4bae1f49533ef5cb512fd1b1ba1513f2e1989, GPL-3.0-or-later); each
//! file names what it took. The RPF9 layout follows rpf9-research
//! (its format notes and `tools/rpf9.py`, commit 144d345191bd37de3f6bd5e85df3e8ae8815552e, GPL-3.0).

pub mod dxbc;
pub mod dxil;
pub mod emit;
pub mod gfx10;
pub mod io;
pub mod ir;
pub mod native;
pub mod psocache;
pub mod resource;
pub mod structure;
pub mod rpf9;
pub mod sgd5;

pub use dxbc::{Container, FourCC, Part};
pub use dxil::{DxilProgram, ShaderKind, XboxExtras};
pub use native::{
    HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource, native_programs, xdat_entries,
};

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderError {
    NotDxbc,
    Malformed(&'static str),
    /// Bad GFX10 code at a byte offset in the program.
    Isa { offset: usize, reason: &'static str },
    /// Valid code the translator cannot handle yet: an opcode name or a construct.
    Unsupported { offset: usize, what: &'static str },
}

impl fmt::Display for ShaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShaderError::NotDxbc => f.write_str("not a DXBC container"),
            ShaderError::Malformed(what) => write!(f, "malformed shader: {what}"),
            ShaderError::Isa { offset, reason } => write!(f, "GFX10 code at {offset:#x}: {reason}"),
            ShaderError::Unsupported { offset, what } => write!(f, "GFX10 code at {offset:#x}: {what} is not supported"),
        }
    }
}

impl std::error::Error for ShaderError {}
