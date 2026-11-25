//! A captured frame outside the title. The capture reader (`load`:
//! `draws.tsv` probe frames and `list.tsv` live lists), the geometry
//! decoders (`decode`), world-space helpers and the OBJ writer (`world`)
//! and the replay driver of our world pass (`replay`: `xnw world-render`,
//! `xnw world-probe-decode`). Moved verbatim out of `apps/xnw/src/worldprobe.rs`
//! (59afddb) so a small binary can use them without the hook, the loader
//! or the HLE crates; the capture side (the hook and the writers) stays there.

pub mod audit;
pub mod bridge;
pub mod bridge_ps;
pub mod decode;
pub mod hair;
pub mod imposter;
pub mod load;
pub mod pack;
pub mod preskin;
pub mod preskin_gpu;
pub mod replay;
pub mod world;

pub use decode::*;
pub use load::*;
pub use replay::*;
pub use world::*;

pub(crate) fn le32(b: &[u8], at: usize) -> u32 {
    b.get(at..at + 4).map_or(0, |s| u32::from_le_bytes(s.try_into().unwrap()))
}
