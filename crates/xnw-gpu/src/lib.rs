//! Host GPU: the D3D12 device everything else draws on.

pub mod d3d12x;
pub use xnw_device as host;
pub mod mesh;
pub mod shared;

pub use host::{Caps, Host};
