//! Opus (RFC 6716), the codec of the console's audio decoder (`xapu.dll`):
//! the `opus-pure` crate (BSD-3-Clause, pure Rust, no dependencies), the
//! decoder VIctory's `apu.rs` uses. Re-exported here so the codec stays in
//! the formats crate.

pub use opus_pure::{Application, OpusDecoder, OpusEncoder};
