//! Runs a title's x86-64 code natively inside our process.
//!
//! Design from VIctory's `victory-loader` (GPL-3.0-or-later), re-implemented
//! without dependencies: [`prepare`] a copy of each image that Windows maps
//! but does not run, load it as a real module, bind every import ourselves
//! (with a naming stub for whatever nobody provides and an optional trace
//! thunk for the rest), then run TLS callbacks and the entry point. Every
//! attached image hears of each thread start and end and of process exit
//! ([`threads`]); Zen 2 instructions this CPU lacks are done in the fault
//! handler ([`amd`]).

mod amd;
pub mod crash;
mod image;
pub mod prepare;
mod record;
pub mod rewrite;
mod stubs;

mod threads;
mod tls;

pub use image::{BindReport, Loaded, Provider, Resolver, host_export, load};
pub use stubs::{Site, Unimplemented, set_policy};
pub use threads::disable_thread_calls;
