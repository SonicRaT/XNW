//! Memory system calls on the runtime's paths, counted and timed
//! (calls, microseconds) for the frame preparing line. A per-call
//! `VirtualQuery` has cost ~4 ms each in a 2.2 GiB heap
//! in other runtimes: every such call of ours is counted
//! here so a run shows what they cost a frame. Always on: two relaxed adds.

use std::sync::atomic::{AtomicU64, Ordering};

/// Where a call is made.
#[derive(Clone, Copy)]
pub enum Call {
    /// `VirtualQuery`: the write watch learning whether memory is a view.
    QueryView,
    /// `VirtualQuery`: the write-fault handler.
    QueryFault,
    /// `VirtualQuery`: a title memory copy step checking both ranges.
    QueryCopy,
    /// `VirtualQuery`: a bindless slot's memory check (cache missed).
    QuerySlot,
    /// `VirtualQuery`: xmem counting committed bytes around a title
    /// `VirtualAlloc`/`VirtualFree`.
    QueryXmem,
    /// `VirtualQuery`: the title's own `XMemVirtualQuery`.
    QueryTitle,
    /// `GetWriteWatch` (with reset).
    WriteWatch,
    /// `VirtualProtect`: guarding blocks again (read-only).
    ProtectGuard,
    /// `VirtualProtect`: opening a guarded block at a write fault.
    ProtectOpen,
    /// `VirtualProtect`: access traps (roles).
    ProtectTrap,
    /// `ReadProcessMemory`: copies of title memory that must not fault.
    ReadCopy,
    /// `ReadProcessMemory`: the app's guarded reads of title objects
    /// (`watch::guarded_read`: the hook's buffer objects and palettes, the
    /// texture table, the bridge taps, the geometry reads without
    /// `geomcopy=direct`).
    ReadProbe,
}

const NAMES: [&str; 12] = [
    "VirtualQuery view check",
    "VirtualQuery fault",
    "VirtualQuery copy step",
    "VirtualQuery slot memory",
    "VirtualQuery xmem commit count",
    "VirtualQuery title's",
    "GetWriteWatch",
    "VirtualProtect guard",
    "VirtualProtect open",
    "VirtualProtect trap",
    "ReadProcessMemory copy",
    "ReadProcessMemory probe",
];

static COUNTS: [[AtomicU64; 2]; NAMES.len()] = [const { [const { AtomicU64::new(0) }; 2] }; NAMES.len()];

/// Counts one call at `call` begun at `since`.
pub fn count(call: Call, since: std::time::Instant) {
    let c = &COUNTS[call as usize];
    c[0].fetch_add(1, Ordering::Relaxed);
    c[1].fetch_add(since.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// Runs `f`, counted at `call`.
pub fn timed<R>(call: Call, f: impl FnOnce() -> R) -> R {
    let at = std::time::Instant::now();
    let r = f();
    count(call, at);
    r
}

/// Per frame over `n` frames, zeroing the counts: "name calls/us, ...".
pub fn text(n: u64) -> String {
    let n = n.max(1);
    NAMES
        .iter()
        .zip(&COUNTS)
        .map(|(name, c)| format!("{name} {}/{} us", c[0].swap(0, Ordering::Relaxed) / n, c[1].swap(0, Ordering::Relaxed) / n))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Windows 10 refuses upload and readback buffers
/// made in the common state, and a title then reads a null object: host
/// resource creations that failed, by kind, for the memory line; the first
/// 8 of each kind are logged with their error.
#[derive(Clone, Copy)]
pub enum Made {
    /// Upload buffers (GENERIC_READ).
    Upload,
    /// CPU-mapped custom-heap buffers (host copies of title memory).
    Shared,
    /// Other host buffers (readback, default).
    Buffer,
    /// Committed host textures.
    Texture,
    /// Host textures placed in our pool heaps.
    Pooled,
    /// Heaps opened over title memory and their buffers.
    Opened,
}

const MADE_NAMES: [&str; 6] = ["upload", "shared", "buffer", "texture", "pooled texture", "opened heap"];
static FAILED: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// A host creation of `kind` failed with `error` (`what` describes it).
pub fn creation_failed(kind: Made, error: &dyn std::fmt::Display, what: &dyn std::fmt::Display) {
    let n = FAILED[kind as usize].fetch_add(1, Ordering::Relaxed);
    if n < 8 {
        crate::warn!("host {} creation failed ({} so far): {error}; {what}", MADE_NAMES[kind as usize], n + 1);
    }
}

/// "kind n, ..." of host creations failed so far.
pub fn failed_text() -> String {
    MADE_NAMES.iter().zip(&FAILED).map(|(k, c)| format!("{k} {}", c.load(Ordering::Relaxed))).collect::<Vec<_>>().join(", ")
}