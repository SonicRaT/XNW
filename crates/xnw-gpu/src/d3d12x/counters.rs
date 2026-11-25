//! Measurement only. Counters for the frame statistics lines
//! (`XNW_FRAME_STATS=1`, see `queue::frame_stats`): what the title calls per
//! frame, lock waits, the cost of each submission, fills by cause, and CPU
//! time per thread. Atomics on the hot paths; everything is swapped at print
//! time and printed as averages per frame over the window.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

// ---- title calls ---------------------------------------------------------

pub(crate) const CALL_NAMES: [&str; 22] = [
    "submissions",
    "lists",
    "draws direct",
    "draws indexed",
    "draws indirect",
    "draws mesh",
    "draws rectangle",
    "dispatches",
    "copies buffer",
    "copies texture",
    "copies resource",
    "copies resolve",
    "copies memory",
    "clears target",
    "clears depth",
    "barriers",
    "pipeline binds",
    "root constants sets",
    "root view sets",
    "descriptor table sets",
    "heap sets",
    "target sets",
];
pub(crate) const SUBMISSIONS: usize = 0;
pub(crate) const LISTS: usize = 1;
pub(crate) const DRAW: usize = 2;
pub(crate) const DRAW_INDEXED: usize = 3;
pub(crate) const DRAW_INDIRECT: usize = 4;
pub(crate) const DRAW_MESH: usize = 5;
/// Rectangle-list draws, counted again among direct and indexed.
pub(crate) const DRAW_RECT: usize = 6;
pub(crate) const DISPATCH: usize = 7;
pub(crate) const COPY_BUFFER: usize = 8;
pub(crate) const COPY_TEXTURE: usize = 9;
pub(crate) const COPY_RESOURCE: usize = 10;
pub(crate) const COPY_RESOLVE: usize = 11;
pub(crate) const COPY_MEMORY: usize = 12;
pub(crate) const CLEAR_TARGET: usize = 13;
pub(crate) const CLEAR_DEPTH: usize = 14;
pub(crate) const BARRIER: usize = 15;
pub(crate) const PIPELINE: usize = 16;
pub(crate) const ROOT_CONSTANTS: usize = 17;
pub(crate) const ROOT_VIEW: usize = 18;
pub(crate) const TABLE: usize = 19;
pub(crate) const HEAPS: usize = 20;
pub(crate) const TARGETS: usize = 21;

static CALLS: [AtomicU64; CALL_NAMES.len()] = [const { AtomicU64::new(0) }; CALL_NAMES.len()];

/// One title call of kind `kind`; with `XNW_FRAME_STATS=1` the
/// returned guard times it to the end of the caller's scope (inclusive of
/// calls it makes; the record-time cost of each method group).
#[inline]
pub(crate) fn call(kind: usize) -> CallTimer {
    CALLS[kind].fetch_add(1, Relaxed);
    CallTimer(kind, enabled().then(std::time::Instant::now))
}

/// See [`call`].
pub(crate) struct CallTimer(usize, Option<std::time::Instant>);
impl Drop for CallTimer {
    fn drop(&mut self) {
        if let Some(at) = self.1 {
            CALL_US[self.0].fetch_add(at.elapsed().as_micros() as u64, Relaxed);
        }
    }
}
static CALL_US: [AtomicU64; CALL_NAMES.len()] = [const { AtomicU64::new(0) }; CALL_NAMES.len()];

/// Record-time microseconds by call kind (inclusive), the busiest
/// first, since the last call (per frame over `n`).
pub(crate) fn call_time_text(n: u64) -> String {
    let mut v: Vec<(u64, &str)> = CALL_US.iter().zip(CALL_NAMES).map(|(c, name)| (c.swap(0, Relaxed) / n.max(1), name)).filter(|(us, _)| *us > 0).collect();
    v.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    if v.is_empty() {
        return "none timed (XNW_FRAME_STATS=1)".into();
    }
    v.iter().take(10).map(|(us, name)| format!("{name} {us} us")).collect::<Vec<_>>().join(", ")
}

/// A submission of `lists` lists.
pub(crate) fn submitted(lists: usize) {
    CALLS[SUBMISSIONS].fetch_add(1, Relaxed);
    CALLS[LISTS].fetch_add(lists as u64, Relaxed);
}

/// Global UAV barriers we add: after ops (`XNW_UAV_BARRIER`), and for
/// the title's pipeline syncs.
pub(crate) static UAV_ADDED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
pub(crate) const UAV_AFTER_OPS: usize = 0;
pub(crate) const UAV_TITLE_SYNCS: usize = 1;

/// Per frame over `n` frames, rounded, zeroing the counts.
fn per_frame(c: &AtomicU64, n: u64) -> u64 {
    (c.swap(0, Relaxed) + n / 2) / n
}

fn calls_text(n: u64) -> String {
    let v: Vec<u64> = CALLS.iter().map(|c| per_frame(c, n)).collect();
    let draws = v[DRAW] + v[DRAW_INDEXED] + v[DRAW_INDIRECT] + v[DRAW_MESH];
    format!(
        "submissions {}, lists {}, draws {draws} (direct {}, indexed {}, indirect {}, mesh {}, rectangle {} of those), dispatches {}, copies {} (buffer {}, texture {}, resource {}, resolve {}, memory {}), clears {} (target {}, depth {}), barriers {}, pipeline binds {}, root-argument sets {} (constants {}, views {}), descriptor-table sets {} (heaps {}, targets {}); UAV barriers added {} (after ops, XNW_UAV_BARRIER={}) and {} (title pipeline syncs); list pads written {}; list memory waits met {}, waited {}, given up {}, forced in skipped frames {}, polled 100 ms or more {}; {}; {}; {}; {}",
        v[SUBMISSIONS],
        v[LISTS],
        v[DRAW],
        v[DRAW_INDEXED],
        v[DRAW_INDIRECT],
        v[DRAW_MESH],
        v[DRAW_RECT],
        v[DISPATCH],
        v[COPY_BUFFER] + v[COPY_TEXTURE] + v[COPY_RESOURCE] + v[COPY_RESOLVE] + v[COPY_MEMORY],
        v[COPY_BUFFER],
        v[COPY_TEXTURE],
        v[COPY_RESOURCE],
        v[COPY_RESOLVE],
        v[COPY_MEMORY],
        v[CLEAR_TARGET] + v[CLEAR_DEPTH],
        v[CLEAR_TARGET],
        v[CLEAR_DEPTH],
        v[BARRIER],
        v[PIPELINE],
        v[ROOT_CONSTANTS] + v[ROOT_VIEW],
        v[ROOT_CONSTANTS],
        v[ROOT_VIEW],
        v[TABLE],
        v[HEAPS],
        v[TARGETS],
        per_frame(&UAV_ADDED[UAV_AFTER_OPS], n),
        super::crumbs::uav_mode_name(),
        per_frame(&UAV_ADDED[UAV_TITLE_SYNCS], n),
        (super::list::pad_written() + n / 2) / n,
        per_frame(&super::queue::WAITS[0], n),
        per_frame(&super::queue::WAITS[1], n),
        per_frame(&super::queue::WAITS[2], n),
        per_frame(&super::queue::WAITS[3], n),
        per_frame(&super::queue::WAITS[4], n),
        super::draw::mesh_indirect_text(n),
        super::draw::root_views_text(n),
        super::draw::topology_text(n),
        super::draw::clamp_text(n),
    )
}

// ---- lock waits ----------------------------------------------------------

pub(crate) const LOCK_SUBMITTED: usize = 0;
pub(crate) const LOCK_PLACED: usize = 1;
const LOCK_NAMES: [&str; 2] = ["submitted", "placed textures"];
/// Per lock: microseconds waited, waits that blocked, longest wait (us).
static WAITS: [[AtomicU64; 3]; 2] = [const { [const { AtomicU64::new(0) }; 3] }; 2];

/// Locks `m`, noting the time spent waiting for it when it was held.
pub(crate) fn timed<T>(lock: usize, m: &Mutex<T>) -> MutexGuard<'_, T> {
    if let Ok(g) = m.try_lock() {
        return g;
    }
    let at = Instant::now();
    let g = m.lock().unwrap_or_else(|e| e.into_inner());
    let us = at.elapsed().as_micros() as u64;
    let w = &WAITS[lock];
    w[0].fetch_add(us, Relaxed);
    w[1].fetch_add(1, Relaxed);
    w[2].fetch_max(us, Relaxed);
    g
}

fn locks_text(n: u64) -> String {
    let parts: Vec<String> = LOCK_NAMES
        .iter()
        .zip(&WAITS)
        .map(|(name, [us, blocked, max])| format!("{name} waited {} us in {} blocked acquisitions, longest {} us", us.swap(0, Relaxed) / n, per_frame(blocked, n), max.swap(0, Relaxed)))
        .collect();
    parts.join("; ")
}

// ---- submissions ---------------------------------------------------------

/// Bytes snapshotted (`Shadow::writing`) so far.
static SNAPSHOT: AtomicU64 = AtomicU64::new(0);
/// Count, preparing microseconds (sum, max), snapshot bytes (sum, max).
static SUBMITTED: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// A snapshot of `bytes` was made or grown.
pub(crate) fn snapshot(bytes: usize) {
    SNAPSHOT.fetch_add(bytes as u64, Relaxed);
}

/// The snapshot bytes so far, to take a submission's share.
pub(crate) fn snapshot_total() -> u64 {
    SNAPSHOT.load(Relaxed)
}

/// One submission prepared in `us` microseconds, with `snapshot_before` the
/// [`snapshot_total`] at its start.
pub(crate) fn submission(us: u64, snapshot_before: u64) {
    let bytes = snapshot_total().saturating_sub(snapshot_before);
    SUBMITTED[0].fetch_add(1, Relaxed);
    SUBMITTED[1].fetch_add(us, Relaxed);
    SUBMITTED[2].fetch_max(us, Relaxed);
    SUBMITTED[3].fetch_add(bytes, Relaxed);
    SUBMITTED[4].fetch_max(bytes, Relaxed);
}

fn submissions_text(n: u64) -> String {
    let [count, sum, max, bytes, bytes_max] = &SUBMITTED;
    let count_n = count.swap(0, Relaxed);
    let c = count_n.max(1);
    format!(
        "{} per frame, preparing {} us average and {} us at most per submission, snapshot {} KiB average and {} KiB at most per submission ({} KiB per frame)",
        count_n.div_ceil(n),
        sum.swap(0, Relaxed) / c,
        max.swap(0, Relaxed),
        bytes.load(Relaxed) / c / 1024,
        bytes_max.swap(0, Relaxed) / 1024,
        bytes.swap(0, Relaxed) / n / 1024
    )
}

/// The copy-back part of the `frame preparing` line: per frame time and
/// bytes written back into title memory after submissions.
pub(crate) fn copy_back_text(n: u64) -> String {
    let c = &super::host::COPY_BACK;
    format!("; copy-back {} us, {} KiB written back", c[0].swap(0, Relaxed) / n, c[1].swap(0, Relaxed) / n / 1024)
}

// ---- fills ---------------------------------------------------------------

pub(crate) const FILL_NAMES: [&str; 4] = ["misaligned constant table", "not resolved", "crosses a window", "other"];
static FILLS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Whether `XNW_FRAME_STATS=1`.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_FRAME_STATS").is_ok_and(|v| v == "1"));
    *ON
}

/// A fill of `len` bytes from title memory at `src` was made: counted by
/// why it was not read in place.
pub(crate) fn fill(src: usize, len: usize) {
    if enabled() {
        FILLS[crate::shared::fill_cause(src, len)].fetch_add(1, Relaxed);
    }
}

fn fills_text(n: u64) -> String {
    let v: Vec<String> = FILL_NAMES.iter().zip(&FILLS).map(|(name, c)| format!("{name} {}", per_frame(c, n))).collect();
    v.join(", ")
}

// ---- threads -------------------------------------------------------------

/// CPU time (100 ns units) of each of the process's threads at the last line.
static THREADS: Mutex<Option<HashMap<u32, u64>>> = Mutex::new(None);

/// The top six threads by CPU time since the last call, `id start ms` per
/// frame; the first call only takes the baseline.
fn threads_text(n: u64) -> String {
    use xnw_core::win::*;
    let mut last = THREADS.lock().unwrap_or_else(|e| e.into_inner());
    let mut now = HashMap::new();
    let mut used: Vec<(u64, u32, usize)> = Vec::new();
    let mut names: HashMap<u32, String> = HashMap::new();
    // SAFETY: no arguments.
    let me = unsafe { GetCurrentProcessId() };
    // SAFETY: a thread snapshot walked with a sized entry; each thread is
    // opened for queries only and closed; out-parameters sized as declared.
    unsafe {
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snapshot == INVALID_HANDLE_VALUE {
            return "no thread snapshot".into();
        }
        let mut entry = ThreadEntry32 { size: size_of::<ThreadEntry32>() as u32, usage: 0, thread_id: 0, owner_process_id: 0, base_priority: 0, delta_priority: 0, flags: 0 };
        let mut more = Thread32First(snapshot, &mut entry) != 0;
        while more {
            if entry.owner_process_id == me {
                let id = entry.thread_id;
                let thread = OpenThread(THREAD_QUERY_INFORMATION, 0, id);
                if !thread.is_null() {
                    let (mut c, mut e, mut k, mut u) = (0, 0, 0, 0);
                    GetThreadTimes(thread, &mut c, &mut e, &mut k, &mut u);
                    now.insert(id, k + u);
                    let delta = (k + u).saturating_sub(last.as_ref().and_then(|l| l.get(&id)).copied().unwrap_or(0));
                    if last.is_some() && delta > 0 {
                        let mut start = 0usize;
                        NtQueryInformationThread(thread, THREAD_QUERY_SET_WIN32_START_ADDRESS, (&raw mut start).cast(), 8, std::ptr::null_mut());
                        used.push((delta, id, start));
                        names.insert(id, thread_name(thread));
                    }
                    CloseHandle(thread);
                }
            }
            more = Thread32Next(snapshot, &mut entry) != 0;
        }
        CloseHandle(snapshot);
    }
    let first = last.is_none();
    *last = Some(now);
    if first {
        return "baseline taken".into();
    }
    used.sort_by(|a, b| b.0.cmp(&a.0));
    let base = super::object::title_base();
    // Every thread over 1 ms a frame, named: ours by their names, the
    // title's by the context their title TLS binds.
    let parts: Vec<String> = used
        .iter()
        .filter(|&&(delta, ..)| delta as f64 / 1e4 / n as f64 >= 1.0)
        .map(|&(delta, id, start)| {
            let at = if base != 0 && start > base { format!("title+{:#x}", start - base) } else { format!("{start:#x}") };
            let name = names.get(&id).filter(|s| !s.is_empty()).cloned().unwrap_or_else(|| title_role(id));
            format!("{id} {name} ({at}) {:.2} ms", delta as f64 / 1e4 / n as f64)
        })
        .collect();
    parts.join("; ")
}

/// CPU milliseconds since the last call, summed over title threads
/// (no description) and over ours (named), for the null line.
pub(crate) fn cpu_split() -> (f64, f64) {
    use xnw_core::win::*;
    static LAST: Mutex<Option<HashMap<u32, (u64, bool)>>> = Mutex::new(None);
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let mut now = HashMap::new();
    let (mut title, mut ours) = (0u64, 0u64);
    // SAFETY: as `threads_text`.
    unsafe {
        let me = GetCurrentProcessId();
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snapshot == INVALID_HANDLE_VALUE {
            return (0.0, 0.0);
        }
        let mut entry = ThreadEntry32 { size: size_of::<ThreadEntry32>() as u32, usage: 0, thread_id: 0, owner_process_id: 0, base_priority: 0, delta_priority: 0, flags: 0 };
        let mut more = Thread32First(snapshot, &mut entry) != 0;
        while more {
            if entry.owner_process_id == me {
                let id = entry.thread_id;
                let thread = OpenThread(THREAD_QUERY_INFORMATION, 0, id);
                if !thread.is_null() {
                    let (mut c, mut e, mut k, mut u) = (0, 0, 0, 0);
                    GetThreadTimes(thread, &mut c, &mut e, &mut k, &mut u);
                    let named = last.as_ref().and_then(|l| l.get(&id)).map_or_else(|| { let n = thread_name(thread); !n.is_empty() && !n.starts_with("title ") }, |&(_, n)| n);
                    let delta = (k + u).saturating_sub(last.as_ref().and_then(|l| l.get(&id)).map_or(k + u, |&(t, _)| t));
                    if named { ours += delta } else { title += delta }
                    now.insert(id, (k + u, named));
                    CloseHandle(thread);
                }
            }
            more = Thread32Next(snapshot, &mut entry) != 0;
        }
        CloseHandle(snapshot);
    }
    *last = Some(now);
    (title as f64 / 1e4, ours as f64 / 1e4)
}

/// A thread's description (Rust thread names are set as one).
fn thread_name(thread: xnw_core::win::HANDLE) -> String {
    use xnw_core::win::*;
    let mut p: *mut u16 = std::ptr::null_mut();
    // SAFETY: a query of an open thread; the string is the system's, freed here.
    unsafe {
        if GetThreadDescription(thread, &mut p) < 0 || p.is_null() {
            return String::new();
        }
        let mut n = 0;
        while *p.add(n) != 0 && n < 256 {
            n += 1;
        }
        let s = String::from_utf16_lossy(std::slice::from_raw_parts(p, n));
        LocalFree(p.cast());
        s
    }
}

/// What a title thread is, by the recording context its title TLS
/// binds at +0x12A8: one of the five sub-render threads' (slot table
/// 0x149C3D670, stride 64, context at +0x28), another context (the render
/// thread, or a drain), or none (a job or other thread).
fn title_role(id: u32) -> String {
    use xnw_core::win::*;
    let base = super::object::title_base();
    let index = xnw_core::TITLE_TLS_INDEX.load(Relaxed) as usize;
    if base == 0 {
        return "unnamed".into();
    }
    #[repr(C)]
    struct Basic {
        exit: i32,
        teb: usize,
        client: [usize; 2],
        affinity: usize,
        priority: i32,
        base_priority: i32,
    }
    let read = |at: usize| -> Option<usize> {
        let mut v = [0u8; 8];
        super::host::read_into(at as *const u8, &mut v).then(|| usize::from_le_bytes(v))
    };
    // SAFETY: a query of a thread opened for it, into a sized local.
    let teb = unsafe {
        let thread = OpenThread(THREAD_QUERY_INFORMATION, 0, id);
        if thread.is_null() {
            return "unnamed".into();
        }
        let mut b: Basic = std::mem::zeroed();
        let ok = NtQueryInformationThread(thread, 0, (&raw mut b).cast(), size_of::<Basic>() as u32, std::ptr::null_mut()) >= 0;
        CloseHandle(thread);
        if !ok {
            return "unnamed".into();
        }
        b.teb
    };
    let Some(ctx) = read(teb + 0x58).and_then(|tls| read(tls + 8 * index)).and_then(|block| read(block + 0x12a8)) else {
        return "title (no TLS)".into();
    };
    if ctx == 0 {
        return "title job or other (no recording context)".into();
    }
    let table = base + (0x149c3_d670 - 0x1_4000_0000);
    for i in 0..5 {
        if read(table + i * 64 + 0x28) == Some(ctx) {
            return format!("title Subrender:{i:02}");
        }
    }
    format!("title render thread or drain (context {ctx:#x})")
}

// ---- printing ------------------------------------------------------------

/// The new lines of the frame statistics, over `n` frames to `frame`.
pub(crate) fn lines(frame: u64, n: u64) {
    xnw_core::info!("frame calls to {frame} (per frame): {}", calls_text(n));
    xnw_core::info!("frame record time to {frame} (per frame, by call, inclusive): {}", call_time_text(n));
    xnw_core::info!("frame locks to {frame} (per frame): {}", locks_text(n));
    xnw_core::info!("frame submissions to {frame}: {}", submissions_text(n));
    xnw_core::info!("frame fills to {frame} (per frame, by cause): {}", fills_text(n));
    xnw_core::info!("frame in-place misses to {frame} (per frame): {}", crate::shared::view_misses_text(n));
    xnw_core::info!("frame threads to {frame} (CPU per frame, top 6: id start): {}", threads_text(n));
}
