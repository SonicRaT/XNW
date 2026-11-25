//! The submission core: one host queue that runs every title submission
//! in order and waits for it (VIctory's decision 0007), host lists the
//! title's lists record into, and the work each list does on the CPU when
//! it is submitted (title memory copies, buffer refreshes, bottom-of-pipe
//! writes) in recorded order.
//!
//! Every list is a host direct list and every title queue submits to the
//! one host queue, whatever its type: submissions are waited for, so the
//! title's cross-queue fences are already passed when it signals them.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap;
//! overlap them with per-submission fences once draws run.
//!
//! Ported from VIctory-base (VIctory main d2a4bae + 26 PRs, branch
//! `xnw/base` @ 5d0516e, GPL-3.0-or-later), `crates/victory-gpu/src/host.rs`
//! lines 1-188 (states, Host, retire, fail), 336-548 (HostList, steps,
//! deferred work), 550-674 (execute), 811-848 (render target and depth view
//! rings), 965-1349 (HostList methods), 1350-1615 (Batch, copy helpers,
//! upload buffers), 2225-2300 (legal states, shared buffers), 2390-2410
//! (transition) and tests 2475-2540; PRs touching that file: #135, #211,
//! #413, #581. Changed: the device is ours (`xnw-gpu/src/host.rs`, Agility);
//! device-removed data, the debug layer, GPU timing and copy statistics
//! (the base's `diagnostics` feature) are dropped; `windows-sys` is
//! `xnw_core::win`; `tracing` is `xnw_core::{info,warn}`.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;

use super::descriptor::host::{BindlessPlan, HostHeap};
use super::object::This;
use super::queue::Written;
use super::resource::HostTexture;
use super::shadow::Shadow;

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

/// The host side of submission: the device, the one queue, its fence.
pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    /// The one queue; the presenter's swap chain is made on it.
    pub(crate) queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    next_rtv: AtomicU32,
    /// The window's swap chain and compositor (`present.rs`, base 58).
    pub(crate) presenter: Mutex<Option<super::present::Presenter>>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

impl Drop for Host {
    fn drop(&mut self) {
        // SAFETY: our event.
        let _ = unsafe { CloseHandle(self.event) };
    }
}

/// A host resource waiting to be dropped after in-flight GPU work finishes.
struct Retired {
    frame: u64,
    _resource: ID3D12Resource,
}

// SAFETY: D3D12 resources are free-threaded and may be released on any thread.
unsafe impl Send for Retired {}

/// Host resources retired on worker threads (a descriptor slot overwritten,
/// a title resource released while a submission runs), kept for several
/// presented frames: a title frame spans many submissions and worker
/// threads record lists ahead of presentation.
static RETIRED: Mutex<Vec<Retired>> = Mutex::new(Vec::new());

/// Holds `resource` alive until several full frames have completed on the GPU.
pub(crate) fn retire(resource: ID3D12Resource) {
    let frame = super::queue::presented();
    RETIRED.lock().unwrap_or_else(|e| e.into_inner()).push(Retired { frame, _resource: resource });
}

/// Retired resources old enough to drop once the current submission
/// finishes; ages are in presented frames.
fn take_retired() -> Vec<Retired> {
    const KEEP_FRAMES: u64 = 4;
    let now = super::queue::presented();
    let mut guard = RETIRED.lock().unwrap_or_else(|e| e.into_inner());
    let mut ready = Vec::new();
    let mut i = 0;
    while i < guard.len() {
        if now.saturating_sub(guard[i].frame) >= KEEP_FRAMES {
            ready.push(guard.swap_remove(i));
        } else {
            i += 1;
        }
    }
    ready
}

/// Time spent in submissions since the last frame line, in microseconds:
/// preparing them on the CPU, then waiting for the GPU.
pub(crate) static SUBMISSION_US: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// After submissions: shader-written ranges copied back into title memory
/// (microseconds, bytes that differed).
pub(crate) static COPY_BACK: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Bindless heaps' slots made ready (microseconds): plans at submission,
/// placed textures as draws are recorded.
pub(crate) static BINDLESS_US: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed).
pub(crate) static PREPARING: [AtomicU64; 4] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

/// Per-stage time (microseconds) and count of host work, for the frame
/// statistics (`XNW_FRAME_STATS`): one row per `STAGE_*`, taken and zeroed
/// by each statistics line. Atomics only; a stage costs two clock reads.
pub(crate) static STAGES: [[AtomicU64; 2]; STAGE_NAMES.len()] = [const { [AtomicU64::new(0), AtomicU64::new(0)] }; STAGE_NAMES.len()];
pub(crate) const STAGE_NAMES: [&str; 16] = [
    "host texture made",
    "placed texture made",
    "staging made",
    "title copy",
    "texture read",
    "hash",
    "detile",
    "write watch",
    "bindless plan",
    "bindless record",
    "pipeline wait",
    "deferred work",
    "prelude",
    "execute",
    "retire",
    "upload buffer made",
];
pub(crate) const STAGE_HOST_TEXTURE: usize = 0;
pub(crate) const STAGE_PLACED: usize = 1;
pub(crate) const STAGE_STAGING: usize = 2;
pub(crate) const STAGE_COPY: usize = 3;
pub(crate) const STAGE_READ: usize = 4;
pub(crate) const STAGE_HASH: usize = 5;
pub(crate) const STAGE_DETILE: usize = 6;
pub(crate) const STAGE_WATCH: usize = 7;
pub(crate) const STAGE_PLAN: usize = 8;
pub(crate) const STAGE_RECORD: usize = 9;
pub(crate) const STAGE_PIPELINE_WAIT: usize = 10;
pub(crate) const STAGE_WORK: usize = 11;
pub(crate) const STAGE_PRELUDE: usize = 12;
pub(crate) const STAGE_EXECUTE: usize = 13;
pub(crate) const STAGE_RETIRE: usize = 14;
pub(crate) const STAGE_UPLOAD_MADE: usize = 15;

/// Preparing a submission by phase, microseconds and count, for the
/// frame statistics: the phases cover `execute` from taking the submission
/// to handing the lists to the queue, so they sum to the preparing time
/// (the stage line's stages are partly inside them, partly at recording).
pub(crate) static PHASES: [[AtomicU64; 2]; PHASE_NAMES.len()] = [const { [AtomicU64::new(0), AtomicU64::new(0)] }; PHASE_NAMES.len()];
pub(crate) const PHASE_NAMES: [&str; 13] = [
    "copy and fill steps",
    "refresh steps queued",
    "work steps",
    "gpu-write steps (flush, snapshot)",
    "texture steps",
    "prelude",
    "refresh flush",
    "bindless plan",
    "bindless constants",
    "bindless residency",
    "bindless reads",
    "bindless write ranges",
    "retired taken",
];
const PH_COPY: usize = 0;
const PH_REFRESH: usize = 1;
const PH_WORK: usize = 2;
const PH_GPU_WRITE: usize = 3;
const PH_TEXTURE: usize = 4;
const PH_PRELUDE: usize = 5;
const PH_FLUSH: usize = 6;
const PH_PLAN: usize = 7;
const PH_CONSTANTS: usize = 8;
const PH_RESIDENT: usize = 9;
const PH_READS: usize = 10;
const PH_WRITES: usize = 11;
const PH_RETIRED: usize = 12;

/// Adds the time since `started` to phase `i`; returns now, the next
/// phase's start.
fn phase(i: usize, started: std::time::Instant) -> std::time::Instant {
    let now = std::time::Instant::now();
    PHASES[i][0].fetch_add((now - started).as_micros() as u64, Ordering::Relaxed);
    PHASES[i][1].fetch_add(1, Ordering::Relaxed);
    now
}

/// The phase part of the frame statistics over `n` frames and `preparing`
/// microseconds per frame, zeroing the counts: each phase, and what none
/// covers.
pub(crate) fn phases_text(n: u64, preparing: u64) -> String {
    let mut sum = 0;
    let parts: Vec<String> = PHASE_NAMES
        .iter()
        .zip(&PHASES)
        .filter_map(|(name, [us, count])| {
            let (us, count) = (us.swap(0, Ordering::Relaxed) / n, count.swap(0, Ordering::Relaxed));
            sum += us;
            (count > 0).then(|| format!("{name} {us} us/{}", count.div_ceil(n)))
        })
        .collect();
    // Copy-back (after the GPU) is the other part of the line.
    format!(
        "{}; not in a phase {} us; memory system calls per frame (calls/us): {}; host resources made per frame {} in {} us, CPU-visible buffers refused (non-local budget) {}, host copies and buffers not made in null-GPU frames {}",
        parts.join(", "),
        preparing.saturating_sub(sum),
        xnw_core::syscalls::text(n),
        CREATION[0].swap(0, Ordering::Relaxed) / n,
        CREATION[1].swap(0, Ordering::Relaxed) / n,
        CREATION[2].swap(0, Ordering::Relaxed),
        super::shadow::NULL_REFUSED.swap(0, Ordering::Relaxed)
    )
}

/// The thread holding the submission lock (0: none), for stall dumps.
pub(crate) static SUBMIT_OWNER: AtomicU64 = AtomicU64::new(0);

/// Who holds our locks now, for a stall dump: the submission, the
/// write guards and traps.
pub fn lock_owners() -> String {
    let guards = super::watch::GUARDS_OWNER.load(Ordering::Relaxed);
    format!(
        "submission lock held by thread {}\nguard/trap lock held by thread {}",
        SUBMIT_OWNER.load(Ordering::Relaxed),
        guards
    )
}

/// Adds the time since `started` and one to stage `i`.
pub(crate) fn stage(i: usize, started: std::time::Instant) {
    STAGES[i][0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
    STAGES[i][1].fetch_add(1, Ordering::Relaxed);
}

/// Placed-texture cache events, for the frame statistics: lookups that
/// found one, textures made, of those made again after a release, and
/// released (overlap, cap, remap); bytes uploaded from staging buffers;
/// cold staging states dropped; upload buffers taken from the pool and
/// made for it.
pub(crate) static TEXTURE_EVENTS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
pub(crate) const EV_HIT: usize = 0;
pub(crate) const EV_MADE: usize = 1;
pub(crate) const EV_REMADE: usize = 2;
pub(crate) const EV_RELEASED: usize = 3;
pub(crate) const EV_UPLOADED: usize = 4;
pub(crate) const EV_STAGING_DROPPED: usize = 5;
pub(crate) const EV_UPLOAD_REUSED: usize = 6;
pub(crate) const EV_UPLOAD_MADE: usize = 7;

/// Title memory copies at submission by kind, for the frame statistics:
/// fills of list upload memory (and of those, fills merged into the one
/// before), the title's buffer copies, refreshes of ranges draws read
/// (tables, roots, vertex and index buffers), bindless plan reads, bounce
/// copies of misaligned bindless constant buffers (and those skipped,
/// unwritten since).
pub(crate) static COPY_KINDS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
/// Bytes those copies moved, by the same kinds.
pub(crate) static COPY_BYTES: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
pub(crate) const COPY_FILL: usize = 0;
pub(crate) const COPY_FILL_MERGED: usize = 1;
pub(crate) const COPY_BUFFER: usize = 2;
pub(crate) const COPY_REFRESH: usize = 3;
pub(crate) const COPY_BINDLESS: usize = 4;
pub(crate) const COPY_BOUNCE: usize = 5;
pub(crate) const COPY_BOUNCE_SKIPPED: usize = 6;

/// The texture-event part of the frame statistics line, over `n` frames,
/// zeroing the counts.
pub(crate) fn texture_events_text(n: u64) -> String {
    let ev: Vec<u64> = TEXTURE_EVENTS.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
    let c: Vec<u64> = COPY_KINDS.iter().map(|c| c.swap(0, Ordering::Relaxed) / n).collect();
    format!(
        "placed textures: {} hits, {} made, {} made again, {} released; uploaded {} KiB; {} cold staging states dropped; upload buffers {} reused, {} made; title copies: {} fills ({} more merged in), {} buffer copies, {} refreshes, {} bindless reads, {} bounces ({} skipped)",
        ev[EV_HIT] / n,
        ev[EV_MADE] / n,
        ev[EV_REMADE] / n,
        ev[EV_RELEASED] / n,
        ev[EV_UPLOADED] / n / 1024,
        ev[EV_STAGING_DROPPED],
        ev[EV_UPLOAD_REUSED] / n,
        ev[EV_UPLOAD_MADE] / n,
        c[COPY_FILL],
        c[COPY_FILL_MERGED],
        c[COPY_BUFFER],
        c[COPY_REFRESH],
        c[COPY_BINDLESS],
        c[COPY_BOUNCE],
        c[COPY_BOUNCE_SKIPPED],
    ) + &crate::shared::stats_text(n)
}

/// A mapped upload buffer from [`upload_pool`]: the pool takes it back when
/// it is dropped, which is once the GPU work that reads it has finished
/// (every submission is waited for; a list's buffers go at its reset).
pub(crate) struct Upload {
    pub(crate) buffer: ID3D12Resource,
    pub(crate) data: *mut u8,
    class: u32,
}

// SAFETY: D3D12 resources are free-threaded; the mapping is written by one
// owner at a time.
unsafe impl Send for Upload {}
// SAFETY: as above.
unsafe impl Sync for Upload {}

/// Free upload buffers by size class: class `c` holds `UPLOAD_MIN << c`
/// bytes. Making an upload heap buffer cost about 8 ms each in the world
/// (one per title texture: `staging made 1328046 us/163` per frame);
/// a buffer from here costs a lock.
struct Pool {
    free: Vec<Vec<(ID3D12Resource, usize)>>,
    /// Bytes in `free`, and in use.
    bytes: [u64; 2],
}

// SAFETY: as for `Upload`.
unsafe impl Send for Pool {}

static POOL: Mutex<Pool> = Mutex::new(Pool { free: Vec::new(), bytes: [0; 2] });

/// The smallest upload class (64 KiB, the placement alignment of buffers).
const UPLOAD_MIN: u64 = 64 << 10;
/// Free upload memory kept at most; past it a buffer given back is released.
const UPLOAD_KEEP: u64 = 256 << 20;

/// The class holding `size` bytes.
fn upload_class(size: u64) -> u32 {
    size.max(UPLOAD_MIN).next_power_of_two().trailing_zeros() - UPLOAD_MIN.trailing_zeros()
}

thread_local! {
    /// Set while our world pass refreshes a texture for its own list
    /// (that list runs in null mode; the title's would not).
    static PASS_UPLOADS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` with the null-mode refusal of [`upload_pool`] lifted on
/// this thread (our world pass records and runs the copies itself).
pub(crate) fn pass_uploads<R>(f: impl FnOnce() -> R) -> R {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            PASS_UPLOADS.set(self.0);
        }
    }
    let _reset = Reset(PASS_UPLOADS.replace(true));
    f()
}

/// An upload buffer of at least `size` bytes, reused when the pool has one
/// of its class.
pub(crate) fn upload_pool(size: u64) -> Option<Upload> {
    let class = upload_class(size);
    let bytes = UPLOAD_MIN << class;
    let reused = {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        let found = pool.free.get_mut(class as usize).and_then(Vec::pop);
        if found.is_some() {
            pool.bytes[0] -= bytes;
        }
        pool.bytes[1] += bytes;
        found
    };
    if let Some((buffer, data)) = reused {
        TEXTURE_EVENTS[EV_UPLOAD_REUSED].fetch_add(1, Ordering::Relaxed);
        return Some(Upload { buffer, data: data as *mut u8, class });
    }
    // None made in a null-GPU frame (nothing runs what is recorded);
    // Except for kept UI work (uikeep::refuse is null_now && !keep_now);
    // Except for our world pass's own uploads (`pass_uploads`, this thread).
    if super::uikeep::refuse() && !PASS_UPLOADS.get() {
        POOL.lock().unwrap_or_else(|e| e.into_inner()).bytes[1] -= bytes;
        super::shadow::NULL_REFUSED.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let started = std::time::Instant::now();
    let made = host().and_then(|h| h.upload_buffer(bytes));
    stage(STAGE_UPLOAD_MADE, started);
    let Some((buffer, data)) = made else {
        POOL.lock().unwrap_or_else(|e| e.into_inner()).bytes[1] -= bytes;
        return None;
    };
    name(&buffer, || format!("pooled upload {bytes:#x}"));
    TEXTURE_EVENTS[EV_UPLOAD_MADE].fetch_add(1, Ordering::Relaxed);
    Some(Upload { buffer, data, class })
}

impl Drop for Upload {
    fn drop(&mut self) {
        let bytes = UPLOAD_MIN << self.class;
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        pool.bytes[1] -= bytes;
        if pool.bytes[0] + bytes > UPLOAD_KEEP {
            return; // released here
        }
        let class = self.class as usize;
        if pool.free.len() <= class {
            pool.free.resize_with(class + 1, Vec::new);
        }
        pool.bytes[0] += bytes;
        pool.free[class].push((self.buffer.clone(), self.data as usize));
    }
}

/// Upload memory the pool holds (free and in use), for the memory line.
pub(crate) fn upload_pool_bytes() -> u64 {
    let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.bytes[0] + pool.bytes[1]
}

/// What skipped submissions did: buffer copies (count, us), title-
/// writing work (count, us), value writes (count, us), the zero-copy dry
/// classification (us).
pub(crate) static SKIPPED_WORK: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// `XNW_GPU_DEADLINE=<ms>`: how long a submission may run before the
/// process ends itself (0: no deadline). Default 1200 on a hardware
/// adapter (Windows' GPU timeout is 2 s), 0 on WARP.
pub(crate) fn deadline_ms() -> u32 {
    static MS: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
        let software = crate::d3d12x::host::host_device().is_some_and(|h| h.caps.software);
        let ms = std::env::var("XNW_GPU_DEADLINE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(if software { 0 } else { 1200 });
        xnw_core::info!("d3d12x: GPU deadline {ms} ms per submission (XNW_GPU_DEADLINE; 0: none)");
        ms
    });
    *MS
}

/// A submission ran past its deadline: what the crumbs saw, the
/// pipeline library saved (at most 300 ms), then the process ends at
/// once, without tearing the device down.
fn deadline_hit(host: &Host, submitted: u64, lists: usize) -> ! {
    // SAFETY: a plain query.
    let completed = unsafe { host.fence.GetCompletedValue() };
    xnw_core::warn!(
        "gpu deadline: submission {submitted} ({lists} lists) not done after {} ms (fence at {completed}): ending the process before the driver's timeout",
        deadline_ms()
    );
    super::crumbs::report(completed, submitted);
    super::loopcount::report();
    let (tx, rx) = std::sync::mpsc::channel();
    let _ = std::thread::Builder::new().name("xnw deadline save".into()).spawn(move || {
        super::pipeline::save_library_now();
        let _ = tx.send(());
    });
    let saved = rx.recv_timeout(std::time::Duration::from_millis(300)).is_ok();
    xnw_core::warn!("gpu deadline: pipeline library {}; terminating", if saved { "saved" } else { "not saved in 300 ms" });
    // SAFETY: ends our own process now.
    unsafe { xnw_core::win::TerminateProcess(xnw_core::win::GetCurrentProcess(), 0xDEAD_0001) };
    std::process::abort()
}

/// The fence value of the last submission sent to the queue.
static SIGNALLED: AtomicU64 = AtomicU64::new(0);

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Logs the validation layer's last stored error messages (`all`: messages
/// of every severity) (`XNW_GPU_DEBUG=layer`): they name the invalid call.
fn layer_report(device: &ID3D12Device, all: bool) {
    use windows::Win32::Graphics::Direct3D12::{D3D12_MESSAGE, D3D12_MESSAGE_SEVERITY_WARNING, ID3D12InfoQueue};
    use windows::core::Interface;
    let Ok(queue) = device.cast::<ID3D12InfoQueue>() else {
        return;
    };
    // SAFETY: plain queries into buffers of the size the queue asked for.
    unsafe {
        let n = queue.GetNumStoredMessages();
        let mut shown = 0;
        for i in (0..n).rev() {
            let mut len = 0usize;
            if queue.GetMessage(i, None, &mut len).is_err() || len == 0 {
                continue;
            }
            let mut buffer = vec![0u64; len.div_ceil(8)];
            let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
            if queue.GetMessage(i, Some(message), &mut len).is_err() {
                continue;
            }
            let m = &*message;
            if !all && m.Severity.0 >= D3D12_MESSAGE_SEVERITY_WARNING.0 {
                continue;
            }
            let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength.saturating_sub(1));
            xnw_core::warn!("d3d12x: validation #{i} (severity {}, id {}): {}", m.Severity.0, m.ID.0, String::from_utf8_lossy(text));
            shown += 1;
            if shown == 24 {
                break;
            }
        }
        xnw_core::warn!("d3d12x: validation: {shown} messages shown of {n} stored");
    }
}

/// Logs what the device-removal data holds (`XNW_GPU_DEBUG=dred`): for each
/// command list that did not finish, the last operations that ran and the
/// first that did not, then the page fault address if there was one.
fn dred_report(device: &ID3D12Device) {
    use windows::Win32::Graphics::Direct3D12::{D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT, D3D12_DRED_PAGE_FAULT_OUTPUT, ID3D12DeviceRemovedExtendedData};
    use windows::core::Interface;
    let Ok(dred) = device.cast::<ID3D12DeviceRemovedExtendedData>() else {
        return;
    };
    // SAFETY: plain queries; the nodes are the runtime's and live with the device.
    unsafe {
        if let Ok(D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT { pHeadAutoBreadcrumbNode: mut node }) = dred.GetAutoBreadcrumbsOutput() {
            while !node.is_null() {
                let n = &*node;
                let (done, count) = (if n.pLastBreadcrumbValue.is_null() { 0 } else { *n.pLastBreadcrumbValue }, n.BreadcrumbCount);
                if done < count && !n.pCommandHistory.is_null() {
                    let ops = std::slice::from_raw_parts(n.pCommandHistory, count as usize);
                    let from = done.saturating_sub(6) as usize;
                    let shown: Vec<String> = ops[from..(done as usize + 3).min(ops.len())]
                        .iter()
                        .enumerate()
                        .map(|(i, op)| format!("{}{}", if from + i == done as usize { ">>" } else { "" }, op.0))
                        .collect();
                    let name = if n.pCommandListDebugNameA.is_null() { String::new() } else { std::ffi::CStr::from_ptr(n.pCommandListDebugNameA.cast()).to_string_lossy().into_owned() };
                    xnw_core::warn!("d3d12x: removal: list {name}: {done} of {count} operations ran; ops around the stop (>> first not run): {}", shown.join(" "));
                }
                node = n.pNext;
            }
        }
        if let Ok(D3D12_DRED_PAGE_FAULT_OUTPUT { PageFaultVA, .. }) = dred.GetPageFaultAllocationOutput() {
            if PageFaultVA != 0 {
                xnw_core::warn!("d3d12x: removal: page fault at GPU address {PageFaultVA:#x}");
            }
        }
    }
}

/// The removal reason of `device` (`None` while it lives), logging the
/// device-removal data and the layer's stored errors when it is gone.
pub fn removal_report(device: &ID3D12Device) -> Option<String> {
    // SAFETY: a plain query.
    let e = unsafe { device.GetDeviceRemovedReason() }.err()?;
    dred_report(device);
    layer_report(device, true);
    Some(format!("device removed: reason {:#010x} ({})", e.code().0 as u32, e.message()))
}

/// Ends the run on a host GPU failure: what failed and the removal reason.
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: reason {:#010x} ({})", e.code().0 as u32, e.message()));
            dred_report(&host.device);
            // Host pipelines compiling now become suspects.
            super::pipeline::note_device_lost(&format!("host device removed ({:#010x})", e.code().0 as u32));
        }
        // Where each list of the last submissions was (`XNW_GPU_CRUMBS=1`);
        // With the queue's state (completed against signalled).
        // SAFETY: a plain query.
        super::crumbs::report(unsafe { host.fence.GetCompletedValue() }, SIGNALLED.load(Ordering::Relaxed));
        // The shader loops that ran longest (`XNW_LOOP_COUNTS=1`).
        super::loopcount::report();
        // Any failure: the validation layer names an invalid command even
        // when the device survives it (a list that did not close).
        layer_report(&host.device, false);
    }
    xnw_core::warn!("d3d12x: host GPU failure: {text}");
    // What the pipeline workers built survives the failure.
    super::pipeline::flush_library("host GPU failure");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

/// The one host device for the process (the console has one too), made
/// on first use: the Agility SDK's device factory when `XNW_AGILITY` names
/// its folder, else the inbox runtime. (Was `d3d12x/gpu.rs`; moved here by
/// once the presenter stopped needing that file's queue and list.)
pub fn host_device() -> Option<&'static crate::host::Host> {
    struct Device(crate::host::Host);
    // SAFETY: D3D12 devices are free-threaded.
    unsafe impl Send for Device {}
    // SAFETY: as above.
    unsafe impl Sync for Device {}
    static DEVICE: OnceLock<Option<Device>> = OnceLock::new();
    DEVICE
        .get_or_init(|| {
            let agility = std::env::var_os("XNW_AGILITY").map(std::path::PathBuf::from).filter(|p| !p.as_os_str().is_empty());
            match crate::host::Host::new(agility.as_deref()) {
                Ok(h) => {
                    xnw_core::info!(
                        "d3d12x: host device {} ({} MiB), shader model {}.{}, agility {:?}",
                        h.caps.adapter,
                        h.caps.vram_bytes >> 20,
                        h.caps.shader_model.0,
                        h.caps.shader_model.1,
                        h.caps.agility
                    );
                    Some(Device(h))
                }
                Err(e) => {
                    xnw_core::warn!("d3d12x: no host device: {e}");
                    None
                }
            }
        })
        .as_ref()
        .map(|d| &d.0)
}

/// The submission host, made on first use over the one host device;
/// `None` without one (lists then record nothing and nothing runs).
pub(crate) fn host() -> Option<&'static Host> {
    static HOST: OnceLock<Option<Host>> = OnceLock::new();
    HOST.get_or_init(|| {
        let device = host_device()?.device.clone();
        match create(device) {
            Ok(host) => Some(host),
            Err(e) => {
                xnw_core::warn!("d3d12x: no host submission queue: {e}; command lists will not run");
                None
            }
        }
    })
    .as_ref()
}

fn create(device: ID3D12Device) -> windows::core::Result<Host> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let rtvs: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: RTV_RING, ..Default::default() })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: RTV_RING, ..Default::default() })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        xnw_core::info!("d3d12x: host submission queue ready (one direct queue, submissions waited for)");
        Ok(Host {
            device,
            queue,
            fence,
            submitted: Mutex::new(0),
            event,
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(None),
        })
    }
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// The list that runs those transitions just before this one.
    prelude: (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// Its breadcrumb slot pair (`XNW_GPU_CRUMBS=1`); its
    /// prelude's.
    crumbs: super::crumbs::ListCrumbs,
    prelude_crumbs: super::crumbs::ListCrumbs,
    /// `ui=keep`: it holds kept work since its reset (uikeep.rs).
    pub(crate) kept: std::sync::atomic::AtomicBool,
}

// SAFETY: D3D12 lists are used by one thread at a time (the title's rule);
// the rest is behind locks.
unsafe impl Send for HostList {}
// SAFETY: as above.
unsafe impl Sync for HostList {}

type States = std::collections::HashMap<usize, (u32, u32, Arc<HostTexture>)>;

/// The render target view of `resource` the host can take. A view of a
/// typeless texture left typeless or unnamed is a null view (its writes
/// lost): it takes the family's first typed format, the rule VIctory
/// (GPL-3.0-or-later, Xvania7, commit 2be93af) applies to
/// shader views; packed 16-bit colour stored as R16_TYPELESS writes
/// R16_UNORM. A format of another family is refused (counted) and the
/// family's format used instead.
pub(crate) fn target_view(resource: &D3D12_RESOURCE_DESC, desc: Option<D3D12_RENDER_TARGET_VIEW_DESC>) -> Option<D3D12_RENDER_TARGET_VIEW_DESC> {
    use super::descriptor::{TYPELESS_RTV, VIEWS_REFUSED, count_typeless};
    use super::resource::{family, first_concrete, is_typeless};
    let own = resource.Format.0 as u32;
    if !is_typeless(own) {
        return desc;
    }
    let mut d = desc.unwrap_or_else(|| {
        let layers = u32::from(resource.DepthOrArraySize);
        let mut d = D3D12_RENDER_TARGET_VIEW_DESC::default();
        if resource.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
            d.Anonymous.Texture3D = D3D12_TEX3D_RTV { WSize: u32::MAX, ..Default::default() };
        } else if resource.SampleDesc.Count > 1 {
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
        } else if layers > 1 {
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
            d.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_RTV { ArraySize: layers, ..Default::default() };
        } else {
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        }
        d
    });
    let view = d.Format.0 as u32;
    let used = if own == 53 && matches!(view, 0 | 53 | 86 | 115) {
        56
    } else if view == 0 || is_typeless(view) {
        first_concrete(own)
    } else if family(view) != family(own) {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| xnw_core::warn!("d3d12x: render target view format {view} of another family than the texture's {own}"));
        count_typeless(VIEWS_REFUSED);
        d.Format = DXGI_FORMAT(first_concrete(own) as i32);
        return Some(d);
    } else {
        return Some(d);
    };
    d.Format = DXGI_FORMAT(used as i32);
    count_typeless(TYPELESS_RTV);
    Some(d)
}

/// The depth view of `resource` the host can take: a typeless depth
/// texture (made so for shaders to read it) is viewed in its depth target
/// format, whatever of its family (or none) the view names.
pub(crate) fn depth_view(resource: &D3D12_RESOURCE_DESC, desc: Option<D3D12_DEPTH_STENCIL_VIEW_DESC>) -> Option<D3D12_DEPTH_STENCIL_VIEW_DESC> {
    use super::descriptor::{TYPELESS_DSV, VIEWS_REFUSED, count_typeless};
    let own = resource.Format.0 as u32;
    let Some(target) = super::resource::depth_target(own) else {
        return desc;
    };
    let mut d = desc.unwrap_or_else(|| {
        let layers = u32::from(resource.DepthOrArraySize);
        let mut d = D3D12_DEPTH_STENCIL_VIEW_DESC::default();
        match (resource.SampleDesc.Count > 1, layers > 1) {
            (true, false) => d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS,
            (true, true) => {
                d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY;
                d.Anonymous.Texture2DMSArray = D3D12_TEX2DMS_ARRAY_DSV { FirstArraySlice: 0, ArraySize: layers };
            }
            (false, true) => {
                d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
                d.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_DSV { ArraySize: layers, ..Default::default() };
            }
            (false, false) => d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D,
        }
        d
    });
    let view = d.Format.0 as u32;
    if view == target {
        return Some(d);
    }
    if view != 0 && super::resource::family(view) != own {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| xnw_core::warn!("d3d12x: depth view format {view} of another family than the texture's {own}"));
        count_typeless(VIEWS_REFUSED);
    } else {
        count_typeless(TYPELESS_DSV);
    }
    d.Format = DXGI_FORMAT(target as i32);
    Some(d)
}

/// Names a host object for a device-removed report (only with
/// device-removed data, which this build does not collect).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    let _ = (object, text);
}

/// Upload chunks handed out front to back: the chunks, the one in use and
/// the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<(ID3D12Resource, usize, u64)>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

const SCRATCH_CHUNK: u64 = 4 << 20;

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(Arc<Shadow>, usize, usize),
    /// Work that may write title memory.
    Run(Box<dyn Fn() + Send>),
    /// Work that writes host memory only.
    Host(Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy.
    GpuWrite(Arc<Shadow>, usize, usize),
    /// Brings a texture's staging buffer up to date and, when it changed,
    /// copies it into the host texture on the list's prelude: once per
    /// submission, whichever lists sample it, until a `Copy` or `Run`
    /// intervenes (VIctory).
    Texture(Arc<super::texture::Staging>, Arc<super::resource::HostTexture>),
}

/// Textures whose contents changed, detiled into pooled upload buffers, to
/// copy on a prelude; the buffers go back to the pool after the submission.
type Uploads = Vec<(Arc<super::texture::Staging>, Arc<super::resource::HostTexture>, Upload)>;

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    copies: Vec<Step>,
    uploads: Vec<Upload>,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize)>,
    /// View heaps the list's draws index directly, and whether any of their
    /// shaders may write memory: their slots are brought up to date once per
    /// submission, after every other step.
    bindless: Vec<(Arc<HostHeap>, bool)>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, Written, usize)>,
    /// VIctory: the list's memory waits, in order: address,
    /// comparison (D3D12_COMPARISON_FUNC), reference, 64-bit.
    waits: Vec<(u64, u32, u64, bool)>,
    /// Whether a step zeroes the indirect-work budget.
    budget_zeroed: bool,
    /// Where in `copies` the last fill is, if no step that writes title
    /// memory came after it: a fill that continues it is merged in.
    last_fill: Option<usize>,
    /// `XNW_LATE_LOG=1`: culling inputs noted when recorded.
    late: Vec<super::late::Note>,
}


/// Host resource creations (count, microseconds) since the last frame
/// line, and CPU-visible buffers refused for the non-local budget.
pub(crate) static CREATION: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// One host resource creation of `kind` (`bytes`) begun at
/// `since`: counted, and logged with the video memory now when it took
/// over 50 ms (one took tens of seconds in the kernel).
pub(crate) fn note_creation(kind: &str, bytes: u64, since: std::time::Instant) {
    let us = since.elapsed().as_micros() as u64;
    CREATION[0].fetch_add(1, Ordering::Relaxed);
    CREATION[1].fetch_add(us, Ordering::Relaxed);
    if us > 50_000 {
        xnw_core::warn!("d3d12x: host {kind} resource of {} KiB took {} ms to make; {}", bytes >> 10, us / 1000, super::resource::video_memory_text());
    }
}

/// `XNW_ZEROCOPY_DRY=1`: skipped submissions classify what they would
/// have copied ([`HostList::run_skipped`]).
fn zero_copy_dry() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_ZEROCOPY_DRY").is_ok_and(|v| v.trim() == "1"));
    *ON
}

impl Host {
    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
            let prelude_allocator: ID3D12CommandAllocator = self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let prelude: ID3D12GraphicsCommandList = self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &prelude_allocator, None)?;
            prelude.Close()?;
            Ok(HostList {
                allocator,
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                prelude: (prelude_allocator, prelude),
                pipelines: Mutex::default(),
                budget: OnceLock::new(),
                crumbs: super::crumbs::ListCrumbs::new(),
                prelude_crumbs: super::crumbs::ListCrumbs::new(),
                kept: std::sync::atomic::AtomicBool::new(false),
            })
            .inspect(|l: &HostList| {
                l.crumbs.mark(&l.list, "list begin");
                super::crumbs::list_start(&l.list);
            })
        }
    }

    /// Runs closed lists in order and waits for them, then does what the
    /// lists' shaders wrote back and their bottom-of-pipe writes.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut submitted = super::counters::timed(super::counters::LOCK_SUBMITTED, &self.submitted);
        // Shader loop counts start from zero for each submission.
        super::loopcount::clear();
        // Who holds the submission, for stall dumps.
        struct Owner;
        impl Drop for Owner {
            fn drop(&mut self) {
                SUBMIT_OWNER.store(0, Ordering::Relaxed);
            }
        }
        // SAFETY: a plain query.
        SUBMIT_OWNER.store(u64::from(unsafe { xnw_core::win::GetCurrentThreadId() }), Ordering::Relaxed);
        let _owner = Owner;
        let started = std::time::Instant::now();
        let snapshots = super::counters::snapshot_total();
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        let mut batch = Batch::default();
        // The upload buffers the preludes copy from, given back to the pool
        // once the submission has run.
        let mut uploaded: Vec<Uploads> = Vec::with_capacity(runs.len());
        // The copies below see title memory as at this submission: each
        // page's written state is asked of the kernel once for all of them
        // (again after a step that writes title memory; VIctory).
        let sweep = super::watch::Sweep::begin();
        let mut crumbs: Vec<(u32, Option<u32>)> = Vec::new();
        for list in runs {
            let uploads = list.run_deferred(&mut batch);
            let at = std::time::Instant::now();
            let prelude = list.prelude(&uploads);
            if super::crumbs::on() || super::crumbs::stepping() {
                crumbs.push((list.crumbs.slot(), prelude.is_some().then(|| list.prelude_crumbs.slot())));
            }
            if let Some(prelude) = prelude {
                lists.push(Some(prelude));
            }
            stage(STAGE_PRELUDE, at);
            lists.push(list.list.cast().ok());
            uploaded.push(uploads);
            phase(PH_PRELUDE, at);
        }
        batch.prepare_bindless();
        drop(sweep);
        if super::late::on() {
            super::late::submitting(runs.iter().flat_map(|l| std::mem::take(&mut l.deferred().late)).collect());
        }
        let at = std::time::Instant::now();
        let retired = take_retired();
        phase(PH_RETIRED, at);
        let prepared = std::time::Instant::now();
        *submitted += 1;
        // The lists' breadcrumb slots, remembered for a failure report;
        // Zeroed, and the submission bracketed by two lists of ours.
        let crumb_submission = if super::crumbs::on() { super::crumbs::submitted(*submitted, crumbs.clone()) } else { 0 };
        if let Some([first, last]) = super::crumbs::bracket(&self.device, crumb_submission) {
            lists.insert(0, Some(first));
            lists.push(Some(last));
        }
        SIGNALLED.store(*submitted, Ordering::Relaxed);
        // SAFETY: closed lists; the fence and event are ours.
        unsafe {
            if !lists.is_empty() {
                self.queue.ExecuteCommandLists(&lists);
            }
            stage(STAGE_EXECUTE, prepared);
            let waited = self.queue.Signal(&self.fence, *submitted).and_then(|()| self.fence.SetEventOnCompletion(*submitted, self.event));
            match waited {
                // `XNW_GPU_STEP`: the wait times each op from its crumbs.
                Ok(()) if super::crumbs::stepping() => {
                    super::crumbs::watch(|| WaitForSingleObject(self.event, 0).0 == 0, &crumbs);
                }
                Ok(()) if super::crumbs::sync() => {
                    // `XNW_GPU_SYNC=1`: a submission past 2 s is named
                    // while it runs (a hang, or the removal it ends in).
                    const WAIT_TIMEOUT: u32 = 0x102;
                    let mut waited_s = 0;
                    while WaitForSingleObject(self.event, 2000).0 == WAIT_TIMEOUT {
                        waited_s += 2;
                        xnw_core::warn!(
                            "gpu sync: submission {} ({} lists) not done after {waited_s} s; {}",
                            *submitted,
                            runs.len(),
                            if super::crumbs::on() { format!("crumbs submission {crumb_submission}: {}", super::crumbs::submission_text(&crumbs)) } else { "no crumbs (XNW_GPU_CRUMBS=1)".into() }
                        );
                    }
                }
                // `XNW_GPU_DEADLINE`: a submission not done in time ends
                // the process before the driver's own timeout.
                Ok(()) if deadline_ms() > 0 => {
                    const WAIT_TIMEOUT: u32 = 0x102;
                    if WaitForSingleObject(self.event, deadline_ms()).0 == WAIT_TIMEOUT {
                        deadline_hit(self, *submitted, runs.len());
                    }
                }
                Ok(()) => {
                    WaitForSingleObject(self.event, INFINITE);
                }
                Err(e) => xnw_core::warn!("d3d12x: host submission not waited for: {e}"),
            }
        }
        let gpu = prepared.elapsed();
        SUBMISSION_US[0].fetch_add((prepared - started).as_micros() as u64, Ordering::Relaxed);
        super::counters::submission((prepared - started).as_micros() as u64, snapshots);
        SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs.iter().flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone()).collect();
            xnw_core::warn!("d3d12x: slow host submission: {} ms, {} lists, pipelines {pipelines:?}", gpu.as_millis(), runs.len());
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!("host GPU device removed after a submission of {} lists", runs.len()));
        }
        let at = std::time::Instant::now();
        drop(retired);
        drop(uploaded);
        stage(STAGE_RETIRE, at);
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it.
        let copying = std::time::Instant::now();
        let mut back: Vec<(Arc<Shadow>, usize, usize)> = Vec::new();
        for list in runs {
            for step in &list.deferred().copies {
                if let Step::GpuWrite(shadow, address, len) = step
                    && !back.iter().any(|(s, a, l)| Arc::ptr_eq(s, shadow) && (*a, *l) == (*address, *len))
                {
                    back.push((shadow.clone(), *address, *len));
                }
            }
        }
        for (shadow, address, len) in batch.bindless_writes.drain(..) {
            if !back.iter().any(|(s, a, l)| Arc::ptr_eq(s, &shadow) && (*a, *l) == (address, len)) {
                back.push((shadow, address, len));
            }
        }
        for (shadow, address, len) in &back {
            let written = shadow.copy_back(*address, *len);
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        for list in runs {
            for &(at, what, bytes) in &list.deferred().writes {
                what.write(at as *mut u8, bytes);
            }
        }
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(&self, resource: &ID3D12Resource, desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            let patched = target_view(&resource.GetDesc(), desc.copied());
            self.device.CreateRenderTargetView(resource, patched.as_ref().map(|d| d as *const _), handle);
            handle
        }
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(&self, resource: &ID3D12Resource, desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            let patched = depth_view(&resource.GetDesc(), desc.copied());
            self.device.CreateDepthStencilView(resource, patched.as_ref().map(|d| d as *const _), handle);
            handle
        }
    }

    /// A buffer in the upload heap, mapped for writing.
    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() };
        self.mapped_buffer(&heap, size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, "upload")
    }

    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        self.mapped_buffer(&heap, size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, "shared")
    }

    fn mapped_buffer(&self, heap: &D3D12_HEAP_PROPERTIES, size: u64, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES, what: &str) -> Option<(ID3D12Resource, *mut u8)> {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        // CPU-visible buffers live in the PC's memory (the video memory
        // manager's non-local segment): refused past 9/10 of its budget
        // rather than made, counted.
        if !super::resource::room_nonlocal(size) {
            CREATION[2].fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        let creating = std::time::Instant::now();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut buffer).and_then(|()| {
                let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                b.Map(0, None, Some(&mut data))
            })
        };
        note_creation(what, size, creating);
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                let kind = if what == "upload" { xnw_core::syscalls::Made::Upload } else { xnw_core::syscalls::Made::Shared };
                xnw_core::syscalls::creation_failed(kind, &e, &format_args!("{what} buffer of {size} bytes, state {:#x}", state.0));
                None
            }
        }
    }
}

impl HostList {
    /// The list's breadcrumb state.
    pub(crate) fn crumbs(&self) -> &super::crumbs::ListCrumbs {
        &self.crumbs
    }

    pub(crate) fn close(&self) {
        self.crumbs.mark(&self.list, "list end");
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!("a host command list did not close ({e}): a command recorded into it was invalid"));
        }
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        self.kept.store(false, Ordering::Relaxed);
        *self.deferred() = Deferred::default();
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            (scratch.current, scratch.used) = (0, 0);
        }
        // SAFETY: the list is not executing.
        let reset = unsafe { self.allocator.Reset().and_then(|()| self.list.Reset(&self.allocator, None)) };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
        self.crumbs.mark(&self.list, "list begin");
        super::crumbs::list_start(&self.list);
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    pub(crate) fn transition(&self, resource: This, wanted: u32) -> Option<(&'static ID3D12Resource, u32)> {
        let texture = super::resource::host_resource(resource)?;
        let copy = super::resource::host_copy(resource)?;
        let wanted = legal_state(texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(Arc::as_ptr(&copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some((texture, before))
            }
        }
    }

    /// At submission: the list that copies newly changed textures from
    /// their staging buffers into their host textures (`uploads`) and moves
    /// each texture this list uses from its real state to the one the list
    /// starts with; the textures' real states become those the list ends
    /// with (VIctory).
    fn prelude(&self, uploads: &Uploads) -> Option<ID3D12CommandList> {
        const COPY_DEST: u32 = D3D12_RESOURCE_STATE_COPY_DEST.0 as u32;
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        super::resource::resident(states.values().map(|(_, _, copy)| copy).chain(uploads.iter().map(|(_, copy, _)| copy)));
        let mut pre_barriers = Vec::new();
        for (_, copy, _) in uploads {
            let real = copy.state.swap(COPY_DEST, Ordering::Relaxed);
            if real != COPY_DEST {
                pre_barriers.push(transition(&copy.texture, D3D12_RESOURCE_STATES(real as i32), D3D12_RESOURCE_STATE_COPY_DEST));
            }
        }
        let barriers: Vec<D3D12_RESOURCE_BARRIER> = states
            .values()
            .filter_map(|(first, last, copy)| {
                let real = copy.state.swap(*last, Ordering::Relaxed);
                (real != *first).then(|| transition(&copy.texture, D3D12_RESOURCE_STATES(real as i32), D3D12_RESOURCE_STATES(*first as i32)))
            })
            .collect();
        if uploads.is_empty() && barriers.is_empty() {
            return None;
        }
        let (allocator, list) = &self.prelude;
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures and staging buffers are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                self.prelude_crumbs.mark(list, "prelude begin");
                if !pre_barriers.is_empty() {
                    list.ResourceBarrier(&pre_barriers);
                }
                {
                    let _crumb = self.prelude_crumbs.begin(list, || format!("prelude: {} texture uploads", uploads.len()));
                    for (staging, copy, upload) in uploads {
                        let bytes = staging.record_copies(list, &copy.texture, upload);
                        TEXTURE_EVENTS[EV_UPLOADED].fetch_add(bytes, Ordering::Relaxed);
                    }
                }
                if !barriers.is_empty() {
                    let _crumb = self.prelude_crumbs.begin(list, || format!("prelude: {} state moves", barriers.len()));
                    list.ResourceBarrier(&barriers);
                }
                self.prelude_crumbs.mark(list, "prelude end");
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// Notes a pipeline the list binds.
    pub(crate) fn used_pipeline(&self, name: &str) {
        self.crumbs.pipeline(name);
        super::loopcount::note_pipeline(name);
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if !pipelines.contains(name) {
            pipelines.insert(name.to_owned());
        }
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        deferred.copies.push(Step::Copy(dst as usize, src as usize, len));
        deferred.refreshed.clear();
        deferred.last_fill = None;
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    /// Keeps a culling input noted at recording for the submission.
    pub(crate) fn note_late(&self, note: super::late::Note) {
        self.deferred().late.push(note);
    }

    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        let (dst, src) = (dst as usize, src as usize);
        let mut deferred = self.deferred();
        let d = &mut *deferred;
        // Draws fill constants piece by piece (one copy through the kernel
        // each, ~8 us); a piece right after the last one in upload memory
        // (past only its 256-byte padding, which is that piece's own) whose
        // source lies as far after the last one's becomes one copy with it.
        // Only steps that read title memory came since, so the order holds.
        if let Some(i) = d.last_fill
            && let Some(Step::Fill(at, from, n)) = d.copies.get_mut(i)
            && dst == (*at + *n).next_multiple_of(256)
            && src.checked_sub(*from) == Some(dst - *at)
        {
            *n = dst + len - *at;
            COPY_KINDS[COPY_FILL_MERGED].fetch_add(1, Ordering::Relaxed);
            return;
        }
        d.last_fill = Some(d.copies.len());
        d.copies.push(Step::Fill(dst, src, len));
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(&self, shadow: &Arc<Shadow>, address: usize, len: usize) {
        let mut deferred = self.deferred();
        let key = (Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred.copies.push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run.
    pub(crate) fn defer_gpu_write(&self, shadow: &Arc<Shadow>, address: usize, len: usize) {
        self.deferred().copies.push(Step::GpuWrite(shadow.clone(), address, len));
    }

    /// Debug: whether the list declared, before now, shaders writing
    /// `[address, address + len)` of `shadow`'s copy, and whether any of its
    /// draws index a heap with shaders that write (their ranges are known
    /// only at submission).
    pub(crate) fn writes_declared(&self, shadow: &Arc<Shadow>, address: usize, len: usize) -> (bool, bool) {
        let d = self.deferred();
        let declared = d.copies.iter().any(|s| matches!(s, Step::GpuWrite(s2, a, l) if Arc::ptr_eq(s2, shadow) && *a < address + len && address < *a + *l));
        (declared, d.bindless.iter().any(|(_, w)| *w))
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// A memory wait recorded in the list (see `queue::execute`).
    pub(crate) fn defer_wait(&self, address: u64, comparison: u32, reference: u64, wide: bool) {
        self.deferred().waits.push((address, comparison, reference, wide));
    }

    /// The list's memory waits.
    pub(crate) fn waits(&self) -> Vec<(u64, u32, u64, bool)> {
        self.deferred().waits.clone()
    }

    /// The addresses of the list's bottom-of-pipe writes.
    pub(crate) fn write_addresses(&self) -> Vec<u64> {
        self.deferred().writes.iter().map(|&(at, _, _)| at as u64).collect()
    }

    /// Whether the list's bottom-of-pipe writes cover `address`.
    pub(crate) fn writes_to(&self, address: u64) -> bool {
        let a = address as usize;
        self.deferred().writes.iter().any(|&(at, _, bytes)| at <= a && a < at + bytes)
    }

    /// A mapped upload buffer of at least `size` bytes from the pool,
    /// alive until the list is reset (then the pool takes it back).
    pub(crate) fn upload(&self, _host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let upload = upload_pool(size)?;
        let out = (upload.buffer.clone(), upload.data);
        self.deferred().uploads.push(upload);
        Some(out)
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            let upload = upload_pool(len)?;
            // SAFETY: a live buffer.
            let gpu = unsafe { upload.buffer.GetGPUVirtualAddress() };
            let data = upload.data;
            self.deferred().uploads.push(upload);
            return Some((gpu, data));
        }
        if s.used + len > SCRATCH_CHUNK && s.current < s.chunks.len() {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            if super::uikeep::refuse() {
                super::shadow::NULL_REFUSED.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            let (buffer, data) = host.upload_buffer(SCRATCH_CHUNK)?;
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            s.chunks.push((buffer, data as usize, gpu));
            s.used = 0;
        }
        let (_, data, gpu) = s.chunks[s.current];
        let at = s.used;
        s.used += len;
        Some((gpu + at, (data + at as usize) as *mut u8))
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped word, written before it runs.
            deferred.copies.push(Step::Host(Box::new(move || unsafe { (at as *mut u32).write_volatile(0) })));
        }
        // SAFETY: a live buffer.
        Some(unsafe { buffer.GetGPUVirtualAddress() })
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again).
    pub(crate) fn defer(&self, work: impl Fn() + Send + 'static) {
        let mut deferred = self.deferred();
        deferred.copies.push(Step::Run(Box::new(work)));
        deferred.last_fill = None;
        deferred.refreshed.clear();
    }

    /// Queues submission-time materialization and refresh of the bindless
    /// slots in `heap`, once per list; `writes`: the draw's shaders may write
    /// memory, so the slots' unordered access ranges go back to title memory.
    pub(crate) fn defer_bindless(&self, heap: &Arc<HostHeap>, writes: bool) {
        // A heap-indexing draw while ring slots are left unmade.
        if super::descriptor::host::plan_skip_ring() {
            super::descriptor::host::INDEXED_WHILE_SKIPPING.fetch_add(1, Ordering::Relaxed);
        }
        let mut deferred = self.deferred();
        match deferred.bindless.iter_mut().find(|(h, _)| Arc::ptr_eq(h, heap)) {
            Some((_, w)) => *w |= writes,
            None => deferred.bindless.push((heap.clone(), writes)),
        }
    }

    /// Whether each heap queued for bindless preparation may be written.
    #[cfg(test)]
    pub(crate) fn bindless_queued(&self) -> Vec<bool> {
        self.deferred().bindless.iter().map(|&(_, w)| w).collect()
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work)));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted, keeping it alive until the list is reset.
    pub(crate) fn defer_texture(&self, staging: Arc<super::texture::Staging>, copy: Arc<super::resource::HostTexture>) {
        self.deferred().copies.push(Step::Texture(staging, copy));
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers are read by most lists of a submission,
    /// through many ranges).
    /// Returns the textures whose staging buffers changed, for the prelude.
    /// `XNW_FRAMESKIP`: a submission of a skipped frame: only what
    /// the title's CPU and later frames need from it: buffer copies into
    /// title memory and work that may write title memory, in order, then
    /// the bottom-of-pipe value writes (fences, markers the title waits
    /// on). No refresh, fill, texture, bindless or GPU work: the write
    /// guards keep what the title wrote meanwhile dirty for the next
    /// rendered frame.
    pub(crate) fn run_skipped(&self) {
        let deferred = self.deferred();
        // `XNW_ZEROCOPY_DRY=1`: the page classification of
        // what preparing this submission would have copied, without copying:
        // fills, buffer copies and refreshes from the steps, bindless reads
        // and bounces from each heap's last plan (made before skipping
        // began: the plan itself makes host slots, so none is made here).
        // Root views, indirect arguments and table roots are classified when
        // draws are recorded, skipped or not.
        use std::sync::atomic::Ordering::Relaxed;
        let dry_at = std::time::Instant::now();
        if zero_copy_dry() {
            for step in &deferred.copies {
                match step {
                    &Step::Copy(_, src, len) => crate::shared::zero_copy_note(COPY_BUFFER, src, len),
                    &Step::Fill(_, src, len) => crate::shared::zero_copy_note(COPY_FILL, src, len),
                    Step::Refresh(_, address, len) => crate::shared::zero_copy_note(COPY_REFRESH, *address, *len),
                    _ => {}
                }
            }
            for (heap, _) in &deferred.bindless {
                if let Some(plan) = heap.last_plan() {
                    plan.zero_copy_dry();
                }
            }
        }
        SKIPPED_WORK[6].fetch_add(dry_at.elapsed().as_micros() as u64, Relaxed);
        for step in &deferred.copies {
            let at = std::time::Instant::now();
            match step {
                &Step::Copy(dst, src, len) => {
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    SKIPPED_WORK[0].fetch_add(1, Relaxed);
                    SKIPPED_WORK[1].fetch_add(at.elapsed().as_micros() as u64, Relaxed);
                }
                Step::Run(work) => {
                    work();
                    SKIPPED_WORK[2].fetch_add(1, Relaxed);
                    SKIPPED_WORK[3].fetch_add(at.elapsed().as_micros() as u64, Relaxed);
                }
                _ => {}
            }
        }
        let at = std::time::Instant::now();
        for &(to, what, bytes) in &deferred.writes {
            what.write(to as *mut u8, bytes);
        }
        SKIPPED_WORK[4].fetch_add(deferred.writes.len() as u64, Relaxed);
        SKIPPED_WORK[5].fetch_add(at.elapsed().as_micros() as u64, Relaxed);
    }

    fn run_deferred(&self, batch: &mut Batch) -> Uploads {
        let mut uploads = Vec::new();
        let deferred = self.deferred();
        for (heap, writes) in &deferred.bindless {
            batch.bindless(heap, *writes);
        }
        for step in &deferred.copies {
            let started = std::time::Instant::now();
            let which = match step {
                Step::Copy(..) | Step::Fill(..) => PH_COPY,
                Step::Refresh(..) => PH_REFRESH,
                Step::Run(_) | Step::Host(_) => PH_WORK,
                Step::GpuWrite(..) => PH_GPU_WRITE,
                Step::Texture(..) => PH_TEXTURE,
            };
            // Timed whole, whichever way the step ends.
            struct Timed(usize, std::time::Instant);
            impl Drop for Timed {
                fn drop(&mut self) {
                    phase(self.0, self.1);
                }
            }
            let _timed = Timed(which, started);
            let copied = match step {
                &Step::Copy(dst, src, len) => {
                    COPY_KINDS[COPY_BUFFER].fetch_add(1, Ordering::Relaxed);
                    crate::shared::would_share(COPY_BUFFER, src, len);
                    // Both ends are title memory the CPU reads and the host copies mirror.
                    crate::shared::left_because(crate::shared::VIEW_COPY, crate::shared::WHY_HOST);
                    batch.title_written_range(dst, len);
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    len
                }
                &Step::Fill(dst, src, len) => {
                    COPY_KINDS[COPY_FILL].fetch_add(1, Ordering::Relaxed);
                    crate::shared::would_share(COPY_FILL, src, len);
                    super::counters::fill(src, len);
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    len
                }
                Step::Refresh(shadow, address, len) => {
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Run(work) | Step::Host(work) => {
                    if matches!(step, Step::Run(_)) {
                        batch.title_written();
                    }
                    let at = std::time::Instant::now();
                    work();
                    stage(STAGE_WORK, at);
                    continue;
                }
                Step::GpuWrite(shadow, address, len) => {
                    // The shaders start from the title's data: the pending
                    // refreshes over the range first (all of them were
                    // done at each of 326 such steps a frame, 168 ms with
                    // copy-back off).
                    batch.flush_range(*address, *len);
                    shadow.writing(*address, *len);
                    continue;
                }
                Step::Texture(staging, copy) => {
                    if batch.texture_due(staging)
                        && let Some(upload) = staging.refresh()
                    {
                        uploads.push((staging.clone(), copy.clone(), upload));
                    }
                    continue;
                }
            };
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
            stage(STAGE_COPY, started);
        }
        uploads
    }
}

/// A submission's refreshes not done yet, merged per buffer copy into the
/// range that covers them: done before a step that writes title memory or
/// gives a range to shaders, and before the lists run.
///
/// Also the textures brought up to date in the submission so far: checked
/// once, a texture is checked again only after a step that writes title
/// memory.
#[derive(Default)]
struct Batch {
    refreshes: Vec<(Arc<Shadow>, usize, usize)>,
    /// Staging buffers (by address) refreshed since the last title write.
    textures: std::collections::HashSet<usize>,
    /// View heaps the submission's draws index directly, and whether their
    /// shaders may write memory.
    bindless: Vec<(Arc<HostHeap>, bool)>,
    /// Their plans, which hold the host resources the slots view until the
    /// submission has run.
    plans: Vec<Arc<BindlessPlan>>,
    /// Bindless unordered access ranges, copied back after the submission.
    bindless_writes: Vec<(Arc<Shadow>, usize, usize)>,
}

impl Batch {
    fn add(&mut self, shadow: &Arc<Shadow>, address: usize, len: usize) {
        let end = address + len;
        match self.refreshes.iter_mut().find(|(s, ..)| Arc::ptr_eq(s, shadow)) {
            Some((_, from, to)) => (*from, *to) = ((*from).min(address), (*to).max(end)),
            None => self.refreshes.push((shadow.clone(), address, end)),
        }
    }

    /// Whether `staging` is still to be refreshed in this submission (and
    /// notes it refreshed).
    fn texture_due<T>(&mut self, staging: &Arc<T>) -> bool {
        self.textures.insert(Arc::as_ptr(staging) as *const () as usize)
    }

    /// A list of the submission indexes `heap` directly.
    fn bindless(&mut self, heap: &Arc<HostHeap>, writes: bool) {
        match self.bindless.iter_mut().find(|(h, _)| Arc::ptr_eq(h, heap)) {
            Some((_, w)) => *w |= writes,
            None => self.bindless.push((heap.clone(), writes)),
        }
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self) {
        self.flush();
        self.textures.clear();
        super::watch::Sweep::title_written();
    }

    /// Before a step that writes title memory `[at, at + len)` only (a
    /// buffer copy): the pending refreshes over it are done first and its
    /// pages asked about again; the others wait. Before, every buffer copy
    /// flushed all pending refreshes and the whole sweep (80 copies
    /// of 216 KiB a frame cost 654 ms of 'copy and fill steps').
    fn title_written_range(&mut self, at: usize, len: usize) {
        self.flush_range(at, len);
        self.textures.clear();
        super::watch::Sweep::title_written_range(at, len);
    }

    /// The pending refreshes meeting `[at, at + len)`, done now; the others
    /// wait.
    fn flush_range(&mut self, at: usize, len: usize) {
        let end = at.saturating_add(len);
        let (hit, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut self.refreshes).into_iter().partition(|&(_, from, to)| from < end && at < to);
        self.refreshes = keep;
        for (shadow, from, to) in hit {
            update(&shadow, from, to - from, COPY_REFRESH);
        }
    }

    fn flush(&mut self) {
        for (shadow, from, to) in std::mem::take(&mut self.refreshes) {
            update(&shadow, from, to - from, COPY_REFRESH);
        }
    }

    /// After every list's steps (no title copy follows): each directly
    /// indexed heap's slots, once per submission however many lists and
    /// draws use it. What they read is refreshed; their unordered access
    /// ranges are kept for copy-back only when a shader that may write
    /// memory uses the heap.
    fn prepare_bindless(&mut self) {
        let at = std::time::Instant::now();
        self.flush();
        let mut at = phase(PH_FLUSH, at);
        // The write ranges already declared, by (copy, address, bytes): a
        // look per range through all before (quadratic in the ranges).
        let mut declared: std::collections::HashSet<(usize, usize, usize)> =
            self.bindless_writes.iter().map(|(s, a, l)| (Arc::as_ptr(s) as usize, *a, *l)).collect();
        for (heap, writes) in std::mem::take(&mut self.bindless) {
            let started = at;
            // Slots of views released since are null before the plan;
            // And slots in place over shared windows released since.
            heap.null_released();
            heap.null_gone_windows();
            let plan = heap.bindless_plan();
            gpu_check(&heap);
            BINDLESS_US[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            stage(STAGE_PLAN, started);
            at = phase(PH_PLAN, started);
            let Some(plan) = plan else {
                continue;
            };
            plan.copy_constants();
            at = phase(PH_CONSTANTS, at);
            super::resource::resident_views(plan.retained());
            at = phase(PH_RESIDENT, at);
            plan.refresh_reads(|shadow, address, len| update(shadow, address, len, COPY_BINDLESS));
            // Owed GPU writes under what slots read in place go back first.
            plan.settle_in_place();
            at = phase(PH_READS, at);
            if writes {
                for (shadow, address, len) in plan.writes() {
                    if declared.insert((Arc::as_ptr(shadow) as usize, *address, *len)) {
                        shadow.writing(*address, *len);
                        self.bindless_writes.push((shadow.clone(), *address, *len));
                    }
                }
            }
            self.plans.push(plan);
            at = phase(PH_WRITES, at);
        }
    }
}

/// `XNW_GPU_CHECK=1`: before the submission runs, the first slot of a
/// heap it indexes that names a freed title resource or a released view
/// (the first 32 such submissions are logged).
fn gpu_check(heap: &HostHeap) {
    static LOGGED: AtomicU32 = AtomicU32::new(0);
    if !gpu_check_on() {
        return;
    }
    // Releases since the plan's sweep are swept again, and what the
    // check still finds is made null in place: the submission skips it.
    heap.null_released();
    if let Some((first, n)) = heap.check_live(true)
        && LOGGED.fetch_add(1, Ordering::Relaxed) < 32
    {
        xnw_core::warn!("d3d12x: XNW_GPU_CHECK: submission in frame {}: heap of {} slots, {n} slots stale (made null); first {first}", super::queue::presented(), heap.count());
    }
}

/// `XNW_GPU_CHECK=1`.
pub(crate) fn gpu_check_on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_GPU_CHECK").is_ok_and(|v| v.trim() == "1"));
    *ON
}

/// Brings `[from, from + len)` of a host copy up to date now.
fn update(shadow: &Shadow, from: usize, len: usize, kind: usize) {
    COPY_KINDS[kind].fetch_add(1, Ordering::Relaxed);
    crate::shared::would_share(kind, from, len);
    let started = std::time::Instant::now();
    let copied = shadow.update(from, len);
    COPY_BYTES[kind].fetch_add(copied as u64, Ordering::Relaxed);
    PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
    PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
    stage(STAGE_COPY, started);
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped is left alone.
///
/// The source is read through the kernel (`ReadProcessMemory`), which
/// reports an unmapped page instead of faulting. Asking `VirtualQuery` first
/// cost about 2 ms per copy in the title's big pools (it walks the page
/// tables to the region's end), which made copies run at 100-200 MB/s.
pub fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    // No VirtualQuery of a title destination first (milliseconds per
    // copy in the big pools: 'copy and fill steps' 333 ms a frame in run
    // 29): the kernel reports an unmapped destination as it does an unmapped
    // source, and a piece that fails there is left alone (zeros are written
    // only into our own memory).
    if len == 0 {
        return;
    }
    // The kernel reads the source: its traps go first.
    super::watch::reading(src as usize, len);
    if title_dst {
        // The kernel writes the destination: it must not be guarded.
        super::watch::writing(dst as usize, len);
    }
    if read(src, dst, len) {
        return;
    }
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read.
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !read(at as *const u8, to, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !read(at as *const u8, to, n) && !title_dst {
                    // SAFETY: inside the destination, our own mapped memory.
                    unsafe { to.write_bytes(0, n) };
                }
            });
        }
    });
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    super::watch::reading(src as usize, dst.len());
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Copies `len` bytes from `src` to `dst` unless a source page is not
/// readable.
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use xnw_core::win::{GetCurrentProcess, ReadProcessMemory};
    let mut done = 0;
    // SAFETY: the destination is mapped writable memory of `len` bytes; the
    // kernel checks the source.
    xnw_core::syscalls::timed(xnw_core::syscalls::Call::ReadCopy, || unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 })
}

/// `NtWriteVirtualMemory`, looked up once. `WriteProcessMemory` asks the
/// kernel about the destination's region first (to flush the instruction
/// cache of executable memory), which in the title's big pools walks page
/// tables for about 2 ms a call; the system call itself writes the same
/// bytes and reports a page it cannot write the same way. Ported from
/// VIctory (`crates/victory-gpu/src/host.rs`, GPL-3.0-or-later;
/// their copy-back went from 1.5 s to 0.13 s); `windows-sys` is
/// `xnw_core::win`.
fn nt_write_virtual_memory() -> Option<NtWrite> {
    static CALL: std::sync::OnceLock<Option<NtWrite>> = std::sync::OnceLock::new();
    *CALL.get_or_init(|| {
        // SAFETY: ntdll is always loaded; the symbol has this signature.
        unsafe {
            let ntdll = xnw_core::win::LoadLibraryA(c"ntdll.dll".as_ptr().cast());
            let f = xnw_core::win::GetProcAddress(ntdll, c"NtWriteVirtualMemory".as_ptr().cast());
            (!f.is_null()).then(|| std::mem::transmute::<*const std::ffi::c_void, NtWrite>(f))
        }
    })
}

type NtWrite = unsafe extern "system" fn(process: xnw_core::win::HANDLE, base: *mut std::ffi::c_void, buffer: *const std::ffi::c_void, len: usize, written: *mut usize) -> i32;

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    use xnw_core::win::{GetCurrentProcess, WriteProcessMemory};
    let nt = nt_write_virtual_memory();
    let write = |at: usize, n: usize| {
        let from = src.wrapping_add(at - dst as usize);
        let mut done = 0;
        // SAFETY: the source is our mapped host copy; the kernel checks the
        // destination.
        unsafe {
            match nt {
                // STATUS_SUCCESS only: a partial copy is a failure here too.
                Some(call) => call(GetCurrentProcess(), at as _, from.cast(), n, &mut done) == 0,
                None => WriteProcessMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) != 0,
            }
        }
    };
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                write(at, n);
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [(0x1, false, 0x4), (0x4, false, 0x8), (0x2, false, 0x10 | 0x20), (0x8, true, 0x40 | 0x80)];
    FORBIDDEN.iter().filter(|&&(flag, deny, _)| (flags & flag != 0) == deny).fold(state, |s, &(_, _, bits)| s & !bits)
}

pub(crate) fn transition(resource: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

#[cfg(test)]
mod batch {
    use super::Batch;
    use std::sync::Arc;

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    #[test]
    fn legal_states_drop_what_the_flags_forbid() {
        // A depth target that is not shader-readable: no shader resource
        // states; a plain texture: no render target or depth states.
        assert_eq!(super::legal_for(0x2 | 0x8, 0x10 | 0x40 | 0x80), 0x10);
        assert_eq!(super::legal_for(0, 0x4 | 0x10 | 0x40), 0x40);
        assert_eq!(super::texture_state(0x0200_0001 | 0x4), 0x4);
    }

    /// A skipped frame's submission does the bottom-of-pipe value
    /// writes (what the title's fence waits read) and the copies into title
    /// memory, and nothing the GPU would do.
    #[test]
    fn a_skipped_submission_signals_and_copies() {
        let Some(host) = super::host() else { return };
        let list = host.new_list().unwrap();
        let mut fence = 0u64;
        let (src, mut dst) = ([5u8; 32], [0u8; 32]);
        list.defer_copy(dst.as_mut_ptr(), src.as_ptr(), 32);
        list.defer_write(std::ptr::from_mut(&mut fence).cast(), super::super::queue::Written::Value(42), 8);
        list.close();
        list.run_skipped();
        assert_eq!(fence, 42, "the value write the title waits on");
        assert_eq!(dst, src, "the copy into title memory");
    }

    /// VIctory's test: mapped pages are written, a page not
    /// committed is skipped, through `NtWriteVirtualMemory`.
    #[test]
    fn write_mapped_writes_mapped_pages_and_skips_the_rest() {
        use xnw_core::win::*;
        const PAGE: usize = 4096;
        // SAFETY: a fresh reservation of ours, two of its pages committed.
        let base = unsafe {
            let base = VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_READWRITE);
            assert!(!base.is_null());
            assert!(!VirtualAlloc(base, 2 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            base.cast::<u8>()
        };
        assert!(super::nt_write_virtual_memory().is_some());
        let data = vec![0xabu8; 3 * PAGE];
        super::write_mapped(base, data.as_ptr(), 3 * PAGE);
        // SAFETY: the two committed pages.
        let written = unsafe { std::slice::from_raw_parts(base, 2 * PAGE) };
        assert!(written.iter().all(|&b| b == 0xab));
        // SAFETY: our reservation.
        unsafe { VirtualFree(base.cast(), 0, 0x8000) };
    }

    #[test]
    fn copies_read_unmapped_pages_as_zeros() {
        let src = vec![7u8; 100];
        let mut dst = vec![1u8; 100];
        super::copy_mapped(dst.as_mut_ptr(), src.as_ptr(), 100, false);
        assert!(dst.iter().all(|&b| b == 7));
        // Address 0 is never mapped.
        super::copy_mapped(dst.as_mut_ptr(), std::ptr::null(), 100, false);
        assert!(dst.iter().all(|&b| b == 0));
    }
}

/// On the host GPU (skipped without `XNW_AGILITY`): a title list clears a
/// render target, copies a buffer and writes a value; one submission runs
/// them in order, and the target reads back as the clear colour.
#[cfg(test)]
mod gpu {
    use std::sync::atomic::Ordering;

    use super::{upload_class, upload_pool};
    use windows::core::Interface;

    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;

    use crate::d3d12x::object::{HRESULT, S_OK, This};
    use crate::d3d12x::resource::{self, HeapProperties, ResourceDesc};
    use crate::d3d12x::{descriptor, list, queue::Written};

    fn slot<F: Copy>(object: This, offset: usize) -> F {
        // SAFETY: our own objects, called with their slots' signatures.
        unsafe { crate::d3d12x::object::slot(object, offset) }
    }

    fn committed(desc: &ResourceDesc) -> This {
        let heap = HeapProperties { kind: 1, cpu_pages: 0, pool: 0, creation_nodes: 1, visible_nodes: 1 };
        let mut out = std::ptr::null();
        let hr = resource::create_committed(std::ptr::null(), &heap, 0, desc, 0, std::ptr::null(), &resource::RESOURCE_IIDS[0], &mut out);
        assert_eq!(hr, S_OK);
        out
    }

    #[test]
    fn upload_classes_and_the_pool_reuse_buffers() {
        assert_eq!(upload_class(1), 0);
        assert_eq!(upload_class(64 << 10), 0);
        assert_eq!(upload_class((64 << 10) + 1), 1);
        assert_eq!(upload_class(3 << 20), 6);
        if std::env::var_os("XNW_AGILITY").is_none_or(|d| d.is_empty()) || super::host().is_none() {
            println!("SKIPPED: set XNW_AGILITY to the Agility SDK folder to run on the host GPU");
            return;
        }
        // A class no other test uses (5 MiB rounds to 8 MiB).
        let a = upload_pool(5 << 20).expect("an upload buffer");
        let raw = a.buffer.as_raw();
        drop(a);
        let b = upload_pool(6 << 20).expect("an upload buffer");
        assert_eq!(b.buffer.as_raw(), raw, "given back, then reused");
    }

    /// Constant fills that continue each other in upload memory and in
    /// title memory become one copy; one that does not stays apart.
    #[test]
    fn adjacent_fills_merge_into_one_copy() {
        if std::env::var_os("XNW_AGILITY").is_none_or(|d| d.is_empty()) {
            println!("SKIPPED: set XNW_AGILITY to the Agility SDK folder to run on the host GPU");
            return;
        }
        let Some(host) = super::host() else { panic!("no host submission queue") };
        let list = host.new_list().unwrap();
        let title: Vec<u8> = (0..2048u32).map(|i| i as u8).collect();
        let base = title.as_ptr();
        let mut pieces = Vec::new();
        // 100 bytes at 0, 100 at 256 (upload padding = title gap), 16 at 700.
        for (offset, len) in [(0usize, 100usize), (256, 100), (700, 16)] {
            let (_, to) = list.scratch(len as u64).unwrap();
            // SAFETY: inside `title`.
            list.defer_fill(to, unsafe { base.add(offset) }, len);
            pieces.push((to, offset, len));
        }
        assert_eq!(list.deferred().copies.len(), 2, "the second fill merged into the first");
        list.close();
        host.execute(&[&list]);
        for (to, offset, len) in pieces {
            // SAFETY: the list's upload memory, filled at submission.
            assert_eq!(unsafe { std::slice::from_raw_parts(to, len) }, &title[offset..offset + len]);
        }
        list.reset();
    }

    /// Culling constants are written after the dispatch is
    /// recorded): constants filled from title memory are read at
    /// submission, not when recorded, through either view of memory mapped
    /// twice (the paired reservation's CPU and GPU views): written through
    /// one view after recording, the other view's fill has the new bytes.
    #[test]
    fn constants_written_after_recording_reach_the_gpu_through_both_views() {
        if std::env::var_os("XNW_AGILITY").is_none_or(|d| d.is_empty()) {
            println!("SKIPPED: set XNW_AGILITY to the Agility SDK folder to run on the host GPU");
            return;
        }
        let Some(host) = super::host() else { panic!("no host submission queue") };
        use xnw_core::win;
        const LEN: usize = 64 << 10;
        // SAFETY: plain section creation and two mappings of it.
        let (primary, alias) = unsafe {
            let section = win::CreateFileMappingW(win::INVALID_HANDLE_VALUE, std::ptr::null(), win::PAGE_READWRITE, 0, LEN as u32, std::ptr::null());
            assert!(!section.is_null());
            let a = win::MapViewOfFile(section, win::FILE_MAP_WRITE, 0, 0, LEN).cast::<u8>();
            let b = win::MapViewOfFile(section, win::FILE_MAP_WRITE, 0, 0, LEN).cast::<u8>();
            win::CloseHandle(section);
            (a, b)
        };
        assert!(!primary.is_null() && !alias.is_null());
        let list = host.new_list().unwrap();
        // A 456-byte block at +96 of a 576-byte one, zero when recorded.
        let block = 0x1000;
        let (_, through_alias) = list.scratch(456).unwrap();
        let (_, through_primary) = list.scratch(456).unwrap();
        // SAFETY: inside both views.
        unsafe {
            list.defer_fill(through_alias, alias.add(block + 96), 456);
            list.defer_fill(through_primary, primary.add(block + 2048 + 96), 456);
        }
        list.close();
        // Written after recording, before submission: through the primary
        // view for the alias's fill, through the alias for the primary's.
        let late: Vec<u8> = (0..456u32).map(|i| (i * 7 + 1) as u8).collect();
        // SAFETY: inside both views.
        unsafe {
            std::ptr::copy_nonoverlapping(late.as_ptr(), primary.add(block + 96), 456);
            std::ptr::copy_nonoverlapping(late.as_ptr(), alias.add(block + 2048 + 96), 456);
        }
        host.execute(&[&list]);
        // SAFETY: the list's upload memory, filled at submission.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(through_alias, 456), &late[..], "read through the alias at submission");
            assert_eq!(std::slice::from_raw_parts(through_primary, 456), &late[..], "read through the primary at submission");
        }
        list.reset();
        // SAFETY: our own views.
        unsafe {
            win::UnmapViewOfFile(primary.cast());
            win::UnmapViewOfFile(alias.cast());
        }
    }

    #[test]
    fn clears_copies_and_writes_run_in_order() {
        if std::env::var_os("XNW_AGILITY").is_none_or(|d| d.is_empty()) {
            println!("SKIPPED: set XNW_AGILITY to the Agility SDK folder to run on the host GPU");
            return;
        }
        let Some(host) = super::host() else { panic!("no host submission queue") };
        let null = std::ptr::null();
        // A 64x64 RGBA8 render target and two 256-byte buffers.
        let target = committed(&ResourceDesc { dimension: 3, alignment: 0, width: 64, height: 64, depth_or_array_size: 1, mip_levels: 1, format: 28, sample_count: 1, sample_quality: 0, layout: 0, flags: 1 });
        let buffer = ResourceDesc { dimension: 1, alignment: 0, width: 256, height: 1, depth_or_array_size: 1, mip_levels: 1, format: 0, sample_count: 1, sample_quality: 0, layout: 1, flags: 0 };
        let (src, dst) = (committed(&buffer), committed(&buffer));
        let (from, _) = resource::buffer_bytes(src).unwrap();
        let (to, _) = resource::buffer_bytes(dst).unwrap();
        // SAFETY: 256-byte buffers of ours.
        unsafe { from.write_bytes(0x5a, 256) };
        // A render target view in a descriptor slot of ours.
        let view = vec![0u8; 128];
        descriptor::create_rtv(null, target, std::ptr::null(), view.as_ptr() as u64);
        // The list: clear (0x1a0), buffer copy (0x78), close (0x48).
        let mut allocator = null;
        assert_eq!(list::create_allocator(null, 0, &list::ALLOCATOR_IIDS[0], &mut allocator), S_OK);
        let mut l = null;
        assert_eq!(list::create_list(null, 0, 0, allocator, null, &list::LIST_IIDS[1], &mut l), S_OK);
        let before = list::RECORDED[0].load(Ordering::Relaxed);
        let colour = [0.25f32, 0.5, 0.75, 1.0];
        slot::<extern "system" fn(This, u64, *const [f32; 4], u32, *const u8)>(l, 0x1a0)(l, view.as_ptr() as u64, &colour, 0, null.cast());
        assert!(list::RECORDED[0].load(Ordering::Relaxed) > before, "the clear was recorded");
        slot::<extern "system" fn(This, This, u64, This, u64, u64)>(l, 0x78)(l, dst, 0, src, 0, 256);
        let mut flag = 0u64;
        list::host_list(l).unwrap().defer_write((&raw mut flag).cast(), Written::Value(9), 8);
        assert_eq!(slot::<extern "system" fn(This) -> HRESULT>(l, 0x48)(l), S_OK);
        // SAFETY: not written before the submission.
        assert_eq!(unsafe { to.read() }, 0, "the copy waits for the submission");
        host.execute(&[list::host_list(l).unwrap()]);
        assert_eq!(flag, 9, "the bottom-of-pipe write landed");
        // SAFETY: 256 bytes copied.
        assert!(unsafe { std::slice::from_raw_parts(to, 256) }.iter().all(|&b| b == 0x5a), "the buffer copy ran");

        // Read the target back: its host copy is in the state the list left.
        let copy = resource::host_copy(target).unwrap();
        let state = copy.state.load(Ordering::Relaxed);
        assert_eq!(state, 0x4, "the clear left it a render target");
        let readback = super::host_device().unwrap().buffer(256 * 64, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST).unwrap();
        let back = host.new_list().unwrap();
        let location = |r: &ID3D12Resource, footprint: bool| D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::ManuallyDrop::new(Some(r.clone())),
            Type: if footprint { D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT } else { D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX },
            Anonymous: if footprint {
                D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: DXGI_FORMAT_R8G8B8A8_UNORM, Width: 64, Height: 64, Depth: 1, RowPitch: 256 },
                    },
                }
            } else {
                D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 }
            },
        };
        let (mut a, mut b) = (location(&readback, true), location(&copy.texture, false));
        // SAFETY: an open list, live resources; the locations are released once.
        unsafe {
            back.list.ResourceBarrier(&[super::transition(&copy.texture, D3D12_RESOURCE_STATES(state as i32), D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            back.list.CopyTextureRegion(&a, 0, 0, 0, &b, None);
            back.list.ResourceBarrier(&[super::transition(&copy.texture, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATES(state as i32))]);
            std::mem::ManuallyDrop::drop(&mut a.pResource);
            std::mem::ManuallyDrop::drop(&mut b.pResource);
        }
        back.close();
        host.execute(&[&back]);
        let mut data = std::ptr::null_mut();
        // SAFETY: a read-back buffer the finished list copied into.
        let texel = unsafe {
            readback.Map(0, None, Some(&mut data)).unwrap();
            std::slice::from_raw_parts(data.cast::<u8>().add(256 * 33 + 4 * 17), 4).to_vec()
        };
        assert_eq!(texel, [64, 128, 191, 255], "the target holds the clear colour");
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        for o in [l, allocator, target, src, dst] {
            release(o);
        }
    }
}
