//! Command queues. Every title queue submits to the one host queue
//! (`host.rs`): lists run in submission order and are waited for, so fence
//! signals that follow complete at once, and the lists' bottom-of-pipe
//! memory writes land when the submission returns. `PresentX` shows every
//! plane's host texture, composited, in the title's window (`present.rs`),
//! logs the frame with what was dropped since the last line, and paces.
//!
//! Ported from VIctory-base (VIctory main d2a4bae + 26 PRs, branch
//! `xnw/base` @ 5d0516e, GPL-3.0-or-later), `crates/victory-gpu/src/
//! queue.rs`: layout, both slot layouts, descriptions, signals (1-235),
//! `Written` (186-232), `Plane` (135-145), `PresentX` (248-343), page
//! mappings (382-450), tests (506-588); PRs touching that file: #135,
//! #211, #413, #638. Changed: the frame line of host work and the
//! `PresentX` log lines are ours; planes carry their destination rectangle.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicU64, Ordering};

use super::object::{self, Done, E_INVALIDARG, HRESULT, Iid, Object, QUEUE, S_OK, Slot, Table, This, guid};
use super::resource::PAGEABLE_IIDS;

pub(crate) const QUEUE_IIDS: [Iid; 4] = [
    guid(0x0ec870a6, 0x5d7e, 0x4c22, [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed]),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QueueDesc {
    pub kind: u32,
    pub priority: i32,
    pub flags: u32,
    pub nodes: u32,
}

/// The title reads the list type right after the header.
#[repr(C)]
struct Queue {
    object: Object,
    kind: u32,
    desc: QueueDesc,
}
const _: () = assert!(std::mem::offset_of!(Queue, kind) == 0x10);

fn queue<'a>(this: This) -> &'a Queue {
    // SAFETY: called from queue methods only.
    unsafe { &*this.cast::<Queue>() }
}

/// Device method 0x40.
pub(crate) extern "system" fn create(_: This, desc: *const QueueDesc, iid: *const Iid, out: *mut This) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let functions = if super::device::newer() { &NEWER_FUNCTIONS } else { &FUNCTIONS };
    let queue = Box::new(Queue { object: Object::new(functions), kind: desc.kind, desc });
    object::hand_out(QUEUE, queue, iid, &QUEUE_IIDS, out)
}

/// The console's queue description: an engine or pipe and a queue on it
/// in place of PC's priority and node mask.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QueueDescX {
    pub kind: u32,
    pub flags: u32,
    pub engine_or_pipe: u32,
    pub queue: u32,
}

/// Device method 0x180 `CreateCommandQueueX`; the host picks engines itself.
pub(crate) extern "system" fn create_x(this: This, desc: *const QueueDescX, iid: *const Iid, out: *mut This) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&x) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let desc = QueueDesc { kind: x.kind, priority: 0, flags: x.flags, nodes: 1 };
    create(this, &desc, iid, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(QUEUE, iid, object::is(iid, &QUEUE_IIDS).then_some(this), out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Queue>(this)
}

/// `ExecuteCommandLists`: runs the lists' host lists in order on the one
/// host queue and waits, then does their deferred memory writes.
extern "system" fn execute(this: This, count: u32, lists: *const This) {
    execute_token(this, count, lists, 0);
}

/// Both submissions, with the frame token (0: none).
fn execute_token(_: This, count: u32, lists: *const This, token: u64) {
    if lists.is_null() {
        return;
    }
    let Some(host) = super::host::host() else {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| xnw_core::warn!("d3d12x: command lists submitted; no host GPU runs them"));
        return;
    };
    // SAFETY: the title passes `count` lists.
    let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
    let host_lists: Vec<_> = lists.iter().filter_map(|&l| super::list::host_list(l)).collect();
    // The lists' pad canaries (`XNW_OBJECT_GUARD`).
    if super::guard::mode() != 0 {
        lists.iter().for_each(|&l| super::list::guard_check(l));
    }
    // VIctory: a list's memory waits are met before it runs: a
    // value an earlier list of this submission writes is met in order;
    // else the value is waited for on this thread (no lock held: another
    // thread's submission writes it), up to 2 s.
    // In a skipped (or null) frame nothing waited on may ever be
    // written (its writer's GPU work never runs, or comes later): a wait
    // not met is forced, never slept on.
    let skipped = super::frameskip::skip(token);
    let number = SUBMITTED_NO.fetch_add(1, Ordering::Relaxed) + 1;
    for (k, list) in host_lists.iter().enumerate() {
        for (address, comparison, reference, wide) in list.waits() {
            if host_lists[..k].iter().any(|l| l.writes_to(address)) {
                continue;
            }
            if skipped {
                force_wait(address, comparison, reference, wide);
            } else {
                wait_for(address, comparison, reference, wide, number, k);
            }
        }
    }
    // What our records say wrote each address (for wait diagnoses).
    for list in &host_lists {
        for address in list.write_addresses() {
            writer_note(address, Writer::Submitted(number));
        }
    }
    // `XNW_FRAMESKIP`: a skipped frame's submission does only what
    // the title's CPU and later frames need; no host GPU work.
    if skipped {
        // The frame lines go on in skipped frames.
        frame_line();
        let at = std::time::Instant::now();
        // `ui=keep`: lists holding kept UI work run on the host
        // (their dropped stretches hold state and barriers only); the
        // others do their CPU part.
        let keep_on = super::uikeep::on();
        let mut kept = Vec::new();
        for list in &host_lists {
            if keep_on && list.kept.load(Ordering::Relaxed) {
                kept.push(*list);
            } else {
                list.run_skipped();
            }
        }
        if keep_on {
            super::uikeep::SKIPPED.fetch_add((host_lists.len() - kept.len()) as u64, Ordering::Relaxed);
            if !kept.is_empty() {
                super::uikeep::EXECUTED.fetch_add(kept.len() as u64, Ordering::Relaxed);
                super::resource::release_held();
                super::uikeep::executing(|| host.execute(&kept));
            }
        }
        SUBMISSIONS.fetch_add(1, Ordering::Relaxed);
        super::frameskip::ours(at.elapsed().as_micros() as u64);
        return;
    }
    // References put off by views made since, before any lock is taken.
    super::resource::release_held();
    frame_line();
    host.execute(&host_lists);
    SUBMISSIONS.fetch_add(1, Ordering::Relaxed);
    LISTS.fetch_add(host_lists.len() as u64, Ordering::Relaxed);
    super::counters::submitted(host_lists.len());
    if FIRST.swap(false, Ordering::Relaxed) {
        xnw_core::info!("d3d12x: first submission executed on the host ({count} lists, {} with host lists)", host_lists.len());
    }
}

static FIRST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Memory waits met at once, waited for, given up on (2 s);
/// forced in skipped frames, polled 100 ms or more.
pub(crate) static WAITS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Waits until title memory at `address` passes the comparison.
/// The value a wait reads at `address` (32 or 64 bits).
fn wait_read(address: u64, wide: bool) -> Option<u64> {
    if wide {
        let mut v = [0u8; 8];
        super::host::read_into(address as *const u8, &mut v).then(|| u64::from_le_bytes(v))
    } else {
        let mut v = [0u8; 4];
        super::host::read_into(address as *const u8, &mut v).then(|| u64::from(u32::from_le_bytes(v)))
    }
}

/// A wait of a skipped or null frame: met now, or forced (counted,
/// the first 8 logged); never slept on.
fn force_wait(address: u64, comparison: u32, reference: u64, wide: bool) -> bool {
    let value = wait_read(address, wide);
    if value.is_some_and(|v| super::list::compare(comparison, v, reference)) {
        WAITS[0].fetch_add(1, Ordering::Relaxed);
        return false;
    }
    if WAITS[3].fetch_add(1, Ordering::Relaxed) < 8 {
        xnw_core::warn!(
            "d3d12x: memory wait forced in a skipped frame: {address:#x} (comparison {comparison}, reference {reference:#x}, value {}; last writer {})",
            value.map_or("unreadable".into(), |v| format!("{v:#x}")),
            writer_text(address)
        );
    }
    true
}

/// A rendered frame's wait: polled up to 2 s (every one over 100 ms is
/// logged with what our records say wrote the address last).
fn wait_for(address: u64, comparison: u32, reference: u64, wide: bool, submission: u64, list: usize) {
    let Some(first) = wait_read(address, wide) else { return };
    if super::list::compare(comparison, first, reference) {
        WAITS[0].fetch_add(1, Ordering::Relaxed);
        return;
    }
    let started = std::time::Instant::now();
    let mut met = false;
    let mut last = first;
    while started.elapsed() < std::time::Duration::from_secs(2) {
        std::thread::sleep(std::time::Duration::from_micros(200));
        if let Some(v) = wait_read(address, wide) {
            last = v;
            if super::list::compare(comparison, v, reference) {
                met = true;
                break;
            }
        }
    }
    let ms = started.elapsed().as_millis();
    WAITS[if met { 1 } else { 2 }].fetch_add(1, Ordering::Relaxed);
    if ms >= 100 {
        WAITS[4].fetch_add(1, Ordering::Relaxed);
        xnw_core::warn!(
            "d3d12x: memory wait {} after {ms} ms: {address:#x} (comparison {comparison}, reference {reference:#x}), value {first:#x} at start, {last:#x} at end; submission {submission} list {list}; last writer {}",
            if met { "met" } else { "NOT met" },
            writer_text(address)
        );
    }
}

/// Submissions numbered, for wait diagnoses.
static SUBMITTED_NO: AtomicU64 = AtomicU64::new(0);

/// What our records say last wrote an address a wait may look at.
#[derive(Clone, Copy)]
pub(crate) enum Writer {
    /// A bottom-of-pipe write recorded in a list not yet submitted.
    Recorded,
    /// Written by submission `n` (run or skipped).
    Submitted(u64),
}
static WRITERS: std::sync::Mutex<Option<std::collections::HashMap<u64, Writer>>> = std::sync::Mutex::new(None);

/// Notes a bottom-of-pipe write to `address`.
pub(crate) fn writer_note(address: u64, w: Writer) {
    let mut g = WRITERS.lock().unwrap_or_else(|e| e.into_inner());
    let map = g.get_or_insert_with(Default::default);
    if map.len() > 1 << 16 {
        map.clear();
    }
    map.insert(address, w);
}

fn writer_text(address: u64) -> String {
    match WRITERS.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|m| m.get(&address).copied()) {
        Some(Writer::Recorded) => "a write recorded in a list not yet submitted (list +0x238/+0x240)".into(),
        Some(Writer::Submitted(n)) => format!("the bottom-of-pipe write of submission {n} (list +0x238/+0x240)"),
        None => "nobody (no bottom-of-pipe write to it recorded)".into(),
    }
}
/// Submissions and lists executed on the host since the last frame line.
static SUBMISSIONS: AtomicU64 = AtomicU64::new(0);
static LISTS: AtomicU64 = AtomicU64::new(0);
/// Draws and dispatches run on the host up to the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

/// Once per presented frame, at the frame's first submission: what the
/// frame before executed on the host, for frames 1-4 and every power of
/// two (as `PresentX` logs what was dropped).
fn frame_line() {
    static LAST: AtomicU64 = AtomicU64::new(0);
    let frame = presented();
    let last = LAST.swap(frame, Ordering::Relaxed);
    if frame == last {
        return;
    }
    let recorded: Vec<u64> = super::list::RECORDED.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
    let (submissions, lists) = (SUBMISSIONS.swap(0, Ordering::Relaxed), LISTS.swap(0, Ordering::Relaxed));
    let dropped = super::draw::take_dropped();
    let indirect = super::list::take_indirect();
    let (drawn, dispatched) = (super::draw::drawn(), super::draw::dispatched());
    let drawn = drawn - DRAWN.swap(drawn, Ordering::Relaxed);
    let dispatched = dispatched - DISPATCHED.swap(dispatched, Ordering::Relaxed);
    let us: Vec<u64> = super::host::SUBMISSION_US.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
    frame_stats(frame, us[0], us[1]);
    super::pipeline::note_present(frame);
    if frame <= 4 || frame.is_power_of_two() {
        let what: Vec<String> = super::list::RECORDED_NAMES.iter().zip(&recorded).map(|(n, c)| format!("{c} {n}")).collect();
        xnw_core::info!(
            "d3d12x: frame {frame} executed on the host: {submissions} submissions, {lists} lists, {}; drawn {drawn}, dispatched {dispatched}; dropped {dropped} draws and dispatches, {indirect} indirect executions; {} us preparing, {} us on the GPU",
            what.join(", "),
            us[0],
            us[1]
        );
    }
}

/// With `XNW_FRAME_STATS=1`, every 30 frames or 10 s: the interval between frames
/// (min/avg/max ms), host preparing and GPU time per frame (avg/max), and
/// per frame the title memory copied, texture memory hashed and texture
/// memory detiled (uploaded).
fn frame_stats(frame: u64, preparing_us: u64, gpu_us: u64) {
    use std::sync::Mutex;
    use std::time::Instant;
    struct Window {
        last: Option<Instant>,
        frames: u64,
        min: f64,
        max: f64,
        sum: f64,
        prep: u64,
        prep_max: u64,
        gpu: u64,
        gpu_max: u64,
        base: [u64; 3],
        pacing: [u64; 4],
        over50: u64,
    }
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_FRAME_STATS").is_ok_and(|v| v == "1"));
    static W: Mutex<Option<Window>> = Mutex::new(None);
    if !*ON {
        return;
    }
    let counters = || {
        let p = &super::host::PREPARING;
        [p[1].load(Ordering::Relaxed), p[3].load(Ordering::Relaxed), super::texture::DETILED.load(Ordering::Relaxed)]
    };
    let mut w = W.lock().unwrap_or_else(|e| e.into_inner());
    let w = w.get_or_insert_with(|| Window { last: None, frames: 0, min: f64::MAX, max: 0.0, sum: 0.0, prep: 0, prep_max: 0, gpu: 0, gpu_max: 0, base: counters(), pacing: [0; 4], over50: 0 });
    let now = Instant::now();
    if let Some(last) = w.last.replace(now) {
        let ms = (now - last).as_secs_f64() * 1000.0;
        w.min = w.min.min(ms);
        w.max = w.max.max(ms);
        w.sum += ms;
        w.frames += 1;
        w.prep += preparing_us;
        w.prep_max = w.prep_max.max(preparing_us);
        w.gpu += gpu_us;
        w.gpu_max = w.gpu_max.max(gpu_us);
        // Where a frame's time went, and every interval over 100 ms.
        let pacing: Vec<u64> = super::frame::PACING.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
        for (sum, p) in w.pacing.iter_mut().zip(&pacing) {
            *sum += p;
        }
        w.over50 += u64::from(ms > 50.0);
        if ms > 100.0 {
            let parts: Vec<String> = super::frame::PACING_NAMES.iter().zip(&pacing).map(|(n, us)| format!("{n} {:.1}", *us as f64 / 1000.0)).collect();
            xnw_core::info!("frame spike: frame {frame} {ms:.1} ms after the one before; in it (ms): {}, preparing {:.1}, GPU {:.1}", parts.join(", "), preparing_us as f64 / 1000.0, gpu_us as f64 / 1000.0);
        }
    }
    // 30 frames, or 10 s of slow frames: a heavy stretch still reports.
    if w.frames >= 30 || w.sum >= 10_000.0 {
        let c = counters();
        let n = w.frames;
        let per = |i: usize| (c[i] - w.base[i]) / n / 1024;
        xnw_core::info!(
            "frame stats to {frame}: interval {:.1}/{:.1}/{:.1} ms (min/avg/max); preparing {}/{} us, GPU {}/{} us (avg/max); per frame: copied {} KiB, textures hashed {} KiB, detiled {} KiB",
            w.min,
            w.sum / n as f64,
            w.max,
            w.prep / n,
            w.prep_max,
            w.gpu / n,
            w.gpu_max,
            per(0),
            per(1),
            per(2)
        );
        // Where the preparing goes: per frame, each stage's time and count.
        let stages: Vec<String> = super::host::STAGE_NAMES
            .iter()
            .zip(&super::host::STAGES)
            .filter_map(|(name, [us, count])| {
                let (us, count) = (us.swap(0, Ordering::Relaxed), count.swap(0, Ordering::Relaxed));
                (count > 0).then(|| format!("{name} {} us/{}", us / n, count.div_ceil(n)))
            })
            .collect();
        let pacing: Vec<String> = super::frame::PACING_NAMES.iter().zip(&w.pacing).map(|(name, us)| format!("{name} {} us", us / n)).collect();
        xnw_core::info!("frame pacing to {frame} (per frame): {}; {} intervals over 50 ms; {}", pacing.join(", "), w.over50, super::frame::refreshes_text());
        xnw_core::info!("frame stages to {frame} (per frame, us/count): {}; {}", stages.join(", "), super::host::texture_events_text(n));
        super::counters::lines(frame, n);
        xnw_core::info!("frame prebuild to {frame}: {}", super::pipeline::prebuild_text());
        xnw_core::info!("frame bindless to {frame} (per frame): {}; {}", super::descriptor::host::stats_text(n), super::resource::held_text() + "; " + &super::descriptor::host::view_fix_text() + "; " + &super::descriptor::gate::stats_text());
        xnw_core::info!("frame addresses to {frame} (totals): {}", super::draw::addresses_text());
        if crate::shared::page_table_on() {
            xnw_core::info!("frame page table to {frame}: {}", super::draw::page_table_text());
        }
        if super::descriptor::host::gpu_mode() {
            // XNW: GPU-side descriptors, totals since start.
            let (draws, no_window, remade) = super::draw::State::gpu_counts();
            let nulls = super::descriptor::null_records_text();
            xnw_core::info!("frame gpu descriptors to {frame} (totals): {draws} draws and dispatches bound, {no_window} dropped (table not in a shared window), {remade} view slots remade at a draw; records written as nulls by kind: {nulls}; constant records as raw views {}, left null {}", super::descriptor::host::RAW_CONSTANTS[0].load(Ordering::Relaxed), super::descriptor::host::RAW_CONSTANTS[1].load(Ordering::Relaxed));
        }
        xnw_core::info!("frame preparing to {frame} (per frame, us/count): {}", super::host::phases_text(n, w.prep / n) + &super::counters::copy_back_text(n));
        let roles = super::roles::stats_text(n);
        if !roles.is_empty() {
            xnw_core::info!("frame roles to {frame} (per frame): {roles}");
        }
        xnw_core::info!("frame origins to {frame} (per frame): {}", super::texture::origin_events_text(n));
        xnw_core::info!("frame zero-copy to {frame}: {}", crate::shared::zero_copy_text(n));
        (w.frames, w.min, w.max, w.sum, w.prep, w.prep_max, w.gpu, w.gpu_max, w.base, w.pacing, w.over50) = (0, f64::MAX, 0.0, 0.0, 0, 0, 0, 0, c, [0; 4], 0);
    }
}

/// `ExecuteCommandListsX(count, lists, frame token, flags)`.
extern "system" fn execute_x(this: This, count: u32, lists: *const This, token: u64, _flags: u32) {
    execute_token(this, count, lists, token);
}
/// Signal with sync settings for the top and bottom of the pipe; our
/// submissions are complete when they return, so it signals at once.
extern "system" fn signal2_x(this: This, fence: This, value: u64, _bottom: *const u8, _top: *const u8) -> HRESULT {
    signal(this, fence, value)
}
/// Markers and events only matter to profilers.
extern "system" fn marker(_: This) {}
extern "system" fn signal(_: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    super::fence::signal(fence, value)
}
/// Every earlier submission has completed, so a queue wait never blocks.
extern "system" fn wait(_: This, fence: This, _value: u64) -> HRESULT {
    if fence.is_null() { E_INVALIDARG } else { S_OK }
}
const TIMESTAMP_FREQUENCY: u64 = 100_000_000;
extern "system" fn timestamp_frequency(_: This, out: *mut u64) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    unsafe { out.write(TIMESTAMP_FREQUENCY) };
    S_OK
}
/// Returns its struct through a hidden pointer after `this`.
extern "system" fn desc(this: This, out: *mut QueueDesc) -> *mut QueueDesc {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(queue(this).desc) };
    }
    out
}

/// What a list's bottom-of-pipe write stores when the list has run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Written {
    Value(u64),
    /// The GPU clock: [`TIMESTAMP_FREQUENCY`] ticks per second since the
    /// first write (our GPU work is done on the CPU's timeline).
    GpuTimestamp,
    /// The CPU's performance counter.
    SystemTimestamp,
}

impl Written {
    /// Stores the low `bytes` (4 or 8) of the value at `at`.
    pub(crate) fn write(self, at: *mut u8, bytes: usize) {
        let value = match self {
            Written::Value(v) => v,
            Written::GpuTimestamp => {
                static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
                (elapsed.as_nanos() / (1_000_000_000 / u128::from(TIMESTAMP_FREQUENCY))) as u64
            }
            Written::SystemTimestamp => {
                let mut count = 0;
                // SAFETY: a valid out pointer.
                let _ = unsafe { windows::Win32::System::Performance::QueryPerformanceCounter(&mut count) };
                count as u64
            }
        };
        // SAFETY: the title names writable GPU memory (unified with ours).
        unsafe {
            if bytes == 4 {
                at.cast::<u32>().write_unaligned(value as u32);
            } else {
                at.cast::<u64>().write_unaligned(value);
            }
        }
    }
}

/// A run of destination pages: how they are mapped (0 pool pages from
/// `start` on, 1 pool page `start` each, 2 unmapped, 3 left as they are)
/// and how many.
#[repr(C)]
pub(crate) struct PageRange {
    kind: u32,
    start: u32,
    count: u32,
}

/// The ranges mapped from a destination address on.
#[repr(C)]
pub(crate) struct PageBatch {
    destination: u64,
    count: u32,
    ranges: *const PageRange,
}

/// Bytes per mapped page.
const PAGE: u64 = 64 << 10;

/// `CopyPageMappingsBatchX(count, batches, pool, pool pages, flags)`: maps
/// pages of a pool (memory the title mapped physical pages into) at the
/// batches' addresses. The title streams textures into pool pages and maps
/// them into its tiled heap this way (title+0xf2c470). Done at once: every
/// earlier submission has run.
/// Without a page mapper (the runtime wires `resource::use_page_mapper` from xmem)
/// it fails with E_INVALIDARG, as the base does.
extern "system" fn copy_page_mappings(_: This, count: u32, batches: *const PageBatch, pool: u64, pool_pages: u32, _flags: u32) -> HRESULT {
    let Some(map) = super::resource::page_mapper() else {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| xnw_core::warn!("d3d12x: page mapping copies refused (no page mapper yet)"));
        return E_INVALIDARG;
    };
    if batches.is_null() && count > 0 {
        return E_INVALIDARG;
    }
    let batches = (count > 0).then(|| {
        // SAFETY: the title passes `count` batches.
        unsafe { std::slice::from_raw_parts(batches, count as usize) }
    });
    let mut ok = true;
    for b in batches.into_iter().flatten() {
        if b.ranges.is_null() {
            continue;
        }
        // SAFETY: each batch holds `count` ranges.
        let ranges = unsafe { std::slice::from_raw_parts(b.ranges, b.count as usize) };
        let mut at = b.destination;
        for r in ranges {
            let len = u64::from(r.count) * PAGE;
            let inside = |pages: u32| r.start.checked_add(pages).is_some_and(|e| e <= pool_pages);
            let source = pool + u64::from(r.start) * PAGE;
            ok &= match r.kind {
                0 if inside(r.count) => map(at, source, len),
                1 if inside(1) => (0..u64::from(r.count)).all(|k| map(at + k * PAGE, source, PAGE)),
                2 => map(at, 0, len),
                3 => true,
                _ => false,
            };
            at += len;
        }
    }
    if !ok {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| xnw_core::warn!("d3d12x: a page mapping failed (pool {pool:#x})"));
    }
    S_OK
}

/// Frames presented so far.
static FRAMES: AtomicU64 = AtomicU64::new(0);

pub(crate) fn presented() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// Frames presented so far (the runner's stall watch).
pub fn frames_presented() -> u64 {
    presented()
}

/// A frame passes (tests that check per-frame work).
#[cfg(test)]
pub(crate) fn pass_frame() {
    FRAMES.fetch_add(1, Ordering::Relaxed);
}

/// One plane of a `PresentX` call (`D3D12XBOX_PRESENT_PLANE_PARAMETERS`,
/// 0x48 bytes): frame token, resource array, view rects, color space,
/// extended descriptors, and plane flags. Base queue.rs 135-145; the
/// fields past `resources` named from five-star d3d12x-shim.md
/// "PresentX" and `PresentPlane.cs` (the title fills them at
/// 0x142482313 [I]).
#[repr(C)]
pub(crate) struct Plane {
    pub(crate) token: u64,
    pub(crate) count: u32,
    pub(crate) resources: *const This,
    /// `RECT*`: the part of the resource to show, {0, 0, width, height}.
    pub(crate) source: *const [i32; 4],
    /// `float[4]*`: where it goes, normalized; {0, 0, 1, 1}.
    pub(crate) destination: *const [f32; 4],
    pub(crate) color_space: u32,
    _rest: [u32; 7],
}
const _: () = assert!(std::mem::size_of::<Plane>() == 0x48);
const _: () = assert!(std::mem::offset_of!(Plane, source) == 0x18 && std::mem::offset_of!(Plane, destination) == 0x20 && std::mem::offset_of!(Plane, color_space) == 0x28);

impl Plane {
    /// The plane's source rectangle, if it names one.
    pub(crate) fn source(&self) -> Option<[i32; 4]> {
        // SAFETY: null or the title's rectangle, alive through the call.
        unsafe { self.source.as_ref() }.copied()
    }

    /// The plane's destination rectangle; the whole back buffer without
    /// one or for one that is empty, outside [0, 1] or not a number.
    pub(crate) fn placement(&self) -> super::present::Placement {
        // SAFETY: null or the title's four floats, alive through the call.
        let Some(&r) = (unsafe { self.destination.as_ref() }) else { return super::present::FULL };
        let ok = r.iter().all(|v| (0.0..=1.0).contains(v)) && r[2] > r[0] && r[3] > r[1];
        if ok { r } else { super::present::FULL }
    }
}

/// `PresentX(planes, plane parameters, parameters)`: shows every plane's
/// host texture (composited, `present.rs`), logs the first frames and
/// every power of two after, with the commands dropped since the last
/// line, then paces to the frame interval. Base queue.rs 248-343 (the
/// base's diagnostics-build show, dump and blink modes are not taken).
extern "system" fn present(_: This, planes: u32, plane_parameters: *const Plane, _parameters: *const u8) -> HRESULT {
    // The frame's decision (its token, as its submissions'), taken
    // before the frame count moves (untokened frames are keyed by it).
    let skipped = {
        // SAFETY: the title passes `planes` plane descriptions (null tolerated).
        let token = if planes > 0 && !plane_parameters.is_null() { unsafe { (*plane_parameters).token } } else { 0 };
        super::frameskip::skip_present(token)
    };
    let at = std::time::Instant::now();
    let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    super::frameskip::presented(skipped);
    // `ui=keep`: a null frame with nothing dropped (front end,
    // loading) is shown as the title drew it; one with dropped world work
    // shows ours under the title's UI plane.
    let mut skipped = skipped;
    if skipped && super::uikeep::on() {
        super::uikeep::tick();
        if super::uikeep::frame_all_kept() {
            skipped = false;
        } else if planes == 1 {
            super::uikeep::single_plane();
            skipped = false;
        } else if planes >= 2
            && !plane_parameters.is_null()
            && let Some(host) = super::host::host()
        {
            const COPY_SOURCE: u32 = 0x800;
            const RENDER_TARGET: u32 = 0x4;
            // SAFETY: the title passes `planes` plane descriptions.
            let slice = unsafe { std::slice::from_raw_parts(plane_parameters, planes.min(8) as usize) };
            // SDR is {scene, UI}; HDR {UI, scene} with the scene in colour space 12.
            let (scene, ui) = if slice[1].color_space == 12 { (1, 0) } else { (0, 1) };
            let texture = |p: &Plane| -> Option<(This, &'static windows::Win32::Graphics::Direct3D12::ID3D12Resource)> {
                if p.count == 0 || p.resources.is_null() {
                    return None;
                }
                // SAFETY: the plane lists `count` resources.
                let r = unsafe { *p.resources };
                super::resource::host_resource(r).map(|t| (r, t))
            };
            let ours = super::screen_source().and_then(|f| f());
            let mut shown: Vec<(This, u32)> = Vec::new();
            let mut sources = Vec::new();
            if let Some(o) = &ours {
                sources.push((o, RENDER_TARGET, super::present::FULL));
            } else if let Some((r, t)) = texture(&slice[scene]) {
                let state = super::resource::host_transition(r, COPY_SOURCE).map_or(COPY_SOURCE, |(_, b)| b);
                shown.push((r, state));
                sources.push((t, state, slice[scene].placement()));
            }
            if let Some((r, t)) = texture(&slice[ui]) {
                super::resource::resident(super::resource::host_copy(r).iter());
                let state = super::resource::host_transition(r, COPY_SOURCE).map_or(COPY_SOURCE, |(_, b)| b);
                shown.push((r, state));
                sources.push((t, state, slice[ui].placement()));
            }
            if frame <= 4 || frame.is_power_of_two() || frame.is_multiple_of(100) {
                xnw_core::info!("PresentX frame {frame} ({planes} planes): null GPU, ui=keep: {} under the title's UI plane {ui}", if ours.is_some() { "our picture" } else { "the title's scene plane" });
            }
            if !sources.is_empty() {
                let _ = host.present(&sources, frame);
            }
            for (r, state) in shown {
                super::resource::host_transition(r, state);
            }
            super::frameskip::ours(at.elapsed().as_micros() as u64);
            return S_OK;
        }
    }
    if skipped {
        // The frame count is visible in null mode too (`xnw smoke`).
        if frame <= 4 || frame.is_power_of_two() || frame.is_multiple_of(100) {
            xnw_core::info!("PresentX frame {frame} ({planes} planes): skipped (frameskip / null GPU)");
        }
        // `XNW_WORLD_RENDER=screen`: our world pass's picture is shown
        // as the scene plane of this skipped (null-mode) frame.
        if let Some(f) = super::screen_source()
            && let (Some(host), Some(ours)) = (super::host::host(), f())
        {
            const RENDER_TARGET: u32 = 0x4;
            let _ = host.present(&[(&ours, RENDER_TARGET, super::present::FULL)], frame);
        }
        // The last picture stays; no pacing: a skipped frame runs at once.
        super::frameskip::ours(at.elapsed().as_micros() as u64);
        return S_OK;
    }
    if frame <= 4 || frame.is_power_of_two() {
        let c = super::list::take_counts();
        xnw_core::info!(
            "PresentX frame {frame} ({planes} planes): the title made {} draws, {} dispatches; {} other commands not recorded; {} pipelines so far",
            c.draws,
            c.dispatches,
            c.other,
            super::pipeline::created()
        );
    }
    if planes > 0
        && !plane_parameters.is_null()
        && let Some(host) = super::host::host()
    {
        const COPY_SOURCE: u32 = 0x800;
        // SAFETY: the title passes `planes` plane descriptions.
        let slice = unsafe { std::slice::from_raw_parts(plane_parameters, planes.min(8) as usize) };
        super::present::log_planes(slice);
        let mut resources = Vec::with_capacity(slice.len());
        for plane in slice {
            if plane.count == 0 || plane.resources.is_null() {
                continue;
            }
            // SAFETY: the plane lists `count` resources.
            let resource = unsafe { *plane.resources };
            if let Some(texture) = super::resource::host_resource(resource) {
                super::resource::resident(super::resource::host_copy(resource).iter());
                let state = super::resource::host_transition(resource, COPY_SOURCE).map_or(COPY_SOURCE, |(_, before)| before);
                resources.push((resource, texture, state, plane.placement()));
            }
        }
        // `XNW_SHOW`: a chosen render target in place of the planes.
        if let Some(r) = super::draw::shown_target()
            && let Some(texture) = super::resource::host_resource(r)
        {
            for &(resource, _, state, _) in &resources {
                super::resource::host_transition(resource, state);
            }
            let state = super::resource::host_transition(r, COPY_SOURCE).map_or(COPY_SOURCE, |(_, before)| before);
            resources = vec![(r, texture, state, super::present::FULL)];
        }
        if !resources.is_empty() {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                let written = resources.iter().filter(|r| super::texture::is_host_written(r.0)).count();
                xnw_core::info!("present: frame {frame} shown from the host GPU's copies of {} planes ({written} written on the host)", resources.len());
            });
            let sources: Vec<(&_, u32, super::present::Placement)> = resources.iter().map(|(_, tex, state, place)| (*tex, *state, *place)).collect();
            let at = std::time::Instant::now();
            let shown = host.present(&sources, frame);
            super::frame::PACING[super::frame::PACE_PRESENT].fetch_add(at.elapsed().as_micros() as u64, Ordering::Relaxed);
            for &(resource, _, state, _) in &resources {
                super::resource::host_transition(resource, state);
            }
            let _ = shown;
        }
    }
    super::frame::present_pace();
    S_OK
}

/// The queue methods by their slot in older titles' queues: IUnknown 3,
/// object 4, device child 1, queue 26 (12 as on PC, then console ones).
const METHODS: [Slot; 15] = [
    (0, query as *const (), Done, "queue.rs:422"),
    (2, release as *const (), Done, "queue.rs:423"),
    (7, object::parent_device as *const (), Done, "queue.rs:424"),
    (10, execute as *const (), Done, "queue.rs:425"),
    (11, marker as *const (), Done, "queue.rs:426"),
    (12, marker as *const (), Done, "queue.rs:427"),
    (13, marker as *const (), Done, "queue.rs:428"),
    (14, signal as *const (), Done, "queue.rs:429"),
    (15, wait as *const (), Done, "queue.rs:430"),
    (16, timestamp_frequency as *const (), Done, "queue.rs:431"),
    (26, desc as *const (), Done, "queue.rs:432"),
    (29, copy_page_mappings as *const (), Done, "queue.rs:433"),
    (30, present as *const (), Done, "queue.rs:434"),
    (32, signal2_x as *const (), Done, "queue.rs:436"),
    (33, execute_x as *const (), Done, "queue.rs:435"),
];

/// Newer titles' queues have two more methods before Signal: every slot
/// from 0x70 on is two later (VIctory, from the title's calls: Signal
/// 0x80, timestamp frequency 0x90 at title+0x24841a6, PresentX 0x100,
/// the frame-token submission 0x118 at title+0x24821dd).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 14 { slot + 2 } else { slot }
}

const fn layout(newer: bool) -> [Slot; 15] {
    let mut m = METHODS;
    let mut i = 0;
    while newer && i < m.len() {
        m[i].0 = newer_slot(m[i].0);
        i += 1;
    }
    m
}

pub(crate) const SLOTS: [Slot; 15] = layout(false);
pub(crate) const NEWER_SLOTS: [Slot; 15] = layout(true);
pub(crate) const LEN: usize = 36;
static FUNCTIONS: Table<LEN> = object::table::<QUEUE, LEN>(true, &SLOTS);
static NEWER_FUNCTIONS: Table<LEN> = object::table::<QUEUE, LEN>(true, &NEWER_SLOTS);

#[cfg(test)]
mod tests {
    use super::*;

    /// Slot `offset` of one of our objects as `F`.
    fn slot<F: Copy>(object: This, offset: usize) -> F {
        // SAFETY: the tests call our own objects with their slots' signatures.
        unsafe { crate::d3d12x::object::slot(object, offset) }
    }

    /// Where a queue's signal is: newer layouts have it two slots later.
    /// Another test may switch layouts while this one runs, so the queue's
    /// own table decides.
    fn signal_slot(q: This) -> usize {
        // SAFETY: one of our queues; its first field is the table.
        let table = unsafe { *q.cast::<*const *const ()>() };
        if table == NEWER_FUNCTIONS.0.as_ptr() { 0x80 } else { 0x70 }
    }

    /// CPU only: a skipped submission's value write, then a later
    /// submission's wait on it: met at once, no sleep; a wait nobody
    /// writes is forced in a skipped frame and counted.
    #[test]
    fn skipped_waits_never_sleep() {
        let mut cell = 0u64;
        let at = std::ptr::from_mut(&mut cell) as u64;
        // The skipped writer: its bottom-of-pipe write done on the CPU.
        Written::Value(0x519).write(at as *mut u8, 8);
        let started = std::time::Instant::now();
        assert!(!force_wait(at, 7, 0x519, true), "met by the skipped write");
        let mut never = 0u64;
        let forced_before = WAITS[3].load(Ordering::Relaxed);
        assert!(force_wait(std::ptr::from_mut(&mut never) as u64, 7, 0x519, true), "nobody writes it: forced");
        assert_eq!(WAITS[3].load(Ordering::Relaxed), forced_before + 1);
        assert!(started.elapsed() < std::time::Duration::from_millis(50), "no sleep");
        // A rendered frame's wait already met returns at once too.
        wait_for(at, 7, 0x518, true, 1, 0);
        assert!(started.elapsed() < std::time::Duration::from_millis(50));
        assert_eq!(writer_text(std::ptr::from_mut(&mut never) as u64), "nobody (no bottom-of-pipe write to it recorded)");
    }

    #[test]
    fn queue_signals_complete_at_once() {
        let null = std::ptr::null();
        let desc = QueueDesc { kind: 0, priority: 0, flags: 0, nodes: 0 };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        // SAFETY: the title reads the type at +0x10.
        assert_eq!(unsafe { q.cast::<u8>().add(0x10).cast::<u32>().read() }, 0);
        let mut fence = null;
        let fence_iid = &crate::d3d12x::fence::FENCE_IIDS[0];
        assert_eq!(crate::d3d12x::fence::create(null, 0, 0, fence_iid, &mut fence), S_OK);
        let signal = slot::<extern "system" fn(This, This, u64) -> HRESULT>(q, signal_slot(q));
        assert_eq!(signal(q, fence, 7), S_OK);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(fence, 0x40)(fence), 7);
        assert_eq!(signal(q, null, 8), E_INVALIDARG);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(q, 0x10)(q), 0);
    }

    /// The title's compute queue: `{COMPUTE, 0, pipe 1, queue 0}`.
    #[test]
    fn console_queues_are_created_as_on_pc() {
        let null = std::ptr::null();
        let desc = QueueDescX { kind: 2, flags: 0, engine_or_pipe: 1, queue: 0 };
        let mut q = null;
        assert_eq!(create_x(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        assert_eq!(queue(q).desc.kind, 2);
        let none = std::ptr::null();
        assert_eq!(create_x(null, none, &QUEUE_IIDS[0], &mut q), E_INVALIDARG);
    }

    #[test]
    fn newer_queues_have_their_methods_two_slots_later() {
        let (old, new) = (&SLOTS, &NEWER_SLOTS);
        let at = |m: &[Slot], f: *const ()| m.iter().find(|s| s.1 == f).map(|s| s.0);
        assert_eq!(at(old, execute as *const ()), at(new, execute as *const ())); // 0x50 stays
        assert_eq!(at(old, signal as *const ()), Some(14)); // 0x70
        assert_eq!(at(new, signal as *const ()), Some(16)); // 0x80
        assert_eq!(at(new, present as *const ()), Some(32)); // 0x100
        assert_eq!(at(new, execute_x as *const ()), Some(35)); // 0x118
    }

    /// The title's planes (five-star d3d12x-shim.md "PresentX"): source
    /// rect at +0x18 and destination at +0x20 are pointers; read through.
    #[test]
    fn planes_are_read_with_their_rectangles() {
        let resource: This = std::ptr::null();
        let (source, destination) = ([0, 0, 1920, 1080], [0.0f32, 0.0, 1.0, 1.0]);
        let mut bytes = [0u64; 9];
        bytes[0] = 7;
        bytes[1] = 1;
        bytes[2] = (&raw const resource) as u64;
        bytes[3] = (&raw const source) as u64;
        bytes[4] = (&raw const destination) as u64;
        // SAFETY: 0x48 bytes laid out as the title's plane.
        let plane = unsafe { &*bytes.as_ptr().cast::<Plane>() };
        assert_eq!((plane.token, plane.count, plane.color_space), (7, 1, 0));
        assert_eq!(plane.source(), Some(source));
        assert_eq!(plane.placement(), destination);
        let corner = [0.5f32, 0.5, 1.0, 1.0];
        bytes[4] = (&raw const corner) as u64;
        // SAFETY: as above.
        assert_eq!(unsafe { &*bytes.as_ptr().cast::<Plane>() }.placement(), corner);
        let bad = [0.5f32, 0.5, 0.25, 1.0];
        bytes[4] = (&raw const bad) as u64;
        // SAFETY: as above.
        assert_eq!(unsafe { &*bytes.as_ptr().cast::<Plane>() }.placement(), super::super::present::FULL);
        bytes[4] = 0;
        // SAFETY: as above.
        assert_eq!(unsafe { &*bytes.as_ptr().cast::<Plane>() }.placement(), super::super::present::FULL);
    }

    #[test]
    fn timestamps_and_values_are_written() {
        let mut at = [0u64; 2];
        Written::Value(0x1_0000_0002).write(at.as_mut_ptr().cast(), 4);
        assert_eq!(at[0], 2);
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        let first = at[1];
        std::thread::sleep(std::time::Duration::from_millis(2));
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        assert!(at[1] > first);
    }
}
