//! Host pipelines are made by workers, not by the title thread that first
//! binds or draws with them: translation and the host driver's compile take
//! from milliseconds to seconds each.
//!
//! Ported from VIctory `crates/victory-gpu/src/pipeline/make.rs` 1-511 at
//! xnw/base 5d0516e (VIctory d2a4bae; no merged PR touches it),
//! GPL-3.0-or-later. Changed: xnw-core log; `Pending` keys by our shader
//! hashes (hash128); the frame is the presented count; `VICTORY_PSO_WAIT_MS`
//! is `XNW_PSO_WAIT_MS`; workers are also capped by free memory (one per
//! GiB above a 6 GiB reserve, another runtime's idea); `ready` does
//! not wait when the caller says so (bind time: start only).
//!
//! A first use queues the pipeline and waits for it within an atomic per-frame
//! budget shared across all recording threads, so cheap or one-off passes
//! during startup/warmup are ready immediately while steady-state frames stay
//! responsive without multi-hundred-millisecond freezes. Past the budget, a
//! draw or dispatch whose pipeline is not ready is dropped while the worker
//! finishes compiling it in the background.
//!
//! By default the budget is adaptive: [`WARMUP_BUDGET_MS`] (500 ms) during
//! early warmup frames (`frame < 60`) so boot, menu, and one-shot init passes
//! compile reliably, then [`STEADY_BUDGET_MS`] (8 ms) once presenting steadily
//! so live gameplay avoids large compile hitches. Setting
//! `XNW_PSO_WAIT_MS` (`0..=5000`) overrides the per-frame budget for every
//! frame (`0` disables synchronous waiting completely).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};
use super::library;

/// What identifies a title pipeline across runs: its shaders' stages and
/// content hashes and its graphics state (the console description, not the
/// translation, so a translator change does not move it).
pub(crate) fn key_of(shaders: &[Arc<Shader>], graphics: Option<&GraphicsState>) -> String {
    let state = graphics.map(|g| g.describe()).unwrap_or_default();
    let mut parts: Vec<Vec<u8>> = shaders.iter().map(|s| [s.stage.as_bytes(), &s.hash.to_le_bytes()].concat()).collect();
    parts.push(state.into_bytes());
    let parts: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    format!("{:032x}", xnw_core::hash128(&parts))
}

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queued: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    /// Made before the null point, or (KEEP1) made after it for kept
    /// UI work (only those draw kept UI work).
    before_null: AtomicBool,
    /// KEEP1: a kept draw of a named UI-class program asked for it in null
    /// mode ([`make_kept`]): the one kind made after the null point.
    kept_ui: AtomicBool,
    /// Per-pipeline wake pair so unrelated background compiles do not wake
    /// threads waiting on a different pipeline.
    wait_lock: Mutex<()>,
    wait_done: Condvar,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw has used it yet.
    key: String,
    used: AtomicBool,
    /// A draw has asked for it (it is on the pipeline list).
    asked: AtomicBool,
    /// Queued by the prebuilder (not by a bind or a draw).
    prebuilt: AtomicBool,
    /// A draw found it not made (it is built late, in a frame).
    late: AtomicBool,
    /// Microseconds its making took.
    build_us: AtomicU64,
    /// Too slow to wait for: listed so by an earlier run, or past the
    /// compile deadline in this one; its draws drop.
    slow: Arc<AtomicBool>,
    /// A drop for want of it was logged.
    drop_logged: AtomicBool,
}

/// The programs of a pipeline, for logs.
pub(crate) fn label(shaders: &[Arc<Shader>]) -> String {
    shaders.iter().map(|s| format!("{} {:032x}", s.stage, s.hash)).collect::<Vec<_>>().join(" + ")
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let key = key_of(&shaders, graphics.as_deref());
        Pending {
            shaders,
            graphics,
            queued: AtomicBool::new(false),
            made: OnceLock::new(),
            before_null: AtomicBool::new(false),
            kept_ui: AtomicBool::new(false),
            wait_lock: Mutex::new(()),
            wait_done: Condvar::new(),
            used: AtomicBool::new(false),
            asked: AtomicBool::new(false),
            prebuilt: AtomicBool::new(false),
            late: AtomicBool::new(false),
            build_us: AtomicU64::new(0),
            // A suspect of an earlier device loss is never built in-run.
            slow: Arc::new(AtomicBool::new(super::compile::listed_slow(&key) || super::compile::listed_suspect(&key))),
            drop_logged: AtomicBool::new(false),
            key,
        }
    }

    /// The gate the host pipeline was refused at, once it was refused
    /// (translations already made; nothing is translated here).
    pub(crate) fn gate(&self) -> Option<usize> {
        if !matches!(self.made.get(), Some(None)) {
            return None;
        }
        self.shaders.iter().find_map(|s| s.translated.get()?.as_ref().ok().and_then(|t| super::host::gate(&t.host)))
    }

    fn make(&self) {
        // No host pipeline is begun once null mode began (any path),
        // but (KEEP1) one a kept UI draw asked for.
        let kept_ui = self.kept_ui.load(Ordering::Relaxed);
        if super::super::frameskip::null_now() && self.made.get().is_none() && !kept_ui {
            if STARTED_IN_NULL.fetch_add(1, Ordering::Relaxed) == 0 {
                xnw_core::warn!("d3d12x: a host pipeline was asked to be made after the null point: refused ({})", label(&self.shaders));
            }
            return;
        }
        self.made.get_or_init(|| {
            let started = Instant::now();
            // Watched while it runs (slow log, deadline, exit report).
            let ticket = super::compile::begin(&self.key, label(&self.shaders), self.slow.clone());
            let made = HostPipeline::new(&self.shaders, self.graphics.as_deref());
            let sizes: Vec<String> = self.shaders.iter().map(|s| s.translated().map_or_else(|| format!("{} -", s.stage), |t| format!("{} {} B", s.stage, t.host.dxil.len()))).collect();
            ticket.done(&format!("DXIL {}; {}", sizes.join(", "), if made.is_some() { "made" } else { "refused" }));
            if let Some(p) = &made {
                super::note_ready(p);
                // In flight at the null point: made, never used (uikeep.rs).
                if !super::super::frameskip::null_now() {
                    self.before_null.store(true, Ordering::Relaxed);
                } else if kept_ui {
                    self.before_null.store(true, Ordering::Relaxed);
                    let n = UI_MADE_IN_NULL.fetch_add(1, Ordering::Relaxed);
                    if n < 200 {
                        xnw_core::info!("null ui: host pipeline made for kept UI work: {} [{}]", label_of(self), sizes.join(", "));
                    }
                } else {
                    MADE_IN_NULL.fetch_add(1, Ordering::Relaxed);
                }
            }
            self.build_us.store(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            // What rebuilds it without the title, for the pipeline list.
            if let Some(recipe) = super::warm::recipe_of(&self.shaders, self.graphics.as_deref()) {
                library::note_recipe(&self.key, recipe);
            }
            made
        });
        let _held = self.wait_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.wait_done.notify_all();
    }
}


/// Number of initial presented frames that receive [`WARMUP_BUDGET_MS`] so
/// boot, menus, and one-shot initialization passes compile synchronously.
const WARMUP_FRAMES: u64 = 60;
/// Per-frame wait budget during initial warmup frames.
const WARMUP_BUDGET_MS: u32 = 500;
/// Per-frame wait budget during steady-state presentation (keeps cached/fast
/// PSOs immediate without causing multi-frame hitches on slow compiles).
const STEADY_BUDGET_MS: u32 = 8;
/// Upper bound on `XNW_PSO_WAIT_MS` (5 seconds).
const MAX_WAIT_MS: u32 = 5_000;

/// Bit packing for [`FrameBudget`]: upper 36 bits hold the frame tag and lower
/// 28 bits hold reserved microseconds (up to ~268 s per frame).
const RESERVED_BITS: u32 = 28;
const RESERVED_MASK: u64 = (1u64 << RESERVED_BITS) - 1;
const FRAME_TAG_MASK: u64 = (1u64 << (64 - RESERVED_BITS)) - 1;
const MAX_RESERVED_US: u32 = RESERVED_MASK as u32;
/// Sentinel frame tag before any frame has waited.
const NO_FRAME_TAG: u64 = FRAME_TAG_MASK;

/// Lock-free per-frame wait-budget tracker shared across recording threads.
///
/// Packs `(frame_tag, reserved_us)` into a single `AtomicU64` so concurrent
/// callers can never observe a torn frame reset or collectively reserve more
/// than the frame's wait budget. Callers reserve remaining budget upfront and
/// refund unused microseconds if the pipeline finishes early (only if the same
/// frame is still active).
struct FrameBudget {
    state: AtomicU64,
}

impl FrameBudget {
    const fn new() -> Self {
        Self {
            state: AtomicU64::new(NO_FRAME_TAG << RESERVED_BITS),
        }
    }

    fn frame_tag(frame: u64) -> u64 {
        // Modulo `NO_FRAME_TAG` (`(1 << 36) - 1`) keeps the tag strictly in
        // `0..NO_FRAME_TAG` without collisions at wraparound.
        frame % NO_FRAME_TAG
    }

    fn is_newer_tag(tag: u64, cur_tag: u64) -> bool {
        if cur_tag == NO_FRAME_TAG {
            return true;
        }
        let diff = if tag >= cur_tag {
            tag - cur_tag
        } else {
            NO_FRAME_TAG - cur_tag + tag
        };
        diff > 0 && diff < (NO_FRAME_TAG / 2)
    }

    fn pack(tag: u64, reserved_us: u32) -> u64 {
        (tag << RESERVED_BITS) | u64::from(reserved_us)
    }

    fn unpack(raw: u64) -> (u64, u32) {
        (raw >> RESERVED_BITS, (raw & RESERVED_MASK) as u32)
    }

    /// Atomically reserves up to the remaining budget (in microseconds) for
    /// `frame`, resetting the counter only when `frame` advances past the
    /// active frame tag (stale tags from earlier frames receive 0 without
    /// altering active state).
    fn reserve(&self, frame: u64, limit_us: u32) -> u32 {
        let limit = limit_us.min(MAX_RESERVED_US);
        if limit == 0 {
            return 0;
        }
        let tag = Self::frame_tag(frame);
        let mut granted = 0u32;
        let _ = self
            .state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                let (cur_tag, cur_used) = Self::unpack(cur);
                let used = if cur_tag == tag {
                    cur_used
                } else if Self::is_newer_tag(tag, cur_tag) {
                    0
                } else {
                    granted = 0;
                    return None;
                };
                let avail = limit.saturating_sub(used);
                granted = avail;
                if avail == 0 && cur_tag == tag {
                    return None;
                }
                Some(Self::pack(tag, used + avail))
            });
        granted
    }

    /// Returns `unused_us` back to `frame`'s budget if `frame` is still active.
    fn refund(&self, frame: u64, unused_us: u32) {
        if unused_us == 0 {
            return;
        }
        let tag = Self::frame_tag(frame);
        let _ = self
            .state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                let (cur_tag, cur_used) = Self::unpack(cur);
                if cur_tag != tag {
                    return None;
                }
                Some(Self::pack(tag, cur_used.saturating_sub(unused_us)))
            });
    }
}

static FRAME_BUDGET: FrameBudget = FrameBudget::new();

fn parse_wait_ms(value: Option<&str>) -> Option<u32> {
    value?
        .trim()
        .parse::<u32>()
        .ok()
        .map(|ms| ms.min(MAX_WAIT_MS))
}

fn configured_wait_ms() -> Option<u32> {
    static OVERRIDE: OnceLock<Option<u32>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        let raw = std::env::var("XNW_PSO_WAIT_MS").ok();
        let parsed = parse_wait_ms(raw.as_deref());
        if let Some(ms) = parsed {
            xnw_core::info!("d3d12x: XNW_PSO_WAIT_MS: per-frame pipeline wait budget {ms} ms");
        } else if let Some(raw) = raw.filter(|s| !s.trim().is_empty()) {
            xnw_core::warn!("d3d12x: ignoring invalid XNW_PSO_WAIT_MS {raw:?} (expected integer milliseconds)");
        }
        parsed
    })
}

/// `XNW_PIPELINE_WAIT=1`: draws wait for their pipeline within the
/// per-frame budget below, as before. Without it (the default) nothing
/// waits: a pipeline not made is queued and the draw dropped and counted.
/// `XNW_PSO_WAIT_MS` still sets a budget explicitly.
pub(crate) fn blocking() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XNW_PIPELINE_WAIT").is_ok_and(|v| v == "1"))
}

fn frame_budget_us(frame: u64, override_ms: Option<u32>) -> u32 {
    budget_us(frame, override_ms, blocking())
}

fn budget_us(frame: u64, override_ms: Option<u32>, blocking: bool) -> u32 {
    if override_ms.is_none() && !blocking {
        return 0;
    }
    let ms = override_ms.unwrap_or(if frame < WARMUP_FRAMES {
        WARMUP_BUDGET_MS
    } else {
        STEADY_BUDGET_MS
    });
    ms.saturating_mul(1_000)
}

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
/// The host pipeline if it is made already; nothing started.
pub(crate) fn made_now(p: &Arc<Pending>) -> Option<&HostPipeline> {
    if !p.before_null.load(Ordering::Relaxed) {
        return None;
    }
    p.made.get()?.as_ref()
}

pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    // Listed at the first ask, made or not (a pipeline that fails is
    // retried by the prebuild after a translator change).
    if !p.asked.swap(true, Ordering::Relaxed) {
        library::note_used(&p.key);
    }
    // Too slow (listed, or past the deadline): never waited for.
    if p.slow.load(Ordering::Relaxed) {
        note_drop(p, "too slow to compile in-run, or a device-loss suspect");
        return None;
    }
    if p.made.get().is_none() {
        p.late.store(true, Ordering::Relaxed);
        start(p);
        wait(p, super::super::presented());
    }
    let Some(made) = p.made.get() else {
        note_drop(p, "still compiling");
        return None;
    };
    let made = made.as_ref();
    if made.is_some() && !p.used.swap(true, Ordering::Relaxed) {
        // How this pipeline met its first draw.
        if p.late.load(Ordering::Relaxed) {
            LATE.fetch_add(1, Ordering::Relaxed);
            LATE_US.fetch_add(p.build_us.load(Ordering::Relaxed), Ordering::Relaxed);
        } else if p.prebuilt.load(Ordering::Relaxed) {
            PREBUILT_USED.fetch_add(1, Ordering::Relaxed);
        } else {
            BOUND_READY.fetch_add(1, Ordering::Relaxed);
        }
    }
    made
}

/// The first drop for want of `p` is logged with its programs (64
/// pipelines at most).
fn note_drop(p: &Pending, why: &str) {
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if !p.drop_logged.swap(true, Ordering::Relaxed) && LOGGED.fetch_add(1, Ordering::Relaxed) < 64 {
        xnw_core::info!("d3d12x: draws dropped for want of a host pipeline ({why}): {} key {}", label(&p.shaders), p.key);
    }
}

/// `p`'s host pipeline if made (no queueing, no waiting).
pub(crate) fn made(p: &Pending) -> Option<&HostPipeline> {
    p.made.get()?.as_ref()
}

/// Whether `p`'s host pipeline is made (not queued, not waited for).
pub(crate) fn is_made(p: &Pending) -> bool {
    p.made.get().is_some_and(Option::is_some)
}

/// Queues `p` for the workers without waiting (a list binding it).
pub(crate) fn begin(p: &Arc<Pending>) {
    if p.made.get().is_none() {
        start(p);
    }
}

/// Queues `p` for the workers, once.
fn start(p: &Arc<Pending>) {
    // Too slow or a suspect: never built in-run. None in null
    // mode (nothing will draw with it).
    if p.slow.load(Ordering::Relaxed) || super::super::frameskip::null_now() {
        return;
    }
    if !p.queued.swap(true, Ordering::Relaxed) {
        enqueue(p);
    }
}

fn enqueue(p: &Arc<Pending>) {
    PEAK.fetch_max(
        WAITING.fetch_add(1, Ordering::Relaxed) + 1,
        Ordering::Relaxed,
    );
    queue(p.clone());
}

/// The first draws that found their pipeline made by the prebuilder,
/// made by a bind (before the draw), or not made (built late, in a frame,
/// with the microseconds their making took).
pub(crate) static PREBUILT_USED: AtomicU64 = AtomicU64::new(0);
pub(crate) static BOUND_READY: AtomicU64 = AtomicU64::new(0);
pub(crate) static LATE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LATE_US: AtomicU64 = AtomicU64::new(0);
/// Pipelines the prebuilder queued, and of those finished (made or refused).
static PRE_QUEUED: AtomicU64 = AtomicU64::new(0);
pub(crate) static PRE_DONE: AtomicU64 = AtomicU64::new(0);

/// Prebuilds queued and not finished.
pub(crate) fn prebuild_inflight() -> u64 {
    PRE_QUEUED.load(Ordering::Relaxed).saturating_sub(PRE_DONE.load(Ordering::Relaxed))
}

/// Pipelines queued by binds and draws (not by the prebuilder) and not made.
pub(crate) fn demand_waiting() -> u64 {
    waiting().saturating_sub(prebuild_inflight())
}

/// Queues `p` for the workers as a prebuild (once; a bind or draw that
/// came first keeps it).
pub(crate) fn prebuild(p: &Arc<Pending>) {
    // No prebuild in null mode either.
    if super::super::frameskip::null_now() {
        return;
    }
    if p.made.get().is_none() && !p.slow.load(Ordering::Relaxed) && !p.queued.swap(true, Ordering::Relaxed) {
        p.prebuilt.store(true, Ordering::Relaxed);
        PRE_QUEUED.fetch_add(1, Ordering::Relaxed);
        enqueue(p);
    }
}

/// Waits for `p` while frame `frame` has budget left, reserving budget
/// atomically so concurrent recording threads cannot collectively overshoot it.
fn wait(p: &Pending, frame: u64) {
    let limit_us = frame_budget_us(frame, configured_wait_ms());
    let reserved_us = FRAME_BUDGET.reserve(frame, limit_us);
    if reserved_us == 0 {
        return;
    }
    let left = Duration::from_micros(u64::from(reserved_us));
    let started = Instant::now();
    let mut held = p.wait_lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = p
            .wait_done
            .wait_timeout(held, rest)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
    drop(held);
    super::super::host::stage(super::super::host::STAGE_PIPELINE_WAIT, started);
    let elapsed_us = started.elapsed().as_micros().min(u128::from(reserved_us)) as u32;
    FRAME_BUDGET.refund(frame, reserved_us - elapsed_us);
}

/// Pipeline workers once the world draws: half the logical processors, 2
/// to 8 (as before).
fn world_workers() -> usize {
    // XNW_COMPILES (default 3) host pipelines compile at once once
    // the world draws.
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::env::var("XNW_COMPILES").ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|&n| n > 0).unwrap_or(3))
}

/// Pipeline workers before the world draws: logical processors - 2, at
/// least [`world_workers`]; both at most one per GiB of available memory
/// above 6 GiB.
pub(crate) fn workers() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::thread::available_parallelism().map_or(2, |n| n.get().saturating_sub(2)).max(world_workers()).min(memory_workers()))
}

/// Hands `p` to the workers ([`workers`] at normal priority until the world
/// draws, then [`world_workers`] below normal: the rest stop taking work),
/// started on first use.
fn queue(p: Arc<Pending>) {
    static SENDER: OnceLock<std::sync::mpsc::Sender<Arc<Pending>>> = OnceLock::new();
    let sender = SENDER.get_or_init(|| {
        let (sender, receiver) = std::sync::mpsc::channel::<Arc<Pending>>();
        let receiver = Arc::new(Mutex::new(receiver));
        let workers = workers();
        let in_world_workers = world_workers().min(workers);
        xnw_core::info!("d3d12x: {workers} pipeline workers ({in_world_workers} once the world draws)");
        for i in 0..workers {
            let receiver = receiver.clone();
            let spawned = std::thread::Builder::new()
                .name(format!("xnw-pipelines-{i}"))
                .spawn(move || {
                    let set = |below: bool| {
                        // SAFETY: a plain call on our own thread.
                        unsafe {
                            use windows::Win32::System::Threading::*;
                            let _ = SetThreadPriority(GetCurrentThread(), if below { THREAD_PRIORITY_BELOW_NORMAL } else { THREAD_PRIORITY_NORMAL });
                        }
                    };
                    // Normal priority in the front end and the load;
                    // below the title's own threads once the world draws.
                    let mut below = super::prebuild::in_world();
                    set(below);
                    loop {
                        if super::prebuild::in_world() {
                            if i >= in_world_workers {
                                // Not needed in the world: stops for good.
                                loop {
                                    std::thread::park();
                                }
                            }
                            if !below {
                                below = true;
                                set(below);
                            }
                        }
                        let next = receiver.lock().unwrap_or_else(|e| e.into_inner()).recv();
                        let Ok(p) = next else { return };
                        // Queued before null mode began: not made now
                        // (it may be asked for again if frames render).
                        if super::super::frameskip::null_now() && !p.kept_ui.load(Ordering::Relaxed) {
                            p.queued.store(false, Ordering::Relaxed);
                            NULL_DROPPED.fetch_add(1, Ordering::Relaxed);
                            WAITING.fetch_sub(1, Ordering::Relaxed);
                            continue;
                        }
                        p.make();
                        if p.prebuilt.load(Ordering::Relaxed) {
                            PRE_DONE.fetch_add(1, Ordering::Relaxed);
                        }
                        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                            drained();
                        } else {
                            library::save(false);
                        }
                    }
                });
            if let Err(e) = spawned {
                xnw_core::warn!("d3d12x: no pipeline worker: {e}");
            }
        }
        sender
    });
    if let Err(e) = sender.send(p) {
        // No worker: made here instead.
        e.0.make();
        WAITING.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Queued pipelines not made because null mode began.
pub(crate) static NULL_DROPPED: AtomicU64 = AtomicU64::new(0);
/// Host pipelines made after the null point (expected 0).
pub(crate) static MADE_IN_NULL: AtomicU64 = AtomicU64::new(0);
/// KEEP1: host pipelines made after the null point for kept UI work.
pub(crate) static UI_MADE_IN_NULL: AtomicU64 = AtomicU64::new(0);

/// KEEP1: `p` is asked for by a kept draw of a named UI-class program in
/// null mode: queued once for the workers, never waited for (the draw is
/// dropped until it is made). Not the too-slow or suspect ones.
pub(crate) fn make_kept(p: &Arc<Pending>) {
    p.kept_ui.store(true, Ordering::Relaxed);
    if p.made.get().is_none() && !p.slow.load(Ordering::Relaxed) && !p.queued.swap(true, Ordering::Relaxed) {
        static ASKED: AtomicU64 = AtomicU64::new(0);
        if ASKED.fetch_add(1, Ordering::Relaxed) < 200 {
            xnw_core::info!("null ui: host pipeline asked for kept UI work: {}", label_of(p));
        }
        enqueue(p);
    }
}

/// KEEP1: the stage names and the pack name of the last program of `p`
/// (the pixel or compute one), before anything is made.
pub(crate) fn programs(p: &Pending) -> (Vec<&'static str>, Option<String>) {
    (p.shaders.iter().map(|s| s.stage).collect(), p.shaders.last().and_then(|s| s.pack_name()))
}

/// The texture views `p`'s programs read through tables ([`super::texture_tables`]).
pub(crate) fn texture_tables(p: &Pending) -> Option<Vec<(u8, u32)>> {
    let mut v = Vec::new();
    for s in &p.shaders {
        let t = s.translated()?;
        v.extend(t.used.iter().filter(|u| u.kind == xnw_shader::ResourceType::Srv && u.root == xnw_shader::RootType::Table && u.dwords == 8).map(|u| (u.slot, u.table_offset_dwords)));
    }
    Some(v)
}

/// The hash of `p`'s last program.
pub(crate) fn last_hash(p: &Pending) -> Option<u128> {
    p.shaders.last().map(|s| s.hash)
}

/// The programs of `p` by pack name (hash where the pack has none), for a
/// log line.
pub(crate) fn label_of(p: &Pending) -> String {
    let v: Vec<String> = p.shaders.iter().map(|s| format!("{} {}", s.stage, s.pack_name().unwrap_or_else(|| format!("{:032x}", s.hash)))).collect();
    v.join(" + ")
}

/// Whether this is the first note about `p` (one log line per pipeline).
pub(crate) fn first_note(p: &Pending) -> bool {
    !p.drop_logged.swap(true, Ordering::Relaxed)
}

/// Host pipeline makings asked for after the null point, refused.
pub(crate) static STARTED_IN_NULL: AtomicU64 = AtomicU64::new(0);

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use library::{LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        xnw_core::info!(
            "d3d12x: host pipelines all made ({} made, {} loaded, {} prebuilt)",
            MADE.load(Ordering::Relaxed),
            LOADED.load(Ordering::Relaxed),
            PRE_DONE.load(Ordering::Relaxed)
        );
    }
    library::save(false);
}

/// Workers the free memory allows: one per GiB available above a 6 GiB
/// reserve, at least 1 (a driver compile can take gigabytes).
fn memory_workers() -> usize {
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut m = MEMORYSTATUSEX { dwLength: size_of::<MEMORYSTATUSEX>() as u32, ..Default::default() };
    // SAFETY: a plain query into our own struct.
    if unsafe { GlobalMemoryStatusEx(&mut m) }.is_err() {
        return 2;
    }
    ((m.ullAvailPhys >> 30).saturating_sub(6) as usize).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert!(p.queued.load(Ordering::Relaxed));
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    #[test]
    fn concurrent_budget_reservation_never_overshoots() {
        use std::sync::Barrier;

        let budget = FrameBudget::new();
        let limit_us = 8_000u32;
        let threads = 8;
        let barrier = Barrier::new(threads);
        let total_granted = AtomicU64::new(0);

        std::thread::scope(|s| {
            for _ in 0..threads {
                let budget = &budget;
                let barrier = &barrier;
                let total_granted = &total_granted;
                s.spawn(move || {
                    barrier.wait();
                    let g = budget.reserve(100, limit_us);
                    total_granted.fetch_add(u64::from(g), Ordering::SeqCst);
                });
            }
        });
        assert_eq!(
            total_granted.load(Ordering::SeqCst),
            u64::from(limit_us),
            "total reserved across all concurrent threads never exceeds the frame limit"
        );
        // No more budget remains on frame 100 until a caller refunds unused time.
        assert_eq!(budget.reserve(100, limit_us), 0);

        // Refunding unused microseconds makes them available to subsequent waits
        // on the same frame.
        budget.refund(100, 3_500);
        assert_eq!(budget.reserve(100, limit_us), 3_500);

        // Advancing to a new frame resets the budget atomically.
        assert_eq!(budget.reserve(101, limit_us), limit_us);

        // A stale refund from frame 100 must not alter frame 101's budget.
        budget.refund(100, 5_000);
        assert_eq!(budget.reserve(101, limit_us), 0);
    }

    #[test]
    fn stale_reservation_never_rewinds_active_frame_or_wraparound() {
        let budget = FrameBudget::new();
        let limit_us = 8_000u32;

        // Deterministic descheduled-thread interleaving: frame 101 reserves
        // first, then a delayed thread from frame 100 attempts to reserve.
        assert_eq!(budget.reserve(101, limit_us), limit_us);
        assert_eq!(
            (budget.reserve(100, limit_us), budget.reserve(101, limit_us)),
            (0, 0),
        );

        // Partial budget on frame 102 stays intact when stale frame 101 arrives.
        assert_eq!(budget.reserve(102, limit_us), limit_us);
        budget.refund(102, 2_500);
        assert_eq!(budget.reserve(101, limit_us), 0);
        assert_eq!(budget.reserve(102, limit_us), 2_500);

        // Wraparound across `NO_FRAME_TAG` (`(1 << 36) - 1`) boundary.
        let wrap_budget = FrameBudget::new();
        assert_eq!(wrap_budget.reserve(NO_FRAME_TAG - 1, limit_us), limit_us);
        assert_eq!(wrap_budget.reserve(NO_FRAME_TAG, limit_us), limit_us);
        assert_eq!(
            (
                wrap_budget.reserve(NO_FRAME_TAG - 1, limit_us),
                wrap_budget.reserve(NO_FRAME_TAG, limit_us),
            ),
            (0, 0),
        );
        assert_eq!(wrap_budget.reserve(NO_FRAME_TAG + 1, limit_us), limit_us);
        assert_eq!(wrap_budget.reserve(NO_FRAME_TAG, limit_us), 0);
    }

    #[test]
    fn wait_ms_parsing_clamping_and_adaptive_budget() {
        assert_eq!(parse_wait_ms(None), None);
        assert_eq!(parse_wait_ms(Some("")), None);
        assert_eq!(parse_wait_ms(Some("abc")), None);
        assert_eq!(parse_wait_ms(Some("-5")), None);
        assert_eq!(parse_wait_ms(Some("0")), Some(0));
        assert_eq!(parse_wait_ms(Some(" 16 ")), Some(16));
        assert_eq!(parse_wait_ms(Some("999999")), Some(MAX_WAIT_MS));

        // Adaptive defaults: generous warmup, tight steady-state.
        assert_eq!(budget_us(0, None, true), WARMUP_BUDGET_MS * 1_000);
        assert_eq!(
            budget_us(WARMUP_FRAMES - 1, None, true),
            WARMUP_BUDGET_MS * 1_000
        );
        assert_eq!(
            budget_us(WARMUP_FRAMES, None, true),
            STEADY_BUDGET_MS * 1_000
        );
        assert_eq!(budget_us(10_000, None, true), STEADY_BUDGET_MS * 1_000);

        // By default nothing waits.
        assert_eq!(budget_us(0, None, false), 0);
        assert_eq!(budget_us(10_000, None, false), 0);
        // Explicit override applies to every frame (including 0 = never block).
        assert_eq!(budget_us(0, Some(0), false), 0);
        assert_eq!(budget_us(100, Some(25), false), 25_000);
    }
}
