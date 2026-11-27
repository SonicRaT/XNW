//! Where a frame's WALL time goes (WALL1): one `frame wall:` line per 60
//! presents, from wall-clock sums taken at the wait sites of the frame's
//! thread graph. Sums are atomics (nanoseconds, `Instant` = the performance
//! counter); a site adds with [`add`] or a [`Scope`]; nothing locks.
//!
//! The graph:
//! - the title's game thread runs its update, then kicks the render thread
//!   and waits for it (`update`, `kick+wait`: the two sum to the interval);
//! - the title's render thread drains the frame's stages (waiting for its
//!   Subrender workers, which run our draw hook and our D3D12.X list
//!   methods), then ends the frame (waiting for its submissions);
//! - the thread that submits calls our `ExecuteCommandLists` and our
//!   `PresentX`; in the screen mode PresentX runs our whole pass (record,
//!   submit, fence);
//! - our recorder thread builds each frame's list off the title's threads;
//! - PASS1 (`passthread=on`, the default): our pass thread draws each list
//!   and leaves the finished picture in a slot, PresentX only takes it; the
//!   show's compose is sent without a wait (`showwait=off`), and so are the
//!   kept UI lists (`uiwait=off`): a later submission, a list reset or our
//!   retire thread waits for them instead (`kept retire wait`).
//!
//! The title rows come from the title meter (`titlemeter.rs`) and the stage
//! meter (`stagemeter.rs`) when they are installed; a row nobody feeds reads 0.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::OnceLock;
use std::time::Instant;

/// Title game thread: the update (wall).
pub const UPDATE: usize = 0;
/// Title game thread: the kick of the render thread and the wait for it.
pub const KICK_WAIT: usize = 1;
/// Title render thread: the stage drain (wall; holds the frame end).
pub const DRAIN: usize = 2;
/// Title render thread: the drain's own CPU.
pub const DRAIN_CPU: usize = 3;
/// Title render thread: the frame end inside the drain (wall).
pub const FRAME_END: usize = 4;
/// Stage callbacks, summed over the threads that ran them: wall and CPU.
pub const STAGE_WALL: usize = 5;
pub const STAGE_CPU: usize = 6;
/// Our hooks on the title's threads (wall): the geometry draw hook, our
/// list's indirect draws, mesh draws, other draws and dispatches, the view maker hook.
pub const HOOK_DRAW: usize = 7;
pub const HOOK_INDIRECT: usize = 8;
pub const HOOK_MESH: usize = 9;
pub const HOOK_OTHER: usize = 10;
pub const HOOK_VIEW: usize = 11;
/// Our `PresentX` (wall), of it our pass and the show (compose + swap chain).
pub const PRESENT: usize = 12;
pub const PASS: usize = 13;
pub const SHOW: usize = 14;
/// Our `ExecuteCommandLists` (wall), of it the kept lists run on the host
/// (waited for), and the part on a thread other than the presenting one.
pub const EXECUTE: usize = 15;
pub const EXECUTE_KEPT: usize = 16;
pub const EXECUTE_OFF: usize = 17;
/// Our recorder thread: a frame's list build (wall).
pub const LIST_BUILD: usize = 18;
/// Our pass: record, fence waits, and the rest (submit, retire, lines).
pub const PASS_RECORD: usize = 19;
pub const PASS_FENCE: usize = 20;
pub const PASS_SUBMIT: usize = 21;
/// PASS1: the pass thread busy (draw, retire, put the picture up), waiting
/// for a list, and of the busy part waiting for the show to let go of the
/// picture before the one after it is written.
pub const PASS_THREAD: usize = 22;
pub const PASS_IDLE: usize = 23;
pub const PASS_PUT: usize = 24;
/// The show's parts: the compose's submission (its GPU wait with
/// `showwait=on`) and the swap chain's Present.
pub const SHOW_COMPOSE: usize = 25;
pub const SHOW_SWAP: usize = 26;
/// Waits for a submission sent without one (a kept UI list, a compose):
/// at a later submission, a list reset or release, a compose.
pub const KEPT_WAIT: usize = 27;
/// PERF4: of our list draws (indirect, mesh, other), the draw tap's sinks
/// (binding gathered and handed to the consumers: draws a consumer wants
/// only, so the timer stays off the other calls), and of them the bridged
/// rows and the pre-skin joins.
pub const SINK: usize = 28;
pub const SINK_BRIDGE: usize = 29;
pub const SINK_PRESKIN: usize = 30;
/// PERF5: the bridged sink by part (summed over the title's workers):
/// root constants, frame-table views (of it the big copies, read once a
/// frame, and the sinks that waited for another's copy), constant buffers,
/// kept views (table and root views kept across frames), the draw words
/// and their heap views, the sampler heap.
pub const SINK_ROOTS: usize = 31;
pub const SINK_FVIEWS: usize = 32;
pub const SINK_BIG_COPY: usize = 33;
pub const SINK_BIG_WAIT: usize = 34;
pub const SINK_CB: usize = 35;
pub const SINK_KEPT: usize = 36;
pub const SINK_WORDS: usize = 37;
pub const SINK_SAMPLERS: usize = 38;
/// PERF5 `bigcopy=drain`: the big frame-table views copied at the drain cut (render thread).
pub const DRAIN_BIG_COPY: usize = 39;
pub const N: usize = 40;

/// Presents per line.
pub const EVERY: u64 = 60;

static NS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static PRESENTS: AtomicU64 = AtomicU64::new(0);
/// Lists the pass thread drew in the window.
static PICTURES: AtomicU64 = AtomicU64::new(0);
/// Nanoseconds since [`START`] at the window's first present.
static WINDOW_AT: AtomicU64 = AtomicU64::new(0);
static START: OnceLock<Instant> = OnceLock::new();
/// The thread of the last present (0: none yet).
static PRESENT_THREAD: AtomicU32 = AtomicU32::new(0);

/// `ns` more at `slot`.
#[inline]
pub fn add(slot: usize, ns: u64) {
    NS[slot].fetch_add(ns, Relaxed);
}

/// Milliseconds more at `slot`.
#[inline]
pub fn add_ms(slot: usize, ms: f64) {
    add(slot, (ms.max(0.0) * 1e6) as u64);
}

/// Adds its lifetime to a slot.
pub struct Scope(usize, Instant);

#[inline]
pub fn scope(slot: usize) -> Scope {
    Scope(slot, Instant::now())
}

impl Drop for Scope {
    #[inline]
    fn drop(&mut self) {
        add(self.0, self.1.elapsed().as_nanos() as u64);
    }
}

/// The pass thread drew a list (PASS1).
#[inline]
pub fn picture() {
    PICTURES.fetch_add(1, Relaxed);
}

/// An `ExecuteCommandLists` of ours ended after `ns`, `kept_ns` of it on the host.
pub fn executed(ns: u64, kept_ns: u64) {
    add(EXECUTE, ns);
    add(EXECUTE_KEPT, kept_ns);
    // SAFETY: a plain query of the calling thread.
    if PRESENT_THREAD.load(Relaxed) != unsafe { crate::win::GetCurrentThreadId() } {
        add(EXECUTE_OFF, ns);
    }
}

/// A window's means: milliseconds a frame per slot, the present interval, the presents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Means {
    pub ms: [f64; N],
    pub interval: f64,
    pub frames: u64,
    /// Lists the pass thread drew in the window (PASS1; 0 with the pass in PresentX).
    pub pictures: u64,
}

/// One `PresentX` begins: the line of the last [`EVERY`] presents when due.
/// `XNW_FRAME_WALL=off` gives none.
pub fn present_tick() -> Option<String> {
    static OFF: OnceLock<bool> = OnceLock::new();
    if *OFF.get_or_init(|| matches!(std::env::var("XNW_FRAME_WALL").as_deref().map(str::trim), Ok("off" | "0"))) {
        return None;
    }
    // SAFETY: a plain query of the calling thread.
    PRESENT_THREAD.store(unsafe { crate::win::GetCurrentThreadId() }, Relaxed);
    let now = START.get_or_init(Instant::now).elapsed().as_nanos() as u64;
    let n = PRESENTS.fetch_add(1, Relaxed) + 1;
    if n == 1 {
        WINDOW_AT.store(now, Relaxed);
        return None;
    }
    if n <= EVERY {
        return None;
    }
    // `EVERY` intervals lie between the window's first present and this one, the next window's first.
    PRESENTS.store(1, Relaxed);
    let since = WINDOW_AT.swap(now, Relaxed);
    let mut m = Means { ms: [0.0; N], interval: now.saturating_sub(since) as f64 / 1e6 / EVERY as f64, frames: EVERY, pictures: PICTURES.swap(0, Relaxed) };
    for (a, c) in m.ms.iter_mut().zip(&NS) {
        *a = c.swap(0, Relaxed) as f64 / 1e6 / EVERY as f64;
    }
    // Nothing of ours or of the title measured (front end without a pass): no line.
    (m.ms[PASS_RECORD] > 0.0 || m.ms[UPDATE] > 0.0 || m.ms[HOOK_DRAW] > 0.0).then(|| line(&m))
}

/// The submitting side's parts by size: (name, ms). With the pass thread
/// (PASS1) the pass is not on the submitting side: the picture's take is.
fn submit_parts(m: &Means) -> [(&'static str, f64); 5] {
    let ms = &m.ms;
    if ms[PASS_THREAD] > 0.0 {
        return [
            ("present: take our picture", ms[PASS]),
            ("present: show compose", ms[SHOW_COMPOSE]),
            ("present: show swap chain", ms[SHOW_SWAP]),
            ("present: show other", (ms[SHOW] - ms[SHOW_COMPOSE] - ms[SHOW_SWAP]).max(0.0)),
            ("execute", ms[EXECUTE]),
        ];
    }
    [
        ("present: our pass record", ms[PASS_RECORD]),
        ("present: our pass gpu fence", ms[PASS_FENCE]),
        ("present: our pass submit", ms[PASS_SUBMIT]),
        ("present: show", ms[SHOW]),
        ("execute", ms[EXECUTE]),
    ]
}

fn largest(parts: &[(&'static str, f64)]) -> &'static str {
    parts.iter().copied().fold(("-", f64::MIN), |a, b| if b.1 > a.1 { b } else { a }).0
}

/// The render thread's wait for its Subrender workers: the drain less its frame end and its own CPU.
pub fn subrender_wait(m: &Means) -> f64 {
    (m.ms[DRAIN] - m.ms[FRAME_END] - m.ms[DRAIN_CPU]).max(0.0)
}

/// The wait that bounds the frame: a serial thread busy for most of the
/// interval, named by its largest part.
/// 1. our submissions and presents fill 80 % of the interval: the submitting thread;
/// 2. the render thread's drain fills 80 %: its larger wait, the workers
///    (ours when our hooks are half of the stages' wall) or the frame end;
/// 3. else the title's game update.
pub fn critical(m: &Means) -> &'static str {
    let ms = &m.ms;
    let i = m.interval;
    if i <= 0.0 {
        return "unknown";
    }
    if ms[PRESENT] + ms[EXECUTE] >= 0.8 * i {
        return largest(&submit_parts(m));
    }
    if ms[UPDATE] == 0.0 && ms[DRAIN] == 0.0 {
        return "title (not metered: XNW_TITLE_TIME)";
    }
    if ms[DRAIN] >= 0.8 * i {
        let sub = subrender_wait(m);
        if ms[FRAME_END] >= sub {
            return if ms[PRESENT] + ms[EXECUTE] >= 0.5 * ms[FRAME_END] { largest(&submit_parts(m)) } else { "frame end (title)" };
        }
        let hooks = [("subrender: our draw hook", ms[HOOK_DRAW]), ("subrender: our indirect draws", ms[HOOK_INDIRECT]), ("subrender: our mesh draws", ms[HOOK_MESH]), ("subrender: our other draws", ms[HOOK_OTHER]), ("subrender: our view hook", ms[HOOK_VIEW])];
        let all: f64 = hooks.iter().map(|h| h.1).sum();
        return if ms[STAGE_WALL] > 0.0 && all >= 0.5 * ms[STAGE_WALL] { largest(&hooks) } else { "subrender stages (title)" };
    }
    "title game update"
}

/// The line of a window.
pub fn line(m: &Means) -> String {
    let ms = &m.ms;
    let hooks = ms[HOOK_DRAW] + ms[HOOK_INDIRECT] + ms[HOOK_MESH] + ms[HOOK_OTHER] + ms[HOOK_VIEW];
    format!(
        "frame wall: {} frames, interval {:.1} ms = title game {:.1} + render {:.1} (render thread: drain {:.1}, of it own cpu {:.1}, waiting on subrender {:.1}, frame end {:.1}; stages {:.1} wall / {:.1} cpu over the workers, waiting in our hooks {:.1} [draw {:.1}, indirect {:.1}, mesh {:.1}, other draws {:.1}, view {:.1}; of the draws in our sinks {:.1} (bridged {:.1}, pre-skin {:.1})]; present {:.1} [pass {:.1}, show {:.1} (compose {:.1}, swap chain {:.1})], execute {:.1} [kept {:.1}, off the present thread {:.1}]) ; ours: list build {:.1}, record {:.1}, pass submit {:.1}, gpu fence {:.1} ; pass thread: busy {:.1}, idle {:.1}, of busy waiting for the show {:.1}, {} lists drawn ; kept retire wait {:.1} ; bridged sink: roots {:.1}, frame views {:.1} (big copy {:.1}, waited {:.1}; at the drain {:.1}), constant buffers {:.1}, kept views {:.1}, draw words {:.1}, samplers {:.1} ; critical: {}",
        m.frames,
        m.interval,
        ms[UPDATE],
        ms[KICK_WAIT],
        ms[DRAIN],
        ms[DRAIN_CPU],
        subrender_wait(m),
        ms[FRAME_END],
        ms[STAGE_WALL],
        ms[STAGE_CPU],
        hooks,
        ms[HOOK_DRAW],
        ms[HOOK_INDIRECT],
        ms[HOOK_MESH],
        ms[HOOK_OTHER],
        ms[HOOK_VIEW],
        ms[SINK],
        ms[SINK_BRIDGE],
        ms[SINK_PRESKIN],
        ms[PRESENT],
        ms[PASS],
        ms[SHOW],
        ms[SHOW_COMPOSE],
        ms[SHOW_SWAP],
        ms[EXECUTE],
        ms[EXECUTE_KEPT],
        ms[EXECUTE_OFF],
        ms[LIST_BUILD],
        ms[PASS_RECORD],
        ms[PASS_SUBMIT],
        ms[PASS_FENCE],
        ms[PASS_THREAD],
        ms[PASS_IDLE],
        ms[PASS_PUT],
        m.pictures,
        ms[KEPT_WAIT],
        ms[SINK_ROOTS],
        ms[SINK_FVIEWS],
        ms[SINK_BIG_COPY],
        ms[SINK_BIG_WAIT],
        ms[DRAIN_BIG_COPY],
        ms[SINK_CB],
        ms[SINK_KEPT],
        ms[SINK_WORDS],
        ms[SINK_SAMPLERS],
        critical(m)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn means(interval: f64, rows: &[(usize, f64)]) -> Means {
        let mut m = Means { ms: [0.0; N], interval, frames: 60, pictures: 0 };
        for &(i, v) in rows {
            m.ms[i] = v;
        }
        m
    }

    /// a test run's busy window at 88 s (title time, title stages, frameskip, pass means), as the line prints it.
    #[test]
    fn the_line_of_a_frame_bound_by_our_pass() {
        let m = means(
            44.0,
            &[
                (UPDATE, 14.5), (KICK_WAIT, 29.8), (DRAIN, 42.3), (DRAIN_CPU, 3.0), (FRAME_END, 16.8), (STAGE_WALL, 124.7), (STAGE_CPU, 71.3),
                (HOOK_DRAW, 3.1), (HOOK_INDIRECT, 27.5), (HOOK_MESH, 8.8), (HOOK_OTHER, 0.4), (HOOK_VIEW, 0.1), (SINK, 30.2), (SINK_BRIDGE, 25.0), (SINK_PRESKIN, 1.5),
                (SINK_ROOTS, 0.5), (SINK_FVIEWS, 9.0), (SINK_BIG_COPY, 6.0), (SINK_BIG_WAIT, 2.5), (SINK_CB, 3.0), (SINK_KEPT, 4.0), (SINK_WORDS, 2.0), (SINK_SAMPLERS, 0.4),
                (PRESENT, 32.0), (PASS, 31.6), (SHOW, 0.4), (EXECUTE, 10.1), (EXECUTE_KEPT, 9.0),
                (LIST_BUILD, 16.0), (PASS_RECORD, 31.2), (PASS_FENCE, 0.1), (PASS_SUBMIT, 0.3),
            ],
        );
        assert_eq!(
            line(&m),
            "frame wall: 60 frames, interval 44.0 ms = title game 14.5 + render 29.8 (render thread: drain 42.3, of it own cpu 3.0, waiting on subrender 22.5, frame end 16.8; stages 124.7 wall / 71.3 cpu over the workers, waiting in our hooks 39.9 [draw 3.1, indirect 27.5, mesh 8.8, other draws 0.4, view 0.1; of the draws in our sinks 30.2 (bridged 25.0, pre-skin 1.5)]; present 32.0 [pass 31.6, show 0.4 (compose 0.0, swap chain 0.0)], execute 10.1 [kept 9.0, off the present thread 0.0]) ; ours: list build 16.0, record 31.2, pass submit 0.3, gpu fence 0.1 ; pass thread: busy 0.0, idle 0.0, of busy waiting for the show 0.0, 0 lists drawn ; kept retire wait 0.0 ; bridged sink: roots 0.5, frame views 9.0 (big copy 6.0, waited 2.5; at the drain 0.0), constant buffers 3.0, kept views 4.0, draw words 2.0, samplers 0.4 ; critical: present: our pass record"
        );
    }

    #[test]
    fn the_critical_wait_is_named_by_the_busy_thread() {
        // Our pass waits for its fence most of the frame.
        assert_eq!(critical(&means(40.0, &[(PRESENT, 34.0), (PASS_RECORD, 9.0), (PASS_FENCE, 24.0), (UPDATE, 14.0), (DRAIN, 38.0)])), "present: our pass gpu fence");
        // The submitting side is light; the render thread waits for its workers, who sit in our list methods.
        let hooks = means(30.0, &[(UPDATE, 8.0), (KICK_WAIT, 22.0), (DRAIN, 28.0), (DRAIN_CPU, 3.0), (FRAME_END, 2.0), (STAGE_WALL, 100.0), (HOOK_INDIRECT, 45.0), (HOOK_DRAW, 10.0), (PRESENT, 3.0)]);
        assert_eq!(subrender_wait(&hooks), 23.0);
        assert_eq!(critical(&hooks), "subrender: our indirect draws");
        // The same with small hooks: the title's own stages.
        let mut title = hooks;
        (title.ms[HOOK_INDIRECT], title.ms[HOOK_DRAW]) = (5.0, 3.0);
        assert_eq!(critical(&title), "subrender stages (title)");
        // The frame end waits and nothing of ours explains it.
        assert_eq!(critical(&means(30.0, &[(UPDATE, 8.0), (DRAIN, 28.0), (DRAIN_CPU, 3.0), (FRAME_END, 20.0), (PRESENT, 2.0)])), "frame end (title)");
        // The frame end waits behind our executes.
        assert_eq!(critical(&means(30.0, &[(UPDATE, 8.0), (DRAIN, 28.0), (DRAIN_CPU, 3.0), (FRAME_END, 20.0), (PRESENT, 2.0), (EXECUTE, 14.0)])), "execute");
        // The render thread idles: the game update is the frame (a test run's quiet windows are drain-bound, not this).
        assert_eq!(critical(&means(20.0, &[(UPDATE, 15.0), (KICK_WAIT, 5.0), (DRAIN, 9.0), (PRESENT, 2.0)])), "title game update");
        // No title meter and a light pass.
        assert_eq!(critical(&means(20.0, &[(PRESENT, 3.0), (PASS_RECORD, 2.0)])), "title (not metered: XNW_TITLE_TIME)");
        assert_eq!(critical(&means(0.0, &[])), "unknown");
        // PASS1: the pass on its thread; the present is the show's compose wait.
        assert_eq!(critical(&means(30.0, &[(PRESENT, 20.0), (PASS, 0.1), (SHOW, 19.5), (SHOW_COMPOSE, 13.0), (SHOW_SWAP, 6.0), (EXECUTE, 5.0), (PASS_THREAD, 28.0), (PASS_RECORD, 20.0)])), "present: show compose");
    }

    /// Sixty presents give one line; the sums start again.
    #[test]
    fn a_line_every_sixty_presents() {
        add(PASS_RECORD, 60 * 2_000_000);
        {
            let _s = scope(PRESENT);
        }
        executed(60 * 1_000_000, 0);
        let lines: Vec<String> = (0..=2 * EVERY).filter_map(|_| present_tick()).collect();
        assert_eq!(lines.len(), 1, "the second window has nothing measured: {lines:?}");
        assert!(lines[0].starts_with("frame wall: 60 frames, interval ") && lines[0].contains("record 2.0, "), "{}", lines[0]);
        // The execute came before any present: on another thread than the presenting one.
        assert!(lines[0].contains("execute 1.0 [kept 0.0, off the present thread 1.0]"), "{}", lines[0]);
    }
}
