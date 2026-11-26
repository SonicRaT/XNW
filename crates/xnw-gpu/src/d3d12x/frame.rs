//! Frame pacing: the title sets a frame interval and waits for each frame
//! to begin. Ours is a clock started by the first interval set, ticking
//! once per interval; the token is the frame number. Each event kind
//! (origin, vblank, flip) happens once per tick, so two waits for the same
//! kind never end on the same tick. `PresentX` paces too when the title
//! has not waited since the last present (see [`present_pace`]).
//!
//! The ticks are the display's: a thread of ours counts the host
//! output's vertical blanks (`IDXGIOutput::WaitForVBlank`) and a tick is
//! every `divider`-th of them, the divider being the title's interval in
//! display refreshes (2 for 30 fps at 60 Hz, 4 at 120 Hz, 5 at 144 Hz). The
//! compositor presents with that sync interval ([`sync_interval`]), so each
//! title frame stays on screen the same number of refreshes. A free-running
//! clock beats against the display: frames 33.3 ms apart reached a 60 Hz
//! screen 16.7 or 50 ms apart (world-run8: intervals 12.6-16.8 / 33.3 /
//! 50.2-53.3 ms). Where the output gives no vertical blank (no output, a
//! remote session's failing or instant waits), a high-resolution timer at
//! the measured refresh period, aligned to the last blank, stands in.
//! `XNW_PACING=timer` keeps the free-running clock and `Present(0, 0)`.
//! VIctory main (ffe310e) still has the free-running clock and
//! `Present(0, 0)`: no upstream answer to take.
//!
//! Ported from VIctory `crates/victory-gpu/src/frame.rs` at d2a4bae
//! (GPL-3.0-or-later). Added: pacing at present; our display clock.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use super::object::{HRESULT, S_FALSE, S_OK, This};

/// What a wait returns when its time is up first.
const TIMEOUT: HRESULT = S_FALSE;
/// The frame-origin event: the one that starts frames and makes tokens.
const ORIGIN: u32 = 0;
/// Make the frame's token even if the wait times out.
const TOKEN_ON_TIMEOUT: u32 = 2;
pub(crate) const SIXTIETH: Duration = Duration::from_micros(16_667);

struct Clock {
    start: Instant,
    interval: Duration,
    /// The tick number at `start` (ticks go on across interval changes).
    first: u64,
}

static CLOCK: Mutex<Option<Clock>> = Mutex::new(None);
static FRAME: AtomicU64 = AtomicU64::new(0);

fn clock() -> std::sync::MutexGuard<'static, Option<Clock>> {
    CLOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Device method 0x270: a frame every `length` microseconds.
pub(crate) extern "system" fn set_interval(_: This, _output: This, length: u32, _period: u32, _flags: u32) -> HRESULT {
    let interval = match length {
        0 => SIXTIETH,
        us => Duration::from_micros(u64::from(us)),
    };
    xnw_core::info!("d3d12x: frame interval {interval:?}");
    let first = LAST.iter().map(|t| t.load(Ordering::Relaxed)).max().unwrap_or(0);
    *clock() = Some(Clock { start: Instant::now(), interval, first });
    S_OK
}

/// Device method 0x278: events are scheduled at every tick.
pub(crate) extern "system" fn schedule(_: This, _kind: u32, _offset: u32, _list: *const u8, _flags: u32) -> HRESULT {
    S_OK
}

/// Frames begun so far (origin waits that handed out a token).
pub(crate) fn current() -> u64 {
    FRAME.load(Ordering::Relaxed)
}

/// Per event kind (origin, vblank, flip and later kinds; the last slot is
/// presents): the last tick a wait for it ended on.
static LAST: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
const PRESENT: u32 = 3;

fn last(kind: u32) -> &'static AtomicU64 {
    &LAST[(kind as usize).min(LAST.len() - 2)]
}

/// The next tick after now and after the last one `last` ended on, and how
/// long until it.
fn next_tick(last: &AtomicU64) -> (u64, Duration) {
    let mut clock = clock();
    let c = clock.get_or_insert_with(|| Clock { start: Instant::now(), interval: SIXTIETH, first: 0 });
    let l = last.load(Ordering::Relaxed);
    let (tick, wait, rebase) = plan_tick(c.start.elapsed().as_nanos(), c.interval.as_nanos(), c.first, l, late_pacing());
    if rebase {
        last.fetch_max(tick, Ordering::Relaxed);
        // Only the origin wait restarts the clock; the vblank and flip waits just
        // end at once and catch up to the tick in progress.
        if !std::ptr::eq(last, &LAST[ORIGIN as usize]) {
            return (tick, Duration::ZERO);
        }
        let first = LAST.iter().map(|t| t.load(Ordering::Relaxed)).max().unwrap_or(tick);
        *c = Clock { start: Instant::now(), interval: c.interval, first };
    }
    (tick, Duration::from_nanos(wait as u64))
}

/// `XNW_PACING=late`: the timer clock, except that a frame which overran
/// its tick ends its wait at once and the clock restarts at now, instead of
/// waiting for the next tick.
pub(crate) fn late_pacing() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_PACING").is_ok_and(|v| v == "late"));
    *ON
}

/// Tick arithmetic: at `elapsed_ns` on a clock of `interval_ns` whose tick
/// number at start is `first`, with `last` the tick this kind last ended on.
/// Returns the tick the wait ends on, nanoseconds to wait, and whether the
/// clock must restart at now (late mode, the frame overran: the tick in
/// progress is already past `last`).
fn plan_tick(elapsed_ns: u128, interval_ns: u128, first: u64, last: u64, late: bool) -> (u64, u128, bool) {
    let interval = interval_ns.max(1);
    let current = first + (elapsed_ns / interval) as u64;
    if late && current > last {
        return (current, 0, true);
    }
    let tick = (current + 1).max(last + 1);
    let at = u128::from(tick - first) * interval;
    (tick, at.saturating_sub(elapsed_ns), false)
}

/// Blocks until the next tick after now and after the last one `last`
/// ended on, or `limit`; true if the tick came.
fn wait_tick(last: &AtomicU64, limit: Duration) -> bool {
    // `XNW_PACING=free`: no wait at all (measurement: the title's
    // own CPU frame time, unquantised).
    if free_pacing() {
        last.fetch_add(1, Ordering::Relaxed);
        return true;
    }
    if timer_pacing() || late_pacing() {
        let (tick, wait) = next_tick(last);
        if wait > limit {
            std::thread::sleep(limit);
            return false;
        }
        std::thread::sleep(wait);
        last.fetch_max(tick, Ordering::Relaxed);
        return true;
    }
    // Display ticks: `last` holds the blank count a wait ended on.
    start_blanks();
    let d = divider();
    let deadline = Instant::now() + limit;
    let mut count = BLANKS.lock().unwrap_or_else(|e| e.into_inner());
    let target = ((*count).max(last.load(Ordering::Relaxed)) / d + 1) * d;
    while *count < target {
        let now = Instant::now();
        if now >= deadline {
            return false;
        }
        count = BLANK.wait_timeout(count, deadline - now).unwrap_or_else(|e| e.into_inner()).0;
    }
    last.fetch_max(target, Ordering::Relaxed);
    true
}

/// `XNW_PACING=free` (and `freecap`, which also presents at sync interval 1): frame waits return at once, and presents do not
/// wait for a blank. The console (and our default) ends a frame-origin wait
/// at the next interval boundary after now, so a frame that overran its
/// interval waits to the following one: a 34 ms frame at a 31 ms display
/// interval (a 32 Hz output) shows as 62.5 ms. Free shows the 34.
pub(crate) fn free_pacing() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_PACING").is_ok_and(|v| v == "free" || v == "freecap"));
    *ON
}

/// `XNW_PACING=timer`: the free-running clock (the earlier behaviour).
pub(crate) fn timer_pacing() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_PACING").is_ok_and(|v| v == "timer"));
    *ON
}

/// Vertical blanks counted since the display clock started, and the
/// signal at each.
static BLANKS: Mutex<u64> = Mutex::new(0);
static BLANK: Condvar = Condvar::new();
/// The measured refresh period, nanoseconds.
static PERIOD_NS: AtomicU64 = AtomicU64::new(16_666_667);
/// The blanks come from a timer, not the output.
static SOFTWARE: AtomicBool = AtomicBool::new(false);

/// Display refreshes per title frame: the title's interval over the
/// refresh period, rounded, at least 1.
pub(crate) fn divider() -> u64 {
    let interval = clock().as_ref().map_or(SIXTIETH, |c| c.interval).as_nanos() as u64;
    divider_for(interval, PERIOD_NS.load(Ordering::Relaxed))
}

/// Refreshes of `period_ns` per frame of `interval_ns`, rounded, at least 1.
fn divider_for(interval_ns: u64, period_ns: u64) -> u64 {
    let period = period_ns.max(1);
    ((interval_ns + period / 2) / period).max(1)
}

/// The sync interval the compositor presents with: the divider (1-4), or
/// 0 with `XNW_PACING=timer`.
pub(crate) fn sync_interval() -> u32 {
    static CAP: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_PACING").is_ok_and(|v| v == "freecap"));
    sync_for(timer_pacing() || late_pacing() || free_pacing(), *CAP, divider())
}

/// The sync interval: 0 for the free-running modes, 1 for `XNW_PACING=freecap`
/// (no frame waits, each present waits at most for the display's next refresh,
/// none shown twice in one refresh), else the divider capped at 4.
fn sync_for(free_running: bool, cap: bool, divider: u64) -> u32 {
    if cap { 1 } else if free_running { 0 } else { divider.min(4) as u32 }
}

/// Blanks counted so far (starts the counter).
pub(crate) fn blanks() -> u64 {
    start_blanks();
    *BLANKS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The output the host adapter drives (else the primary one).
fn host_output() -> Option<windows::Win32::Graphics::Dxgi::IDXGIOutput> {
    use windows::Win32::Graphics::Dxgi::*;
    // SAFETY: plain DXGI object queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
        let by_luid = super::host::host()
            .and_then(|h| factory.EnumAdapterByLuid::<IDXGIAdapter1>(h.device.GetAdapterLuid()).ok())
            .and_then(|a| a.EnumOutputs(0).ok());
        by_luid.or_else(|| factory.EnumAdapters1(0).ok()?.EnumOutputs(0).ok())
    }
}

/// Starts the thread that counts the display's vertical blanks: the
/// output's `WaitForVBlank`, or where that fails or returns at once eight
/// times in a row (a remote session), a high-resolution timer at the
/// measured period aligned to the last blank. The period is measured from
/// the blanks (moving average of plausible intervals).
fn start_blanks() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let spawned = std::thread::Builder::new().name("xnw vblank".into()).spawn(|| {
            {
                use windows::Win32::System::Threading::*;
                // SAFETY: this thread's own priority.
                let _ = unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST) };
            }
            let output = host_output();
            let mut source = if output.is_some() { "the host output's vertical blank" } else { "a timer (no output)" };
            SOFTWARE.store(output.is_none(), Ordering::Relaxed);
            let mut instant = 0;
            let mut last = Instant::now();
            let mut said = false;
            loop {
                let period = Duration::from_nanos(PERIOD_NS.load(Ordering::Relaxed));
                match output.as_ref().filter(|_| !SOFTWARE.load(Ordering::Relaxed)) {
                    Some(o) => {
                        let at = Instant::now();
                        // SAFETY: a live output.
                        let ok = unsafe { o.WaitForVBlank() }.is_ok();
                        instant = if ok && at.elapsed() >= Duration::from_micros(500) { 0 } else { instant + 1 };
                        if instant >= 8 {
                            SOFTWARE.store(true, Ordering::Relaxed);
                            source = "a timer (WaitForVBlank failed or returned at once: remote session?)";
                            said = false;
                        }
                    }
                    None => {
                        let next = last + period;
                        let now = Instant::now();
                        if next > now {
                            std::thread::sleep(next - now);
                        }
                    }
                }
                let now = Instant::now();
                let dt = (now - last).as_nanos() as u64;
                last = now;
                if !SOFTWARE.load(Ordering::Relaxed) && (4_000_000..50_000_000).contains(&dt) {
                    let p = PERIOD_NS.load(Ordering::Relaxed);
                    PERIOD_NS.store(p - p / 16 + dt / 16, Ordering::Relaxed);
                }
                let n = {
                    let mut count = BLANKS.lock().unwrap_or_else(|e| e.into_inner());
                    *count += 1;
                    *count
                };
                BLANK.notify_all();
                if !said && n >= 120 {
                    said = true;
                    let hz = 1e9 / PERIOD_NS.load(Ordering::Relaxed) as f64;
                    xnw_core::info!("frame clock: {hz:.2} Hz from {source}; {} refreshes per title frame, sync interval {}", divider(), sync_interval());
                }
            }
        });
        if spawned.is_err() {
            xnw_core::warn!("frame clock: no vblank thread; frame waits use the timer");
        }
    });
}

/// For `XNW_FRAME_STATS`: presents by the refreshes since the one
/// before (blank count at our `Present`), and frames shown by the
/// refreshes they stayed on screen (DXGI frame statistics); buckets 0
/// (same refresh), 1, 2, 3, 4+. Taken by the frame line.
pub(crate) static PRESENTED: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
pub(crate) static SHOWN: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// After the compositor's `Present`: counts the refreshes since the last
/// present, and what DXGI says the frames shown since stayed for.
pub(crate) fn host_presented(swap_chain: &windows::Win32::Graphics::Dxgi::IDXGISwapChain3) {
    static LAST_BLANK: AtomicU64 = AtomicU64::new(u64::MAX);
    static LAST_STATS: Mutex<(u32, u32)> = Mutex::new((0, 0));
    let n = blanks();
    let before = LAST_BLANK.swap(n, Ordering::Relaxed);
    if before != u64::MAX {
        PRESENTED[n.saturating_sub(before).min(4) as usize].fetch_add(1, Ordering::Relaxed);
    }
    let mut stats = windows::Win32::Graphics::Dxgi::DXGI_FRAME_STATISTICS::default();
    // SAFETY: a live swap chain; an out-parameter.
    if unsafe { swap_chain.GetFrameStatistics(&mut stats) }.is_err() || stats.PresentCount == 0 {
        return;
    }
    let mut last = LAST_STATS.lock().unwrap_or_else(|e| e.into_inner());
    let (count, refresh) = *last;
    if count != 0 && stats.PresentCount > count && stats.PresentRefreshCount >= refresh {
        let (k, r) = (stats.PresentCount - count, stats.PresentRefreshCount - refresh);
        SHOWN[((r + k / 2) / k).min(4) as usize].fetch_add(u64::from(k), Ordering::Relaxed);
    }
    if stats.PresentCount != count {
        *last = (stats.PresentCount, stats.PresentRefreshCount);
    }
}

/// The histograms as text, taken and zeroed.
pub(crate) fn refreshes_text() -> String {
    let take = |h: &[AtomicU64; 5]| h.iter().map(|c| c.swap(0, Ordering::Relaxed).to_string()).collect::<Vec<_>>().join("/");
    format!(
        "presents after 0/1/2/3/4+ refreshes: {}; frames shown for 0/1/2/3/4+ refreshes (DXGI): {}; display {:.2} Hz{}, divider {}, sync interval {}",
        take(&PRESENTED),
        take(&SHOWN),
        1e9 / PERIOD_NS.load(Ordering::Relaxed).max(1) as f64,
        if SOFTWARE.load(Ordering::Relaxed) { " (timer)" } else { "" },
        divider(),
        sync_interval()
    )
}

/// Device method 0x280: blocks until the next tick (or the time-out); an
/// origin wait then hands out the new frame's token.
pub(crate) extern "system" fn wait(_: This, kind: u32, timeout_ms: u32, _list: *const u8, flags: u32, token: *mut u64) -> HRESULT {
    let started = Instant::now();
    let limit = Duration::from_millis(u64::from(timeout_ms));
    let hr = if wait_tick(last(kind), limit) { S_OK } else { TIMEOUT };
    if kind == ORIGIN && (hr == S_OK || flags & TOKEN_ON_TIMEOUT != 0) {
        let frame = FRAME.fetch_add(1, Ordering::Relaxed) + 1;
        if !token.is_null() {
            // SAFETY: checked non-null; the title passes its token.
            unsafe { token.write(frame) };
        }
    }
    let slot = if kind == ORIGIN { PACE_ORIGIN } else { PACE_OTHER };
    PACING[slot].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
    hr
}

/// For `XNW_FRAME_STATS`: microseconds since the last frame line spent
/// in frame-origin waits, other frame-event waits, compositing presents and
/// pacing at present; taken by the frame line.
pub(crate) static PACING: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
const PACE_ORIGIN: usize = 0;
const PACE_OTHER: usize = 1;
pub(crate) const PACE_PRESENT: usize = 2;
const PACE_AT_PRESENT: usize = 3;
pub(crate) const PACING_NAMES: [&str; 4] = ["origin waits", "other event waits", "composite", "pace at present"];

/// Device method 0x288: no statistics are kept, so none are returned.
pub(crate) extern "system" fn statistics(_: This, _token: u64, _kinds: u32, count: *mut u32, _out: *mut u8) -> HRESULT {
    if !count.is_null() {
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
    }
    S_OK
}

/// Frames begun at the last present.
static FRAME_AT_PRESENT: AtomicU64 = AtomicU64::new(u64::MAX);

/// Called by `PresentX`: if no frame wait began a frame since the last
/// present, waits for the next tick here, so a title that never waits still
/// runs at the interval. Returns how long it slept.
pub(crate) fn present_pace() -> Duration {
    super::resource::frame_presented(super::queue::presented());
    let now = current();
    if FRAME_AT_PRESENT.swap(now, Ordering::Relaxed) != now {
        return Duration::ZERO;
    }
    let at = Instant::now();
    wait_tick(&LAST[PRESENT as usize], Duration::from_secs(1));
    let wait = at.elapsed();
    PACING[PACE_AT_PRESENT].fetch_add(wait.as_micros() as u64, Ordering::Relaxed);
    wait
}

#[cfg(test)]
mod tests {
    use super::{plan_tick, sync_for};

    #[test]
    fn freecap_presents_at_one_refresh() {
        assert_eq!(sync_for(true, true, 2), 1);
        assert_eq!(sync_for(true, false, 2), 0);
        assert_eq!(sync_for(false, false, 2), 2);
        assert_eq!(sync_for(false, false, 9), 4);
    }
    const I: u128 = 33_000_000;

    #[test]
    fn late_pacing_rebases_only_overrun_frames() {
        // Early in tick 0 (last = 0): wait to tick 1, timer or late.
        assert_eq!(plan_tick(10_000_000, I, 0, 0, false), (1, 23_000_000, false));
        assert_eq!(plan_tick(10_000_000, I, 0, 0, true), (1, 23_000_000, false));
        // 40 ms in, last = 0: timer waits to tick 2 (26 ms); late ends at once.
        assert_eq!(plan_tick(40_000_000, I, 0, 0, false), (2, 26_000_000, false));
        assert_eq!(plan_tick(40_000_000, I, 0, 0, true), (1, 0, true));
        // 70 ms in, last = 0 (two ticks missed): late ends at once on the tick in progress (2),
        // so a vblank or flip wait that lagged catches up in one call, not one tick a call.
        assert_eq!(plan_tick(70_000_000, I, 0, 0, true), (2, 0, true));
        assert_eq!(plan_tick(70_000_000, I, 0, 1, true), (2, 0, true));
        assert_eq!(plan_tick(70_000_000, I, 0, 2, true), (3, 29_000_000, false));
        // Within the tick after last: not an overrun.
        assert_eq!(plan_tick(40_000_000, I, 0, 1, true), (2, 26_000_000, false));
        // A tick number already used by another wait kind still moves on.
        assert_eq!(plan_tick(10_000_000, I, 0, 5, false).0, 6);
    }

    #[test]
    fn a_30_fps_title_takes_whole_refreshes() {
        let thirtieth = 33_333_333;
        assert_eq!(super::divider_for(thirtieth, 16_666_667), 2); // 60 Hz
        assert_eq!(super::divider_for(thirtieth, 16_683_350), 2); // 59.94 Hz
        assert_eq!(super::divider_for(thirtieth, 8_333_333), 4); // 120 Hz
        assert_eq!(super::divider_for(thirtieth, 6_944_444), 5); // 144 Hz
        assert_eq!(super::divider_for(16_666_667, 16_666_667), 1);
        assert_eq!(super::divider_for(16_666_667, 33_333_333), 1); // 30 Hz display: never 0
    }
}
