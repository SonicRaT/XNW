//! `XNW_FRAMESKIP=<n>` or `auto:<ms>` (default off): render one title
//! frame in `n` (or one each time `<ms>` have passed since the last
//! rendered one ended); for the others the title still runs and records,
//! but a submission does only what the title's CPU or later frames need
//! from it (`HostList::run_skipped`: copies into title memory, work that
//! writes title memory, bottom-of-pipe value writes) and no host GPU work,
//! and Present keeps the last picture. The title's clock moves a fixed
//! 1/15 s per frame however long the frame took, so the game advances
//! `n` steps per rendered frame.
//!
//! Which title frame a submission belongs to: `ExecuteCommandListsX`'s
//! frame token, the same value `PresentX` passes in its planes (title
//! 0x1424822xx: the token submission at title+0x24821dd). A token is
//! decided once, the first time it is seen (a submission or its present),
//! and forgotten at its present, so every submission of a rendered frame
//! renders and every one of a skipped frame is skipped. Submissions
//! without a token (`ExecuteCommandLists`) belong to the presented-frame
//! epoch they come in (key `1 << 63 | presented frame`).
//!
//! Null mode (`XNW_GPU=null[@world|@card|@<seconds>|@frame+<n>]`,
//! `XNW_FRAMESKIP=all` = `XNW_GPU=null`): from that point every frame is
//! skipped as above, so no title work reaches the host GPU again and it
//! cannot hang; before it everything runs normally (the front end, movies
//! and loading screens to navigate by). `@world` is capture.rs's world
//! start detector, `@card` the first frame after the first mesh draw (the
//! mission card is the first thing drawn with mesh shaders; I), `@<s>` the
//! log clock, `@frame+<n>` the presented frame.
//!
//! Cost unset: one cached test per submission and present.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// The switch: 0 off; `Fixed(n)`; `Auto(ms)`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Mode {
    Off,
    Fixed(u64),
    Auto(u64),
    Null(Point),
}

/// Where null mode starts.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Point {
    Start,
    World,
    Card,
    Seconds(u64),
    Frame(u64),
}

/// `XNW_GPU=null[@...]`.
pub(crate) fn parse_null(v: Option<&str>) -> Option<Point> {
    // Options after a comma (`,ui=keep`, uikeep.rs).
    let v = v?.split(',').next()?.trim();
    let at = v.strip_prefix("null")?;
    match at {
        "" => Some(Point::Start),
        "@world" => Some(Point::World),
        "@card" => Some(Point::Card),
        _ => {
            let at = at.strip_prefix('@')?;
            match at.strip_prefix("frame+") {
                Some(n) => n.parse().ok().map(Point::Frame),
                None => at.parse().ok().map(Point::Seconds),
            }
        }
    }
}

/// Whether null mode's point is reached (it stays reached).
fn null_reached(point: Point) -> bool {
    static REACHED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if REACHED.load(Ordering::Relaxed) {
        return true;
    }
    let now = super::queue::presented();
    let reached = match point {
        Point::Start => true,
        Point::World => super::capture::world_start().is_some_and(|w| now >= w),
        Point::Card => crate::mesh::MESH_DRAWS.load(Ordering::Relaxed) > 0,
        Point::Seconds(s) => xnw_core::uptime().as_secs() >= s,
        Point::Frame(n) => now >= n,
    };
    if reached && !REACHED.swap(true, Ordering::Relaxed) {
        xnw_core::warn!("null GPU: from presented frame {now} no title work reaches the host GPU (XNW_GPU=null, {point:?})");
        super::present::null_caption();
    }
    reached
}

/// Whether the world start must be detected (capture.rs).
pub(crate) fn wants_world() -> bool {
    mode() == Mode::Null(Point::World)
}

pub(crate) fn parse(v: Option<&str>) -> Mode {
    match v.map(str::trim) {
        Some("all") => Mode::Null(Point::Start),
        Some(v) if v.starts_with("auto:") => v[5..].parse().map_or(Mode::Off, Mode::Auto),
        Some(v) => match v.parse::<u64>() {
            Ok(n) if n > 1 => Mode::Fixed(n),
            _ => Mode::Off,
        },
        None => Mode::Off,
    }
}

/// Set by [`render_all`]: frameskip's `n`/`auto` modes stop skipping.
static RENDER_ALL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Ends `XNW_FRAMESKIP=<n>|auto:<ms>` for the rest of the run: every frame
/// decided from now on renders (null mode is not affected). The world
/// probe calls it when its capture window opens, so the captured frames
/// have the title's own picture (REF1, profile ref-warp: the loading
/// screens are skipped, the world is not). No effect without frameskip.
pub fn render_all() {
    if matches!(mode(), Mode::Fixed(_) | Mode::Auto(_)) && !RENDER_ALL.swap(true, Ordering::Relaxed) {
        xnw_core::warn!("frameskip: off from presented frame {}: every frame renders (asked by the world probe)", super::queue::presented());
    }
}

pub(crate) fn mode() -> Mode {
    static MODE: std::sync::LazyLock<Mode> = std::sync::LazyLock::new(|| {
        let m = match parse_null(std::env::var("XNW_GPU").ok().as_deref()) {
            Some(p) => Mode::Null(p),
            None => parse(std::env::var("XNW_FRAMESKIP").ok().as_deref()),
        };
        if m != Mode::Off {
            xnw_core::warn!("frameskip: {m:?} (XNW_FRAMESKIP): skipped frames record but do not render");
        }
        m
    });
    *MODE
}

/// Decisions by frame key, the count of keys decided, when the last
/// rendered frame ended.
struct Decider {
    decided: HashMap<u64, bool>,
    seen: u64,
    last_rendered: std::time::Instant,
}

impl Decider {
    fn new() -> Decider {
        Decider { decided: HashMap::new(), seen: 0, last_rendered: std::time::Instant::now() }
    }

    /// Whether frame `key` is skipped (decided at first sight).
    fn skip(&mut self, mode: Mode, key: u64) -> bool {
        if let Some(&s) = self.decided.get(&key) {
            return s;
        }
        let k = self.seen;
        self.seen += 1;
        let skip = match mode {
            Mode::Off => false,
            // `auto` ends once asked (`render_all`): every frame from then renders.
            Mode::Fixed(_) | Mode::Auto(_) if RENDER_ALL.load(Ordering::Relaxed) => false,
            Mode::Null(p) => null_reached(p),
            Mode::Fixed(n) => !k.is_multiple_of(n),
            Mode::Auto(ms) => self.last_rendered.elapsed() < std::time::Duration::from_millis(ms),
        };
        // Bounded: tokens never presented (a title path we do not know).
        if self.decided.len() > 64 {
            self.decided.clear();
        }
        self.decided.insert(key, skip);
        skip
    }

    /// Frame `key`'s present: its decision, then forgotten.
    fn present(&mut self, mode: Mode, key: u64) -> bool {
        let skip = self.skip(mode, key);
        self.decided.remove(&key);
        if !skip {
            self.last_rendered = std::time::Instant::now();
        }
        skip
    }
}

static DECIDER: Mutex<Option<Decider>> = Mutex::new(None);

fn key(token: u64) -> u64 {
    if token != 0 { token } else { 1 << 63 | super::queue::presented() }
}

/// Whether the null-GPU point is reached (every frame from now on is
/// skipped: host copies and buffers made at recording would never be read).
pub(crate) fn null_now() -> bool {
    matches!(mode(), Mode::Null(p) if null_reached(p))
}

/// Whether a submission of frame `token` (0: none) is skipped.
pub(crate) fn skip(token: u64) -> bool {
    let mode = mode();
    if mode == Mode::Off {
        return false;
    }
    // Null mode decides by the point alone (once reached, all frames).
    if let Mode::Null(p) = mode {
        return null_reached(p);
    }
    DECIDER.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Decider::new).skip(mode, key(token))
}

/// Whether the present of frame `token` (0: none) is skipped; counts.
pub(crate) fn skip_present(token: u64) -> bool {
    let mode = mode();
    if mode == Mode::Off {
        return false;
    }
    if let Mode::Null(p) = mode {
        let skip = null_reached(p);
        STATS[if skip { SKIPPED } else { RENDERED }].fetch_add(1, Ordering::Relaxed);
        return skip;
    }
    let skip = DECIDER.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Decider::new).present(mode, key(token));
    STATS[if skip { SKIPPED } else { RENDERED }].fetch_add(1, Ordering::Relaxed);
    skip
}

/// Frames rendered and skipped, microseconds of ours spent in skipped
/// submissions and presents, wall microseconds of skipped frames.
static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
const RENDERED: usize = 0;
const SKIPPED: usize = 1;
const OURS_US: usize = 2;
const WALL_US: usize = 3;
static MIN_US: AtomicU64 = AtomicU64::new(u64::MAX);
static MAX_US: AtomicU64 = AtomicU64::new(0);

/// Time spent by us on a skipped submission or present.
pub(crate) fn ours(us: u64) {
    STATS[OURS_US].fetch_add(us, Ordering::Relaxed);
}

/// At each present (frameskip on): the wall time of a skipped frame, and a
/// line every 30 presented frames: rendered, skipped, a skipped frame's
/// cost (ours, wall), game seconds per wall second.
pub(crate) fn presented(skipped: bool) {
    if mode() == Mode::Off {
        return;
    }
    static LAST: Mutex<Option<(std::time::Instant, std::time::Instant, u64)>> = Mutex::new(None);
    let now = std::time::Instant::now();
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let (prev, since, frames) = last.get_or_insert((now, now, 0));
    if skipped {
        let us = now.duration_since(*prev).as_micros() as u64;
        STATS[WALL_US].fetch_add(us, Ordering::Relaxed);
        // The title's own CPU frame time (null mode: nothing of ours).
        MIN_US.fetch_min(us, Ordering::Relaxed);
        MAX_US.fetch_max(us, Ordering::Relaxed);
    }
    *prev = now;
    *frames += 1;
    if *frames >= 30 {
        let [r, s, ours, wall] = [0, 1, 2, 3].map(|i| STATS[i].swap(0, Ordering::Relaxed));
        let elapsed = now.duration_since(*since).as_secs_f64();
        let per = |us: u64| if s == 0 { 0.0 } else { us as f64 / s as f64 / 1000.0 };
        let (lo, hi) = (MIN_US.swap(u64::MAX, Ordering::Relaxed), MAX_US.swap(0, Ordering::Relaxed));
        let w: Vec<u64> = super::host::SKIPPED_WORK.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
        let k = s.max(1);
        let (title_ms, ours_ms) = super::counters::cpu_split();
        let per_frame = (r + s).max(1) as f64;
        xnw_core::info!("frameskip: CPU per frame: title threads {:.1} ms, ours {:.1} ms; {}", title_ms / per_frame, ours_ms / per_frame, super::watch::veh_text(r + s));
        xnw_core::info!(
            "frameskip: ours per skipped frame: {} buffer copies {} us, {} title-writing steps {} us, {} value writes {} us, zero-copy dry {} us; record time by call: {}",
            w[0] / k, w[1] / k, w[2] / k, w[3] / k, w[4] / k, w[5] / k, w[6] / k,
            super::counters::call_time_text(r + s)
        );
        xnw_core::info!(
            "frameskip: {r} frames rendered, {s} skipped{}; a skipped frame {:.1} ms ours, {:.1} ms wall (min {:.1}, max {:.1}); game {:.2} s per wall second",
            if matches!(mode(), Mode::Null(_)) { " (null GPU: the wall time is the title's own CPU frame)" } else { "" },
            per(ours),
            per(wall),
            if s == 0 { 0.0 } else { lo as f64 / 1000.0 },
            hi as f64 / 1000.0,
            (r + s) as f64 / 15.0 / elapsed.max(1e-6)
        );
        *since = now;
        *frames = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_null_switch_parses() {
        assert_eq!(parse_null(Some("null")), Some(Point::Start));
        assert_eq!(parse_null(Some("null@world")), Some(Point::World));
        assert_eq!(parse_null(Some("null@card")), Some(Point::Card));
        assert_eq!(parse_null(Some("null@90")), Some(Point::Seconds(90)));
        assert_eq!(parse_null(Some("null@frame+300")), Some(Point::Frame(300)));
        assert_eq!(parse_null(Some("hardware")), None);
        assert_eq!(parse(Some("all")), Mode::Null(Point::Start));
    }

    #[test]
    fn the_switch_parses() {
        assert_eq!(parse(None), Mode::Off);
        assert_eq!(parse(Some("1")), Mode::Off);
        assert_eq!(parse(Some("4")), Mode::Fixed(4));
        assert_eq!(parse(Some(" auto:500 ")), Mode::Auto(500));
        assert_eq!(parse(Some("auto:x")), Mode::Off);
    }

    /// One frame in three renders; every submission of a frame gets its
    /// frame's decision, and a token is decided anew after its present.
    #[test]
    fn a_frame_is_decided_once_for_all_its_submissions() {
        let mut d = Decider::new();
        let m = Mode::Fixed(3);
        let frames: Vec<bool> = (10..16).map(|t| d.skip(m, t)).collect();
        assert_eq!(frames, [false, true, true, false, true, true]);
        assert!(d.skip(m, 11) && d.skip(m, 11), "the same decision for every submission");
        assert!(!d.present(m, 10) && d.present(m, 11));
        assert!(!d.decided.contains_key(&10), "forgotten at its present");
    }
}
