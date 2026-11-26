//! One run, two answers (FAST1): `XNW_AB=<n>` turns on, at the end of the
//! title's n-th world frame (stage 64, `stagemeter.rs`), what these name:
//!
//! - `XNW_AB_PATCH=<names>`: patches of `patches.rs` that are safe to write
//!   while the title runs (`patches::LIVE`: every changed byte of every
//!   entry inside one aligned 8-byte word, written by one locked
//!   compare-exchange, whole instructions for whole instructions, and a
//!   switch the title reads every frame rather than a size made at start);
//! - `XNW_AB_SKIP=<stage ids>`: stages whose callbacks are skipped from
//!   then on (the meter's own flags, as `XNW_STAGE_SKIP`);
//! - `XNW_AB_WORLD=<name=on|off,...>`: our world pass switches that are
//!   read every frame (`LIVE_WORLD`); the others are read once at start.
//!
//! The stage meter closes its window at the switch, so its lines before
//! are `title stages [A]: ...` and after `title stages [B]: ...`; the line
//! `ab: B on at world frame n (...)` splits every other meter (frame stats,
//! pass lines) by log time.
//!
//! `XNW_STOP_WORLD_FRAMES=<n>`: after n world frames the meter prints its
//! last window, `title stages: n world frames measured; run ends`, and the
//! process ends with code 0: a run stops as soon as it has what it came for.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;

/// World pass switches (`XNW_WORLD_RENDER` items) whose readers ask every
/// frame (`worldprobe.rs` `render_switch` outside a cache).
pub(crate) const LIVE_WORLD: &[&str] = &["gpucull", "repeatdrop", "grass", "instfallback"];

static BASE: AtomicUsize = AtomicUsize::new(0);
/// The world frame of the switch (0: none).
static AT: AtomicU64 = AtomicU64::new(0);
/// The world frame that ends the run (0: none).
static STOP: AtomicU64 = AtomicU64::new(0);
/// Whether B is on.
static B: AtomicBool = AtomicBool::new(false);

struct Plan {
    patches: Vec<String>,
    skip: Vec<usize>,
    world: String,
}
static PLAN: OnceLock<Plan> = OnceLock::new();

/// A positive frame count, or None for empty; Err for anything else.
fn parse_frames(v: &str) -> Result<Option<u64>, ()> {
    match v.trim() {
        "" => Ok(None),
        s => s.parse::<u64>().ok().filter(|&n| n > 0).map(Some).ok_or(()),
    }
}

/// The plan of the A/B switches: patches that are live-safe, stage ids,
/// world items that are live; and the items refused, with why.
fn plan(patch: &str, skip: &str, world: &str) -> (Plan, Vec<String>) {
    let mut bad = Vec::new();
    let mut patches = Vec::new();
    for name in patch.split(',').map(|s| s.trim().to_ascii_lowercase()).filter(|s| !s.is_empty()) {
        if crate::patches::LIVE.contains(&name.as_str()) {
            patches.push(name);
        } else {
            bad.push(format!("XNW_AB_PATCH {name}: not a live-safe patch (patches::LIVE: {})", crate::patches::LIVE.join(",")));
        }
    }
    let mut ids = Vec::new();
    for s in skip.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match s.parse::<usize>() {
            Ok(n) if n < 128 => ids.push(n),
            _ => bad.push(format!("XNW_AB_SKIP {s}: not a stage id (0..128)")),
        }
    }
    let mut items = Vec::new();
    for item in world.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let name = item.split('=').next().unwrap_or("");
        if LIVE_WORLD.contains(&name) && item.contains('=') {
            items.push(item.to_owned());
        } else {
            bad.push(format!("XNW_AB_WORLD {item}: not a live switch name=on|off ({})", LIVE_WORLD.join(",")));
        }
    }
    (Plan { patches, skip: ids, world: items.join(",") }, bad)
}

/// Reads `XNW_AB*` and `XNW_STOP_WORLD_FRAMES`; whether the stage detour
/// is needed for them.
pub(crate) fn configure(base: usize) -> bool {
    let env = |k: &str| std::env::var(k).unwrap_or_default();
    let stop = match parse_frames(&env("XNW_STOP_WORLD_FRAMES")) {
        Ok(n) => n.unwrap_or(0),
        Err(()) => {
            xnw_core::warn!("XNW_STOP_WORLD_FRAMES={}: not a frame count; the run is not stopped", env("XNW_STOP_WORLD_FRAMES").trim());
            0
        }
    };
    STOP.store(stop, Ordering::Relaxed);
    if stop != 0 {
        xnw_core::info!("ab: the run ends after {stop} world frames (XNW_STOP_WORLD_FRAMES)");
    }
    let at = match parse_frames(&env("XNW_AB")) {
        Ok(n) => n.unwrap_or(0),
        Err(()) => {
            xnw_core::warn!("XNW_AB={}: not a frame count; no A/B", env("XNW_AB").trim());
            0
        }
    };
    if at != 0 {
        let (p, bad) = plan(&env("XNW_AB_PATCH"), &env("XNW_AB_SKIP"), &env("XNW_AB_WORLD"));
        for b in &bad {
            xnw_core::warn!("{b}; ignored");
        }
        xnw_core::info!("ab: A until world frame {at}, then B: patches [{}], skipped stages {:?}, world [{}]", p.patches.join(","), p.skip, p.world);
        let _ = PLAN.set(p);
        BASE.store(base, Ordering::Relaxed);
        AT.store(at, Ordering::Relaxed);
    }
    stop != 0 || at != 0
}

/// Whether the A/B switch is set (the meter tags its lines).
pub(crate) fn armed() -> bool {
    AT.load(Ordering::Relaxed) != 0
}

/// "A" or "B" for the meter's lines.
pub(crate) fn phase() -> &'static str {
    if B.load(Ordering::Relaxed) { "B" } else { "A" }
}

/// The world items B adds to `XNW_WORLD_RENDER` (empty under A).
pub(crate) fn world_overlay() -> Option<&'static str> {
    if B.load(Ordering::Relaxed) { PLAN.get().map(|p| p.world.as_str()).filter(|w| !w.is_empty()) } else { None }
}

/// What the meter does at the end of world frame `n`.
#[derive(Debug, PartialEq)]
pub(crate) enum At {
    Nothing,
    Switch,
    Stop,
}

pub(crate) fn at(n: u64) -> At {
    decide(n, AT.load(Ordering::Relaxed), STOP.load(Ordering::Relaxed), B.load(Ordering::Relaxed))
}

/// Frame `n` against the switch frame `ab` and the stop frame `stop` (0:
/// none), with B on or not.
fn decide(n: u64, ab: u64, stop: u64, b: bool) -> At {
    if stop != 0 && n >= stop {
        At::Stop
    } else if ab != 0 && n == ab && !b {
        At::Switch
    } else {
        At::Nothing
    }
}

/// Turns B on (the meter's thread, at a frame end): skip flags, world
/// overlay, then the live patches.
pub(crate) fn switch(n: u64) {
    let Some(p) = PLAN.get() else { return };
    crate::stagemeter::skip_more(&p.skip);
    B.store(true, Ordering::Relaxed);
    let base = BASE.load(Ordering::Relaxed);
    // SAFETY: the mapped title of this build (configure came from detours::install);
    // apply_live writes only LIVE entries, each with one atomic word.
    let done = if base == 0 { Vec::new() } else { unsafe { crate::patches::apply_live(base, &p.patches) } };
    xnw_core::info!("ab: B on at world frame {n} (patches applied [{}], skipped stages {:?}, world [{}])", done.join(","), p.skip, p.world);
}

/// Ends the run after the last window was printed.
pub(crate) fn stop(n: u64) -> ! {
    xnw_core::info!("title stages: {n} world frames measured; run ends (XNW_STOP_WORLD_FRAMES)");
    // SAFETY: ends our own process; the log file is unbuffered.
    unsafe { xnw_core::win::TerminateProcess(xnw_core::win::GetCurrentProcess(), 0) };
    unreachable!("TerminateProcess returned")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_counts_and_the_plan_parse() {
        assert_eq!((parse_frames(""), parse_frames(" 600 "), parse_frames("0"), parse_frames("x")), (Ok(None), Ok(Some(600)), Err(()), Err(())));
        let (p, bad) = plan("nolocalshadows, NoPrepass,watchdog,cbalign", "21,x,65", "gpucull=on,geomasync=on,grass");
        assert_eq!(p.patches, ["nolocalshadows", "noprepass"]);
        assert_eq!(p.skip, [21, 65]);
        assert_eq!(p.world, "gpucull=on");
        assert_eq!(bad.len(), 5, "{bad:?}");
    }

    #[test]
    fn the_switch_comes_once_and_the_stop_wins() {
        let d = |n, b| decide(n, 3, 5, b);
        assert_eq!([d(1, false), d(2, false), d(3, false)], [At::Nothing, At::Nothing, At::Switch]);
        assert_eq!([d(3, true), d(4, true), d(5, true), d(6, true)], [At::Nothing, At::Nothing, At::Stop, At::Stop]);
        assert_eq!([decide(3, 3, 3, false), decide(9, 0, 0, false), decide(9, 9, 0, false)], [At::Stop, At::Nothing, At::Switch]);
        // Unset in a test process: the meter's frame ends never act.
        assert_eq!(at(1), At::Nothing);
    }
}
