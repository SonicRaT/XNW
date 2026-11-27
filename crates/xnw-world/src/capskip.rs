//! PERF4 cut 2 (`XNW_WORLD_RENDER=...,capskip=on|off`, default on): capture
//! only the frames whose list the pass thread will draw. With the pass
//! slower than the title (41 lists drawn per 60 frames), every list
//! published while an older one still waits replaces it unseen; its hooks,
//! sinks and list build (19 of 60 frames, ~45 ms summed of the title's
//! Subrender workers) were spent for nothing (ENG1 b2).
//!
//! The rule runs at a frame's cut (the drain's flip) and decides the NEXT
//! frame, before any of its draws: from the pass thread's state (busy since
//! when, the mean time per list, a list waiting) and the recorder's means
//! (frame interval, cut to publish), the time the pass will want a list
//! after the ones it already has or will get (the waiting one, the frame
//! just cut). When that is past the arrival of the list after next, the
//! next frame's list would be replaced before it is taken: skip it. A
//! quarter frame of margin: on doubt, capture.
//!
//! Simulated (the test below; frame 33.3, cut to publish 19, +-30 % jitter):
//! a pass at 47 ms a list replaces 1,149 of 4,000 lists with every frame
//! captured; with the rule 134, at 27 % fewer captures and 2.4 % fewer lists
//! drawn (the jitter: a pass done early finds nothing). Pass at 80 ms: 0
//! replaced, 59 % fewer captures. Pass faster than the title: no skip.

use std::sync::OnceLock;
use std::time::Instant;

/// Nanoseconds since the first call: the clock the rule's times share
/// (one per process; never 0 after the first call).
pub fn clock_ns() -> u64 {
    static T: OnceLock<Instant> = OnceLock::new();
    T.get_or_init(Instant::now).elapsed().as_nanos() as u64 + 1
}

/// Exponential mean with 1/8 of a new sample (the first sample alone).
pub fn mean(old: u64, sample: u64) -> u64 {
    if old == 0 { sample } else { old - old / 8 + sample / 8 }
}

/// The pass thread as the rule sees it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PassView {
    /// When it took the list it is drawing (None: waiting for one).
    pub busy_since: Option<u64>,
    /// Mean time per list drawn (0: none drawn yet).
    pub step: u64,
    /// A published list waits for it.
    pub waiting: bool,
}

/// The recorder at a frame cut (ns, [`clock_ns`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cut {
    pub now: u64,
    /// Mean interval between cuts.
    pub frame: u64,
    /// Mean time from a cut to its list's publish.
    pub build: u64,
    /// The frame just cut was captured (its list is on its way).
    pub captured: bool,
}

/// Whether the frame after the cut `c` would only be replaced unseen.
pub fn skip_next(p: &PassView, c: &Cut) -> bool {
    if p.step == 0 || c.frame == 0 {
        return false;
    }
    // The pass is done with the list it draws.
    let free = p.busy_since.map_or(c.now, |s| (s + p.step).max(c.now));
    // The cut frame's list arrives.
    let arrives = c.now + c.build;
    // When the pass wants a list after the waiting one and the cut frame's.
    let need = match (p.waiting, c.captured) {
        // The cut frame's list replaces the waiting one before it is taken.
        (true, true) if arrives <= free => free + p.step,
        (true, true) => (free + p.step).max(arrives) + p.step,
        (true, false) => free + p.step,
        (false, true) => free.max(arrives) + p.step,
        (false, false) => free,
    };
    need >= c.now + 2 * c.frame + c.build + c.frame / 4
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: u64 = 1_000_000;

    fn cut(now: u64, captured: bool) -> Cut {
        Cut { now, frame: 33 * MS, build: 19 * MS, captured }
    }

    #[test]
    fn no_skip_without_the_pass_means_or_with_an_idle_pass() {
        let c = cut(1000 * MS, true);
        assert!(!skip_next(&PassView::default(), &c), "no list drawn yet");
        assert!(!skip_next(&PassView { busy_since: None, step: 47 * MS, waiting: false }, &c), "idle pass");
        assert!(!skip_next(&PassView { busy_since: Some(990 * MS), step: 20 * MS, waiting: true }, &c), "a pass faster than the title");
    }

    #[test]
    fn a_list_waiting_behind_a_busy_slow_pass_skips_the_next_frame() {
        // Free at 1007 ms (47 ms a list), a list waiting, the cut frame's arriving at 1019: the
        // pass takes the waiting one at 1007, the cut frame's at 1054, wants a list at 1101;
        // list k+2 arrives at 1000 + 66 + 19 = 1085 (+ 8.25 margin): k+1 would be replaced.
        let p = PassView { busy_since: Some(960 * MS), step: 47 * MS, waiting: true };
        assert!(skip_next(&p, &cut(1000 * MS, true)));
        // The frame just cut was skipped: the waiting list is the pass's last, it wants one at 1054: capture.
        assert!(!skip_next(&p, &cut(1000 * MS, false)));
        // Nothing waiting: the pass takes the cut frame's list at 1019, wants one at 1066: capture.
        assert!(!skip_next(&PassView { waiting: false, ..p }, &cut(1000 * MS, true)));
        // Free only at 1037: the cut frame's list (1019) replaces the waiting one, the pass wants
        // its next at 1084, before k+2: capture.
        assert!(!skip_next(&PassView { busy_since: Some(990 * MS), ..p }, &cut(1000 * MS, true)));
    }

    /// A pass and a title running at their own pace, every list through the
    /// one-slot hand-over (the newest list replaces a waiting one), with the
    /// rule deciding each next frame from the means.
    fn run(pass: f64, skip: bool) -> (u32, u32, u32) {
        let (frame, build, n) = (33.3, 19.0, 4000u32);
        // A small LCG: +-30 % jitter on the pass and the build.
        let mut seed = 1u64;
        let mut jitter = |x: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            x * (1.0 + 0.6 * ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5))
        };
        let ns = |ms: f64| (ms * 1e6) as u64;
        let (mut drawn, mut replaced, mut captured) = (0, 0, 0);
        let mut slot = false;
        let mut busy: Option<(f64, f64)> = None;
        let mut publishes: Vec<f64> = Vec::new();
        let mut skipped = vec![false; n as usize + 1];
        // Events in time order: cuts every frame, publishes, the pass done.
        let mut t_cut = 0.0;
        let mut k = 0usize;
        loop {
            let next_pub = publishes.iter().copied().fold(f64::INFINITY, f64::min);
            let next_done = busy.map_or(f64::INFINITY, |b| b.1);
            let next_cut = if k < n as usize { t_cut } else { f64::INFINITY };
            let t = next_cut.min(next_pub).min(next_done);
            if t == f64::INFINITY {
                break;
            }
            if t == next_done {
                busy = None;
            } else if t == next_pub {
                publishes.retain(|p| *p != t);
                if slot {
                    replaced += 1;
                }
                slot = true;
            } else {
                let cap = !skipped[k];
                if cap {
                    captured += 1;
                    publishes.push(t + jitter(build));
                }
                let p = PassView { busy_since: busy.map(|b| ns(b.0)), step: ns(pass), waiting: slot };
                skipped[k + 1] = skip && skip_next(&p, &Cut { now: ns(t), frame: ns(frame), build: ns(build), captured: cap });
                k += 1;
                t_cut += frame;
            }
            if busy.is_none() && slot {
                slot = false;
                drawn += 1;
                busy = Some((t, t + jitter(pass)));
            }
        }
        (drawn, replaced, captured)
    }

    #[test]
    fn the_rule_sheds_the_replaced_lists_and_keeps_the_picture_rate() {
        for pass in [20.0, 30.0] {
            let (on, off) = (run(pass, true), run(pass, false));
            assert_eq!(on.2, off.2, "a pass at {pass} ms keeps up: every frame captured ({on:?} {off:?})");
        }
        for pass in [47.0, 65.0, 80.0, 100.0] {
            let (on, off) = (run(pass, true), run(pass, false));
            // Lists drawn within 4 % of every-frame capture; at most a third of the replaced
            // lists left; the captures fall by at least 60 % of the lists replaced today.
            assert!(on.0 * 100 >= off.0 * 96, "pass {pass}: drawn {on:?} vs {off:?}");
            assert!(on.1 * 3 <= off.1, "pass {pass}: replaced {on:?} vs {off:?}");
            assert!((off.2 - on.2) as f64 >= off.1 as f64 * 0.6, "pass {pass}: captured {on:?} vs {off:?}");
            println!("pass {pass} ms: rule on (drawn, replaced, captured) {on:?}, off {off:?}");
        }
    }

    #[test]
    fn mean_starts_at_the_first_sample() {
        assert_eq!(mean(0, 800), 800);
        assert_eq!(mean(800, 1600), 900);
    }
}
