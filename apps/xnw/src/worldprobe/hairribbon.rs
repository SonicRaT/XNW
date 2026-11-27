//! HAIR10: our guide ribbons in the live pass (`XNW_WORLD_RENDER=...,
//! hairribbon=on`, default off). The title's strand chain stays an offline
//! oracle (`hair=on`); here:
//!
//! - the frame pack writer's live ribbon mode (`pack::enable_ribbon`)
//!   records at the list sink only the main-view strand draws, each with
//!   its root constants, the frame table's worlds and group records (slots
//!   18, 21) and its group's remap, strand asset and current palette window
//!   (`strand_groups` lean); the frame's pack is handed over in memory
//!   (`pack::take_live`);
//! - on the list thread every guide of each drawn group is posed on the CPU
//!   (`xnw_frame::hair_ribbon::build`) and the points go with the list
//!   (`FrameList::hair_ribbons`);
//! - our pass draws them as camera-facing ribbons under its depth
//!   (`xnw_world::hair::HairPass::record_ribbons`).
//!
//! No title program runs and nothing binds title memory on the GPU. The
//! cost (list thread ms, guides, points) is logged every 60 frames, every
//! frame in a burst; the title thread's part is the burst's "live pack"
//! line (strand records, tables).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use xnw_formats::framepack as fp;
use xnw_world::FrameList;

/// `hairribbon=on` in `XNW_WORLD_RENDER` (read once).
pub fn on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| xnw_world::hair::ribbon_switch_on(&v)));
    *ON
}

/// Half widths by asset (kept for the run) and the counts since the last line.
#[derive(Default)]
struct State {
    widths: HashMap<u128, f32>,
    motion: xnw_frame::hair_ribbon::Motion,
    frames: u32,
    with_draws: u32,
    guides: u64,
    points: u64,
    ms: f64,
    worst_ms: f64,
    last: String,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// Frame `frame`'s ribbons into `list` (the list thread, before the list is published).
pub(super) fn frame(frame: u64, list: &mut FrameList, data: fp::FramePack, blobs: Arc<HashMap<u128, Vec<u8>>>) {
    let t0 = Instant::now();
    let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let st = g.get_or_insert_with(State::default);
    st.frames += 1;
    let p = xnw_frame::pack::Pack::in_memory(&PathBuf::from("out"), data, blobs).reading(|a, n| super::dcopy(a, n as usize));
    // HAIR11: the sway's clock, the title's frames at 30 a second.
    st.motion.t = frame as f32 / 30.0;
    let mut b = xnw_frame::hair_ribbon::build(&p, &mut st.widths, &mut st.motion);
    // The strand draws' worlds are relative to their own camera; our pass
    // draws with the list's main camera (pk8a f1028-1029: not the same one).
    let ours = list.main_camera().map(|c| { let v = &c.view; [0, 1, 2].map(|i| -(v[12] * v[4 * i] + v[13] * v[4 * i + 1] + v[14] * v[4 * i + 2])) });
    let moved = match (b.origin, ours) {
        (Some(from), Some(to)) => format!("; moved {:.3} m onto the list camera", xnw_frame::hair_ribbon::rebase(&mut b.points, from, to)),
        _ => "; not moved (no camera)".to_string(),
    };
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    if b.draws > 0 {
        st.with_draws += 1;
        st.ms += ms;
        st.worst_ms = st.worst_ms.max(ms);
        st.guides += b.guides as u64;
        st.points += b.points.points.len() as u64;
        st.last = format!("{}{moved}", b.line());
    }
    if super::burst::serving() {
        println!("  hair ribbons frame {frame}: {}{moved} in {ms:.2} ms", b.line());
        for (sink, g, n, points, half, lo, hi) in &b.rows {
            println!("    sink {sink} group {g}: {n} guides x {points} points, half width {:.2} mm, roots {lo:.3?} .. {hi:.3?}", half * 1e3);
        }
    }
    if !b.points.points.is_empty() {
        list.hair_ribbons = Some(Arc::new(b.points));
    }
    if st.frames >= 60 || super::burst::serving() {
        let per = |v: f64| if st.with_draws > 0 { v / f64::from(st.with_draws) } else { 0.0 };
        xnw_core::info!(
            "world render: frame {frame}: hair ribbons: {} frames, {} with strand draws; per such frame {:.0} guides, {:.0} points, {:.2} ms on the list thread (worst {:.2} ms); last: {}",
            st.frames,
            st.with_draws,
            per(st.guides as f64),
            per(st.points as f64),
            per(st.ms),
            st.worst_ms,
            st.last
        );
        let (widths, motion) = (std::mem::take(&mut st.widths), std::mem::take(&mut st.motion));
        *st = State { widths, motion, ..State::default() };
    }
}
