//! Host pipelines built before the world needs them (XNW).
//!
//! The title creates all its pipelines at boot (about 60,000) and draws with
//! a few hundred in the first minutes. An earlier run listed the ones it
//! drew with, by key and first-use time (`library.rs`). This run's
//! prebuilder gets every pipeline the title creates (a channel, so the
//! title's thread does no work), computes its key off the title's threads,
//! and for the listed ones, once the title has stopped creating (no new
//! pipeline for [`IDLE`]) hands them to the pipeline workers, earliest first,
//! at most [`inflight`] at a time and not while draws wait for their own
//! pipelines (logical processors - 2 at normal priority until the world
//! draws, then 3 at the lowest). A prebuild is a library load when the host library holds it
//! (`library.rs`), else a compile; mesh pairs (compute and draw) are one
//! pipeline here, so they are built together.
//!
//! `XNW_PREBUILD=off` turns it off (the list is still recorded).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::time::{Duration, Instant};

use super::{Desc, library, make};

/// No new pipeline for this long: the title has created them all.
const IDLE: Duration = Duration::from_millis(1500);
/// Prebuilds in the workers at once once the world draws (the workers are
/// then below normal priority; the title keeps the other cores).
const INFLIGHT_WORLD: u64 = 3;

/// The first mesh draw after the story load started (the world draws
/// with mesh programs): from then on prebuilds yield to the title.
static WORLD: AtomicBool = AtomicBool::new(false);

/// Whether the world draws ([`note_mesh_draw`]).
pub(crate) fn in_world() -> bool {
    WORLD.load(Ordering::Relaxed)
}

/// A mesh draw was recorded.
pub(crate) fn note_mesh_draw() {
    if !WORLD.load(Ordering::Relaxed) && library::STORY_MS.load(Ordering::Relaxed) != 0 && !WORLD.swap(true, Ordering::Relaxed) {
        xnw_core::info!("prebuild: the world draws (frame {}): {}; now {INFLIGHT_WORLD} at a time, lowest priority", super::super::presented(), stats_text());
    }
}

/// Prebuilds at once: logical processors - 2 in the front end and the
/// load, [`INFLIGHT_WORLD`] once the world draws.
fn inflight() -> u64 {
    if in_world() { INFLIGHT_WORLD } else { (make::workers() as u64).max(INFLIGHT_WORLD) }
}

/// Milliseconds since the start at which the first prebuild was queued.
static STARTED_MS: AtomicU64 = AtomicU64::new(0);

/// Prebuilds finished per second since the first was queued.
fn rate() -> f64 {
    let start = STARTED_MS.load(Ordering::Relaxed);
    if start == 0 {
        return 0.0;
    }
    let secs = (xnw_core::uptime().as_millis() as u64).saturating_sub(start) as f64 / 1000.0;
    if secs > 0.0 { make::PRE_DONE.load(Ordering::Relaxed) as f64 / secs } else { 0.0 }
}

fn set_priority(lowest: bool) {
    // SAFETY: a plain call on our own thread.
    unsafe {
        use windows::Win32::System::Threading::*;
        let _ = SetThreadPriority(GetCurrentThread(), if lowest { THREAD_PRIORITY_LOWEST } else { THREAD_PRIORITY_NORMAL });
    }
}

/// Listed pipelines the title has created (key found in the list).
static MATCHED: AtomicU64 = AtomicU64::new(0);

/// A pipeline the title created: sent to the prebuilder (once the list is
/// known to be non-empty).
pub(super) fn offer(d: &Arc<Desc>) {
    static SENDER: OnceLock<Option<mpsc::Sender<Arc<Desc>>>> = OnceLock::new();
    if d.stages.is_empty() {
        return;
    }
    let sender = SENDER.get_or_init(|| {
        super::init_cache_dir();
        if !super::enabled() || std::env::var("XNW_PREBUILD").is_ok_and(|v| v == "off") || library::listed() == 0 {
            return None;
        }
        let (tx, rx) = mpsc::channel();
        match std::thread::Builder::new().name("xnw-prebuild".into()).spawn(move || run(rx)) {
            Ok(_) => Some(tx),
            Err(e) => {
                xnw_core::warn!("d3d12x: no prebuild thread: {e}");
                None
            }
        }
    });
    if let Some(tx) = sender {
        let _ = tx.send(d.clone());
    }
}

fn run(rx: mpsc::Receiver<Arc<Desc>>) {
    // Normal priority until the world draws.
    let mut lowest = false;
    set_priority(lowest);
    let listed = library::listed() as u64;
    // Listed pipelines not yet handed over, latest first (popped from the end).
    let mut todo: Vec<(i64, Arc<Desc>)> = Vec::new();
    let mut last_arrival = Instant::now();
    let mut started = false;
    loop {
        let wait = if started && !todo.is_empty() { Duration::from_millis(20) } else { Duration::from_millis(250) };
        let mut batch = Vec::new();
        if let Ok(d) = rx.recv_timeout(wait) {
            batch.push(d);
            batch.extend(rx.try_iter().take(4096));
        }
        if !batch.is_empty() {
            last_arrival = Instant::now();
        }
        for d in batch {
            let (shaders, g) = (super::shaders_of(&d), d.graphics_state());
            let key = super::make::key_of(&shaders, g.as_ref());
            if let Some(t) = library::wanted(&key) {
                // The recipe, so `xnw prebuild` can make it next time.
                if !library::has_recipe(&key)
                    && let Some(r) = super::warm::early_recipe(&shaders, g.as_ref())
                {
                    library::note_recipe(&key, r);
                }
                MATCHED.fetch_add(1, Ordering::Relaxed);
                todo.push((t, d));
                if started {
                    todo.sort_by_key(|(t, _)| std::cmp::Reverse(*t));
                }
            }
        }
        if in_world() != lowest {
            lowest = in_world();
            set_priority(lowest);
        }
        if !started {
            let all = MATCHED.load(Ordering::Relaxed) >= listed;
            if todo.is_empty() || (!all && last_arrival.elapsed() < IDLE) {
                continue;
            }
            started = true;
            todo.sort_by_key(|(t, _)| std::cmp::Reverse(*t));
            let _ = STARTED_MS.compare_exchange(0, (xnw_core::uptime().as_millis() as u64).max(1), Ordering::Relaxed, Ordering::Relaxed);
            xnw_core::info!("prebuild: {} of {listed} listed pipelines created; building them earliest first, {} at a time", todo.len(), inflight());
        }
        while !todo.is_empty() && make::prebuild_inflight() < inflight() && make::demand_waiting() == 0 {
            if let Some((_, d)) = todo.pop() {
                make::prebuild(super::pending(&d));
            }
        }
    }
}

/// A phase of the title's start ("story load", "world entry"): the first
/// call of each logs how many of the listed pipelines are ready.
pub fn note_phase(label: &str) {
    static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
    {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if seen.iter().any(|s| s == label) {
            return;
        }
        seen.push(label.to_owned());
    }
    if label.starts_with("story") {
        library::story_started();
    }
    xnw_core::info!(
        "prebuild: {} of {} ready ({label}; {} listed, {} still building; {:.1} pipelines/s)",
        make::PRE_DONE.load(Ordering::Relaxed),
        MATCHED.load(Ordering::Relaxed),
        library::listed(),
        make::prebuild_inflight(),
        rate()
    );
}

/// The counters, for the frame statistics line.
pub(crate) fn stats_text() -> String {
    let late = make::LATE.load(Ordering::Relaxed);
    format!(
        "recorded {}; listed {} matched {}; prebuilt {} ({:.1} pipelines/s, {} at a time) (first draw found {} ready), ready by bind {}; built late {late} ({:.1} ms); still building {} (waiting {})",
        library::RECORDED.load(Ordering::Relaxed),
        library::listed(),
        MATCHED.load(Ordering::Relaxed),
        make::PRE_DONE.load(Ordering::Relaxed),
        rate(),
        inflight(),
        make::PREBUILT_USED.load(Ordering::Relaxed),
        make::BOUND_READY.load(Ordering::Relaxed),
        make::LATE_US.load(Ordering::Relaxed) as f64 / 1000.0,
        make::prebuild_inflight(),
        make::waiting()
    )
}
