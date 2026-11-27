//! The head graph in the live pass (FACE1; `XNW_WORLD_RENDER=...,head=on`,
//! default off; live switch `head`). The lab's `xnw-lab head`, run per frame
//! on the live title:
//!
//! - the frame pack writer in its live head mode (`pack::enable_head`)
//!   records only the head graph's dispatches (`compute_graph_ops:*`,
//!   `ped_standard_head_cs:*`, the capture's HEAD_PROGRAMS) with their
//!   roots, tables and unordered buffers (the network's arena holds the
//!   title's expression inputs); the frame's pack is handed over in memory
//!   (`pack::take_live`);
//! - the graph runs translated through the shared compute runner
//!   (`xnw_frame::head_graph::run_on`, one runner for the run: its pipelines
//!   and translations stay, its resources go each frame) on our pass's
//!   device, the network first;
//! - the head position table it fills (the HSS extract's output view) is
//!   kept for the frame; the pre-skin's faces (variant 11) whose t74 is that
//!   table take their positions from it (`preskin::frame_for`) instead of
//!   their bind shape.
//!
//! On the hardware adapter the title's programs run only with
//! `XNW_BRIDGE_HW=1` and the per-submission GPU watchdog (started here,
//! `XNW_GPU_WATCH_MS`, default 500). A frame without a head dispatch does
//! no work and its faces stay neutral; a device failure turns the graph off
//! for the run. The cost per frame (record at the sinks is the title's
//! thread; the graph here, on the list thread) is logged every 60 frames.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use xnw_formats::framepack as fp;
use xnw_frame::head_graph;

/// `head=on` in `XNW_WORLD_RENDER` (read once).
pub fn on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| switch_on(&v)));
    *ON
}

/// `head=on` (the last `head=` item; anything but `off` is on).
pub fn switch_on(v: &str) -> bool {
    v.split(',').map(str::trim).filter_map(|p| p.strip_prefix("head=")).last().is_some_and(|s| s != "off")
}

/// The live switch `head` (SW1): 0 runs no graph, the faces stay neutral.
pub fn shown() -> bool {
    xnw_core::live::on(xnw_core::live::HEAD, || u32::from(on()))
}

struct Live {
    runner: xnw_world::compute::Runner<'static>,
    programs: head_graph::Programs,
    persist: head_graph::Persist,
    names: HashMap<u128, String>,
    dir: PathBuf,
}

// SAFETY: D3D12 objects are free-threaded; the live state is used under its lock.
unsafe impl Send for Live {}

/// Counts since the last line.
#[derive(Default)]
struct Stats {
    frames: u32,
    with_graph: u32,
    /// Frames with head dispatches but no face read by the pre-skin: not run.
    no_face: u32,
    whole: u32,
    graph_ms: f64,
    worst_ms: f64,
    /// Faces the pre-skin offered, of them given a table, found none at their address.
    faces_seen: u32,
    faces: u32,
    faces_none: u32,
    faces_unfilled: u32,
    faces_empty: u32,
    /// The first face not given since the last line, and why.
    face_first: String,
    last: String,
}

static LIVE: Mutex<Option<Result<Live, String>>> = Mutex::new(None);
static STATS: Mutex<Option<Stats>> = Mutex::new(None);
static OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The newest filled head table per title address (modulo the alias): the
/// frame of the record it came from, its bytes. The title writes the table
/// at two addresses in turn (pk8a f1027-1032); a face reads the one at its
/// own t74 address.
static TABLES: Mutex<Option<HashMap<u64, (u64, Arc<Vec<u8>>)>>> = Mutex::new(None);
/// Frames a table stays usable after its record's frame.
const TABLE_AGE: u64 = 120;
/// The frame of the newest record the graph ran from.
static NEWEST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn names_dir() -> PathBuf {
    std::env::var("XNW_SHADER_NAMES").ok().and_then(|f| PathBuf::from(f).parent().map(PathBuf::from)).unwrap_or_else(|| PathBuf::from("out"))
}

fn live() -> Result<Live, String> {
    let host = xnw_gpu::d3d12x::host::host_device().ok_or("no host device")?;
    if !host.caps.software && xnw_gpu::gpuwatch::budget().is_none() && std::env::var("XNW_BRIDGE_HW").ok().as_deref() == Some("1") {
        let ms = std::env::var("XNW_GPU_WATCH_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(500u64);
        xnw_gpu::gpuwatch::start(std::time::Duration::from_millis(ms));
        xnw_core::info!("world render: head on {}: GPU watchdog {ms} ms per submission", host.caps.adapter);
    }
    let runner = xnw_world::compute::Runner::new(host)?;
    // The network's output stays on the device (a burst reads it back for its line).
    head_graph::READ_NETWORK.store(super::burst::serving(), std::sync::atomic::Ordering::Relaxed);
    let dir = names_dir();
    let names = xnw_frame::hair_chain::names_in(&dir).ok_or_else(|| format!("no shader-names.tsv in {} (set XNW_SHADER_NAMES)", dir.display()))?;
    Ok(Live { runner, programs: HashMap::new(), persist: head_graph::Persist::new(), names, dir: dir.join("head-live").join("live") })
}

/// The newest table the graph filled at `addr` (alias off), not older than
/// [`TABLE_AGE`] frames before the newest record run.
pub fn table_at(addr: u64) -> Option<Arc<Vec<u8>>> {
    let newest = NEWEST.load(std::sync::atomic::Ordering::Relaxed);
    lock(&TABLES).as_ref()?.get(&addr).filter(|t| t.0 + TABLE_AGE >= newest).map(|t| t.1.clone())
}

/// Counts the pre-skin's faces this list (the list thread): offered, given a
/// table, without a table at their address (and one such address).
pub fn faces_given(seen: u32, g: &super::preskin::Given) {
    if let Some(st) = lock(&STATS).as_mut() {
        st.faces_seen += seen;
        st.faces += g.given;
        st.faces_none += g.none;
        st.faces_unfilled += g.unfilled;
        st.faces_empty += g.empty;
        if st.face_first.is_empty() {
            st.face_first.clone_from(&g.first);
        }
    }
}

fn head_dispatches(data: &fp::FramePack) -> usize {
    data.sinks.iter().filter(|s| s.op == fp::op::DISPATCH).count()
}

/// Frame `frame`'s head graph (the list thread, before the list's pre-skin is taken).
pub(super) fn frame(frame: u64, data: fp::FramePack, blobs: Arc<HashMap<u128, Vec<u8>>>) {
    let mut sg = lock(&STATS);
    let st = sg.get_or_insert_with(Stats::default);
    st.frames += 1;
    if OFF.load(std::sync::atomic::Ordering::Relaxed) || head_dispatches(&data) == 0 {
        return line(frame, st);
    }
    st.with_graph += 1;
    // FACE1 (3): a frame whose pre-skin reads no face draws no head: no graph.
    // `XNW_HEAD_ALWAYS=on` runs it anyway (a measure).
    // A frame the pre-skin no longer holds (the list behind the title) runs: unknown is not none.
    let record = data.manifest.frame;
    if super::preskin::on() && super::preskin::faces_read(record) == Some(0) && std::env::var("XNW_HEAD_ALWAYS").map_or(true, |v| v != "on") {
        st.no_face += 1;
        return line(frame, st);
    }
    let mut lg = lock(&LIVE);
    let l = match lg.get_or_insert_with(live) {
        Ok(l) => l,
        Err(e) => {
            xnw_core::warn!("world render: head: off: {e}");
            OFF.store(true, std::sync::atomic::Ordering::Relaxed);
            return;
        }
    };
    let p = xnw_frame::pack::Pack::in_memory(&l.dir, data, blobs).reading(|a, n| super::dcopy(a, n as usize));
    let t0 = Instant::now();
    let g = head_graph::run_on(&p, &mut l.runner, &l.names, &mut l.programs, &mut l.persist, &|_, _| true);
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    st.graph_ms += ms;
    st.worst_ms = st.worst_ms.max(ms);
    match g {
        Err(e) => {
            // SAFETY: a plain query.
            if xnw_gpu::d3d12x::host::host_device().is_some_and(|h| unsafe { h.device.GetDeviceRemovedReason() }.is_err()) {
                xnw_core::warn!("world render: frame {frame}: head: the device failed in the head graph ({e}); head off for the run");
                OFF.store(true, std::sync::atomic::Ordering::Relaxed);
                return;
            }
            st.last = format!("graph: {e}");
        }
        Ok(g) => {
            if super::burst::serving() {
                for l in g.lines.iter().filter(|l| l.contains("dropped")) {
                    println!("  head {l}");
                }
                for l in &g.lacks {
                    println!("  head lacks: {l}");
                }
            }
            // The whole graph or nothing: a dropped dispatch leaves the faces neutral.
            match &g.table {
                Some((at, b)) if g.dropped.is_empty() => {
                    st.whole += 1;
                    NEWEST.fetch_max(record, std::sync::atomic::Ordering::Relaxed);
                    lock(&TABLES).get_or_insert_with(HashMap::new).insert(*at, (record, Arc::new(b.clone())));
                }
                _ => {}
            }
            st.last = g.line();
        }
    }
    line(frame, st);
}

/// The head line, every 60 frames taken (a burst: every frame).
fn line(frame: u64, st: &mut Stats) {
    if st.frames < 60 && !super::burst::serving() {
        return;
    }
    let ran = st.with_graph - st.no_face;
    let per = if ran > 0 { st.graph_ms / f64::from(ran) } else { 0.0 };
    let tables: Vec<String> = lock(&TABLES).as_ref().map(|m| m.iter().map(|(a, t)| format!("{a:#x} (frame {})", t.0)).collect()).unwrap_or_default();
    xnw_core::info!(
        "world render: frame {frame}: head: {} frames, {} with head dispatches ({} without a face drawn: not run), {} graphs whole; faces {} offered, {} from a table, {} with no table at their address, {} whose header there is unfilled, {} whose entries lie outside it (first not given: {}; tables {tables:?}); per frame with the graph {per:.1} ms (worst {:.1} ms); last: {}",
        st.frames, st.with_graph, st.no_face, st.whole, st.faces_seen, st.faces, st.faces_none, st.faces_unfilled, st.faces_empty, st.face_first, st.worst_ms, st.last
    );
    *st = Stats::default();
}

#[cfg(test)]
mod tests {
    #[test]
    fn switch_grammar() {
        assert!(super::switch_on("hair=on,head=on"));
        assert!(!super::switch_on("head=off"));
        assert!(!super::switch_on("headgraph=on"));
    }
}
