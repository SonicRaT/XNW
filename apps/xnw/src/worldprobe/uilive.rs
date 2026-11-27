//! UI1: the title's UI in the live pass (`XNW_WORLD_RENDER=...,ui=on`,
//! default off). The replay's path (`xnw_frame::ui`), run per frame on the
//! live title:
//!
//! - the frame pack writer in its live UI mode (`pack::enable_ui`) records,
//!   at the list sink, only the Draw/DrawIndexed sinks of UI-class pipelines
//!   (Scaleform `F*`, `PS_Bink*`, ...: their pixel program's name) with their
//!   tables, IA records (v2.12) and pipeline state; the frame's pack is
//!   handed over in memory (`pack::take_live`), no frame is written;
//! - every UI row runs through the title's translated vertex and pixel
//!   programs into one RGBA8 layer at the pass's size (`xnw_frame::ui::draw`,
//!   translations kept for the run), which goes with the list
//!   (`FrameList::ui`); our pass draws it over its picture with the plane
//!   blend (`xnw_world::ui::UiPass`).
//!
//! On the hardware adapter the title's programs run only with
//! `XNW_BRIDGE_HW=1` and the per-submission GPU watchdog (started here,
//! `XNW_GPU_WATCH_MS`, default 500). A frame without UI rows does no work; a
//! device failure turns the UI off for the run. A `ui:` line every 60 frames.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use xnw_formats::framepack as fp;
use xnw_world::FrameList;

/// The pass's size (`xnw_world::screen_pass`): the layer is made at it.
const SIZE: (u32, u32) = (1920, 1080);

/// `ui=on` in `XNW_WORLD_RENDER` (read once).
pub fn on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| xnw_world::ui::switch_on(&v)));
    *ON
}

/// Where `shader-names.tsv` is: `XNW_SHADER_NAMES`'s directory, else `out`.
fn names_dir() -> PathBuf {
    std::env::var("XNW_SHADER_NAMES").ok().and_then(|f| PathBuf::from(f).parent().map(PathBuf::from)).unwrap_or_else(|| PathBuf::from("out"))
}

/// The program names the UI test and the rows read.
pub(super) fn names() -> Result<HashMap<u128, String>, String> {
    let dir = names_dir();
    xnw_frame::hair_chain::names_in(&dir).ok_or_else(|| format!("no shader-names.tsv in {} (set XNW_SHADER_NAMES)", dir.display()))
}

struct Live {
    names: HashMap<u128, String>,
    cache: xnw_frame::ui::Translated,
    dir: PathBuf,
}

#[derive(Default)]
struct Stats {
    frames: u32,
    with_rows: u32,
    drawn: u32,
    rows: u64,
    ms: f64,
    worst_ms: f64,
    last: String,
    /// LIVE1: frames that showed the last drawn layer again (rows failed or absent).
    held: u32,
    /// Frames that waited for their program pairs' translation.
    translating: u32,
    /// Rows not drawn, by reason: off the plane (by TLS stage), other.
    off_plane: std::collections::BTreeMap<String, u32>,
    refused: u32,
}

/// LIVE1: the last drawn layer and its frame. The title's UI plane holds
/// until the title redraws it: a frame whose rows draw nothing (or fail)
/// shows it again; a frame without UI rows shows it for [`HOLD_ABSENT`]
/// frames, then the plane is empty (the UI is gone).
/// LIVE1: program pairs asked of the translation thread, and those it finished.
static ASKED: Mutex<Option<std::collections::HashSet<(u128, u128)>>> = Mutex::new(None);
static DONE: std::sync::LazyLock<Mutex<xnw_frame::ui::Translated>> = std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

static LAST: Mutex<Option<(u64, Arc<xnw_world::ui::UiLayer>)>> = Mutex::new(None);
const HOLD_ABSENT: u64 = 30;

/// The last drawn layer into `list` (`absent`: the frame has no UI rows).
fn hold(frame: u64, list: &mut FrameList, absent: bool, st: &mut Stats) {
    let g = lock(&LAST);
    if let Some((at, layer)) = g.as_ref()
        && (!absent || frame.saturating_sub(*at) <= HOLD_ABSENT)
    {
        list.ui = Some(layer.clone());
        st.held += 1;
    }
}

static LIVE: Mutex<Option<Result<Live, String>>> = Mutex::new(None);
static STATS: Mutex<Option<Stats>> = Mutex::new(None);
static OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn live() -> Result<Live, String> {
    let host = xnw_gpu::d3d12x::host::host_device().ok_or("no host device")?;
    if !host.caps.software && xnw_gpu::gpuwatch::budget().is_none() && std::env::var("XNW_BRIDGE_HW").ok().as_deref() == Some("1") {
        let ms = std::env::var("XNW_GPU_WATCH_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(500u64);
        xnw_gpu::gpuwatch::start(std::time::Duration::from_millis(ms));
        xnw_core::info!("world render: ui on {}: GPU watchdog {ms} ms per submission", host.caps.adapter);
    }
    let dir = names_dir();
    Ok(Live { names: names()?, cache: HashMap::new(), dir: dir.join("ui-live").join("live") })
}

/// Frame `frame`'s UI layer into `list` (the list thread, before the list is published).
pub(super) fn frame(frame: u64, list: &mut FrameList, data: fp::FramePack, blobs: Arc<HashMap<u128, Vec<u8>>>) {
    let mut sg = lock(&STATS);
    let st = sg.get_or_insert_with(Stats::default);
    st.frames += 1;
    let draws = data.sinks.iter().filter(|s| matches!(s.op, fp::op::DRAW | fp::op::INDEXED)).count();
    if OFF.load(std::sync::atomic::Ordering::Relaxed) {
        return line(frame, st);
    }
    if draws == 0 {
        hold(frame, list, true, st);
        return line(frame, st);
    }
    let mut lg = lock(&LIVE);
    let l = match lg.get_or_insert_with(live) {
        Ok(l) => l,
        Err(e) => {
            xnw_core::warn!("world render: ui: off: {e}");
            OFF.store(true, std::sync::atomic::Ordering::Relaxed);
            return;
        }
    };
    let Some(host) = xnw_gpu::d3d12x::host::host_device() else { return };
    // LIVE1: program pairs first seen are translated on a thread of their own
    // 'worst 583.7 ms', slow frame 1414 'list 628.6 ms' with 4 draws:
    // the translation on the list thread); the frame shows the last layer meanwhile.
    if let Ok(mut done) = DONE.lock() {
        l.cache.extend(done.drain());
    }
    let p = xnw_frame::pack::Pack::in_memory(&l.dir, data, blobs.clone());
    let rows = xnw_frame::ui::rows_of(&p, &l.names);
    let new: Vec<xnw_frame::ui::UiSink> = rows.iter().filter(|r| xnw_frame::ui::drawn_row(r) && !l.cache.contains_key(&(r.vs, r.ps))).cloned().collect();
    if !new.is_empty() {
        let mut asked = lock(&ASKED);
        let ask: Vec<xnw_frame::ui::UiSink> = new.iter().filter(|r| asked.get_or_insert_with(std::collections::HashSet::new).insert((r.vs, r.ps))).cloned().collect();
        drop(asked);
        if !ask.is_empty() {
            let (dir, pack) = (l.dir.clone(), p.data.clone());
            let blobs = blobs.clone();
            let spawned = std::thread::Builder::new().name("xnw-ui-translate".into()).spawn(move || {
                let p = xnw_frame::pack::Pack::in_memory(&dir, pack, blobs);
                for r in &ask {
                    let t = xnw_frame::ui::translate(&p, r);
                    if let Ok(mut d) = DONE.lock() {
                        d.insert((r.vs, r.ps), t);
                    }
                }
            });
            if let Err(e) = spawned {
                xnw_core::warn!("world render: ui: no translation thread: {e}");
            }
        }
        st.with_rows += 1;
        st.rows += rows.len() as u64;
        st.translating += 1;
        st.last = format!("{} rows, {} rows wait for their programs' translation: the last layer stays", rows.len(), new.len());
        hold(frame, list, false, st);
        return line(frame, st);
    }
    let t0 = Instant::now();
    // LIVE1: the live layer stays on the GPU (no wait, readback or upload);
    // a burst reads it back for its layer hash (`XNW_UI_GPU=1`: the live way, to time it).
    let drawn = if super::burst::serving() && std::env::var("XNW_UI_GPU").ok().as_deref() != Some("1") { xnw_frame::ui::draw(&p, host, SIZE, &l.names, &mut l.cache) } else { xnw_frame::ui::draw_gpu(&p, host, SIZE, &l.names, &mut l.cache) };
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    st.ms += ms;
    st.worst_ms = st.worst_ms.max(ms);
    match drawn {
        Ok(d) => {
            // A burst (LIVE1) prints the rows' lines, as the replay does.
            if super::burst::serving() {
                for l in &d.lines {
                    println!("{l}");
                }
            }
            if d.rows > 0 {
                st.with_rows += 1;
                st.rows += d.rows as u64;
            }
            let mut off_here = 0;
            for l in &d.lines {
                if let Some(at) = l.find("TLS stage ") {
                    let stage: String = l[at + 10..].chars().take_while(char::is_ascii_digit).collect();
                    *st.off_plane.entry(stage).or_default() += 1;
                    off_here += 1;
                } else if l.contains("not drawn") || l.contains("not translated") || l.contains("not bound") {
                    st.refused += 1;
                }
            }
            if let Some((g, n)) = d.gpu {
                st.drawn += 1;
                st.last = format!("{} rows, {n} drawn (on the GPU); {}", d.rows, d.lines.first().cloned().unwrap_or_default());
                let layer = Arc::new(xnw_world::ui::UiLayer::Gpu(g));
                *lock(&LAST) = Some((frame, layer.clone()));
                list.ui = Some(layer);
            } else if let Some(layer) = d.layer {
                st.drawn += 1;
                st.last = format!("{} rows, {} drawn, {} px covered, layer hash {:08x}; {}", d.rows, layer.drawn, xnw_world::ui::covered(&layer.rgba), xnw_frame::ui::layer_hash(&layer.rgba), d.lines.first().cloned().unwrap_or_default());
                let layer = Arc::new(xnw_world::ui::UiLayer::Bytes { size: SIZE, rgba: layer.rgba });
                *lock(&LAST) = Some((frame, layer.clone()));
                list.ui = Some(layer);
            } else {
                // Rows all off the plane (stage 8: Scaleform into a texture, V; stage 6:
                // Bink): the title drew no plane this frame: as a frame without UI rows.
                hold(frame, list, d.rows == off_here, st);
                st.last = format!("{} rows, 0 drawn; {}", d.rows, d.lines.first().cloned().unwrap_or_default());
            }
        }
        Err(e) => {
            // SAFETY: a plain query.
            if unsafe { host.device.GetDeviceRemovedReason() }.is_err() {
                xnw_core::warn!("world render: frame {frame}: ui: the device failed in the UI draws ({e}); ui off for the run");
                OFF.store(true, std::sync::atomic::Ordering::Relaxed);
                return;
            }
            st.last = format!("ui draws: {e}");
            hold(frame, list, false, st);
        }
    }
    line(frame, st);
}

/// The UI line, every 60 frames taken (and at the first frame drawn).
fn line(frame: u64, st: &mut Stats) {
    static FIRST: std::sync::Once = std::sync::Once::new();
    if st.drawn == 1 {
        FIRST.call_once(|| xnw_core::info!("world render: frame {frame}: ui: first frame drawn: {}", st.last));
    }
    // A burst (LIVE1) prints every frame.
    if st.frames < 60 && !super::burst::serving() {
        return;
    }
    let per = if st.with_rows > 0 { st.ms / f64::from(st.with_rows) } else { 0.0 };
    let off: Vec<String> = st.off_plane.iter().map(|(k, v)| format!("stage {k}: {v}")).collect();
    xnw_core::info!(
        "world render: frame {frame}: ui: {} frames, {} with UI rows ({} rows), {} drawn, {} showed the last layer again ({} waiting for translations); rows not drawn: off the plane {{{}}}, refused {}; {per:.1} ms a frame with rows (worst {:.1} ms); last: {}",
        st.frames,
        st.with_rows,
        st.rows,
        st.drawn,
        st.held,
        st.translating,
        off.join(", "),
        st.refused,
        st.worst_ms,
        st.last
    );
    *st = Stats::default();
}
