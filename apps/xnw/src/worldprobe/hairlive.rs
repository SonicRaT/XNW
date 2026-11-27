//! Strand hair in the live pass (HAIR6; `XNW_WORLD_RENDER=...,hair=on`,
//! default off). The replay's path, run per frame on the live title:
//!
//! - the frame pack writer in its live hair mode (`pack::enable_hair`)
//!   records, at the list sink, only the strand chain's dispatches
//!   (`smh_pose_N`, `smh_gen_N`, `smh_sim_N`) and the main-view strand draws
//!   (`hair::STRAND_VS`), with the hook draws' palette spans; the frame's
//!   pack is handed over in memory (`pack::take_live`), no frame is written;
//! - the title's strand chain runs translated through the shared compute
//!   runner (`xnw_frame::hair_chain::run_on`, one runner for the run: its
//!   pipelines stay, its resources go each frame) on our pass's device;
//! - every strand draw runs through the title's vertex program, depth only
//!   (`xnw_frame::hair::draw_strands`, the replay's `--hair-bridge`), and the
//!   nearest strand per pixel goes with the list (`FrameList::hair`); our
//!   pass draws those pixels in hair brown under its depth test
//!   (`xnw_world::hair::HairPass`).
//!
//! HAIR7: the title's thread reads only small records: a hair sink's buffer
//! and root views of 64 KiB or more are left in place (address only,
//! `pack.rs` IN_PLACE); the list thread binds them in title memory
//! (`XNW_ZEROCOPY`, the runner's `in_place`) or reads them there. The pool
//! stays on the runner's device (`hair_chain::run_on_device`); every strand
//! draw reads it there, all in one submission into one depth target
//! (`hair::draw_strands_device`), and that depth stays on the device for our
//! pass (`FrameList::hair_depth`): no pool copy, no readback.
//! `XNW_HAIR_READBACK=on` reads it back too (pixels and depth hash, a check);
//! `XNW_HAIR_DEVICE=off` is HAIR6's path (an A/B).
//!
//! On the hardware adapter the title's programs run only with
//! `XNW_BRIDGE_HW=1` and the per-submission GPU watchdog (started here,
//! `XNW_GPU_WATCH_MS`, default 500). A frame without a strand draw does no
//! work. A chain that does not run whole draws nothing that frame; a device
//! failure turns hair off for the run. The cost per frame (read at the sink
//! is the title's; chain and strands here, on the list thread) is logged
//! every 60 frames.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use xnw_formats::framepack as fp;
use xnw_frame::{hair, hair_chain};
use xnw_world::FrameList;


/// The pass's size (`xnw_world::screen_pass`): the strand pixels are made at it.
const SIZE: (u32, u32) = (1920, 1080);

/// `hair=on` in `XNW_WORLD_RENDER` (read once).
pub fn on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| xnw_world::hair::switch_on(&v)));
    *ON
}

struct Live {
    runner: xnw_world::compute::Runner<'static>,
    programs: hair_chain::Programs,
    vertex: hair::Vertex,
    names: HashMap<u128, String>,
    dir: PathBuf,
}

// SAFETY: D3D12 objects are free-threaded; the live state is used under its lock.
unsafe impl Send for Live {}

/// Counts since the last line: frames taken, with a strand draw, chains run
/// whole, frames drawn, pixels, ms in the chain, ms in the strand draws.
#[derive(Default)]
struct Stats {
    frames: u32,
    with_draws: u32,
    whole: u32,
    drawn: u32,
    pixels: u64,
    chain_ms: f64,
    draw_ms: f64,
    worst_ms: f64,
    last: String,
}

static LIVE: Mutex<Option<Result<Live, String>>> = Mutex::new(None);
static STATS: Mutex<Option<Stats>> = Mutex::new(None);
static OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Where `shader-names.tsv` is looked for: `XNW_SHADER_NAMES` (its
/// directory), else `out` under the working directory.
fn names_dir() -> PathBuf {
    std::env::var("XNW_SHADER_NAMES").ok().and_then(|f| PathBuf::from(f).parent().map(PathBuf::from)).unwrap_or_else(|| PathBuf::from("out"))
}

fn live() -> Result<Live, String> {
    let host = xnw_gpu::d3d12x::host::host_device().ok_or("no host device")?;
    if !host.caps.software && xnw_gpu::gpuwatch::budget().is_none() && std::env::var("XNW_BRIDGE_HW").ok().as_deref() == Some("1") {
        let ms = std::env::var("XNW_GPU_WATCH_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(500u64);
        xnw_gpu::gpuwatch::start(std::time::Duration::from_millis(ms));
        xnw_core::info!("world render: hair on {}: GPU watchdog {ms} ms per submission", host.caps.adapter);
    }
    let mut runner = xnw_world::compute::Runner::new(host)?;
    // HAIR7: the views the pack left in place are bound in title memory (`XNW_ZEROCOPY`).
    runner.in_place = true;
    let dir = names_dir();
    let names = hair_chain::names_in(&dir).ok_or_else(|| format!("no shader-names.tsv in {} (set XNW_SHADER_NAMES)", dir.display()))?;
    Ok(Live { runner, programs: HashMap::new(), vertex: HashMap::new(), names, dir: dir.join("hair-live").join("live") })
}

/// Whether the frame's pack holds a main-view strand draw.
fn strand_draws(data: &fp::FramePack) -> usize {
    let vs: HashMap<u64, u128> = data.programs.iter().filter(|g| g.stage == "vertex").map(|g| (g.pipeline, g.container)).collect();
    data.sinks.iter().filter(|s| s.op == fp::op::DRAW && vs.get(&s.pipeline) == Some(&xnw_world::hair::STRAND_VS)).count()
}

/// Frame `frame`'s strand pixels into `list` (the list thread, before the
/// list is published).
pub(super) fn frame(frame: u64, list: &mut FrameList, data: fp::FramePack, blobs: Arc<HashMap<u128, Vec<u8>>>) {
    let mut sg = lock(&STATS);
    let st = sg.get_or_insert_with(Stats::default);
    st.frames += 1;
    if OFF.load(std::sync::atomic::Ordering::Relaxed) || strand_draws(&data) == 0 {
        return line(frame, st);
    }
    st.with_draws += 1;
    let mut lg = lock(&LIVE);
    let l = match lg.get_or_insert_with(live) {
        Ok(l) => l,
        Err(e) => {
            xnw_core::warn!("world render: hair: off: {e}");
            OFF.store(true, std::sync::atomic::Ordering::Relaxed);
            return;
        }
    };
    let Some(host) = xnw_gpu::d3d12x::host::host_device() else { return };
    // HAIR7: views the title's thread left in place are read here (list thread) when not bound in place.
    let p = xnw_frame::pack::Pack::in_memory(&l.dir, data, blobs).reading(|a, n| super::dcopy(a, n as usize));
    let t0 = Instant::now();
    // HAIR7: the pool stays on the device; the strand draws read it there.
    // HAIR7 `XNW_HAIR_DEVICE=off` (an A/B): HAIR6's path, the pool read back and one submission a strand draw.
    // Probes: `readpool` the pool also read back after the chain; `pixels` our pass draws the pixels read back.
    let probe = std::env::var("XNW_HAIR_DEVICE").unwrap_or_default();
    let old = probe == "off";
    let chain = if old || probe == "readpool" { hair_chain::run_on(&p, &mut l.runner, &l.names, &mut l.programs, false) } else { hair_chain::run_on_device(&p, &mut l.runner, &l.names, &mut l.programs) };
    let chain_ms = t0.elapsed().as_secs_f64() * 1e3;
    st.chain_ms += chain_ms;
    let chain = match chain {
        Ok(c) => c,
        Err(e) => {
            st.last = format!("chain: {e}");
            return line(frame, st);
        }
    };
    // A burst (LIVE1) prints the steps left out, as the replay does.
    if super::burst::serving() {
        for s in chain.steps.iter().filter(|s| s.error.is_some() || s.skipped.is_some()) {
            println!("  chain sink {} {}: {}", s.sink, s.name, s.error.as_deref().or(s.skipped.as_deref()).unwrap_or(""));
        }
    }
    if chain.removed {
        let e = chain.steps.iter().find_map(|s| s.error.clone()).unwrap_or_default();
        xnw_core::warn!("world render: frame {frame}: hair: the device failed in the strand chain ({e}); hair off for the run");
        OFF.store(true, std::sync::atomic::Ordering::Relaxed);
        return;
    }
    if let Some(s) = chain.steps.iter().find(|s| s.error.is_some()) {
        // The whole chain or nothing (UP3: a part draws lines across the screen).
        st.last = format!("{}; sink {} {}: {}", chain.line(), s.sink, s.name, s.error.as_deref().unwrap_or(""));
        return line(frame, st);
    }
    st.whole += 1;
    let t1 = Instant::now();
    if super::burst::serving() {
        println!("  hair frame {frame}: chain {chain_ms:.1} ms; the strand draws next");
    }
    let placed = (l.runner.placed, xnw_frame::compute::in_place_taken());
    let pool = if old { Ok(None) } else { l.runner.resource(xnw_frame::compute::key(chain.pool_at)) };
    // HAIR7 `XNW_HAIR_READBACK=on` (a check, default off): the strand depth is also read back (pixels, hash).
    let read = std::env::var("XNW_HAIR_READBACK").is_ok_and(|v| v == "on");
    let drawn = if old {
        hair::draw_strands(&p, host, SIZE, Some(&chain), false, &mut l.vertex).map(|(u, lines, draws)| hair::Drawn { read: Some(u), device: None, lines, draws })
    } else {
        pool.and_then(|pool| hair::draw_strands_device(&p, host, SIZE, &chain, pool.as_ref(), &mut l.vertex, read))
    };
    let draw_ms = t1.elapsed().as_secs_f64() * 1e3;
    st.draw_ms += draw_ms;
    st.worst_ms = st.worst_ms.max(chain_ms + draw_ms);
    match drawn {
        Ok(d) => {
            st.drawn += 1;
            let seen = match &d.read {
                Some(strands) => {
                    let px = strands.pixels(SIZE.0);
                    st.pixels += px.len() as u64;
                    let text = format!("{} pixels, depth hash {:08x}", px.len(), strands.hash());
                    // More than one strand program: their union, as pixels (HAIR6's path).
                    if d.device.is_none() || !read || probe == "pixels" {
                        list.hair = Some(Arc::new(px));
                    }
                    text
                }
                None => "depth on the device".to_string(),
            };
            if let Some(dev) = d.device.filter(|_| list.hair.is_none()) {
                list.hair_depth = Some(Arc::new(dev));
            }
            st.last = format!("{}; {} strand draws, {seen}; chain views in place {} (left by the pack {}); {}", chain.line(), d.draws, placed.0, placed.1, d.lines.join("; "));
        }
        Err(e) => {
            // SAFETY: a plain query.
            if unsafe { host.device.GetDeviceRemovedReason() }.is_err() {
                xnw_core::warn!("world render: frame {frame}: hair: the device failed in the strand draws ({e}{}); hair off for the run", xnw_world::bridge::layer_errors(&host.device));
                OFF.store(true, std::sync::atomic::Ordering::Relaxed);
                return;
            }
            st.last = format!("strand draws: {e}");
        }
    }
    line(frame, st);
}

/// The hair line, every 60 frames taken (and at the first frame drawn).
fn line(frame: u64, st: &mut Stats) {
    static FIRST: std::sync::Once = std::sync::Once::new();
    if st.drawn == 1 {
        FIRST.call_once(|| xnw_core::info!("world render: frame {frame}: hair: first frame drawn: {}", st.last));
    }
    // A burst (LIVE1) prints every frame.
    if st.frames < 60 && !super::burst::serving() {
        return;
    }
    let per = |ms: f64| if st.whole > 0 { ms / f64::from(st.with_draws) } else { 0.0 };
    xnw_core::info!(
        "world render: frame {frame}: hair: {} frames, {} with strand draws, {} chains whole, {} drawn, {} strand pixels; per frame with strands: chain {:.1} ms, strand draws {:.1} ms (worst {:.1} ms); last: {}",
        st.frames,
        st.with_draws,
        st.whole,
        st.drawn,
        st.pixels,
        per(st.chain_ms),
        per(st.draw_ms),
        st.worst_ms,
        st.last
    );
    *st = Stats::default();
}
