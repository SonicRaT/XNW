//! `xnw world-render <frame dir> --adapter warp --bridgems on|<hash>+<hash>
//! [--bridgeps on|<hash>+...] [--out <png>] [--bridgems-repeat N]` (or
//! `XNW_WORLD_RENDER=...,bridgems=...`): the frame through our pass
//! twice. First ours alone; then with every main-view row whose mesh
//! program is named drawn from the title's mesh program in compute form
//! inside our pass (`xnw_world::bridge_ms`), with our pixel stage, or with
//! the title's pixel program where `--bridgeps` names it too. Bindings are
//! the bridge audit's (`bridge::load`, `bridge::bind`).
//!
//! Per bridged row: the pixels it covers in each picture (the id views) and
//! the depth where both show it; the verdict as the audit's (at most 8 px
//! or 2 % on one side alone, median relative depth within 1e-3). Writes
//! `--out <png>`: ours above, bridged below (our albedo view;
//! `--bridgems-lit`: our shading). `--bridgems-repeat N`: GPU ms per frame
//! (median of N) of ours, ours without the bridged rows, and bridged.
//! `--bridgems-tsv <file>`: the rows; `--bridgems-layouts`: each program's
//! record elements and the join.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use xnw_world::bridge;
use xnw_world::bridge_ms::read_depth;
use xnw_world::{FrameList, GeomKey, HostGeometry, Pass, RAW_SHADER, TexMode};

use crate::bridge::{Row, load, programs};
use crate::load::load_list;
use crate::pack::Pack;

/// Pixels per draw in an id picture (RGBA8 rows, id = list index + 1 in
/// the low 24 bits), as (id -> pixel indices); ids past `max` are the clear.
pub fn ids(rgba: &[u8], max: u32) -> HashMap<u32, Vec<u32>> {
    let mut out: HashMap<u32, Vec<u32>> = HashMap::new();
    for (i, px) in rgba.chunks_exact(4).enumerate() {
        let id = u32::from(px[0]) | u32::from(px[1]) << 8 | u32::from(px[2]) << 16;
        if id != 0 && id <= max {
            out.entry(id).or_default().push(i as u32);
        }
    }
    out
}

/// The audit's verdict for one row: ours and bridged pixel counts, pixels
/// both show, median relative depth difference over those.
pub fn verdict(ours: usize, theirs: usize, both: usize, rel: f64) -> &'static str {
    let lone = (ours - both).max(theirs - both);
    match (ours, theirs) {
        (0, 0) => "hidden",
        (0, _) => "ours none",
        (_, 0) => "theirs none",
        _ if lone as f64 > 8f64.max(0.02 * (ours + theirs - both) as f64) => "cover",
        _ if rel > 1e-3 => "depth",
        _ => "agree",
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// `--bridgems`.
pub fn bridgems_cli(args: &[String], dir: &Path, host: &xnw_gpu::host::Host, pass: &mut Pass, wanted: &[u32], wanted_ps: &[u32]) -> Result<(), String> {
    xnw_world::bridge::title_programs_here(host).map_err(|e| format!("bridgems: {e}"))?;
    let (mut list, geoms) = load_list(dir)?;
    let pack = Pack::open(dir)?.ok_or("--bridgems needs a v2 capture (pack\\manifest.tsv)")?;
    let mut args = args.to_vec();
    if !args.iter().any(|a| a == "--bridge-class") {
        args.extend(["--bridge-class".to_string(), "static".to_string()]);
    }
    let rows = crate::bridge::audit_rows(&args, &list, &pack)?;
    if !args.iter().any(|a| a == "--bridgems-lit") {
        pass.set_raw_shader(&format!("#define LAB_VIEW 6\n{RAW_SHADER}"))?;
    }
    (pass.sky, pass.water, pass.terrain.opts.on) = (false, false, false);
    pass.tex.opts.mode = TexMode::Provided;
    let counts = Arc::new([const { AtomicU64::new(0) }; 3]);
    pass.tex.set_provider(Some(pack.texel_provider(counts.clone())));
    let mut texels = pack.texel_provider(counts);
    let mut decode = |key: &GeomKey| geoms.get(key).map(|(vb, idx)| HostGeometry { vertices: Vec::new(), indices: idx.clone(), raw_vb: Some(vb.clone()), vb_place: 0 });
    let (w, h, n) = (pass.width, pass.height, list.draws.len() as u32);

    // The batch: programs once per pipeline, pipelines once per pair, one binding per row.
    let dev = &host.device;
    let t0 = std::time::Instant::now();
    let mut batch = pass.bridge_ms.take().unwrap_or_default();
    // The uploads' submissions under the watchdog, with their line.
    crate::replay::watchdog_arm();
    batch.begin(dev)?;
    let mut progs: HashMap<u64, Result<_, String>> = HashMap::new();
    // Pipelines whose title pixel program this adapter may not run, and those said so.
    let mut refused: HashMap<u64, Option<String>> = HashMap::new();
    let mut noted: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let empty_ps = xnw_shader::emit::HostShader { stage: xnw_shader::HwStage::Ps, dxil: Vec::new(), user_data: Vec::new(), samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: Vec::new(), root_srvs: Vec::new(), root_uavs: Vec::new(), half_views: Vec::new(), address_map: false, descriptor_words: false };
    let (mut bridged, mut failed, mut other) = (Vec::new(), Vec::new(), 0u32);
    let mut layouts: Vec<u32> = Vec::new();
    // `XNW_ZEROCOPY` (ZC2): the title ranges every row will bind, first, so
    // views of one span of title memory share one buffer as they share the
    // title's memory (rows name overlapping views of one buffer at their
    // own addresses: MSB1's per-row large views).
    if xnw_gpu::shared::world_part(xnw_gpu::shared::WORLD_BRIDGE) {
        let mut ranges = Vec::new();
        for &(k, draw, sink) in &rows {
            let row = &list.draws[k];
            let pipeline = pack.data.sinks.iter().find(|s| s.index == sink).map_or(0, |s| s.pipeline);
            let Ok((_, used, hs)) = progs.entry(pipeline).or_insert_with(|| programs(&pack, sink)) else { continue };
            if !wanted.contains(&((hs[0] >> 96) as u32)) {
                continue;
            }
            let loaded = load(&pack, sink, &Row::with_palette(row, &list, &pack, draw), geoms.get(&row.geom).map(|g| &g.0[..]), used, &mut texels);
            ranges.extend(loaded.ranges(bridge::KEEP_MIN));
        }
        batch.use_ranges(ranges);
    }
    for &(k, draw, sink) in &rows {
        let row = &list.draws[k];
        let pipeline = pack.data.sinks.iter().find(|s| s.index == sink).map_or(0, |s| s.pipeline);
        let entry = progs.entry(pipeline).or_insert_with(|| programs(&pack, sink));
        let Ok((p, used, hs)) = entry else {
            other += 1;
            continue;
        };
        let ms = (hs[0] >> 96) as u32;
        if !wanted.contains(&ms) {
            other += 1;
            continue;
        }
        // The row's mesh program runs either way; the title's pixel program
        // only where this adapter may run it (else ours shades the records).
        let mut title_ps = wanted_ps.contains(&((hs[1] >> 96) as u32));
        if title_ps && let Some(why) = refused.entry(pipeline).or_insert_with(|| crate::bridge::pixel_refused(&pack, hs, host.caps.software)) {
            if noted.insert(pipeline) {
                println!("bridgems: {why}: its rows are shaded by our pixel program");
            }
            title_ps = false;
        }
        if args.iter().any(|a| a == "--bridgems-layouts") && !layouts.contains(&ms) {
            layouts.push(ms);
            let lay = &p.layout;
            let els: Vec<String> = lay.elements.iter().map(|e| format!("{}{}@{}.{}", e.name, e.index, e.row, e.col)).collect();
            println!("bridgems: mesh {ms:08x}: {} vertices x {} rows, {} primitives; record {} B; elements {}; join {:?}", lay.max_vertices, lay.rows, lay.max_prims, lay.record_bytes(), els.join(" "), xnw_world::bridge_ms::Join::of(lay));
        }
        if let Some(gap) = bridge::gap(&host.caps, &p.needs()) {
            failed.push(format!("row {k}: {gap}"));
            continue;
        }
        let mut loaded = load(&pack, sink, &Row::with_palette(row, &list, &pack, draw), geoms.get(&row.geom).map(|g| &g.0[..]), used, &mut texels);
        loaded.direct_meshlets();
        let d = loaded.draw();
        let addrs = loaded.addrs();
        let added = bridge::bind(&p.ms, if title_ps { &p.ps } else { &empty_ps }, &d).and_then(|b| {
            let pk = batch.program(dev, (*hs, title_ps), || Ok(p.clone()))?;
            if !title_ps && batch.prog(pk).join.is_none() {
                return Err("the mesh program writes no SV_Position".into());
            }
            batch.add(dev, k as u32, pk, &b, &addrs)
        });
        match added {
            Ok(()) => bridged.push((k, title_ps)),
            Err(e) => failed.push(format!("row {k}: {e}")),
        }
    }
    batch.finish(dev, &host.queue)?;
    crate::replay::watchdog_disarm();
    println!("  {}", crate::replay::watch_line("bridgems uploads"));
    let st = batch.stats.clone();
    println!(
        "bridgems: {} rows of {} mesh programs ({} with the title's pixel program), {} groups; {} pipelines made in {:.0} ms; bindings {:.1} MB ({:.1} MB shared, {} large buffers merged, largest view heap {}; new MB by kind: large views {:.1}, small views {:.1}, textures {:.1}, root views {:.1}) in {:.0} ms; ring {:.1} MB; {} failed; {other} rows of other programs; prepared in {:.2} s; title memory (XNW_ZEROCOPY): in place {} views, on title spans {} views ({:.1} MB made of {} spans, {:.1} MB), {} conflicts",
        st.rows,
        st.programs,
        bridged.iter().filter(|b| b.1).count(),
        st.groups,
        st.pipelines_made,
        st.pipeline_ms,
        st.upload_bytes as f64 / 1e6,
        st.shared_bytes as f64 / 1e6,
        st.merged,
        st.heap_slots,
        st.new[0] as f64 / 1e6,
        st.new[1] as f64 / 1e6,
        st.new[2] as f64 / 1e6,
        st.new[3] as f64 / 1e6,
        st.upload_ms,
        st.ring_bytes as f64 / 1e6,
        failed.len(),
        t0.elapsed().as_secs_f64(),
        st.in_place,
        st.by_addr[0],
        st.by_addr[1] as f64 / 1e6,
        st.spans.0,
        st.spans.1 as f64 / 1e6,
        st.by_addr[2]
    );
    let o = st.off_span;
    println!(
        "bridgems: large views off the spans (views, MB): no title address {} ({:.1}), outside every span {} ({:.1}), a byte another row holds differently {} ({:.1}), offset not a multiple of the unit {} ({:.1}); own words shadow {} B of other rows' copies, wrote over {} B",
        o[0].0,
        o[0].1 as f64 / 1e6,
        o[1].0,
        o[1].1 as f64 / 1e6,
        o[2].0,
        o[2].1 as f64 / 1e6,
        o[3].0,
        o[3].1 as f64 / 1e6,
        st.shadowed,
        st.overrode
    );
    println!("bridgems: device memory after the uploads: {}; this process: {}", xnw_gpu::d3d12x::resource::video_memory_text(), process_memory());
    for f in failed.iter().take(8) {
        println!("  {f}");
    }

    // Ours, then bridged: colour, ids, depth.
    let mut frame = 0u64;
    let mut run = |pass: &mut Pass, list: &mut FrameList, id: bool| -> Result<(xnw_world::PassStats, Vec<u8>), String> {
        frame += 1;
        (list.frame, pass.id_mode) = (frame, id);
        let st = pass.draw(list, &mut decode, true);
        pass.id_mode = false;
        let st = st?;
        Ok((st, pass.read_rgba()?))
    };
    crate::replay::watchdog_arm();
    let r = run(pass, &mut list, false).and_then(|a| run(pass, &mut list, true).map(|b| (a, b)));
    crate::replay::watchdog_disarm();
    println!("  {}", crate::replay::watch_line("bridgems ours (colour, ids)"));
    let ((st_ours, ours), (_, ours_id)) = r?;
    let into = pass.bridge_into();
    let ours_z = read_depth(host, &into.depth, w, h)?;
    pass.bridge_ms = Some(batch);
    // `--warm`: the bridged frame once first (its pictures thrown away), then the timed one.
    if args.iter().any(|a| a == "--warm") {
        crate::replay::watchdog_arm();
        let r = run(pass, &mut list, false);
        crate::replay::watchdog_disarm();
        println!("  {}", crate::replay::watch_line("bridgems bridged warm-up"));
        r?;
    }
    crate::replay::watchdog_arm();
    let r = run(pass, &mut list, false).and_then(|a| run(pass, &mut list, true).map(|b| (a, b)));
    crate::replay::watchdog_disarm();
    println!("  {}", crate::replay::watch_line("bridgems bridged (colour, ids)"));
    let ((st_br, theirs), (_, theirs_id)) = r?;
    let theirs_z = read_depth(host, &into.depth, w, h)?;
    println!("bridgems: ours {} draws; bridged {} rows from records ({} with the title's pixel program) + {} of ours", st_ours.draws, st_br.bridged, st_br.bridged_title_ps, st_br.draws);

    // Coverage and depth per bridged row.
    let (oi, ti) = (ids(&ours_id, n), ids(&theirs_id, n));
    let mut verdicts: std::collections::BTreeMap<&str, u32> = Default::default();
    let mut lines = vec!["row\tmesh\ttitle_ps\tours\ttheirs\tboth\tmedian_rel\tverdict".to_string()];
    let mesh_of = |k: usize| rows.iter().find(|r| r.0 == k).and_then(|r| pack.data.sinks.iter().find(|s| s.index == r.2)).and_then(|s| progs.get(&s.pipeline)).and_then(|e| e.as_ref().ok()).map_or(0, |e| (e.2[0] >> 96) as u32);
    let mut first = Vec::new();
    // The frame's constant depth offset (theirs - ours: +5.0e-6 on pk3 f1241, a
    // projection with a far plane against our infinite one), taken out.
    let offset = median(bridged.iter().flat_map(|&(k, _)| {
        let t: std::collections::HashSet<u32> = ti.get(&(k as u32 + 1)).map(|v| v.iter().copied().collect()).unwrap_or_default();
        oi.get(&(k as u32 + 1)).map(|v| v.iter().filter(|p| t.contains(p)).map(|&p| f64::from(theirs_z[p as usize]) - f64::from(ours_z[p as usize])).collect::<Vec<_>>()).unwrap_or_default()
    }).collect());
    for &(k, tps) in &bridged {
        let id = k as u32 + 1;
        let (o, t) = (oi.get(&id).map_or(&[][..], |v| &v[..]), ti.get(&id).map_or(&[][..], |v| &v[..]));
        let theirs_set: std::collections::HashSet<u32> = t.iter().copied().collect();
        let rel: Vec<f64> = o.iter().filter(|p| theirs_set.contains(p)).map(|&p| {
            let (a, b) = (f64::from(ours_z[p as usize]), f64::from(theirs_z[p as usize]));
            (b - a - offset).abs() / a.abs().max(1e-12)
        }).collect();
        let both = rel.len();
        let rel = median(rel);
        let v = verdict(o.len(), t.len(), both, rel);
        *verdicts.entry(v).or_default() += 1;
        if v != "agree" && first.len() < 6 {
            first.push(format!("row {k} ({:08x}): {v}, ours {}, bridged {}, both {both}, depth {rel:.2e}", mesh_of(k), o.len(), t.len()));
        }
        lines.push(format!("{k}\t{:08x}\t{}\t{}\t{}\t{both}\t{rel:.3e}\t{v}", mesh_of(k), u8::from(tps), o.len(), t.len()));
    }
    let changed = ours.chunks_exact(4).zip(theirs.chunks_exact(4)).filter(|(a, b)| a[..3] != b[..3]).count();
    println!("bridgems: depth offset bridged - ours {offset:.4e}; rows by verdict (ours vs bridged, id views of the whole frame; hidden: no pixel in either): {}; {changed} pixels of the picture changed", verdicts.iter().map(|(k, n)| format!("{k} {n}")).collect::<Vec<_>>().join(", "));
    for f in &first {
        println!("  {f}");
    }
    if let Some(t) = args.iter().position(|a| a == "--bridgems-tsv").and_then(|i| args.get(i + 1)) {
        std::fs::write(t, lines.join("\n") + "\n").map_err(|e| format!("{t}: {e}"))?;
    }
    if let Some(out) = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)) {
        let mut sheet = ours;
        sheet.extend_from_slice(&theirs);
        crate::replay::write_png(Path::new(out), w, 2 * h, &sheet).map_err(|e| e.to_string())?;
        println!("bridgems: sheet (ours above, bridged below) -> {out}");
    }

    // GPU ms per frame: ours, ours without the bridged rows, bridged.
    if let Some(reps) = args.iter().position(|a| a == "--bridgems-repeat").and_then(|i| args.get(i + 1)).and_then(|v| v.parse::<usize>().ok()) {
        let batch = pass.bridge_ms.take();
        let mut time = |pass: &mut Pass, list: &mut FrameList| -> Result<f64, String> {
            let mut ms = Vec::new();
            for _ in 0..reps {
                ms.push(run(pass, list, false)?.0.gpu_ms);
            }
            Ok(median(ms))
        };
        let full = time(pass, &mut list)?;
        let keep = list.draws.clone();
        // Without the rows: moved out of the main view by stage (another view).
        let mut without = list.clone();
        for &(k, _) in &bridged {
            without.draws[k].stage = 12;
        }
        let less = time(pass, &mut without)?;
        list.draws = keep;
        pass.bridge_ms = batch;
        crate::replay::watchdog_arm();
        let br = time(pass, &mut list);
        crate::replay::watchdog_disarm();
        println!("  {}", crate::replay::watch_line(&format!("bridgems repeat x{reps}")));
        let br = br?;
        println!("bridgems: GPU ms per frame (median of {reps}) on {}: ours {full:.3}, ours without the {} rows {less:.3}, bridged {br:.3}; the rows cost ours {:.3} ms, bridged {:.3} ms ({:.1} us a row)", host.caps.adapter, bridged.len(), full - less, br - less, (br - less) * 1e3 / bridged.len().max(1) as f64);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_and_verdicts() {
        let rgba = [2, 0, 0, 255, 23, 23, 26, 255, 2, 0, 0, 255, 0, 1, 0, 255];
        let m = ids(&rgba, 300);
        assert_eq!(m.get(&2), Some(&vec![0, 2]));
        assert_eq!(m.get(&256), Some(&vec![3]));
        assert_eq!(m.len(), 2);
        assert_eq!(verdict(1000, 1004, 1000, 1e-5), "agree");
        assert_eq!(verdict(1000, 1100, 1000, 1e-5), "cover");
        assert_eq!(verdict(1000, 1000, 1000, 1e-2), "depth");
        assert_eq!(verdict(0, 50, 0, 0.0), "ours none");
        assert_eq!(verdict(0, 0, 0, 0.0), "hidden");
    }
}

/// This process's commit and peak working set (bridgems' footprint, ZC2).
fn process_memory() -> String {
    use xnw_core::win::{GetCurrentProcess, K32GetProcessMemoryInfo, ProcessMemoryCounters};
    // SAFETY: a plain query into a local of the size given.
    let mut info: ProcessMemoryCounters = unsafe { std::mem::zeroed() };
    info.cb = std::mem::size_of::<ProcessMemoryCounters>() as u32;
    // SAFETY: as above.
    if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut info, info.cb) } == 0 {
        return "unknown".into();
    }
    format!("commit {} MiB (peak {}), working set peak {} MiB", info.pagefile_usage >> 20, info.peak_pagefile_usage >> 20, info.peak_working_set_size >> 20)
}
