//! The ped pre-skin on the device in the replay. The capture's pre-skin
//! dispatches become items of the list's pre-skin frame and their joined
//! PreSkinD rows its draws; the world pass skins them with our compute
//! program (`xnw_world::preskin_gpu`) and draws each row
//! from its copy. `--preskin-gpu=check` also runs every item through the
//! program on its own and compares the positions with the CPU reference.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use xnw_world::FrameList;
use xnw_world::preskin::{self as ps, PosFormat};
use xnw_world::preskin_gpu::{self as pg, Item};

use crate::pack::{Pack, PreskinDispatch};
use crate::preskin::{load, range_decode, variant_of};

/// What [`fill`] did.
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// PreSkinD list rows; joined to a dispatch; given to an item, by variant.
    pub rows: usize,
    pub joined: usize,
    pub drawn: BTreeMap<u32, usize>,
    /// Rows left as they were, by reason.
    pub left: BTreeMap<String, usize>,
    /// The items with their sinks and variants (for the check).
    pub sinks: Vec<(u32, u32)>,
    /// What the live reader relies on: the b7 slot of each item's dispatch
    /// (root x 96 + slot), and the joined rows by (ped flag, palette).
    pub b7_slots: BTreeMap<u32, usize>,
    pub row_kinds: BTreeMap<(bool, bool), usize>,
    /// Joined rows whose own vertex buffer (address, length) is the
    /// dispatch's vertex view, and those whose is not.
    pub vb_same: (usize, usize),
    /// Joined rows whose hook-time record (draws.tsv `cur_idx`, `range`)
    /// agrees with the row's own record (the sink's, +0x30 and +0x54 in
    /// the range mask): cur_idx same, differs; range word same, differs
    /// (the range only in captures with the column).
    pub hook_rec: [usize; 4],
    /// List rows drawn from a variant-11 item (the faces; `--frame-on` them).
    pub face_rows: Vec<usize>,
}

/// draws.tsv per draw: the hook-time record's cur_idx and range word
/// (None where the capture has no such column or value).
fn hook_words(dir: &Path) -> HashMap<u32, (Option<u32>, Option<u32>)> {
    let Ok(t) = std::fs::read_to_string(dir.join("draws.tsv")) else { return HashMap::new() };
    let mut lines = t.lines();
    let head: Vec<&str> = lines.next().unwrap_or("").split('\t').collect();
    let col = |n: &str| head.iter().position(|h| *h == n);
    let (cd, cc, cr) = (col("draw"), col("cur_idx"), col("range"));
    let num = |s: &str| s.strip_prefix("0x").map_or_else(|| s.parse().ok(), |h| u32::from_str_radix(h, 16).ok());
    lines
        .filter_map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            let get = |c: Option<usize>| c.and_then(|c| f.get(c)).and_then(|s| num(s));
            Some((get(cd)?, (get(cc), get(cr))))
        })
        .collect()
}

/// The list's pre-skin frame from the capture in `dir`: one item per joined
/// dispatch of a covered variant, one draw per joined PreSkinD row (`bits`:
/// the live `preskinbits`).
pub fn fill(dir: &Path, list: &mut FrameList, bits: u32) -> Result<Report, String> {
    let Some(pack) = Pack::open(dir)? else { return Err("pre-skin: no pack".into()) };
    let names = crate::audit::program_names(dir).ok_or_else(|| format!("pre-skin: no shader-names.tsv above {}", dir.display()))?;
    let v1 = pack.list_draws().filter(|v| v.len() == list.draws.len()).ok_or("pre-skin: no list-draws.tsv covering the list")?;
    let joined = pack.preskin_join(&|h| names.get(&h).is_some_and(|n| n.contains("PreSkinD")));
    let by_draw: HashMap<u32, (u32, Option<u32>, u32)> = joined.iter().filter_map(|j| Some((j.draw?, (j.word?, j.dispatch, j.sink)))).collect();
    let hook = hook_words(dir);
    let dispatches: HashMap<u32, PreskinDispatch> = pack.dispatches().into_iter().map(|d| (d.sink, d)).collect();
    let mut rep = Report::default();
    let mut item_of: HashMap<u32, Result<(u32, u32), String>> = HashMap::new();
    let mut frame = pg::PreskinFrame::default();
    for i in 0..list.draws.len() {
        let Some(&(word, dispatch, own)) = by_draw.get(&v1[i]) else { continue };
        rep.rows += 1;
        let Some(sink) = dispatch else {
            *rep.left.entry("no dispatch".into()).or_default() += 1;
            continue;
        };
        rep.joined += 1;
        let made = item_of.entry(sink).or_insert_with(|| {
            let d = &dispatches[&sink];
            let variant = names.get(&d.program).and_then(|n| variant_of(n)).ok_or_else(|| format!("program {:032x} is no standard variant", d.program))?;
            let inp = load(&pack, d)?;
            let (rule, neutral, unmapped) = pg::rule_for(variant, &d.rec.b7, bits, || ps::table_unfilled(&inp.inputs())).map_err(|e| format!("{e} (variant {variant})"))?;
            let mut delta = if pg::takes_delta(variant, bits) { inp.t76.clone() } else { Vec::new() };
            // HEAD2: variant 11 with its table filled (the head graph's) runs its table step.
            let (table, table_normals) = if variant == 11 && !neutral { pg::face_table(&inp.inputs()) } else { (Vec::new(), 0) };
            if variant == 11 && !neutral && crate::le32(&inp.b6, 0xc) == 0 {
                delta = inp.t76.clone();
            }
            frame.items.push(Item { neutral, unmapped, delta, table, table_normals, ..Item::from_inputs(&inp.inputs(), rule) });
            rep.sinks.push((sink, variant));
            *rep.b7_slots.entry(d.rec.b7_slot).or_default() += 1;
            Ok(((frame.items.len() - 1) as u32, variant))
        });
        match made {
            Ok((item, variant)) => {
                // The draw's own range word says whether it reads the extra blocks.
                frame.draws.push((i as u32, *item, ps::range(word).3));
                let r = &list.draws[i];
                let vv = pack.heap_views(sink).into_iter().find(|h| h.k == 0).and_then(|h| h.view);
                if vv.is_some_and(|v| v.addr == r.geom.vb_at && v.bytes == r.geom.vb_len) {
                    rep.vb_same.0 += 1;
                } else {
                    rep.vb_same.1 += 1;
                }
                *rep.row_kinds.entry((r.ped, r.palette != xnw_world::NO_PALETTE)).or_default() += 1;
                // The hook's record against the row's own (the live join key).
                let own_cur = pack.record(own).and_then(|b| Some(u32::from_le_bytes(b.get(0x30..0x34)?.try_into().ok()?)));
                if let Some(&(cur, range)) = hook.get(&v1[i]) {
                    if let (Some(h), Some(o)) = (cur, own_cur) {
                        rep.hook_rec[usize::from(h != o)] += 1;
                    }
                    if let Some(h) = range {
                        rep.hook_rec[2 + usize::from(h & xnw_formats::framepack::RANGE_MASK != word & xnw_formats::framepack::RANGE_MASK)] += 1;
                    }
                }
                *rep.drawn.entry(*variant).or_default() += 1;
                if *variant == 11 {
                    rep.face_rows.push(i);
                }
            }
            Err(e) => *rep.left.entry(e.clone()).or_default() += 1,
        }
    }
    list.preskin = frame;
    Ok(rep)
}

/// One quantisation step of a range format (the largest of its axes).
fn step(format: PosFormat) -> f32 {
    match format {
        PosFormat::Float3 => 0.0,
        PosFormat::Packed10 => f32::from_bits(0x3bc0_300c),
        PosFormat::Packed24 => f32::from_bits(0x38c0_00c0),
    }
}

/// Every item through our program on `host` against the CPU reference:
/// positions within one step of the range format (plus 1e-4 m), NaN where
/// the reference hides a vertex; normals within 0.03 of the reference's
/// 8:8 octahedral ones. Prints one line; Err when a vertex is out.
pub fn check(dir: &Path, list: &FrameList, sinks: &[(u32, u32)], host: &xnw_gpu::host::Host) -> Result<(), String> {
    let Some(pack) = Pack::open(dir)? else { return Err("pre-skin: no pack".into()) };
    let dispatches: HashMap<u32, PreskinDispatch> = pack.dispatches().into_iter().map(|d| (d.sink, d)).collect();
    let items = &list.preskin.items;
    let (gpu, gpu_ms) = pg::run(host, items)?;
    let (mut worst, mut over, mut nan_diff, mut vertices, mut nworst, mut nover) = (0f32, 0usize, 0usize, 0usize, 0f32, 0usize);
    let mut cpu_ms = 0.0;
    let mut per: Vec<String> = Vec::new();
    for ((item, (pos, nrm)), &(sink, variant)) in items.iter().zip(&gpu).zip(sinks) {
        let inp = load(&pack, &dispatches[&sink])?;
        let t = std::time::Instant::now();
        // A neutral face is the reference's variant 11 without its table step.
        // Variant 2's item carries t76: its delta step first. A face with its
        // table (HEAD2): the whole variant-11 rule.
        let before = if variant == 11 && !item.neutral { ps::Before::Tables } else if item.delta.is_empty() { ps::Before::Nothing } else { ps::Before::Delta };
        let w = ps::cpu_rule(&inp.inputs(), item.rule, before);
        cpu_ms += t.elapsed().as_secs_f64() * 1e3;
        let (cpos, cnrm) = range_decode(&w.bytes, item.word());
        let tol = step(ps::range(item.word()).2) + 1e-4;
        let mut e_item = 0f32;
        for (v, (g, c)) in pos.iter().zip(&cpos).enumerate() {
            vertices += 1;
            if c[0].is_nan() || g[0].is_nan() {
                nan_diff += usize::from(c[0].is_nan() != g[0].is_nan());
                continue;
            }
            let e = (0..3).map(|k| (g[k] - c[k]).abs()).fold(0f32, f32::max);
            e_item = e_item.max(e);
            over += usize::from(!(e <= tol));
            if let (Some(gn), Some(cn)) = (nrm.get(v), cnrm.get(v)) {
                let e = (0..3).map(|k| (gn[k] - cn[k]).abs()).fold(0f32, f32::max);
                nworst = nworst.max(e);
                nover += usize::from(!(e <= 0.03));
            }
        }
        if pos.len() != cpos.len() {
            over += pos.len().abs_diff(cpos.len());
        }
        worst = worst.max(e_item);
        per.push(format!("{sink}:v{variant}:{e_item:.2e}"));
    }
    let pass = over == 0 && nan_diff == 0 && nover == 0;
    println!(
        "pre-skin check: {} dispatches, {vertices} vertices: positions max |gpu - cpu| {worst:.3e} m, over one format step {over}, hidden mismatches {nan_diff}; normals max {nworst:.3e}, over 0.03 {nover}; GPU {gpu_ms:.3} ms, CPU reference {cpu_ms:.2} ms ({}) -> {}",
        items.len(),
        host.caps.adapter,
        if pass { "PASS" } else { "FAIL" }
    );
    println!("pre-skin check per dispatch (sink:variant:max error): {}", per.join(" "));
    if pass { Ok(()) } else { Err("pre-skin check: GPU positions differ from the CPU reference".into()) }
}

/// `--preskin-gpu` (the covered variants on the device) or
/// `--preskin-gpu=check` (the same, and the comparison with the CPU
/// reference first), `--preskin-bits <hex>` as the live `preskinbits`.
/// Prints one line. Without the option the list is untouched.
pub fn gpu_arg(args: &[String], dir: &Path, list: &mut FrameList) -> Result<Option<Report>, String> {
    let check_too = match args.iter().find_map(|a| a.strip_prefix("--preskin-gpu")) {
        Some("") => false,
        Some("=check") => true,
        Some(o) => return Err(format!("--preskin-gpu{o}: use --preskin-gpu or --preskin-gpu=check")),
        None => return Ok(None),
    };
    let mut bits = args.iter().position(|a| a == "--preskin-bits").and_then(|i| args.get(i + 1)).and_then(|v| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok()).unwrap_or(0);
    // `headgraph=on`: the faces from the head graph's table (HEAD2).
    if crate::head_graph::switch_on(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default()) {
        bits |= pg::BIT_FACE_TABLE;
    }
    let r = fill(dir, list, bits)?;
    let per = r.drawn.iter().map(|(v, n)| format!("v{v} {n}")).collect::<Vec<_>>().join(", ");
    let left = r.left.iter().map(|(why, n)| format!("{n} {why}")).collect::<Vec<_>>().join("; ");
    println!("pre-skin (gpu): {} PreSkinD rows, {} joined to a dispatch; given to the device pass {} ({}); left as they were: {}; {} items ({} neutral faces: variant 11 from its bind shape; {} unmapped: variants 7-10 without their maps; {} with t76's delta (variant 2); {} faces from their table; bits {bits:#x})", r.rows, r.joined, r.drawn.values().sum::<usize>(), per, if left.is_empty() { "none".into() } else { left }, list.preskin.items.len(), list.preskin.items.iter().filter(|i| i.neutral).count(), list.preskin.items.iter().filter(|i| i.unmapped).count(), list.preskin.items.iter().filter(|i| !i.delta.is_empty()).count(), list.preskin.items.iter().filter(|i| !i.table.is_empty()).count());
    println!("  pre-skin (gpu): dispatches by b7 slot (root x 96 + slot): {:?}; rows by (ped, has palette): {:?}; row vertex buffer = dispatch vertex view: {} rows, differs {}; hook record vs own: cur_idx same {}, differs {}; range same {}, differs {}; face rows {:?}", r.b7_slots, r.row_kinds, r.vb_same.0, r.vb_same.1, r.hook_rec[0], r.hook_rec[1], r.hook_rec[2], r.hook_rec[3], r.face_rows);
    if check_too {
        let host = xnw_gpu::host::Host::new(None).map_err(|e| e.to_string())?;
        check(dir, list, &r.sinks, &host)?;
    }
    Ok(Some(r))
}

/// The part of `frame` whose draws are among `rows` (list indices), the
/// draws renumbered to their places in `rows`; the items stay whole.
pub fn subset(frame: &pg::PreskinFrame, rows: &[usize]) -> pg::PreskinFrame {
    if frame.draws.is_empty() {
        return pg::PreskinFrame::default();
    }
    let draws = frame.draws.iter().filter_map(|&(d, item, normals)| rows.iter().position(|&r| r == d as usize).map(|p| (p as u32, item, normals))).collect();
    pg::PreskinFrame { items: frame.items.clone(), draws, unreadable: frame.unreadable, read_ms: frame.read_ms, reads: frame.reads.clone() }
}
