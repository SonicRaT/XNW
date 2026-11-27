//! `xnw-lab geodiff <frame dir> --adapter warp [--class a,b] [--rows a-b,c]
//! [--tsv <file>] [--resume] [--limit N] [--masks <dir>]`: the title's own mesh
//! programs as the oracle for our geometry, row by row.
//!
//! Per main-view row paired with a sink whose mesh program translates:
//! - theirs: the mesh program in compute form (`xnw_world::bridge::run`,
//!   depth only, records read back): the covered pixels with their depth,
//!   and from the records the vertices and primitives it asked for, the
//!   primitives it culled (bit 31) and every vertex's clip position;
//! - ours: the row alone through our pass, twice: with the frame's TAA
//!   jitter kept (what the title's view constants hold) and without it (the
//!   live pass), each an id draw (pixels) and a depth-bits draw.
//!
//! One TSV line a row ([`HEADER`]): counts on both sides, the two covers,
//! the distance in pixels between them (`c_max`/`c_mean`: every pixel one
//! side covers alone to the nearest pixel of the other; the mean is over
//! the union), the distance of the title's vertices to our cover
//! (`v_max`/`v_mean`: vertices its own raster shows within a pixel), the
//! depth difference theirs - ours over the pixels both cover (mean,
//! smallest, largest, and how many pixels the title's depth is behind ours:
//! those fail a GREATER_EQUAL test against our depth), the same against
//! our pass without the jitter, and what the run cost. The summary by
//! class and by mesh program follows on stdout; names and the report are
//! `tools/geodiff_report.py`'s.
//!
//! Two processes, as the bridge audit: `--side ours --ours <file>`
//! first, on the inbox software adapter (no `XNW_WARP`: our pass removes the
//! WARP 1.0.22 device at its first draw), writes our covers; the run of
//! the title's programs (`XNW_WARP` set) reads them with `--ours <file>`.
//!
//! Limits: instance lists and the visible-meshlet list are GPU
//! outputs no capture holds: theirs draws instance 0 and every meshlet in
//! order. A program that removes the software device ends the process:
//! `--resume` goes on after the row it was on (marked `device removed`).

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::PathBuf;

use xnw_world::bridge;
use xnw_world::textures::Provider;
use xnw_world::{DECAL_STAGES, DrawRecord, FrameList, GeomKey, HostGeometry, NO_PALETTE};

use crate::bridge::{Row, depth_shader, load, mesh_program};
use crate::le32;
use crate::load::load_list;
use crate::pack::Pack;

pub const HEADER: &str = "frame\trow\tdraw\tsink\tclass\tstage\tmesh\tpixel\tinst\tbones\tgroups\tt_groups_live\tt_verts\tt_distinct\tt_prims\tt_culled\tt_behind\to_verts\to_tris\to_draws\to_inst\to_skinned\tt_px\to_px\tboth\tc_max\tc_mean\tv_n\tv_max\tv_mean\tdz_mean\tdz_min\tdz_max\tdz_fail\tonj_px\tnj_both\tnj_dz_mean\tnj_dz_min\tnj_dz_max\tnj_fail\tseconds\trec_bytes\tverdict\tnote";

/// The G-buffer stage a decal row is drawn as for its depth (no bias of ours).
const OPAQUE_STAGE: u32 = 70;
/// The depth prepass (65) and the alpha surfaces' first two passes (12 alpha prepass, 13: the same rows
/// come again at stage 14, which our pass draws): here their rows are drawn as G-buffer rows.
const PREPASS_STAGES: [u32; 3] = [65, 12, 13];

/// What kind of row our pass takes a list row for.
pub fn class(list: &FrameList, k: usize) -> &'static str {
    let r = &list.draws[k];
    if DECAL_STAGES.contains(&r.stage) {
        "decal"
    } else if r.ped {
        "ped"
    } else if list.rigid.iter().any(|(i, _)| *i as usize == k) {
        "rigid part"
    } else if r.palette != NO_PALETTE {
        "skinned"
    } else if r.inst_count != 0 {
        "instanced"
    } else {
        "static"
    }
}

/// Squared Euclidean distance of every cell of a `w` x `h` grid to the
/// nearest set cell (Felzenszwalb and Huttenlocher's two passes); all
/// `f32::MAX` when none is set.
pub fn distance(set: &[bool], w: usize, h: usize) -> Vec<f32> {
    const INF: f32 = 1e18;
    fn line(f: &[f32], d: &mut [f32], v: &mut [usize], z: &mut [f32]) {
        let n = f.len();
        let (mut k, mut q) = (0usize, 1usize);
        (v[0], z[0], z[1]) = (0, -INF, INF);
        while q < n {
            let s = loop {
                let p = v[k];
                let s = ((f[q] + (q * q) as f32) - (f[p] + (p * p) as f32)) / (2.0 * q as f32 - 2.0 * p as f32);
                if s <= z[k] && k > 0 {
                    k -= 1;
                } else {
                    break s;
                }
            };
            k += 1;
            (v[k], z[k], z[k + 1]) = (q, s, INF);
            q += 1;
        }
        k = 0;
        for (q, o) in d.iter_mut().enumerate() {
            while z[k + 1] < q as f32 {
                k += 1;
            }
            let p = v[k];
            *o = (q as f32 - p as f32) * (q as f32 - p as f32) + f[p];
        }
    }
    let mut g: Vec<f32> = set.iter().map(|s| if *s { 0.0 } else { INF }).collect();
    let n = w.max(h);
    let (mut f, mut d, mut v, mut z) = (vec![0.0; n], vec![0.0; n], vec![0usize; n], vec![0.0; n + 1]);
    for x in 0..w {
        for y in 0..h {
            f[y] = g[y * w + x];
        }
        line(&f[..h], &mut d[..h], &mut v, &mut z);
        for y in 0..h {
            g[y * w + x] = d[y];
        }
    }
    for y in 0..h {
        f[..w].copy_from_slice(&g[y * w..(y + 1) * w]);
        line(&f[..w], &mut d[..w], &mut v, &mut z);
        g[y * w..(y + 1) * w].copy_from_slice(&d[..w]);
    }
    if set.iter().any(|s| *s) { g } else { vec![f32::MAX; w * h] }
}

/// The pixels a draw covers with their depth: (y * width + x, z), by pixel.
pub type Cover = Vec<(u32, f32)>;

/// Two covers and the title's vertices (screen x, y) against each other.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Diff {
    pub both: usize,
    /// Pixels one side covers alone to the other's nearest: largest and
    /// the mean over the union (None: one side is empty).
    pub c: Option<(f32, f32)>,
    /// The title's vertices its own cover shows (within a pixel) to our
    /// cover: how many, largest, mean.
    pub v: Option<(usize, f32, f32)>,
    /// z theirs - z ours over the pixels both cover: mean, smallest,
    /// largest, pixels below zero.
    pub dz: Option<(f64, f32, f32, usize)>,
}

pub fn diff(theirs: &Cover, ours: &Cover, verts: &[(f32, f32)], width: u32) -> Diff {
    let mut out = Diff::default();
    let (mut i, mut j, mut sum, mut lo, mut hi, mut fail) = (0, 0, 0f64, f32::MAX, f32::MIN, 0usize);
    while i < theirs.len() && j < ours.len() {
        match theirs[i].0.cmp(&ours[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                let d = theirs[i].1 - ours[j].1;
                sum += f64::from(d);
                (lo, hi) = (lo.min(d), hi.max(d));
                fail += usize::from(d < 0.0);
                out.both += 1;
                (i, j) = (i + 1, j + 1);
            }
        }
    }
    if out.both > 0 {
        out.dz = Some((sum / out.both as f64, lo, hi, fail));
    }
    if theirs.is_empty() || ours.is_empty() {
        return out;
    }
    // The box of both covers, one pixel wider.
    let xy = |p: u32| ((p % width) as usize, (p / width) as usize);
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for (p, _) in theirs.iter().chain(ours) {
        let (x, y) = xy(*p);
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
    }
    (x0, y0) = (x0.saturating_sub(1), y0.saturating_sub(1));
    let (w, h) = (x1 - x0 + 2, y1 - y0 + 2);
    let grid = |c: &Cover| {
        let mut g = vec![false; w * h];
        for (p, _) in c {
            let (x, y) = xy(*p);
            g[(y - y0) * w + x - x0] = true;
        }
        g
    };
    let (gt, go) = (grid(theirs), grid(ours));
    let (dt, d_o) = (distance(&gt, w, h), distance(&go, w, h));
    let (mut max, mut total) = (0f32, 0f64);
    for k in 0..w * h {
        let d = match (gt[k], go[k]) {
            (true, false) => d_o[k].sqrt(),
            (false, true) => dt[k].sqrt(),
            _ => continue,
        };
        max = max.max(d);
        total += f64::from(d);
    }
    out.c = Some((max, (total / (theirs.len() + ours.len() - out.both) as f64) as f32));
    let (mut n, mut vmax, mut vsum) = (0usize, 0f32, 0f64);
    for &(x, y) in verts {
        let (xi, yi) = (x.floor() as isize - x0 as isize, y.floor() as isize - y0 as isize);
        if xi < 0 || yi < 0 || xi as usize >= w || yi as usize >= h {
            continue;
        }
        let k = yi as usize * w + xi as usize;
        if dt[k] > 2.0 {
            continue;
        }
        let d = d_o[k].sqrt();
        n += 1;
        vmax = vmax.max(d);
        vsum += f64::from(d);
    }
    if n > 0 {
        out.v = Some((n, vmax, (vsum / n as f64) as f32));
    }
    out
}

/// What a mesh program's records hold, over all groups.
#[derive(Clone, Debug, Default)]
pub struct Records {
    pub live: usize,
    pub verts: u64,
    pub distinct: usize,
    pub prims: u64,
    pub culled: u64,
    /// Vertices with w <= 0 or a position that is not finite.
    pub behind: u64,
    /// Screen positions (pixels) of the vertices in front of the camera.
    pub screen: Vec<(f32, f32)>,
    /// Their box in normalised device coordinates: x, y, z smallest, then largest.
    pub ndc: [f32; 6],
}

pub fn records(rec: &[u8], lay: &xnw_shader::emit::mesh::Layout, size: (u32, u32)) -> Option<Records> {
    let (bytes, stride) = (lay.record_bytes() as usize, lay.vertex_stride() as usize);
    let row = lay.position_row()? as usize;
    let f = |b: &[u8], at: usize| f32::from_bits(le32(b, at));
    let mut out = Records { ndc: [f32::MAX, f32::MAX, f32::MAX, f32::MIN, f32::MIN, f32::MIN], ..Records::default() };
    let mut seen = std::collections::HashSet::new();
    for g in rec.chunks_exact(bytes) {
        let (nv, np) = (le32(g, 0) as usize, le32(g, 4) as usize);
        out.live += usize::from(nv > 0);
        out.verts += nv as u64;
        out.prims += np as u64;
        for v in 0..nv.min(lay.max_vertices as usize) {
            let at = 16 + v * stride + 16 * row;
            let p = [f(g, at), f(g, at + 4), f(g, at + 8), f(g, at + 12)];
            seen.insert([le32(g, at), le32(g, at + 4), le32(g, at + 8), le32(g, at + 12)]);
            if p.iter().any(|x| !x.is_finite()) || p[3] <= 0.0 {
                out.behind += 1;
                continue;
            }
            for (k, v) in [p[0] / p[3], p[1] / p[3], p[2] / p[3]].into_iter().enumerate() {
                (out.ndc[k], out.ndc[k + 3]) = (out.ndc[k].min(v), out.ndc[k + 3].max(v));
            }
            out.screen.push(((p[0] / p[3] * 0.5 + 0.5) * size.0 as f32, (0.5 - p[1] / p[3] * 0.5) * size.1 as f32));
        }
        for p in 0..np.min(lay.max_prims as usize) {
            out.culled += u64::from(le32(g, lay.prim_offset() as usize + 4 * p) >> 31 == 1);
        }
    }
    out.distinct = seen.len();
    Some(out)
}

/// `agree`, or what differs (as `bridge::Agreement::verdict`, with the
/// depth in depth units: more than 1e-4 from the frame's constant offset).
fn verdict(t: usize, o: usize, d: &Diff) -> &'static str {
    let union = t + o - d.both;
    let single = union - d.both;
    match (t, o) {
        (0, 0) => "empty",
        (0, _) => "theirs none",
        (_, 0) => "ours none",
        _ if single > 8 && single * 50 > union => "cover",
        _ if d.dz.is_some_and(|z| (z.0 - 5e-6).abs() > 1e-4) => "depth",
        _ => "agree",
    }
}

fn rows_arg(args: &[String]) -> Option<Vec<(usize, usize)>> {
    args.iter().position(|a| a == "--rows").and_then(|i| args.get(i + 1)).map(|v| {
        v.split(',').filter_map(|p| match p.split_once('-') {
            Some((a, b)) => Some((a.trim().parse().ok()?, b.trim().parse().ok()?)),
            None => p.trim().parse().ok().map(|k| (k, k)),
        }).collect()
    })
}

/// Our side of one row: unique vertices, the pass's triangles, draws,
/// instances and skinned draws, and the covers with and without the jitter.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ours {
    pub counts: [u64; 5],
    pub with: Cover,
    pub without: Cover,
}

fn put_ours(out: &mut impl Write, row: u32, o: &Ours) -> std::io::Result<()> {
    out.write_all(&row.to_le_bytes())?;
    for c in o.counts {
        out.write_all(&c.to_le_bytes())?;
    }
    for cover in [&o.with, &o.without] {
        out.write_all(&(cover.len() as u32).to_le_bytes())?;
        let mut b = Vec::with_capacity(8 * cover.len());
        for (p, z) in cover {
            b.extend_from_slice(&p.to_le_bytes());
            b.extend_from_slice(&z.to_bits().to_le_bytes());
        }
        out.write_all(&b)?;
    }
    Ok(())
}

/// The rows of an ours file; a cut last row (a process that ended) is dropped.
pub fn read_ours(b: &[u8]) -> HashMap<u32, Ours> {
    let (mut out, mut at) = (HashMap::new(), 0usize);
    'rows: while at + 44 <= b.len() {
        let row = le32(b, at);
        let mut o = Ours { counts: std::array::from_fn(|i| u64::from_le_bytes(b[at + 4 + 8 * i..at + 12 + 8 * i].try_into().unwrap())), ..Ours::default() };
        at += 44;
        for k in 0..2 {
            if at + 4 > b.len() {
                break 'rows;
            }
            let n = le32(b, at) as usize;
            let Some(pairs) = b.get(at + 4..at + 4 + 8 * n) else { break 'rows };
            let cover = pairs.chunks_exact(8).map(|c| (le32(c, 0), f32::from_bits(le32(c, 4)))).collect();
            if k == 0 { o.with = cover } else { o.without = cover }
            at += 4 + 8 * n;
        }
        out.insert(row, o);
    }
    out
}

struct Theirs {
    groups: u32,
    seconds: f64,
    cover: Cover,
    rec: Records,
    rec_bytes: usize,
    lacks: usize,
}

pub fn geodiff_cmd(args: &[String]) -> Result<(), String> {
    let dir = PathBuf::from(args.first().ok_or("usage: xnw-lab geodiff <frame dir> --adapter warp [--class a,b] [--rows a-b,c] [--tsv <file>] [--resume] [--limit N] [pass options: --skinned --skin-rules hex --preskin ...]")?);
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let (host, mut pass) = crate::pass_from_args(args)?;
    if !host.caps.software {
        return Err("geodiff runs the title's mesh programs: --adapter warp only".into());
    }
    let ours_side = opt("--side").as_deref() == Some("ours");
    let ours_path = PathBuf::from(opt("--ours").ok_or("geodiff: --ours <file> (written by --side ours, read by the run of the title's programs)")?);
    let size = (pass.width, pass.height);
    let fname = dir.file_name().and_then(|s| s.to_str()).unwrap_or("?").to_string();
    let tsv_path = opt("--tsv").map_or_else(|| PathBuf::from(format!("geodiff-{fname}.tsv")), PathBuf::from);
    let cur_path = tsv_path.with_extension("cur");

    let (mut list, mut geoms) = load_list(&dir)?;
    crate::audit::skin_args(args, &dir, &mut list, &mut geoms)?;
    crate::preskin::preskin_arg(args, &dir, &mut list, &mut geoms)?;
    crate::list_options(args, &mut list);
    let pack = Pack::open(&dir)?.ok_or("geodiff needs a v2 capture (pack\\manifest.tsv)")?;
    let draws = pack.list_draws().ok_or("no list-draws.tsv")?;
    let vps = list.main_viewports();
    let (wanted, classes) = (rows_arg(args), opt("--class"));
    let classes: Option<Vec<&str>> = classes.as_deref().map(|c| c.split(',').collect());
    let rows: Vec<(usize, u32, u32)> = (0..list.draws.len()).filter(|&k| vps.contains(&list.draws[k].viewport) && wanted.as_ref().is_none_or(|w| w.iter().any(|(a, b)| (*a..=*b).contains(&k))) && classes.as_ref().is_none_or(|c| c.contains(&class(&list, k)))).filter_map(|k| {
        let draw = *draws.get(k)?;
        Some((k, draw, pack.sink_of(draw)?))
    }).collect();

    if ours_side {
        pass.set_raw_shader(&depth_shader()?)?;
        (pass.sky, pass.water, pass.terrain.opts.on) = (false, false, false);
        pass.decal_hold = false;
        (list.imposters, list.terrain, list.bridged, list.hair, list.hair_depth, list.ui) = (None, None, None, None, None, None);
        let all = std::mem::take(&mut list.draws);
        let preskin = std::mem::take(&mut list.preskin);
        let mut decode = |key: &GeomKey| geoms.get(key).map(|(vb, idx)| HostGeometry { vertices: Vec::new(), indices: idx.clone(), raw_vb: Some(vb.clone()), vb_place: 0 });
        let mut file = std::io::BufWriter::new(std::fs::File::create(&ours_path).map_err(|e| format!("{}: {e}", ours_path.display()))?);
        let (t0, mut frame_no) = (std::time::Instant::now(), 0u64);
        for &(k, _, _) in &rows {
            let row = &all[k];
            let o_verts = geoms.get(&row.geom).map_or(0, |g| {
                let mut v = g.1.clone();
                v.sort_unstable();
                v.dedup();
                v.len()
            });
            // A decal as an opaque draw (no depth bias of ours); a depth prepass row too (our pass leaves that stage out).
            let stage = if DECAL_STAGES.contains(&row.stage) || PREPASS_STAGES.contains(&row.stage) { OPAQUE_STAGE } else { row.stage };
            list.draws = vec![DrawRecord { state: None, stage, ..row.clone() }];
            list.preskin = crate::preskin_gpu::subset(&preskin, &[k]);
            let mut ours = |jitter: bool| -> Result<(xnw_world::PassStats, Cover), String> {
                xnw_world::keep_jitter(jitter);
                frame_no += 2;
                list.frame = frame_no;
                crate::replay::watchdog_arm();
                let st = pass.draw(&list, &mut decode, true).and_then(|st| Ok((st, pass.read_rgba()?)));
                (list.frame, pass.id_mode) = (frame_no + 1, true);
                let ids = pass.draw(&list, &mut decode, true).and_then(|_| pass.read_rgba());
                pass.id_mode = false;
                crate::replay::watchdog_disarm();
                let ((st, bits), ids) = (st?, ids?);
                Ok((st, ids.chunks_exact(4).zip(bits.chunks_exact(4)).enumerate().filter(|(_, (id, _))| id[..3] == [1, 0, 0]).map(|(i, (_, z))| (i as u32, f32::from_bits(le32(z, 0)))).collect()))
            };
            let ((st, with), (_, without)) = (ours(true)?, ours(false)?);
            put_ours(&mut file, k as u32, &Ours { counts: [o_verts as u64, st.triangles, st.draws as u64, st.instances, st.skinned as u64], with, without }).map_err(|e| e.to_string())?;
        }
        file.flush().map_err(|e| e.to_string())?;
        println!("geodiff (ours) {} on {}: {} rows in {:.1} s -> {}", dir.display(), host.caps.adapter, rows.len(), t0.elapsed().as_secs_f64(), ours_path.display());
        return Ok(());
    }
    let ours_rows = read_ours(&std::fs::read(&ours_path).map_err(|e| format!("{}: {e}", ours_path.display()))?);

    // `--resume`: the rows the file holds stay; the row the last process was on removed the device.
    let mut lines: Vec<String> = Vec::new();
    let mut removes: Vec<String> = Vec::new();
    if args.iter().any(|a| a == "--resume") {
        lines = std::fs::read_to_string(&tsv_path).unwrap_or_default().lines().skip(1).map(str::to_owned).collect();
        if let Ok(cur) = std::fs::read_to_string(&cur_path) {
            let c: Vec<&str> = cur.trim().split('\t').collect();
            if c.len() == 8 && !lines.iter().any(|l| l.split('\t').nth(1) == Some(c[1])) {
                lines.push(format!("{}{}\tdevice removed\tthe process ended on this row", cur.trim(), "\t-".repeat(34)));
            }
        }
        removes = lines.iter().filter(|l| l.contains("\tdevice removed\t")).filter_map(|l| l.split('\t').nth(6).map(str::to_owned)).collect();
    }
    let done: Vec<usize> = lines.iter().filter_map(|l| l.split('\t').nth(1)?.parse().ok()).collect();
    let mut file = std::fs::File::create(&tsv_path).map_err(|e| format!("{}: {e}", tsv_path.display()))?;
    writeln!(file, "{HEADER}").and_then(|_| lines.iter().try_for_each(|l| writeln!(file, "{l}"))).map_err(|e| e.to_string())?;
    let limit: usize = opt("--limit").and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);

    drop(pass);
    let all = std::mem::take(&mut list.draws);
    let mut whole = list.clone();
    whole.draws = all.clone();
    let mut programs: HashMap<u128, Result<(bridge::Programs, HashMap<(u16, u32), u32>), String>> = HashMap::new();
    let mut texels: Provider = Box::new(|_| None);
    let masks = opt("--masks").map(PathBuf::from);
    let t0 = std::time::Instant::now();
    let mut n = 0usize;
    println!("geodiff {} on {}: {} main-view rows with a sink ({} done before)", dir.display(), host.caps.adapter, rows.len(), done.len());
    for &(k, draw, sink) in rows.iter().filter(|r| !done.contains(&r.0)) {
        if n >= limit {
            break;
        }
        n += 1;
        let row = &all[k];
        let pipeline = pack.data.sinks.iter().find(|s| s.index == sink).map(|s| s.pipeline);
        let stage = |name: &str| pack.data.programs.iter().find(|p| Some(p.pipeline) == pipeline && p.stage == name).map(|p| p.container);
        let (mesh, pixel) = (stage("mesh"), stage("pixel"));
        let head = format!("{fname}\t{k}\t{draw}\t{sink}\t{}\t{}\t{}\t{}", class(&whole, k), row.stage, mesh.map_or("-".into(), |h| format!("{h:032x}")), pixel.map_or("-".into(), |h| format!("{h:032x}")));
        std::fs::write(&cur_path, &head).map_err(|e| e.to_string())?;

        // Theirs.
        let mut one = || -> Result<Theirs, String> {
            let hash = mesh.ok_or("the pipeline has no mesh program")?;
            if removes.contains(&format!("{hash:032x}")) {
                return Err("not run: this program removed the device on an earlier row".into());
            }
            let (p, used) = programs.entry(hash).or_insert_with(|| mesh_program(&pack, hash)).as_ref().map_err(|e| format!("translate: {e}"))?;
            if let Some(gap) = bridge::gap(&host.caps, &p.needs()) {
                return Err(gap);
            }
            let mut loaded = load(&pack, sink, &Row { handles: Vec::new(), ..Row::with_palette(row, &whole, &pack, draw) }, geoms.get(&row.geom).map(|g| &g.0[..]), used, &mut texels);
            loaded.direct_meshlets();
            if loaded.groups == 0 {
                return Err("no group count".into());
            }
            let b = bridge::bind(&p.ms, &p.ps, &loaded.draw())?;
            let target = bridge::Target { width: size.0, height: size.1, depth: Some((true, 5)), limit: 120.0, depth_only: true, records: true };
            crate::replay::watchdog_arm();
            let frame = bridge::run(&host, p, &b, &target);
            crate::replay::watchdog_disarm();
            let frame = frame?;
            let rec = records(&frame.records, &p.layout, size).ok_or("no SV_Position row")?;
            Ok(Theirs { groups: loaded.groups, seconds: frame.seconds, cover: frame.depth.iter().enumerate().filter(|(_, z)| **z != frame.far).map(|(i, z)| (i as u32, *z)).collect(), rec, rec_bytes: frame.records.len() + loaded.groups as usize * p.layout.index_bytes() as usize, lacks: loaded.lacks.len() })
        };
        let theirs = one();
        // SAFETY: the device call takes no arguments.
        if unsafe { host.device.GetDeviceRemovedReason() }.is_err() {
            writeln!(file, "{head}{}\tdevice removed\t{}", "\t-".repeat(34), theirs.as_ref().err().cloned().unwrap_or_default().replace(['\t', '\n'], " ")).map_err(|e| e.to_string())?;
            let _ = std::fs::remove_file(&cur_path);
            return Err(format!("row {k}: the device was removed; run again with --resume"));
        }

        // Ours: the row alone, with the jitter and without (the other process).
        let none = Ours::default();
        let os = ours_rows.get(&(k as u32)).unwrap_or(&none);
        let (with, without) = (&os.with, &os.without);
        let o = os.counts.map(|c| c.to_string()).join("\t");
        let num = |v: Option<String>| v.unwrap_or_else(|| "-".into());
        let mut their_cover = None;
        let line = match theirs {
            Ok(t) => {
                their_cover = Some(t.cover.clone());
                let (d, nj) = (diff(&t.cover, with, &t.rec.screen, size.0), diff(&t.cover, without, &[], size.0));
                let dz = |d: &Diff| d.dz.map_or("-\t-\t-\t-".into(), |z| format!("{:.4e}\t{:.4e}\t{:.4e}\t{}", z.0, z.1, z.2, z.3));
                format!("{head}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{o}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}\t{}\t{}", row.inst_count, row.bone_count, t.groups, t.rec.live, t.rec.verts, t.rec.distinct, t.rec.prims, t.rec.culled, t.rec.behind, t.cover.len(), with.len(), d.both, num(d.c.map(|c| format!("{:.2}", c.0))), num(d.c.map(|c| format!("{:.3}", c.1))), d.v.map_or(0, |v| v.0), num(d.v.map(|v| format!("{:.2}", v.1))), num(d.v.map(|v| format!("{:.3}", v.2))), dz(&d), without.len(), nj.both, dz(&nj), t.seconds, t.rec_bytes, if !ours_rows.contains_key(&(k as u32)) { "ours not run" } else if os.counts[2] == 0 { "ours skips" } else { verdict(t.cover.len(), with.len(), &d) }, format!("{}{}", if t.lacks > 0 { format!("{} inputs stood in", t.lacks) } else { String::new() }, if t.cover.is_empty() && !t.rec.screen.is_empty() { format!("; vertices in front: NDC x {:.2}..{:.2} y {:.2}..{:.2} z {:.2e}..{:.2e}", t.rec.ndc[0], t.rec.ndc[3], t.rec.ndc[1], t.rec.ndc[4], t.rec.ndc[2], t.rec.ndc[5]) } else { String::new() }))
            }
            Err(e) => format!("{head}\t{}\t{}{}\t{o}\t-\t{}{}\t{}{}\tnot run\t{}", row.inst_count, row.bone_count, "\t-".repeat(7), with.len(), "\t-".repeat(10), without.len(), "\t-".repeat(7), e.replace(['\t', '\n'], " ")),
        };
        writeln!(file, "{line}").map_err(|e| e.to_string())?;
        // `--masks <dir>`: a picture of each row that does not agree: red theirs alone, green ours alone,
        // grey both (blue where the title's depth is behind ours).
        if let Some(dir) = masks.as_ref().filter(|_| !line.contains("	agree	") && !line.contains("	empty	")) {
            if let Some(t) = &their_cover {
                let mut px = vec![0u8; 4 * (size.0 * size.1) as usize];
                px.chunks_exact_mut(4).for_each(|p| p[3] = 255);
                let zs: HashMap<u32, f32> = t.iter().copied().collect();
                for (p, _) in t {
                    px[4 * *p as usize] = 255;
                }
                for (p, z) in with {
                    let at = 4 * *p as usize;
                    match zs.get(p) {
                        Some(tz) if tz < z => px[at..at + 3].copy_from_slice(&[60, 60, 255]),
                        Some(_) => px[at..at + 3].copy_from_slice(&[150, 150, 150]),
                        None => px[at + 1] = 255,
                    }
                }
                std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
                crate::replay::write_png(&dir.join(format!("{fname}-r{k}.png")), size.0, size.1, &px).map_err(|e| e.to_string())?;
            }
        }
        lines.push(line);
        if n % 100 == 0 {
            println!("  {n} rows, {:.0} s", t0.elapsed().as_secs_f64());
        }
    }
    let _ = std::fs::remove_file(&cur_path);
    println!("geodiff: {n} rows in {:.1} s -> {}", t0.elapsed().as_secs_f64(), tsv_path.display());
    summary(&lines);
    Ok(())
}

/// Rows per verdict by class and by class and mesh program, with the
/// largest cover distance and the worst depth of each.
fn summary(lines: &[String]) {
    let mut sums: [BTreeMap<String, (BTreeMap<String, u32>, f32, f32)>; 2] = Default::default();
    for l in lines {
        let c: Vec<&str> = l.split('\t').collect();
        if c.len() < 44 {
            continue;
        }
        let num = |i: usize| c[i].parse::<f32>().ok();
        for (sum, key) in sums.iter_mut().zip([c[4].to_string(), format!("{} {}", c[4], &c[6][..c[6].len().min(8)])]) {
            let e = sum.entry(key).or_default();
            *e.0.entry(c[42].to_string()).or_default() += 1;
            e.1 = e.1.max(num(25).unwrap_or(0.0));
            e.2 = e.2.max(num(31).map_or(0.0, |z| -z));
        }
    }
    for (name, sum) in ["class", "class and mesh program"].iter().zip(&sums) {
        for (key, (counts, c_max, behind)) in sum {
            println!("geodiff by {name}: {key}: {}; cover distance max {c_max:.1} px, deepest behind ours {behind:.2e}", counts.iter().map(|(v, n)| format!("{v} {n}")).collect::<Vec<_>>().join(", "));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_is_euclidean() {
        let (w, h) = (7, 5);
        let mut set = vec![false; w * h];
        set[2 * w + 1] = true;
        set[4 * w + 6] = true;
        let d = distance(&set, w, h);
        for y in 0..h {
            for x in 0..w {
                let want = [(1i32, 2i32), (6, 4)].iter().map(|(a, b)| ((x as i32 - a).pow(2) + (y as i32 - b).pow(2)) as f32).fold(f32::MAX, f32::min);
                assert_eq!(d[y * w + x], want, "at {x},{y}");
            }
        }
        assert!(distance(&vec![false; 6], 3, 2).iter().all(|v| *v == f32::MAX));
    }

    #[test]
    fn diff_measures_shift_and_depth() {
        // Theirs: a 4x4 block at (10,10); ours the same block 3 pixels right, 1e-3 nearer.
        let w = 64u32;
        let block = |x0: u32, z: f32| -> Cover { (10..14).flat_map(|y| (x0..x0 + 4).map(move |x| (y * w + x, z))).collect() };
        let (t, o) = (block(10, 0.5), block(13, 0.501));
        let d = diff(&t, &o, &[(10.5, 10.5), (13.5, 12.5), (40.0, 40.0)], w);
        assert_eq!(d.both, 4);
        assert_eq!(d.c.map(|c| c.0), Some(3.0));
        let dz = d.dz.unwrap();
        assert!((dz.0 + 1e-3).abs() < 1e-6 && dz.3 == 4);
        // Vertex (10.5,10.5) is 3 px from ours, (13.5,12.5) on it; (40,40) is not on theirs.
        assert_eq!(d.v, Some((2, 3.0, 1.5)));
        assert_eq!(verdict(16, 16, &d), "cover");
        assert_eq!(verdict(0, 16, &Diff::default()), "theirs none");
    }
}
