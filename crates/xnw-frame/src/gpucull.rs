//! The device instance cull in the replay (`--gpucull[=check]`, PERF3). A
//! live list's instanced rows are rebuilt as cull batches from the probe
//! dump of the same frame (draws.tsv `inst_*` columns, `blobs\inst-*.bin`,
//! `blobs\lod-*.bin`, the list's cameras), so the world pass draws them from
//! its compute's output instead of the list's CPU-made instances.
//!
//! What it checks: the CPU reference ([`xnw_world::gpucull::cull_cpu`]) of
//! each row's batch against the instances the live list drew for that row
//! (instances.bin), and with `=check` the device output of every batch
//! against the reference, bit for bit. The hide bitset is not dumped: a
//! row the live hook hid instances of shows up as a reference mismatch.
//! A row apart only by records at a frustum plane within the text camera's
//! rounding is counted "at a plane" ([`at_plane`]); captures after
//! write the cameras exactly.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use xnw_world::FrameList;
use xnw_world::gpucull::{Batch, cull_cpu, frustum_planes};

/// What [`fill`] did.
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// Instanced rows (inst_count > 0), and of them made device-culled.
    pub rows: usize,
    pub culled: usize,
    /// Rows left on the CPU's instances, by reason.
    pub left: HashMap<&'static str, usize>,
    /// Rows whose reference instances equal the live list's, and differ (count, worlds).
    pub same: usize,
    pub differ: Vec<(u32, usize, usize)>,
    /// Rows whose counts differ by no more than their records at a frustum
    /// plane within the text camera's rounding ([`at_plane`]): (row, live, reference, such records).
    pub edge: Vec<(u32, usize, usize, usize)>,
    /// Rows with equal counts and worlds within 5 cm (the text camera's origin).
    pub near: usize,
    /// Largest |reference - live| over the rows' world floats (m).
    pub max_diff: f32,
    /// Records tested, instances kept by the reference.
    pub tested: u64,
    pub kept: u64,
    /// Record bytes of the batches: kept-on-device blocks and per-frame blocks (each block once).
    pub kept_bytes: u64,
    pub frame_bytes: u64,
    pub cpu_ms: f64,
    /// The CPU's per-instance test and expansion of all batches (the cost the device takes), best of 3.
    pub ref_ms: f64,
}

/// Blocks by (length, content hash), shared by the frames of a replay.
fn shared(b: Vec<u8>) -> Arc<Vec<u8>> {
    type Blocks = HashMap<(usize, u64), Arc<Vec<u8>>>;
    static BLOCKS: std::sync::Mutex<Option<Blocks>> = std::sync::Mutex::new(None);
    let key = (b.len(), crate::decode::fnv(&b));
    let Ok(mut g) = BLOCKS.lock() else { return Arc::new(b) };
    g.get_or_insert_with(HashMap::new).entry(key).or_insert_with(|| Arc::new(b)).clone()
}

fn f32s<const N: usize>(b: &[u8]) -> [f32; N] {
    std::array::from_fn(|i| f32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap()))
}

/// `--near-hash h[,h..]`: geometry hashes (hex) for [`fill`]'s `near_hashes`.
pub fn hashes_arg(args: &[String]) -> Vec<u64> {
    args.iter().position(|a| a == "--near-hash").and_then(|i| args.get(i + 1)).map(|v| v.split(',').filter_map(|h| u64::from_str_radix(h.trim().trim_start_matches("0x"), 16).ok()).collect()).unwrap_or_default()
}

/// A capture's bitset (blobs/mask-N.bin, u32 words LSB first) named by a
/// draws.tsv cell; empty for "-", a missing column or a missing file.
fn mask_blob(blobs: &Path, cache: &mut HashMap<u32, Vec<u32>>, cell: Option<&str>) -> Vec<u32> {
    let Some(i) = cell.and_then(|s| s.parse::<u32>().ok()) else { return Vec::new() };
    cache.entry(i).or_insert_with(|| std::fs::read(blobs.join(format!("mask-{i}.bin"))).map(|b| b.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect()).unwrap_or_default()).clone()
}

/// Rebuilds `list`'s instanced rows as device-culled batches (see the module).
/// `device` false (`--gpucull=cpu`): the rows draw the reference's instances
/// from the list's CPU buffer instead (the same batches and origins), so a
/// picture against `--gpucull` isolates the device path.
/// The batch's hide and near fade-in bitsets come from the capture
/// (draws.tsv inst_hide / inst_near naming blobs/mask-N.bin, VEH3c), as the
/// live cull takes them; captures made before VEH3c have neither.
/// `near_rows` (`--near-rows k[,k..]`, list rows) and `near_hashes`
/// (`--near-hash h[,h..]`, geometry hashes: stable across a sequence's
/// frames): every instance of these batches counts as flagged in the near
/// fade-in bitset (VEH3), for those older captures. The fp16 the rule reads
/// is set on most batch props (h = 60, 150, 300 in pk8a f1030's stage 21),
/// so "all flagged" for every batch is wrong; this is a lab instrument for
/// a batch whose flagged instance is known, not a rule.
pub fn fill(dir: &Path, list: &mut FrameList, device: bool, near_rows: &[u32], near_hashes: &[u64]) -> Result<Report, String> {
    let mut rep = Report::default();
    let rows_v1: HashMap<u32, usize> = std::fs::read_to_string(dir.join("list-draws.tsv")).map_err(|e| format!("list-draws.tsv: {e}"))?.lines().skip(1).filter_map(|l| l.split_once('\t')).filter_map(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?))).collect();
    let tsv = std::fs::read_to_string(dir.join("draws.tsv")).map_err(|e| format!("draws.tsv: {e}"))?;
    let mut lines = tsv.lines();
    let head: Vec<&str> = lines.next().ok_or("draws.tsv empty")?.split('\t').collect();
    let col = |n: &str| head.iter().position(|h| *h == n).ok_or(format!("draws.tsv: no {n} column"));
    let (cvp, cb, cl, cf, cr, cg) = (col("viewport")?, col("inst_block")?, col("inst_lod")?, col("inst_first_lod")?, col("inst_lodrec")?, col("inst_gpu")?);
    let (chide, cnear) = (head.iter().position(|h| *h == "inst_hide"), head.iter().position(|h| *h == "inst_near"));
    let draws: Vec<Vec<&str>> = lines.map(|l| l.split('\t').collect()).collect();
    let hex = |s: &str| u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0);
    let blobs = dir.join("blobs");
    let mut blocks: HashMap<u32, Arc<Vec<u8>>> = HashMap::new();
    let mut lods: HashMap<u32, [f32; 35]> = HashMap::new();
    let mut masks: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut batch_of: HashMap<(u32, u64, u32, u32), u32> = HashMap::new();
    let mut seen_blocks: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let t0 = std::time::Instant::now();
    for row in 0..list.draws.len() {
        let r = list.draws[row].clone();
        if r.inst_count == 0 {
            continue;
        }
        rep.rows += 1;
        let mut left = |why: &'static str| *rep.left.entry(why).or_default() += 1;
        let Some(d) = rows_v1.get(&(row as u32)).and_then(|v| draws.get(*v)) else {
            left("no v1 draw");
            continue;
        };
        let get = |c: usize| d.get(c).copied().unwrap_or("-");
        let Ok(block) = get(cb).parse::<u32>() else {
            // Wheels and other rows whose instances the hook made itself.
            left("no instance block");
            continue;
        };
        let vp = hex(get(cvp));
        let (lod, first_lod, lodrec) = (get(cl).parse::<u32>().unwrap_or(0), get(cf).parse::<u32>().unwrap_or(255), get(cr).parse::<u32>().unwrap_or(u32::MAX));
        let Some(cam) = list.cameras.iter().find(|c| c.viewport == vp).copied() else {
            left("no camera");
            continue;
        };
        let records = match blocks.get(&block) {
            Some(b) => b.clone(),
            None => {
                let Ok(b) = std::fs::read(blobs.join(format!("inst-{block}.bin"))) else {
                    left("no block file");
                    continue;
                };
                // One allocation per content across the frames of a sequence, as the
                // live hook keeps a static block (so the kept store is reused).
                let b = shared(b);
                blocks.insert(block, b.clone());
                b
            }
        };
        if !lods.contains_key(&lodrec)
            && let Ok(b) = std::fs::read(blobs.join(format!("lod-{lodrec}.bin")))
            && b.len() >= 140
        {
            lods.insert(lodrec, f32s::<35>(&b));
        }
        let slot = (first_lod != 255 && lod >= first_lod).then(|| lod - first_lod);
        let t = lods.get(&lodrec).copied();
        let b = *match batch_of.entry((block, vp, lod, lodrec)) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let origin = xnw_world::camera_position(&cam.view);
                let planes = frustum_planes(&xnw_world::Camera { viewport: 0, view: cam.view, proj: cam.proj }.view_proj_relative());
                let keep = hex(get(cg)) != 0;
                let lod = match (t, slot) {
                    (Some(t), Some(s)) => Some((t, s)),
                    _ => None,
                };
                let n = (records.len() / 64) as u32;
                // The live batch test already passed (the row is listed): every record is tested against the planes.
                let hide = mask_blob(&blobs, &mut masks, chide.and_then(|c| d.get(c).copied()));
                let mut near = mask_blob(&blobs, &mut masks, cnear.and_then(|c| d.get(c).copied()));
                if near_rows.contains(&(row as u32)) || near_hashes.contains(&r.geom.hash) {
                    near = vec![u32::MAX; n.div_ceil(32) as usize];
                }
                let Some(i) = list.cull.push(Batch { records: records.clone(), len: n, keep, lod, hide, near, origin, planes: Some(planes), out_first: 0 }) else {
                    left("cull frame full");
                    continue;
                };
                if seen_blocks.insert(block) {
                    if keep {
                        rep.kept_bytes += records.len() as u64;
                    } else {
                        rep.frame_bytes += records.len() as u64;
                    }
                }
                e.insert(i)
            }
        };
        let bt = &list.cull.batches[b as usize];
        let want = cull_cpu(bt);
        let live = list.instances.get(12 * r.inst_first as usize..12 * (r.inst_first + r.inst_count) as usize).unwrap_or(&[]);
        // camera.txt holds the view as text: the origin differs from the live one in the last bits.
        let diff = want.iter().zip(live).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        rep.max_diff = rep.max_diff.max(diff);
        let edge = at_plane(bt);
        if want.len() == live.len() && diff < 1e-3 {
            rep.same += 1;
        } else if want.len() == live.len() && diff < 0.05 {
            rep.near += 1;
        } else if want.len() != live.len() && want.len().abs_diff(live.len()) / 12 <= edge {
            // Only records a camera.txt's rounding can move across a plane.
            rep.edge.push((row as u32, live.len() / 12, want.len() / 12, edge));
        } else {
            rep.differ.push((row as u32, live.len() / 12, want.len() / 12));
        }
        let (first, len) = if device {
            list.cull.rows.push((row as u32, b));
            (bt.out_first, bt.len)
        } else {
            let first = (list.instances.len() / 12) as u32;
            list.instances.extend_from_slice(&want);
            (first, (want.len() / 12) as u32)
        };
        let r = &mut list.draws[row];
        r.inst_first = first;
        r.inst_count = len;
        rep.culled += 1;
    }
    rep.cpu_ms = t0.elapsed().as_secs_f64() * 1e3;
    // The work the device takes over: each batch's per-instance test and worlds, once (best of 3).
    rep.ref_ms = f64::MAX;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let kept: usize = list.cull.batches.iter().map(|b| cull_cpu(b).len() / 12).sum();
        rep.ref_ms = rep.ref_ms.min(t.elapsed().as_secs_f64() * 1e3);
        rep.kept = kept as u64;
    }
    rep.tested = list.cull.batches.iter().map(|b| u64::from(b.len)).sum();
    Ok(rep)
}

/// Records of a batch whose frustum verdict the camera's text rounding
/// can flip: they pass the hide bit, the sanity test and the LOD slot, and
/// their nearest plane lies within `1.5e-5 (d + |origin|) + 1 mm` of the
/// sphere (camera.txt before kept 5 decimals: each view element off
/// by up to 5e-6, three to a dot product: the planes by 1.5e-5 of the distance, the origin by
/// 1.5e-5 of its size). pk5 f977 row 431: 3 records 1.5-2.3 mm inside the
/// top plane at 157 m, outside in the live list [V].
pub fn at_plane(b: &Batch) -> usize {
    let (Some((t, slot)), Some(planes)) = (&b.lod, &b.planes) else { return 0 };
    let o = b.origin;
    let size = (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt();
    let mut n = 0;
    for (i, rb) in b.records.chunks_exact(64).take(b.len as usize).enumerate() {
        let r = f32s::<16>(rb);
        if b.hide.get(i >> 5).is_some_and(|w| w & (1 << (i & 31)) != 0) || !xnw_world::gpucull::plausible_record(&r) {
            continue;
        }
        let v = [r[3] - o[0], r[7] - o[1], r[11] - o[2]];
        let d = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if xnw_world::gpucull::lod_slot(t, &r, d) != Some(*slot as usize) {
            continue;
        }
        let s = r[12].min(r[14]);
        let (c, rad) = ([t[2], t[3], t[4]], t[5] / s);
        let p: [f32; 3] = std::array::from_fn(|k| r[4 * k] * c[0] + r[4 * k + 1] * c[1] + r[4 * k + 2] * c[2] + r[4 * k + 3] - o[k]);
        let margin = planes.iter().map(|pl| pl[0] * p[0] + pl[1] * p[1] + pl[2] * p[2] + pl[3] + rad).fold(f32::MAX, f32::min);
        n += usize::from(margin.abs() <= 1.5e-5 * (d + size) + 1e-3);
    }
    n
}

/// The report as lines.
pub fn lines(rep: &Report, f: &xnw_world::gpucull::CullFrame) -> Vec<String> {
    let mut left: Vec<String> = rep.left.iter().map(|(k, v)| format!("{k} {v}")).collect();
    left.sort();
    vec![
        format!("gpu cull: {} instanced rows, {} device-culled ({} batches; left: {}); {} records tested, {} kept by the reference", rep.rows, rep.culled, f.batches.len(), if left.is_empty() { "none".into() } else { left.join(", ") }, rep.tested, rep.kept),
        format!(
            "gpu cull: reference = live list (counts equal, worlds within 1 mm) on {} rows, within 5 cm on {}, at a plane on {} (counts apart by no more than their records within the text camera's rounding of a plane){}, differs on {} (max difference {:.2e}){}",
            rep.same,
            rep.near,
            rep.edge.len(),
            rep.edge.iter().take(6).map(|(r, a, b, n)| format!(" [row {r}: live {a}, reference {b}, {n} at a plane]")).collect::<String>(),
            rep.differ.len(),
            rep.max_diff,
            rep.differ.iter().take(6).map(|(r, a, b)| format!(" [row {r}: live {a}, reference {b}]")).collect::<String>()
        ),
        format!("gpu cull: record bytes kept on the device {} KiB, per frame {} KiB; the CPU path's instances {} KiB; CPU per-instance test and worlds {:.2} ms", rep.kept_bytes >> 10, rep.frame_bytes >> 10, rep.kept * 48 >> 10, rep.ref_ms),
    ]
}

/// `=check`: every batch on the device alone against the reference, bit for bit.
pub fn check(pass: &xnw_world::Pass, f: &xnw_world::gpucull::CullFrame) -> Result<String, String> {
    let counts = vec![3u32; f.rows.len()];
    let (outs, _, ms, st) = xnw_world::gpucull::run(pass.device(), f, &counts)?;
    let (mut same, mut differ, mut first) = (0usize, 0usize, None);
    let mut has_row = vec![false; f.batches.len()];
    for &(_, b) in &f.rows {
        has_row[b as usize] = true;
    }
    for (i, b) in f.batches.iter().enumerate().filter(|(i, _)| has_row[*i]) {
        let want = cull_cpu(b);
        if outs[i].len() == want.len() && outs[i].iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()) {
            same += 1;
        } else {
            differ += 1;
            first.get_or_insert((i, outs[i].len() / 12, want.len() / 12));
        }
    }
    Ok(format!("gpu cull check: device = reference on {same} batches, differs on {differ}{}; GPU {ms:.3} ms (second run: {} KiB uploaded, {} KiB kept)", first.map_or(String::new(), |(i, a, b)| format!(" (first: batch {i}, device {a}, reference {b})")), st.uploaded >> 10, st.kept >> 10))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(x: f32, y: f32, z: f32) -> [f32; 16] {
        [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, z, 1.0, 0.0, 1.0, 0.0]
    }

    /// Only drawable records within the rounding band of a plane count: one
    /// just outside the plane y >= 0 (radius 1), not one far out, not one
    /// well inside, not a hidden one at the plane.
    #[test]
    fn at_plane_counts_the_band() {
        let mut t = [0f32; 35];
        (t[5], t[18], t[19], t[20], t[21], t[22], t[23]) = (1.0, 20.0, 40.0, -1.0, -1.0, 150.0, 160.0);
        let recs = [rec(0.0, -1.0005, 0.0), rec(0.0, -1.5, 0.0), rec(0.0, 5.0, 0.0), rec(0.0, -1.0, 0.0)];
        let far = [0.0, 0.0, 0.0, 1e9];
        let b = Batch { records: Arc::new(recs.iter().flatten().flat_map(|v| v.to_le_bytes()).collect()), len: 4, lod: Some((t, 0)), hide: vec![0b1000], planes: Some([[0.0, 1.0, 0.0, 0.0], far, far, far, far]), ..Default::default() };
        assert_eq!(at_plane(&b), 1);
        assert_eq!(at_plane(&Batch { planes: None, ..b }), 0, "no planes: nothing to flip");
    }

    /// A draws.tsv cell names blobs/mask-N.bin (u32 words); "-", no column
    /// or no file give an empty bitset (VEH3c); `--near-hash` reads hex.
    #[test]
    fn mask_blobs_and_hashes() {
        let dir = std::env::temp_dir().join(format!("xnw-maskblob-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("mask-3.bin"), [0x20u8, 0, 0, 0, 1, 0, 0, 0x80]).unwrap();
        let mut cache = HashMap::new();
        assert_eq!(mask_blob(&dir, &mut cache, Some("3")), vec![0x20, 0x8000_0001]);
        assert_eq!(mask_blob(&dir, &mut cache, Some("-")), Vec::<u32>::new());
        assert_eq!(mask_blob(&dir, &mut cache, None), Vec::<u32>::new());
        assert_eq!(mask_blob(&dir, &mut cache, Some("7")), Vec::<u32>::new());
        let _ = std::fs::remove_dir_all(&dir);
        let args: Vec<String> = ["burst", "--near-hash", "0x815ea3bf98ab200,ff"].iter().map(|s| s.to_string()).collect();
        assert_eq!(hashes_arg(&args), vec![0x815ea3bf98ab200, 0xff]);
    }
}
