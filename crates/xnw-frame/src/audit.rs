//! Geometry audit of live-list sequence dumps (`xnw world-render <dir>
//! --seq --audit`). Every main-view draw the pass would draw is transformed
//! on the CPU exactly as RAW_SHADER's vertex shader does (decode, skin with
//! the frame's palette buffer and the ped remap, instance, world), then
//! measured: (a) exploded edges, (b) far / behind the camera, (c) NaN/inf,
//! (d) collapsed, (e) frame-to-frame jumps, (f) present in N and N+2 but not
//! N+1. Results per class (kind, stage, remap, bone count) to `audit.tsv`,
//! the worst draws to `audit-worst.tsv` and a contact sheet of them.

use std::collections::HashMap;
use std::fmt::Write as _;

use xnw_world::{DrawRecord, FrameList, GeomKey, Mat, NO_PALETTE, NO_REMAP, OTHER_VIEW_STAGES, PREPASS_STAGES};

use super::Streams;

/// Flags (a)..(f).
pub const FLAGS: [&str; 7] = ["exploded", "far/behind", "nan", "collapsed", "jump", "flicker", "double"];
/// (a): an edge longer than this many local bounding-box diagonals.
const EXPLODE: f32 = 4.0;
/// (b): camera-relative distance (m) counted as far, and the share of far
/// vertices that flags a draw.
const FAR_M: f32 = 20_000.0;
const FAR_SHARE: f32 = 0.1;
/// Vertices measured per draw at most (spread over the index list) and
/// instances per instanced draw.
const MAX_VERTS: usize = 4096;
const MAX_INSTANCES: usize = 8;

/// One draw's measurement.
#[derive(Clone, Debug, Default)]
pub struct Measure {
    pub(crate) frame: u64,
    pub(crate) draw: usize,
    pub(crate) class: String,
    pub(crate) key: (GeomKey, u32, u32),
    /// Absolute (camera-added) world bbox centre and its diagonal.
    pub(crate) centre: [f32; 3],
    pub(crate) diag: f32,
    pub(crate) local_diag: f32,
    /// Longest edge / local diagonal.
    pub(crate) stretch: f32,
    pub(crate) flags: [bool; 7],
    /// How bad (for the worst list): a per-flag score.
    pub(crate) score: f32,
    /// The absolute world matrix (translation + camera) rounded, for double
    /// drawing and the placement table; the stage.
    pub(crate) abs_world: [f32; 12],
    pub(crate) stage: u32,
    pub(crate) triangles: u32,
    /// The title's state indices (blend, depth-stencil, raster) and, for a
    /// repeat, the stage of the first draw of the same geometry and placement.
    pub(crate) state: [u16; 3],
    pub(crate) repeat_of: Option<u32>,
    /// Share of the screen the measured vertices' projected box
    /// covers (0..1; 1 when they straddle the camera plane), and the effect.
    pub(crate) cover: f32,
    pub(crate) effect: u64,
}

/// Draws whose projected bounds exceed this share of the screen.
pub const BIG_COVER: f32 = 0.25;

/// The share of the screen covered by the projected box of
/// camera-relative points `ps` under `vp` (row vectors): 1 when some lie in
/// front and some behind the camera plane, 0 when all are behind.
pub fn screen_cover(ps: &[[f32; 3]], vp: &Mat) -> f32 {
    let (mut lo, mut hi, mut front, mut back) = ([f32::MAX; 2], [f32::MIN; 2], 0, 0);
    for p in ps {
        let c: [f32; 4] = std::array::from_fn(|j| p[0] * vp[j] + p[1] * vp[4 + j] + p[2] * vp[8 + j] + vp[12 + j]);
        if !(c[3] > 1e-4) {
            back += 1;
            continue;
        }
        front += 1;
        for k in 0..2 {
            let v = (c[k] / c[3]).clamp(-1.0, 1.0);
            lo[k] = lo[k].min(v);
            hi[k] = hi[k].max(v);
        }
    }
    match (front, back) {
        (0, _) => 0.0,
        (_, 0) => (hi[0] - lo[0]) * (hi[1] - lo[1]) / 4.0,
        _ => 1.0,
    }
}

/// The class of a draw: kind (skinned-ped, skinned, instanced, rigid),
/// stage, remap, bone bucket.
fn class_of(r: &DrawRecord) -> String {
    let kind = if r.palette != NO_PALETTE {
        if r.ped { "skinned-ped" } else { "skinned" }
    } else if r.inst_count > 0 {
        "instanced"
    } else {
        "rigid"
    };
    let bones = match r.bone_count {
        0 => "n0",
        1..=64 => "n<=64",
        65..=128 => "n<=128",
        _ => "n>128",
    };
    let remap = if r.remap != NO_REMAP { "remap" } else { "-" };
    if kind.starts_with("skinned") { format!("{kind} st{} {remap} {bones}", r.stage) } else { format!("{kind} st{}", r.stage) }
}

/// The frame's palette buffer as the pass builds it (each palette cut to
/// whole 3x4 rows; `first` in float4 units).
fn palette_buffer(list: &FrameList) -> (Vec<f32>, Vec<u32>) {
    let mut floats = Vec::new();
    let mut first = Vec::new();
    for p in &list.palettes {
        first.push((floats.len() / 4) as u32);
        floats.extend_from_slice(&p[..p.len() / 12 * 12]);
    }
    floats.resize(floats.len().max(4), 0.0);
    (floats, first)
}

fn row_mul(p: [f32; 3], w: f32, m: &Mat) -> [f32; 3] {
    std::array::from_fn(|j| p[0] * m[j] + p[1] * m[4 + j] + p[2] * m[8 + j] + w * m[12 + j])
}

/// The vertex shader's skin of vertex `v` (the shader's rules: ped weights
/// normalised, others raw w/255 up to the first zero weight after the
/// first; remap by `remap_at` dwords; palette rows bounded by the buffer).
fn skin(s: &Streams, vb: &[u8], v: u32, p: [f32; 3], pal: &[f32], pal_first: u32, remap_at: Option<u32>, ped: bool) -> [f32; 3] {
    let (terms, n) = skin_terms(s, vb, v, pal, remap_at, ped, 0);
    skin_apply(&terms[..n], p, pal, pal_first, ped)
}

/// The influences the vertex shader uses for vertex `v`: (bone after the
/// remap, weight w/255), in its order and with its stop rules.
fn skin_terms(s: &Streams, vb: &[u8], v: u32, pal: &[f32], remap_at: Option<u32>, ped: bool, rules: u32) -> ([(u32, f32); 8], usize) {
    let pu = |i: u32| -> u32 { pal.get(i as usize).map_or(0, |f| f.to_bits()) };
    let (mut out, mut n, mut stop) = ([(0u32, 0f32); 8], 0usize, false);
    let le = |at: usize| vb.get(at..at + 4).map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()));
    for k in 0..2 {
        let (ws, is) = (s.weights[k], s.indices[k]);
        if ws.stride == 0 || is.stride == 0 {
            continue;
        }
        let wb = le((ws.stride * v + ws.offset) as usize);
        let ia = (is.stride * v + is.offset) as usize;
        for j in 0..4u32 {
            let wt = ((wb >> (8 * j)) & 0xFF) as f32 / 255.0;
            // Rules bit 0: UINT8x4 is one byte per index (the title's
            // tbuffer_load_format BUF_FMT_8_8_8_8_UINT, ped_posonly 0x270).
            let u8x4 = if rules & SKIN_U8 != 0 { 0xFF } else { 0xFFF };
            let mut b = if is.format == 5 { le(ia) >> (8 * j) & u8x4 } else { le(ia + 4 * j as usize) } & 0xFFF;
            if let Some(r) = remap_at {
                b = pu(r + b) & 0xFFF;
            }
            if !ped && (k as u32 + j) > 0 && (stop || wt == 0.0) {
                stop = true;
                continue;
            }
            if ped && wt == 0.0 {
                continue;
            }
            out[n] = (b, wt);
            n += 1;
        }
    }
    (out, n)
}

/// Bone `b`'s three palette rows (zeros past the buffer, as the shader's `pl`).
fn bone_rows(pal: &[f32], pal_first: u32, b: u32) -> [[f32; 4]; 3] {
    let pl = |i: u32| -> [f32; 4] {
        let a = i as usize * 4;
        if a + 4 <= pal.len() { [pal[a], pal[a + 1], pal[a + 2], pal[a + 3]] } else { [0.0; 4] }
    };
    [pl(pal_first + 3 * b), pl(pal_first + 3 * b + 1), pl(pal_first + 3 * b + 2)]
}

fn skin_apply(terms: &[(u32, f32)], p: [f32; 3], pal: &[f32], pal_first: u32, ped: bool) -> [f32; 3] {
    let (mut sp, mut total) = ([0f32; 3], 0f32);
    for &(b, wt) in terms {
        for (c, r) in bone_rows(pal, pal_first, b).iter().enumerate() {
            sp[c] += wt * (r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + r[3]);
        }
        total += wt;
    }
    if ped {
        if total > 0.0 { sp.map(|x| x / total) } else { p }
    } else {
        sp
    }
}

/// Skin rules (the pass's `skin_rules`, flags bits 8..): bit 0, a
/// UINT8x4 index is its byte (else the default 12 bits of the shifted dword).
pub const SKIN_U8: u32 = 1;

/// The per-draw skin sanity measure.
#[derive(Clone, Debug, Default)]
pub struct SkinCheck {
    pub draw: usize,
    pub verts: usize,
    pub components: usize,
    /// Largest spectral norm of the reached bones' 3x3.
    pub s: f32,
    /// max over components of the skinned radius about the component's first
    /// vertex / (s e_C) (an earlier bound was diag(skinned bbox) / (2 s e_C)); pass needs <= 1.
    pub q: f32,
    /// Reached bones that are non-finite or singular (det 0).
    pub bad_bones: usize,
    /// Distinct bones reached; how many lie past the draw's bone_count.
    pub bones: usize,
    pub past_count: usize,
    /// Largest raw (pre-remap) index & 0xFFF over the used influences.
    pub max_raw: u32,
    /// Reached bones whose row 0 has unit length and row 2.w is 0 (a DQ entry's look).
    pub dq_like: usize,
    /// 99th percentile of |a'-b'| / (s |a-b|) over the edges (not gated).
    pub edge_p99: f32,
    /// Largest |A A^T - I| entry over the reached bones (0: rigid; not gated)
    /// and the reached bones that are exactly the identity.
    pub ortho: f32,
    pub identity: usize,
    /// Components that left every component they touched in the bind
    /// pose (`torn_components`); pass needs 0.
    pub torn: usize,
    pub pass: bool,
}

/// Spectral norm of a 3x3 (rows r[i][0..3]) by power iteration on A^T A.
fn spectral(r: &[[f32; 4]; 3]) -> f32 {
    let a = |i: usize, j: usize| f64::from(r[i][j]);
    let ata: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a(k, i) * a(k, j)).sum()));
    let mut x = [1.0f64, 0.7, 0.3];
    let mut l = 0.0;
    for _ in 0..64 {
        let y: [f64; 3] = std::array::from_fn(|i| (0..3).map(|j| ata[i][j] * x[j]).sum());
        let n = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
        if n == 0.0 || !n.is_finite() {
            return n as f32;
        }
        l = n;
        x = y.map(|v| v / n);
    }
    l.sqrt() as f32
}

fn det3(r: &[[f32; 4]; 3]) -> f32 {
    r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
}

/// The palette buffer exactly as the pass builds it (each palette whole,
/// bones then any remap entries, padded to a float4).
fn pass_palette(list: &FrameList) -> (Vec<f32>, Vec<u32>) {
    let mut floats: Vec<f32> = Vec::new();
    let mut first = Vec::new();
    for p in &list.palettes {
        first.push((floats.len() / 4) as u32);
        floats.extend_from_slice(p);
        floats.resize(floats.len().div_ceil(4) * 4, 0.0);
    }
    floats.resize(floats.len().max(4), 0.0);
    (floats, first)
}

/// The skin sanity of one skinned draw over its whole geometry.
pub fn skin_check_draw(r: &DrawRecord, vb: &[u8], idx: &[u32], pal: &[f32], first: &[u32], rules: u32) -> Option<SkinCheck> {
    let s = Streams::parse(vb)?;
    if r.palette == NO_PALETTE || s.weights[0].stride == 0 || s.indices[0].stride == 0 {
        return None;
    }
    let pf = *first.get(r.palette as usize)?;
    let pal_first = if r.bone_count == 0 { pf } else { pf + 3 * r.bone_first };
    let remap_at = (r.remap != NO_REMAP).then(|| 4 * pf + r.remap);
    // Vertices of the triangles, their bind and skinned positions.
    let mut verts: Vec<u32> = idx.to_vec();
    verts.sort_unstable();
    verts.dedup();
    let at: HashMap<u32, usize> = verts.iter().enumerate().map(|(i, v)| (*v, i)).collect();
    let mut bind = Vec::with_capacity(verts.len());
    let mut posed = Vec::with_capacity(verts.len());
    let mut reached: std::collections::BTreeSet<u32> = Default::default();
    let mut max_raw = 0u32;
    let raw_s = |v: u32| skin_terms(&s, vb, v, pal, None, r.ped, rules);
    for &v in &verts {
        let p = s.position(vb, v).unwrap_or([0.0; 3]);
        let (t, n) = skin_terms(&s, vb, v, pal, remap_at, r.ped, rules);
        reached.extend(t[..n].iter().map(|x| x.0));
        let (rt, rn) = raw_s(v);
        max_raw = rt[..rn].iter().map(|x| x.0).fold(max_raw, u32::max);
        bind.push(p);
        posed.push(skin_apply(&t[..n], p, pal, pal_first, r.ped));
    }
    let mut c = SkinCheck { draw: 0, verts: verts.len(), bones: reached.len(), max_raw, ..Default::default() };
    for &b in &reached {
        let m = bone_rows(pal, pal_first, b);
        let fin = m.iter().flatten().all(|x| x.is_finite());
        let d = det3(&m);
        if !fin || d == 0.0 || !d.is_finite() {
            c.bad_bones += 1;
            continue;
        }
        c.s = c.s.max(spectral(&m));
        // Palette plausibility apart from the maths: |A A^T - I| and all-identity bones.
        let o = (0..3).flat_map(|i| (0..3).map(move |j| (i, j))).map(|(i, j)| ((0..3).map(|k| m[i][k] * m[j][k]).sum::<f32>() - if i == j { 1.0 } else { 0.0 }).abs()).fold(0f32, f32::max);
        c.ortho = c.ortho.max(o);
        c.identity += usize::from((0..3).all(|i| (0..4).all(|j| m[i][j] == if i == j { 1.0 } else { 0.0 })));
        let l0 = (m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[0][2] * m[0][2] + m[0][3] * m[0][3]).sqrt();
        c.dq_like += usize::from((l0 - 1.0).abs() < 1e-3 && m[2][3] == 0.0);
        c.past_count += usize::from(r.bone_count != 0 && b >= r.bone_count);
    }
    // Bind-pose edges; components by union-find; eccentricity by Dijkstra.
    let n = verts.len();
    let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    let dist = |a: [f32; 3], b: [f32; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let mut ratios = Vec::new();
    for t in idx.chunks_exact(3) {
        for e in 0..3 {
            let (a, b) = (at[&t[e]], at[&t[(e + 1) % 3]]);
            if a == b {
                continue;
            }
            let l = dist(bind[a], bind[b]);
            adj[a].push((b, l));
            adj[b].push((a, l));
            if l > 0.0 && c.s > 0.0 {
                ratios.push(dist(posed[a], posed[b]) / (c.s * l));
            }
        }
    }
    // Welded: vertices split only by seams (the same bind position bits) are one surface.
    let mut same: HashMap<[u32; 3], usize> = HashMap::new();
    for (i, p) in bind.iter().enumerate() {
        if let Some(&j) = same.get(&p.map(f32::to_bits)) {
            adj[i].push((j, 0.0));
            adj[j].push((i, 0.0));
        } else {
            same.insert(p.map(f32::to_bits), i);
        }
    }
    ratios.sort_by(f32::total_cmp);
    c.edge_p99 = ratios.get(ratios.len() * 99 / 100).copied().unwrap_or(0.0);
    let mut comp = vec![usize::MAX; n];
    let mut d = vec![f32::INFINITY; n];
    for start in 0..n {
        if comp[start] != usize::MAX {
            continue;
        }
        let id = c.components;
        c.components += 1;
        // Dijkstra from the component's first vertex (positive f32 bits order as the floats).
        let mut heap = std::collections::BinaryHeap::new();
        d[start] = 0.0;
        heap.push(std::cmp::Reverse((0u32, start)));
        let mut members = Vec::new();
        while let Some(std::cmp::Reverse((db, u))) = heap.pop() {
            if comp[u] == id {
                continue;
            }
            comp[u] = id;
            members.push(u);
            let du = f32::from_bits(db);
            for &(w, l) in &adj[u] {
                let nd = du + l;
                if comp[w] != id && nd < d[w] {
                    d[w] = nd;
                    heap.push(std::cmp::Reverse((nd.to_bits(), w)));
                }
            }
        }
        let e = members.iter().map(|&u| d[u]).fold(0f32, f32::max);
        // The posed radius about the first
        // vertex, not the bbox diagonal over 2: a rigid bone keeps every
        // |p - p0| <= geodesic <= e, so radius <= s e holds exactly, while a
        // ball of radius e has a bbox diagonal up to 2 sqrt(3) e (the old
        // bound failed whole shapes whose first vertex sits mid-shape).
        let mut finite = true;
        let mut radius = 0f32;
        for &u in &members {
            finite &= posed[u].iter().all(|v| v.is_finite());
            radius = radius.max(dist(posed[u], posed[start]));
        }
        let q = if !finite {
            f32::INFINITY
        } else if e > 0.0 {
            radius / (c.s * e)
        } else if radius > 0.0 {
            f32::INFINITY
        } else {
            0.0
        };
        if !(q <= c.q) {
            c.q = q;
        }
    }
    c.torn = torn_components(&comp, c.components, &bind, &posed);
    // Q is exactly 1 on any straight run from the first vertex (geodesic
    // = Euclidean); one weight step (1/255: raw u8 weights, not normalised)
    // covers the title's own per-vertex shrink and the f32 rounding
    // (three draws of one captured frame, whole on the sheet: q 1.0017-1.0026).
    c.pass = c.bad_bones == 0 && c.q <= 1.0 + 1.0 / 255.0 && c.torn == 0;
    Some(c)
}

/// Torn components (a vehicle in floating pieces, lamps off their
/// bases): each welded component can be rigid and yet sit metres from the
/// parts it touched. Components whose bind-pose boxes intersect are
/// neighbours; a component with neighbours whose posed box meets none of
/// their posed boxes has left them. A rigid or articulated pose (one bone,
/// or parts turning about a joint they share) keeps every contact; a stale
/// window moves each part by its own unrelated bone. Returns how many left.
fn torn_components(comp: &[usize], n: usize, bind: &[[f32; 3]], posed: &[[f32; 3]]) -> usize {
    if n < 2 || n > 4096 {
        return 0;
    }
    let mut bb = vec![([f32::MAX; 3], [f32::MIN; 3]); n];
    let mut pb = bb.clone();
    for (v, &k) in comp.iter().enumerate() {
        for i in 0..3 {
            (bb[k].0[i], bb[k].1[i]) = (bb[k].0[i].min(bind[v][i]), bb[k].1[i].max(bind[v][i]));
            (pb[k].0[i], pb[k].1[i]) = (pb[k].0[i].min(posed[v][i]), pb[k].1[i].max(posed[v][i]));
        }
    }
    let meet = |a: &([f32; 3], [f32; 3]), b: &([f32; 3], [f32; 3])| (0..3).all(|i| a.0[i] <= b.1[i] && b.0[i] <= a.1[i]);
    (0..n)
        .filter(|&a| {
            let mut neighbours = (0..n).filter(|&b| b != a && meet(&bb[a], &bb[b])).peekable();
            neighbours.peek().is_some() && neighbours.all(|b| !meet(&pb[a], &pb[b]))
        })
        .count()
}

/// The skin check of every ped draw of a live list (`ped_only`: else
/// every skinned draw), with the frame's palette as the pass builds it.
pub fn skin_check(list: &FrameList, geoms: &HashMap<GeomKey, (Vec<u8>, Vec<u32>)>, ped_only: bool, rules: u32) -> Vec<SkinCheck> {
    let (pal, first) = pass_palette(list);
    let mut out = Vec::new();
    for (i, r) in list.draws.iter().enumerate() {
        if ped_only && !r.ped {
            continue;
        }
        let Some((vb, idx)) = geoms.get(&r.geom) else { continue };
        if let Some(mut c) = skin_check_draw(r, vb, idx, &pal, &first, rules) {
            c.draw = i;
            out.push(c);
        }
    }
    out
}

/// For the sheet: draw `i`'s vertices in the list's camera-relative space:
/// (bind pose x world, posed x world). Skinned draws are posed with the
/// shader's maths under `rules`; others have posed = bind. None without geometry.
pub fn draw_points(list: &FrameList, geoms: &HashMap<GeomKey, (Vec<u8>, Vec<u32>)>, i: usize, rules: u32) -> Option<(Vec<[f32; 3]>, Vec<[f32; 3]>)> {
    let r = list.draws.get(i)?;
    let (vb, idx) = geoms.get(&r.geom)?;
    let s = Streams::parse(vb)?;
    let mut verts: Vec<u32> = idx.to_vec();
    verts.sort_unstable();
    verts.dedup();
    let skinned = r.palette != NO_PALETTE && s.weights[0].stride != 0 && s.indices[0].stride != 0;
    let (pal, first) = if skinned { pass_palette(list) } else { (Vec::new(), Vec::new()) };
    let pf = first.get(r.palette as usize).copied().unwrap_or(0);
    let pal_first = if r.bone_count == 0 { pf } else { pf + 3 * r.bone_first };
    let remap_at = (r.remap != NO_REMAP).then(|| 4 * pf + r.remap);
    let (mut bind, mut posed) = (Vec::with_capacity(verts.len()), Vec::with_capacity(verts.len()));
    for &v in &verts {
        let p = s.position(vb, v)?;
        let q = if skinned {
            let (t, n) = skin_terms(&s, vb, v, &pal, remap_at, r.ped, rules);
            skin_apply(&t[..n], p, &pal, pal_first, r.ped)
        } else {
            p
        };
        bind.push(row_mul(p, 1.0, &r.world));
        posed.push(row_mul(q, 1.0, &r.world));
    }
    // An instanced draw: its first instance (12 floats: x, y, z axes, translation; as the shader).
    if r.inst_count != 0
        && let Some(w) = list.instances.get(12 * r.inst_first as usize..12 * r.inst_first as usize + 12)
    {
        let t = |p: [f32; 3]| -> [f32; 3] { std::array::from_fn(|k| p[0] * w[k] + p[1] * w[3 + k] + p[2] * w[6 + k] + w[9 + k]) };
        bind = bind.into_iter().map(t).collect();
        posed = posed.into_iter().map(t).collect();
    }
    Some((bind, posed))
}

/// For the lab: draw `i`'s skin sanity under `rules` (None: not a skinned
/// draw with geometry). Builds the palette buffer per call: for one draw.
pub fn skin_sane(list: &FrameList, geoms: &HashMap<GeomKey, (Vec<u8>, Vec<u32>)>, i: usize, rules: u32) -> Option<bool> {
    let r = list.draws.get(i)?;
    let (vb, idx) = geoms.get(&r.geom)?;
    let (pal, first) = pass_palette(list);
    skin_check_draw(r, vb, idx, &pal, &first, rules).map(|c| c.pass)
}

/// `world-render <frame> --skin-check [all] [--out-tsv f] [--skin-rules hex]`:
/// the table to `f` (default skin-check.tsv), the pass rate printed.
pub fn skin_check_cli(list: &FrameList, geoms: &HashMap<GeomKey, (Vec<u8>, Vec<u32>)>, ped_only: bool, rules: u32, out: &std::path::Path) -> Result<(), String> {
    let all = skin_check(list, geoms, ped_only, rules);
    let mut t = String::from("draw\tstage\tremap\tbone_first\tbone_count\tverts\tcomponents\tbones\tpast_count\tmax_raw\tbad_bones\tdq_like\ts\tq\tedge_p99\tortho\tidentity\tpass\ttorn\n");
    for c in &all {
        let r = &list.draws[c.draw];
        let _ = writeln!(t, "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{}", c.draw, r.stage, r.remap as i32, r.bone_first, r.bone_count, c.verts, c.components, c.bones, c.past_count, c.max_raw, c.bad_bones, c.dq_like, c.s, c.q, c.edge_p99, c.ortho, c.identity, u8::from(c.pass), c.torn);
    }
    std::fs::write(out, t).map_err(|e| e.to_string())?;
    let pass = all.iter().filter(|c| c.pass).count();
    let remapped = all.iter().filter(|c| list.draws[c.draw].remap != NO_REMAP).count();
    let rp = all.iter().filter(|c| c.pass && list.draws[c.draw].remap != NO_REMAP).count();
    println!("skin check (rules {rules:#x}): {pass} of {} {} draws pass ({:.1} %); with remap {rp} of {remapped}, without {} of {} -> {}", all.len(), if ped_only { "ped" } else { "skinned" }, 100.0 * pass as f64 / all.len().max(1) as f64, pass - rp, all.len() - remapped, out.display());
    if !ped_only {
        skin_kinds(list, geoms, &all, rules, &out.with_extension("kinds.tsv"))?;
    }
    Ok(())
}

/// The records the pass's `skinned` skip removes from the drawn main view
/// (main view, not the prepass or another view's stage, not ped, a palette,
/// more than one bone: worldrender.rs `Pass::draw` filters).
pub fn skin_skip_target(list: &FrameList) -> Vec<usize> {
    let main: Vec<u64> = list.main_draws().map(|r| r.viewport).collect();
    list.draws
        .iter()
        .enumerate()
        .filter(|(_, r)| main.contains(&r.viewport) && !PREPASS_STAGES.contains(&r.stage) && !OTHER_VIEW_STAGES.contains(&r.stage) && !r.ped && r.palette != NO_PALETTE && r.bone_count > 1)
        .map(|(i, _)| i)
        .collect()
}

/// The kind of a skinned non-ped record: `identity` (every bone its
/// vertices reach is exactly the identity: the title's rest pose),
/// `one bone` (all vertices on one non-identity bone), `blend` (several),
/// `no stream` (no weight/index stream: the shader draws it unskinned),
/// `no geom` (not captured).
pub fn skin_kind(c: Option<&SkinCheck>, has_geom: bool) -> &'static str {
    match c {
        None if !has_geom => "no geom",
        None => "no stream",
        Some(c) if c.bones > 0 && c.identity == c.bones => "identity",
        Some(c) if c.bones <= 1 => "one bone",
        Some(_) => "blend",
    }
}

/// The skip target's records by kind x verdict (printed) and one row per
/// record to `out`.
fn skin_kinds(list: &FrameList, geoms: &HashMap<GeomKey, (Vec<u8>, Vec<u32>)>, all: &[SkinCheck], rules: u32, out: &std::path::Path) -> Result<(), String> {
    let target = skin_skip_target(list);
    let by: HashMap<usize, &SkinCheck> = all.iter().map(|c| (c.draw, c)).collect();
    let mut t = String::from("draw\tstage\tkind\tverdict\trigid\tbone_first\tbone_count\tbones\tidentity\tortho\ts\tq\tmesh_at\n");
    let mut table: std::collections::BTreeMap<(&str, &str), [usize; 2]> = Default::default();
    let mut stages: std::collections::BTreeMap<u32, usize> = Default::default();
    for &i in &target {
        let r = &list.draws[i];
        let c = by.get(&i).copied();
        let kind = skin_kind(c, geoms.contains_key(&r.geom));
        let verdict = match c {
            Some(c) if c.pass => "sane",
            Some(_) => "broken",
            None => "not judged",
        };
        let rigid = list.rigid.iter().any(|(k, _)| *k as usize == i);
        let e = table.entry((kind, verdict)).or_default();
        e[0] += 1;
        e[1] += usize::from(rigid);
        *stages.entry(r.stage).or_default() += 1;
        let _ = writeln!(t, "{i}\t{}\t{kind}\t{verdict}\t{:?}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:#x}", r.stage, xnw_world::skin_window(list, r), u8::from(rigid), r.bone_first, r.bone_count, c.map_or(0, |c| c.bones), c.map_or(0, |c| c.identity), c.map_or(0.0, |c| c.ortho), c.map_or(0.0, |c| c.s), c.map_or(0.0, |c| c.q), r.geom.mesh_at);
    }
    std::fs::write(out, t).map_err(|e| e.to_string())?;
    println!("skinned-other skip target (rules {rules:#x}): {} records; by stage {}", target.len(), stages.iter().map(|(s, n)| format!("{s}:{n}")).collect::<Vec<_>>().join(" "));
    for ((kind, verdict), [n, rigid]) in &table {
        println!("  {kind:<10} {verdict:<10} {n:>5}  (rigid.tsv {rigid})");
    }
    println!("  -> {}", out.display());
    Ok(())
}

/// Measures one frame's drawn main-view records.
pub fn measure_frame(frame: u64, list: &FrameList, geoms: &HashMap<GeomKey, (Vec<u8>, Vec<u32>)>) -> Vec<Measure> {
    let Some(cam) = list.main_camera().copied() else { return Vec::new() };
    let origin = xnw_world::camera_position(&cam.view);
    let rot = xnw_world::rotation_only(&cam.view);
    let vp = cam.view_proj_relative();
    let (pal, first) = palette_buffer(list);
    let mut out = Vec::new();
    let mut ordinal: HashMap<(GeomKey, u32), u32> = HashMap::new();
    for (di, r) in list.main_draws().enumerate() {
        // Static world only (skinned draws are left out of every table).
        if PREPASS_STAGES.contains(&r.stage) || OTHER_VIEW_STAGES.contains(&r.stage) || r.palette != NO_PALETTE {
            continue;
        }
        let Some((vb, idx)) = geoms.get(&r.geom) else { continue };
        let Some(s) = Streams::parse(vb) else { continue };
        let n = ordinal.entry((r.geom, r.stage)).or_default();
        let key = (r.geom, r.stage, *n);
        *n += 1;
        // Skin parameters as the pass sets them.
        let (pal_first, bones) = match list.palettes.get(r.palette as usize) {
            Some(p) if r.palette != NO_PALETTE => match r.bone_count {
                0 => (first[r.palette as usize], (p.len() / 12) as u32),
                c => (first[r.palette as usize] + 3 * r.bone_first, c),
            },
            _ => (0, 0),
        };
        let remap_at = (r.remap != NO_REMAP && bones > 0 && (r.palette as usize) < first.len()).then(|| 4 * first[r.palette as usize] + r.remap);
        let skinned = bones > 0 && s.weights[0].stride != 0 && s.indices[0].stride != 0;
        // Vertices: the indexed ones, spread.
        let step = (idx.len() / MAX_VERTS).max(1);
        let verts: Vec<u32> = idx.iter().step_by(step).copied().collect();
        let mut local = ([f32::MAX; 3], [f32::MIN; 3]);
        let worlds: Vec<Mat> = if r.inst_count > 0 {
            let total = (list.instances.len() / 12) as u32;
            let n = r.inst_count.min(total.saturating_sub(r.inst_first)) as usize;
            (0..n.min(MAX_INSTANCES)).map(|k| {
                let i = (r.inst_first as usize + k * n / n.min(MAX_INSTANCES).max(1)) * 12;
                let w = &list.instances[i..i + 12];
                [w[0], w[1], w[2], 0.0, w[3], w[4], w[5], 0.0, w[6], w[7], w[8], 0.0, w[9], w[10], w[11], 1.0]
            }).collect()
        } else {
            vec![r.world]
        };
        let mut pos: HashMap<u32, Vec<[f32; 3]>> = HashMap::new();
        for &v in &verts {
            let Some(p) = s.position(vb, v) else { continue };
            for i in 0..3 {
                local.0[i] = local.0[i].min(p[i]);
                local.1[i] = local.1[i].max(p[i]);
            }
            let p = if skinned { skin(&s, vb, v, p, &pal, pal_first, remap_at, r.ped) } else { p };
            pos.insert(v, worlds.iter().map(|w| row_mul(p, 1.0, w)).collect());
        }
        if pos.is_empty() {
            continue;
        }
        let local_diag = ((0..3).map(|i| (local.1[i] - local.0[i]).powi(2)).sum::<f32>()).sqrt().max(1e-3);
        let mut m = Measure { frame, draw: di, class: class_of(r), key, local_diag, stage: r.stage, triangles: (idx.len() / 3) as u32, state: r.material.state_index, effect: r.effect, ..Default::default() };
        let w = &r.world;
        m.abs_world = [w[0], w[1], w[2], w[4], w[5], w[6], w[8], w[9], w[10], w[12] + origin[0], w[13] + origin[1], w[14] + origin[2]];
        // (a) the longest edge of the measured triangles.
        let mut longest = 0f32;
        for t in idx.chunks_exact(3).step_by((idx.len() / 3 / MAX_VERTS).max(1)) {
            for e in 0..3 {
                let (Some(a), Some(b)) = (pos.get(&t[e]), pos.get(&t[(e + 1) % 3])) else { continue };
                for (pa, pb) in a.iter().zip(b) {
                    longest = longest.max(((0..3).map(|i| (pa[i] - pb[i]).powi(2)).sum::<f32>()).sqrt());
                }
            }
        }
        // Instances scale their geometry (up to 15x): the edge test is
        // against the local diagonal times the largest axis scale.
        let scale = worlds.iter().map(|w| (0..3).map(|r| (w[4 * r] * w[4 * r] + w[4 * r + 1] * w[4 * r + 1] + w[4 * r + 2] * w[4 * r + 2]).sqrt()).fold(0f32, f32::max)).fold(0f32, f32::max).max(1e-6);
        m.stretch = longest / (local_diag * scale.max(1.0));
        m.flags[0] = m.stretch > EXPLODE;
        // (b), (c), (d) and the bbox.
        let all: Vec<[f32; 3]> = pos.values().flatten().copied().collect();
        let nan = all.iter().any(|p| p.iter().any(|x| !x.is_finite()));
        m.flags[2] = nan;
        let fin: Vec<&[f32; 3]> = all.iter().filter(|p| p.iter().all(|x| x.is_finite())).collect();
        m.cover = screen_cover(&fin.iter().map(|p| **p).collect::<Vec<_>>(), &vp);
        let far = fin.iter().filter(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() > FAR_M).count();
        let behind = fin.iter().filter(|p| row_mul(***p, 0.0, &rot)[2] > 0.0).count();
        m.flags[1] = !fin.is_empty() && (far as f32 > FAR_SHARE * fin.len() as f32 || behind == fin.len());
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in &fin {
            for i in 0..3 {
                lo[i] = lo[i].min(p[i]);
                hi[i] = hi[i].max(p[i]);
            }
        }
        if !fin.is_empty() {
            m.diag = ((0..3).map(|i| (hi[i] - lo[i]).powi(2)).sum::<f32>()).sqrt();
            m.centre = std::array::from_fn(|i| (lo[i] + hi[i]) * 0.5 + origin[i]);
        }
        // Collapsed: the transformed extent under 1 % of the local one, with
        // more than 8 vertices (single points and quads are left alone).
        m.flags[3] = fin.len() > 8 && m.diag < 0.01 * local_diag * worlds.len() as f32;
        m.score = if m.flags[2] { 1e9 } else { 0.0 } + if m.flags[0] { m.stretch } else { 0.0 } + if m.flags[3] { 50.0 } else { 0.0 } + if m.flags[1] { 20.0 } else { 0.0 };
        out.push(m);
    }
    out
}

/// (e) and (f) over a sequence: jumps of a key's centre between consecutive
/// frames, far above that frame's median jump; keys present in N and N+2
/// but missing in N+1.
pub fn cross_frame(frames: &mut [Vec<Measure>]) {
    // Double drawing: one geometry with the same placement drawn in two kept
    // stages (or twice in one) within a frame (instanced draws excluded).
    for f in frames.iter_mut() {
        let mut seen: HashMap<(GeomKey, [i64; 12]), usize> = HashMap::new();
        for j in 0..f.len() {
            if f[j].class.starts_with("instanced") {
                continue;
            }
            let k = (f[j].key.0, f[j].abs_world.map(|x| (x * 100.0).round() as i64));
            if let Some(&i) = seen.get(&k) {
                f[j].flags[6] = true;
                f[i].flags[6] = true;
                f[j].score += 2.0;
                f[j].repeat_of = Some(f[i].stage);
            } else {
                seen.insert(k, j);
            }
        }
    }
    for i in 1..frames.len() {
        let prev: HashMap<_, ([f32; 3], f32)> = frames[i - 1].iter().map(|m| (m.key, (m.centre, m.diag))).collect();
        let mut jumps: Vec<(usize, f32)> = Vec::new();
        for (j, m) in frames[i].iter().enumerate() {
            if let Some((c, _)) = prev.get(&m.key) {
                jumps.push((j, ((0..3).map(|k| (m.centre[k] - c[k]).powi(2)).sum::<f32>()).sqrt()));
            }
        }
        let mut sorted: Vec<f32> = jumps.iter().map(|j| j.1).collect();
        sorted.sort_by(f32::total_cmp);
        let median = sorted.get(sorted.len().saturating_sub(1) / 2).copied().unwrap_or(0.0);
        for (j, d) in jumps {
            let m = &mut frames[i][j];
            if d > (1.0f32).max(0.5 * m.diag) && d > 5.0 * median.max(0.01) && !m.flags[1] {
                m.flags[4] = true;
                m.score += 10.0 + d.min(1000.0) / 10.0;
            }
        }
    }
    for i in 1..frames.len().saturating_sub(1) {
        // A frame much smaller than both neighbours is a split frame (draw
        // counts alternate at the flush), not a flicker: skip it.
        let (a, b, c) = (frames[i - 1].len(), frames[i].len(), frames[i + 1].len());
        if (b as f32) < 0.8 * a.min(c) as f32 {
            continue;
        }
        let before: std::collections::HashSet<_> = frames[i - 1].iter().map(|m| m.key).collect();
        let now: std::collections::HashSet<_> = frames[i].iter().map(|m| m.key).collect();
        let after_keys: Vec<_> = frames[i + 1].iter().map(|m| m.key).collect();
        for (j, k) in after_keys.iter().enumerate() {
            if before.contains(k) && !now.contains(k) {
                frames[i + 1][j].flags[5] = true;
                frames[i + 1][j].score += 5.0;
            }
        }
    }
}

/// The class table: class, draws, flagged per (a)..(f), worst example.
pub fn table(frames: &[Vec<Measure>]) -> String {
    let mut by: HashMap<&str, (usize, [usize; 7], usize, Option<&Measure>)> = HashMap::new();
    for m in frames.iter().flatten() {
        let e = by.entry(&m.class).or_insert((0, [0; 7], 0, None));
        e.0 += 1;
        for (k, f) in m.flags.iter().enumerate() {
            e.1[k] += usize::from(*f);
        }
        e.2 += usize::from(m.flags.iter().any(|f| *f));
        if m.score > 0.0 && e.3.is_none_or(|w| m.score > w.score) {
            e.3 = Some(m);
        }
    }
    let mut rows: Vec<_> = by.into_iter().collect();
    rows.sort_by(|a, b| b.1.2.cmp(&a.1.2).then(b.1.0.cmp(&a.1.0)));
    let mut s = String::from("class\tdraws\tflagged\texploded\tfar_behind\tnan\tcollapsed\tjump\tflicker\tdouble\tworst_frame\tworst_draw\tworst_stretch\n");
    for (c, (n, f, any, w)) in rows {
        let (wf, wd, ws) = w.map_or(("-".into(), "-".into(), "-".into()), |w| (w.frame.to_string(), w.draw.to_string(), format!("{:.1}", w.stretch)));
        let _ = writeln!(s, "{c}\t{n}\t{any} ({:.0} %)\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{wf}\t{wd}\t{ws}", 100.0 * any as f32 / n.max(1) as f32, f[0], f[1], f[2], f[3], f[4], f[5], f[6]);
    }
    s
}

/// `xnw world-render <dir> --seq --audit`: measures every frame of a
/// live-list sequence, writes `audit.tsv` (the class table), `audit-worst.tsv`
/// (the 12 worst draws) and `audit-sheet.png` (each of them drawn alone, 4 x 3)
/// to `out`; prints the table.
pub fn run(frames: &[(u64, std::path::PathBuf)], out: &std::path::Path, pass: &mut xnw_world::Pass) -> Result<(), String> {
    let mut all = Vec::new();
    let mut masks: Vec<(usize, usize)> = Vec::new();
    // Coverage per (kind, stage): recorded, other camera, filtered (rule),
    // no geometry, drawn, triangles drawn; skinned left out.
    let mut cov: std::collections::BTreeMap<String, [u64; 7]> = Default::default();
    for (n, dir) in frames {
        let (list, geoms) = super::load_list(dir)?;
        let main: std::collections::HashSet<*const DrawRecord> = list.main_draws().map(|r| r as *const DrawRecord).collect();
        for r in &list.draws {
            if r.palette != NO_PALETTE {
                continue;
            }
            let kind = if r.inst_count > 0 { "instanced" } else { "rigid" };
            let e = cov.entry(format!("{kind} st{}", r.stage)).or_default();
            e[0] += 1;
            if !main.contains(&(r as *const DrawRecord)) {
                e[1] += 1;
            } else if PREPASS_STAGES.contains(&r.stage) {
                e[2] += 1;
            } else if OTHER_VIEW_STAGES.contains(&r.stage) {
                e[3] += 1;
            } else if let Some((_, idx)) = geoms.get(&r.geom) {
                e[5] += 1;
                e[6] += (idx.len() / 3) as u64 * u64::from(r.inst_count.max(1));
            } else {
                e[4] += 1;
            }
        }
        all.push(measure_frame(*n, &list, &geoms));
        // Per frame, the drawn draws with a sample mask, and stage 12/13 draws still kept.
        let kept: Vec<&DrawRecord> = list.main_draws().filter(|r| r.palette == NO_PALETTE && !PREPASS_STAGES.contains(&r.stage) && !OTHER_VIEW_STAGES.contains(&r.stage)).collect();
        masks.push((kept.iter().filter(|r| r.material.stipple != 0).count(), kept.iter().filter(|r| r.stage == 12 || r.stage == 13).count()));
    }
    let (mx, kx) = (masks.iter().map(|m| m.0).max().unwrap_or(0), masks.iter().map(|m| m.1).max().unwrap_or(0));
    println!("draws with a sample mask per frame: max {mx} (total {}); stage 12/13 draws kept per frame: max {kx}", masks.iter().map(|m| m.0).sum::<usize>());
    // Per frame and per pair of consecutive frames: static draws drawn (a
    // split frame halves one frame, not the pair).
    let per: Vec<usize> = all.iter().map(Vec::len).collect();
    let pairs: Vec<usize> = all.windows(2).map(|w| w[0].iter().chain(&w[1]).map(|m| m.key).collect::<std::collections::HashSet<_>>().len()).collect();
    let stat = |v: &[usize]| -> String {
        let mut s = v.to_vec();
        s.sort_unstable();
        if s.is_empty() { "-".into() } else { format!("min {} median {} max {}", s[0], s[s.len() / 2], s[s.len() - 1]) }
    };
    println!("static draws measured per frame: {}; distinct per pair of frames: {}; first 12 frames: {:?}", stat(&per), stat(&pairs), &per[..per.len().min(12)]);
    let mut c = String::from("class\trecorded\tother_camera\tfiltered_prepass\tfiltered_other_view\tno_geometry\tdrawn\ttriangles_drawn\n");
    for (k, v) in &cov {
        let _ = writeln!(c, "{k}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
    }
    std::fs::write(out.join("audit-coverage.tsv"), &c).map_err(|e| e.to_string())?;
    print!("{c}");
    cross_frame(&mut all);
    // Placement: absolute position of every static draw per frame.
    let mut pos = String::from("frame\tdraw\tclass\tmesh_at\thash\tordinal\tx\ty\tz\tcentre_x\tcentre_y\tcentre_z\n");
    for m in all.iter().flatten() {
        let _ = writeln!(pos, "{}\t{}\t{}\t{:#x}\t{:#x}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}", m.frame, m.draw, m.class, m.key.0.mesh_at, m.key.0.hash, m.key.2, m.abs_world[9], m.abs_world[10], m.abs_world[11], m.centre[0], m.centre[1], m.centre[2]);
    }
    std::fs::write(out.join("static-positions.tsv"), &pos).map_err(|e| e.to_string())?;
    // Draws covering more than BIG_COVER of the screen (black polygons seen
    // in one capture were such draws): per frame count, and every one listed.
    let mut big = String::from("frame	draw	class	stage	hash	effect	cover
");
    let mut per_frame: Vec<usize> = Vec::new();
    for f in &all {
        let b: Vec<&Measure> = f.iter().filter(|m| m.cover > BIG_COVER).collect();
        per_frame.push(b.len());
        for m in b {
            let _ = writeln!(big, "{}	{}	{}	{}	{:#x}	{:#x}	{:.2}", m.frame, m.draw, m.class, m.stage, m.key.0.hash, m.effect, m.cover);
        }
    }
    std::fs::write(out.join("audit-bigcover.tsv"), &big).map_err(|e| e.to_string())?;
    println!("big-cover draws (> {:.0} % of the screen) per frame: {}; total {}", BIG_COVER * 100.0, stat(&per_frame), per_frame.iter().sum::<usize>());
    let t = table(&all);
    std::fs::write(out.join("audit.tsv"), &t).map_err(|e| e.to_string())?;
    // The repeats: (first stage, repeating stage, the repeat's state indices) and how often.
    let mut reps: HashMap<(u32, u32, [u16; 3]), usize> = HashMap::new();
    for m in all.iter().flatten() {
        if let Some(s) = m.repeat_of {
            *reps.entry((s, m.stage, m.state)).or_default() += 1;
        }
    }
    let mut reps: Vec<_> = reps.into_iter().collect();
    reps.sort_by(|a, b| b.1.cmp(&a.1));
    let mut rt = String::from("first_stage\trepeat_stage\trepeat_blend/dss/raster\trepeats\n");
    for ((a, b, s), n) in &reps {
        let _ = writeln!(rt, "{a}\t{b}\t{}/{}/{}\t{n}", s[0], s[1], s[2]);
    }
    std::fs::write(out.join("audit-repeats.tsv"), &rt).map_err(|e| e.to_string())?;
    print!("{rt}");
    let mut worst: Vec<&Measure> = all.iter().flatten().filter(|m| m.score > 0.0).collect();
    worst.sort_by(|a, b| b.score.total_cmp(&a.score));
    worst.truncate(12);
    let mut w = String::from("frame\tdraw\tclass\tstage\tscore\tstretch\tlocal_diag\ttriangles\tflags\tmesh_at\thash\n");
    for m in &worst {
        let flags: Vec<&str> = FLAGS.iter().zip(m.flags).filter(|(_, f)| *f).map(|(n, _)| *n).collect();
        let _ = writeln!(w, "{}\t{}\t{}\t{}\t{:.1}\t{:.1}\t{:.2}\t{}\t{}\t{:#x}\t{:#x}", m.frame, m.draw, m.class, m.stage, m.score, m.stretch, m.local_diag, m.triangles, flags.join(","), m.key.0.mesh_at, m.key.0.hash);
    }
    std::fs::write(out.join("audit-worst.tsv"), &w).map_err(|e| e.to_string())?;
    // The sheet: each worst draw alone (same camera, palettes, instances), a quarter size.
    let (tw, th) = (pass.width / 4, pass.height / 4);
    let mut sheet = vec![0u8; (tw * 4 * th * 3 * 4) as usize];
    for (i, m) in worst.iter().enumerate() {
        let Some((_, dir)) = frames.iter().find(|(n, _)| *n == m.frame) else { continue };
        let (mut list, geoms) = super::load_list(dir)?;
        let Some(r) = list.main_draws().nth(m.draw).copied() else { continue };
        list.draws = vec![r];
        list.sky = None;
        list.frame = 1_000_000 + i as u64;
        let mut decode = |k: &GeomKey| geoms.get(k).map(|(vb, idx)| xnw_world::HostGeometry { vertices: Vec::new(), indices: idx.clone(), raw_vb: Some(vb.clone()), vb_place: 0 });
        super::watchdog_arm();
        if pass.draw(&list, &mut decode, true).is_err() {
            super::watchdog_disarm();
            continue;
        }
        let px = pass.read_rgba()?;
        super::watchdog_disarm();
        let (ox, oy) = ((i as u32 % 4) * tw, (i as u32 / 4) * th);
        for y in 0..th {
            for x in 0..tw {
                let s = (((y * 4) * pass.width + x * 4) * 4) as usize;
                let d = (((oy + y) * tw * 4 + ox + x) * 4) as usize;
                sheet[d..d + 4].copy_from_slice(&px[s..s + 4]);
            }
        }
    }
    super::write_png(&out.join("audit-sheet.png"), tw * 4, th * 3, &sheet).map_err(|e| e.to_string())?;
    print!("{t}");
    println!("worst:\n{w}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A float-position geometry (format 3, stride 12 at 0x80) with 4 bone
    /// weights/indices streams (u8x4 at slots 6/8), one triangle.
    fn vb(weights: [u8; 4], index: [u8; 4]) -> Vec<u8> {
        let word = |stride: u32, offset: u32, format: u32| (stride / 4) << 5 | (offset / 4) << 9 | format;
        let mut b = vec![0u8; 0x80];
        b[0..4].copy_from_slice(&word(12, 0x80, 3).to_le_bytes());
        b[0x18..0x1C].copy_from_slice(&word(4, 0xA4, 6).to_le_bytes());
        b[0x20..0x24].copy_from_slice(&word(4, 0xB0, 5).to_le_bytes());
        for v in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for x in v {
                b.extend(x.to_le_bytes());
            }
        }
        for _ in 0..3 {
            b.extend(weights);
        }
        for _ in 0..3 {
            b.extend(index);
        }
        b
    }

    fn list_with(rec: DrawRecord, palette: Vec<f32>) -> FrameList {
        let id: Mat = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let proj: Mat = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.1, 0.0];
        FrameList { frame: 1, draws: vec![rec], cameras: vec![xnw_world::Camera { viewport: 1, view: id, proj }], main: Some(1), palettes: vec![palette], ..Default::default() }
    }

    fn rec(palette: u32) -> DrawRecord {
        let id: Mat = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -5.0, 1.0];
        DrawRecord { geom: GeomKey { mesh_at: 1, ..Default::default() }, world: id, viewport: 1, effect: 0, albedo: xnw_world::Albedo::Flat([1.0; 3]), palette, stage: 67, inst_first: 0, inst_count: 0, bone_first: 0, bone_count: 0, material: Default::default(), state: None, remap: NO_REMAP, ped: false }
    }

    /// Identity bones: nothing flagged. A bone with a 100x scale: exploded.
    /// A zero bone: collapsed is not flagged for 3 vertices (too few), but a
    /// NaN bone is.
    #[test]
    fn audit_flags_the_classes() {
        let ident = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut geoms = HashMap::new();
        geoms.insert(GeomKey { mesh_at: 1, ..Default::default() }, (vb([255, 0, 0, 0], [0, 0, 0, 0]), vec![0, 1, 2]));
        // Skinned draws are not measured (static world only).
        assert!(measure_frame(1, &list_with(rec(0), ident.to_vec()), &geoms).is_empty());
        let rigid = measure_frame(1, &list_with(rec(NO_PALETTE), Vec::new()), &geoms);
        assert_eq!((rigid[0].flags, rigid[0].class.as_str()), ([false; 7], "rigid st67"));
        // A world with a 100x scale on one axis only stretches edges past 4 bbox diagonals? No: the
        // bbox is local; a 100x world scale is a legitimate placement, so measure with a NaN world.
        let mut bad = rec(NO_PALETTE);
        bad.world[0] = f32::NAN;
        let m = measure_frame(1, &list_with(bad, Vec::new()), &geoms);
        assert!(m[0].flags[2], "{m:?}");
        // The skin replica itself: a 100x bone stretches the triangle.
        let s = Streams::parse(&geoms.values().next().unwrap().0).unwrap();
        let vbb = &geoms.values().next().unwrap().0;
        let big = [100.0f32, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0];
        assert_eq!(skin(&s, vbb, 1, [1.0, 0.0, 0.0], &big, 0, None, false), [100.0, 0.0, 0.0]);
        let _ = ident;
    }

    /// A two-bone strip (a 4 m rod of 5 vertices along x, a quad strip
    /// in z): vertices 0..2 on bone 0, 3..4 on bone 1. A rigid pose (bone 1
    /// rotated 90 degrees about the joint at x = 2) passes with q <= 1; bone 1
    /// pushed 50 m away (a wrong bone) fails; a NaN bone fails; a zero bone fails.
    #[test]
    fn skin_check_two_bones() {
        let word = |stride: u32, offset: u32, format: u32| (stride / 4) << 5 | (offset / 4) << 9 | format;
        // 10 vertices: x = 0..4 at z = 0 and z = 0.1.
        let pts: Vec<[f32; 3]> = (0..2).flat_map(|z| (0..5).map(move |x| [x as f32, 0.0, z as f32 * 0.1])).collect();
        let n = pts.len() as u32;
        let mut b = vec![0u8; 0x80];
        b[0..4].copy_from_slice(&word(12, 0x80, 3).to_le_bytes());
        b[0x18..0x1C].copy_from_slice(&word(4, 0x80 + 12 * n, 6).to_le_bytes());
        b[0x20..0x24].copy_from_slice(&word(4, 0x80 + 16 * n, 5).to_le_bytes());
        for p in &pts {
            for x in p {
                b.extend(x.to_le_bytes());
            }
        }
        for _ in &pts {
            b.extend([255u8, 0, 0, 0]);
        }
        for p in &pts {
            b.extend([u8::from(p[0] > 2.5), 0, 0, 0]);
        }
        let idx: Vec<u32> = (0..4).flat_map(|i| [i, i + 1, i + 5, i + 1, i + 6, i + 5]).collect();
        let ident = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        // Rows (X.r, Y.r, Z.r, T.r): rotate 90 deg about z through (2, 0, 0): x' = 2 - y, y' = x - 2.
        let rot = [0.0f32, -1.0, 0.0, 2.0, 1.0, 0.0, 0.0, -2.0, 0.0, 0.0, 1.0, 0.0];
        let far = [1.0f32, 0.0, 0.0, 50.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut r = rec(0);
        (r.ped, r.bone_count) = (true, 2);
        let check = |bone1: [f32; 12]| {
            let pal: Vec<f32> = ident.iter().chain(bone1.iter()).copied().collect();
            skin_check_draw(&r, &b, &idx, &pal, &[0], 0).unwrap()
        };
        let ok = check(rot);
        assert!(ok.pass && ok.q <= 1.0 && ok.components == 1 && ok.bones == 2, "{ok:?}");
        assert!((ok.s - 1.0).abs() < 1e-4);
        let bad = check(far);
        assert!(!bad.pass && bad.q > 1.0, "{bad:?}");
        let mut nan = ident;
        nan[0] = f32::NAN;
        assert!(!check(nan).pass);
        assert!(!check([0.0; 12]).pass);
        // The identity pose: the same as the bind pose.
        assert!(check(ident).pass);
    }

    /// A float-position geometry of `pts`, every vertex on bone 0 (weight
    /// 255) when `skinned`, else no weight/index stream.
    fn vb_of(pts: &[[f32; 3]], skinned: bool) -> Vec<u8> {
        let word = |stride: u32, offset: u32, format: u32| (stride / 4) << 5 | (offset / 4) << 9 | format;
        let n = pts.len() as u32;
        let mut b = vec![0u8; 0x80];
        b[0..4].copy_from_slice(&word(12, 0x80, 3).to_le_bytes());
        if skinned {
            b[0x18..0x1C].copy_from_slice(&word(4, 0x80 + 12 * n, 6).to_le_bytes());
            b[0x20..0x24].copy_from_slice(&word(4, 0x80 + 16 * n, 5).to_le_bytes());
        }
        for x in pts.iter().flatten() {
            b.extend(x.to_le_bytes());
        }
        if skinned {
            for _ in pts {
                b.extend([255u8, 0, 0, 0]);
            }
            b.extend(std::iter::repeat_n(0u8, 4 * pts.len()));
        }
        b
    }

    /// A star of six unit arms (+-x, +-y, +-z) about its first vertex under
    /// identity bones is the bind pose. The earlier bound (bbox diagonal / 2 s e)
    /// read sqrt(3) for it; the radius bound reads 1 and passes, and a 2x
    /// stretched arm still fails.
    #[test]
    fn skin_check_star_is_whole() {
        let pts = [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
        let b = vb_of(&pts, true);
        let idx = [0u32, 1, 2, 0, 3, 4, 0, 5, 6];
        let ident = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut r = rec(0);
        r.bone_count = 1;
        let c = skin_check_draw(&r, &b, &idx, &ident, &[0], 0).unwrap();
        assert!(c.pass && (c.q - 1.0).abs() < 1e-6 && c.components == 1, "{c:?}");
        // The old bound on the same posed points (bbox [-1, 1]^3, e = 1) failed it.
        assert!((2.0f32 * 3f32.sqrt()) / (2.0 * c.s * 1.0) > 1.7);
        // A rigid move of the whole component stays whole; torn parts still
        // fail (skin_check_two_bones: a bone 50 m away, NaN, zero).
        let moved = skin_check_draw(&r, &b, &idx, &[0.0, -1.0, 0.0, 50.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[0], 0).unwrap();
        assert!(moved.pass, "{moved:?}");
    }

    /// Torn components: a
    /// base (bone 0, a flat triangle) and a part resting on it (bone 1, a
    /// standing triangle whose foot is inside the base's box), not welded.
    /// Both bones equal, or the part turned about its foot: whole. The part's
    /// bone 2 m up: each piece is rigid (q passes) but it left the base: torn.
    #[test]
    fn skin_check_torn_part() {
        let word = |stride: u32, offset: u32, format: u32| (stride / 4) << 5 | (offset / 4) << 9 | format;
        let pts = [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.2, 0.0], [0.3, 0.2, 0.0], [0.2, 0.2, 0.5]];
        let n = pts.len() as u32;
        let mut b = vec![0u8; 0x80];
        b[0..4].copy_from_slice(&word(12, 0x80, 3).to_le_bytes());
        b[0x18..0x1C].copy_from_slice(&word(4, 0x80 + 12 * n, 6).to_le_bytes());
        b[0x20..0x24].copy_from_slice(&word(4, 0x80 + 16 * n, 5).to_le_bytes());
        for x in pts.iter().flatten() {
            b.extend(x.to_le_bytes());
        }
        for _ in 0..n {
            b.extend([255u8, 0, 0, 0]);
        }
        for v in 0..n {
            b.extend([u8::from(v >= 3), 0, 0, 0]);
        }
        let idx = [0u32, 1, 2, 3, 4, 5];
        let ident = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut r = rec(0);
        r.bone_count = 2;
        let check = |b1: [f32; 12]| skin_check_draw(&r, &b, &idx, &[ident, b1].concat(), &[0], 0).unwrap();
        let rest = check(ident);
        assert!(rest.pass && rest.torn == 0 && rest.components == 2, "{rest:?}");
        // Turned 90 degrees about z through its foot (0.2, 0.2): x' = 0.4 - y, y' = x.
        let turned = check([0.0, -1.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert!(turned.pass && turned.torn == 0, "{turned:?}");
        let lifted = check([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0]);
        // Both the part and the base it left count (each has lost its only neighbour).
        assert!(lifted.q <= 1.0 && lifted.torn == 2 && !lifted.pass, "{lifted:?}");
    }

    /// (e) a key jumping 100 m while others stay; (f) a key missing in the
    /// middle frame.
    #[test]
    fn screen_cover_shares() {
        // Identity view-projection: x, y in clip space, w = 1 (row 3 = [0,0,0,1]) ...
        let mut vp = [0f32; 16];
        (vp[0], vp[5], vp[10], vp[15]) = (1.0, 1.0, 1.0, 1.0);
        assert!((screen_cover(&[[-1.0, -1.0, 0.5], [1.0, 1.0, 0.5]], &vp) - 1.0).abs() < 1e-6);
        assert!((screen_cover(&[[0.0, 0.0, 0.5], [0.5, 0.5, 0.5]], &vp) - 0.0625).abs() < 1e-6);
        // ... and a w from z: points behind (w < 0) and in front straddle the camera.
        let mut persp = vp;
        (persp[11], persp[15]) = (1.0, 0.0);
        assert_eq!(screen_cover(&[[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], &persp), 1.0);
        assert_eq!(screen_cover(&[[0.0, 0.0, -1.0]], &persp), 0.0);
    }

    #[test]
    fn audit_cross_frame() {
        let mk = |k: u64, x: f32| Measure { key: (GeomKey { mesh_at: k, ..Default::default() }, 67, 0), centre: [x, 0.0, 0.0], diag: 1.0, class: "rigid st67".into(), ..Default::default() };
        let mut frames = vec![vec![mk(1, 0.0), mk(2, 0.0), mk(3, 0.0)], vec![mk(1, 0.0), mk(2, 100.0), mk(4, 0.0)], vec![mk(1, 0.0), mk(2, 100.0), mk(3, 0.0)]];
        cross_frame(&mut frames);
        assert!(frames[1][1].flags[4] && !frames[1][0].flags[4], "key 2 jumped");
        assert!(frames[2][2].flags[5], "key 3 flickered");
        let t = table(&frames);
        assert!(t.contains("rigid st67\t9\t2 (22 %)"), "{t}");
        // Double drawing: the same geometry and placement twice in a frame.
        let mut f = vec![vec![mk(9, 0.0), mk(9, 0.0)]];
        cross_frame(&mut f);
        assert!(f[0][0].flags[6] && f[0][1].flags[6]);
    }
}

/// The v1 draw (draws.tsv index) of each list.tsv row, matched by
/// geometry address, geometry hash and world translation (first unused).
pub fn list_v1_draws(dir: &std::path::Path, list: &FrameList) -> Vec<Option<u32>> {
    let Ok(t) = std::fs::read_to_string(dir.join("draws.tsv")) else { return vec![None; list.draws.len()] };
    let mut lines = t.lines();
    let head: Vec<&str> = lines.next().unwrap_or("").split('\t').collect();
    let col = |n: &str| head.iter().position(|h| *h == n);
    let (Some(cd), Some(cg), Some(cw), Some(ch)) = (col("draw"), col("mesh_data"), col("world"), col("ghash")) else { return vec![None; list.draws.len()] };
    let hex = |s: &str| u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0);
    let mut by: HashMap<(u64, u64), Vec<(u32, [f32; 3], bool)>> = HashMap::new();
    for l in lines {
        let f: Vec<&str> = l.split('\t').collect();
        let (Some(d), Some(g), Some(w), Some(h)) = (f.get(cd), f.get(cg), f.get(cw), f.get(ch)) else { continue };
        let w: Vec<f32> = w.replace('|', " ").split_whitespace().filter_map(|x| x.parse().ok()).collect();
        if w.len() < 16 {
            continue;
        }
        by.entry((hex(g), hex(h))).or_default().push((d.parse().unwrap_or(u32::MAX), [w[12], w[13], w[14]], false));
    }
    list.draws.iter().map(|r| {
        let c = by.get_mut(&(r.geom.mesh_at, r.geom.hash))?;
        let t = [r.world[12], r.world[13], r.world[14]];
        let e = c.iter_mut().filter(|e| !e.2).min_by(|a, b| {
            let d = |p: [f32; 3]| (0..3).map(|i| (p[i] - t[i]).abs()).sum::<f32>();
            d(a.1).total_cmp(&d(b.1))
        })?;
        e.2 = true;
        Some(e.0)
    }).collect()
}

/// What a v2 pack says about the first `n` ped rows: the rec (tbl9+0,
/// slot 0) and remap (tbl9+0x20, slot 1) views and bytes, hook palettes.
pub fn pack_probe(dir: &std::path::Path, list: &FrameList, n: usize) -> Result<(), String> {
    let Some(pack) = crate::pack::Pack::open(dir)? else { return Err("no pack".into()) };
    let v1 = list_v1_draws(dir, list);
    println!("pack: {} of {} list rows matched to a v1 draw", v1.iter().flatten().count(), v1.len());
    let peds: Vec<usize> = (0..list.draws.len()).filter(|&i| list.draws[i].ped).collect();
    let sinked = peds.iter().filter(|&&i| v1[i].and_then(|d| pack.sink_of(d)).is_some()).count();
    let paled = peds.iter().filter(|&&i| v1[i].is_some_and(|d| pack.hook_palettes(d)[0].is_some())).count();
    println!("pack: ped rows {}, with a 1:1 sink {sinked}, with a hook palette {paled}", peds.len());
    // Older captures' lists carry no remap column; every ped row with a sink then.
    let v2_rows = pack.list_draws().is_some();
    for (i, r) in list.draws.iter().enumerate().filter(|(i, r)| r.ped && (r.remap != NO_REMAP || v2_rows) && v1[*i].and_then(|d| pack.sink_of(d)).is_some()).take(n) {
        let Some(d) = v1[i] else {
            println!("row {i}: no v1 draw");
            continue;
        };
        let u = |b: &[u8], at: usize| b.get(at..at + 4).map_or(u32::MAX, |s| u32::from_le_bytes(s.try_into().unwrap()));
        let pal = pack.hook_palettes(d).map(|p| p.map(|(a, b)| (a, b.len() / 48)));
        print!("row {i} v1 {d} remap {} bf {} bc {}: pal {pal:?}", r.remap as i32, r.bone_first, r.bone_count);
        // Every table root of the sink: (root, slots as slot:kind:bytes).
        for (root, h, views) in pack.tables(pack.sink_of(d).unwrap_or(u32::MAX)) {
            print!("; root {root} {h:#x} [{}]", views.iter().map(|v| format!("{}:{}:{}", v.slot, v.kind, v.bytes)).collect::<Vec<_>>().join(" "));
        }
        let mut remap74: Option<u32> = None;
        for slot in [0u32, 1] {
            let sink = pack.sink_of(d).unwrap_or(u32::MAX);
            match pack.slot_bytes(sink, 1, slot).map(|(v, b)| (1u16, v, b)) {
                Ok((root, v, b)) => {
                    print!("; slot {slot} root {root} kind {} addr {:#x} bytes {} res {:#x} width {} fmt {} words {:x?} len {}", v.kind, v.addr, v.bytes, v.resource, v.width, v.format, &v.words[..12], b.len());
                    if slot == 0 {
                        print!(" rec+0x30 {} +0x74 {} +0x78 {}", u(&b, 0x30), u(&b, 0x74) as i32, u(&b, 0x78) as i32);
                        remap74 = (u(&b, 0x74) != u32::MAX).then(|| u(&b, 0x74));
                    } else {
                        print!(" first {:?}", (0..8).map(|k| u(&b, 4 * k)).collect::<Vec<_>>());
                        if let Some(r74) = remap74 {
                            print!(" at rec74 {:?}", (0..12).map(|k| u(&b, 4 * (r74 as usize + k)) & 0xFFF).collect::<Vec<_>>());
                        }
                    }
                }
                Err(e) => print!("; slot {slot}: {e}"),
            }
        }
        println!();
    }
    Ok(())
}

/// Palette timing. Replaces each skinned row's bones by the
/// palette span the hook read (v2 pack, kind 38 current) and prints how many
/// differ from the end-of-frame copy (pal-N.bin slice) and by how much.
/// Rows with the ped remap keep their palette (the remap entries live in it).
/// The swapped palettes are marked draw-time (`palette_at_draw`); the
/// join is the capture's `list-draws.tsv` when it covers the list, else
/// [`list_v1_draws`]. `rows`: the per-row non-ped lines. Returns the rows swapped.
pub fn use_hook_palettes(dir: &std::path::Path, list: &mut FrameList, rows: bool) -> Result<usize, String> {
    let Some(pack) = crate::pack::Pack::open(dir)? else { return Err("no pack".into()) };
    let v1: Vec<Option<u32>> = match pack.list_draws().filter(|v| v.len() == list.draws.len()) {
        Some(v) => v.into_iter().map(Some).collect(),
        None => list_v1_draws(dir, list),
    };
    list.palette_at_draw.resize(list.palettes.len(), false);
    let (mut same, mut differ, mut missing, mut len_diff) = (0usize, 0usize, 0usize, 0usize);
    let (mut bones_eq, mut bones_near, mut bones_all) = (0usize, 0usize, 0usize);
    let mut diffs: Vec<f32> = Vec::new();
    let mut new_pals: HashMap<u64, u32> = HashMap::new();
    for i in 0..list.draws.len() {
        let r = &list.draws[i];
        if r.palette == NO_PALETTE || r.remap != NO_REMAP {
            continue;
        }
        let Some((addr, bytes)) = v1[i].and_then(|d| pack.hook_palettes(d)[0].clone()) else {
            missing += 1;
            continue;
        };
        let hook: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let n = (hook.len() / 12) as u32;
        let end = list.palettes.get(r.palette as usize).map(|p| {
            let (a, b) = if r.bone_count == 0 { (0, p.len()) } else { (12 * r.bone_first as usize, 12 * (r.bone_first + r.bone_count) as usize) };
            p.get(a.min(p.len())..b.min(p.len())).unwrap_or(&[]).to_vec()
        }).unwrap_or_default();
        if end.len() != hook.len() {
            len_diff += 1;
        }
        // Per bone: how many hook bones equal the end-of-frame bone (bitwise), and the share whose 3x3 matches but translation differs.
        let nb = hook.len().min(end.len()) / 12;
        let eq = (0..nb).filter(|&k| hook[12 * k..12 * k + 12].iter().zip(&end[12 * k..12 * k + 12]).all(|(a, b)| a.to_bits() == b.to_bits())).count();
        let near = (0..nb).filter(|&k| hook[12 * k..12 * k + 12].iter().zip(&end[12 * k..12 * k + 12]).all(|(a, b)| (a - b).abs() < 1e-3)).count();
        bones_eq += eq;
        bones_near += near;
        bones_all += nb;
        let d = hook.iter().zip(&end).map(|(a, b)| if a.is_finite() && b.is_finite() { (a - b).abs() } else if a.to_bits() == b.to_bits() { 0.0 } else { f32::INFINITY }).fold(0f32, f32::max);
        if d == 0.0 && end.len() == hook.len() { same += 1 } else { differ += 1; diffs.push(d) }
        // Non-ped rows one by one (the props' bones: hook vs end of frame).
        if rows && !r.ped {
            let id =|p: &[f32]| p.chunks_exact(12).all(|m| m.iter().enumerate().all(|(j, v)| *v == if j % 5 == 0 { 1.0 } else { 0.0 }));
            println!("  non-ped row {i} stage {} bones {nb}: bit-equal {eq}, within 1e-3 {near}, max diff {d:.4}; hook all identity {}, end all identity {}", r.stage, id(&hook), id(&end));
        }
        let key = addr ^ (u64::from(n) << 48);
        let idx = *new_pals.entry(key).or_insert_with(|| {
            list.palettes.push(hook.clone());
            list.palette_at_draw.push(true);
            (list.palettes.len() - 1) as u32
        });
        let r = &mut list.draws[i];
        (r.palette, r.bone_first, r.bone_count) = (idx, 0, n);
    }
    diffs.sort_by(f32::total_cmp);
    let q = |p: usize| diffs.get(diffs.len().saturating_sub(1) * p / 100).copied().unwrap_or(0.0);
    println!("hook palettes: {} rows: same {same}, differ {differ} (span length differs {len_diff}; max |hook-end| p50 {:.4} p90 {:.4} max {:.4}), no hook palette {missing}; {} distinct hook spans; bones bit-equal {bones_eq}, within 1e-3 {bones_near}, of {bones_all}", same + differ, q(50), q(90), q(100), new_pals.len());
    Ok(same + differ)
}

/// Draw-time bones are the replay's source whenever the capture has
/// them (a v2 pack with hook palettes); `--hook-palettes=off` keeps the
/// end-of-frame copy, `--hook-palettes` adds the per-row lines. Frames
/// without a pack are untouched.
pub fn hook_palettes_arg(args: &[String], dir: &std::path::Path, list: &mut FrameList) -> Result<usize, String> {
    if args.iter().any(|a| a == "--hook-palettes=off") || crate::pack::Pack::open(dir)?.is_none() {
        return Ok(0);
    }
    use_hook_palettes(dir, list, args.iter().any(|a| a == "--hook-palettes"))
}

// ---- vehicle parts (replay side) ----

/// `skin_rules` bit 4: a skinned row's class comes from the mesh
/// program its sink bound ([`prog_class`]), not from the bone count: only
/// Ped* / PreSkin programs (and rows with the ped remap) keep `ped`; rows of
/// a program that reads no palette lose theirs (the title draws them rigid).
pub const SKIN_PROGRAM_CLASS: u32 = 0x10;
/// bit 5: the MSR one-bone skin (one bone per vertex or per mesh, no
/// weights; a4a1cd89 0x208-0x2c0) is decoded into ordinary skin streams
/// ([`one_bone_streams`]), so the pass and the audit skin those parts.
pub const SKIN_ONE_BONE: u32 = 0x20;
/// bit 6: `b += rec+0x54` where the program does it (only under
/// `rec+0x74 != -1`: a4a1cd89 0x7d0-0x7d8, f9dd5454 0x940-0x960).
pub const SKIN_REC_OFFSET: u32 = 0x40;
/// bit 7: the mask|rank compaction of the bone index through the
/// arena words at `rec+0x74` ([`mask_rank`]).
pub const SKIN_MASK_RANK: u32 = 0x80;

/// The vehicle programs' compaction of bone index `b` (after the
/// `rec+0x54` add) through the arena words `mask16 | rank << 16`
/// (a4a1cd89 0x7dc-0x83c, f9dd5454 0x964-0x9c0): a set bit gives
/// `rank + popcount(mask below the bit) + 1`, a clear one entry 0 (the exact
/// zero the title writes at the head of a compacted palette). Words past the
/// slice read 0. The program keeps bits 12.. of `b` and masks them off at
/// the palette load, so the result is taken `& 0xFFF`.
pub fn mask_rank(b: u32, words: &[u32]) -> u32 {
    let w = words.get((b >> 4 & 0xFF) as usize).copied().unwrap_or(0);
    let bit = 1u32 << (b & 15);
    if w & 0xFFFF & bit == 0 {
        return 0;
    }
    ((w >> 16) + (w & 0xFFFF & (bit - 1)).count_ones() + 1) & 0xFFF
}

/// What a mesh program does with bones, from its name
/// (`shader-names.tsv`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgClass {
    /// Ped* and *PreSkin programs.
    Ped,
    /// MSR_*: one bone per vertex or per mesh, no weights; mask at rec+0x74.
    OneBone,
    /// Other *Skin* programs: four indices and weights; mask at rec+0x74.
    Blend,
    /// Reads no palette.
    Rigid,
}

pub fn prog_class(name: &str) -> ProgClass {
    if name.contains("PedTransform") || name.contains("PreSkin") {
        ProgClass::Ped
    } else if name.starts_with("MSR_") {
        ProgClass::OneBone
    } else if name.contains("Skin") {
        ProgClass::Blend
    } else {
        ProgClass::Rigid
    }
}

/// `shader-names.tsv` (container hash, name, ..) from the nearest
/// ancestor of the frame directory that has one (`out` for captures
/// under it).
pub(crate) fn program_names(dir: &std::path::Path) -> Option<HashMap<u128, String>> {
    let text = dir.ancestors().skip(1).take(6).find_map(|d| std::fs::read_to_string(d.join("shader-names.tsv")).ok())?;
    Some(text.lines().filter_map(|l| {
        let mut f = l.split('\t');
        Some((u128::from_str_radix(f.next()?, 16).ok()?, f.next()?.to_string()))
    }).collect())
}

/// The MSR one-bone skin of a mesh as ordinary streams appended to
/// its vertex buffer: one u32 index per vertex (stream word 0x20: stride 4,
/// format 12) and the weight 255,0,0,0 (word 0x18: stride 4, format 6).
/// `word0` is the meshlet blob's first dword: bits 29 and 30 both set: one
/// bone for the whole mesh, bits 17-28; else an array at the old word 0x20's
/// offset with elements of `1 << bits 17-21` bytes (0: u8, 1: u16; a4a1cd89
/// 0x208-0x2c0). False (buffer untouched): no header, streams already there,
/// or an element size the program has no load for.
pub fn one_bone_streams(vb: &mut Vec<u8>, word0: u32, verts: u32) -> bool {
    let Some(s) = Streams::parse(vb) else { return false };
    if s.weights[0].stride != 0 || s.indices[0].stride != 0 {
        return false;
    }
    let field = word0 >> 17 & 0xFFF;
    let whole = word0 >> 29 & 3 == 3;
    let shift = field & 31;
    if !whole && shift > 1 {
        return false;
    }
    let at = s.indices[0].offset as usize;
    let bone = |v: u32| -> u32 {
        if whole {
            return field;
        }
        let a = at + ((v as usize) << shift);
        match shift {
            0 => vb.get(a).map_or(0, |b| u32::from(*b)),
            _ => vb.get(a..a + 2).map_or(0, |b| u32::from(u16::from_le_bytes([b[0], b[1]]))),
        }
    };
    let bones: Vec<u32> = (0..verts).map(bone).collect();
    vb.resize(vb.len().div_ceil(4) * 4, 0);
    let (io, wo) = (vb.len() as u32, vb.len() as u32 + 4 * verts);
    if u64::from(wo) + 4 * u64::from(verts) > 0x1FF_FFFC {
        return false;
    }
    for b in bones {
        vb.extend_from_slice(&b.to_le_bytes());
    }
    for _ in 0..verts {
        vb.extend_from_slice(&[255, 0, 0, 0]);
    }
    vb[0x20..0x24].copy_from_slice(&(io << 7 | 4 << 3 | 12).to_le_bytes());
    vb[0x18..0x1C].copy_from_slice(&(wo << 7 | 4 << 3 | 6).to_le_bytes());
    true
}

/// The vehicle rules of `rules` (bits 4-7) applied to a loaded list
/// from its v2 pack: per skinned row with one sink, the mesh program's
/// class, the slot-0 record (+0x54, +0x74), the arena words and the meshlet
/// blob's word 0. The index maths of bits 6 and 7 becomes a 4096-entry table
/// after the row's palette (the pass's `remap`: `b = table[b] & 0xFFF`).
/// Call after [`hook_palettes_arg`] and before `list_options`. Prints one line.
pub fn vehicle_skin(dir: &std::path::Path, list: &mut FrameList, geoms: &mut HashMap<GeomKey, (Vec<u8>, Vec<u32>)>, rules: u32) -> Result<(), String> {
    use xnw_formats::framepack::kind;
    if rules & (SKIN_PROGRAM_CLASS | SKIN_ONE_BONE | SKIN_REC_OFFSET | SKIN_MASK_RANK) == 0 {
        return Ok(());
    }
    let Some(pack) = crate::pack::Pack::open(dir)? else {
        println!("vehicle skin: no pack: rules {rules:#x} bits 4-7 not applied");
        return Ok(());
    };
    let Some(names) = program_names(dir) else {
        println!("vehicle skin: no shader-names.tsv above {}: rules {rules:#x} bits 4-7 not applied", dir.display());
        return Ok(());
    };
    let v1: Vec<Option<u32>> = match pack.list_draws().filter(|v| v.len() == list.draws.len()) {
        Some(v) => v.into_iter().map(Some).collect(),
        None => list_v1_draws(dir, list),
    };
    let u = |b: &[u8], at: usize| b.get(at..at + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()));
    let (mut class_n, mut no_sink, mut unnamed) = ([0usize; 4], 0usize, 0usize);
    let (mut ped_off, mut rigid_off, mut mask_rows, mut no_words, mut no_rec) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut forms, mut no_blob) = ([0usize; 3], 0usize);
    let mut patched: HashMap<GeomKey, bool> = HashMap::new();
    let mut tables: HashMap<(u32, u32, u128), u32> = HashMap::new();
    for i in 0..list.draws.len() {
        let r = list.draws[i];
        if r.palette == NO_PALETTE {
            continue;
        }
        let Some(sink) = v1[i].and_then(|d| pack.sink_of(d)) else {
            no_sink += 1;
            continue;
        };
        let pipeline = pack.data.sinks.iter().find(|s| s.index == sink).map_or(0, |s| s.pipeline);
        let Some(name) = pack.data.programs.iter().find(|p| p.pipeline == pipeline && p.stage == "mesh").and_then(|p| names.get(&p.container)) else {
            unnamed += 1;
            continue;
        };
        let class = prog_class(name);
        class_n[class as usize] += 1;
        if rules & SKIN_PROGRAM_CLASS != 0 {
            let ped = class == ProgClass::Ped || r.remap != NO_REMAP;
            ped_off += usize::from(r.ped && !ped);
            list.draws[i].ped = ped;
            if class == ProgClass::Rigid && r.remap == NO_REMAP {
                rigid_off += 1;
                (list.draws[i].palette, list.draws[i].bone_first, list.draws[i].bone_count) = (NO_PALETTE, 0, 0);
                continue;
            }
        }
        if class != ProgClass::OneBone && class != ProgClass::Blend {
            continue;
        }
        if class == ProgClass::OneBone && rules & SKIN_ONE_BONE != 0 {
            let done = *patched.entry(r.geom).or_insert_with(|| {
                let blob = pack.data.refs_of(sink).find(|x| x.kind == kind::BUFFER && x.addr as u32 == r.geom.mesh_at as u32 && u64::from(x.len) == r.geom.mesh_len as u64).and_then(|x| pack.blob(x.hash));
                let (Some(w0), Some((vb, idx))) = (blob.as_deref().and_then(|b| u(b, 0)), geoms.get_mut(&r.geom)) else { return false };
                let verts = idx.iter().max().map_or(0, |m| m + 1);
                let ok = one_bone_streams(vb, w0, verts);
                if ok {
                    forms[if w0 >> 29 & 3 == 3 { 0 } else { 1 + (w0 >> 17 & 1) as usize }] += 1;
                }
                ok
            });
            no_blob += usize::from(!done);
        }
        if rules & (SKIN_REC_OFFSET | SKIN_MASK_RANK) == 0 {
            continue;
        }
        let Ok((_, rec)) = pack.slot_bytes(sink, 1, 0) else {
            no_rec += 1;
            continue;
        };
        let (Some(off), Some(at)) = (u(&rec, 0x54), u(&rec, 0x74)) else {
            no_rec += 1;
            continue;
        };
        if at == u32::MAX {
            continue;
        }
        mask_rows += 1;
        let arena = pack.data.refs_of(sink).find(|x| x.kind == kind::ARENA).and_then(|x| Some((x.hash, pack.blob(x.hash)?)));
        let Some((hash, bytes)) = arena else {
            no_words += 1;
            continue;
        };
        let p = r.palette as usize;
        let at = *tables.entry((r.palette, off, hash)).or_insert_with(|| {
            let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            let start = list.palettes[p].len() as u32;
            list.palettes[p].extend((0..4096u32).map(|b| {
                let b = if rules & SKIN_REC_OFFSET != 0 { b.wrapping_add(off) } else { b };
                f32::from_bits(if rules & SKIN_MASK_RANK != 0 { mask_rank(b, &words) } else { b & 0xFFF })
            }));
            start
        });
        list.draws[i].remap = at;
    }
    println!(
        "vehicle skin (rules {rules:#x}): skinned rows by program: ped {}, one-bone {}, blend {}, rigid {}; no 1:1 sink {no_sink}, unnamed {unnamed}; ped flag dropped {ped_off}, palette dropped (rigid program) {rigid_off}; one-bone geometries: whole-mesh {}, u8 array {}, u16 array {}, rows not decoded {no_blob}; mask on {mask_rows} rows ({} tables, {no_words} without words, {no_rec} without a record)",
        class_n[0], class_n[1], class_n[2], class_n[3], forms[0], forms[1], forms[2], tables.len()
    );
    Ok(())
}

/// `skin_rules` bit 8: the instanced skin programs (vehicle wheels and
/// tyres, `*_inst`) draw `groups / meshlets` instances, each with its own
/// 128-B record (world) and palette base; [`vehicle_instances`] gives the
/// list row those instances and the bones at the base.
pub const SKIN_INSTANCES: u32 = 0x100;

/// An instanced skin program's T9 table slots (byte offset / 32), read from
/// its code: meshlet info (dword 0 & 0x1FFFF = meshlets per instance),
/// per-instance records (128 B), palette bases (u32 per instance, -1: the
/// slot-0 record's +0x30), the view constants (camera at +0xd0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstSlots {
    pub meshlets: u32,
    pub records: u32,
    pub bases: u32,
    pub view: u32,
}

/// The main-view instanced skin programs by the top 32 bits of the
/// container hash: MS_VS_TransformSkinD_NoDeform eddda273 (wheel: info
/// T9+0x80, records T9+0xa0 at 0x878, bases T9+0x100 at 0xb6c, camera
/// T9+0x180+0xd0 at 0x844) and f9dd5454 (tyre: T9+0xc0, T9+0xe0 at 0x858,
/// T9+0x120 at 0xb38, T9+0x1a0+0xd0 at 0x848). The palette is T8+0xa0
/// (root 0 slot 5) in both, index base + bone.
pub const INST_PROGRAMS: [(u32, InstSlots); 2] = [
    (0xeddd_a273, InstSlots { meshlets: 4, records: 5, bases: 8, view: 12 }),
    (0xf9dd_5454, InstSlots { meshlets: 6, records: 7, bases: 9, view: 13 }),
];

/// The camera position the instanced programs subtract: floats at view
/// +0xd0 (x, y, z) plus 32 x the two i16 at +0xdc (coarse x, y); z has no
/// coarse part (f9dd5454 0x874-0x910).
pub fn inst_camera(view: &[u8]) -> Option<[f64; 3]> {
    let f = |o: usize| view.get(o..o + 4).map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap())));
    let w = view.get(0xdc..0xe0).map(|b| u32::from_le_bytes(b.try_into().unwrap()))?;
    Some([f(0xd0)? + 32.0 * f64::from(w as u16 as i16), f(0xd4)? + 32.0 * f64::from((w >> 16) as u16 as i16), f(0xd8)?])
}

/// One instance's camera-relative world as the list's 12 floats (x, y, z
/// axes, translation) from its record: the program computes
/// `out = d0..2 * x + d4..6 * y + d8..10 * z + (d3, d7, d11) - camera`
/// (f9dd5454 0x103c-0x1068).
pub fn inst_world(rec: &[u8], cam: [f64; 3]) -> Option<[f32; 12]> {
    let d = |i: usize| rec.get(4 * i..4 * i + 4).map(|b| f32::from_le_bytes(b.try_into().unwrap()));
    let mut w = [0f32; 12];
    for (k, i) in [0, 1, 2, 4, 5, 6, 8, 9, 10].into_iter().enumerate() {
        w[k] = d(i)?;
    }
    for (k, i) in [3, 7, 11].into_iter().enumerate() {
        w[9 + k] = (f64::from(d(i)?) - cam[k]) as f32;
    }
    w.iter().all(|v| v.is_finite()).then_some(w)
}

/// Bones taken from the palette buffer per instanced row (u8 indices).
const INST_BONES: u32 = 256;

/// `skin_rules` bit 8 over a loaded list (after [`vehicle_skin`]): per
/// skinned row drawn directly by a program of [`INST_PROGRAMS`], the
/// `groups / meshlets` instances become the row's instances (world
/// identity), and the bones at the palette base are appended to the row's
/// palette and reached through a remap table, so the row keeps its palette
/// window (one tile with its body). Rows whose instances have different
/// bases or a mask (`rec+0x74 != -1`) are counted and left as they were.
/// Prints one line.
pub fn vehicle_instances(dir: &std::path::Path, list: &mut FrameList, rules: u32) -> Result<(), String> {
    use xnw_formats::framepack::op;
    if rules & SKIN_INSTANCES == 0 {
        return Ok(());
    }
    let Some(pack) = crate::pack::Pack::open(dir)? else {
        println!("vehicle instances: no pack: rule 0x100 not applied");
        return Ok(());
    };
    let v1: Vec<Option<u32>> = match pack.list_draws().filter(|v| v.len() == list.draws.len()) {
        Some(v) => v.into_iter().map(Some).collect(),
        None => list_v1_draws(dir, list),
    };
    let u = |b: &[u8], at: usize| b.get(at..at + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()));
    let (mut rows, mut placed, mut not_direct, mut short, mut mixed, mut masked, mut far, mut from_window) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut palettes: HashMap<u128, Vec<u8>> = HashMap::new();
    let mut detail: Vec<String> = Vec::new();
    // Rows whose live reads agree, differ, were refused.
    let mut live = [0usize; 3];
    for i in 0..list.draws.len() {
        let r = list.draws[i];
        if r.palette == NO_PALETTE || r.inst_count != 0 {
            continue;
        }
        // The draw's sink drawn by an instanced program (a draw may have
        // more sinks: the tyre draw is followed by two compute sinks).
        let inst_slots = |s: &&xnw_formats::framepack::Sink| pack.data.programs.iter().find(|p| p.pipeline == s.pipeline && p.stage == "mesh").and_then(|p| INST_PROGRAMS.iter().find(|t| t.0 == (p.container >> 96) as u32)).map(|t| t.1);
        let Some(d) = v1[i] else { continue };
        let Some((sink, slots)) = pack.data.sinks.iter().filter(|s| s.hook_draw == Some(d)).find_map(|s| inst_slots(&s).map(|t| (s, t))) else { continue };
        rows += 1;
        if sink.op != op::MESH {
            not_direct += 1;
            continue;
        }
        let slot = |root: u16, s: u32| pack.slot_bytes(sink.index, root, s).ok().map(|x| x.1);
        let (Some(info), Some(recs), Some(bases), Some(view), Some(rec0)) = (slot(1, slots.meshlets), slot(1, slots.records), slot(1, slots.bases), slot(1, slots.view), slot(1, 0)) else {
            short += 1;
            continue;
        };
        let per = u(&info, 0).map_or(0, |w| w & 0x1FFFF);
        let n = if per == 0 { 0 } else { sink.count / per };
        let cam = inst_camera(&view);
        let worlds: Option<Vec<[f32; 12]>> = cam.and_then(|c| (0..n as usize).map(|k| recs.get(128 * k..128 * k + 128).and_then(|b| inst_world(b, c))).collect());
        let base: Option<Vec<u32>> = (0..n as usize).map(|k| u(&bases, 4 * k).map(|b| if b == u32::MAX { u(&rec0, 0x30).unwrap_or(0) } else { b })).collect();
        let (Some(worlds), Some(base)) = (worlds.filter(|w| !w.is_empty()), base) else {
            short += 1;
            continue;
        };
        if base.iter().any(|b| *b != base[0]) {
            mixed += 1;
            continue;
        }
        if u(&rec0, 0x74).is_some_and(|m| m != u32::MAX) {
            masked += 1;
            continue;
        }
        // The bones at the base from the palette buffer (T8+0xa0: root 0 slot 5).
        let Some(hash) = pack.data.refs_of(sink.index).find(|x| x.kind == xnw_formats::framepack::kind::BUFFER && x.slot == 5).map(|x| x.hash) else {
            short += 1;
            continue;
        };
        if !palettes.contains_key(&hash) {
            palettes.insert(hash, pack.slot_bytes(sink.index, 0, 5).map(|x| x.1).unwrap_or_default());
        }
        // The bones at the base: from the slot's bytes, else (the slot cut
        // short of the base) from the draw's own palette window (v2.3).
        let at = 48 * base[0] as usize;
        let window;
        let bones = match palettes[&hash].get(at..at + 48 * INST_BONES as usize) {
            Some(b) => b,
            None => {
                window = pack.palette_windows(sink.index).into_iter().find(|w| w.0 == base[0]).and_then(|w| w.1.ok()).filter(|b| b.len() >= 48 * INST_BONES as usize);
                match &window {
                    Some(b) => {
                        from_window += 1;
                        &b[..48 * INST_BONES as usize]
                    }
                    None => {
                        short += 1;
                        continue;
                    }
                }
            }
        };
        // The bones go at bone index e of the palette; the remap entry for
        // bone b is its index relative to the row's bone_first. Index tables
        // appended before may leave the length off a whole bone: pad first.
        let p = r.palette as usize;
        let len = list.palettes[p].len().next_multiple_of(12);
        list.palettes[p].resize(len, 0.0);
        let e = (len / 12) as u32;
        let Some(rel) = e.checked_sub(r.bone_first).filter(|x| x + INST_BONES <= 0x1000) else {
            far += 1;
            continue;
        };
        list.palettes[p].extend(bones.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())));
        let remap = list.palettes[p].len() as u32;
        list.palettes[p].extend((0..INST_BONES).map(|b| f32::from_bits(rel + b)));
        let first = (list.instances.len() / 12) as u32;
        for w in &worlds {
            list.instances.extend_from_slice(w);
        }
        let d = &mut list.draws[i];
        (d.inst_first, d.inst_count, d.remap) = (first, worlds.len() as u32, remap);
        d.world = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        placed += 1;
        // The live pass's reads (`wheel_reads`) over the capture's bytes of the
        // same views: the same instances and bones, or a reason.
        let mut views: HashMap<(u16, u32), (u64, u64)> = HashMap::new();
        let mut regions: Vec<(u64, Vec<u8>)> = Vec::new();
        for (root, s) in [(1u16, slots.meshlets), (1, slots.records), (1, slots.bases), (1, slots.view), (1, 0), (0, WHEEL_PALETTE)] {
            if let Ok((v, b)) = pack.slot_bytes(sink.index, root, s) {
                views.insert((root, s), (v.addr, v.bytes));
                regions.push((v.addr, b));
            }
        }
        if let Some(&(pa, _)) = views.get(&(0, WHEEL_PALETTE)) {
            regions.extend(pack.palette_windows(sink.index).into_iter().filter_map(|(b, w)| w.ok().map(|w| (pa + 48 * u64::from(b), w))));
        }
        let mut read = |at: u64, len: usize| regions.iter().rev().find_map(|(a, b)| at.checked_sub(*a).and_then(|o| b.get(o as usize..o as usize + len)).map(<[u8]>::to_vec));
        match wheel_reads(slots, sink.count, &|r, s| views.get(&(r, s)).copied(), &mut read) {
            Ok(w) if w.worlds == worlds && w.bones.iter().zip(bones.chunks_exact(4)).all(|(a, b)| a.to_bits() == u32::from_le_bytes(b.try_into().unwrap())) => live[0] += 1,
            Ok(_) => live[1] += 1,
            Err(_) => live[2] += 1,
        }
        if detail.len() < 16 {
            detail.push(format!("row {i} draw {} sink {} instances {} base {}", v1[i].map_or(-1, i64::from), sink.index, worlds.len(), base[0]));
        }
    }
    if !detail.is_empty() {
        println!("vehicle instances placed: {}", detail.join("; "));
    }
    // Instanced-program sinks no skinned row reached, and why.
    let mut rows_of: HashMap<u32, usize> = HashMap::new();
    for (i, d) in v1.iter().enumerate() {
        if let Some(d) = d {
            rows_of.insert(*d, i);
        }
    }
    let missed: Vec<String> = pack
        .data
        .sinks
        .iter()
        .filter(|s| !pack.palette_windows(s.index).is_empty() || pack.data.programs.iter().any(|p| p.pipeline == s.pipeline && p.stage == "mesh" && INST_PROGRAMS.iter().any(|t| t.0 == (p.container >> 96) as u32)))
        .filter_map(|s| {
            let why = match s.hook_draw.map(|d| (d, rows_of.get(&d))) {
                None => "no hook draw".to_string(),
                Some((d, None)) => format!("draw {d} has no list row"),
                Some((d, Some(&i))) if list.draws[i].palette == NO_PALETTE => format!("draw {d} row {i} not skinned"),
                Some(_) => return None,
            };
            Some(format!("sink {} ({})", s.index, why))
        })
        .take(16)
        .collect();
    if !missed.is_empty() {
        println!("vehicle instances: instanced sinks without a skinned row: {}", missed.join("; "));
    }
    println!("vehicle instances (rules {rules:#x}): instanced skin rows {rows}, placed {placed} ({from_window} from palette windows); not direct {not_direct}, slots or bytes missing {short}, bases differ {mixed}, mask on {masked}, palette too far {far}; live reads agree {}, differ {}, refused {}", live[0], live[1], live[2]);
    Ok(())
}

/// The live pass's reads of an instanced draw (`skinrules` bit 8), shared
/// with the replay, which checks them against [`vehicle_instances`].
/// Instances read per draw, at most; bones read at the palette base.
pub const WHEEL_MAX: usize = 64;
pub const WHEEL_BONES: usize = 256;
/// The frame table's (root 0) palette slot, 48 B a bone.
const WHEEL_PALETTE: u32 = 5;

/// One instanced draw's reads: per instance its camera-relative world (the
/// list's 12 floats), and the bones at the draw's palette base.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WheelInst {
    pub worlds: Vec<[f32; 12]>,
    pub bones: Vec<f32>,
}

/// The reads of one instanced draw of `groups` mesh groups: `view(root,
/// slot)` the slot's (address, bytes), `read(address, len)` a bounded,
/// fault-fixed copy. Err: the reason, 0 slot views missing, 1 no
/// instances, 2 bases differ, 3 mask on, 4 unreadable.
pub fn wheel_reads(slots: InstSlots, groups: u32, view: &dyn Fn(u16, u32) -> Option<(u64, u64)>, read: &mut dyn FnMut(u64, usize) -> Option<Vec<u8>>) -> Result<WheelInst, usize> {
    let u = |b: &[u8], at: usize| b.get(at..at + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()));
    let (Some(info), Some(recs), Some(bases), Some(cam), Some(rec0), Some(pal)) = (view(1, slots.meshlets), view(1, slots.records), view(1, slots.bases), view(1, slots.view), view(1, 0), view(0, WHEEL_PALETTE)) else {
        return Err(0);
    };
    let per = read(info.0, 4).and_then(|b| u(&b, 0)).ok_or(4usize)? & 0x1FFFF;
    let n = if per == 0 { 0 } else { (groups / per) as usize }.min(WHEEL_MAX).min((recs.1 / 128) as usize).min((bases.1 / 4) as usize);
    if n == 0 {
        return Err(1);
    }
    let rec0 = read(rec0.0, 0x78).ok_or(4usize)?;
    if u(&rec0, 0x74).is_some_and(|m| m != u32::MAX) {
        return Err(3);
    }
    let b = read(bases.0, 4 * n).ok_or(4usize)?;
    let base: Vec<u32> = (0..n).map(|k| u(&b, 4 * k).map_or(0, |v| if v == u32::MAX { u(&rec0, 0x30).unwrap_or(0) } else { v })).collect();
    if base.iter().any(|v| *v != base[0]) {
        return Err(2);
    }
    let c = read(cam.0, 0xe0).ok_or(4usize)?;
    let c = inst_camera(&c).ok_or(4usize)?;
    let r = read(recs.0, 128 * n).ok_or(4usize)?;
    let worlds: Vec<[f32; 12]> = (0..n).map(|k| inst_world(&r[128 * k..128 * k + 128], c)).collect::<Option<_>>().ok_or(4usize)?;
    let off = 48 * u64::from(base[0]);
    if off + 48 * WHEEL_BONES as u64 > pal.1 {
        return Err(4);
    }
    let bones = read(pal.0 + off, 48 * WHEEL_BONES).ok_or(4usize)?;
    Ok(WheelInst { worlds, bones: bones.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() })
}

/// The replay's skin sources for a loaded list: draw-time bones
/// ([`hook_palettes_arg`]), then the vehicle rules of `--skin-rules`.
pub fn skin_args(args: &[String], dir: &std::path::Path, list: &mut FrameList, geoms: &mut HashMap<GeomKey, (Vec<u8>, Vec<u32>)>) -> Result<usize, String> {
    let n = hook_palettes_arg(args, dir, list)?;
    let rules = crate::replay::skin_rules_arg(args);
    vehicle_skin(dir, list, geoms, rules)?;
    vehicle_instances(dir, list, rules)?;
    Ok(n)
}

#[cfg(test)]
mod geo10_tests {
    use super::*;

    /// The compaction formula on made-up words: word k = mask | rank << 16, rank the
    /// number of set bits in the words before it.
    #[test]
    fn mask_rank_formula() {
        // Bones 0-15: bits 0, 2, 3 set; bones 16-31: bits 1 and 15; rank 3.
        let words = [0b1101u32, (1 << 1 | 1 << 15) | 3 << 16];
        assert_eq!(mask_rank(0, &words), 1);
        assert_eq!(mask_rank(1, &words), 0, "hidden: the zero entry");
        assert_eq!(mask_rank(2, &words), 2);
        assert_eq!(mask_rank(3, &words), 3);
        assert_eq!(mask_rank(4, &words), 0);
        assert_eq!(mask_rank(16, &words), 0);
        assert_eq!(mask_rank(17, &words), 4);
        assert_eq!(mask_rank(31, &words), 5);
        assert_eq!(mask_rank(32, &words), 0, "past the words");
        // Bits 12.. of the index pick no other word and do not survive.
        assert_eq!(mask_rank(0x1000 | 17, &words), 4);
        // One captured frame's first words (2a17|0, ff90|7): bone 4 is the 4th visible one.
        let pk3 = [0x2a17u32, 0xff90 | 7 << 16];
        assert_eq!((mask_rank(4, &pk3), mask_rank(3, &pk3), mask_rank(20, &pk3)), (4, 0, 8));
    }

    /// A made-up view block and instance record: camera 1832, 143, 3 as fine
    /// 8, 15, 3 plus coarse 57, 4 (x 32); a wheel at 1834.5, 141, 2 with a
    /// half-size 90-degree yaw.
    #[test]
    fn instance_world_and_camera() {
        let mut view = vec![0u8; 0xe0];
        for (k, v) in [8.0f32, 15.0, 3.0].into_iter().enumerate() {
            view[0xd0 + 4 * k..0xd4 + 4 * k].copy_from_slice(&v.to_le_bytes());
        }
        view[0xdc..0xe0].copy_from_slice(&(57u32 | 4 << 16).to_le_bytes());
        assert_eq!(inst_camera(&view), Some([1832.0, 143.0, 3.0]));
        // Negative coarse parts are signed.
        view[0xdc..0xe0].copy_from_slice(&(0xFFFFu32 | 0xFFFE << 16).to_le_bytes());
        assert_eq!(inst_camera(&view), Some([8.0 - 32.0, 15.0 - 64.0, 3.0]));
        assert_eq!(inst_camera(&view[..0xdc]), None);
        // d0..2 = image of x, d4..6 = image of y, d8..10 = image of z, translation in d3, d7, d11.
        let d: [f32; 12] = [0.0, -0.5, 0.0, 1834.5, 0.5, 0.0, 0.0, 141.0, 0.0, 0.0, 0.5, 2.0];
        let rec: Vec<u8> = d.iter().flat_map(|v| v.to_le_bytes()).chain([0u8; 80]).collect();
        let w = inst_world(&rec, [1832.0, 143.0, 3.0]).unwrap();
        assert_eq!(w, [0.0, -0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 2.5, -2.0, -1.0]);
        // As the pass applies it (p.x * ax + p.y * ay + p.z * az + tr): local x 1 -> 0.5 to the right of -y.
        let p = [1.0f32, 0.0, 0.0];
        let o: Vec<f32> = (0..3).map(|k| p[0] * w[k] + p[1] * w[3 + k] + p[2] * w[6 + k] + w[9 + k]).collect();
        assert_eq!(o, vec![2.5, -2.5, -1.0]);
        assert_eq!(inst_world(&rec[..44], [0.0; 3]), None, "short record");
        assert_eq!(INST_PROGRAMS.iter().find(|t| t.0 == (0xf9dd_5454_30b0_753b_962c_87bc_9cb5_3c76u128 >> 96) as u32).map(|t| t.1.records), Some(7));
    }

    #[test]
    fn program_classes() {
        assert_eq!(prog_class("MSR_VS_TransformSkin_dc8b9b7f_Wrapped"), ProgClass::OneBone);
        assert_eq!(prog_class("MS_VS_TransformSkinD_NoDeform_34f92402_Wrapped"), ProgClass::Blend);
        assert_eq!(prog_class("MS_VS_PedTransformPreSkinD_6a40bffd_Wrapped"), ProgClass::Ped);
        assert_eq!(prog_class("MSNC_VS_PedTransformSkinAlpha_af24a1cd_Wrapped"), ProgClass::Ped);
        assert_eq!(prog_class("MS_VS_LinearDepthPreSkin_4f9e4bb2_Wrapped"), ProgClass::Ped);
        assert_eq!(prog_class("MS_VS_TransformD_c5fd36bb_Wrapped"), ProgClass::Rigid);
    }

    #[test]
    fn one_bone_streams_forms() {
        let head = |idx_word: u32, tail: &[u8]| {
            let mut vb = vec![0u8; 0x80];
            vb[0x20..0x24].copy_from_slice(&idx_word.to_le_bytes());
            vb.extend_from_slice(tail);
            vb
        };
        let terms = |vb: &[u8], v: u32| {
            let s = Streams::parse(vb).unwrap();
            let (t, n) = skin_terms(&s, vb, v, &[], None, false, 0);
            t[..n].to_vec()
        };
        // A u8 array at 0x80 (word: offset only, stride 0), three vertices.
        let mut vb = head(0x80 << 7, &[7, 0, 9]);
        assert!(one_bone_streams(&mut vb, 2 << 29 | 5, 3));
        assert_eq!((terms(&vb, 0), terms(&vb, 1), terms(&vb, 2)), (vec![(7, 1.0)], vec![(0, 1.0)], vec![(9, 1.0)]));
        // A u16 array (field 1).
        let mut vb = head(0x80 << 7, &[0x34, 0x01, 0x02, 0x00]);
        assert!(one_bone_streams(&mut vb, 2 << 29 | 1 << 17 | 5, 2));
        assert_eq!((terms(&vb, 0), terms(&vb, 1)), (vec![(0x134, 1.0)], vec![(2, 1.0)]));
        // One bone for the whole mesh (bits 29 and 30): 170, no array.
        let mut vb = head(0, &[]);
        assert!(one_bone_streams(&mut vb, 0xe154_0001, 2));
        assert_eq!((terms(&vb, 0), terms(&vb, 1)), (vec![(170, 1.0)], vec![(170, 1.0)]));
        // Streams already there: untouched.
        let before = vb.clone();
        assert!(!one_bone_streams(&mut vb, 0xe154_0001, 2));
        assert_eq!(vb, before);
    }
}
