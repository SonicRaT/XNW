//! Imposters: far trees and props the title draws as flat octahedral cards,
//! drawn here as one camera-facing card per instance.
//!
//! The title submits a batch per imposter type (0x1451D0AC0: ecx type, edx
//! stage, IcRec) and lets a compute pass (`CS_*_BatchCull_ImposterOnly`)
//! pick the instances and write the draw arguments; with the title's GPU
//! work nulled nothing is drawn [V 0x14013EFB0 reached, I nothing drawn].
//! The CPU holds the inputs of that choice:
//!
//! | where | what |
//! |---|---|
//! | desc+0x08 | n, the batch's instances |
//! | desc+0x10 / desc+0x20 | local box min / max (xyz) |
//! | desc+0x30 | local sphere (centre xyz, radius) |
//! | T+0x48 | LOD distances D[4] (< 0 absent) |
//! | T+0x58 | far fade F0, F1 |
//! | 64-B records | rows r0..r2 (rK.xyz the image of model axis K, translation in .w, absolute world); +0x30/+0x38 > 0 or dropped |
//!
//! Rule (the LOD rule of the instanced props, applied to the batch's own
//! table): an instance is a card when its distance d to the camera is at
//! least the last present D (and the type record's start R+0x58 [I]) and
//! below F1; it fades out from F0 to F1. The card stands upright on the
//! world z axis, turned to face the camera, as wide as the record's box seen
//! from the camera (the box's axes projected on the card's horizontal) and
//! as tall as the box's vertical extent. The title's own card is a square of
//! half size R+0x4C around R+0x40 whose atlas alpha draws the silhouette; we
//! have no atlas, so the box gives the outline. Flat-shaded: a crown over a
//! stem for types taller than wide, a round blob for near-cubic ones, a
//! rounded block for low wide ones (hedges, rooftop boxes); lit by the sun
//! and fogged as the rest of the pass.

use windows::Win32::Graphics::Direct3D12::*;

use super::{Camera, FrameList, RenderState};

/// Bytes of one instance record.
pub const RECORD_BYTES: usize = 64;
/// Cards per frame at most.
pub const MAX_CARDS: usize = 65_536;
/// Distance histogram: 50 m bins, the last one open.
pub const BIN_METRES: f32 = 50.0;
pub const BINS: usize = 21;

/// One imposter submit: the batch's table and its records.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Batch {
    pub kind: u32,
    pub stage: u32,
    pub n: u32,
    pub box_min: [f32; 3],
    pub box_max: [f32; 3],
    pub sphere: [f32; 4],
    pub lods: [f32; 4],
    pub fade: [f32; 2],
    /// 16 floats per instance (the record as read).
    pub records: Vec<[f32; 16]>,
    /// Where cards start (m): the type record's R+0x58 (90, 75, 120 ... in
    /// pk5) [I: read as the mesh-to-card distance]; 0 without a type record.
    pub start: f32,
}

/// The frame's imposter submits.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Imposters {
    pub batches: Vec<Batch>,
    /// Submit calls met (kept or skipped).
    pub calls: u32,
    /// Calls not kept, by reason (`SKIP_NAMES`).
    pub skipped: [u32; SKIPS],
}

/// Why a call has no batch: its IcRec or desc did not read, n > 0 without
/// records, past `MAX_CALLS` in one frame.
pub const SKIPS: usize = 3;
pub const SKIP_NAMES: [&str; SKIPS] = ["unreadable", "without records", "over the cap"];
pub const SKIP_UNREADABLE: usize = 0;
pub const SKIP_NO_RECORDS: usize = 1;
pub const SKIP_CAP: usize = 2;
/// Calls kept per frame at most (pk5: 2-51).
pub const MAX_CALLS: u32 = 1024;

impl Imposters {
    /// `skipped S (r1 a, r2 b)`, reasons with a count only.
    pub fn skipped_text(&self) -> String {
        let total: u32 = self.skipped.iter().sum();
        let parts: Vec<String> = self.skipped.iter().zip(SKIP_NAMES).filter(|(n, _)| **n > 0).map(|(n, w)| format!("{w} {n}")).collect();
        if parts.is_empty() { format!("skipped {total}") } else { format!("skipped {total} ({})", parts.join(", ")) }
    }
}

/// One card: absolute centre (the box centre), half width and height, fade
/// (1 opaque .. 0 gone), its batch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Card {
    pub centre: [f64; 3],
    pub half_width: f32,
    pub half_height: f32,
    /// `SHAPE_TREE`, `SHAPE_BLOB` or `SHAPE_BLOCK` (from the local box).
    pub shape: u32,
    pub fade: f32,
    pub distance: f32,
    pub batch: u32,
    /// The record's index in its batch.
    pub index: u32,
    /// The record's translation (absolute).
    pub root: [f64; 3],
}

/// What the selection found.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImposterStats {
    pub batches: usize,
    pub candidates: usize,
    /// Batches whose records were not captured (n > 0, no records).
    pub without_records: usize,
    /// Records with +0x30/+0x38 <= 0 or a non-finite translation.
    pub dropped: usize,
    /// Nearer than the last LOD distance (a mesh draws them).
    pub near: usize,
    /// At or past F1.
    pub far: usize,
    /// At the translation of an instance the list draws as a mesh.
    pub meshed: usize,
    pub cards: usize,
    /// Batches with a start distance from their type record.
    pub with_start: usize,
    pub per_bin: [u32; BINS],
    pub drawn: usize,
    /// Drawn cards with a corner in view ([`in_view`]).
    pub in_view: usize,
    /// Of `near`: in view with no mesh of the list at their translation
    /// ([`near_unmeshed`]): a card withheld where no mesh stands, or an
    /// instance the title culls (its mesh-vs-card choice is on the GPU).
    pub near_unmeshed: usize,
    pub record_ms: f64,
}

/// `--imposters` (replay, lab) / `imposters=on` (screen mode): off by default.
/// `show` (`--imposters-show`, `imposters=show`): the cards in magenta
/// without the depth test, to see where they stand (a diagnostic).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImposterOptions {
    pub on: bool,
    pub show: bool,
}

pub fn parse_screen(v: &str) -> ImposterOptions {
    let last = v.split(',').map(str::trim).filter_map(|p| p.strip_prefix("imposters=")).last();
    ImposterOptions { on: matches!(last, Some("on" | "show")), show: last == Some("show") }
}

/// The card's corners as the vertex shader places them (`rel` its centre
/// minus the camera), in view of `vp` (row vectors): some corner in front
/// of the camera and the corners' screen box over the screen.
pub fn in_view(c: &Card, rel: [f32; 3], vp: &super::Mat) -> bool {
    let h = rel[0] * rel[0] + rel[1] * rel[1];
    let right = if h > 1e-6 { [-rel[1] / h.sqrt(), rel[0] / h.sqrt()] } else { [1.0, 0.0] };
    let (mut lo, mut hi, mut front) = ([f32::MAX; 2], [f32::MIN; 2], false);
    for (x, y) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        let p = [rel[0] + x * c.half_width * right[0], rel[1] + x * c.half_width * right[1], rel[2] + y * c.half_height, 1.0];
        let clip: [f32; 4] = std::array::from_fn(|j| (0..4).map(|i| p[i] * vp[4 * i + j]).sum());
        if clip[3] <= 1e-4 {
            continue;
        }
        front = true;
        for k in 0..2 {
            lo[k] = lo[k].min(clip[k] / clip[3]);
            hi[k] = hi[k].max(clip[k] / clip[3]);
        }
    }
    front && (0..2).all(|k| lo[k] <= 1.0 && hi[k] >= -1.0)
}

/// The last present LOD distance (0 when none is present).
pub fn last_lod(lods: &[f32; 4]) -> f32 {
    lods.iter().copied().filter(|d| *d >= 0.0).fold(0.0, f32::max)
}

/// Card outlines: a crown over a stem, a round blob, a rounded block.
pub const SHAPE_TREE: u32 = 0;
pub const SHAPE_BLOB: u32 = 1;
pub const SHAPE_BLOCK: u32 = 2;

/// The outline of a type from its local box: taller than wide a tree, at
/// least half as tall as wide a blob, else a low block.
pub fn shape_of(b: &Batch) -> u32 {
    let e: [f32; 3] = std::array::from_fn(|k| (b.box_max[k] - b.box_min[k]).max(0.0));
    let wide = e[0].max(e[1]);
    if e[2] >= wide {
        SHAPE_TREE
    } else if e[2] >= 0.5 * wide {
        SHAPE_BLOB
    } else {
        SHAPE_BLOCK
    }
}

/// One record's card seen from `cam` (absolute; None: dropped), without the
/// distance rule. The record's row K (xyz) is the image of model axis K; the
/// box seen from the camera is sum_K e_K |row_K . u| along a unit `u`: `u` =
/// the card's horizontal (perpendicular to the camera's bearing) for the
/// width, world z for the height. A long low prop seen end-on is narrow.
pub fn card_of(b: &Batch, r: &[f32; 16], batch: u32, cam: [f64; 3]) -> Option<Card> {
    if !(r[12] > 0.0 && r[14] > 0.0) {
        return None;
    }
    let c: [f32; 3] = std::array::from_fn(|k| 0.5 * (b.box_min[k] + b.box_max[k]));
    let e: [f32; 3] = std::array::from_fn(|k| (b.box_max[k] - b.box_min[k]).max(0.0));
    // World = c.x * r0.xyz + c.y * r1.xyz + c.z * r2.xyz + (r0.w, r1.w, r2.w).
    let centre: [f64; 3] = std::array::from_fn(|k| f64::from(c[0] * r[k] + c[1] * r[4 + k] + c[2] * r[8 + k]) + f64::from(r[4 * k + 3]));
    if !centre.iter().all(|v| v.is_finite()) {
        return None;
    }
    // The card's horizontal: perpendicular to the bearing from the camera
    // (the shader's `right`); straight above or below: world x.
    let (hx, hy) = ((centre[0] - cam[0]) as f32, (centre[1] - cam[1]) as f32);
    let h = (hx * hx + hy * hy).sqrt();
    let right = if h > 1e-3 { [-hy / h, hx / h] } else { [1.0, 0.0] };
    let half_width = 0.5 * (0..3).map(|k| e[k] * (r[4 * k] * right[0] + r[4 * k + 1] * right[1]).abs()).sum::<f32>();
    let half_height = 0.5 * (0..3).map(|k| e[k] * r[4 * k + 2].abs()).sum::<f32>();
    (half_width > 0.0 && half_height > 0.0 && half_width.is_finite() && half_height.is_finite()).then_some(Card { centre, half_width, half_height, shape: shape_of(b), fade: 1.0, distance: 0.0, batch, index: 0, root: std::array::from_fn(|k| f64::from(r[4 * k + 3])) })
}

/// Translations of the instances the list draws as meshes (0.25 m cells):
/// a card is never put where the mesh rule already drew that instance.
#[derive(Clone, Debug, Default)]
pub struct Meshed {
    cells: std::collections::HashMap<[i64; 3], Vec<[f64; 3]>>,
}

/// Same-instance tolerance (m): the same record gives the same translation.
pub const SAME_INSTANCE: f64 = 0.25;

impl Meshed {
    /// From the list's 12-float worlds (translation last) relative to `origin`.
    pub fn new(instances: &[f32], origin: [f64; 3]) -> Meshed {
        let mut cells: std::collections::HashMap<[i64; 3], Vec<[f64; 3]>> = std::collections::HashMap::new();
        for w in instances.chunks_exact(12) {
            let p: [f64; 3] = std::array::from_fn(|k| f64::from(w[9 + k]) + origin[k]);
            cells.entry(cell(p)).or_default().push(p);
        }
        Meshed { cells }
    }

    pub fn contains(&self, p: [f64; 3]) -> bool {
        let c = cell(p);
        let close = |q: &[f64; 3]| (0..3).map(|k| (q[k] - p[k]).powi(2)).sum::<f64>() < SAME_INSTANCE * SAME_INSTANCE;
        (-1..=1).any(|x| (-1..=1).any(|y| (-1..=1).any(|z| self.cells.get(&[c[0] + x, c[1] + y, c[2] + z]).is_some_and(|v| v.iter().any(close)))))
    }
}

fn cell(p: [f64; 3]) -> [i64; 3] {
    std::array::from_fn(|k| (p[k] / SAME_INSTANCE).floor() as i64)
}

/// The frame's cards for a camera at `cam` (absolute), nearest first;
/// instances in `meshed` are left to their meshes.
pub fn select(im: &Imposters, cam: [f64; 3], meshed: Option<&Meshed>, st: &mut ImposterStats) -> Vec<Card> {
    let mut out = Vec::new();
    st.batches = im.batches.len();
    for (bi, b) in im.batches.iter().enumerate() {
        st.candidates += b.n as usize;
        if b.n > 0 && b.records.is_empty() {
            st.without_records += 1;
            continue;
        }
        let last = last_lod(&b.lods);
        let (f0, f1) = (b.fade[0], b.fade[1]);
        let last = last.max(b.start);
        st.with_start += usize::from(b.start > 0.0);
        for (ri, r) in b.records.iter().enumerate() {
            let Some(mut c) = card_of(b, r, bi as u32, cam) else {
                st.dropped += 1;
                continue;
            };
            let d = (0..3).map(|k| (c.centre[k] - cam[k]).powi(2)).sum::<f64>().sqrt() as f32;
            if d < last {
                st.near += 1;
                continue;
            }
            if f1 > 0.0 && d >= f1 {
                st.far += 1;
                continue;
            }
            if meshed.is_some_and(|m| m.contains(c.root)) {
                st.meshed += 1;
                continue;
            }
            c.distance = d;
            c.index = ri as u32;
            c.fade = if f1 > f0 && d > f0 { ((f1 - d) / (f1 - f0)).clamp(0.0, 1.0) } else { 1.0 };
            st.per_bin[((d / BIN_METRES) as usize).min(BINS - 1)] += 1;
            out.push(c);
        }
    }
    out.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    out.truncate(MAX_CARDS);
    st.cards = out.len();
    out
}

/// Instances [`select`] counts as `near` (inside the last LOD distance or
/// the type's start) that are in view of `vp` (camera-relative) and have no
/// mesh of the list at their translation.
pub fn near_unmeshed(im: &Imposters, cam: [f64; 3], meshed: &Meshed, vp: &super::Mat) -> usize {
    let mut n = 0;
    for (bi, b) in im.batches.iter().enumerate() {
        let last = last_lod(&b.lods).max(b.start);
        for r in &b.records {
            let Some(c) = card_of(b, r, bi as u32, cam) else { continue };
            let rel: [f32; 3] = std::array::from_fn(|k| (c.centre[k] - cam[k]) as f32);
            let d = rel.iter().map(|v| v * v).sum::<f32>().sqrt();
            n += usize::from(d < last && !meshed.contains(c.root) && in_view(&c, rel, vp));
        }
    }
    n
}

/// Cards whose centre lies within `radius` m of a drawn instance's
/// translation (the list's 12-float worlds, relative to `origin`): a card
/// beside a mesh the list draws (another instance or the same one).
pub fn over_meshes(cards: &[Card], instances: &[f32], origin: [f64; 3], radius: f32) -> usize {
    let t: Vec<[f64; 3]> = instances.chunks_exact(12).map(|w| std::array::from_fn(|k| f64::from(w[9 + k]) + origin[k])).collect();
    let r2 = f64::from(radius).powi(2);
    cards.iter().filter(|c| t.iter().any(|p| (0..3).map(|k| (p[k] - c.centre[k]).powi(2)).sum::<f64>() < r2)).count()
}

pub const IMPOSTER_SHADER: &str = r#"
#define RS "RootConstants(num32BitConstants=32, b0)"
cbuffer Card : register(b0) {
    row_major float4x4 vp;   // 0: camera-relative view-projection
    float3 centre;           // 16: the card's centre minus the camera
    float halfWidth;         // 19
    float halfHeight;        // 20
    float fade;              // 21: 1 opaque .. 0 gone (dithered)
    uint shape;              // 22: 0 tree, 1 blob, 2 block
    uint show;               // 23: 1 magenta, no depth test (`imposters=show`)
    float4 sunDir;           // 24: light direction (xyz), ambient (w)
    float4 fog;              // 28: fog colour, 1 / distance (0: off)
};
struct VsOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
[RootSignature(RS)]
VsOut vs(uint v : SV_VertexID) {
    // (-1,-1) (1,-1) (1,1) | (-1,-1) (1,1) (-1,1)
    static const float2 corner[6] = { float2(-1, -1), float2(1, -1), float2(1, 1), float2(-1, -1), float2(1, 1), float2(-1, 1) };
    float2 c = corner[v % 6];
    float2 h = centre.xy;
    float3 right = dot(h, h) > 1e-6 ? normalize(float3(-h.y, h.x, 0)) : float3(1, 0, 0);
    float3 p = centre + c.x * halfWidth * right + float3(0, 0, c.y * halfHeight);
    VsOut o;
    o.pos = mul(float4(p, 1), vp);
    o.uv = c;
    return o;
}
float4 shade(VsOut i) {
    // Tree: an ellipse over the upper part, a thin stem below. Blob: one
    // ellipse. Block: a rectangle with rounded corners.
    float2 u = i.uv;
    float ey = shape == 0 ? (u.y - 0.25) / 0.75 : u.y;
    bool crown = shape == 2 ? length(max(abs(u) - 0.75, 0)) <= 0.25 : u.x * u.x + ey * ey <= 1;
    bool stem = shape == 0 && abs(u.x) < 0.08 && u.y < -0.4;
    if (!crown && !stem) discard;
    uint2 q = uint2(i.pos.xy) & 3;
    float dither = (float((q.x * 4 + q.y) * 7 % 16) + 0.5) / 16;
    if (fade < dither) discard;
    float3 base = crown ? float3(0.20, 0.27, 0.13) * (0.85 + 0.15 * u.y) : float3(0.25, 0.19, 0.12);
    float3 c = base * (sunDir.w > 0 ? sunDir.w + (1 - sunDir.w) * 0.6 : 0.8);
    float4 o = float4(fog.w > 0 ? lerp(c, fog.rgb, 1 - exp(-i.pos.w * fog.w)) : c, 1);
    return show != 0 ? float4(1, 0, 1, 1) : o;
}
[RootSignature(RS)]
float4 ps(VsOut i) : SV_Target { return shade(i); }
"#;

/// Our card pipeline (root constants only, no heap).
pub struct ImposterPass {
    pub opts: ImposterOptions,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// `show`: no depth test.
    pso_show: ID3D12PipelineState,
    pub last: ImposterStats,
}

// SAFETY: D3D12 objects are free-threaded; used by its pass's owner.
unsafe impl Send for ImposterPass {}

impl ImposterPass {
    pub fn new(device: &ID3D12Device) -> std::result::Result<ImposterPass, String> {
        let e = |e: windows::core::Error| e.to_string();
        let (vs, ps) = (super::compile_source(IMPOSTER_SHADER, c"vs", c"vs_5_1")?, super::compile_source(IMPOSTER_SHADER, c"ps", c"ps_5_1")?);
        // SAFETY: object creation with descriptions that outlive each call.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, super::blob_bytes(&vs)).map_err(e)?;
            let state = RenderState { depth_enable: true, depth_write: true, depth_func: 5, ..Default::default() };
            let pso = super::pipeline_state_with(device, &root, &vs, &ps, &state, &[]).map_err(e)?;
            let pso_show = super::pipeline_state_with(device, &root, &vs, &ps, &RenderState::default(), &[]).map_err(e)?;
            Ok(ImposterPass { opts: ImposterOptions::default(), root, pso, pso_show, last: ImposterStats::default() })
        }
    }

    /// Records the frame's cards on `l` (open, our pass's targets bound).
    pub fn record(&mut self, l: &ID3D12GraphicsCommandList, list: &FrameList, cam: &Camera) -> ImposterStats {
        let t0 = std::time::Instant::now();
        let mut st = ImposterStats::default();
        let Some(im) = list.imposters.as_deref().filter(|_| self.opts.on) else {
            self.last = st.clone();
            return st;
        };
        let campos = super::camera_position(&cam.view).map(f64::from);
        // The list's instance worlds are relative to the main camera.
        let meshed = Meshed::new(&list.instances, campos);
        let cards = select(im, campos, Some(&meshed), &mut st);
        let vp = cam.view_proj_relative();
        st.near_unmeshed = near_unmeshed(im, campos, &meshed, &vp);
        let mut k = [0u32; 32];
        k[..16].iter_mut().zip(&vp).for_each(|(d, s)| *d = s.to_bits());
        if let Some(s) = list.sun {
            for (j, v) in s.iter().take(4).enumerate() {
                k[24 + j] = v.to_bits();
            }
        }
        if let Some(s) = list.sky {
            for (j, v) in s.fog.iter().enumerate() {
                k[28 + j] = v.to_bits();
            }
        }
        // SAFETY: an open list and our live objects.
        unsafe {
            l.SetGraphicsRootSignature(&self.root);
            l.SetPipelineState(if self.opts.show { &self.pso_show } else { &self.pso });
        }
        k[23] = u32::from(self.opts.show);
        for c in &cards {
            for j in 0..3 {
                k[16 + j] = ((c.centre[j] - campos[j]) as f32).to_bits();
            }
            (k[19], k[20], k[21], k[22]) = (c.half_width.to_bits(), c.half_height.to_bits(), c.fade.to_bits(), c.shape);
            let rel: [f32; 3] = std::array::from_fn(|j| (c.centre[j] - campos[j]) as f32);
            st.in_view += usize::from(in_view(c, rel, &vp));
            // SAFETY: as above.
            unsafe {
                l.SetGraphicsRoot32BitConstants(0, 32, k.as_ptr().cast(), 0);
                l.DrawInstanced(6, 1, 0, 0);
            }
            st.drawn += 1;
        }
        st.record_ms = t0.elapsed().as_secs_f64() * 1e3;
        self.last = st.clone();
        st
    }

    /// The live line (every 60 frames with `imposters=on`): the frame's
    /// submit calls and skips, and the cards the last record drew.
    pub fn frame_line(&self, frame: u64, im: Option<&Imposters>) -> String {
        let none = (Imposters::default(), ImposterStats::default());
        let (im, s) = match im {
            Some(im) => (im, &self.last),
            None => (&none.0, &none.1),
        };
        format!(
            "world render: frame {frame}: imposters: calls {}, cards {} (in view {}), {}; batches {}, nearer than the start {} (in view without a mesh {}), past the far fade {}, left to their meshes {}, record {:.2} ms",
            im.calls,
            s.drawn,
            s.in_view,
            im.skipped_text(),
            im.batches.len(),
            s.near,
            s.near_unmeshed,
            s.far,
            s.meshed,
            s.record_ms
        )
    }

    /// The pass's line for the last frame.
    pub fn log_line(&self) -> String {
        let s = &self.last;
        format!(
            "world render: imposters: batches {}, candidates {}, batches without records {}, dropped {}, nearer than the last LOD {}, past the far fade {}, left to their meshes {}, cards {} (per 50 m {:?}), batches with a start distance {}, drawn {} (in view {}); record {:.2} ms",
            s.batches, s.candidates, s.without_records, s.dropped, s.near, s.far, s.meshed, s.cards, s.per_bin, s.with_start, s.drawn, s.in_view, s.record_ms
        )
    }
}

/// A made-up record: identity axes times `scale`, translation `at`.
pub fn synthetic_record(at: [f32; 3], scale: f32) -> [f32; 16] {
    let mut r = [0f32; 16];
    for k in 0..3 {
        r[5 * k] = scale;
        r[4 * k + 3] = at[k];
    }
    (r[12], r[14]) = (1.0, 1.0);
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tree type as pk4's first batch: box (-5,-5,0)..(5,5,14), no mesh
    /// LODs in its own table, far fade 400..420 m.
    fn tree(records: Vec<[f32; 16]>) -> Batch {
        Batch { kind: 0xb003939, stage: 76, n: records.len() as u32, box_min: [-5.0, -5.0, 0.0], box_max: [5.0, 5.0, 14.0], sphere: [0.0, 0.0, 7.0, 10.0], lods: [-1.0; 4], fade: [400.0, 420.0], records, start: 0.0 }
    }

    #[test]
    fn rule_sizes_and_fades() {
        let mut dropped = synthetic_record([100.0, 0.0, 0.0], 1.0);
        dropped[12] = 0.0;
        let b = tree(vec![
            synthetic_record([100.0, 0.0, 0.0], 1.0),
            synthetic_record([0.0, 410.0, 0.0], 2.0),
            synthetic_record([0.0, 0.0, 0.0], 1.0),
            synthetic_record([1000.0, 0.0, 0.0], 1.0),
            dropped,
        ]);
        let mut near = tree(vec![synthetic_record([50.0, 0.0, 0.0], 1.0), synthetic_record([250.0, 0.0, 0.0], 1.0)]);
        near.lods = [30.0, 80.0, 150.0, -1.0];
        let mut b = b;
        b.start = 50.0;
        let im = Imposters { batches: vec![b, near, Batch { n: 9, ..tree(Vec::new()) }], ..Default::default() };
        let mut st = ImposterStats::default();
        let cards = select(&im, [0.0, 0.0, 0.0], None, &mut st);
        assert_eq!((st.batches, st.candidates, st.without_records, st.dropped, st.near, st.far, st.cards), (3, 16, 1, 1, 2, 1, 3), "{st:?}");
        // Nearest first: 100 m, 250 m, 410 m; the one at the camera is inside the type's start (50 m).
        let d: Vec<u32> = cards.iter().map(|c| c.distance.round() as u32).collect();
        assert_eq!(d, [100, 250, 410]);
        assert_eq!(cards[0].centre, [100.0, 0.0, 7.0], "the box centre, not the root");
        assert_eq!((cards[0].half_width, cards[0].half_height, cards[0].fade), (5.0, 7.0, 1.0));
        let c = cards[2];
        assert_eq!((c.half_width, c.half_height, c.batch), (10.0, 14.0, 0), "scale 2");
        assert!((c.fade - 0.5).abs() < 0.05, "half way from 400 to 420: {}", c.fade);
        assert_eq!(cards[1].batch, 1, "past the last LOD (150 m)");
        assert_eq!(st.with_start, 1);
        assert_eq!(st.per_bin[2] + st.per_bin[5] + st.per_bin[8], 3);
        assert_eq!(last_lod(&[30.0, 80.0, 150.0, -1.0]), 150.0);
        assert_eq!(last_lod(&[-1.0; 4]), 0.0);
    }

    /// A long low prop (pk5 f977 calls 15-17: box 5.6 x 1.2 x 1.6, the two
    /// "discs" of f977-y0-on.png): a block, 5.6 m wide seen from the side,
    /// 1.2 m seen end-on, 2.6 m turned 30 degrees; a tree type stays a tree.
    #[test]
    fn card_width_follows_the_view() {
        let b = Batch { box_min: [-2.8, -0.6, 0.0], box_max: [2.8, 0.6, 1.6], ..tree(Vec::new()) };
        let r = synthetic_record([100.0, 0.0, 0.0], 1.0);
        let w = |cam: [f64; 3]| card_of(&b, &r, 0, cam).map(|c| (2.0 * c.half_width, 2.0 * c.half_height, c.shape)).unwrap();
        let near = |a: (f32, f32, u32), e: (f32, f32, u32)| (a.0 - e.0).abs() < 1e-3 && (a.1 - e.1).abs() < 1e-3 && a.2 == e.2;
        assert!(near(w([100.0, -50.0, 0.0]), (5.6, 1.6, SHAPE_BLOCK)), "side-on: {:?}", w([100.0, -50.0, 0.0]));
        assert!(near(w([0.0, 0.0, 0.0]), (1.2, 1.6, SHAPE_BLOCK)), "end-on: {:?}", w([0.0, 0.0, 0.0]));
        // Turned 30 degrees about z: 5.6 sin 30 + 1.2 cos 30 seen end-on.
        let (s, c) = (30f32.to_radians().sin(), 30f32.to_radians().cos());
        let mut t = r;
        (t[0], t[1], t[4], t[5]) = (c, s, -s, c);
        let got = card_of(&b, &t, 0, [0.0; 3]).unwrap();
        assert!((2.0 * got.half_width - (5.6 * s + 1.2 * c)).abs() < 1e-3, "{got:?}");
        assert_eq!(shape_of(&tree(Vec::new())), SHAPE_TREE);
        assert_eq!(shape_of(&Batch { box_min: [-3.0, -3.0, 0.0], box_max: [3.0, 3.0, 4.0], ..tree(Vec::new()) }), SHAPE_BLOB);
        let im = Imposters { skipped: [1, 0, 2], ..Default::default() };
        assert_eq!(im.skipped_text(), "skipped 3 (unreadable 1, over the cap 2)");
        assert_eq!(Imposters::default().skipped_text(), "skipped 0");
    }

    #[test]
    fn cards_over_meshes_and_switch() {
        let cards = [Card { centre: [100.0, 0.0, 7.0], half_width: 5.0, half_height: 7.0, shape: SHAPE_TREE, fade: 1.0, distance: 100.0, batch: 0, index: 0, root: [100.0, 0.0, 0.0] }, Card { centre: [300.0, 0.0, 7.0], half_width: 5.0, half_height: 7.0, shape: SHAPE_TREE, fade: 1.0, distance: 300.0, batch: 0, index: 1, root: [300.0, 0.0, 0.0] }];
        let mut w = vec![0f32; 24];
        // Instance worlds relative to origin (10, 0, 0): one at 99, 0, 6 absolute.
        (w[9], w[10], w[11]) = (89.0, 0.0, 6.0);
        (w[21], w[22], w[23]) = (0.0, 0.0, 0.0);
        assert_eq!(over_meshes(&cards, &w, [10.0, 0.0, 0.0], 5.0), 1);
        // The same instance drawn as a mesh (translation 100, 0, 0): no card; 0.6 m away: a card.
        let im = Imposters { batches: vec![tree(vec![synthetic_record([100.0, 0.0, 0.0], 1.0), synthetic_record([100.6, 0.0, 0.0], 1.0)])], ..Default::default() };
        let mut w = vec![0f32; 12];
        (w[9], w[10], w[11]) = (90.0, 0.0, 0.0);
        let m = Meshed::new(&w, [10.0, 0.0, 0.0]);
        let mut st = ImposterStats::default();
        let cards = select(&im, [0.0; 3], Some(&m), &mut st);
        assert_eq!((st.meshed, cards.len(), cards[0].index), (1, 1, 1), "{st:?}");
        assert!(!parse_screen("screen,terrain=on").on, "off by default");
        assert!(parse_screen("screen, imposters=on").on);
        assert_eq!(parse_screen("screen,imposters=show"), ImposterOptions { on: true, show: true });
        assert_eq!(parse_screen("imposters=show,imposters=on"), ImposterOptions { on: true, show: false }, "the last one wins");
    }

    /// A card ahead of a camera looking along +x is in view; behind it, or
    /// far to the side, it is not. View: x forward -> clip w; y -> clip x
    /// (negated: right-handed), z -> clip y.
    #[test]
    fn cards_in_view() {
        let mut vp = [0f32; 16];
        (vp[4 * 1], vp[4 * 2 + 1], vp[3], vp[4 * 3 + 2]) = (-1.0, 1.0, 1.0, 0.1);
        let c = Card { centre: [0.0; 3], half_width: 5.0, half_height: 7.0, shape: SHAPE_TREE, fade: 1.0, distance: 100.0, batch: 0, index: 0, root: [0.0; 3] };
        assert!(in_view(&c, [100.0, 0.0, 7.0], &vp));
        assert!(!in_view(&c, [-100.0, 0.0, 7.0], &vp), "behind");
        assert!(!in_view(&c, [100.0, 300.0, 7.0], &vp), "off to the side");
        assert!(in_view(&c, [100.0, 103.0, 7.0], &vp), "its edge reaches the screen");
    }

    /// A near instance in view with no mesh is counted; one behind the
    /// camera, one with a mesh, and one past the start are not.
    #[test]
    fn near_in_view_without_a_mesh() {
        let mut vp = [0f32; 16];
        (vp[4 * 1], vp[4 * 2 + 1], vp[3], vp[4 * 3 + 2]) = (-1.0, 1.0, 1.0, 0.1);
        let mut b = tree(vec![synthetic_record([20.0, 0.0, 0.0], 1.0), synthetic_record([-20.0, 0.0, 0.0], 1.0), synthetic_record([30.0, 0.0, 0.0], 1.0), synthetic_record([200.0, 0.0, 0.0], 1.0)]);
        b.start = 50.0;
        let im = Imposters { batches: vec![b], ..Default::default() };
        let mut w = vec![0f32; 12];
        (w[9], w[10], w[11]) = (30.0, 0.0, 0.0);
        let m = Meshed::new(&w, [0.0; 3]);
        let mut st = ImposterStats::default();
        select(&im, [0.0; 3], Some(&m), &mut st);
        assert_eq!((st.near, near_unmeshed(&im, [0.0; 3], &m, &vp)), (3, 1));
    }

    /// Two made-up trees seen from 300 m south at eye height 7 m, through
    /// the whole pass on WARP: one 200 m ahead (a card, centre of the
    /// picture), one 30 m ahead with a mesh LOD to 150 m (no card).
    /// `--imposters` off: nothing.
    #[test]
    fn cards_on_warp() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        // SAFETY: inside call_once, before this test makes a device.
        ONCE.call_once(|| unsafe { std::env::set_var("XNW_ADAPTER", "warp") });
        let Some(host) = xnw_gpu::d3d12x::host::host_device() else { return };
        let cam = [0.0f32, -300.0, 7.0];
        let mut near = tree(vec![synthetic_record([-20.0, -270.0, 0.0], 1.0)]);
        near.lods = [50.0, 150.0, -1.0, -1.0];
        let im = Imposters { batches: vec![tree(vec![synthetic_record([0.0, -100.0, 0.0], 1.0)]), near], ..Default::default() };
        // Looking along +y, z up: camera x = world x, camera y = world z, camera -z = world y.
        let mut view = [0f32; 16];
        (view[0], view[6], view[9], view[15]) = (1.0, -1.0, 1.0, 1.0);
        (view[12], view[13], view[14]) = (-cam[0], -cam[2], cam[1]);
        let mut proj = [0f32; 16];
        (proj[0], proj[5], proj[11], proj[14]) = (1.0, 1.0, -1.0, 0.5);
        let list = FrameList { frame: 1, cameras: vec![Camera { viewport: 1, view, proj }], main: Some(1), imposters: Some(std::sync::Arc::new(im)), ..Default::default() };
        let mut pass = super::super::Pass::new(host, 128, 128).unwrap();
        pass.imposters.opts.on = true;
        pass.draw(&list, &mut |_| None, true).unwrap();
        assert_eq!((pass.imposters.last.cards, pass.imposters.last.near, pass.imposters.last.drawn), (1, 1, 1), "{:?}", pass.imposters.last);
        let px = pass.read_rgba().unwrap();
        let at = |col: usize, row: usize| -> [u8; 4] { px[4 * (128 * row + col)..4 * (128 * row + col) + 4].try_into().unwrap() };
        // The crown centre: box centre z 7 + 0.25 * 7 = 8.75 m, 200 m ahead: 1.75/200 above the centre (~1 px).
        let c = at(64, 63);
        assert!(c[1] > c[2] && c[1] > 20, "a green crown at the centre: {c:?}");
        // 5 m half width at 200 m: 1.6 px each side; 4 px out is the clear colour; the near tree (at -20 m: column 64 - 20/30*64 = 21) has no card.
        let side = at(70, 63);
        let near_px = at(21, 63);
        assert!(side[1] < 30 && near_px[1] < 30, "outside the card and no card on the near tree: {side:?} {near_px:?}");
        pass.imposters.opts.on = false;
        pass.draw(&list, &mut |_| None, true).unwrap();
        let px = pass.read_rgba().unwrap();
        assert!(px[4 * (128 * 63 + 64) + 1] < 30, "off: the clear colour");
    }

    #[test]
    fn imposter_shader_compiles() {
        for (entry, target) in [(c"vs", c"vs_5_1"), (c"ps", c"ps_5_1")] {
            if let Err(e) = super::super::compile_source(IMPOSTER_SHADER, entry, target) {
                panic!("{e}");
            }
        }
    }
}
