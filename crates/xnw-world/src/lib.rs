//! Our own world renderer, first step (`XNW_WORLD_RENDER=1`). The title's
//! geometry draw is hooked
//! (apps/xnw worldprobe.rs). Each draw becomes a [`DrawRecord`]: the geometry's
//! identity, its camera-relative transform and the DiffuseTex identity. The
//! records go into one [`FrameList`] per presented frame. A host pass of
//! ours draws the finished list into our own colour and depth target with
//! one pipeline ([`SHADER`]). The presenter then shows that target in place
//! of the title's scene plane.
//!
//! Built here: the record and list types, the hand-over between the
//! recording thread and the pass ([`Lists`]), the geometry cache with its keys
//! ([`GeometryCache`]), the matrices, vertex packing, the shader (FXC at run
//! time, as the presenter's compositor), and the device side: [`Pass`]
//! with one pipeline, colour + reverse-Z depth targets, upload-heap geometry
//! buffers and a readback (`xnw world-render` draws probe dumps with it on
//! WARP). Not built: pack textures, the swap into the title's scene plane.
//!
//! Space [V, one capture's frame 1700]: a draw's transform is relative to its
//! viewport's camera. Adding the camera position `-T R^T` of the viewport's
//! view matrix gives one absolute position per object across viewports, so
//! the pass uses `world_rel * R * P`, where `R` is the view with its
//! translation dropped. That keeps the floats small near the camera.
//! Projection: row vectors (`clip = p * M`), reversed infinite depth
//! (`P[2][2]` ~ 0, `P[3][2]` = near, `P[2][3]` = -1). The main camera of
//! frame 1700 has a 33.4 degree vertical FOV, 16:9, near 0.15 m.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

/// Albedo textures: our table, the slot resolve, the budget.
pub mod textures;
pub use textures::{TEX_SLOTS, TexMode, TexOptions};
/// The geometry budget, eviction and the recorder hand-shake.
pub mod geometry;
/// The title's lighting and output chain for our pixel shader.
pub mod lighting;
/// The heightmap terrain (shader family 4): tile records, our grid.
pub mod terrain;
/// Camera cuts: coverage, hold of the last complete picture, recovery budget.
pub mod cuts;
/// The title's ped pre-skin compute programs run on captured inputs.
pub mod preskin;
/// Far trees and props as camera-facing cards (the title's imposter batches).
pub mod imposter;
/// One captured draw through the title's own mesh and pixel program.
pub mod bridge;
/// One compute runner for captured title dispatches over buffers and textures, results kept by title address.
pub mod compute;
/// The title's pixel programs per row in our pass, live and replayed.
pub mod bridged;
/// The title's mesh programs in our pass (`bridgems`).
pub mod bridge_ms;
/// The ped pre-skin as our own compute pass: the frame's items, the draws' copies.
pub mod preskin_gpu;
/// Strand hair: the title's strand draws as data, the strand vertex rule.
pub mod hair;
/// The instance cull on the device (`gpucull=on`): per-instance frustum and LOD, indirect draws.
pub mod gpucull;
/// `gbuf=on`: the title's G-buffer layout beside our picture (LIT2).
pub mod gbuf;

pub type Mat = [f32; 16];

/// `a * b`, row-major, row vectors.
/// The inverse of a 4x4 matrix (None when singular).
pub fn invert(m: &Mat) -> Option<Mat> {
    let a = |r: usize, c: usize| f64::from(m[4 * r + c]);
    let mut inv = [0f64; 16];
    // Cofactors (adjugate, transposed) by the standard expansion.
    for r in 0..4 {
        for c in 0..4 {
            let rows: Vec<usize> = (0..4).filter(|&i| i != c).collect();
            let cols: Vec<usize> = (0..4).filter(|&j| j != r).collect();
            let d = a(rows[0], cols[0]) * (a(rows[1], cols[1]) * a(rows[2], cols[2]) - a(rows[1], cols[2]) * a(rows[2], cols[1]))
                - a(rows[0], cols[1]) * (a(rows[1], cols[0]) * a(rows[2], cols[2]) - a(rows[1], cols[2]) * a(rows[2], cols[0]))
                + a(rows[0], cols[2]) * (a(rows[1], cols[0]) * a(rows[2], cols[1]) - a(rows[1], cols[1]) * a(rows[2], cols[0]));
            inv[4 * r + c] = if (r + c) % 2 == 0 { d } else { -d };
        }
    }
    let det: f64 = (0..4).map(|j| a(0, j) * inv[4 * j]).sum();
    (det.abs() > 1e-30).then(|| std::array::from_fn(|k| (inv[k] / det) as f32))
}

pub fn mul(a: &Mat, b: &Mat) -> Mat {
    std::array::from_fn(|k| {
        let (i, j) = (k / 4, k % 4);
        (0..4).map(|n| a[4 * i + n] * b[4 * n + j]).sum()
    })
}

/// The camera position of a view matrix `p_view = p R + T`: `-T R^T`.
pub fn camera_position(view: &Mat) -> [f32; 3] {
    std::array::from_fn(|i| -(view[12] * view[4 * i] + view[13] * view[4 * i + 1] + view[14] * view[4 * i + 2]))
}

/// The view without its translation (for camera-relative positions).
pub fn rotation_only(view: &Mat) -> Mat {
    let mut r = *view;
    r[12] = 0.0;
    r[13] = 0.0;
    r[14] = 0.0;
    r
}

/// What a reversed-depth perspective projection says: vertical field of
/// view (degrees), aspect (width / height), near plane; far is infinite
/// when `P[2][2]` is about 0.
pub fn projection_params(p: &Mat) -> (f32, f32, f32) {
    let fovy = 2.0 * (1.0 / p[5]).atan().to_degrees();
    let aspect = p[5] / p[0];
    let near = p[14] / (1.0 - p[10]);
    (fovy, aspect, near)
}

/// A camera as the title's viewport holds it (+0x40 view, +0x80 projection).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub viewport: u64,
    pub view: Mat,
    pub proj: Mat,
}

/// The title's projection carries its TAA jitter in row 2
/// x/y (proj[8], proj[9]: non-zero in 296 of 300 frames, median 0.74 px, max
/// 1.5 px at 1080p, new every frame): we have no TAA resolve, so it only
/// shakes the whole picture. Dropped.
pub fn unjittered(proj: &Mat) -> Mat {
    let mut p = *proj;
    (p[8], p[9]) = (0.0, 0.0);
    p
}

/// Keep the jitter (default off): the bridge audit compares our pass with
/// the title's own mesh programs, which draw with it (the rim of
/// pk4 f887, pk3 f1248/f1250 was the jitter, -p8 * W/2, -p9 * H/2 px).
static KEEP_JITTER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `--jitter` of `world-render`, and the audit's ours side: the projection
/// as captured, TAA jitter included.
pub fn keep_jitter(on: bool) {
    KEEP_JITTER.store(on, std::sync::atomic::Ordering::Relaxed);
}

impl Camera {
    /// The matrix a camera-relative world matrix is multiplied with.
    pub fn view_proj_relative(&self) -> Mat {
        self.view_proj(KEEP_JITTER.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// The same, with the TAA jitter kept or dropped.
    pub fn view_proj(&self, jitter: bool) -> Mat {
        mul(&rotation_only(&self.view), &if jitter { self.proj } else { unjittered(&self.proj) })
    }
}

/// The content identity of a title geometry: where its meshlet data and
/// vertex buffer live, and their sizes. A geometry streamed out and
/// another streamed into the same memory changes at least one of these
/// in practice [I]. The origin registry's stamps would make it exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct GeomKey {
    pub mesh_at: u64,
    pub mesh_len: u64,
    pub vb_at: u64,
    pub vb_len: u64,
    /// A classic geometry (no meshlets): index count << 8 | index size; 0 otherwise.
    pub ib: u64,
    /// The geometry's hash-like qword +0x30: the title reuses
    /// buffer memory for another geometry of the same size, so addresses
    /// and sizes alone keyed stale buffers (terrain shards in a test run).
    pub hash: u64,
}

/// Albedo: a texture of our texture pack (`texpack::file_id` of its
/// `.ttpf`, texture index), else a flat colour per effect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Albedo {
    Pack { file: u64, texture: u32 },
    Flat([f32; 3]),
}

/// A stable flat colour for an effect (shader) pointer or hash.
pub fn flat_colour(effect: u64) -> [f32; 3] {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    effect.hash(&mut h);
    let v = h.finish();
    let c = |s: u32| 0.35 + 0.55 * ((v >> s) & 0xFF) as f32 / 255.0;
    [c(0), c(8), c(16)]
}

/// One draw of the frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawRecord {
    pub geom: GeomKey,
    /// Camera-relative world matrix (first half of the title's transform).
    pub world: Mat,
    pub viewport: u64,
    pub effect: u64,
    pub albedo: Albedo,
    /// Index into [`FrameList::palettes`] for a skinned draw, else `NO_PALETTE`.
    pub palette: u32,
    /// The pass stage of the draw (0x140369A80 edx), for `pass=` views.
    pub stage: u32,
    /// Instanced draws (0x14006B210): `inst_count` worlds at
    /// `FrameList::instances[12 * inst_first ..]`; `world` is then identity.
    pub inst_first: u32,
    pub inst_count: u32,
    /// A skinned draw's bones in its palette: the title's curIndex
    /// and curCount (0: the whole palette entry, as before).
    pub bone_first: u32,
    pub bone_count: u32,
    /// Reserved material fields, not yet filled or read.
    pub material: Material,
    /// The render state the title set for this draw (decoded from
    /// its tables); None: our defaults.
    pub state: Option<RenderState>,
    /// The ped bone remap: the float index in this draw's palette
    /// floats of remap[base] (u32 entries), or NO_REMAP.
    pub remap: u32,
    /// A Ped* program: weights divided by their sum (generic
    /// TransformSkin: raw w/255, stopping at the first zero weight).
    pub ped: bool,
}

pub const NO_REMAP: u32 = u32::MAX;

/// A: `skin_rules` bit 1: a skinned non-ped draw whose bones are all
/// exactly the identity (the rest pose) leaves the `skinned` skip.
/// It stays skinned: with identity bones that is the static shape wherever the
/// weights sum to 255, and the title's own shrink where not.
pub const SKIN_IDENTITY_STATIC: u32 = 2;
/// `skin_rules` bit 2: a skinned non-ped draw whose bones are all finite
/// rigid transforms (|A A^T - I| < 1e-3) leaves the `skinned` skip as well.
pub const SKIN_RIGID_BONES: u32 = 4;
/// `skin_rules` bit 3: a skinned non-ped draw whose palette is a copy
/// made at its draw ([`FrameList::palette_at_draw`]: the bones the title drew
/// with, not the end-of-frame ring copy) leaves the `skinned` skip.
pub const SKIN_DRAW_TIME: u32 = 8;

/// What a skinned draw's bone window holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkinWindow {
    Identity,
    Rigid,
    Other,
}

/// The bones `bone_first .. + bone_count` of a skinned draw's palette:
/// all identity, all rigid (orthonormal 3x3, finite), or other (no palette,
/// a whole-palette draw, a zero, sheared or non-finite bone).
pub fn skin_window(list: &FrameList, r: &DrawRecord) -> SkinWindow {
    let Some(p) = list.palettes.get(r.palette as usize).filter(|_| r.palette != NO_PALETTE && r.bone_count > 0) else { return SkinWindow::Other };
    let Some(w) = p.get(12 * r.bone_first as usize..12 * (r.bone_first + r.bone_count) as usize) else { return SkinWindow::Other };
    if w.chunks_exact(12).all(|m| m.iter().enumerate().all(|(j, v)| *v == if j % 5 == 0 { 1.0 } else { 0.0 })) {
        return SkinWindow::Identity;
    }
    let rigid = |m: &[f32]| m.iter().all(|v| v.is_finite()) && (0..3).all(|i| (0..3).all(|j| ((0..3).map(|k| m[4 * i + k] * m[4 * j + k]).sum::<f32>() - if i == j { 1.0 } else { 0.0 }).abs() < 1e-3));
    if w.chunks_exact(12).all(rigid) { SkinWindow::Rigid } else { SkinWindow::Other }
}

/// Whether `skin_rules` lets a skinned non-ped draw out of the
/// `skinned` skip (identity or rigid window, or draw-time bones).
pub fn skin_lifted(list: &FrameList, r: &DrawRecord, rules: u32) -> bool {
    if rules & SKIN_DRAW_TIME != 0 && r.palette != NO_PALETTE && list.palette_at_draw.get(r.palette as usize).copied().unwrap_or(false) {
        return true;
    }
    match skin_window(list, r) {
        SkinWindow::Identity => rules & (SKIN_IDENTITY_STATIC | SKIN_RIGID_BONES) != 0,
        SkinWindow::Rigid => rules & SKIN_RIGID_BONES != 0,
        SkinWindow::Other => false,
    }
}

/// The sky stand-in: sun direction (xyz) and the cosine of the
/// disc radius (w), sun colour, zenith and horizon colours, moon direction
/// (w 1: draw its disc), fog colour (rgb) and 1 / fog distance (w; 0: no fog).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sky {
    pub sun: [f32; 4],
    pub sun_col: [f32; 4],
    pub zenith: [f32; 4],
    pub horizon: [f32; 4],
    pub moon: [f32; 4],
    pub fog: [f32; 4],
}

/// A water grid in absolute world coordinates (built from the
/// title's water height map; rebuilt only when the map changes: `id`).
#[derive(Debug, Default)]
pub struct WaterGrid {
    pub id: u64,
    pub vertices: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

/// The frame's water: the grid, the main camera's absolute position and
/// the tint (rgb, alpha).
#[derive(Clone, Debug)]
pub struct Water {
    pub grid: std::sync::Arc<WaterGrid>,
    pub origin: [f32; 3],
    pub tint: [f32; 4],
}

/// The water pass: positions minus the camera, the camera-relative
/// view-projection; lit by the sun, blended, depth test without write.
pub const WATER_SHADER: &str = r#"
#define RS "RootConstants(num32BitConstants=28, b0), RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), CBV(b1)"
cbuffer Water : register(b0) {
    row_major float4x4 vp;
    float4 origin; float4 tint; float4 sunDir;
};
// The frame's Light buffer (lighting.rs `constants`): light dir (towards it, w 1: on),
// radiance colour x mult (w: exposure), ambient sky (w 1: the old look), ambient ground,
// Hable A B C D, E F W 1/Hable(W), sky radiance scale (x).
cbuffer Light : register(b1) { float4 lDir; float4 lRad; float4 lSky; float4 lGround; float4 tone0; float4 tone1; float4 lPad; };
float3 hableC(float3 x) { return (x * (tone0.x * x + tone0.z * tone0.y) + tone0.w * tone1.x) / (x * (tone0.x * x + tone0.y) + tone0.w * tone1.y) - tone1.x / tone1.y; }
bool titleChain() { return lDir.w > 0.5 && lSky.w < 0.5; }
// Linear radiance -> exposure -> Hable / Hable(W) -> pow 1/2.2.
float3 tonemap(float3 hdr) { return pow(saturate(hableC(max(hdr * lRad.w, 0)) * tone1.w), 1.0 / 2.2); }
// A stand-in colour made for the screen (sky, fog) as radiance: linearised, x the sky scale.
float3 skyRadiance(float3 c) { return pow(max(c, 0), 2.2) * lPad.x; }

struct VsOut { float4 pos : SV_Position; };
[RootSignature(RS)]
VsOut vs(float3 p : POSITION) {
    VsOut o;
    o.pos = mul(float4(p - origin.xyz, 1), vp);
    return o;
}
float4 shade(VsOut i) {
    // The tint as stored albedo on a flat (+z) surface, lit and tonemapped like the scene.
    if (titleChain()) {
        float nl = saturate(normalize(lDir.xyz + 1e-9).z);
        return float4(tonemap(tint.rgb * (lRad.rgb * (nl * 0.31830989) + lSky.rgb)), tint.a);
    }
    float lit = 0.45 + 0.55 * saturate(abs(normalize(sunDir.xyz + 1e-6).z));
    return float4(tint.rgb * lit, tint.a);
}
// One return in the entry: the newer WARP (1.0.22) removes the device
// (no layer message) on a pixel shader that returns inside a branch.
[RootSignature(RS)]
float4 ps(VsOut i) : SV_Target { return shade(i); }
"#;

/// The sky pass: one full-screen triangle at the far plane (reverse Z:
/// depth 0), view rays from the inverse camera-relative view-projection.
pub const SKY_SHADER: &str = r#"
#define RS "RootConstants(num32BitConstants=40, b0), CBV(b1)"
cbuffer Sky : register(b0) {
    row_major float4x4 inv;
    float4 sun; float4 sunCol; float4 zenith; float4 horizon; float4 moon; float4 fog;
};
// The frame's Light buffer (lighting.rs `constants`): light dir (towards it, w 1: on),
// radiance colour x mult (w: exposure), ambient sky (w 1: the old look), ambient ground,
// Hable A B C D, E F W 1/Hable(W), sky radiance scale (x).
cbuffer Light : register(b1) { float4 lDir; float4 lRad; float4 lSky; float4 lGround; float4 tone0; float4 tone1; float4 lPad; };
float3 hableC(float3 x) { return (x * (tone0.x * x + tone0.z * tone0.y) + tone0.w * tone1.x) / (x * (tone0.x * x + tone0.y) + tone0.w * tone1.y) - tone1.x / tone1.y; }
bool titleChain() { return lDir.w > 0.5 && lSky.w < 0.5; }
// Linear radiance -> exposure -> Hable / Hable(W) -> pow 1/2.2.
float3 tonemap(float3 hdr) { return pow(saturate(hableC(max(hdr * lRad.w, 0)) * tone1.w), 1.0 / 2.2); }
// A stand-in colour made for the screen (sky, fog) as radiance: linearised, x the sky scale.
float3 skyRadiance(float3 c) { return pow(max(c, 0), 2.2) * lPad.x; }

struct VsOut { float4 pos : SV_Position; float2 ndc : TEXCOORD0; };
[RootSignature(RS)]
VsOut vs(uint id : SV_VertexID) {
    VsOut o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.ndc = uv * float2(2, -2) + float2(-1, 1);
    o.pos = float4(o.ndc, 0, 1);
    return o;
}
float4 shade(VsOut i) {
    float4 p = mul(float4(i.ndc, 1, 1), inv);
    float3 d = normalize(p.xyz / p.w + 1e-9);
    float h = saturate(d.z);
    float3 c = lerp(horizon.rgb, zenith.rgb, sqrt(h));
    if (d.z < 0) c = horizon.rgb * (1 - 0.5 * saturate(-d.z * 4));
    float3 s = normalize(sun.xyz + 1e-9);
    float sd = dot(d, s);
    c += sunCol.rgb * (pow(saturate(sd), 64) * 0.35 + (sd > sun.w ? 4.0 : 0.0));
    if (moon.w > 0.5) {
        float md = dot(d, normalize(moon.xyz + 1e-9));
        c += md > sun.w ? float3(0.75, 0.78, 0.85) : float3(0, 0, 0);
    }
    // Through the same exposure and tonemap as the scene. At night
    // (no sun radiance, the sky scale at its floor) that crushed the stand-in
    // to black; it never goes below 0.6 x the stand-in colour.
    if (titleChain()) return float4(max(tonemap(skyRadiance(c)), saturate(c) * 0.6), 1);
    return float4(saturate(c), 1);
}
// One return in the entry: the newer WARP (1.0.22) removes the device
// (no layer message) on a pixel shader that returns inside a branch.
[RootSignature(RS)]
float4 ps(VsOut i) : SV_Target { return shade(i); }
"#;

/// A draw's render state from the title's records, in D3D12 enum
/// values: depth-stencil (24 B: enable, write, func), raster (16 B: i8 bias
/// +3, f32 clamp +4, f32 slope +8; cull and fill are recorded by the probe but
/// not applied: our winding is not the title's [U]) and blend RT0 (68 B:
/// +4 {enable, src, dst, op, srcA, dstA, opA, mask}).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct RenderState {
    pub depth_enable: bool,
    pub depth_write: bool,
    pub depth_func: u8,
    pub depth_bias: i32,
    /// f32 bits.
    pub bias_clamp: u32,
    pub slope_bias: u32,
    pub blend: [u8; 8],
}

/// Stages that draw other views (cube/probe capture 36, local-light
/// shadows 21, sun cascades 23-28 and 48, puddle view 42, 44, 60) or
/// particles (13): never in the main view.
pub const OTHER_VIEW_STAGES: [u32; 14] = [36, 21, 23, 24, 25, 26, 27, 28, 48, 42, 44, 60, 13, 12];
/// Stage 14 is transparent (bucket 23): after opaque and decals,
/// back to front.
pub const TRANSPARENT_STAGES: [u32; 1] = [14];
/// Pipelines made from title states at most (then our defaults).
const MAX_STATE_PSOS: usize = 64;

/// Reserved per-draw material fields (all 0 until filled).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Material {
    /// The albedo texture key after EDUP (pack name hash).
    pub albedo_key: u64,
    /// The effect's alpha-clip (cut-out) flag.
    pub alpha_clip: bool,
    /// The tint palette row (0: none).
    pub tint_row: u16,
    /// The effect name hash.
    pub effect_name: u32,
    /// The albedo view's descriptor slot (`view+0x18` at the
    /// hook; 0: none or not resident), resolved by the pass ([`textures`]).
    pub albedo_desc: u64,
    /// A layered effect's layer diffuse descriptor slots (0: none).
    pub layers: [u64; 4],
    /// The 2x2-quad sample mask as [`stipple_lanes`] packs it
    /// (0: solid; else 0x10 | the lanes kept).
    pub stipple: u8,
    /// The title's state indices at the draw (blend, depth-stencil,
    /// raster; 0xFFFF: not read), for the repeat table offline.
    pub state_index: [u16; 3],
}

/// The title's u64 sample mask [V data] (four 16-bit
/// lanes, each 0 or 0xFFFF: lane = (x & 1) | (y & 1) << 1 of the 2x2 quad
/// [I which lane is X1Y0]) as 0 (solid: all lanes, or -1) or 0x10 | lanes.
/// Set 0 always (no temporal rotation): one lane = {0}, three = {1, 2, 3};
/// two lanes stay as given (the 50 % pair is {1, 2} / {0, 3}, complements).
pub fn stipple_lanes(mask: u64) -> u8 {
    let lanes = (0..4).fold(0u8, |a, k| a | u8::from((mask >> (16 * k)) & 0xFFFF != 0) << k);
    match lanes.count_ones() {
        4 => 0,
        1 => 0x10 | 0b0001,
        3 => 0x10 | 0b1110,
        _ => 0x10 | lanes,
    }
}

pub const NO_PALETTE: u32 = u32::MAX;

/// A frame's draws and its cameras.
#[derive(Clone, Debug, Default)]
pub struct FrameList {
    pub frame: u64,
    pub draws: Vec<DrawRecord>,
    pub cameras: Vec<Camera>,
    /// The main view when the recorder knows it (counted over every
    /// draw, not only those with a transform); else [`FrameList::main_camera`] picks.
    pub main: Option<u64>,
    /// Skin palettes of this frame (48-byte 3x4 rows per bone, as f32).
    pub palettes: Vec<Vec<f32>>,
    /// Per palette, true when its bones were copied at the draw (hook
    /// time); missing or false: the end-of-frame copy.
    pub palette_at_draw: Vec<bool>,
    /// Geometries first seen in this frame, with their bytes; the pass
    /// uploads them before drawing (kept across replaced lists).
    pub new_geoms: Vec<(GeomKey, HostGeometry)>,
    /// Geometries offered from residency (no draw asked yet): the ahead pool.
    pub ahead_geoms: Vec<(GeomKey, HostGeometry)>,
    /// Rigid parts' bones (draw index, 3 float4 rows), for dumps and replays.
    pub rigid: Vec<(u32, [f32; 12])>,
    /// Per-instance camera-relative worlds, 12 floats each: the rows of
    /// x, y, z (3 floats each) then the translation (row-vector 4x3).
    pub instances: Vec<f32>,
    /// The frame's sun from the timecycle block: direction xyz,
    /// ambient, colour rgb, 1.0.
    pub sun: Option<[f32; 8]>,
    /// Skinned draws the hook saw (curCount 1..4096): the pass line
    /// says how many of them were drawn skinned.
    pub skin_hooked: u32,
    /// The ped pre-skin of this frame (no items: none).
    pub preskin: preskin_gpu::PreskinFrame,
    /// The sky stand-in and fog for this frame (None: none).
    pub sky: Option<Sky>,
    /// The water stand-in (None: none).
    pub water: Option<Water>,
    /// Lighting from the timecycle (None: the earlier look).
    pub light: Option<lighting::Lighting>,
    /// The heightmap terrain's tile records of this frame (None: not read).
    pub terrain: Option<std::sync::Arc<terrain::Terrain>>,
    /// The frame's imposter batches (None: not read).
    pub imposters: Option<std::sync::Arc<imposter::Imposters>>,
    /// Rows drawn again by the title's own pixel programs (`bridgeps`; None: off).
    pub bridged: Option<std::sync::Arc<bridged::Frame>>,
    /// Instanced rows culled on the device (`gpucull=on`; empty: none): their
    /// `inst_first`/`inst_count` index the cull's output, not `instances`.
    pub cull: gpucull::CullFrame,
}

/// An instance record (64 B: rows 0..2 of a row-major 3x4, translation
/// in the .w lanes) as our 12 floats, translated by `-origin` (the
/// camera position when records are absolute).
pub fn instance_world(rec: &[f32; 16], origin: [f32; 3]) -> [f32; 12] {
    let r = |i: usize, j: usize| rec[4 * i + j];
    // [V MS_VS_PropFoliageDeferred 0x1390]: p' = x * r0.xyz + y * r1.xyz
    // + z * r2.xyz + (r0.w, r1.w, r2.w): row k holds the image of model axis k
    // (an earlier version had the transpose: rotated props turned by -yaw).
    [r(0, 0), r(0, 1), r(0, 2), r(1, 0), r(1, 1), r(1, 2), r(2, 0), r(2, 1), r(2, 2), r(0, 3) - origin[0], r(1, 3) - origin[1], r(2, 3) - origin[2]]
}

impl FrameList {
    /// The main camera: 16:9 and the most draws (the frame's shadow and
    /// reflection viewports are square [V frame 1700]).
    pub fn main_camera(&self) -> Option<&Camera> {
        if let Some(m) = self.main {
            return self.cameras.iter().find(|c| c.viewport == m);
        }
        let count = |c: &Camera| self.draws.iter().filter(|d| d.viewport == c.viewport).count();
        let wide = |c: &&Camera| c.proj[0] > 0.0 && (c.proj[5] / c.proj[0] - 16.0 / 9.0).abs() < 0.01;
        self.cameras.iter().filter(wide).max_by_key(|c| count(c)).or_else(|| self.cameras.iter().max_by_key(|c| count(c)))
    }

    /// The draws the main pass draws: those of every viewport with the main
    /// camera's view and projection. In a test run the main view's passes 65, 67,
    /// 70, 71, 73-75 each have their own viewport object with one camera;
    /// An earlier version drew only one of them (the props only) [V camera.txt].
    pub fn main_draws(&self) -> impl Iterator<Item = &DrawRecord> {
        let vps = self.main_viewports();
        self.draws.iter().filter(move |d| vps.contains(&d.viewport))
    }

    /// The viewports whose camera is the main camera (view and projection).
    pub fn main_viewports(&self) -> Vec<u64> {
        let main = self.main_camera().copied();
        let same = |a: &Mat, b: &Mat| a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-4 * (1.0 + x.abs()));
        self.cameras.iter().filter(|c| main.is_some_and(|m| same(&c.view, &m.view) && same(&c.proj, &m.proj))).map(|c| c.viewport).collect()
    }
}

/// The hand-over between the recording side (the title's render thread via
/// the probe's writer) and the pass: the newest finished list; an older
/// one not yet drawn is replaced (the pass draws the latest).
#[derive(Default)]
pub struct Lists {
    ready: Mutex<Option<FrameList>>,
    published: std::sync::atomic::AtomicU64,
    dropped: std::sync::atomic::AtomicU64,
}

impl Lists {
    pub fn publish(&self, list: FrameList) {
        use std::sync::atomic::Ordering::Relaxed;
        if let Ok(mut r) = self.ready.lock() {
            let mut list = list;
            if let Some(old) = r.take() {
                // Its new geometries were never uploaded: keep them.
                list.new_geoms.extend(old.new_geoms);
                list.ahead_geoms.extend(old.ahead_geoms);
                self.dropped.fetch_add(1, Relaxed);
            }
            *r = Some(list);
            self.published.fetch_add(1, Relaxed);
        }
    }

    pub fn take(&self) -> Option<FrameList> {
        self.ready.lock().ok()?.take()
    }

    /// (published, replaced before being drawn).
    pub fn counts(&self) -> (u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        (self.published.load(Relaxed), self.dropped.load(Relaxed))
    }
}

/// The process's lists.
pub fn lists() -> &'static Lists {
    static L: std::sync::OnceLock<Lists> = std::sync::OnceLock::new();
    L.get_or_init(Lists::default)
}

/// Decoded geometries (or their GPU buffers) by key, with the last frame
/// each was used; evicts what no frame used for `keep` frames.
pub struct GeometryCache<T> {
    items: HashMap<GeomKey, (T, u64)>,
    pub made: u64,
    pub evicted: u64,
}

impl<T> Default for GeometryCache<T> {
    fn default() -> Self {
        GeometryCache { items: HashMap::new(), made: 0, evicted: 0 }
    }
}

impl<T> GeometryCache<T> {
    /// The item for `key`, made by `make` the first time (None: not cached).
    pub fn get_or_make(&mut self, key: GeomKey, frame: u64, make: impl FnOnce() -> Option<T>) -> Option<&T> {
        if !self.items.contains_key(&key) {
            let v = make()?;
            self.made += 1;
            self.items.insert(key, (v, frame));
        }
        let e = self.items.get_mut(&key)?;
        e.1 = frame;
        Some(&e.0)
    }

    pub fn remove(&mut self, key: &GeomKey) -> Option<T> {
        self.items.remove(key).map(|(v, _)| v)
    }

    pub fn contains(&self, key: &GeomKey) -> bool {
        self.items.contains_key(key)
    }

    pub fn evict(&mut self, frame: u64, keep: u64) {
        let before = self.items.len();
        self.items.retain(|_, (_, used)| frame.saturating_sub(*used) <= keep);
        self.evicted += (before - self.items.len()) as u64;
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// Bytes per vertex of the pass's vertex buffer: position, normal (float3
/// each), uv0 (float2).
pub const VERTEX_BYTES: usize = 32;

/// Interleaves a decoded mesh for the pass (missing normals: +z; missing
/// uv: 0).
pub fn pack_vertices(pos: &[[f32; 3]], normal: &[[f32; 3]], uv: &[[f32; 2]]) -> Vec<f32> {
    let mut v = Vec::with_capacity(pos.len() * 8);
    for (i, p) in pos.iter().enumerate() {
        v.extend_from_slice(p);
        v.extend_from_slice(normal.get(i).unwrap_or(&[0.0, 0.0, 1.0]));
        v.extend_from_slice(uv.get(i).unwrap_or(&[0.0, 0.0]));
    }
    v
}

/// The per-draw root constants: the relative world-view-projection, the
/// world's 3x3 rows (normals), the flat colour and whether a texture is
/// bound (as in [`SHADER`]'s `Draw`).
pub fn draw_constants(rec: &DrawRecord, cam: &Camera) -> [f32; 32] {
    let mvp = mul(&rec.world, &cam.view_proj_relative());
    let mut c = [0f32; 32];
    c[..16].copy_from_slice(&mvp);
    for r in 0..3 {
        c[16 + 4 * r..19 + 4 * r].copy_from_slice(&rec.world[4 * r..4 * r + 3]);
    }
    let (rgb, textured) = match rec.albedo {
        Albedo::Flat(c) => (c, 0.0),
        Albedo::Pack { .. } => ([1.0; 3], 1.0),
    };
    c[28..31].copy_from_slice(&rgb);
    c[31] = textured;
    c
}

/// The pass's one pipeline: SM 5.1 HLSL compiled at run time by FXC (no dxc
/// dependency), as the presenter's compositor is. Row vectors as the
/// title's matrices; root constants hold [`draw_constants`]. Albedo is the
/// flat colour for now (pack textures are the next step); `albedo.w`
/// is kept for it.
pub const SHADER: &str = r#"
#define RS "RootConstants(num32BitConstants=32, b0), RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)"
cbuffer Draw : register(b0) {
    row_major float4x4 mvp;     // camera-relative world * view rotation * projection
    float4 world3[3];           // the world's 3x3 rows (xyz)
    float4 albedo;              // rgb flat colour, w = 1 when a texture will be bound
};
struct VsIn { float3 pos : POSITION; float3 normal : NORMAL; float2 uv : TEXCOORD0; };
struct VsOut { float4 pos : SV_Position; float3 normal : NORMAL; float2 uv : TEXCOORD0; };
[RootSignature(RS)]
VsOut vs(VsIn i) {
    VsOut o;
    o.pos = mul(float4(i.pos, 1), mvp);
    o.normal = i.normal.x * world3[0].xyz + i.normal.y * world3[1].xyz + i.normal.z * world3[2].xyz;
    o.uv = i.uv;
    return o;
}
[RootSignature(RS)]
float4 ps(VsOut i) : SV_Target {
    float3 sun = normalize(float3(0.4, -0.3, 0.866));
    float3 n = normalize(i.normal);
    float lit = 0.3 + 0.7 * abs(dot(n, sun));   // two-sided: meshlet winding is not known [U]
    return float4(albedo.rgb * lit, 1);
}
"#;

// ---- the device side ----

use std::mem::ManuallyDrop;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::{Interface, PCSTR, Result};

/// A geometry for the pass: [`pack_vertices`] floats and u32 triangle
/// indices (decoded on the CPU), or the title's raw vertex buffer
/// bytes with indices into it (decoded by [`RAW_SHADER`]).
#[derive(Clone, Debug, Default)]
pub struct HostGeometry {
    pub vertices: Vec<f32>,
    pub indices: Vec<u32>,
    pub raw_vb: Option<Vec<u8>>,
    /// `XNW_ZEROCOPY` (ZC2): the raw vertex buffer's bytes left in title
    /// memory at the key's `vb_at` (`raw_vb` then holds only its header):
    /// the pass binds them in place. 0: all in `raw_vb`.
    pub vb_place: u64,
}


/// The same pass decoding the title's own vertex buffer in the vertex
/// shader (the CPU decoder in apps/xnw worldprobe.rs is
/// the reference). The index buffer holds vertex-buffer indices (the
/// meshlets expanded once on the CPU). Every read at a title-supplied
/// offset goes through `ld` / `pl`, which return 0 past the buffer's byte
/// size (root constant, set from the buffer we made), and the SRVs are raw
/// views sized to the buffers, so the hardware bound holds too. Bone
/// indices are checked against the palette's bone count.
pub const RAW_SHADER: &str = r#"
#define RS "RootConstants(num32BitConstants=58, b0), DescriptorTable(SRV(t0), visibility=SHADER_VISIBILITY_VERTEX), DescriptorTable(SRV(t1), visibility=SHADER_VISIBILITY_VERTEX), DescriptorTable(SRV(t2), visibility=SHADER_VISIBILITY_VERTEX), DescriptorTable(SRV(t3, numDescriptors=1024), visibility=SHADER_VISIBILITY_PIXEL), CBV(b1, visibility=SHADER_VISIBILITY_PIXEL), StaticSampler(s0, filter=FILTER_ANISOTROPIC, maxAnisotropy=8, visibility=SHADER_VISIBILITY_PIXEL)"
cbuffer Draw : register(b0) {
    row_major float4x4 mvp;
    float4 world3[3];
    float4 albedo;
    uint vbBytes;     // 32: the vertex buffer's size in bytes
    uint palFirst;    // 33: first float4 of this draw's palette in t1
    uint palBones;    // 34: its bones (0: not skinned)
    uint flags;       // 35: bit 0 wireframe colour
    uint palBytes;    // 36: t1's size in bytes
    uint instFirst;   // 37: first instance (12 floats each) in t2
    uint instCount;   // 38: instances (0: not instanced)
    uint instBytes;   // 39: t2's size in bytes
    float4 sunDir;    // 40: the timecycle's light direction (xyz), ambient (w)
    float4 sunCol;    // 44: its colour (rgb), w = 1 when set (else the fixed sun)
    uint texIndex;    // 48: albedo entry in our table t3 (0: flat), checked on the CPU
    uint matFlags;    // 49: bit 0 alpha clip (a < 0.5 discards), bit 1 sRGB view
    uint layerTex;    // 50: four layer entries of t3, 8 bits each (0: none)
    uint layerFlags;  // 51: bit 0 layered, bits 1-2 colour stream (1: COLOR0 at 0x30, 2: COLOR1 at 0x34), bit 3 debug colours
    float4 fog;       // 52: fog colour (rgb), 1 / distance (w; 0: off)
    uint remapAt;     // 56: dword index in t1 of remap[base] (0xFFFFFFFF: none)
    uint skinPad;     // 57
};
// The frame's Light buffer (lighting.rs `constants`): light dir (towards it, w 1: on),
// radiance colour x mult (w: exposure), ambient sky (w 1: the old look), ambient ground,
// Hable A B C D, E F W 1/Hable(W), sky radiance scale (x).
cbuffer Light : register(b1) { float4 lDir; float4 lRad; float4 lSky; float4 lGround; float4 tone0; float4 tone1; float4 lPad; };
float3 hableC(float3 x) { return (x * (tone0.x * x + tone0.z * tone0.y) + tone0.w * tone1.x) / (x * (tone0.x * x + tone0.y) + tone0.w * tone1.y) - tone1.x / tone1.y; }
bool titleChain() { return lDir.w > 0.5 && lSky.w < 0.5; }
// Linear radiance -> exposure -> Hable / Hable(W) -> pow 1/2.2.
float3 tonemap(float3 hdr) { return pow(saturate(hableC(max(hdr * lRad.w, 0)) * tone1.w), 1.0 / 2.2); }
// A stand-in colour made for the screen (sky, fog) as radiance: linearised, x the sky scale.
float3 skyRadiance(float3 c) { return pow(max(c, 0), 2.2) * lPad.x; }
// Under the title chain the fog colour (the sky's horizon) is tonemapped like the sky.
float4 fogged(float3 c, float w) { float3 f = titleChain() ? tonemap(skyRadiance(fog.rgb)) : fog.rgb; return float4(fog.w > 0 ? lerp(c, f, 1 - exp(-w * fog.w)) : c, 1); }
// Albedo as the title stores it -> lit in that space -> exposure -> Hable / Hable(W) -> pow 1/2.2 (one conversion).
float3 titleLit(float3 base, float3 n) {
    float nl = saturate(dot(n, normalize(lDir.xyz + 1e-9)));
    float3 amb = lerp(lGround.rgb, lSky.rgb, saturate(n.z * 0.5 + 0.5));
    float3 hdr = base * (lRad.rgb * (nl * 0.31830989) + amb);
    return tonemap(hdr);
}
ByteAddressBuffer vb : register(t0);
ByteAddressBuffer pal : register(t1);
ByteAddressBuffer inst : register(t2);
Texture2D albedoTex[1024] : register(t3); // TEX_SLOTS (test `albedo_table_matches_tex_slots`)
SamplerState wrapAniso : register(s0);
// `?:` runs both sides: the condition picks the address (0 is inside every
// view), then the value; a load past a view is never issued (GBV id 1005, HANG1).
float3 il(uint i, uint k) { uint a = (i * 12 + k) * 4; bool ok = a + 12 <= instBytes && a < 0x7FFFFFF0; float3 v = asfloat(inst.Load3(ok ? a : 0)); return ok ? v : float3(0, 0, 0); }
uint ld(uint a) { bool ok = a + 4 <= vbBytes && a < 0x7FFFFFF0; uint v = vb.Load(ok ? a : 0); return ok ? v : 0; }
float4 pl(uint i) { uint a = i * 16; bool ok = a + 16 <= palBytes && a < 0x7FFFFFF0; float4 v = asfloat(pal.Load4(ok ? a : 0)); return ok ? v : float4(0, 0, 0, 0); }
uint pu(uint i) { uint a = i * 4; bool ok = a + 4 <= palBytes && a < 0x7FFFFFF0; uint v = pal.Load(ok ? a : 0); return ok ? v : 0; }
uint at(uint w, uint v) { return ((w >> 3) & 0x3C) * v + ((w >> 7) & 0x1FFFFFC); }
float3 position(uint v) {
    uint w = ld(0), a = at(w, v);
    if ((w & 15) == 3) return float3(asfloat(ld(a)), asfloat(ld(a + 4)), asfloat(ld(a + 8)));
    uint A = ld(a), B = ld(a + 4);
    float3 c = float3(A >> 8, (A & 0xFF) | ((B & 0xFFFF) << 8), B >> 16);
    c = c * float3(asfloat(ld(0x70)), asfloat(ld(0x74)), asfloat(ld(0x78))) + float3(asfloat(ld(0x60)), asfloat(ld(0x64)), asfloat(ld(0x68)));
    uint mode = ld(0x6C);
    return mode == 0 ? c.zyx : mode == 1 ? float3(c.y, c.z, c.x) : c;
}
float3 normal(uint v) {
    uint w = ld(8);
    if (((w >> 3) & 0x3C) == 0) return float3(0, 0, 1);
    uint n = ld(at(w, v));
    // Format 3 is three floats [V].
    if ((w & 15) == 3) { uint a = at(w, v); return normalize(float3(asfloat(ld(a)), asfloat(ld(a + 4)), asfloat(ld(a + 8))) + 1e-6); }
    if ((w & 15) == 7) return normalize(float3(n & 0x3FF, (n >> 10) & 0x3FF, (n >> 20) & 0x3FF) / 1023.0 * 2.0 - 1.0 + 1e-6);
    float2 e = (n >> 31) != 0 ? float2(n & 0x3FF, (n >> 10) & 0x3FF) * (2.0 / 1023.0) - 1.0 : float2(n & 0xFF, (n >> 8) & 0xFF) * (2.0 / 255.0) - 1.0;
    float z = 1.0 - abs(e.x) - abs(e.y);
    float t = saturate(-z);
    e += float2(e.x >= 0 ? -t : t, e.y >= 0 ? -t : t);
    return normalize(float3(e, z) + 1e-6);
}
float2 uv(uint v) {
    uint w = ld(12);
    uint a = at(w, v), x = ld(a);
    if ((w & 15) == 2 || (w & 15) == 3) return float2(asfloat(x), asfloat(ld(a + 4)));
    return max(float2(int(x << 16) >> 16, int(x) >> 16) / 32767.0, -1.0) * asfloat(ld(0x7C));
}
float3 bone(uint b, float3 p, float w4) {
    float4 r0 = pl(palFirst + 3 * b), r1 = pl(palFirst + 3 * b + 1), r2 = pl(palFirst + 3 * b + 2);
    float4 q = float4(p, w4);
    return float3(dot(r0, q), dot(r1, q), dot(r2, q));
}
struct VsOut { float4 pos : SV_Position; float3 normal : NORMAL; float2 uv : TEXCOORD0; float4 col : COLOR0; };
[RootSignature(RS)]
VsOut vs(uint v : SV_VertexID, uint instance : SV_InstanceID) {
    float3 p = position(v), n = normal(v);
    uint ws = ld(0x18), is = ld(0x20);
    if (palBones != 0 && ((ws >> 3) & 0x3C) != 0 && ((is >> 3) & 0x3C) != 0) {
        // As the title's programs do: index = i & 0xFFF (u8x4 or u32x4);
        // Ped*: k = remap[base + k] & 0xFFF when base != -1, weights / their sum;
        // generic TransformSkin: raw w/255, w0 always, the rest up to the first
        // zero weight. No bound against curCount (reads stay inside our copy).
        bool ped = (flags & 2) != 0;
        float3 sp = 0, sn = 0;
        float total = 0;
        bool stop = false;
        [unroll] for (uint k = 0; k < 2; k++) {
            uint wsk = ld(0x18 + 4 * k), isk = ld(0x20 + 4 * k);
            if (((wsk >> 3) & 0x3C) == 0 || ((isk >> 3) & 0x3C) == 0) continue;
            uint wb = ld(at(wsk, v)), ia = at(isk, v);
            [unroll] for (uint j = 0; j < 4; j++) {
                float wt = ((wb >> (8 * j)) & 0xFF) / 255.0;
                // Flags bit 8: a UINT8x4 index is its byte (the title's BUF_FMT_8_8_8_8_UINT load).
                uint b = ((isk & 15) == 5 ? (ld(ia) >> (8 * j)) & ((flags & 0x100) != 0 ? 0xFF : 0xFFF) : ld(ia + 4 * j)) & 0xFFF;
                if (remapAt != 0xFFFFFFFF) b = pu(remapAt + b) & 0xFFF;
                if (!ped && (k + j) > 0 && (stop || wt == 0)) { stop = true; continue; }
                if (ped && wt == 0) continue;
                sp += wt * bone(b, p, 1);
                sn += wt * bone(b, n, 0);
                total += wt;
            }
        }
        if (ped) { if (total > 0) p = sp / total; } else { p = sp; }
        n = normalize(sn + 1e-6);
    }
    if (instCount != 0) {
        uint i = instFirst + min(instance, instCount - 1);
        float3 ax = il(i, 0), ay = il(i, 3), az = il(i, 6), tr = il(i, 9);
        p = p.x * ax + p.y * ay + p.z * az + tr;
        n = n.x * ax + n.y * ay + n.z * az;
    }
    VsOut o;
    o.pos = mul(float4(p, 1), mvp);
    o.normal = n.x * world3[0].xyz + n.y * world3[1].xyz + n.z * world3[2].xyz;
    o.uv = uv(v);
    // The vertex colour C (unorm8x4) of the chosen stream, for layer weights.
    o.col = float4(0, 0, 0, 0);
    uint chan = (layerFlags >> 1) & 3;
    if ((layerFlags & 1) != 0 && chan != 0) {
        uint cw = ld(chan == 1 ? 0x30 : 0x34);
        if (((cw >> 3) & 0x3C) != 0) { uint c = ld(at(cw, v)); o.col = float4(c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF, c >> 24) / 255.0; }
    }
    return o;
}
// A row drawn from the title's mesh program (`bridgems`): t0 is the
// frame's record ring, vbBytes its size; palFirst this row's first record
// byte, palBytes a record's bytes; instFirst max vertices | vertex stride << 16;
// instBytes position row | uv row << 8 | uv channel << 16 | uv on << 24;
// remapAt normal row | channel << 8 | on << 16 (bridge_ms::Join). A vertex at
// or past its group's count gets a NaN position (dropped).
float4 rrow(uint a, uint row, uint ch) { uint b = a + 16 * row + 4 * ch; return asfloat(uint4(ld(b), ld(b + 4), ld(b + 8), ld(b + 12))); }
[RootSignature(RS)]
VsOut vs_rec(uint id : SV_VertexID) {
    uint maxv = max(instFirst & 0xFFFF, 1), stride = instFirst >> 16;
    uint rec = palFirst + (id / maxv) * palBytes, v = id % maxv;
    uint a = rec + 16 + v * stride;
    VsOut o;
    o.pos = v < ld(rec) ? rrow(a, instBytes & 0xFF, 0) : asfloat(0x7FC00000).xxxx;
    o.uv = (instBytes >> 24) != 0 ? rrow(a, (instBytes >> 8) & 0xFF, (instBytes >> 16) & 3).xy : float2(0, 0);
    o.normal = (remapAt >> 16) != 0 ? rrow(a, remapAt & 0xFF, (remapAt >> 8) & 3).xyz : float3(0, 0, 1);
    o.col = float4(0, 0, 0, 0);
    return o;
}
#ifdef GBUF
// `gbuf=on` (gbuf.rs): the title's G-buffer beside the picture. shade() leaves
// its albedo (linear) and unit normal here.
static float3 gbBase = 0;
static float3 gbN = float3(0, 0, 1);
// The title's RT1 (PS_DeferredTextured ff2c46ab 0x1d0..0x2ec): octahedral normal,
// 12 + 12 bits, word = y' << 12 | x', as (0, byte 0, byte 1, byte 2) / 255.
float4 gbufNormal(float3 n) {
    float s = max(abs(n.x) + abs(n.y) + abs(n.z), 1e-6);
    float f = saturate(-n.z / s);
    float2 p = n.xy / s;
    p += float2(p.x >= 0 ? f : -f, p.y >= 0 ? f : -f);
    uint2 q = min(uint2(p * 2047.5 + 2047.5), 4095);
    uint w = (q.y << 12) | q.x;
    return float4(0, w & 255, (w >> 8) & 255, (w >> 16) & 255) / 255.0;
}
#endif
float4 shade(VsOut i) {
    if (flags & 1) return float4(0.95, 0.95, 0.9, 1);
    // ID mode (world-render --pick): the draw's index + 1 as the colour.
    if (flags & 4) { uint id = skinPad; return float4((id & 255) / 255.0, ((id >> 8) & 255) / 255.0, ((id >> 16) & 255) / 255.0, 1); }
    // The title's 2x2-quad sample mask (LOD cross-fade): lanes in bits 4-7, bit 8 on.
    if ((layerFlags & 0x100) != 0) {
        uint lane = (uint(i.pos.x) & 1) | ((uint(i.pos.y) & 1) << 1);
        if (((layerFlags >> (4 + lane)) & 1) == 0) discard;
    }
    float3 base = albedo.rgb;
    float a = 1;
    bool chain = titleChain();
    if (texIndex != 0) {
        // Our own table only; the index is checked on the CPU and here.
        float4 t = albedoTex[min(texIndex, 1023)].Sample(wrapAniso, i.uv);
        // a = Diffuse.a * matCB[0]; matCB[0] is not read at the hook (no
        // single cheap read: the alpha-scale variable is unnamed), so 1.0.
        a = t.a;
        if ((matFlags & 1) != 0 && a < 0.5) discard;
        // The title stores the sampled value as albedo (an
        // sRGB view already decodes to linear): no re-encode; the final pow
        // 1/2.2 is the one conversion. `light=old` keeps the older re-encode.
        base = (!chain && (matFlags & 2) != 0) ? pow(saturate(t.rgb), 1.0 / 2.2) : t.rgb;
    }
    // First cut: w = (C.r, C.g, C.b, sat(1 - r - g - b)), normalised;
    // albedo = sum w_i * layer_i (mask texture, C.a mix and height terms later).
    if ((layerFlags & 1) != 0 && ((layerFlags >> 1) & 3) != 0) {
        float4 C = i.col;
        float4 w = float4(C.rgb, saturate(1 - C.r - C.g - C.b));
        float s = dot(w, 1);
        w = s > 1e-4 ? w / s : float4(0, 0, 0, 1);
        const float3 dbg[4] = { float3(0.80, 0.30, 0.20), float3(0.25, 0.70, 0.30), float3(0.25, 0.35, 0.85), float3(0.85, 0.80, 0.30) };
        float3 acc = 0;
        [unroll] for (uint k = 0; k < 4; k++) {
            uint ti = (layerTex >> (8 * k)) & 0xFF;
            float3 c = (layerFlags & 8) != 0 ? dbg[k] : (ti != 0 ? albedoTex[min(ti, 255)].Sample(wrapAniso, i.uv).rgb : base);
            acc += w[k] * c;
        }
        base = acc;
    }
    float3 n = normalize(i.normal + 1e-6);
#ifdef GBUF
    // Linear albedo: the old path re-encoded an sRGB view's value (undone here).
    gbBase = (!chain && texIndex != 0 && (matFlags & 2) != 0) ? pow(saturate(base), 2.2) : base;
    gbN = n;
#endif
#ifdef LAB_VIEW
    // The lab's view modes (xnw-lab prepends the define). Geometry first:
    // 1 shaded by normal, no textures; 2 bones per draw (grey: none; hue by count);
    // 3 draw class (static blue-grey, instanced green, skinned ped red, skinned other
    // orange); 4 normals; 5 depth (reverse Z); 6 albedo only; 7 flat (no table entry).
    float3 L = normalize(float3(0.4, -0.3, 0.866));
    float shade = 0.35 + 0.65 * abs(dot(n, L));
    if (LAB_VIEW == 1) return float4(shade.xxx * 0.85, 1);
    if (LAB_VIEW == 2) { if (palBones == 0) return float4(shade.xxx * 0.3, 1); float h = frac(palBones * 0.161); return float4(shade * saturate(abs(h * 6 - float3(3, 2, 4)) * float3(1, -1, -1) + float3(-1, 2, 2)), 1); }
    if (LAB_VIEW == 3) { float3 c = instCount != 0 ? float3(0.3, 0.8, 0.35) : palBones == 0 ? float3(0.55, 0.6, 0.7) : (flags & 2) != 0 ? float3(0.9, 0.25, 0.2) : float3(0.95, 0.6, 0.15); return float4(c * shade, 1); }
    if (LAB_VIEW == 4) return float4(n * 0.5 + 0.5, 1);
    if (LAB_VIEW == 5) return float4(pow(saturate(i.pos.z * 256), 0.25).xxx, 1);
    if (LAB_VIEW == 6) return float4(base, 1);
    if (LAB_VIEW == 7) return texIndex == 0 ? float4(albedo.rgb, 1) : float4(0.25, 0.25, 0.25, 1);
#endif
    // Alpha out: the texture's (decals blend with the title's state, SRC_ALPHA).
    if (chain) { float4 o = fogged(titleLit(base, n), i.pos.w); o.a = a; return o; }
    if (sunCol.w > 0.5) {
        float3 d = normalize(sunDir.xyz + 1e-6);
        float3 lit = sunDir.w + (1.0 - sunDir.w) * abs(dot(n, d)) * saturate(sunCol.rgb);
        return fogged(base * lit, i.pos.w);
    }
    float3 sun = normalize(float3(0.4, -0.3, 0.866));
    float lit = 0.3 + 0.7 * abs(dot(n, sun));
    return fogged(base * lit, i.pos.w);
}
// One return in the entry: the newer WARP (1.0.22) removes the device
// (no layer message) on a pixel shader that returns inside a branch.
#ifdef GBUF
struct GbufOut { float4 c : SV_Target0; float4 albedo : SV_Target1; float4 normal : SV_Target2; };
[RootSignature(RS)]
GbufOut ps(VsOut i) { GbufOut o; o.c = shade(i); o.albedo = float4(gbBase, 0); o.normal = gbufNormal(gbN); return o; }
#else
[RootSignature(RS)]
float4 ps(VsOut i) : SV_Target { return shade(i); }
#endif
"#;

/// A geometry on the device: vertex and index buffers (upload heap, read in
/// place: the in-title step can move them to the default heap) and their
/// views; a raw geometry has its SRV slot in the pass's heap instead of a
/// vertex view.
pub struct GpuGeometry {
    /// Its range in a device page (page, offset, bytes): vertex bytes,
    /// then indices at a 256-byte offset (an earlier version made one buffer per geometry).
    place: (u32, u64, u64),
    vbv: D3D12_VERTEX_BUFFER_VIEW,
    ibv: D3D12_INDEX_BUFFER_VIEW,
    indices: u32,
    bytes: u64,
    raw: Option<(u32, u32)>,
    /// The largest index + 1 (a pre-skin item must have that many vertices).
    top: u32,
    /// Its vertex buffer read in place (ZC2): its entry in `Pass::zc_geoms`.
    in_place: Option<(u64, u64, u64)>,
}

/// Device pages (default heap, 64 MiB; a larger geometry gets a page
/// of its own size) sub-allocated first fit at 256-byte granularity; freed
/// ranges coalesce. One CreateCommittedResource per page, not per geometry.
#[derive(Default)]
pub struct Pages {
    pages: Vec<(ID3D12Resource, u64, Vec<(u64, u64)>)>,
}

pub const GEOM_PAGE: u64 = 64 << 20;

impl Pages {
    fn buffer(&self, page: u32) -> &ID3D12Resource {
        &self.pages[page as usize].0
    }

    /// Bytes of all pages.
    pub fn bytes(&self) -> u64 {
        self.pages.iter().map(|p| p.1).sum()
    }

    fn alloc(&mut self, device: &ID3D12Device, bytes: u64) -> Result<(u32, u64, u64)> {
        let size = bytes.next_multiple_of(256);
        for (i, (_, _, free)) in self.pages.iter_mut().enumerate() {
            if let Some(k) = free.iter().position(|r| r.1 >= size) {
                let (at, len) = free[k];
                if len == size {
                    free.remove(k);
                } else {
                    free[k] = (at + size, len - size);
                }
                return Ok((i as u32, at, size));
            }
        }
        let page = size.max(GEOM_PAGE);
        let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: page, Height: 1, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_UNKNOWN, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: descriptions outlive the call.
        unsafe { device.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut out)? };
        let free = if page > size { vec![(size, page - size)] } else { Vec::new() };
        self.pages.push((out.expect("resource on success"), page, free));
        Ok(((self.pages.len() - 1) as u32, 0, size))
    }

    fn free(&mut self, (page, at, size): (u32, u64, u64)) {
        let Some((_, _, free)) = self.pages.get_mut(page as usize) else { return };
        let k = free.partition_point(|r| r.0 < at);
        free.insert(k, (at, size));
        // Coalesce with the next, then the previous.
        if k + 1 < free.len() && free[k].0 + free[k].1 == free[k + 1].0 {
            free[k].1 += free[k + 1].1;
            free.remove(k + 1);
        }
        if k > 0 && free[k - 1].0 + free[k - 1].1 == free[k].0 {
            free[k - 1].1 += free[k].1;
            free.remove(k);
        }
    }
}

/// 4: one-second cost window: hook ms per frame (the recorder reports
/// it), our pass ms per frame, upload and pipeline ms, frames over 50 ms.
#[derive(Default)]
struct CostWindow {
    start: Option<std::time::Instant>,
    hook: Vec<f64>,
    pass: Vec<f64>,
    upload: f64,
    pipelines: f64,
    slow: u32,
}

static COST: Mutex<CostWindow> = Mutex::new(CostWindow { start: None, hook: Vec::new(), pass: Vec::new(), upload: 0.0, pipelines: 0.0, slow: 0 });

fn p95(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[((v.len() - 1) as f64 * 0.95) as usize]
}

/// The recorder's hook time of one frame (summed over the recording threads).
pub fn cost_hook(ms: f64) {
    if let Ok(mut c) = COST.lock() {
        c.hook.push(ms);
        c.slow += u32::from(ms > 50.0);
    }
}

/// Our pass's time of one frame; once a second the summary line.
fn cost_pass(ms: f64, upload: f64, pipelines: f64) {
    let Ok(mut c) = COST.lock() else { return };
    c.pass.push(ms);
    c.upload += upload;
    c.pipelines += pipelines;
    c.slow += u32::from(ms > 50.0);
    let start = *c.start.get_or_insert_with(std::time::Instant::now);
    if start.elapsed().as_secs_f64() >= 1.0 {
        let (hook_n, pass_n) = (c.hook.len(), c.pass.len());
        let (hook_sum, pass_sum): (f64, f64) = (c.hook.iter().sum(), c.pass.iter().sum());
        let (h95, p95v) = (p95(&mut c.hook), p95(&mut c.pass));
        xnw_core::info!(
            "world render: cost: hook {:.1} ms/frame (p95 {h95:.1}, {hook_n} frames), pass {:.1} ms/frame (p95 {p95v:.1}, {pass_n} frames), upload {:.1} ms, pipelines {:.1} ms on the frame path, frames over 50 ms {}",
            hook_sum / hook_n.max(1) as f64,
            pass_sum / pass_n.max(1) as f64,
            c.upload,
            c.pipelines,
            c.slow
        );
        *c = CostWindow { start: Some(std::time::Instant::now()), ..Default::default() };
    }
}

/// A D3D12 object moved to or from the pipeline worker thread (the
/// device and pipeline objects are free-threaded).
pub struct SendBox<T>(pub T);
// SAFETY: D3D12 devices, root signatures, blobs and pipeline states may be
// used from any thread.
unsafe impl<T> Send for SendBox<T> {}

/// The previous list's decals (record, absolute translation) not in this
/// list, moved to this camera; `held` becomes this list's decals (a decal is
/// held for one list only: the union of two consecutive lists).
fn decal_union(held: &mut Vec<(DrawRecord, [f32; 3])>, draws: &[&DrawRecord], cam: &Camera) -> Vec<DrawRecord> {
    let origin = camera_position(&cam.view);
    let key = |g: &GeomKey, a: [f32; 3]| (g.hash, g.mesh_at, a.map(|v| (v * 100.0).round() as i64));
    let now: Vec<(DrawRecord, [f32; 3])> = draws
        .iter()
        .filter(|r| DECAL_STAGES.contains(&r.stage) && r.inst_count == 0)
        .map(|r| (**r, [r.world[12] + origin[0], r.world[13] + origin[1], r.world[14] + origin[2]]))
        .collect();
    let present: std::collections::HashSet<_> = now.iter().map(|(r, a)| key(&r.geom, *a)).collect();
    let out = held
        .iter()
        .filter(|(r, a)| !present.contains(&key(&r.geom, *a)))
        .map(|(r, a)| {
            let mut r = *r;
            (r.world[12], r.world[13], r.world[14]) = (a[0] - origin[0], a[1] - origin[1], a[2] - origin[2]);
            r.viewport = cam.viewport;
            r
        })
        .collect();
    *held = now;
    out
}

/// The main view's draws the pass left out (stage filter, peds),
/// never below zero.
fn filtered_count(main: usize, kept: usize) -> usize {
    main.saturating_sub(kept)
}

/// What a [`Pass::draw`] did.
#[derive(Debug, Default, Clone)]
pub struct PassStats {
    /// Geometry copies skipped (outside the page or the staging).
    pub copies_refused: usize,
    /// `XNW_ZEROCOPY` (ZC2): geometries read in place, and those dropped
    /// this frame because their window went or the title remapped their
    /// pages (read again).
    pub in_place_geoms: usize,
    pub in_place_dropped: usize,
    /// Rows drawn from the title's mesh programs (`bridgems`); of them with the title's pixel program.
    pub bridged: usize,
    pub bridged_title_ps: usize,
    pub draws: usize,
    /// Submissions of the frame (1, more with `chunk`), the longest fence
    /// wait of one and the longest timestamped GPU time of one (ms).
    pub submits: u32,
    pub submit_wait_ms: f64,
    pub submit_gpu_ms: f64,
    pub triangles: u64,
    pub skipped: usize,
    pub skinned: usize,
    /// Draws left out by the `pass=` filter.
    pub filtered: usize,
    /// Instanced draws and their instances.
    pub instanced: usize,
    /// Draws with a pipeline from the title's state.
    pub stated: usize,
    pub instances: u64,
    /// Re-uploads of a key already on the device (content changed).
    pub replaced: usize,
    /// Indices clamped (past MAX_INDICES or not a multiple of 3).
    pub clamped: u64,
    pub uploaded_geometries: usize,
    pub uploaded_bytes: u64,
    pub palette_bytes: u64,
    pub record_ms: f64,
    pub gpu_ms: f64,
    /// The texture table's time in the record (reads, copies, views).
    pub tex_ms: f64,
    /// Slow frames: time in uploads and in new pipelines, the
    /// pipelines made, geometries left for later frames.
    pub upload_ms: f64,
    /// Draws blended from layers.
    pub layered: usize,
    /// Draws with the title's sample mask (cross-fade).
    pub stippled: usize,
    /// Draws without geometry by reason: waiting for the frame's upload
    /// budget, waiting for device room, not delivered by the recorder; and
    /// geometries freed on the device this frame.
    pub missing: [usize; 3],
    pub evicted: usize,
    pub pso_ms: f64,
    pub new_psos: usize,
    /// 3: the measured upload cost (ms per upload, ms per MiB).
    pub upload_cost: (f64, f64),
    /// 2: ms the pipeline worker spent on the pipelines taken this frame (off the frame path).
    pub pso_ms_worker: f64,
    pub pending: usize,
    /// The sky was drawn.
    pub sky: bool,
    /// The water was drawn.
    pub water: bool,
    /// Skinned draws with the ped bone remap on.
    pub remapped: usize,
    /// Ped skin draws left out (`peds=on` draws them).
    pub peds_skipped: usize,
    /// Skinned non-ped draws left out (`skinned=on` draws them).
    pub skinned_skipped: usize,
    /// Decals drawn from the previous list (alternate-frame decals).
    pub decals_held: usize,
    /// The terrain's frame.
    pub terrain: terrain::TerrainStats,
    /// The main view's triangles with geometry on the device (0..1), and
    /// whether this frame's picture was held back (the last complete one shown).
    pub coverage: f32,
    pub held: bool,
}

const COLOUR: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;
const DEPTH: DXGI_FORMAT = DXGI_FORMAT_D32_FLOAT;
/// Shader-visible SRV slots: raw vertex buffers (slot 0 and 1 are the two
/// per-frame palette buffers).
/// The albedo table on top (geometry keeps its 16,128 slots).
const HEAP_SLOTS: u32 = 16384 - 256 + TEX_SLOTS;
/// Heap slots for raw vertex buffers (0-3 are the per-frame buffers).
pub(crate) const GEOM_SLOTS: u32 = 16384 - 4;
/// Main-view pass stages left out by default: 74 holds decals and road
/// markings lying on the ground of pass 65/71, which z-fight with it in
/// our single depth pass (in one capture the colour bands on the street are
/// pass 74 alone; passes 65 and 71 alone are clean). `pass=74` still shows it.
pub const DECAL_STAGES: [u32; 1] = [74];
/// The ground's stages (a census: uber_4lyr at 71/75), for
/// `layers_debug` on dumps without layer slots.
pub const LAYER_DEBUG_STAGES: [u32; 2] = [71, 75];
/// Stage 65 is the depth prepass (depth only): its draws repeat the
/// opaque G-buffer stages, so the default view leaves them out.
pub const PREPASS_STAGES: [u32; 1] = [65];
/// Submissions per frame that carry a pair of timestamps (`chunk` can make more; those go untimed).
const STAMP_PAIRS: u32 = 256;
/// Instances drawn per draw at most.
pub const MAX_INSTANCES: u32 = 65_536;
/// Largest index count drawn per geometry (clamp) and vertex buffer taken.
const MAX_INDICES: usize = 3 << 20;
const MAX_VB_BYTES: usize = 64 << 20;

/// Our world pass: the decoded-vertex pipeline and the raw-buffer
/// pipeline (solid and wireframe), one colour and depth target, a
/// geometry cache on the device.
pub struct Pass {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    raw_root: ID3D12RootSignature,
    raw_pso: ID3D12PipelineState,
    wire_pso: ID3D12PipelineState,
    heap: ID3D12DescriptorHeap,
    heap_step: u32,
    next_slot: u32,
    free_slots: Vec<u32>,
    /// Geometries whose vertex buffer is read in place (ZC2), by title
    /// address (then mesh address and hash): bytes, GPU address of the view, key.
    zc_geoms: std::collections::BTreeMap<(u64, u64, u64), (u64, u64, GeomKey)>,
    palettes: [Option<(ID3D12Resource, *mut u8, usize)>; 2],
    /// Per-frame instance buffers (heap slots 2 and 3), alternating.
    instances: [Option<(ID3D12Resource, *mut u8, usize)>; 2],
    palette_turn: usize,
    /// Per-frame Light constant buffers (256 B, mapped), alternating.
    light_cb: [Option<(ID3D12Resource, *mut u8)>; 2],
    /// `light=old`: the earlier lighting (bisecting).
    pub light_old: bool,
    /// Layer blending: 0 off, 1 COLOR0 (stream word 0x30), 2 COLOR1
    /// (0x34); `layers_debug`: ground stages without layer slots (old dumps)
    /// get four fixed layer colours, to judge the weights.
    pub layers: u32,
    pub layers_debug: bool,
    /// Apply the title's sample mask (`stipple=off`).
    pub stipple: bool,
    /// This frame's Light buffer (the sky and water passes bind it too).
    light_va: u64,
    colour: ID3D12Resource,
    _depth: ID3D12Resource,
    rtv: ID3D12DescriptorHeap,
    dsv: ID3D12DescriptorHeap,
    readback: ID3D12Resource,
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    fence_value: u64,
    /// Draws per submission (`--chunk N`; 0: the frame in one list):
    /// the list is closed, sent and waited for every N draws, so no single
    /// submission can run near the driver's 2 s timeout on the card.
    pub chunk: u32,
    /// A pair of timestamps per submission (query heap, readback), made at the first frame.
    stamps: Option<(ID3D12QueryHeap, ID3D12Resource)>,
    pub width: u32,
    pub height: u32,
    pub cache: GeometryCache<GpuGeometry>,
    /// Draw the wireframe over the solid picture.
    pub wire: bool,
    /// Draws write their list index + 1 as the colour (`world-render --pick`).
    pub id_mode: bool,
    /// Debug views: only this pass stage; skinning off.
    pub only_stage: Option<u32>,
    /// Draw stage 74 (decals) last, depth test GE, no depth write, biased.
    pub decals: bool,
    decal_pso: ID3D12PipelineState,
    pub skin: bool,
    /// Draw ped skin draws (`peds=on`); off: skipped.
    pub peds: bool,
    /// Draw skinned non-ped draws with more than one bone (screen:
    /// `skinned=on`; off: skipped (in a test run their bones put
    /// screen-size polygons in the picture). Single-bone ones (most map pieces,
    /// the road) always draw.
    pub skinned: bool,
    /// Skin rule bits into flags bits 8..15 (bit 0: UINT8x4 index = its byte); 0: the default maths.
    /// CPU side only: [`SKIN_IDENTITY_STATIC`], [`SKIN_RIGID_BONES`], [`SKIN_DRAW_TIME`].
    /// Live: `XNW_WORLD_RENDER=screen,skinrules=<hex>` ([`parse_skin_rules`]).
    pub skin_rules: u32,
    /// `decalhold=off` turns the two-list decal union off.
    pub decal_hold: bool,
    held_decals: Vec<(DrawRecord, [f32; 3])>,
    /// Albedo textures (off unless set: `tex.opts.mode`); the table is
    /// the top [`TEX_SLOTS`] of the heap.
    pub tex: textures::TexTable,
    /// Title render states applied (raw path); their pipelines.
    pub states: bool,
    state_psos: HashMap<RenderState, ID3D12PipelineState>,
    raw_shaders: (ID3DBlob, ID3DBlob),
    /// `gbuf=on`: the raw pixel entry compiled without `GBUF` (dual-source states).
    raw_plain_ps: Option<ID3DBlob>,
    /// Per-frame budgets (screen: bounded so a camera cut cannot
    /// stall a frame; offline: unbounded): upload bytes, new state pipelines.
    /// Geometries over the budget wait in `pending` (drawn from a later frame).
    pub pso_budget: usize,
    /// 2: screen mode makes title-state pipelines on a worker thread,
    /// asked for when a state is first seen in a list; draws use our default
    /// pipeline until theirs is ready (167 ms each on the presenting path).
    pub pso_async: bool,
    pso_jobs: Option<(std::sync::mpsc::Sender<RenderState>, std::sync::mpsc::Receiver<SendBox<(RenderState, Option<ID3D12PipelineState>, f64)>>)>,
    pso_asked: std::collections::HashSet<RenderState>,
    /// 3: screen mode sets the frame's upload budget from the measured
    /// cost (ms per upload, ms per MiB) to this many ms (None: bytes only).
    pub upload_ms_budget: Option<f64>,
    upload_cost: (f64, f64),
    /// The sky stand-in and fog (`sky=off`).
    pub sky: bool,
    sky_root: ID3D12RootSignature,
    sky_pso: ID3D12PipelineState,
    /// The water stand-in (`water=off`) and the uploaded grid (id, vb, ib, indices).
    pub water: bool,
    water_root: ID3D12RootSignature,
    water_pso: ID3D12PipelineState,
    water_gpu: Option<(u64, ID3D12Resource, ID3D12Resource, u32)>,
    /// What is on the device, what waits, the budgets ([geometry]).
    pub geom: geometry::GeomPolicy,
    /// The frame's upload buffer (mapped) and the copies recorded from it.
    staging: Option<(ID3D12Resource, *mut u8, usize)>,
    staged: usize,
    /// This frame's ms making device pages and staging buffers (left out
    /// of the upload cost sample: a one-off, not a per-MiB cost; CUT1).
    make_ms: f64,
    copies: Vec<(ID3D12Resource, ID3D12Resource, u64, u64, u64)>,
    /// The device pages geometry ranges live in.
    pages: Pages,
    /// The heightmap terrain (`terrain=off`): pipeline, own heap and table.
    pub terrain: terrain::TerrainPass,
    /// Imposter cards (`--imposters`, off by default).
    pub imposters: imposter::ImposterPass,
    /// The title's pixel programs on a list's bridged rows (`bridgeps`, off
    /// by default): on, what the last frame's record did, the objects its
    /// list uses (to the fence) and the grey stand-in texture.
    pub bridge_on: bool,
    pub bridge_stats: bridged::Stats,
    /// The bridged rows' textures are asked for although the rows are not
    /// drawn (`bridge_on` off): the replay's first frame warms them, so the
    /// watched frame with the rows reads no texels.
    pub bridge_warm: bool,
    /// The bridged rows' device objects (heaps, kept inputs, uploads), kept across frames.
    pub bridge_gpu: bridged::Gpu,
    bridge_keep: Vec<windows::core::IUnknown>,
    bridge_grey: Option<ID3D12Resource>,
    /// The words buffer (upload memory, mapped): the frame table's slot 86 for the bridged rows.
    bridge_words: Option<(ID3D12Resource, *mut u8, u64)>,
    /// The picture shown (a copy of `colour` from the last frame that was
    /// complete enough) and the cut watch (`hold=off`: `colour` is shown).
    shown: ID3D12Resource,
    pub cuts: cuts::CutWatch,
    /// The ped pre-skin compute pass (made at its first item).
    preskin: preskin_gpu::GpuPreskin,
    /// Rows drawn from the title's mesh programs (`bridgems`): set
    /// before [`Pass::draw`] (its rows by list index), kept across frames
    /// for its pipelines.
    pub bridge_ms: Option<bridge_ms::Batch>,
    /// The raw shader's source (`vs_rec` is compiled from it on demand),
    /// its record-drawing pipeline and the heap slot of the record ring.
    raw_source: String,
    /// `gbuf=on`: the title's G-buffer targets beside the colour target.
    gbuf: Option<gbuf::Targets>,
    rec_pso: Option<ID3D12PipelineState>,
    rec_slot: Option<u32>,
    /// The instance cull on the device (made at its first batch).
    pub cull: gpucull::GpuCull,
}

/// Screen-mode budgets per frame.
pub const SCREEN_UPLOAD_BUDGET: u64 = 32 << 20;
pub const SCREEN_PSO_BUDGET: usize = 2;
/// 3: screen mode's upload time per frame.
pub const SCREEN_UPLOAD_MS: f64 = 4.0;

/// Which raw-path pipeline is bound.
#[derive(Clone, Copy, PartialEq)]
enum Bound {
    Raw,
    Wire,
    Decal,
    State(RenderState),
}

// SAFETY: the mapped palette pointers are only used under the owner's
// `&mut self`; D3D12 objects are free-threaded.
unsafe impl Send for Pass {}

fn blob_bytes(b: &ID3DBlob) -> &[u8] {
    // SAFETY: the blob's own buffer, alive with it.
    unsafe { std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize()) }
}

/// Compiles an entry of `source` with FXC.
pub fn compile_source(source: &str, entry: &std::ffi::CStr, target: &std::ffi::CStr) -> std::result::Result<ID3DBlob, String> {
    let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
    // SAFETY: the source and names live through the call; out pointers ours.
    let r = unsafe { D3DCompile(source.as_ptr().cast(), source.len(), PCSTR(c"world_render".as_ptr().cast()), None, None, PCSTR(entry.as_ptr().cast()), PCSTR(target.as_ptr().cast()), 0, 0, &mut code, Some(&mut errors)) };
    match (r, code) {
        (Ok(()), Some(c)) => Ok(c),
        (r, _) => Err(format!("{entry:?}: {r:?} {}", errors.map(|e| String::from_utf8_lossy(blob_bytes(&e)).into_owned()).unwrap_or_default())),
    }
}

/// Compiles an entry of [`SHADER`] with FXC.
pub fn compile(entry: &std::ffi::CStr, target: &std::ffi::CStr) -> std::result::Result<ID3DBlob, String> {
    compile_source(SHADER, entry, target)
}

fn transition(list: &ID3D12GraphicsCommandList, r: &ID3D12Resource, from: D3D12_RESOURCE_STATES, to: D3D12_RESOURCE_STATES) {
    let b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER { pResource: ManuallyDrop::new(Some(r.clone())), Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: from, StateAfter: to }),
        },
    };
    // SAFETY: an open list; the barrier's reference is released after.
    unsafe {
        list.ResourceBarrier(std::slice::from_ref(&b));
        drop(ManuallyDrop::into_inner(ManuallyDrop::into_inner(b.Anonymous.Transition).pResource));
    }
}

/// A graphics pipeline: `elements` empty for the raw path.
fn pipeline(device: &ID3D12Device, root: &ID3D12RootSignature, vs: &ID3DBlob, ps: &ID3DBlob, elements: &[D3D12_INPUT_ELEMENT_DESC], wire: bool) -> Result<ID3D12PipelineState> {
    pipeline_with(device, root, vs, ps, elements, wire, false)
}

/// `decal`: depth test GE without write and a depth bias towards the camera
/// (reverse Z: positive), as the title's decal raster state [I values].
fn pipeline_with(device: &ID3D12Device, root: &ID3D12RootSignature, vs: &ID3DBlob, ps: &ID3DBlob, elements: &[D3D12_INPUT_ELEMENT_DESC], wire: bool, decal: bool) -> Result<ID3D12PipelineState> {
    let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        // SAFETY: the blobs outlive the call.
        VS: unsafe { D3D12_SHADER_BYTECODE { pShaderBytecode: vs.GetBufferPointer(), BytecodeLength: vs.GetBufferSize() } },
        // SAFETY: as above.
        PS: unsafe { D3D12_SHADER_BYTECODE { pShaderBytecode: ps.GetBufferPointer(), BytecodeLength: ps.GetBufferSize() } },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: if wire { D3D12_FILL_MODE_WIREFRAME } else { D3D12_FILL_MODE_SOLID },
            CullMode: D3D12_CULL_MODE_NONE,
            DepthClipEnable: true.into(),
            DepthBias: if decal { 64 } else { 0 },
            SlopeScaledDepthBias: if decal { 1.0 } else { 0.0 },
            ..Default::default()
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: if wire || decal { D3D12_DEPTH_WRITE_MASK_ZERO } else { D3D12_DEPTH_WRITE_MASK_ALL },
            DepthFunc: if wire || decal { D3D12_COMPARISON_FUNC_GREATER_EQUAL } else { D3D12_COMPARISON_FUNC_GREATER },
            ..Default::default()
        },
        InputLayout: D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: if elements.is_empty() { std::ptr::null() } else { elements.as_ptr() }, NumElements: elements.len() as u32 },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        DSVFormat: DEPTH,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    desc.RTVFormats[0] = COLOUR;
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
    gbuf::layout(&mut desc, blob_bytes(ps));
    // SAFETY: the description and what it points at outlive the call.
    let pso = unsafe { device.CreateGraphicsPipelineState::<ID3D12PipelineState>(&desc) };
    drop(ManuallyDrop::into_inner(desc.pRootSignature));
    pso
}

/// The raw-path pipeline for a title render state. Values outside
/// the D3D12 enum ranges fall back to ours (depth GREATER with write, no blend).
/// `plain`: with `gbuf=on`, the pixel entry without the G-buffer outputs,
/// for states that blend with source 1 ([`gbuf::pixel_for`]).
fn pipeline_state(device: &ID3D12Device, root: &ID3D12RootSignature, vs: &ID3DBlob, ps: &ID3DBlob, plain: Option<&ID3DBlob>, s: &RenderState) -> Result<ID3D12PipelineState> {
    pipeline_state_with(device, root, vs, gbuf::pixel_for(ps, plain, &s.blend), s, &[])
}

/// [`pipeline_state`] with an input layout (empty: none).
fn pipeline_state_with(device: &ID3D12Device, root: &ID3D12RootSignature, vs: &ID3DBlob, ps: &ID3DBlob, s: &RenderState, elements: &[D3D12_INPUT_ELEMENT_DESC]) -> Result<ID3D12PipelineState> {
    let func = if (1..=8).contains(&s.depth_func) { D3D12_COMPARISON_FUNC(i32::from(s.depth_func)) } else { D3D12_COMPARISON_FUNC_GREATER };
    let b = s.blend;
    let blend_ok = b[0] != 0 && [b[1], b[2], b[4], b[5]].iter().all(|v| (1..=19).contains(v)) && [b[3], b[6]].iter().all(|v| (1..=5).contains(v));
    let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        // SAFETY: the blobs outlive the call.
        VS: unsafe { D3D12_SHADER_BYTECODE { pShaderBytecode: vs.GetBufferPointer(), BytecodeLength: vs.GetBufferSize() } },
        // SAFETY: as above.
        PS: unsafe { D3D12_SHADER_BYTECODE { pShaderBytecode: ps.GetBufferPointer(), BytecodeLength: ps.GetBufferSize() } },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            DepthClipEnable: true.into(),
            DepthBias: s.depth_bias,
            DepthBiasClamp: f32::from_bits(s.bias_clamp),
            SlopeScaledDepthBias: f32::from_bits(s.slope_bias),
            ..Default::default()
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: s.depth_enable.into(),
            DepthWriteMask: if s.depth_enable && s.depth_write { D3D12_DEPTH_WRITE_MASK_ALL } else { D3D12_DEPTH_WRITE_MASK_ZERO },
            DepthFunc: func,
            ..Default::default()
        },
        InputLayout: D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: if elements.is_empty() { std::ptr::null() } else { elements.as_ptr() }, NumElements: elements.len() as u32 },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        DSVFormat: DEPTH,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    desc.RTVFormats[0] = COLOUR;
    let rt = &mut desc.BlendState.RenderTarget[0];
    rt.RenderTargetWriteMask = if blend_ok { b[7] & 0xF } else { D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8 };
    if blend_ok {
        rt.BlendEnable = true.into();
        (rt.SrcBlend, rt.DestBlend, rt.BlendOp) = (D3D12_BLEND(i32::from(b[1])), D3D12_BLEND(i32::from(b[2])), D3D12_BLEND_OP(i32::from(b[3])));
        (rt.SrcBlendAlpha, rt.DestBlendAlpha, rt.BlendOpAlpha) = (D3D12_BLEND(i32::from(b[4])), D3D12_BLEND(i32::from(b[5])), D3D12_BLEND_OP(i32::from(b[6])));
    }
    gbuf::layout(&mut desc, blob_bytes(ps));
    // SAFETY: the description and what it points at outlive the call.
    let pso = unsafe { device.CreateGraphicsPipelineState::<ID3D12PipelineState>(&desc) };
    drop(ManuallyDrop::into_inner(desc.pRootSignature));
    pso
}

/// An upload-heap buffer holding `bytes`.
fn upload_buffer(device: &ID3D12Device, bytes: &[u8]) -> Result<ID3D12Resource> {
    let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: bytes.len().max(16) as u64, Height: 1, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_UNKNOWN, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
    let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() };
    let mut out: Option<ID3D12Resource> = None;
    // SAFETY: descriptions outlive the call; the mapping covers `bytes`.
    unsafe {
        device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut out)?;
        let r = out.expect("resource on success");
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        r.Map(0, None, Some(&mut p))?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.cast::<u8>(), bytes.len());
        r.Unmap(0, None);
        Ok(r)
    }
}

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain-old-data viewed as bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

impl Pass {
    /// Replaces the world pass's shader ([`RAW_SHADER`], the one live
    /// lists draw with) by `source`: compiles `vs`/`ps` with FXC and rebuilds
    /// the root, the raw, wire and decal pipelines; the per-state pipelines
    /// are made again on demand. On a compile or pipeline error nothing
    /// changes and the FXC message is returned. Call between frames ([`Pass::draw`]
    /// waits for its own work).
    pub fn set_raw_shader(&mut self, source: &str) -> std::result::Result<(), String> {
        let e = |e: windows::core::Error| e.to_string();
        let plain = if gbuf::on() { Some(compile_source(source, c"ps", c"ps_5_1")?) } else { None };
        let source = &gbuf::source(source);
        let (vs, ps) = (compile_source(source, c"vs", c"vs_5_1")?, compile_source(source, c"ps", c"ps_5_1")?);
        // SAFETY: object creation from blobs that outlive each call.
        let (root, raw, wire, decal) = unsafe {
            let root: ID3D12RootSignature = self.device.CreateRootSignature(0, blob_bytes(&vs)).map_err(e)?;
            let raw = pipeline(&self.device, &root, &vs, &ps, &[], false).map_err(e)?;
            let wire = pipeline(&self.device, &root, &vs, &ps, &[], true).map_err(e)?;
            let decal = pipeline_with(&self.device, &root, &vs, &ps, &[], false, true).map_err(e)?;
            (root, raw, wire, decal)
        };
        (self.raw_root, self.raw_pso, self.wire_pso, self.decal_pso, self.raw_shaders, self.raw_plain_ps) = (root, raw, wire, decal, (vs, ps), plain);
        (self.raw_source, self.rec_pso) = (source.clone(), None);
        self.state_psos.clear();
        self.pso_jobs = None;
        self.pso_asked.clear();
        Ok(())
    }
}

impl Pass {
    /// The pass on `host`'s device: pipelines, `width` x `height` targets.
    pub fn new(host: &xnw_gpu::host::Host, width: u32, height: u32) -> std::result::Result<Pass, String> {
        let e = |e: windows::core::Error| e.to_string();
        // `gbuf=on`: before any pipeline is made (they read it).
        gbuf::set(gbuf::parse_switch(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default()));
        let raw_source = gbuf::source(RAW_SHADER);
        let (vs, ps) = (compile(c"vs", c"vs_5_1")?, compile(c"ps", c"ps_5_1")?);
        let (rvs, rps) = (compile_source(&raw_source, c"vs", c"vs_5_1")?, compile_source(&raw_source, c"ps", c"ps_5_1")?);
        let raw_plain_ps = if gbuf::on() { Some(compile_source(RAW_SHADER, c"ps", c"ps_5_1")?) } else { None };
        let device = host.device.clone();
        // SAFETY: object creation with descriptions that outlive each call.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, blob_bytes(&vs)).map_err(e)?;
            let raw_root: ID3D12RootSignature = device.CreateRootSignature(0, blob_bytes(&rvs)).map_err(e)?;
            let elements = [
                D3D12_INPUT_ELEMENT_DESC { SemanticName: PCSTR(c"POSITION".as_ptr().cast()), SemanticIndex: 0, Format: DXGI_FORMAT_R32G32B32_FLOAT, InputSlot: 0, AlignedByteOffset: 0, InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, InstanceDataStepRate: 0 },
                D3D12_INPUT_ELEMENT_DESC { SemanticName: PCSTR(c"NORMAL".as_ptr().cast()), SemanticIndex: 0, Format: DXGI_FORMAT_R32G32B32_FLOAT, InputSlot: 0, AlignedByteOffset: 12, InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, InstanceDataStepRate: 0 },
                D3D12_INPUT_ELEMENT_DESC { SemanticName: PCSTR(c"TEXCOORD".as_ptr().cast()), SemanticIndex: 0, Format: DXGI_FORMAT_R32G32_FLOAT, InputSlot: 0, AlignedByteOffset: 24, InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, InstanceDataStepRate: 0 },
            ];
            let pso = pipeline(&device, &root, &vs, &ps, &elements, false).map_err(e)?;
            let raw_pso = pipeline(&device, &raw_root, &rvs, &rps, &[], false).map_err(e)?;
            let wire_pso = pipeline(&device, &raw_root, &rvs, &rps, &[], true).map_err(e)?;
            let decal_pso = pipeline_with(&device, &raw_root, &rvs, &rps, &[], false, true).map_err(e)?;
            // The sky: no depth test or write (drawn first, behind everything).
            let (svs, sps) = (compile_source(SKY_SHADER, c"vs", c"vs_5_1")?, compile_source(SKY_SHADER, c"ps", c"ps_5_1")?);
            let sky_root: ID3D12RootSignature = device.CreateRootSignature(0, blob_bytes(&svs)).map_err(e)?;
            let sky_pso = pipeline_state(&device, &sky_root, &svs, &sps, None, &RenderState::default()).map_err(e)?;
            // Water: depth test GREATER (reverse Z) without write, straight alpha.
            let (wvs, wps) = (compile_source(WATER_SHADER, c"vs", c"vs_5_1")?, compile_source(WATER_SHADER, c"ps", c"ps_5_1")?);
            let water_root: ID3D12RootSignature = device.CreateRootSignature(0, blob_bytes(&wvs)).map_err(e)?;
            let water_state = RenderState { depth_enable: true, depth_write: false, depth_func: 5, blend: [1, 5, 6, 1, 2, 1, 1, 0xF], ..Default::default() };
            let water_pso = pipeline_state_with(&device, &water_root, &wvs, &wps, &water_state, &[D3D12_INPUT_ELEMENT_DESC { SemanticName: PCSTR(c"POSITION".as_ptr().cast()), SemanticIndex: 0, Format: DXGI_FORMAT_R32G32B32_FLOAT, InputSlot: 0, AlignedByteOffset: 0, InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, InstanceDataStepRate: 0 }]).map_err(e)?;
            let heap: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: HEAP_SLOTS, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 }).map_err(e)?;
            let heap_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            let texture = |format, flags, state, clear: D3D12_CLEAR_VALUE| -> Result<ID3D12Resource> {
                let d = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: u64::from(width), Height: height, DepthOrArraySize: 1, MipLevels: 1, Format: format, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Flags: flags, ..Default::default() };
                let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
                let mut out: Option<ID3D12Resource> = None;
                device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &d, state, Some(&clear), &mut out)?;
                Ok(out.expect("resource on success"))
            };
            let colour = texture(COLOUR, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_CLEAR_VALUE { Format: COLOUR, Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.09, 0.09, 0.1, 1.0] } }).map_err(e)?;
            let shown = texture(COLOUR, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_CLEAR_VALUE { Format: COLOUR, Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.09, 0.09, 0.1, 1.0] } }).map_err(e)?;
            let depth = texture(DEPTH, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_CLEAR_VALUE { Format: DEPTH, Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 0.0, Stencil: 0 } } }).map_err(e)?;
            let small = |kind, n| device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC { Type: kind, NumDescriptors: n, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE, NodeMask: 0 });
            let rtv = small(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 3).map_err(e)?;
            let dsv = small(D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1).map_err(e)?;
            device.CreateRenderTargetView(&colour, None, rtv.GetCPUDescriptorHandleForHeapStart());
            // `gbuf=on`: the title's RT0 and RT1 at RTV slots 1 and 2.
            let gbuf_targets = if gbuf::on() {
                let rstep = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
                let black = |f: DXGI_FORMAT| D3D12_CLEAR_VALUE { Format: f, Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0; 4] } };
                let t = gbuf::Targets {
                    albedo: texture(gbuf::ALBEDO, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, black(gbuf::ALBEDO)).map_err(e)?,
                    normal: texture(gbuf::NORMAL, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, black(gbuf::NORMAL)).map_err(e)?,
                };
                let at = |k: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: rtv.GetCPUDescriptorHandleForHeapStart().ptr + k * rstep };
                device.CreateRenderTargetView(&t.albedo, None, at(1));
                device.CreateRenderTargetView(&t.normal, None, at(2));
                Some(t)
            } else {
                None
            };
            device.CreateDepthStencilView(&depth, None, dsv.GetCPUDescriptorHandleForHeapStart());
            let readback = host.buffer(u64::from(width) * 4 * u64::from(height), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST).map_err(e)?;
            let alloc: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
            let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(e)?;
            list.Close().map_err(e)?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
            let terrain = terrain::TerrainPass::new(&device)?;
            let imposters = imposter::ImposterPass::new(&device)?;
            Ok(Pass {
                device,
                queue: host.queue.clone(),
                root,
                pso,
                raw_root,
                raw_pso,
                wire_pso,
                heap,
                heap_step,
                next_slot: 4,
                free_slots: Vec::new(),
                zc_geoms: Default::default(),
                palettes: [None, None],
                instances: [None, None],
                light_cb: [None, None],
                light_old: false,
                layers: 2,
                layers_debug: false,
                stipple: true,
                light_va: 0,
                palette_turn: 0,
                colour,
                _depth: depth,
                rtv,
                dsv,
                readback,
                alloc,
                list,
                fence,
                fence_value: 0,
                chunk: 0,
                stamps: None,
                width,
                height,
                cache: GeometryCache::default(),
                wire: false,
                id_mode: false,
                only_stage: None,
                decals: true,
                decal_pso,
                skin: true,
                peds: true,
                skinned: true,
                skin_rules: 0,
                decal_hold: true,
                held_decals: Vec::new(),
                tex: textures::TexTable::new(TexOptions::default(), HEAP_SLOTS - TEX_SLOTS),
                states: true,
                state_psos: HashMap::new(),
                raw_shaders: (rvs, rps),
                raw_plain_ps,
                pso_budget: usize::MAX,
                pso_async: false,
                pso_jobs: None,
                pso_asked: std::collections::HashSet::new(),
                upload_ms_budget: None,
                upload_cost: (0.0, 0.0),
                sky: true,
                sky_root,
                sky_pso,
                water: true,
                water_root,
                water_pso,
                water_gpu: None,
                geom: geometry::GeomPolicy::new(geometry::GeomBudget::UNBOUNDED),
                staging: None,
                staged: 0,
                make_ms: 0.0,
                copies: Vec::new(),
                pages: Pages::default(),
                terrain,
                imposters,
                bridge_on: false,
                bridge_stats: bridged::Stats::default(),
                bridge_warm: false,
                bridge_gpu: bridged::Gpu::default(),
                bridge_keep: Vec::new(),
                bridge_grey: None,
                bridge_words: None,
                shown,
                cuts: cuts::CutWatch::new(false),
                preskin: preskin_gpu::GpuPreskin::default(),
                bridge_ms: None,
                raw_source,
                rec_pso: None,
                gbuf: gbuf_targets,
                rec_slot: None,
                cull: gpucull::GpuCull::default(),
            })
        }
    }

    /// The pass's device (offline checks run beside it).
    pub fn device(&self) -> &ID3D12Device {
        &self.device
    }

    /// The colour target (render-target state between draws).
    pub fn colour(&self) -> &ID3D12Resource {
        &self.colour
    }

    /// What the presenter shows: the last picture complete enough (hold
    /// on), else this frame's (render-target state).
    pub fn shown(&self) -> &ID3D12Resource {
        if self.cuts.hold { &self.shown } else { &self.colour }
    }

    fn cpu_slot(&self, slot: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        // SAFETY: our heap.
        let base = unsafe { self.heap.GetCPUDescriptorHandleForHeapStart() };
        D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base.ptr + (slot * self.heap_step) as usize }
    }

    fn gpu_slot(&self, slot: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: our heap.
        let base = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        D3D12_GPU_DESCRIPTOR_HANDLE { ptr: base.ptr + u64::from(slot * self.heap_step) }
    }

    /// A raw (ByteAddressBuffer) view of `buffer`'s first `bytes` at `slot`.
    fn raw_view(&self, buffer: &ID3D12Resource, bytes: u64, slot: u32) {
        self.raw_view_at(buffer, 0, bytes, slot);
    }

    /// A raw view of `bytes` at byte `offset` (a multiple of 4) of `buffer`.
    fn raw_view_at(&self, buffer: &ID3D12Resource, offset: u64, bytes: u64, slot: u32) {
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Buffer: D3D12_BUFFER_SRV { FirstElement: offset / 4, NumElements: (bytes / 4).max(1) as u32, StructureByteStride: 0, Flags: D3D12_BUFFER_SRV_FLAG_RAW } },
        };
        // SAFETY: a live buffer and our heap's slot.
        unsafe { self.device.CreateShaderResourceView(buffer, Some(&desc), self.cpu_slot(slot)) };
    }

    /// A device buffer (default heap, COMMON) holding `vb` then `ib` at a
    /// 256-byte offset; the bytes go through the frame's staging buffer and
    /// a copy recorded before the draws (no upload-heap buffer is kept, no
    /// CPU copy either: the caller drops its bytes).
    fn stage_buffer(&mut self, vb: &[u8], ib: &[u8]) -> Result<((u32, u64, u64), u64)> {
        let ib_at = (vb.len() as u64).next_multiple_of(256);
        let total = ib_at + ib.len().max(4) as u64;
        let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_UNKNOWN, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
        // A range of a device page, not a committed resource per geometry
        // 1,230 creations in one frame, 680 ms on the hardware).
        let (tm, pages_before) = (std::time::Instant::now(), self.pages.bytes());
        let place = self.pages.alloc(&self.device, total)?;
        if self.pages.bytes() != pages_before {
            self.make_ms += tm.elapsed().as_secs_f64() * 1e3;
        }
        let need = self.staged + total as usize;
        if self.staging.as_ref().is_none_or(|s| s.2 < need) {
            let tm = std::time::Instant::now();
            // A larger staging buffer; copies already recorded keep the old one.
            let size = need.next_power_of_two().max(4 << 20).max(self.staging.as_ref().map_or(0, |s| 2 * s.2));
            let staging = {
                let d = D3D12_RESOURCE_DESC { Width: size as u64, ..desc };
                let mut o: Option<ID3D12Resource> = None;
                // SAFETY: as above; an upload buffer stays mapped for its life.
                unsafe { self.device.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut o)? };
                o.expect("resource on success")
            };
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
            // SAFETY: as above.
            unsafe { staging.Map(0, None, Some(&mut p))? };
            self.staging = Some((staging, p.cast::<u8>(), size));
            self.staged = 0;
            self.make_ms += tm.elapsed().as_secs_f64() * 1e3;
        }
        let (staging, mapped, _) = self.staging.as_ref().expect("made above");
        // SAFETY: `total` bytes from `staged` fit in the mapping (checked above).
        unsafe {
            std::ptr::copy_nonoverlapping(vb.as_ptr(), mapped.add(self.staged), vb.len());
            std::ptr::copy_nonoverlapping(ib.as_ptr(), mapped.add(self.staged + ib_at as usize), ib.len());
        }
        self.copies.push((self.pages.buffer(place.0).clone(), staging.clone(), self.staged as u64, total, place.1));
        self.staged = (self.staged + total as usize).next_multiple_of(256);
        Ok((place, ib_at))
    }

    /// 2: takes the pipelines the worker made and asks it for every
    /// state of this list not made or asked for yet (at most MAX_STATE_PSOS).
    fn pipelines_ahead(&mut self, list: &FrameList, st: &mut PassStats) {
        if self.pso_jobs.is_none() {
            let (ask_tx, ask_rx) = std::sync::mpsc::channel::<RenderState>();
            let (done_tx, done_rx) = std::sync::mpsc::channel();
            let parts = SendBox((self.device.clone(), self.raw_root.clone(), self.raw_shaders.0.clone(), self.raw_shaders.1.clone(), self.raw_plain_ps.clone()));
            std::thread::spawn(move || {
                let SendBox((device, root, vs, ps, plain)) = parts;
                for s in ask_rx {
                    let t0 = std::time::Instant::now();
                    let p = pipeline_state(&device, &root, &vs, &ps, plain.as_ref(), &s).map_err(|e| xnw_core::warn!("world render: state pipeline {s:?} failed: {e}")).ok();
                    if done_tx.send(SendBox((s, p, t0.elapsed().as_secs_f64() * 1e3))).is_err() {
                        break;
                    }
                }
            });
            self.pso_jobs = Some((ask_tx, done_rx));
        }
        let Some((ask, done)) = &self.pso_jobs else { return };
        while let Ok(SendBox((s, p, ms))) = done.try_recv() {
            st.pso_ms_worker += ms;
            if let Some(p) = p {
                self.state_psos.insert(s, p);
                st.new_psos += 1;
            }
        }
        for r in &list.draws {
            if let Some(s) = r.state
                && self.pso_asked.len() < MAX_STATE_PSOS
                && self.pso_asked.insert(s)
            {
                let _ = ask.send(s);
            }
        }
    }

    /// `XNW_ZEROCOPY` (ZC2): geometries whose in-place vertex buffer lies in
    /// a window heap released since the last frame, or in title pages
    /// remapped since (the view reads the section's old page), are dropped
    /// and read again. Their number.
    fn zc_changes(&mut self) -> usize {
        let c = xnw_gpu::shared::world_changes();
        if self.zc_geoms.is_empty() || c.gone.is_empty() && c.remapped.is_empty() && !c.overflow {
            return 0;
        }
        let mut drop: Vec<GeomKey> = Vec::new();
        if c.overflow || !c.gone.is_empty() {
            for &(len, gpu, key) in self.zc_geoms.values() {
                if c.overflow || c.gone.iter().any(|&(g, n)| gpu < g + n && gpu + len > g) {
                    drop.push(key);
                }
            }
        }
        // A geometry starts at most MAX_VB_BYTES before a remapped range it overlaps.
        for &(a, n) in &c.remapped {
            let from = (a.saturating_sub(MAX_VB_BYTES as u64), 0, 0);
            for (&(at, _, _), &(len, _, key)) in self.zc_geoms.range(from..(a + n, 0, 0)) {
                if at + len > a {
                    drop.push(key);
                }
            }
        }
        drop.sort_unstable_by_key(|k| (k.vb_at, k.mesh_at, k.hash));
        drop.dedup();
        let known = geometry::known();
        for key in &drop {
            self.geom.drop_key(known, key);
            if let Some(old) = self.cache.remove(key) {
                self.release(old);
            }
        }
        drop.len()
    }

    /// Drops a cached geometry: its heap slot and its page range.
    fn release(&mut self, g: GpuGeometry) {
        if let Some((slot, _)) = g.raw {
            self.free_slots.push(slot);
        }
        if let Some(k) = g.in_place {
            self.zc_geoms.remove(&k);
        }
        self.pages.free(g.place);
    }

    fn upload(&mut self, key: &GeomKey, g: &HostGeometry, clamped: &mut u64) -> Result<Option<GpuGeometry>> {
        let indices = &g.indices[..g.indices.len().min(MAX_INDICES) / 3 * 3];
        *clamped += (g.indices.len() - indices.len()) as u64;
        if let Some(raw) = &g.raw_vb {
            let raw = &raw[..raw.len().min(MAX_VB_BYTES) & !3];
            // `XNW_ZEROCOPY` (ZC1, ZC2): the vertex buffer read in place in
            // title memory (a raw view needs a 16-byte offset); only the
            // indices (meshlets expanded on the CPU) go to the device. With
            // `vb_place` the reader copied only its header: in place or not
            // at all (asked again; the reader then copies it whole).
            let len = if g.vb_place > 0 { g.vb_place.min(MAX_VB_BYTES as u64) & !3 } else { raw.len() as u64 };
            let in_place = if xnw_gpu::shared::world_part(xnw_gpu::shared::WORLD_GEOM) { xnw_gpu::shared::world_range(key.vb_at, len).filter(|r| r.offset.is_multiple_of(16)) } else { None };
            if g.vb_place > 0 && in_place.is_none() {
                return Ok(None);
            }
            if let Some(r) = in_place {
                let Some(slot) = self.free_slots.pop().or_else(|| (self.next_slot < HEAP_SLOTS - TEX_SLOTS).then(|| {
                    self.next_slot += 1;
                    self.next_slot - 1
                })) else {
                    return Ok(None);
                };
                let (place, ib_at) = match self.stage_buffer(&[], as_bytes(indices)) {
                    Ok(b) => b,
                    Err(e) => {
                        self.free_slots.push(slot);
                        return Err(e);
                    }
                };
                // View bounds: exactly the vertex buffer's bytes (the window
                // holds them whole: `world_range`).
                self.raw_view_at(&r.buffer, r.offset, len, slot);
                // SAFETY: live buffers.
                let (ia, gpu) = unsafe { (self.pages.buffer(place.0).GetGPUVirtualAddress() + place.1 + ib_at, r.buffer.GetGPUVirtualAddress() + r.offset) };
                self.zc_geoms.insert((key.vb_at, key.mesh_at, key.hash), (len, gpu, *key));
                return Ok(Some(GpuGeometry {
                    vbv: D3D12_VERTEX_BUFFER_VIEW::default(),
                    ibv: D3D12_INDEX_BUFFER_VIEW { BufferLocation: ia, SizeInBytes: (indices.len() * 4) as u32, Format: DXGI_FORMAT_R32_UINT },
                    indices: indices.len() as u32,
                    bytes: (indices.len() * 4) as u64,
                    raw: Some((slot, len as u32)),
                    top: indices.iter().max().map_or(0, |m| m + 1),
                    place,
                    in_place: Some((key.vb_at, key.mesh_at, key.hash)),
                }));
            }
            let Some(slot) = self.free_slots.pop().or_else(|| (self.next_slot < HEAP_SLOTS - TEX_SLOTS).then(|| {
                self.next_slot += 1;
                self.next_slot - 1
            })) else {
                return Ok(None);
            };
            let (place, ib_at) = match self.stage_buffer(raw, as_bytes(indices)) {
                Ok(b) => b,
                Err(e) => {
                    self.free_slots.push(slot);
                    return Err(e);
                }
            };
            let buf = self.pages.buffer(place.0).clone();
            self.raw_view_at(&buf, place.1, raw.len() as u64, slot);
            // SAFETY: a live buffer.
            let ia = unsafe { buf.GetGPUVirtualAddress() } + place.1 + ib_at;
            return Ok(Some(GpuGeometry {
                vbv: D3D12_VERTEX_BUFFER_VIEW::default(),
                ibv: D3D12_INDEX_BUFFER_VIEW { BufferLocation: ia, SizeInBytes: (indices.len() * 4) as u32, Format: DXGI_FORMAT_R32_UINT },
                indices: indices.len() as u32,
                bytes: (raw.len() + indices.len() * 4) as u64,
                raw: Some((slot, raw.len() as u32)),
                top: indices.iter().max().map_or(0, |m| m + 1),
                place,
                in_place: None,
            }));
        }
        let (place, ib_at) = self.stage_buffer(as_bytes(&g.vertices), as_bytes(indices))?;
        // SAFETY: a live buffer.
        let va = unsafe { self.pages.buffer(place.0).GetGPUVirtualAddress() } + place.1;
        let vb_bytes = g.vertices.len() * 4;
        // Indices past the vertex count would read past the vertex buffer: clamp to none.
        let vertices = (g.vertices.len() * 4 / VERTEX_BYTES) as u32;
        let count = if indices.iter().all(|&i| i < vertices) { indices.len() } else { 0 };
        Ok(Some(GpuGeometry {
            vbv: D3D12_VERTEX_BUFFER_VIEW { BufferLocation: va, SizeInBytes: vb_bytes.max(4) as u32, StrideInBytes: VERTEX_BYTES as u32 },
            ibv: D3D12_INDEX_BUFFER_VIEW { BufferLocation: va + ib_at, SizeInBytes: (indices.len() * 4) as u32, Format: DXGI_FORMAT_R32_UINT },
            indices: count as u32,
            bytes: (vb_bytes + indices.len() * 4) as u64,
            raw: None,
            top: 0,
            place,
            in_place: None,
        }))
    }
    /// Draws `list`'s main view: the list's new geometries first, then any
    /// missing one from `decode` (None: that draw is skipped). With
    /// `readback` the target is copied for [`Pass::read_rgba`]. Waits for
    /// the GPU; the colour target is left in the render-target state.
    pub fn draw(&mut self, list: &FrameList, decode: &mut dyn FnMut(&GeomKey) -> Option<HostGeometry>, readback: bool) -> std::result::Result<PassStats, String> {
        // The bridged rows stay with the pass whatever the frame does.
        let batch = self.bridge_ms.take();
        let r = self.draw_with(list, decode, readback, batch.as_ref().filter(|b| !b.is_empty()));
        self.bridge_ms = batch;
        r
    }

    /// The record-drawing pipeline (`vs_rec` of the raw shader's source,
    /// our pixel shader and default state) and the record ring's heap
    /// slot, made at their first use.
    fn rec_ready(&mut self) -> std::result::Result<(), String> {
        if self.rec_pso.is_none() {
            let vs = compile_source(&self.raw_source, c"vs_rec", c"vs_5_1")?;
            self.rec_pso = Some(pipeline(&self.device, &self.raw_root, &vs, &self.raw_shaders.1, &[], false).map_err(|e| format!("bridgems record pipeline: {e}"))?);
        }
        if self.rec_slot.is_none() {
            let slot = self.free_slots.pop().or_else(|| (self.next_slot < HEAP_SLOTS - TEX_SLOTS).then(|| {
                self.next_slot += 1;
                self.next_slot - 1
            }));
            self.rec_slot = Some(slot.ok_or("bridgems: no heap slot for the record ring")?);
        }
        Ok(())
    }

    fn draw_with(&mut self, list: &FrameList, decode: &mut dyn FnMut(&GeomKey) -> Option<HostGeometry>, readback: bool, ms_rows: Option<&bridge_ms::Batch>) -> std::result::Result<PassStats, String> {
        if ms_rows.is_some() {
            self.rec_ready()?;
        }
        // Every failure names its call, sizes and the frame.
        let frame = list.frame;
        let at = |what: String| move |e: windows::core::Error| format!("frame {frame}: {what}: {e}");
        let cam = *list.main_camera().ok_or("no camera")?;
        let mut st = PassStats::default();
        // `XNW_ZEROCOPY` (ZC2): no window heap is released from here to the
        // fence; one released or remapped since the last frame is dropped first.
        let _in_flight = xnw_gpu::shared::world_frame();
        st.in_place_dropped = self.zc_changes();
        let t0 = std::time::Instant::now();
        let now = cuts::now_ms();
        self.cuts.camera(list.frame, &cam, now);
        let mut new_psos = 0usize;
        let only = self.only_stage;
        let peds = self.peds;
        let draws: Vec<&DrawRecord> = list.main_draws().filter(|r| match only {
            Some(s) => s == r.stage,
            None => !PREPASS_STAGES.contains(&r.stage) && !OTHER_VIEW_STAGES.contains(&r.stage) && (self.decals || !DECAL_STAGES.contains(&r.stage)),
        }).collect();
        // Ped skin draws are skipped unless `peds=on`.
        let before = draws.len();
        // Rows the pre-skin gives positions are drawn with `peds` and `skinned` off too.
        let pre_rows = list.preskin.rows();
        let pre_row = |r: &DrawRecord| !pre_rows.is_empty() && pre_rows.contains(&(((r as *const DrawRecord as usize).wrapping_sub(list.draws.as_ptr() as usize) / std::mem::size_of::<DrawRecord>()) as u32));
        let draws: Vec<&DrawRecord> = draws.into_iter().filter(|r| peds || !r.ped || pre_row(r)).collect();
        st.peds_skipped = before - draws.len();
        let before = draws.len();
        let skinned = self.skinned;
        let rules = self.skin_rules;
        let lifted = |r: &DrawRecord| skin_lifted(list, r, rules);
        let mut draws: Vec<&DrawRecord> = draws.into_iter().filter(|r| skinned || r.ped || r.palette == NO_PALETTE || r.bone_count <= 1 || lifted(r) || pre_row(r)).collect();
        st.skinned_skipped = before - draws.len();
        // The title submits some stage-74 decals on alternate frames only
        // (standard_decal 0xd6385a01c3119400 in 111 of 325 frames in a test run),
        // leaving its TAA to blend them; we draw the union of this list's
        // decals and the previous list's (re-placed for this camera).
        let held = if self.decal_hold { decal_union(&mut self.held_decals, &draws, &cam) } else { Vec::new() };
        st.decals_held = held.len();
        // Counted before the held decals join (they are not in this
        // list's main view: main - draws underflowed by their number).
        st.filtered = filtered_count(list.main_draws().count(), draws.len());
        draws.extend(held.iter());
        // Opaque, then decals (74), then transparent (14)
        // back to front (camera-relative translation: world row 3).
        let mut draws = draws;
        let order = |r: &DrawRecord| -> (u8, i64) {
            if DECAL_STAGES.contains(&r.stage) {
                (1, 0)
            } else if TRANSPARENT_STAGES.contains(&r.stage) {
                let d = (r.world[12] * r.world[12] + r.world[13] * r.world[13] + r.world[14] * r.world[14]).sqrt();
                (2, -((d * 1000.0) as i64))
            } else {
                (0, 0)
            }
        };
        draws.sort_by_key(|r| order(r));
        if self.pso_async && self.states {
            self.pipelines_ahead(list, &mut st);
        }
        // Geometry through the policy: arrivals queued, then the frame's
        // plan (evictions, uploads by priority under the budgets).
        let tu = std::time::Instant::now();
        let known = geometry::known();
        for (key, g) in &list.ahead_geoms {
            self.geom.offer_ahead(known, *key, g.clone(), list.frame);
        }
        for (key, g) in &list.new_geoms {
            self.geom.offer(known, *key, g.clone(), list.frame);
        }
        for r in &draws {
            if !self.geom.is_resident(&r.geom)
                && !self.geom.is_pending(&r.geom)
                && let Some(g) = decode(&r.geom)
            {
                self.geom.offer(known, r.geom, g, list.frame);
            }
        }
        let refs: Vec<(GeomKey, f32)> = draws.iter().map(|r| (r.geom, r.world[12] * r.world[12] + r.world[13] * r.world[13] + r.world[14] * r.world[14])).collect();
        // 3: the frame's upload budget from time: the measured ms per
        // upload and per MiB (each taken as if it were the whole cost, so
        // either bound alone keeps the frame under the time).
        if let Some(ms) = self.upload_ms_budget
            && self.upload_cost.0 > 0.0
        {
            // While a cut recovers the frame is held anyway: spend more
            // (`cutfast=on`: what drains the main view's queue in 2 frames).
            let (per_geom, per_mib) = self.upload_cost;
            let need = if self.cuts.drain_frames().is_some() {
                let (n, bytes) = self.geom.pending_for(draws.iter().map(|r| &r.geom));
                (per_geom * n as f64).max(per_mib * bytes as f64 / f64::from(1 << 20))
            } else {
                0.0
            };
            let ms = self.cuts.upload_ms(ms, need);
            let (max_count, max_bytes) = if self.cuts.drain_frames().is_some() { (8192, cuts::DRAIN_MAX_BYTES) } else { (2048, SCREEN_UPLOAD_BUDGET) };
            self.geom.budget.frame_count = ((ms / per_geom) as u32).clamp(8, max_count);
            // The ahead pool's 2 ms a frame.
            self.geom.budget.ahead_frame_count = ((geometry::AHEAD_FRAME_MS / per_geom) as u32).clamp(1, 512);
            self.geom.budget.frame_bytes = (((ms / per_mib.max(1e-6)) * f64::from(1 << 20)) as u64).clamp(1 << 20, max_bytes);
        }
        // Geometry of title objects unloaded since the last frame.
        for key in geometry::take_evict() {
            if self.geom.drop_key(known, &key)
                && let Some(old) = self.cache.remove(&key)
            {
                self.release(old);
                st.evicted += 1;
            }
        }
        let plan = self.geom.plan(known, list.frame, &refs);
        for key in &plan.evict {
            if let Some(old) = self.cache.remove(key) {
                self.release(old);
                st.evicted += 1;
            }
        }
        self.staged = 0;
        self.make_ms = 0.0;
        self.copies.clear();
        let tup = std::time::Instant::now();
        let before = (st.uploaded_geometries, st.uploaded_bytes);
        for (key, g) in plan.upload {
            if let Some(old) = self.cache.remove(&key) {
                self.release(old);
                st.replaced += 1;
            }
            match self.upload(&key, &g, &mut st.clamped) {
                Ok(Some(gpu)) => {
                    st.uploaded_geometries += 1;
                    st.uploaded_bytes += gpu.bytes;
                    self.cache.get_or_make(key, list.frame, || Some(gpu));
                }
                Ok(None) => self.geom.failed(known, &key),
                Err(err) => {
                    self.geom.failed(known, &key);
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| xnw_core::warn!("world render: geometry upload failed ({err}); asked again"));
                }
            }
            // `g` (the CPU bytes) goes here.
        }
        st.pending = self.geom.pending_count();
        st.in_place_geoms = self.zc_geoms.len();
        st.upload_ms = tu.elapsed().as_secs_f64() * 1e3;
        let (n, b) = (st.uploaded_geometries - before.0, st.uploaded_bytes - before.1);
        if n >= 4 {
            // `cutfast=on`: without page and staging creation (a one-off at a cut's first frame).
            let made = if self.cuts.fast { self.make_ms } else { 0.0 };
            let ms = (tup.elapsed().as_secs_f64() * 1e3 - made).max(0.01);
            let sample = (ms / n as f64, ms / (b as f64 / f64::from(1 << 20)).max(1e-3));
            let c = &mut self.upload_cost;
            *c = if c.0 == 0.0 { sample } else { (0.8 * c.0 + 0.2 * sample.0, 0.8 * c.1 + 0.2 * sample.1) };
        }
        st.upload_cost = self.upload_cost;
        // Coverage of the main view, then whether this frame is shown.
        let cover: Vec<(u32, bool)> = draws.iter().map(|r| match self.cache.get_or_make(r.geom, list.frame, || None) {
            Some(g) => (g.indices, true),
            None => (self.geom.pending_indices(&r.geom), false),
        }).collect();
        st.coverage = cuts::coverage(&cover);
        st.held = self.cuts.frame(list.frame, st.coverage, st.uploaded_bytes, t0.elapsed().as_secs_f64() * 1e3, now) == cuts::Show::Hold;
        // This frame's palettes: one buffer, a float4 index per palette.
        let mut first = Vec::with_capacity(list.palettes.len());
        let mut floats: Vec<f32> = Vec::new();
        for p in &list.palettes {
            first.push((floats.len() / 4) as u32);
            // Whole (bones, then any remap entries), padded to a float4.
            floats.extend_from_slice(p);
            floats.resize(floats.len().div_ceil(4) * 4, 0.0);
        }
        floats.resize(floats.len().max(4), 0.0);
        let turn = self.palette_turn;
        // This frame's Light buffer (zeros: the old look).
        if self.light_cb[turn].is_none() {
            let b = upload_buffer(&self.device, &[0u8; 256]).map_err(at("light buffer create (256 B)".into()))?;
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
            // SAFETY: an upload buffer stays mapped for its life.
            unsafe { b.Map(0, None, Some(&mut p)) }.map_err(at("light buffer Map".into()))?;
            self.light_cb[turn] = Some((b, p.cast::<u8>()));
        }
        let lc = list.light.map_or([0f32; 28], |l| l.constants(self.light_old));
        let (light_buf, light_map) = self.light_cb[turn].as_ref().expect("made above");
        // SAFETY: 112 bytes into the 256-byte mapping.
        unsafe { std::ptr::copy_nonoverlapping(lc.as_ptr().cast::<u8>(), *light_map, 112) };
        // SAFETY: a live buffer.
        let light_va = unsafe { light_buf.GetGPUVirtualAddress() };
        self.light_va = light_va;
        self.palette_turn ^= 1;
        // Two persistent mapped upload buffers (alternating; the GPU
        // finished with the other one: every draw waits), grown when needed;
        // An earlier version made a new committed buffer each frame.
        let need = floats.len() * 4;
        if self.palettes[turn].as_ref().is_none_or(|p| p.2 < need) {
            let size = need.next_power_of_two().max(1 << 20);
            let buffer = upload_buffer(&self.device, &vec![0u8; size]).map_err(at(format!("palette buffer create ({size} B for {need} B, {} palettes)", list.palettes.len())))?;
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
            // SAFETY: an upload buffer stays mapped for its life.
            unsafe { buffer.Map(0, None, Some(&mut p)) }.map_err(at(format!("palette buffer Map ({size} B)")))?;
            self.palettes[turn] = Some((buffer, p.cast::<u8>(), size));
        }
        let (palette, mapped, _) = self.palettes[turn].as_ref().expect("made above");
        // SAFETY: `need` bytes fit in the mapping (checked above).
        unsafe { std::ptr::copy_nonoverlapping(floats.as_ptr().cast::<u8>(), *mapped, need) };
        let palette = palette.clone();
        self.raw_view(&palette, need as u64, turn as u32);
        st.palette_bytes = need as u64;
        let pal_bytes = need as u32;
        // This frame's instance worlds, the same way (heap slot 2 + turn).
        let ineed = (list.instances.len() * 4).max(16);
        if self.instances[turn].as_ref().is_none_or(|p| p.2 < ineed) {
            let size = ineed.next_power_of_two().max(1 << 20);
            let buffer = upload_buffer(&self.device, &vec![0u8; size]).map_err(at(format!("instance buffer create ({size} B for {} instances)", list.instances.len() / 12)))?;
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
            // SAFETY: an upload buffer stays mapped for its life.
            unsafe { buffer.Map(0, None, Some(&mut p)) }.map_err(at(format!("instance buffer Map ({size} B)")))?;
            self.instances[turn] = Some((buffer, p.cast::<u8>(), size));
        }
        let (ibuf, imapped, _) = self.instances[turn].as_ref().expect("made above");
        // SAFETY: the instances fit in the mapping (checked above).
        unsafe { std::ptr::copy_nonoverlapping(list.instances.as_ptr().cast::<u8>(), *imapped, list.instances.len() * 4) };
        let ibuf = ibuf.clone();
        self.raw_view(&ibuf, ineed as u64, 2 + turn as u32);
        let inst_bytes = (list.instances.len() * 4) as u32;
        let inst_total = (list.instances.len() / 12) as u32;
        // SAFETY: recording on our own list and objects; every pointer passed outlives its call.
        unsafe {
            self.alloc.Reset().map_err(at("allocator Reset".into()))?;
            self.list.Reset(&self.alloc, &self.pso).map_err(at("list Reset".into()))?;
            let list_com = self.list.clone();
            let l = &list_com;
            // One timestamp pair per submission; `chunk`: the frame in several.
            self.stamps_ready()?;
            let (mut sub_k, mut chunk_from, mut chunk_wait) = (0u32, 0usize, 0f64);
            self.stamp_open(l, sub_k);
            let (rtv, dsv) = (self.rtv.GetCPUDescriptorHandleForHeapStart(), self.dsv.GetCPUDescriptorHandleForHeapStart());
            l.ClearRenderTargetView(rtv, &[0.09, 0.09, 0.1, 1.0], None);
            l.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 0.0, 0, None);
            // `gbuf=on`: the G-buffer targets follow the colour target (one range of three).
            let targets = if self.gbuf.is_some() {
                let rstep = self.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
                for k in 1..3 {
                    l.ClearRenderTargetView(D3D12_CPU_DESCRIPTOR_HANDLE { ptr: rtv.ptr + k * rstep }, &[0.0; 4], None);
                }
                3
            } else {
                1
            };
            l.OMSetRenderTargets(targets, Some(&rtv), true, Some(&dsv));
            l.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: self.width as f32, Height: self.height as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
            l.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: self.width as i32, bottom: self.height as i32 }]);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            l.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            // This frame's geometry copies, then their buffers to the read states.
            if !self.copies.is_empty() {
                // A copy outside either buffer would fail the list at Close; skipped and named.
                for (dst, src, from, len, to) in &self.copies {
                    let (dw, sw) = (dst.GetDesc().Width, src.GetDesc().Width);
                    if to.checked_add(*len).is_none_or(|x| x > dw) || from.checked_add(*len).is_none_or(|x| x > sw) {
                        st.copies_refused += 1;
                        xnw_core::warn!("world render: frame {frame}: geometry copy skipped: {len} B from staging {from:#x} (of {sw:#x}) to page {to:#x} (of {dw:#x})");
                        continue;
                    }
                    l.CopyBufferRegion(dst, *to, src, *from, *len);
                }
                let read = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_INDEX_BUFFER | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                // Each page copied into once moves to the read state (buffers decay to COMMON after the list).
                let mut done: Vec<&ID3D12Resource> = Vec::new();
                for (dst, ..) in &self.copies {
                    if !done.iter().any(|d| *d == dst) {
                        transition(l, dst, D3D12_RESOURCE_STATE_COPY_DEST, read);
                        done.push(dst);
                    }
                }
            }
            // The ped pre-skin: draw index -> its vertex-buffer copy (slot, bytes).
            let pre = self.preskin_record(list, l);
            // every bridged row's mesh program, then one barrier; the
            // record ring's raw view for `vs_rec`.
            if let Some(b) = ms_rows {
                b.record_compute(l);
                l.SetDescriptorHeaps(&[Some(self.heap.clone())]);
                if let (Some((ring, bytes)), Some(slot)) = (b.records(), self.rec_slot) {
                    self.raw_view(ring, bytes, slot);
                }
            }
            // The device cull of the instanced rows (`gpucull=on`): per list draw its argument slot.
            let culled = self.cull_record(list, l);
            // This frame's albedo table (uploads recorded on this list first).
            let (heap_cpu, step) = (self.heap.GetCPUDescriptorHandleForHeapStart().ptr, self.heap_step as usize);
            let cpu = |slot: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: heap_cpu + slot as usize * step };
            // The layers' textures share the table.
            // Layer slots take the first entries (k[50] packs them in 8 bits each; at most 192 per frame in the captures).
            // Timed (CUT1); a fast camera cut caps its new reads (cuts::CUT_TEX_MS).
            let tt = std::time::Instant::now();
            let tex_base = self.tex.opts.frame_ms;
            self.tex.opts.frame_ms = self.cuts.tex_ms(tex_base);
            // The bridged rows' textures last (our rows' come first in the table).
            let bridged = list.bridged.as_deref().filter(|_| self.bridge_on || self.bridge_warm);
            self.tex.prepare(&self.device, l, &mut draws.iter().flat_map(|r| r.material.layers).chain(draws.iter().map(|r| r.material.albedo_desc)).chain(bridged.into_iter().flat_map(|f| f.rows.iter().flat_map(|r| r.1.handles()))), &cpu);
            self.tex.opts.frame_ms = tex_base;
            st.tex_ms = tt.elapsed().as_secs_f64() * 1e3;
            let tex_table = self.gpu_slot(self.tex.base());
            if let Some(s) = list.sky.filter(|_| self.sky)
                && let Some(inv) = invert(&cam.view_proj_relative())
            {
                let mut k = [0f32; 40];
                k[..16].copy_from_slice(&inv);
                for (j, v) in [s.sun, s.sun_col, s.zenith, s.horizon, s.moon, s.fog].iter().enumerate() {
                    k[16 + 4 * j..20 + 4 * j].copy_from_slice(v);
                }
                l.SetGraphicsRootSignature(&self.sky_root);
                l.SetGraphicsRootConstantBufferView(1, light_va);
                l.SetPipelineState(&self.sky_pso);
                l.SetGraphicsRoot32BitConstants(0, 40, k.as_ptr().cast(), 0);
                l.DrawInstanced(3, 1, 0, 0);
                st.sky = true;
            }
            // The heightmap terrain (opaque, depth) on its own heap; ours again after.
            if list.terrain.is_some() && self.terrain.opts.on {
                st.terrain = self.terrain.record(&self.device, l, list, &cam);
                l.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            }
            // Imposter cards (opaque, depth; root constants only); `show`: after the draws.
            if list.imposters.is_some() && self.imposters.opts.on && !self.imposters.opts.show {
                self.imposters.record(l, list, &cam);
            }
            let passes: &[bool] = if self.wire { &[false, true] } else { &[false] };
            // `gbuf=on`: targets bound (3; 1 for transparent rows and dual-source states).
            let (mut rts, mut gbuf_done) = (if self.gbuf.is_some() { 3u32 } else { 1 }, false);
            for &wire in passes {
                let mut raw_bound: Option<bool> = None;
                // Table 3 holds the cull's output (a device-culled row) rather than the frame's instances.
                let mut inst_cull_bound = false;
                let mut indirect: Option<u32> = None;
                let mut bound = Bound::Raw;
                let mut water_done = wire;
                for r in &draws {
                    // `chunk`: every N draws the list so far is sent and waited for; the new
                    // list gets the frame's targets, viewport and heap again, the rest at the next draw.
                    if self.chunk > 0 && st.draws + st.bridged >= chunk_from + self.chunk as usize {
                        chunk_from = st.draws + st.bridged;
                        self.stamp_close(l, sub_k);
                        l.Close().map_err(at(format!("chunk list Close after {chunk_from} draws")))?;
                        let (w, g) = self.send_wait(l, sub_k, "pass chunk", frame)?;
                        (chunk_wait, st.submits, sub_k) = (chunk_wait + w, st.submits + 1, sub_k + 1);
                        (st.submit_wait_ms, st.submit_gpu_ms) = (st.submit_wait_ms.max(w), st.submit_gpu_ms.max(g.unwrap_or(0.0)));
                        self.alloc.Reset().map_err(at("chunk allocator Reset".into()))?;
                        self.list.Reset(&self.alloc, &self.pso).map_err(at("chunk list Reset".into()))?;
                        self.stamp_open(l, sub_k);
                        let (rtv, dsv) = (self.rtv.GetCPUDescriptorHandleForHeapStart(), self.dsv.GetCPUDescriptorHandleForHeapStart());
                        l.OMSetRenderTargets(rts, Some(&rtv), true, Some(&dsv));
                        l.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: self.width as f32, Height: self.height as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
                        l.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: self.width as i32, bottom: self.height as i32 }]);
                        l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                        l.SetDescriptorHeaps(&[Some(self.heap.clone())]);
                        raw_bound = None;
                        inst_cull_bound = false;
                    }
                    // Water after opaque and decals, before transparent (14).
                    if !water_done && TRANSPARENT_STAGES.contains(&r.stage) {
                        water_done = true;
                        st.water = self.record_water(list, &cam).map_err(at("water grid upload".into()))?;
                        raw_bound = None;
                        // `gbuf=on`: transparent rows come after the title's G-buffer.
                        gbuf_done = true;
                    }
                    // a bridged row draws from its records, not our geometry.
                    let idx = ((*r as *const DrawRecord as usize).wrapping_sub(list.draws.as_ptr() as usize) / std::mem::size_of::<DrawRecord>()) as u32;
                    if let Some(b) = ms_rows.filter(|_| !wire && (idx as usize) < list.draws.len())
                        && let Some(it) = b.item(idx)
                    {
                        let pr = b.prog(it.prog);
                        if pr.title_ps {
                            b.record_title(l, it);
                            l.SetDescriptorHeaps(&[Some(self.heap.clone())]);
                            st.bridged_title_ps += 1;
                        } else if let (Some(pso), Some(slot), Some(j), Some((ibv, n)), Some((_, ring_bytes))) = (&self.rec_pso, self.rec_slot, pr.join, b.indices(it), b.records()) {
                            let c = draw_constants(r, &cam);
                            let mut k = [0u32; 58];
                            for (i, v) in c.iter().enumerate() {
                                k[i] = v.to_bits();
                            }
                            let (ti, srgb) = self.tex.lookup(r.material.albedo_desc);
                            k[48] = ti.min(TEX_SLOTS - 1);
                            k[49] = u32::from(r.material.alpha_clip) | u32::from(srgb) << 1;
                            self.tex.count_draw(r.material.albedo_desc);
                            let jc = j.constants(&pr.p.layout);
                            (k[32], k[33], k[36], k[37], k[39], k[56]) = (ring_bytes.min(u64::from(u32::MAX)) as u32, b.record_at(it) as u32, pr.p.layout.record_bytes(), jc[0], jc[1], jc[2]);
                            k[35] = u32::from(r.ped) << 1 | u32::from(self.id_mode) << 2;
                            k[57] = idx + 1;
                            if let Some(s) = list.sun {
                                for (j, v) in s.iter().enumerate() {
                                    k[40 + j] = v.to_bits();
                                }
                            }
                            if let Some(s) = list.sky.filter(|_| self.sky) {
                                for (j, v) in s.fog.iter().enumerate() {
                                    k[52 + j] = v.to_bits();
                                }
                            }
                            l.SetGraphicsRootSignature(&self.raw_root);
                            l.SetPipelineState(pso);
                            l.SetGraphicsRootDescriptorTable(1, self.gpu_slot(slot));
                            l.SetGraphicsRootDescriptorTable(2, self.gpu_slot(turn as u32));
                            l.SetGraphicsRootDescriptorTable(3, self.gpu_slot(2 + turn as u32));
                            l.SetGraphicsRootDescriptorTable(4, tex_table);
                            l.SetGraphicsRootConstantBufferView(5, light_va);
                            l.SetGraphicsRoot32BitConstants(0, 58, k.as_ptr().cast(), 0);
                            l.IASetIndexBuffer(Some(&ibv));
                            l.DrawIndexedInstanced(n, 1, 0, 0, 0);
                        } else {
                            continue;
                        }
                        // Our root signature, heap tables and pipeline are bound again at the next raw draw.
                        raw_bound = None;
                        st.bridged += 1;
                        continue;
                    }
                    let Some((ibv, vbv, indices, raw)) = self.cache.get_or_make(r.geom, list.frame, || None).map(|g| (g.ibv, g.vbv, g.indices, g.raw)) else {
                        if !wire {
                            st.skipped += 1;
                            // Why, per draw.
                            st.missing[match self.geom.missing(&r.geom) {
                                Some(geometry::Missing::Waiting) => 0,
                                Some(geometry::Missing::Full) => 1,
                                _ => 2,
                            }] += 1;
                        }
                        continue;
                    };
                    if indices == 0 || (wire && raw.is_none()) {
                        continue;
                    }
                    let c = draw_constants(r, &cam);
                    let mut instances = 1u32;
                    // The title's state for this draw (raw path, solid
                    // pass); without one, the decal pipeline for stage 74.
                    let want = if wire {
                        Bound::Wire
                    } else if self.id_mode {
                        Bound::Raw
                    } else if let Some(s) = r.state.filter(|_| self.states && raw.is_some()) {
                        // New pipelines within the frame's budget; the rest draw with
                        // our defaults until a later frame makes them.
                        if !self.pso_async && !self.state_psos.contains_key(&s) && self.state_psos.len() < MAX_STATE_PSOS && new_psos < self.pso_budget {
                            let tp = std::time::Instant::now();
                            new_psos += 1;
                            match pipeline_state(&self.device, &self.raw_root, &self.raw_shaders.0, &self.raw_shaders.1, self.raw_plain_ps.as_ref(), &s) {
                                Ok(p) => {
                                    self.state_psos.insert(s, p);
                                }
                                Err(err) => xnw_core::warn!("world render: state pipeline {s:?} failed: {err}"),
                            }
                            st.pso_ms += tp.elapsed().as_secs_f64() * 1e3;
                        }
                        if self.state_psos.contains_key(&s) {
                            st.stated += 1;
                            Bound::State(s)
                        } else if DECAL_STAGES.contains(&r.stage) {
                            Bound::Decal
                        } else {
                            Bound::Raw
                        }
                    } else if raw.is_some() && DECAL_STAGES.contains(&r.stage) {
                        Bound::Decal
                    } else {
                        Bound::Raw
                    };
                    if raw_bound == Some(true) && want != bound {
                        match want {
                            Bound::Raw => l.SetPipelineState(&self.raw_pso),
                            Bound::Wire => l.SetPipelineState(&self.wire_pso),
                            Bound::Decal => l.SetPipelineState(&self.decal_pso),
                            Bound::State(s) => l.SetPipelineState(&self.state_psos[&s]),
                        }
                        bound = want;
                    }
                    // A dual-source state blends into one target only (bound with three, WARP drew it wrong).
                    if self.gbuf.is_some() {
                        let n = if gbuf_done || matches!(want, Bound::State(s) if gbuf::dual_source(&s.blend)) { 1 } else { 3 };
                        if n != rts {
                            let (rtv, dsv) = (self.rtv.GetCPUDescriptorHandleForHeapStart(), self.dsv.GetCPUDescriptorHandleForHeapStart());
                            l.OMSetRenderTargets(n, Some(&rtv), true, Some(&dsv));
                            rts = n;
                        }
                    }
                    l.IASetIndexBuffer(Some(&ibv));
                    match raw {
                        Some((slot, vb_bytes)) => {
                            if raw_bound != Some(true) {
                                l.SetGraphicsRootSignature(&self.raw_root);
                                match want {
                                    Bound::Raw => l.SetPipelineState(&self.raw_pso),
                                    Bound::Wire => l.SetPipelineState(&self.wire_pso),
                                    Bound::Decal => l.SetPipelineState(&self.decal_pso),
                                    Bound::State(s) => l.SetPipelineState(&self.state_psos[&s]),
                                }
                                bound = want;
                                l.SetGraphicsRootDescriptorTable(2, self.gpu_slot(turn as u32));
                                l.SetGraphicsRootDescriptorTable(3, self.gpu_slot(2 + turn as u32));
                                inst_cull_bound = false;
                                l.SetGraphicsRootDescriptorTable(4, tex_table);
                                l.SetGraphicsRootConstantBufferView(5, light_va);
                                raw_bound = Some(true);
                            }
                            let (pf, pb) = match list.palettes.get(r.palette as usize) {
                                Some(p) if r.palette != NO_PALETTE && self.skin => match r.bone_count {
                                    0 => (first[r.palette as usize], (p.len() / 12) as u32),
                                    // Bone k of the draw is palette entry curIndex + k (bounded by palBytes in the shader).
                                    n => (first[r.palette as usize] + 3 * r.bone_first, n),
                                },
                                _ => (0, 0),
                            };
                            st.skinned += usize::from(pb > 0 && !wire);
                            let mut k = [0u32; 58];
                            for (i, v) in c.iter().enumerate() {
                                k[i] = v.to_bits();
                            }
                            // The albedo entry of our table (0: flat) and the material flags.
                            let (ti, srgb) = self.tex.lookup(r.material.albedo_desc);
                            k[48] = ti.min(TEX_SLOTS - 1);
                            k[49] = u32::from(r.material.alpha_clip) | u32::from(srgb) << 1;
                            // Layer entries (each checked < 256 by the table) and flags.
                            let has_layers = r.material.layers.iter().any(|s| *s != 0);
                            let debug = self.layers_debug && !has_layers && LAYER_DEBUG_STAGES.contains(&r.stage);
                            if self.layers != 0 && (has_layers || debug) {
                                k[50] = r.material.layers.iter().enumerate().map(|(j, s)| textures::layer_entry(self.tex.lookup(*s).0) << (8 * j)).fold(0, |a, b| a | b);
                                k[51] = 1 | (self.layers & 3) << 1 | u32::from(debug) << 3;
                                st.layered += usize::from(!wire);
                            }
                            // The sample mask: lanes kept in bits 4-7, bit 8 on.
                            if self.stipple && r.material.stipple != 0 {
                                k[51] |= 0x100 | u32::from(r.material.stipple & 0xF) << 4;
                                st.stippled += usize::from(!wire);
                            }
                            if !wire {
                                self.tex.count_draw(r.material.albedo_desc);
                            }
                            k[32] = vb_bytes;
                            k[33] = pf;
                            k[34] = pb;
                            k[35] = u32::from(wire) | u32::from(r.ped) << 1 | u32::from(self.id_mode) << 2 | (self.skin_rules & 0xFF) << 8;
                            // The draw's index in the list + 1 (ID mode).
                            k[57] = ((*r as *const DrawRecord as usize).wrapping_sub(list.draws.as_ptr() as usize) / std::mem::size_of::<DrawRecord>()) as u32 + 1;
                            // The remap's dword index in t1 (relative to the draw's palette floats).
                            k[56] = match list.palettes.get(r.palette as usize) {
                                Some(_) if r.remap != NO_REMAP && pb > 0 => 4 * first[r.palette as usize] + r.remap,
                                _ => u32::MAX,
                            };
                            st.remapped += usize::from(k[56] != u32::MAX && !wire);
                            k[36] = pal_bytes;
                            // Instances: clamped to what this frame's buffer holds.
                            let first = r.inst_first.min(inst_total);
                            let count = r.inst_count.min(inst_total - first).min(MAX_INSTANCES);
                            k[37] = first;
                            k[38] = count;
                            k[39] = inst_bytes;
                            // A device-culled row reads the cull's output: its reserved run, the survivors drawn indirectly.
                            let cull_slot = culled.as_ref().and_then(|c| c.slot_of(k[57].wrapping_sub(1)));
                            if let (Some(c), Some(_)) = (&culled, cull_slot) {
                                k[37] = r.inst_first;
                                k[38] = r.inst_count.min(MAX_INSTANCES);
                                k[39] = c.out_bytes.min(u64::from(u32::MAX)) as u32;
                            }
                            if cull_slot.is_some() != inst_cull_bound {
                                inst_cull_bound = cull_slot.is_some();
                                l.SetGraphicsRootDescriptorTable(3, match (&culled, inst_cull_bound) {
                                    (Some(c), true) => self.gpu_slot(c.view_slot),
                                    _ => self.gpu_slot(2 + turn as u32),
                                });
                            }
                            indirect = cull_slot;
                            if let Some(s) = list.sun {
                                for (j, v) in s.iter().enumerate() {
                                    k[40 + j] = v.to_bits();
                                }
                            }
                            instances = count.max(1);
                            st.instanced += usize::from((count > 0 || indirect.is_some()) && !wire);
                            st.instances += u64::from(count) * u64::from(!wire && indirect.is_none());
                            if let Some(s) = list.sky.filter(|_| self.sky) {
                                for (j, v) in s.fog.iter().enumerate() {
                                    k[52 + j] = v.to_bits();
                                }
                            }
                            // A pre-skinned draw reads its copy: positions already skinned, no palette.
                            let slot = match pre.get(&k[57].wrapping_sub(1)) {
                                Some(&(s, bytes)) => {
                                    (k[32], k[34], k[56]) = (bytes, 0, u32::MAX);
                                    s
                                }
                                None => slot,
                            };
                            l.SetGraphicsRoot32BitConstants(0, 58, k.as_ptr().cast(), 0);
                            l.SetGraphicsRootDescriptorTable(1, self.gpu_slot(slot));
                        }
                        None => {
                            if raw_bound != Some(false) {
                                l.SetGraphicsRootSignature(&self.root);
                                l.SetPipelineState(&self.pso);
                                raw_bound = Some(false);
                            }
                            l.SetGraphicsRoot32BitConstants(0, 32, c.as_ptr().cast(), 0);
                            l.IASetVertexBuffers(0, Some(&[vbv]));
                        }
                    }
                    match (indirect.take(), &culled) {
                        (Some(slot), Some(c)) => l.ExecuteIndirect(&c.signature, 1, &c.args, u64::from(slot) * u64::from(gpucull::ARG_BYTES), None, 0),
                        _ => l.DrawIndexedInstanced(indices, instances, 0, 0, 0),
                    }
                    if !wire {
                        st.draws += 1;
                        st.triangles += u64::from(indices / 3);
                    }
                }
                if !water_done {
                    st.water = self.record_water(list, &cam).map_err(at("water grid upload".into()))?;
                }
            }
            if list.imposters.is_some() && self.imposters.opts.show {
                self.imposters.record(l, list, &cam);
            }
            // The title's own pixel programs on the bridged rows, over ours.
            if self.bridge_on
                && let Some(f) = list.bridged.as_deref()
            {
                self.bridge_stats = self.record_bridged(l, f).map_err(|x| format!("frame {frame}: bridgeps: {x}"))?;
            }
            // A frame complete enough becomes the shown picture.
            if self.cuts.hold && !st.held {
                transition(l, &self.colour, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
                transition(l, &self.shown, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);
                l.CopyResource(&self.shown, &self.colour);
                transition(l, &self.shown, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET);
                transition(l, &self.colour, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            }
            if readback {
                transition(l, &self.colour, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
                let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: 0, Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: COLOUR, Width: self.width, Height: self.height, Depth: 1, RowPitch: self.width * 4 } };
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: ManuallyDrop::new(Some(self.readback.clone())), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: ManuallyDrop::new(Some(self.colour.clone())), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
                l.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                drop(ManuallyDrop::into_inner(dst.pResource));
                drop(ManuallyDrop::into_inner(src.pResource));
                transition(l, &self.colour, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            }
            self.stamp_close(l, sub_k);
            l.Close().map_err(at(format!(
                "list Close (an invalid recorded call): {} geometry copies ({} skipped), {} texture uploads this frame, {} draws, {} instances, palettes {} B, terrain {}",
                self.copies.len(),
                st.copies_refused,
                self.tex.stats.uploads,
                st.draws,
                list.instances.len() / 12,
                st.palette_bytes,
                self.terrain.opts.on
            )))?;
            st.record_ms = t0.elapsed().as_secs_f64() * 1e3 - chunk_wait;
            let (w, g) = self.send_wait(l, sub_k, "pass", frame)?;
            st.submits += 1;
            (st.submit_wait_ms, st.submit_gpu_ms) = (st.submit_wait_ms.max(w), st.submit_gpu_ms.max(g.unwrap_or(0.0)));
            // The copies ran: the staging space is free again (and a large
            // one made for a big offline frame goes).
            self.copies.clear();
            self.bridge_keep.clear();
            if self.staging.as_ref().is_some_and(|s| s.2 > 256 << 20) {
                self.staging = None;
            }
            self.tex.finished();
            self.terrain.finished();
            st.gpu_ms = w + chunk_wait;
        }
        st.new_psos += new_psos;
        self.cuts.parts([st.upload_ms, st.tex_ms, (st.record_ms - st.upload_ms - st.tex_ms).max(0.0), st.gpu_ms]);
        self.cuts.end(t0.elapsed().as_secs_f64() * 1e3, cuts::now_ms());
        Ok(st)
    }

    /// The pass's timestamp heap and readback, made once.
    fn stamps_ready(&mut self) -> std::result::Result<(), String> {
        if self.stamps.is_none() {
            let mut h: Option<ID3D12QueryHeap> = None;
            // SAFETY: our device; the heap is kept with the pass.
            unsafe { self.device.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2 * STAMP_PAIRS, NodeMask: 0 }, &mut h) }.map_err(|e| format!("pass query heap: {e}"))?;
            let rb = bridge::buffer(&self.device, 16 * u64::from(STAMP_PAIRS), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
            self.stamps = Some((h.ok_or("pass query heap: none")?, rb));
        }
        Ok(())
    }

    /// The first timestamp of submission `k` on the open list (none past [`STAMP_PAIRS`]).
    fn stamp_open(&self, l: &ID3D12GraphicsCommandList, k: u32) {
        if let Some((h, _)) = self.stamps.as_ref().filter(|_| k < STAMP_PAIRS) {
            // SAFETY: an open list, a query index inside the heap.
            unsafe { l.EndQuery(h, D3D12_QUERY_TYPE_TIMESTAMP, 2 * k) };
        }
    }

    /// The second timestamp of submission `k` and the pair's resolve, before the list closes.
    fn stamp_close(&self, l: &ID3D12GraphicsCommandList, k: u32) {
        if let Some((h, rb)) = self.stamps.as_ref().filter(|_| k < STAMP_PAIRS) {
            // SAFETY: as above; the readback holds 2 * STAMP_PAIRS values.
            unsafe {
                l.EndQuery(h, D3D12_QUERY_TYPE_TIMESTAMP, 2 * k + 1);
                l.ResolveQueryData(h, D3D12_QUERY_TYPE_TIMESTAMP, 2 * k, 2, rb, 16 * u64::from(k));
            }
        }
    }

    /// Sends the closed list `l` and waits for its fence under the GPU
    /// watchdog (`xnw_gpu::gpuwatch`: the card's time of this submission
    /// alone, not the CPU's recording); its fence wait and timestamped GPU ms.
    fn send_wait(&mut self, l: &ID3D12GraphicsCommandList, k: u32, label: &'static str, frame: u64) -> std::result::Result<(f64, Option<f64>), String> {
        let at = |what: &'static str| move |e: windows::core::Error| format!("frame {frame}: {label}: {what}: {e}");
        // SAFETY: our queue, fence and a closed list of ours; the event is closed here.
        let wait = unsafe {
            self.queue.ExecuteCommandLists(&[Some(l.cast().map_err(at("list cast"))?)]);
            let sent = xnw_gpu::gpuwatch::begin(label);
            self.fence_value += 1;
            self.queue.Signal(&self.fence, self.fence_value).map_err(at("queue Signal"))?;
            let event: HANDLE = CreateEventW(None, false, false, None).map_err(at("CreateEvent"))?;
            self.fence.SetEventOnCompletion(self.fence_value, event).map_err(at("SetEventOnCompletion"))?;
            WaitForSingleObject(event, INFINITE);
            let _ = CloseHandle(event);
            if let Some(r) = xnw_gpu::d3d12x::host::removal_report(&self.device) {
                xnw_gpu::gpuwatch::end(label, sent, None);
                return Err(format!("frame {frame}: after the {label} wait: {r}"));
            }
            let stamp = self.stamp_ms(k);
            (xnw_gpu::gpuwatch::end(label, sent, stamp), stamp)
        };
        Ok(wait)
    }

    /// Submission `k`'s GPU milliseconds by its timestamps, once its fence was reached.
    fn stamp_ms(&self, k: u32) -> Option<f64> {
        let (_, rb) = self.stamps.as_ref().filter(|_| k < STAMP_PAIRS)?;
        // SAFETY: the copy is done (the fence was reached); 16 B of the readback read.
        let t = unsafe {
            let mut p = std::ptr::null_mut();
            rb.Map(0, Some(&D3D12_RANGE { Begin: 16 * k as usize, End: 16 * k as usize + 16 }), Some(&mut p)).ok()?;
            let t = std::slice::from_raw_parts(p.cast::<u64>().add(2 * k as usize), 2).to_vec();
            rb.Unmap(0, Some(&D3D12_RANGE::default()));
            t
        };
        // SAFETY: our queue.
        let f = unsafe { self.queue.GetTimestampFrequency() }.ok()?;
        Some(t[1].saturating_sub(t[0]) as f64 * 1e3 / f.max(1) as f64)
    }

    /// Records the frame's water (uploading its grid when it changed):
    /// whether it was drawn.
    /// [`bridged::record`] on our open list `l`, into our colour and depth;
    /// the grey stand-in made on first use.
    fn record_bridged(&mut self, l: &ID3D12GraphicsCommandList, f: &bridged::Frame) -> std::result::Result<bridged::Stats, String> {
        let e = |x: windows::core::Error| x.message().to_string();
        // SAFETY: our device and open list; the upload goes to the fence with `bridge_keep`.
        unsafe {
            if self.bridge_grey.is_none() {
                let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 1, Height: 1, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8G8B8A8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() };
                let mut t: Option<ID3D12Resource> = None;
                self.device.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut t).map_err(e)?;
                let t = t.ok_or("no grey texture")?;
                let bdesc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: 256, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
                let mut up: Option<ID3D12Resource> = None;
                self.device.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &bdesc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut up).map_err(e)?;
                let up = up.ok_or("no grey upload")?;
                let mut at = std::ptr::null_mut();
                up.Map(0, None, Some(&mut at)).map_err(e)?;
                std::ptr::copy_nonoverlapping([128u8, 128, 128, 255].as_ptr(), at.cast::<u8>(), 4);
                up.Unmap(0, None);
                let fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: 0, Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: DXGI_FORMAT_R8G8B8A8_UNORM, Width: 1, Height: 1, Depth: 1, RowPitch: 256 } };
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: ManuallyDrop::new(Some(t.clone())), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: ManuallyDrop::new(Some(up.clone())), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
                l.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                drop(ManuallyDrop::into_inner(dst.pResource));
                drop(ManuallyDrop::into_inner(src.pResource));
                transition(l, &t, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                self.bridge_keep.push(windows::core::Interface::cast(&up).map_err(e)?);
                self.bridge_grey = Some(t);
            }
            // The words buffer: grown to the frame's largest offset (at most
            // 256 MiB), kept; each frame writes only its rows' words.
            let want = f.words_end.next_multiple_of(1 << 20).min(256 << 20);
            if self.bridge_words.as_ref().is_none_or(|w| w.2 < want) {
                if let Some((old, _, _)) = self.bridge_words.take() {
                    old.Unmap(0, None);
                    self.bridge_keep.push(windows::core::Interface::cast(&old).map_err(e)?);
                }
                let bdesc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: want.max(1 << 20), Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
                let mut r: Option<ID3D12Resource> = None;
                self.device.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &bdesc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut r).map_err(e)?;
                let r = r.ok_or("no words buffer")?;
                let mut at = std::ptr::null_mut();
                r.Map(0, None, Some(&mut at)).map_err(e)?;
                self.bridge_words = Some((r, at.cast::<u8>(), want.max(1 << 20)));
            }
            let (words, wp, wlen) = self.bridge_words.clone().ok_or("no words buffer")?;
            let grey = self.bridge_grey.clone().ok_or("no grey texture")?;
            let tex = &self.tex;
            let texture = |h: u64| tex.host_view(h);
            let flat = |h: u64| tex.flat_reason(h) as usize;
            let t = bridged::Target { rtv: self.rtv.GetCPUDescriptorHandleForHeapStart(), dsv: self.dsv.GetCPUDescriptorHandleForHeapStart(), width: self.width, height: self.height, texture: &texture, flat: &flat, grey: &grey, words: (&words, wlen, wp) };
            self.bridge_gpu.record(&self.device, l, f, &t, &mut self.bridge_keep)
        }
    }

    fn record_water(&mut self, list: &FrameList, cam: &Camera) -> Result<bool> {
        let Some(w) = list.water.as_ref().filter(|w| self.water && !w.grid.indices.is_empty()) else { return Ok(false) };
        if self.water_gpu.as_ref().is_none_or(|g| g.0 != w.grid.id) {
            let n = w.grid.indices.len().min(MAX_INDICES) / 3 * 3;
            let vb = upload_buffer(&self.device, as_bytes(&w.grid.vertices))?;
            let ib = upload_buffer(&self.device, as_bytes(&w.grid.indices[..n]))?;
            self.water_gpu = Some((w.grid.id, vb, ib, n as u32));
        }
        let Some((_, vb, ib, n)) = &self.water_gpu else { return Ok(false) };
        let mut k = [0f32; 28];
        k[..16].copy_from_slice(&cam.view_proj_relative());
        k[16..19].copy_from_slice(&w.origin);
        k[20..24].copy_from_slice(&w.tint);
        if let Some(s) = list.sun {
            k[24..27].copy_from_slice(&s[..3]);
        } else {
            k[26] = 1.0;
        }
        // SAFETY: our open list and live buffers.
        unsafe {
            let l = &self.list;
            l.SetGraphicsRootSignature(&self.water_root);
            l.SetGraphicsRootConstantBufferView(1, self.light_va);
            l.SetPipelineState(&self.water_pso);
            l.SetGraphicsRoot32BitConstants(0, 28, k.as_ptr().cast(), 0);
            let vbv = D3D12_VERTEX_BUFFER_VIEW { BufferLocation: vb.GetGPUVirtualAddress(), SizeInBytes: (w.grid.vertices.len() * 12) as u32, StrideInBytes: 12 };
            let ibv = D3D12_INDEX_BUFFER_VIEW { BufferLocation: ib.GetGPUVirtualAddress(), SizeInBytes: n * 4, Format: DXGI_FORMAT_R32_UINT };
            l.IASetVertexBuffers(0, Some(&[vbv]));
            l.IASetIndexBuffer(Some(&ibv));
            l.DrawIndexedInstanced(*n, 1, 0, 0, 0);
        }
        Ok(true)
    }

    /// The colour target as RGBA rows (after a [`Pass::draw`] with readback).
    /// Our colour and depth targets, for the title's own programs to draw
    /// into (`bridgeps`: `bridge::run_into`).
    pub fn bridge_into(&self) -> bridge::Into {
        bridge::Into { colour: self.colour.clone(), depth: self._depth.clone() }
    }

    /// Our depth target read back as float rows (after a [`Pass::draw`]):
    /// reverse depth, what the title's own programs test against (HAIR4: the
    /// strand pixels `hair=on` paints).
    pub fn read_depth(&self) -> std::result::Result<Vec<f32>, String> {
        bridge_ms::read_depth_on(&self.device, &self.queue, &self._depth, self.width, self.height)
    }

    /// `gbuf=on`: the G-buffer targets and the depth read back (after a
    /// [`Pass::draw`]); None with the G-buffer off.
    pub fn read_gbuf(&self) -> Option<std::result::Result<gbuf::Read, String>> {
        let t = self.gbuf.as_ref()?;
        Some(gbuf::read(&self.device, &self.queue, t, &self._depth, self.width, self.height))
    }

    /// The validation layer's stored messages on the pass's device
    /// ([`bridge::layer_messages`]; `XNW_GPU_DEBUG=layer`).
    pub fn layer_messages(&self) -> Vec<String> {
        bridge::layer_messages(&self.device)
    }

    pub fn read_rgba(&self) -> std::result::Result<Vec<u8>, String> {
        let n = (self.width * 4 * self.height) as usize;
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: the readback buffer is `n` bytes; mapped and unmapped here.
        unsafe {
            self.readback.Map(0, None, Some(&mut p)).map_err(|e| format!("readback Map: {e}{}", xnw_gpu::d3d12x::host::removal_report(&self.device).map_or(String::new(), |r| format!("; {r}"))))?;
            let v = std::slice::from_raw_parts(p.cast::<u8>(), n).to_vec();
            self.readback.Unmap(0, None);
            Ok(v)
        }
    }
}

// ---- on screen (`XNW_WORLD_RENDER=screen[,wire]`) ----

/// `XNW_WORLD_RENDER=screen[,wire][,pass=<stage>][,skin=off]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenOptions {
    pub wire: bool,
    pub stage: Option<u32>,
    pub skin: bool,
    /// `state=off` draws with our defaults instead of the title's states.
    pub states: bool,
    /// `sky=off`: no sky stand-in and no fog.
    pub sky: bool,
    /// `water=off`: no water stand-in.
    pub water: bool,
    /// `decalhold=off`: no two-list decal union.
    pub decal_hold: bool,
    /// `peds=on` draws ped skin draws (skipped by default until
    /// skinning is right: broken humans threw polygons across the picture).
    pub peds: bool,
}

pub fn parse_screen(v: &str) -> Option<ScreenOptions> {
    let mut parts = v.split(',').map(str::trim);
    if parts.next() != Some("screen") {
        return None;
    }
    let mut o = ScreenOptions { wire: false, stage: None, skin: true, states: true, sky: true, water: true, peds: false, decal_hold: true };
    for p in parts {
        match p {
            "wire" => o.wire = true,
            "skin=off" => o.skin = false,
            "state=off" => o.states = false,
            "sky=off" => o.sky = false,
            "water=off" => o.water = false,
            "peds=on" => o.peds = true,
            "decalhold=off" => o.decal_hold = false,
            _ => o.stage = p.strip_prefix("pass=").and_then(|s| s.parse().ok()).or(o.stage),
        }
    }
    Some(o)
}

/// `XNW_WORLD_RENDER=screen,...,skinrules=<hex>`: the pass's
/// `skin_rules` live (bit 0 UINT8x4 index byte, 1 identity windows, 2 rigid
/// windows, 3 draw-time bones leave the skip); absent or unparsable: 0. The
/// last one given wins.
pub fn parse_skin_rules(v: &str) -> u32 {
    v.split(',').filter_map(|p| p.trim().strip_prefix("skinrules=")).filter_map(|h| u32::from_str_radix(h.trim_start_matches("0x"), 16).ok()).last().unwrap_or(0) & 0xFF
}

/// The texture switches of `XNW_WORLD_RENDER=screen,...`: `tex=off`
/// (flat colours), `tex=<n>` (table entries, at most [`TEX_SLOTS`]),
/// `texn=<uploads per frame>`, `texmb=<upload MiB per frame>`, `texms=<ms>`
/// (new reads stop after that much of the frame's table; 0: none); textures on
/// (live) by default.
pub fn parse_screen_tex(v: &str) -> TexOptions {
    // Texture reads off the frame only with `texasync=on` (test runs had
    // black regions and missing crowns with it on, and none before it).
    let mut o = TexOptions { mode: TexMode::Live, async_reads: v.split(',').any(|p| p.trim() == "texasync=on"), ..Default::default() };
    for p in v.split(',').map(str::trim) {
        if p == "tex=off" {
            o.mode = TexMode::Off;
        } else if let Some(n) = p.strip_prefix("tex=").and_then(|s| s.parse::<u32>().ok()) {
            o.slots = n.clamp(1, TEX_SLOTS);
        } else if let Some(n) = p.strip_prefix("texn=").and_then(|s| s.parse().ok()) {
            o.uploads = n;
        } else if let Some(n) = p.strip_prefix("texmb=").and_then(|s| s.parse::<u64>().ok()) {
            o.upload_bytes = n << 20;
        } else if let Some(ms) = p.strip_prefix("texms=").and_then(|s| s.parse::<f64>().ok()) {
            o.frame_ms = ms.max(0.0);
        }
    }
    o
}

/// `XNW_WORLD_RENDER=screen,...` geometry budgets: `geom=<device MiB>`,
/// `geomcpu=<queued MiB>`, `geomframe=<upload MiB per frame>`,
/// `geomkeep=<frames>` (defaults [`geometry::GeomBudget::SCREEN`]).
pub fn parse_geom_budget(v: &str) -> geometry::GeomBudget {
    let mut b = geometry::GeomBudget::SCREEN;
    for p in v.split(',').map(str::trim) {
        let Some((k, n)) = p.split_once('=') else { continue };
        let Ok(n) = n.parse::<u64>() else { continue };
        match k {
            "geom" => b.device_bytes = n.max(16) << 20,
            "geomcpu" => b.cpu_bytes = n.max(16) << 20,
            "geomframe" => b.frame_bytes = n.max(1) << 20,
            "geomkeep" => b.keep = n.max(2),
            _ => {}
        }
    }
    b
}

impl Pass {
    /// The geometry line every 60 frames and the memory line every 10 s.
    fn log_geometry(&mut self, frame: u64, st: &PassStats) {
        let g = &self.geom;
        if frame % 60 == 0 {
            xnw_core::info!(
                "world render: geometry: cached {} ({} MiB), uploaded this frame {} ({} KiB), waiting {}, evicted {} (total {}), without geometry {} (reasons: waiting for the frame budget {}, device budget full {}, not delivered {}); failed {}, dropped from the queue {}; uploaded ahead {}, of which later drawn {}, never drawn (evicted) {}; dropped on a title unload {}; ahead pool {} MiB of {} MiB ({} uploads into it)",
                g.resident_count(),
                g.device >> 20,
                st.uploaded_geometries,
                st.uploaded_bytes >> 10,
                g.pending_count(),
                st.evicted,
                g.stats.evicted,
                st.skipped,
                st.missing[0],
                st.missing[1],
                st.missing[2],
                g.stats.failed,
                g.stats.dropped,
                g.stats.ahead,
                g.stats.ahead_drawn,
                g.stats.ahead_evicted,
                g.stats.unloaded,
                g.ahead_pool_bytes() >> 20,
                g.budget.ahead_pool >> 20,
                g.stats.pool_uploads
            );
        }
        static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
        let due = LAST.lock().is_ok_and(|mut l| {
            let now = std::time::Instant::now();
            let d = l.is_none_or(|t| now - t >= std::time::Duration::from_secs(10));
            if d {
                *l = Some(now);
            }
            d
        });
        if due {
            let buffers: usize = self.palettes.iter().chain(&self.instances).flatten().map(|p| p.2).sum();
            xnw_core::info!(
                "world render: memory: geometry {} entries, device {} MiB (peak {} MiB, budget {} MiB), heap slots {} of {}; queued CPU {} MiB in {} (peak {} MiB, budget {} MiB); staging {} MiB; palette+instance buffers {} MiB; state pipelines {}; recorder hand-shake keys {}",
                g.resident_count(),
                g.device >> 20,
                g.stats.peak_device >> 20,
                g.budget.device_bytes >> 20,
                self.next_slot as usize - self.free_slots.len(),
                HEAP_SLOTS,
                g.cpu >> 20,
                g.pending_count(),
                g.stats.peak_cpu >> 20,
                g.budget.cpu_bytes >> 20,
                self.staging.as_ref().map_or(0, |s| s.2) >> 20,
                buffers >> 20,
                self.state_psos.len(),
                geometry::known().len()
            );
        }
    }
}

/// `layers=c0` (COLOR0, stream word 0x30), `layers=c1` (COLOR1, 0x34:
/// the default, chosen offline on one capture's ground: smoother weights)
/// or `layers=off`.
pub fn parse_layers(v: &str) -> u32 {
    let mut m = 2;
    for p in v.split(',').map(str::trim) {
        match p {
            "layers=c0" => m = 1,
            "layers=c1" => m = 2,
            "layers=off" => m = 0,
            _ => {}
        }
    }
    m
}

/// The screen options when `XNW_WORLD_RENDER` asks for the picture on screen.
pub fn screen_mode() -> Option<ScreenOptions> {
    static MODE: std::sync::OnceLock<Option<ScreenOptions>> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| parse_screen(&std::env::var("XNW_WORLD_RENDER").ok()?))
}

/// Hands our picture to the presenter (xnw-gpu's PresentX) when `XNW_WORLD_RENDER` asks for it on screen. Called
/// once at start, before the title's first present.
pub fn install_screen() {
    if screen_mode().is_some() {
        xnw_gpu::d3d12x::set_screen_source(screen_frame);
    }
}

/// The pass of the screen path, on the host device; 1920x1080 (plane 0's size).
static SCREEN: Mutex<Option<Pass>> = Mutex::new(None);

/// Draws the newest finished list into our target and returns it (render-
/// target state) for the presenter; `None` before the first list, or when
/// the pass cannot be made. Called on the PresentX thread of a skipped
/// (null-mode) frame; the GPU work is waited for, so the presenter's queue
/// reads a finished target.
pub(crate) fn screen_frame() -> Option<ID3D12Resource> {
    // Called from PresentX (an extern boundary that cannot unwind):
    // a panic in the pass is logged once with the frame and turns the pass
    // off; the process goes on.
    if SCREEN_PANICKED.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    match std::panic::catch_unwind(screen_frame_inner) {
        Ok(r) => r,
        Err(p) => {
            SCREEN_PANICKED.store(true, std::sync::atomic::Ordering::Relaxed);
            let what = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_default();
            xnw_core::warn!("world render: pass panicked at frame {} ({what}); off", SCREEN_FRAME.load(std::sync::atomic::Ordering::Relaxed));
            *SCREEN.lock().unwrap_or_else(|e| e.into_inner()) = None;
            SCREEN.clear_poison();
            None
        }
    }
}

/// The pass panicked once: off for the rest of the run.
static SCREEN_PANICKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The frame of the list the screen pass last took (for the panic line).
static SCREEN_FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
static SCREEN_FORCE_PANIC: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn screen_frame_inner() -> Option<ID3D12Resource> {
    #[cfg(test)]
    if SCREEN_FORCE_PANIC.load(std::sync::atomic::Ordering::Relaxed) {
        SCREEN_FRAME.store(77, std::sync::atomic::Ordering::Relaxed);
        let (main, kept) = (std::hint::black_box(1usize), std::hint::black_box(3usize));
        let _ = main - kept;
    }
    // The only place the pass is made: never without `XNW_WORLD_RENDER=screen..`
    // (`capture` and `1` record and dump only; no device work of ours).
    screen_mode()?;
    let list = lists().take();
    if let Some(l) = &list {
        SCREEN_FRAME.store(l.frame, std::sync::atomic::Ordering::Relaxed);
    }
    let mut guard = SCREEN.lock().ok()?;
    if guard.is_none() {
        let host = xnw_gpu::d3d12x::host::host_device()?;
        match screen_pass(host) {
            Ok(p) => *guard = Some(p),
            Err(e) => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| xnw_core::warn!("world render: no pass: {e}"));
                return None;
            }
        }
    }
    let pass = guard.as_mut()?;
    // A test run's "lists 4150 published, 193 replaced": the recorder finished two
    // frames between presents; the older was dropped and the newer drawn (one
    // slot, one recorder thread: never an older list after a newer one). A
    // present without a new list shows the last picture again. Guard the order.
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let list = list.filter(|l| {
        let last = LAST.load(std::sync::atomic::Ordering::Relaxed);
        if l.frame <= last && last != 0 {
            xnw_core::warn!("world render: list of frame {} after frame {last}: not drawn", l.frame);
            return false;
        }
        LAST.store(l.frame, std::sync::atomic::Ordering::Relaxed);
        true
    });
    if let Some(list) = list
        && let Err(e) = screen_draw(pass, list, false)
    {
        xnw_core::warn!("world render: pass failed: {e}; off");
        *guard = None;
        return None;
    }
    Some(pass.shown().clone())
}

/// The screen pass as the live path makes it: 1920x1080 on `host`, every
/// option from `XNW_WORLD_RENDER` (`xnw burst` makes its pass here too).
pub fn screen_pass(host: &xnw_gpu::host::Host) -> std::result::Result<Pass, String> {
    // Parsed again, not `screen_mode()`: `xnw burst` sets the variable after main read it.
    let opts = parse_screen(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default()).ok_or("XNW_WORLD_RENDER does not ask for the screen")?;
    let mut p = Pass::new(host, 1920, 1080)?;
    (p.wire, p.only_stage, p.skin, p.states, p.sky, p.water, p.peds, p.decal_hold) = (opts.wire, opts.stage, opts.skin, opts.states, opts.sky, opts.water, opts.peds, opts.decal_hold);
    // Skinned non-ped draws only with `skinned=on`.
    p.skinned = std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| v.split(',').any(|s| s.trim() == "skinned=on"));
    // `skinrules=<hex>` (default 0: the default maths and skips).
    p.skin_rules = parse_skin_rules(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default());
    p.pso_budget = SCREEN_PSO_BUDGET;
    p.pso_async = true;
    p.upload_ms_budget = Some(SCREEN_UPLOAD_MS);
    p.geom.budget = parse_geom_budget(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default());
    p.tex.opts = parse_screen_tex(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default());
    // `terrain=on` (off by default), `terraintri=`; its textures follow tex= (live or off).
    p.terrain.opts = terrain::parse_screen(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default());
    // `imposters=on` (off by default): cards for the title's imposter batches.
    p.imposters.opts = imposter::parse_screen(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default());
    // `bridgeps=on|<hashes>` (off by default): those rows again with the title's own
    // programs (the list sink's tap reads their bindings); on the hardware
    // adapter only with XNW_BRIDGE_HW=1 (a scheduled run).
    let bridgeps = bridge::parse_switch(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default());
    if !bridgeps.is_empty() {
        let hw = std::env::var("XNW_BRIDGE_HW").ok().as_deref() == Some("1");
        p.bridge_on = host.caps.software || hw;
        xnw_core::info!(
            "world render: bridgeps: {} pixel programs ({bridgeps:08x?}) {} on {}",
            bridgeps.len(),
            if p.bridge_on { "drawn by the title's own programs over ours" } else { "not drawn: the hardware adapter needs XNW_BRIDGE_HW=1" },
            host.caps.adapter
        );
    }
    // `bridgems=on|<hashes>` (off by default): the replay draws those rows from the title's mesh programs (`bridge_ms`); live waits for a binding source too.
    let bridgems = bridge_ms::parse_switch(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default());
    if !bridgems.is_empty() {
        xnw_core::info!("world render: bridgems asks for {} mesh programs; the live pass has no source of their bindings yet: drawn by our own programs", bridgems.len());
    }
    p.terrain.table.opts.mode = p.tex.opts.mode;
    p.terrain.table.opts.async_reads = p.tex.opts.async_reads;
    // Every per-frame budget of the pass on one line (geometry, textures, pipelines).
    xnw_core::info!(
        "world render: our pass on {} (1920x1080, {opts:?}); budgets per frame: geometry {} MiB / {} uploads (device {} MiB), textures {} MiB / {} uploads, pipelines {} new; total uploads {} MiB",
        host.caps.adapter,
        p.geom.budget.frame_bytes >> 20,
        p.geom.budget.frame_count,
        p.geom.budget.device_bytes >> 20,
        p.tex.opts.upload_bytes >> 20,
        p.tex.opts.uploads,
        p.pso_budget,
        (p.geom.budget.frame_bytes + p.tex.opts.upload_bytes) >> 20
    );
    xnw_core::info!("world render: textures {:?}, geometry {:?}", p.tex.opts, p.geom.budget);
    xnw_core::info!("world render: skin: rules {:#x}, skinned {}, peds {}", p.skin_rules, if p.skinned { "on" } else { "off" }, if p.peds { "on" } else { "off" });
    // The ped pre-skin (`preskin=on`): its program is made here, not in a frame.
    if preskin_gpu::screen_switch(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default()) {
        match p.preskin_prepare() {
            Ok(ms) => xnw_core::info!("world render: pre-skin on (our compute pass; program made in {ms:.0} ms)"),
            Err(e) => xnw_core::warn!("world render: pre-skin program failed: {e}"),
        }
    }
    // `light=old` keeps the earlier lighting.
    p.light_old = std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| v.split(',').any(|s| s.trim() == "light=old"));
    // `layers=c0|c1|off` (default c1).
    p.layers = parse_layers(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default());
    // Hold partial pictures at cuts (`hold=off`: show every frame as drawn).
    p.cuts = cuts::CutWatch::new(!std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| v.split(',').any(|s| s.trim() == "hold=off")));
    // `cutfast=on` (off by default, CUT1): a camera cut's queue drained in 2 frames, its texture reads capped.
    p.cuts.fast = std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| v.split(',').any(|s| s.trim() == "cutfast=on"));
    xnw_core::info!(
        "world render: cuts: hold {} (a picture under {:.0}% of the main view's triangles is not shown, at most {HOLD} ms), complete at {:.0}%, cut at {} m / {} deg / coverage under {:.0}%; upload time {} ms a frame, x{} while a cut recovers; pipelines off the frame (async), {} a frame; fast {} (queue drained in {} frames, at most {} ms a frame; texture reads {} ms); texture reads cap {} ms",
        p.cuts.hold, 100.0 * cuts::HOLD_COVERAGE, 100.0 * cuts::COMPLETE_COVERAGE, cuts::CUT_METRES, cuts::CUT_DEGREES, 100.0 * cuts::CUT_COVERAGE, SCREEN_UPLOAD_MS, p.cuts.boost, p.pso_budget, p.cuts.fast, cuts::DRAIN_FRAMES, cuts::DRAIN_MAX_MS, cuts::CUT_TEX_MS, p.tex.opts.frame_ms, HOLD = cuts::HOLD_MS
    );
    // `stipple=off` draws cross-fading LODs solid (bisecting).
    p.stipple = !std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| v.split(',').any(|s| s.trim() == "stipple=off"));
    Ok(p)
}

/// One finished list into the screen pass, as the live path draws it: its new
/// geometries offered to the queue, drawn, and the live log lines (cut, slow
/// frame, every 60 frames). `xnw burst` draws its lists here too (`readback`:
/// the picture copied for [`Pass::read_rgba`]).
pub fn screen_draw(pass: &mut Pass, mut list: FrameList, readback: bool) -> std::result::Result<PassStats, String> {
    let t = std::time::Instant::now();
    // The new geometries move into the queue (no copy of their bytes).
    for (key, g) in std::mem::take(&mut list.ahead_geoms) {
        pass.geom.offer_ahead(geometry::known(), key, g, list.frame);
    }
    for (key, g) in std::mem::take(&mut list.new_geoms) {
        pass.geom.offer(geometry::known(), key, g, list.frame);
    }
    // The lighting in use, once at the first lit frame and every 3,600 frames.
    if let Some(l) = list.light {
        static FIRST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
        if FIRST.swap(false, std::sync::atomic::Ordering::Relaxed) || list.frame % 3600 == 0 {
            xnw_core::info!("{}{}", l.line(), if pass.light_old { " (light=old: not used)" } else { "" });
        }
    }
    let st = pass.draw(&list, &mut |_| None, readback)?;
    // One line per camera cut, when its picture is complete.
    if let Some(l) = pass.cuts.line.take() {
        xnw_core::info!("{l}");
    }
    if pass.tex.opts.mode != TexMode::Off
        && let Some(lines) = pass.tex.log_line()
    {
        for l in lines {
            xnw_core::info!("{l}");
        }
    }
    pass.log_geometry(list.frame, &st);
    let total = t.elapsed().as_secs_f64() * 1e3;
    cost_pass(total, st.upload_ms, st.pso_ms);
    if total > 50.0 {
        let steps = [("upload", st.upload_ms), ("pipelines", st.pso_ms), ("textures", st.tex_ms), ("record", st.record_ms - st.upload_ms - st.pso_ms - st.tex_ms), ("GPU", st.gpu_ms)];
        let worst = steps.iter().copied().fold(("-", 0.0), |a, b| if b.1 > a.1 { b } else { a });
        xnw_core::warn!(
            "world render: slow frame {}: {total:.1} ms in our pass, mostly {} ({:.1} ms): upload {:.1} ms ({} geometries, {} KiB, {} pending), pipelines {} new in {:.1} ms, textures {:.1} ms, record {:.1} ms, GPU {:.1} ms, {} draws{}",
            list.frame,
            worst.0,
            worst.1,
            st.upload_ms,
            st.uploaded_geometries,
            st.uploaded_bytes >> 10,
            st.pending,
            st.new_psos,
            st.pso_ms,
            st.tex_ms,
            st.record_ms,
            st.gpu_ms,
            st.draws,
            if st.held { " (held)" } else { "" }
        );
    }
    if list.skin_hooked > 0 && st.skinned == 0 && pass.skin {
        static WARNED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if WARNED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 60 == 0 {
            xnw_core::warn!("world render: frame {}: skinned 0 of {} hooked: no skin palette reached the pass", list.frame, list.skin_hooked);
        }
    }
    if list.frame % 60 == 0 && (!list.preskin.items.is_empty() || list.preskin.unreadable > 0) {
        let s = pass.preskin_stats();
        xnw_core::info!("{}", preskin_gpu::line(list.frame, &list.preskin, &s));
    }
    if list.frame % 60 == 0 && !list.cull.rows.is_empty() {
        let s = pass.cull_stats();
        xnw_core::info!("{}", gpucull::line(list.frame, &s));
    }
    if list.frame % 60 == 0 && pass.bridge_on {
        let none = bridged::Frame::default();
        xnw_core::info!("{}", bridged::line(list.frame, list.bridged.as_deref().unwrap_or(&none), &pass.bridge_stats));
    }
    if list.frame % 60 == 0 && pass.imposters.opts.on {
        xnw_core::info!("{}", pass.imposters.frame_line(list.frame, list.imposters.as_deref()));
    }
    if list.frame % 60 == 0 && pass.terrain.opts.on {
        for l in pass.terrain.log_lines() {
            xnw_core::info!("{l}");
        }
    }
    if list.frame % 60 == 0 {
        xnw_core::info!(
            "world render: frame {}: {} draws (skinned {} of {} hooked, {} with the ped remap, {} peds skipped, {} skinned skipped, {} decals held, {} without geometry, {} filtered, {} with the title's state, {} state pipelines), {} triangles; {} geometries uploaded ({} replaced, {} indices clamped) ({} KiB), palettes {} B; record {:.1} ms, GPU {:.1} ms, total {:.1} ms",
            list.frame,
            st.draws,
            st.skinned,
            list.skin_hooked,
            st.remapped,
            st.peds_skipped,
            st.skinned_skipped,
            st.decals_held,
            st.skipped,
            st.filtered,
            st.stated,
            pass.state_psos.len(),
            st.triangles,
            st.uploaded_geometries,
            st.replaced,
            st.clamped,
            st.uploaded_bytes >> 10,
            st.palette_bytes,
            st.record_ms,
            st.gpu_ms,
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    Ok(st)
}


#[cfg(test)]
mod tests {
    use super::*;

    /// The body of each pixel entry (`: SV_Target {` to its closing brace).
    fn pixel_entries(src: &str) -> Vec<&str> {
        let mut out = Vec::new();
        let mut rest = src;
        while let Some(at) = rest.find(": SV_Target {") {
            let body = &rest[at + ": SV_Target {".len()..];
            let mut depth = 1;
            let end = body.char_indices().find(|&(_, c)| {
                depth += match c {
                    '{' => 1,
                    '}' => -1,
                    _ => 0,
                };
                depth == 0
            });
            let end = end.map_or(body.len(), |(i, _)| i);
            out.push(&body[..end]);
            rest = &body[end..];
        }
        out
    }

    /// Every pixel entry of the pass returns once, at its end: the newer
    /// WARP removes the device on a pixel shader that returns inside a branch.
    #[test]
    fn pixel_entries_return_once() {
        let shaders = [("SHADER", SHADER), ("RAW_SHADER", RAW_SHADER), ("WATER_SHADER", WATER_SHADER), ("SKY_SHADER", SKY_SHADER), ("IMPOSTER_SHADER", imposter::IMPOSTER_SHADER), ("TERRAIN_SHADER", terrain::TERRAIN_SHADER)];
        for (name, src) in shaders {
            let entries = pixel_entries(src);
            assert_eq!(entries.len(), 1, "{name}: one pixel entry");
            for body in entries {
                assert_eq!(body.matches("return").count(), 1, "{name}: {body}");
                assert!(body.trim_end().trim_end_matches(';').trim_end().rfind("return").is_some_and(|r| !body[r..].contains('}')), "{name}: the return ends the entry");
            }
        }
        // The checker sees a return inside a branch.
        let early = "float4 ps(VsOut i) : SV_Target { if (a) return 1; return 0; }";
        assert_eq!(pixel_entries(early)[0].matches("return").count(), 2);
    }

    /// HANG1: no load is the arm of a `?:` (HLSL runs both arms, so the
    /// load runs whatever the condition says; through a root view it reads
    /// past its buffer on the card). Every guarded load picks its address.
    #[test]
    fn no_load_in_a_conditional_arm() {
        let shaders = [("SHADER", SHADER), ("RAW_SHADER", RAW_SHADER), ("WATER_SHADER", WATER_SHADER), ("SKY_SHADER", SKY_SHADER), ("IMPOSTER_SHADER", imposter::IMPOSTER_SHADER), ("TERRAIN_SHADER", terrain::TERRAIN_SHADER), ("gpucull", gpucull::SHADER), ("preskin_gpu", preskin_gpu::SHADER)];
        // Both arms of each `?`: the text after it up to the end of its
        // expression (a `;`, a `,` or `)` closing what encloses it).
        let arm = |src: &str| {
            src.lines().filter(|l| !l.trim_start().starts_with("//")).any(|l| {
                l.match_indices('?').any(|(at, _)| {
                    let mut depth = 0i32;
                    let end = l[at + 1..].char_indices().find(|&(_, c)| {
                        depth += i32::from(c == '(') - i32::from(c == ')');
                        c == ';' || depth < 0 || (c == ',' && depth == 0)
                    });
                    l[at + 1..at + 1 + end.map_or(l.len() - at - 1, |e| e.0)].contains(".Load")
                })
            })
        };
        for (name, src) in shaders {
            assert!(!arm(src), "{name}: a load in a conditional arm");
        }
        assert!(arm("uint ld(uint a) { return a < n ? vb.Load(a) : 0; }"), "the checker sees one");
        assert!(arm("float4 r(bool f, uint a) { return asfloat(f ? 0 : kept.Load4(a)); }"), "and the second arm");
        assert!(!arm("uint ld(uint a) { bool ok = a < n; uint v = vb.Load(ok ? a : 0); return ok ? v : 0; }"));
        assert!(!arm("float4 f = asfloat(frame.Load4(c ? a : 0)), k = asfloat(kept.Load4(c ? 0 : a));"));
    }

    /// Frame 1700's main viewport (camera.txt).
    const VIEW: Mat = [0.75284, -0.01068, -0.65811, 0.0, -0.65820, -0.01222, -0.75274, 0.0, 0.0, 0.99987, -0.01623, 0.0, 54.06376, -255.99721, -304.74072, 1.0];
    const PROJ: Mat = [1.875, 0.0, 0.0, 0.0, 0.0, 3.33333, 0.0, 0.0, 0.0, 0.0, 0.00002, -1.0, 0.0, 0.0, 0.15, 0.0];

    fn translation(t: [f32; 3]) -> Mat {
        let mut m = [0f32; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[10] = 1.0;
        m[15] = 1.0;
        m[12..15].copy_from_slice(&t);
        m
    }

    #[test]
    fn camera_of_frame_1700() {
        let c = camera_position(&VIEW);
        assert!((c[0] + 243.99).abs() < 0.05 && (c[1] + 196.93).abs() < 0.05 && (c[2] - 251.02).abs() < 0.05, "{c:?}");
        // The camera maps to the view origin.
        let o = mul(&translation(c), &VIEW);
        assert!(o[12].abs() < 1e-2 && o[13].abs() < 1e-2 && o[14].abs() < 1e-2, "{o:?}");
        let (fovy, aspect, near) = projection_params(&PROJ);
        assert!((fovy - 33.4).abs() < 0.1 && (aspect - 16.0 / 9.0).abs() < 0.01 && (near - 0.15).abs() < 1e-3, "{fovy} {aspect} {near}");
    }

    /// Relative world * rotation * P equals absolute world * view * P.
    #[test]
    fn relative_equals_absolute() {
        let cam = Camera { viewport: 1, view: VIEW, proj: PROJ };
        let rel = translation([2.85, 2.28, -1.02]);
        let c = camera_position(&VIEW);
        let abs = translation([2.85 + c[0], 2.28 + c[1], -1.02 + c[2]]);
        let a = mul(&rel, &cam.view_proj_relative());
        let b = mul(&mul(&abs, &VIEW), &PROJ);
        for k in 0..16 {
            assert!((a[k] - b[k]).abs() < 2e-2 * (1.0 + b[k].abs()), "{k}: {} vs {}", a[k], b[k]);
        }
        // The object is in front: w = its view distance (3.6 m).
        assert!((a[15] - 3.575).abs() < 0.01, "{a:?}");
    }

    #[test]
    fn lists_keep_the_newest() {
        let l = Lists::default();
        assert!(l.take().is_none());
        l.publish(FrameList { frame: 1, ..Default::default() });
        l.publish(FrameList { frame: 2, ..Default::default() });
        assert_eq!(l.take().map(|f| f.frame), Some(2));
        assert!(l.take().is_none());
        assert_eq!(l.counts(), (2, 1));
    }

    #[test]
    fn main_camera_is_the_wide_one() {
        let square = Camera { viewport: 7, view: VIEW, proj: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -0.01, 0.0] };
        let wide = Camera { viewport: 9, view: VIEW, proj: PROJ };
        let rec = |vp| DrawRecord { geom: GeomKey::default(), world: translation([0.0; 3]), viewport: vp, effect: 0, albedo: Albedo::Flat([1.0; 3]), palette: NO_PALETTE, stage: 0, inst_first: 0, inst_count: 0, bone_first: 0, bone_count: 0, material: Material::default(), state: None, remap: NO_REMAP, ped: false };
        let f = FrameList { frame: 3, draws: vec![rec(7), rec(7), rec(9)], cameras: vec![square, wide], main: None, ..Default::default() };
        assert_eq!(f.main_camera().map(|c| c.viewport), Some(9));
        assert_eq!(f.main_draws().count(), 1);
    }

    #[test]
    fn cache_makes_once_and_evicts() {
        let mut c: GeometryCache<u32> = GeometryCache::default();
        let k = |a| GeomKey { mesh_at: a, mesh_len: 16, vb_at: a + 0x1000, vb_len: 256, ib: 0, hash: 0 };
        assert_eq!(c.get_or_make(k(1), 10, || Some(5)), Some(&5));
        assert_eq!(c.get_or_make(k(1), 11, || panic!("made twice")), Some(&5));
        assert_eq!(c.get_or_make(k(2), 11, || None), None);
        assert_eq!(c.get_or_make(k(3), 12, || Some(7)), Some(&7));
        c.evict(72, 60);
        assert_eq!((c.len(), c.made, c.evicted), (1, 2, 1));
        assert!(c.contains(&k(3)) && !c.contains(&k(1)));
    }

    #[test]
    fn constants_and_vertices() {
        let cam = Camera { viewport: 1, view: VIEW, proj: PROJ };
        let rec = DrawRecord { geom: GeomKey::default(), world: translation([1.0, 2.0, 3.0]), viewport: 1, effect: 0x1234, albedo: Albedo::Flat(flat_colour(0x1234)), palette: NO_PALETTE, stage: 0, inst_first: 0, inst_count: 0, bone_first: 0, bone_count: 0, material: Material::default(), state: None, remap: NO_REMAP, ped: false };
        let c = draw_constants(&rec, &cam);
        assert_eq!(&c[16..19], &[1.0, 0.0, 0.0]);
        assert_eq!(c[31], 0.0);
        assert!(c[28..31].iter().all(|v| (0.35..=0.9).contains(v)));
        assert_eq!(flat_colour(0x1234), flat_colour(0x1234));
        let v = pack_vertices(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &[], &[[0.5, 0.25]]);
        assert_eq!(v.len() * 4, 2 * VERTEX_BYTES);
        assert_eq!(&v[..8], &[1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 0.5, 0.25]);
        assert_eq!(&v[8..], &[4.0, 5.0, 6.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    /// The pass on WARP (software; `XNW_TEST_GPU=1`): one triangle 3 m
    /// in front of an identity view lands in the middle of the target with
    /// its flat colour lit, the corner keeps the clear colour.
    #[test]
    fn pass_draws_on_warp() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let (_gpu, host) = warp_host();
        assert!(host.caps.software, "{}", host.caps.adapter);
        let mut pass = Pass::new(&host, 64, 36).unwrap();
        let id: Mat = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let cam = Camera { viewport: 1, view: id, proj: PROJ };
        let geom = GeomKey { mesh_at: 1, ..Default::default() };
        let rec = DrawRecord { geom, world: id, viewport: 1, effect: 0, albedo: Albedo::Flat([1.0, 0.0, 0.0]), palette: NO_PALETTE, stage: 0, inst_first: 0, inst_count: 0, bone_first: 0, bone_count: 0, material: Material::default(), state: None, remap: NO_REMAP, ped: false };
        let list = FrameList { frame: 1, draws: vec![rec], cameras: vec![cam], main: Some(1), ..Default::default() };
        let tri = HostGeometry { vertices: pack_vertices(&[[-1.0, -1.0, -3.0], [1.0, -1.0, -3.0], [0.0, 1.0, -3.0]], &[[0.0, 0.0, 1.0]; 3], &[]), indices: vec![0, 1, 2], raw_vb: None, vb_place: 0 };
        let mut once = Some(tri);
        let st = pass.draw(&list, &mut |_| once.take(), true).unwrap();
        assert_eq!((st.draws, st.triangles, st.uploaded_geometries), (1, 1, 1));
        let px = pass.read_rgba().unwrap();
        let at = |x: usize, y: usize| &px[(y * 64 + x) * 4..][..4];
        assert!(at(32, 18)[0] > 100 && at(32, 18)[1] < 10, "centre {:?}", at(32, 18));
        assert_eq!(at(0, 0)[..3], [23, 23, 26], "corner {:?}", at(0, 0));
    }
    /// The raw-buffer path on WARP: a title-style vertex buffer (stream
    /// table, float positions in slot 0, weights/indices in slots 6/8)
    /// drawn with and without a palette that moves it.
    #[test]
    fn raw_path_draws_and_skins_on_warp() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let (_gpu, host) = warp_host();
        let mut pass = Pass::new(&host, 64, 36).unwrap();
        let word = |stride: u32, offset: u32, format: u32| (stride / 4) << 5 | (offset / 4) << 9 | format;
        let mut vb = vec![0u8; 0x80];
        vb[0..4].copy_from_slice(&word(12, 0x80, 3).to_le_bytes());
        vb[0x18..0x1C].copy_from_slice(&word(4, 0x80 + 36, 6).to_le_bytes());
        vb[0x20..0x24].copy_from_slice(&word(4, 0x80 + 48, 5).to_le_bytes());
        for v in [[-1.0f32, -1.0, -3.0], [1.0, -1.0, -3.0], [0.0, 1.0, -3.0]] {
            for x in v {
                vb.extend(x.to_le_bytes());
            }
        }
        vb.extend([255u8, 0, 0, 0].repeat(3)); // weights: bone 0 only
        vb.extend([0u8, 0, 0, 0].repeat(3)); // indices
        let id: Mat = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let cam = Camera { viewport: 1, view: id, proj: PROJ };
        let geom = GeomKey { mesh_at: 7, ..Default::default() };
        let g = HostGeometry { vertices: Vec::new(), indices: vec![0, 1, 2, 2, 1, 900], raw_vb: Some(vb), vb_place: 0 };
        let rec = DrawRecord { geom, world: id, viewport: 1, effect: 0, albedo: Albedo::Flat([0.0, 1.0, 0.0]), palette: NO_PALETTE, stage: 0, inst_first: 0, inst_count: 0, bone_first: 0, bone_count: 0, material: Material::default(), state: None, remap: NO_REMAP, ped: false };
        let mut list = FrameList { frame: 1, draws: vec![rec], cameras: vec![cam], main: Some(1), new_geoms: vec![(geom, g)], ..Default::default() };
        let st = pass.draw(&list, &mut |_| None, true).unwrap();
        assert_eq!((st.draws, st.uploaded_geometries, st.skinned), (1, 1, 0));
        let px = pass.read_rgba().unwrap();
        let at = |px: &[u8], x: usize, y: usize| px[(y * 64 + x) * 4 + 1];
        assert!(at(&px, 32, 18) > 100, "unskinned centre {}", at(&px, 32, 18));
        // A palette that moves bone 0 by +10 in x: the centre empties.
        list.palettes = vec![vec![1.0, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]];
        list.draws[0].palette = 0;
        list.new_geoms.clear();
        list.frame = 2;
        let st = pass.draw(&list, &mut |_| None, true).unwrap();
        assert_eq!((st.draws, st.skinned, st.uploaded_geometries), (1, 1, 0));
        let px = pass.read_rgba().unwrap();
        assert!(at(&px, 32, 18) < 40, "skinned centre {}", at(&px, 32, 18));
        // The out-of-range index 900 read zeros (bounded), nothing faulted.
        // The ped remap: index 0 -> remap[0] & 0xFFF = bone 1 (identity):
        // the centre is back. The remap entries follow the bones in the palette floats.
        let mut pal = vec![1.0, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        pal.extend([f32::from_bits(0x7001), f32::from_bits(0)]);
        list.palettes = vec![pal];
        (list.draws[0].remap, list.draws[0].ped) = (24, true);
        list.frame = 3;
        let st = pass.draw(&list, &mut |_| None, true).unwrap();
        assert_eq!((st.skinned, st.remapped), (1, 1));
        let px = pass.read_rgba().unwrap();
        assert!(at(&px, 32, 18) > 100, "remapped to bone 1: centre {}", at(&px, 32, 18));
    }
    /// An IcInstance record (3x4 rows, translation in .w) as our
    /// row-vector 4x3, relative to an origin.
    #[test]
    fn instance_record_layout() {
        // A 90 degree yaw (model x -> world y, model y -> world -x), as the
        // title's program reads it: r0.xyz = image of x, r1.xyz = image of y.
        let rec: [f32; 16] = [0.0, 1.0, 0.0, 100.0, -1.0, 0.0, 0.0, 200.0, 0.0, 0.0, 1.0, 30.0, 0.5, 0.5, 0.0, 0.0];
        let w = instance_world(&rec, [90.0, 190.0, 30.0]);
        // p'' = x * (w0..2) + y * (w3..5) + z * (w6..8) + (w9..11): x axis to +y.
        let p = [1.0f32, 0.0, 0.0];
        let q: [f32; 3] = std::array::from_fn(|i| p[0] * w[i] + p[1] * w[3 + i] + p[2] * w[6 + i] + w[9 + i]);
        assert_eq!(q, [10.0, 11.0, 0.0]);
        let p = [0.0f32, 1.0, 0.0];
        let q: [f32; 3] = std::array::from_fn(|i| p[0] * w[i] + p[1] * w[3 + i] + p[2] * w[6 + i] + w[9 + i]);
        assert_eq!(q, [9.0, 10.0, 0.0], "model y to world -x");
    }

    /// One raw geometry drawn with two instances on WARP: instance 0 is
    /// moved off screen, instance 1 sits in front of the camera; with the
    /// count clamped to 1 only instance 0 draws (empty centre).
    #[test]
    fn instanced_raw_path_on_warp() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let (_gpu, host) = warp_host();
        let mut pass = Pass::new(&host, 64, 36).unwrap();
        let word = |stride: u32, offset: u32, format: u32| (stride / 4) << 5 | (offset / 4) << 9 | format;
        let mut vb = vec![0u8; 0x80];
        vb[0..4].copy_from_slice(&word(12, 0x80, 3).to_le_bytes());
        for v in [[-1.0f32, -1.0, -3.0], [1.0, -1.0, -3.0], [0.0, 1.0, -3.0]] {
            for x in v {
                vb.extend(x.to_le_bytes());
            }
        }
        let id: Mat = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let geom = GeomKey { mesh_at: 9, ..Default::default() };
        let mut instances = Vec::new();
        for tx in [50.0f32, 0.0] {
            let rec: [f32; 16] = [1.0, 0.0, 0.0, tx, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            instances.extend(instance_world(&rec, [0.0; 3]));
        }
        let rec = DrawRecord { geom, world: id, viewport: 1, effect: 0, albedo: Albedo::Flat([0.0, 0.0, 1.0]), palette: NO_PALETTE, stage: 0, inst_first: 0, inst_count: 2, bone_first: 0, bone_count: 0, material: Material::default(), state: None, remap: NO_REMAP, ped: false };
        let g = HostGeometry { vertices: Vec::new(), indices: vec![0, 1, 2], raw_vb: Some(vb), vb_place: 0 };
        let mut list = FrameList { frame: 1, draws: vec![rec], cameras: vec![Camera { viewport: 1, view: id, proj: PROJ }], main: Some(1), new_geoms: vec![(geom, g)], instances, ..Default::default() };
        let st = pass.draw(&list, &mut |_| None, true).unwrap();
        assert_eq!((st.instanced, st.instances), (1, 2));
        let blue = |px: &[u8]| px[(18 * 64 + 32) * 4 + 2];
        assert!(blue(&pass.read_rgba().unwrap()) > 100, "instance 1 in the centre");
        list.draws[0].inst_count = 1;
        list.new_geoms.clear();
        list.frame = 2;
        pass.draw(&list, &mut |_| None, true).unwrap();
        assert!(blue(&pass.read_rgba().unwrap()) < 60, "only instance 0 (off screen)");
        // A first instance past the buffer is clamped to none: drawn once with instance 0 of nothing (zeros: degenerate).
        list.draws[0].inst_first = 1_000;
        list.frame = 3;
        let st = pass.draw(&list, &mut |_| None, true).unwrap();
        assert_eq!(st.instanced, 0);
    }
    /// One raw triangle in front of the camera (64x36 target), drawn with
    /// a flat green colour and albedo slot `slot`.
    /// Lane packing: -1 and all lanes solid; set 0 for 25 % / 75 %;
    /// the 50 % pair keeps its lanes (complements).
    #[test]
    fn stipple_lane_packing() {
        assert_eq!(stipple_lanes(u64::MAX), 0);
        assert_eq!(stipple_lanes(0xFFFF_0000_0000_0000), 0x11, "any single lane: set 0's lane 0");
        assert_eq!(stipple_lanes(0x0000_FFFF_FFFF_FFFF), 0x1E, "three lanes: 1, 2, 3");
        assert_eq!(stipple_lanes(0x0000_FFFF_FFFF_0000), 0x16);
        assert_eq!(stipple_lanes(0xFFFF_0000_0000_FFFF), 0x19);
        assert_eq!(stipple_lanes(0x0000_FFFF_FFFF_0000) & 0xF ^ stipple_lanes(0xFFFF_0000_0000_FFFF) & 0xF, 0xF, "the pair covers each quad pixel once");
    }

    /// On WARP: with lane 0 kept, pixel (32, 18) (lane 0) is drawn
    /// and (33, 18) (lane 1) keeps the clear colour; `stipple` off: both drawn.
    #[test]
    fn stipple_discard_on_warp() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let (_gpu, host) = warp_host();
        let mut pass = Pass::new(&host, 64, 36).unwrap();
        let mut list = textured_list(0, 31, false);
        list.draws[0].material.stipple = 0x11;
        pass.draw(&list, &mut |_| None, true).unwrap();
        let px = pass.read_rgba().unwrap();
        let g = |x: usize, y: usize| px[(y * 64 + x) * 4 + 1];
        assert!(g(32, 18) > 150 && g(33, 18) < 60 && g(32, 19) < 60, "{} {} {}", g(32, 18), g(33, 18), g(32, 19));
        pass.stipple = false;
        list.new_geoms.clear();
        list.frame = 2;
        pass.draw(&list, &mut |_| None, true).unwrap();
        let px = pass.read_rgba().unwrap();
        assert!(px[(18 * 64 + 33) * 4 + 1] > 150, "stipple off: solid");
    }

    fn textured_list(slot: u64, mesh: u64, clip: bool) -> FrameList {
        let word = |stride: u32, offset: u32, format: u32| (stride / 4) << 5 | (offset / 4) << 9 | format;
        let mut vb = vec![0u8; 0x80];
        vb[0..4].copy_from_slice(&word(12, 0x80, 3).to_le_bytes());
        for v in [[-1.0f32, -1.0, -3.0], [1.0, -1.0, -3.0], [0.0, 1.0, -3.0]] {
            for x in v {
                vb.extend(x.to_le_bytes());
            }
        }
        let id: Mat = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let geom = GeomKey { mesh_at: mesh, ..Default::default() };
        let material = Material { albedo_desc: slot, alpha_clip: clip, ..Default::default() };
        let rec = DrawRecord { geom, world: id, viewport: 1, effect: 0, albedo: Albedo::Flat([0.0, 1.0, 0.0]), palette: NO_PALETTE, stage: 0, inst_first: 0, inst_count: 0, bone_first: 0, bone_count: 0, material, state: None, remap: NO_REMAP, ped: false };
        let g = HostGeometry { vertices: Vec::new(), indices: vec![0, 1, 2], raw_vb: Some(vb), vb_place: 0 };
        FrameList { frame: 1, draws: vec![rec], cameras: vec![Camera { viewport: 1, view: id, proj: PROJ }], main: Some(1), new_geoms: vec![(geom, g)], ..Default::default() }
    }

    fn centre(px: &[u8]) -> [u8; 4] {
        let i = (18 * 64 + 32) * 4;
        [px[i], px[i + 1], px[i + 2], px[i + 3]]
    }

    /// Checker mode (offline dumps): a slot gets its own texture from
    /// our table (not the flat green); slot 0 stays flat.
    #[test]
    fn checker_albedo_on_warp() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let (_gpu, host) = warp_host();
        let mut pass = Pass::new(&host, 64, 36).unwrap();
        pass.tex.opts.mode = TexMode::Checker;
        let mut list = textured_list(0x4444_0000, 21, false);
        pass.draw(&list, &mut |_| None, true).unwrap();
        let c = centre(&pass.read_rgba().unwrap());
        assert!(c[1] < 200 && c[0] > 40, "a checker colour, not the flat green: {c:?}");
        assert_eq!((pass.tex.stats.bound_draws, pass.tex.stats.flat_draws, pass.tex.stats.distinct), (1, 0, 1));
        list.draws[0].material.albedo_desc = 0;
        list.new_geoms.clear();
        list.frame = 2;
        pass.draw(&list, &mut |_| None, true).unwrap();
        let c = centre(&pass.read_rgba().unwrap());
        assert!(c[1] > 150 && c[0] < 30, "flat green: {c:?}");
        assert_eq!(pass.tex.stats.flat_draws, 1);
    }

    /// The live path on WARP: a title texture placed in memory
    /// (16x16 RGBA8, linear, all red), its view written to a descriptor slot
    /// as the title does; our pass finds the host copy through the slot,
    /// uploads it with `texture::for_pass` on its own list, samples it. The
    /// next frame reuses it without another upload; a slot without a record
    /// draws flat.
    #[test]
    fn live_albedo_upload_on_warp() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let _gpu = warp_host();
        let Some(host) = xnw_gpu::d3d12x::host::host_device() else { return };
        let mem = xnw_gpu::d3d12x::resource::Memory::zeroed(64 << 10, 64 << 10).unwrap();
        let fill = |rgba: [u8; 4]| {
            // SAFETY: our own 64 KiB.
            let m = unsafe { std::slice::from_raw_parts_mut(mem.address() as *mut u8, 64 << 10) };
            for p in m.chunks_exact_mut(4) {
                p.copy_from_slice(&rgba);
            }
        };
        fill([255, 0, 0, 255]);
        let resource: [u32; 14] = [3, 0, 0, 0, 16, 0, 16, 0x0001_0001, DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32, 1, 0, 0x100, 0, 0];
        let mut desc = [0u32; 16];
        desc[..6].copy_from_slice(&[DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32, 4, 0x1688, 0, 0, 1]);
        (desc[10], desc[11]) = (mem.address() as u32, (mem.address() >> 32) as u32);
        let slots = [0u64; 8];
        let slot = slots.as_ptr() as u64;
        xnw_gpu::d3d12x::descriptor::create_placed_srv(std::ptr::null(), &resource, &desc, slot);
        let mut pass = Pass::new(host, 64, 36).unwrap();
        pass.tex.opts.mode = TexMode::Live;
        let mut list = textured_list(slot, 22, false);
        pass.draw(&list, &mut |_| None, true).unwrap();
        let c = centre(&pass.read_rgba().unwrap());
        let s = pass.tex.stats;
        assert_eq!((s.placed, s.uploads, s.bound_draws), (1, 1, 1), "{s:?}");
        assert!(c[0] > 150 && c[1] < 30 && c[2] < 30, "the texture's red, not the flat green: {c:?}");
        // Frame 2: known, complete: sampled without another upload.
        list.new_geoms.clear();
        list.frame = 2;
        pass.draw(&list, &mut |_| None, true).unwrap();
        assert!(centre(&pass.read_rgba().unwrap())[0] > 150);
        assert_eq!((pass.tex.stats.uploads, pass.tex.stats.bound_draws), (1, 2));
        // A slot without a record: flat.
        let empty = [0u64; 4];
        list.draws[0].material.albedo_desc = empty.as_ptr() as u64;
        list.frame = 3;
        pass.draw(&list, &mut |_| None, true).unwrap();
        assert!(centre(&pass.read_rgba().unwrap())[1] > 150, "flat green");
        assert_eq!(pass.tex.stats.no_record, 1);
        drop(mem);
    }

    /// A test run had a cold pack read of 154 ms inside our record step. With
    /// `async_reads` the frame never reads: the first frame draws flat and
    /// queues the read; a later frame takes the finished upload and samples it.
    #[test]
    fn async_texture_read_on_warp() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let _gpu = warp_host();
        let Some(host) = xnw_gpu::d3d12x::host::host_device() else { return };
        let mem = Box::leak(Box::new(xnw_gpu::d3d12x::resource::Memory::zeroed(64 << 10, 64 << 10).unwrap()));
        // SAFETY: our own 64 KiB.
        let m = unsafe { std::slice::from_raw_parts_mut(mem.address() as *mut u8, 64 << 10) };
        for p in m.chunks_exact_mut(4) {
            p.copy_from_slice(&[255, 0, 0, 255]);
        }
        let resource: [u32; 14] = [3, 0, 0, 0, 16, 0, 16, 0x0001_0001, DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32, 1, 0, 0x100, 0, 0];
        let mut desc = [0u32; 16];
        desc[..6].copy_from_slice(&[DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32, 4, 0x1688, 0, 0, 1]);
        (desc[10], desc[11]) = (mem.address() as u32, (mem.address() >> 32) as u32);
        let slot = Box::leak(Box::new([0u64; 8])).as_ptr() as u64;
        xnw_gpu::d3d12x::descriptor::create_placed_srv(std::ptr::null(), &resource, &desc, slot);
        let mut pass = Pass::new(host, 64, 36).unwrap();
        pass.tex.opts.mode = TexMode::Live;
        pass.tex.opts.async_reads = true;
        let mut list = textured_list(slot, 23, false);
        pass.draw(&list, &mut |_| None, true).unwrap();
        let s = pass.tex.stats;
        assert_eq!((s.uploads, s.bound_draws, s.flat_draws), (0, 0, 1), "frame 1: queued, drawn flat: {s:?}");
        assert!(centre(&pass.read_rgba().unwrap())[1] > 150, "flat green");
        list.new_geoms.clear();
        let mut red = false;
        for f in 2..60 {
            std::thread::sleep(std::time::Duration::from_millis(5));
            list.frame = f;
            pass.draw(&list, &mut |_| None, true).unwrap();
            if centre(&pass.read_rgba().unwrap())[0] > 150 {
                red = true;
                break;
            }
        }
        assert!(red, "the read came back and was sampled");
        assert_eq!(pass.tex.stats.uploads, 1);
        let (_, _, reads, _) = xnw_gpu::d3d12x::texture::pass_reader_counts();
        assert!(reads >= 1);
    }

    #[test]
    fn screen_texture_switches() {
        assert_eq!(parse_screen_tex("screen").mode, TexMode::Live);
        assert!(!parse_screen_tex("screen").async_reads && parse_screen_tex("screen,texasync=on").async_reads);
        assert_eq!((parse_screen_tex("screen").frame_ms, parse_screen_tex("screen,texms=8").frame_ms), (0.0, 8.0));
        assert_eq!(parse_screen_tex("screen,tex=off").mode, TexMode::Off);
        let o = parse_screen_tex("screen,wire,tex=64,texn=8,texmb=16");
        assert_eq!((o.mode, o.slots, o.uploads, o.upload_bytes), (TexMode::Live, 64, 8, 16 << 20));
        assert_eq!(parse_screen_tex("screen,tex=9999").slots, TEX_SLOTS);
        assert_eq!(parse_screen("screen,tex=off").map(|o| o.stage), Some(None), "not a pass filter");
    }


    #[test]
    fn projection_jitter_dropped() {
        let mut p = PROJ;
        (p[8], p[9]) = (0.00107, 0.00096);
        let a = Camera { viewport: 1, view: VIEW, proj: p }.view_proj_relative();
        let b = Camera { viewport: 1, view: VIEW, proj: PROJ }.view_proj_relative();
        assert_eq!(a, b, "the jitter of a captured frame changes nothing we draw");
        // Kept (the audit): a point straight ahead moves by -p8, -p9 in NDC.
        let k = Camera { viewport: 1, view: VIEW, proj: p }.view_proj(true);
        let d = Camera { viewport: 1, view: VIEW, proj: p }.view_proj(false);
        let ahead = |m: &Mat| {
            // The view's forward axis (-z in view space) in world terms, relative to the camera.
            let f = [-VIEW[2], -VIEW[6], -VIEW[10]];
            let c: [f32; 4] = std::array::from_fn(|j| (0..3).map(|i| f[i] * m[4 * i + j]).sum::<f32>());
            (c[0] / c[3], c[1] / c[3])
        };
        let ((kx, ky), (dx, dy)) = (ahead(&k), ahead(&d));
        assert!((kx - dx + 0.00107).abs() < 1e-5 && (ky - dy + 0.00096).abs() < 1e-5, "{kx} {dx} {ky} {dy}");
    }

    /// A decal drawn in list 1, absent in list 2, present in list 3 is
    /// drawn in all three; one absent two lists running is dropped after one;
    /// a held decal keeps its world place when the camera moves.
    /// Two held decals and one kept draw of one main draw: the
    /// count that overflowed is zero now.
    #[test]
    fn filtered_never_underflows() {
        assert_eq!(filtered_count(1, 3), 0);
        assert_eq!(filtered_count(5, 3), 2);
    }

    /// A panic inside the screen pass is caught at the PresentX
    /// boundary, logged with the frame, and the pass stays off.
    #[test]
    fn screen_pass_panic_is_caught() {
        SCREEN_FORCE_PANIC.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(screen_frame().is_none());
        assert!(SCREEN_PANICKED.load(std::sync::atomic::Ordering::Relaxed));
        assert_eq!(SCREEN_FRAME.load(std::sync::atomic::Ordering::Relaxed), 77);
        SCREEN_FORCE_PANIC.store(false, std::sync::atomic::Ordering::Relaxed);
        assert!(screen_frame().is_none(), "off for the rest of the run");
        SCREEN_PANICKED.store(false, std::sync::atomic::Ordering::Relaxed);
    }

    #[test]
    fn decal_union_two_lists() {
        let id: Mat = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let cam = |x: f32| {
            let mut v = id;
            v[12] = -x;
            Camera { viewport: 1, view: v, proj: PROJ }
        };
        let decal = |abs: f32, c: f32| {
            let mut w = id;
            w[12] = abs - c;
            DrawRecord { geom: GeomKey { hash: 7, mesh_at: 1, ..Default::default() }, world: w, viewport: 1, effect: 0, albedo: Albedo::Flat([1.0; 3]), palette: NO_PALETTE, stage: 74, inst_first: 0, inst_count: 0, bone_first: 0, bone_count: 0, material: Material::default(), state: None, remap: NO_REMAP, ped: false }
        };
        let mut held = Vec::new();
        let d1 = decal(100.0, 0.0);
        assert!(decal_union(&mut held, &[&d1], &cam(0.0)).is_empty());
        let out = decal_union(&mut held, &[], &cam(2.0));
        assert_eq!(out.len(), 1, "held through the absent list");
        assert!((out[0].world[12] - 98.0).abs() < 1e-4, "re-placed for the moved camera");
        let d3 = decal(100.0, 3.0);
        assert!(decal_union(&mut held, &[&d3], &cam(3.0)).is_empty(), "present again: nothing extra");
        assert_eq!(decal_union(&mut held, &[], &cam(3.0)).len(), 1);
        assert!(decal_union(&mut held, &[], &cam(3.0)).is_empty(), "two absent lists: gone");
    }

    #[test]
    fn invert_undoes_the_camera() {
        let m = mul(&rotation_only(&VIEW), &PROJ);
        let i = invert(&m).unwrap();
        let id = mul(&m, &i);
        assert!(id.iter().enumerate().all(|(k, v)| (v - if k % 5 == 0 { 1.0 } else { 0.0 }).abs() < 1e-3), "{id:?}");
        assert!(invert(&[0.0; 16]).is_none());
    }

    /// A camera cut, scene A (resident) -> scene B (all new), with run
    /// 93/95's sizes: 1,300 geometries, about 31 MiB, delivered at the cut.
    /// Per variant: frames and ms until the main view is complete (coverage
    /// >= 0.98), held frames, the worst frame on our side (CPU: record incl.
    /// uploads; WARP's GPU time is reported apart, it is not the card's).
    #[test]
    fn cut_benchmark_on_warp() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let (_gpu, host) = warp_host();
        let n = 1_300u64;
        let scene = |base: u64, at: f32| -> (Vec<(GeomKey, HostGeometry)>, Vec<DrawRecord>) {
            let mut geoms = Vec::new();
            let mut draws = Vec::new();
            for i in 0..n {
                let key = GeomKey { mesh_at: base + i, hash: base + i, ..Default::default() };
                // Sizes as in a test run: most small, every 50th large (1.2 MiB).
                let vb = if i % 50 == 0 { 1_200_000 } else { 1_000 + (i as usize * 7919) % 30_000 };
                let tris = if i % 50 == 0 { 2_000 } else { 32 + (i as u32 * 31) % 400 };
                geoms.push((key, HostGeometry { vertices: Vec::new(), indices: vec![0; 3 * tris as usize], raw_vb: Some(vec![0u8; vb]), vb_place: 0 }));
                let mut world: Mat = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
                world[12] = at + (i % 37) as f32;
                world[14] = -5.0 - (i % 91) as f32 * 3.0;
                draws.push(DrawRecord { geom: key, world, viewport: 1, effect: 0, albedo: Albedo::Flat([0.5; 3]), palette: NO_PALETTE, stage: 67, inst_first: 0, inst_count: 0, bone_first: 0, bone_count: 0, material: Material::default(), state: None, remap: NO_REMAP, ped: false });
            }
            (geoms, draws)
        };
        let mib = |g: &[(GeomKey, HostGeometry)]| g.iter().map(|(_, g)| g.raw_vb.as_ref().unwrap().len() + 4 * g.indices.len()).sum::<usize>() as f64 / f64::from(1 << 20);
        let id: Mat = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mut moved = id;
        moved[12] = -1000.0;
        // (name, time budget, boost, hold, frames B is announced before the cut: residency)
        let variants: [(&str, Option<f64>, f64, bool, u64); 6] = [
            ("bytes (32 MiB/2,048)", None, 1.0, false, 0),
            ("time 4 ms", Some(4.0), 1.0, false, 0),
            ("+ recovery boost x3", Some(4.0), cuts::RECOVERY_BOOST, false, 0),
            ("+ hold", Some(4.0), cuts::RECOVERY_BOOST, true, 0),
            ("+ ahead 10 frames", Some(4.0), cuts::RECOVERY_BOOST, true, 10),
            ("+ ahead 120 frames (4 s lead)", Some(4.0), cuts::RECOVERY_BOOST, true, 120),
        ];
        for (name, budget, boost, hold, ahead) in variants {
            let mut pass = Pass::new(&host, 64, 36).unwrap();
            pass.geom.budget = geometry::GeomBudget::SCREEN;
            pass.upload_ms_budget = budget;
            pass.cuts = cuts::CutWatch::new(hold);
            pass.cuts.boost = boost;
            // Scene A, warmed up until complete (and the upload cost measured).
            let (ga, da) = scene(0x10_0000, 0.0);
            let mut list = FrameList { frame: 1, draws: da, cameras: vec![Camera { viewport: 1, view: id, proj: PROJ }], main: Some(1), new_geoms: ga, ..Default::default() };
            for f in 1..400 {
                list.frame = f;
                let st = pass.draw(&list, &mut |_| None, false).unwrap();
                list.new_geoms.clear();
                if st.coverage >= 1.0 && f > 3 {
                    break;
                }
            }
            // The cut: scene B, all new; delivered at the cut frame, or `ahead`
            // frames before it while A is still drawn (resident before the cut).
            let (gb, db) = scene(0x90_0000, 1000.0);
            let size = mib(&gb);
            let mut gb = Some(gb);
            if ahead > 0 {
                let mut la = FrameList { frame: 1000 - ahead, draws: list.draws.clone(), cameras: list.cameras.clone(), main: Some(1), ahead_geoms: gb.take().unwrap(), ..Default::default() };
                let mut worst_a = 0f64;
                for f in 1000 - ahead..1000 {
                    la.frame = f;
                    let t = std::time::Instant::now();
                    let st = pass.draw(&la, &mut |_| None, false).unwrap();
                    worst_a = worst_a.max(t.elapsed().as_secs_f64() * 1e3 - st.gpu_ms);
                    la.ahead_geoms.clear();
                }
                eprintln!("cut benchmark [{name}]: before the cut: uploaded ahead {} geometries, worst frame ours {worst_a:.1} ms, {} still queued", pass.geom.stats.ahead, pass.geom.pending_count());
            }
            let mut list = FrameList { frame: 1000, draws: db, cameras: vec![Camera { viewport: 1, view: moved, proj: PROJ }], main: Some(1), new_geoms: gb.unwrap_or_default(), ..Default::default() };
            let (mut frames, mut held, mut worst, mut gpu, mut total) = (0u32, 0u32, 0f64, 0f64, 0f64);
            let mut first_shown_cov = None;
            loop {
                let t = std::time::Instant::now();
                let st = pass.draw(&list, &mut |_| None, false).unwrap();
                let ms = t.elapsed().as_secs_f64() * 1e3 - st.gpu_ms;
                list.new_geoms.clear();
                list.frame += 1;
                frames += 1;
                held += u32::from(st.held);
                worst = worst.max(ms);
                gpu = gpu.max(st.gpu_ms);
                total += ms + st.gpu_ms;
                if !st.held && first_shown_cov.is_none() {
                    first_shown_cov = Some(st.coverage);
                }
                if st.coverage >= cuts::COMPLETE_COVERAGE || frames >= 400 {
                    break;
                }
            }
            let line = pass.cuts.line.take().unwrap_or_default();
            eprintln!("cut benchmark [{name}]: ahead {} of which drawn {}; {size:.1} MiB, 1,300 geometries: complete after {frames} frames / {total:.0} ms, held {held}, worst frame ours {worst:.1} ms (WARP GPU max {gpu:.1} ms), first shown frame at coverage {:.2}; upload cost {:.3} ms/geometry {:.2} ms/MiB | {line}", pass.geom.stats.ahead, pass.geom.stats.ahead_drawn, first_shown_cov.unwrap_or(0.0), pass.upload_cost.0, pass.upload_cost.1);
            assert!(frames < 400, "[{name}] never complete");
            if hold {
                assert!(first_shown_cov.is_none_or(|c| c >= cuts::HOLD_COVERAGE) || held > 0, "[{name}] a partial picture shown without a hold");
            }
        }
    }

    /// A test run's frame 1711 uploaded 1,230 geometries (32.8 MiB) in
    /// 680 ms. The same burst on WARP, with the CPU side split: buffer
    /// creation alone vs the whole upload path.
    #[test]
    fn upload_burst_cost_on_warp() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let (_gpu, host) = warp_host();
        let mut pass = Pass::new(&host, 64, 36).unwrap();
        let id: Mat = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let n = 1_230u64;
        let new_geoms: Vec<(GeomKey, HostGeometry)> = (0..n)
            .map(|i| (GeomKey { mesh_at: 0x1000 + i, hash: i, ..Default::default() }, HostGeometry { vertices: Vec::new(), indices: (0..1_536).collect(), raw_vb: Some(vec![0u8; 21_200]), vb_place: 0 }))
            .collect();
        let mib = new_geoms.iter().map(|(_, g)| g.raw_vb.as_ref().unwrap().len() + 4 * g.indices.len()).sum::<usize>() as f64 / f64::from(1 << 20);
        let list = FrameList { frame: 1, cameras: vec![Camera { viewport: 1, view: id, proj: PROJ }], main: Some(1), new_geoms, ..Default::default() };
        let st = pass.draw(&list, &mut |_| None, false).unwrap();
        // Buffer creation alone, for the same count and sizes.
        let t0 = std::time::Instant::now();
        let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: 27_392, Height: 1, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_UNKNOWN, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
        let mut keep = Vec::new();
        for _ in 0..n {
            let mut out: Option<ID3D12Resource> = None;
            // SAFETY: a plain buffer description.
            unsafe { host.device.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut out) }.unwrap();
            keep.push(out);
        }
        let create = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!("upload burst: {} geometries, {mib:.1} MiB: upload {:.1} ms ({:.2} ms/MiB, {:.3} ms each); committed buffer creation alone {create:.1} ms ({:.3} ms each); record {:.1} ms, GPU {:.1} ms", st.uploaded_geometries, st.upload_ms, st.upload_ms / mib, st.upload_ms / n as f64, create / n as f64, st.record_ms, st.gpu_ms);
        assert_eq!(st.uploaded_geometries as u64, n);
        assert_eq!(pass.pages.bytes(), GEOM_PAGE, "32 MiB in one 64 MiB page");
        // 3: the budget by time: with the burst's measured cost, a 4 ms
        // frame takes only what fits (WARP's cost, so a few hundred at most).
        assert!(st.upload_cost.0 > 0.0 && st.upload_cost.1 > 0.0, "{:?}", st.upload_cost);
        let mut timed = Pass::new(&host, 64, 36).unwrap();
        timed.geom.budget = geometry::GeomBudget::SCREEN;
        timed.upload_ms_budget = Some(4.0);
        timed.upload_cost = st.upload_cost;
        let mut l2 = list.clone();
        l2.frame = 1;
        let s2 = timed.draw(&l2, &mut |_| None, false).unwrap();
        let allowed = (4.0 / st.upload_cost.0) as usize;
        eprintln!("upload by time: cost {:.3} ms/upload, {:.2} ms/MiB: {} uploads in {:.1} ms (budget 4 ms, about {allowed})", st.upload_cost.0, st.upload_cost.1, s2.uploaded_geometries, s2.upload_ms);
        assert!(s2.uploaded_geometries <= allowed.clamp(8, 2048) && s2.uploaded_geometries < n as usize);
        // Free and coalesce: three ranges back to one.
        let mut pages = Pages::default();
        let a = pages.alloc(&host.device, 1000).unwrap();
        let b = pages.alloc(&host.device, 3000).unwrap();
        let c = pages.alloc(&host.device, 5000).unwrap();
        assert_eq!((a.1, b.1, c.1), (0, 1024, 4096));
        pages.free(b);
        assert_eq!(pages.alloc(&host.device, 2000).unwrap().1, 1024, "first fit reuses the hole");
        pages.free((0, 1024, 2048));
        pages.free(a);
        pages.free(c);
        assert_eq!(pages.pages[0].2, vec![(0, GEOM_PAGE)], "all coalesced");
        let big = pages.alloc(&host.device, GEOM_PAGE + 1).unwrap();
        assert_eq!((big.0, pages.bytes()), (1, GEOM_PAGE + GEOM_PAGE + 256));
    }

    #[test]
    fn water_on_warp() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let (_gpu, host) = warp_host();
        let mut pass = Pass::new(&host, 64, 36).unwrap();
        let id: Mat = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        // A quad 3 m in front of a camera at (100, 200, 0) looking along -z (absolute coordinates).
        let o = [100.0f32, 200.0, 0.0];
        let grid = WaterGrid { id: 7, vertices: vec![[o[0] - 2.0, o[1] - 2.0, -3.0], [o[0] + 2.0, o[1] - 2.0, -3.0], [o[0] + 2.0, o[1] + 2.0, -3.0], [o[0] - 2.0, o[1] + 2.0, -3.0]], indices: vec![0, 1, 2, 0, 2, 3] };
        let water = Water { grid: std::sync::Arc::new(grid), origin: o, tint: [0.0, 0.0, 1.0, 1.0] };
        let list = FrameList { frame: 1, cameras: vec![Camera { viewport: 1, view: id, proj: PROJ }], main: Some(1), water: Some(water), ..Default::default() };
        let st = pass.draw(&list, &mut |_| None, true).unwrap();
        assert!(st.water);
        let px = pass.read_rgba().unwrap();
        let at = (18 * 64 + 32) * 4;
        assert!(px[at + 2] > 80 && px[at] < 40, "water in the centre: {:?}", &px[at..at + 4]);
        pass.water = false;
        let st = pass.draw(&list, &mut |_| None, true).unwrap();
        assert!(!st.water);
    }

    #[test]
    fn sky_on_warp() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let (_gpu, host) = warp_host();
        let mut pass = Pass::new(&host, 64, 36).unwrap();
        let id: Mat = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        // The camera looks along -z: the centre is below the horizon (horizon colour, red here).
        let sky = Sky { sun: [0.0, 0.0, 1.0, 0.9999], sun_col: [1.0, 1.0, 1.0, 1.0], zenith: [0.0, 0.0, 1.0, 0.0], horizon: [1.0, 0.0, 0.0, 0.0], moon: [0.0; 4], fog: [1.0, 0.0, 0.0, 1e-4] };
        let mut list = FrameList { frame: 1, cameras: vec![Camera { viewport: 1, view: id, proj: PROJ }], main: Some(1), sky: Some(sky), ..Default::default() };
        let st = pass.draw(&list, &mut |_| None, true).unwrap();
        assert!(st.sky);
        let px = pass.read_rgba().unwrap();
        let at = (18 * 64 + 32) * 4;
        assert!(px[at] > 80 && px[at + 2] < 40, "horizon colour below the horizon: {:?}", &px[at..at + 4]);
        pass.sky = false;
        list.frame = 2;
        let st = pass.draw(&list, &mut |_| None, true).unwrap();
        let px = pass.read_rgba().unwrap();
        assert!(!st.sky && px[at] < 40, "sky=off: the clear colour {:?}", &px[at..at + 4]);
    }

    #[test]
    fn title_state_and_stage_filter_on_warp() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let (_gpu, host) = warp_host();
        let mut pass = Pass::new(&host, 64, 36).unwrap();
        let word = |stride: u32, offset: u32, format: u32| (stride / 4) << 5 | (offset / 4) << 9 | format;
        let mut vb = vec![0u8; 0x80];
        vb[0..4].copy_from_slice(&word(12, 0x80, 3).to_le_bytes());
        for v in [[-1.0f32, -1.0, -3.0], [1.0, -1.0, -3.0], [0.0, 1.0, -3.0]] {
            for x in v {
                vb.extend(x.to_le_bytes());
            }
        }
        let id: Mat = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let geom = GeomKey { mesh_at: 9, ..Default::default() };
        let rec = |rgb: [f32; 3], stage: u32, state: Option<RenderState>| DrawRecord { geom, world: id, viewport: 1, effect: 0, albedo: Albedo::Flat(rgb), palette: NO_PALETTE, stage, inst_first: 0, inst_count: 0, bone_first: 0, bone_count: 0, material: Material::default(), state, remap: NO_REMAP, ped: false };
        // Additive blend (ONE, ONE, ADD), depth test GE without write: the decal's red adds to the opaque blue.
        let add = RenderState { depth_enable: true, depth_write: false, depth_func: 7, depth_bias: 0, bias_clamp: 0, slope_bias: 0, blend: [1, 2, 2, 1, 2, 2, 1, 0xF] };
        let g = HostGeometry { vertices: Vec::new(), indices: vec![0, 1, 2], raw_vb: Some(vb), vb_place: 0 };
        let draws = vec![rec([1.0, 0.0, 0.0], 74, Some(add)), rec([0.0, 1.0, 0.0], 36, None), rec([0.0, 0.0, 1.0], 67, None)];
        let list = FrameList { frame: 1, draws, cameras: vec![Camera { viewport: 1, view: id, proj: PROJ }], main: Some(1), new_geoms: vec![(geom, g)], ..Default::default() };
        let st = pass.draw(&list, &mut |_| None, true).unwrap();
        assert_eq!((st.draws, st.stated, st.filtered), (2, 1, 1), "stage 36 left out; the decal with the title's state");
        // 2: with the pipeline worker, a new state draws with our default
        // first (never made on the frame path), then with its own once ready.
        {
            let mut p2 = Pass::new(&host, 64, 36).unwrap();
            p2.pso_async = true;
            let mut l2 = list.clone();
            let mut seen = Vec::new();
            for f in 1..200u64 {
                l2.frame = f;
                let st = p2.draw(&l2, &mut |_| None, false).unwrap();
                assert_eq!(st.pso_ms, 0.0, "no pipeline made on the frame path");
                seen.push(st.stated);
                if st.stated == 1 {
                    break;
                }
                l2.new_geoms.clear();
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            assert_eq!((seen[0], *seen.last().unwrap()), (0, 1), "default first, the title's state once ready: {seen:?}");
        }
        let px = pass.read_rgba().unwrap();
        let at = (18 * 64 + 32) * 4;
        assert!(px[at] > 60 && px[at + 2] > 60 && px[at + 1] < 60, "opaque blue, then the decal added (drawn after it): {:?}", &px[at..at + 4]);
        pass.states = false;
        let mut list = list;
        list.new_geoms.clear();
        list.frame = 2;
        let st = pass.draw(&list, &mut |_| None, true).unwrap();
        assert_eq!(st.stated, 0);
    }
    /// A WARP device for the device tests: XNW_ADAPTER set once, before any
    /// test makes a device (set_var racing other threads' reads crashed the
    /// parallel run).
    /// One WARP device at a time: parallel device tests crashed WARP
    /// intermittently (0xC0000005, 1 run in 3 on the remote box).
    /// The title's chain on WARP. A flat mid-grey (stored albedo
    /// 0.5) surface facing the light (normal +z, light towards +z) with
    /// radiance pi, exposure 1, no ambient: hdr = 0.5 * pi / pi = 0.5, so the
    /// pixel is display(0.5) = (Hable(0.5)/Hable(4))^(1/2.2) (one conversion).
    /// The light behind it (towards -z): unlit (0 with no ambient).
    /// light_dir_mult x2 brightens it; `light_old` gives the earlier look.
    #[test]
    fn title_lighting_on_warp() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let (_gpu, host) = warp_host();
        let mut pass = Pass::new(&host, 64, 36).unwrap();
        let word = |stride: u32, offset: u32, format: u32| (stride / 4) << 5 | (offset / 4) << 9 | format;
        let mut vb = vec![0u8; 0x80];
        vb[0..4].copy_from_slice(&word(12, 0x80, 3).to_le_bytes());
        for v in [[-1.0f32, -1.0, -3.0], [1.0, -1.0, -3.0], [0.0, 1.0, -3.0]] {
            for x in v {
                vb.extend(x.to_le_bytes());
            }
        }
        let id: Mat = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let geom = GeomKey { mesh_at: 13, ..Default::default() };
        let rec = DrawRecord { geom, world: id, viewport: 1, effect: 0, albedo: Albedo::Flat([0.5, 0.5, 0.5]), palette: NO_PALETTE, stage: 0, inst_first: 0, inst_count: 0, bone_first: 0, bone_count: 0, material: Material::default(), state: None, remap: NO_REMAP, ped: false };
        let g = HostGeometry { vertices: Vec::new(), indices: vec![0, 1, 2], raw_vb: Some(vb), vb_place: 0 };
        let curve = [0.22, 0.3, 0.1, 0.2, 0.01, 0.3, 4.0];
        let pi = std::f32::consts::PI;
        let light = lighting::Lighting { dir: [0.0, 0.0, 1.0], radiance: [pi; 3], mult: pi, amb_sky: [0.0; 3], amb_ground: [0.0; 3], exposure: 1.0, ev: 0.0, bias: 0.0, strength: 1.0, moon: false, sky_scale: 1.0, curve };
        let mut list = FrameList { frame: 1, draws: vec![rec], cameras: vec![Camera { viewport: 1, view: id, proj: PROJ }], main: Some(1), new_geoms: vec![(geom, g)], light: Some(light), ..Default::default() };
        let grey = |pass: &mut Pass, list: &mut FrameList| -> i32 {
            pass.draw(list, &mut |_| None, true).unwrap();
            list.new_geoms.clear();
            list.frame += 1;
            i32::from(pass.read_rgba().unwrap()[(18 * 64 + 32) * 4])
        };
        let want = (lighting::display(0.5, &curve) * 255.0).round() as i32;
        let lit = grey(&mut pass, &mut list);
        assert!((lit - want).abs() <= 3, "mid-grey lit {lit}, expected {want} (display(0.5))");
        list.light.as_mut().unwrap().dir = [0.0, 0.0, -1.0];
        let back = grey(&mut pass, &mut list);
        assert!(back <= 2, "a back-facing surface is unlit: {back}");
        let l = list.light.as_mut().unwrap();
        (l.dir, l.radiance) = ([0.0, 0.0, 1.0], [2.0 * pi; 3]);
        let brighter = grey(&mut pass, &mut list);
        assert!(brighter > lit + 20, "light_dir_mult x2: {brighter} vs {lit}");
        pass.light_old = true;
        let old = grey(&mut pass, &mut list);
        assert!(old != brighter, "light=old: the old path ({old})");
    }
    /// Layer weights from the vertex colour (COLOR1 stream word 0x34,
    /// unorm8x4): every vertex (0, 0, 255, 0) is all layer 2; with
    /// `layers_debug` on a ground stage the pixel is layer 2's debug blue;
    /// with COLOR0 chosen (no stream there) C = 0 puts all weight on layer 3.
    #[test]
    fn layer_weights_on_warp() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let (_gpu, host) = warp_host();
        let mut pass = Pass::new(&host, 64, 36).unwrap();
        let word = |stride: u32, offset: u32, format: u32| (stride / 4) << 5 | (offset / 4) << 9 | format;
        let mut vb = vec![0u8; 0x80];
        vb[0..4].copy_from_slice(&word(12, 0x80, 3).to_le_bytes());
        vb[0x34..0x38].copy_from_slice(&word(4, 0xA4, 6).to_le_bytes());
        for v in [[-1.0f32, -1.0, -3.0], [1.0, -1.0, -3.0], [0.0, 1.0, -3.0]] {
            for x in v {
                vb.extend(x.to_le_bytes());
            }
        }
        for _ in 0..3 {
            vb.extend([0u8, 0, 255, 0]);
        }
        let id: Mat = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let geom = GeomKey { mesh_at: 15, ..Default::default() };
        let rec = DrawRecord { geom, world: id, viewport: 1, effect: 0, albedo: Albedo::Flat([0.9, 0.9, 0.9]), palette: NO_PALETTE, stage: 71, inst_first: 0, inst_count: 0, bone_first: 0, bone_count: 0, material: Material::default(), state: None, remap: NO_REMAP, ped: false };
        let g = HostGeometry { vertices: Vec::new(), indices: vec![0, 1, 2], raw_vb: Some(vb), vb_place: 0 };
        let mut list = FrameList { frame: 1, draws: vec![rec], cameras: vec![Camera { viewport: 1, view: id, proj: PROJ }], main: Some(1), new_geoms: vec![(geom, g)], ..Default::default() };
        pass.layers_debug = true;
        let st = pass.draw(&list, &mut |_| None, true).unwrap();
        let px = pass.read_rgba().unwrap();
        let c = &px[(18 * 64 + 32) * 4..(18 * 64 + 32) * 4 + 3];
        assert_eq!(st.layered, 1);
        assert!(c[2] > c[0] + 60 && c[2] > c[1] + 40, "layer 2's blue: {c:?}");
        pass.layers = 1;
        list.new_geoms.clear();
        list.frame = 2;
        pass.draw(&list, &mut |_| None, true).unwrap();
        let px = pass.read_rgba().unwrap();
        let c = &px[(18 * 64 + 32) * 4..(18 * 64 + 32) * 4 + 3];
        assert!(c[0] > c[2] + 60, "COLOR0: no stream, C = 0, all weight on layer 3 (debug yellow): {c:?}");
    }

    pub(super) fn warp_host() -> (std::sync::MutexGuard<'static, ()>, xnw_gpu::host::Host) {
        static ONCE: std::sync::Once = std::sync::Once::new();
        static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let guard = GPU.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: inside call_once; every test reaches its device through here.
        ONCE.call_once(|| unsafe { std::env::set_var("XNW_ADAPTER", "warp") });
        (guard, xnw_gpu::host::Host::new(None).expect("WARP device"))
    }
    #[test]
    fn screen_switch() {
        assert_eq!(parse_screen("1"), None);
        assert_eq!(parse_screen("screen"), Some(ScreenOptions { wire: false, stage: None, skin: true, states: true, sky: true, water: true, peds: false, decal_hold: true }));
        assert_eq!(parse_screen("screen,decalhold=off").map(|o| o.decal_hold), Some(false));
        assert_eq!(parse_screen("screen,peds=on").map(|o| o.peds), Some(true));
        // Skinrules= is not a stage; default 0; hex; last wins; junk is 0.
        assert_eq!(parse_screen("screen,skinrules=b").map(|o| o.stage), Some(None));
        assert_eq!(parse_skin_rules("screen,peds=on"), 0);
        assert_eq!(parse_skin_rules("screen,skinrules=b"), 0xB);
        assert_eq!(parse_skin_rules("screen,skinrules=0x3, skinrules=9"), 9);
        assert_eq!(parse_skin_rules("screen,skinrules=zz"), 0);
        assert_eq!(parse_screen("screen,wire,pass=67,skin=off"), Some(ScreenOptions { wire: true, stage: Some(67), skin: false, states: true, sky: true, water: true, peds: false, decal_hold: true }));
        assert_eq!(parse_screen("screen,water=off").map(|o| o.water), Some(false));
        assert_eq!(parse_screen("screen,sky=off").map(|o| o.sky), Some(false));
        assert_eq!(parse_screen("screen,state=off").map(|o| o.states), Some(false));
    }
    #[test]
    fn shader_compiles_through_compile() {
        assert!(compile(c"vs", c"vs_5_1").is_ok() && compile(c"ps", c"ps_5_1").is_ok());
    }

    /// The shader compiles with FXC (CPU only, as the presenter's does).
    #[test]
    fn shader_compiles() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::core::PCSTR;
        for (entry, target) in [(c"vs", c"vs_5_1"), (c"ps", c"ps_5_1")] {
            let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
            // SAFETY: static source and strings; out pointers ours.
            let r = unsafe { D3DCompile(SHADER.as_ptr().cast(), SHADER.len(), PCSTR(c"world_render".as_ptr().cast()), None, None, PCSTR(entry.as_ptr().cast()), PCSTR(target.as_ptr().cast()), 0, 0, &mut code, Some(&mut errors)) };
            let text = errors.map(|e| {
                // SAFETY: the blob's bytes.
                String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(e.GetBufferPointer().cast::<u8>(), e.GetBufferSize()) }).into_owned()
            });
            assert!(r.is_ok(), "{entry:?}: {text:?}");
            assert!(code.is_some());
        }
    }

    /// Identity, rigid (a rotation) and sheared / zero windows.
    #[test]
    fn skin_window_kinds() {
        let id = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let rot = [0.0, -1.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0];
        let shear = [1.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let list = FrameList { palettes: vec![[id, id, rot, shear, [0.0; 12]].concat()], ..Default::default() };
        let r = |first: u32, count: u32| DrawRecord { palette: 0, bone_first: first, bone_count: count, geom: GeomKey::default(), world: translation([0.0; 3]), viewport: 1, effect: 0, albedo: Albedo::Flat([1.0; 3]), stage: 67, inst_first: 0, inst_count: 0, material: Material::default(), state: None, remap: NO_REMAP, ped: false };
        assert_eq!(skin_window(&list, &r(0, 2)), SkinWindow::Identity);
        assert_eq!(skin_window(&list, &r(1, 2)), SkinWindow::Rigid);
        assert_eq!(skin_window(&list, &r(2, 2)), SkinWindow::Other);
        assert_eq!(skin_window(&list, &r(4, 1)), SkinWindow::Other);
        assert_eq!(skin_window(&list, &r(4, 2)), SkinWindow::Other);
        assert_eq!(skin_window(&list, &DrawRecord { palette: NO_PALETTE, ..r(0, 2) }), SkinWindow::Other);
        // Draw-time bones leave the skip under bit 3 only, whatever they hold.
        let mut l2 = FrameList { palettes: vec![shear.to_vec(), shear.to_vec()], palette_at_draw: vec![false, true], ..Default::default() };
        let at = |p: u32| DrawRecord { palette: p, ..r(0, 1) };
        assert!(!skin_lifted(&l2, &at(1), SKIN_IDENTITY_STATIC | SKIN_RIGID_BONES));
        assert!(skin_lifted(&l2, &at(1), SKIN_DRAW_TIME));
        assert!(!skin_lifted(&l2, &at(0), SKIN_DRAW_TIME));
        l2.palette_at_draw.clear();
        assert!(!skin_lifted(&l2, &at(1), SKIN_DRAW_TIME));
        assert!(skin_lifted(&list, &r(0, 2), SKIN_IDENTITY_STATIC));
    }
}
