//! The title's heightmap terrain drawn by our world pass (shader
//! family 4, system A).
//!
//! The title draws it at stage 72 (0x14096A610 -> 0x143288FB0,
//! DrawInstancedIndirect) from patches a compute pass writes; that never
//! reaches our draw hooks and draws nothing in null-GPU mode. What is CPU
//! data [V 0x143288320]: the array of 372-byte tile records rebuilt every
//! frame at 0x148420858 {u32 cap, u32 count, rec*}. A record
//! (VS/PS_RasterizePatches read whole):
//!
//! | off | field |
//! |---|---|
//! | +0x00/+0x10/+0x20 | rows 0..2: `P = L.x*row0 + L.y*row1 + h*row2 + row3` (L tile-local metres 0..512) |
//! | +0x30 | row 3: the tile's world corner |
//! | +0x40/+0x80/+0xC0/+0x100 | bakedA..D: 16 bindless indices each, a 4x4 grid of 128 m cells |
//! | +0x148 | heightmap (bindless index): `h = 1600 * (s - 0.5)` m |
//! | +0x158 | hole mask (bindless index): a triangle whose centroid samples > 0.5 is dropped (lod < 5) |
//! | +0x168 | nBaked: cell clamp, 0 = no baked set |
//! | +0x16C/+0x170 | lod (distance level), and the hole mask mip clamp |
//!
//! A bindless index is a 32-byte descriptor slot of the title's heap:
//! `[0x14A943AA8] + (index << 5)` [V 0x144DA90A0]. The pass never
//! hands an index to a shader: each slot goes through the albedo [`TexTable`]
//! (guarded record copy, generation check, our host copy, the null-mode
//! upload of `texture::for_pass`), and the shader indexes only our own
//! 256-entry table with indices our CPU code picked (entry 0 = none).
//!
//! Geometry: our own grid per visible tile, no vertex buffer
//! (`SV_VertexID` -> quad, corner): `(256 >> lod)` quads a side, 8..64
//! (64 = 8 m quads; the title's patches are 16x finer at lod 0). Tiles are
//! culled against the frustum on the CPU (their box over h in -800..800 m),
//! drawn nearest first under a triangle cap (a tile over the cap is
//! coarsened, down to 4 quads, then left out). Heights and normals (two
//! taps each way, as the title) are the vertex shader's; holes the pixel
//! shader's (the title culls in its VS; ours did too until WARP lost the
//! device with heights and holes both in the VS); colour is
//! the title's not-resident fallback, the baked texture of the 128 m cell
//! (bakedA), else a flat earth tone; lit and fogged as the rest of the pass.
//! Camera-relative: row 3 minus the camera in f64 on the CPU.

use std::collections::HashMap;
use std::sync::Arc;

use windows::Win32::Graphics::Direct3D12::*;

use super::textures::{self, Slot, TexMode, TexOptions, TexTable};
use super::{Camera, FrameList, RenderState};

/// Record size and the records read at most.
pub const TILE_BYTES: usize = 372;
pub const MAX_TILES: usize = 4096;
/// Bindless indices above this are not slots of the title's heap [I bound;
/// the slot's record and generation are checked again anyway].
pub const MAX_BINDLESS: u32 = 1 << 20;
/// Triangles per frame at most (`terraintri=<thousands>`).
pub const MAX_TRIANGLES: u64 = 2_000_000;
/// Our table of SRVs (own heap; entry 0 is "none").
const SLOTS: u32 = 256;
/// h = 1600 * (s - 0.5): the box of a tile over h.
const H_RANGE: f64 = 800.0;

/// One tile record, parsed: the bindless indices turned into slot addresses
/// (0: none or out of range).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TerrainTile {
    pub rows: [[f32; 4]; 4],
    pub lod: u32,
    pub hole_mip: u32,
    pub height: u64,
    pub holes: u64,
    pub baked: [u64; 16],
    pub n_baked: u32,
    /// Rows 0..2 were degenerate: identity used instead.
    pub identity: bool,
}

/// The frame's tile array (as read on the recording side).
#[derive(Clone, Debug, Default)]
pub struct Terrain {
    pub cap: u32,
    pub count: u32,
    pub heap_base: u64,
    pub tiles: Vec<TerrainTile>,
    /// The raw records (dumps), and the read's cost.
    pub raw: Vec<u8>,
    pub read_ms: f64,
}

fn le32(b: &[u8], at: usize) -> u32 {
    b.get(at..at + 4).map_or(0, |s| u32::from_le_bytes(s.try_into().unwrap()))
}

/// The slot of bindless index `i` in the heap at `base` (0: none).
pub fn slot_of(base: u64, i: u32) -> u64 {
    if base == 0 || i >= MAX_BINDLESS { 0 } else { base + (u64::from(i) << 5) }
}

/// Parses `raw` (whole records) with the title's heap base `heap_base`.
pub fn parse(cap: u32, count: u32, raw: Vec<u8>, heap_base: u64) -> Terrain {
    let mut tiles = Vec::new();
    for r in raw.chunks_exact(TILE_BYTES).take(MAX_TILES) {
        let f = |o: usize| f32::from_bits(le32(r, o));
        let mut rows: [[f32; 4]; 4] = std::array::from_fn(|k| std::array::from_fn(|j| f(16 * k + 4 * j)));
        let len = |v: &[f32; 4]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let ok = rows.iter().flatten().all(|v| v.is_finite()) && rows[..3].iter().all(|v| len(v) > 1e-3 && len(v) < 1e3);
        if !ok {
            for (k, row) in rows.iter_mut().take(3).enumerate() {
                *row = [0.0; 4];
                row[k] = 1.0;
            }
            if !rows[3].iter().all(|v| v.is_finite()) {
                rows[3] = [0.0; 4];
            }
        }
        tiles.push(TerrainTile {
            rows,
            lod: le32(r, 0x16C),
            hole_mip: le32(r, 0x170),
            height: slot_of(heap_base, le32(r, 0x148)),
            holes: slot_of(heap_base, le32(r, 0x158)),
            baked: std::array::from_fn(|c| slot_of(heap_base, le32(r, 0x40 + 4 * c))),
            n_baked: le32(r, 0x168),
            identity: !ok,
        });
    }
    Terrain { cap, count, heap_base, tiles, raw, read_ms: 0.0 }
}

/// Quads a side for a tile's lod (the title: 16 * (64 >> lod)).
pub fn quads_for(lod: u32) -> u32 {
    (256u32.checked_shr(lod).unwrap_or(0)).clamp(8, 64)
}

/// `XNW_WORLD_RENDER=screen,...`: `terrain=on` (off by default: the title's tile array is never
/// filled in this build, the ground is ordinary mesh), `terrain=off`, `terraintri=<thousands>`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainOptions {
    pub on: bool,
    pub max_triangles: u64,
}

impl Default for TerrainOptions {
    fn default() -> Self {
        TerrainOptions { on: false, max_triangles: MAX_TRIANGLES }
    }
}

pub fn parse_screen(v: &str) -> TerrainOptions {
    let mut o = TerrainOptions::default();
    for p in v.split(',').map(str::trim) {
        if p == "terrain=on" {
            o.on = true;
        } else if p == "terrain=off" {
            o.on = false;
        } else if let Some(n) = p.strip_prefix("terraintri=").and_then(|s| s.parse::<u64>().ok()) {
            o.max_triangles = n.max(1) * 1000;
        }
    }
    o
}

/// One frame's plan: tile index, quads a side, camera-relative row 3.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Planned {
    pub tile: usize,
    pub quads: u32,
    pub origin: [f32; 3],
}

/// What a frame did (and what the read found).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TerrainStats {
    pub tiles: usize,
    pub count: u32,
    pub cap: u32,
    pub per_lod: [u32; 8],
    pub drawn: usize,
    pub culled: usize,
    pub over_budget: usize,
    pub coarsened: usize,
    pub identity: usize,
    /// Title patches the drawn tiles stand for ((64 >> lod)^2 each).
    pub patches: u64,
    pub triangles: u64,
    pub heights: usize,
    pub baked: usize,
    pub baked_cells: usize,
    pub holes: usize,
    pub read_ms: f64,
    pub record_ms: f64,
}

/// The tiles to draw: frustum culled (row-vector `vp`, reverse Z), nearest
/// first, coarsened then dropped over `max_triangles`.
pub fn plan(t: &Terrain, cam: [f64; 3], vp: &super::Mat, max_triangles: u64, st: &mut TerrainStats) -> Vec<Planned> {
    let mut vis: Vec<(f64, Planned)> = Vec::new();
    for (i, tile) in t.tiles.iter().enumerate() {
        st.per_lod[(tile.lod as usize).min(7)] += 1;
        st.identity += usize::from(tile.identity);
        let r = &tile.rows;
        let o: [f64; 3] = std::array::from_fn(|k| f64::from(r[3][k]) - cam[k]);
        let corner = |lx: f64, ly: f64, h: f64| -> [f64; 3] { std::array::from_fn(|k| lx * f64::from(r[0][k]) + ly * f64::from(r[1][k]) + h * f64::from(r[2][k]) + o[k]) };
        let mut out = [true; 5];
        let mut centre = [0f64; 3];
        for (lx, ly, h) in [0.0, 512.0].iter().flat_map(|&x| [0.0, 512.0].map(move |y| (x, y))).flat_map(|(x, y)| [-H_RANGE, H_RANGE].map(move |h| (x, y, h))) {
            let p = corner(lx, ly, h);
            for k in 0..3 {
                centre[k] += p[k] / 8.0;
            }
            let c: [f64; 4] = std::array::from_fn(|j| p[0] * f64::from(vp[j]) + p[1] * f64::from(vp[4 + j]) + p[2] * f64::from(vp[8 + j]) + f64::from(vp[12 + j]));
            let inside = [c[3] > 0.0, c[0] >= -c[3], c[0] <= c[3], c[1] >= -c[3], c[1] <= c[3]];
            for k in 0..5 {
                out[k] &= !inside[k];
            }
        }
        // Inside the box: never culled.
        let near = centre.iter().map(|v| v * v).sum::<f64>().sqrt() < 512.0;
        if !near && out.iter().any(|&o| o) {
            st.culled += 1;
            continue;
        }
        vis.push((centre.iter().map(|v| v * v).sum(), Planned { tile: i, quads: quads_for(tile.lod), origin: o.map(|v| v as f32) }));
    }
    vis.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out = Vec::with_capacity(vis.len());
    let mut total = 0u64;
    for (_, mut p) in vis {
        let tri = |n: u32| 2 * u64::from(n) * u64::from(n);
        let first = p.quads;
        while total + tri(p.quads) > max_triangles && p.quads > 4 {
            p.quads /= 2;
        }
        if total + tri(p.quads) > max_triangles {
            st.over_budget += 1;
            continue;
        }
        st.coarsened += usize::from(p.quads != first);
        total += tri(p.quads);
        let lod = t.tiles[p.tile].lod;
        st.patches += u64::from(64u32.checked_shr(lod).unwrap_or(0).max(1)).pow(2);
        out.push(p);
    }
    st.triangles = total;
    out
}

pub const TERRAIN_SHADER: &str = r#"
#define RS "RootConstants(num32BitConstants=56, b0), DescriptorTable(SRV(t0, numDescriptors=256)), StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP), StaticSampler(s1, filter=FILTER_ANISOTROPIC, maxAnisotropy=8), StaticSampler(s2, filter=FILTER_MIN_MAG_MIP_POINT, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP)"
cbuffer Tile : register(b0) {
    row_major float4x4 vp;   // 0: camera-relative view-projection
    float4 row0, row1, row2; // 16: P = L.x*row0 + L.y*row1 + h*row2 + origin
    float4 origin;           // 28: row 3 minus the camera
    uint quads;              // 32: quads a side
    uint heightIx;           // 33: our table entry of the heightmap (0: none, h = 0)
    uint holeIx;             // 34: of the hole mask (0: none)
    uint holeMip;            // 35: the mask's mip (holes only tested when the tile's lod < 5)
    uint4 baked;             // 36: 16 x u8 table entries of bakedA (0: none)
    uint nBaked;             // 40: the cell clamp (0: no baked set)
    uint srgbMask;           // 41: bit c: cell c's view is sRGB
    uint2 pad;
    float4 sunDir;           // 44: light direction (xyz), ambient (w)
    float4 sunCol;           // 48: colour (rgb), w = 1 when set
    float4 fog;              // 52: fog colour, 1 / distance (0: off)
};
Texture2D tex[256] : register(t0);
SamplerState clampLinear : register(s0);
SamplerState wrapAniso : register(s1);
SamplerState clampPoint : register(s2);
float height(float2 L) {
    if (heightIx == 0) return 0;
    uint w, h;
    tex[min(heightIx, 255)].GetDimensions(w, h);
    float2 uv = float2(L.x / 512 + 0.5 / max(w, 1), 1 - L.y / 512 - 0.5 / max(h, 1));
    return 1600 * (tex[min(heightIx, 255)].SampleLevel(clampLinear, uv, 0).r - 0.5);
}
struct VsOut { float4 pos : SV_Position; float3 normal : NORMAL; float2 L : TEXCOORD0; };
[RootSignature(RS)]
VsOut vs(uint v : SV_VertexID) {
    uint tri = v / 3, corner = v % 3, q = tri >> 1, second = tri & 1;
    uint n = max(quads, 1), qx = q % n, qy = q / n;
    // (0,0) (1,0) (1,1) | (0,0) (1,1) (0,1)
    uint2 c = second == 0 ? uint2(corner != 0, corner == 2) : uint2(corner == 1, corner != 0);
    float step = 512.0 / n;
    float2 L = float2(qx + c.x, qy + c.y) * step;
    VsOut o;
    float h = height(L);
    float dx = (height(L + float2(step, 0)) - height(L - float2(step, 0))) / (2 * step);
    float dy = (height(L + float2(0, step)) - height(L - float2(0, step))) / (2 * step);
    float3 p = L.x * row0.xyz + L.y * row1.xyz + h * row2.xyz + origin.xyz;
    o.pos = mul(float4(p, 1), vp);
    o.normal = normalize(cross(row0.xyz + dx * row2.xyz, row1.xyz + dy * row2.xyz) + 1e-6);
    o.L = L;
    return o;
}
[RootSignature(RS)]
float4 ps(VsOut i) : SV_Target {
    if (holeIx != 0) {
        // The title drops a triangle whose centroid samples > 0.5 (its VS);
        // here per pixel, with the centroid of the pixel's grid triangle
        // (both in the VS crashed WARP).
        float n = max(quads, 1), step = 512.0 / n;
        float2 g = i.L / step, q = min(floor(g), n - 1), f = g - q;
        float2 cen = (q + (f.y <= f.x ? float2(2, 1) : float2(1, 2)) / 3) * step;
        if (tex[min(holeIx, 255)].SampleLevel(clampPoint, float2(cen.x / 512, 1 - cen.y / 512), holeMip).r > 0.5) discard;
    }
    float3 base = float3(0.36, 0.31, 0.24);
    if (nBaked != 0) {
        float u = i.L.x / 512, v = 1 - i.L.y / 512;
        uint cell = min(min(4 * uint(clamp(4 * (1 - v), 0, 3.999)) + uint(clamp(4 * u, 0, 3.999)), nBaked), 15);
        uint ix = (baked[cell >> 2] >> (8 * (cell & 3))) & 0xFF;
        if (ix != 0) {
            float3 t = tex[NonUniformResourceIndex(ix)].Sample(wrapAniso, float2(frac(4 * u), frac(4 * v))).rgb;
            base = ((srgbMask >> cell) & 1) != 0 ? pow(saturate(t), 1.0 / 2.2) : t;
        }
    }
    float3 n = normalize(i.normal + 1e-6);
    float3 c;
    if (sunCol.w > 0.5) {
        float3 d = normalize(sunDir.xyz + 1e-6);
        c = base * (sunDir.w + (1.0 - sunDir.w) * abs(dot(n, d)) * saturate(sunCol.rgb));
    } else {
        c = base * (0.3 + 0.7 * abs(dot(n, normalize(float3(0.4, -0.3, 0.866)))));
    }
    return float4(fog.w > 0 ? lerp(c, fog.rgb, 1 - exp(-i.pos.w * fog.w)) : c, 1);
}
"#;

/// What a heightmap slot holds (the null-mode question: loaded or rendered).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HeightKind {
    #[default]
    NoRecord,
    /// A streamed texture: its host copy has a pack origin.
    Streamed,
    /// Loaded into title memory (no pack origin; detiled from memory).
    TitleMemory,
    /// Its resource is a render target or UAV: written on the GPU.
    GpuWritable,
    /// A view that is not a texture, or no host copy.
    Other,
}

/// The heightmap kinds of the last frame's drawn tiles, and the table's counters.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KindCounts {
    pub kinds: [u32; 5],
}

/// Our terrain pipeline, its heap and table, and the probe of heightmap slots.
pub struct TerrainPass {
    pub opts: TerrainOptions,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    heap: ID3D12DescriptorHeap,
    step: u32,
    pub table: TexTable,
    kinds: HashMap<u64, ((u32, u32), HeightKind)>,
    pub last: TerrainStats,
    pub last_kinds: KindCounts,
}

// SAFETY: D3D12 objects are free-threaded; used by its pass's owner.
unsafe impl Send for TerrainPass {}

impl TerrainPass {
    pub fn new(device: &ID3D12Device) -> std::result::Result<TerrainPass, String> {
        let e = |e: windows::core::Error| e.to_string();
        let (vs, ps) = (super::compile_source(TERRAIN_SHADER, c"vs", c"vs_5_1")?, super::compile_source(TERRAIN_SHADER, c"ps", c"ps_5_1")?);
        // SAFETY: object creation with descriptions that outlive each call.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, super::blob_bytes(&vs)).map_err(e)?;
            let state = RenderState { depth_enable: true, depth_write: true, depth_func: 5, ..Default::default() };
            let pso = super::pipeline_state_with(device, &root, &vs, &ps, &state, &[]).map_err(e)?;
            let heap: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: SLOTS, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 }).map_err(e)?;
            let step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            let opts = TexOptions { mode: TexMode::Off, slots: SLOTS, uploads: 16, upload_bytes: 16 << 20, async_reads: false, frame_ms: 0.0 };
            Ok(TerrainPass { opts: TerrainOptions::default(), root, pso, heap, step, table: TexTable::new(opts, 0), kinds: HashMap::new(), last: TerrainStats::default(), last_kinds: KindCounts::default() })
        }
    }

    /// After the frame's fence.
    pub fn finished(&mut self) {
        self.table.finished();
    }

    /// The probe of one heightmap slot (cached per generation).
    fn kind(&mut self, slot: u64) -> HeightKind {
        let (id, view) = match textures::classify(slot) {
            Slot::Placed(id, v) => (id, Some(v)),
            Slot::Shader(id, r, d) => (id, Some(xnw_gpu::d3d12x::descriptor::View::Shader(r, d))),
            Slot::Other => return HeightKind::Other,
            _ => return HeightKind::NoRecord,
        };
        if let Some((known, k)) = self.kinds.get(&slot)
            && *known == id
        {
            return *k;
        }
        let k = view.map_or(HeightKind::Other, |v| probe_view(&v));
        if self.kinds.len() > 8192 {
            self.kinds.clear();
        }
        self.kinds.insert(slot, (id, k));
        k
    }

    /// Records the frame's terrain on `l` (open, our pass's targets bound):
    /// sets its own descriptor heap (the caller sets its own again after).
    pub fn record(&mut self, device: &ID3D12Device, l: &ID3D12GraphicsCommandList, list: &FrameList, cam: &Camera) -> TerrainStats {
        let t0 = std::time::Instant::now();
        let mut st = TerrainStats::default();
        let Some(t) = list.terrain.as_deref().filter(|_| self.opts.on) else {
            self.last = st.clone();
            return st;
        };
        (st.tiles, st.count, st.cap, st.read_ms) = (t.tiles.len(), t.count, t.cap, t.read_ms);
        let campos = super::camera_position(&cam.view).map(f64::from);
        let vp = cam.view_proj_relative();
        let planned = plan(t, campos, &vp, self.opts.max_triangles, &mut st);
        // Slots: heightmaps and holes first (geometry), then baked cells nearest first.
        let checker = self.table.opts.mode == TexMode::Checker;
        let mut slots: Vec<u64> = Vec::new();
        let mut kinds = KindCounts::default();
        for p in &planned {
            let tile = &t.tiles[p.tile];
            if !checker && tile.height != 0 {
                slots.push(tile.height);
                let k = self.kind(tile.height);
                kinds.kinds[k as usize] += 1;
            }
            if !checker && tile.holes != 0 && tile.lod < 5 {
                slots.push(tile.holes);
            }
        }
        for p in &planned {
            let tile = &t.tiles[p.tile];
            if tile.n_baked != 0 {
                slots.extend(tile.baked.iter().take(tile.n_baked.min(15) as usize + 1).filter(|&&s| s != 0));
            }
        }
        let base = {
            // SAFETY: our heap.
            unsafe { self.heap.GetCPUDescriptorHandleForHeapStart() }.ptr
        };
        let step = self.step as usize;
        let cpu = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base + i as usize * step };
        self.table.prepare(device, l, &mut slots.into_iter(), &cpu);
        let mut k = [0u32; 56];
        k[..16].iter_mut().zip(&vp).for_each(|(d, s)| *d = s.to_bits());
        if let Some(s) = list.sun {
            for (j, v) in s.iter().enumerate() {
                k[44 + j] = v.to_bits();
            }
        }
        if let Some(s) = list.sky {
            for (j, v) in s.fog.iter().enumerate() {
                k[52 + j] = v.to_bits();
            }
        }
        // SAFETY: an open list and our live objects.
        unsafe {
            l.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            l.SetGraphicsRootSignature(&self.root);
            l.SetPipelineState(&self.pso);
            l.SetGraphicsRootDescriptorTable(1, self.heap.GetGPUDescriptorHandleForHeapStart());
        }
        for p in &planned {
            let tile = &t.tiles[p.tile];
            for r in 0..3 {
                for j in 0..3 {
                    k[16 + 4 * r + j] = tile.rows[r][j].to_bits();
                }
            }
            for j in 0..3 {
                k[28 + j] = p.origin[j].to_bits();
            }
            k[32] = p.quads;
            let look = |s: u64| if s == 0 { (0, false) } else { self.table.lookup(s) };
            k[33] = if checker { 0 } else { look(tile.height).0.min(SLOTS - 1) };
            k[34] = if checker || tile.lod >= 5 { 0 } else { look(tile.holes).0.min(SLOTS - 1) };
            k[35] = tile.lod.min(tile.hole_mip).min(15);
            let (mut packed, mut srgb, mut cells) = ([0u32; 4], 0u32, 0usize);
            if tile.n_baked != 0 {
                for c in 0..16 {
                    let (i, s) = look(tile.baked[c]);
                    let i = i.min(SLOTS - 1);
                    packed[c / 4] |= i << (8 * (c % 4));
                    srgb |= u32::from(s && i != 0) << c;
                    cells += usize::from(i != 0);
                }
            }
            k[36..40].copy_from_slice(&packed);
            k[40] = tile.n_baked;
            k[41] = srgb;
            st.heights += usize::from(k[33] != 0);
            st.holes += usize::from(k[34] != 0);
            st.baked += usize::from(cells > 0);
            st.baked_cells += cells;
            // SAFETY: as above.
            unsafe {
                l.SetGraphicsRoot32BitConstants(0, 56, k.as_ptr().cast(), 0);
                l.DrawInstanced(6 * p.quads * p.quads, 1, 0, 0);
            }
            st.drawn += 1;
        }
        st.record_ms = t0.elapsed().as_secs_f64() * 1e3;
        self.last = st.clone();
        self.last_kinds = kinds;
        st
    }

    /// The two log lines of the last frame (the pass logs them every 60 frames).
    pub fn log_lines(&mut self) -> [String; 2] {
        let s = &self.last;
        let ts = std::mem::take(&mut self.table.stats);
        let k = self.last_kinds.kinds;
        [
            format!(
                "world render: terrain: tiles {} (per LOD {:?}), patches {}, triangles {}, heightmaps resident {} of {}, baked resident {} ({} cells), holes {}; read {} of {} (cap {}), culled {}, coarsened {}, over budget {}, identity rows {}; read {:.2} ms, record {:.2} ms",
                s.drawn, s.per_lod, s.patches, s.triangles, s.heights, s.drawn, s.baked, s.baked_cells, s.holes, s.tiles, s.count, s.cap, s.culled, s.coarsened, s.over_budget, s.identity, s.read_ms, s.record_ms
            ),
            format!(
                "world render: terrain: heightmaps: streamed {}, title memory {}, GPU-writable {}, other {}, no record {}; textures: uploads {} ({:.1} MiB), from pack {}, pages {}, memory {}, hashed memory {}, unknown {}, waiting {}; slots placed {}, shader {}, stale {}, no record {}, not 2D {}, table full {}",
                k[HeightKind::Streamed as usize], k[HeightKind::TitleMemory as usize], k[HeightKind::GpuWritable as usize], k[HeightKind::Other as usize], k[HeightKind::NoRecord as usize],
                ts.uploads, ts.upload_bytes as f64 / (1 << 20) as f64, ts.sources[0], ts.sources[1], ts.sources[2], ts.sources[3], ts.sources[4], ts.waiting, ts.placed, ts.shader, ts.stale, ts.no_record, ts.not_2d, ts.full
            ),
        ]
    }
}

/// Loaded or rendered: the host copy's origin and its resource flags.
fn probe_view(view: &xnw_gpu::d3d12x::descriptor::View) -> HeightKind {
    use xnw_gpu::d3d12x::descriptor::View;
    use xnw_gpu::d3d12x::{object, resource};
    let (this, _held_placed, _held_live) = match view {
        View::Placed(..) => {
            let Some((v, held)) = xnw_gpu::d3d12x::descriptor::host::placed_texture(view) else { return HeightKind::Other };
            let View::Shader(this, _) = v else { return HeightKind::Other };
            (this, Some(held), None)
        }
        View::Shader(r, _) => {
            let Some(h) = resource::hold_live(*r as object::This) else { return HeightKind::NoRecord };
            (*r, None, Some(h))
        }
        _ => return HeightKind::Other,
    };
    let Some(copy): Option<Arc<resource::HostTexture>> = resource::host_copy(this as object::This) else { return HeightKind::Other };
    // SAFETY: a live resource.
    let d = unsafe { copy.texture.GetDesc() };
    if d.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) || d.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) || xnw_gpu::d3d12x::texture::is_host_written(this as object::This) {
        HeightKind::GpuWritable
    } else if copy.origin.get().is_some() {
        HeightKind::Streamed
    } else {
        HeightKind::TitleMemory
    }
}

/// A synthetic record (tests and the offline fixture): rows identity,
/// origin, the bindless indices, lod, nBaked.
pub fn synthetic_record(origin: [f32; 3], height: u32, holes: u32, baked: &[u32], lod: u32) -> Vec<u8> {
    let mut r = vec![0u8; TILE_BYTES];
    let mut put = |o: usize, v: u32| r[o..o + 4].copy_from_slice(&v.to_le_bytes());
    for k in 0..3 {
        put(16 * k + 4 * k, 1f32.to_bits());
    }
    for (k, v) in origin.iter().enumerate() {
        put(0x30 + 4 * k, v.to_bits());
    }
    put(0x3C, 1f32.to_bits());
    for (c, b) in baked.iter().take(16).enumerate() {
        put(0x40 + 4 * c, *b);
    }
    put(0x148, height);
    put(0x158, holes);
    put(0x168, baked.len().min(16) as u32);
    put(0x16C, lod);
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id_rows() -> [[f32; 4]; 3] {
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    }

    #[test]
    fn records_parse_into_tiles_and_slots() {
        let mut raw = synthetic_record([1024.0, -512.0, 3.0], 7, 9, &[11, 12], 2);
        raw.extend(vec![0xFFu8; TILE_BYTES]); // NaN rows: identity
        raw.extend(vec![0u8; 100]); // a partial record is not one
        let t = parse(8, 2, raw, 0x1000_0000);
        assert_eq!(t.tiles.len(), 2);
        let a = &t.tiles[0];
        assert_eq!(a.rows[..3], id_rows());
        assert_eq!(a.rows[3][..3], [1024.0, -512.0, 3.0]);
        assert_eq!((a.height, a.holes, a.baked[0], a.baked[1], a.baked[2], a.n_baked, a.lod), (0x1000_0000 + 7 * 32, 0x1000_0000 + 9 * 32, 0x1000_0000 + 11 * 32, 0x1000_0000 + 12 * 32, 0x1000_0000, 2, 2));
        assert!(!a.identity && t.tiles[1].identity);
        assert_eq!(t.tiles[1].rows[..3], id_rows());
        assert_eq!(t.tiles[1].rows[3], [0.0; 4]);
        assert_eq!(slot_of(0, 5), 0, "no heap base: no slot");
        assert_eq!(slot_of(0x1000, MAX_BINDLESS), 0, "out of range");
    }

    #[test]
    fn lod_quads_and_switches() {
        assert_eq!([0, 1, 2, 3, 4, 5, 6, 40].map(quads_for), [64, 64, 64, 32, 16, 8, 8, 8]);
        assert!(!TerrainOptions::default().on, "off by default");
        assert_eq!(parse_screen("screen"), TerrainOptions::default(), "off unless asked");
        assert!(parse_screen("screen,terrain=on").on);
        assert!(!parse_screen("screen,terrain=off").on);
        assert_eq!(parse_screen("screen,terraintri=300").max_triangles, 300_000);
    }

    /// Camera at the origin looking along -z (identity view), the test projection.
    fn vp() -> super::super::Mat {
        let mut p = [0f32; 16];
        (p[0], p[5], p[11], p[14]) = (1.0, 1.0, -1.0, 0.1);
        p
    }

    #[test]
    fn plan_culls_orders_and_caps() {
        // Tiles below the camera (looking down -z): one under it, one far
        // away ahead (-z is "ahead": world y is up the screen), one behind (+z).
        let raw = [synthetic_record([-256.0, -256.0, -1000.0], 1, 0, &[], 0), synthetic_record([-256.0, -256.0, -20000.0], 1, 0, &[], 4), synthetic_record([-256.0, -256.0, 5000.0], 1, 0, &[], 0)].concat();
        let t = parse(3, 3, raw, 0x1000);
        let mut st = TerrainStats::default();
        let p = plan(&t, [0.0; 3], &vp(), MAX_TRIANGLES, &mut st);
        assert_eq!(p.iter().map(|p| p.tile).collect::<Vec<_>>(), [0, 1], "nearest first; the one behind culled");
        assert_eq!((st.culled, st.triangles), (1, 2 * 64 * 64 + 2 * 16 * 16));
        assert_eq!(st.per_lod[..5], [2, 0, 0, 0, 1]);
        assert_eq!(p[0].origin, [-256.0, -256.0, -1000.0]);
        // A cap of 3,000 triangles: the first is coarsened to 32 (2,048), the second to 16 (512): 2,560.
        let mut st = TerrainStats::default();
        let p = plan(&t, [0.0; 3], &vp(), 3000, &mut st);
        assert_eq!(p.iter().map(|p| p.quads).collect::<Vec<_>>(), [32, 16]);
        assert_eq!((st.coarsened, st.over_budget), (1, 0));
        let mut st = TerrainStats::default();
        let p = plan(&t, [0.0; 3], &vp(), 40, &mut st);
        assert_eq!((p.len(), st.over_budget), (1, 1), "4 quads (32 triangles) fit once");
    }

    /// A title texture placed in memory (64x64 RGBA8, linear) and its view
    /// written to a descriptor slot as the title does (the albedo recipe); both
    /// leaked (a freed address could come back in another test).
    fn placed_texture(texel: impl Fn(usize, usize) -> [u8; 4]) -> u64 {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;
        let mem = Box::leak(Box::new(xnw_gpu::d3d12x::resource::Memory::zeroed(64 << 10, 64 << 10).unwrap()));
        // SAFETY: our own 64 KiB.
        let m = unsafe { std::slice::from_raw_parts_mut(mem.address() as *mut u8, 64 << 10) };
        for y in 0..64 {
            for x in 0..64 {
                m[4 * (64 * y + x)..4 * (64 * y + x) + 4].copy_from_slice(&texel(x, y));
            }
        }
        let resource: [u32; 14] = [3, 0, 0, 0, 64, 0, 64, 0x0001_0001, DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32, 1, 0, 0x100, 0, 0];
        let mut desc = [0u32; 16];
        desc[..6].copy_from_slice(&[DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32, 4, 0x1688, 0, 0, 1]);
        (desc[10], desc[11]) = (mem.address() as u32, (mem.address() >> 32) as u32);
        let slot = Box::leak(Box::new([0u64; 8])).as_ptr() as u64;
        xnw_gpu::d3d12x::descriptor::create_placed_srv(std::ptr::null(), &resource, &desc, slot);
        slot
    }

    /// Offline proof (no dump holds the tile array yet): one tile at
    /// (1000, 2000, 50) seen from 1500 m straight above its centre, on
    /// WARP, through the live texture path (descriptor slot -> host copy ->
    /// upload on our list). Heightmap: a ramp along x, s = (128 + column) /
    /// 255, so h = 1600 (s - 0.5) runs from 3 m to 398 m: the raised east
    /// edge lies 8 px further out than a flat tile's would. A hole mask over
    /// L in [256, 384) x [0, 128) leaves the clear colour; baked cell 0 is red.
    #[test]
    fn heightmap_tile_on_warp() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _gpu = GPU.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        static ONCE: std::sync::Once = std::sync::Once::new();
        // SAFETY: inside call_once, before this test makes a device.
        ONCE.call_once(|| unsafe { std::env::set_var("XNW_ADAPTER", "warp") });
        let Some(host) = xnw_gpu::d3d12x::host::host_device() else { return };
        let height = placed_texture(|x, _| [128 + x as u8, 0, 0, 255]);
        let holes = placed_texture(|x, y| [if (32..48).contains(&x) && y >= 48 { 255 } else { 0 }, 0, 0, 255]);
        let red = placed_texture(|_, _| [255, 0, 0, 255]);
        let origin = [1000.0f32, 2000.0, 50.0];
        let mut rows = [[0f32; 4]; 4];
        for (k, row) in rows.iter_mut().take(3).enumerate() {
            row[k] = 1.0;
        }
        rows[3][..3].copy_from_slice(&origin);
        let mut baked = [0u64; 16];
        baked[0] = red;
        let tile = TerrainTile { rows, lod: 0, hole_mip: 0, height, holes, baked, n_baked: 1, identity: false };
        let terrain = Terrain { cap: 1, count: 1, heap_base: 0, tiles: vec![tile], raw: Vec::new(), read_ms: 0.0 };
        let cam = [origin[0] + 256.0, origin[1] + 256.0, origin[2] + 1500.0];
        let mut view = [0f32; 16];
        (view[0], view[5], view[10], view[15]) = (1.0, 1.0, 1.0, 1.0);
        (view[12], view[13], view[14]) = (-cam[0], -cam[1], -cam[2]);
        let mut proj = [0f32; 16];
        (proj[0], proj[5], proj[11], proj[14]) = (1.0, 1.0, -1.0, 0.5);
        let list = FrameList { frame: 1, cameras: vec![Camera { viewport: 1, view, proj }], main: Some(1), terrain: Some(Arc::new(terrain)), ..Default::default() };
        let mut pass = super::super::Pass::new(host, 256, 256).unwrap();
        pass.terrain.opts.on = true;
        pass.terrain.table.opts.mode = TexMode::Live;
        let st = pass.draw(&list, &mut |_| None, true).unwrap().terrain;
        assert_eq!((st.drawn, st.heights, st.holes, st.baked, st.triangles), (1, 1, 1, 1, 2 * 64 * 64), "{st:?}");
        assert_eq!(pass.terrain.last_kinds.kinds[HeightKind::TitleMemory as usize], 1, "a placed texture in title memory");
        let px = pass.read_rgba().unwrap();
        let at = |col: usize, row: usize| -> [u8; 4] { px[4 * (256 * row + col)..4 * (256 * row + col) + 4].try_into().unwrap() };
        // Where a world point lands (top-down, f = 1): (column, row).
        let screen = |lx: f32, ly: f32, h: f32| {
            let d = cam[2] - (origin[2] + h);
            (((origin[0] + lx - cam[0]) / d + 1.0) * 128.0, (1.0 - (origin[1] + ly - cam[1]) / d) * 128.0)
        };
        let ramp = |lx: f32| 1600.0 * ((128.0 + (lx / 8.0).min(63.0)) / 255.0 - 0.5);
        let terrain_px = |c: [u8; 4]| c[0] > 40;
        let clear_px = |c: [u8; 4]| c[0] < 30 && c[1] < 30;
        let (west, row) = screen(0.0, 256.0, ramp(0.0));
        let (east, _) = screen(512.0, 256.0, ramp(512.0));
        let (flat_east, _) = screen(512.0, 256.0, 0.0);
        assert!(east - flat_east > 6.0, "the ramp moves the east edge out: {east} vs flat {flat_east}");
        let r = row as usize;
        assert!(clear_px(at(west as usize - 3, r)) && terrain_px(at(west as usize + 3, r)), "west edge at {west}: {:?} {:?}", at(west as usize - 3, r), at(west as usize + 3, r));
        assert!(terrain_px(at(east as usize - 3, r)) && clear_px(at(east as usize + 3, r)), "east edge at {east} (flat would be {flat_east}): {:?} {:?}", at(east as usize - 3, r), at(east as usize + 3, r));
        assert!(terrain_px(at(flat_east as usize + 3, r)), "terrain past the flat edge: the heights are applied");
        // The hole: the clear colour; beside it (L.y 192) terrain.
        let (hc, hr) = screen(320.0, 64.0, ramp(320.0));
        assert!(clear_px(at(hc as usize, hr as usize)), "hole at ({hc}, {hr}): {:?}", at(hc as usize, hr as usize));
        let (nc, nr) = screen(320.0, 192.0, ramp(320.0));
        assert!(terrain_px(at(nc as usize, nr as usize)), "no hole at ({nc}, {nr})");
        // Baked cell 0 (L < 128 both ways) is red; cell 1 has none: earth.
        let (bc, br) = screen(64.0, 64.0, ramp(64.0));
        let c = at(bc as usize, br as usize);
        assert!(c[0] > 70 && c[1] < 20 && c[2] < 20, "baked red at ({bc}, {br}): {c:?}");
        let (ec, er) = screen(192.0, 320.0, ramp(192.0));
        let c = at(ec as usize, er as usize);
        assert!(c[0] > 40 && c[1] > 30 && c[0] > c[2], "earth tone at ({ec}, {er}): {c:?}");
        // terrain=off: nothing drawn.
        pass.terrain.opts.on = false;
        let st = pass.draw(&list, &mut |_| None, true).unwrap().terrain;
        assert_eq!(st.drawn, 0);
        let px = pass.read_rgba().unwrap();
        let i = 4 * (256 * r + 128);
        assert!(clear_px(px[i..i + 4].try_into().unwrap()), "terrain=off: the clear colour");
    }

    #[test]
    fn terrain_shader_compiles() {
        for (entry, target) in [(c"vs", c"vs_5_1"), (c"ps", c"ps_5_1")] {
            if let Err(e) = super::super::compile_source(TERRAIN_SHADER, entry, target) {
                panic!("{e}");
            }
        }
    }
}
