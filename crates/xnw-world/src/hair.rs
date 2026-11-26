//! Strand hair (HAIR1): the title's strand draws as data, and the strand
//! program's vertex rule on the CPU, read from its native code.
//!
//! The title draws ped hair as strands: plain `DrawInstanced` calls (sink op
//! 4) of `VS_HairStrandGeometry_8e7634e5` (container f76935e0, the main view;
//! the shadow cascades use 58f72c1e) with `PS_HairStrandVBuffer` (7e040f4d),
//! one instance per strand, two vertices per point (a ribbon), then the
//! `hair_resolve` compute pass shades the visibility buffer. None of them is a
//! hook draw, so no list row exists for them.
//!
//! What the vertex program reads (f76935e0, `out\shaders\disasm`, V):
//!
//! | where | what |
//! |---|---|
//! | root 2 constants | LOD, a width divisor scale (1.0), f16 minimum width per metre in the high half, group record index |
//! | table 8 slot 21 (+0x2a0) | group records, 336 B each ([`Group`]) |
//! | table 8 slot 18 (+0x240) | worlds, 176 B each: a 3x4 at +0x30 (camera-relative), width scale at +0x9c |
//! | table 8 slot 15 (+0x1e0) | the points: the 47 MB frame pool, 16 B a point |
//! | table 8 slot 29 (+0x3a0) | width profile image, 64 x 320, 15 steps along a strand |
//! | table 9 slot 1 (+0x20) | the view block: forward at +0xa0, camera-relative view-projection at +0x160 |
//!
//! The points are written on the GPU by the `smh` compute chain
//! (`smh_pose_N:CS_PreSkinSimGuides` and friends, the simulation, then
//! `smh_gen_N:CS_InterpolateStrands` into the pool), so with the title's GPU
//! work nulled they are zero, and so is the profile image (pk3 f1241, pk4
//! f887, pk5 f975, pk6 f1212: 0 non-zero bytes in the 16 MB captured), and
//! the program culls every strand itself (the last point's arc length at most
//! 1e-4, f76935e0 0x2d0-0x308, 0x5fc-0x614). Hair in our pass is the title's
//! own chain through the bridge (the user's direction, 2026-11-25): the smh
//! compute, then this vertex program, per draw; this module is its data.

/// The main-view strand vertex program (container hash).
pub const STRAND_VS: u128 = 0xf76935e03c6e77f2ab85441f194e4240;
/// Its pixel program (the visibility buffer).
pub const STRAND_PS: u128 = 0x7e040f4d1d02c04db8e7fb072ebafe27;
/// The shadow-cascade strand vertex program (no pixel program).
pub const STRAND_SHADOW_VS: u128 = 0x58f72c1e72ee590a7d84ce1cf6538a49;
/// `smh_pose_1:CS_PreSkinSimGuides` (HAIR3: [`sim_guide`]).
pub const SIM_GUIDES: u128 = 0xed403b6d88dab0c1689e31f52fa043cd;
/// Frame table (root 0) slots the main-view program reads.
pub const POINTS_SLOT: u32 = 15;
pub const WORLDS_SLOT: u32 = 18;
pub const GROUPS_SLOT: u32 = 21;
pub const PROFILE_SLOT: u32 = 29;
/// Root 1 slot of the view block.
pub const VIEW_SLOT: u32 = 1;
pub const GROUP_BYTES: usize = 336;
pub const WORLD_BYTES: usize = 176;
pub const POINT_BYTES: usize = 16;

/// `hair=on` in an `XNW_WORLD_RENDER` value (default off, HAIR2): the
/// title's strand chain and strand draws. The replay runs them
/// (`world-render`); the live path is a stub while the card is frozen.
pub fn switch_on(v: &str) -> bool {
    v.split(',').map(str::trim).filter_map(|p| p.strip_prefix("hair=")).last().is_some_and(|s| s != "off")
}

/// IEEE half to f32.
pub fn f16(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    let mag = match e {
        0 => m as f32 * (1.0 / 16_777_216.0),
        31 => {
            if m == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => f32::from_bits((e + 112) << 23 | m << 13),
    };
    if s == 1 { -mag } else { mag }
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    b.get(o..o + 4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).unwrap_or(0)
}

fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_bits(u32_at(b, o))
}

/// One strand group record (336 B), the fields the vertex program reads.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Group {
    /// +0x50 (bit 0: the arc length is scaled by `t_scale`, else by the
    /// strand's own length).
    pub flags: u32,
    /// +0x54 bits 16-23: points a strand at LOD 0; bits 24-31 at LOD > 0.
    pub points: [u32; 2],
    /// +0x58: the world record.
    pub world: u32,
    /// +0x5c: f16 width divisor (low) and arc-length scale (high).
    pub width_div: f32,
    pub t_scale: f32,
    /// +0x60/+0x64/+0x68: per axis the box corner (f16 low) and extent (f16 high).
    pub lo: [f32; 3],
    pub ext: [f32; 3],
    /// +0x6c: LOD 0's first byte in the pool; +0x7c/+0x80 the LOD > 0 base and step.
    pub base: u32,
    pub lod_base: u32,
    pub lod_step: u32,
    /// +0x84 low half: the width profile's image row.
    pub profile_row: u32,
    /// +0x104 + +0x10c: strands (the next group's base is this one's plus
    /// strands x points x 16 B: V on pk4 f887 groups 0 -> 1 -> 2).
    pub strands: u32,
}

impl Group {
    pub fn parse(r: &[u8]) -> Group {
        let w = |o| u32_at(r, o);
        let pair = |o| (f16(w(o) as u16), f16((w(o) >> 16) as u16));
        let (width_div, t_scale) = pair(0x5c);
        let (x, y, z) = (pair(0x60), pair(0x64), pair(0x68));
        Group {
            flags: w(0x50),
            points: [(w(0x54) >> 16) & 0xff, w(0x54) >> 24],
            world: w(0x58),
            width_div,
            t_scale,
            lo: [x.0, y.0, z.0],
            ext: [x.1, y.1, z.1],
            base: w(0x6c),
            lod_base: w(0x7c),
            lod_step: w(0x80),
            profile_row: w(0x84) & 0xffff,
            strands: w(0x104).wrapping_add(w(0x10c)),
        }
    }

    /// (first byte of strand 0 in the pool, points a strand) at `lod`.
    pub fn layout(&self, lod: u32) -> (u64, u32) {
        if lod == 0 {
            (u64::from(self.base), self.points[0])
        } else {
            (u64::from(self.lod_base) + u64::from(self.lod_step) * u64::from(lod - 1), self.points[1])
        }
    }

    /// Point bytes of strand `s` at `lod`: (first byte, length).
    pub fn strand_bytes(&self, lod: u32, s: u32) -> (u64, u64) {
        let (base, n) = self.layout(lod);
        let len = u64::from(n) * POINT_BYTES as u64;
        (base + u64::from(s) * len, len)
    }

    /// A point's position in the group's space: the box corner plus the
    /// three u16 (low halves of dwords 0-2) over 65535 times the extent.
    pub fn position(&self, p: &[u8]) -> [f32; 3] {
        std::array::from_fn(|k| self.lo[k] + (u32_at(p, 4 * k) & 0xffff) as f32 * (1.0 / 65535.0) * self.ext[k])
    }
}

/// One strand draw: its root constants and its world record.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Draw {
    pub lod: u32,
    /// Root dword 1 (1.0 in every capture): multiplies `Group::width_div`.
    pub div_scale: f32,
    /// Root dword 2 high half (f16) / 256: minimum width per metre of distance.
    pub min_width: f32,
    pub group: u32,
    /// Vertices a strand (2 a point).
    pub vertices: u32,
    /// World rows (3x4, camera-relative translation in .w).
    pub world: [f32; 12],
    /// World +0x9c: the width scale.
    pub width_scale: f32,
}

impl Draw {
    /// From root 2's four dwords, the vertex count and the world record.
    pub fn new(roots: &[u32], vertices: u32, world: &[u8]) -> Draw {
        let r = |i: usize| roots.get(i).copied().unwrap_or(0);
        Draw {
            lod: r(0),
            div_scale: f32::from_bits(r(1)),
            min_width: f16((r(2) >> 16) as u16) / 256.0,
            group: r(3),
            vertices,
            world: std::array::from_fn(|i| f32_at(world, 0x30 + 4 * i)),
            width_scale: f32_at(world, 0x9c),
        }
    }
}

/// HAIR3: `CS_PreSkinSimGuides` (ed403b6d, `smh_pose_1`; 0479e16e and
/// 243b071d the same code) on the CPU, read from its native code (V). One
/// lane per guide `g` of record `rec`'s asset (heap index rec+0x120; guides
/// asset+0x58 low half, points a guide asset+0x5c low byte):
/// - 8 bone bytes at asset[+0x10] + 8g, 8 weight bytes at asset[+0x14] + 8g,
///   weights w = b/255 / (sum/255 + 1e-6) (0x120-0x2a4);
/// - the guide's root rest point: f16 x, y, z at asset[+0x44] + 8 P g;
/// - per bone j: palette bone = remap[rec+0xe4 + bone_j] & 0xfff (16-bit,
///   `buffer_load_format_x`), M += w_j palette[rec+0xe8 + bone], M' += w_j
///   palette[rec+0xec + bone] (3x4 row-major, 48 B; loop 0x2d8-0x4ac, two
///   bones a turn: j and j+4);
/// - out = M p - (R (M' p) + T), R|T the record's own 3x4 at +0x00
///   (0x4b0-0x5a4); all three zero unless all three are finite;
/// - stored as 3 f32 at pool[rec+0x144] + 12 g.
///
/// So the output is a displacement, not a position: with the same palette
/// at +0xe8 and +0xec and R = I it is -T for every guide whatever the bones.
/// `remap` holds u16 elements. `None` past the guide count.
pub fn sim_guide(rec: &[u8], asset: &[u8], remap: &[u8], palette: &[u8], g: u32) -> Option<[f32; 3]> {
    if g >= u32_at(asset, 0x58) & 0xffff {
        return None;
    }
    let points = u32_at(asset, 0x5c) & 0xff;
    let (bo, wo, ro) = (u32_at(asset, 0x10) as usize, u32_at(asset, 0x14) as usize, u32_at(asset, 0x44) as usize);
    let byte = |o: usize| asset.get(o).copied().unwrap_or(0);
    let half = |o: usize| f16(asset.get(o..o + 2).map_or(0, |b| u16::from_le_bytes([b[0], b[1]])));
    let g8 = 8 * g as usize;
    let sum: f32 = (0..8).map(|j| f32::from(byte(wo + g8 + j))).sum();
    let norm = 1.0 / (sum * (1.0 / 255.0) + 1e-6);
    let at = ro + 8 * (g * points) as usize;
    let p = [half(at), half(at + 2), half(at + 4)];
    let (mut m, mut mp) = ([0f32; 12], [0f32; 12]);
    let (remap_at, base, prev) = (u32_at(rec, 0xe4) as usize, u32_at(rec, 0xe8) as usize, u32_at(rec, 0xec) as usize);
    for j in 0..8 {
        let w = f32::from(byte(wo + g8 + j)) * (1.0 / 255.0) * norm;
        let e = 2 * (remap_at + byte(bo + g8 + j) as usize);
        let bone = (remap.get(e..e + 2).map_or(0, |b| u16::from_le_bytes([b[0], b[1]])) & 0xfff) as usize;
        for k in 0..12 {
            m[k] += w * f32_at(palette, 48 * (base + bone) + 4 * k);
            mp[k] += w * f32_at(palette, 48 * (prev + bone) + 4 * k);
        }
    }
    let apply = |m: &[f32; 12], r: usize| m[4 * r] * p[0] + m[4 * r + 1] * p[1] + m[4 * r + 2] * p[2] + m[4 * r + 3];
    let (now, before) = ([0, 1, 2].map(|r| apply(&m, r)), [0, 1, 2].map(|r| apply(&mp, r)));
    let out: [f32; 3] = std::array::from_fn(|r| {
        let row = |k: usize| f32_at(rec, 16 * r + 4 * k);
        now[r] - row(3) - (row(0) * before[0] + row(1) * before[1] + row(2) * before[2])
    });
    Some(if out.iter().all(|v| v.is_finite()) { out } else { [0.0; 3] })
}

/// A frame's strand draws, for the census line.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Census {
    /// Main-view strand draws, and shadow-cascade ones.
    pub draws: usize,
    pub shadow_draws: usize,
    /// Strands of the main-view draws (by their groups' counts).
    pub strands: u64,
    /// Strands whose point bytes lie in the captured pool, and of those
    /// with a non-zero byte.
    pub captured: u64,
    pub with_points: u64,
    /// Draws whose group or world record is missing.
    pub unreadable: usize,
}

impl Census {
    pub fn line(&self) -> String {
        format!(
            "hair: strand draws {} (shadow {}), strands {}, in the captured pool {}, with points {}, draws without records {}",
            self.draws, self.shadow_draws, self.strands, self.captured, self.with_points, self.unreadable
        )
    }
}

/// HAIR6: the strand pixels of a frame as our pass draws them over its
/// picture: per pixel (x, y, the nearest strand's depth, 0), at the pass's
/// size. The live path makes them from the title's own strand chain and
/// strand draws (`apps/xnw` `worldprobe/hairlive.rs`), as the replay does.
pub type Pixels = Vec<[f32; 4]>;

/// The strand pixel shader: one quad per pixel at its depth, a flat hair
/// brown (the replay's overlay colour, 96/64/40), depth tested
/// GREATER_EQUAL against our pass's depth without writing it (reverse
/// depth: the title's own strand test, HAIR4).
pub const HAIR_SHADER: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0)"
cbuffer Size : register(b0) { float2 size; float2 pad; };
StructuredBuffer<float4> pixels : register(t0);
[RootSignature(RS)]
float4 vs(uint v : SV_VertexID) : SV_Position {
    static const float2 corner[6] = { float2(0, 0), float2(1, 0), float2(1, 1), float2(0, 0), float2(1, 1), float2(0, 1) };
    float4 p = pixels[v / 6];
    float2 q = p.xy + corner[v % 6];
    return float4(q.x / size.x * 2 - 1, 1 - q.y / size.y * 2, p.z, 1);
}
[RootSignature(RS)]
float4 ps() : SV_Target { return float4(96.0 / 255, 64.0 / 255, 40.0 / 255, 1); }
"#;

/// Our strand pixel pipeline and the upload buffers of the frames in flight.
pub struct HairPass {
    root: windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
    pso: windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
    /// Upload buffers of the last frames (the oldest goes when a fourth comes).
    bufs: std::collections::VecDeque<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    /// Pixels drawn by the last record.
    pub drawn: usize,
}

// SAFETY: D3D12 objects are free-threaded; used by its pass's owner.
unsafe impl Send for HairPass {}

impl HairPass {
    pub fn new(device: &windows::Win32::Graphics::Direct3D12::ID3D12Device) -> Result<HairPass, String> {
        let e = |e: windows::core::Error| e.to_string();
        let (vs, ps) = (super::compile_source(HAIR_SHADER, c"vs", c"vs_5_1")?, super::compile_source(HAIR_SHADER, c"ps", c"ps_5_1")?);
        // SAFETY: object creation with descriptions that outlive each call.
        unsafe {
            let root: windows::Win32::Graphics::Direct3D12::ID3D12RootSignature = device.CreateRootSignature(0, super::blob_bytes(&vs)).map_err(e)?;
            // Depth GREATER_EQUAL (7), no write.
            let state = super::RenderState { depth_enable: true, depth_write: false, depth_func: 7, ..Default::default() };
            let pso = super::pipeline_state_with(device, &root, &vs, &ps, &state, &[]).map_err(e)?;
            Ok(HairPass { root, pso, bufs: std::collections::VecDeque::new(), drawn: 0 })
        }
    }

    /// Records `pixels` on `l` (open, our colour and depth targets bound).
    pub fn record(&mut self, device: &windows::Win32::Graphics::Direct3D12::ID3D12Device, l: &windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList, pixels: &[[f32; 4]], size: (u32, u32)) -> Result<(), String> {
        use windows::Win32::Graphics::Direct3D12::*;
        self.drawn = 0;
        if pixels.is_empty() {
            return Ok(());
        }
        let bytes = (pixels.len() * 16) as u64;
        let buf = super::bridge::buffer(device, bytes.next_multiple_of(256), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
        // SAFETY: a mapped upload buffer of `bytes`; an open list and our live objects.
        unsafe {
            let mut to = std::ptr::null_mut();
            buf.Map(0, None, Some(&mut to)).map_err(|e| e.to_string())?;
            std::ptr::copy_nonoverlapping(pixels.as_ptr().cast::<u8>(), to.cast::<u8>(), bytes as usize);
            buf.Unmap(0, None);
            let k = [(size.0 as f32).to_bits(), (size.1 as f32).to_bits(), 0, 0];
            l.SetGraphicsRootSignature(&self.root);
            l.SetPipelineState(&self.pso);
            l.SetGraphicsRoot32BitConstants(0, 4, k.as_ptr().cast(), 0);
            l.SetGraphicsRootShaderResourceView(1, buf.GetGPUVirtualAddress());
            l.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            l.DrawInstanced(6 * pixels.len() as u32, 1, 0, 0);
        }
        self.bufs.push_back(buf);
        while self.bufs.len() > 3 {
            self.bufs.pop_front();
        }
        self.drawn = pixels.len();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(x: f32) -> u16 {
        // Exact for the values below (small integers and halves).
        let b = x.to_bits();
        if x == 0.0 {
            return 0;
        }
        (((b >> 16) & 0x8000) | ((((b >> 23) & 0xff) - 112) << 10) | ((b >> 13) & 0x3ff)) as u16
    }

    fn group_bytes() -> Vec<u8> {
        let mut r = vec![0u8; GROUP_BYTES];
        let mut put = |o: usize, v: u32| r[o..o + 4].copy_from_slice(&v.to_le_bytes());
        put(0x50, 0);
        put(0x54, 3 << 16 | 2 << 24);
        put(0x58, 1);
        put(0x5c, u32::from(h(0.5)) | u32::from(h(2.0)) << 16);
        for k in 0..3 {
            put(0x60 + 4 * k, u32::from(h(-1.0)) | u32::from(h(2.0)) << 16);
        }
        put(0x6c, 0x100);
        put(0x7c, 0x1000);
        put(0x80, 0x200);
        put(0x104, 3);
        put(0x10c, 2);
        r
    }

    /// A made-up guide record for [`sim_guide`]: one guide of 2 points,
    /// bones (local 0 weight 3, local 1 weight 1), the remap shifting local
    /// bones by 5, palettes at bone 10 (now) and 20 (before), R a turn of
    /// 90 degrees about z and T (1, 2, 3).
    pub(crate) fn guide_case() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        let put = |b: &mut Vec<u8>, o: usize, v: u32| b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        let mut rec = vec![0u8; GROUP_BYTES];
        for (k, v) in [0.0f32, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0].iter().enumerate() {
            put(&mut rec, 4 * k, v.to_bits());
        }
        put(&mut rec, 0xe4, 1);
        put(&mut rec, 0xe8, 10);
        put(&mut rec, 0xec, 20);
        let mut asset = vec![0u8; 0x200];
        put(&mut asset, 0x10, 0x100);
        put(&mut asset, 0x14, 0x140);
        put(&mut asset, 0x44, 0x180);
        put(&mut asset, 0x58, 1);
        put(&mut asset, 0x5c, 2);
        asset[0x101] = 1;
        asset[0x140] = 3;
        asset[0x141] = 1;
        for (k, v) in [1.0f32, 2.0, 0.5].iter().enumerate() {
            asset[0x180 + 2 * k..0x182 + 2 * k].copy_from_slice(&h(*v).to_le_bytes());
        }
        // Remap element 1 + local: local 0 -> 5, local 1 -> 6.
        let remap: Vec<u8> = [0u16, 5, 6, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut palette = vec![0u8; 48 * 32];
        // Now: bone 15 a shift by (10, 0, 0), bone 16 scale 2. Before: bone 25 identity, 26 a shift by (0, 4, 0).
        for (at, m) in [
            (15, [1.0f32, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            (16, [2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]),
            (25, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            (26, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0]),
        ] {
            for (k, v) in m.iter().enumerate() {
                put(&mut palette, 48 * at + 4 * k, v.to_bits());
            }
        }
        (rec, asset, remap, palette)
    }

    /// By hand: p = (1, 2, 0.5); M p = 0.75 (11, 2, 0.5) + 0.25 (2, 4, 1)
    /// = (8.75, 2.5, 0.625); M' p = 0.75 p + 0.25 (1, 6, 0.5) = (1, 3, 0.5);
    /// R (1, 3, 0.5) = (-3, 1, 0.5); out = M p - (-3, 1, 0.5) - (1, 2, 3).
    pub(crate) const GUIDE_EXPECTED: [f32; 3] = [10.75, -0.5, -2.875];

    pub(crate) fn near3(a: [f32; 3], b: [f32; 3]) -> bool {
        (0..3).all(|k| (a[k] - b[k]).abs() < 2e-3)
    }

    #[test]
    fn sim_guide_made_up_case() {
        let (rec, asset, remap, palette) = guide_case();
        let got = sim_guide(&rec, &asset, &remap, &palette, 0).unwrap();
        assert!(near3(got, GUIDE_EXPECTED), "{got:?}");
        assert_eq!(sim_guide(&rec, &asset, &remap, &palette, 1), None, "past the guide count");
        // The same palette before and now, R = I: -T whatever the bones (HAIR2's collapse).
        let mut same = rec.clone();
        same[0xec..0xf0].copy_from_slice(&10u32.to_le_bytes());
        for (k, v) in [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0].iter().enumerate() {
            let o = 16 * (k / 3) + 4 * (k % 3);
            same[o..o + 4].copy_from_slice(&v.to_bits().to_le_bytes());
        }
        let got = sim_guide(&same, &asset, &remap, &palette, 0).unwrap();
        assert!(near3(got, [-1.0, -2.0, -3.0]), "{got:?}");
    }

    /// The translated ed403b6d on the made-up record against [`sim_guide`]
    /// (HAIR3, rung 2): record 0 in slot 21, the palette in slot 1, the pool
    /// in slot 36, the asset at heap index 3 (raw) and the remap at 4
    /// (R16_UINT), root 1 dword 3 the record. Skips without `XNW_TEST_GPU=1`,
    /// the pack, or an adapter below the program (newer WARP: `XNW_WARP`).
    #[test]
    fn translated_sim_guide_made_up_case() {
        use crate::preskin::{self as ps, Kind, View};
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let Some(c) = ps::container(std::path::Path::new(r"game\shaders.rpf"), Some(std::path::Path::new(r"out\cache\pack-index.bin")), SIM_GUIDES) else {
            eprintln!("hair: no shader pack: skipped");
            return;
        };
        let shader = ps::translate(&c).expect("ed403b6d translates");
        let wave = ps::wave_size(&c).expect("wave size");
        let (_gpu, host) = crate::tests::warp_host();
        if let Some(gap) = ps::adapter_gap(&host.caps, wave) {
            eprintln!("hair: ed403b6d not run: {gap}");
            return;
        }
        let (mut rec, asset, remap, palette) = guide_case();
        let put = |b: &mut Vec<u8>, o: usize, v: u32| b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        put(&mut rec, 0xe0, 4);
        put(&mut rec, 0x120, 3);
        put(&mut rec, 0x144, 0x40);
        let ud = shader.user_data.iter().find(|u| u.root == xnw_shader::RootType::Table && u.slot == 0).map_or(0, |u| u.index);
        let d = ps::Dispatch {
            groups: [1, 1, 1],
            tables: vec![(ud, vec![
                (1, View { kind: Kind::Structured(48), bytes: &palette }),
                (21, View { kind: Kind::Structured(GROUP_BYTES as u32), bytes: &rec }),
                (36, View { kind: Kind::RawUav { len: 0x100 }, bytes: &[] }),
            ])],
            heap: vec![(3, View { kind: Kind::Raw, bytes: &asset }), (4, View { kind: Kind::Typed { format: 57, bytes: 2 }, bytes: &remap })],
            constants: vec![(1, vec![u32::MAX, 0, 0, 0])],
            texels: Vec::new(),
        };
        let b = ps::bind(&shader, &d).expect("binding");
        let (outs, _) = ps::run(&host, &shader, wave, &b, d.groups).expect("dispatch");
        let got: [f32; 3] = std::array::from_fn(|k| f32_at(&outs[0], 0x40 + 4 * k));
        eprintln!("hair: translated ed403b6d {got:?}, CPU {:?}", sim_guide(&rec, &asset, &remap, &palette, 0));
        assert!(near3(got, GUIDE_EXPECTED), "translated {got:?} against {GUIDE_EXPECTED:?}");
        assert!(outs[0][0x4c..].iter().all(|b| *b == 0), "one guide only");
    }

    #[test]
    fn hair_switch_default_off() {
        assert!(!switch_on(""));
        assert!(!switch_on("gpucull=on,hair=off"));
        assert!(switch_on("gpucull=on, hair=on"));
        assert!(!switch_on("hair=on,hair=off"));
        assert!(!switch_on("hairx=on"));
    }

    #[test]
    fn half_decodes() {
        assert_eq!(f16(0x3c00), 1.0);
        assert_eq!(f16(0xc000), -2.0);
        assert_eq!(f16(0x3800), 0.5);
        assert_eq!(f16(0), 0.0);
    }

    #[test]
    fn group_layout_by_lod() {
        let g = Group::parse(&group_bytes());
        assert_eq!((g.points, g.world, g.strands), ([3, 2], 1, 5));
        assert_eq!((g.width_div, g.t_scale, g.lo, g.ext), (0.5, 2.0, [-1.0; 3], [2.0; 3]));
        assert_eq!(g.layout(0), (0x100, 3));
        assert_eq!(g.layout(2), (0x1200, 2));
        assert_eq!(g.strand_bytes(0, 2), (0x100 + 2 * 48, 48));
        let mut p = [0u8; 16];
        p[0..2].copy_from_slice(&65535u16.to_le_bytes());
        assert_eq!(g.position(&p), [1.0, -1.0, -1.0]);
    }

}
