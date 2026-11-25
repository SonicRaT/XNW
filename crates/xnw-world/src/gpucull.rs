//! The instance cull on the device (`gpucull=on`, PERF3): the per-instance
//! half of the hook's instance prefilter (apps/xnw worldprobe.rs
//! `prefilter_why`) as a compute pass of ours, so the list build no longer
//! tests and expands every instance on the CPU.
//!
//! A [`Batch`] is one instanced call's record block (the title's 64-B
//! IcInstance records), its LOD record and slot, the hide bitset, and the
//! draw's camera (origin and frustum planes). The list build keeps the
//! batch-level test (one sphere per batch, cached) and hands every batch
//! that survives it to the pass with `inst_count` = its records. The pass's
//! compute writes, per batch, the instances that pass the same rule as the
//! CPU (hide bit, sane record, sphere in the frustum, LOD slot) as the
//! 12-float camera-relative worlds [`super::instance_world`] makes, in record
//! order, from the batch's reserved first output instance; and per list row
//! of the batch the indirect draw arguments (index count, survivors). The row
//! is then drawn with ExecuteIndirect from those arguments, its vertex
//! program reading the cull output in place of the CPU instance buffer.
//!
//! Record blocks that are the same allocation frame after frame (the static
//! map lists and grass, kept by the hook by address) stay on the device:
//! uploaded once into one buffer of ours ([`Kept`], keyed by the block's
//! `Arc`), so a frame uploads only the per-frame CPU lists (0.1-0.6 MB in
//! the captures against 6-7 MB of static records).
//!
//! The rule's functions ([`lod_slot`], [`in_frustum`], [`plausible_record`],
//! [`frustum_planes`]) moved here from apps/xnw worldprobe.rs (FLK3 rule,
//! d067720) so both paths use one copy; [`cull_cpu`] is the reference the
//! device output is compared with.

use std::collections::HashMap;
use std::sync::Arc;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// Instance indices are 16 bits: at most 65,536 per batch.
pub const MAX_BATCH: u32 = 65_536;
/// Output instances per frame at most (48 B each: 48 MiB).
pub const OUT_CAP: u32 = 1 << 20;
/// Batches per frame at most (one thread group each).
pub const MAX_BATCHES: usize = 65_535;
/// A batch's descriptor in the frame buffer: 48 dwords.
const DESC_WORDS: usize = 48;
/// Bytes of one indirect draw (D3D12_DRAW_INDEXED_ARGUMENTS).
pub const ARG_BYTES: u32 = 20;
const NONE: u32 = u32::MAX;

/// One instanced call's cull input.
#[derive(Clone, Debug, Default)]
pub struct Batch {
    /// The records (64 B each); `len` of them are tested.
    pub records: Arc<Vec<u8>>,
    pub len: u32,
    /// The block is the same allocation across frames (kept on the device).
    pub keep: bool,
    /// The LOD record (35 floats) and this call's slot; None: every record draws.
    pub lod: Option<([f32; 35], u32)>,
    /// The title's hide bitset (instance i hidden when bit i&31 of word i>>5 is set).
    pub hide: Vec<u32>,
    /// The draw's camera position; its planes (None: the batch sphere is inside every plane).
    pub origin: [f32; 3],
    pub planes: Option<[[f32; 4]; 5]>,
    /// First output instance (reserved: `len` of them).
    pub out_first: u32,
}

/// The frame's batches and the list rows drawn from them (draw index, batch).
#[derive(Clone, Debug, Default)]
pub struct CullFrame {
    pub batches: Vec<Batch>,
    pub rows: Vec<(u32, u32)>,
    /// Output instances reserved (the sum of the batches' `len`).
    pub reserved: u32,
}

impl CullFrame {
    /// Adds a batch, reserving its output; None when the frame's output is full.
    pub fn push(&mut self, mut b: Batch) -> Option<u32> {
        b.len = b.len.min(MAX_BATCH).min((b.records.len() / 64) as u32);
        if self.batches.len() >= MAX_BATCHES || self.reserved + b.len > OUT_CAP {
            return None;
        }
        b.out_first = self.reserved;
        self.reserved += b.len;
        self.batches.push(b);
        Some((self.batches.len() - 1) as u32)
    }
}

/// Little-endian f32s.
fn f32s<const N: usize>(b: &[u8]) -> [f32; N] {
    std::array::from_fn(|i| f32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap()))
}

/// A record's 3x4 is finite and its axes are neither degenerate nor huge.
pub fn plausible_record(r: &[f32; 16]) -> bool {
    r[..12].iter().all(|v| v.is_finite()) && [0, 4, 8].iter().all(|&k| (1e-6..1e6).contains(&(r[k] * r[k] + r[k + 1] * r[k + 1] + r[k + 2] * r[k + 2])))
}

/// The LOD slot the culling shader gives instance `r` (16
/// floats) at distance `d` under LOD record `t` (35 floats), or None when it
/// is dropped (scale fields, far fade). Draw it iff the slot equals this
/// call's lod - firstLod.
pub fn lod_slot(t: &[f32; 35], r: &[f32; 16], d: f32) -> Option<usize> {
    // 2. rec_i+0x30 / +0x38.
    if r[12].min(r[14]) <= 0.0 {
        return None;
    }
    // 3. Far fade T+0x58 {s0, s1}: gone at d >= s1.
    let (s0, s1) = (t[22], t[23]);
    let e = s0.min(s1 - 1e-7);
    if d >= e && (s1 <= e || 1.0 - (d - e) / (s1 - e) <= 0.0) {
        return None;
    }
    // 4. Slot from the distances T+0x48 (< 0: absent); the last has no upper bound.
    let dd = [t[18], t[19], t[20], t[21]];
    let last = if dd[3] >= 0.0 {
        3
    } else if dd[2] >= 0.0 {
        2
    } else if dd[1] >= 0.0 {
        1
    } else {
        0
    };
    let mut k = if last == 0 || dd[1] < 0.0 {
        last
    } else if dd[0] >= 0.0 && d < dd[0] {
        0
    } else {
        (1..last).find(|&k| dd[k] >= 0.0 && d < dd[k]).unwrap_or(last)
    };
    // 5. Cross-fade band T+0x30: the hard switch (T+0x44 bit 0) at mid-band,
    // and the band's far end, move to the next slot (the band's double draw: U).
    if k < last {
        let (w, gap) = (t[12 + k], dd[k] - d);
        if w > 0.0 && gap < w {
            let x = gap / w;
            if (t[17].to_bits() & 1 != 0 && x < 0.5) || x < 1.0 / 255.0 {
                k += 1;
            }
        }
    }
    Some(k)
}

/// The planes of a camera-relative view-projection
/// (row vectors, reverse Z): left, right, bottom, top, near; (n, d) with |n| = 1.
pub fn frustum_planes(vp: &[f32; 16]) -> [[f32; 4]; 5] {
    let col = |j: usize| [vp[j], vp[4 + j], vp[8 + j], vp[12 + j]];
    let (x, y, z, w) = (col(0), col(1), col(2), col(3));
    let pl = |a: [f32; 4], s: f32, b: [f32; 4]| {
        let p: [f32; 4] = std::array::from_fn(|k| a[k] + s * b[k]);
        let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(1e-12);
        p.map(|v| v / l)
    };
    [pl(w, 1.0, x), pl(w, -1.0, x), pl(w, 1.0, y), pl(w, -1.0, y), pl(w, -1.0, z)]
}

/// An instance's bounding sphere (LOD record T: centre +0x08,
/// radius +0x14; scale 1 / min(rec +0x30, +0x38)) against the planes, in
/// camera-relative space; false when outside (or a scale <= 0).
pub fn in_frustum(t: &[f32; 35], r: &[f32; 16], cam: [f32; 3], planes: &[[f32; 4]; 5]) -> bool {
    let s = r[12].min(r[14]);
    if !(s > 0.0) {
        return false;
    }
    let (c, rad) = ([t[2], t[3], t[4]], t[5] / s);
    // Rows 0..2 of the 3x4 record, translation in .w (absolute), made camera-relative.
    let p: [f32; 3] = std::array::from_fn(|k| r[4 * k] * c[0] + r[4 * k + 1] * c[1] + r[4 * k + 2] * c[2] + r[4 * k + 3] - cam[k]);
    planes.iter().all(|pl| pl[0] * p[0] + pl[1] * p[1] + pl[2] * p[2] + pl[3] >= -rad)
}

/// Whether record `i` of a batch draws: the per-instance test of
/// worldprobe's `prefilter_why` (hide bit, sane record, frustum before the
/// LOD as BatchCull orders them, LOD slot).
pub fn keeps(b: &Batch, i: usize, r: &[f32; 16]) -> bool {
    if b.hide.get(i >> 5).is_some_and(|w| w & (1 << (i & 31)) != 0) || !plausible_record(r) {
        return false;
    }
    let Some((t, slot)) = &b.lod else { return true };
    if let Some(pl) = &b.planes
        && !in_frustum(t, r, b.origin, pl)
    {
        return false;
    }
    let v = [r[3] - b.origin[0], r[7] - b.origin[1], r[11] - b.origin[2]];
    let d = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    lod_slot(t, r, d) == Some(*slot as usize)
}

/// The CPU reference: a batch's surviving instances as 12-float worlds, in record order.
pub fn cull_cpu(b: &Batch) -> Vec<f32> {
    let mut out = Vec::new();
    for (i, rb) in b.records.chunks_exact(64).take(b.len as usize).enumerate() {
        let r = f32s::<16>(rb);
        if keeps(b, i, &r) {
            out.extend(super::instance_world(&r, b.origin));
        }
    }
    out
}

/// The compute program: one group of 64 per batch, the batch's records in
/// chunks of 64, survivors compacted in record order by a group prefix sum;
/// then the indirect arguments of the batch's rows. Every comparison is the
/// CPU's (`precise`: no contraction into fused multiply-adds).
pub const SHADER: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer K : register(b0) { uint batchCount; uint outBytes; uint rowsAt; uint argBytes; };
ByteAddressBuffer kept : register(t0);   // records kept on the device
ByteAddressBuffer frame : register(t1);  // descriptors, rows, hide words, this frame's records
RWByteAddressBuffer outInst : register(u0);
RWByteAddressBuffer args : register(u1);
groupshared uint scan[64];
groupshared uint base;
bool finite1(float x) { return (asuint(x) & 0x7F800000) != 0x7F800000; }
// HLSL evaluates both sides of `?:`: a load the condition rules out still
// runs, and through a root view (no bounds on the card) it reads wherever
// its address points. So the condition picks the address, every load stays
// inside its buffer (offset 0 is), then the value (HANG1).
float4 rec4(bool fromFrame, uint a) { float4 f = asfloat(frame.Load4(fromFrame ? a : 0)), k = asfloat(kept.Load4(fromFrame ? 0 : a)); return fromFrame ? f : k; }
uint dw(uint b, uint k) { return frame.Load(b * 192 + 4 * k); }
float df(uint b, uint k) { return asfloat(dw(b, k)); }
float4 d4(uint b, uint k) { return asfloat(frame.Load4(b * 192 + 4 * k)); }
// lod_slot (gpucull.rs): -1 dropped.
int lodSlot(uint b, float s, float d) {
    if (s <= 0) return -1;
    precise float s0 = df(b, 44), s1 = df(b, 45);
    precise float e = min(s0, s1 - 1e-7);
    if (d >= e) {
        if (s1 <= e) return -1;
        precise float f = 1.0 - (d - e) / (s1 - e);
        if (f <= 0) return -1;
    }
    float4 dd = d4(b, 40);
    int last = dd.w >= 0 ? 3 : dd.z >= 0 ? 2 : dd.y >= 0 ? 1 : 0;
    int k;
    if (last == 0 || dd.y < 0) k = last;
    else if (dd.x >= 0 && d < dd.x) k = 0;
    else {
        k = last;
        [loop] for (int j = 1; j < last; j++) { if (dd[j] >= 0 && d < dd[j]) { k = j; break; } }
    }
    if (k < last) {
        float w = d4(b, 36)[k];
        precise float gap = dd[k] - d;
        if (w > 0 && gap < w) {
            precise float x = gap / w;
            if (((dw(b, 46) & 1) != 0 && x < 0.5) || x < 1.0 / 255.0) k++;
        }
    }
    return k;
}
bool keeps(uint b, uint i, float4 r0, float4 r1, float4 r2, float4 r3, uint flags, float3 o) {
    uint hideAt = dw(b, 5), hideWords = dw(b, 6), hw = i >> 5;
    uint hide = frame.Load(hideAt + 4 * (hw < hideWords ? hw : 0));
    if (hw < hideWords && ((hide >> (i & 31)) & 1) != 0) return false;
    [unroll] for (int k = 0; k < 4; k++) { if (!finite1(r0[k]) || !finite1(r1[k]) || !finite1(r2[k])) return false; }
    precise float l0 = r0.x * r0.x + r0.y * r0.y + r0.z * r0.z;
    precise float l1 = r1.x * r1.x + r1.y * r1.y + r1.z * r1.z;
    precise float l2 = r2.x * r2.x + r2.y * r2.y + r2.z * r2.z;
    if (!(l0 >= 1e-6 && l0 < 1e6) || !(l1 >= 1e-6 && l1 < 1e6) || !(l2 >= 1e-6 && l2 < 1e6)) return false;
    if ((flags & 2) == 0) return true;
    float s = min(r3.x, r3.z);
    if ((flags & 4) != 0) {
        if (!(s > 0)) return false;
        float4 cr = d4(b, 32);
        precise float rad = cr.w / s;
        precise float3 p = float3(r0.x * cr.x + r0.y * cr.y + r0.z * cr.z + r0.w - o.x,
                                  r1.x * cr.x + r1.y * cr.y + r1.z * cr.z + r1.w - o.y,
                                  r2.x * cr.x + r2.y * cr.y + r2.z * cr.z + r2.w - o.z);
        [loop] for (uint q = 0; q < 5; q++) {
            float4 pl = d4(b, 12 + 4 * q);
            precise float dist = pl.x * p.x + pl.y * p.y + pl.z * p.z + pl.w;
            if (!(dist >= -rad)) return false;
        }
    }
    precise float3 v = float3(r0.w - o.x, r1.w - o.y, r2.w - o.z);
    precise float d = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return lodSlot(b, s, d) == (int)dw(b, 4);
}
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void cs(uint3 g : SV_GroupID, uint t : SV_GroupThreadID) {
    // One exit (no return inside a branch: WARP 1.0.22 removes the device
    // on that form, WFIX1, and GPU-based validation's patched copy too): a
    // group past the batches tests nothing and writes no row.
    uint b = g.x;
    bool live = b < batchCount;
    uint recAt = dw(b, 0), n = live ? dw(b, 1) : 0, outFirst = dw(b, 2), flags = dw(b, 3);
    float3 o = float3(df(b, 9), df(b, 10), df(b, 11));
    bool fromFrame = (flags & 1) != 0;
    if (t == 0) base = 0;
    GroupMemoryBarrierWithGroupSync();
    [loop] for (uint c = 0; c < n; c += 64) {
        uint i = c + t;
        float4 r0 = 0, r1 = 0, r2 = 0, r3 = 0;
        bool keep = false;
        if (i < n) {
            uint a = recAt + 64 * i;
            r0 = rec4(fromFrame, a); r1 = rec4(fromFrame, a + 16); r2 = rec4(fromFrame, a + 32); r3 = rec4(fromFrame, a + 48);
            keep = keeps(b, i, r0, r1, r2, r3, flags, o);
        }
        scan[t] = keep ? 1 : 0;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint s = 1; s < 64; s <<= 1) {
            uint add = t >= s ? scan[t - s] : 0;
            GroupMemoryBarrierWithGroupSync();
            scan[t] += add;
            GroupMemoryBarrierWithGroupSync();
        }
        uint at = (outFirst + base + scan[t] - 1) * 48;
        if (keep && at + 48 <= outBytes) {
            outInst.Store3(at, asuint(r0.xyz));
            outInst.Store3(at + 12, asuint(r1.xyz));
            outInst.Store3(at + 24, asuint(r2.xyz));
            precise float3 tr = float3(r0.w - o.x, r1.w - o.y, r2.w - o.z);
            outInst.Store3(at + 36, asuint(tr));
        }
        GroupMemoryBarrierWithGroupSync();
        if (t == 63) base += scan[63];
        GroupMemoryBarrierWithGroupSync();
    }
    uint rowFirst = dw(b, 7), rowCount = live ? dw(b, 8) : 0;
    [loop] for (uint r = t; r < rowCount; r += 64) {
        uint2 row = frame.Load2(rowsAt + 8 * (rowFirst + r));
        uint a = row.x * 20;
        if (a + 20 <= argBytes) {
            args.Store4(a, uint4(row.y, base, 0, 0));
            args.Store(a + 16, 0);
        }
    }
}
"#;

/// One batch's 48-dword descriptor (see [`SHADER`]).
fn descriptor(b: &Batch, rec_at: u32, from_frame: bool, hide_at: u32, rows: (u32, u32)) -> [u32; DESC_WORDS] {
    let mut d = [0u32; DESC_WORDS];
    let (t, slot) = b.lod.map_or(([0f32; 35], NONE), |(t, s)| (t, s));
    d[0] = rec_at;
    d[1] = b.len;
    d[2] = b.out_first;
    d[3] = u32::from(from_frame) | u32::from(b.lod.is_some()) << 1 | u32::from(b.lod.is_some() && b.planes.is_some()) << 2;
    d[4] = slot;
    d[5] = hide_at;
    d[6] = b.hide.len() as u32;
    (d[7], d[8]) = rows;
    for k in 0..3 {
        d[9 + k] = b.origin[k].to_bits();
    }
    if let Some(p) = &b.planes {
        for (q, pl) in p.iter().enumerate() {
            for k in 0..4 {
                d[12 + 4 * q + k] = pl[k].to_bits();
            }
        }
    }
    for k in 0..4 {
        d[32 + k] = t[2 + k].to_bits();
        d[36 + k] = t[12 + k].to_bits();
        d[40 + k] = t[18 + k].to_bits();
    }
    d[44] = t[22].to_bits();
    d[45] = t[23].to_bits();
    d[46] = t[17].to_bits();
    d
}

/// Record blocks kept on the device across frames, by their allocation.
#[derive(Default)]
pub struct Kept {
    at: HashMap<(usize, usize), (Arc<Vec<u8>>, u64)>,
    pub used: u64,
    pub size: u64,
}

/// Where this frame's blocks lie: per batch (offset, in the frame buffer),
/// the kept blocks to copy (batch index, kept offset), and whether the kept
/// buffer was remade (everything in it copied again).
struct Placement {
    at: Vec<(u64, bool)>,
    new: Vec<(usize, u64)>,
    rebuild: bool,
}

impl Kept {
    fn place(&mut self, batches: &[Batch]) -> Placement {
        let key = |b: &Batch| (Arc::as_ptr(&b.records) as usize, b.records.len());
        let need: u64 = {
            let mut seen = std::collections::HashSet::new();
            batches.iter().filter(|b| b.keep && seen.insert(key(b))).map(|b| b.records.len() as u64).sum()
        };
        let fresh: u64 = batches.iter().filter(|b| b.keep && !self.at.contains_key(&key(b))).map(|b| b.records.len() as u64).sum();
        let mut rebuild = false;
        if self.used + fresh > self.size {
            // Start over with this frame's blocks only, in a buffer twice their size.
            self.at.clear();
            self.used = 0;
            self.size = (2 * need).next_power_of_two().max(16 << 20);
            rebuild = true;
        }
        let (mut at, mut new) = (Vec::with_capacity(batches.len()), Vec::new());
        let mut frame_at: HashMap<(usize, usize), u64> = HashMap::new();
        let mut frame_used = 0u64;
        for (i, b) in batches.iter().enumerate() {
            if b.keep {
                let off = match self.at.get(&key(b)) {
                    Some((_, o)) => *o,
                    None => {
                        let o = self.used;
                        self.used += (b.records.len() as u64).next_multiple_of(64);
                        self.at.insert(key(b), (b.records.clone(), o));
                        new.push((i, o));
                        o
                    }
                };
                at.push((off, false));
            } else {
                let off = *frame_at.entry(key(b)).or_insert_with(|| {
                    let o = frame_used;
                    frame_used += (b.records.len() as u64).next_multiple_of(64);
                    o
                });
                at.push((off, true));
            }
        }
        Placement { at, new, rebuild }
    }
}

/// What the last frame's cull did.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub batches: usize,
    pub rows: usize,
    pub tested: u64,
    /// Bytes uploaded this frame (descriptors, rows, hide words, per-frame
    /// records, newly kept records) and bytes kept on the device.
    pub uploaded: u64,
    pub kept: u64,
    pub record_ms: f64,
    pub gpu_ms: f64,
}

/// The cull pass's device objects.
#[derive(Default)]
pub struct GpuCull {
    pipeline: Option<(ID3D12RootSignature, ID3D12PipelineState, ID3D12CommandSignature)>,
    upload: Option<(ID3D12Resource, *mut u8, usize)>,
    store: Option<ID3D12Resource>,
    out: Option<(ID3D12Resource, u64)>,
    args: Option<(ID3D12Resource, u64)>,
    kept: Kept,
    stamps: Option<(ID3D12QueryHeap, ID3D12Resource)>,
    stamped: bool,
    /// The heap slot of the output's view (the world pass's heap).
    pub slot: Option<u32>,
    pub stats: Stats,
}

fn buffer(device: &ID3D12Device, size: u64, heap: D3D12_HEAP_TYPE, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES) -> Result<ID3D12Resource, String> {
    let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: size.max(256), Height: 1, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_UNKNOWN, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, Flags: flags, ..Default::default() };
    let mut r: Option<ID3D12Resource> = None;
    // SAFETY: descriptions outlive the call.
    unsafe { device.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r) }.map_err(|e| format!("cull buffer ({size} B): {e}"))?;
    r.ok_or_else(|| "cull buffer: none".into())
}

fn barrier(list: &ID3D12GraphicsCommandList, r: &ID3D12Resource, from: D3D12_RESOURCE_STATES, to: D3D12_RESOURCE_STATES) {
    let b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer for the call; not released (ManuallyDrop).
                pResource: unsafe { std::mem::transmute_copy(r) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: from,
                StateAfter: to,
            }),
        },
    };
    // SAFETY: an open list and a live resource.
    unsafe { list.ResourceBarrier(&[b]) };
}

/// What [`GpuCull::record`] leaves for the draws: the output instances
/// (12 floats each, readable by vertex shaders), its bytes, the argument
/// buffer (one [`ARG_BYTES`] slot per row, indirect-argument state) and the
/// command signature of one indexed draw.
pub struct Recorded {
    pub out: ID3D12Resource,
    pub out_bytes: u64,
    pub args: ID3D12Resource,
    pub signature: ID3D12CommandSignature,
}

impl GpuCull {
    fn pipeline(&mut self, device: &ID3D12Device) -> Result<(ID3D12RootSignature, ID3D12PipelineState, ID3D12CommandSignature), String> {
        if let Some(p) = &self.pipeline {
            return Ok(p.clone());
        }
        let cs = super::compile_source(SHADER, c"cs", c"cs_5_1")?;
        // SAFETY: the blob and descriptions live through the calls.
        let p = unsafe {
            let bytes = std::slice::from_raw_parts(cs.GetBufferPointer().cast::<u8>(), cs.GetBufferSize());
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).map_err(|e| format!("cull root: {e}"))?;
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: std::mem::transmute_copy(&root), CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.GetBufferPointer(), BytecodeLength: cs.GetBufferSize() }, ..Default::default() };
            let pso: ID3D12PipelineState = device.CreateComputePipelineState(&desc).map_err(|e| format!("cull pipeline: {e}"))?;
            let arg = D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED, ..Default::default() };
            let sd = D3D12_COMMAND_SIGNATURE_DESC { ByteStride: ARG_BYTES, NumArgumentDescs: 1, pArgumentDescs: &arg, NodeMask: 0 };
            let mut sig: Option<ID3D12CommandSignature> = None;
            device.CreateCommandSignature(&sd, None, &mut sig).map_err(|e| format!("cull command signature: {e}"))?;
            (root, pso, sig.ok_or("cull command signature: none")?)
        };
        self.pipeline = Some(p.clone());
        Ok(p)
    }

    /// Records the frame's cull on `list` before the draws. `rows`: per row
    /// of `frame.rows`, the index count its draw will use (0: the row is not
    /// drawn; its arguments draw nothing).
    pub fn record(&mut self, device: &ID3D12Device, list: &ID3D12GraphicsCommandList, frame: &CullFrame, index_counts: &[u32]) -> Result<Recorded, String> {
        let t0 = std::time::Instant::now();
        let (root, pso, signature) = self.pipeline(device)?;
        let batches = &frame.batches;
        let place = self.kept.place(batches);
        if place.rebuild || self.store.is_none() {
            // The buffer before is free: the caller waits for each frame's list.
            self.store = Some(buffer(device, self.kept.size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON)?);
        }
        // Rows grouped by batch: per batch (first, count) in the rows table of (arg slot, index count).
        let mut by_batch: Vec<Vec<(u32, u32)>> = vec![Vec::new(); batches.len()];
        for (slot, &(_, b)) in frame.rows.iter().enumerate() {
            if let Some(v) = by_batch.get_mut(b as usize) {
                v.push((slot as u32, index_counts.get(slot).copied().unwrap_or(0)));
            }
        }
        // The frame buffer: descriptors | rows | hide words | per-frame records | newly kept records.
        let rows_at = (batches.len() * DESC_WORDS * 4).next_multiple_of(64);
        let hide_at = (rows_at + 8 * frame.rows.len()).next_multiple_of(64);
        let hide_words: usize = batches.iter().map(|b| b.hide.len()).sum();
        let recs_at = (hide_at + 4 * hide_words).next_multiple_of(64);
        let frame_recs: u64 = place.at.iter().zip(batches).filter(|(a, _)| a.1).map(|(a, b)| a.0 + (b.records.len() as u64).next_multiple_of(64)).max().unwrap_or(0);
        let new_at = (recs_at as u64 + frame_recs).next_multiple_of(256);
        let new_bytes: u64 = place.new.iter().map(|&(i, _)| (batches[i].records.len() as u64).next_multiple_of(64)).sum();
        // 64 bytes past the end: a batch without hide words reads one at its offset.
        let total = (new_at + new_bytes) as usize + 64;
        if self.upload.as_ref().is_none_or(|u| u.2 < total) {
            let size = total.next_power_of_two().max(1 << 20);
            let r = buffer(device, size as u64, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
            let mut p = std::ptr::null_mut();
            // SAFETY: an upload buffer stays mapped for its life.
            unsafe { r.Map(0, None, Some(&mut p)) }.map_err(|e| format!("cull upload Map: {e}"))?;
            self.upload = Some((r, p.cast::<u8>(), size));
        }
        let out_bytes = u64::from(frame.reserved.max(1)) * 48;
        if self.out.as_ref().is_none_or(|o| o.1 < out_bytes) {
            let size = out_bytes.next_power_of_two().max(1 << 20);
            self.out = Some((buffer(device, size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON)?, size));
        }
        let arg_bytes = (frame.rows.len().max(1) as u64) * u64::from(ARG_BYTES);
        if self.args.as_ref().is_none_or(|a| a.1 < arg_bytes) {
            let size = arg_bytes.next_power_of_two().max(64 << 10);
            self.args = Some((buffer(device, size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON)?, size));
        }
        if self.stamps.is_none() {
            let mut h: Option<ID3D12QueryHeap> = None;
            // SAFETY: as above.
            unsafe { device.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2, NodeMask: 0 }, &mut h) }.map_err(|e| format!("cull query heap: {e}"))?;
            let rb = buffer(device, 16, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
            self.stamps = Some((h.ok_or("cull query heap: none")?, rb));
        }
        let (ubuf, mapped, _) = self.upload.as_ref().expect("made above");
        // SAFETY: every write below lies inside `total` bytes of the mapping (sized above); the GPU is done with the last frame's bytes (the caller waits).
        unsafe {
            let put = |at: usize, b: &[u8]| std::ptr::copy_nonoverlapping(b.as_ptr(), mapped.add(at), b.len());
            let mut row_first = 0u32;
            let mut hide_next = hide_at;
            let mut rows_bytes: Vec<u8> = Vec::with_capacity(8 * frame.rows.len());
            for (i, b) in batches.iter().enumerate() {
                let (off, from_frame) = place.at[i];
                let rec_at = if from_frame { recs_at as u64 + off } else { off };
                let rows = &by_batch[i];
                let d = descriptor(b, rec_at as u32, from_frame, hide_next as u32, (row_first, rows.len() as u32));
                put(i * DESC_WORDS * 4, std::slice::from_raw_parts(d.as_ptr().cast::<u8>(), DESC_WORDS * 4));
                if !b.hide.is_empty() {
                    put(hide_next, std::slice::from_raw_parts(b.hide.as_ptr().cast::<u8>(), 4 * b.hide.len()));
                    hide_next += 4 * b.hide.len();
                }
                for &(slot, n) in rows {
                    rows_bytes.extend_from_slice(&slot.to_le_bytes());
                    rows_bytes.extend_from_slice(&n.to_le_bytes());
                }
                row_first += rows.len() as u32;
            }
            put(rows_at, &rows_bytes);
            // Per-frame records, each block once.
            let mut done = std::collections::HashSet::new();
            for (i, b) in batches.iter().enumerate() {
                let (off, from_frame) = place.at[i];
                if from_frame && done.insert(off) {
                    put(recs_at + off as usize, &b.records);
                }
            }
            let mut at = new_at as usize;
            for &(i, _) in &place.new {
                put(at, &batches[i].records);
                at += batches[i].records.len().next_multiple_of(64);
            }
        }
        let (out, _) = self.out.as_ref().expect("made above");
        let (args, _) = self.args.as_ref().expect("made above");
        let store = self.store.as_ref().expect("made above");
        let (qh, qrb) = self.stamps.as_ref().expect("made above");
        // SAFETY: recording on the caller's open list; every object outlives the list's execution (held here).
        unsafe {
            // Buffers decay to COMMON after each list.
            if !place.new.is_empty() {
                barrier(list, store, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
                let mut from = new_at;
                for &(i, to) in &place.new {
                    let len = batches[i].records.len() as u64;
                    list.CopyBufferRegion(store, to, ubuf, from, len);
                    from += len.next_multiple_of(64);
                }
                barrier(list, store, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            } else {
                barrier(list, store, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            barrier(list, out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            barrier(list, args, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            list.SetComputeRootSignature(&root);
            list.SetPipelineState(&pso);
            let k = [batches.len() as u32, out_bytes.min(u64::from(u32::MAX)) as u32, rows_at as u32, arg_bytes as u32];
            list.SetComputeRoot32BitConstants(0, 4, k.as_ptr().cast(), 0);
            list.SetComputeRootShaderResourceView(1, store.GetGPUVirtualAddress());
            list.SetComputeRootShaderResourceView(2, ubuf.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(3, out.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(4, args.GetGPUVirtualAddress());
            list.EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            if !batches.is_empty() {
                list.Dispatch(batches.len() as u32, 1, 1);
            }
            list.EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            list.ResolveQueryData(qh, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, qrb, 0);
            barrier(list, out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            barrier(list, args, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        }
        self.stamped = true;
        self.stats = Stats { batches: batches.len(), rows: frame.rows.len(), tested: batches.iter().map(|b| u64::from(b.len)).sum(), uploaded: total as u64, kept: self.kept.used, record_ms: t0.elapsed().as_secs_f64() * 1e3, gpu_ms: 0.0 };
        Ok(Recorded { out: out.clone(), out_bytes, args: args.clone(), signature })
    }

    /// [`Self::prepare`] with a device handle of its own (the pass's, cloned).
    pub fn prepare_on(&mut self, device: ID3D12Device) -> Result<f64, String> {
        self.prepare(&device)
    }

    /// Makes the pipeline now (an FXC compile); the milliseconds it took.
    pub fn prepare(&mut self, device: &ID3D12Device) -> Result<f64, String> {
        let t = std::time::Instant::now();
        self.pipeline(device)?;
        Ok(t.elapsed().as_secs_f64() * 1e3)
    }

    /// The GPU milliseconds of the last recorded dispatch, once its list has finished.
    pub fn gpu_ms(&mut self, frequency: u64) -> Option<f64> {
        if !std::mem::take(&mut self.stamped) {
            return None;
        }
        let (_, rb) = self.stamps.as_ref()?;
        let mut p = std::ptr::null_mut();
        // SAFETY: a readback buffer of 16 B, mapped and unmapped here.
        let t = unsafe {
            rb.Map(0, None, Some(&mut p)).ok()?;
            let t = std::slice::from_raw_parts(p.cast::<u64>(), 2).to_vec();
            rb.Unmap(0, None);
            t
        };
        let ms = t[1].saturating_sub(t[0]) as f64 * 1e3 / frequency.max(1) as f64;
        self.stats.gpu_ms = ms;
        Some(ms)
    }
}

/// The 60-frame line.
pub fn line(frame: u64, s: &Stats) -> String {
    format!("world render: frame {frame}: gpu cull: {} batches, {} rows, {} instances tested on the device; uploaded {} KiB, kept {} KiB; record {:.2} ms, GPU {:.3} ms", s.batches, s.rows, s.tested, s.uploaded >> 10, s.kept >> 10, s.record_ms, s.gpu_ms)
}

/// Runs `frame` on `dev` alone (twice: the second run finds the
/// kept blocks on the device) and reads back each batch's survivors (12
/// floats each, record order) and each row's arguments; with the GPU
/// milliseconds of the second run. For checks against [`cull_cpu`].
#[allow(clippy::type_complexity)]
pub fn run(dev: &ID3D12Device, frame: &CullFrame, index_counts: &[u32]) -> Result<(Vec<Vec<f32>>, Vec<[u32; 5]>, f64, Stats), String> {
    let e = |e: windows::core::Error| format!("cull run: {e}");
    let mut pass = GpuCull::default();
    // SAFETY: D3D12 calls on our own objects; the fence wait ends the GPU's use of every buffer before they drop.
    unsafe {
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(e)?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        let mut result = None;
        for pass_no in 1..=2u64 {
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(e)?;
            let rec = pass.record(dev, &list, frame, index_counts)?;
            let (ob, ab) = (rec.out_bytes, frame.rows.len().max(1) as u64 * u64::from(ARG_BYTES));
            let rb = buffer(dev, ob + ab, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
            barrier(&list, &rec.out, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
            barrier(&list, &rec.args, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list.CopyBufferRegion(&rb, 0, &rec.out, 0, ob);
            list.CopyBufferRegion(&rb, ob, &rec.args, 0, ab);
            list.Close().map_err(e)?;
            queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(&list).map_err(e)?)]);
            let sent = xnw_gpu::gpuwatch::begin("cull check");
            queue.Signal(&fence, pass_no).map_err(e)?;
            while fence.GetCompletedValue() < pass_no {
                dev.GetDeviceRemovedReason().map_err(|x| format!("device removed: {}", e(x)))?;
                std::thread::yield_now();
            }
            xnw_gpu::gpuwatch::end("cull check", sent, None);
            let ms = pass.gpu_ms(queue.GetTimestampFrequency().map_err(e)?).unwrap_or(0.0);
            let mut p = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut p)).map_err(e)?;
            let bytes = std::slice::from_raw_parts(p.cast::<u8>(), (ob + ab) as usize).to_vec();
            rb.Unmap(0, None);
            result = Some((bytes, ms, ob));
        }
        let (bytes, ms, ob) = result.ok_or("no run")?;
        let w = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let args: Vec<[u32; 5]> = (0..frame.rows.len()).map(|r| std::array::from_fn(|k| w(ob as usize + 20 * r + 4 * k))).collect();
        // A batch's survivor count is in any of its rows' arguments; a batch
        // without rows is read up to its reservation and compared by the caller.
        let mut count: Vec<Option<u32>> = vec![None; frame.batches.len()];
        for (r, &(_, b)) in frame.rows.iter().enumerate() {
            count[b as usize] = Some(args[r][1]);
        }
        let outs = frame
            .batches
            .iter()
            .zip(&count)
            .map(|(b, c)| {
                let n = c.unwrap_or(b.len) as usize;
                (0..12 * n).map(|k| f32::from_bits(w(48 * b.out_first as usize + 4 * k))).collect()
            })
            .collect();
        Ok((outs, args, ms, pass.stats))
    }
}

/// What the draws of a frame need from its cull: the output's view slot
/// and bytes, the arguments, the signature, and per list draw its argument slot.
pub struct Culled {
    pub view_slot: u32,
    pub out_bytes: u64,
    pub args: ID3D12Resource,
    pub signature: ID3D12CommandSignature,
    slot: Vec<u32>,
}

impl Culled {
    /// The argument slot of list draw `d` (None: not a device-culled row).
    pub fn slot_of(&self, d: u32) -> Option<u32> {
        self.slot.get(d as usize).copied().filter(|s| *s != NONE)
    }
}

impl super::Pass {
    /// Records the frame's device cull on the pass's open list, before the
    /// draws (None: no culled rows, or the cull failed: logged once, the
    /// rows then draw nothing).
    pub(super) fn cull_record(&mut self, list: &super::FrameList, l: &ID3D12GraphicsCommandList) -> Option<Culled> {
        let cf = &list.cull;
        if cf.rows.is_empty() {
            return None;
        }
        // Each row's index count as the draw loop will take it (raw geometries only).
        let counts: Vec<u32> = cf.rows.iter().map(|&(d, _)| list.draws.get(d as usize).and_then(|r| self.cache.get_or_make(r.geom, list.frame, || None)).filter(|g| g.raw.is_some()).map_or(0, |g| g.indices)).collect();
        let rec = match self.cull.record(&self.device, l, cf, &counts) {
            Ok(r) => r,
            Err(e) => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| xnw_core::warn!("world render: gpu cull not recorded: {e}"));
                return None;
            }
        };
        if self.cull.slot.is_none() {
            self.cull.slot = self.free_slots.pop().or_else(|| {
                (self.next_slot < super::HEAP_SLOTS - super::TEX_SLOTS).then(|| {
                    self.next_slot += 1;
                    self.next_slot - 1
                })
            });
        }
        let view_slot = self.cull.slot?;
        self.raw_view(&rec.out, rec.out_bytes, view_slot);
        let mut slot = vec![NONE; list.draws.len()];
        for (a, &(d, _)) in cf.rows.iter().enumerate() {
            if let Some(s) = slot.get_mut(d as usize) {
                *s = a as u32;
            }
        }
        Some(Culled { view_slot, out_bytes: rec.out_bytes, args: rec.args, signature: rec.signature, slot })
    }

    /// The last frame's cull, with the GPU time of its dispatch (read once its list has finished).
    pub fn cull_stats(&mut self) -> Stats {
        // SAFETY: a live queue.
        let freq = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(0);
        self.cull.gpu_ms(freq);
        self.cull.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(x: f32, y: f32, z: f32, s: f32) -> [f32; 16] {
        [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, z, s, 0.0, s, 0.0]
    }

    fn block(recs: &[[f32; 16]]) -> Arc<Vec<u8>> {
        Arc::new(recs.iter().flatten().flat_map(|v| v.to_le_bytes()).collect())
    }

    /// A LOD record: sphere centre 0, radius 1, slots at 20 / 40 m, fade 150-160 m.
    fn lodrec() -> [f32; 35] {
        let mut t = [0f32; 35];
        t[5] = 1.0;
        t[18] = 20.0;
        t[19] = 40.0;
        t[20] = -1.0;
        t[21] = -1.0;
        t[22] = 150.0;
        t[23] = 160.0;
        t
    }

    /// The reference keeps what worldprobe's rule keeps: LOD slot by distance,
    /// hide bits, insane records, the frustum (a camera looking down +y).
    #[test]
    fn reference_rule() {
        let recs = [rec(0.0, 5.0, 0.0, 1.0), rec(0.0, 30.0, 0.0, 1.0), rec(0.0, 10.0, 0.0, 0.0), rec(0.0, -10.0, 0.0, 1.0), rec(0.0, 8.0, 0.0, 1.0), rec(f32::NAN, 8.0, 0.0, 1.0)];
        // Planes: only y >= 0 counts (the other four never cut).
        let far = [0.0, 0.0, 0.0, 1e9];
        let planes = [[0.0, 1.0, 0.0, 0.0], far, far, far, far];
        let b = Batch { records: block(&recs), len: 6, lod: Some((lodrec(), 0)), hide: vec![0b1_0000], planes: Some(planes), ..Default::default() };
        let out = cull_cpu(&b);
        // rec 0 slot 0 kept; 1 slot 1; 2 scale 0; 3 behind; 4 hidden; 5 NaN.
        assert_eq!(out.len(), 12);
        assert_eq!(out[10], 5.0);
        let b1 = Batch { lod: Some((lodrec(), 1)), ..b.clone() };
        assert_eq!(cull_cpu(&b1)[10], 30.0);
        let none = Batch { lod: None, hide: Vec::new(), ..b };
        assert_eq!(cull_cpu(&none).len(), 12 * 5, "without a LOD record every sane record draws");
    }

    #[test]
    fn push_reserves_and_caps() {
        let mut f = CullFrame::default();
        let b = Batch { records: block(&[rec(0.0, 0.0, 0.0, 1.0); 3]), len: 10, ..Default::default() };
        assert_eq!(f.push(b.clone()), Some(0));
        assert_eq!(f.push(b), Some(1));
        assert_eq!((f.reserved, f.batches[1].out_first, f.batches[1].len), (6, 3, 3));
    }

    #[test]
    fn kept_blocks_upload_once() {
        let a = block(&[rec(0.0, 0.0, 0.0, 1.0); 4]);
        let c = block(&[rec(1.0, 0.0, 0.0, 1.0); 2]);
        let bs = vec![Batch { records: a.clone(), len: 4, keep: true, ..Default::default() }, Batch { records: a.clone(), len: 4, keep: true, ..Default::default() }, Batch { records: c.clone(), len: 2, ..Default::default() }];
        let mut k = Kept::default();
        let p = k.place(&bs);
        assert!(p.rebuild);
        assert_eq!(p.new.len(), 1, "one copy of a block two batches share");
        assert_eq!((p.at[0], p.at[1], p.at[2]), ((0, false), (0, false), (0, true)));
        let p2 = k.place(&bs);
        assert!(!p2.rebuild && p2.new.is_empty(), "the next frame uploads no kept block");
    }

    /// The instrument: one trivial dispatch (one store through a root UAV)
    /// runs to its fence. Under `XNW_GPU_DEBUG=layer,gbv` it tells whether
    /// GPU-based validation itself works on this adapter (HANG1).
    #[test]
    fn trivial_dispatch_runs() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let (_gpu, host) = super::super::tests::warp_host();
        let dev = &host.device;
        let src = "#define RS \"UAV(u0)\"\nRWByteAddressBuffer o : register(u0);\n[RootSignature(RS)][numthreads(1, 1, 1)] void cs() { o.Store(0, 7); }\n";
        let cs = super::super::compile_source(src, c"cs", c"cs_5_1").expect("compile");
        let e = |e: windows::core::Error| e.to_string();
        // SAFETY: as in `run`.
        let got = unsafe {
            let bytes = std::slice::from_raw_parts(cs.GetBufferPointer().cast::<u8>(), cs.GetBufferSize());
            let root: ID3D12RootSignature = dev.CreateRootSignature(0, bytes).map_err(e).expect("root");
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: std::mem::transmute_copy(&root), CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.GetBufferPointer(), BytecodeLength: cs.GetBufferSize() }, ..Default::default() };
            let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&desc).map_err(e).expect("pso");
            let out = buffer(dev, 256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON).expect("out");
            let rb = buffer(dev, 256, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST).expect("rb");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(e).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e).expect("fence");
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e).expect("alloc");
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(e).expect("list");
            barrier(&list, &out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            list.SetComputeRootSignature(&root);
            list.SetPipelineState(&pso);
            list.SetComputeRootUnorderedAccessView(0, out.GetGPUVirtualAddress());
            list.Dispatch(1, 1, 1);
            barrier(&list, &out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list.CopyBufferRegion(&rb, 0, &out, 0, 4);
            list.Close().map_err(e).expect("close");
            queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(&list).map_err(e).expect("cast"))]);
            queue.Signal(&fence, 1).map_err(e).expect("signal");
            while fence.GetCompletedValue() < 1 {
                dev.GetDeviceRemovedReason().map_err(e).expect("device kept");
                std::thread::yield_now();
            }
            let mut p = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut p)).map_err(e).expect("map");
            let v = *p.cast::<u32>();
            rb.Unmap(0, None);
            v
        };
        assert_eq!(got, 7);
    }

    /// The device output equals the reference, instance for instance, on a
    /// made-up frame of mixed batches (WARP unless XNW_ADAPTER says otherwise).
    #[test]
    fn device_matches_reference() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let (_gpu, host) = super::super::tests::warp_host();
        let mut f = CullFrame::default();
        let mut seed = 12345u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        let far = [0.0, 0.0, 0.0, 1e9];
        for k in 0..40 {
            let n = [1usize, 63, 64, 65, 200, 1000][k % 6];
            let recs: Vec<[f32; 16]> = (0..n).map(|_| rec(rnd() * 200.0 - 100.0, rnd() * 200.0 - 20.0, rnd() * 10.0, if rnd() < 0.05 { 0.0 } else { 0.5 + rnd() })).collect();
            let hide = if k % 5 == 0 { vec![0xF0F0_F0F0; 4] } else { Vec::new() };
            let planes = (k % 3 != 0).then_some([[0.0, 1.0, 0.0, 0.0], [0.6, 0.8, 0.0, 0.0], far, far, far]);
            let lod = (k % 7 != 0).then(|| (lodrec(), (k % 3) as u32));
            let i = f.push(Batch { records: block(&recs), len: n as u32, keep: k % 2 == 0, lod, hide, origin: [rnd(), rnd(), 0.5], planes, out_first: 0 }).unwrap();
            f.rows.push((k as u32, i));
        }
        let counts: Vec<u32> = (0..f.rows.len() as u32).map(|r| 36 + r).collect();
        let (outs, args, ms, st) = run(&host.device, &f, &counts).expect("run");
        let mut kept = 0;
        for (i, b) in f.batches.iter().enumerate() {
            let want = cull_cpu(b);
            assert_eq!(outs[i].len(), want.len(), "batch {i}: count");
            assert!(outs[i].iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "batch {i}: worlds");
            kept += want.len() / 12;
        }
        assert_eq!(args[3], [39, args[3][1], 0, 0, 0]);
        eprintln!("gpu cull: {} batches, {} tested, {kept} kept; GPU {ms:.3} ms; uploaded {} KiB (second run), kept {} KiB", f.batches.len(), st.tested, st.uploaded >> 10, st.kept >> 10);
    }
}
