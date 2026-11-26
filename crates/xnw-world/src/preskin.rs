//! The ped pre-skin pass (`ped_skinning_cs`)
//! outside the title, so the PreSkinD draws find the positions the console's
//! GPU would have written. Two routes over the same inputs (one captured
//! dispatch) and the same output layout:
//!
//! - [`cpu`]: our own reference of what `CS_PedStandardSkinning_0`
//!   (a45da713, 13,376 B of GFX10 code) computes, checked against the
//!   program's output. Needs no adapter.
//! - [`run`]: the title's own program, translated ([`translate`]) and
//!   dispatched with the binding [`bind`] builds from the record. Needs a
//!   device with shader model 6.6, `WaveSize(32)` and doubles;
//!   the software adapter here has 6.2 and 4 lanes, so [`adapter_gap`]
//!   says why it cannot run and the device test skips.
//!
//! Output range (both routes; V a45da713 0x108-0x1a0, 0x2198-0x2480):
//! `w` = the range word (b7+0x34, later rec+0x54); bytes from
//! `(w << 4) & 0x0ffffff0` in buffer A (u7, ud4+0x300) when bits 24-25 are
//! 0, else B (u8, ud4+0x320): a 32-B header `{n, 0xdeadbeef, 0 x 6}`, n
//! positions (float3 12 B if bit 29, 10:10:10 4 B if bit 28, else 24:24:16
//! 8 B), then n extra blocks when bit 30 (primary) is set: a packed frame
//! dword (normal octahedral 8:8, tangent 7:7, sign bit 30) and the previous
//! frame's position in the same format (16, 8 or 12 B). Without bit 30
//! the console writes every vertex's extra block at the same place, just
//! past the range (stride 0; the allocator 0x1427C2090 reserves no extra
//! bytes then): [`cpu`] leaves it out, nothing reads it.

use std::path::Path;

use xnw_shader::emit::{self, HostShader, Options};

/// `CS_PedStandardSkinning_0`: the smallest of the
/// 13 variants.
pub const STANDARD_0: u128 = 0xa45da7131ba3596484a96440c5f20afb;

/// What one pre-skin dispatch reads, as bytes from each view's first byte
/// (a read past a view's end is 0, as on the console).
#[derive(Clone, Debug, Default)]
pub struct Inputs<'a> {
    /// The 96-B constant buffer b7 (item+32..+127; ud5+0x100).
    pub b7: &'a [u8],
    /// Heap view b7+0x40: the geometry's vertex streams (raw), with the
    /// position bias at +0x60 and scale at +0x70.
    pub vb: &'a [u8],
    /// Heap view b7+0x44 decoded to one value per element (it is read with
    /// `buffer_load_format_x`): the bone remap.
    pub remap: &'a [u32],
    /// ud4+0x20: the bone palette, 48-B rows (3 x float4).
    pub palette: &'a [u8],
    /// ud4+0x120: 48-B rows; dword 7 of row b7+0x3c, >> 16, picks the t71 record.
    pub t7: &'a [u8],
    /// ud4+0: 416-B records; dword +0xa0 of the picked one indexes t78 (< 0: no mask).
    pub t71: &'a [u8],
    /// ud4+0xc0: 1,280-B hide masks (two halves of 20 x 256 bits).
    pub t78: &'a [u8],
    /// The second table (empty: not bound). b5 and b6: constant buffers;
    /// t73: 16-B rows of 8 half floats; t74 and t75: replacement positions
    /// of this and the previous frame; t76: position deltas.
    pub b5: &'a [u8],
    pub b6: &'a [u8],
    pub t73: &'a [u8],
    pub t74: &'a [u8],
    pub t75: &'a [u8],
    pub t76: &'a [u8],
}

/// What a dispatch writes: which output buffer (0 A, 1 B), the byte offset
/// in it and the bytes from there (header, positions, extra blocks).
#[derive(Clone, Debug, PartialEq)]
pub struct Written {
    pub buffer: u8,
    pub offset: u32,
    pub bytes: Vec<u8>,
}

/// Position formats (range word bits 29 and 28).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PosFormat {
    Float3,
    Packed10,
    Packed24,
}

/// The range word's fields: buffer, byte offset, format, primary.
pub fn range(word: u32) -> (u8, u32, PosFormat, bool) {
    let format = if word & 0x2000_0000 != 0 {
        PosFormat::Float3
    } else if word & 0x1000_0000 != 0 {
        PosFormat::Packed10
    } else {
        PosFormat::Packed24
    };
    (u8::from(word >> 24 & 3 != 0), (word << 4) & 0x0fff_fff0, format, word & 0x4000_0000 != 0)
}

/// Bytes per position and per extra block (0 without the primary bit).
pub fn strides(format: PosFormat, primary: bool) -> (u32, u32) {
    let (pos, extra) = match format {
        PosFormat::Float3 => (12, 16),
        PosFormat::Packed10 => (4, 8),
        PosFormat::Packed24 => (8, 12),
    };
    (pos, if primary { extra } else { 0 })
}

fn dw(b: &[u8], at: u64) -> u32 {
    let at = at as usize;
    b.get(at..at + 4).map_or(0, |s| u32::from_le_bytes(s.try_into().unwrap_or_default()))
}

fn fl(b: &[u8], at: u64) -> f32 {
    f32::from_bits(dw(b, at))
}

/// `v_mul_legacy_f32`: 0 times anything is 0.
fn ml(a: f32, b: f32) -> f32 {
    if a == 0.0 || b == 0.0 { 0.0 } else { a * b }
}

/// `v_max_f32 d, -x, -x clamp`: max(0, -x) clamped to [0, 1] (NaN 0).
fn clamp01(x: f32) -> f32 {
    if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) }
}

/// The octahedral fold shared by both decodes: `v += (v >= 0 ? -t : t)`.
fn fold(v: f32, t: f32) -> f32 {
    (if v >= 0.0 { -t } else { t }) + v
}

/// The vertex's two frame vectors from its frame word (0x4dc-0x6f4):
/// `n` from bytes 0/1 (or the 10:10 octahedral pair), `t` from bits
/// 16-29 (or the angle form when bit 31 is set).
fn frame(word: u32) -> ([f32; 3], [f32; 3]) {
    if (word as i32) < 0 {
        // 10:10 octahedral normal and a 10-bit angle (0x4dc-0x5f0).
        let x = (word & 0x3ff) as f32 * f32::from_bits(0x3b00_2008) - 1.0;
        let y = (word >> 10 & 0x3ff) as f32 * f32::from_bits(0x3b00_2008) - 1.0;
        let a = ((word >> 20 & 0x3ff) as f32).mul_add(f32::from_bits(0x3bc9_422c), f32::from_bits(0xc049_0fdb));
        let z = 1.0 - x.abs() - y.abs();
        let t = clamp01(-z);
        let (y, x) = (fold(y, t), fold(x, t));
        let r = 1.0 / (y * y + x * x + z * z).sqrt();
        let (nz, ny, nx) = (r * z, y * r, x * r);
        let s = if nz >= 0.0 { 1.0 } else { -1.0 };
        let k = 1.0 / r.mul_add(z, s);
        let rev = a.mul_add(0.159_154_94, -ml(0.159_154_94, a).trunc());
        let (sin, cos) = ((rev * std::f32::consts::TAU).sin(), (rev * std::f32::consts::TAU).cos());
        let q = ml(ny, k);
        let sx = ml(s, nx);
        let qx = ml(q, nx);
        let b = ml(k, nx).mul_add(sx, -1.0);
        let c = q.mul_add(ny, -s);
        let t28 = ml(s, qx).mul_add(cos, -ml(c, sin));
        let t9 = b.mul_add(cos, -ml(qx, sin));
        let t8 = sx.mul_add(cos, -ml(ny, sin));
        ([nx, ny, nz], [t9, t28, t8])
    } else {
        // Two octahedral pairs, 8:8 and 7:7 (0x600-0x6f4).
        let u = |v: u32, k: u32| v as f32 * f32::from_bits(k) - 1.0;
        let (x, y) = (u(word & 0xff, 0x3c00_8081), u(word >> 8 & 0xff, 0x3c00_8081));
        let (p, q) = (u(word >> 16 & 0x7f, 0x3c81_0204), u(word >> 23 & 0x7f, 0x3c81_0204));
        let z = 1.0 - x.abs() - y.abs();
        let w = 1.0 - p.abs() - q.abs();
        let (tz, tw) = (clamp01(-z), clamp01(-w));
        let (y, x, q, p) = (fold(y, tz), fold(x, tz), fold(q, tw), fold(p, tw));
        let r1 = 1.0 / (y * y + x * x + z * z).sqrt();
        let r2 = 1.0 / (q * q + p * p + w * w).sqrt();
        ([x * r1, y * r1, r1 * z], [p * r2, q * r2, r2 * w])
    }
}

/// A blended 3x4 matrix (rows of 4).
type M34 = [[f32; 4]; 3];

/// Palette row `i` (48 B).
fn row(palette: &[u8], i: u32) -> M34 {
    let at = u64::from(i) * 48;
    std::array::from_fn(|r| std::array::from_fn(|c| fl(palette, at + 16 * r as u64 + 4 * c as u64)))
}

/// `sum w_k M_k` as the programs accumulate it: bone 0 always; with `stop`,
/// bone k only while every weight before it is non-zero (variant 0
/// 0xb90-0xd08; variant 3 0xd44-0x103c over eight), else every bone
/// (variant 5's loop 0xec8-0x1108, which adds `w_k (M_k p - M_0 p)` to
/// `M_0 p`: the same sum once the weights add up to 1).
fn blend(palette: &[u8], base: u32, bones: &[u32], w: &[f32], stop: bool) -> M34 {
    let mut m = row(palette, base.wrapping_add(bones[0]));
    for r in &mut m {
        for v in r.iter_mut() {
            *v = ml(*v, w[0]);
        }
    }
    for k in 1..bones.len() {
        if w[k] == 0.0 {
            if stop {
                break;
            }
            continue;
        }
        let b = row(palette, base.wrapping_add(bones[k]));
        for r in 0..3 {
            for c in 0..4 {
                m[r][c] = b[r][c].mul_add(w[k], m[r][c]);
            }
        }
    }
    m
}

/// How a variant gathers its bones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bones {
    /// Variant 0: one index/weight pair of streams (b7+0xc, +0x10).
    Four,
    /// Variants 3 and 4: a second pair (b7+0x14, +0x18), the eight weights
    /// divided by their sum, the chain stopping at the first zero weight
    /// (a1bddae1 0xa58-0x103c). Variant 4's extra block (54afc93e
    /// 0x14f8-0x3638) acts only on vertices whose fourth index dword has
    /// 0xb2 in its top byte, which a 4-byte index stream cannot carry.
    Eight,
    /// Variants 5 and 9: the same eight, every one applied (0d96d9b4
    /// 0xec8-0x1108), and the frame vectors corrected from the stream at
    /// b7+0x1c ([`frame_all`]).
    EightAll,
    /// Variant 11 with bit 1 of b7+0x5c set and bit 0 set too: the four
    /// bones of the first pair, every one applied (659b134d 0x1fdc-0x2234).
    FourAll,
}

/// What a variant does to a vertex before the bones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Before {
    Nothing,
    /// Variant 11 (659b134d 0xb00-0xf70, 0x12c4-0x1ec4): vertices tagged in
    /// the top byte of the fourth index dword take their position from a
    /// table (t74, the previous frame's from t75; tag 0xb6 by the entry in
    /// the second dword's high half, with a normal; tag 0x54 by a 16-bit
    /// entry spread over dwords 2 and 3, plus three half-float offsets in
    /// thousandths) or, when b6+0xc is 0, add a delta from t76.
    Tables,
    /// Variant 2 (448e13a4 0xa68-0xb40, float entries at 0x1090): t76's
    /// delta step of variant 11 alone, with no b6 to choose it: a vertex
    /// tagged 0xb6 in the top byte of its fourth index dword (a 16-B index
    /// stream) adds entry `(dword 2 >> 16) - 1` of t76 to its position (and
    /// its previous position) when t76 is bound.
    Delta,
}

/// A variant's rule: how it gathers bones and what comes before them.
/// Variant 11 chooses its bones by b7+0x5c: bit 0 clear four bones, set
/// eight (the second pair's layout is variant 3's by inference); bit 1 set
/// every bone applied. None: a variant this reference does not cover.
pub fn rule_of(variant: u32, b7: &[u8]) -> Option<(Bones, Before)> {
    if variant == 11 {
        let bones = match dw(b7, 0x5c) & 3 {
            0 => Bones::Four,
            2 => Bones::FourAll,
            1 => Bones::Eight,
            _ => Bones::EightAll,
        };
        return Some((bones, Before::Tables));
    }
    bones_of(variant).map(|b| (b, Before::Nothing))
}

/// The rule of a standard variant past the covered ones, read from the
/// resources and code each program adds to a covered one (sizes are native
/// code bytes, from `shader-info`): 1 is 0 (13,376) plus 4's
/// block (t72, t73, b5: 25,088); 2 is 0 plus t76's delta (14,400); 6 is 5
/// plus the block (36,544); 7, 8, 9, 10 are 3, 4, 5, 6 plus the
/// displacement maps (samplers, root 0 slots 2-5, 8, 12-22, root 1 slots 4,
/// 6; each about 29,000 more). The block moves no position (v4) and
/// the maps are not read here, so 7 to 10 are their base variant's maths
/// without the displacement. 6 is 5's maths (the block left out as for 4;
/// seen live only, a test run). None: 12 (not seen).
pub fn rule_more(variant: u32) -> Option<(Bones, Before)> {
    match variant {
        1 => Some((Bones::Four, Before::Nothing)),
        2 => Some((Bones::Four, Before::Delta)),
        7 => Some((Bones::Eight, Before::Nothing)),
        8 => Some((Bones::Eight, Before::Nothing)),
        6 | 9 | 10 => Some((Bones::EightAll, Before::Nothing)),
        _ => None,
    }
}

/// A half float as f32.
fn half(h: u32) -> f32 {
    let (sign, exp, man) = (h >> 15 & 1, h >> 10 & 0x1f, h & 0x3ff);
    let bits = match exp {
        0 if man == 0 => 0,
        0 => {
            let shift = man.leading_zeros() - 21;
            (113 - shift) << 23 | (man << shift & 0x3ff) << 13
        }
        31 => 0x7f80_0000 | man << 13,
        _ => (exp + 112) << 23 | man << 13,
    };
    f32::from_bits(sign << 31 | bits)
}

/// A 24:24:16 position through a bias, a scale and the axis order word
/// (0x41c-0x4dc; the tables of variant 11 carry their own three).
fn unpack_pos(d0: u32, d1: u32, bias: [f32; 3], scale: [f32; 3], order: u32) -> [f32; 3] {
    let q = [(d0 >> 8) as f32, (d0 & 0xff | (d1 & 0xffff) << 8) as f32, (d1 >> 16) as f32];
    let x2 = scale[0].mul_add(q[0], bias[0]);
    let x3 = scale[1].mul_add(q[1], bias[1]);
    let x4 = scale[2].mul_add(q[2], bias[2]);
    match order {
        0 => [x4, x3, x2],
        1 => [x3, x4, x2],
        _ => [x2, x3, x4],
    }
}

/// Entry `e` of a position table whose header (bias, axis order, scale: 28
/// B) stands at byte `base` and whose 8-B entries follow it.
pub fn table_pos(t: &[u8], base: u32, e: u32) -> [f32; 3] {
    let b = u64::from(base);
    let at = b + 28 + 8 * u64::from(e);
    let f3 = |o: u64| [fl(t, b + o), fl(t, b + o + 4), fl(t, b + o + 8)];
    unpack_pos(dw(t, at), dw(t, at + 4), f3(0), f3(16), dw(t, b + 12))
}

/// Whether the position table a dispatch names (b6+0xc set, header at byte
/// b6+0 of t74) has a zero scale: every entry then decodes to the bias, so
/// the table holds no shape. The console fills that buffer on the GPU; a
/// title memory read without those passes finds it zero.
pub fn table_unfilled(inp: &Inputs) -> bool {
    let base = u64::from(dw(inp.b6, 0));
    dw(inp.b6, 0xc) != 0 && !inp.b6.is_empty() && (0..3).all(|k| fl(inp.t74, base + 16 + 4 * k) == 0.0)
}

/// Variant 11's step before the bones on one vertex: `ib` the four raw
/// index values, `p` the position (returned with the previous frame's),
/// `nv`/`tv` the frame vectors.
fn tables(inp: &Inputs, ib: [u32; 4], p: [f32; 3], nv: &mut [f32; 3], tv: &mut [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let tag = ib[3] & 0xff00_0000;
    if dw(inp.b6, 0xc) == 0 {
        let moved = delta(inp.t76, ib, p);
        return (moved, moved);
    }
    if inp.t74.is_empty() {
        return (p, p);
    }
    let (base, normals) = (dw(inp.b6, 0), dw(inp.b6, 4));
    let (mut cur, mut prev) = (p, p);
    // 0xb3c-0xd6c: tag 0xb6.
    let e = if tag != 0xb600_0000 || ib[1] == 0xffff { 0xffff } else { (ib[1] >> 16).wrapping_sub(1) };
    if e != 0xffff {
        cur = table_pos(inp.t74, base, e);
        prev = table_pos(inp.t75, base, e);
        if ib[0] < 0xffff_0000 {
            let w = dw(inp.t74, u64::from(base.wrapping_add(normals)) + 4 * u64::from(e));
            let k = f32::from_bits(0x3b00_2008);
            let n = [(w & 0x3ff) as f32 * k - 1.0, (w >> 10 & 0x3ff) as f32 * k - 1.0, (w >> 20 & 0x3ff) as f32 * k - 1.0];
            // t = [v10, v7, v6]; the dot starts at its second component.
            let d = tv[2].mul_add(n[2], tv[0].mul_add(n[0], ml(tv[1], n[1])));
            let t = [(-n[0]).mul_add(d, tv[0]), (-n[1]).mul_add(d, tv[1]), (-n[2]).mul_add(d, tv[2])];
            let r = 1.0 / t[2].mul_add(t[2], t[0].mul_add(t[0], ml(t[1], t[1]))).sqrt();
            *tv = [ml(t[0], r), ml(t[1], r), ml(t[2], r)];
            *nv = n;
        }
    }
    // 0xd74-0xf50: tag 0x54.
    if tag == 0x5400_0000 {
        let e = ib[3] >> 12 & 0xfff | ib[2] >> 16 & 0xf000;
        let k = f32::from_bits(0x3a83_126f);
        let off = [half(ib[0] >> 12 & 0xffff) * k, half(ib[1] >> 12 & 0xffff) * k, half(ib[2] >> 12 & 0xffff) * k];
        let add = |q: [f32; 3]| [q[0] + off[0], q[1] + off[1], q[2] + off[2]];
        cur = add(table_pos(inp.t74, base, e));
        prev = add(table_pos(inp.t75, base, e));
    }
    (cur, prev)
}

/// HAIR3: the position table's entries a variant-11 dispatch reads, as its
/// own bind positions (no blend shape): (entry, position) per vertex tagged
/// 0xb6 (entry = high half of index dword 1 - 1, as [`tables`] reads it).
/// The console fills the table on the GPU (the head compute graph);
/// a capture holds it zero. The strand chain's `CS_UpdatePosePosition`
/// reads the same table (frame table slot 7 at record +0x13c) for the
/// scalp vertices its guides grow from.
pub fn neutral_table(inp: &Inputs) -> Vec<(u32, [f32; 3])> {
    let c = |off: u64| dw(inp.b7, off);
    let stream = |desc: u32, v: u32| u64::from(desc >> 8) + u64::from(desc & 0xff) * u64::from(v);
    let (bias, swizzle, scale) = ([fl(inp.vb, 0x60), fl(inp.vb, 0x64), fl(inp.vb, 0x68)], dw(inp.vb, 0x6c), [fl(inp.vb, 0x70), fl(inp.vb, 0x74), fl(inp.vb, 0x78)]);
    let idesc = c(0xc);
    let mut out = Vec::new();
    for v in 0..c(0x38) {
        let ib = if idesc & 0xff == 16 {
            let at = u64::from(idesc >> 8) + 16 * u64::from(v);
            [dw(inp.vb, at), dw(inp.vb, at + 4), dw(inp.vb, at + 8), dw(inp.vb, at + 12)]
        } else {
            let w = dw(inp.vb, stream(idesc, v));
            [w & 0xff, w >> 8 & 0xff, w >> 16 & 0xff, w >> 24]
        };
        if ib[3] & 0xff00_0000 != 0xb600_0000 || ib[1] == 0xffff {
            continue;
        }
        let at = stream(c(0), v);
        out.push(((ib[1] >> 16).wrapping_sub(1), unpack_pos(dw(inp.vb, at), dw(inp.vb, at + 4), bias, scale, swizzle)));
    }
    out
}

/// A position table at byte `base` of `t` (grown as needed) holding
/// `entries`: the header (bias, axis order 2 = as is, scale; 28 B) and the
/// 24:24:16 entries [`table_pos`] reads, over +-3 m.
pub fn put_table(t: &mut Vec<u8>, base: u32, entries: &[(u32, [f32; 3])]) {
    let (b, top) = (base as usize, entries.iter().map(|e| e.0 + 1).max().unwrap_or(0) as usize);
    if t.len() < b + 28 + 8 * top {
        t.resize(b + 28 + 8 * top, 0);
    }
    let scale = [6.0 / 16_777_215.0, 6.0 / 16_777_215.0, 6.0 / 65_535.0f32];
    for (k, w) in [(-3.0f32).to_bits(), (-3.0f32).to_bits(), (-3.0f32).to_bits(), 2, scale[0].to_bits(), scale[1].to_bits(), scale[2].to_bits()].iter().enumerate() {
        t[b + 4 * k..b + 4 * k + 4].copy_from_slice(&w.to_le_bytes());
    }
    for &(e, p) in entries {
        let q = |v: f32, k: usize, max: f32| ((v + 3.0) / scale[k]).round().clamp(0.0, max) as u32;
        let (x, y, z) = (q(p[0], 0, 16_777_215.0), q(p[1], 1, 16_777_215.0), q(p[2], 2, 65_535.0));
        let at = b + 28 + 8 * e as usize;
        t[at..at + 4].copy_from_slice(&(x << 8 | y & 0xff).to_le_bytes());
        t[at + 4..at + 8].copy_from_slice(&(y >> 8 & 0xffff | z << 16).to_le_bytes());
    }
}

/// t76's delta on a position (variant 11 0x12c4-0x1ec4, variant 2
/// 0xa68-0xb40): a vertex tagged 0xb6 adds entry `(ib[2] >> 16) - 1`,
/// packed 8 B with the table's own bias and scale when its first dword is
/// set, else three floats; the entries start at byte 32.
fn delta(t: &[u8], ib: [u32; 4], p: [f32; 3]) -> [f32; 3] {
    let e = if ib[3] & 0xff00_0000 == 0xb600_0000 { (ib[2] >> 16).wrapping_sub(1) } else { 0xffff };
    if t.is_empty() || e >= 0xffff {
        return p;
    }
    let d = if dw(t, 0) != 0 {
        let at = 32 + 8 * u64::from(e);
        let (d0, d1) = (dw(t, at), dw(t, at + 4));
        let q = [(d0 >> 8) as f32, (d0 & 0xff | (d1 & 0xffff) << 8) as f32, (d1 >> 16) as f32];
        [q[0].mul_add(fl(t, 16), fl(t, 4)), q[1].mul_add(fl(t, 20), fl(t, 8)), q[2].mul_add(fl(t, 24), fl(t, 12))]
    } else {
        let at = 32 + 12 * u64::from(e);
        [fl(t, at), fl(t, at + 4), fl(t, at + 8)]
    };
    [d[0] + p[0], d[1] + p[1], d[2] + p[2]]
}

/// The bone rule of `CS_PedStandardSkinning_<variant>` (None: a variant
/// this reference does not cover).
pub fn bones_of(variant: u32) -> Option<Bones> {
    match variant {
        0 => Some(Bones::Four),
        3 | 4 => Some(Bones::Eight),
        5 => Some(Bones::EightAll),
        _ => None,
    }
}

/// `m * (p, 1)`, the program's operand order (0xd18-0xdc0).
fn apply(m: &M34, p: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|r| m[r][2].mul_add(p[2], m[r][1].mul_add(p[1], ml(m[r][0], p[0]))) + m[r][3])
}

/// `m`'s 3x3 part times `v`, normalised (0xd14-0xdd8).
fn turn(m: &M34, v: [f32; 3]) -> [f32; 3] {
    let t: [f32; 3] = std::array::from_fn(|r| m[r][2].mul_add(v[2], m[r][1].mul_add(v[1], ml(m[r][0], v[0]))));
    let k = 1.0 / (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
    [ml(k, t[0]), ml(k, t[1]), ml(k, t[2])]
}

/// [`turn`] as the variants past 0 round it (a1bddae1 0x1044-0x10f4).
fn turn_fused(m: &M34, v: [f32; 3]) -> [f32; 3] {
    let t: [f32; 3] = std::array::from_fn(|r| m[r][2].mul_add(v[2], m[r][1].mul_add(v[1], ml(m[r][0], v[0]))));
    let k = 1.0 / t[2].mul_add(t[2], t[0].mul_add(t[0], ml(t[1], t[1]))).sqrt();
    [ml(k, t[0]), ml(k, t[1]), ml(k, t[2])]
}

/// The frame of the variants that apply every bone (0d96d9b4
/// 0xc94-0x1258). The stream at b7+0x1c holds, per vertex, a half-float
/// magnitude and seven byte pairs (a_k, b_k), one per bone after the first:
/// the change of that bone's weight along the tangent and the bitangent.
/// With d_k = M_k p - M_0 p: the tangent is `M t + min(1, 0.9 / |A|) A`,
/// A = sum a_k d_k; the bitangent `M (h t x n) - min(1, 0.9 / |B|) B`,
/// B = sum b_k d_k; the normal `h (bitangent x tangent)`, all normalised
/// (`h`: -1 when bit 30 of the frame word is set). Returns (normal, tangent).
#[allow(clippy::too_many_arguments)]
fn frame_all(inp: &Inputs, at: u64, palette_base: u32, bones: &[u32], m: &M34, p: [f32; 3], n: [f32; 3], t: [f32; 3], h: f32) -> ([f32; 3], [f32; 3]) {
    let f = f32::from_bits;
    let d0 = dw(inp.vb, at);
    let mag = { let x = half(d0 & 0xffff); if x > 0.0 { x } else { 0.0 } };
    let byte = |i: u64| dw(inp.vb, at + (i & !3)) >> (8 * (i & 3)) & 0xff;
    let coef = |i: u64| if i < 4 { ml(f(0x3b80_8081).mul_add(byte(i) as f32, f(0xbf00_8081)), mag) } else { ml(mag, byte(i) as f32 / 255.0 + f(0xbf00_8081)) };
    let dot = |r: &[f32; 4], v: [f32; 3]| r[2].mul_add(v[2], r[1].mul_add(v[1], ml(r[0], v[0])));
    let p0 = apply(&row(inp.palette, palette_base.wrapping_add(bones[0])), p);
    let (mut a, mut b) = ([0f32; 3], [0f32; 3]);
    for k in 1..bones.len() {
        let mk = row(inp.palette, palette_base.wrapping_add(bones[k]));
        let (ca, cb) = (coef(2 * k as u64), coef(2 * k as u64 + 1));
        for r in 0..3 {
            let d = (mk[r][3] - p0[r]) + dot(&mk[r], p);
            a[r] = ca.mul_add(d, a[r]);
            b[r] = cb.mul_add(d, b[r]);
        }
    }
    let len2 = |v: [f32; 3]| v[2].mul_add(v[2], v[0].mul_add(v[0], ml(v[1], v[1])));
    let (sa, sb) = (ml(f(0x3f66_6666), 1.0 / len2(a).sqrt()).min(1.0), ml(f(0x3f66_6666), 1.0 / len2(b).sqrt()).min(1.0));
    // h (t x n), then both through the blended matrix.
    let bi = [ml(n[2].mul_add(t[1], -ml(n[1], t[2])), h), ml(n[0].mul_add(t[2], -ml(n[2], t[0])), h), ml(n[1].mul_add(t[0], -ml(n[0], t[1])), h)];
    let tan: [f32; 3] = std::array::from_fn(|r| sa.mul_add(a[r], dot(&m[r], t)));
    let bit: [f32; 3] = std::array::from_fn(|r| (-sb).mul_add(b[r], dot(&m[r], bi)));
    let unit = |v: [f32; 3], k: f32| [ml(k, v[0]), ml(k, v[1]), ml(k, v[2])];
    let (tan, bit) = (unit(tan, 1.0 / len2(tan).sqrt()), unit(bit, 1.0 / len2(bit).sqrt()));
    let c = [bit[1].mul_add(tan[2], -ml(bit[2], tan[1])), bit[2].mul_add(tan[0], -ml(bit[0], tan[2])), bit[0].mul_add(tan[1], -ml(bit[1], tan[0]))];
    (unit(c, ml(1.0 / len2(c).sqrt(), h)), tan)
}

/// The frame word written to the extra block (0x2370-0x2474): `n` as
/// octahedral 8:8 in bits 0-15, `t` as 7:7 in bits 16-29, `sign` (bit 30
/// of the input word) kept.
fn pack_frame(n: [f32; 3], t: [f32; 3], sign: u32) -> u32 {
    let oct = |v: [f32; 3]| {
        let l = 1.0 / (v[0].abs() + v[1].abs() + v[2].abs());
        let z = clamp01(ml(v[2], -l));
        let (x, y) = (ml(v[0], l), ml(v[1], l));
        ((if x >= 0.0 { z } else { -z }) + x, (if y >= 0.0 { z } else { -z }) + y)
    };
    let q = |v: f32, k: f32| clamp01(v.mul_add(0.5, 0.5)).mul_add(k, 0.5) as u32;
    let (nx, ny) = oct(n);
    let (tx, ty) = oct(t);
    q(nx, 255.0) | sign | q(ny, 255.0) << 8 | q(tx, 127.0) << 16 | q(ty, 127.0) << 23
}

/// The frame word of the variants past 0 (a1bddae1 0x30d8-0x3d80): the
/// frame is packed both ways, as [`pack_frame`]'s two octahedral pairs and
/// as a 10:10 octahedral normal with the tangent's 10-bit angle about it
/// (bit 31 set), each is decoded again, and the word whose normal and
/// tangent lie closer to the inputs is kept (`(2 - n.n') - t.t'`; the
/// angle form only when strictly closer).
fn pack_frame_best(n: [f32; 3], t: [f32; 3], sign: u32) -> u32 {
    let f = f32::from_bits;
    let side = |v: f32, z: f32| if 0.0 <= v { z } else { -z };
    let rsq = |x: f32| 1.0 / x.sqrt();
    // The normal's octahedral pair, kept unquantised for the angle's basis.
    let l = 1.0 / (n[1].abs() + n[0].abs() + n[2].abs());
    let zc = clamp01(ml(n[2], -l));
    let ox = n[0].mul_add(l, side(ml(n[0], l), zc));
    let oy = n[1].mul_add(l, side(ml(n[1], l), zc));
    let z = (1.0 - ox.abs()) - oy.abs();
    let tt = clamp01(-z);
    let (fy, fx) = (side(oy, -tt) + oy, side(ox, -tt) + ox);
    let (uy, ux) = (clamp01(oy.mul_add(0.5, 0.5)), clamp01(ox.mul_add(0.5, 0.5)));
    let (qy10, qx10) = (1023f32.mul_add(uy, 0.5) as u32, 1023f32.mul_add(ux, 0.5) as u32);
    let (qy8, qx8) = (255f32.mul_add(uy, 0.5) as u32, 255f32.mul_add(ux, 0.5) as u32);
    let r = rsq(z.mul_add(z, fx.mul_add(fx, ml(fy, fy))));
    let (nz, ny, nx) = (ml(r, z), ml(fy, r), ml(fx, r));
    let k10 = f(0x3b00_2008);
    let dx = k10.mul_add((qx10 & 0x3ff) as f32, -1.0);
    let s = if 0.0 <= nz { 1.0 } else { -1.0 };
    let k = 1.0 / r.mul_add(z, s);
    let snx = ml(s, nx);
    let q = ml(ny, k);
    let qx = ml(q, nx);
    let b = ml(k, nx).mul_add(snx, -1.0);
    let c = q.mul_add(ny, -s);
    let lt = 1.0 / (t[1].abs() + t[0].abs() + t[2].abs());
    // The tangent's angle in that basis: acos by a cubic, signed by the
    // second basis vector, in turns about 0.5.
    let cosv = t[2].mul_add(snx, t[0].mul_add(b, ml(t[1], ml(s, qx))));
    let dy = k10.mul_add((qy10 & 0x3ff) as f32, -1.0);
    let xc = if cosv.is_nan() { -1.0 } else { cosv.clamp(-1.0, 1.0) };
    let ax = xc.abs();
    let poly = ax.mul_add(ax.mul_add(ax.mul_add(f(0xbc99_6e30), f(0x3d98_1627)), f(0xbe59_3484)), f(0x3fc9_0da4));
    let root = (1.0 - ax).sqrt();
    let d2 = t[2].mul_add(ny, t[0].mul_add(qx, ml(t[1], c)));
    let angle = if 0.0 <= xc { root * poly } else { (-root).mul_add(poly, f(0x4049_0fdb)) };
    let turns = if 0.0 > d2 { 0.159_154_94 } else { f(0xbe22_f983) };
    let qa = 1023f32.mul_add(clamp01(turns.mul_add(angle, 0.5)), 0.5) as u32;
    let angled = 0x8000_0000 | sign | qa << 20 | qy10 << 10 | qx10;
    // The tangent's octahedral pair.
    let tzc = clamp01(ml(t[2], -lt));
    let tox = t[0].mul_add(lt, side(ml(t[0], lt), tzc));
    let toy = t[1].mul_add(lt, side(ml(t[1], lt), tzc));
    let q7 = |v: f32| 127f32.mul_add(clamp01(v.mul_add(0.5, 0.5)), 0.5) as u32;
    let (qtx, qty) = (q7(tox), q7(toy));
    let pairs = qty << 23 | qtx << 16 | qy8 << 8 | qx8 | sign;
    // The angle form decoded.
    let dz = (1.0 - dx.abs()) - dy.abs();
    let td = clamp01(-dz);
    let (fdy, fdx) = (side(dy, -td) + dy, side(dx, -td) + dx);
    let rd = rsq(dz.mul_add(dz, fdx.mul_add(fdx, ml(fdy, fdy))));
    let (by, bz, bx) = (ml(fdy, rd), ml(rd, dz), ml(fdx, rd));
    let s2 = if 0.0 <= bz { 1.0 } else { -1.0 };
    let a2 = f(0x3bc9_422c).mul_add(qa as f32, f(0xc049_0fdb));
    let k2 = 1.0 / rd.mul_add(dz, s2);
    let rev = a2.mul_add(0.159_154_94, -ml(0.159_154_94, a2).trunc());
    let (sin, cos) = ((rev * std::f32::consts::TAU).sin(), (rev * std::f32::consts::TAU).cos());
    let q2 = ml(by, k2);
    let s2nx = ml(s2, bx);
    let qx2 = ml(q2, bx);
    let b2 = ml(k2, bx).mul_add(s2nx, -1.0);
    let c2 = q2.mul_add(by, -s2);
    let tb = [b2.mul_add(cos, -ml(qx2, sin)), ml(s2, qx2).mul_add(cos, -ml(c2, sin)), s2nx.mul_add(cos, -ml(by, sin))];
    // The two pairs decoded.
    let (x, y) = (f(0x3c00_8081).mul_add(qx8 as f32, -1.0), f(0x3c00_8081).mul_add(qy8 as f32, -1.0));
    let (p, q) = (f(0x3c81_0204).mul_add(qtx as f32, -1.0), f(0x3c81_0204).mul_add(qty as f32, -1.0));
    let (az, aw) = ((1.0 - x.abs()) - y.abs(), (1.0 - p.abs()) - q.abs());
    let (tz, tw) = (clamp01(-az), clamp01(-aw));
    let (ay, aq, ax2, ap) = (side(y, -tz) + y, side(q, -tw) + q, side(x, -tz) + x, side(p, -tw) + p);
    let r1 = rsq(az.mul_add(az, ax2.mul_add(ax2, ml(ay, ay))));
    let r2 = rsq(aw.mul_add(aw, ap.mul_add(ap, ml(aq, aq))));
    let na = [ml(ax2, r1), ml(ay, r1), ml(r1, az)];
    let ta = [ml(ap, r2), ml(aq, r2), ml(r2, aw)];
    // Distances; the handedness (bit 30) enters both tangent sums alike.
    let h = if sign >> 30 & 1 == 1 { -1.0f32 } else { 1.0 };
    let dot_n = |m: [f32; 3]| n[2].mul_add(m[2], n[0].mul_add(m[0], ml(n[1], m[1])));
    let dot_t = |m: [f32; 3]| t[2].mul_add(m[2], h.mul_add(h, t[0].mul_add(m[0], ml(t[1], m[1]))));
    let far_angled = (2.0 - dot_n([bx, by, bz])) - dot_t(tb);
    let far_pairs = (2.0 - dot_n(na)) - dot_t(ta);
    if far_angled < far_pairs { angled } else { pairs }
}

/// A position in the output format (0xfb8-0x104c, 0x207c-0x20ec).
fn put_pos(out: &mut Vec<u8>, p: [f32; 3], format: PosFormat) {
    match format {
        PosFormat::Float3 => p.iter().for_each(|v| out.extend(v.to_le_bytes())),
        PosFormat::Packed10 => {
            let q = |v: f32| clamp01(ml(f32::from_bits(0x3e2a_aaab), v + 3.0)).mul_add(1023.0, 0.5) as u32;
            out.extend((q(p[0]) | q(p[1]) << 10 | q(p[2]) << 20 | 0x8000_0000).to_le_bytes());
        }
        PosFormat::Packed24 => {
            let q = |v: f32, k: u32, max: u32| {
                let x = ml(f32::from_bits(k), v + 3.0);
                if x.is_nan() { 0 } else { x.clamp(0.0, f32::from_bits(max)) as u32 }
            };
            let (x, y, z) = (q(p[0], 0x4a2a_aaaa, 0x4b7f_ffff), q(p[1], 0x4a2a_aaaa, 0x4b7f_ffff), q(p[2], 0x462a_aa00, 0x477f_ff00) & 0xffff);
            out.extend((y & 0xff | x << 8).to_le_bytes());
            out.extend((y >> 8 & 0xffff | z << 16).to_le_bytes());
        }
    }
}

/// Bit `i` of the 256-bit mask at byte `at` of t78 (record `rec`), when
/// `set < 20` and `i < 256` (0x21c-0x3c4).
fn masked(t78: &[u8], rec: u32, set: u32, half: u64, i: u32) -> bool {
    if set >= 20 || i >= 256 {
        return false;
    }
    let at = u64::from(rec) * 1280 + half + u64::from(set) * 32 + u64::from(i >> 5) * 4;
    dw(t78, at) >> (i & 31) & 1 != 0
}

/// `CS_PedStandardSkinning_0` on the CPU: what the dispatch writes.
pub fn cpu(inp: &Inputs) -> Written {
    cpu_with(inp, Bones::Four)
}

/// A standard variant on the CPU under its bone rule ([`bones_of`]).
pub fn cpu_with(inp: &Inputs, rule: Bones) -> Written {
    cpu_rule(inp, rule, Before::Nothing)
}

/// A standard variant on the CPU under its whole rule ([`rule_of`]).
pub fn cpu_rule(inp: &Inputs, rule: Bones, before: Before) -> Written {
    let cb = inp.b7;
    let c = |off: u64| dw(cb, off);
    let word = c(0x34);
    let n = c(0x38);
    let (buffer, offset, format, primary) = range(word);
    let (pos_stride, extra_stride) = strides(format, primary);
    let extra_at = 32 + n * pos_stride;
    let size = extra_at + n * extra_stride;
    let mut out = vec![0u8; size as usize];
    out[..8].copy_from_slice(&[n.to_le_bytes(), 0xdead_beefu32.to_le_bytes()].concat());
    // Per-dispatch constants (0x30-0x1dc).
    let bias = [fl(inp.vb, 0x60), fl(inp.vb, 0x64), fl(inp.vb, 0x68)];
    let swizzle = dw(inp.vb, 0x6c);
    let scale = [fl(inp.vb, 0x70), fl(inp.vb, 0x74), fl(inp.vb, 0x78)];
    let pick = dw(inp.t7, u64::from(c(0x3c)).wrapping_mul(48) + 28) >> 16;
    let rec = dw(inp.t71, u64::from(pick) * 416 + 0xa0);
    let (sets, mask_stream) = ((c(0x20), c(0x24)), c(0x8));
    let masks = (rec as i32) >= 0 && mask_stream != 0 && sets.0 & sets.1 != u32::MAX;
    let stream = |desc: u32, v: u32| u64::from(desc >> 8) + u64::from(desc & 0xff) * u64::from(v);
    // Only variants 0, 1 and 2 (four bones, no table before them) pack the
    // frame one way (oracle: 0 of 1,938 and 0 of 1,680 words differ, pk5 f975).
    let best = !(rule == Bones::Four && before != Before::Tables);
    for v in 0..n {
        let pos_at = (32 + v * pos_stride) as usize;
        if masks {
            let b = dw(inp.vb, stream(mask_stream, v));
            let hidden = masked(inp.t78, rec, sets.0, 0, (b & 0xff).wrapping_sub(1)) || masked(inp.t78, rec, sets.1, 640, (b >> 8 & 0xff).wrapping_sub(1));
            if hidden {
                // 0x3dc-0x410, 0x2060-0x21f4: all ones, no extra block.
                out[pos_at..pos_at + pos_stride as usize].fill(0xff);
                continue;
            }
        }
        // Position, 24:24:16 (0x41c-0x4a8).
        let at = stream(c(0), v);
        let (d0, d1) = (dw(inp.vb, at), dw(inp.vb, at + 4));
        let p = unpack_pos(d0, d1, bias, scale, swizzle);
        let fw = dw(inp.vb, stream(c(4), v));
        let (mut nv, mut tv) = frame(fw);
        // Bones (0xa0c-0xafc): four indices through the remap, four unorm8
        // weights; a second set of each under the eight-bone rules. Both
        // index streams are read by the first one's stride test.
        let wide = c(0xc) & 0xff == 16;
        let four = |idesc: u32| -> [u32; 4] {
            if wide {
                let at = u64::from(idesc >> 8) + 16 * u64::from(v);
                [dw(inp.vb, at), dw(inp.vb, at + 4), dw(inp.vb, at + 8), dw(inp.vb, at + 12)]
            } else {
                let w = dw(inp.vb, stream(idesc, v));
                [w & 0xff, w >> 8 & 0xff, w >> 16 & 0xff, w >> 24]
            }
        };
        let (p, p_prev) = match before {
            Before::Nothing => (p, p),
            Before::Tables => tables(inp, four(c(0xc)), p, &mut nv, &mut tv),
            Before::Delta => {
                let moved = delta(inp.t76, four(c(0xc)), p);
                (moved, moved)
            }
        };
        let sets = if matches!(rule, Bones::Four | Bones::FourAll) { 1 } else { 2 };
        let (mut bones, mut b) = ([0u32; 8], [0f32; 8]);
        for set in 0..sets {
            let ib = four(c(0xc + 8 * set as u64));
            let ww = dw(inp.vb, stream(c(0x10 + 8 * set as u64), v));
            for k in 0..4 {
                bones[4 * set + k] = inp.remap.get(((ib[k] & 0xfff) + c(0x30)) as usize).copied().unwrap_or(0) & 0xfff;
                b[4 * set + k] = (ww >> (8 * k) & 0xff) as f32;
            }
        }
        let n_bones = 4 * sets;
        let k = f32::from_bits(0x3b80_8081);
        let sum = if sets == 1 { b[1] + b[0] + b[2] + b[3] } else { b[1] + b[0] + b[2] + b[3] + b[4] + b[5] + b[6] + b[7] };
        let r = 1.0 / ml(k, sum);
        let w = b.map(|x| ml(ml(k, x), r));
        let stop = !matches!(rule, Bones::EightAll | Bones::FourAll);
        let cur = blend(inp.palette, c(0x28), &bones[..n_bones], &w[..n_bones], stop);
        let prev = blend(inp.palette, c(0x2c), &bones[..n_bones], &w[..n_bones], true);
        let mut pos = Vec::with_capacity(12);
        put_pos(&mut pos, apply(&cur, p), format);
        out[pos_at..pos_at + pos.len()].copy_from_slice(&pos);
        if !primary {
            continue;
        }
        let e = (extra_at + v * extra_stride) as usize;
        let (fn_, ft) = if !stop && c(0x1c) != 0 {
            let h = if fw & 0x4000_0000 != 0 { -1.0 } else { 1.0 };
            frame_all(inp, stream(c(0x1c), v), c(0x28), &bones[..n_bones], &cur, p, nv, tv, h)
        } else if best {
            (turn_fused(&cur, nv), turn_fused(&cur, tv))
        } else {
            (turn(&cur, nv), turn(&cur, tv))
        };
        let word = if best { pack_frame_best(fn_, ft, fw & 0x4000_0000) } else { pack_frame(fn_, ft, fw & 0x4000_0000) };
        let mut extra = word.to_le_bytes().to_vec();
        put_pos(&mut extra, apply(&prev, p_prev), format);
        out[e..e + extra.len()].copy_from_slice(&extra);
    }
    Written { buffer, offset, bytes: out }
}

/// The container whose bytes hash to `hash`, from the title's shader pack
/// (`index`: the saved pack index, built when missing).
pub fn container(pack: &Path, index: Option<&Path>, hash: u128) -> Option<Vec<u8>> {
    let pack = xnw_gpu::d3d12x::pack::PackIndex::open(pack, index).ok()?;
    let place = pack.by_hash(hash)?.clone();
    pack.container(&place)
}

/// The program translated as the runtime translates compute programs
/// (descriptors planned, exact multiply-adds).
pub fn translate(container: &[u8]) -> Result<HostShader, String> {
    emit::translate_with(container, Options::default()).map_err(|e| e.to_string())
}

/// The lanes per wave the program was compiled for (its emitted
/// `WaveSize`): 64 or 32.
pub fn wave_size(container: &[u8]) -> Option<u32> {
    let c = xnw_shader::Container::parse(container).ok()?;
    let p = xnw_shader::native_programs(c.part(xnw_shader::FourCC::DXIL)?.data).ok()?;
    p.first().map(|p| if p.wave64 { 64 } else { 32 })
}

/// Why `caps` cannot run a translated program of `wave` lanes (None: it
/// can). The title's translated programs need shader model 6.6 (heap
/// indexing), the program's wave size and doubles (exact multiply-adds).
pub fn adapter_gap(caps: &xnw_gpu::host::Caps, wave: u32) -> Option<String> {
    let mut gap = Vec::new();
    if caps.shader_model < (6, 6) {
        gap.push(format!("shader model {}.{} < 6.6", caps.shader_model.0, caps.shader_model.1));
    }
    if !(caps.wave_lanes.0..=caps.wave_lanes.1).contains(&wave) {
        gap.push(format!("wave lanes {}..{} lack {wave}", caps.wave_lanes.0, caps.wave_lanes.1));
    }
    if !caps.double_ops {
        gap.push("no double ops".into());
    }
    (!gap.is_empty()).then(|| format!("{}: {}", caps.adapter, gap.join(", ")))
}

/// How the host views a heap slot, as the program reads it (emit/mod.rs
/// module notes): raw (`R32_TYPELESS`), structured with the program's
/// stride, typed (the descriptor's DXGI format, read as `buffer_load_format`),
/// a constant buffer, or a raw UAV of `len` bytes (an output, no bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Raw,
    Structured(u32),
    Typed { format: i32, bytes: u32 },
    Cbv,
    RawUav { len: u32 },
}

impl Kind {
    /// The buffer a view of this kind over `len` bytes needs and the count
    /// its description takes (elements; for a constant buffer its size in
    /// bytes), or `None` when the slot keeps its null view: a buffer view
    /// of zero elements is invalid (the layer's `NumElements (value = 0)`),
    /// and a constant buffer view needs a size of a 256-byte multiple.
    /// A raw UAV's length is its own `len`, not `len` here; an empty one
    /// still gets one element. Short buffers are padded to one whole element.
    pub fn extent(self, len: u64) -> Option<(u64, u32)> {
        let whole = |unit: u64| {
            let size = len.max(unit).next_multiple_of(unit);
            (size, (size / unit).min(u64::from(u32::MAX)) as u32)
        };
        match self {
            // An output keeps its place in the outputs read back: one element.
            Kind::RawUav { .. } => Some(whole(4)),
            _ if len == 0 => None,
            Kind::Raw => Some(whole(4)),
            Kind::Structured(stride) => Some(whole(u64::from(stride.max(4)))),
            Kind::Typed { bytes, .. } => Some(whole(u64::from(bytes.max(1)))),
            Kind::Cbv => {
                let size = len.next_multiple_of(256);
                Some((size, size.min(65_536) as u32))
            }
        }
    }
}

/// One view: its kind and the bytes from its first byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct View<'a> {
    pub kind: Kind,
    pub bytes: &'a [u8],
}

/// A captured dispatch, as far as the binding needs it: the group counts,
/// the descriptor tables the program's user data names (by user-data
/// dword: the views at their offsets in the table, in slots of 32 B), and
/// the views the program picks from the heap by index (b7+0x40.. for the
/// pre-skin), at their heap indices.
#[derive(Clone, Debug, Default)]
pub struct Dispatch<'a> {
    pub groups: [u32; 3],
    pub tables: Vec<(u32, Vec<(u32, View<'a>)>)>,
    pub heap: Vec<(u32, View<'a>)>,
    /// Root constants by root parameter (HAIR2: the strand programs' root 1).
    pub constants: Vec<(u32, Vec<u32>)>,
    /// HAIR3: heap indices that hold a 1x1 RGBA8 texture of this texel: a
    /// stand-in for a texture whose texels the capture lacks.
    pub texels: Vec<(u32, u32)>,
}

/// Root arguments and the view heap for one dispatch: what [`run`] binds.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Binding<'a> {
    /// Root parameter 0: the user-data dwords.
    pub constants: Vec<u32>,
    /// Views by heap slot (a slot not listed is a typed null).
    pub slots: Vec<(u32, View<'a>)>,
    /// View heap slots, the five guards (null 2D SRV, raw SRV, raw UAV, 2D
    /// UAV, CBV) at the end.
    pub heap_slots: u32,
    /// The runtime stride table (t0 space 3): per slot, the stride and the
    /// raw twin (StrideEntry words).
    pub strides: Vec<[u32; 2]>,
    /// The heap bounds (b0 space 7).
    pub bounds: [u32; 8],
    /// [`Dispatch::texels`].
    pub texels: Vec<(u32, u32)>,
}

/// Bytes per element of a typed view's format (formats the pre-skin
/// remap may carry); 4 for others.
fn format_bytes(format: i32) -> u32 {
    use windows::Win32::Graphics::Dxgi::Common::*;
    match windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(format) {
        DXGI_FORMAT_R8_UINT | DXGI_FORMAT_R8_UNORM => 1,
        DXGI_FORMAT_R16_UINT | DXGI_FORMAT_R16_UNORM | DXGI_FORMAT_R16_FLOAT => 2,
        DXGI_FORMAT_R32G32_UINT | DXGI_FORMAT_R32G32_FLOAT | DXGI_FORMAT_R16G16B16A16_UINT => 8,
        DXGI_FORMAT_R32G32B32A32_UINT | DXGI_FORMAT_R32G32B32A32_FLOAT => 16,
        _ => 4,
    }
}

/// The binding of `d` for `shader` (pure). The program's tables go past
/// the largest heap index the record names, each table's views at its
/// offsets; a user-data dword of a table holds the table's first slot
/// times 32 (emit/mod.rs: the byte offset in the heap). Every non-raw
/// buffer view gets a raw twin after the tables, named in the stride
/// table; the guards come last.
pub fn bind<'a>(shader: &HostShader, d: &Dispatch<'a>) -> Result<Binding<'a>, String> {
    if !shader.root_cbvs.is_empty() || !shader.root_srvs.is_empty() || !shader.root_uavs.is_empty() || shader.address_map || shader.descriptor_words {
        return Err("a program with root views, an address map or descriptor words".into());
    }
    let mut b = Binding { constants: vec![0; shader.user_data_dwords().max(1) as usize], ..Binding::default() };
    let mut next = d.heap.iter().map(|&(s, _)| s + 1).chain(d.texels.iter().map(|t| t.0 + 1)).max().unwrap_or(0);
    if d.texels.iter().any(|t| d.heap.iter().any(|h| h.0 == t.0)) {
        return Err("a texel stand-in at a view's heap index".into());
    }
    b.texels = d.texels.clone();
    b.slots.extend(d.heap.iter().copied());
    for u in &shader.user_data {
        if u.root == xnw_shader::RootType::Constants {
            let words = d.constants.iter().find(|(r, _)| *r == u32::from(u.slot)).ok_or_else(|| format!("the record has no constants at root {}", u.slot))?;
            b.constants[u.index as usize] = *words.1.get(u.dword as usize).ok_or_else(|| format!("root {} has {} constants, user data {} wants dword {}", u.slot, words.1.len(), u.index, u.dword))?;
            continue;
        }
        if u.root != xnw_shader::RootType::Table {
            return Err(format!("user data {} is a {:?}, not a table", u.index, u.root));
        }
        let Some((_, views)) = d.tables.iter().find(|(ud, _)| *ud == u.index) else {
            return Err(format!("the record has no table for user data {}", u.index));
        };
        b.constants[u.index as usize] = next * 32;
        b.slots.extend(views.iter().map(|&(k, v)| (next + k, v)));
        next += views.iter().map(|&(k, _)| k + 1).max().unwrap_or(1);
    }
    b.slots.sort_by_key(|&(s, _)| s);
    if b.slots.windows(2).any(|w| w[0].0 == w[1].0) {
        return Err("two views at one slot".into());
    }
    let mut strides = vec![[0u32; 2]; next as usize];
    let mut twins = Vec::new();
    for &(s, v) in &b.slots {
        let stride = match v.kind {
            Kind::Raw | Kind::RawUav { .. } => 4,
            Kind::Structured(n) => n,
            Kind::Typed { format, .. } => format_bytes(format),
            Kind::Cbv => 0,
        };
        let raw = match v.kind {
            Kind::Structured(_) | Kind::Typed { .. } => {
                twins.push((next, View { kind: Kind::Raw, bytes: v.bytes }));
                next += 1;
                Some(next - 1)
            }
            _ => None,
        };
        strides[s as usize] = emit::StrideEntry { stride, raw_slot: raw, bias: 0 }.words();
    }
    strides.resize(next as usize, [4, 0]);
    b.slots.extend(twins);
    b.strides = strides;
    b.heap_slots = next + 5;
    b.bounds = [b.heap_slots, next, next + 1, next + 2, next + 3, next + 4, 1, 0];
    Ok(b)
}

/// Runs `shader` (compiled for `wave` lanes) with binding `b` on `host`'s
/// device; returns each raw UAV's bytes after the dispatch, in slot order,
/// and the seconds the dispatch took on the GPU (timestamps). Refuses a
/// hardware adapter (the title's programs run on WARP only) and an
/// adapter [`adapter_gap`] names.
pub fn run(host: &xnw_gpu::host::Host, shader: &HostShader, wave: u32, b: &Binding, groups: [u32; 3]) -> Result<(Vec<Vec<u8>>, f64), String> {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    if !host.caps.software {
        return Err("the title's translated programs run on the software adapter only".into());
    }
    if let Some(gap) = adapter_gap(&host.caps, wave) {
        return Err(gap);
    }
    let e = |e: windows::core::Error| e.message().to_string();
    let dev = &host.device;
    let (root, pso) = emit::host::Host::from_device(dev.clone()).compute_pipeline(shader)?;
    // SAFETY: D3D12 calls with descriptions that outlive them; every
    // resource lives until the fence says the GPU is done.
    unsafe {
        let buffer = |size: u64, heap: D3D12_HEAP_TYPE, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES| -> Result<ID3D12Resource, String> {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(256),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let mut r: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r).map_err(e)?;
            r.ok_or_else(|| "no resource".to_string())
        };
        let upload = |bytes: &[u8], size: u64| -> Result<ID3D12Resource, String> {
            let r = buffer(size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
            let mut p = std::ptr::null_mut();
            r.Map(0, None, Some(&mut p)).map_err(e)?;
            std::ptr::write_bytes(p.cast::<u8>(), 0, size.max(256) as usize);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.cast::<u8>(), bytes.len().min(size as usize));
            r.Unmap(0, None);
            Ok(r)
        };
        let heap = |ty, n: u32| -> Result<ID3D12DescriptorHeap, String> {
            dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: ty, NumDescriptors: n.max(1), Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, ..Default::default() }).map_err(e)
        };
        let views = heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, b.heap_slots)?;
        let samplers = heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1)?;
        let step = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        let at = |slot: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: views.GetCPUDescriptorHandleForHeapStart().ptr + slot as usize * step };
        let srv = |format, first: u64, n: u32, stride: u32, flags| D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Buffer: D3D12_BUFFER_SRV { FirstElement: first, NumElements: n, StructureByteStride: stride, Flags: flags } },
        };
        let raw_uav = |n: u32| D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: D3D12_BUFFER_UAV { NumElements: n, Flags: D3D12_BUFFER_UAV_FLAG_RAW, ..Default::default() } },
        };
        // Nulls everywhere first, then the guards' kinds, then the views.
        for s in 0..b.heap_slots {
            dev.CreateShaderResourceView(None, Some(&srv(DXGI_FORMAT_R32_TYPELESS, 0, 0, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at(s));
        }
        let g = &b.bounds;
        let tex = D3D12_SHADER_RESOURCE_VIEW_DESC { Format: DXGI_FORMAT_R8G8B8A8_UNORM, ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } } };
        dev.CreateShaderResourceView(None, Some(&tex), at(g[1]));
        dev.CreateUnorderedAccessView(None, None, Some(&raw_uav(0)), at(g[3]));
        let tex_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: DXGI_FORMAT_R8G8B8A8_UNORM, ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D, ..Default::default() };
        dev.CreateUnorderedAccessView(None, None, Some(&tex_uav), at(g[4]));
        dev.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC::default()), at(g[5]));
        let mut keep = Vec::new();
        let mut outs = Vec::new();
        let mut seeds: Vec<(ID3D12Resource, ID3D12Resource, u64)> = Vec::new();
        // HAIR3: 1x1 RGBA8 textures, copied from an upload row before the dispatch.
        let mut texels: Vec<(ID3D12Resource, ID3D12Resource)> = Vec::new();
        for &(s, texel) in &b.texels {
            let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 1, Height: 1, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8G8B8A8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() };
            let mut r: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut r).map_err(e)?;
            let r = r.ok_or("no texture")?;
            dev.CreateShaderResourceView(&r, Some(&tex), at(s));
            texels.push((r, upload(&texel.to_le_bytes(), 256)?));
        }
        for &(s, v) in &b.slots {
            let len = match v.kind {
                Kind::RawUav { len } => u64::from(len),
                _ => v.bytes.len() as u64,
            };
            // An empty slot keeps its null view (`Kind::extent`).
            let Some((size, n)) = v.kind.extent(len) else { continue };
            match v.kind {
                Kind::Raw => {
                    let r = upload(v.bytes, size)?;
                    dev.CreateShaderResourceView(&r, Some(&srv(DXGI_FORMAT_R32_TYPELESS, 0, n, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at(s));
                    keep.push(r);
                }
                Kind::Structured(stride) => {
                    let r = upload(v.bytes, size)?;
                    dev.CreateShaderResourceView(&r, Some(&srv(DXGI_FORMAT_UNKNOWN, 0, n, stride.max(4), D3D12_BUFFER_SRV_FLAG_NONE)), at(s));
                    keep.push(r);
                }
                Kind::Typed { format, .. } => {
                    let r = upload(v.bytes, size)?;
                    dev.CreateShaderResourceView(&r, Some(&srv(DXGI_FORMAT(format), 0, n, 0, D3D12_BUFFER_SRV_FLAG_NONE)), at(s));
                    keep.push(r);
                }
                Kind::Cbv => {
                    let r = upload(v.bytes, size)?;
                    dev.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: r.GetGPUVirtualAddress(), SizeInBytes: n }), at(s));
                    keep.push(r);
                }
                Kind::RawUav { .. } => {
                    // Bytes given: the output starts as them (HAIR2: the strand pool
                    // carried from one dispatch of the chain to the next).
                    let first = if v.bytes.is_empty() { D3D12_RESOURCE_STATE_COMMON } else { D3D12_RESOURCE_STATE_COPY_DEST };
                    let r = buffer(size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, first)?;
                    if !v.bytes.is_empty() {
                        let up = upload(v.bytes, size)?;
                        seeds.push((r.clone(), up, size));
                    }
                    dev.CreateUnorderedAccessView(&r, None, Some(&raw_uav(n)), at(s));
                    let rb = buffer(size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
                    outs.push((r, rb, size, len));
                }
            }
        }
        let words: Vec<u8> = b.strides.iter().flatten().flat_map(|w| w.to_le_bytes()).collect();
        let table = upload(&words, words.len() as u64)?;
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso).map_err(e)?;
        let stamps: ID3D12QueryHeap = {
            let mut h: Option<ID3D12QueryHeap> = None;
            dev.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2, NodeMask: 0 }, &mut h).map_err(e)?;
            h.ok_or("no query heap")?
        };
        let times = buffer(16, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
        for (r, up, size) in &seeds {
            list.CopyBufferRegion(r, 0, up, 0, *size);
            list.ResourceBarrier(&[transition(r, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
        }
        for (r, up) in &texels {
            let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: 0, Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: DXGI_FORMAT_R8G8B8A8_UNORM, Width: 1, Height: 1, Depth: 1, RowPitch: 256 } };
            // SAFETY: borrowed pointers the copy locations never release.
            let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
            let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(up), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint } };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            list.ResourceBarrier(&[transition(r, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);
        }
        list.SetDescriptorHeaps(&[Some(views.clone()), Some(samplers.clone())]);
        list.SetComputeRootSignature(&root);
        list.SetComputeRoot32BitConstants(0, b.constants.len() as u32, b.constants.as_ptr().cast(), 0);
        // Root parameters after the user data (emit/host.rs root_signature_records):
        // the stride table, then the heap bounds.
        let mut param = 1;
        if shader.reads_stride_table() {
            list.SetComputeRootShaderResourceView(param, table.GetGPUVirtualAddress());
            param += 1;
        }
        if shader.reads_heap_bounds() {
            list.SetComputeRoot32BitConstants(param, 8, b.bounds.as_ptr().cast(), 0);
        }
        list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        list.Dispatch(groups[0], groups[1], groups[2]);
        list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
        list.ResolveQueryData(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &times, 0);
        for (r, rb, size, _) in &outs {
            list.ResourceBarrier(&[transition(r, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            list.CopyBufferRegion(rb, 0, r, 0, *size);
        }
        list.Close().map_err(e)?;
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(e)?;
        queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(&list).map_err(e)?)]);
        let sent = xnw_gpu::gpuwatch::begin("preskin");
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        queue.Signal(&fence, 1).map_err(e)?;
        while fence.GetCompletedValue() < 1 {
            dev.GetDeviceRemovedReason().map_err(|x| format!("device removed: {}", e(x)))?;
            if xnw_gpu::gpuwatch::overdue("preskin", sent) {
                return Err("preskin: the submission was given up".into());
            }
            std::thread::yield_now();
        }
        xnw_gpu::gpuwatch::end("preskin", sent, None);
        let read = |r: &ID3D12Resource, n: usize| -> Result<Vec<u8>, String> {
            let mut p = std::ptr::null_mut();
            r.Map(0, None, Some(&mut p)).map_err(e)?;
            let v = std::slice::from_raw_parts(p.cast::<u8>(), n).to_vec();
            r.Unmap(0, None);
            Ok(v)
        };
        let t = read(&times, 16)?;
        let ticks = u64::from_le_bytes(t[8..16].try_into().unwrap_or_default()).saturating_sub(u64::from_le_bytes(t[..8].try_into().unwrap_or_default()));
        let seconds = ticks as f64 / queue.GetTimestampFrequency().map_err(e)? as f64;
        let mut bytes = Vec::new();
        for (_, rb, size, len) in &outs {
            let mut out = read(rb, *size as usize)?;
            out.truncate(len.next_multiple_of(4) as usize);
            bytes.push(out);
        }
        drop(keep);
        drop(seeds);
        drop(texels);
        Ok((bytes, seconds))
    }
}

/// A transition barrier of all of `r`'s subresources.
fn transition(r: &windows::Win32::Graphics::Direct3D12::ID3D12Resource, from: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES, to: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES) -> windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER {
    use windows::Win32::Graphics::Direct3D12::*;
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            // SAFETY: a borrowed pointer the barrier never releases.
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER { pResource: unsafe { std::mem::transmute_copy(r) }, Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: from, StateAfter: to }),
        },
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    const PACK: &str = r"game\shaders.rpf";
    const INDEX: &str = r"out\cache\pack-index.bin";

    /// Empty and short slots: never a view of zero elements or a constant
    /// buffer view of a size that is not a 256-byte multiple.
    #[test]
    fn view_extents_never_empty() {
        let kinds = [Kind::Raw, Kind::Structured(1280), Kind::Structured(0), Kind::Typed { format: 2, bytes: 16 }, Kind::Typed { format: 0, bytes: 0 }, Kind::Cbv];
        for k in kinds {
            assert_eq!(k.extent(0), None, "{k:?} empty stays null");
            for len in [1u64, 3, 4, 15, 16, 17, 255, 1279, 1280, 1281, 70_000] {
                let (size, n) = k.extent(len).expect("non-empty");
                assert!(n >= 1 && size >= len, "{k:?} {len}: size {size}, n {n}");
                match k {
                    Kind::Cbv => assert!(size % 256 == 0 && n % 256 == 0 && n <= 65_536, "{len}: {size} {n}"),
                    _ => assert_eq!(size % u64::from(n), 0, "{k:?} {len}: whole elements"),
                }
            }
        }
        assert_eq!(Kind::Raw.extent(3), Some((4, 1)));
        assert_eq!(Kind::Structured(1280).extent(16), Some((1280, 1)));
        assert_eq!(Kind::Typed { format: 2, bytes: 16 }.extent(17), Some((32, 2)));
        assert_eq!(Kind::Cbv.extent(1), Some((256, 256)));
        // An output keeps its place in the outputs, also when empty.
        assert_eq!(Kind::RawUav { len: 0 }.extent(0), Some((4, 1)));
        assert_eq!(Kind::RawUav { len: 6 }.extent(6), Some((8, 2)));
    }

    /// The made-up case: two vertices, one bone, float3, primary, buffer A
    /// at byte 0x100. Owned bytes for [`Inputs`].
    pub(crate) struct Case {
        pub b7: Vec<u8>,
        pub vb: Vec<u8>,
        pub remap: Vec<u32>,
        pub palette: Vec<u8>,
        pub t7: Vec<u8>,
        pub t71: Vec<u8>,
    }

    pub(crate) fn case() -> Case {
        let put = |b: &mut Vec<u8>, at: usize, v: u32| {
            if b.len() < at + 4 {
                b.resize(at + 4, 0);
            }
            b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        };
        let mut b7 = vec![0u8; 96];
        // Streams (stride | offset << 8): position 8 B at 0x80, frame word
        // 4 B at 0x90, indices 4 B at 0x98, weights 4 B at 0xa0.
        put(&mut b7, 0x0, 8 | 0x80 << 8);
        put(&mut b7, 0x4, 4 | 0x90 << 8);
        put(&mut b7, 0xc, 4 | 0x98 << 8);
        put(&mut b7, 0x10, 4 | 0xa0 << 8);
        put(&mut b7, 0x20, u32::MAX); // no hide-mask sets
        put(&mut b7, 0x24, u32::MAX);
        put(&mut b7, 0x28, 2); // current palette base row
        put(&mut b7, 0x2c, 5); // previous palette base row
        put(&mut b7, 0x30, 1); // remap base
        put(&mut b7, 0x34, 0x6000_0000 | 0x100 >> 4); // float3, primary, A, byte 0x100
        put(&mut b7, 0x38, 2);
        let mut vb = vec![0u8; 0x80];
        for (k, v) in [0.5f32, 0.25, 0.125].iter().enumerate() {
            put(&mut vb, 0x60 + 4 * k, v.to_bits()); // bias
        }
        put(&mut vb, 0x6c, 2); // natural order
        for (k, v) in [1.0f32 / 256.0, 1.0 / 512.0, 1.0 / 1024.0].iter().enumerate() {
            put(&mut vb, 0x70 + 4 * k, v.to_bits()); // scale
        }
        // Vertex 0: q = (256, 512, 1024): p = (1.5, 1.25, 1.125).
        // Vertex 1: q = (0, 0, 0):        p = (0.5, 0.25, 0.125).
        let pos = |x: u32, y: u32, z: u32| [(x << 8) | (y & 0xff), (y >> 8) | z << 16];
        let [a, b] = pos(256, 512, 1024);
        put(&mut vb, 0x80, a);
        put(&mut vb, 0x84, b);
        let [a, b] = pos(0, 0, 0);
        put(&mut vb, 0x88, a);
        put(&mut vb, 0x8c, b);
        // Frame word 0: octahedral (-1, -1) for both vectors, which folds
        // to (0, 0, -1).
        put(&mut vb, 0x90, 0);
        put(&mut vb, 0x94, 0);
        // Indices: byte 0 = 3 -> remap[3 + 1] = 7. Weights (255, 0, 0, 0).
        put(&mut vb, 0x98, 3);
        put(&mut vb, 0x9c, 3);
        put(&mut vb, 0xa0, 255);
        put(&mut vb, 0xa4, 255);
        let remap = vec![0, 0, 0, 0, 7, 0];
        // Palette: row 2 + 7 = 9 (current): identity rotation, translation
        // (10, 20, 30); row 5 + 7 = 12 (previous): x2 scale, translation
        // (-1, -2, -3).
        let mut palette = vec![0u8; 48 * 13];
        let rows = |p: &mut Vec<u8>, i: usize, m: [[f32; 4]; 3]| {
            for (r, row) in m.iter().enumerate() {
                for (c, v) in row.iter().enumerate() {
                    p[i * 48 + r * 16 + c * 4..][..4].copy_from_slice(&v.to_le_bytes());
                }
            }
        };
        rows(&mut palette, 9, [[1.0, 0.0, 0.0, 10.0], [0.0, 1.0, 0.0, 20.0], [0.0, 0.0, 1.0, 30.0]]);
        rows(&mut palette, 12, [[2.0, 0.0, 0.0, -1.0], [0.0, 2.0, 0.0, -2.0], [0.0, 0.0, 2.0, -3.0]]);
        // t7 row 0 dword 7 >> 16 = 0 -> t71 record 0, dword 0xa0 = -1: no masks.
        let t7 = vec![0u8; 48];
        let mut t71 = vec![0u8; 416];
        put(&mut t71, 0xa0, u32::MAX);
        Case { b7, vb, remap, palette, t7, t71 }
    }

    impl Case {
        pub(crate) fn inputs(&self) -> Inputs<'_> {
            Inputs { b7: &self.b7, vb: &self.vb, remap: &self.remap, palette: &self.palette, t7: &self.t7, t71: &self.t71, ..Default::default() }
        }
    }

    fn f3(b: &[u8], at: usize) -> [f32; 3] {
        std::array::from_fn(|k| f32::from_le_bytes(b[at + 4 * k..at + 4 * k + 4].try_into().unwrap()))
    }

    fn near(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-5 * y.abs().max(1.0))
    }

    /// Hand values (see [`case`]): positions = bias + scale * q in the
    /// program's natural order (swizzle 2), weight 255/255 = 1 on bone 7,
    /// current row 9 = translate (10, 20, 30), previous row 12 = 2x then
    /// (-1, -2, -3).
    pub(crate) fn expected() -> ([f32; 3], [f32; 3], [f32; 3], [f32; 3]) {
        let p0 = [1.5f32, 1.25, 1.125];
        let p1 = [0.5f32, 0.25, 0.125];
        (
            [11.5, 21.25, 31.125],
            [10.5, 20.25, 30.125],
            [2.0 * p0[0] - 1.0, 2.0 * p0[1] - 2.0, 2.0 * p0[2] - 3.0],
            [2.0 * p1[0] - 1.0, 2.0 * p1[1] - 2.0, 2.0 * p1[2] - 3.0],
        )
    }

    /// Checks a dispatch's output range against [`expected`].
    pub(crate) fn check(w: &Written) {
        let (c0, c1, p0, p1) = expected();
        assert_eq!((w.buffer, w.offset, w.bytes.len()), (0, 0x100, 32 + 2 * 12 + 2 * 16));
        assert_eq!(&w.bytes[..8], &[2, 0, 0, 0, 0xef, 0xbe, 0xad, 0xde], "header n, 0xdeadbeef");
        assert!(w.bytes[8..32].iter().all(|&b| b == 0), "header zeros");
        assert!(near(f3(&w.bytes, 32), c0), "vertex 0 {:?} vs {c0:?}", f3(&w.bytes, 32));
        assert!(near(f3(&w.bytes, 44), c1), "vertex 1 {:?} vs {c1:?}", f3(&w.bytes, 44));
        assert!(near(f3(&w.bytes, 56 + 4), p0), "previous 0 {:?} vs {p0:?}", f3(&w.bytes, 60));
        assert!(near(f3(&w.bytes, 72 + 4), p1), "previous 1 {:?} vs {p1:?}", f3(&w.bytes, 76));
    }

    #[test]
    fn cpu_made_up_case() {
        let case = case();
        let w = cpu(&case.inputs());
        check(&w);
        // Frame word 0: n = t = (0, 0, -1), unchanged by a pure
        // translation; re-encoded as octahedral (1, 1): 255, 255, 127, 127.
        for at in [56, 72] {
            let fw = u32::from_le_bytes(w.bytes[at..at + 4].try_into().unwrap());
            assert_eq!(fw, 0x3fff_ffff, "frame word {fw:#x}");
        }
    }

    /// Variant 11's step before the bones on the made-up case with a 16-B
    /// index stream. Tables with unit-256 scales: t74 (bias (1, 2, 3)) entry
    /// 0 = (2, 2, 3), entry 1 = (1, 4, 4) with normal code (1023, 0, 0);
    /// t75 (no bias) entry 0 = (2, 0, 0), entry 1 = (0, 0, 1). Vertex 0 is
    /// tagged 0xb6 with entry 1: (1, 4, 4) + (10, 20, 30), previous
    /// 2 x (0, 0, 1) - (1, 2, 3), normal (1, -1, -1) / sqrt 3. Vertex 1 is
    /// tagged 0x54 with entry 0 and half-float offsets (1000, 2000, -1000)
    /// thousandths: (3, 4, 2) + (10, 20, 30), previous 2 x (3, 2, -1) -
    /// (1, 2, 3). With b6+0xc = 0 the tagged vertex 0 adds t76's delta 0
    /// instead (floats (0.5, 1, 2); packed (1.5, 2.5, 0.5)) and vertex 1
    /// stays where the bones put it.
    /// Variant 2: t76's delta before the bones with no b6 (a tagged vertex
    /// moves by its entry, packed or float; an untagged one does not), and
    /// the frame packed one way, as variants 0 and 1.
    #[test]
    fn delta_before_the_bones_variant_2() {
        let mut c = case();
        let put = |b: &mut Vec<u8>, at: usize, v: u32| {
            if b.len() < at + 4 {
                b.resize(at + 4, 0);
            }
            b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        };
        put(&mut c.b7, 0xc, 16 | 0x100 << 8);
        for (k, d) in [3u32, 2 << 16, 1 << 16, 0xb600_0000, 3, 0, 0, 0].iter().enumerate() {
            put(&mut c.vb, 0x100 + 4 * k, *d);
        }
        assert_eq!(rule_more(2), Some((Bones::Four, Before::Delta)));
        let base = cpu_rule(&c.inputs(), Bones::Four, Before::Nothing);
        let mut t76 = vec![0u8; 32];
        for (k, x) in [0.5f32, 1.0, 2.0].iter().enumerate() {
            put(&mut t76, 32 + 4 * k, x.to_bits());
        }
        let w = cpu_rule(&Inputs { t76: &t76, ..c.inputs() }, Bones::Four, Before::Delta);
        let none = cpu_rule(&c.inputs(), Bones::Four, Before::Delta);
        assert_eq!(none.bytes, base.bytes, "no t76: no step");
        let (b0, w0) = (f3(&base.bytes, 32), f3(&w.bytes, 32));
        assert!(w0 != b0, "the tagged vertex moved: {b0:?} -> {w0:?}");
        assert_eq!(f3(&w.bytes, 44), f3(&base.bytes, 44), "vertex 1 untagged");
        assert_eq!(w.bytes[56..60], base.bytes[56..60], "frame words as variant 0's");
    }

    #[test]
    fn tables_before_the_bones_made_up_case() {
        let mut c = case();
        let put = |b: &mut Vec<u8>, at: usize, v: u32| {
            if b.len() < at + 4 {
                b.resize(at + 4, 0);
            }
            b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        };
        let putf = |b: &mut Vec<u8>, at: usize, v: [f32; 3]| {
            for (k, x) in v.iter().enumerate() {
                put(b, at + 4 * k, x.to_bits());
            }
        };
        let pos = |x: u32, y: u32, z: u32| [(x << 8) | (y & 0xff), (y >> 8) | z << 16];
        put(&mut c.b7, 0xc, 16 | 0x100 << 8);
        for (k, d) in [3u32, 2 << 16, 1 << 16, 0xb600_0000, 0x063d_0003, 0x067d_0000, 0x0e3d_0000, 0x5400_0000].iter().enumerate() {
            put(&mut c.vb, 0x100 + 4 * k, *d);
        }
        let table = |bias: [f32; 3], entries: [[u32; 2]; 2]| {
            let mut t = vec![0u8; 8];
            putf(&mut t, 8, bias);
            put(&mut t, 20, 2);
            putf(&mut t, 24, [1.0 / 256.0; 3]);
            for (e, d) in entries.iter().enumerate() {
                put(&mut t, 36 + 8 * e, d[0]);
                put(&mut t, 40 + 8 * e, d[1]);
            }
            t
        };
        let mut t74 = table([1.0, 2.0, 3.0], [pos(256, 0, 0), pos(0, 512, 256)]);
        put(&mut t74, 8 + 64 + 4, 1023);
        let t75 = table([0.0; 3], [pos(512, 0, 0), pos(0, 0, 256)]);
        let mut b6 = vec![0u8; 16];
        put(&mut b6, 0, 8);
        put(&mut b6, 4, 64);
        put(&mut b6, 12, 1);
        let at = |w: &Written, o: usize| -> [f32; 3] { f3(&w.bytes, o) };
        let rule = rule_of(11, &c.b7).unwrap();
        assert_eq!(rule, (Bones::Four, Before::Tables));
        let w = cpu_rule(&Inputs { b6: &b6, t74: &t74, t75: &t75, ..c.inputs() }, rule.0, rule.1);
        assert!(near(at(&w, 32), [11.0, 24.0, 34.0]), "{:?}", at(&w, 32));
        assert!(near(at(&w, 44), [13.0, 24.0, 32.0]), "{:?}", at(&w, 44));
        assert!(near(at(&w, 60), [-1.0, -2.0, -1.0]), "{:?}", at(&w, 60));
        assert!(near(at(&w, 76), [5.0, 2.0, -5.0]), "{:?}", at(&w, 76));
        // Octahedral (2/3, -2/3) of the table's normal; vertex 1 keeps its own.
        let fw = |o: usize| u32::from_le_bytes(w.bytes[o..o + 4].try_into().unwrap());
        assert!((212..=213).contains(&(fw(56) & 0xff)) && (42..=43).contains(&(fw(56) >> 8 & 0xff)), "{:#x}", fw(56));
        assert_eq!(fw(72), 0x3fff_ffff);
        assert!(!table_unfilled(&Inputs { b6: &b6, t74: &t74, t75: &t75, ..c.inputs() }));
        assert!(table_unfilled(&Inputs { b6: &b6, t74: &vec![0u8; t74.len()], ..c.inputs() }));
        assert!(!table_unfilled(&c.inputs()));
        // Deltas: three floats, then the packed form with its bias and scale.
        b6[12] = 0;
        let mut t76 = vec![0u8; 32];
        putf(&mut t76, 32, [0.5, 1.0, 2.0]);
        let w = cpu_rule(&Inputs { b6: &b6, t76: &t76, ..c.inputs() }, rule.0, rule.1);
        assert!(near(at(&w, 32), [12.0, 22.25, 33.125]), "{:?}", at(&w, 32));
        assert!(near(at(&w, 44), [10.5, 20.25, 30.125]), "{:?}", at(&w, 44));
        let mut t76 = vec![0u8; 32];
        put(&mut t76, 0, 1);
        putf(&mut t76, 4, [0.5; 3]);
        putf(&mut t76, 16, [1.0 / 256.0; 3]);
        let [d0, d1] = pos(256, 512, 0);
        put(&mut t76, 32, d0);
        put(&mut t76, 36, d1);
        let w = cpu_rule(&Inputs { b6: &b6, t76: &t76, ..c.inputs() }, rule.0, rule.1);
        assert!(near(at(&w, 32), [13.0, 23.75, 31.625]), "{:?}", at(&w, 32));
        // No table bound: the plain four-bone result.
        check(&cpu_rule(&Inputs { b6: &b6, ..c.inputs() }, rule.0, rule.1));
        assert_eq!((half(0x3c00), half(0xc400), half(0x63d0), half(0x0001)), (1.0, -4.0, 1000.0, 2f32.powi(-24)));
    }

    /// The eight-bone rules on the made-up case with a second index/weight
    /// pair (b7+0x14, +0x18) and a second bone (row 2 + 8: identity moved by
    /// (100, 200, 300)). Vertex 0 has weights (102, 0, 0, 0 | 153, 0, 0, 0):
    /// the chained rule stops at the first zero weight and keeps 0.4 of bone
    /// 7 alone, 0.4 x (11.5, 21.25, 31.125); the unchained rule adds 0.6 of
    /// bone 8, p + (64, 128, 192). Vertex 1 has (128, 127, 0, 0 | 0): both
    /// rules give p + (128 x (10, 20, 30) + 127 x (100, 200, 300)) / 255.
    #[test]
    fn eight_bone_rules_made_up_case() {
        let mut c = case();
        let put = |b: &mut Vec<u8>, at: usize, v: u32| {
            if b.len() < at + 4 {
                b.resize(at + 4, 0);
            }
            b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        };
        put(&mut c.b7, 0x14, 4 | 0xa8 << 8);
        put(&mut c.b7, 0x18, 4 | 0xb0 << 8);
        c.remap = vec![0, 0, 0, 0, 7, 8];
        for (r, t) in [100f32, 200.0, 300.0].iter().enumerate() {
            c.palette[10 * 48 + r * 16 + r * 4..][..4].copy_from_slice(&1f32.to_le_bytes());
            c.palette[10 * 48 + r * 16 + 12..][..4].copy_from_slice(&t.to_le_bytes());
        }
        put(&mut c.vb, 0x98, 3); // vertex 0: first index 3 -> bone 7
        put(&mut c.vb, 0x9c, 3 | 4 << 8); // vertex 1: 3 -> 7, 4 -> 8
        put(&mut c.vb, 0xa0, 102);
        put(&mut c.vb, 0xa4, 128 | 127 << 8);
        put(&mut c.vb, 0xa8, 4); // vertex 0, second set: 4 -> bone 8
        put(&mut c.vb, 0xac, 0);
        put(&mut c.vb, 0xb0, 153);
        put(&mut c.vb, 0xb4, 0);
        let pos = |w: &Written, v: usize| -> [f32; 3] { std::array::from_fn(|k| f32::from_le_bytes(w.bytes[32 + 12 * v + 4 * k..][..4].try_into().unwrap())) };
        let near = |a: [f32; 3], b: [f32; 3]| (0..3).all(|k| (a[k] - b[k]).abs() < 2e-3);
        let v1 = [0.5 + (128.0 * 10.0 + 127.0 * 100.0) / 255.0, 0.25 + (128.0 * 20.0 + 127.0 * 200.0) / 255.0, 0.125 + (128.0 * 30.0 + 127.0 * 300.0) / 255.0];
        let chained = cpu_with(&c.inputs(), Bones::Eight);
        assert!(near(pos(&chained, 0), [4.6, 8.5, 12.45]), "{:?}", pos(&chained, 0));
        assert!(near(pos(&chained, 1), v1), "{:?}", pos(&chained, 1));
        let all = cpu_with(&c.inputs(), Bones::EightAll);
        assert!(near(pos(&all, 0), [65.5, 129.25, 193.125]), "{:?}", pos(&all, 0));
        assert!(near(pos(&all, 1), v1), "{:?}", pos(&all, 1));
        // Variant 0 reads the first pair only: vertex 0 is all of bone 7.
        assert!(near(pos(&cpu(&c.inputs()), 0), [11.5, 21.25, 31.125]));
        assert_eq!([0, 3, 4, 5, 8, 11].map(bones_of), [Some(Bones::Four), Some(Bones::Eight), Some(Bones::Eight), Some(Bones::EightAll), None, None]);
    }

    /// The frame word of the variants past 0. A frame the pairs hold exactly
    /// (both vectors (0, 0, -1): 255, 255, 127, 127) stays in pairs; n = +z
    /// with t = +x is closer as 10:10 and an angle (bit 31), bit 30 kept;
    /// over a sweep of frames the word decodes to within a degree of the
    /// input and is never farther from it than the pairs alone; a frame read
    /// from a pair word packs to that word again.
    #[test]
    fn best_frame_word_made_up_cases() {
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        assert_eq!(pack_frame_best([0.0, 0.0, -1.0], [0.0, 0.0, -1.0], 0), 0x3fff_ffff);
        let w = pack_frame_best([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 0x4000_0000);
        assert_eq!(w >> 30, 3, "{w:#x}");
        let (n, t) = frame(w);
        assert!(dot(n, [0.0, 0.0, 1.0]) > 0.9999 && dot(t, [1.0, 0.0, 0.0]) > 0.9999, "{n:?} {t:?}");
        let (mut angled, mut pairs) = (0u32, 0u32);
        for i in 0..400 {
            let (a, b, c) = (0.37 * i as f32, 0.11 * i as f32 + 1.0, 0.23 * i as f32);
            let n = [a.sin() * b.cos(), a.sin() * b.sin(), a.cos()];
            let e = if n[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
            let u = [n[1] * e[2] - n[2] * e[1], n[2] * e[0] - n[0] * e[2], n[0] * e[1] - n[1] * e[0]];
            let l = dot(u, u).sqrt();
            let u = u.map(|x| x / l);
            let v = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]];
            let t: [f32; 3] = std::array::from_fn(|k| u[k] * c.cos() + v[k] * c.sin());
            let w = pack_frame_best(n, t, 0);
            let (dn, dt) = frame(w);
            assert!(dot(dn, n) > 0.99985 && dot(dt, t) > 0.99985, "frame {i}: {w:#x} {dn:?} {dt:?} for {n:?} {t:?}");
            let (pn, pt) = frame(pack_frame(n, t, 0));
            assert!(dot(dn, n) + dot(dt, t) >= dot(pn, n) + dot(pt, t) - 1e-5, "frame {i}");
            angled += w >> 31;
        }
        // Frames read from pair words go back into the same pair words.
        for i in 0..200u32 {
            let w = i.wrapping_mul(0x9e37_79b9) & 0x3fff_ffff;
            let (n, t) = frame(w);
            pairs += u32::from(pack_frame_best(n, t, 0) == w);
        }
        assert!(angled > 300 && pairs > 100, "{angled} angled, {pairs} in pairs");
    }

    /// The every-bone frame on the eight-bone case. Vertex 0: n = +z, t = +x
    /// (pairs 128, 128, 127, 64), bones 7 (0.4) and 8 (0.6), both pure
    /// translations, so the matrix is the identity and d of bone 8 (the
    /// fifth) is (90, 180, 270). Its pair (bytes 8, 9 of the 16-B stream at
    /// b7+0x1c) is (255, 128) under magnitude 1: a = 127 / 255, b = 0. The
    /// tangent is t + 0.9 A / |A| = (1, 0, 0) + 0.9 (1, 2, 3) / sqrt 14,
    /// normalised; the bitangent stays t x n = (0, -1, 0); the normal is
    /// their cross product. With every byte 128 the frame is unchanged.
    #[test]
    fn every_bone_frame_made_up_case() {
        let mut c = case();
        let put = |b: &mut Vec<u8>, at: usize, v: u32| {
            if b.len() < at + 4 {
                b.resize(at + 4, 0);
            }
            b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        };
        put(&mut c.b7, 0x14, 4 | 0xa8 << 8);
        put(&mut c.b7, 0x18, 4 | 0xb0 << 8);
        put(&mut c.b7, 0x1c, 16 | 0xc0 << 8);
        c.remap = vec![0, 0, 0, 0, 7, 8];
        for (r, t) in [100f32, 200.0, 300.0].iter().enumerate() {
            c.palette[10 * 48 + r * 16 + r * 4..][..4].copy_from_slice(&1f32.to_le_bytes());
            c.palette[10 * 48 + r * 16 + 12..][..4].copy_from_slice(&t.to_le_bytes());
        }
        put(&mut c.vb, 0x90, 128 | 128 << 8 | 127 << 16 | 64 << 23);
        put(&mut c.vb, 0x98, 3);
        put(&mut c.vb, 0xa0, 102);
        put(&mut c.vb, 0xa8, 4);
        put(&mut c.vb, 0xb0, 153);
        for (k, d) in [0x3c00 | 0x8080 << 16, 0x8080_8080, 0x8080_80ff, 0x8080_8080, 0x3c00 | 0x8080 << 16, 0x8080_8080, 0x8080_8080, 0x8080_8080].iter().enumerate() {
            put(&mut c.vb, 0xc0 + 4 * k, *d);
        }
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let unit = |a: [f32; 3]| a.map(|x| x / dot(a, a).sqrt());
        let word = |w: &Written| u32::from_le_bytes(w.bytes[56..60].try_into().unwrap());
        let (n, t) = frame(word(&cpu_with(&c.inputs(), Bones::EightAll)));
        let k = 0.9 / 14f32.sqrt();
        let want_t = unit([1.0 + k, 2.0 * k, 3.0 * k]);
        let want_n = unit([-want_t[2], 0.0, want_t[0]]);
        assert!(dot(t, want_t) > 0.9998 && dot(n, want_n) > 0.9998, "{n:?} {t:?} want {want_n:?} {want_t:?}");
        // Without the pair the frame is the bones' alone.
        put(&mut c.vb, 0xc8, 0x8080_8080);
        let (n, t) = frame(word(&cpu_with(&c.inputs(), Bones::EightAll)));
        assert!(dot(t, [1.0, 0.0, 0.0]) > 0.9998 && dot(n, [0.0, 0.0, 1.0]) > 0.9998, "{n:?} {t:?}");
        // The chained rule does not read the stream.
        put(&mut c.vb, 0xc8, 0x8080_80ff);
        let (n, t) = frame(word(&cpu_with(&c.inputs(), Bones::Eight)));
        assert!(dot(t, [1.0, 0.0, 0.0]) > 0.9998 && dot(n, [0.0, 0.0, 1.0]) > 0.9998, "{n:?} {t:?}");
    }

    /// The made-up case as a captured dispatch (variant 0's table slots:
    /// ud4 +0 t71, +0x20 palette, +0xc0 t78, +0x120 t7, +0x300 u7, +0x320
    /// u8; ud5 +0x100 b7; heap 3 vertex streams, heap 4 remap). Buffer A is
    /// 0x200 B (the range ends at 0x158), B 0x100 B.
    fn dispatch<'a>(c: &'a Case, b7: &'a [u8], remap: &'a [u8]) -> Dispatch<'a> {
        let v = |kind, bytes| View { kind, bytes };
        Dispatch {
            groups: [1, 1, 1],
            tables: vec![
                (4, vec![
                    (0, v(Kind::Structured(416), &c.t71)),
                    (1, v(Kind::Structured(48), &c.palette)),
                    (6, v(Kind::Structured(1280), &[])),
                    (9, v(Kind::Structured(48), &c.t7)),
                    (24, v(Kind::RawUav { len: 0x200 }, &[])),
                    (25, v(Kind::RawUav { len: 0x100 }, &[])),
                ]),
                (5, vec![(8, v(Kind::Cbv, b7))]),
            ],
            heap: vec![(3, v(Kind::Raw, &c.vb)), (4, v(Kind::Typed { format: 42, bytes: 4 }, remap))],
            constants: Vec::new(),
            texels: Vec::new(),
        }
    }

    fn gpu_b7(c: &Case) -> Vec<u8> {
        let mut b7 = c.b7.clone();
        b7[0x40..0x44].copy_from_slice(&3u32.to_le_bytes());
        b7[0x44..0x48].copy_from_slice(&4u32.to_le_bytes());
        b7
    }

    /// A stand-in for the translation's binding needs (variant 0: user data
    /// 4 and 5 are tables, 8 dwords, stride table and heap bounds read).
    fn shader_like_0() -> HostShader {
        let ud = |index, slot| emit::UserData { index, slot, root: xnw_shader::RootType::Table, dword: 0 };
        HostShader { stage: xnw_shader::HwStage::Cs, dxil: Vec::new(), user_data: vec![ud(4, 0), ud(5, 1)], samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: Vec::new(), root_srvs: Vec::new(), root_uavs: Vec::new(), half_views: Vec::new(), address_map: false, descriptor_words: false }
    }

    #[test]
    fn binding_from_record() {
        let c = case();
        let (b7, remap) = (gpu_b7(&c), c.remap.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let b = bind(&shader_like_0(), &dispatch(&c, &b7, &remap)).unwrap();
        // Heap indices 3 and 4 first; ud4's table at 5 (26 slots), ud5's at 31 (9 slots).
        assert_eq!(b.constants, vec![0, 0, 0, 0, 5 * 32, 31 * 32, 0, 0]);
        let at: Vec<u32> = b.slots.iter().map(|s| s.0).collect();
        // Views, then the raw twins of the remap and the four structured views.
        assert_eq!(at, vec![3, 4, 5, 6, 11, 14, 29, 30, 39, 40, 41, 42, 43, 44]);
        assert_eq!(b.heap_slots, 45 + 5);
        assert_eq!(b.bounds, [50, 45, 46, 47, 48, 49, 1, 0]);
        assert_eq!(b.strides[4], emit::StrideEntry { stride: 4, raw_slot: Some(40), bias: 0 }.words());
        assert_eq!(b.strides[6], emit::StrideEntry { stride: 48, raw_slot: Some(42), bias: 0 }.words());
        assert_eq!(b.strides[3], [4, 0], "raw: the slot itself");
        assert_eq!(b.slots[9].1.bytes, &remap[..], "twin of the remap");
        assert!(bind(&shader_like_0(), &Dispatch::default()).is_err(), "a record without the tables");
    }

    /// Step 1 on a device: the translated variant 0 on the made-up case,
    /// against the same hand values as the CPU reference. Skips (printing
    /// why) without `XNW_TEST_GPU=1`, without the pack, or on an adapter
    /// below what the program needs (this PC's WARP: shader model 6.2,
    /// 4 lanes).
    #[test]
    fn translated_made_up_case() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let Some(container) = container(Path::new(PACK), Some(Path::new(INDEX)), STANDARD_0) else {
            eprintln!("preskin: no shader pack at {PACK}: skipped");
            return;
        };
        let t0 = std::time::Instant::now();
        let shader = translate(&container).expect("variant 0 translates");
        let translated = t0.elapsed();
        let wave = wave_size(&container).expect("wave size");
        let (_gpu, host) = super::super::tests::warp_host();
        if let Some(gap) = adapter_gap(&host.caps, wave) {
            eprintln!("preskin: translated program not run: {gap} (translation {translated:?})");
            return;
        }
        let c = case();
        let (b7, remap) = (gpu_b7(&c), c.remap.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let d = dispatch(&c, &b7, &remap);
        let b = bind(&shader, &d).unwrap();
        let t1 = std::time::Instant::now();
        let (outs, gpu) = run(&host, &shader, wave, &b, d.groups).expect("dispatch");
        eprintln!("preskin: translation {translated:?}, pipeline + dispatch {:?}, dispatch {gpu:.6} s", t1.elapsed());
        let a = &outs[0];
        check(&Written { buffer: 0, offset: 0x100, bytes: a[0x100..0x158].to_vec() });
        assert_eq!(&a[0x100..0x158], &cpu(&c.inputs()).bytes[..], "translated program and CPU reference, byte for byte");
    }

    #[test]
    fn frame_decodes() {
        // 8:8 (255, 255) is octahedral (1, 1): z = -1 folds to (0, 0, -1).
        let (n, _) = frame(0xffff);
        assert!(near(n, [0.0, 0.0, -1.0]), "{n:?}");
        // 7:7 (127, 127) likewise, for t.
        let (_, t) = frame(0x3fff_0000);
        assert!(near(t, [0.0, 0.0, -1.0]), "{t:?}");
        assert_eq!(pack_frame([0.0, 0.0, -1.0], [0.0, 0.0, -1.0], 0), 0xffff | 0x7f << 16 | 0x7f << 23);
    }

    #[test]
    fn range_word() {
        assert_eq!(range(0x6000_0010), (0, 0x100, PosFormat::Float3, true));
        assert_eq!(range(0x1100_0001), (1, 0x10, PosFormat::Packed10, false));
        assert_eq!(strides(PosFormat::Packed24, true), (8, 12));
    }

    /// What the translation of variant 0 asks the host to bind (by hand:
    /// `cargo test -p xnw-gpu preskin_dump -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn preskin_dump() {
        let c = container(Path::new(PACK), Some(Path::new(INDEX)), STANDARD_0).expect("container");
        let t = translate(&c).expect("translates");
        println!("user data {:?}", t.user_data);
        println!("dwords {} root cbvs {:?} srvs {:?} uavs {:?} map {} words {} stride {} bounds {} samplers {}", t.user_data_dwords(), t.root_cbvs, t.root_srvs, t.root_uavs, t.address_map, t.descriptor_words, t.reads_stride_table(), t.reads_heap_bounds(), t.static_samplers.len());
        if let Ok(dir) = std::env::var("PRESKIN_DUMP") {
            std::fs::write(Path::new(&dir).join("cs0.dxil"), &t.dxil).unwrap();
        }
    }

    /// HAIR3: a table `put_table` writes reads back through `table_pos`
    /// (the decode both variant 11 and the strand chain use) within its step.
    #[test]
    fn put_table_reads_back() {
        let mut t = vec![0u8; 8];
        let entries = [(0u32, [0.1f32, -0.25, 1.7]), (3, [-2.9, 2.9, 0.0])];
        put_table(&mut t, 0x40, &entries);
        assert_eq!(t.len(), 0x40 + 28 + 8 * 4);
        for (e, p) in entries {
            let got = table_pos(&t, 0x40, e);
            assert!((0..3).all(|k| (got[k] - p[k]).abs() < 1e-4), "{e}: {got:?} against {p:?}");
        }
        assert_eq!(table_pos(&t, 0x40, 1), [-3.0; 3], "an entry not written is the bias");
    }
}
