//! The ped pre-skin as a compute pass of ours: the maths of the CPU
//! reference ([`super::preskin::cpu_with`]) in our own HLSL, one dispatch per
//! item, so the skinned positions of a ped are made on the device each frame
//! and its PreSkinD draws read them there.
//!
//! An item is one pre-skin dispatch's inputs (the 96-B constant block with
//! its palette bases pointing at the rows given, the vertex view, the bone
//! remap, the palette rows, the hide-mask record). A job is one draw that
//! takes an item's output: the compute pass copies the draw's own vertex
//! buffer into a buffer of ours, appends the skinned positions as float3
//! (and the turned normals when the draw reads the extra blocks), points
//! stream words 0 and 8 at them and clears the skin stream words (0x18 to
//! 0x24), so the world pass draws the copy as it is, without skinning it
//! again. Positions go through the output range's own format (24:24:16,
//! 10:10:10 or float3) and back, as the PreSkinD program reads them; a
//! hidden vertex is NaN, so its triangles are not drawn.
//!
//! The vertex views and remaps are kept on the device: each is uploaded
//! once per key (its title address, length and head; or its content) into
//! one buffer of ours ([`Kept`]); a frame uploads only its constant
//! blocks, palette rows and mask records, and the streams it has not seen.

use std::collections::HashMap;
use std::sync::Arc;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::preskin::{self as ps, Before, Bones, Inputs};

/// Our compute program: one thread per dword of the draw's vertex buffer
/// (the copy) and per vertex (the skin), whichever is more.
pub const SHADER: &str = r#"
#define RS "RootConstants(num32BitConstants=24, b0), SRV(t1), UAV(u0), DescriptorTable(SRV(t0)), SRV(t2)"
cbuffer Item : register(b0) {
    uint b7At;       // 0: the 96-B constant block in t1 (palette bases rebased)
    uint vbAt;       // 1: the dispatch's vertex view in t2
    uint vbBytes;    // 2: its bytes
    uint remapAt;    // 3: the bone remap in t2, one uint per element
    uint remapCount; // 4
    uint palAt;      // 5: the palette rows in t1 (3 float4 each)
    uint palRows;    // 6
    uint maskAt;     // 7: the 1,280-B hide-mask record in t1 (0xFFFFFFFF: no masks)
    uint outAt;      // 8: the draw's vertex-buffer copy in u0
    uint copyDwords; // 9: dwords copied from t0 (0: positions only)
    uint posAt;      // 10: float3 positions in u0
    uint nrmAt;      // 11: float3 normals in u0 (0xFFFFFFFF: none)
    uint rule;       // 12: 0 four bones, 1 eight up to the first zero weight, 2 eight all, 3 four all
    uint posWord;    // 13: stream word 0 of the copy
    uint nrmWord;    // 14: stream word 8 of the copy
    uint threads;    // 15: max(copyDwords, vertices)
    uint deltaAt;    // 16: variant 2's t76 delta table in t1
    uint deltaBytes; // 17: its bytes (0: no delta step)
    uint tableAt;    // 18: variant 11's face (t74 from b6+0, rebased to 0) in t1
    uint tableBytes; // 19: its bytes (0: no table step)
    uint tableNrm;   // 20: its normals (b6+4) from the face's header
    uint rangeAt;    // 21: the title's range of this item in u0 (header; positions at posAt = rangeAt + 32), 0xFFFFFFFF: none
    uint extraAt;    // 22: its extra blocks (16 B a vertex: frame word, position)
    uint pad;        // 23
};
ByteAddressBuffer geom : register(t0);
ByteAddressBuffer inp : register(t1);
ByteAddressBuffer kept : register(t2);
RWByteAddressBuffer outb : register(u0);

uint cb(uint off) { return inp.Load(b7At + off); }
// A dword of the vertex view at any byte; 0 when it does not lie inside.
// Both loads stay inside the view whatever the compiler makes of the
// branches (an unaligned dword's second word is read only when it lies inside).
uint vw(uint a) {
    bool ok = a < vbBytes && vbBytes - a >= 4;
    uint at = ok ? a : 0, s = (at & 3) * 8, b = vbAt + (at & ~3u);
    uint lo = kept.Load(b), hi = kept.Load(b + (s == 0 ? 0 : 4));
    return ok ? (s == 0 ? lo : (lo >> s) | (hi << (32 - s))) : 0;
}
uint stream(uint desc, uint v) { return (desc >> 8) + (desc & 0xFF) * v; }
// v_mul_legacy_f32: 0 times anything is 0.
float ml(float a, float b) { return (a == 0 || b == 0) ? 0 : a * b; }
float4 ml4(float4 a, float b) { return float4(ml(a.x, b), ml(a.y, b), ml(a.z, b), ml(a.w, b)); }
float clamp01(float x) { return isnan(x) ? 0 : clamp(x, 0, 1); }
float fold(float v, float t) { return (v >= 0 ? -t : t) + v; }

// The normal and tangent of a vertex's frame word: a 10:10 octahedral
// normal and a 10-bit angle when bit 31 is set, else 8:8 and 7:7 pairs.
void frameOf(uint word, out float3 n, out float3 t) {
    if ((word >> 31) != 0) {
        float x = (word & 0x3FF) * asfloat(0x3B002008) - 1.0;
        float y = ((word >> 10) & 0x3FF) * asfloat(0x3B002008) - 1.0;
        float a = mad((float)((word >> 20) & 0x3FF), asfloat(0x3BC9422C), asfloat(0xC0490FDB));
        float z = 1.0 - abs(x) - abs(y);
        float tz = clamp01(-z);
        y = fold(y, tz);
        x = fold(x, tz);
        float r = 1.0 / sqrt(y * y + x * x + z * z);
        float nz = r * z, ny = y * r, nx = x * r;
        float s = nz >= 0 ? 1.0 : -1.0;
        float k = 1.0 / mad(r, z, s);
        float rev = mad(a, 0.15915494, -trunc(ml(0.15915494, a)));
        float sn, cs;
        sincos(rev * 6.2831855, sn, cs);
        float q = ml(ny, k), sx = ml(s, nx), qx = ml(q, nx);
        float b = mad(ml(k, nx), sx, -1.0);
        float c = mad(q, ny, -s);
        n = float3(nx, ny, nz);
        t = float3(mad(b, cs, -ml(qx, sn)), mad(ml(s, qx), cs, -ml(c, sn)), mad(sx, cs, -ml(ny, sn)));
    } else {
        float x = (word & 0xFF) * asfloat(0x3C008081) - 1.0;
        float y = ((word >> 8) & 0xFF) * asfloat(0x3C008081) - 1.0;
        float p = ((word >> 16) & 0x7F) * asfloat(0x3C810204) - 1.0;
        float q = ((word >> 23) & 0x7F) * asfloat(0x3C810204) - 1.0;
        float z = 1.0 - abs(x) - abs(y);
        float w = 1.0 - abs(p) - abs(q);
        float tz = clamp01(-z), tw = clamp01(-w);
        y = fold(y, tz); x = fold(x, tz); q = fold(q, tw); p = fold(p, tw);
        float r1 = 1.0 / sqrt(y * y + x * x + z * z);
        float r2 = 1.0 / sqrt(q * q + p * p + w * w);
        n = float3(x * r1, y * r1, r1 * z);
        t = float3(p * r2, q * r2, r2 * w);
    }
}

float len2(float3 v) { return mad(v.z, v.z, mad(v.x, v.x, ml(v.y, v.y))); }
float3 unit(float3 v, float k) { return float3(ml(k, v.x), ml(k, v.y), ml(k, v.z)); }
float rdot(float4 m, float3 v) { return mad(m.z, v.z, mad(m.y, v.y, ml(m.x, v.x))); }
// Byte i of the gradient stream at a: (i < 4 rounds as the program does).
float coef(uint a, uint i, float mag) {
    float b = (float)((vw(a + (i & ~3u)) >> (8 * (i & 3))) & 0xFF);
    return i < 4 ? ml(mad(asfloat(0x3B808081), b, asfloat(0xBF008081)), mag) : ml(mag, b / 255.0 + asfloat(0xBF008081));
}

// A dword of the delta table; 0 past its end (the title's bounded load).
// `?:` runs both sides: the condition picks the address, then the value (a
// root view has no bounds on the card; HANG1).
uint dl(uint a) { bool ok = a + 4 <= deltaBytes; uint v = inp.Load(deltaAt + (ok ? a : 0)); return ok ? v : 0; }
// A dword of the face table; 0 past its end.
uint tl(uint a) { bool ok = a < tableBytes && tableBytes - a >= 4; uint v = inp.Load(tableAt + (ok ? a : 0)); return ok ? v : 0; }
// Entry e of the face table: 24:24:16 through the header's bias, scale and axis order.
float3 tpos(uint e) {
    bool ok = e < 0x10000000;
    uint at = 28 + 8 * (ok ? e : 0);
    uint d0 = ok ? tl(at) : 0, d1 = ok ? tl(at + 4) : 0;
    float3 q = float3(d0 >> 8, (d0 & 0xFF) | ((d1 & 0xFFFF) << 8), d1 >> 16);
    float x2 = mad(asfloat(tl(16)), q.x, asfloat(tl(0))), x3 = mad(asfloat(tl(20)), q.y, asfloat(tl(4))), x4 = mad(asfloat(tl(24)), q.z, asfloat(tl(8)));
    uint o = tl(12);
    return o == 0 ? float3(x4, x3, x2) : o == 1 ? float3(x3, x4, x2) : float3(x2, x3, x4);
}
float4 prow(uint i, uint r) { bool ok = i < palRows; float4 v = asfloat(inp.Load4(palAt + (ok ? i * 48 + r * 16 : 0))); return ok ? v : float4(0, 0, 0, 0); }

bool masked(uint set, uint half_, uint i) {
    if (set >= 20 || i >= 256) return false;
    return ((inp.Load(maskAt + half_ + set * 32 + (i >> 5) * 4) >> (i & 31)) & 1) != 0;
}

// A position through the range format and back, as the PreSkinD program reads it.
float3 quant(float3 p, uint word) {
    if ((word & 0x20000000) != 0) return p;
    float3 o;
    if ((word & 0x10000000) != 0) {
        [unroll] for (uint k = 0; k < 3; k++) {
            uint q = (uint)mad(clamp01(ml(asfloat(0x3E2AAAAB), p[k] + 3.0)), 1023.0, 0.5);
            o[k] = q * asfloat(0x3BC0300C) - 3.0;
        }
        return o;
    }
    [unroll] for (uint j = 0; j < 2; j++) {
        float f = ml(asfloat(0x4A2AAAAA), p[j] + 3.0);
        uint q = isnan(f) ? 0 : (uint)clamp(f, 0, asfloat(0x4B7FFFFF));
        o[j] = q * asfloat(0x34C00001) - 3.0;
    }
    float fz = ml(asfloat(0x462AAA00), p.z + 3.0);
    uint qz = (isnan(fz) ? 0 : (uint)clamp(fz, 0, asfloat(0x477FFF00))) & 0xFFFF;
    o.z = qz * asfloat(0x38C000C0) - 3.0;
    return o;
}

// The extra block's frame word (preskin.rs `pack_frame`, a45da713
// 0x2370-0x2474): n as octahedral 8:8 in bits 0-15, t as 7:7 in bits
// 16-29, `sign` (bit 30 of the vertex's frame word) kept.
float2 oct(float3 v) {
    float l = 1.0 / (abs(v.x) + abs(v.y) + abs(v.z));
    float z = clamp01(ml(v.z, -l));
    float x = ml(v.x, l), y = ml(v.y, l);
    return float2((x >= 0 ? z : -z) + x, (y >= 0 ? z : -z) + y);
}
uint q8(float v, float k) { return (uint)mad(clamp01(mad(v, 0.5, 0.5)), k, 0.5); }
uint packFrame(float3 n, float3 t, uint sign) {
    float2 a = oct(n), b = oct(t);
    return q8(a.x, 255.0) | sign | (q8(a.y, 255.0) << 8) | (q8(b.x, 127.0) << 16) | (q8(b.y, 127.0) << 23);
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= threads) return;
    if (i < copyDwords) {
        uint w = geom.Load(4 * i);
        if (i == 0) w = posWord;
        if (i == 2 && nrmAt != 0xFFFFFFFF) w = nrmWord;
        if (i >= 6 && i < 10) w = 0;
        outb.Store(outAt + 4 * i, w);
    }
    uint n = cb(0x38);
    // The range's 32-B header {n, 0xdeadbeef, 0 x 6}, as the title's pass writes it.
    if (i == 0 && rangeAt != 0xFFFFFFFF) {
        outb.Store4(rangeAt, uint4(n, 0xDEADBEEF, 0, 0));
        outb.Store4(rangeAt + 16, uint4(0, 0, 0, 0));
    }
    if (i >= n) return;
    uint v = i;
    uint word = cb(0x34);
    if (maskAt != 0xFFFFFFFF) {
        uint b = vw(stream(cb(0x8), v));
        bool hidden = masked(cb(0x20), 0, (b & 0xFF) - 1);
        hidden = hidden || masked(cb(0x24), 640, ((b >> 8) & 0xFF) - 1);
        if (hidden) {
            outb.Store3(posAt + 12 * v, uint3(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF));
            if (nrmAt != 0xFFFFFFFF) outb.Store3(nrmAt + 12 * v, asuint(float3(0, 0, 1)));
            return;
        }
    }
    // Position, 24:24:16, then bias, scale and the axis order.
    uint at = stream(cb(0), v);
    uint d0 = vw(at), d1 = vw(at + 4);
    float3 q = float3(d0 >> 8, (d0 & 0xFF) | ((d1 & 0xFFFF) << 8), d1 >> 16);
    float3 bias = asfloat(uint3(vw(0x60), vw(0x64), vw(0x68)));
    float3 scale = asfloat(uint3(vw(0x70), vw(0x74), vw(0x78)));
    float x2 = mad(scale.x, q.x, bias.x), x3 = mad(scale.y, q.y, bias.y), x4 = mad(scale.z, q.z, bias.z);
    uint swz = vw(0x6C);
    float3 p = swz == 0 ? float3(x4, x3, x2) : swz == 1 ? float3(x3, x4, x2) : float3(x2, x3, x4);
    // Variant 2: a vertex tagged 0xb6 in its fourth index dword (16-B index
    // stream) adds entry (dword 2 >> 16) - 1 of t76: 8 B through the table's
    // scale and bias when its first dword is set, else three floats.
    if (deltaBytes != 0 && (cb(0xC) & 0xFF) == 16) {
        uint ia = (cb(0xC) >> 8) + 16 * v;
        uint e = (vw(ia + 12) & 0xFF000000) == 0xB6000000 ? (vw(ia + 8) >> 16) - 1 : 0xFFFF;
        if (e < 0xFFFF) {
            float3 d;
            if (dl(0) != 0) {
                uint e0 = dl(32 + 8 * e), e1 = dl(36 + 8 * e);
                float3 dq = float3(e0 >> 8, (e0 & 0xFF) | ((e1 & 0xFFFF) << 8), e1 >> 16);
                d = float3(mad(dq.x, asfloat(dl(16)), asfloat(dl(4))), mad(dq.y, asfloat(dl(20)), asfloat(dl(8))), mad(dq.z, asfloat(dl(24)), asfloat(dl(12))));
            } else {
                d = asfloat(uint3(dl(32 + 12 * e), dl(36 + 12 * e), dl(40 + 12 * e)));
            }
            p = d + p;
        }
    }
    // Variant 11 with its face table filled (659b134d 0xb3c-0xf50): a
    // vertex tagged 0xb6 takes entry (dword 1 >> 16) - 1 of the table (and,
    // below dword 0 0xFFFF0000, the entry's 10:10:10 normal); one tagged
    // 0x54 the 16-bit entry of dwords 2 and 3 plus three half-float
    // offsets in thousandths. As `tables` in the CPU reference.
    bool tabN = false;
    float3 tN = float3(0, 0, 0);
    if (tableBytes != 0 && (cb(0xC) & 0xFF) == 16) {
        uint ia = (cb(0xC) >> 8) + 16 * v;
        uint i0 = vw(ia), i1 = vw(ia + 4), i2 = vw(ia + 8), i3 = vw(ia + 12);
        uint tag = i3 & 0xFF000000;
        uint e = (tag != 0xB6000000 || i1 == 0xFFFF) ? 0xFFFF : (i1 >> 16) - 1;
        if (e != 0xFFFF) {
            p = tpos(e);
            if (i0 < 0xFFFF0000) {
                bool ok = e < 0x10000000;
                uint w = ok ? tl(tableNrm + 4 * e) : 0;
                float k = asfloat(0x3B002008);
                tN = float3((w & 0x3FF) * k - 1.0, ((w >> 10) & 0x3FF) * k - 1.0, ((w >> 20) & 0x3FF) * k - 1.0);
                tabN = true;
            }
        }
        if (tag == 0x54000000) {
            uint e2 = ((i3 >> 12) & 0xFFF) | ((i2 >> 16) & 0xF000);
            float k2 = asfloat(0x3A83126F);
            float3 off = float3(ml(f16tof32((i0 >> 12) & 0xFFFF), k2), ml(f16tof32((i1 >> 12) & 0xFFFF), k2), ml(f16tof32((i2 >> 12) & 0xFFFF), k2));
            p = tpos(e2) + off;
        }
    }
    // Bones: four (or eight) indices through the remap, unorm8 weights over their sum.
    bool wide = (cb(0xC) & 0xFF) == 16;
    uint sets = (rule == 0 || rule == 3) ? 1 : 2;
    uint bones[8];
    float b[8];
    [unroll] for (uint s = 0; s < 2; s++) {
        uint idesc = cb(0xC + 8 * s);
        uint ib[4];
        if (wide) {
            uint a = (idesc >> 8) + 16 * v;
            ib[0] = vw(a); ib[1] = vw(a + 4); ib[2] = vw(a + 8); ib[3] = vw(a + 12);
        } else {
            uint w4 = vw(stream(idesc, v));
            ib[0] = w4 & 0xFF; ib[1] = (w4 >> 8) & 0xFF; ib[2] = (w4 >> 16) & 0xFF; ib[3] = w4 >> 24;
        }
        uint ww = vw(stream(cb(0x10 + 8 * s), v));
        [unroll] for (uint k = 0; k < 4; k++) {
            uint e = (ib[k] & 0xFFF) + cb(0x30);
            uint re = kept.Load(remapAt + 4 * (e < remapCount ? e : 0));
            bones[4 * s + k] = e < remapCount ? re & 0xFFF : 0;
            b[4 * s + k] = s < sets ? (float)((ww >> (8 * k)) & 0xFF) : 0;
        }
    }
    float kw = asfloat(0x3B808081);
    float sum = sets == 1 ? b[1] + b[0] + b[2] + b[3] : b[1] + b[0] + b[2] + b[3] + b[4] + b[5] + b[6] + b[7];
    float r = 1.0 / ml(kw, sum);
    float w[8];
    [unroll] for (uint j = 0; j < 8; j++) w[j] = ml(ml(kw, b[j]), r);
    // The blended matrix: bone 0 always; with the stop, bone k only while no weight before it is 0.
    uint base = cb(0x28), count = 4 * sets;
    bool stop = rule != 2 && rule != 3, done = false;
    float4 m[3];
    [unroll] for (uint r0 = 0; r0 < 3; r0++) m[r0] = ml4(prow(base + bones[0], r0), w[0]);
    [unroll] for (uint k2 = 1; k2 < 8; k2++) {
        if (k2 < count && !done) {
            if (w[k2] == 0) {
                done = stop;
            } else {
                [unroll] for (uint r1 = 0; r1 < 3; r1++) m[r1] = mad(prow(base + bones[k2], r1), w[k2], m[r1]);
            }
        }
    }
    float3 o;
    [unroll] for (uint r2 = 0; r2 < 3; r2++) o[r2] = mad(m[r2].z, p.z, mad(m[r2].y, p.y, ml(m[r2].x, p.x))) + m[r2].w;
    float3 po = quant(o, word);
    outb.Store3(posAt + 12 * v, asuint(po));
    if (nrmAt != 0xFFFFFFFF || rangeAt != 0xFFFFFFFF) {
        uint fw = vw(stream(cb(4), v));
        float3 nv, tv, nOut, tOut;
        frameOf(fw, nv, tv);
        if (tabN) {
            // The table's normal; the tangent made orthogonal to it.
            float d = mad(tv.z, tN.z, mad(tv.x, tN.x, ml(tv.y, tN.y)));
            float3 t = float3(mad(-tN.x, d, tv.x), mad(-tN.y, d, tv.y), mad(-tN.z, d, tv.z));
            tv = unit(t, 1.0 / sqrt(len2(t)));
            nv = tN;
        }
        if (!stop && cb(0x1C) != 0) {
            // Every bone applied: the frame from the weight gradients (seven
            // pairs of bytes after a half-float magnitude) times each bone's
            // displacement of the position; the normal is their cross product.
            uint fa = stream(cb(0x1C), v);
            float hm = f16tof32(vw(fa) & 0xFFFF);
            float mag = hm > 0 ? hm : 0;
            float h = (fw & 0x40000000) != 0 ? -1.0 : 1.0;
            float3 p0, ga = float3(0, 0, 0), gb = float3(0, 0, 0);
            [unroll] for (uint r4 = 0; r4 < 3; r4++) { float4 m0 = prow(base + bones[0], r4); p0[r4] = rdot(m0, p) + m0.w; }
            [unroll] for (uint k3 = 1; k3 < 8; k3++) {
                if (k3 < count) {
                    float ca = coef(fa, 2 * k3, mag), cb2 = coef(fa, 2 * k3 + 1, mag);
                    [unroll] for (uint r5 = 0; r5 < 3; r5++) {
                        float4 mk = prow(base + bones[k3], r5);
                        float d = (mk.w - p0[r5]) + rdot(mk, p);
                        ga[r5] = mad(ca, d, ga[r5]);
                        gb[r5] = mad(cb2, d, gb[r5]);
                    }
                }
            }
            float sa = min(ml(asfloat(0x3F666666), 1.0 / sqrt(len2(ga))), 1.0);
            float sb = min(ml(asfloat(0x3F666666), 1.0 / sqrt(len2(gb))), 1.0);
            float3 bi = float3(ml(mad(nv.z, tv.y, -ml(nv.y, tv.z)), h), ml(mad(nv.x, tv.z, -ml(nv.z, tv.x)), h), ml(mad(nv.y, tv.x, -ml(nv.x, tv.y)), h));
            float3 tn, bt;
            [unroll] for (uint r6 = 0; r6 < 3; r6++) { tn[r6] = mad(sa, ga[r6], rdot(m[r6], tv)); bt[r6] = mad(-sb, gb[r6], rdot(m[r6], bi)); }
            tn = unit(tn, 1.0 / sqrt(len2(tn)));
            bt = unit(bt, 1.0 / sqrt(len2(bt)));
            float3 c = float3(mad(bt.y, tn.z, -ml(bt.z, tn.y)), mad(bt.z, tn.x, -ml(bt.x, tn.z)), mad(bt.x, tn.y, -ml(bt.y, tn.x)));
            nOut = unit(c, ml(1.0 / sqrt(len2(c)), h));
            tOut = tn;
        } else {
            float3 t, u;
            [unroll] for (uint r3 = 0; r3 < 3; r3++) { t[r3] = rdot(m[r3], nv); u[r3] = rdot(m[r3], tv); }
            nOut = unit(t, 1.0 / sqrt(dot(t, t)));
            tOut = unit(u, 1.0 / sqrt(dot(u, u)));
        }
        if (nrmAt != 0xFFFFFFFF) outb.Store3(nrmAt + 12 * v, asuint(nOut));
        // The extra block; the previous frame's position is this one's
        // (our pass has no previous palette: no motion).
        if (rangeAt != 0xFFFFFFFF) outb.Store4(extraAt + 16 * v, uint4(packFrame(nOut, tOut, fw & 0x40000000), asuint(po)));
    }
}
"#;

/// Root constants per dispatch (the `Item` block of [`SHADER`]).
const CONSTANTS: usize = 24;
/// No hide masks / no normals.
const NONE: u32 = u32::MAX;
/// The hide-mask record: two halves of 20 x 256 bits.
pub const MASK_RECORD: usize = 1280;

fn le32(b: &[u8], at: usize) -> u32 {
    b.get(at..at + 4).map_or(0, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

/// One pre-skin dispatch's inputs, as the compute pass takes them. The
/// vertex view and the remap are shared (static per model; a live reader
/// keeps them by address).
#[derive(Clone, Debug)]
pub struct Item {
    /// The 96-B constant block; +0x28 is the first row of `palette`.
    pub b7: Vec<u8>,
    pub vb: Arc<Vec<u8>>,
    pub remap: Arc<Vec<u32>>,
    /// Palette rows from the block's base (48 B each).
    pub palette: Vec<u8>,
    /// The 1,280-B hide-mask record when the dispatch hides vertices.
    pub mask: Option<Vec<u8>>,
    pub rule: Bones,
    /// Variant 11's face run without its unfilled position table: the bind
    /// shape under the bones (see [`rule_for`]).
    pub neutral: bool,
    /// Variant 8 or 9 run under 4's or 5's rule without its displacement
    /// maps ([`BIT_UNMAPPED`]).
    pub unmapped: bool,
    /// The keys `vb` and `remap` are kept on the device by ([`Kept`]).
    pub vb_key: u64,
    pub remap_key: u64,
    /// Variant 2's delta table (t76, from its header; empty: no delta step).
    pub delta: Vec<u8>,
    /// Variant 11's face from its filled position table (t74 from b6+0,
    /// rebased to 0: [`face_table`]) and where its normals lie (b6+4);
    /// empty: no table step ([`BIT_FACE_TABLE`]).
    pub table: Vec<u8>,
    pub table_normals: u32,
}

impl Item {
    /// The item of a dispatch whose inputs the CPU reference takes (the
    /// mask record picked here as [`super::preskin::cpu_with`] picks it).
    pub fn from_inputs(inp: &Inputs, rule: Bones) -> Item {
        let c = |off: usize| le32(inp.b7, off);
        let pick = le32(inp.t7, (c(0x3c) as usize).wrapping_mul(48).wrapping_add(28)) >> 16;
        let rec = le32(inp.t71, pick as usize * 416 + 0xa0);
        let mask = ((rec as i32) >= 0 && c(0x8) != 0 && c(0x20) & c(0x24) != u32::MAX).then(|| {
            let at = rec as usize * MASK_RECORD;
            let mut m = vec![0u8; MASK_RECORD];
            if let Some(s) = inp.t78.get(at.min(inp.t78.len())..(at + MASK_RECORD).min(inp.t78.len())) {
                m[..s.len()].copy_from_slice(s);
            }
            m
        });
        let mut b7 = inp.b7.to_vec();
        b7.resize(b7.len().max(96), 0);
        let remap_bytes: Vec<u8> = inp.remap.iter().flat_map(|v| v.to_le_bytes()).collect();
        Item {
            b7,
            vb: Arc::new(inp.vb.to_vec()),
            remap: Arc::new(inp.remap.to_vec()),
            palette: inp.palette[..inp.palette.len() / 48 * 48].to_vec(),
            mask,
            rule,
            neutral: false,
            unmapped: false,
            vb_key: content_key(inp.vb),
            remap_key: content_key(&remap_bytes),
            delta: Vec::new(),
            table: Vec::new(),
            table_normals: 0,
        }
    }

    pub fn vertices(&self) -> u32 {
        le32(&self.b7, 0x38)
    }

    /// The range word (b7+0x34).
    pub fn word(&self) -> u32 {
        le32(&self.b7, 0x34)
    }

    fn rule_code(&self) -> u32 {
        match self.rule {
            Bones::Four => 0,
            Bones::Eight => 1,
            Bones::EightAll => 2,
            Bones::FourAll => 3,
        }
    }
}

/// `preskinbits` bit 0: variants 8 and 9 (4 and 5 plus a displacement
/// read from maps the oracle cannot bind yet) run under 4's and 5's
/// rule without the displacement. With the maps unbound the title's own
/// program gives the same positions (0 of 3,201 and 2 of 12,850 frame words
/// differ, pk4 f889); the title switches a ped's clothes between
/// 4 and 8 from frame to frame (pk4 f887: 26 rows v4, f889: the same 26 v8).
pub const BIT_UNMAPPED: u32 = 1;
/// `preskinbits` bit 1: variant 11 whose face is not run from an unfilled
/// table (a filled table, or the delta step from t76) runs from its bind
/// shape too, counted with the neutral faces.
pub const BIT_FACE_ANY: u32 = 2;
/// `preskinbits` bit 2: variants 1, 2, 6, 7 and 10 ([`ps::rule_more`]): 1
/// as 0, 2 as 0 with t76's delta first, 6 as 5, 7 and 10 as 3 and 5
/// without their displacement maps (counted with the unmapped).
pub const BIT_MORE: u32 = 4;
/// `preskinbits` bit 3: ped-flagged list rows that are no ped program (no
/// ped remap: dumpster, trolley, crate, door; non-ped effects) are drawn
/// as the skinned props they are ([`props_unflag`]).
pub const BIT_PROPS: u32 = 8;

/// `preskinbits` bit 4 (HEAD2): variant 11 whose position table is filled
/// (the head compute graph's, `headgraph=on`) runs its table step on the
/// device from the item's [`Item::table`] (b6+0xc = 0: t76's delta in
/// [`Item::delta`]), not from its bind shape. An unfilled table still runs
/// neutral.
pub const BIT_FACE_TABLE: u32 = 16;

/// The face a variant-11 dispatch reads from t74: the bytes from b6+0
/// (header and entries) through its last entry and normal, rebased to 0,
/// and the normals' offset (b6+4). Empty when b6+0xc is 0 (the delta step)
/// or t74 is empty.
pub fn face_table(inp: &Inputs) -> (Vec<u8>, u32) {
    let (base, normals, on) = (le32(inp.b6, 0) as usize, le32(inp.b6, 4), le32(inp.b6, 0xc));
    if on == 0 || inp.t74.is_empty() || base >= inp.t74.len() {
        return (Vec::new(), 0);
    }
    let c = |off: usize| le32(inp.b7, off);
    let idesc = c(0xc);
    let mut top = 0u64;
    if idesc & 0xff == 16 {
        for v in 0..c(0x38) {
            let at = (idesc >> 8) as usize + 16 * v as usize;
            let ib = [le32(inp.vb, at), le32(inp.vb, at + 4), le32(inp.vb, at + 8), le32(inp.vb, at + 12)];
            let e = match ib[3] & 0xff00_0000 {
                0xb600_0000 if ib[1] != 0xffff && ib[1] >> 16 != 0 => (ib[1] >> 16) - 1,
                0x5400_0000 => ib[3] >> 12 & 0xfff | ib[2] >> 16 & 0xf000,
                _ => continue,
            };
            top = top.max(u64::from(e) + 1);
        }
    }
    let span = (28 + 8 * top).max(u64::from(normals) + 4 * top) as usize;
    (inp.t74[base..(base + span).min(inp.t74.len())].to_vec(), normals)
}

/// The variants [`BIT_MORE`] adds.
const MORE: [u32; 5] = [1, 2, 6, 7, 10];

/// What a dispatch our pass does not run is counted under.
pub const WHY_FACE: &str = "face table or deltas";

/// The bone rule a pre-skin variant runs under in our pass, whether it is
/// variant 11's face drawn from its bind shape, and whether it is a
/// displacement variant without its maps (`bits`: [`BIT_UNMAPPED`],
/// [`BIT_FACE_ANY`]). Variant 11 runs only while its position table is
/// unfilled (`unfilled`, asked only for it): then without the table step,
/// as the CPU reference does ([`ps::table_unfilled`]); with a filled table,
/// or the delta step, it is not covered unless [`BIT_FACE_ANY`].
pub fn rule_for(variant: u32, b7: &[u8], bits: u32, unfilled: impl FnOnce() -> bool) -> Result<(Bones, bool, bool), &'static str> {
    if bits & BIT_UNMAPPED != 0 && (variant == 8 || variant == 9) {
        return Ok((if variant == 8 { Bones::Eight } else { Bones::EightAll }, false, true));
    }
    if bits & BIT_MORE != 0 && MORE.contains(&variant) {
        return ps::rule_more(variant).map(|(b, _)| (b, false, variant >= 7)).ok_or("variant not covered");
    }
    match ps::rule_of(variant, b7) {
        None => Err("variant not covered"),
        Some((b, Before::Nothing)) => Ok((b, false, false)),
        Some((b, Before::Tables)) if bits & BIT_FACE_ANY != 0 || unfilled() => Ok((b, true, false)),
        Some((b, Before::Tables)) if bits & BIT_FACE_TABLE != 0 => Ok((b, false, false)),
        Some(_) => Err(WHY_FACE),
    }
}

/// Whether our pass runs `variant` under `bits` (the face by its table, asked per dispatch).
pub fn covered(variant: u32, bits: u32) -> bool {
    ps::bones_of(variant).is_some() || variant == 11 || (bits & BIT_UNMAPPED != 0 && (variant == 8 || variant == 9)) || (bits & BIT_MORE != 0 && MORE.contains(&variant))
}

/// Whether `variant`'s item takes t76's delta (variant 2 under [`BIT_MORE`]).
pub fn takes_delta(variant: u32, bits: u32) -> bool {
    variant == 2 && bits & BIT_MORE != 0
}

/// The fewest bones of a full ped skeleton (pk8a f1027-1032, every flagged row by
/// bone count and effect: 389 and 496 ped_standard_*; 205/206 standard_vehicle*,
/// vehicle glass and wipers; 152 ped_standard_lod (props today); 9-46 shadowZCM
/// and standard props).
pub const PED_SKELETON_BONES: u32 = 300;

/// [`BIT_PROPS`]: the ped flag off on rows with the flag but no ped remap
/// (it came from the bone count alone: a non-ped program, pk6 f1205:
/// all 92 such rows are non-ped effects, all 24 with a remap ped_standard_*),
/// so they are drawn as skinned props. Returns how many.
/// A row on a full ped skeleton ([`PED_SKELETON_BONES`] or more) keeps the flag
/// with no remap too (VEH2, pk8a f1027-1032: 822 of 962 flagged 389/496-bone rows
/// have no remap, all ped_standard_*; unflagged, a near ped without a pre-skin item
/// was drawn with our skinning and exploded, f1029/f1032).
pub fn props_unflag(list: &mut super::FrameList) -> usize {
    let mut n = 0;
    for r in list.draws.iter_mut().filter(|r| r.ped && r.remap == super::NO_REMAP && r.bone_count < PED_SKELETON_BONES) {
        r.ped = false;
        n += 1;
    }
    n
}

/// The key of a stream kept by its content (never 0).
pub fn content_key(bytes: &[u8]) -> u64 {
    xnw_core::hash128(&[bytes]) as u64 | 1
}

/// The key of a stream kept by its title address, length and head (never 0).
pub fn address_key(addr: u64, len: u64, head: &[u8]) -> u64 {
    xnw_core::hash128(&[&addr.to_le_bytes(), &len.to_le_bytes(), head]) as u64 | 1
}

/// A frame's pre-skin: its items and the draws that take them.
#[derive(Clone, Debug, Default)]
pub struct PreskinFrame {
    pub items: Vec<Item>,
    /// (list draw index, item index, normals): the draw takes the item's
    /// positions (and normals when it reads the extra blocks).
    pub draws: Vec<(u32, u32, bool)>,
    /// Dispatches whose inputs could not be read, and the reads' time on
    /// the title's threads (live only).
    pub unreadable: u32,
    pub read_ms: f64,
    /// Live only: what the dispatches not run were, and where the reads' time went.
    pub reads: Reads,
    /// [`Hold`]: rows given a held item (this frame had none for them).
    pub held: u32,
}

/// The live reads of a frame's pre-skin dispatches, for the log line.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Reads {
    /// Dispatches of covered variants not run, by reason (`unreadable` is their sum).
    pub why: Vec<(&'static str, u32)>,
    /// Dispatches of the standard variants our pass does not run, by variant.
    pub uncovered: [u32; 13],
    /// The slowest dispatch's read (ms).
    pub max_ms: f64,
    /// Streams read whole this frame (not yet kept): count, KB and ms.
    pub new: u32,
    pub new_kb: u64,
    pub new_ms: f64,
}

impl Reads {
    /// One more dispatch not run for `why`.
    pub fn count(&mut self, why: &'static str) {
        match self.why.iter_mut().find(|w| w.0 == why) {
            Some(w) => w.1 += 1,
            None => self.why.push((why, 1)),
        }
    }

    /// The line's tail: `; not run: <why> n, ..; uncovered: v8 n, ..; new streams n (KB, ms); slowest x ms`.
    pub fn text(&self) -> String {
        let why: Vec<String> = self.why.iter().map(|(w, n)| format!("{w} {n}")).collect();
        let unc: Vec<String> = self.uncovered.iter().enumerate().filter(|x| *x.1 > 0).map(|(v, n)| format!("v{v} {n}")).collect();
        format!(
            "; not run: {}; uncovered: {}; new streams {} ({} KB, {:.2} ms), slowest read {:.2} ms",
            if why.is_empty() { "none".into() } else { why.join(", ") },
            if unc.is_empty() { "none".into() } else { unc.join(", ") },
            self.new,
            self.new_kb,
            self.new_ms,
            self.max_ms
        )
    }
}

impl PreskinFrame {
    /// The list rows that take an item.
    pub fn rows(&self) -> std::collections::HashSet<u32> {
        self.draws.iter().map(|d| d.0).collect()
    }
}

/// `pedhold=N` (default [`PED_HOLD`]; 0: off): lists a ped row keeps
/// its last item when its own frame has none.
pub const PED_HOLD: u32 = 3;

/// `XNW_WORLD_RENDER=...,pedhold=N` (the last one wins).
pub fn hold_switch(v: &str) -> u32 {
    v.split(',').filter_map(|p| p.trim().strip_prefix("pedhold=")).filter_map(|n| n.trim().parse().ok()).last().unwrap_or(PED_HOLD)
}

/// How far (m, camera-relative translation) a row may be from the held
/// item's row of the same geometry and still take it.
pub const HOLD_NEAR: f32 = 5.0;

/// One ped part's last item: the geometry and place of the row that took it.
#[derive(Clone, Debug)]
struct Held {
    geom: super::GeomKey,
    at: [f32; 3],
    item: Item,
    normals: bool,
    /// Lists since it was last joined (0: this one).
    age: u32,
}

/// The ped rows' last pre-skin items. Some lists have no item for
/// a ped the title draws (a test run f4020: 74 ped rows with their own record,
/// 0 joined, 78 peds skipped, against 1-6 in the frames around it; pk8a
/// f1029/f1032 whole near peds), and the pass then leaves a hole. A main-view
/// row with no item takes the item last joined to a row of the same geometry
/// within [`HOLD_NEAR`] for up to N lists (a held pose under this frame's
/// draw, as the title would draw the ped).
#[derive(Debug, Default)]
pub struct Hold {
    held: Vec<Held>,
    /// The last list's frame: a list from before it (a new sequence) starts over.
    frame: u64,
}

fn row_at(r: &super::DrawRecord) -> [f32; 3] {
    [r.world[12], r.world[13], r.world[14]]
}

fn far2(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

impl Hold {
    /// Gives `pre`'s unjoined main-view rows of `list` held items (up to
    /// `n` lists old), then keeps this list's joined items. Returns the rows
    /// held (also in `pre.held`).
    pub fn apply(&mut self, list: &super::FrameList, pre: &mut PreskinFrame, n: u32) -> u32 {
        pre.held = 0;
        if n == 0 || list.frame < self.frame {
            self.held.clear();
        }
        self.frame = list.frame;
        if n == 0 {
            return 0;
        }
        for h in &mut self.held {
            h.age += 1;
        }
        let vps = list.main_viewports();
        let main = |r: &super::DrawRecord| vps.contains(&r.viewport) && (r.ped || r.palette != super::NO_PALETTE);
        let nearest = |held: &[Held], r: &super::DrawRecord, ok: &dyn Fn(usize) -> bool| {
            let at = row_at(r);
            held.iter().enumerate().filter(|(k, h)| ok(*k) && h.geom == r.geom).map(|(k, h)| (k, far2(h.at, at))).filter(|x| x.1 <= HOLD_NEAR * HOLD_NEAR).min_by(|a, b| a.1.total_cmp(&b.1)).map(|x| x.0)
        };
        // This list's rows without an item: the nearest held item of their geometry.
        let joined = pre.rows();
        let fresh = pre.draws.len();
        let mut given: HashMap<usize, u32> = HashMap::new();
        for (row, r) in list.draws.iter().enumerate() {
            if joined.contains(&(row as u32)) || !main(r) {
                continue;
            }
            let Some(k) = nearest(&self.held, r, &|k| self.held[k].age <= n) else { continue };
            let h = &self.held[k];
            let item = *given.entry(k).or_insert_with(|| {
                pre.items.push(h.item.clone());
                pre.items.len() as u32 - 1
            });
            pre.draws.push((row as u32, item, h.normals));
        }
        pre.held = (pre.draws.len() - fresh) as u32;
        // This list's joined items replace their held ones (one per item, at its first main-view row).
        let mut kept: Vec<bool> = vec![false; self.held.len()];
        let mut seen = std::collections::HashSet::new();
        for &(row, item, normals) in &pre.draws[..fresh] {
            let Some(r) = list.draws.get(row as usize).filter(|r| main(r)) else { continue };
            if !seen.insert(item) {
                continue;
            }
            let Some(it) = pre.items.get(item as usize) else { continue };
            let h = Held { geom: r.geom, at: row_at(r), item: it.clone(), normals, age: 0 };
            let k = nearest(&self.held, r, &|k| !kept.get(k).copied().unwrap_or(true));
            match k {
                Some(k) => {
                    self.held[k] = h;
                    kept[k] = true;
                }
                None => {
                    self.held.push(h);
                    kept.push(true);
                }
            }
        }
        self.held.retain(|h| h.age <= n);
        pre.held
    }
}

/// `XNW_WORLD_RENDER=...,preskin=on` (default off; the last one wins).
pub fn screen_switch(v: &str) -> bool {
    v.split(',').map(str::trim).filter_map(|p| p.strip_prefix("preskin=")).last() == Some("on")
}

/// `XNW_WORLD_RENDER=...,preskinbits=<hex>` ([`BIT_UNMAPPED`], [`BIT_FACE_ANY`]; default 0, the last one wins).
pub fn screen_bits(v: &str) -> u32 {
    v.split(',').filter_map(|p| p.trim().strip_prefix("preskinbits=")).filter_map(|h| u32::from_str_radix(h.trim_start_matches("0x"), 16).ok()).last().unwrap_or(0)
}

/// The log line of a frame's pre-skin: the items, the draws drawn from
/// them, the unreadable dispatches and the GPU time of the dispatches.
pub fn line(frame: u64, pre: &PreskinFrame, s: &Stats) -> String {
    format!(
        "world render: frame {frame}: pre-skin: items {}, drawn {}, unreadable {}, ms {:.3} (joined {}, copies {}, no geometry {}, short {}, vertices {}, neutral faces {}, unmapped {}, reads {:.2} ms, record {:.3} ms, uploaded {} KB, kept {} KB{})",
        pre.items.len(),
        s.drawn,
        pre.unreadable,
        s.ms,
        pre.draws.len(),
        s.jobs,
        s.no_geometry,
        s.short,
        s.vertices,
        pre.items.iter().filter(|i| i.neutral).count(),
        pre.items.iter().filter(|i| i.unmapped).count(),
        pre.read_ms,
        s.record_ms,
        s.uploaded.div_ceil(1024),
        s.kept.div_ceil(1024),
        pre.reads.text()
    ) + &s.bind.map_or(String::new(), |b| b.text())
}

/// What the last frame's pre-skin did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    pub items: usize,
    /// Distinct (geometry, item) copies made, and draws given one.
    pub jobs: usize,
    pub drawn: usize,
    /// Draws whose geometry was not on the device or not a raw buffer.
    pub no_geometry: usize,
    /// Draws reading past their item's vertices (left out).
    pub short: usize,
    pub vertices: u64,
    /// The dispatches on the GPU (timestamps), of the frame before.
    pub ms: f64,
    /// The recording on the CPU (layout, uploads, commands).
    pub record_ms: f64,
    /// Bytes written to the upload buffer this frame; bytes of streams
    /// kept on the device.
    pub uploaded: u64,
    pub kept: u64,
    /// `preskinbind=on`: the bridged rows bound to our range this frame.
    pub bind: Option<Bind>,
}

/// One dispatch: an item, the draw's vertex buffer (its view in the heap
/// and bytes; None: positions only) and whether normals are written.
#[derive(Clone, Copy, Debug)]
pub struct Job {
    pub item: u32,
    pub geom: Option<(D3D12_GPU_DESCRIPTOR_HANDLE, u32)>,
    pub normals: bool,
    /// `preskinbind`: the job also writes the item's output as the
    /// title's range (float3 positions, an extra block a vertex), for the
    /// bridged rows that read it.
    pub range: bool,
}

/// Where a job's output lies in the output buffer: the copy (or the
/// positions) from `at`, `bytes` long; positions and normals (NONE).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Placed {
    pub at: u64,
    pub bytes: u64,
    pub pos_at: u64,
    pub nrm_at: u64,
    /// The title's range (its header; positions follow at `pos_at`
    /// = `range_at` + 32), NONE without [`Job::range`].
    pub range_at: u64,
}

/// The appended streams must stay addressable by a stream word (dword
/// offset in bits 9-31).
const MAX_STREAM_AT: u64 = 0x1FF_FFFC;
/// A range word's byte offset reach (bits 0-23 in 16-B units).
pub const RANGE_REACH: u32 = 0x0FFF_FFF0;

/// A block's place in a buffer: 16-B aligned, a zero dword past its end
/// (the unaligned read).
fn padded(bytes: u64) -> u64 {
    (bytes + 4).next_multiple_of(16)
}

fn put(input: &mut Vec<u8>, b: &[u8]) -> u32 {
    let at = input.len() as u32;
    input.extend_from_slice(b);
    input.resize(at as usize + padded(b.len() as u64) as usize, 0);
    at
}

fn remap_bytes(r: &[u32]) -> Vec<u8> {
    r.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Smallest buffer of kept streams.
const KEPT_MIN: u64 = 8 << 20;
/// Largest: past it a frame's streams are refused.
const KEPT_MAX: u64 = 1 << 30;

/// The streams kept on the device (pure bookkeeping): key to byte offset
/// in one buffer of `size` bytes, filled up to `used`. A frame whose new
/// streams do not fit starts the buffer again with only its own streams
/// (twice their size at least), so streams no longer drawn go then.
#[derive(Default, Debug)]
pub struct Kept {
    pub size: u64,
    pub used: u64,
    map: HashMap<u64, u64>,
}

/// Where one frame's streams lie in the kept buffer: per item [vertex
/// view, remap]; the new ones to upload, (item, 0 vertex view or 1 remap)
/// in the order they lie from `from`, `bytes` long; and the new size when
/// the buffer starts again.
#[derive(Debug, Default, PartialEq)]
pub struct Placement {
    pub at: Vec<[u32; 2]>,
    pub new: Vec<(usize, usize)>,
    pub from: u64,
    pub bytes: u64,
    pub rebuild: Option<u64>,
}

impl Kept {
    pub fn place(&mut self, items: &[Item]) -> Result<Placement, String> {
        let streams = |it: &Item| [(it.vb_key, padded(it.vb.len() as u64)), (it.remap_key, padded(4 * it.remap.len() as u64))];
        let mut seen = std::collections::HashSet::new();
        let (mut fresh, mut all) = (0u64, 0u64);
        for (k, n) in items.iter().flat_map(streams) {
            if seen.insert(k) {
                all += n;
                fresh += if self.map.contains_key(&k) { 0 } else { n };
            }
        }
        let mut p = Placement::default();
        if self.used + fresh > self.size {
            let size = if 2 * all <= self.size { self.size } else { (2 * all).next_power_of_two().max(KEPT_MIN) };
            if size > KEPT_MAX {
                return Err(format!("pre-skin: {all} B of streams in one frame"));
            }
            *self = Kept { size, used: 0, map: HashMap::new() };
            p.rebuild = Some(size);
        }
        p.from = self.used;
        for (i, it) in items.iter().enumerate() {
            let mut at = [0u32; 2];
            for (w, (k, n)) in streams(it).into_iter().enumerate() {
                at[w] = *self.map.entry(k).or_insert_with(|| {
                    p.new.push((i, w));
                    self.used += n;
                    self.used - n
                }) as u32;
            }
            p.at.push(at);
        }
        p.bytes = self.used - p.from;
        Ok(p)
    }
}

/// The input and output layouts of `jobs` (pure): the input bytes and the
/// constants per job, the output places and the output size. `kept`: per
/// item, where its vertex view and remap lie in the kept buffer. A job
/// whose appended streams lie past a stream word's reach is left out (None).
#[allow(clippy::type_complexity)]
pub fn layout(items: &[Item], kept: &[[u32; 2]], jobs: &[Job]) -> (Vec<u8>, Vec<Option<([u32; CONSTANTS], Placed)>>, u64) {
    let mut input: Vec<u8> = Vec::new();
    // Per item: [b7, vb, vb bytes, remap, remap n, palette, rows, mask], [delta, delta bytes, table, table bytes, table normals].
    let mut blocks = Vec::with_capacity(items.len());
    for (it, at) in items.iter().zip(kept) {
        let b7 = put(&mut input, &it.b7);
        let pal = put(&mut input, &it.palette);
        // SKIN2 `XNW_REDUCE=..,range=nomask`: no item hides a vertex.
        let nomask = crate::reduce::get().is_some_and(|r| r.range_nomask);
        let mask = it.mask.as_ref().filter(|_| !nomask).map_or(NONE, |m| put(&mut input, m));
        let delta = if it.delta.is_empty() { 0 } else { put(&mut input, &it.delta) };
        let table = if it.table.is_empty() { 0 } else { put(&mut input, &it.table) };
        blocks.push(([b7, at[0], it.vb.len() as u32, at[1], it.remap.len() as u32, pal, (it.palette.len() / 48) as u32, mask], [delta, it.delta.len() as u32, table, it.table.len() as u32, it.table_normals]));
    }
    let mut out_at = 0u64;
    let placed = jobs
        .iter()
        .map(|j| {
            let it = items.get(j.item as usize)?;
            let n = u64::from(it.vertices());
            let copy = j.geom.map_or(0, |g| u64::from(g.1) & !3);
            if j.geom.is_some() && (copy < 0x80 || copy + 24 * n > MAX_STREAM_AT) {
                return None;
            }
            let normals = j.normals && it.word() & 0x4000_0000 != 0;
            // With the title's range, its header goes 16-B aligned before
            // the positions, the extra blocks after them (16 B a vertex), ours after those.
            // A range past the word's reach, or streams past a stream word's, goes without (our draw keeps its copy).
            let range = j.range && (out_at + copy).next_multiple_of(16) <= u64::from(RANGE_REACH) && copy + 48 + 40 * n <= MAX_STREAM_AT;
            let range_at = if range { (out_at + copy).next_multiple_of(16) } else { u64::from(NONE) };
            let pos_at = if range { range_at + 32 } else { out_at + copy };
            let extra_at = pos_at + 12 * n;
            let nrm_at = if !normals {
                u64::from(NONE)
            } else if range {
                extra_at + 16 * n
            } else {
                pos_at + 12 * n
            };
            let end = if normals { nrm_at + 12 * n } else if range { extra_at + 16 * n } else { pos_at + 12 * n };
            let p = Placed { at: out_at, bytes: end - out_at, pos_at, nrm_at, range_at };
            let stream_word = |at: u64| 3 | 3 << 5 | (((at - out_at) / 4) as u32) << 9;
            let (b, d) = blocks[j.item as usize];
            let mut k = [0u32; CONSTANTS];
            k[..8].copy_from_slice(&b);
            k[16..21].copy_from_slice(&d);
            k[8] = out_at as u32;
            k[9] = (copy / 4) as u32;
            k[10] = pos_at as u32;
            k[11] = if normals { nrm_at as u32 } else { NONE };
            k[12] = it.rule_code();
            k[13] = stream_word(pos_at);
            k[14] = if normals { stream_word(nrm_at) } else { 0 };
            k[15] = (copy / 4).max(n) as u32;
            k[21] = if range { range_at as u32 } else { NONE };
            k[22] = if range { extra_at as u32 } else { 0 };
            out_at = end.next_multiple_of(256);
            Some((k, p))
        })
        .collect();
    (input, placed, out_at.max(256))
}

/// The compute pass: pipeline, the frame's input (upload, mapped) and
/// output (default heap, unordered access) buffers, timestamps, and the
/// heap slots of the world pass it holds for the draws' views.
#[derive(Default)]
pub struct GpuPreskin {
    pipeline: Option<(ID3D12RootSignature, ID3D12PipelineState)>,
    input: Option<(ID3D12Resource, *mut u8, usize)>,
    output: Option<(ID3D12Resource, u64)>,
    /// The kept streams: their buffer and its state, and the bookkeeping.
    store: Option<(ID3D12Resource, D3D12_RESOURCE_STATES)>,
    pub kept: Kept,
    stamps: Option<(ID3D12QueryHeap, ID3D12Resource)>,
    stamped: bool,
    /// Bytes uploaded by the last record and bytes kept.
    pub stats_upload: (u64, u64),
    pub slots: Vec<u32>,
    pub stats: Stats,
    /// `preskinbind=on`: the jobs write the title's range too; the
    /// frame's places by list draw index, the output buffer and its size,
    /// the stand-in bytes the bridged rows' range slots take ([`bind_rows`]).
    pub bind: bool,
    pub ranges: HashMap<u32, u64>,
    pub range_buf: Option<(ID3D12Resource, u64)>,
    pub standins: Vec<Arc<Vec<u8>>>,
    /// Pixel programs (first 32 bits) whose rows keep the title's empty
    /// range ([`BIND_DENY`] less `XNW_BRIDGE_ALLOW`; set with the switch).
    pub deny: Vec<u32>,
}

impl GpuPreskin {
    /// Zero bytes of `len` that stand for our output buffer in a bridged
    /// row's range slot (kept across frames by length; only their address
    /// and length are used: the bridge binds the buffer there, nothing is uploaded).
    pub fn standin(&mut self, len: u64) -> Arc<Vec<u8>> {
        if let Some(a) = self.standins.iter().find(|a| a.len() as u64 == len) {
            return a.clone();
        }
        if self.standins.len() >= 4 {
            self.standins.remove(0);
        }
        let a = Arc::new(vec![0u8; len as usize]);
        self.standins.push(a.clone());
        a
    }

    /// The bridge's view of our output buffer for the stand-ins' bytes
    /// ([`super::bridged::Target::preskin`]); None without a range this frame.
    pub fn target(&self) -> Option<(&ID3D12Resource, Vec<usize>)> {
        let (b, _) = self.range_buf.as_ref()?;
        Some((b, self.standins.iter().map(|a| a.as_ptr() as usize).collect()))
    }
}

/// `XNW_WORLD_RENDER=...,preskinbind=on|off` (default off; the last one wins).
pub fn bind_switch(v: &str) -> bool {
    v.split(',').map(str::trim).filter_map(|p| p.strip_prefix("preskinbind=")).last() == Some("on")
}

/// Pixel programs (first 32 bits of the container hash) whose bridged rows
/// keep the title's empty range under `preskinbind=on`, with why; on any
/// adapter. `XNW_BRIDGE_ALLOW=<hash>+..` lets one through (as `bridge::denied`).
pub const BIND_DENY: &[(u32, &str)] = &[(
    0x957f_c3f6,
    "ped skin with our range bound (real coverage) ran past the 600 s GPU watchdog on WARP 1.0.22, alone with main's 15, pk8a f1027 (twice); without the range its rows draw 0 px and pass",
)];

/// [`BIND_DENY`] less the programs `allow` (`XNW_BRIDGE_ALLOW`) names.
pub fn bind_deny(allow: &str) -> Vec<u32> {
    let allowed: Vec<u32> = allow.split('+').filter_map(|h| u32::from_str_radix(h.trim().trim_start_matches("0x"), 16).ok()).collect();
    BIND_DENY.iter().map(|d| d.0).filter(|d| !allowed.contains(d)).collect()
}

/// The frame table's (root 0) slots of the title's pre-skin output: T8+0x420
/// (33, buffer B) and T8+0xa80 (84, buffer A) [V 40943a69 0x710-0x718].
pub const RANGE_SLOTS: [u32; 2] = [33, 84];
/// The PreSkinD record: root 1's slot 0 (the descriptor at s[24:25]+0); its range word at +0x54 [V 40943a69 0x6cc-0x6ec].
pub const RECORD_ROOT: usize = 1;
pub const RECORD_WORD: usize = 0x54;

/// What [`bind_rows`] did: bridged rows that read the pre-skin range, those
/// given ours, and those left with the title's (empty) buffers, by why.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bind {
    pub rows: u32,
    pub bound: u32,
    /// The record's range word was the item's own (the join agrees).
    pub same_word: u32,
    /// Not a pre-skin draw of the list (no dispatch joined, or not covered).
    pub no_item: u32,
    /// A pre-skin draw without our range (no geometry, short, past reach, `preskin` failed).
    pub no_job: u32,
    /// No record at root 1 slot 0 (or shorter than its word).
    pub no_record: u32,
    /// A range slot read as a constant buffer (64 KiB: ours does not fit).
    pub cbv: u32,
    /// Its pixel program in [`BIND_DENY`].
    pub denied: u32,
}

impl Bind {
    pub fn text(&self) -> String {
        format!("; bind (preskinbind): rows {}, bound {} (range word the item's {}), unbound: no item {}, no job {}, no record {}, constant-buffer slot {}, denied {}", self.rows, self.bound, self.same_word, self.no_item, self.no_job, self.no_record, self.cbv, self.denied)
    }
}

/// Whether a bridged row reads the title's pre-skin output (a range slot of
/// its frame table with bytes: a PreSkinD mesh program).
pub fn reads_range(o: &super::bridged::Owned) -> bool {
    use super::bridged::{ORoot, OSlot};
    matches!(o.roots.first(), Some(ORoot::Table(t)) if t.iter().any(|(k, s)| RANGE_SLOTS.contains(k) && matches!(s, OSlot::Buffer(..))))
}

/// pure: the bridged rows of `f` that read the pre-skin range and
/// whose list draw has our range (`ranges`: draw -> byte offset in our
/// output buffer of `buf_len` bytes) read ours: the record's range word
/// points at it (float3, the record's buffer and primary bits kept: our
/// range always has its extra blocks), each range slot takes `standin`'s
/// bytes (as long as the buffer, in whole units of the slot's kind: the
/// bridge binds our buffer for them), and the row has no identity (its
/// binding is made each frame, never held or replayed). Rows whose pixel
/// program is in `deny` are left as they are. None: no row bound.
pub fn bind_rows(f: &super::bridged::Frame, ranges: &HashMap<u32, u64>, pre: &PreskinFrame, buf_len: u64, deny: &[u32], standin: &mut dyn FnMut(u64) -> Arc<Vec<u8>>) -> (Option<super::bridged::Frame>, Bind) {
    use super::bridged::{ORoot, OSlot};
    let mut st = Bind::default();
    let mut rows = Vec::new();
    for (i, (row, o)) in f.rows.iter().enumerate() {
        if !reads_range(o) {
            continue;
        }
        st.rows += 1;
        if deny.contains(&((o.key.1 >> 96) as u32)) {
            st.denied += 1;
            continue;
        }
        let item = pre.draws.iter().find(|d| d.0 == *row).map(|d| d.1);
        let Some(&at) = ranges.get(row).filter(|a| **a <= u64::from(RANGE_REACH)) else {
            if item.is_some() {
                st.no_job += 1;
            } else {
                st.no_item += 1;
            }
            continue;
        };
        let rec = match o.roots.get(RECORD_ROOT) {
            Some(ORoot::Table(t)) => t.iter().position(|(k, s)| *k == 0 && matches!(s, OSlot::Buffer(_, b) if b.len() >= RECORD_WORD + 4)),
            _ => None,
        };
        let Some(rec) = rec else {
            st.no_record += 1;
            continue;
        };
        let ORoot::Table(t0) = &o.roots[0] else { continue };
        if t0.iter().any(|(k, s)| RANGE_SLOTS.contains(k) && matches!(s, OSlot::Buffer(ps::Kind::Cbv, _))) {
            st.cbv += 1;
            continue;
        }
        let mut n = o.clone();
        if let ORoot::Table(t) = &mut n.roots[RECORD_ROOT]
            && let OSlot::Buffer(kind, b) = &t[rec].1
        {
            let old = le32(b, RECORD_WORD);
            if item.and_then(|i| pre.items.get(i as usize)).is_some_and(|it| it.word() == old) {
                st.same_word += 1;
            }
            let word = (old & 0x4300_0000) | 0x2000_0000 | (at >> 4) as u32;
            let mut v = b.to_vec();
            v[RECORD_WORD..RECORD_WORD + 4].copy_from_slice(&word.to_le_bytes());
            t[rec].1 = OSlot::Buffer(*kind, Arc::new(v));
        }
        if let ORoot::Table(t) = &mut n.roots[0] {
            for (k, s) in t.iter_mut() {
                if let OSlot::Buffer(kind, _) = s
                    && RANGE_SLOTS.contains(k)
                {
                    let unit = super::bridge::unit(*kind);
                    *s = OSlot::Buffer(*kind, standin(buf_len / unit * unit));
                }
            }
        }
        n.ident = 0;
        st.bound += 1;
        rows.push((i, n));
    }
    if rows.is_empty() {
        return (None, st);
    }
    let mut all: Vec<(u32, super::bridged::Owned)> = f.rows.clone();
    for (i, n) in rows {
        all[i].1 = n;
    }
    let mut g = super::bridged::Frame::new(all);
    (g.not_ready, g.unread) = (f.not_ready, f.unread);
    (Some(g), st)
}

/// SKIN2: under `XNW_REDUCE`, each bound row of the reduced program is
/// named (its range offset, the buffer's length, its record word, groups);
/// `range=zero` gives its range slots zeros of their length.
fn reduce_bound(g: &mut super::bridged::Frame, ranges: &HashMap<u32, u64>, pre: &PreskinFrame, buf_len: u64) {
    use super::bridged::{ORoot, OSlot};
    for (row, o) in g.rows.iter_mut() {
        let Some(&at) = ranges.get(row) else { continue };
        if o.ident != 0 || !reads_range(o) || crate::reduce::get().is_none_or(|r| (o.key.1 >> 96) as u32 != r.hash) {
            continue;
        }
        let word = match o.roots.get(RECORD_ROOT) {
            Some(ORoot::Table(t)) => t.iter().find_map(|(k, s)| match s {
                OSlot::Buffer(_, b) if *k == 0 && b.len() >= RECORD_WORD + 4 => Some(le32(b, RECORD_WORD)),
                _ => None,
            }),
            _ => None,
        };
        let zero = crate::reduce::range_zero(o.key.1);
        if zero && let ORoot::Table(t) = &mut o.roots[0] {
            for (k, s) in t.iter_mut() {
                if let OSlot::Buffer(_, b) = s
                    && RANGE_SLOTS.contains(k)
                {
                    *b = Arc::new(vec![0; b.len()]);
                }
            }
        }
        let item = pre.draws.iter().find(|d| d.0 == *row).and_then(|d| pre.items.get(d.1 as usize));
        let what = item.map_or("no item".to_string(), |it| format!("item: {} vertices, hide mask {}, rule {:?}", it.vertices(), it.mask.is_some(), it.rule));
        println!("reduce: bound row {row}: range at {at:#x} of {buf_len:#x}, record word {:#010x}, groups {}, mesh {:08x}, {what}{}", word.unwrap_or(0), o.groups, (o.key.0 >> 96) as u32, if zero { ", range slots zeroed" } else { "" });
    }
}

fn buffer(device: &ID3D12Device, size: u64, heap: D3D12_HEAP_TYPE, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES) -> Result<ID3D12Resource, String> {
    let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: size.max(256), Height: 1, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_UNKNOWN, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, Flags: flags, ..Default::default() };
    let mut r: Option<ID3D12Resource> = None;
    // SAFETY: descriptions outlive the call.
    unsafe { device.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r) }.map_err(|e| format!("pre-skin buffer ({size} B): {e}"))?;
    r.ok_or_else(|| "pre-skin buffer: none".into())
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

impl GpuPreskin {
    fn pipeline(&mut self, device: &ID3D12Device) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
        if let Some(p) = &self.pipeline {
            return Ok(p.clone());
        }
        let cs = super::compile_source(SHADER, c"cs", c"cs_5_1")?;
        // SAFETY: the blob lives through the calls.
        let p = unsafe {
            let bytes = std::slice::from_raw_parts(cs.GetBufferPointer().cast::<u8>(), cs.GetBufferSize());
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).map_err(|e| format!("pre-skin root: {e}"))?;
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: std::mem::transmute_copy(&root), CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.GetBufferPointer(), BytecodeLength: cs.GetBufferSize() }, ..Default::default() };
            let pso: ID3D12PipelineState = device.CreateComputePipelineState(&desc).map_err(|e| format!("pre-skin pipeline: {e}"))?;
            (root, pso)
        };
        self.pipeline = Some(p.clone());
        Ok(p)
    }

    /// Records the dispatches of `jobs` on `list` (its shader-visible heap
    /// set): uploads the inputs, runs one dispatch per job and leaves the
    /// output buffer readable by shaders. `table`: any view of that heap,
    /// bound for positions-only jobs (never read by them). Returns the
    /// output buffer and each job's place (None: left out).
    pub fn record(&mut self, device: &ID3D12Device, list: &ID3D12GraphicsCommandList, items: &[Item], jobs: &[Job], table: D3D12_GPU_DESCRIPTOR_HANDLE) -> Result<(ID3D12Resource, Vec<Option<Placed>>), String> {
        let (root, pso) = self.pipeline(device)?;
        let placement = self.kept.place(items)?;
        if placement.rebuild.is_some() || self.store.is_none() {
            // The buffer before is free: the caller waits for each frame's list.
            self.store = Some((buffer(device, self.kept.size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON)?, D3D12_RESOURCE_STATE_COMMON));
        }
        let (mut input, consts, out_bytes) = layout(items, &placement.at, jobs);
        // The new streams after the blocks, as they will lie in the kept buffer.
        let fresh_at = input.len().next_multiple_of(256);
        input.resize(fresh_at, 0);
        for &(i, w) in &placement.new {
            if w == 0 {
                put(&mut input, &items[i].vb);
            } else {
                put(&mut input, &remap_bytes(&items[i].remap));
            }
        }
        debug_assert_eq!((input.len() - fresh_at) as u64, placement.bytes);
        self.stats_upload = (input.len() as u64, self.kept.used);
        if self.input.as_ref().is_none_or(|i| i.2 < input.len()) {
            let size = input.len().next_power_of_two().max(1 << 20);
            let r = buffer(device, size as u64, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
            let mut p = std::ptr::null_mut();
            // SAFETY: an upload buffer stays mapped for its life.
            unsafe { r.Map(0, None, Some(&mut p)) }.map_err(|e| format!("pre-skin input Map: {e}"))?;
            self.input = Some((r, p.cast::<u8>(), size));
        }
        if self.output.as_ref().is_none_or(|o| o.1 < out_bytes) {
            let size = out_bytes.next_power_of_two().max(4 << 20);
            self.output = Some((buffer(device, size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON)?, size));
        }
        if self.stamps.is_none() {
            let mut h: Option<ID3D12QueryHeap> = None;
            // SAFETY: as above.
            unsafe { device.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2, NodeMask: 0 }, &mut h) }.map_err(|e| format!("pre-skin query heap: {e}"))?;
            let rb = buffer(device, 16, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
            self.stamps = Some((h.ok_or("pre-skin query heap: none")?, rb));
        }
        let (ibuf, mapped, _) = self.input.as_ref().expect("made above");
        // SAFETY: `input` fits in the mapping (checked above); the GPU is done with it (the caller waits each frame).
        unsafe { std::ptr::copy_nonoverlapping(input.as_ptr(), *mapped, input.len()) };
        let (obuf, _) = self.output.as_ref().expect("made above");
        let (qh, qrb) = self.stamps.as_ref().expect("made above");
        let (store, state) = self.store.as_mut().expect("made above");
        // SAFETY: recording on the caller's open list; every object outlives the list's execution (held here).
        unsafe {
            if placement.bytes > 0 {
                if *state != D3D12_RESOURCE_STATE_COPY_DEST {
                    barrier(list, store, *state, D3D12_RESOURCE_STATE_COPY_DEST);
                }
                list.CopyBufferRegion(&*store, placement.from, ibuf, fresh_at as u64, placement.bytes);
                *state = D3D12_RESOURCE_STATE_COPY_DEST;
            }
            if *state != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE {
                barrier(list, store, *state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                *state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            }
            barrier(list, obuf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            list.SetComputeRootSignature(&root);
            list.SetPipelineState(&pso);
            list.SetComputeRootShaderResourceView(1, ibuf.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2, obuf.GetGPUVirtualAddress());
            list.SetComputeRootDescriptorTable(3, table);
            list.SetComputeRootShaderResourceView(4, store.GetGPUVirtualAddress());
            list.EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            for (j, c) in jobs.iter().zip(&consts) {
                let Some((k, _)) = c else { continue };
                if let Some((h, _)) = j.geom {
                    list.SetComputeRootDescriptorTable(3, h);
                }
                list.SetComputeRoot32BitConstants(0, CONSTANTS as u32, k.as_ptr().cast(), 0);
                list.Dispatch(k[15].div_ceil(64).max(1), 1, 1);
            }
            list.EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            list.ResolveQueryData(qh, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, qrb, 0);
            barrier(list, obuf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        self.stamped = true;
        Ok((obuf.clone(), consts.into_iter().map(|c| c.map(|c| c.1)).collect()))
    }

    /// The GPU milliseconds of the last recorded dispatches, once their list
    /// has finished (`frequency`: the queue's timestamp ticks per second).
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
        Some(t[1].saturating_sub(t[0]) as f64 * 1e3 / frequency.max(1) as f64)
    }
}

/// The skinned positions (and normals, when the range is a primary one)
/// of every item on `host`'s device, read back; and the GPU milliseconds
/// of all dispatches. For checks against the CPU reference.
#[allow(clippy::type_complexity)]
pub fn run(host: &xnw_gpu::host::Host, items: &[Item]) -> Result<(Vec<(Vec<[f32; 3]>, Vec<[f32; 3]>)>, f64), String> {
    let (out, placed, ms) = dispatch(host, items, false)?;
    let f3 = |at: u64| -> [f32; 3] { std::array::from_fn(|k| f32::from_bits(le32(&out, at as usize + 4 * k))) };
    let res = items
        .iter()
        .zip(&placed)
        .map(|(it, p)| {
            let Some(p) = p else { return (Vec::new(), Vec::new()) };
            let n = u64::from(it.vertices());
            let pos = (0..n).map(|v| f3(p.pos_at + 12 * v)).collect();
            let nrm = if p.nrm_at == u64::from(NONE) { Vec::new() } else { (0..n).map(|v| f3(p.nrm_at + 12 * v)).collect() };
            (pos, nrm)
        })
        .collect();
    Ok((res, ms))
}

/// Each item's output as the title's range our pass writes for the
/// bridged rows (`preskinbind`), read back: header, float3 positions, the
/// extra blocks (32 + 28 n bytes; empty when left out).
pub fn run_ranges(host: &xnw_gpu::host::Host, items: &[Item]) -> Result<Vec<Vec<u8>>, String> {
    let (out, placed, _) = dispatch(host, items, true)?;
    Ok(items
        .iter()
        .zip(&placed)
        .map(|(it, p)| match p.filter(|p| p.range_at != u64::from(NONE)) {
            Some(p) => out[p.range_at as usize..][..32 + 28 * it.vertices() as usize].to_vec(),
            None => Vec::new(),
        })
        .collect())
}

/// Every item once (positions only, with the title's range when `range`)
/// on `host`'s device: the output buffer read back, each job's place, the
/// GPU milliseconds.
fn dispatch(host: &xnw_gpu::host::Host, items: &[Item], range: bool) -> Result<(Vec<u8>, Vec<Option<Placed>>, f64), String> {
    let dev = &host.device;
    let e = |e: windows::core::Error| format!("pre-skin run: {e}");
    let mut pass = GpuPreskin::default();
    // SAFETY: D3D12 calls on our own objects; the fence wait ends the GPU's use of every buffer before they drop.
    unsafe {
        let heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 1, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 }).map_err(e)?;
        let null = D3D12_SHADER_RESOURCE_VIEW_DESC { Format: DXGI_FORMAT_R32_TYPELESS, ViewDimension: D3D12_SRV_DIMENSION_BUFFER, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Buffer: D3D12_BUFFER_SRV { FirstElement: 0, NumElements: 0, StructureByteStride: 0, Flags: D3D12_BUFFER_SRV_FLAG_RAW } } };
        dev.CreateShaderResourceView(None, Some(&null), heap.GetCPUDescriptorHandleForHeapStart());
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(e)?;
        list.SetDescriptorHeaps(&[Some(heap.clone())]);
        // Positions only: the null view at slot 0 is the table (never read).
        let jobs: Vec<Job> = (0..items.len() as u32).map(|i| Job { item: i, geom: None, normals: true, range }).collect();
        let (_, placed) = pass.record(dev, &list, items, &jobs, heap.GetGPUDescriptorHandleForHeapStart())?;
        let out_bytes = placed.iter().flatten().map(|p| p.at + p.bytes).max().unwrap_or(0).max(256);
        let (obuf, _) = pass.output.as_ref().ok_or("no output")?;
        let rb = buffer(dev, out_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
        barrier(&list, obuf, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list.CopyBufferRegion(&rb, 0, obuf, 0, out_bytes);
        list.Close().map_err(e)?;
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(e)?;
        queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(&list).map_err(e)?)]);
        let sent = xnw_gpu::gpuwatch::begin("preskin gpu");
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        queue.Signal(&fence, 1).map_err(e)?;
        while fence.GetCompletedValue() < 1 {
            dev.GetDeviceRemovedReason().map_err(|x| format!("device removed: {}", e(x)))?;
            if xnw_gpu::gpuwatch::overdue("preskin gpu", sent) {
                return Err("preskin gpu: the submission was given up".into());
            }
            std::thread::yield_now();
        }
        xnw_gpu::gpuwatch::end("preskin gpu", sent, None);
        let ms = pass.gpu_ms(queue.GetTimestampFrequency().map_err(e)?).unwrap_or(0.0);
        let mut p = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut p)).map_err(e)?;
        let out = std::slice::from_raw_parts(p.cast::<u8>(), out_bytes as usize).to_vec();
        rb.Unmap(0, None);
        Ok((out, placed, ms))
    }
}


impl super::Pass {
    /// Records the frame's pre-skin on the pass's open list, before the
    /// draws: one job per distinct (geometry, item, normals) of the draws
    /// that take an item, and a view of each job's copy in a heap slot the
    /// pre-skin holds. Returns, per list draw index, the copy's slot and
    /// bytes; a draw left out keeps its own vertex buffer. A failure is
    /// counted and logged once, never a failed frame.
    pub(super) fn preskin_record(&mut self, list: &super::FrameList, l: &ID3D12GraphicsCommandList) -> std::collections::HashMap<u32, (u32, u32)> {
        let t0 = std::time::Instant::now();
        let mut views = std::collections::HashMap::new();
        let pre = &list.preskin;
        let mut st = Stats { items: pre.items.len(), vertices: pre.items.iter().map(|i| u64::from(i.vertices())).sum(), bind: self.preskin.bind.then(Bind::default), ..Stats::default() };
        let range = self.preskin.bind;
        self.preskin.ranges.clear();
        self.preskin.range_buf = None;
        if pre.draws.is_empty() {
            self.preskin.stats = st;
            return views;
        }
        let mut jobs: Vec<Job> = Vec::new();
        let mut job_of: std::collections::HashMap<(super::GeomKey, u32, bool), usize> = std::collections::HashMap::new();
        let mut draw_job = Vec::with_capacity(pre.draws.len());
        for &(d, item, normals) in &pre.draws {
            let Some(r) = list.draws.get(d as usize) else { continue };
            if item as usize >= pre.items.len() {
                continue;
            }
            let Some(((slot, len), top)) = self.cache.get_or_make(r.geom, list.frame, || None).and_then(|g| Some((g.raw?, g.top))) else {
                st.no_geometry += 1;
                continue;
            };
            // A draw reading past the item's vertices is joined to the wrong
            // item (another frame's range word): left out, counted.
            if top > pre.items[item as usize].vertices() {
                st.short += 1;
                continue;
            }
            let h = self.gpu_slot(slot);
            let j = *job_of.entry((r.geom, item, normals)).or_insert_with(|| {
                jobs.push(Job { item, geom: Some((h, len)), normals, range });
                jobs.len() - 1
            });
            draw_job.push((d, j));
        }
        let table = self.gpu_slot(0);
        let (obuf, placed) = match self.preskin.record(&self.device, l, &pre.items, &jobs, table) {
            Ok(x) => {
                (st.uploaded, st.kept) = self.preskin.stats_upload;
                x
            }
            Err(e) => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| xnw_core::warn!("world render: pre-skin not recorded: {e}"));
                self.preskin.stats = st;
                return views;
            }
        };
        while self.preskin.slots.len() < jobs.len() {
            let slot = self.free_slots.pop().or_else(|| {
                (self.next_slot < super::HEAP_SLOTS - super::TEX_SLOTS).then(|| {
                    self.next_slot += 1;
                    self.next_slot - 1
                })
            });
            match slot {
                Some(s) => self.preskin.slots.push(s),
                None => break,
            }
        }
        let mut job_view = vec![None; jobs.len()];
        for (j, p) in placed.iter().enumerate() {
            if let (Some(p), Some(&slot)) = (p, self.preskin.slots.get(j)) {
                self.raw_view_at(&obuf, p.at, p.bytes, slot);
                job_view[j] = Some((slot, p.bytes as u32));
            }
        }
        st.jobs = job_view.iter().flatten().count();
        if range {
            // Each draw's range, for the bridged rows of the frame ([`bind_rows`]).
            for &(d, j) in &draw_job {
                if let Some(p) = placed[j].filter(|p| p.range_at != u64::from(NONE)) {
                    self.preskin.ranges.insert(d, p.range_at);
                }
            }
            self.preskin.range_buf = self.preskin.output.as_ref().map(|o| (o.0.clone(), o.1));
        }
        for (d, j) in draw_job {
            if let Some(v) = job_view[j] {
                views.insert(d, v);
                st.drawn += 1;
            }
        }
        st.record_ms = t0.elapsed().as_secs_f64() * 1e3;
        self.preskin.stats = st;
        views
    }

    /// `preskinbind=on`: the frame's bridged rows with our range bound
    /// where they read the title's ([`bind_rows`]; None: none bound or the
    /// switch off); the counts go to the pre-skin line.
    pub(super) fn preskin_bind_rows(&mut self, list: &super::FrameList, f: &super::bridged::Frame) -> Option<super::bridged::Frame> {
        if !self.preskin.bind {
            return None;
        }
        let buf_len = self.preskin.range_buf.as_ref().map_or(0, |b| b.1);
        let ranges = std::mem::take(&mut self.preskin.ranges);
        let (mut g, b) = bind_rows(f, &ranges, &list.preskin, buf_len, &self.preskin.deny.clone(), &mut |len| self.preskin.standin(len));
        if crate::reduce::get().is_some()
            && let Some(g) = g.as_mut()
        {
            reduce_bound(g, &ranges, &list.preskin, buf_len);
        }
        self.preskin.ranges = ranges;
        self.preskin.stats.bind = Some(b);
        g
    }

    /// Makes the compute pipeline now (an FXC compile), so the first frame
    /// with items does not pay for it; returns the milliseconds it took.
    pub fn preskin_prepare(&mut self) -> Result<f64, String> {
        let t = std::time::Instant::now();
        self.preskin.pipeline(&self.device)?;
        Ok(t.elapsed().as_secs_f64() * 1e3)
    }

    /// What the last frame's pre-skin did, with the GPU time of its
    /// dispatches (read once its list has finished).
    pub fn preskin_stats(&mut self) -> Stats {
        // SAFETY: a live queue.
        let freq = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(0);
        if let Some(ms) = self.preskin.gpu_ms(freq) {
            self.preskin.stats.ms = ms;
        }
        self.preskin.stats
    }
}

#[cfg(test)]
mod tests {
    use super::super::preskin::{self as ps, tests::case};
    use super::*;

    /// The layout: inputs 16-B aligned with a zero dword after each block,
    /// the kept streams' places, a job's copy then its positions then its
    /// normals, the stream words pointing at them, jobs 256-B apart; a copy
    /// too short for a stream header is left out.
    #[test]
    fn layout_places_copy_positions_normals() {
        let c = case();
        let it = Item::from_inputs(&c.inputs(), ps::Bones::Four);
        assert!(it.mask.is_none());
        assert_eq!((it.vertices(), it.palette.len()), (2, 48 * 13));
        let h = D3D12_GPU_DESCRIPTOR_HANDLE { ptr: 64 };
        let jobs = [Job { item: 0, geom: Some((h, 0x102)), normals: true, range: false }, Job { item: 0, geom: None, normals: false, range: false }, Job { item: 0, geom: Some((h, 0x40)), normals: false, range: false }];
        let (input, placed, out) = layout(std::slice::from_ref(&it), &[[0x1000, 0x2000]], &jobs);
        let (k0, p0) = placed[0].unwrap();
        assert_eq!(p0, Placed { at: 0, bytes: 0x100 + 24 + 24, pos_at: 0x100, nrm_at: 0x118, range_at: u64::from(NONE) });
        assert_eq!((k0[9], k0[10], k0[11], k0[13], k0[14], k0[15]), (0x40, 0x100, 0x118, 3 | 3 << 5 | 0x40 << 9, 3 | 3 << 5 | 0x46 << 9, 0x40));
        let (k1, p1) = placed[1].unwrap();
        assert_eq!(p1, Placed { at: 0x200, bytes: 24, pos_at: 0x200, nrm_at: u64::from(NONE), range_at: u64::from(NONE) });
        assert_eq!((k1[9], k1[11], k1[15], k1[21]), (0, NONE, 2, NONE));
        assert!(placed[2].is_none());
        assert_eq!(out, 0x300);
        assert_eq!(k0[0] % 16, 0);
        assert_eq!(&input[k0[0] as usize..][..96], &it.b7[..]);
        assert_eq!((k0[1], k0[2], k0[3], k0[4]), (0x1000, it.vb.len() as u32, 0x2000, it.remap.len() as u32));
        assert_eq!(&input[k0[5] as usize..][..it.palette.len()], &it.palette[..]);
        assert_eq!(le32(&input, k0[5] as usize + it.palette.len()), 0);
    }

    /// A job with the title's range lays its header 16-B aligned after
    /// the copy, the float3 positions 32 B on, an extra block (16 B) a vertex
    /// after them, our normals last; the stream words still point at ours.
    #[test]
    fn layout_places_title_range() {
        let c = case();
        let it = Item::from_inputs(&c.inputs(), ps::Bones::Four);
        let h = D3D12_GPU_DESCRIPTOR_HANDLE { ptr: 64 };
        let jobs = [Job { item: 0, geom: Some((h, 0x102)), normals: true, range: true }, Job { item: 0, geom: Some((h, 0x100)), normals: false, range: true }];
        let (_, placed, out) = layout(std::slice::from_ref(&it), &[[0x1000, 0x2000]], &jobs);
        let (k0, p0) = placed[0].unwrap();
        // copy 0x100 (0x102 & !3), header at 0x100, positions 0x120 (2 x 12), extra 0x138 (2 x 16), normals 0x158.
        assert_eq!(p0, Placed { at: 0, bytes: 0x158 + 24, pos_at: 0x120, nrm_at: 0x158, range_at: 0x100 });
        assert_eq!((k0[10], k0[11], k0[13], k0[14], k0[21], k0[22]), (0x120, 0x158, 3 | 3 << 5 | 0x48 << 9, 3 | 3 << 5 | 0x56 << 9, 0x100, 0x138));
        let (k1, p1) = placed[1].unwrap();
        assert_eq!(p1, Placed { at: 0x200, bytes: 0x100 + 32 + 24 + 32, pos_at: 0x320, nrm_at: u64::from(NONE), range_at: 0x300 });
        assert_eq!((k1[11], k1[21], k1[22]), (NONE, 0x300, 0x338));
        assert_eq!(out, 0x400);
        assert_eq!(p0.range_at % 16, 0);
    }

    /// `bind_rows`: a row reading the range whose draw has ours gets
    /// its record word pointed there (float3, the record's buffer and
    /// primary bits kept), stand-ins as long as the buffer in whole units
    /// at both range slots, no identity; the others are counted by why and
    /// left as they were; a row reading no range is not counted.
    #[test]
    fn bind_rows_points_records_at_our_range() {
        use super::super::bridged::{Frame, ORoot, OSlot, Owned};
        let word = 0x4100_0010u32;
        let mut it = Item::from_inputs(&case().inputs(), ps::Bones::Four);
        it.b7[0x34..0x38].copy_from_slice(&word.to_le_bytes());
        let mut rec = vec![0u8; 0x60];
        rec[0x54..0x58].copy_from_slice(&word.to_le_bytes());
        let rec = Arc::new(rec);
        let frame_table = || ORoot::Table(vec![(33, OSlot::Buffer(ps::Kind::Raw, Arc::new(vec![0; 64]))), (84, OSlot::Buffer(ps::Kind::Structured(12), Arc::new(vec![0; 48]))), (90, OSlot::Texture(7))]);
        let reader = |with_rec: bool| Owned { roots: vec![frame_table(), ORoot::Table(if with_rec { vec![(0, OSlot::Buffer(ps::Kind::Cbv, rec.clone()))] } else { Vec::new() })], ident: 3, ..Owned::default() };
        let other = Owned { roots: vec![ORoot::Table(vec![(3, OSlot::Buffer(ps::Kind::Cbv, rec.clone()))])], ident: 4, ..Owned::default() };
        let denied = Owned { key: (0, 0x957f_c3f6_u128 << 96 | 1), ..reader(true) };
        let f = Frame::new(vec![(5, reader(true)), (6, other), (7, reader(true)), (8, reader(true)), (9, reader(false)), (10, denied)]);
        let pre = PreskinFrame { items: vec![it], draws: vec![(5, 0, true), (8, 0, true), (9, 0, true)], ..PreskinFrame::default() };
        let ranges: HashMap<u32, u64> = [(5, 0x1230), (9, 0x40), (10, 0x2000)].into_iter().collect();
        let mut made = Vec::new();
        let (g, b) = bind_rows(&f, &ranges, &pre, 0x1_0000, &bind_deny(""), &mut |len| {
            made.push(len);
            Arc::new(vec![0; len as usize])
        });
        assert_eq!(b, Bind { rows: 5, bound: 1, same_word: 1, no_item: 1, no_job: 1, no_record: 1, cbv: 0, denied: 1 });
        assert_eq!(made, [0x1_0000, 0x1_0000 / 12 * 12]);
        let g = g.expect("a row bound");
        assert_eq!(g.rows.len(), 6);
        let (row, o) = &g.rows[0];
        assert_eq!((*row, o.ident), (5, 0));
        let ORoot::Table(t1) = &o.roots[1] else { panic!() };
        let OSlot::Buffer(_, r) = &t1[0].1 else { panic!() };
        assert_eq!(le32(r, 0x54), 0x4100_0000 | 0x2000_0000 | 0x123);
        let ORoot::Table(t0) = &o.roots[0] else { panic!() };
        assert!(matches!(&t0[0].1, OSlot::Buffer(ps::Kind::Raw, x) if x.len() == 0x1_0000));
        assert!(matches!(&t0[1].1, OSlot::Buffer(ps::Kind::Structured(12), x) if x.len() == 0x1_0000 / 12 * 12));
        assert!(matches!(t0[2].1, OSlot::Texture(7)));
        // The record the source handed over is untouched; the rows not bound are as they were.
        assert_eq!(le32(&rec, 0x54), word);
        assert_eq!(g.rows.iter().map(|r| r.1.ident).collect::<Vec<_>>(), [0, 4, 3, 3, 3, 3]);
        assert!(bind_rows(&f, &HashMap::new(), &pre, 0, &[], &mut |len| Arc::new(vec![0; len as usize])).0.is_none());
        assert_eq!((bind_deny(""), bind_deny("957fc3f6"), bind_deny("x+0x957FC3F6")), (vec![0x957f_c3f6], vec![], vec![]));
        assert!(bind_switch("x,preskinbind=on") && !bind_switch("preskinbind=on,preskinbind=off") && !bind_switch("preskin=on"));
    }

    /// Kept streams: a frame uploads only the streams it has not seen (a
    /// stream two items share once); the next frame with the same items
    /// uploads nothing; a frame that does not fit starts the buffer again
    /// with its own streams only.
    #[test]
    fn kept_streams_upload_once() {
        let c = case();
        let a = Item::from_inputs(&c.inputs(), ps::Bones::Four);
        let b = Item { vb: Arc::new(vec![7u8; 100]), vb_key: 5, ..a.clone() };
        let mut k = Kept::default();
        let p = k.place(&[a.clone(), a.clone(), b.clone()]).unwrap();
        assert_eq!(p.rebuild, Some(KEPT_MIN));
        assert_eq!(p.new, [(0, 0), (0, 1), (2, 0)]);
        let (vb, rm) = (padded(a.vb.len() as u64), padded(4 * a.remap.len() as u64));
        assert_eq!(p.at, [[0, vb as u32], [0, vb as u32], [(vb + rm) as u32, vb as u32]]);
        assert_eq!((p.from, p.bytes, k.used), (0, vb + rm + 112, vb + rm + 112));
        let q = k.place(&[b.clone(), a.clone()]).unwrap();
        assert_eq!((q.new.len(), q.bytes, q.rebuild), (0, 0, None));
        assert_eq!(q.at, [p.at[2], p.at[0]]);
        let big = Item { vb: Arc::new(vec![1u8; KEPT_MIN as usize]), vb_key: 9, ..a.clone() };
        let r = k.place(&[big, b]).unwrap();
        assert_eq!(r.rebuild, Some(4 * KEPT_MIN));
        assert_eq!((r.from, r.new.len()), (0, 3));
        assert!(!k.map.contains_key(&a.vb_key));
    }

    #[test]
    fn switch_default_off() {
        assert!(!screen_switch("screen,skinrules=b"));
        assert!(screen_switch("screen, preskin=on"));
        assert!(!screen_switch("screen,preskin=on,preskin=off"));
        assert_eq!(screen_bits("screen,preskin=on"), 0);
        assert_eq!(screen_bits("screen,preskin=on,preskinbits=3"), BIT_UNMAPPED | BIT_FACE_ANY);
    }

    /// Variants 8 and 9 run under 4's and 5's rule only with bit 0; the
    /// face without an unfilled table only with bit 1; the reasons and the
    /// uncovered variants reach the line.
    #[test]
    fn bits_cover_displacement_and_any_face() {
        let b7 = vec![0u8; 96];
        assert_eq!(rule_for(8, &b7, 0, || false), Err("variant not covered"));
        assert_eq!(rule_for(8, &b7, BIT_UNMAPPED, || false), Ok((Bones::Eight, false, true)));
        assert_eq!(rule_for(9, &b7, BIT_UNMAPPED, || false), Ok((Bones::EightAll, false, true)));
        assert_eq!(rule_for(4, &b7, BIT_UNMAPPED, || false), Ok((Bones::Eight, false, false)));
        assert_eq!(rule_for(11, &b7, 0, || false), Err(WHY_FACE));
        assert_eq!(rule_for(11, &b7, 0, || true), Ok((Bones::Four, true, false)));
        assert_eq!(rule_for(11, &b7, BIT_FACE_ANY, || false), Ok((Bones::Four, true, false)));
        assert!(!covered(8, 0) && covered(8, BIT_UNMAPPED) && covered(9, BIT_UNMAPPED) && !covered(10, 3) && covered(11, 0));
        let mut r = Reads::default();
        r.count(WHY_FACE);
        r.count(WHY_FACE);
        r.count("palette unreadable");
        r.uncovered[10] = 1;
        assert!(r.text().starts_with("; not run: face table or deltas 2, palette unreadable 1; uncovered: v10 1; new streams 0"));
    }

    /// `preskinbits` bit 2: variants 1, 2, 6, 7, 10 by their base variant's
    /// rule (7 and 10 counted unmapped), 2 with t76 handed to the pass (its
    /// place and size in constants 16 and 17); bit 3: ped-flagged rows
    /// without a remap lose the flag, rows with one keep it.
    #[test]
    fn more_variants_and_props() {
        let b7 = vec![0u8; 96];
        assert_eq!(rule_for(1, &b7, 0, || false), Err("variant not covered"));
        assert_eq!(rule_for(1, &b7, BIT_MORE, || false), Ok((Bones::Four, false, false)));
        assert_eq!(rule_for(2, &b7, BIT_MORE, || false), Ok((Bones::Four, false, false)));
        assert_eq!(rule_for(6, &b7, BIT_MORE, || false), Ok((Bones::EightAll, false, false)));
        assert_eq!(rule_for(7, &b7, BIT_MORE, || false), Ok((Bones::Eight, false, true)));
        assert_eq!(rule_for(10, &b7, BIT_MORE, || false), Ok((Bones::EightAll, false, true)));
        assert_eq!(rule_for(12, &b7, 0xf, || false), Err("variant not covered"));
        assert!(covered(10, BIT_MORE) && !covered(10, BIT_UNMAPPED) && !covered(12, 0xf));
        assert!(takes_delta(2, BIT_MORE) && !takes_delta(2, 0) && !takes_delta(1, BIT_MORE));
        let c = case();
        let it = Item { delta: vec![1u8; 40], ..Item::from_inputs(&c.inputs(), Bones::Four) };
        let jobs = [Job { item: 0, geom: None, normals: false, range: false }];
        let (input, placed, _) = layout(std::slice::from_ref(&it), &[[0, 0]], &jobs);
        let k = placed[0].unwrap().0;
        assert_eq!(k[17], 40);
        assert_eq!(&input[k[16] as usize..k[16] as usize + 40], &[1u8; 40][..]);
        let plain = Item::from_inputs(&c.inputs(), Bones::Four);
        let (_, placed, _) = layout(std::slice::from_ref(&plain), &[[0, 0]], &jobs);
        assert_eq!(placed[0].unwrap().0[16..], [0, 0, 0, 0, 0, NONE, 0, 0]);
        use super::super::{Albedo, DrawRecord, FrameList, GeomKey, Material, NO_PALETTE, NO_REMAP};
        let rec = |ped: bool, remap: u32| DrawRecord { geom: GeomKey::default(), world: [0.0; 16], viewport: 1, effect: 0, albedo: Albedo::Flat([1.0; 3]), palette: 0, stage: 67, inst_first: 0, inst_count: 0, bone_first: 4, bone_count: 13, material: Material::default(), state: None, remap, ped };
        let mut list = FrameList { draws: vec![rec(true, NO_REMAP), rec(true, 24), rec(false, NO_REMAP), DrawRecord { bone_count: 496, ..rec(true, NO_REMAP) }], ..Default::default() };
        assert_eq!(props_unflag(&mut list), 1);
        assert_eq!(list.draws.iter().map(|r| r.ped).collect::<Vec<_>>(), [false, true, false, true]);
        assert_eq!(list.draws[0].palette, 0, "{NO_PALETTE}: the bones stay");
    }

    /// The hide-mask record is picked as the reference picks it.
    /// HEAD2: a variant-11 face is taken from b6+0 through its last entry
    /// (tag 0xb6 by dword 1, tag 0x54 by dwords 2 and 3) and normal; a
    /// filled table runs under [`BIT_FACE_TABLE`], an unfilled one neutral.
    #[test]
    fn face_table_from_its_base() {
        let put = |b: &mut Vec<u8>, at: usize, v: u32| b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        let mut b7 = vec![0u8; 96];
        put(&mut b7, 0xc, 16 | 0x40 << 8);
        put(&mut b7, 0x38, 2);
        let mut vb = vec![0u8; 0x60];
        put(&mut vb, 0x44, 3 << 16);
        put(&mut vb, 0x4c, 0xb600_0000);
        put(&mut vb, 0x5c, 0x5400_0000 | 5 << 12);
        let mut b6 = vec![0u8; 16];
        put(&mut b6, 0, 8);
        put(&mut b6, 4, 100);
        put(&mut b6, 0xc, 1);
        let t74: Vec<u8> = (0..300u32).map(|i| i as u8).collect();
        let inp = Inputs { b7: &b7, vb: &vb, b6: &b6, t74: &t74, ..Inputs::default() };
        let (t, n) = face_table(&inp);
        // Entries 0..=5: 28 + 8 * 6 = 76 B of positions, normals to 100 + 4 * 6.
        assert_eq!((t.len(), n, t[0]), (124, 100, 8));
        let mut b6z = b6.clone();
        put(&mut b6z, 0xc, 0);
        assert!(face_table(&Inputs { b6: &b6z, ..inp }).0.is_empty(), "b6+0xc = 0: the delta step, no table");
        assert_eq!(rule_for(11, &b7, BIT_FACE_TABLE, || false), Ok((Bones::Four, false, false)));
        assert_eq!(rule_for(11, &b7, BIT_FACE_TABLE, || true), Ok((Bones::Four, true, false)));
        assert_eq!(rule_for(11, &b7, 0, || false), Err(WHY_FACE));
    }

    #[test]
    fn mask_record_picked() {
        let c = case();
        let mut b7 = c.b7.clone();
        b7[0x8..0xc].copy_from_slice(&(4u32 | 0x90 << 8).to_le_bytes());
        b7[0x20..0x24].copy_from_slice(&0u32.to_le_bytes());
        b7[0x3c..0x40].copy_from_slice(&1u32.to_le_bytes());
        let mut t7 = vec![0u8; 96];
        t7[48 + 28..48 + 32].copy_from_slice(&(2u32 << 16).to_le_bytes());
        let mut t71 = vec![0u8; 3 * 416];
        t71[2 * 416 + 0xa0..][..4].copy_from_slice(&1u32.to_le_bytes());
        let mut t78 = vec![0u8; 2 * MASK_RECORD];
        t78[MASK_RECORD] = 0x5a;
        let inp = Inputs { b7: &b7, t7: &t7, t71: &t71, t78: &t78, ..c.inputs() };
        let it = Item::from_inputs(&inp, ps::Bones::Four);
        assert_eq!(it.mask.as_ref().map(|m| (m.len(), m[0])), Some((MASK_RECORD, 0x5a)));
    }

    /// on a device (`XNW_TEST_GPU=1`): our range against the CPU
    /// reference's (the title's bytes): header and float3 positions equal,
    /// the frame word's fields within one step, the extra block's position
    /// our current one.
    #[test]
    fn device_title_range() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let (_gpu, host) = super::super::tests::warp_host();
        let c = case();
        let mut b7 = c.b7.clone();
        b7[0x34..0x38].copy_from_slice(&0x6000_0010u32.to_le_bytes());
        let it = Item::from_inputs(&Inputs { b7: &b7, ..c.inputs() }, ps::Bones::Four);
        let got = run_ranges(&host, std::slice::from_ref(&it)).expect("dispatch").remove(0);
        let want = ps::cpu_with(&Inputs { b7: &it.b7, ..c.inputs() }, it.rule).bytes;
        assert_eq!(got.len(), 32 + 28 * 2);
        assert_eq!(&got[..32], &want[..32], "header");
        for v in 0..2usize {
            let (gp, wp) = (32 + 12 * v, 32 + 12 * v);
            for k in 0..3 {
                let (a, b) = (f32::from_bits(le32(&got, gp + 4 * k)), f32::from_bits(le32(&want, wp + 4 * k)));
                assert!((a - b).abs() <= 1e-5, "vertex {v} position {k}: {a} vs {b}");
                assert_eq!(le32(&got, 32 + 24 + 16 * v + 4 + 4 * k), le32(&got, gp + 4 * k), "extra block position");
            }
            let (gw, ww) = (le32(&got, 32 + 24 + 16 * v), le32(&want, 32 + 24 + 16 * v));
            for (sh, m) in [(0, 0xff), (8, 0xff), (16, 0x7f), (23, 0x7f), (30, 1)] {
                let (a, b) = ((gw >> sh & m) as i32, (ww >> sh & m) as i32);
                assert!((a - b).abs() <= 1, "vertex {v} frame word {gw:#x} vs {ww:#x} (field at {sh})");
            }
        }
    }

    /// On a device (`XNW_TEST_GPU=1`, the software adapter): the made-up
    /// case through our program gives the reference's positions; every
    /// rule, the 10:10:10 and 24:24:16 formats within one step.
    #[test]
    fn device_made_up_case() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let (_gpu, host) = super::super::tests::warp_host();
        let c = case();
        let mut items = Vec::new();
        for (bits, rule) in [(0x2000_0000u32, ps::Bones::Four), (0x2000_0000, ps::Bones::Eight), (0x2000_0000, ps::Bones::EightAll), (0x1000_0000, ps::Bones::Four), (0, ps::Bones::Four)] {
            let mut b7 = c.b7.clone();
            let word = 0x4000_0010 | bits;
            b7[0x34..0x38].copy_from_slice(&word.to_le_bytes());
            items.push(Item::from_inputs(&Inputs { b7: &b7, ..c.inputs() }, rule));
        }
        let (out, ms) = run(&host, &items).expect("dispatch");
        eprintln!("pre-skin gpu: {} items, {ms:.4} ms", items.len());
        for (it, (pos, nrm)) in items.iter().zip(&out) {
            let w = ps::cpu_with(&Inputs { b7: &it.b7, ..c.inputs() }, it.rule);
            let (format, step) = match ps::range(it.word()).2 {
                ps::PosFormat::Float3 => (0, 1e-5),
                ps::PosFormat::Packed10 => (1, 6.0 / 1023.0),
                ps::PosFormat::Packed24 => (2, 6.0 / 65535.0),
            };
            let (pw, _) = ps::strides(ps::range(it.word()).2, true);
            assert_eq!(pos.len(), 2);
            for v in 0..2 {
                let at = 32 + v * pw as usize;
                let want: [f32; 3] = match format {
                    0 => std::array::from_fn(|k| f32::from_bits(le32(&w.bytes, at + 4 * k))),
                    1 => {
                        let d = le32(&w.bytes, at);
                        [d & 0x3ff, d >> 10 & 0x3ff, d >> 20 & 0x3ff].map(|q| q as f32 * f32::from_bits(0x3bc0_300c) - 3.0)
                    }
                    _ => {
                        let (d0, d1) = (le32(&w.bytes, at), le32(&w.bytes, at + 4));
                        [(d0 >> 8) as f32 * f32::from_bits(0x34c0_0001) - 3.0, (d0 & 0xff | (d1 & 0xffff) << 8) as f32 * f32::from_bits(0x34c0_0001) - 3.0, (d1 >> 16) as f32 * f32::from_bits(0x38c0_00c0) - 3.0]
                    }
                };
                assert!((0..3).all(|k| (pos[v][k] - want[k]).abs() <= step), "{:?} {:?}: {:?} vs {want:?}", it.rule, ps::range(it.word()).2, pos[v]);
                assert!((nrm[v].iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-3, "{:?}", nrm[v]);
            }
        }
    }

    /// `pedhold`: a main-view ped row with no item this frame takes the
    /// item last joined to a row of its geometry nearby, up to N lists old;
    /// a fresh item always wins; other views, far rows and never-joined
    /// rows are left alone.
    #[test]
    fn hold_keeps_a_ped_rows_last_item_for_n_lists() {
        use super::super::{Albedo, Camera, DrawRecord, FrameList, GeomKey, Material, NO_PALETTE, NO_REMAP};
        let rec = |geom: u64, viewport: u64, ped: bool, x: f32| {
            let mut world = [0.0f32; 16];
            (world[0], world[5], world[10], world[15], world[12]) = (1.0, 1.0, 1.0, 1.0, x);
            DrawRecord { geom: GeomKey { mesh_at: geom, ..GeomKey::default() }, world, viewport, effect: 0, albedo: Albedo::Flat([1.0; 3]), palette: NO_PALETTE, stage: 67, inst_first: 0, inst_count: 0, bone_first: 0, bone_count: 389, material: Material::default(), state: None, remap: NO_REMAP, ped }
        };
        let eye = { let mut m = [0.0f32; 16]; (m[0], m[5], m[10], m[15]) = (1.0, 1.0, 1.0, 1.0); m };
        let mut other = eye;
        other[12] = 9.0;
        let list = |frame: u64, draws: Vec<DrawRecord>| FrameList { frame, draws, cameras: vec![Camera { viewport: 1, view: eye, proj: eye }, Camera { viewport: 2, view: other, proj: eye }], main: Some(1), ..Default::default() };
        let c = case();
        let item = |tag: u8| Item { palette: vec![tag; 48], ..Item::from_inputs(&c.inputs(), Bones::Four) };
        let tag = |pre: &PreskinFrame, row: u32| pre.draws.iter().find(|d| d.0 == row).map(|d| (pre.items[d.1 as usize].palette[0], d.2));
        // Rows: ped A in two main-view stages, ped B, ped A's row in another view, a never-joined ped C, an unskinned prop of A's geometry.
        let rows = |ax: f32| vec![rec(1, 1, true, ax), DrawRecord { stage: 74, ..rec(1, 1, true, ax) }, rec(2, 1, true, 3.0), rec(1, 2, true, ax), rec(3, 1, true, 0.0), rec(1, 1, false, ax)];
        let mut hold = Hold::default();
        // f10: both peds joined (A's two rows share one item).
        let mut pre = PreskinFrame { items: vec![item(1), item(2)], draws: vec![(0, 0, true), (1, 0, true), (2, 1, false), (3, 0, true)], ..PreskinFrame::default() };
        assert_eq!(hold.apply(&list(10, rows(0.0)), &mut pre, 3), 0);
        assert_eq!(pre.draws.len(), 4, "nothing added to a joined list");
        // f11: no item (a test run f4020); A moved 0.5 m: both peds held, A's rows on one copied item.
        let mut pre = PreskinFrame::default();
        assert_eq!(hold.apply(&list(11, rows(0.5)), &mut pre, 3), 3);
        assert_eq!((tag(&pre, 0), tag(&pre, 1), tag(&pre, 2)), (Some((1, true)), Some((1, true)), Some((2, false))));
        assert_eq!(pre.items.len(), 2, "one copy per held item");
        assert_eq!((tag(&pre, 3), tag(&pre, 4), tag(&pre, 5)), (None, None, None), "other view, never joined, unskinned prop");
        assert_eq!(pre.held, 3);
        // f12, f13: still held (2 and 3 lists old); f14: gone.
        for (f, want) in [(12, 3), (13, 3), (14, 0)] {
            let mut pre = PreskinFrame::default();
            assert_eq!(hold.apply(&list(f, rows(0.5)), &mut pre, 3), want, "f{f}");
        }
        // f15: A joined afresh (tag 7), B not: next list A holds the new item, B (expired) nothing.
        let mut pre = PreskinFrame { items: vec![item(7)], draws: vec![(0, 0, true)], ..PreskinFrame::default() };
        assert_eq!(hold.apply(&list(15, rows(0.0)), &mut pre, 3), 0);
        let mut pre = PreskinFrame::default();
        assert_eq!(hold.apply(&list(16, rows(0.0)), &mut pre, 3), 2);
        assert_eq!((tag(&pre, 0), tag(&pre, 1), tag(&pre, 2)), (Some((7, true)), Some((7, true)), None));
        // A row too far from the held one (10 m) takes nothing.
        let mut pre = PreskinFrame::default();
        assert_eq!(hold.apply(&list(17, rows(10.0)), &mut pre, 3), 0);
        // Two peds of one geometry: each takes the nearer held item.
        let mut hold = Hold::default();
        let two = |a: f32, b: f32| vec![rec(1, 1, true, a), rec(1, 1, true, b)];
        let mut pre = PreskinFrame { items: vec![item(1), item(2)], draws: vec![(0, 0, true), (1, 1, true)], ..PreskinFrame::default() };
        hold.apply(&list(20, two(0.0, 4.0)), &mut pre, 3);
        let mut pre = PreskinFrame::default();
        assert_eq!(hold.apply(&list(21, two(3.8, 0.2)), &mut pre, 3), 2);
        assert_eq!((tag(&pre, 0), tag(&pre, 1)), (Some((2, true)), Some((1, true))));
        // A list from before the last one (a new sequence) starts over; n = 0 holds nothing.
        let mut pre = PreskinFrame::default();
        assert_eq!(hold.apply(&list(5, two(0.0, 4.0)), &mut pre, 3), 0);
        let mut hold = Hold::default();
        let mut pre = PreskinFrame { items: vec![item(1)], draws: vec![(0, 0, true)], ..PreskinFrame::default() };
        hold.apply(&list(30, two(0.0, 4.0)), &mut pre, 0);
        let mut pre = PreskinFrame::default();
        assert_eq!(hold.apply(&list(31, two(0.0, 4.0)), &mut pre, 0), 0);
        // The switch: default 3, the last one wins, junk keeps the default.
        assert_eq!((hold_switch("screen,preskin=on"), hold_switch("pedhold=0"), hold_switch("pedhold=5, pedhold=2"), hold_switch("pedhold=x")), (PED_HOLD, 0, 2, PED_HOLD));
    }
}
