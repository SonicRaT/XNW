// Stand-in for a console mesh program and the draw of its records
// (crates/xnw-gpu/src/mesh.rs). Compiled by the test
// with DXC: cs_main as cs_6_6, vs_main as vs_6_6, ps_main as ps_6_6;
// INDEXED=1 selects DrawMode::Indexed, 0 DrawMode::VertexId.
//
// Group g draws tile (g % grid_w, g / grid_w), `tile` pixels square, as an
// 8x8 grid of quads over 9x9 shared vertices (81 vertices, 128 triangles).
// Vertices 81..127 are written but never referenced. Primitive t is triangle
// t & 1 of quad t >> 1; the program writes all 128 primitive slots whatever
// the count says, so a draw that ignores the count paints too much.
// EXTRA (0..16) adds that many zero float4 attributes after the two real
// ones, to measure records of the size real programs need; IGNORE_COUNTS=1
// is the negative control (the draw ignores the primitive count).

#ifndef EXTRA
#define EXTRA 0
#endif
#ifndef IGNORE_COUNTS
#define IGNORE_COUNTS 0
#endif

cbuffer C : register(b0) {
    uint rec_bytes, vstride, prim_off, ngroups;
    uint grid_w, tile, count_mode, np_fixed;
    uint nv_fixed, cull_mode, rt_w, rt_h;
};

RWByteAddressBuffer rec : register(u0);
RWByteAddressBuffer ib : register(u1);
ByteAddressBuffer rec_ro : register(t0);

groupshared uint g_alloc;

uint2 counts(uint g) {
    if (count_mode == 0) return uint2(nv_fixed, np_fixed);
    uint np = 2 * (g % 65);
    return uint2(np ? 81 : 0, np);
}

float4 quad_color(uint g, uint q) {
    return float4(q * 4, g & 255, ((g >> 8) & 15) * 16 + 15, 255) / 255.0;
}

[WaveSize(64)]
[numthreads(128, 1, 1)]
void cs_main(uint t : SV_GroupIndex, uint3 gid : SV_GroupID) {
    uint g = gid.x;
    // GS_ALLOC_REQ: thread 0 publishes m0 = prims << 12 | verts.
    if (t == 0) {
        uint2 c = counts(g);
        g_alloc = (c.y << 12) | c.x;
    }
    GroupMemoryBarrierWithGroupSync();
    uint nv = g_alloc & 0xFFF, np = g_alloc >> 12;
    uint base = g * rec_bytes;
    if (t == 0) {
        rec.Store4(base, uint4(nv, np, WaveGetLaneCount(), 0x4D534631u ^ g));
    }

    float2 origin = float2(g % grid_w, g / grid_w) * tile;
    float cell = tile / 8.0;
    // Vertex export: thread t -> vertex t.
    if (t < nv) {
        uint i = t % 9, j = t / 9;
        float2 p = t < 81 ? origin + float2(i, j) * cell : origin + tile * 0.5;
        float4 pos = float4(p.x / rt_w * 2 - 1, 1 - p.y / rt_h * 2, 0.5, 1);
        float4 a0 = quad_color(g, j * 8 + i);
        float4 a1 = float4((p - origin) / tile, 0, 0);
        uint a = base + 16 + t * vstride;
        rec.Store4(a, asuint(pos));
        rec.Store4(a + 16, asuint(a0));
        rec.Store4(a + 32, asuint(a1));
        for (uint k = 0; k < EXTRA; k++) {
            rec.Store4(a + 48 + k * 16, uint4(0, 0, 0, 0));
        }
    }
    // Primitive export: thread t -> primitive t, console packing.
    uint q = t >> 1, v00 = (q / 8) * 9 + q % 8;
    uint3 tri = (t & 1) ? uint3(v00, v00 + 10, v00 + 9) : uint3(v00, v00 + 1, v00 + 10);
    bool cull = cull_mode == 1 || (cull_mode == 2 && g % 7 == 3 && q % 3 == 1);
    uint packed = tri.x | (tri.y << 10) | (tri.z << 20) | (cull ? 0x80000000u : 0);
    rec.Store(base + prim_off + t * 4, packed);
#if INDEXED
    // Epilogue: record primitive t -> global triangle t; dead ones degenerate.
    uint p = rec.Load(base + prim_off + t * 4);
    uint first = g * 128;
    uint3 idx = ((IGNORE_COUNTS || t < np) && (p >> 31) == 0)
        ? first + uint3(p & 0x1FF, (p >> 10) & 0x1FF, (p >> 20) & 0x1FF)
        : uint3(first, first, first);
    ib.Store3((g * 128 + t) * 12, idx);
#endif
}

struct VO {
    float4 pos : SV_Position;
    nointerpolation float4 a0 : ATTR0;
    float4 a1 : ATTR1;
#if EXTRA
    float4 ex[EXTRA] : XATTR;
#endif
};

VO vs_main(uint vid : SV_VertexID) {
    VO o;
#if INDEXED
    uint g = vid >> 7, local = vid & 127;
    bool live = true;
#else
    uint g = vid / 384, r = vid % 384, prim = r / 3, corner = r % 3;
    uint base = g * rec_bytes;
    uint np = IGNORE_COUNTS ? 128 : rec_ro.Load(base + 4);
    uint p = rec_ro.Load(base + prim_off + prim * 4);
    bool live = prim < np && (p >> 31) == 0;
    uint local = (p >> (corner * 10)) & 0x1FF;
#endif
    if (!live) {
        // All three corners equal and outside the depth range: no pixels.
        o.pos = float4(0, 0, 2, 1);
        o.a0 = 0;
        o.a1 = 0;
#if EXTRA
        for (uint k = 0; k < EXTRA; k++) {
            o.ex[k] = 0;
        }
#endif
        return o;
    }
    uint a = g * rec_bytes + 16 + local * vstride;
    o.pos = asfloat(rec_ro.Load4(a));
    o.a0 = asfloat(rec_ro.Load4(a + 16));
    o.a1 = asfloat(rec_ro.Load4(a + 32));
#if EXTRA
    for (uint k = 0; k < EXTRA; k++) {
        o.ex[k] = asfloat(rec_ro.Load4(a + 48 + k * 16));
    }
#endif
    return o;
}

float4 ps_main(VO i) : SV_Target {
    float3 s = 0;
#if EXTRA
    for (uint k = 0; k < EXTRA; k++) {
        s += i.ex[k].xyz + i.ex[k].w;
    }
#endif
    return float4(i.a0.rgb + s, 0.5 * (i.a1.x + i.a1.y));
}
