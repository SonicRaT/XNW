//! `XNW_WORLD_RENDER=...,ourlight=on` (LITE1, default off; turns `gbuf=on`
//! on): our own lighting over our G-buffer, one full-screen pass after the
//! opaque rows (before the water and the transparent rows). Our HLSL only:
//! - the rows before the pass (flags bit 3) write their albedo, pow 1/2.2,
//!   to the colour target (decals blend into the albedo there), and their
//!   normal to the G-buffer's RT1 with .r = (1 + 254 x fog visibility) / 255:
//!   0 (the clear) marks a pixel no row of ours wrote (sky, terrain, cards);
//! - the pass copies the colour target and reads the copy and RT1 at its
//!   pixel (clamped to the targets: every read inside them, none in a
//!   conditional arm, one exit), and writes tonemap(albedo x (sun x N.L / pi +
//!   hemisphere ambient)) towards the fog colour; other pixels keep their colour;
//! - the light: the frame's Light buffer, [`crate::lighting::Lighting::ours`]
//!   (the rows after the pass light themselves from the same buffer).
//!
//! `ourlight=metal` (LITE2): as `on`, and the title's bridged rows
//! (`bridgeps`) draw before the pass, their target 2 (F0) into a metal
//! target of ours ([`METAL`], cleared to alpha 1; the rows leave alpha 0).
//! The title's `PS_DeferredTextured` (3ed68103) writes a metal workflow:
//! target 0 = base x (1 - metal) (0x108c..0x109c, `exp mrt0` 0x10d0),
//! target 2 = sqrt(1 - sqrt(1 - F0)) per channel with F0 = lerp(dielectric,
//! base, metal) (0x10a0..0x10b8, 0x10f0..0x1124, `exp mrt2` 0x11ac) [V
//! disasm]. At a pixel a bridged row wrote, the pass takes target 0 as the
//! diffuse colour (linear, as the row wrote it), F0 decoded, and adds a
//! specular sun lobe (GGX, one roughness) and an environment term (the
//! hemisphere ambient along the reflection, Schlick with roughness): we
//! have no reflection probes. Other pixels are lit as with `on`.
//!
//! `ournormals=title` (NORM1, with `ourlight=metal`): the bridged rows'
//! target 1 also lands in a target of ours ([`TITLE_NORMAL`], cleared to
//! 0), and at their pixels the pass takes the normal and the roughness
//! from it instead of our G-buffer's normal and [`ROUGH`]. The title's
//! RT1 (`PS_DeferredTextured` 3ed68103 0xf0c..0xfb0, 07b19da9
//! 0x2988..0x2a2c, `exp mrt1` [V disasm]): R = 1 - roughness (perceptual,
//! widened by the normal's variance over the quad, 0xe88..0xf04), G B A =
//! bytes 0 1 2 of the octahedral word y' << 12 | x' (the same packing as
//! ours, [`crate::gbuf::pack_normal`]), world space, Z up [I: every covered
//! pixel of the reference f1633 / f2002 decodes to z > 0.9 on ground rows].

use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// Process-wide: set by `Pass::new` before it compiles the raw shader
/// ([`OFF`], [`ON`] or [`METAL_ON`]).
static MODE: AtomicU8 = AtomicU8::new(OFF);

pub const OFF: u8 = 0;
pub const ON: u8 = 1;
pub const METAL_ON: u8 = 2;

/// `ourlight=on` or `metal`.
pub fn on() -> bool {
    MODE.load(Ordering::Relaxed) != OFF
}

/// `ourlight=metal`.
pub fn metal() -> bool {
    MODE.load(Ordering::Relaxed) == METAL_ON
}

pub(crate) fn set(v: u8) {
    MODE.store(v, Ordering::Relaxed);
}

/// Process-wide, as [`MODE`]: `ournormals=title`.
static TITLE_NORMALS: AtomicBool = AtomicBool::new(false);

/// `ourlight=metal` with `ournormals=title`: the pass lights the bridged
/// rows with the title's normals and roughness (their target 1).
pub fn title_normals() -> bool {
    metal() && TITLE_NORMALS.load(Ordering::Relaxed)
}

pub(crate) fn set_normals(v: bool) {
    TITLE_NORMALS.store(v, Ordering::Relaxed);
}

/// Process-wide, as [`MODE`]: `emissive=on`.
static EMISSIVE_ON: AtomicBool = AtomicBool::new(false);

/// `ourlight=metal` with `emissive=on` (LGT1): the bridged rows' target 5
/// (the title's G-buffer RT5, LIGHT1 section 3 (c)) lands in a target of
/// ours ([`EMISSIVE`], cleared to 0) and the pass adds it, decoded, to
/// their lit colour. Our own rows have no emissive source.
pub fn emissive() -> bool {
    metal() && EMISSIVE_ON.load(Ordering::Relaxed)
}

pub(crate) fn set_emissive(v: bool) {
    EMISSIVE_ON.store(v, Ordering::Relaxed);
}

/// `emissive=on|off` in `XNW_WORLD_RENDER` (the last one wins; default off).
pub fn parse_emissive(v: &str) -> bool {
    v.split(',').filter_map(|p| p.trim().strip_prefix("emissive=")).next_back() == Some("on")
}

/// `ournormals=title|ours` in `XNW_WORLD_RENDER` (the last one wins; default ours).
pub fn parse_normals(v: &str) -> bool {
    v.split(',').filter_map(|p| p.trim().strip_prefix("ournormals=")).next_back() == Some("title")
}

/// `ourlight=on|metal|off` in `XNW_WORLD_RENDER` (the last one wins; default off).
pub fn parse_switch(v: &str) -> u8 {
    match v.split(',').filter_map(|p| p.trim().strip_prefix("ourlight=")).next_back() {
        Some("on") => ON,
        Some("metal") => METAL_ON,
        _ => OFF,
    }
}

/// The switch as the environment asks it now (the frame loaders pick the
/// light with it: [`crate::lighting::Lighting::ours`]).
pub fn asked() -> bool {
    parse_switch(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default()) != OFF
}

/// SW1 live `ourlight` (capped; start: asked): 0 draws the frame without
/// the pass (taken once a frame when the walk starts).
pub fn live_on() -> bool {
    xnw_core::live::on(xnw_core::live::OURLIGHT, || u32::from(asked()))
}

/// SW1 live `ournormals` (capped; start: `ournormals=title`): 0 clears the
/// title normal target before the pass reads it (taken once a frame).
pub fn live_normals() -> bool {
    xnw_core::live::on(xnw_core::live::OURNORMALS, || u32::from(parse_normals(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default())))
}

/// The metal target's format (the title's target 2 as the bridged rows write it).
pub const METAL: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;

/// `ournormals=title`: the title normal target's format (the title's RT1, f28).
pub const TITLE_NORMAL: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;

/// `emissive=on`: the emissive target's format (the title's RT5, f28: the
/// packed word's four bytes).
pub const EMISSIVE: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;

/// The roughness of every bridged pixel without the title's normals.
pub const ROUGH: f32 = 0.45;

/// The raw shader's flags bit: this row comes before the lighting pass
/// (its colour output is its albedo).
pub const DEFERRED: u32 = 1 << 3;

/// The pass. Root: the Light buffer (b0), a table of two views (t0 the
/// colour copy, t1 the G-buffer's RT1); with `#define METAL 1` three (t2
/// the metal target) and the camera's clip-to-world (b1, camera-relative);
/// with `#define TITLE_NORMALS 1` too, four (t3 the title normal target).
pub const SHADER: &str = r#"
#ifdef TITLE_NORMALS
#define RS "CBV(b0), DescriptorTable(SRV(t0, numDescriptors=4), visibility=SHADER_VISIBILITY_PIXEL), RootConstants(num32BitConstants=16, b1)"
#elif defined(METAL)
#define RS "CBV(b0), DescriptorTable(SRV(t0, numDescriptors=3), visibility=SHADER_VISIBILITY_PIXEL), RootConstants(num32BitConstants=16, b1)"
#else
#define RS "CBV(b0), DescriptorTable(SRV(t0, numDescriptors=2), visibility=SHADER_VISIBILITY_PIXEL)"
#endif
// The frame's Light buffer (lighting.rs `constants`): light dir (towards it), radiance (w: exposure),
// ambient sky, ambient ground, Hable A B C D, E F W 1/Hable(W), sky radiance scale (x), fog colour (yzw).
cbuffer Light : register(b0) { float4 lDir; float4 lRad; float4 lSky; float4 lGround; float4 tone0; float4 tone1; float4 lPad; };
Texture2D<float4> colourIn : register(t0);
Texture2D<float4> normalIn : register(t1);
float3 hableC(float3 x) { return (x * (tone0.x * x + tone0.z * tone0.y) + tone0.w * tone1.x) / (x * (tone0.x * x + tone0.y) + tone0.w * tone1.y) - tone1.x / tone1.y; }
float3 tonemap(float3 hdr) { return pow(saturate(hableC(max(hdr * lRad.w, 0)) * tone1.w), 1.0 / 2.2); }
float3 skyRadiance(float3 c) { return pow(max(c, 0), 2.2) * lPad.x; }
// `post=on` (post.rs): the pass writes scene-linear colour; the composite exposes and tone-maps it.
#ifdef POST
#define TONE(x) (x)
#else
#define TONE(x) tonemap(x)
#endif
// RT1 (0, b0, b1, b2) / 255: the title's octahedral normal, word = y' << 12 | x' (gbuf.rs `unpack_normal`).
float3 unpackNormal(float3 b) {
    uint3 q = uint3(saturate(b) * 255 + 0.5);
    uint w = q.x | (q.y << 8) | (q.z << 16);
    float2 e = (float2(w & 4095, (w >> 12) & 4095) + 0.5 - 2047.5) / 2047.5;
    float z = 1 - abs(e.x) - abs(e.y);
    float2 f = (1 - abs(e.yx)) * float2(e.x >= 0 ? 1 : -1, e.y >= 0 ? 1 : -1);
    return normalize(float3(z < 0 ? f : e, z) + 1e-6);
}
#ifdef METAL
Texture2D<float4> metalIn : register(t2);
cbuffer Eye : register(b1) { row_major float4x4 clipToWorld; };
// The title's target 2: e = sqrt(1 - sqrt(1 - F0)) per channel (3ed68103 0x10f0..0x1124).
float3 decodeF0(float3 e) { float3 t = 1 - saturate(e) * saturate(e); return 1 - t * t; }
// One roughness for every bridged pixel without the title's normals (ourlight.rs `ROUGH`).
#define ROUGH 0.45
// Sun specular: GGX distribution, height-correlated Smith visibility (approximate), Schlick Fresnel.
float3 sunSpecular(float3 f0, float nl, float nv, float nh, float vh, float rough) {
    float a = rough * rough, a2 = a * a;
    float d = nh * nh * (a2 - 1) + 1;
    float D = a2 / (3.14159265 * d * d);
    float vis = 0.5 / (nl * (nv * (1 - a) + a) + nv * (nl * (1 - a) + a) + 1e-5);
    float3 F = f0 + (1 - f0) * pow(1 - vh, 5);
    return D * vis * F * nl;
}
// Environment: Schlick's Fresnel towards max(1 - roughness, F0) at grazing angles, times F0's share.
float3 envSpecular(float3 f0, float nv, float rough) { return f0 + (max(1 - rough, f0) - f0) * pow(1 - nv, 5); }
#endif
#ifdef TITLE_NORMALS
// The title's RT1 as the bridged rows wrote it: R = 1 - roughness, G B A the octahedral word (as ours).
Texture2D<float4> titleNormalIn : register(t3);
#endif
#ifdef SUNSHADOW
// `sunshadow=on` (sunshadow.rs): the screen term, .r the sun's lit fraction at the pixel.
Texture2D<float4> shadowIn : register(SUNSHADOW);
#endif
#ifdef LAMPS
// `lights=on` (lamps.rs): our depth's copy (reverse Z) and the frame's kept lights, lampHead.x of them (at most 64),
// four float4 each: position - camera, 1 / r^2; colour x intensity x scale, R^2; axis, cone s; cone o, ramp k, ramp o, type.
Texture2D<float> lampDepth : register(LAMPS);
cbuffer Lamps : register(b2) { uint4 lampHead; float4 lampData[256]; };
#endif
#ifdef EMISSIVE
// `emissive=on` (LGT1): the bridged rows' target 5, the title's RT5 word u = bytes (r, g, b, a): 9-bit mantissas
// R = u & 0x1FE, G = (u >> 9) & 0x1FE, B = (u >> 18) & 0x1FE (bit 0 dropped), exponent e = (u >> 27) - 24
// (TL_PS_PBR_MATERIAL_EMISSIVE 81470aa6 @0x184..0x614, LIGHT1 3 (c); the writer's encode: aa741d98 0x1268..0x1490).
// Title units; EMISSIVE_SCALE (the lights' scale, lamps.rs) brings them to ours. The clear (0) decodes to 0.
Texture2D<float4> emissiveIn : register(EMISSIVE);
float3 decodeEmissive(float4 b) {
    uint4 q = uint4(saturate(b) * 255 + 0.5);
    uint u = q.x | (q.y << 8) | (q.z << 16) | (q.w << 24);
    return float3(u & 0x1FE, (u >> 9) & 0x1FE, (u >> 18) & 0x1FE) * exp2(float(int(u >> 27) - 24));
}
#endif
struct VsOut { float4 pos : SV_Position; };
[RootSignature(RS)]
VsOut vs(uint id : SV_VertexID) {
    VsOut o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}
// One return; both loads first, at a pixel clamped inside the targets.
[RootSignature(RS)]
float4 ps(VsOut i) : SV_Target {
    uint w, h;
    colourIn.GetDimensions(w, h);
    int3 p = int3(min(uint2(i.pos.xy), uint2(max(w, 1), max(h, 1)) - 1), 0);
    float4 c = colourIn.Load(p);
    float4 g = normalIn.Load(p);
    float3 n = unpackNormal(g.gba);
    float3 base = pow(saturate(c.rgb), 2.2);
    float nl = saturate(dot(n, normalize(lDir.xyz + 1e-9)));
    float3 amb = lerp(lGround.rgb, lSky.rgb, saturate(n.z * 0.5 + 0.5));
#ifdef SUNSHADOW
    float sh = shadowIn.Load(p).r;
#else
    float sh = 1;
#endif
    float3 hdr = base * (lRad.rgb * (nl * sh * 0.31830989) + amb);
#ifdef CARSHADE
    hdr = carShade(hdr, p, w, h, n, base, sh);
#endif
    float3 lit = TONE(hdr);
#ifdef METAL
    // A bridged row wrote here (alpha 0): its target 0 is the diffuse colour, linear; F0 from the metal target.
    float4 m = metalIn.Load(p);
#ifdef TITLE_NORMALS
    // Where a bridged row wrote (alpha 0 on the metal target) a normal word (not the clear): its normal and roughness.
    float4 tn = titleNormalIn.Load(p);
    bool own = m.a < 0.5 && any(tn.gba > 0.5 / 255);
    float3 nb = own ? unpackNormal(tn.gba) : n;
    float rough = own ? max(1 - saturate(tn.r), 0.045) : ROUGH;
    float nlb = saturate(dot(nb, normalize(lDir.xyz + 1e-9)));
    float3 ambb = lerp(lGround.rgb, lSky.rgb, saturate(nb.z * 0.5 + 0.5));
#else
    float3 nb = n;
    float rough = ROUGH;
    float nlb = nl;
    float3 ambb = amb;
#endif
    float2 ndc = (float2(p.xy) + 0.5) / float2(max(w, 1), max(h, 1)) * float2(2, -2) + float2(-1, 1);
    float4 q = mul(float4(ndc, 0.5, 1), clipToWorld);
    float3 v = -normalize(q.xyz / q.w + 1e-9);
    float3 nf = dot(nb, v) < 0 ? -nb : nb;
    float3 ld = normalize(lDir.xyz + 1e-9);
    float3 hv = normalize(ld + v);
    float nlf = saturate(dot(nf, ld)), nv = saturate(dot(nf, v)), nh = saturate(dot(nf, hv)), vh = saturate(dot(v, hv));
    float3 f0 = decodeF0(m.rgb);
    float3 r = reflect(-v, nf);
    float3 env = lerp(lGround.rgb, lSky.rgb, saturate(r.z * 0.5 + 0.5)) * envSpecular(f0, nv, rough);
    float3 hdrMetal = saturate(c.rgb) * (lRad.rgb * (nlb * sh * 0.31830989) + ambb) + lRad.rgb * sunSpecular(f0, nlf, nv, nh, vh, rough) * sh + env;
#ifdef LAMPS
    // The title's local lights (TL_PS's model, lamps.rs) at the pixel's position from our depth, camera-relative as
    // the records. One load, before the loop; the loop reads constants only, its bound the count clamped to 64.
    float zl = lampDepth.Load(p);
    float4 ql = mul(float4(ndc, zl, 1), clipToWorld);
    float3 wp = ql.xyz / (abs(ql.w) > 1e-12 ? ql.w : 1e-12);
    // A row of ours: F0 0.04, roughness 0.6.
    float3 f0l = m.a < 0.5 ? f0 : 0.04;
    float roughl = m.a < 0.5 ? rough : 0.6;
    float3 lampD = 0, lampS = 0;
    uint lampN = min(lampHead.x, 64);
    [loop] for (uint k = 0; k < lampN; k++) {
        float4 a = lampData[4 * k], b = lampData[4 * k + 1], ax = lampData[4 * k + 2], e = lampData[4 * k + 3];
        float3 dl = a.xyz - wp;
        float d2 = dot(dl, dl);
        float3 ll = dl * rsqrt(max(d2, 1e-8));
        float x = d2 * a.w;
        float win = saturate(1 - x * x);
        float t = saturate(dot(-ll, ax.xyz) * ax.w + e.x);
        float t2 = t * t;
        float wgt = win * win / (d2 + b.w) * saturate(d2 * e.y + e.z) * t * (t - t2 + t2 * t2 * (3 - 2 * t2));
        float3 hl = normalize(ll + v);
        float nll = saturate(dot(nf, ll));
        lampD += b.rgb * (wgt * saturate(dot(nb, ll)));
        lampS += b.rgb * wgt * sunSpecular(f0l, nll, nv, saturate(dot(nf, hl)), saturate(dot(v, hl)), roughl);
    }
    hdrMetal += saturate(c.rgb) * lampD * 0.31830989 + lampS;
    lit = TONE(hdr + base * lampD * 0.31830989 + lampS);
#endif
#ifdef EMISSIVE
    // Only bridged rows write it, and only their pixels take hdrMetal.
    hdrMetal += decodeEmissive(emissiveIn.Load(p)) * EMISSIVE_SCALE;
#endif
    float3 metalLit = TONE(hdrMetal);
    lit = m.a < 0.5 ? metalLit : lit;
#endif
    float vis = saturate((g.r * 255 - 1) / 254);
#ifdef POST
    // Scene-linear; .a the fog visibility of a lit pixel (the composite fogs after its curve), below 0: not lit.
    float4 o = float4(lit, g.r > 0.5 / 255 ? vis : -1.0);
#else
    lit = lerp(tonemap(skyRadiance(lPad.yzw)), lit, vis);
    float4 o = float4(g.r > 0.5 / 255 ? lit : c.rgb, c.a);
#endif
    return o;
}
"#;

/// The pass's source: `metal` and `titled` as the defines above; `shadow`
/// (`sunshadow=on`): one more view after the others (`SUNSHADOW` is its
/// register), the root's table one longer; `lamps` (`lights=on`, with
/// `metal`): our depth's view after those (`LAMPS` its register) and the
/// lights' constant buffer (b2, root parameter 3).
pub fn source(metal: bool, titled: bool, shadow: bool, lamps: bool) -> String {
    source_post(metal, titled, shadow, lamps, false)
}

/// [`source`]; `post` (`post=on`): the pass's scene-linear variant (`#define POST`).
pub fn source_post(metal: bool, titled: bool, shadow: bool, lamps: bool, post: bool) -> String {
    source_full(metal, titled, shadow, lamps, None, post)
}

/// [`source_post`]; `emissive` (`emissive=on`, with `metal`; the scale):
/// the emissive target's view after all the others (`EMISSIVE` its register).
pub fn source_full(metal: bool, titled: bool, shadow: bool, lamps: bool, emissive: Option<f32>, post: bool) -> String {
    let titled = metal && titled;
    let lamps = metal && lamps;
    let mut src = match (metal, titled) {
        (true, true) => format!("#define METAL 1\n#define TITLE_NORMALS 1\n{SHADER}"),
        (true, false) => format!("#define METAL 1\n{SHADER}"),
        _ => SHADER.to_string(),
    };
    if shadow {
        // Each root variant's table grows by one (largest first: no count is replaced twice).
        for n in [4, 3, 2] {
            src = src.replace(&format!("SRV(t0, numDescriptors={n})"), &format!("SRV(t0, numDescriptors={})", n + 1));
        }
        src = format!("#define SUNSHADOW t{}\n{src}", 2 + u32::from(metal) + u32::from(titled));
    }
    if lamps {
        for n in [5, 4, 3] {
            src = src.replace(&format!("SRV(t0, numDescriptors={n})"), &format!("SRV(t0, numDescriptors={})", n + 1));
        }
        src = src.replace("RootConstants(num32BitConstants=16, b1)\"", "RootConstants(num32BitConstants=16, b1), CBV(b2, visibility=SHADER_VISIBILITY_PIXEL)\"");
        src = format!("#define LAMPS t{}\n{src}", 3 + u32::from(titled) + u32::from(shadow));
    }
    if let Some(scale) = emissive.filter(|_| metal) {
        for n in [6, 5, 4, 3] {
            src = src.replace(&format!("SRV(t0, numDescriptors={n})"), &format!("SRV(t0, numDescriptors={})", n + 1));
        }
        src = format!("#define EMISSIVE t{}\n#define EMISSIVE_SCALE {scale:.6}\n{src}", 3 + u32::from(titled) + u32::from(shadow) + u32::from(lamps));
    }
    if post {
        src = format!("#define POST 1\n{src}");
    }
    src
}

/// The metal target's clear: alpha 1 (the bridged rows write alpha 0).
const CLEAR_METAL: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

/// The pass's objects: root, pipeline, the colour copy (its view at
/// `slot`, RT1's at `slot + 1`).
pub struct Lit {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    copy: ID3D12Resource,
    pub slot: u32,
    /// `ourlight=metal`: the metal target and the RTV heap (its view at `slot + 2`).
    metal: Option<(ID3D12Resource, ID3D12DescriptorHeap)>,
    /// `ournormals=title`: the title normal target (RTV 1 of the metal
    /// target's heap, its view at `slot + 3`) and the RTV step.
    normal: Option<(ID3D12Resource, usize)>,
    /// `lights=on`: our depth's copy (its view after the pass's others) and
    /// the lights' constants, four [`LAMP_SLICE`] slices of one mapped
    /// upload buffer (a frame takes the next one), the slice in use.
    lamps: Option<(ID3D12Resource, ID3D12Resource, *mut u8, u64)>,
    /// `post=on`: the pass's scene-linear pipeline (same root) and the composite with its target.
    post: Option<(ID3D12PipelineState, crate::post::Composite)>,
    /// `emissive=on`: the emissive target (RTV 2 of the metal target's heap,
    /// its view after the pass's others) and the RTV step.
    emissive: Option<(ID3D12Resource, usize)>,
    /// CAR1 `carshade=on`: the root parameter of the paint table, and its view ([`Lit::set_car`]).
    car_param: Option<u32>,
    car: Option<crate::carshade::View>,
}

/// A frame's slice of the lights' constant buffer (4112 bytes used).
const LAMP_SLICE: usize = (crate::lamps::CB_BYTES + 255) & !255;

impl Lit {
    /// Made once by `Pass::new`: `colour_format` x `width` x `height` copy, the
    /// two views written at `cpu(slot)` and `cpu(slot + 1)`; with
    /// `metal` the metal target too, its view at `cpu(slot + 2)`; with
    /// `titled` (and `metal`) the title normal target, its view at `cpu(slot + 3)`;
    /// `emissive` (and `metal`; the scale) the emissive target, its view after the others.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(device: &ID3D12Device, normal: &ID3D12Resource, colour_format: DXGI_FORMAT, width: u32, height: u32, slot: u32, metal: bool, titled: bool, shadow: Option<&ID3D12Resource>, lamps: bool, emissive: Option<f32>, post: bool, cpu: &dyn Fn(u32) -> D3D12_CPU_DESCRIPTOR_HANDLE) -> Result<Lit, String> {
        let e = |e: windows::core::Error| format!("ourlight: {e}");
        let titled = metal && titled;
        let lamps = metal && lamps;
        let emissive = emissive.filter(|_| metal);
        // CAR1 (`carshade=on`, with metal): the paint term and its table (carshade.rs).
        let car = crate::carshade::on() && metal;
        let src = crate::carshade::pass_source(source_full(metal, titled, shadow.is_some(), lamps, emissive, false), car);
        let (vs, ps) = (crate::compile_source(&src, c"vs", c"vs_5_1")?, crate::compile_source(&src, c"ps", c"ps_5_1")?);
        // SAFETY: object creation with descriptions and blobs that outlive each call.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, crate::blob_bytes(&vs)).map_err(e)?;
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(root.clone())),
                VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.GetBufferPointer(), BytecodeLength: vs.GetBufferSize() },
                PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.GetBufferPointer(), BytecodeLength: ps.GetBufferSize() },
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            desc.RTVFormats[0] = colour_format;
            desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            let pso = device.CreateGraphicsPipelineState::<ID3D12PipelineState>(&desc);
            drop(ManuallyDrop::into_inner(desc.pRootSignature));
            let pso = pso.map_err(e)?;
            // `post=on`: the same pass with `#define POST` into the scene-linear format.
            let pso_post = if post {
                let ps = crate::compile_source(&crate::carshade::pass_source(source_full(metal, titled, shadow.is_some(), lamps, emissive, true), car), c"ps", c"ps_5_1")?;
                desc.pRootSignature = ManuallyDrop::new(Some(root.clone()));
                desc.PS = D3D12_SHADER_BYTECODE { pShaderBytecode: ps.GetBufferPointer(), BytecodeLength: ps.GetBufferSize() };
                desc.RTVFormats[0] = crate::post::HDR;
                let p = device.CreateGraphicsPipelineState::<ID3D12PipelineState>(&desc);
                drop(ManuallyDrop::into_inner(desc.pRootSignature));
                Some(p.map_err(e)?)
            } else {
                None
            };
            let d = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: u64::from(width), Height: height, DepthOrArraySize: 1, MipLevels: 1, Format: colour_format, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() };
            let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
            let mut copy: Option<ID3D12Resource> = None;
            device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, None, &mut copy).map_err(e)?;
            let copy = copy.ok_or("ourlight: no colour copy")?;
            let view = |format| D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } },
            };
            device.CreateShaderResourceView(&copy, Some(&view(colour_format)), cpu(slot));
            device.CreateShaderResourceView(normal, Some(&view(crate::gbuf::NORMAL)), cpu(slot + 1));
            let metal = if metal {
                let d = D3D12_RESOURCE_DESC { Format: METAL, Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, ..d };
                let clear = D3D12_CLEAR_VALUE { Format: METAL, Anonymous: D3D12_CLEAR_VALUE_0 { Color: CLEAR_METAL } };
                let mut t: Option<ID3D12Resource> = None;
                device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_RENDER_TARGET, Some(&clear), &mut t).map_err(e)?;
                let t = t.ok_or("ourlight: no metal target")?;
                let h: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 3, ..Default::default() }).map_err(e)?;
                device.CreateRenderTargetView(&t, None, h.GetCPUDescriptorHandleForHeapStart());
                device.CreateShaderResourceView(&t, Some(&view(METAL)), cpu(slot + 2));
                Some((t, h))
            } else {
                None
            };
            // `ournormals=title`: the title normal target, cleared to 0 (no bridged row's word here).
            let normal = match &metal {
                Some((_, h)) if titled => {
                    let d = D3D12_RESOURCE_DESC { Format: TITLE_NORMAL, Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, ..d };
                    let clear = D3D12_CLEAR_VALUE { Format: TITLE_NORMAL, Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0; 4] } };
                    let mut t: Option<ID3D12Resource> = None;
                    device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_RENDER_TARGET, Some(&clear), &mut t).map_err(e)?;
                    let t = t.ok_or("ourlight: no title normal target")?;
                    let step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
                    device.CreateRenderTargetView(&t, None, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: h.GetCPUDescriptorHandleForHeapStart().ptr + step });
                    device.CreateShaderResourceView(&t, Some(&view(TITLE_NORMAL)), cpu(slot + 3));
                    Some((t, step))
                }
                _ => None,
            };
            // `sunshadow=on`: the screen term after the pass's other views.
            if let Some(t) = shadow {
                device.CreateShaderResourceView(t, Some(&view(crate::sunshadow::TERM)), cpu(slot + 2 + u32::from(metal.is_some()) + u32::from(normal.is_some())));
            }
            // `lights=on`: our depth's copy after those, and the lights' constants (no light until a frame sets them).
            let lamps = if lamps {
                let dd = D3D12_RESOURCE_DESC { Format: DXGI_FORMAT_R32_FLOAT, ..d };
                let mut t: Option<ID3D12Resource> = None;
                device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &dd, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, None, &mut t).map_err(e)?;
                let t = t.ok_or("ourlight: no depth copy")?;
                device.CreateShaderResourceView(&t, Some(&view(DXGI_FORMAT_R32_FLOAT)), cpu(slot + 2 + u32::from(metal.is_some()) + u32::from(normal.is_some()) + u32::from(shadow.is_some())));
                let cb = crate::upload_buffer(device, &[0u8; 4 * LAMP_SLICE]).map_err(e)?;
                let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
                cb.Map(0, None, Some(&mut p)).map_err(e)?;
                Some((t, cb, p.cast::<u8>(), 0))
            } else {
                None
            };
            // `emissive=on`: the emissive target, cleared to 0 (no emissive), its view after the depth copy's.
            let after = slot + 2 + u32::from(metal.is_some()) + u32::from(normal.is_some()) + u32::from(shadow.is_some()) + u32::from(lamps.is_some());
            let emissive = match (&metal, emissive) {
                (Some((_, h)), Some(_)) => {
                    let d = D3D12_RESOURCE_DESC { Format: EMISSIVE, Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, ..d };
                    let clear = D3D12_CLEAR_VALUE { Format: EMISSIVE, Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0; 4] } };
                    let mut t: Option<ID3D12Resource> = None;
                    device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_RENDER_TARGET, Some(&clear), &mut t).map_err(e)?;
                    let t = t.ok_or("ourlight: no emissive target")?;
                    let step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
                    device.CreateRenderTargetView(&t, None, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: h.GetCPUDescriptorHandleForHeapStart().ptr + 2 * step });
                    device.CreateShaderResourceView(&t, Some(&view(EMISSIVE)), cpu(after));
                    Some((t, step))
                }
                _ => None,
            };
            let post = match pso_post {
                Some(p) => Some((p, crate::post::Composite::new(device, colour_format, width, height, after + u32::from(emissive.is_some()), cpu)?)),
                None => None,
            };
            let car_param = car.then(|| crate::carshade::root_param(lamps.is_some()));
            Ok(Lit { root, pso, copy, slot, metal, normal, lamps, post, emissive, car_param, car: None })
        }
    }

    /// `ourlight=metal`: the metal target's RTV (the bridged rows' target 2).
    pub(crate) fn metal_rtv(&self) -> Option<D3D12_CPU_DESCRIPTOR_HANDLE> {
        // SAFETY: a live heap.
        self.metal.as_ref().map(|(_, h)| unsafe { h.GetCPUDescriptorHandleForHeapStart() })
    }

    /// `ournormals=title`: the title normal target's RTV (the bridged rows' target 1).
    pub(crate) fn normal_rtv(&self) -> Option<D3D12_CPU_DESCRIPTOR_HANDLE> {
        let step = self.normal.as_ref()?.1;
        self.metal_rtv().map(|h| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: h.ptr + step })
    }

    /// `emissive=on`: the emissive target's RTV (the bridged rows' target 5).
    pub(crate) fn emissive_rtv(&self) -> Option<D3D12_CPU_DESCRIPTOR_HANDLE> {
        let step = self.emissive.as_ref()?.1;
        self.metal_rtv().map(|h| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: h.ptr + 2 * step })
    }

    /// `emissive=on`: the pass has the emissive target (one heap view more).
    pub(crate) fn has_emissive(&self) -> bool {
        self.emissive.is_some()
    }

    /// CAR1: with `carshade=on` the albedo target's view at `cpu(slot)` (a heap slot of its own).
    pub(crate) fn set_car(&mut self, device: &ID3D12Device, albedo: &ID3D12Resource, slot: u32, cpu: &dyn Fn(u32) -> D3D12_CPU_DESCRIPTOR_HANDLE) {
        self.car = self.car_param.map(|param| crate::carshade::View::new(device, albedo, slot, param, cpu));
    }

    /// CAR1: the pass reads the paint table (one heap slot more).
    pub(crate) fn has_car(&self) -> bool {
        self.car_param.is_some()
    }

    /// `post=on`: the pass has the scene-linear pipeline and the composite.
    pub(crate) fn has_post(&self) -> bool {
        self.post.is_some()
    }

    /// `post=on`: the heap slot of the scene-linear target's view (the composite's table).
    pub(crate) fn post_slot(&self) -> u32 {
        self.post.as_ref().map_or(0, |p| p.1.slot)
    }

    /// `lights=on`: the pass has the lights' objects.
    pub(crate) fn has_lamps(&self) -> bool {
        self.lamps.is_some()
    }

    /// `lights=on`: this frame's lights (`bytes`: `lamps::constants`), into the next slice.
    pub(crate) fn set_lamps(&mut self, bytes: &[u8]) {
        if let Some((_, _, map, turn)) = &mut self.lamps {
            *turn = (*turn + 1) % 4;
            // SAFETY: at most a slice's bytes into its place in the 4-slice mapping of a live buffer.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), map.add(*turn as usize * LAMP_SLICE), bytes.len().min(LAMP_SLICE)) };
        }
    }

    /// The metal and title normal targets (render-target state between
    /// frames), for the readback of `--bridgeps-dump`.
    pub fn targets(&self) -> (Option<&ID3D12Resource>, Option<&ID3D12Resource>) {
        (self.metal.as_ref().map(|m| &m.0), self.normal.as_ref().map(|n| &n.0))
    }

    /// The frame's clear of the metal target (alpha 1: no bridged row here)
    /// and of the title normal target (0: no word).
    pub(crate) fn clear(&self, l: &ID3D12GraphicsCommandList) {
        if let Some(h) = self.metal_rtv() {
            // SAFETY: an open list; the target is in the render-target state between passes.
            unsafe { l.ClearRenderTargetView(h, &CLEAR_METAL, None) };
        }
        if let Some(h) = self.normal_rtv() {
            // SAFETY: as above.
            unsafe { l.ClearRenderTargetView(h, &[0.0; 4], None) };
        }
        if let Some(h) = self.emissive_rtv() {
            // SAFETY: as above.
            unsafe { l.ClearRenderTargetView(h, &[0.0; 4], None) };
        }
    }

    /// Records the pass on `l`: the colour target copied, RT1 read, the
    /// colour target written (bound alone, no depth). Leaves both targets
    /// in the render-target state and the list's targets unbound: the
    /// caller binds its own again. `clip_to_world`: the inverse of the
    /// camera-relative view-projection (`ourlight=metal`'s view rays).
    pub(crate) fn record(&self, l: &ID3D12GraphicsCommandList, colour: &ID3D12Resource, normal: &ID3D12Resource, rtv: D3D12_CPU_DESCRIPTOR_HANDLE, light_va: u64, table: D3D12_GPU_DESCRIPTOR_HANDLE, clip_to_world: &crate::Mat, depth: &ID3D12Resource, post: Option<(&[f32; 12], D3D12_GPU_DESCRIPTOR_HANDLE)>) {
        use crate::transition;
        // SAFETY: an open list; every resource lives with the pass.
        unsafe {
            // SW1 live `ournormals=0`: the bridged rows' words gone, every pixel takes ours.
            if let Some(h) = self.normal_rtv().filter(|_| !live_normals()) {
                l.ClearRenderTargetView(h, &[0.0; 4], None);
            }
            if let Some((m, _)) = &self.metal {
                transition(l, m, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            }
            if let Some((t, _)) = &self.normal {
                transition(l, t, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            }
            if let Some((t, _)) = &self.emissive {
                transition(l, t, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            }
            transition(l, colour, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
            transition(l, &self.copy, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
            l.CopyResource(&self.copy, colour);
            transition(l, &self.copy, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            transition(l, colour, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            transition(l, normal, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            if let Some(c) = &self.car {
                transition(l, &c.albedo, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            }
            // `lights=on`: our depth as the opaque rows left it.
            if let Some((copy, ..)) = &self.lamps {
                transition(l, depth, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
                transition(l, copy, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
                l.CopyResource(copy, depth);
                transition(l, copy, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
                transition(l, depth, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_DEPTH_WRITE);
            }
            // `post=on` (this frame: `post` given): into the scene-linear target with its pipeline, the composite after.
            let posted = self.post.as_ref().zip(post);
            match &posted {
                Some(((_, c), _)) => l.OMSetRenderTargets(1, Some(&c.begin(l)), true, None),
                None => l.OMSetRenderTargets(1, Some(&rtv), true, None),
            }
            l.SetGraphicsRootSignature(&self.root);
            l.SetPipelineState(posted.as_ref().map_or(&self.pso, |((p, _), _)| p));
            l.SetGraphicsRootConstantBufferView(0, light_va);
            l.SetGraphicsRootDescriptorTable(1, table);
            if self.metal.is_some() {
                l.SetGraphicsRoot32BitConstants(2, 16, clip_to_world.as_ptr().cast(), 0);
            }
            if let Some((_, cb, _, turn)) = &self.lamps {
                l.SetGraphicsRootConstantBufferView(3, cb.GetGPUVirtualAddress() + *turn * LAMP_SLICE as u64);
            }
            if let Some(c) = &self.car {
                c.bind(l, table, self.slot);
            }
            l.DrawInstanced(3, 1, 0, 0);
            if let Some(((_, c), (k, table))) = posted {
                c.record(l, rtv, table, k);
            }
            transition(l, normal, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            if let Some(c) = &self.car {
                transition(l, &c.albedo, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            }
            if let Some((m, _)) = &self.metal {
                transition(l, m, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            }
            if let Some((t, _)) = &self.normal {
                transition(l, t, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            }
            if let Some((t, _)) = &self.emissive {
                transition(l, t, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switch_last_wins() {
        assert_eq!(parse_switch("screen,gbuf=on"), OFF);
        assert_eq!(parse_switch("screen,gbuf=on,ourlight=on"), ON);
        assert_eq!(parse_switch("ourlight=on,ourlight=off"), OFF);
        assert_eq!(parse_switch("ourlight=on,ourlight=metal"), METAL_ON);
        assert!(!parse_normals("ourlight=metal"));
        assert!(parse_normals("ourlight=metal,ournormals=title"));
        assert!(!parse_normals("ournormals=title,ournormals=ours"));
        assert!(!parse_emissive("ourlight=metal"));
        assert!(parse_emissive("ourlight=metal,emissive=on"));
        assert!(!parse_emissive("emissive=on,emissive=off"));
    }

    /// The title's RT5 word (LIGHT1 3 (c)) as the pass decodes it: mantissa x 2^(e - 24), bit 0 dropped.
    #[test]
    fn emissive_word_decodes() {
        let decode = |u: u32| [u & 0x1FE, (u >> 9) & 0x1FE, (u >> 18) & 0x1FE].map(|m| m as f32 * ((u >> 27) as f32 - 24.0).exp2());
        // 1.0 in red: mantissa 256 at e = 16; blue 0.5: 256 at 15 does not share an exponent, so 128 at 16.
        let u = (16 << 27) | (128 << 18) | 256;
        assert_eq!(decode(u), [1.0, 0.0, 0.5]);
        assert_eq!(decode(0), [0.0; 3], "the clear");
    }

    /// The title's RT1 (3ed68103 0xf0c..0xfb0): the normal word in G B A is
    /// our packing; R = 1 - roughness does not touch it.
    #[test]
    fn title_normal_word_is_ours() {
        let n = [0.3f32, -0.2, 0.93];
        let mut b = crate::gbuf::pack_normal(n);
        b[0] = 200;
        let m = crate::gbuf::unpack_normal(b);
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        for k in 0..3 {
            assert!((m[k] - n[k] / l).abs() < 2e-3, "{m:?}");
        }
    }

    /// The title's F0 encoding (3ed68103 0x10f0..0x1124) and our decode round-trip.
    #[test]
    fn f0_encoding_round_trips() {
        for f0 in [0.0f32, 0.02, 0.04, 0.3, 0.56, 0.9, 1.0] {
            let e = (1.0 - (1.0 - f0).sqrt()).sqrt();
            let t = 1.0 - e * e;
            assert!((1.0 - t * t - f0).abs() < 1e-5, "{f0}");
        }
    }

    /// The pass and both raw-shader variants it needs compile (FXC).
    #[test]
    fn shaders_compile() {
        let metal = format!("#define METAL 1\n{SHADER}");
        let titled = format!("#define METAL 1\n#define TITLE_NORMALS 1\n{SHADER}");
        for entry in [(c"vs", c"vs_5_1"), (c"ps", c"ps_5_1")] {
            crate::compile_source(SHADER, entry.0, entry.1).unwrap();
            crate::compile_source(&metal, entry.0, entry.1).unwrap();
            crate::compile_source(&titled, entry.0, entry.1).unwrap();
            // `sunshadow=on`: every variant with the term's view.
            for (m, t) in [(false, false), (true, false), (true, true)] {
                crate::compile_source(&source(m, t, true, false), entry.0, entry.1).unwrap();
            }
            // `lights=on` (with metal): with and without the title's normals and the term.
            for (t, s) in [(false, false), (true, false), (false, true), (true, true)] {
                let src = source(true, t, s, true);
                assert!(src.contains(&format!("#define LAMPS t{}", 3 + u32::from(t) + u32::from(s))) && src.contains(&format!("numDescriptors={})", 4 + u32::from(t) + u32::from(s))), "{t} {s}");
                crate::compile_source(&src, entry.0, entry.1).unwrap();
            }
            assert!(!source(false, false, true, true).contains("LAMPS t"), "lights need metal");
            // `emissive=on` (with metal): after every other view, with and without the others.
            for (t, s, la) in [(false, false, false), (true, true, true), (false, false, true)] {
                let src = source_full(true, t, s, la, Some(0.213), false);
                let n = 3 + u32::from(t) + u32::from(s) + u32::from(la);
                assert!(src.contains(&format!("#define EMISSIVE t{n}")) && src.contains(&format!("numDescriptors={})", n + 1)), "{t} {s} {la}");
                crate::compile_source(&src, entry.0, entry.1).unwrap();
                if entry.0 == c"ps" {
                    crate::compile_source(&source_full(true, t, s, la, Some(0.213), true), entry.0, entry.1).unwrap();
                }
            }
            assert!(!source_full(false, false, false, false, Some(1.0), false).contains("EMISSIVE t"), "emissive needs metal");
            // `post=on`: the scene-linear variant of the plain, the metal and the full pass.
            if entry.0 == c"ps" {
                for (m, t, s, la) in [(false, false, false, false), (true, false, false, false), (true, true, true, true)] {
                    crate::compile_source(&source_post(m, t, s, la, true), entry.0, entry.1).unwrap();
                }
            }
        }
        let with = format!("#define OURLIGHT 1\n#define GBUF 1\n{}", crate::RAW_SHADER);
        let plain = format!("#define OURLIGHT 1\n{}", crate::RAW_SHADER);
        for src in [&with, &plain] {
            crate::compile_source(src, c"ps", c"ps_5_1").unwrap();
            crate::compile_source(src, c"vs", c"vs_5_1").unwrap();
        }
        // The G-buffer variant still writes three targets.
        let ps = crate::compile_source(&with, c"ps", c"ps_5_1").unwrap();
        assert_eq!(crate::gbuf::targets(crate::blob_bytes(&ps)), 7);
    }
}
