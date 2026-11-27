//! `XNW_WORLD_RENDER=...,post=on|key` (LIGHT2 after POST1, default off;
//! needs `ourlight=on|metal`): our lighting pass writes scene-linear colour
//! to an RGBA16F target of ours (no exposure, no tone map), and ONE
//! composite pass of ours takes it to the 8-bit target, as the analytic
//! path of the title's composite does (PS_CompositeNoise b4045fba 0x11a8..
//! 0x1268, 0x13d8):
//!   x = E x colour;
//!   y = sat(((x (A x + CB) + DE) / (x (A x + B) + DF) - E/F) / Hable(W)),
//!       the set P = lerp(set 0, set 1, t), t = sat(E x k + o)
//!       (PS_CalculateExposure d80a7c70);
//!   the sRGB OETF (12.92 below 0.0031308, else 1.055 y^(1/2.4) - 0.055);
//!   + (noise - 0.5) / 255: an ordered dither of ours (interleaved gradient
//!     noise, no image), where the title adds blue noise.
//!
//! The curve sets: the title's block b6 holds set 0 at +0x4A0 and set 1 at
//! +0x4C0; the timecycle render copy holds the same values at floats
//! 847..853 and 838..844 [V pk8a f1030: 0.22 0.3 0.1 0.2 0.01 0.3 11.2 and
//! the same with E 0, both places], so they are read from the timecycle
//! (live: no hook). k, o = b6+0x4E0 / +0x4E4 = 0.4 / 2.0 [V f1030; their
//! timecycle floats are not found]: [`BLEND`]. With them t = 1: set 1.
//!
//! E: the title's exposure exists on the GPU only (it read 0 in the
//! captures). `post=on` (POST3): an adapted EV, the title's own rule
//! ([`adapt`], PS_CalculateExposure d80a7c70) towards [`target_ev`]: our
//! key held to the timecycle's EV range (floats 796..797) shifted by [`K`],
//! so a free frame (range -32..32) aims at our key and a pinned one at the
//! title's pin; the step is the game's (0x14ABBEF4C, [`STEP_MAX`]); a
//! camera cut or the first frame starts at the target (PS_ResetExposure).
//! (Before POST3 `post=on` took exp2 of the timecycle's EV, 794 + 795 clamped
//! to 796..797: E stepped 1 <-> 0.354 frame to frame, pin <-> free.)
//! `post=key`: our lighting's own keyed exposure (the picture's level as
//! without `post`), not adapted. Live switch `exposure`: n > 0 sets E =
//! n / 100, not adapted. No luminance pass yet.
//!
//! What stays as it was: the rows after the lighting pass (water,
//! transparent rows, the sky) light and tone-map themselves into the 8-bit
//! target; they take the same E (the frame's Light buffer is given it, the
//! sky stand-in's scale adjusted so its picture does not move). The fog
//! towards the sky stand-in's colour is blended after the curve, in
//! display values, as before. Pixels the lighting pass does not light keep
//! their colour (the composite blends with the lit flag).

use std::mem::ManuallyDrop;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::lighting::Lighting;

pub const OFF: u32 = 0;
/// E adapted towards our key in the timecycle's EV range ([`adapt`], [`target_ev`]).
pub const SEED: u32 = 1;
/// E = our lighting's keyed exposure.
pub const KEY: u32 = 2;

/// `post=on|key|off` in `XNW_WORLD_RENDER` (the last one wins; default off).
pub fn parse_switch(v: &str) -> u32 {
    match v.split(',').filter_map(|p| p.trim().strip_prefix("post=")).next_back() {
        Some("on") => SEED,
        Some("key") => KEY,
        _ => OFF,
    }
}

/// The switch as the environment asks it now.
pub fn asked() -> u32 {
    parse_switch(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default())
}

/// Live `post` (taken once a frame; start: 1 when asked): 0 the lighting pass tone-maps itself, as without `post`.
pub fn live_on() -> bool {
    xnw_core::live::on(xnw_core::live::POST, || u32::from(asked() != OFF))
}

/// Live `exposure` (taken once a frame; start 0): n > 0 is E = n / 100.
pub fn live_exposure() -> Option<f32> {
    match xnw_core::live::value(xnw_core::live::EXPOSURE, || 0) {
        0 => None,
        n => Some(n as f32 / 100.0),
    }
}

/// The scene-linear target's format (the title's scene target is the same, POST1 5a).
pub const HDR: DXGI_FORMAT = DXGI_FORMAT_R16G16B16A16_FLOAT;

/// (k, o) of t = sat(E k + o): b6+0x4E0, +0x4E4 [V pk8a f1030].
pub const BLEND: [f32; 2] = [0.4, 2.0];

/// PS_CalculateExposure d80a7c70's adaptation constants (b6+0x3A8, +0x3AC,
/// +0x3B0, +0x3B4: rate min, rate max, speed s, snap): 0.01 100 0.65 0
/// [V pk8a f1027 sink 3522, f1029 sink 1038; read at 0x1b8/0x1c0]. b6+0x3A4,
/// the shader's a (1.0, once 0.0667 = 1/15), is taken as the step [I].
pub const ADAPT: [f32; 4] = [0.01, 100.0, 0.65, 0.0];

/// The longest step [`adapt`] takes, s: the title's step clamp 1/15 (b6+0x3A4 = 0.0667 at f1029 sink 1038) [I].
pub const STEP_MAX: f32 = 1.0 / 15.0;

/// The EV offset between our scene-linear units and the title's: the pk8a
/// f1030 pin (-1.5) + K = log2 of our key there (1.711): K = 0.775 + 1.5
/// [I: one frame's calibration, no title evidence].
pub const K: f32 = 2.27;

/// The title's exposure step (PS_CalculateExposure d80a7c70 0x1f4..0x27c [V]):
///   d = target - prev; rate = clamp(|d| sat(a s) / a, rate min, rate max);
///   EV = target when rate <= snap or d = 0, else prev + sign(d) a rate,
///   never past the target (med3 of the step, prev and target).
/// `a` is the step in seconds ([`STEP_MAX`] at most); a = 0 (paused) holds
/// prev [ours: the shader divides by a]; prev not finite takes the target.
pub fn adapt(prev: f32, target: f32, a: f32) -> f32 {
    let [rmin, rmax, s, snap] = ADAPT;
    if !prev.is_finite() {
        return target;
    }
    // (f32::min takes a NaN step as STEP_MAX: tested first.)
    if !target.is_finite() || a.is_nan() || a <= 0.0 {
        return prev;
    }
    let a = a.min(STEP_MAX);
    let d = target - prev;
    let rate = (d.abs() * (a * s).clamp(0.0, 1.0) / a).clamp(rmin, rmax);
    if rate <= snap || d == 0.0 {
        return target;
    }
    (prev + d.signum() * a * rate).clamp(prev.min(target), prev.max(target))
}

/// ComputeLuxCompressionInfoSmooth c62dbccf 0x068..0x0d0 [V]: E + (1 -
/// exp(-dt)) (target - E), dt = CB+0xE4 (the game step, 0 when paused;
/// exp(-dt) as exp2(dt x lit 0xbfb8aa3b = -1/ln 2)). The title's composite
/// multiplier chain (B, POST2); its target's constants are unread, so it is
/// not in our chain yet.
pub fn smooth(prev: f32, target: f32, dt: f32) -> f32 {
    prev + (1.0 - (-dt.max(0.0)).exp()) * (target - prev)
}

fn ok(e: f32) -> f32 {
    if e.is_finite() && e > 0.0 { e.clamp(1e-4, 1e4) } else { 1.0 }
}

/// [`SEED`]'s target EV: clamp(log2(our key) - K, the timecycle's range) + K.
/// Free (-32..32): our key; pinned (low = high): the title's pin + K.
pub fn target_ev(light: &Lighting) -> f32 {
    let [lo, hi] = light.ev_range;
    let (lo, hi) = (lo.min(hi), lo.max(hi));
    let key = ok(light.exposure).log2();
    if lo.is_finite() && hi.is_finite() { (key - K).clamp(lo, hi) + K } else { key }
}

/// The frame's E, the adapted EV to keep (None: nothing adapted) and the
/// source's name: the live value; else by `mode` ([`SEED`]: [`adapt`] from
/// `prev` (None: start at the target) towards [`target_ev`] over the step
/// `dt` s; [`KEY`]: the lighting's exposure).
pub fn exposure(light: &Lighting, mode: u32, live: Option<f32>, prev: Option<f32>, dt: f32) -> (f32, Option<f32>, &'static str) {
    match (live, mode) {
        (Some(e), _) => (ok(e), None, "live exposure"),
        (None, KEY) => (ok(light.exposure), None, "our key"),
        _ => {
            let target = target_ev(light);
            let ev = prev.map_or(target, |p| adapt(p, target, dt)).clamp(-13.0, 13.0);
            (ok(ev.exp2()), Some(ev), "adapted")
        }
    }
}

/// The curve set at exposure `e`: lerp(set 0, set 1, sat(e k + o)), and t.
pub fn curve(light: &Lighting, e: f32) -> ([f32; 7], f32) {
    let t = (e * BLEND[0] + BLEND[1]).clamp(0.0, 1.0);
    (std::array::from_fn(|i| light.curve_dark[i] + (light.curve[i] - light.curve_dark[i]) * t), t)
}

/// The title's curve at `x` (exposed, linear) for the set `c` (A B C D E F W): 0..1.
pub fn tone(x: f32, c: &[f32; 7]) -> f32 {
    (crate::lighting::hable(x.max(0.0), c) / crate::lighting::hable(c[6], c)).clamp(0.0, 1.0)
}

/// The sRGB OETF.
pub fn srgb(y: f32) -> f32 {
    if y < 0.003_130_8 { y * 12.92 } else { y.powf(1.0 / 2.4) * 1.055 - 0.055 }
}

/// The composite's constants: (E, A, B, 1 / Hable(W)), (CB, DE, DF, E/F)
/// (the title's two rows, with E in the place of its 1/W), the fog's
/// display colour and the dither's frame offset.
pub fn params(light: &Lighting, e: f32, frame: u64) -> [f32; 12] {
    let (c, _) = curve(light, e);
    // The fog as the lighting pass showed it: the sky stand-in's colour through the Light buffer's own chain.
    let fog = light.fog.map(|f| crate::lighting::display(f.max(0.0).powf(2.2) * light.sky_scale * light.exposure, &light.curve));
    [e, c[0], c[1], 1.0 / crate::lighting::hable(c[6], &c), c[2] * c[1], c[3] * c[4], c[3] * c[5], c[4] / c[5], fog[0], fog[1], fog[2], (frame & 31) as f32]
}

/// The frame's probe line; `dt` the step [`exposure`] was given.
pub fn line(light: &Lighting, e: f32, source: &str, dt: f32) -> String {
    let (c, t) = curve(light, e);
    let [lo, hi] = light.ev_range;
    format!(
        "post: E {e:.3} EV {:.3} ({source}; EV target {:.3}, range {lo}..{hi} {}, step {dt:.4} s, K {K}; timecycle EV {:.2}, our key {:.3}), curve A {} B {} C {} D {} E {} F {} W {} (t = sat({} E + {}) = {t:.2} between set 0 E {} W {} and set 1 E {} W {}), scene-linear RGBA16F, sRGB OETF, ordered dither",
        ok(e).log2(),
        target_ev(light),
        if lo == hi { "pinned" } else { "free" },
        light.ev,
        light.exposure, c[0], c[1], c[2], c[3], c[4], c[5], c[6], BLEND[0], BLEND[1], light.curve_dark[4], light.curve_dark[6], light.curve[4], light.curve[6]
    )
}

/// The composite. b0: [`params`]; t0 the scene-linear target (.a: the fog
/// visibility of a lit pixel, below 0 where the lighting pass lit nothing).
/// One load at a clamped texel, no branch, one return; the blend keeps the
/// target's colour where alpha is 0, and its alpha everywhere.
pub const SHADER: &str = r#"
#define RS "RootConstants(num32BitConstants=12, b0, visibility=SHADER_VISIBILITY_PIXEL), DescriptorTable(SRV(t0, numDescriptors=1), visibility=SHADER_VISIBILITY_PIXEL)"
cbuffer K : register(b0) { float4 k0; float4 k1; float4 k2; };
Texture2D<float4> hdrIn : register(t0);
struct VsOut { float4 pos : SV_Position; };
[RootSignature(RS)]
VsOut vs(uint id : SV_VertexID) {
    VsOut o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}
[RootSignature(RS)]
float4 ps(VsOut i) : SV_Target {
    uint w, h;
    hdrIn.GetDimensions(w, h);
    int3 p = int3(min(uint2(i.pos.xy), uint2(max(w, 1), max(h, 1)) - 1), 0);
    float4 s = hdrIn.Load(p);
    float3 x = max(s.rgb, 0) * k0.x;
    float3 y = saturate(((x * (k0.y * x + k1.x) + k1.y) / (x * (k0.y * x + k0.z) + k1.z) - k1.w) * k0.w);
    float3 lo = y * 12.92, hi = pow(max(y, 1e-7), 1.0 / 2.4) * 1.055 - 0.055;
    float3 d = float3(y.x < 0.0031308 ? lo.x : hi.x, y.y < 0.0031308 ? lo.y : hi.y, y.z < 0.0031308 ? lo.z : hi.z);
    d = lerp(k2.rgb, d, saturate(s.a));
    float n = frac(52.9829189 * frac(dot(float2(p.xy) + 5.588238 * k2.w, float2(0.06711056, 0.00583715))));
    return float4(saturate(d + (n - 0.5) / 255.0), s.a < 0 ? 0.0 : 1.0);
}
"#;

/// The composite's objects and the scene-linear target (shader-resource
/// state between passes; its view at `slot`).
pub struct Composite {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    hdr: ID3D12Resource,
    rtv: ID3D12DescriptorHeap,
    pub slot: u32,
}

impl Composite {
    pub(crate) fn new(device: &ID3D12Device, colour_format: DXGI_FORMAT, width: u32, height: u32, slot: u32, cpu: &dyn Fn(u32) -> D3D12_CPU_DESCRIPTOR_HANDLE) -> Result<Composite, String> {
        let e = |e: windows::core::Error| format!("post: {e}");
        let (vs, ps) = (crate::compile_source(SHADER, c"vs", c"vs_5_1")?, crate::compile_source(SHADER, c"ps", c"ps_5_1")?);
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
            // Colour by the lit flag; the target's alpha kept.
            desc.BlendState.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
                BlendEnable: true.into(),
                SrcBlend: D3D12_BLEND_SRC_ALPHA,
                DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
                BlendOp: D3D12_BLEND_OP_ADD,
                SrcBlendAlpha: D3D12_BLEND_ZERO,
                DestBlendAlpha: D3D12_BLEND_ONE,
                BlendOpAlpha: D3D12_BLEND_OP_ADD,
                RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                ..Default::default()
            };
            let pso = device.CreateGraphicsPipelineState::<ID3D12PipelineState>(&desc);
            drop(ManuallyDrop::into_inner(desc.pRootSignature));
            let pso = pso.map_err(e)?;
            let d = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: u64::from(width), Height: height, DepthOrArraySize: 1, MipLevels: 1, Format: HDR, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, ..Default::default() };
            let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
            let clear = D3D12_CLEAR_VALUE { Format: HDR, Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0; 4] } };
            let mut hdr: Option<ID3D12Resource> = None;
            device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, Some(&clear), &mut hdr).map_err(e)?;
            let hdr = hdr.ok_or("post: no scene-linear target")?;
            let rtv: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() }).map_err(e)?;
            device.CreateRenderTargetView(&hdr, None, rtv.GetCPUDescriptorHandleForHeapStart());
            let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: HDR,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } },
            };
            device.CreateShaderResourceView(&hdr, Some(&view), cpu(slot));
            Ok(Composite { root, pso, hdr, rtv, slot })
        }
    }

    /// The lighting pass's target for this frame: the scene-linear target, made a render target.
    pub(crate) fn begin(&self, l: &ID3D12GraphicsCommandList) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        crate::transition(l, &self.hdr, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
        // SAFETY: a live heap.
        unsafe { self.rtv.GetCPUDescriptorHandleForHeapStart() }
    }

    /// The composite on `l` into `rtv` (the 8-bit target, bound alone);
    /// `table`: the heap's handle at [`Composite::slot`].
    pub(crate) fn record(&self, l: &ID3D12GraphicsCommandList, rtv: D3D12_CPU_DESCRIPTOR_HANDLE, table: D3D12_GPU_DESCRIPTOR_HANDLE, k: &[f32; 12]) {
        crate::transition(l, &self.hdr, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        // SAFETY: an open list; every resource lives with the pass.
        unsafe {
            l.OMSetRenderTargets(1, Some(&rtv), true, None);
            l.SetGraphicsRootSignature(&self.root);
            l.SetPipelineState(&self.pso);
            l.SetGraphicsRoot32BitConstants(0, 12, k.as_ptr().cast(), 0);
            l.SetGraphicsRootDescriptorTable(1, table);
            l.DrawInstanced(3, 1, 0, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SET0: [f32; 7] = [0.22, 0.3, 0.1, 0.2, 0.01, 0.3, 11.2];
    const SET1: [f32; 7] = [0.22, 0.3, 0.1, 0.2, 0.0, 0.3, 11.2];

    fn light() -> Lighting {
        Lighting { curve: SET1, curve_dark: SET0, exposure: 1.7, ev: -1.5, ev_range: [-1.5, -1.5], sky_scale: 1.0, ..Default::default() }
    }

    #[test]
    fn switch_and_exposure() {
        assert_eq!((parse_switch("screen"), parse_switch("post=on"), parse_switch("post=on,post=key"), parse_switch("post=key,post=off")), (OFF, SEED, KEY, OFF));
        let l = light();
        // Pinned -1.5: clamp(log2 1.7 - 2.27, -1.5, -1.5) + 2.27 = 0.77, E = 1.70530; no prev: the target itself.
        let (e, ev, s) = exposure(&l, SEED, None, None, 1.0 / 30.0);
        assert!((e - 1.705_30).abs() < 1e-4 && (ev.unwrap() - 0.77).abs() < 1e-5 && s == "adapted", "{e} {ev:?}");
        // Free: our key.
        let free = Lighting { ev_range: [-32.0, 32.0], ..l };
        assert!((exposure(&free, SEED, None, None, 0.0).0 - 1.7).abs() < 1e-5);
        assert!((exposure(&Lighting { ev_range: [f32::NAN; 2], ..l }, SEED, None, None, 0.0).0 - 1.7).abs() < 1e-5);
        assert_eq!(exposure(&l, KEY, None, Some(3.0), 1.0).0, 1.7);
        assert_eq!(exposure(&l, KEY, None, None, 0.0).1, None);
        assert_eq!(exposure(&l, SEED, Some(0.25), Some(3.0), 1.0).0, 0.25);
        // With a prev the step is the title's: 1 EV below the free target, 1/30 s: 0.65 / 30 EV towards it.
        let (_, ev, _) = exposure(&free, SEED, None, Some(1.7f32.log2() - 1.0), 1.0 / 30.0);
        assert!((ev.unwrap() - (1.7f32.log2() - 1.0 + 0.65 / 30.0)).abs() < 1e-5, "{ev:?}");
    }

    /// The rate rule: never past the target, never back from prev, symmetric in the sign of d;
    /// 2 EV at 1/30 s moves 2 x 0.65 / 30 = 0.0433; the rate floor (0.01 EV/s) and cap (100 EV/s); a = 0 holds.
    #[test]
    fn adapt_is_the_titles_rule() {
        for prev in [-6.0f32, -1.5, 0.0, 0.77, 3.0] {
            for d in [-40.0f32, -2.0, -0.5, -0.01, -1e-5, 0.0, 1e-5, 0.01, 0.5, 2.0, 40.0] {
                for a in [1.0 / 120.0f32, 1.0 / 30.0, 1.0 / 15.0, 0.5] {
                    let up = adapt(prev, prev + d, a);
                    assert!(up >= prev.min(prev + d) && up <= prev.max(prev + d), "past the target: {prev} {d} {a} -> {up}");
                    let down = adapt(prev, prev - d, a);
                    assert!(((up - prev) + (down - prev)).abs() < 1e-5, "not symmetric: {prev} {d} {a}: {up} {down}");
                }
            }
        }
        assert!((adapt(0.0, 2.0, 1.0 / 30.0) - 2.0 * 0.65 / 30.0).abs() < 1e-6);
        // a above 1/15 is taken as 1/15: 2 x 0.65 / 15.
        assert!((adapt(0.0, 2.0, 0.5) - 2.0 * 0.65 / 15.0).abs() < 1e-6);
        // Cap: |d| s = 650 EV/s -> 100 EV/s x 1/30 s.
        assert!((adapt(0.0, 1000.0, 1.0 / 30.0) - 100.0 / 30.0).abs() < 1e-3);
        // Floor: |d| s = 0.0065 EV/s -> 0.01 EV/s x 1/30 s.
        assert!((adapt(0.0, 0.01, 1.0 / 30.0) - 0.01 / 30.0).abs() < 1e-7);
        assert_eq!(adapt(0.0, 1e-6, 1.0 / 30.0), 1e-6);
        assert_eq!((adapt(1.0, 3.0, 0.0), adapt(1.0, 3.0, -1.0), adapt(1.0, 3.0, f32::NAN), adapt(f32::NAN, 3.0, 0.1)), (1.0, 1.0, 1.0, 3.0));
        // Exponential with time constant 1 / 0.65 s: 99 % of a 2 EV step in about 7 s of 1/30 s frames.
        let left = |n: usize| 2.0 - (0..n).fold(0.0f32, |ev, _| adapt(ev, 2.0, 1.0 / 30.0));
        assert!(left(200) > 0.02 && left(215) < 0.02, "{} {}", left(200), left(215));
    }

    /// The smooth step (c62dbccf): dt 0 holds, dt 1 s takes 1 - 1/e of the way, 30 frames of 1/30 s about the same.
    #[test]
    fn smooth_is_the_titles_step() {
        assert_eq!(smooth(1.0, 2.0, 0.0), 1.0);
        assert_eq!(smooth(1.0, 2.0, -1.0), 1.0);
        assert!((smooth(0.0, 1.0, 1.0) - 0.632_121).abs() < 1e-6);
        assert!((smooth(4.0, 2.0, 1.0) - (4.0 - 2.0 * 0.632_121)).abs() < 1e-5);
        assert!((smooth(0.0, 1.0, 50.0) - 1.0).abs() < 1e-6);
        let e = (0..30).fold(0.0f32, |e, _| smooth(e, 1.0, 1.0 / 30.0));
        assert!((e - 0.632_121).abs() < 1e-4, "{e}");
    }

    /// pk8a f1027-1032: the title's range -2, -2, -32..32 x 3, -1.5 (b6) and our timecycle copy's
    /// free, -1.5, free (794 = 1), -1.5, free, free [V timecycle.bin]; our key 1.711. The old rule
    /// stepped E by 1.5 EV and more a frame; the adapted E moves less than 0.05 EV a frame
    /// while its target flips by 0.5 EV, at 1/30 s and at the 1/15 s fallback.
    #[test]
    fn jumping_target_gives_a_smooth_e() {
        let key = 1.711f32;
        let ranges = |r: &[[f32; 2]]| r.iter().map(|&ev_range| Lighting { exposure: key, ev_range, ..light() }).collect::<Vec<_>>();
        let title = ranges(&[[-2.0, -2.0], [-2.0, -2.0], [-32.0, 32.0], [-32.0, 32.0], [-32.0, 32.0], [-1.5, -1.5]]);
        let ours = ranges(&[[-32.0, 32.0], [-1.5, -1.5], [-32.0, 32.0], [-1.5, -1.5], [-32.0, 32.0], [-32.0, 32.0]]);
        let old_ev = [0.0f32, -1.5, 1.0, -1.5, 0.0, 0.0];
        assert!(old_ev.windows(2).any(|w| (w[1] - w[0]).abs() > 1.4), "the old rule jumped");
        for frames in [&title, &ours] {
            let targets: Vec<f32> = frames.iter().map(target_ev).collect();
            // The title's ranges flip the target by 0.5 EV (pin -2 + K = 0.27 against our key 0.775);
            // ours by 0.005 (K is calibrated on the -1.5 pin).
            let flip = targets.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
            assert!(if std::ptr::eq(frames, &title) { flip > 0.4 } else { flip < 0.01 }, "{targets:?}");
            for dt in [1.0 / 30.0f32, STEP_MAX] {
                let mut prev = None;
                let mut evs = Vec::new();
                for l in frames.iter() {
                    let (e, ev, _) = exposure(l, SEED, None, prev, dt);
                    assert!((e.log2() - ev.unwrap()).abs() < 1e-4);
                    prev = ev;
                    evs.push(ev.unwrap());
                }
                assert!(evs.windows(2).all(|w| (w[1] - w[0]).abs() < 0.05), "{dt}: {evs:?} for {targets:?}");
                // Towards the target: each step's sign follows the target's side.
                for (i, w) in evs.windows(2).enumerate() {
                    assert!((w[1] - w[0]) * (targets[i + 1] - w[0]) >= 0.0, "{i}: {evs:?} {targets:?}");
                }
            }
        }
    }

    /// Hand values, set 1 (E 0, W 11.2):
    /// Hable(11.2) = 11.2 (2.464 + 0.03) / (11.2 (2.464 + 0.3) + 0.06) = 27.9328 / 31.0168 = 0.900570;
    /// Hable(1) = 0.25 / 0.58 = 0.431034; tone(1) = 0.478624; sRGB = 1.055 x 0.478624^(1/2.4) - 0.055 = 0.7211.
    /// sRGB(0.001) = 0.01292 (the old pow 1/2.2 gave 0.0433: deep shadows sink); sRGB(1) = 1.
    #[test]
    fn curve_and_oetf_match_hand_values() {
        assert!((crate::lighting::hable(11.2, &SET1) - 0.900_570).abs() < 1e-5);
        assert!((tone(1.0, &SET1) - 0.478_624).abs() < 1e-5, "{}", tone(1.0, &SET1));
        assert!((srgb(tone(1.0, &SET1)) - 0.7211).abs() < 1e-3, "{}", srgb(tone(1.0, &SET1)));
        assert!((srgb(0.001) - 0.012_92).abs() < 1e-7 && (srgb(1.0) - 1.0).abs() < 1e-6);
        assert!((srgb(0.003_130_8) - 0.040_45).abs() < 1e-4, "continuous at the knee");
        assert_eq!((tone(0.0, &SET1), tone(11.2, &SET1), tone(100.0, &SET1)), (0.0, 1.0, 1.0));
        // Mid-tones: sRGB against the old pow 1/2.2 within 0.01 above 0.05.
        for y in [0.05f32, 0.18, 0.5, 0.9] {
            assert!((srgb(y) - y.powf(1.0 / 2.2)).abs() < 0.012, "{y}");
        }
    }

    /// t = sat(0.4 E + 2) = 1 for every E >= 0: set 1; the rows are the title's (A, B, 1/Hable(W)), (CB, DE, DF, E/F).
    #[test]
    fn params_take_set_1() {
        let l = Lighting { fog: [0.5, 0.5, 0.5], ..light() };
        for e in [0.0f32, 0.25, 1.0, 8.0] {
            assert_eq!(curve(&l, e), (SET1, 1.0));
        }
        let k = params(&l, 0.5, 37);
        assert_eq!(&k[..3], &[0.5, 0.22, 0.3]);
        assert!((k[3] - 1.0 / 0.900_570).abs() < 1e-4);
        assert!((k[4] - 0.03).abs() < 1e-7 && k[5] == 0.0 && (k[6] - 0.06).abs() < 1e-7 && k[7] == 0.0);
        // The fog: display(0.5^2.2 x 1 x 1.7) with set 1, the lighting pass's own chain.
        let f = crate::lighting::display(0.5f32.powf(2.2) * 1.7, &SET1);
        assert_eq!((k[8], k[9], k[10], k[11]), (f, f, f, 5.0));
        // Between the sets the values are lerped before the rows are made (set 0 E 0.01: E/F 0.0333).
        let (c, t) = (std::array::from_fn::<f32, 7, _>(|i| SET0[i] + (SET1[i] - SET0[i]) * 0.5), 0.5);
        assert!((c[4] - 0.005).abs() < 1e-7 && t == 0.5);
    }

    #[test]
    fn shader_compiles() {
        for entry in [(c"vs", c"vs_5_1"), (c"ps", c"ps_5_1")] {
            crate::compile_source(SHADER, entry.0, entry.1).unwrap();
        }
    }
}
