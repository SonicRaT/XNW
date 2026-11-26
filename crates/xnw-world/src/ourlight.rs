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

use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// Process-wide: set by `Pass::new` before it compiles the raw shader.
static ON: AtomicBool = AtomicBool::new(false);

pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

pub(crate) fn set(v: bool) {
    ON.store(v, Ordering::Relaxed);
}

/// `ourlight=on|off` in `XNW_WORLD_RENDER` (the last one wins; default off).
pub fn parse_switch(v: &str) -> bool {
    v.split(',').filter_map(|p| p.trim().strip_prefix("ourlight=")).next_back() == Some("on")
}

/// The switch as the environment asks it now (the frame loaders pick the
/// light with it: [`crate::lighting::Lighting::ours`]).
pub fn asked() -> bool {
    parse_switch(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default())
}

/// The raw shader's flags bit: this row comes before the lighting pass
/// (its colour output is its albedo).
pub const DEFERRED: u32 = 1 << 3;

/// The pass. Root: the Light buffer (b0), a table of two views (t0 the
/// colour copy, t1 the G-buffer's RT1).
pub const SHADER: &str = r#"
#define RS "CBV(b0), DescriptorTable(SRV(t0, numDescriptors=2), visibility=SHADER_VISIBILITY_PIXEL)"
// The frame's Light buffer (lighting.rs `constants`): light dir (towards it), radiance (w: exposure),
// ambient sky, ambient ground, Hable A B C D, E F W 1/Hable(W), sky radiance scale (x), fog colour (yzw).
cbuffer Light : register(b0) { float4 lDir; float4 lRad; float4 lSky; float4 lGround; float4 tone0; float4 tone1; float4 lPad; };
Texture2D<float4> colourIn : register(t0);
Texture2D<float4> normalIn : register(t1);
float3 hableC(float3 x) { return (x * (tone0.x * x + tone0.z * tone0.y) + tone0.w * tone1.x) / (x * (tone0.x * x + tone0.y) + tone0.w * tone1.y) - tone1.x / tone1.y; }
float3 tonemap(float3 hdr) { return pow(saturate(hableC(max(hdr * lRad.w, 0)) * tone1.w), 1.0 / 2.2); }
float3 skyRadiance(float3 c) { return pow(max(c, 0), 2.2) * lPad.x; }
// RT1 (0, b0, b1, b2) / 255: the title's octahedral normal, word = y' << 12 | x' (gbuf.rs `unpack_normal`).
float3 unpackNormal(float3 b) {
    uint3 q = uint3(saturate(b) * 255 + 0.5);
    uint w = q.x | (q.y << 8) | (q.z << 16);
    float2 e = (float2(w & 4095, (w >> 12) & 4095) + 0.5 - 2047.5) / 2047.5;
    float z = 1 - abs(e.x) - abs(e.y);
    float2 f = (1 - abs(e.yx)) * float2(e.x >= 0 ? 1 : -1, e.y >= 0 ? 1 : -1);
    return normalize(float3(z < 0 ? f : e, z) + 1e-6);
}
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
    float3 lit = tonemap(base * (lRad.rgb * (nl * 0.31830989) + amb));
    float vis = saturate((g.r * 255 - 1) / 254);
    lit = lerp(tonemap(skyRadiance(lPad.yzw)), lit, vis);
    return float4(g.r > 0.5 / 255 ? lit : c.rgb, c.a);
}
"#;

/// The pass's objects: root, pipeline, the colour copy (its view at
/// `slot`, RT1's at `slot + 1`).
pub struct Lit {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    copy: ID3D12Resource,
    pub slot: u32,
}

impl Lit {
    /// Made once by `Pass::new`: `colour_format` x `width` x `height` copy, the
    /// two views written at `cpu(slot)` and `cpu(slot + 1)`.
    pub(crate) fn new(device: &ID3D12Device, normal: &ID3D12Resource, colour_format: DXGI_FORMAT, width: u32, height: u32, slot: u32, cpu: &dyn Fn(u32) -> D3D12_CPU_DESCRIPTOR_HANDLE) -> Result<Lit, String> {
        let e = |e: windows::core::Error| format!("ourlight: {e}");
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
            desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            let pso = device.CreateGraphicsPipelineState::<ID3D12PipelineState>(&desc);
            drop(ManuallyDrop::into_inner(desc.pRootSignature));
            let pso = pso.map_err(e)?;
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
            Ok(Lit { root, pso, copy, slot })
        }
    }

    /// Records the pass on `l`: the colour target copied, RT1 read, the
    /// colour target written (bound alone, no depth). Leaves both targets
    /// in the render-target state and the list's targets unbound: the
    /// caller binds its own again.
    pub(crate) fn record(&self, l: &ID3D12GraphicsCommandList, colour: &ID3D12Resource, normal: &ID3D12Resource, rtv: D3D12_CPU_DESCRIPTOR_HANDLE, light_va: u64, table: D3D12_GPU_DESCRIPTOR_HANDLE) {
        use crate::transition;
        // SAFETY: an open list; every resource lives with the pass.
        unsafe {
            transition(l, colour, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
            transition(l, &self.copy, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
            l.CopyResource(&self.copy, colour);
            transition(l, &self.copy, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            transition(l, colour, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            transition(l, normal, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            l.OMSetRenderTargets(1, Some(&rtv), true, None);
            l.SetGraphicsRootSignature(&self.root);
            l.SetPipelineState(&self.pso);
            l.SetGraphicsRootConstantBufferView(0, light_va);
            l.SetGraphicsRootDescriptorTable(1, table);
            l.DrawInstanced(3, 1, 0, 0);
            transition(l, normal, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switch_last_wins() {
        assert!(!parse_switch("screen,gbuf=on"));
        assert!(parse_switch("screen,gbuf=on,ourlight=on"));
        assert!(!parse_switch("ourlight=on,ourlight=off"));
    }

    /// The pass and both raw-shader variants it needs compile (FXC).
    #[test]
    fn shaders_compile() {
        for entry in [(c"vs", c"vs_5_1"), (c"ps", c"ps_5_1")] {
            crate::compile_source(SHADER, entry.0, entry.1).unwrap();
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
