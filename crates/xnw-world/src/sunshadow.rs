//! `XNW_WORLD_RENDER=...,sunshadow=on|title|own` (SHADOW1, default off;
//! needs `ourlight=on|metal`): sun (or moon) shadows in our lighting pass.
//!
//! The title's chain:
//! stage 23 + i draws cascade i with `MS_VS_CascadeShadows_*` into a depth
//! atlas, `cascadeshadows_rendering:CS_ShadowRenderTiled*` and the shadow
//! mask programs turn it into a screen R8 mask (frame-table slot 95) that
//! TL_PS loads. No capture holds the bindings of those compute programs,
//! and live our own lighting pass ([`crate::ourlight`]) lights the picture,
//! so the chain here is ours, on the title's rows and views:
//! 1. a depth atlas of ours (2 x 2 tiles of [`TILE`], D32, depth growing
//!    away from the light as the title's cascade projections have it);
//! 2. the title's cascade rows (list rows of stage 23..26, their world
//!    matrices relative to their own viewport's camera [V pk6 f1205])
//!    drawn into tile `stage - 23` by our raw vertex path with that
//!    viewport's rotation and orthographic projection, depth only (alpha
//!    clipped rows through our texture table). The main view's pre-skinned
//!    ped rows are drawn into every cascade too (the title's ped cascade
//!    rows are skinned by programs our raw path does not run). A cascade
//!    the title does not draw in a frame keeps its tile and view (the
//!    title redraws them only now and then: pk6 f1205-1206 of 12 frames);
//! 3. one full-screen pass writes the screen term ([`TERM`]: .r = lit
//!    fraction, 16 loads, weighted 3 x 3; .g = (cascade + 1) / 4) from a
//!    copy of our depth; our lighting pass loads it and scales the sun's
//!    diffuse and specular terms.
//!
//! `own` (and `on` when the title has drawn no cascade for [`HOLD`]
//! frames): three cascades of ours around the main camera along the
//! frame's light direction, drawn from the main view's opaque rows (what
//! is on screen casts; rows outside the main frustum are not in the list).
//!
//! Live switch `sunshadow` (0 off: the term is cleared to 1; 1 on, 2 title,
//! 3 own). The objects are made at start: a run started without it has none.

use super::*;

pub const OFF: u32 = 0;
/// The title's cascades while it draws them, else ours.
pub const AUTO: u32 = 1;
pub const TITLE: u32 = 2;
pub const OWN: u32 = 3;

/// `sunshadow=on|title|own|off` in `XNW_WORLD_RENDER` (the last one wins; default off).
pub fn parse_switch(v: &str) -> u32 {
    match v.split(',').filter_map(|p| p.trim().strip_prefix("sunshadow=")).next_back() {
        Some("on") => AUTO,
        Some("title") => TITLE,
        Some("own") => OWN,
        _ => OFF,
    }
}

/// The switch as the environment asks it now.
pub fn asked() -> u32 {
    parse_switch(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default())
}

/// Live `sunshadow` (taken once a frame; start: asked).
pub fn live() -> u32 {
    xnw_core::live::value(xnw_core::live::SUNSHADOW, asked)
}

/// A cascade tile's side in texels; the atlas is 2 x 2 of them.
pub const TILE: u32 = 1024;
pub const CASCADES: usize = 4;
/// The title's cascade i is stage `FIRST_STAGE + i`.
pub const FIRST_STAGE: u32 = 23;
/// `on`: frames without a title cascade draw before ours take over.
pub const HOLD: u64 = 120;
/// The screen term's format (4-byte texels: read back as the other targets).
pub const TERM: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;
/// Our cascades' half widths in metres (`own`).
pub const OWN_HALF: [f32; 3] = [10.0, 40.0, 160.0];
/// The receiver's depth bias: metres, and texels of its cascade.
const BIAS_M: f32 = 0.03;
const BIAS_TEXELS: f32 = 1.5;
/// The casters' slope-scaled depth bias.
const SLOPE: f32 = 1.5;

/// A list row of a title cascade.
pub fn title_stage(stage: u32) -> bool {
    (FIRST_STAGE..FIRST_STAGE + CASCADES as u32).contains(&stage)
}

/// The caster's pixel entry, appended to the raw shader's source: nothing
/// but the alpha clip (one sample outside any branch, no return in a branch).
const CASTER_PS: &str = r#"
[RootSignature(RS)]
void ps_caster(VsOut i) {
    float a = albedoTex[min(texIndex, 4095)].Sample(wrapAniso, i.uv).a;
    if (texIndex != 0 && (matFlags & 1) != 0 && a < 0.5) discard;
}
"#;

/// The term pass. b0: the main camera's clip-to-world (camera-relative),
/// per cascade the camera-relative world to its clip, its bias (x) and
/// whether it holds a picture (y); misc.x the tile side. t0 our depth's
/// copy (reverse Z, 0: nothing drawn), t1 the atlas. Every load is at a
/// clamped texel outside any branch; one return.
pub const TERM_SHADER: &str = r#"
#define RS "CBV(b0, visibility=SHADER_VISIBILITY_PIXEL), DescriptorTable(SRV(t0, numDescriptors=2), visibility=SHADER_VISIBILITY_PIXEL)"
cbuffer K : register(b0) { row_major float4x4 clipToWorld; row_major float4x4 casc[4]; float4 bias[4]; float4 misc; };
Texture2D<float> depthIn : register(t0);
Texture2D<float> atlas : register(t1);
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
    depthIn.GetDimensions(w, h);
    int3 p = int3(min(uint2(i.pos.xy), uint2(max(w, 1), max(h, 1)) - 1), 0);
    float z = depthIn.Load(p);
    float2 ndc = (float2(p.xy) + 0.5) / float2(max(w, 1), max(h, 1)) * float2(2, -2) + float2(-1, 1);
    float4 q = mul(float4(ndc, z, 1), clipToWorld);
    float3 wp = q.xyz / (abs(q.w) > 1e-12 ? q.w : 1e-12);
    int pick = -1;
    [unroll] for (int k = 3; k >= 0; k--) {
        float4 ck = mul(float4(wp, 1), casc[k]);
        bool inside = bias[k].y > 0.5 && abs(ck.x) < 0.985 && abs(ck.y) < 0.985 && ck.z < 1.0;
        pick = inside ? k : pick;
    }
    int ki = max(pick, 0);
    float4 c = mul(float4(wp, 1), casc[ki]);
    int tile = int(misc.x);
    float2 t = (c.xy * float2(0.5, -0.5) + 0.5) * misc.x - 0.5;
    float2 fl = floor(t), fr = t - fl;
    int2 org = int2(ki & 1, ki >> 1) * tile;
    float zr = c.z - bias[ki].x;
    float sum = 0;
    [unroll] for (int y = -1; y <= 2; y++) {
        [unroll] for (int x = -1; x <= 2; x++) {
            int2 s = clamp(int2(fl) + int2(x, y), 0, tile - 1) + org;
            float d = atlas.Load(int3(s, 0));
            float wx = x == -1 ? 1 - fr.x : (x == 2 ? fr.x : 1);
            float wy = y == -1 ? 1 - fr.y : (y == 2 ? fr.y : 1);
            sum += wx * wy * (zr <= d ? 1.0 : 0.0);
        }
    }
    float lit = (pick < 0 || z <= 0) ? 1.0 : sum / 9.0;
    return float4(lit, (pick + 1) * 0.25, 0, 1);
}
"#;

/// A cascade: positions relative to `origin` (absolute) times `rot_proj`
/// are its clip positions (w 1, depth growing away from the light).
#[derive(Clone, Copy, Debug)]
pub struct Cascade {
    pub rot_proj: Mat,
    pub origin: [f32; 3],
    /// Ours ([`own_cascade`]), not a title viewport's.
    pub own: bool,
    /// The pass frame ([`Sun::tick`]) it was drawn in.
    pub drawn_at: u64,
}

impl Cascade {
    /// Metres per texel and clip depth per metre.
    pub fn scales(&self) -> (f32, f32) {
        let m = &self.rot_proj;
        let col = |j: usize| (m[j] * m[j] + m[4 + j] * m[4 + j] + m[8 + j] * m[8 + j]).sqrt();
        (2.0 / (col(0).max(1e-9) * TILE as f32), col(2))
    }

    /// Positions relative to `from` (absolute) to this cascade's clip.
    pub fn from_origin(&self, from: [f32; 3]) -> Mat {
        let mut t: Mat = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        for i in 0..3 {
            t[12 + i] = from[i] - self.origin[i];
        }
        mul(&t, &self.rot_proj)
    }
}

/// A cascade of ours: an orthographic view along `to_light` (unit, towards
/// the light) of the square of half width `half` metres centred `ahead`
/// (relative to `origin`), `reach` metres of depth either side of it.
pub fn own_cascade(to_light: [f32; 3], origin: [f32; 3], ahead: [f32; 3], half: f32, reach: f32) -> Cascade {
    let l = to_light;
    let up = if l[2].abs() > 0.99 { [1.0, 0.0, 0.0] } else { [0.0, 0.0, 1.0] };
    let cross = |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let norm = |a: [f32; 3]| {
        let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-9);
        [a[0] / n, a[1] / n, a[2] / n]
    };
    let x = norm(cross(up, l));
    let y = cross(l, x);
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let (sx, sz) = (1.0 / half, 0.5 / reach);
    // clip = ((p - ahead) . x / half, (p - ahead) . y / half, 0.5 - (p - ahead) . l / (2 reach), 1).
    let mut m = [0f32; 16];
    for i in 0..3 {
        (m[4 * i], m[4 * i + 1], m[4 * i + 2]) = (x[i] * sx, y[i] * sx, -l[i] * sz);
    }
    (m[12], m[13], m[14], m[15]) = (-dot(ahead, x) * sx, -dot(ahead, y) * sx, 0.5 + dot(ahead, l) * sz, 1.0);
    Cascade { rot_proj: m, origin, own: true, drawn_at: 0 }
}

/// One caster as recorded: the raw root's constants and tables.
struct Caster {
    slot: u64,
    t3: u64,
    ibv: D3D12_INDEX_BUFFER_VIEW,
    k: [u32; 58],
    indices: u32,
    instances: u32,
    arg: Option<u32>,
}

/// The frame's counts for the log line.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// Per cascade: rows asked, rows drawn.
    pub rows: [(u32, u32); CASCADES],
    /// Rows not drawn, all cascades: left out by the skin rules (skinned,
    /// ped), geometry not on the device yet (or not a raw one), instances
    /// with the device cull but no output of it.
    pub left: [u32; 3],
    /// Cascades the term pass reads, and how many of them are ours.
    pub valid: u32,
    pub own: u32,
    /// Main-view pre-skinned rows drawn into the title's cascades.
    pub peds: u32,
    pub record_ms: f64,
}

/// The pass's objects and the cascades held.
pub struct Sun {
    atlas: ID3D12Resource,
    dsv: ID3D12DescriptorHeap,
    depth_copy: ID3D12Resource,
    term: ID3D12Resource,
    term_rtv: ID3D12DescriptorHeap,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    caster_pso: ID3D12PipelineState,
    /// The term pass's constants: four 512-byte slices of one mapped upload buffer.
    cb: (ID3D12Resource, *mut u8),
    cb_va: u64,
    /// Heap slots: the depth copy at `slot`, the atlas at `slot + 1`.
    pub slot: u32,
    pub cascades: [Option<Cascade>; CASCADES],
    /// Frames of the pass so far, and the last with a title cascade draw.
    pub tick: u64,
    last_title: u64,
    /// The term pass has a cascade to read this frame.
    valid: bool,
    pub stats: Stats,
}

impl Sun {
    /// Made once by `Pass::new`: the atlas, the depth copy and the term
    /// (`width` x `height`), the term pass and the caster pipeline (the
    /// raw shader's `vs` with [`CASTER_PS`], on the raw root). The depth
    /// copy's view goes to `cpu(slot)`, the atlas's to `cpu(slot + 1)`.
    pub(crate) fn new(device: &ID3D12Device, raw_source: &str, raw_root: &ID3D12RootSignature, width: u32, height: u32, slot: u32, cpu: &dyn Fn(u32) -> D3D12_CPU_DESCRIPTOR_HANDLE) -> std::result::Result<Sun, String> {
        let e = |e: windows::core::Error| format!("sunshadow: {e}");
        let caster_src = format!("{raw_source}\n{CASTER_PS}");
        let (cvs, cps) = (compile_source(&caster_src, c"vs", c"vs_5_1")?, compile_source(&caster_src, c"ps_caster", c"ps_5_1")?);
        let (vs, ps) = (compile_source(TERM_SHADER, c"vs", c"vs_5_1")?, compile_source(TERM_SHADER, c"ps", c"ps_5_1")?);
        let code = |b: &ID3DBlob| D3D12_SHADER_BYTECODE { pShaderBytecode: blob_bytes(b).as_ptr().cast(), BytecodeLength: blob_bytes(b).len() };
        // SAFETY: object creation with descriptions and blobs that outlive each call.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, blob_bytes(&vs)).map_err(e)?;
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(root.clone())),
                VS: code(&vs),
                PS: code(&ps),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            desc.RTVFormats[0] = TERM;
            desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            let pso = device.CreateGraphicsPipelineState::<ID3D12PipelineState>(&desc);
            drop(ManuallyDrop::into_inner(desc.pRootSignature));
            let pso = pso.map_err(e)?;
            // Casters: depth only, LESS, no culling, no depth clip (a caster nearer the light than the cascade's near plane still casts).
            let cdesc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(raw_root.clone())),
                VS: code(&cvs),
                PS: code(&cps),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: false.into(), SlopeScaledDepthBias: SLOPE, ..Default::default() },
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC { DepthEnable: true.into(), DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL, DepthFunc: D3D12_COMPARISON_FUNC_LESS, ..Default::default() },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 0,
                DSVFormat: DXGI_FORMAT_D32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let caster_pso = device.CreateGraphicsPipelineState::<ID3D12PipelineState>(&cdesc);
            drop(ManuallyDrop::into_inner(cdesc.pRootSignature));
            let caster_pso = caster_pso.map_err(e)?;
            let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
            let tex = |w: u32, h: u32, format, flags, state, clear: Option<D3D12_CLEAR_VALUE>| -> std::result::Result<ID3D12Resource, String> {
                let d = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: u64::from(w), Height: h, DepthOrArraySize: 1, MipLevels: 1, Format: format, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Flags: flags, ..Default::default() };
                let mut out: Option<ID3D12Resource> = None;
                device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &d, state, clear.as_ref().map(std::ptr::from_ref), &mut out).map_err(e)?;
                out.ok_or_else(|| "sunshadow: no texture".to_string())
            };
            let far = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_D32_FLOAT, Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 } } };
            let atlas = tex(2 * TILE, 2 * TILE, DXGI_FORMAT_R32_TYPELESS, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_DEPTH_WRITE, Some(far))?;
            let depth_copy = tex(width, height, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, None)?;
            let white = D3D12_CLEAR_VALUE { Format: TERM, Anonymous: D3D12_CLEAR_VALUE_0 { Color: [1.0; 4] } };
            let term = tex(width, height, TERM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, Some(white))?;
            let small = |kind| device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC { Type: kind, NumDescriptors: 1, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE, NodeMask: 0 });
            let dsv = small(D3D12_DESCRIPTOR_HEAP_TYPE_DSV).map_err(e)?;
            let term_rtv = small(D3D12_DESCRIPTOR_HEAP_TYPE_RTV).map_err(e)?;
            let dview = D3D12_DEPTH_STENCIL_VIEW_DESC { Format: DXGI_FORMAT_D32_FLOAT, ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D, ..Default::default() };
            device.CreateDepthStencilView(&atlas, Some(&dview), dsv.GetCPUDescriptorHandleForHeapStart());
            device.CreateRenderTargetView(&term, None, term_rtv.GetCPUDescriptorHandleForHeapStart());
            let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_FLOAT,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } },
            };
            device.CreateShaderResourceView(&depth_copy, Some(&view), cpu(slot));
            device.CreateShaderResourceView(&atlas, Some(&view), cpu(slot + 1));
            let cb = upload_buffer(device, &[0u8; 2048]).map_err(e)?;
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
            cb.Map(0, None, Some(&mut p)).map_err(e)?;
            Ok(Sun { atlas, dsv, depth_copy, term, term_rtv, root, pso, caster_pso, cb: (cb, p.cast()), cb_va: 0, slot, cascades: [None; CASCADES], tick: 0, last_title: 0, valid: false, stats: Stats::default() })
        }
    }

    /// The screen term (shader-resource state between passes): our
    /// lighting pass's view of it is made by `ourlight::Lit::new`.
    pub fn term(&self) -> &ID3D12Resource {
        &self.term
    }

    /// The depth atlas (depth-write state between passes), 2 x [`TILE`] square.
    pub fn atlas(&self) -> &ID3D12Resource {
        &self.atlas
    }

    /// The term pass on `l`, before our lighting pass: our depth copied,
    /// the term written (cleared to 1 without a cascade), left in the
    /// shader-resource state. The list's targets are left unbound.
    /// `table`: the heap's handle at [`Sun::slot`].
    pub(crate) fn record_term(&self, l: &ID3D12GraphicsCommandList, depth: &ID3D12Resource, table: D3D12_GPU_DESCRIPTOR_HANDLE) {
        // SAFETY: an open list; every resource lives with the pass.
        unsafe {
            let rtv = self.term_rtv.GetCPUDescriptorHandleForHeapStart();
            transition(l, &self.term, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            if self.valid {
                transition(l, depth, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
                transition(l, &self.depth_copy, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
                l.CopyResource(&self.depth_copy, depth);
                transition(l, &self.depth_copy, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
                transition(l, depth, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_DEPTH_WRITE);
                transition(l, &self.atlas, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
                l.OMSetRenderTargets(1, Some(&rtv), true, None);
                l.SetGraphicsRootSignature(&self.root);
                l.SetPipelineState(&self.pso);
                l.SetGraphicsRootConstantBufferView(0, self.cb_va);
                l.SetGraphicsRootDescriptorTable(1, table);
                l.DrawInstanced(3, 1, 0, 0);
                transition(l, &self.atlas, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_DEPTH_WRITE);
            } else {
                l.ClearRenderTargetView(rtv, &[1.0; 4], None);
            }
            transition(l, &self.term, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        }
    }

    /// The term pass's constants for this frame: `clip_to_world` and
    /// `main_pos` are the main camera's.
    fn constants(&mut self, clip_to_world: &Mat, main_pos: [f32; 3]) {
        let mut k = [0f32; 100];
        k[..16].copy_from_slice(clip_to_world);
        let (mut valid, mut own) = (0u32, 0u32);
        for (i, c) in self.cascades.iter().enumerate() {
            let Some(c) = c else { continue };
            k[16 + 16 * i..32 + 16 * i].copy_from_slice(&c.from_origin(main_pos));
            let (texel, per_m) = c.scales();
            (k[80 + 4 * i], k[81 + 4 * i]) = ((BIAS_M + BIAS_TEXELS * texel) * per_m, 1.0);
            valid += 1;
            own += u32::from(c.own);
        }
        k[96] = TILE as f32;
        (self.stats.valid, self.stats.own, self.valid) = (valid, own, valid > 0);
        let at = (self.tick % 4) as usize * 512;
        // SAFETY: 400 bytes into a 512-byte slice of the 2048-byte mapping; a live buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(k.as_ptr().cast::<u8>(), self.cb.1.add(at), 400);
            self.cb_va = self.cb.0.GetGPUVirtualAddress() + at as u64;
        }
    }

    /// The frame's log line.
    pub fn line(&self, mode: u32) -> String {
        let name = ["off", "on", "title", "own"][mode.min(3) as usize];
        let rows: Vec<String> = (0..CASCADES)
            .map(|i| match &self.cascades[i] {
                Some(c) => format!("{} {}/{} rows{} ({:.1} cm/texel)", i, self.stats.rows[i].1, self.stats.rows[i].0, if c.drawn_at == self.tick { String::new() } else { format!(" held {} frames", self.tick - c.drawn_at) }, c.scales().0 * 100.0),
                None => format!("{i} none"),
            })
            .collect();
        format!("sunshadow {name}: {} cascades ({} ours): {}; rows left out: skin rules {}, geometry not on the device {}, instances not in the frame {}; main-view pre-skinned rows {}; record {:.2} ms", self.stats.valid, self.stats.own, rows.join(", "), self.stats.left[0], self.stats.left[1], self.stats.left[2], self.stats.peds, self.stats.record_ms)
    }
}

/// The list's cascade rows whose geometry the pass should hold (`on`, `title`).
pub fn title_rows(list: &FrameList) -> impl Iterator<Item = &DrawRecord> {
    list.draws.iter().filter(|r| title_stage(r.stage))
}

impl Pass {
    /// `sunshadow`: the pass has its objects and the live switch is on.
    pub(super) fn sunshadow_mode(&self) -> u32 {
        if self.sunshadow.is_some() && !self.id_mode { live() } else { OFF }
    }

    /// A row as a caster: `mvp` takes its vertex program's positions to
    /// the cascade's clip. Err: 1 not resident or not a raw geometry, 2
    /// its instances are not in this frame's buffers ([`Stats::left`]).
    fn caster(&mut self, f: &precord::Frame, d: &DrawRecord, idx: u32, mvp: &Mat) -> std::result::Result<Caster, usize> {
        let list = f.list;
        let (ibv, indices, raw) = self.cache.get_or_make(d.geom, list.frame, || None).map(|g| (g.ibv, g.indices, g.raw)).ok_or(1usize)?;
        let (slot, vb_bytes) = raw.filter(|_| indices != 0).ok_or(1usize)?;
        let (g0, gs) = (self.gpu_slot(0).ptr, self.gpu_slot(1).ptr - self.gpu_slot(0).ptr);
        let gpu = move |s: u32| g0 + u64::from(s) * gs;
        let mut k = [0u32; 58];
        for (i, v) in mvp.iter().enumerate() {
            k[i] = v.to_bits();
        }
        // As the pass's own rows (precord.rs `resolve`): palette, remap, instances, pre-skin copy.
        let (pf, pb) = match list.palettes.get(d.palette as usize) {
            Some(p) if d.palette != NO_PALETTE && self.skin => match d.bone_count {
                0 => (f.first[d.palette as usize], (p.len() / 12) as u32),
                n => (f.first[d.palette as usize] + 3 * d.bone_first, n),
            },
            _ => (0, 0),
        };
        let (ti, _) = self.tex.lookup(d.material.albedo_desc);
        (k[32], k[33], k[34], k[36]) = (vb_bytes, pf, pb, f.pal_bytes);
        k[35] = u32::from(d.ped) << 1 | (self.skin_rules & 0xFF) << 8;
        (k[48], k[49]) = (ti.min(TEX_SLOTS - 1), u32::from(d.material.alpha_clip));
        k[56] = match list.palettes.get(d.palette as usize) {
            Some(_) if d.remap != NO_REMAP && pb > 0 => 4 * f.first[d.palette as usize] + d.remap,
            _ => u32::MAX,
        };
        let first = d.inst_first.min(f.inst_total);
        let count = d.inst_count.min(f.inst_total - first).min(MAX_INSTANCES);
        (k[37], k[38], k[39]) = (first, count, f.inst_bytes);
        let cull_slot = f.culled.and_then(|c| c.slot_of(idx));
        let mut t3 = gpu(2 + f.turn as u32);
        match (f.culled, cull_slot) {
            (Some(c), Some(_)) => {
                (k[37], k[38], k[39]) = (d.inst_first, d.inst_count.min(MAX_INSTANCES), c.out_bytes.min(u64::from(u32::MAX)) as u32);
                t3 = gpu(c.view_slot);
            }
            // Its instances are the device cull's, and the cull has no output for it.
            _ if f.device_rows.contains(&idx) => return Err(2),
            _ => {}
        }
        let slot = match f.pre.get(&idx) {
            Some(&(s, bytes)) => {
                (k[32], k[34], k[56]) = (bytes, 0, u32::MAX);
                s
            }
            None => slot,
        };
        Ok(Caster { slot: gpu(slot), t3, ibv, k, indices, instances: count.max(1), arg: cull_slot.filter(|_| f.culled.is_some()) })
    }

    /// The frame's cascades on the open list `l` (before the pass's rows):
    /// the title's cascade rows, or ours from `draws` (the main view's
    /// rows as the pass draws them); then the term pass's constants. The
    /// frame's `rts` targets, depth, viewport and scissor are bound again.
    pub(super) fn record_sunshadow(&mut self, l: &ID3D12GraphicsCommandList, f: &precord::Frame, draws: &[&DrawRecord], rts: u32) {
        let mode = self.sunshadow_mode();
        let Some(mut sun) = self.sunshadow.take() else { return };
        let t0 = std::time::Instant::now();
        sun.tick += 1;
        sun.stats = Stats::default();
        if mode == OFF {
            sun.valid = false;
            self.sunshadow = Some(sun);
            return;
        }
        let list = f.list;
        let main_pos = camera_pos(&f.cam.view);
        let index = |d: &DrawRecord| ((d as *const DrawRecord as usize).wrapping_sub(list.draws.as_ptr() as usize) / std::mem::size_of::<DrawRecord>()) as u32;
        let (peds, skinned, rules) = (self.peds, self.skinned, self.skin_rules);
        // Ped cascade rows only without the pre-skin (with it the main view's pre-skinned rows cast below).
        let pre_on = !f.pre.is_empty();
        let casts = |d: &DrawRecord| (!d.ped || (peds && !pre_on)) && (skinned || d.ped || d.palette == NO_PALETTE || d.bone_count <= 1 || skin_lifted(list, d, rules));
        let mut jobs: Vec<(usize, Vec<Caster>)> = Vec::new();
        if mode != OWN {
            for k in 0..CASCADES {
                let stage = FIRST_STAGE + k as u32;
                let rows: Vec<(u32, &DrawRecord)> = list.draws.iter().enumerate().filter(|(_, d)| d.stage == stage).map(|(i, d)| (i as u32, d)).collect();
                // The stage's viewport: orthographic (its rows share it).
                let Some(cam) = rows.first().and_then(|(_, d)| list.cameras.iter().find(|c| c.viewport == d.viewport)).filter(|c| c.proj[11] == 0.0 && c.proj[15] != 0.0) else { continue };
                let c = Cascade { rot_proj: mul(&rotation_only(&cam.view), &cam.proj), origin: camera_pos(&cam.view), own: false, drawn_at: sun.tick };
                let mut casters = Vec::with_capacity(rows.len());
                for (idx, d) in &rows {
                    if d.viewport != cam.viewport || !casts(d) {
                        sun.stats.left[0] += 1;
                        continue;
                    }
                    match self.caster(f, d, *idx, &mul(&d.world, &c.rot_proj)) {
                        Ok(x) => casters.push(x),
                        Err(why) => sun.stats.left[why] += 1,
                    }
                }
                sun.stats.rows[k] = (rows.len() as u32, casters.len() as u32);
                // The main view's pre-skinned rows (peds on screen), placed from the main camera.
                let from_main = c.from_origin(main_pos);
                for d in draws {
                    let idx = index(d);
                    if !DECAL_STAGES.contains(&d.stage)
                        && f.pre.contains_key(&idx)
                        && let Ok(x) = self.caster(f, d, idx, &mul(&d.world, &from_main))
                    {
                        casters.push(x);
                        sun.stats.peds += 1;
                    }
                }
                // A title cascade drawn: ours go.
                if sun.cascades.iter().flatten().any(|c| c.own) {
                    sun.cascades = [None; CASCADES];
                }
                sun.cascades[k] = Some(c);
                sun.last_title = sun.tick;
                jobs.push((k, casters));
            }
        }
        let title_held = sun.cascades.iter().flatten().any(|c| !c.own) && (mode == TITLE || sun.tick - sun.last_title <= HOLD);
        let light = list.light.map(|x| x.dir).or(list.sun.map(|s| [s[0], s[1], s[2]])).filter(|d| d.iter().all(|v| v.is_finite()) && d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > 0.25);
        if mode == TITLE {
            for c in &mut sun.cascades {
                *c = c.filter(|c| !c.own);
            }
        } else if let Some(dir) = light.filter(|_| !title_held) {
            let n = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            let dir = [dir[0] / n, dir[1] / n, dir[2] / n];
            // The main camera looks down -z of its view: the cascades sit ahead of it.
            let r = rotation_only(&f.cam.view);
            let fwd = [-r[2], -r[6], -r[10]];
            sun.cascades = [None; CASCADES];
            let rows: Vec<(u32, &DrawRecord)> = draws.iter().map(|d| (index(d), *d)).filter(|(idx, d)| !DECAL_STAGES.contains(&d.stage) && !TRANSPARENT_STAGES.contains(&d.stage) && f.ms_rows.is_none_or(|b| b.item(*idx).is_none())).collect();
            for (k, half) in OWN_HALF.iter().enumerate() {
                let ahead = [fwd[0] * half * 0.7, fwd[1] * half * 0.7, fwd[2] * half * 0.7];
                let mut c = own_cascade(dir, main_pos, ahead, *half, 2.0 * half + 150.0);
                c.drawn_at = sun.tick;
                let mut casters = Vec::with_capacity(rows.len());
                for (idx, d) in &rows {
                    match self.caster(f, d, *idx, &mul(&d.world, &c.rot_proj)) {
                        Ok(x) => casters.push(x),
                        Err(why) => sun.stats.left[why] += 1,
                    }
                }
                sun.stats.rows[k] = (rows.len() as u32, casters.len() as u32);
                sun.cascades[k] = Some(c);
                jobs.push((k, casters));
            }
        } else if !title_held {
            sun.cascades = [None; CASCADES];
        }
        // SAFETY: recording on our open list; the frame's buffers and the geometry cache live until its fence.
        unsafe {
            // The whole atlas once (a tile no cascade draws is never read, but the readback shows it).
            if sun.tick == 1 {
                l.ClearDepthStencilView(sun.dsv.GetCPUDescriptorHandleForHeapStart(), D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
            }
            if !jobs.is_empty() {
                let dsv = sun.dsv.GetCPUDescriptorHandleForHeapStart();
                l.OMSetRenderTargets(0, None, false, Some(&dsv));
                l.SetGraphicsRootSignature(&self.raw_root);
                l.SetPipelineState(&sun.caster_pso);
                l.SetGraphicsRootDescriptorTable(2, self.gpu_slot(f.turn as u32));
                l.SetGraphicsRootDescriptorTable(4, f.tex_table);
                l.SetGraphicsRootConstantBufferView(5, f.light_va);
                for (k, casters) in &jobs {
                    let (x, y) = ((*k as u32 & 1) * TILE, (*k as u32 >> 1) * TILE);
                    let rect = windows::Win32::Foundation::RECT { left: x as i32, top: y as i32, right: (x + TILE) as i32, bottom: (y + TILE) as i32 };
                    l.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: x as f32, TopLeftY: y as f32, Width: TILE as f32, Height: TILE as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
                    l.RSSetScissorRects(&[rect]);
                    l.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, Some(&[rect]));
                    let mut t3 = 0u64;
                    for c in casters {
                        if c.t3 != t3 {
                            l.SetGraphicsRootDescriptorTable(3, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: c.t3 });
                            t3 = c.t3;
                        }
                        l.IASetIndexBuffer(Some(&c.ibv));
                        l.SetGraphicsRoot32BitConstants(0, 58, c.k.as_ptr().cast(), 0);
                        l.SetGraphicsRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: c.slot });
                        match (c.arg, f.culled) {
                            (Some(a), Some(cull)) => l.ExecuteIndirect(&cull.signature, 1, &cull.args, u64::from(a) * u64::from(gpucull::ARG_BYTES), None, 0),
                            _ => l.DrawIndexedInstanced(c.indices, c.instances, 0, 0, 0),
                        }
                    }
                }
                let (rtv, dsv) = (self.rtv.GetCPUDescriptorHandleForHeapStart(), self.dsv.GetCPUDescriptorHandleForHeapStart());
                l.OMSetRenderTargets(rts, Some(&rtv), true, Some(&dsv));
                l.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: self.width as f32, Height: self.height as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
                l.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: self.width as i32, bottom: self.height as i32 }]);
            }
        }
        sun.constants(f.clip_to_world, main_pos);
        sun.stats.record_ms = t0.elapsed().as_secs_f64() * 1e3;
        static EVERY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let every = *EVERY.get_or_init(|| std::env::var("XNW_SUNSHADOW_LOG").is_ok_and(|v| v.trim() == "every"));
        if every || sun.tick <= 2 || sun.tick % 60 == 0 || !jobs.is_empty() && sun.cascades.iter().flatten().any(|c| !c.own && c.drawn_at == sun.tick) {
            xnw_core::info!("world render: frame {}: {}", list.frame, sun.line(mode));
        }
        self.sunshadow = Some(sun);
    }

    /// `sunshadow`: the screen term read back (after a [`Pass::draw`]):
    /// RGBA8 rows, .r the lit fraction, .g (cascade + 1) / 4. None: off.
    pub fn read_sunshadow(&self) -> Option<std::result::Result<Vec<u8>, String>> {
        let s = self.sunshadow.as_ref()?;
        Some(gbuf::read_targets(&self.device, &self.queue, &[(&s.term, TERM, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)], self.width, self.height).map(|mut v| v.pop().unwrap_or_default()))
    }

    /// `sunshadow`: the depth atlas read back (after a [`Pass::draw`]): 2 x
    /// [`TILE`] square rows of f32, 1.0 where no caster drew. None: off.
    pub fn read_sunshadow_atlas(&self) -> Option<std::result::Result<Vec<f32>, String>> {
        let s = self.sunshadow.as_ref()?;
        Some(gbuf::read_targets(&self.device, &self.queue, &[(&s.atlas, DXGI_FORMAT_R32_TYPELESS, D3D12_RESOURCE_STATE_DEPTH_WRITE)], 2 * TILE, 2 * TILE).map(|mut v| v.pop().unwrap_or_default().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switch_last_wins() {
        assert_eq!(parse_switch("screen,sunshadow=on"), AUTO);
        assert_eq!(parse_switch("screen,sunshadow=on,sunshadow=own"), OWN);
        assert_eq!(parse_switch("screen,sunshadow=title"), TITLE);
        assert_eq!(parse_switch("screen,sunshadow=off"), OFF);
        assert_eq!(parse_switch("screen"), OFF);
    }

    /// Our cascade: its centre lands at clip (0, 0, 0.5); a point towards
    /// the light is nearer (smaller depth); the square's edge is at |x| = 1.
    #[test]
    fn own_cascade_maps_the_square() {
        let l = [0.0, 0.6, 0.8];
        let c = own_cascade(l, [100.0, 200.0, 5.0], [3.0, 4.0, 0.0], 10.0, 50.0);
        let clip = |p: [f32; 3]| {
            let m = &c.rot_proj;
            [p[0] * m[0] + p[1] * m[4] + p[2] * m[8] + m[12], p[0] * m[1] + p[1] * m[5] + p[2] * m[9] + m[13], p[0] * m[2] + p[1] * m[6] + p[2] * m[10] + m[14]]
        };
        let o = clip([3.0, 4.0, 0.0]);
        assert!(o[0].abs() < 1e-5 && o[1].abs() < 1e-5 && (o[2] - 0.5).abs() < 1e-5, "{o:?}");
        let n = clip([3.0, 4.0 + 6.0, 8.0]);
        assert!(n[0].abs() < 1e-4 && n[1].abs() < 1e-4 && (n[2] - 0.4).abs() < 1e-5, "{n:?}");
        // x axis = up x l = (-0.6, 0, 0) normalised: 10 m along -x is the edge.
        let e = clip([3.0 - 10.0, 4.0, 0.0]);
        assert!((e[0] - 1.0).abs() < 1e-5 && e[1].abs() < 1e-5, "{e:?}");
        let (texel, per_m) = c.scales();
        assert!((texel - 20.0 / TILE as f32).abs() < 1e-6 && (per_m - 0.01).abs() < 1e-7);
    }

    /// A title cascade read from another origin: the offset is applied before its rotation.
    #[test]
    fn from_origin_moves_the_point() {
        let c = own_cascade([0.0, 0.0, 1.0], [10.0, 20.0, 30.0], [0.0; 3], 5.0, 50.0);
        let m = c.from_origin([11.0, 20.0, 30.0]);
        // The other origin itself (0 relative to it) is 1 m along +x of the cascade's origin.
        let (x, y) = (m[12], m[13]);
        let direct = [c.rot_proj[0] + c.rot_proj[12], c.rot_proj[1] + c.rot_proj[13]];
        assert!((x - direct[0]).abs() < 1e-6 && (y - direct[1]).abs() < 1e-6);
    }

    #[test]
    fn shaders_compile() {
        for entry in [(c"vs", c"vs_5_1"), (c"ps", c"ps_5_1")] {
            compile_source(TERM_SHADER, entry.0, entry.1).unwrap();
        }
        let caster = format!("#define OURLIGHT 1\n#define GBUF 1\n{RAW_SHADER}\n{CASTER_PS}");
        compile_source(&caster, c"ps_caster", c"ps_5_1").unwrap();
    }
}
