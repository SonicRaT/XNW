//! `PresentX` on the host: every plane's host texture, composited into a
//! flip-model swap chain on the title's own window. One plane is copied
//! to the back buffer (the fast path); two or more (Series S: a 1920x1080
//! scene and a 3840x2160 UI plane) are drawn by a full-screen-triangle
//! compositor, plane 0 opaque and the others blended over it
//! (premultiplied alpha), each into its destination rectangle of a back
//! buffer as large as the largest plane.
//!
//! The window is the title's own (its `CreateWindowExA`: the visible
//! top-level window of this process), sized to the plane's aspect within
//! the desktop, without taking focus; its caption gets the frame rate
//! appended. Only if the title made none do we open ours (closing that one
//! ends the run). Presents with the frame clock's sync interval
//! (`frame.rs`: 2 for a 30 fps title at 60 Hz), so every title frame stays
//! the same number of refreshes; three buffers, so a frame waiting for its
//! refresh does not hold the next one's back buffer (`XNW_PACING=timer`:
//! `Present(0, 0)` as before).
//! `XNW_CAPTURE=<dir>` saves the host image shown (`capture.rs`).
//! `XNW_PRESENT_TEST=1` (diagnostic) fills each plane's host texture with
//! a gradient of its own before compositing, to see the planes' placement.
//!
//! Ported from VIctory-base (VIctory main d2a4bae + 26 PRs, branch
//! `xnw/base` @ 5d0516e, GPL-3.0-or-later), `crates/victory-gpu/src/
//! host.rs` lines 1616-1781 (`swap_format`, `Compositor`, from upstream)
//! and 1782-1792, 1909-2155 (`Presenter`, `Host::present`,
//! `Host::presenter`); PRs touching them: #211, #413, #638. Changed:
//! - planes are placed by their destination rectangle (the base stretches
//!   every plane over the whole back buffer; the plan's "base flaw to
//!   fix"): one viewport per plane; the back buffer is cleared first when
//!   plane 0 does not cover it; the fast path only for one full plane;
//! - the window, its fitting to the desktop and the caption meter are ours
//!   (the base's `Meter` reads counters our frame line takes);
//! - captures in every build, from the back buffer or the one plane's
//!   host texture (the base's diagnostics path);
//! - `XNW_PRESENT_TEST` and the composited-planes log lines are ours.
//!
//! shortcut: source rectangles are not applied (logged once when not the
//! whole texture); no color space, scale filter or HDR.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use windows::Win32::Foundation::{HWND, RECT};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::{Interface, PCSTR, s};

use super::host::{Host, HostList, transition};
use super::queue::Plane;
use std::time::Instant;

/// `XNW_PRESENT`: 0 normal; 1 `off` (no swap chain Present and no
/// blit: frames still count, readbacks come from plane 0's texture); 2
/// `noblit` (Present of a cleared back buffer); 3 `sync0` (sync interval
/// 0, with tearing where the factory allows it).
fn present_mode() -> u8 {
    static MODE: std::sync::LazyLock<u8> = std::sync::LazyLock::new(|| {
        let m = match std::env::var("XNW_PRESENT").as_deref().map(str::trim) {
            Ok("off") => 1,
            Ok("noblit") => 2,
            Ok("sync0") => 3,
            Ok(other) if !other.is_empty() => {
                xnw_core::warn!("XNW_PRESENT={other}: not off, noblit or sync0; ignored");
                0
            }
            _ => 0,
        };
        if m != 0 {
            xnw_core::warn!("present: XNW_PRESENT={} (diagnostic)", ["", "off", "noblit", "sync0"][m as usize]);
        }
        m
    });
    *MODE
}

/// `XNW_READBACK=off`: no per-frame luma or capture readback.
fn readback_on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| !std::env::var("XNW_READBACK").is_ok_and(|v| v.trim() == "off"));
    *ON
}

/// Whether frame `frame` wants a readback (luma, capture, world frame).
fn readback_wanted(frame: u64) -> bool {
    readback_on() && (super::capture::wanted(frame) || super::capture::luma_on() || super::capture::world_frame_wanted(frame))
}

/// Records on `list` a readback of `texture` (in `state`), moved to copy
/// source and back.
unsafe fn read_back_plane(device: &ID3D12Device, list: &ID3D12GraphicsCommandList, texture: &ID3D12Resource, state: u32) -> Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    let state = D3D12_RESOURCE_STATES(state as i32);
    let moved = state != D3D12_RESOURCE_STATE_COPY_SOURCE;
    // SAFETY: an open list and a live texture (the caller's).
    unsafe {
        if moved {
            list.ResourceBarrier(&[transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        }
        let capture = super::capture::read_back(device, list, texture);
        if moved {
            list.ResourceBarrier(&[transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state)]);
        }
        capture
    }
}

/// Saves or measures a readback taken for `frame`.
fn use_read_back(frame: u64, capture: Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)>) {
    if let Some((buffer, footprint)) = capture {
        if super::capture::world_frame_wanted(frame) {
            super::capture::save_world_frame(frame, &buffer, &footprint);
        }
        let flash = super::capture::luma_on() && super::capture::log_luma(frame, &buffer, &footprint);
        if flash || super::capture::wanted(frame) {
            super::capture::save_read_back(frame, &buffer, &footprint);
        }
    }
}

/// A plane's normalized destination rectangle `[left, top, right, bottom]`
/// in the back buffer (`pDestPlacementBase`; the title passes
/// {0, 0, 1, 1}, five-star d3d12x-shim.md "PresentX").
pub(crate) type Placement = [f32; 4];
pub(crate) const FULL: Placement = [0.0, 0.0, 1.0, 1.0];

fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

/// A shader-resource format for showing `format` (typeless and depth
/// formats as their readable form; `XNW_SHOW`).
fn view_format(format: DXGI_FORMAT) -> DXGI_FORMAT {
    match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        1 | 2 => DXGI_FORMAT_R32G32B32A32_FLOAT,
        15 | 16 => DXGI_FORMAT_R32G32_FLOAT,
        19..=22 => DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
        33 | 34 => DXGI_FORMAT_R16G16_FLOAT,
        39..=41 => DXGI_FORMAT_R32_FLOAT,
        44..=47 => DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
        48..=52 => DXGI_FORMAT_R8G8_UNORM,
        53 | 54 => DXGI_FORMAT_R16_FLOAT,
        55 | 56 => DXGI_FORMAT_R16_UNORM,
        60..=64 => DXGI_FORMAT_R8_UNORM,
        _ => format,
    }
}

const BUFFERS: u32 = 3;
const MAX_PLANES: u32 = 8;

/// The viewport of a plane placed at `place` in a `width` x `height` target.
fn viewport(place: &Placement, width: u32, height: u32) -> D3D12_VIEWPORT {
    let (w, h) = (width as f32, height as f32);
    D3D12_VIEWPORT { TopLeftX: place[0] * w, TopLeftY: place[1] * h, Width: (place[2] - place[0]) * w, Height: (place[3] - place[1]) * h, MinDepth: 0.0, MaxDepth: 1.0 }
}

/// Built-in fullscreen-triangle compositor for multi-plane `PresentX` calls
/// (e.g. Series S presenting a 1080p scene plane with a 4K UI overlay plane).
pub(crate) struct Compositor {
    format: DXGI_FORMAT,
    root: ID3D12RootSignature,
    opaque: ID3D12PipelineState,
    overlay: ID3D12PipelineState,
    rtv_heap: ID3D12DescriptorHeap,
    srv_heap: ID3D12DescriptorHeap,
    srv_stride: u32,
}

impl Compositor {
    fn new(device: &ID3D12Device, format: DXGI_FORMAT) -> windows::core::Result<Self> {
        const SOURCE: &str = r#"
            #define RS "DescriptorTable(SRV(t0), visibility=SHADER_VISIBILITY_PIXEL), \
                        StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, \
                            addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP, \
                            visibility=SHADER_VISIBILITY_PIXEL)"
            Texture2D tex : register(t0);
            SamplerState smp : register(s0);
            [RootSignature(RS)]
            void vs(uint id : SV_VertexID, out float4 pos : SV_Position, out float2 uv : TEXCOORD) {
                uv = float2((id << 1) & 2, id & 2);
                pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
            }
            [RootSignature(RS)]
            float4 ps(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target {
                return tex.Sample(smp, uv);
            }
        "#;
        let compile = |entry: PCSTR, target: PCSTR| -> windows::core::Result<ID3DBlob> {
            let mut code: Option<ID3DBlob> = None;
            let mut errors: Option<ID3DBlob> = None;
            // SAFETY: static source and entry/target strings; out pointers ours.
            let res = unsafe { D3DCompile(SOURCE.as_ptr().cast(), SOURCE.len(), s!("present_composite"), None, None, entry, target, 0, 0, &mut code, Some(&mut errors)) };
            if let Err(e) = res {
                if let Some(errors) = errors {
                    // SAFETY: the blob's own bytes.
                    let text = unsafe { std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize()) };
                    xnw_core::warn!("present compositor: {}", String::from_utf8_lossy(text));
                }
                return Err(e);
            }
            code.ok_or_else(windows::core::Error::empty)
        };
        let vs = compile(s!("vs"), s!("vs_5_1"))?;
        let ps = compile(s!("ps"), s!("ps_5_1"))?;
        // SAFETY: compiled shader blobs live through PSO creation.
        unsafe {
            let vs_bytes = std::slice::from_raw_parts(vs.GetBufferPointer().cast::<u8>(), vs.GetBufferSize());
            let ps_bytes = std::slice::from_raw_parts(ps.GetBufferPointer().cast::<u8>(), ps.GetBufferSize());
            let root: ID3D12RootSignature = device.CreateRootSignature(0, vs_bytes)?;
            let make_pso = |blend: bool| -> windows::core::Result<ID3D12PipelineState> {
                let mut rt_blend = D3D12_RENDER_TARGET_BLEND_DESC { RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8, ..Default::default() };
                if blend {
                    rt_blend.BlendEnable = true.into();
                    rt_blend.SrcBlend = D3D12_BLEND_ONE;
                    rt_blend.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
                    rt_blend.BlendOp = D3D12_BLEND_OP_ADD;
                    rt_blend.SrcBlendAlpha = D3D12_BLEND_ONE;
                    rt_blend.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
                    rt_blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
                }
                let mut blend_desc = D3D12_BLEND_DESC::default();
                blend_desc.RenderTarget[0] = rt_blend;
                let mut rtv_formats = [DXGI_FORMAT_UNKNOWN; 8];
                rtv_formats[0] = format;
                let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(root.clone())),
                    VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs_bytes.as_ptr().cast(), BytecodeLength: vs_bytes.len() },
                    PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps_bytes.as_ptr().cast(), BytecodeLength: ps_bytes.len() },
                    BlendState: blend_desc,
                    SampleMask: u32::MAX,
                    RasterizerState: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: false.into(), ..Default::default() },
                    PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                    NumRenderTargets: 1,
                    RTVFormats: rtv_formats,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    ..Default::default()
                };
                let pso = device.CreateGraphicsPipelineState(&desc);
                let _ = std::mem::ManuallyDrop::into_inner(desc.pRootSignature);
                pso
            };
            let opaque = make_pso(false)?;
            let overlay = make_pso(true)?;
            let rtv_heap: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE, NodeMask: 0 })?;
            let srv_heap: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: MAX_PLANES, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 })?;
            let srv_stride = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            Ok(Compositor { format, root, opaque, overlay, rtv_heap, srv_heap, srv_stride })
        }
    }

    /// Records `planes` (host texture, its host state, destination) into
    /// `target`, a `width` x `height` texture of the compositor's format in
    /// state `from`, which it leaves a render target: plane 0 opaque, the
    /// others over it, each in its destination rectangle. The planes are
    /// left in their states. (Base `Host::present`, its multi-plane branch.)
    ///
    /// # Safety
    /// An open list; live textures in the states given.
    unsafe fn composite(&self, device: &ID3D12Device, list: &ID3D12GraphicsCommandList, target: &ID3D12Resource, from: D3D12_RESOURCE_STATES, planes: &[(&ID3D12Resource, u32, Placement)], width: u32, height: u32) {
        // SAFETY: the caller's.
        unsafe {
            let format = self.format;
            let rtv = self.rtv_heap.GetCPUDescriptorHandleForHeapStart();
            device.CreateRenderTargetView(
                target,
                Some(&D3D12_RENDER_TARGET_VIEW_DESC {
                    Format: format,
                    ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                    Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_RTV { MipSlice: 0, PlaneSlice: 0 } },
                }),
                rtv,
            );
            let srv_cpu_base = self.srv_heap.GetCPUDescriptorHandleForHeapStart();
            let srv_gpu_base = self.srv_heap.GetGPUDescriptorHandleForHeapStart();
            let count = planes.len().min(MAX_PLANES as usize);
            let mut before = Vec::with_capacity(count + 1);
            for (i, &(tex, state, _)) in planes[..count].iter().enumerate() {
                let d = tex.GetDesc();
                let srv_fmt = view_format(d.Format);
                let handle = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: srv_cpu_base.ptr + i * self.srv_stride as usize };
                device.CreateShaderResourceView(
                    tex,
                    Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: srv_fmt,
                        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1, PlaneSlice: 0, ResourceMinLODClamp: 0.0 } },
                    }),
                    handle,
                );
                let state = D3D12_RESOURCE_STATES(state as i32);
                if state != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE {
                    before.push(transition(tex, state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
                }
            }
            if from != D3D12_RESOURCE_STATE_RENDER_TARGET {
                before.push(transition(target, from, D3D12_RESOURCE_STATE_RENDER_TARGET));
            }
            if !before.is_empty() {
                list.ResourceBarrier(&before);
            }
            // Ours: what plane 0 does not cover is black.
            if planes[0].2 != FULL {
                list.ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 1.0], None);
            }
            list.SetDescriptorHeaps(&[Some(self.srv_heap.clone())]);
            list.SetGraphicsRootSignature(&self.root);
            list.RSSetScissorRects(&[RECT { left: 0, top: 0, right: width as i32, bottom: height as i32 }]);
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            for (i, (_, _, place)) in planes[..count].iter().enumerate() {
                list.RSSetViewports(&[viewport(place, width, height)]);
                list.SetPipelineState(if i == 0 { &self.opaque } else { &self.overlay });
                list.SetGraphicsRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: srv_gpu_base.ptr + (i as u64) * u64::from(self.srv_stride) });
                list.DrawInstanced(3, 1, 0, 0);
            }
            let mut after = Vec::with_capacity(count);
            for &(tex, state, _) in &planes[..count] {
                let state = D3D12_RESOURCE_STATES(state as i32);
                if state != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE {
                    after.push(transition(tex, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, state));
                }
            }
            if !after.is_empty() {
                list.ResourceBarrier(&after);
            }
        }
    }
}

pub(crate) struct Presenter {
    swap_chain: IDXGISwapChain3,
    /// Its creation flags (tearing under `XNW_PRESENT=sync0`).
    flags: DXGI_SWAP_CHAIN_FLAG,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    /// The caption meter of the window (`hwnd` in the base).
    meter: Meter,
    compositor: Option<Compositor>,
    /// `XNW_PRESENT_TEST`: the gradient uploads, by plane and texture shape.
    patterns: Vec<((usize, u64, u32, i32), ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)>,
}

impl Host {
    /// Shows one or more `planes` (each a live host texture, its current
    /// host state, kept in that state, and its destination rectangle),
    /// compositing planes 1.. on top of plane 0 when multiple planes are
    /// presented, as frame `frame` (saved if captures are on). `None` if
    /// nothing was shown; else whether the frame was black (always `false`:
    /// the base's probe is a diagnostics-build feature).
    pub(crate) fn present(&self, planes: &[(&ID3D12Resource, u32, Placement)], frame: u64) -> Option<bool> {
        // XNW: view-table slots that waited for placed textures (no-op
        // unless XNW_DESC=gpu).
        super::descriptor::host::table_catch_up();
        let &(primary, _, _) = planes.first()?;
        // SAFETY: live resources.
        let primary_desc = unsafe { primary.GetDesc() };
        let show = super::draw::shown_target().is_some();
        let Some(format) = swap_format(primary_desc.Format).or(show.then_some(DXGI_FORMAT_R8G8B8A8_UNORM)) else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| xnw_core::warn!("present: cannot present format {}", primary_desc.Format.0));
            return None;
        };
        let (mut width, mut height) = (primary_desc.Width as u32, primary_desc.Height);
        for &(tex, _, _) in &planes[1..] {
            // SAFETY: live resource.
            let d = unsafe { tex.GetDesc() };
            width = width.max(d.Width as u32);
            height = height.max(d.Height);
        }
        // `XNW_PRESENT=off`: no swap chain at all; a readback (if
        // wanted) from plane 0's texture on a list of its own.
        if present_mode() == 1 {
            super::crumbs::presenting(1, frame);
            if readback_wanted(frame) {
                static LIST: std::sync::OnceLock<Option<HostList>> = std::sync::OnceLock::new();
                let list = LIST
                    .get_or_init(|| {
                        let l = self.new_list().ok()?;
                        l.close();
                        Some(l)
                    })
                    .as_ref()?;
                list.reset();
                let (texture, state, _) = planes[0];
                // SAFETY: an open list and a live texture.
                let capture = unsafe { read_back_plane(&self.device, &list.list, texture, state) };
                list.close();
                self.execute(&[list]);
                use_read_back(frame, capture);
            }
            super::crumbs::presenting(0, frame);
            return Some(false);
        }
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        if presenter.is_none() {
            *presenter = match self.presenter(width, height, format) {
                Ok(p) => Some(p),
                Err(e) => {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| xnw_core::warn!("present: no window or swap chain: {e}; frames are not shown"));
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            // SAFETY: no back buffer is held; every submission was waited for.
            let resized = unsafe { p.swap_chain.ResizeBuffers(BUFFERS, width, height, format, p.flags) };
            match resized {
                Ok(()) => {
                    (p.width, p.height, p.format) = (width, height, format);
                    xnw_core::info!("present: swap chain now {width}x{height} format {}", format.0);
                }
                Err(e) => {
                    xnw_core::warn!("present: swap chain not resized to {width}x{height}: {e}");
                    return None;
                }
            }
        }
        super::capture::note_output(width, height);
        let logged = frame <= 4 || frame.is_power_of_two();
        // SAFETY: our swap chain, list and live source textures.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            p.copy.reset();
            super::crumbs::presenting(1, frame);
            let _crumb = super::crumbs::begin(&p.copy, || format!("present blit frame {frame}, {} planes into {width}x{height}", planes.len()));
            let list = &p.copy.list;
            if test_pattern() {
                fill_patterns(&self.device, list, &mut p.patterns, planes);
            }
            let capture = if present_mode() == 2 {
                // `XNW_PRESENT=noblit`: the back buffer cleared, not drawn.
                list.ResourceBarrier(&[transition(&back, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET)]);
                list.ClearRenderTargetView(self.rtv(&back, None), &[0.0, 0.0, 0.0, 1.0], None);
                list.ResourceBarrier(&[transition(&back, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT)]);
                let (texture, state, _) = planes[0];
                readback_wanted(frame).then(|| read_back_plane(&self.device, list, texture, state)).flatten()
            } else if planes.len() == 1 && planes[0].2 == FULL && !show {
                let (texture, state, _) = planes[0];
                let state = D3D12_RESOURCE_STATES(state as i32);
                let mut before = Vec::with_capacity(2);
                if state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                    before.push(transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE));
                }
                before.push(transition(&back, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST));
                list.ResourceBarrier(&before);
                list.CopyResource(&back, texture);
                let capture = readback_wanted(frame).then(|| super::capture::read_back(&self.device, list, texture)).flatten();
                let mut after = Vec::with_capacity(2);
                if state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                    after.push(transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state));
                }
                after.push(transition(&back, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT));
                list.ResourceBarrier(&after);
                if logged {
                    xnw_core::info!("present: frame {frame}: 1 plane copied into the {width}x{height} back buffer");
                }
                capture
            } else {
                if p.compositor.as_ref().is_none_or(|c| c.format != format) {
                    match Compositor::new(&self.device, format) {
                        Ok(c) => p.compositor = Some(c),
                        Err(e) => {
                            xnw_core::warn!("present: no present compositor: {e}");
                            return None;
                        }
                    }
                }
                let comp = p.compositor.as_ref()?;
                comp.composite(&self.device, list, &back, D3D12_RESOURCE_STATE_PRESENT, planes, width, height);
                let capture = if readback_wanted(frame) {
                    list.ResourceBarrier(&[transition(&back, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                    let capture = super::capture::read_back(&self.device, list, &back);
                    list.ResourceBarrier(&[transition(&back, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT)]);
                    capture
                } else {
                    list.ResourceBarrier(&[transition(&back, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT)]);
                    None
                };
                if logged {
                    let what: Vec<String> = planes
                        .iter()
                        .enumerate()
                        .map(|(i, (tex, _, place))| {
                            let d = tex.GetDesc();
                            let v = viewport(place, width, height);
                            format!("plane {i} {}x{} format {} at ({}, {}) {}x{}", d.Width, d.Height, d.Format.0, v.TopLeftX, v.TopLeftY, v.Width, v.Height)
                        })
                        .collect();
                    xnw_core::info!("present: frame {frame}: {} planes composited into the {width}x{height} back buffer: {}", planes.len(), what.join("; "));
                }
                capture
            };
            drop(_crumb);
            p.copy.close();
            self.execute(&[&p.copy]);
            super::crumbs::presenter_blit(p.copy.crumbs().slot(), frame);
            use_read_back(frame, capture);
            // Each title frame for its number of refreshes (frame.rs).
            super::crumbs::presenting(2, frame);
            let (interval, flags) = if present_mode() == 3 {
                (0, if p.flags.0 & DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0 != 0 { DXGI_PRESENT_ALLOW_TEARING } else { DXGI_PRESENT(0) })
            } else {
                (super::frame::sync_interval(), DXGI_PRESENT(0))
            };
            let _ = p.swap_chain.Present(interval, flags);
            super::crumbs::presenting(0, frame);
            super::frame::host_presented(&p.swap_chain);
            p.meter.frame(planes.len());
            Some(false)
        }
    }

    fn presenter(&self, width: u32, height: u32, format: DXGI_FORMAT) -> windows::core::Result<Presenter> {
        let hwnd = HWND(match window::title_window() {
            Some(w) => {
                window::fit(w, width, height);
                xnw_core::info!("present: into the title's window {w:?}, client sized to fit the desktop");
                w
            }
            None => {
                xnw_core::info!("present: the title made no window; opening our own");
                window::open(width, height)?
            }
        });
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            // `XNW_PRESENT=sync0` asks for tearing where it is allowed.
            let tearing = present_mode() == 3
                && factory.cast::<IDXGIFactory5>().is_ok_and(|f5| {
                    let mut allowed = windows::core::BOOL(0);
                    f5.CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, std::ptr::from_mut(&mut allowed).cast(), size_of::<windows::core::BOOL>() as u32).is_ok() && allowed.as_bool()
                });
            let flags = if tearing { DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING } else { DXGI_SWAP_CHAIN_FLAG(0) };
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                Scaling: DXGI_SCALING_STRETCH,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                Flags: flags.0 as u32,
                ..Default::default()
            };
            let swap_chain = factory.CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)?.cast::<IDXGISwapChain3>()?;
            // The window is the title's or ours; no full-screen switching on Alt+Enter.
            let _ = factory.MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
            let copy = self.new_list()?;
            copy.close();
            xnw_core::info!("present: window and flip-model swap chain {width}x{height} format {} ({BUFFERS} buffers)", format.0);
            // Where the image goes: the output, the session.
            // (The output's name and rectangle need the Gdi feature of the
            // windows crate; its presence and the session are logged.)
            let output = swap_chain.GetContainingOutput().is_ok();
            const SM_REMOTESESSION: i32 = 0x1000;
            let remote = xnw_core::win::GetSystemMetrics(SM_REMOTESESSION) != 0;
            xnw_core::info!(
                "present: windowed in the title's window, containing output {}, sync interval {}, Present flags {}, tearing {}, XNW_PRESENT {}, remote session {remote}; refresh: the frame clock's vertical blank line",
                if output { "found" } else { "none" },
                if present_mode() == 3 { 0 } else { super::frame::sync_interval() },
                if tearing { "ALLOW_TEARING" } else { "0" },
                if tearing { "allowed" } else { "off" },
                ["normal", "off", "noblit", "sync0"][present_mode() as usize]
            );
            Ok(Presenter { swap_chain, flags, width, height, format, copy, meter: Meter::new(hwnd.0), compositor: None, patterns: Vec::new() })
        }
    }
}

/// The window's caption says the GPU is off (null mode), once: the
/// picture stays the last rendered one.
pub(crate) fn null_caption() {
    if let Some(w) = window::title_window() {
        window::set_caption(w as usize, "XNW: null GPU (XNW_GPU=null): the picture is the last rendered frame".into());
    }
}

/// Logs every plane's title resource when what is presented changes.
pub(crate) fn log_planes(planes: &[Plane]) {
    use super::resource::ResourceDesc;
    /// `D3D12_HEAP_FLAG_ALLOW_DISPLAY`.
    const ALLOW_DISPLAY: u32 = 0x8;
    static LAST: std::sync::Mutex<Vec<(u64, u32, u32)>> = std::sync::Mutex::new(Vec::new());
    let shown: Vec<Option<(ResourceDesc, u32, *const u8)>> = planes
        .iter()
        .map(|p| {
            // SAFETY: a plane lists `count` resources.
            (p.count > 0 && !p.resources.is_null()).then(|| super::resource::displayed(unsafe { *p.resources })).flatten()
        })
        .collect();
    let now: Vec<(u64, u32, u32)> = shown.iter().map(|s| s.map_or((0, 0, 0), |(d, _, _)| (d.width, d.height, d.format))).collect();
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if *last == now {
        return;
    }
    *last = now;
    for (i, (plane, s)) in planes.iter().zip(&shown).enumerate() {
        let place = plane.placement();
        match s {
            Some((d, flags, _)) => xnw_core::info!(
                "present: plane {i} of {}: {} resources, first {}x{} format {} layout {:#x} flags {:#x}, heap flags {flags:#x}{}; source {:?}, destination {place:?}, color space {}",
                planes.len(),
                plane.count,
                d.width,
                d.height,
                d.format,
                d.layout,
                d.flags,
                if flags & ALLOW_DISPLAY != 0 { " (allow display)" } else { "" },
                plane.source(),
                plane.color_space
            ),
            None => xnw_core::info!("present: plane {i} of {}: {} resources, not one of ours", planes.len(), plane.count),
        }
        if let (Some(r), Some((d, _, _))) = (plane.source(), s)
            && r != [0, 0, d.width as i32, d.height as i32]
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| xnw_core::warn!("present: plane {i}'s source rectangle {r:?} is not the whole texture; shown whole (shortcut)"));
        }
    }
}

/// `XNW_PRESENT_TEST=1`: fill each plane's host texture with a test
/// gradient before compositing (diagnostic).
fn test_pattern() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XNW_PRESENT_TEST").is_ok_and(|v| v == "1");
        if on {
            xnw_core::info!("present: XNW_PRESENT_TEST: each plane's host texture is filled with a test gradient before compositing");
        }
        on
    })
}

/// Plane `index`'s test texel at `(x, y)` of `width` x `height`, RGBA with
/// premultiplied alpha. Plane 0: an opaque red-green ramp (red left to
/// right, green top to bottom). Plane 1: transparent, but for an opaque
/// blue box over the middle half (green ramping left to right inside it)
/// and a white bar at half coverage along the bottom eighth. Others:
/// opaque magenta.
pub(crate) fn gradient(index: usize, x: u32, y: u32, width: u32, height: u32) -> [u8; 4] {
    let ramp = |v: u32, n: u32| (u64::from(v) * 255 / u64::from(n.max(2) - 1)) as u8;
    match index {
        0 => [ramp(x, width), ramp(y, height), 32, 255],
        1 if (width / 4..width * 3 / 4).contains(&x) && (height / 4..height * 3 / 4).contains(&y) => [0, ramp(x - width / 4, width / 2), 255, 255],
        1 if y >= height * 7 / 8 => [128, 128, 128, 128],
        1 => [0; 4],
        _ => [255, 0, 255, 255],
    }
}

/// An upload buffer holding `texel(x, y)` for every texel of `texture`'s
/// mip 0 (8-bit RGBA or BGRA formats only), and its footprint.
pub(crate) fn pattern_upload(device: &ID3D12Device, texture: &ID3D12Resource, texel: impl Fn(u32, u32) -> [u8; 4]) -> Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    // SAFETY: a live texture.
    let desc = unsafe { texture.GetDesc() };
    let bgra = match desc.Format.0 {
        27..=32 => false,
        87 | 90 | 91 => true,
        _ => return None,
    };
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0;
    // SAFETY: outputs for one subresource.
    unsafe { device.GetCopyableFootprints(&desc, 0, 1, 0, Some(&mut footprint), None, None, Some(&mut total)) };
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() };
    let buffer_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: total,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    let mut at = std::ptr::null_mut();
    // SAFETY: plain creation; the mapping spans `total` bytes.
    unsafe {
        device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut buffer).ok()?;
        let buffer = buffer?;
        buffer.Map(0, None, Some(&mut at)).ok()?;
        let bytes = std::slice::from_raw_parts_mut(at.cast::<u8>(), total as usize);
        let f = footprint.Footprint;
        for y in 0..f.Height {
            let row = &mut bytes[(footprint.Offset + u64::from(y) * u64::from(f.RowPitch)) as usize..];
            for x in 0..f.Width {
                let [r, g, b, a] = texel(x, y);
                row[x as usize * 4..][..4].copy_from_slice(&if bgra { [b, g, r, a] } else { [r, g, b, a] });
            }
        }
        buffer.Unmap(0, None);
        Some((buffer, footprint))
    }
}

/// Records the copy of an upload made by [`pattern_upload`] into
/// `texture`, which is in host state `state` and is left there.
///
/// # Safety
/// An open list; a live texture in `state`.
pub(crate) unsafe fn record_fill(list: &ID3D12GraphicsCommandList, texture: &ID3D12Resource, state: u32, upload: &ID3D12Resource, footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT) {
    let state = D3D12_RESOURCE_STATES(state as i32);
    // SAFETY: the caller's; the locations borrow live resources.
    unsafe {
        if state != D3D12_RESOURCE_STATE_COPY_DEST {
            list.ResourceBarrier(&[transition(texture, state, D3D12_RESOURCE_STATE_COPY_DEST)]);
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(texture),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(upload),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *footprint },
        };
        list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        if state != D3D12_RESOURCE_STATE_COPY_DEST {
            list.ResourceBarrier(&[transition(texture, D3D12_RESOURCE_STATE_COPY_DEST, state)]);
        }
    }
}

/// `XNW_PRESENT_TEST`: overwrites every plane's host texture with its
/// gradient (the title clears them each frame, so every frame).
fn fill_patterns(device: &ID3D12Device, list: &ID3D12GraphicsCommandList, patterns: &mut Vec<((usize, u64, u32, i32), ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)>, planes: &[(&ID3D12Resource, u32, Placement)]) {
    for (i, &(texture, state, _)) in planes.iter().enumerate() {
        // SAFETY: a live texture.
        let d = unsafe { texture.GetDesc() };
        let key = (i, d.Width, d.Height, d.Format.0);
        if !patterns.iter().any(|(k, _, _)| *k == key) {
            let (w, h) = (d.Width as u32, d.Height);
            match pattern_upload(device, texture, |x, y| gradient(i, x, y, w, h)) {
                Some((upload, footprint)) => patterns.push((key, upload, footprint)),
                None => {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| xnw_core::warn!("present: XNW_PRESENT_TEST: plane {i} format {} has no test gradient", d.Format.0));
                    continue;
                }
            }
        }
        if let Some((_, upload, footprint)) = patterns.iter().find(|(k, _, _)| *k == key) {
            // SAFETY: an open list, the plane's live texture in its state.
            unsafe { record_fill(list, texture, state, upload, footprint) };
        }
    }
}

/// Frames per second in the window's caption, once a second, after the
/// caption the window had.
struct Meter {
    since: Instant,
    frames: u32,
    window: usize,
    caption: String,
}

impl Meter {
    fn new(window: *mut std::ffi::c_void) -> Meter {
        let mut text = [0u16; 256];
        // SAFETY: a window handle and a buffer of that length; this form
        // sends no message to the window's thread.
        let n = unsafe { xnw_core::win::InternalGetWindowText(window, text.as_mut_ptr(), text.len() as i32) };
        let caption = String::from_utf16_lossy(&text[..n.max(0) as usize]);
        Meter { since: Instant::now(), frames: 0, window: window as usize, caption }
    }

    /// Counts a presented frame of `planes` planes.
    fn frame(&mut self, planes: usize) {
        self.frames += 1;
        let elapsed = self.since.elapsed();
        if elapsed.as_secs_f64() < 1.0 {
            return;
        }
        let fps = f64::from(self.frames) / elapsed.as_secs_f64();
        let counted = self.frames;
        (self.since, self.frames) = (Instant::now(), 0);
        let frame = super::queue::presented();
        if frame / 300 != (frame - u64::from(counted)) / 300 || frame < 40 {
            xnw_core::info!("present: {fps:.1} fps at frame {frame} ({planes} planes per frame)");
        }
        let text = format!("{} | XNW {fps:.1} fps, frame {frame}", self.caption);
        window::set_caption(self.window, text.trim_start_matches(" | ").to_string());
    }
}

/// The window, on a thread of ours that only pumps its messages.
mod window {
    use xnw_core::win::*;

    const CLASS: &[u16] = &[b'X' as u16, b'N' as u16, b'W' as u16, 0];

    /// A visible top-level window of this process that is not ours: the
    /// one the title made through its own `CreateWindowExA`.
    pub(super) fn title_window() -> Option<*mut std::ffi::c_void> {
        unsafe extern "system" fn visit(window: HANDLE, found: isize) -> i32 {
            let mut process = 0;
            let mut class = [0u16; 8];
            // SAFETY: a window from the enumeration; `found` is ours.
            unsafe {
                GetWindowThreadProcessId(window, &mut process);
                let n = GetClassNameW(window, class.as_mut_ptr(), class.len() as i32);
                if process == GetCurrentProcessId() && IsWindowVisible(window) != 0 && class[..n.max(0) as usize] != CLASS[..3] {
                    *(found as *mut HANDLE) = window;
                    return 0;
                }
            }
            1
        }
        let mut found: HANDLE = std::ptr::null_mut();
        // SAFETY: the callback writes only `found`, which outlives the call.
        unsafe { EnumWindows(visit, (&raw mut found) as isize) };
        (!found.is_null()).then_some(found)
    }

    /// The client size for a `width` x `height` plane: the plane's aspect,
    /// at most nine tenths of the desktop's work area.
    fn fitted(width: u32, height: u32) -> (i32, i32) {
        let mut area = RECT::default();
        // SAFETY: the work area into our rectangle.
        unsafe { SystemParametersInfoW(SPI_GETWORKAREA, 0, (&raw mut area).cast(), 0) };
        let (aw, ah) = (f64::from((area.right - area.left).max(640)) * 0.9, f64::from((area.bottom - area.top).max(480)) * 0.9);
        let scale = (aw / f64::from(width)).min(ah / f64::from(height)).min(1.0);
        ((f64::from(width) * scale) as i32, (f64::from(height) * scale) as i32)
    }

    /// Sizes the title's window to the fitted client size, asynchronously
    /// (its thread pumps its messages) and without activating it.
    pub(super) fn fit(window: *mut std::ffi::c_void, width: u32, height: u32) {
        let (w, h) = fitted(width, height);
        let mut rect = RECT { left: 0, top: 0, right: w, bottom: h };
        // SAFETY: our rectangle, the title's window.
        unsafe {
            AdjustWindowRect(&mut rect, GetWindowLongW(window, GWL_STYLE) as u32, 0);
            SetWindowPos(window, std::ptr::null_mut(), 0, 0, rect.right - rect.left, rect.bottom - rect.top, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_ASYNCWINDOWPOS);
        }
    }

    /// Sets a window's caption from a thread of ours: setting it waits for
    /// the window's thread, which may be busy, and presenting never waits.
    pub(super) fn set_caption(window: usize, text: String) {
        static PENDING: std::sync::Mutex<Option<(usize, String)>> = std::sync::Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((window, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new().name("xnw-caption".into()).spawn(|| {
                loop {
                    let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                    match next {
                        Some((window, text)) => {
                            let wide: Vec<u16> = text.encode_utf16().chain([0]).collect();
                            // SAFETY: a window handle and a NUL-terminated text.
                            unsafe { SetWindowTextW(window as HANDLE, wide.as_ptr()) };
                        }
                        None => std::thread::sleep(std::time::Duration::from_millis(100)),
                    }
                }
            });
        });
    }

    unsafe extern "system" fn procedure(window: HANDLE, message: u32, wparam: usize, lparam: isize) -> isize {
        if message == WM_CLOSE {
            xnw_core::info!("present: window closed; ending the run");
            std::process::exit(0);
        }
        // SAFETY: the default handling for our window.
        unsafe { DefWindowProcW(window, message, wparam, lparam) }
    }

    /// Opens a window of ours, the client fitted to a `width` x `height`
    /// plane, shown without taking focus (when the title made none).
    pub(super) fn open(width: u32, height: u32) -> windows::core::Result<*mut std::ffi::c_void> {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("xnw-window".into())
            .spawn(move || {
                let window = make(width, height);
                let _ = sender.send(window as usize);
                if window.is_null() {
                    return;
                }
                let mut message: MSG = MSG { window: std::ptr::null_mut(), message: 0, wparam: 0, lparam: 0, time: 0, point: [0; 2], private: 0 };
                // SAFETY: the standard message loop of this thread's window.
                unsafe {
                    while GetMessageW(&mut message, std::ptr::null_mut(), 0, 0) > 0 {
                        TranslateMessage(&message);
                        DispatchMessageW(&message);
                    }
                }
            })
            .map_err(|_| windows::core::Error::from_thread())?;
        match receiver.recv() {
            Ok(w) if w != 0 => Ok(w as *mut _),
            _ => Err(windows::core::Error::from_thread()),
        }
    }

    fn make(width: u32, height: u32) -> HANDLE {
        let (width, height) = fitted(width, height);
        // SAFETY: registers our class (again is harmless) and creates one
        // window of it, shown without activation.
        unsafe {
            let instance = GetModuleHandleW(std::ptr::null());
            let class = WNDCLASSEXW {
                size: size_of::<WNDCLASSEXW>() as u32,
                style: 0,
                window_proc: Some(procedure),
                class_extra: 0,
                window_extra: 0,
                instance,
                icon: std::ptr::null_mut(),
                cursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW as *const u16),
                background: std::ptr::null_mut(),
                menu_name: std::ptr::null(),
                class_name: CLASS.as_ptr(),
                small_icon: std::ptr::null_mut(),
            };
            RegisterClassExW(&class);
            let mut rect = RECT { left: 0, top: 0, right: width, bottom: height };
            AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, 0);
            let window = CreateWindowExW(
                0,
                CLASS.as_ptr(),
                CLASS.as_ptr(),
                WS_OVERLAPPEDWINDOW,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                rect.right - rect.left,
                rect.bottom - rect.top,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                instance,
                std::ptr::null(),
            );
            if !window.is_null() {
                ShowWindow(window, SW_SHOWNOACTIVATE);
            }
            window
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewports_follow_the_destination_rectangle() {
        let v = viewport(&[0.5, 0.25, 1.0, 0.75], 3840, 2160);
        assert_eq!((v.TopLeftX, v.TopLeftY, v.Width, v.Height), (1920.0, 540.0, 1920.0, 1080.0));
        let v = viewport(&FULL, 3840, 2160);
        assert_eq!((v.TopLeftX, v.TopLeftY, v.Width, v.Height), (0.0, 0.0, 3840.0, 2160.0));
    }

    #[test]
    fn test_gradients_tell_the_planes_apart() {
        assert_eq!(gradient(0, 0, 0, 1920, 1080), [0, 0, 32, 255]);
        assert_eq!(gradient(0, 1919, 1079, 1920, 1080), [255, 255, 32, 255]);
        assert_eq!(gradient(1, 10, 10, 3840, 2160), [0; 4], "the UI plane is transparent outside its box");
        assert_eq!(gradient(1, 1920, 1080, 3840, 2160)[2..], [255, 255], "an opaque blue box in the middle");
        assert_eq!(gradient(1, 10, 2100, 3840, 2160), [128; 4], "a half-covering bar at the bottom");
    }

    /// Two synthetic planes of different sizes through
    /// the compositor `PresentX` uses, the second placed by its destination
    /// rectangle, read back from the composited image.
    #[test]
    fn two_planes_are_composited_by_destination_rectangle() {
        if std::env::var_os("XNW_AGILITY").is_none_or(|d| d.is_empty()) {
            println!("SKIPPED: set XNW_AGILITY to the Agility SDK folder to run on the host GPU");
            return;
        }
        let host = super::super::host::host().expect("a host submission queue");
        let device = &host.device;
        let make = |width: u32, height: u32, flags: D3D12_RESOURCE_FLAGS| {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(width),
                Height: height,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: flags,
                ..Default::default()
            };
            let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
            let mut texture: Option<ID3D12Resource> = None;
            // SAFETY: plain creation.
            unsafe { device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut texture).unwrap() };
            texture.unwrap()
        };
        // Plane 0: a 64x36 opaque red scene; plane 1: a 128x72 UI plane,
        // its left half opaque blue, its right half transparent, placed in
        // the top-right quarter.
        let (scene, ui) = (make(64, 36, D3D12_RESOURCE_FLAG_NONE), make(128, 72, D3D12_RESOURCE_FLAG_NONE));
        let target = make(128, 72, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        let (scene_up, scene_fp) = pattern_upload(device, &scene, |_, _| [255, 0, 0, 255]).unwrap();
        let (ui_up, ui_fp) = pattern_upload(device, &ui, |x, _| if x < 64 { [0, 0, 255, 255] } else { [0; 4] }).unwrap();
        let compositor = Compositor::new(device, DXGI_FORMAT_R8G8B8A8_UNORM).unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: an open list; live textures in the common state (0).
        let (buffer, footprint) = unsafe {
            record_fill(&list.list, &scene, 0, &scene_up, &scene_fp);
            record_fill(&list.list, &ui, 0, &ui_up, &ui_fp);
            compositor.composite(device, &list.list, &target, D3D12_RESOURCE_STATE_COMMON, &[(&scene, 0, FULL), (&ui, 0, [0.5, 0.0, 1.0, 0.5])], 128, 72);
            list.list.ResourceBarrier(&[transition(&target, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            super::super::capture::read_back(device, &list.list, &target).unwrap()
        };
        list.close();
        host.execute(&[&list]);
        let mut data = std::ptr::null_mut();
        // SAFETY: the finished copy's read-back buffer, mapped whole.
        let rows = unsafe {
            buffer.Map(0, None, Some(&mut data)).unwrap();
            std::slice::from_raw_parts(data.cast::<u8>(), footprint.Footprint.RowPitch as usize * 72).to_vec()
        };
        let at = |x: usize, y: usize| -> [u8; 4] { rows[y * footprint.Footprint.RowPitch as usize + 4 * x..][..4].try_into().unwrap() };
        const RED: [u8; 4] = [255, 0, 0, 255];
        assert_eq!(at(10, 10), RED, "top left: the scene");
        assert_eq!(at(10, 60), RED, "bottom left: the scene, stretched to the back buffer");
        assert_eq!(at(90, 60), RED, "bottom right: outside the UI plane's rectangle");
        assert_eq!(at(72, 10), [0, 0, 255, 255], "top right, left half: the UI plane's opaque part");
        assert_eq!(at(120, 10), RED, "top right, right half: the UI plane is transparent there");
    }
}
