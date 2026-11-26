//! the title's user interface (Scaleform HUD, menus, intro title
//! cards and credits) drawn in our layer from the title's own translated
//! vertex and pixel programs, as the bridge runs a world draw.
//!
//! The title records its UI as plain Draw/DrawIndexed calls with an input
//! layout (`VText`/`FText` glyphs from a 2048x2048 A8 glyph cache,
//! `V*VertexCxformEAlpha`, `V*TexTGCxform*` images, `VS_Bink`/`PS_Bink*`
//! movies) into its UI plane (R8G8B8A8, plane 1). Our pass never drew
//! them. Here each row runs through the bridge's vertex-program mode with
//! the title's input layout, vertex and index bytes and render-target-0
//! blend, into one UI layer (R8G8B8A8, cleared to zero), which is composed
//! over our picture with the console's plane blend: premultiplied, ONE and
//! INV_SRC_ALPHA (present.rs's compositor for planes 1..).
//!
//! Not done: the plane-1 alpha compensation (stage 16,
//! `CS_UIOverlayAlphaCompensation`, alpha^(1/2.2)); the Scaleform stencil
//! masks (no depth-stencil target: masked shapes draw unmasked); the menu
//! backdrop blur (sub-passes 184/185 read the title's scene target).

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::bridge::{self, Binding, Pipes, Programs, Shared};

/// `ui=on` in an `XNW_WORLD_RENDER` value (default off; the last one wins).
pub fn switch_on(v: &str) -> bool {
    v.split(',').map(str::trim).filter_map(|p| p.strip_prefix("ui=")).last().is_some_and(|s| s == "on")
}

/// One input element of the title's layout (as `xnw_gpu`'s `PipeInput`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Input {
    pub name: String,
    pub index: u32,
    pub format: u32,
    pub slot: u32,
    pub offset: u32,
    pub per_instance: bool,
    pub step: u32,
}

/// One UI draw: its programs and binding, the title's state (render
/// target 0's blend as D3D12 numbers: enable, src, dest, op, src alpha,
/// dest alpha, alpha op, write mask; cull), input layout, the vertex
/// bytes per IA slot from the view's start (with the stride), the index
/// bytes from the draw's first index (with the DXGI format), and the draw:
/// count, instances, first vertex (Draw), base vertex (DrawIndexed).
pub struct Row<'a> {
    pub programs: &'a Programs,
    pub binding: Binding<'a>,
    pub blend: [u32; 8],
    pub cull: u32,
    pub inputs: &'a [Input],
    pub vertices: Vec<(u32, &'a [u8], u32)>,
    pub indices: Option<(&'a [u8], u32)>,
    pub draw: (u32, u32, u32, i32),
    /// A triangle strip (else a list).
    pub strip: bool,
}

/// The layer's format.
pub const LAYER: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;

/// Render target 0's blend from the title's numbers.
pub fn blend_desc(b: [u32; 8]) -> D3D12_RENDER_TARGET_BLEND_DESC {
    let f = |v: u32, d: D3D12_BLEND| if v == 0 { d } else { D3D12_BLEND(v as i32) };
    let o = |v: u32| if v == 0 { D3D12_BLEND_OP_ADD } else { D3D12_BLEND_OP(v as i32) };
    D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: (b[0] != 0).into(),
        LogicOpEnable: false.into(),
        SrcBlend: f(b[1], D3D12_BLEND_ONE),
        DestBlend: f(b[2], D3D12_BLEND_ZERO),
        BlendOp: o(b[3]),
        SrcBlendAlpha: f(b[4], D3D12_BLEND_ONE),
        DestBlendAlpha: f(b[5], D3D12_BLEND_ZERO),
        BlendOpAlpha: o(b[6]),
        LogicOp: D3D12_LOGIC_OP_NOOP,
        RenderTargetWriteMask: if b[7] == 0 { D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8 } else { b[7] as u8 },
    }
}

/// What [`draw_layer`] gives back: the layer (RGBA8, premultiplied, rows of
/// `width`), one line per row, rows drawn, the GPU seconds.
pub struct Layer {
    pub rgba: Vec<u8>,
    pub lines: Vec<String>,
    pub drawn: usize,
    pub seconds: f64,
}

/// Draws `rows` in order into one layer of `size`, on one command list, on
/// `host` (the title's programs: [`bridge::title_programs_here`]); a row
/// that cannot be set up is a line and is left out. `limit`: seconds the
/// submission may take.
pub fn draw_layer(host: &xnw_gpu::host::Host, rows: &[Row], size: (u32, u32), limit: f64) -> Result<Layer, String> {
    bridge::title_programs_here(host)?;
    let e = |e: windows::core::Error| e.message().to_string();
    let dev = &host.device;
    let (w, h) = size;
    let mut lines = Vec::new();
    // SAFETY: D3D12 calls with descriptions that outlive them; every
    // resource lives until the fence says the GPU is done.
    unsafe {
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(e)?;
        let rt = D3D12_RESOURCE_STATE_RENDER_TARGET;
        let clear = D3D12_CLEAR_VALUE { Format: LAYER, Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0; 4] } };
        let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: u64::from(w), Height: h, DepthOrArraySize: 1, MipLevels: 1, Format: LAYER, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, ..Default::default() };
        let colour = bridge::committed(dev, &desc, D3D12_HEAP_TYPE_DEFAULT, rt, Some(&clear))?;
        let rtvs: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() }).map_err(e)?;
        let rtv = rtvs.GetCPUDescriptorHandleForHeapStart();
        dev.CreateRenderTargetView(&colour, None, rtv);
        list.ClearRenderTargetView(rtv, &[0.0; 4], None);
        let records = bridge::buffer(dev, 256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)?;
        let mut shared = Shared::default();
        let mut keep: Vec<ID3D12Resource> = Vec::new();
        let mut held = Vec::new();
        let mut drawn = 0;
        for (i, r) in rows.iter().enumerate() {
            if let Some(gap) = bridge::gap(&host.caps, &r.programs.needs()) {
                lines.push(format!("  ui row {i}: not run: {gap}"));
                continue;
            }
            let names: Vec<std::ffi::CString> = r.inputs.iter().map(|x| std::ffi::CString::new(x.name.as_str()).unwrap_or_default()).collect();
            let elems: Vec<D3D12_INPUT_ELEMENT_DESC> = r
                .inputs
                .iter()
                .zip(&names)
                .map(|(x, n)| D3D12_INPUT_ELEMENT_DESC {
                    SemanticName: windows::core::PCSTR(n.as_ptr().cast()),
                    SemanticIndex: x.index,
                    Format: DXGI_FORMAT(x.format as i32),
                    InputSlot: x.slot,
                    AlignedByteOffset: x.offset,
                    InputSlotClass: if x.per_instance { D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA } else { D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA },
                    InstanceDataStepRate: x.step,
                })
                .collect();
            let pipes = match Pipes::new_ia(dev, r.programs, LAYER, blend_desc(r.blend), D3D12_CULL_MODE(r.cull.max(1) as i32), &elems) {
                Ok(p) => p,
                Err(x) => {
                    lines.push(format!("  ui row {i}: no pipeline: {x}"));
                    continue;
                }
            };
            let up = match bridge::upload(dev, &list, &r.binding, &pipes, &mut shared) {
                Ok(u) => u,
                Err(x) => {
                    lines.push(format!("  ui row {i}: binding not uploaded: {x}"));
                    continue;
                }
            };
            let slots = r.inputs.iter().map(|x| x.slot as usize + 1).max().unwrap_or(0).min(16);
            let mut vbs = vec![D3D12_VERTEX_BUFFER_VIEW::default(); slots];
            for &(slot, bytes, stride) in r.vertices.iter().filter(|v| (v.0 as usize) < slots) {
                let size = (bytes.len() as u64).next_multiple_of(256);
                let b = bridge::upload_bytes(dev, bytes, size)?;
                vbs[slot as usize] = D3D12_VERTEX_BUFFER_VIEW { BufferLocation: b.GetGPUVirtualAddress(), SizeInBytes: bytes.len() as u32, StrideInBytes: stride };
                keep.push(b);
            }
            let ibv = match r.indices {
                Some((bytes, format)) => {
                    let b = bridge::upload_bytes(dev, bytes, (bytes.len() as u64).next_multiple_of(256))?;
                    let v = D3D12_INDEX_BUFFER_VIEW { BufferLocation: b.GetGPUVirtualAddress(), SizeInBytes: bytes.len() as u32, Format: DXGI_FORMAT(format as i32) };
                    keep.push(b);
                    Some(v)
                }
                None => None,
            };
            let first = rtv;
            list.OMSetRenderTargets(1, Some(&first), true, None);
            list.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: w as f32, Height: h as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
            list.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: w as i32, bottom: h as i32 }]);
            bridge::record_ia_draw(&list, &pipes, &up, records.GetGPUVirtualAddress(), &vbs, ibv.as_ref(), r.draw, r.strip);
            drawn += 1;
            held.push((pipes, up));
        }
        let pitch = (w * 4).next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
        let back = bridge::buffer(dev, u64::from(pitch) * u64::from(h), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
        list.ResourceBarrier(&[bridge::barrier(&colour, rt, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&back), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: 0, Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: LAYER, Width: w, Height: h, Depth: 1, RowPitch: pitch } } } };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&colour), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        list.Close().map_err(e)?;
        let queue = &host.queue;
        let t0 = std::time::Instant::now();
        queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(&list).map_err(e)?)]);
        let sent = xnw_gpu::gpuwatch::begin("ui");
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        queue.Signal(&fence, 1).map_err(e)?;
        while fence.GetCompletedValue() < 1 {
            dev.GetDeviceRemovedReason().map_err(|x| format!("ui layer: during the submission: {}", e(x)))?;
            if t0.elapsed().as_secs_f64() > limit {
                return Err(format!("ui layer: the submission took over {limit} s"));
            }
            std::thread::yield_now();
        }
        xnw_gpu::gpuwatch::end("ui", sent, None);
        let seconds = t0.elapsed().as_secs_f64();
        let mut from = std::ptr::null_mut();
        back.Map(0, None, Some(&mut from)).map_err(e)?;
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h as usize {
            rgba.extend_from_slice(std::slice::from_raw_parts(from.cast::<u8>().add(y * pitch as usize), (w * 4) as usize));
        }
        back.Unmap(0, None);
        drop((held, keep, records, colour));
        Ok(Layer { rgba, lines, drawn, seconds })
    }
}

/// A host on `device` (a queue of its own) with `caps`: the replay's pass
/// device for [`draw_layer`].
pub fn host_of(device: &ID3D12Device, caps: xnw_gpu::host::Caps) -> Result<xnw_gpu::host::Host, String> {
    // SAFETY: a plain queue description.
    let queue = unsafe { device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()) }.map_err(|e| e.message().to_string())?;
    Ok(xnw_gpu::host::Host { device: device.clone(), caps, queue })
}

/// The console's plane blend of a premultiplied UI layer over `picture`
/// (both RGBA8, same size): `ui + picture * (1 - ui.a)`; the picture's
/// alpha stays opaque.
pub fn compose(picture: &mut [u8], ui: &[u8]) {
    for (p, u) in picture.chunks_exact_mut(4).zip(ui.chunks_exact(4)) {
        let keep = 255 - u32::from(u[3]);
        for c in 0..3 {
            p[c] = (u32::from(u[c]) + (u32::from(p[c]) * keep + 127) / 255).min(255) as u8;
        }
    }
}

/// The layer shader: one full-screen triangle; each pixel reads its texel
/// of the layer (RGBA8 words) and the pipeline blends it premultiplied,
/// ONE and INV_SRC_ALPHA (the console's plane blend), no depth.
pub const UI_SHADER: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0)"
cbuffer Size : register(b0) { uint width; uint height; uint2 pad; };
StructuredBuffer<uint> layer : register(t0);
[RootSignature(RS)]
float4 vs(uint v : SV_VertexID) : SV_Position {
    float2 t = float2((v << 1) & 2, v & 2);
    return float4(t * float2(2, -2) + float2(-1, 1), 0, 1);
}
[RootSignature(RS)]
float4 ps(float4 at : SV_Position) : SV_Target {
    uint2 p = min(uint2(at.xy), uint2(width - 1, height - 1));
    uint c = layer[p.y * width + p.x];
    return float4(c & 255, (c >> 8) & 255, (c >> 16) & 255, c >> 24) / 255.0;
}
"#;

/// UI1: our layer pipeline over the pass's colour target and the upload
/// buffers of the frames in flight.
pub struct UiPass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    bufs: std::collections::VecDeque<ID3D12Resource>,
}

// SAFETY: D3D12 objects are free-threaded; used by its pass's owner.
unsafe impl Send for UiPass {}

impl UiPass {
    pub fn new(device: &ID3D12Device) -> Result<UiPass, String> {
        let e = |e: windows::core::Error| e.to_string();
        let (vs, ps) = (crate::compile_source(UI_SHADER, c"vs", c"vs_5_1")?, crate::compile_source(UI_SHADER, c"ps", c"ps_5_1")?);
        // SAFETY: object creation with descriptions that outlive each call.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, crate::blob_bytes(&vs)).map_err(e)?;
            // ONE (2), INV_SRC_ALPHA (6), ADD (1), all channels.
            let state = crate::RenderState { blend: [1, 2, 6, 1, 2, 6, 1, 15], ..Default::default() };
            let pso = crate::pipeline_state_with(device, &root, &vs, &ps, &state, &[]).map_err(e)?;
            Ok(UiPass { root, pso, bufs: std::collections::VecDeque::new() })
        }
    }

    /// Records the layer `rgba` (premultiplied RGBA8 at `size`) over the
    /// bound colour target of `l` (open).
    pub fn record(&mut self, device: &ID3D12Device, l: &ID3D12GraphicsCommandList, rgba: &[u8], size: (u32, u32)) -> Result<(), String> {
        let bytes = (size.0 as u64) * (size.1 as u64) * 4;
        if rgba.len() as u64 != bytes {
            return Err(format!("layer of {} bytes for {}x{}", rgba.len(), size.0, size.1));
        }
        let buf = bridge::upload_bytes(device, rgba, bytes.next_multiple_of(256))?;
        // SAFETY: an open list and our live objects.
        unsafe {
            let k = [size.0, size.1, 0, 0];
            l.SetGraphicsRootSignature(&self.root);
            l.SetPipelineState(&self.pso);
            l.SetGraphicsRoot32BitConstants(0, 4, k.as_ptr().cast(), 0);
            l.SetGraphicsRootShaderResourceView(1, buf.GetGPUVirtualAddress());
            l.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            l.DrawInstanced(3, 1, 0, 0);
        }
        self.bufs.push_back(buf);
        while self.bufs.len() > 3 {
            self.bufs.pop_front();
        }
        Ok(())
    }
}

/// Pixels of a layer any row wrote (alpha or colour not zero).
pub fn covered(ui: &[u8]) -> usize {
    ui.chunks_exact(4).filter(|p| p.iter().any(|&b| b != 0)).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ui_switch_is_off_unless_on() {
        assert!(switch_on("screen,ui=on"));
        assert!(!switch_on("screen"));
        assert!(!switch_on("ui=on,ui=off"));
        assert!(switch_on(" ui=on "));
    }

    /// Premultiplied over: a clear layer leaves the picture, an opaque
    /// texel replaces it, half alpha adds the layer to half the picture.
    #[test]
    fn plane_blend_is_premultiplied_over() {
        let mut p = vec![200, 100, 50, 255, 200, 100, 50, 255, 200, 100, 50, 255];
        compose(&mut p, &[0, 0, 0, 0, 10, 20, 30, 255, 100, 0, 0, 128]);
        assert_eq!(p, [200, 100, 50, 255, 10, 20, 30, 255, 200, 50, 25, 255]);
        assert_eq!(covered(&[0, 0, 0, 0, 0, 0, 0, 1]), 1);
    }

    /// The title's blend numbers map to D3D12's; zeros (not given) are the
    /// default of no blending.
    #[test]
    fn blend_from_the_title_numbers() {
        let b = blend_desc([1, 2, 6, 1, 2, 6, 1, 15]);
        assert_eq!((b.BlendEnable.as_bool(), b.SrcBlend, b.DestBlend, b.RenderTargetWriteMask), (true, D3D12_BLEND_ONE, D3D12_BLEND_INV_SRC_ALPHA, 15));
        let d = blend_desc([0; 8]);
        assert_eq!((d.BlendEnable.as_bool(), d.SrcBlend, d.DestBlend, d.RenderTargetWriteMask), (false, D3D12_BLEND_ONE, D3D12_BLEND_ZERO, 15));
    }
}
