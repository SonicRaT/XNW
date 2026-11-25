//! Mesh-program fallback: the host has no mesh shaders, so a
//! console mesh program runs as a compute shader that writes one fixed-size
//! record per group (counts, vertices, packed primitives), and a plain vertex
//! shader draws the records.
//!
//! This is the smallest graphics path that proves it: one offscreen colour
//! target (RGBA8), one depth target (D32), one root signature shared by the
//! compute and graphics pipelines, a draw, and readback. Nothing is tuned.
//!
//! Record of group `g`, at `g * record_bytes` in the record buffer:
//!
//! | bytes                      | holds                                        |
//! |----------------------------|----------------------------------------------|
//! | 0..16                      | vertex count, primitive count, two free u32  |
//! | 16..16+V*vertex_stride     | vertex `i`: position float4, then attributes |
//! | prim_offset..+P*4          | primitive `i`: console packing, indices in bits 0-8, 10-18, 20-28, cull in bit 31 |
//!
//! Root signature (both pipelines): `b0` = 12 root constants (`record_bytes`,
//! `vertex_stride`, `prim_offset`, `groups`, then 8 for the program), `u0` =
//! record buffer, `u1` = index buffer (indexed mode), `t0` = record buffer as
//! seen by the vertex shader.

use std::ffi::c_void;
use std::mem::ManuallyDrop;

use windows::Win32::Foundation::{CloseHandle, HANDLE, RECT};
use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R32_UINT, DXGI_SAMPLE_DESC,
};
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::{Interface, Result};

use crate::Host;

/// How the vertex shader finds its vertex.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawMode {
    /// The compute pass also writes a global index buffer (dead primitives
    /// as degenerate triangles); `DrawIndexedInstanced`, and `SV_VertexID` is
    /// `group * max_vertices + vertex`. Shared vertices are shaded once.
    Indexed,
    /// No index buffer; `DrawInstanced(groups * max_prims * 3)`, and the
    /// vertex shader decodes group, primitive and corner from `SV_VertexID`,
    /// reads the counts and the packed primitive itself.
    VertexId,
}

/// Fixed capacity of one record.
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    /// Bytes per vertex: 16 for the position plus 16 per attribute.
    pub vertex_stride: u32,
    pub max_vertices: u32,
    pub max_prims: u32,
}

impl Layout {
    pub const HEADER_BYTES: u32 = 16;

    pub fn prim_offset(&self) -> u32 {
        Self::HEADER_BYTES + self.max_vertices * self.vertex_stride
    }

    pub fn record_bytes(&self) -> u32 {
        self.prim_offset() + self.max_prims * 4
    }

    /// Index-buffer bytes per group in indexed mode.
    pub fn index_bytes(&self) -> u32 {
        self.max_prims * 12
    }
}

/// One frame read back.
pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// RGBA8, red in the low byte, row-major.
    pub color: Vec<u32>,
    pub depth: Vec<f32>,
    /// The record buffer of the drawn groups, as u32.
    pub records: Vec<u32>,
}

/// GPU time of one repetition, in microseconds.
#[derive(Debug, Clone, Copy)]
pub struct Timing {
    pub compute_us: f64,
    pub draw_us: f64,
}

pub struct MeshFallback {
    pub mode: DrawMode,
    pub layout: Layout,
    pub max_groups: u32,
    pub width: u32,
    pub height: u32,
    root: ID3D12RootSignature,
    cs: ID3D12PipelineState,
    gfx: ID3D12PipelineState,
    records: ID3D12Resource,
    indices: ID3D12Resource,
    color: ID3D12Resource,
    depth: ID3D12Resource,
    rtv: ID3D12DescriptorHeap,
    dsv: ID3D12DescriptorHeap,
}

fn bytecode(dxil: &[u8]) -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr().cast(), BytecodeLength: dxil.len() }
}

fn transition(res: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(res.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// Records the transitions and releases the references they hold.
fn barriers(list: &ID3D12GraphicsCommandList, items: &[(&ID3D12Resource, D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATES)]) {
    let v: Vec<D3D12_RESOURCE_BARRIER> = items.iter().map(|(r, b, a)| transition(r, *b, *a)).collect();
    // SAFETY: the barriers reference live resources.
    unsafe { list.ResourceBarrier(&v) };
    for b in v {
        // SAFETY: every barrier above is a transition.
        drop(ManuallyDrop::into_inner(ManuallyDrop::into_inner(unsafe { b.Anonymous.Transition }).pResource));
    }
}

fn texture(host: &Host, w: u32, h: u32, format: DXGI_FORMAT, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES, clear: D3D12_CLEAR_VALUE) -> Result<ID3D12Resource> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(w),
        Height: h,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: flags,
        ..Default::default()
    };
    let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let mut out: Option<ID3D12Resource> = None;
    // SAFETY: the descriptions outlive the call.
    unsafe { host.device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, Some(&clear), &mut out)? };
    Ok(out.expect("resource on success"))
}

/// Copies a 2D texture with 4-byte texels into a new readback buffer.
fn copy_texture(host: &Host, list: &ID3D12GraphicsCommandList, tex: &ID3D12Resource, format: DXGI_FORMAT, w: u32, h: u32) -> Result<ID3D12Resource> {
    let pitch = (w * 4).next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    let back = host.buffer(u64::from(pitch * h), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
    let dst = D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(back.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: format, Width: w, Height: h, Depth: 1, RowPitch: pitch },
            },
        },
    };
    let src = D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(tex.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
    };
    // SAFETY: both locations reference live resources.
    unsafe { list.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
    drop(ManuallyDrop::into_inner(dst.pResource));
    drop(ManuallyDrop::into_inner(src.pResource));
    Ok(back)
}

/// Maps a readback buffer and copies `count` items out, rows of `pitch`
/// bytes holding `row` items each (`pitch == row * 4` for a plain buffer).
fn read_back<T: Copy>(buf: &ID3D12Resource, count: usize, row: usize, pitch: usize) -> Result<Vec<T>> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    // SAFETY: the buffer is at least `count / row * pitch` bytes and the GPU
    // has finished with it.
    unsafe {
        buf.Map(0, None, Some(&mut ptr))?;
        let mut out = Vec::with_capacity(count);
        for r in 0..count / row {
            let line = ptr.cast::<u8>().add(r * pitch).cast::<T>();
            out.extend_from_slice(std::slice::from_raw_parts(line, row));
        }
        buf.Unmap(0, None);
        Ok(out)
    }
}

/// Runs a closed command list on the host's queue and waits for it (a free function: `Host` is xnw-device's).
fn submit_and_wait(host: &Host, list: &ID3D12GraphicsCommandList) -> Result<()> {
    // SAFETY: plain COM calls on live objects.
    unsafe {
        host.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
        let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        host.queue.Signal(&fence, 1)?;
        let event: HANDLE = CreateEventW(None, false, false, None)?;
        fence.SetEventOnCompletion(1, event)?;
        WaitForSingleObject(event, INFINITE);
        let _ = CloseHandle(event);
        host.device.GetDeviceRemovedReason()
    }
}

impl MeshFallback {
    /// Builds both pipelines and every buffer for up to `max_groups` groups.
    /// `cs` is the program (`cs_6_6`), `vs`/`ps` the draw (`vs_6_6`/`ps_6_6`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(host: &Host, cs: &[u8], vs: &[u8], ps: &[u8], mode: DrawMode, layout: Layout, max_groups: u32, width: u32, height: u32) -> Result<MeshFallback> {
        let dev = &host.device;
        // SAFETY: COM calls on live objects; every pointer handed over is to a
        // local that outlives the call.
        unsafe {
            let params = [
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 12 },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                    Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                    Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 1, RegisterSpace: 0 } },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                    Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
                },
            ];
            let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                ..Default::default()
            };
            let mut blob: Option<ID3DBlob> = None;
            D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &mut blob, None)?;
            let blob = blob.expect("blob on success");
            let root: ID3D12RootSignature = dev.CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()))?;

            let cs_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(root.clone())),
                CS: bytecode(cs),
                ..Default::default()
            };
            let cs_pso = dev.CreateComputePipelineState(&cs_desc);
            drop(ManuallyDrop::into_inner(cs_desc.pRootSignature));

            let mut blend = D3D12_BLEND_DESC::default();
            blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            let mut rtv_formats = [DXGI_FORMAT::default(); 8];
            rtv_formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            let gfx_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(root.clone())),
                VS: bytecode(vs),
                PS: bytecode(ps),
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                    DepthFunc: D3D12_COMPARISON_FUNC_LESS,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: rtv_formats,
                DSVFormat: DXGI_FORMAT_D32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let gfx = dev.CreateGraphicsPipelineState(&gfx_desc);
            drop(ManuallyDrop::into_inner(gfx_desc.pRootSignature));
            let (cs_pso, gfx) = (cs_pso?, gfx?);

            let groups = u64::from(max_groups.max(1));
            let uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            let records = host.buffer(groups * u64::from(layout.record_bytes()), D3D12_HEAP_TYPE_DEFAULT, uav, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)?;
            let index_bytes = if mode == DrawMode::Indexed { groups * u64::from(layout.index_bytes()) } else { 256 };
            let indices = host.buffer(index_bytes, D3D12_HEAP_TYPE_DEFAULT, uav, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)?;

            let color = texture(host, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_CLEAR_VALUE {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0; 4] },
            })?;
            let depth = texture(host, width, height, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_CLEAR_VALUE {
                Format: DXGI_FORMAT_D32_FLOAT,
                Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 } },
            })?;
            let rtv: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() })?;
            let dsv: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: 1, ..Default::default() })?;
            dev.CreateRenderTargetView(&color, None, rtv.GetCPUDescriptorHandleForHeapStart());
            dev.CreateDepthStencilView(&depth, None, dsv.GetCPUDescriptorHandleForHeapStart());

            Ok(MeshFallback { mode, layout, max_groups, width, height, root, cs: cs_pso, gfx, records, indices, color, depth, rtv, dsv })
        }
    }

    fn new_list(&self, host: &Host) -> Result<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)> {
        // SAFETY: plain COM calls.
        unsafe {
            let alloc: ID3D12CommandAllocator = host.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = host.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &self.cs)?;
            Ok((alloc, list))
        }
    }

    /// One iteration: program, then draw. Leaves every resource in the state
    /// it started in. `stamps`, when given, gets timestamps `base..base+4`:
    /// before the program, after it, before the draw, after the draw.
    fn record(&self, list: &ID3D12GraphicsCommandList, groups: u32, consts: &[u32; 8], stamps: Option<(&ID3D12QueryHeap, u32)>) {
        let l = &self.layout;
        let mut c = [0u32; 12];
        c[..4].copy_from_slice(&[l.record_bytes(), l.vertex_stride, l.prim_offset(), groups]);
        c[4..].copy_from_slice(consts);
        let stamp = |i: u32| {
            if let Some((heap, base)) = stamps {
                // SAFETY: the heap holds `base + 4` queries or more.
                unsafe { list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, base + i) };
            }
        };
        let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let srv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        let ib = D3D12_RESOURCE_STATE_INDEX_BUFFER;
        // SAFETY: COM calls recording into an open list; every pointer is to
        // a local that outlives the call.
        unsafe {
            stamp(0);
            list.SetPipelineState(&self.cs);
            list.SetComputeRootSignature(&self.root);
            list.SetComputeRoot32BitConstants(0, 12, c.as_ptr().cast(), 0);
            list.SetComputeRootUnorderedAccessView(1, self.records.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2, self.indices.GetGPUVirtualAddress());
            list.Dispatch(groups, 1, 1);
            stamp(1);
            barriers(list, &[(&self.records, uav, srv), (&self.indices, uav, ib)]);

            let rtv = self.rtv.GetCPUDescriptorHandleForHeapStart();
            let dsv = self.dsv.GetCPUDescriptorHandleForHeapStart();
            list.ClearRenderTargetView(rtv, &[0.0; 4], None);
            list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
            stamp(2);
            list.SetPipelineState(&self.gfx);
            list.SetGraphicsRootSignature(&self.root);
            list.SetGraphicsRoot32BitConstants(0, 12, c.as_ptr().cast(), 0);
            list.SetGraphicsRootUnorderedAccessView(1, self.records.GetGPUVirtualAddress());
            list.SetGraphicsRootUnorderedAccessView(2, self.indices.GetGPUVirtualAddress());
            list.SetGraphicsRootShaderResourceView(3, self.records.GetGPUVirtualAddress());
            list.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
            list.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: self.width as f32, Height: self.height as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
            list.RSSetScissorRects(&[RECT { left: 0, top: 0, right: self.width as i32, bottom: self.height as i32 }]);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            let corners = groups * l.max_prims * 3;
            match self.mode {
                DrawMode::Indexed => {
                    let view = D3D12_INDEX_BUFFER_VIEW {
                        BufferLocation: self.indices.GetGPUVirtualAddress(),
                        SizeInBytes: groups * l.index_bytes(),
                        Format: DXGI_FORMAT_R32_UINT,
                    };
                    list.IASetIndexBuffer(Some(&view));
                    list.DrawIndexedInstanced(corners, 1, 0, 0, 0);
                }
                DrawMode::VertexId => list.DrawInstanced(corners, 1, 0, 0),
            }
            stamp(3);
            barriers(list, &[(&self.records, srv, uav), (&self.indices, ib, uav)]);
        }
    }

    /// Runs `groups` groups once and reads back colour, depth and records.
    /// `consts` are the program's eight root constants (`c[4..12]`).
    pub fn frame(&self, host: &Host, groups: u32, consts: &[u32; 8]) -> Result<Frame> {
        assert!(groups <= self.max_groups);
        let (_alloc, list) = self.new_list(host)?;
        self.record(&list, groups, consts, None);
        let (w, h) = (self.width, self.height);
        let rt = D3D12_RESOURCE_STATE_RENDER_TARGET;
        let dw = D3D12_RESOURCE_STATE_DEPTH_WRITE;
        let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let cp = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barriers(&list, &[(&self.color, rt, cp), (&self.depth, dw, cp), (&self.records, uav, cp)]);
        let color = copy_texture(host, &list, &self.color, DXGI_FORMAT_R8G8B8A8_UNORM, w, h)?;
        let depth = copy_texture(host, &list, &self.depth, DXGI_FORMAT_D32_FLOAT, w, h)?;
        let rec_bytes = u64::from(groups.max(1)) * u64::from(self.layout.record_bytes());
        let recs = host.buffer(rec_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
        // SAFETY: both buffers are at least `rec_bytes` long.
        unsafe { list.CopyBufferRegion(&recs, 0, &self.records, 0, rec_bytes) };
        barriers(&list, &[(&self.color, cp, rt), (&self.depth, cp, dw), (&self.records, cp, uav)]);
        // SAFETY: the list is open.
        unsafe { list.Close()? };
        submit_and_wait(host, &list)?;
        let pitch = (w * 4).next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT) as usize;
        let n = (w * h) as usize;
        let words = (rec_bytes / 4) as usize;
        Ok(Frame {
            width: w,
            height: h,
            color: read_back(&color, n, w as usize, pitch)?,
            depth: read_back(&depth, n, w as usize, pitch)?,
            records: read_back(&recs, words, words, words * 4)?,
        })
    }

    /// The indirect form. `cull` (DXIL with its root signature inside:
    /// one root constant at `b0`, a root UAV at `u0`) runs one thread that
    /// writes the group counts into a buffer with the constant `groups`;
    /// the [`IndirectWriter`] (ring of `ring_bytes`) turns them into
    /// commands, clamped to `capacity` groups; two `ExecuteIndirect` run the
    /// program and the draw. The processor never sees the count. Returns
    /// the frame (records of `capacity` groups, from the ring's start) and
    /// the writer's counters after it.
    pub fn frame_indirect(&self, host: &Host, cull: &[u8], groups: u32, capacity: u32, ring_bytes: u64, consts: &[u32; 8]) -> Result<(Frame, WriterStats)> {
        assert!(self.mode == DrawMode::Indexed);
        let writer = IndirectWriter::new(host, ring_bytes)?;
        let sig = writer.signatures(host, (&self.root, &self.root), CommandRoots { records: 1, indices: 2, vertex_records: 3 })?;
        let uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        let ua = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let title_args = host.buffer(256, D3D12_HEAP_TYPE_DEFAULT, uav, ua)?;
        let slot = host.buffer(256, D3D12_HEAP_TYPE_DEFAULT, uav, ua)?;
        let l = &self.layout;
        let mut c = [0u32; 12];
        c[..4].copy_from_slice(&[l.record_bytes(), l.vertex_stride, l.prim_offset(), capacity]);
        c[4..].copy_from_slice(consts);
        let (_alloc, list) = self.new_list(host)?;
        // SAFETY: COM calls recording into an open list; every buffer lives
        // until the list has run.
        unsafe {
            let cull_root: ID3D12RootSignature = host.device.CreateRootSignature(0, cull)?;
            let cull_pso: ID3D12PipelineState = host.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&cull_root),
                CS: bytecode(cull),
                ..Default::default()
            })?;
            list.SetComputeRootSignature(&cull_root);
            list.SetPipelineState(&cull_pso);
            list.SetComputeRoot32BitConstant(0, groups, 0);
            list.SetComputeRootUnorderedAccessView(1, title_args.GetGPUVirtualAddress());
            list.Dispatch(1, 1, 1);
            let plan = Plan { record_bytes: l.record_bytes(), index_bytes: l.index_bytes(), corners: l.max_prims * 3, record: 0, capacity };
            writer.record(&list, plan, title_args.GetGPUVirtualAddress(), None, slot.GetGPUVirtualAddress(), None);
            let ia = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
            barriers(&list, &[(&slot, ua, ia)]);
            list.SetComputeRootSignature(&self.root);
            list.SetPipelineState(&self.cs);
            list.SetComputeRoot32BitConstants(0, 12, c.as_ptr().cast(), 0);
            writer.ring_writable(&list);
            list.ExecuteIndirect(&sig.compute, 1, &slot, 0, None, 0);
            writer.ring_readable(&list);
            let rtv = self.rtv.GetCPUDescriptorHandleForHeapStart();
            let dsv = self.dsv.GetCPUDescriptorHandleForHeapStart();
            list.ClearRenderTargetView(rtv, &[0.0; 4], None);
            list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
            list.SetPipelineState(&self.gfx);
            list.SetGraphicsRootSignature(&self.root);
            list.SetGraphicsRoot32BitConstants(0, 12, c.as_ptr().cast(), 0);
            list.SetGraphicsRootUnorderedAccessView(1, self.records.GetGPUVirtualAddress());
            list.SetGraphicsRootUnorderedAccessView(2, self.indices.GetGPUVirtualAddress());
            list.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
            list.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: self.width as f32, Height: self.height as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
            list.RSSetScissorRects(&[RECT { left: 0, top: 0, right: self.width as i32, bottom: self.height as i32 }]);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            let view = D3D12_INDEX_BUFFER_VIEW { BufferLocation: writer.ring.GetGPUVirtualAddress(), SizeInBytes: writer.ring_bytes() as u32, Format: DXGI_FORMAT_R32_UINT };
            list.IASetIndexBuffer(Some(&view));
            list.ExecuteIndirect(&sig.draw, 1, &slot, DRAW_COMMAND, None, 0);
            writer.ring_released(&list);
        }
        let (w, h) = (self.width, self.height);
        let rt = D3D12_RESOURCE_STATE_RENDER_TARGET;
        let dw = D3D12_RESOURCE_STATE_DEPTH_WRITE;
        let cp = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barriers(&list, &[(&self.color, rt, cp), (&self.depth, dw, cp), (&writer.ring, D3D12_RESOURCE_STATE_COMMON, cp)]);
        let color = copy_texture(host, &list, &self.color, DXGI_FORMAT_R8G8B8A8_UNORM, w, h)?;
        let depth = copy_texture(host, &list, &self.depth, DXGI_FORMAT_D32_FLOAT, w, h)?;
        let rec_bytes = (u64::from(capacity.max(1)) * u64::from(l.record_bytes())).min(writer.ring_bytes());
        let recs = host.buffer(rec_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
        // SAFETY: both buffers are at least `rec_bytes` long; the list is open.
        unsafe {
            list.CopyBufferRegion(&recs, 0, &writer.ring, 0, rec_bytes);
        }
        barriers(&list, &[(&self.color, cp, rt), (&self.depth, cp, dw)]);
        // SAFETY: the list is open.
        unsafe { list.Close()? };
        submit_and_wait(host, &list)?;
        let pitch = (w * 4).next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT) as usize;
        let n = (w * h) as usize;
        let words = (rec_bytes / 4) as usize;
        let frame = Frame {
            width: w,
            height: h,
            color: read_back(&color, n, w as usize, pitch)?,
            depth: read_back(&depth, n, w as usize, pitch)?,
            records: read_back(&recs, words, words, words * 4)?,
        };
        Ok((frame, writer.stats()))
    }

    /// Runs `groups` groups `repeats` times in one submission (after one
    /// warm-up iteration) and returns the GPU time of each repetition.
    pub fn time(&self, host: &Host, groups: u32, consts: &[u32; 8], repeats: u32) -> Result<Vec<Timing>> {
        assert!(groups <= self.max_groups);
        // SAFETY: plain COM calls; the readback range is the resolved one.
        unsafe {
            let n = repeats * 4;
            let mut heap: Option<ID3D12QueryHeap> = None;
            host.device.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: n, NodeMask: 0 }, &mut heap)?;
            let heap = heap.expect("heap on success");
            let back = host.buffer(u64::from(n) * 8, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
            let (_alloc, list) = self.new_list(host)?;
            self.record(&list, groups, consts, None);
            for r in 0..repeats {
                self.record(&list, groups, consts, Some((&heap, r * 4)));
            }
            list.ResolveQueryData(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, n, &back, 0);
            list.Close()?;
            submit_and_wait(host, &list)?;
            let freq = host.queue.GetTimestampFrequency()? as f64;
            let ticks: Vec<u64> = read_back(&back, n as usize, n as usize, n as usize * 8)?;
            let us = |a: u64, b: u64| b.saturating_sub(a) as f64 * 1e6 / freq;
            Ok(ticks.chunks(4).map(|t| Timing { compute_us: us(t[0], t[1]), draw_us: us(t[2], t[3]) }).collect())
        }
    }
}

// ---- Record memory for the title's mesh draws ----

/// Record and index memory of one host command list's mesh draws: default
/// heap buffers, bump-allocated per draw and reused after the list is
/// reset. A chunk is written by the compute pass (unordered access) and
/// read by the draw (vertex-shader resource, index buffer); a draw that
/// writes a chunk the list read before moves it back first.
///
/// Buffers decay to the common state when their list finishes, and a
/// buffer in the common state is promoted to unordered access by the first
/// dispatch writing it, so a chunk starts every list writable.
#[derive(Default)]
pub struct Arena {
    chunks: Vec<Chunk>,
    current: usize,
    /// GPU times of the list's mesh draws (`XNW_MESH_TIME=1`).
    timer: Option<Timer>,
    /// Argument slots of the list's indirect mesh dispatches.
    args: Vec<ArgsChunk>,
    args_current: usize,
}

/// What one of our buffers is in, as recorded in a list: common (each
/// list's start; buffers decay to it after each execution), unordered
/// access, or read (by draws as records and indices, or as indirect
/// arguments). Every change is an explicit barrier from the state noted
/// here (no reliance on promotion, which the validation layer does not
/// see for root views).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tracked {
    #[default]
    Common,
    Writable,
    Readable,
}

/// The state of `Tracked::Readable` for record chunks and the ring:
/// records read by the vertex shader, indices by the input assembler.
pub const READ_RECORDS: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATES(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_INDEX_BUFFER.0);

/// The D3D12 state of `t` for a buffer whose readable state is `read`.
fn state_of(t: Tracked, read: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_STATES {
    match t {
        Tracked::Common => D3D12_RESOURCE_STATE_COMMON,
        Tracked::Writable => D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        Tracked::Readable => read,
    }
}

/// Moves `buffers` (all in `*t`) to `to` on `list`, noting it.
pub fn move_to(list: &ID3D12GraphicsCommandList, buffers: &[&ID3D12Resource], t: &mut Tracked, to: Tracked, read: D3D12_RESOURCE_STATES) {
    if *t == to {
        return;
    }
    let (before, after) = (state_of(*t, read), state_of(to, read));
    let items: Vec<_> = buffers.iter().map(|b| (*b, before, after)).collect();
    barriers(list, &items);
    *t = to;
}

/// A buffer of argument rooms: written by our passes (unordered access),
/// read by `ExecuteIndirect` (indirect argument).
struct ArgsChunk {
    buffer: ID3D12Resource,
    bytes: u64,
    used: u64,
    state: Tracked,
}

/// Room for host indirect arguments (one indirect mesh dispatch's
/// commands; the clamped records of any indirect execution).
#[derive(Debug, Clone, Copy)]
pub struct ArgsSlot {
    pub gpu: u64,
    pub offset: u64,
    chunk: usize,
}

struct Chunk {
    records: ID3D12Resource,
    indices: ID3D12Resource,
    capacity: (u64, u64),
    used: (u64, u64),
    state: Tracked,
}

/// Where one mesh draw writes and reads.
#[derive(Debug, Clone, Copy)]
pub struct Segment {
    pub records: u64,
    pub indices: u64,
    chunk: usize,
}

/// The smallest chunk: 4 MiB of records (about 100 groups at 18 parameters).
const MIN_RECORDS: u64 = 4 << 20;

/// Bytes allocated by every arena (for reports).
pub static ARENA_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Arena {
    /// After the list's reset: every chunk is free, and in the common state
    /// (where the list starts it).
    pub fn reset(&mut self, host: &Host) {
        self.collect_times(host);
        self.current = 0;
        for c in &mut self.chunks {
            c.used = (0, 0);
            c.state = Tracked::Common;
        }
        self.args_current = 0;
        for c in &mut self.args {
            c.used = 0;
            c.state = Tracked::Common;
        }
    }

    /// An argument slot for one indirect mesh dispatch.
    pub fn args_slot(&mut self, host: &Host, list: &ID3D12GraphicsCommandList) -> Result<ArgsSlot> {
        self.args_room(host, list, ARGS_SLOT)
    }

    /// `bytes` of argument room (64-byte aligned), its chunk made writable
    /// on `list`.
    pub fn args_room(&mut self, host: &Host, list: &ID3D12GraphicsCommandList, bytes: u64) -> Result<ArgsSlot> {
        let bytes = bytes.max(1).next_multiple_of(ARGS_SLOT);
        while self.args_current < self.args.len() && self.args[self.args_current].used + bytes > self.args[self.args_current].bytes {
            self.args_current += 1;
        }
        if self.args_current == self.args.len() {
            let size = bytes.max(ARGS_CHUNK).next_multiple_of(ARGS_CHUNK);
            let uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            let buffer = host.buffer(size, D3D12_HEAP_TYPE_DEFAULT, uav, D3D12_RESOURCE_STATE_COMMON)?;
            ARENA_BYTES.fetch_add(size, std::sync::atomic::Ordering::Relaxed);
            self.args.push(ArgsChunk { buffer, bytes: size, used: 0, state: Tracked::Common });
        }
        let c = &mut self.args[self.args_current];
        move_to(list, &[&c.buffer], &mut c.state, Tracked::Writable, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        let offset = c.used;
        c.used += bytes;
        // SAFETY: a live buffer.
        let gpu = unsafe { c.buffer.GetGPUVirtualAddress() } + offset;
        Ok(ArgsSlot { gpu, offset, chunk: self.args_current })
    }

    /// After our pass filled `s`: its chunk readable as arguments; the
    /// buffer to execute from.
    pub fn args_written(&mut self, list: &ID3D12GraphicsCommandList, s: ArgsSlot) -> ID3D12Resource {
        let c = &mut self.args[s.chunk];
        move_to(list, &[&c.buffer], &mut c.state, Tracked::Readable, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        c.buffer.clone()
    }

    /// The tracked states of the argument chunks (tests).
    pub fn args_states(&self) -> Vec<Tracked> {
        self.args.iter().map(|c| c.state).collect()
    }

    /// Room for `records` and `indices` bytes (256-aligned starts), the
    /// chunk made writable on `list`.
    pub fn segment(&mut self, host: &Host, list: &ID3D12GraphicsCommandList, records: u64, indices: u64) -> Result<Segment> {
        let fits = |c: &Chunk| c.used.0 + records <= c.capacity.0 && c.used.1 + indices <= c.capacity.1;
        while self.current < self.chunks.len() && !fits(&self.chunks[self.current]) {
            self.current += 1;
        }
        if self.current == self.chunks.len() {
            let last = self.chunks.last().map_or(MIN_RECORDS / 2, |c| c.capacity.0);
            let rec = (last * 2).min(256 << 20).max(records).next_multiple_of(1 << 16);
            let idx = (rec / 4).max(indices).next_multiple_of(1 << 16);
            let uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            let common = D3D12_RESOURCE_STATE_COMMON;
            let r = host.buffer(rec, D3D12_HEAP_TYPE_DEFAULT, uav, common)?;
            let i = host.buffer(idx, D3D12_HEAP_TYPE_DEFAULT, uav, common)?;
            ARENA_BYTES.fetch_add(rec + idx, std::sync::atomic::Ordering::Relaxed);
            self.chunks.push(Chunk { records: r, indices: i, capacity: (rec, idx), used: (0, 0), state: Tracked::Common });
        }
        let c = &mut self.chunks[self.current];
        move_to(list, &[&c.records, &c.indices], &mut c.state, Tracked::Writable, READ_RECORDS);
        // SAFETY: live buffers.
        let (rb, ib) = unsafe { (c.records.GetGPUVirtualAddress(), c.indices.GetGPUVirtualAddress()) };
        let s = Segment { records: rb + c.used.0, indices: ib + c.used.1, chunk: self.current };
        c.used.0 = (c.used.0 + records).next_multiple_of(256);
        c.used.1 = (c.used.1 + indices).next_multiple_of(256);
        Ok(s)
    }

    /// After the compute pass wrote `s`: its chunk readable by the draw
    /// (records as a shader resource, indices as the index buffer).
    pub fn written(&mut self, list: &ID3D12GraphicsCommandList, s: Segment) {
        let c = &mut self.chunks[s.chunk];
        move_to(list, &[&c.records, &c.indices], &mut c.state, Tracked::Readable, READ_RECORDS);
    }

    /// The tracked states of the record chunks (tests).
    pub fn chunk_states(&self) -> Vec<Tracked> {
        self.chunks.iter().map(|c| c.state).collect()
    }
}

/// The arena of the host list at `key`, reset first when `reset`.
pub fn with_arena<R>(host: &Host, key: usize, reset: bool, f: impl FnOnce(&mut Arena) -> R) -> R {
    static ARENAS: std::sync::Mutex<Vec<(usize, Arena)>> = std::sync::Mutex::new(Vec::new());
    let mut all = ARENAS.lock().unwrap_or_else(|e| e.into_inner());
    let at = match all.iter().position(|a| a.0 == key) {
        Some(at) => at,
        None => {
            all.push((key, Arena::default()));
            all.len() - 1
        }
    };
    let arena = &mut all[at].1;
    if reset {
        arena.reset(host);
    }
    f(arena)
}

/// Mesh draws run through the fallback, and their groups.
pub static MESH_DRAWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static MESH_GROUPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// ---- Indirect mesh dispatches on the GPU ----
//
// The title's culled world geometry is drawn by `DispatchMesh` through
// `ExecuteIndirect`, its group count written by a culling compute pass in
// the same submission. The count stays
// on the GPU: a one-thread pass of ours (the writer) reads the record,
// clamps it, takes room for its groups' records and indices from one ring
// (a GPU bump counter, zeroed before each submission), and writes two
// commands: the mesh program as compute with the room's two root views, and
// the indexed draw with the records' root view and the room's first index.
// Two `ExecuteIndirect` run them. No count reaches the processor, and the
// memory is the ring's, sized once (a test run ran out of memory with a
// 4 MiB segment reserved per dispatch).
//
// After VIctory (`crates/victory-gpu/src/mesh_indirect.rs` @
// 13d3718, GPL-3.0-or-later): their conversion pass, one thread writing
// both command records (root view addresses, then the dispatch or draw)
// from the clamped title record, and the two executions. Changed: one title
// record per pass (the title's signatures are all max count 1); the
// room comes from a ring shared by every list of a submission (an atomic
// bump) instead of a buffer per dispatch; our fallback's records and
// indices are two regions of the room; the list's indirect-work budget of
// `d3d12x::indirect` applies; the pass counts what it clamped, left empty
// and could not place.

/// Bytes per argument slot: the compute command (two root views and a
/// dispatch, 28) at 0, the draw command (a root view and an indexed draw,
/// 28) at [`DRAW_COMMAND`].
pub const ARGS_SLOT: u64 = 64;
/// Where the draw command starts in a slot.
pub const DRAW_COMMAND: u64 = 32;
/// Bytes per command.
const COMMAND_STRIDE: u32 = 32;
/// Bytes per argument chunk.
const ARGS_CHUNK: u64 = 64 << 10;
/// The ring's bytes: records and indices of every indirect mesh dispatch of
/// one submission. Past it, dispatches run empty (counted as not placed).
pub const RING_BYTES: u64 = 64 << 20;
/// The most output (records and indices) one indirect mesh dispatch takes
/// (VIctory's bound, 32 MiB per dispatch in a 64 MiB pool).
pub const DISPATCH_BYTES: u64 = 32 << 20;
/// Byte of the ring's bump counter in the counters buffer.
const RING_COUNTER: usize = 64;

const WRITER: &str = r#"
#define RS "RootConstants(num32BitConstants=10, b0), UAV(u0), UAV(u1), UAV(u2), UAV(u3), UAV(u4)"
cbuffer Plan : register(b0) {
    uint record_bytes; uint index_bytes; uint corners; uint record;
    uint flags; uint capacity; uint2 ring; uint ring_bytes; uint unused;
};
// The title's record (x, y, z groups) and its signature's count, if any.
RWByteAddressBuffer args : register(u0);
RWByteAddressBuffer count : register(u1);
// Our slot: compute command at 0, draw command at 32.
RWByteAddressBuffer slot : register(u2);
// The list's indirect work so far (d3d12x::indirect); flags bit 1.
RWByteAddressBuffer budget : register(u3);
// Dispatches, clamped, empty, groups, not placed (wrapping counters); the
// ring's bump counter at byte 64.
RWByteAddressBuffer stats : register(u4);
static const uint BUDGET = 1u << 27;
uint2 at(uint offset) {
    uint lo = ring.x + offset;
    return uint2(lo, ring.y + (lo < ring.x ? 1u : 0u));
}
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint3 g = args.Load3(0);
    // The console's mesh programs are one-dimensional: y and z run as 1,
    // and a zero there is no work.
    uint x = (g.y == 0 || g.z == 0) ? 0u : g.x;
    if ((flags & 1) != 0 && record >= count.Load(0)) {
        x = 0;
    }
    uint clamped = 0;
    uint most = min(capacity, ring_bytes / (record_bytes + index_bytes + 1));
    if (x > most) {
        x = most;
        clamped = 1;
    }
    if ((flags & 2) != 0) {
        // After VIctory PR #1510: charged only when it fits, so rejected
        // dispatches charge nothing and the word never wraps. One writer
        // runs at a time (each is its own dispatch between UAV barriers).
        uint used = min(budget.Load(0), BUDGET);
        if (x * 128 > BUDGET - used) {
            x = 0;
        } else {
            budget.Store(0, used + x * 128);
        }
    }
    uint records = (x * record_bytes + 255) & ~255u;
    uint need = records + ((x * index_bytes + 255) & ~255u);
    uint start = 0;
    uint unplaced = 0;
    if (x != 0) {
        stats.InterlockedAdd(64, need, start);
        if (start > ring_bytes || need > ring_bytes - start) {
            x = 0;
            start = 0;
            unplaced = 1;
        }
    }
    uint2 rec = at(start);
    uint2 idx = at(start + records);
    slot.Store4(0, uint4(rec, idx));
    slot.Store3(16, uint3(x, 1, 1));
    slot.Store2(32, rec);
    slot.Store4(40, uint4(x * corners, 1, (start + records) / 4, 0));
    slot.Store(56, 0);
    uint unused2;
    stats.InterlockedAdd(0, 1, unused2);
    stats.InterlockedAdd(4, clamped, unused2);
    stats.InterlockedAdd(8, x == 0 ? 1u : 0u, unused2);
    stats.InterlockedAdd(12, x, unused2);
    stats.InterlockedAdd(16, unplaced, unused2);
}
"#;

/// The writer pass, the ring, and the pass's counters (system memory the
/// processor reads and zeroes).
pub struct IndirectWriter {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// The ring: records and indices of a submission's indirect mesh
    /// dispatches. Unordered access while written, then readable by draws.
    pub ring: ID3D12Resource,
    stats: ID3D12Resource,
    stats_data: *mut u32,
    /// Command signatures by (compute root, draw root) signature: the
    /// roots kept so their addresses stay theirs.
    signatures: std::sync::Mutex<Vec<(ID3D12RootSignature, ID3D12RootSignature, Signatures)>>,
}

// SAFETY: D3D12 objects are free-threaded; `stats_data` is a persistent
// mapping, read, and written only before a submission runs.
unsafe impl Send for IndirectWriter {}
// SAFETY: as above.
unsafe impl Sync for IndirectWriter {}

/// The two command signatures of one fallback pipeline's indirect
/// dispatches.
#[derive(Clone)]
pub struct Signatures {
    /// Root UAV records, root UAV indices, dispatch.
    pub compute: ID3D12CommandSignature,
    /// Root SRV records, indexed draw.
    pub draw: ID3D12CommandSignature,
}

/// Root parameters the commands set: compute records and indices, draw
/// records.
#[derive(Debug, Clone, Copy)]
pub struct CommandRoots {
    pub records: u32,
    pub indices: u32,
    pub vertex_records: u32,
}

/// What the writer is told about one title record.
#[derive(Debug, Clone, Copy)]
pub struct Plan {
    pub record_bytes: u32,
    pub index_bytes: u32,
    /// Index-buffer entries per group (`max_prims * 3`).
    pub corners: u32,
    /// The record's index, against the count buffer.
    pub record: u32,
    /// Groups one dispatch may run: larger counts are cut to it.
    pub capacity: u32,
}

/// The writer's counters: dispatches, clamped, empty, groups, not placed
/// (ring full).
pub type WriterStats = [u32; 5];

fn heap_buffer(host: &Host, heap: &D3D12_HEAP_PROPERTIES, bytes: u64) -> Result<ID3D12Resource> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut out: Option<ID3D12Resource> = None;
    // SAFETY: both descriptions outlive the call.
    unsafe { host.device.CreateCommittedResource(heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut out)? };
    out.ok_or_else(windows::core::Error::empty)
}

impl IndirectWriter {
    /// The pass and its counters, and a ring of `ring_bytes`.
    pub fn new(host: &Host, ring_bytes: u64) -> Result<IndirectWriter> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; out pointers ours.
        let compiled = unsafe {
            D3DCompile(WRITER.as_ptr().cast(), WRITER.len(), windows::core::s!("mesh_indirect"), None, None, windows::core::s!("main"), windows::core::s!("cs_5_1"), 0, 0, &mut code, Some(&mut errors))
        };
        if let Err(e) = compiled {
            if let Some(errors) = errors {
                // SAFETY: the blob's own bytes.
                let text = unsafe { std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize()) };
                xnw_core::warn!("mesh indirect writer: {}", String::from_utf8_lossy(text));
            }
            return Err(e);
        }
        let code = code.ok_or_else(windows::core::Error::empty)?;
        let dev = &host.device;
        // SAFETY: the blob's own bytes (the root signature is inside the
        // shader); descriptions point at locals that outlive each call.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = dev.CreateRootSignature(0, bytes)?;
            let pso = dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                ..Default::default()
            })?;
            let custom = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let stats = heap_buffer(host, &custom, 256)?;
            let mut data: *mut c_void = std::ptr::null_mut();
            stats.Map(0, None, Some(&mut data))?;
            std::ptr::write_bytes(data.cast::<u8>(), 0, 256);
            let ring = heap_buffer(host, &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, ring_bytes)?;
            ARENA_BYTES.fetch_add(ring_bytes, std::sync::atomic::Ordering::Relaxed);
            Ok(IndirectWriter { root, pso, ring, stats, stats_data: data.cast(), signatures: std::sync::Mutex::default() })
        }
    }

    /// The ring's bytes.
    pub fn ring_bytes(&self) -> u64 {
        // SAFETY: a live buffer.
        unsafe { self.ring.GetDesc() }.Width
    }

    /// Before a submission runs (no GPU work of ours in flight): the ring
    /// is empty again.
    pub fn ring_reset(&self) {
        // SAFETY: a word of the mapping; the GPU is not using it now.
        unsafe { self.stats_data.add(RING_COUNTER / 4).write_volatile(0) };
    }

    /// The command signatures for a fallback pipeline whose compute half
    /// has root signature `compute` and draw half `draw`, made once.
    pub fn signatures(&self, host: &Host, (compute, draw): (&ID3D12RootSignature, &ID3D12RootSignature), roots: CommandRoots) -> Result<Signatures> {
        let mut made = self.signatures.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, _, s)) = made.iter().find(|(c, d, _)| c.as_raw() == compute.as_raw() && d.as_raw() == draw.as_raw()) {
            return Ok(s.clone());
        }
        let view = |kind, index| D3D12_INDIRECT_ARGUMENT_DESC {
            Type: kind,
            Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 { UnorderedAccessView: D3D12_INDIRECT_ARGUMENT_DESC_0_4 { RootParameterIndex: index } },
        };
        let plain = |kind| D3D12_INDIRECT_ARGUMENT_DESC { Type: kind, ..Default::default() };
        let c = [
            view(D3D12_INDIRECT_ARGUMENT_TYPE_UNORDERED_ACCESS_VIEW, roots.records),
            view(D3D12_INDIRECT_ARGUMENT_TYPE_UNORDERED_ACCESS_VIEW, roots.indices),
            plain(D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH),
        ];
        let d = [view(D3D12_INDIRECT_ARGUMENT_TYPE_SHADER_RESOURCE_VIEW, roots.vertex_records), plain(D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED)];
        let make = |args: &[D3D12_INDIRECT_ARGUMENT_DESC], root: &ID3D12RootSignature| -> Result<ID3D12CommandSignature> {
            let desc = D3D12_COMMAND_SIGNATURE_DESC { ByteStride: COMMAND_STRIDE, NumArgumentDescs: args.len() as u32, pArgumentDescs: args.as_ptr(), NodeMask: 0 };
            let mut s: Option<ID3D12CommandSignature> = None;
            // SAFETY: the description points at live arguments.
            unsafe { host.device.CreateCommandSignature(&desc, root, &mut s)? };
            s.ok_or_else(windows::core::Error::empty)
        };
        let s = Signatures { compute: make(&c, compute)?, draw: make(&d, draw)? };
        made.push((compute.clone(), draw.clone(), s.clone()));
        Ok(s)
    }

    /// Records the writer for one title record: its 12 bytes at host
    /// address `args`, the signature's count at `count` (if any), the slot
    /// at `slot`, the list's budget at `budget` (if any). Every buffer in
    /// unordered access. Replaces the list's compute root signature and
    /// pipeline; a UAV barrier first orders it after the passes that wrote
    /// the record.
    pub fn record(&self, list: &ID3D12GraphicsCommandList, plan: Plan, args: u64, count: Option<u64>, slot: u64, budget: Option<u64>) {
        let flags = u32::from(count.is_some()) | (u32::from(budget.is_some()) << 1);
        // SAFETY: a live buffer.
        let ring = unsafe { self.ring.GetGPUVirtualAddress() };
        let constants = [plan.record_bytes, plan.index_bytes, plan.corners, plan.record, flags, plan.capacity, ring as u32, (ring >> 32) as u32, self.ring_bytes().min(u64::from(u32::MAX)) as u32, 0];
        // SAFETY: an open list; our root signature, pipeline and buffers the
        // caller keeps alive until the list has run.
        unsafe {
            let stats = self.stats.GetGPUVirtualAddress();
            list.ResourceBarrier(&[uav_barrier()]);
            list.SetComputeRootSignature(&self.root);
            list.SetPipelineState(&self.pso);
            list.SetComputeRoot32BitConstants(0, constants.len() as u32, constants.as_ptr().cast(), 0);
            list.SetComputeRootUnorderedAccessView(1, args);
            list.SetComputeRootUnorderedAccessView(2, count.unwrap_or(args));
            list.SetComputeRootUnorderedAccessView(3, slot);
            list.SetComputeRootUnorderedAccessView(4, budget.unwrap_or(stats + 128));
            list.SetComputeRootUnorderedAccessView(5, stats);
            list.Dispatch(1, 1, 1);
            list.ResourceBarrier(&[uav_barrier()]);
        }
    }

    /// The ring is shared by every list of a submission, whose order of
    /// recording is not their order of execution, so each use is a full
    /// cycle that starts and ends in the common state: writable before
    /// the compute command, [`IndirectWriter::ring_readable`] before the
    /// draw, [`IndirectWriter::ring_released`] after it.
    pub fn ring_writable(&self, list: &ID3D12GraphicsCommandList) {
        barriers(list, &[(&self.ring, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
    }

    pub fn ring_readable(&self, list: &ID3D12GraphicsCommandList) {
        barriers(list, &[(&self.ring, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, READ_RECORDS)]);
    }

    pub fn ring_released(&self, list: &ID3D12GraphicsCommandList) {
        barriers(list, &[(&self.ring, READ_RECORDS, D3D12_RESOURCE_STATE_COMMON)]);
    }
    /// The counters so far (they wrap; take differences).
    pub fn stats(&self) -> WriterStats {
        // SAFETY: the mapping holds 64 words; the GPU writes them atomically.
        std::array::from_fn(|i| unsafe { self.stats_data.add(i).read_volatile() })
    }
}

/// A barrier on every unordered access.
fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: ManuallyDrop::new(None) }) },
    }
}

// ---- Switches and GPU times of the title's mesh draws ----

/// `XNW_MESH`, comma-separated: `off` (drop every mesh draw), `only:<s>`
/// and `skip:<s>` (keep or drop draws whose pipeline name contains `s`;
/// repeatable), `max-groups:<n>` (cut each draw to `n` groups),
/// `compute-only` (run the program, skip its draw). `XNW_MESH_LOG=<n>`
/// logs the first `n` mesh draws (default 8); `XNW_MESH_TIME=1` times each
/// mesh draw on the GPU and logs the slow ones and a per-program summary.
/// For bisecting a fault in the fallback.
#[derive(Debug, Default)]
pub struct Switches {
    pub off: bool,
    pub only: Vec<String>,
    pub skip: Vec<String>,
    pub max_groups: Option<u32>,
    pub compute_only: bool,
    pub log: u64,
    pub time: bool,
}

impl Switches {
    pub fn parse(mesh: Option<&str>, log: Option<&str>, time: Option<&str>) -> Switches {
        let mut s = Switches { log: log.and_then(|v| v.trim().parse().ok()).unwrap_or(8), time: time.is_some_and(|v| v == "1"), ..Switches::default() };
        for item in mesh.unwrap_or("").split(',').map(str::trim).filter(|i| !i.is_empty()) {
            match item.split_once(':') {
                Some(("only", v)) => s.only.push(v.to_owned()),
                Some(("skip", v)) => s.skip.push(v.to_owned()),
                Some(("max-groups", v)) => s.max_groups = v.parse().ok(),
                None if item == "off" => s.off = true,
                None if item == "compute-only" => s.compute_only = true,
                _ => xnw_core::warn!("XNW_MESH: unknown item {item:?}"),
            }
        }
        s
    }

    /// Whether a mesh draw of pipeline `name` runs.
    pub fn allows(&self, name: &str) -> bool {
        !self.off && (self.only.is_empty() || self.only.iter().any(|o| name.contains(o.as_str()))) && !self.skip.iter().any(|k| name.contains(k.as_str()))
    }
}

pub fn switches() -> &'static Switches {
    static S: std::sync::OnceLock<Switches> = std::sync::OnceLock::new();
    S.get_or_init(|| {
        let var = |k| std::env::var(k).ok();
        Switches::parse(var("XNW_MESH").as_deref(), var("XNW_MESH_LOG").as_deref(), var("XNW_MESH_TIME").as_deref())
    })
}

/// Timestamp queries per arena: three per mesh draw.
const QUERIES: u32 = 3 * 4096;

/// One arena's timestamps: before the program, after it, after the draw.
pub struct Timer {
    heap: ID3D12QueryHeap,
    back: ID3D12Resource,
    used: u32,
    draws: Vec<(String, u32, u32)>,
}

impl Timer {
    fn new(host: &Host) -> Result<Timer> {
        let mut heap: Option<ID3D12QueryHeap> = None;
        // SAFETY: plain COM calls.
        unsafe { host.device.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: QUERIES, NodeMask: 0 }, &mut heap)? };
        let back = host.buffer(u64::from(QUERIES) * 8, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
        Ok(Timer { heap: heap.ok_or_else(windows::core::Error::empty)?, back, used: 0, draws: Vec::new() })
    }
}

/// Per-program GPU time since the last summary: (draws, groups, us).
static TIMES: std::sync::Mutex<Vec<(String, u64, u64, f64)>> = std::sync::Mutex::new(Vec::new());

impl Arena {
    /// Starts timing a mesh draw (`XNW_MESH_TIME=1`): the first query's
    /// index, or `None` when timing is off or the arena's queries are used.
    pub fn time_begin(&mut self, host: &Host, list: &ID3D12GraphicsCommandList) -> Option<u32> {
        if !switches().time {
            return None;
        }
        if self.timer.is_none() {
            self.timer = Timer::new(host).ok();
        }
        let t = self.timer.as_mut()?;
        if t.used + 3 > QUERIES {
            return None;
        }
        let q = t.used;
        t.used += 3;
        // SAFETY: an open list; `q` is inside the heap.
        unsafe { list.EndQuery(&t.heap, D3D12_QUERY_TYPE_TIMESTAMP, q) };
        Some(q)
    }

    /// Timestamp `k` (1 after the program, 2 after the draw) of the draw
    /// started at `q`; at 2 the three are resolved.
    pub fn time_mark(&mut self, list: &ID3D12GraphicsCommandList, q: u32, k: u32, name: &str, groups: u32) {
        let Some(t) = self.timer.as_mut() else { return };
        // SAFETY: an open list; the queries are inside the heap and the
        // readback buffer holds all of them.
        unsafe {
            list.EndQuery(&t.heap, D3D12_QUERY_TYPE_TIMESTAMP, q + k);
            if k == 2 {
                list.ResolveQueryData(&t.heap, D3D12_QUERY_TYPE_TIMESTAMP, q, 3, &t.back, u64::from(q) * 8);
                t.draws.push((name.to_owned(), groups, q));
            }
        }
    }

    /// At the list's reset (its last recording has run): the times of its
    /// mesh draws into the per-program table; slow draws logged.
    fn collect_times(&mut self, host: &Host) {
        let Some(t) = self.timer.as_mut() else { return };
        let draws = std::mem::take(&mut t.draws);
        let used = std::mem::replace(&mut t.used, 0);
        if draws.is_empty() {
            return;
        }
        // SAFETY: a plain query.
        let Ok(freq) = (unsafe { host.queue.GetTimestampFrequency() }) else { return };
        let Ok(ticks) = read_back::<u64>(&t.back, used as usize, used as usize, used as usize * 8) else { return };
        let us = |a: u64, b: u64| b.saturating_sub(a) as f64 * 1e6 / freq as f64;
        static SLOW: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let mut times = TIMES.lock().unwrap_or_else(|e| e.into_inner());
        for (name, groups, q) in draws {
            let q = q as usize;
            let (cs, draw) = (us(ticks[q], ticks[q + 1]), us(ticks[q + 1], ticks[q + 2]));
            if cs + draw > 20_000.0 && SLOW.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 64 {
                xnw_core::warn!("mesh: slow draw {groups} groups, program {:.1} ms, draw {:.1} ms ({name})", cs / 1000.0, draw / 1000.0);
            }
            match times.iter_mut().find(|e| e.0 == name) {
                Some(e) => {
                    e.1 += 1;
                    e.2 += u64::from(groups);
                    e.3 += cs + draw;
                }
                None => times.push((name, 1, u64::from(groups), cs + draw)),
            }
        }
        static LAST: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if last.is_none_or(|l| l.elapsed().as_secs() >= 5) {
            *last = Some(std::time::Instant::now());
            times.sort_by(|a, b| b.3.total_cmp(&a.3));
            let total: f64 = times.iter().map(|e| e.3).sum();
            let top: Vec<String> = times.iter().take(6).map(|e| format!("{:.1} ms {} draws {} groups {}", e.3 / 1000.0, e.1, e.2, e.0)).collect();
            xnw_core::info!("mesh: GPU {:.1} ms in {} programs over the last 5 s; top {top:?}", total / 1000.0, times.len());
            times.clear();
        }
    }
}
