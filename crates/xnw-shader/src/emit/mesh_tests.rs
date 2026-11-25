//! XNW: the mesh fallback on real mesh programs of the title's pack
//! (`XNW_SHADER_PACK`, else `game\shaders.rpf`), on the host GPU
//! (`XNW_AGILITY`); each test skips without them.
//!
//! * `real_position_only_program_matches_its_cpu_model`: the smallest mesh
//!   program of the pack (`MS_VS_TransformRDP_PositionOnly_9bfb7f5f`, 1,024
//!   bytes of code) with synthetic meshlets, constants and a matrix; every
//!   record (counts, M0, positions, packed primitives) and every index
//!   triple is compared with a model of the program written from its
//!   disassembly (`xnw shader-info ... --disasm`).
//! * `real_skinned_and_wide_programs_keep_the_invariants`: a skinned
//!   program (`MS_VS_CascadeShadows_PedTransformPreSkinD_c3cef4f3`) and one
//!   with 21 output rows (`MS_VS_MirrorReflectionTransformD_ed3bd86e`):
//!   counts within bounds, indices in their group, written vertices finite,
//!   records identical across two runs, drawn to a capture (`out/mesh/`;
//!   empty: with these synthetic streams every vertex of a group lands on
//!   one point, so nothing has area).
//! * `real_program_cost_per_group` (ignored, timing): GPU time of the
//!   compute pass and of the draw per group on these programs.

use std::collections::HashMap;
use std::mem::ManuallyDrop;

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::host::{self, Host};
use super::mesh::{self, Layout};
use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, RootType};
use crate::rpf9::{Archive, EntryKind};
use crate::sgd5;

fn pack_path() -> Option<std::path::PathBuf> {
    let p = std::env::var_os("XNW_SHADER_PACK").map_or_else(|| std::path::PathBuf::from(r"game\shaders.rpf"), Into::into);
    p.exists().then_some(p)
}

/// The containers of the pack's mesh records named `names` (first match each).
fn containers(names: &[&str]) -> Option<Vec<Vec<u8>>> {
    let data = std::fs::read(pack_path()?).ok()?;
    let archive = Archive::parse(&data).ok()?;
    let mut out: Vec<Option<Vec<u8>>> = vec![None; names.len()];
    for (i, e) in archive.entries.iter().enumerate() {
        if e.kind != EntryKind::Sgd5 {
            continue;
        }
        for r in sgd5::records(archive.read(i)?) {
            let Some(name) = r.name else { continue };
            let Some(k) = names.iter().position(|n| name.starts_with(n)) else { continue };
            if out[k].is_some() {
                continue;
            }
            let Some(bytes) = r.unpack() else { continue };
            let ms = Container::parse(&bytes).ok().and_then(|c| c.part(FourCC::DXIL).map(|d| d.data.to_vec()));
            if ms.and_then(|d| crate::native_programs(&d).ok()).is_some_and(|p| p.iter().any(|p| p.stage == HwStage::Ms)) {
                out[k] = Some(bytes.to_vec());
            }
        }
    }
    out.into_iter().collect()
}

/// A tiny deterministic generator (xorshift).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
    fn float(&mut self) -> f32 {
        (self.next() % 2001) as f32 / 1000.0 - 1.0
    }
}

/// What the program reads: user data by title root parameter (a table's
/// first heap index, or constants), raw buffer or constant-buffer views at
/// heap indices, root SRVs by title root parameter.
#[derive(Default)]
struct Inputs {
    tables: HashMap<u8, u32>,
    constants: HashMap<u8, Vec<u32>>,
    raw: Vec<(u32, Vec<u32>)>,
    cbv: Vec<(u32, Vec<u32>)>,
    root_srvs: Vec<(u8, Vec<u32>)>,
    groups: u32,
}

struct Output {
    layout: Layout,
    records: Vec<u32>,
    indices: Vec<u32>,
    /// RGBA32F target, cleared to 1; the stub pixel shader writes 0.
    target: Vec<[f32; 4]>,
    /// GPU microseconds: compute passes (one per repeat), then the draw.
    compute_us: Vec<f64>,
    draw_us: f64,
}

const SIZE: u32 = 256;
const POISON: u32 = 0xDEAD_BEEF;

fn err(e: windows::core::Error) -> String {
    e.message().to_string()
}

fn buffer(dev: &ID3D12Device, heap: D3D12_HEAP_TYPE, size: u64, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES) -> Result<ID3D12Resource, String> {
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
    let props = D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() };
    let mut r: Option<ID3D12Resource> = None;
    // SAFETY: descriptions outlive the call.
    unsafe { dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r) }.map_err(err)?;
    r.ok_or_else(|| "no resource".into())
}

/// An upload buffer holding `data` (at least `min` bytes, zeros after).
fn upload(dev: &ID3D12Device, data: &[u32], min: u64) -> Result<ID3D12Resource, String> {
    let size = (data.len() as u64 * 4).max(min).next_multiple_of(256);
    let b = buffer(dev, D3D12_HEAP_TYPE_UPLOAD, size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
    // SAFETY: a mapped upload buffer of `size` bytes.
    unsafe {
        let mut p = std::ptr::null_mut();
        b.Map(0, None, Some(&mut p)).map_err(err)?;
        std::ptr::write_bytes(p.cast::<u8>(), 0, size as usize);
        std::ptr::copy_nonoverlapping(data.as_ptr(), p.cast::<u32>(), data.len());
        b.Unmap(0, None);
    }
    Ok(b)
}

fn read(b: &ID3D12Resource, words: usize) -> Result<Vec<u32>, String> {
    // SAFETY: a finished readback buffer of at least `words` dwords.
    unsafe {
        let mut p = std::ptr::null_mut();
        b.Map(0, None, Some(&mut p)).map_err(err)?;
        let v = std::slice::from_raw_parts(p.cast::<u32>(), words).to_vec();
        b.Unmap(0, None);
        Ok(v)
    }
}

type Transition<'a> = (&'a ID3D12Resource, D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATES);

fn transition(list: &ID3D12GraphicsCommandList, items: &[Transition]) {
    let v: Vec<D3D12_RESOURCE_BARRIER> = items
        .iter()
        .map(|(r, b, a)| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: ManuallyDrop::new(Some((*r).clone())),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: *b,
                    StateAfter: *a,
                }),
            },
        })
        .collect();
    // SAFETY: live resources.
    unsafe { list.ResourceBarrier(&v) };
    for b in v {
        // SAFETY: transitions only.
        drop(ManuallyDrop::into_inner(ManuallyDrop::into_inner(unsafe { b.Anonymous.Transition }).pResource));
    }
}

/// Runs the program's fallback over `inp.groups` groups (`repeats` compute
/// passes, then one draw) and reads everything back.
fn run(h: &Host, container: &[u8], inp: &Inputs, repeats: u32) -> Result<Output, String> {
    let (cs, layout) = mesh::translate_fallback(container, super::Options::default()).map_err(|e| e.to_string())?;
    let ps = host::stub_ps();
    let vs = mesh::passthrough_vs(&layout, Some(&ps.dxil)).map_err(|e| e.to_string())?;
    let [(root, pso), (groot, gpso)] = h.fallback_pipelines(&cs, &vs, &ps)?;
    let dev = &h.device;
    let groups = inp.groups;
    let rec_bytes = u64::from(groups) * u64::from(layout.record_bytes());
    let idx_bytes = u64::from(groups) * u64::from(layout.index_bytes());
    // SAFETY: D3D12 calls on live objects with descriptions that outlive
    // them; every resource lives until the fence wait at the end.
    unsafe {
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(err)?;
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(err)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso).map_err(err)?;
        let mut keep: Vec<ID3D12Resource> = Vec::new();

        // Views in a shader-visible heap.
        let count = inp.raw.iter().chain(&inp.cbv).map(|v| v.0 + 1).max().unwrap_or(1);
        let heap: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: count, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, ..Default::default() })
            .map_err(err)?;
        let step = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: heap.GetCPUDescriptorHandleForHeapStart().ptr + i as usize * step };
        for (i, data) in &inp.raw {
            let b = upload(dev, data, 1 << 16)?;
            let n = b.GetDesc().Width / 4;
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Buffer: D3D12_BUFFER_SRV { FirstElement: 0, NumElements: n as u32, StructureByteStride: 0, Flags: D3D12_BUFFER_SRV_FLAG_RAW } },
            };
            dev.CreateShaderResourceView(&b, Some(&desc), at(*i));
            keep.push(b);
        }
        for (i, data) in &inp.cbv {
            let b = upload(dev, data, 1 << 16)?;
            dev.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: b.GetGPUVirtualAddress(), SizeInBytes: 1 << 16 }), at(*i));
            keep.push(b);
        }
        // User data as translated shaders read it.
        let mut user = vec![0u32; cs.user_data_dwords().max(1) as usize];
        for u in &cs.user_data {
            user[u.index as usize] = match u.root {
                RootType::Table => inp.tables.get(&u.slot).copied().unwrap_or(0) * 32,
                _ => inp.constants.get(&u.slot).and_then(|c| c.get(u.dword as usize)).copied().unwrap_or(0),
            };
        }
        let mut views: Vec<(u8, bool)> = cs.root_cbvs.iter().map(|&r| (r, false)).chain(cs.root_srvs.iter().map(|&r| (r, true))).collect();
        views.sort_unstable();
        views.dedup();

        // Records and indices start poisoned, so stray and missing writes show.
        let records = buffer(dev, D3D12_HEAP_TYPE_DEFAULT, rec_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON)?;
        let indices = buffer(dev, D3D12_HEAP_TYPE_DEFAULT, idx_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON)?;
        let poison = upload(dev, &vec![POISON; (rec_bytes.max(idx_bytes) / 4) as usize], 0)?;
        list.CopyBufferRegion(&records, 0, &poison, 0, rec_bytes);
        list.CopyBufferRegion(&indices, 0, &poison, 0, idx_bytes);
        let (cd, uav, srv, ib, cpsrc) = (
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_INDEX_BUFFER,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        transition(&list, &[(&records, cd, uav), (&indices, cd, uav)]);

        let mut qh: Option<ID3D12QueryHeap> = None;
        let stamps = 2 * repeats + 2;
        dev.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: stamps, NodeMask: 0 }, &mut qh).map_err(err)?;
        let qh = qh.ok_or("no query heap")?;

        list.SetDescriptorHeaps(&[Some(heap.clone())]);
        list.SetComputeRootSignature(&root);
        list.SetPipelineState(&pso);
        list.SetComputeRoot32BitConstants(0, user.len() as u32, user.as_ptr().cast(), 0);
        for (n, (slot, is_srv)) in views.iter().enumerate() {
            let data = inp.root_srvs.iter().find(|r| r.0 == *slot).map_or(&[][..], |r| &r.1[..]);
            let b = upload(dev, data, 32 << 20)?;
            if *is_srv {
                list.SetComputeRootShaderResourceView(1 + n as u32, b.GetGPUVirtualAddress());
            } else {
                list.SetComputeRootConstantBufferView(1 + n as u32, b.GetGPUVirtualAddress());
            }
            keep.push(b);
        }
        let rp = 1 + views.len() as u32;
        list.SetComputeRootUnorderedAccessView(rp, records.GetGPUVirtualAddress());
        list.SetComputeRootUnorderedAccessView(rp + 1, indices.GetGPUVirtualAddress());
        // The dispatch's first console group (one piece, from 0).
        let first = upload(dev, &[0u32], 0)?;
        list.SetComputeRootShaderResourceView(rp + 2, first.GetGPUVirtualAddress());
        keep.push(first);
        // Root SRV lengths: past every read.
        if !cs.root_srvs.is_empty() {
            let lens = vec![32u32 << 20; cs.root_srvs.len().next_multiple_of(4)];
            list.SetComputeRoot32BitConstants(rp + 3, lens.len() as u32, lens.as_ptr().cast(), 0);
        }
        for r in 0..repeats {
            list.EndQuery(&qh, D3D12_QUERY_TYPE_TIMESTAMP, 2 * r);
            list.Dispatch(groups, 1, 1);
            list.EndQuery(&qh, D3D12_QUERY_TYPE_TIMESTAMP, 2 * r + 1);
            let b = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: ManuallyDrop::new(None) }) },
                ..Default::default()
            };
            list.ResourceBarrier(&[b]);
        }
        transition(&list, &[(&records, uav, srv), (&indices, uav, ib)]);

        // The draw: pass-through vertex shader, stub pixel shader.
        let target = {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(SIZE),
                Height: SIZE,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let clear = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_R32G32B32A32_FLOAT, Anonymous: D3D12_CLEAR_VALUE_0 { Color: [1.0; 4] } };
            let mut r: Option<ID3D12Resource> = None;
            let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
            dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_RENDER_TARGET, Some(&clear), &mut r).map_err(err)?;
            r.ok_or("no target")?
        };
        let rtv_heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() }).map_err(err)?;
        let rtv = rtv_heap.GetCPUDescriptorHandleForHeapStart();
        dev.CreateRenderTargetView(&target, None, rtv);
        list.ClearRenderTargetView(rtv, &[1.0; 4], None);
        list.OMSetRenderTargets(1, Some(&rtv), false, None);
        list.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: SIZE as f32, Height: SIZE as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
        list.RSSetScissorRects(&[RECT { left: 0, top: 0, right: SIZE as i32, bottom: SIZE as i32 }]);
        list.SetGraphicsRootSignature(&groot);
        list.SetPipelineState(&gpso);
        // Parameters: vertex user data, pixel user data, then t0.
        list.SetGraphicsRootShaderResourceView(2, records.GetGPUVirtualAddress());
        list.IASetIndexBuffer(Some(&D3D12_INDEX_BUFFER_VIEW { BufferLocation: indices.GetGPUVirtualAddress(), SizeInBytes: idx_bytes as u32, Format: DXGI_FORMAT_R32_UINT }));
        list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list.EndQuery(&qh, D3D12_QUERY_TYPE_TIMESTAMP, 2 * repeats);
        list.DrawIndexedInstanced(groups * layout.max_prims * 3, 1, 0, 0, 0);
        list.EndQuery(&qh, D3D12_QUERY_TYPE_TIMESTAMP, 2 * repeats + 1);

        // Read back.
        let rt = D3D12_RESOURCE_STATE_RENDER_TARGET;
        transition(&list, &[(&records, srv, cpsrc), (&indices, ib, cpsrc), (&target, rt, cpsrc)]);
        let rb = |n: u64| buffer(dev, D3D12_HEAP_TYPE_READBACK, n, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        let (rrec, ridx, rtex, rts) = (rb(rec_bytes)?, rb(idx_bytes)?, rb(u64::from(SIZE * SIZE * 16))?, rb(u64::from(stamps) * 8)?);
        list.CopyBufferRegion(&rrec, 0, &records, 0, rec_bytes);
        list.CopyBufferRegion(&ridx, 0, &indices, 0, idx_bytes);
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(rtex.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: DXGI_FORMAT_R32G32B32A32_FLOAT, Width: SIZE, Height: SIZE, Depth: 1, RowPitch: SIZE * 16 },
                },
            },
        };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: ManuallyDrop::new(Some(target.clone())), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        drop(ManuallyDrop::into_inner(dst.pResource));
        drop(ManuallyDrop::into_inner(src.pResource));
        list.ResolveQueryData(&qh, D3D12_QUERY_TYPE_TIMESTAMP, 0, stamps, &rts, 0);
        list.Close().map_err(err)?;
        queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(&list).map_err(err)?)]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(err)?;
        queue.Signal(&fence, 1).map_err(err)?;
        while fence.GetCompletedValue() < 1 {
            dev.GetDeviceRemovedReason().map_err(err)?;
            std::thread::yield_now();
        }
        dev.GetDeviceRemovedReason().map_err(err)?;
        let freq = queue.GetTimestampFrequency().map_err(err)? as f64;
        let ticks = read(&rts, stamps as usize * 2)?;
        let t = |i: u32| u64::from(ticks[2 * i as usize]) | u64::from(ticks[2 * i as usize + 1]) << 32;
        let us = |a: u32, b: u32| t(b).saturating_sub(t(a)) as f64 * 1e6 / freq;
        let texels = read(&rtex, (SIZE * SIZE * 4) as usize)?;
        drop(keep);
        Ok(Output {
            records: read(&rrec, (rec_bytes / 4) as usize)?,
            indices: read(&ridx, (idx_bytes / 4) as usize)?,
            target: texels.chunks(4).map(|c| [0, 1, 2, 3].map(|k| f32::from_bits(c[k]))).collect(),
            compute_us: (0..repeats).map(|r| us(2 * r, 2 * r + 1)).collect(),
            draw_us: us(2 * repeats, 2 * repeats + 1),
            layout,
        })
    }
}

impl Output {
    fn record(&self, g: u32) -> &[u32] {
        let n = (self.layout.record_bytes() / 4) as usize;
        &self.records[g as usize * n..(g as usize + 1) * n]
    }

    fn covered(&self) -> usize {
        self.target.iter().filter(|t| t[0] == 0.0).count()
    }

    /// The target as a binary PPM (covered pixels black).
    fn capture(&self, path: &std::path::Path) {
        let mut out = format!("P6 {SIZE} {SIZE} 255\n").into_bytes();
        for t in &self.target {
            let v = (t[0].clamp(0.0, 1.0) * 255.0) as u8;
            out.extend([v, v, v]);
        }
        let _ = std::fs::create_dir_all(path.parent().expect("a directory"));
        let _ = std::fs::write(path, out);
    }

    /// The invariants every program's records keep: counts within the
    /// maxima and M0 consistent with them, index triples inside their group
    /// and below the vertex count (or degenerate), primitives past the count
    /// degenerate, vertices below the count written (not poison) with a
    /// finite position. Returns the vertices and primitives requested.
    fn check_invariants(&self, groups: u32) -> Result<(u32, u32), String> {
        let l = &self.layout;
        let pos = l.position_row().ok_or("no position row")?;
        let (mut verts, mut prims) = (0, 0);
        for g in 0..groups {
            let r = self.record(g);
            let (nv, np, m0) = (r[0], r[1], r[2]);
            if nv > l.max_vertices || np > l.max_prims || m0 != (np << 12 | nv) {
                return Err(format!("group {g}: counts {nv} {np} m0 {m0:#x}"));
            }
            verts += nv;
            prims += np;
            for v in 0..nv {
                let at = (16 + v * l.vertex_stride() + 16 * pos) as usize / 4;
                for c in 0..4 {
                    let x = r[at + c];
                    if x == POISON || !f32::from_bits(x).is_finite() {
                        return Err(format!("group {g} vertex {v}: position {c} {x:#x}"));
                    }
                }
            }
            let base = g * l.max_vertices;
            for p in 0..l.max_prims {
                let t = &self.indices[((g * l.max_prims + p) * 3) as usize..][..3];
                let degenerate = t.iter().all(|&i| i == base);
                if !degenerate && (p >= np || t.iter().any(|&i| i < base || i >= base + nv)) {
                    return Err(format!("group {g} primitive {p}: {t:?} with {nv} vertices, {np} primitives"));
                }
            }
        }
        Ok((verts, prims))
    }
}

fn gpu() -> Option<Host> {
    let h = Host::for_tests();
    if h.is_none() {
        eprintln!("skipped: no SM 6.6 device (set XNW_AGILITY)");
    }
    h
}

// ---- The smallest program, against a model of it ----

const POSITION_ONLY: &str = "MS_VS_TransformRDP_PositionOnly_9bfb7f5f";

/// Synthetic meshlets for [`POSITION_ONLY`] and what it must make of them.
///
/// The program (byte offsets of its disassembly): with bit 28 of root
/// constant 0 clear (0x64: set reads the meshlet number from a table the
/// second constant names), meshlet `m = group` if `group < count`, else 0, where
/// `count` is dword 0 of the raw buffer at table entry 1 (& 0x1FFFF).
/// Header dwords at byte `m * 16 + 12`: `d0` bits 0-6 vertices - 1, 7-13
/// primitives - 1 (bit 15, as `<< 1 & 0x10000`, and `d1 >> 16` the first
/// primitive word); primitive words at byte `(count * 16 | 12) + 4 * (first
/// + p)`, three 7-bit indices (0x304-0x320: a primitive naming a vertex at
/// or past the count is culled, 0x80000000). Every vertex's position:
/// `x * row0 + y * row1 + z * row2 + row3` (one multiply, two fused
/// multiply-adds, one add per column) with `(x, y, z)` at byte 0x30 of
/// table entry 3 and the rows at byte 0x160 of table entry 4.
struct PositionOnly {
    inputs: Inputs,
    meshlets: Vec<(u32, u32, Vec<u32>)>,
    position: [f32; 4],
}

fn position_only_inputs(groups: u32) -> PositionOnly {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let count = 61u32;
    let mut buf = vec![0u32; 1 << 14];
    buf[0] = count;
    let words_at = ((count * 16) | 12) / 4;
    let mut next = 0u32;
    let mut meshlets = Vec::new();
    for m in 0..count {
        let (nv, np) = (1 + rng.below(128), 1 + rng.below(128));
        let h = (m * 16 + 12) / 4;
        buf[h as usize] = (nv - 1) | (np - 1) << 7;
        buf[h as usize + 1] = next << 16;
        let mut words = Vec::new();
        for _ in 0..np {
            // Mostly valid triangles; one in eight may name a vertex past nv.
            let lim = if rng.below(8) == 0 { 128 } else { nv };
            let w = rng.below(lim) | rng.below(lim) << 7 | rng.below(lim) << 14;
            buf[(words_at + next) as usize] = w;
            words.push(w);
            next += 1;
        }
        meshlets.push((nv, np, words));
    }
    let mut vec_cb = vec![0u32; 64];
    let xyz = [rng.float(), rng.float(), rng.float()];
    for (k, x) in xyz.iter().enumerate() {
        vec_cb[0x30 / 4 + k] = x.to_bits();
    }
    let mut mat_cb = vec![0u32; 0x200 / 4];
    let mat: Vec<f32> = (0..16).map(|_| rng.float()).collect();
    for (k, m) in mat.iter().enumerate() {
        mat_cb[0x160 / 4 + k] = m.to_bits();
    }
    let position = [0, 1, 2, 3].map(|j| mat[8 + j].mul_add(xyz[2], mat[4 + j].mul_add(xyz[1], mat[j] * xyz[0])) + mat[12 + j]);
    let mut inputs = Inputs { groups, ..Default::default() };
    inputs.tables.insert(1, 0);
    inputs.constants.insert(3, vec![0, 0]);
    inputs.raw.push((1, buf));
    inputs.raw.push((3, vec_cb));
    inputs.cbv.push((4, mat_cb));
    PositionOnly { inputs, meshlets, position }
}

#[test]
fn real_position_only_program_matches_its_cpu_model() {
    let Some(h) = gpu() else { return };
    let Some(c) = containers(&[POSITION_ONLY]) else {
        eprintln!("skipped: no shader pack");
        return;
    };
    let groups = 80;
    let model = position_only_inputs(groups);
    let out = run(&h, &c[0], &model.inputs, 1).unwrap();
    let l = &out.layout;
    assert_eq!((l.rows, l.max_vertices, l.max_prims, l.record_bytes()), (1, 128, 128, 2576));
    let mut checked = (0, 0, 0);
    for g in 0..groups {
        let m = if g < model.meshlets.len() as u32 { g } else { 0 };
        let (nv, np, words) = &model.meshlets[m as usize];
        let r = out.record(g);
        assert_eq!(&r[..3], &[*nv, *np, np << 12 | nv], "group {g}: header");
        for v in 0..128 {
            let at = (16 + v * 16) as usize / 4;
            let got = &r[at..at + 4];
            if v < *nv {
                assert_eq!(got, &model.position.map(f32::to_bits), "group {g} vertex {v}");
                checked.0 += 1;
            } else {
                assert_eq!(got, &[POISON; 4], "group {g} vertex {v}: not exported, not written");
            }
        }
        let base = g * 128;
        for p in 0..128u32 {
            let t = &out.indices[((g * 128 + p) * 3) as usize..][..3];
            let word = r[(l.prim_offset() / 4 + p) as usize];
            if p < *np {
                let w = words[p as usize];
                let i = [w & 0x7F, w >> 7 & 0x7F, w >> 14 & 0x7F];
                let culled = i.iter().any(|&i| i >= *nv);
                let packed = if culled { 0x8000_0000 } else { i[0] | i[1] << 10 | i[2] << 20 };
                assert_eq!(word, packed, "group {g} primitive {p}: packed word");
                let want = if culled { [base; 3] } else { i.map(|i| base + i) };
                assert_eq!(t, &want, "group {g} primitive {p}: indices");
                checked.1 += 1;
                checked.2 += usize::from(culled);
            } else {
                assert_eq!(word, POISON, "group {g} primitive {p}: not exported");
                assert_eq!(t, &[base; 3], "group {g} primitive {p}: degenerate");
            }
        }
    }
    out.check_invariants(groups).unwrap();
    // Every vertex of the program is the same point: nothing has area.
    assert_eq!(out.covered(), 0);
    eprintln!("{POSITION_ONLY}: {groups} groups, {} vertices and {} primitives ({} culled) equal to the model", checked.0, checked.1, checked.2);
}

// ---- A skinned program and a 21-row one: invariants and a capture ----

const SKINNED: &str = "MS_VS_CascadeShadows_PedTransformPreSkinD_c3cef4f3";
const WIDE: &str = "MS_VS_MirrorReflectionTransformD_ed3bd86e";

/// Inputs for the other programs: [`position_only_inputs`]'s meshlet
/// buffer at every raw view of stride 1 the program reads through table 1
/// (where the smallest program reads it),
/// small random floats in its other views (bone matrices, constants, vertex
/// streams), zeros in root SRVs and root constants.
fn generic_inputs(container: &[u8], groups: u32, seed: u64) -> Inputs {
    let base = position_only_inputs(groups);
    let meshlets = base.inputs.raw[0].1.clone();
    let mut rng = Rng(seed);
    let c = Container::parse(container).unwrap();
    let p = crate::native_programs(c.part(FourCC::DXIL).unwrap().data).unwrap().into_iter().find(|p| p.stage == HwStage::Ms).unwrap();
    let mut inputs = Inputs { groups, ..Default::default() };
    let floats = |rng: &mut Rng, n: usize| (0..n).map(|_| (rng.float() * 0.5).to_bits()).collect::<Vec<u32>>();
    for u in &p.used {
        match u.root {
            RootType::Table => {
                // Table k starts at heap index 64 * k.
                let first = 64 * u32::from(u.slot);
                inputs.tables.insert(u.slot, first);
                let i = first + u.table_offset_dwords / 8;
                match u.kind {
                    crate::native::ResourceType::Cbv => inputs.cbv.push((i, floats(&mut rng, 1 << 14))),
                    _ if u.stride == 1 && u.slot == 1 => inputs.raw.push((i, meshlets.clone())),
                    _ => inputs.raw.push((i, floats(&mut rng, 1 << 16))),
                }
            }
            RootType::Constants => {
                inputs.constants.insert(u.slot, vec![0; 4]);
            }
            _ => inputs.root_srvs.push((u.slot, vec![0; 1 << 16])),
        }
    }
    inputs
}

#[test]
fn real_skinned_and_wide_programs_keep_the_invariants() {
    let Some(h) = gpu() else { return };
    let Some(c) = containers(&[SKINNED, WIDE]) else {
        eprintln!("skipped: no shader pack");
        return;
    };
    let groups = 80;
    for (name, container) in [SKINNED, WIDE].iter().zip(&c) {
        let inputs = generic_inputs(container, groups, 7);
        let a = run(&h, container, &inputs, 1).unwrap_or_else(|e| panic!("{name}: {e}"));
        let b = run(&h, container, &inputs, 1).unwrap();
        let (verts, prims) = a.check_invariants(groups).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(a.records == b.records && a.indices == b.indices && a.target == b.target, "{name}: two runs differ");
        assert!(verts > 0 && prims > 0, "{name}: the program exported nothing");
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out/mesh").join(format!("{name}.ppm"));
        a.capture(&path);
        eprintln!(
            "{name}: rows {}, record {} B; {groups} groups, {verts} vertices, {prims} primitives; {} pixels covered; capture {}",
            a.layout.rows,
            a.layout.record_bytes(),
            a.covered(),
            path.display()
        );
    }
}

#[test]
#[ignore = "timing: run alone with --ignored --nocapture"]
fn real_program_cost_per_group() {
    let Some(h) = gpu() else { return };
    let Some(c) = containers(&[POSITION_ONLY, SKINNED, WIDE]) else { return };
    for (name, container) in [POSITION_ONLY, SKINNED, WIDE].iter().zip(&c) {
        for groups in [1u32, 256, 4096] {
            let inputs = if *name == POSITION_ONLY { position_only_inputs(groups).inputs } else { generic_inputs(container, groups, 7) };
            let mut best = (f64::MAX, f64::MAX);
            for _ in 0..3 {
                let o = run(&h, container, &inputs, 50).unwrap();
                let mut t = o.compute_us.clone();
                t.sort_by(f64::total_cmp);
                best = (best.0.min(t[t.len() / 2]), best.1.min(o.draw_us));
            }
            let rows = mesh::record_layout(container).unwrap().rows;
            eprintln!(
                "{name} ({rows} rows): {groups:5} groups compute {:8.1} us ({:.3} us/group), draw {:8.1} us ({:.3} us/group)",
                best.0,
                best.0 / f64::from(groups),
                best.1,
                best.1 / f64::from(groups)
            );
        }
    }
}

/// VIctory: a mesh fallback program reading through a root SRV
/// past the bytes its lengths say are behind it reads offset 0 instead, so
/// a read 16 MiB past a 64 KiB buffer (unmapped memory on the host) neither
/// faults nor hangs; reads inside are untouched.
#[test]
fn root_srv_reads_past_their_length_stay_inside() {
    use super::bitcode::{Attr, Cast, Term};
    let Some(h) = gpu() else { return };
    let mut s = super::shader::Shader::new(false);
    s.fallback = true;
    s.threads = Some([1, 1, 1]);
    let b = s.entry;
    let view = s.root_srv_handle(0);
    let out = s.raw_view(true, mesh::RECORD_REGISTER);
    let offsets = [0u32, 4096, 65520, 65536, 16 << 20];
    for (k, &at) in offsets.iter().enumerate() {
        let (u, mask, al) = (s.m.undef(s.t.i32), s.m.i8(1), s.m.i32(4));
        let a = s.m.i32(at);
        let r = s.op(b, "rawBufferLoad.f32", Attr::ReadOnly, s.t.res_ret, 139, &[view, a, u, mask, al]);
        let x = s.m.extract(b, r, 0);
        let x = s.m.cast(b, Cast::Bitcast, x, s.t.i32);
        let o = s.m.i32(4 * k as u32);
        let u = s.m.undef(s.t.i32);
        let (m1, al) = (s.m.i8(1), s.m.i32(4));
        s.op(b, "rawBufferStore.i32", Attr::NoUnwind, s.t.void, 140, &[out, o, u, x, u, u, u, m1, al]);
    }
    s.m.terminate(b, Term::Ret);
    let cs = super::HostShader { stage: HwStage::Ms, dxil: s.finish(), user_data: Vec::new(), samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: Vec::new(), root_srvs: vec![0], root_uavs: Vec::new(), address_map: false, descriptor_words: false };
    let root = h.root_signature_records(&[(&cs, D3D12_SHADER_VISIBILITY_ALL)], &[(true, mesh::RECORD_REGISTER)]).unwrap();
    let pso = h.compute_pipeline_with(&root, &cs.dxil).unwrap();
    let dev = &h.device;
    // Dword i holds i + 1, over 64 KiB.
    let data: Vec<u32> = (1..=16384).collect();
    let src = upload(dev, &data, 0).unwrap();
    let dst = buffer(dev, D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS).unwrap();
    let back = buffer(dev, D3D12_HEAP_TYPE_READBACK, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST).unwrap();
    // SAFETY: D3D12 calls on live objects; every resource lives to the wait.
    unsafe {
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso).unwrap();
        list.SetComputeRootSignature(&root);
        // Parameters: user data, the root SRV, the record UAV, the lengths.
        list.SetComputeRootShaderResourceView(1, src.GetGPUVirtualAddress());
        list.SetComputeRootUnorderedAccessView(2, dst.GetGPUVirtualAddress());
        let lens = [65536u32, 0, 0, 0];
        list.SetComputeRoot32BitConstants(3, 4, lens.as_ptr().cast(), 0);
        list.Dispatch(1, 1, 1);
        transition(&list, &[(&dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        list.CopyResource(&back, &dst);
        list.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(&list).unwrap())]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.Signal(&fence, 1).unwrap();
        while fence.GetCompletedValue() < 1 {
            dev.GetDeviceRemovedReason().unwrap();
            std::thread::yield_now();
        }
        dev.GetDeviceRemovedReason().unwrap();
    }
    // Inside: their dwords; the last 16 bytes still inside; past: offset 0.
    assert_eq!(read(&back, 5).unwrap(), vec![1, 1025, 16381, 1, 1]);
}
/// A minimal BLAKE3 (hash mode), for matching victory-fork's shader
/// hashes (the first 16 hex digits of BLAKE3 of the container) to ours.
fn blake3(input: &[u8]) -> [u8; 32] {
    const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];
    const PERM: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
    const START: u32 = 1;
    const END: u32 = 2;
    const PARENT: u32 = 4;
    const ROOT: u32 = 8;
    fn compress(cv: &[u32; 8], m: &[u32; 16], ctr: u64, len: u32, flags: u32) -> [u32; 16] {
        let mut s = [cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7], IV[0], IV[1], IV[2], IV[3], ctr as u32, (ctr >> 32) as u32, len, flags];
        let mut m = *m;
        let g = |s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32| {
            s[a] = s[a].wrapping_add(s[b]).wrapping_add(x);
            s[d] = (s[d] ^ s[a]).rotate_right(16);
            s[c] = s[c].wrapping_add(s[d]);
            s[b] = (s[b] ^ s[c]).rotate_right(12);
            s[a] = s[a].wrapping_add(s[b]).wrapping_add(y);
            s[d] = (s[d] ^ s[a]).rotate_right(8);
            s[c] = s[c].wrapping_add(s[d]);
            s[b] = (s[b] ^ s[c]).rotate_right(7);
        };
        for r in 0..7 {
            g(&mut s, 0, 4, 8, 12, m[0], m[1]);
            g(&mut s, 1, 5, 9, 13, m[2], m[3]);
            g(&mut s, 2, 6, 10, 14, m[4], m[5]);
            g(&mut s, 3, 7, 11, 15, m[6], m[7]);
            g(&mut s, 0, 5, 10, 15, m[8], m[9]);
            g(&mut s, 1, 6, 11, 12, m[10], m[11]);
            g(&mut s, 2, 7, 8, 13, m[12], m[13]);
            g(&mut s, 3, 4, 9, 14, m[14], m[15]);
            if r < 6 {
                m = std::array::from_fn(|i| m[PERM[i]]);
            }
        }
        for i in 0..8 {
            s[i] ^= s[i + 8];
            s[i + 8] ^= cv[i];
        }
        s
    }
    let words = |b: &[u8]| -> [u32; 16] {
        let mut p = [0u8; 64];
        p[..b.len()].copy_from_slice(b);
        std::array::from_fn(|i| u32::from_le_bytes(p[4 * i..4 * i + 4].try_into().unwrap()))
    };
    let first8 = |s: [u32; 16]| -> [u32; 8] { std::array::from_fn(|i| s[i]) };
    // A node's output: (chaining value in, block, counter, length, flags).
    type Out = ([u32; 8], [u32; 16], u64, u32, u32);
    let chunk = |i: u64, c: &[u8]| -> Out {
        let blocks: Vec<&[u8]> = if c.is_empty() { vec![&[][..]] } else { c.chunks(64).collect() };
        let mut cv = IV;
        let last = blocks.len() - 1;
        for (j, b) in blocks.iter().enumerate() {
            let flags = if j == 0 { START } else { 0 } | if j == last { END } else { 0 };
            if j < last {
                cv = first8(compress(&cv, &words(b), i, 64, flags));
            } else {
                return (cv, words(b), i, b.len() as u32, flags);
            }
        }
        unreachable!()
    };
    let cv_of = |o: &Out| first8(compress(&o.0, &o.1, o.2, o.3, o.4));
    let parent = |l: [u32; 8], r: [u32; 8]| -> Out {
        let mut m = [0u32; 16];
        m[..8].copy_from_slice(&l);
        m[8..].copy_from_slice(&r);
        (IV, m, 0, 64, PARENT)
    };
    let chunks: Vec<&[u8]> = if input.is_empty() { vec![&[][..]] } else { input.chunks(1024).collect() };
    let n = chunks.len();
    let mut stack: Vec<[u32; 8]> = Vec::new();
    for (i, c) in chunks[..n - 1].iter().enumerate() {
        let mut cv = cv_of(&chunk(i as u64, c));
        let mut total = i as u64 + 1;
        while total & 1 == 0 {
            cv = cv_of(&parent(stack.pop().unwrap(), cv));
            total >>= 1;
        }
        stack.push(cv);
    }
    let mut out = chunk(n as u64 - 1, chunks[n - 1]);
    while let Some(l) = stack.pop() {
        out = parent(l, cv_of(&out));
    }
    let s = compress(&out.0, &out.1, out.2, out.3, out.4 | ROOT);
    let mut h = [0u8; 32];
    for i in 0..8 {
        h[4 * i..4 * i + 4].copy_from_slice(&s[i].to_le_bytes());
    }
    h
}

#[test]
fn blake3_matches_its_test_vectors() {
    let hex = |h: [u8; 32]| h.iter().map(|b| format!("{b:02x}")).collect::<String>();
    assert_eq!(hex(blake3(b"")), "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262");
    assert_eq!(hex(blake3(b"abc")), "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85");
    // The official vectors (test_vectors.json: bytes 0..250 repeating).
    let input = |n: u32| -> Vec<u8> { (0..n).map(|i| (i % 251) as u8).collect() };
    for (n, want) in [(1024, "42214739f095a406"), (1025, "d00278ae47eb27b3"), (2048, "e776b6028c7cd22a")] {
        let h = hex(blake3(&input(n)));
        println!("blake3 of {n} bytes: {h}");
        assert!(h.starts_with(want), "{n}: {h}");
    }
}

/// Which of the pack's containers victory-fork's skip list names
/// (`VICTORY_SKIP_PIPELINES`: the first 16 hex digits of BLAKE3 of the
/// container), with our hash (`xnw_core::hash128`, as the log and
/// `out\shader-names.tsv` name them). Run by hand:
/// `cargo test -p xnw-shader fork_skip_hashes -- --ignored --nocapture`.
#[test]
#[ignore]
fn fork_skip_hashes() {
    let wanted = ["eeed82b877b44b5f", "269b802f878c24e1", "b83629b90cd2b7bc"];
    let data = std::fs::read(pack_path().expect("the pack")).unwrap();
    let archive = Archive::parse(&data).unwrap();
    for (i, e) in archive.entries.iter().enumerate() {
        if e.kind != EntryKind::Sgd5 {
            continue;
        }
        let Some(raw) = archive.read(i) else { continue };
        for r in sgd5::records(raw) {
            let Some(bytes) = r.unpack() else { continue };
            let h: String = blake3(&bytes)[..8].iter().map(|b| format!("{b:02x}")).collect();
            if wanted.contains(&h.as_str()) {
                println!("{h} ours {:032x} name {:?} bytes {}", xnw_core::hash128(&[&bytes]), r.name, bytes.len());
            }
        }
    }
}