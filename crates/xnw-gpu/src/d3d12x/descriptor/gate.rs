//! Shader-visible descriptor writes never overlap a running submission
//!
//! WARP 1.0.22 reads a view-heap slot as several loads; a slot rewritten
//! while a submission reads it, even as the very same view, can be seen
//! half written and the device is removed (`reading address 0x388`; lane
//! ref2, `xnw-shader` test `null_kind_one` kinds 13-16, 7 of 7 runs).
//! Descriptor writes into the host heaps behind the title's heaps come from
//! the title's recording threads while the submit thread waits on the GPU
//! (`host::Host::execute`), and a hardware GPU has no promise either: the
//! D3D12 rule is that a descriptor is not changed while work that may read
//! it runs.
//!
//! So while a submission runs, a write into a registered (shader-visible)
//! heap is made into a slot of a CPU-only staging heap instead and copied
//! into place when the submission has completed, before the next one is
//! made (all submissions go through `execute`, one at a time). Writes when
//! no submission runs go straight to the slot. A write therefore takes
//! effect for work submitted after it, which is what the console gives the
//! title: its descriptor writes are memory the GPU reads when the work runs.
//!
//! `XNW_SLOT_GATE=off` writes straight to the slot always (the old way).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface as _;

/// Gating is on (`XNW_SLOT_GATE`, default on).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("XNW_SLOT_GATE").as_deref(), Ok("off" | "0")))
}

/// Taken shared by every gated write for the length of one D3D12 call, and
/// exclusively by the submit thread to start and to end a submission's
/// window: no write is half done when a submission starts or ends.
static GATE: RwLock<()> = RwLock::new(());
/// A submission runs (changed only under [`GATE`] held exclusively).
static IN_FLIGHT: AtomicBool = AtomicBool::new(false);

/// Writes made into staging slots, staging copies made at submission ends,
/// writes past the staging space made in place, and the microseconds spent
/// copying staged writes into place.
static STAGED: AtomicU64 = AtomicU64::new(0);
static COPIES: AtomicU64 = AtomicU64::new(0);
static UNSTAGED: AtomicU64 = AtomicU64::new(0);
static DRAIN_US: AtomicU64 = AtomicU64::new(0);

/// A gated heap: its CPU range and type (the heap is held, so a write
/// staged for it lands in a live heap even if its owner drops it first).
struct Heap {
    start: usize,
    end: usize,
    kind: D3D12_DESCRIPTOR_HEAP_TYPE,
    heap: ID3D12DescriptorHeap,
}

/// A CPU-only staging heap: start, slots, bytes per slot, slots used.
struct Stage {
    _heap: ID3D12DescriptorHeap,
    start: usize,
    len: u32,
    step: usize,
    used: u32,
}

/// A staged write: `n` slots at `src` (staging) go to `dst`; `order`: when
/// it was last made (staged writes go into place in that order).
struct Write {
    order: u64,
    dst: usize,
    src: usize,
    n: u32,
    kind: D3D12_DESCRIPTOR_HEAP_TYPE,
    _heap: ID3D12DescriptorHeap,
}

struct Pending {
    heaps: Vec<Heap>,
    /// Staging heaps for views (0) and samplers (1), kept for reuse.
    stages: [Vec<Stage>; 2],
    /// Staged writes, one per (slot, count).
    writes: Vec<Write>,
    /// Per (slot, count) staged: its entry in `writes`, so a slot written
    /// many times in one submission takes one staging range.
    staged: Option<std::collections::HashMap<(usize, u32), usize>>,
    /// Writes staged so far (the next `order`).
    made: u64,
}

// SAFETY: D3D12 descriptor heaps are free-threaded; the lists are only
// touched under the mutex.
unsafe impl Send for Pending {}

static PENDING: Mutex<Pending> = Mutex::new(Pending { heaps: Vec::new(), stages: [Vec::new(), Vec::new()], writes: Vec::new(), staged: None, made: 0 });

/// Slots per staging heap: views, samplers (a CPU-only heap of either kind).
const STAGE_SLOTS: [u32; 2] = [1 << 16, 1 << 12];
/// Staging slots at most (views, samplers): past them a write waits for
/// the submission to end and goes in place (counted).
const STAGE_MAX: [u32; 2] = [1 << 22, 1 << 16];

fn pending() -> std::sync::MutexGuard<'static, Pending> {
    PENDING.lock().unwrap_or_else(|e| e.into_inner())
}

/// Gates writes into `heap` (shader-visible) from now on.
pub(crate) fn register(heap: &ID3D12DescriptorHeap, kind: D3D12_DESCRIPTOR_HEAP_TYPE, slots: u32, step: u32) {
    // SAFETY: a live heap.
    let start = unsafe { heap.GetCPUDescriptorHandleForHeapStart().ptr };
    let end = start + slots as usize * step as usize;
    pending().heaps.push(Heap { start, end, kind, heap: heap.clone() });
}

/// Stops gating the heap starting at `start` (its owner drops it).
pub(crate) fn forget(start: usize) {
    pending().heaps.retain(|h| h.start != start);
}

impl Pending {
    /// `n` staging slots for a write of `n` slots at `dst`, the write
    /// recorded; none when `dst` is not in a gated heap or staging is full.
    fn stage(&mut self, dst: usize, n: u32) -> Option<usize> {
        let h = self.heaps.iter().find(|h| (h.start..h.end).contains(&dst))?;
        let (kind, heap) = (h.kind, h.heap.clone());
        self.made += 1;
        if let Some(&index) = self.staged.get_or_insert_with(Default::default).get(&(dst, n)) {
            let w = &mut self.writes[index];
            w.order = self.made;
            return Some(w.src);
        }
        let which = usize::from(kind == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        let stages = &mut self.stages[which];
        let at = match stages.iter_mut().find(|s| s.len - s.used >= n) {
            Some(s) => s,
            None => {
                let total: u32 = stages.iter().map(|s| s.len).sum();
                let len = STAGE_SLOTS[which].max(n);
                if total + len > STAGE_MAX[which] {
                    return None;
                }
                let device = &super::super::host::host()?.device;
                // SAFETY: plain heap creation.
                let heap: ID3D12DescriptorHeap = unsafe { device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: kind, NumDescriptors: len, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE, NodeMask: 0 }) }.ok()?;
                // SAFETY: a live heap.
                let (start, step) = unsafe { (heap.GetCPUDescriptorHandleForHeapStart().ptr, device.GetDescriptorHandleIncrementSize(kind) as usize) };
                stages.push(Stage { _heap: heap, start, len, step, used: 0 });
                stages.last_mut().expect("pushed")
            }
        };
        let src = at.start + at.used as usize * at.step;
        at.used += n;
        self.staged.get_or_insert_with(Default::default).insert((dst, n), self.writes.len());
        self.writes.push(Write { order: self.made, dst, src, n, kind, _heap: heap });
        Some(src)
    }
}

/// Makes a write of `n` slots at `at` with `write`, which is handed the
/// handle to write: `at`, or a staging slot while a submission runs.
/// `write` is one D3D12 call (no lock of ours is taken inside it).
pub(crate) fn put(at: D3D12_CPU_DESCRIPTOR_HANDLE, n: u32, write: impl FnOnce(D3D12_CPU_DESCRIPTOR_HANDLE)) {
    if !on() {
        return write(at);
    }
    loop {
        let _shared = GATE.read().unwrap_or_else(|e| e.into_inner());
        if !IN_FLIGHT.load(Ordering::Relaxed) {
            return write(at);
        }
        let mut p = pending();
        if !p.heaps.iter().any(|h| (h.start..h.end).contains(&at.ptr)) {
            // A CPU-only heap: no submission reads it.
            drop(p);
            return write(at);
        }
        if let Some(ptr) = p.stage(at.ptr, n) {
            drop(p);
            STAGED.fetch_add(u64::from(n), Ordering::Relaxed);
            // Several threads may write one staging range (the same slot):
            // the copy into place waits for every write under way.
            return write(D3D12_CPU_DESCRIPTOR_HANDLE { ptr });
        }
        // Staging full (or no device): wait for the submission's end.
        drop(p);
        drop(_shared);
        UNSTAGED.fetch_add(1, Ordering::Relaxed);
        while IN_FLIGHT.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
    }
}

/// A submission's window: from [`begin`] until dropped (after its wait).
pub(crate) struct Flight(());

/// Marks a submission as running; writes from here until the returned
/// window is dropped are staged. None when gating is off.
pub(crate) fn begin() -> Option<Flight> {
    if !on() {
        return None;
    }
    let _only = GATE.write().unwrap_or_else(|e| e.into_inner());
    IN_FLIGHT.store(true, Ordering::Relaxed);
    Some(Flight(()))
}

impl Drop for Flight {
    /// The submission completed: staged writes go into place, in the order
    /// they were made, before any write goes straight in again.
    fn drop(&mut self) {
        let _only = GATE.write().unwrap_or_else(|e| e.into_inner());
        let started = std::time::Instant::now();
        let mut p = pending();
        if !p.writes.is_empty()
            && let Some(host) = super::super::host::host()
        {
            let mut copies = 0u64;
            // In the order last made: a later write over a range wins.
            p.writes.sort_unstable_by_key(|w| w.order);
            for w in p.writes.drain(..) {
                // SAFETY: a staging range of ours and a slot range of a heap held by the write.
                unsafe { host.device.CopyDescriptorsSimple(w.n, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: w.dst }, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: w.src }, w.kind) };
                copies += 1;
            }
            COPIES.fetch_add(copies, Ordering::Relaxed);
            DRAIN_US.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        p.writes.clear();
        if let Some(staged) = p.staged.as_mut() {
            staged.clear();
        }
        for s in p.stages.iter_mut().flatten() {
            s.used = 0;
        }
        IN_FLIGHT.store(false, Ordering::Relaxed);
    }
}

/// For the memory line: staged writes, copies into place and their time,
/// writes that waited for a submission's end for want of staging space.
pub(crate) fn stats_text() -> String {
    format!(
        "slot gate {}: {} writes staged, {} copies into place in {} ms, {} waited",
        if on() { "on" } else { "off" },
        STAGED.load(Ordering::Relaxed),
        COPIES.load(Ordering::Relaxed),
        DRAIN_US.load(Ordering::Relaxed) / 1000,
        UNSTAGED.load(Ordering::Relaxed)
    )
}

/// Gated `CreateShaderResourceView`.
pub(crate) unsafe fn srv<P0: windows::core::Param<ID3D12Resource>>(device: &ID3D12Device, resource: P0, desc: Option<*const D3D12_SHADER_RESOURCE_VIEW_DESC>, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    // XNW: the resource as a plain reference, for the slot watch.
    // SAFETY: the caller's; the value lives until the write is made.
    let value = unsafe { resource.param() };
    let raw = value.abi();
    // SAFETY: a borrowed interface pointer (or null) held by `value`.
    let r = unsafe { ID3D12Resource::from_raw_borrowed(&raw) };
    // SAFETY: the caller's description, read once.
    super::slotwatch::write(at.ptr, 1, "SRV", |_| format!("{}, {}", super::slotwatch::srv_text(desc.and_then(|d| unsafe { d.as_ref() })), super::slotwatch::resource_text(r)));
    // SAFETY: the caller's.
    put(at, 1, |h| unsafe { device.CreateShaderResourceView(r, desc, h) });
}

/// Gated `CreateUnorderedAccessView`.
pub(crate) unsafe fn uav<P0: windows::core::Param<ID3D12Resource>, P1: windows::core::Param<ID3D12Resource>>(
    device: &ID3D12Device,
    resource: P0,
    counter: P1,
    desc: Option<*const D3D12_UNORDERED_ACCESS_VIEW_DESC>,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    // SAFETY: the caller's; the values live until the write is made.
    let (value, cvalue) = unsafe { (resource.param(), counter.param()) };
    let (raw, craw) = (value.abi(), cvalue.abi());
    // SAFETY: borrowed interface pointers (or null) held by the values.
    let (r, c) = unsafe { (ID3D12Resource::from_raw_borrowed(&raw), ID3D12Resource::from_raw_borrowed(&craw)) };
    // SAFETY: the caller's description, read once.
    super::slotwatch::write(at.ptr, 1, "UAV", |_| format!("{}, {}{}", super::slotwatch::uav_text(desc.and_then(|d| unsafe { d.as_ref() })), super::slotwatch::resource_text(r), if c.is_some() { ", with a counter" } else { "" }));
    // SAFETY: the caller's.
    put(at, 1, |h| unsafe { device.CreateUnorderedAccessView(r, c, desc, h) });
}

/// Gated `CreateConstantBufferView`.
pub(crate) unsafe fn cbv(device: &ID3D12Device, desc: Option<*const D3D12_CONSTANT_BUFFER_VIEW_DESC>, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    super::slotwatch::write(at.ptr, 1, "CBV", |_| {
        // SAFETY: the caller's description, read once.
        match desc.and_then(|d| unsafe { d.as_ref() }) {
            Some(d) if d.BufferLocation != 0 => format!("location {:#x} size {}", d.BufferLocation, d.SizeInBytes),
            _ => "null (location 0)".into(),
        }
    });
    // SAFETY: the caller's.
    put(at, 1, |h| unsafe { device.CreateConstantBufferView(desc, h) });
}

/// Gated `CreateSampler`.
pub(crate) unsafe fn sampler(device: &ID3D12Device, desc: *const D3D12_SAMPLER_DESC, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    // SAFETY: the caller's.
    put(at, 1, |h| unsafe { device.CreateSampler(desc, h) });
}

/// Gated `CopyDescriptorsSimple` (the source is a CPU-only heap).
pub(crate) unsafe fn copy(device: &ID3D12Device, n: u32, dst: D3D12_CPU_DESCRIPTOR_HANDLE, src: D3D12_CPU_DESCRIPTOR_HANDLE, kind: D3D12_DESCRIPTOR_HEAP_TYPE) {
    if kind == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV {
        super::slotwatch::write(dst.ptr, n, "copy", |k| {
            // SAFETY: plain query.
            let step = unsafe { device.GetDescriptorHandleIncrementSize(kind) };
            let from = src.ptr + k as usize * step as usize;
            format!("from CPU slot {from:#x}: {}", super::slotwatch::warp_words(from, step))
        });
    }
    // SAFETY: the caller's.
    put(dst, n, |h| unsafe { device.CopyDescriptorsSimple(n, h, src, kind) });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::d3d12x;
    use crate::d3d12x::descriptor::host::HostHeap;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::core::{Interface, s};

    /// The repro (`xnw-shader` `null_kind_one`, kinds 13-16, mode 30)
    /// through our layer: a long dispatch reads view slots 0 and 2 of a host
    /// heap, lane by lane, while another thread rewrites them through the
    /// gated writes (as the same view, kind 15). Gated, the submission runs
    /// to its end on WARP 1.0.22; `XNW_SLOT_GATE=off` is the control, which
    /// removed the device in 7 of 7 runs in ref2. The tables are
    /// DESCRIPTORS_VOLATILE as the layer's (`pipeline/host.rs`): the writer
    /// also writes while the list is recorded. A view written while the
    /// submission runs is in its slot for the next submission. Run:
    /// `XNW_TEST_GPU=1 XNW_ADAPTER=warp XNW_WARP=<dir> XNW_AGILITY=<dir>
    /// cargo test -p xnw-gpu --lib gate::tests -- --nocapture`
    /// (`XNW_GPU_DEBUG=layer`: the validation layer counts errors too).
    #[test]
    fn slot_rewritten_while_read() {
        if std::env::var_os("XNW_AGILITY").is_none_or(|d| d.is_empty()) {
            println!("SKIPPED: set XNW_AGILITY to the Agility SDK folder to run on the host GPU");
            return;
        }
        let host = d3d12x::host::host().expect("a host GPU");
        let device = &host.device;
        const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), DescriptorTable(SRV(t0, numDescriptors=3, flags=DESCRIPTORS_VOLATILE)), DescriptorTable(UAV(u0, numDescriptors=1, flags=DESCRIPTORS_VOLATILE))"
cbuffer K : register(b0) { uint iterations; uint pad; };
Texture2D<float4> tex[3] : register(t0);
RWByteAddressBuffer result : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 t : SV_DispatchThreadID) {
    uint v = 0;
    for (uint i = 0; i < iterations; i++) {
        uint k = ((t.x + i) & 1) * 2;
        uint w, h, n;
        tex[NonUniformResourceIndex(k)].GetDimensions(0, w, h, n);
        v += asuint(tex[NonUniformResourceIndex(k)].Load(int3(i & 7, 0, 0)).x) + w;
    }
    uint o;
    result.InterlockedAdd(0, v & 1, o);
    if (t.x == 0) { uint w, h, n; tex[0].GetDimensions(0, w, h, n); result.Store(4, w); }
}
"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: static source and names; out pointer ours; the blob outlives the creation calls.
        let (root, pso) = unsafe {
            D3DCompile(SOURCE.as_ptr().cast(), SOURCE.len(), s!("gate_probe"), None, None, s!("main"), s!("cs_5_1"), 0, 0, &mut code, None).unwrap();
            let code = code.unwrap();
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        };
        let resource = |width: u64, height: u32, texture: bool, kind: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES| {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: if texture { D3D12_RESOURCE_DIMENSION_TEXTURE2D } else { D3D12_RESOURCE_DIMENSION_BUFFER },
                Width: width,
                Height: height,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: if texture { DXGI_FORMAT_R8G8B8A8_UNORM } else { DXGI_FORMAT_UNKNOWN },
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: if texture { D3D12_TEXTURE_LAYOUT_UNKNOWN } else { D3D12_TEXTURE_LAYOUT_ROW_MAJOR },
                Flags: if kind == D3D12_HEAP_TYPE_DEFAULT && !texture { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
                ..Default::default()
            };
            let mut r: Option<ID3D12Resource> = None;
            // SAFETY: plain creation.
            unsafe { device.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r).unwrap() };
            r.unwrap()
        };
        let errors = || {
            let Ok(queue) = device.cast::<ID3D12InfoQueue>() else { return 0 };
            let mut n = 0;
            // SAFETY: plain queries into buffers of the size the queue asked for.
            unsafe {
                for i in 0..queue.GetNumStoredMessages() {
                    let mut len = 0usize;
                    if queue.GetMessage(i, None, &mut len).is_err() || len == 0 {
                        continue;
                    }
                    let mut buf = vec![0u64; len.div_ceil(8)];
                    let m = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                    if queue.GetMessage(i, Some(m), &mut len).is_ok() && (*m).Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 {
                        let text = std::slice::from_raw_parts((*m).pDescription, (*m).DescriptionByteLength.saturating_sub(1));
                        println!("validation: {}", String::from_utf8_lossy(text));
                        n += 1;
                    }
                }
            }
            n
        };
        let before = errors();
        let heap = HostHeap::new(false, 4, 0).unwrap();
        // SAFETY: plain queries of a live heap.
        let (cpu, gpu, step) = unsafe { (heap.heap.GetCPUDescriptorHandleForHeapStart().ptr, heap.heap.GetGPUDescriptorHandleForHeapStart().ptr, device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize) };
        let at = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu + i * step };
        let a = resource(16, 16, true, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        let b = resource(32, 32, true, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        let result = resource(256, 1, false, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        let readback = resource(256, 1, false, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        let raw = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: D3D12_BUFFER_UAV { NumElements: 64, Flags: D3D12_BUFFER_UAV_FLAG_RAW, ..Default::default() } },
        };
        // SAFETY: live resources and slots of our heap; no submission runs yet.
        unsafe {
            srv(device, &a, None, at(0));
            srv(device, &a, None, at(2));
            uav(device, &result, None::<&ID3D12Resource>, Some(&raw), at(3));
        }
        let run = |iterations: u32, groups: u32| {
            let list = host.new_list().unwrap();
            // SAFETY: an open list; live objects of ours.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                list.list.SetComputeRoot32BitConstants(0, 2, [iterations, 0].as_ptr().cast(), 0);
                list.list.SetComputeRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: gpu });
                list.list.SetComputeRootDescriptorTable(2, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: gpu + 3 * step as u64 });
                list.list.Dispatch(groups, 1, 1);
                list.list.ResourceBarrier(&[d3d12x::host::transition(&result, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                list.list.CopyBufferRegion(&readback, 0, &result, 0, 8);
                list.list.ResourceBarrier(&[d3d12x::host::transition(&result, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
            }
            list.close();
            let started = std::time::Instant::now();
            host.execute(&[&list]);
            // SAFETY: a plain query.
            if let Err(e) = unsafe { device.GetDeviceRemovedReason() } {
                println!("device removed after the submission: {e}; validation errors {}", errors());
                panic!("device removed: {e}");
            }
            let mut mapped = std::ptr::null_mut();
            // SAFETY: the finished copy's 8 bytes.
            let got: [u32; 2] = unsafe {
                readback.Map(0, None, Some(&mut mapped)).expect("the readback maps (no device removal)");
                let got = mapped.cast::<[u32; 2]>().read();
                readback.Unmap(0, None);
                got
            };
            (got, started.elapsed())
        };
        // The writer: slots 0 and 2 as the same view, over and over; after
        // 200 writes made while the submission runs, slot 0 once as a view
        // of `b` (32 wide), then slot 2 only.
        struct Shared(ID3D12Device, ID3D12Resource, ID3D12Resource, usize, usize);
        // SAFETY: D3D12 objects are free-threaded.
        unsafe impl Send for Shared {}
        // SAFETY: as above.
        unsafe impl Sync for Shared {}
        let shared = Arc::new(Shared(device.clone(), a.clone(), b.clone(), cpu, step));
        let stop = Arc::new(AtomicBool::new(false));
        // Writes in flight, and whether `b` went in while in flight.
        let flown = Arc::new(AtomicU32::new(0));
        let b_in_flight = Arc::new(AtomicBool::new(false));
        let staged = STAGED.load(Ordering::Relaxed);
        let writer = {
            let (shared, stop, flown, b_in_flight) = (shared.clone(), stop.clone(), flown.clone(), b_in_flight.clone());
            // `XNW_GATE_QUIET=1`: no writes (the probe alone).
            let quiet = std::env::var_os("XNW_GATE_QUIET").is_some();
            std::thread::spawn(move || {
                let s = &*shared;
                let at = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: s.3 + i * s.4 };
                let mut wrote_b = false;
                let mut n = 0u64;
                while !stop.load(Ordering::Relaxed) && !quiet {
                    let flying = IN_FLIGHT.load(Ordering::Relaxed);
                    // SAFETY: live resources; slots of the test heap (kept until the thread is joined).
                    unsafe {
                        if !wrote_b {
                            srv(&s.0, &s.1, None, at(0));
                        }
                        srv(&s.0, &s.1, None, at(2));
                        if flying && !wrote_b && flown.load(Ordering::Relaxed) >= 200 {
                            srv(&s.0, &s.2, None, at(0));
                            wrote_b = true;
                            b_in_flight.store(IN_FLIGHT.load(Ordering::Relaxed), Ordering::Relaxed);
                        }
                    }
                    if flying {
                        flown.fetch_add(1, Ordering::Relaxed);
                    }
                    n += 1;
                }
                n
            })
        };
        let groups = std::env::var("XNW_GATE_GROUPS").ok().and_then(|g| g.parse().ok()).unwrap_or(2048);
        let (got, took) = run(4000, groups);
        stop.store(true, Ordering::Relaxed);
        let writes = writer.join().unwrap();
        println!(
            "gate {}: long submission {groups} groups in {} ms, read {got:?}; writer {writes} writes, {} while it ran; b written in flight {}; {}",
            if on() { "on" } else { "off" },
            took.as_millis(),
            flown.load(Ordering::Relaxed),
            b_in_flight.load(Ordering::Relaxed),
            stats_text()
        );
        assert!(flown.load(Ordering::Relaxed) >= 200, "the writer wrote while the submission ran");
        // The next submission sees the view written while the last ran.
        let (after, _) = run(1, 1);
        println!("next submission reads slot 0 {} wide", after[1]);
        assert_eq!(after[1], 32, "slot 0 holds the view written in flight");
        if on() {
            assert!(b_in_flight.load(Ordering::Relaxed), "b written while the submission ran");
            assert!(STAGED.load(Ordering::Relaxed) > staged, "writes in flight were staged");
        }
        assert_eq!(errors(), before, "no new validation errors");
    }
}
