//! XNW: which null descriptions the software adapter (WARP) survives when a
//! shader reads the slot as a kind other than the one it was made with
//! (the title's CS_Prop_BatchCull faulted WARP 1.0.22 reading
//! address 0x388). A compute program (`nk.cso`, HLSL in `NK_HLSL`, compiled
//! with dxc -T cs_6_6) reads heap slot 0 as the kind `mode` names and writes
//! what it read to slot 1. Each (slot, mode) runs in a child process, since a
//! fault ends the process: `XNW_NK_CS=<nk.cso> cargo test -p xnw-shader
//! null_kinds -- --ignored --nocapture` (XNW_ADAPTER=warp XNW_WARP=<dir>).
//! Result on WARP 1.0.22: a stable null SRV of any dimension
//! reads zeros under every read; a null UAV or CBV read as a resource, or a
//! slot never written, removes the device. Kinds 13-16 with modes 30/31 and
//! `XNW_NK_GROUPS=2048`: a slot rewritten by another thread while the
//! dispatch reads it faults every time, even rewritten as the same view
//! (`reading 0x388`, the title's fault) or copied in from a CPU heap.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::host::{Dispatch, Host, Root, View};

#[allow(dead_code)]
const NK_HLSL: &str = r#"
struct C4 { uint4 a; };
RWByteAddressBuffer ru : register(u0, space1);
cbuffer U : register(b0, space0) { uint src; uint dst; uint mode; uint pad; };
[numthreads(64, 1, 1)]
void main(uint3 t : SV_DispatchThreadID) {
    uint v = 7;
    if (mode == 0) { ByteAddressBuffer b = ResourceDescriptorHeap[src]; v = b.Load(t.x * 4); }
    else if (mode == 1) { StructuredBuffer<uint4> b = ResourceDescriptorHeap[src]; v = b[t.x].x; }
    else if (mode == 2) { Buffer<float4> b = ResourceDescriptorHeap[src]; v = asuint(b[t.x].x); }
    else if (mode == 3) { RWByteAddressBuffer b = ResourceDescriptorHeap[src]; b.Store(t.x * 4, 5); v = 3; }
    else if (mode == 4) { RWByteAddressBuffer b = ResourceDescriptorHeap[src]; uint o; b.InterlockedAdd(0, 1, o); v = o; }
    else if (mode == 5) { RWStructuredBuffer<uint4> b = ResourceDescriptorHeap[src]; v = b[t.x].y; }
    else if (mode == 6) { Texture2D<float4> b = ResourceDescriptorHeap[src]; v = asuint(b.Load(int3(0, 0, 0)).x); }
    else if (mode == 7) { Texture2D<float4> b = ResourceDescriptorHeap[src]; uint w, h, n; b.GetDimensions(0, w, h, n); v = w; }
    else if (mode == 8) { RWTexture2D<uint> b = ResourceDescriptorHeap[src]; v = b[uint2(0, 0)]; }
    else if (mode == 9) { ConstantBuffer<C4> b = ResourceDescriptorHeap[src]; v = b.a.x; }
    else if (mode == 10) { v = ru.Load(t.x * 4); ru.Store(t.x * 4 + 256, v + 1); }
    else if (mode == 11) { Texture2D<float4> b = ResourceDescriptorHeap[src]; v = asuint(b.Load(int3(t.x, 1, 3)).x); }
    else if (mode == 12) { Texture2D<float4> b = ResourceDescriptorHeap[src]; uint w, h, n; b.GetDimensions(3, w, h, n); v = w + h + n; }
    else if (mode == 13) { Texture2D<float> b = ResourceDescriptorHeap[src]; v = asuint(b.Load(int3(t.x, 1, 0))); }
    else if (mode == 14) { ByteAddressBuffer b = ResourceDescriptorHeap[NonUniformResourceIndex(src + (t.x >> 12))]; v = b.Load(t.x * 4); }
    else if (mode == 15) { Texture2D<float4> b = ResourceDescriptorHeap[NonUniformResourceIndex(src + (t.x >> 12))]; v = asuint(b.Load(int3(0, 0, 0)).x); }
    else if (mode == 16) { Texture2D<float4> b = ResourceDescriptorHeap[NonUniformResourceIndex(src + (t.x >> 12))]; uint w, h, n; b.GetDimensions(0, w, h, n); v = w + n; }
    else if (mode == 17) { RWByteAddressBuffer b = ResourceDescriptorHeap[NonUniformResourceIndex(src + (t.x >> 12))]; uint o; b.InterlockedAdd(0, 1, o); v = o; }
    else if (mode == 19) { ByteAddressBuffer b = ResourceDescriptorHeap[src]; uint n; b.GetDimensions(n); v = n; }
    else if (mode == 20) { ByteAddressBuffer b = ResourceDescriptorHeap[NonUniformResourceIndex(src + (t.x >> 12))]; uint n; b.GetDimensions(n); v = n; }
    else if (mode == 21) { StructuredBuffer<uint4> b = ResourceDescriptorHeap[src]; uint n, st; b.GetDimensions(n, st); v = n; }
    else if (mode == 22) { RWByteAddressBuffer b = ResourceDescriptorHeap[src]; uint n; b.GetDimensions(n); v = n; }
    else if (mode == 23) { Texture2D<float4> b = ResourceDescriptorHeap[NonUniformResourceIndex((t.x & 1) * 2)]; v = asuint(b.Load(int3(0, 0, 0)).x); }
    else if (mode == 24) { ByteAddressBuffer b = ResourceDescriptorHeap[NonUniformResourceIndex((t.x & 1) * 2)]; v = b.Load(t.x * 4); }
    else if (mode == 25) { Texture2D<float4> b = ResourceDescriptorHeap[NonUniformResourceIndex((t.x & 1) * 2)]; uint w, h, n; b.GetDimensions(0, w, h, n); v = w + n; }
    else if (mode == 26) { StructuredBuffer<uint4> b = ResourceDescriptorHeap[NonUniformResourceIndex((t.x & 1) * 2)]; v = b[t.x].x; }
    else if (mode == 27) { ConstantBuffer<C4> b = ResourceDescriptorHeap[NonUniformResourceIndex((t.x & 1) * 2)]; v = b.a.x; }
    else if (mode == 28) { RWByteAddressBuffer b = ResourceDescriptorHeap[NonUniformResourceIndex((t.x & 1) * 2)]; uint o; b.InterlockedAdd(0, 1, o); v = o; }
    else if (mode == 29) { ByteAddressBuffer b = ResourceDescriptorHeap[NonUniformResourceIndex((t.x & 1) * 2)]; uint n; b.GetDimensions(n); v = n; }
    else if (mode == 30) { for (uint i = 0; i < 4000; i++) { Texture2D<float4> b = ResourceDescriptorHeap[NonUniformResourceIndex(((t.x + i) & 1) * 2)]; uint w, h, n; b.GetDimensions(0, w, h, n); v += asuint(b.Load(int3(i & 7, 0, 0)).x) + w; } }
    else if (mode == 31) { for (uint i = 0; i < 4000; i++) { ByteAddressBuffer b = ResourceDescriptorHeap[NonUniformResourceIndex(((t.x + i) & 1) * 2)]; v += b.Load((i & 7) * 4); } }
    else if (mode == 18) { StructuredBuffer<uint4> b = ResourceDescriptorHeap[NonUniformResourceIndex(src + (t.x >> 12))]; v = b[t.x].x; }
    RWByteAddressBuffer o = ResourceDescriptorHeap[dst];
    o.Store(t.x * 4, v);
}"#;

const MODES: [&str; 10] = ["raw SRV load", "structured SRV load", "typed SRV load", "raw UAV store", "raw UAV atomic", "structured UAV load", "2D texture load", "2D texture size", "2D UAV load", "constant buffer load"];

fn srv(dev: &ID3D12Device, at: D3D12_CPU_DESCRIPTOR_HANDLE, d: D3D12_SHADER_RESOURCE_VIEW_DESC) {
    // SAFETY: a slot of the test heap; a null resource with a description.
    unsafe { dev.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&d), at) }
}

fn uav(dev: &ID3D12Device, at: D3D12_CPU_DESCRIPTOR_HANDLE, d: D3D12_UNORDERED_ACCESS_VIEW_DESC) {
    // SAFETY: as above.
    unsafe { dev.CreateUnorderedAccessView(None::<&ID3D12Resource>, None::<&ID3D12Resource>, Some(&d), at) }
}

fn buffer_srv(elements: u32, format: DXGI_FORMAT, stride: u32, flags: D3D12_BUFFER_SRV_FLAGS) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Buffer: D3D12_BUFFER_SRV { NumElements: elements, StructureByteStride: stride, Flags: flags, ..Default::default() } },
    }
}

/// Slot 0 as xnw-gpu's null views make it (descriptor/host.rs), and others.
fn write_slot(kind: u32, dev: &ID3D12Device, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    match kind {
        // null_view: what fill_null puts in every slot, and slots not made.
        0 => srv(dev, at, D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } },
        }),
        // null_buffer_view: raw, one element.
        1 => srv(dev, at, buffer_srv(1, DXGI_FORMAT_R32_TYPELESS, 0, D3D12_BUFFER_SRV_FLAG_RAW)),
        // raw, no elements.
        2 => srv(dev, at, buffer_srv(0, DXGI_FORMAT_R32_TYPELESS, 0, D3D12_BUFFER_SRV_FLAG_RAW)),
        // null_uav_buffer_view(stride 0): raw, one element.
        3 => uav(dev, at, D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: D3D12_BUFFER_UAV { NumElements: 1, Flags: D3D12_BUFFER_UAV_FLAG_RAW, ..Default::default() } },
        }),
        // null_uav_view: 2D R32_UINT.
        4 => uav(dev, at, D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: DXGI_FORMAT_R32_UINT, ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D, ..Default::default() }),
        // never written.
        5 => {}
        // structured SRV, 16 B, one element.
        6 => srv(dev, at, buffer_srv(1, DXGI_FORMAT_UNKNOWN, 16, D3D12_BUFFER_SRV_FLAG_NONE)),
        // CBV with a zero location.
        // SAFETY: a slot of the test heap.
        7 => unsafe { dev.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC::default()), at) },
        // Views of live resources (kept for the process).
        8 => {
            let b = resource(dev, D3D12_HEAP_TYPE_DEFAULT, true, false);
            // SAFETY: a live buffer and a slot of the test heap.
            unsafe {
                dev.CreateUnorderedAccessView(&b, None::<&ID3D12Resource>, Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: D3D12_BUFFER_UAV { NumElements: 1024, Flags: D3D12_BUFFER_UAV_FLAG_RAW, ..Default::default() } },
                }), at)
            };
            std::mem::forget(b);
        }
        9 => {
            let b = resource(dev, D3D12_HEAP_TYPE_DEFAULT, false, false);
            // SAFETY: as above.
            unsafe { dev.CreateShaderResourceView(&b, Some(&buffer_srv(1024, DXGI_FORMAT_R32_TYPELESS, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at) };
            std::mem::forget(b);
        }
        10 => {
            let b = resource(dev, D3D12_HEAP_TYPE_UPLOAD, false, false);
            // SAFETY: as above.
            unsafe { dev.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: b.GetGPUVirtualAddress(), SizeInBytes: 4096 }), at) };
            std::mem::forget(b);
        }
        11 => {
            let t = resource(dev, D3D12_HEAP_TYPE_DEFAULT, false, true);
            // SAFETY: as above.
            unsafe { dev.CreateShaderResourceView(&t, None, at) };
            std::mem::forget(t);
        }
        // 13, 14: a live 2D SRV, rewritten by another thread for 3 s, over and
        // over, to a null view (13) or to a raw view of a zeroed buffer (14,
        // as XNW_NULL_SRV=dummy) or to the same view again (15), while the
        // dispatch reads it.
        13..=16 => {
            let t = resource(dev, D3D12_HEAP_TYPE_DEFAULT, false, true);
            let zero = resource(dev, D3D12_HEAP_TYPE_DEFAULT, false, false);
            // SAFETY: as above.
            unsafe { dev.CreateShaderResourceView(&t, None, at) };
            struct Send2(ID3D12Device, ID3D12Resource, ID3D12Resource, usize);
            // SAFETY: D3D12 objects are free-threaded; the slot outlives the thread (3 s, heap kept by the test).
            unsafe impl Send for Send2 {}
            let s = Send2(dev.clone(), t, zero, at.ptr);
            std::thread::spawn(move || {
                let s = s;
                let at = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: s.3 };
                // 16: the view is made in a CPU-only heap and copied into the slot.
                // SAFETY: plain heap creation.
                let staging: ID3D12DescriptorHeap = unsafe { s.0.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 1, ..Default::default() }) }.unwrap();
                // SAFETY: a live heap.
                let stage = unsafe { staging.GetCPUDescriptorHandleForHeapStart() };
                let start = std::time::Instant::now();
                let mut n = 0u64;
                while start.elapsed().as_secs() < 3 {
                    // SAFETY: live resources and a slot of the test heap.
                    unsafe {
                        if kind == 13 {
                            write_slot(0, &s.0, at);
                        } else if kind == 15 {
                            s.0.CreateShaderResourceView(&s.1, None, at);
                        } else if kind == 16 {
                            s.0.CreateShaderResourceView(&s.1, None, stage);
                            s.0.CopyDescriptorsSimple(1, at, stage, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                        } else {
                            s.0.CreateShaderResourceView(&s.2, Some(&buffer_srv(1024, DXGI_FORMAT_R32_TYPELESS, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at);
                        }
                        if kind != 16 {
                            s.0.CreateShaderResourceView(&s.1, None, at);
                        }
                    }
                    n += 1;
                }
                println!("NKT rewrote the slot {n} times");
            });
        }
        _ => {
            let t = resource(dev, D3D12_HEAP_TYPE_DEFAULT, true, true);
            // SAFETY: as above.
            unsafe { dev.CreateUnorderedAccessView(&t, None::<&ID3D12Resource>, None, at) };
            std::mem::forget(t);
        }
    }
}

/// A 4 KiB buffer or a 16x16 R32_UINT texture.
fn resource(dev: &ID3D12Device, heap: D3D12_HEAP_TYPE, uav: bool, texture: bool) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: if texture { D3D12_RESOURCE_DIMENSION_TEXTURE2D } else { D3D12_RESOURCE_DIMENSION_BUFFER },
        Width: if texture { 16 } else { 4096 },
        Height: if texture { 16 } else { 1 },
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: if texture { DXGI_FORMAT_R32_UINT } else { DXGI_FORMAT_UNKNOWN },
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: if texture { D3D12_TEXTURE_LAYOUT_UNKNOWN } else { D3D12_TEXTURE_LAYOUT_ROW_MAJOR },
        Flags: if uav { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
        ..Default::default()
    };
    let state = if heap == D3D12_HEAP_TYPE_UPLOAD { D3D12_RESOURCE_STATE_GENERIC_READ } else { D3D12_RESOURCE_STATE_COMMON };
    let mut r: Option<ID3D12Resource> = None;
    // SAFETY: a plain committed resource.
    unsafe { dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r) }.unwrap();
    r.unwrap()
}

const KINDS: [&str; 17] = ["null_view (2D SRV)", "null_buffer_view (raw SRV, 1 element)", "raw SRV, 0 elements", "null_uav_buffer_view (raw UAV, 1)", "null_uav_view (2D UAV)", "never written", "structured SRV (16 B, 1)", "null CBV", "live raw UAV buffer", "live raw SRV buffer", "live CBV", "live 2D SRV (R32_UINT)", "live 2D UAV (R32_UINT)", "live 2D SRV <-> null, rewritten while read", "live 2D SRV <-> zero buffer, rewritten while read", "live 2D SRV rewritten as itself while read", "live 2D SRV copied over itself from a CPU heap while read"];

/// Prints the first access violation (WARP faults inside its shader code
/// and removes the device; the title's run logged `reading address 0x388`).
unsafe extern "system" fn on_fault(p: *mut xnw_core::win::ExceptionPointers) -> i32 {
    // SAFETY: the system passes a live record.
    let r = unsafe { &*(*p).record };
    static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if r.code == 0xC000_0005 && !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
        println!("NKAV at {:#x} reading {:#x}", r.address, r.params[1]);
    }
    0
}

/// One (kind, mode) in this process: `XNW_NK=kind,mode`.
#[test]
#[ignore]
fn null_kind_one() {
    let Ok(nk) = std::env::var("XNW_NK") else { return };
    let (k, m) = nk.split_once(',').map(|(a, b)| (a.parse::<u32>().unwrap(), b.parse::<u32>().unwrap())).unwrap();
    let cs = std::fs::read(std::env::var("XNW_NK_CS").expect("XNW_NK_CS")).unwrap();
    // SAFETY: a handler that reads the record only.
    unsafe { xnw_core::win::AddVectoredExceptionHandler(1, on_fault) };
    let host = Host::for_tests().expect("an SM 6.6 device");
    let shader = super::HostShader {
        stage: crate::HwStage::Cs,
        dxil: cs,
        user_data: (0..4).map(|i| super::UserData { index: i, slot: 0, root: crate::RootType::Constants, dword: i }).collect(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: vec![0],
        address_map: false,
        descriptor_words: false,
    };
    let (root, pso) = host.compute_pipeline(&shader).unwrap();
    // The root UAV (`XNW_NK_ROOT`): a live buffer, address 0, a released
    // buffer's address, or past a live buffer's end.
    let live = resource(&host.device, D3D12_HEAP_TYPE_DEFAULT, true, false);
    // SAFETY: a live buffer.
    let live_va = unsafe { live.GetGPUVirtualAddress() };
    let va = match std::env::var("XNW_NK_ROOT").as_deref() {
        Ok("zero") => 0,
        Ok("freed") => {
            let gone = resource(&host.device, D3D12_HEAP_TYPE_DEFAULT, true, false);
            // SAFETY: as above; released right after.
            let va = unsafe { gone.GetGPUVirtualAddress() };
            drop(gone);
            va
        }
        Ok("past") => live_va + (1 << 20),
        _ => live_va,
    };
    println!("NKR root UAV at {va:#x} (live buffer at {live_va:#x})");
    let write = move |dev: &ID3D12Device, at: D3D12_CPU_DESCRIPTOR_HANDLE| {
        write_slot(k, dev, at);
        if std::env::var_os("XNW_NK_BYTES").is_some() {
            // WARP's descriptors are CPU memory: the slot's bytes.
            // SAFETY: the slot's own bytes in a live heap (software adapter only).
            let n = unsafe { dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) } as usize;
            let b = unsafe { std::slice::from_raw_parts(at.ptr as *const u64, n / 8) };
            println!("NKD {k}: {}", b.iter().map(|w| format!("{w:016x}")).collect::<Vec<_>>().join(" "));
        }
    };
    // Slot 2 as slot 0, for reads whose lanes index different slots.
    let views = [View::Slot(&write), View::RawUav(256), View::Slot(&write)];
    let d = Dispatch { roots: vec![(0, Root::Constants(vec![0, 1, m, 0])), (1, Root::UavAt(va))], views: &views, samplers: &[], groups: [std::env::var("XNW_NK_GROUPS").ok().and_then(|g| g.parse().ok()).unwrap_or(4), 1, 1] };
    match host.dispatch(&root, &pso, &d) {
        Ok(out) => println!("NK {k} {m}: ok, read {:?}", &out[0][..4]),
        Err(e) => println!("NK {k} {m}: error {e}"),
    }
}

/// Every (kind, mode) in a child process; prints a table.
#[test]
#[ignore]
fn null_kinds() {
    if std::env::var_os("XNW_NK_CS").is_none() {
        eprintln!("XNW_NK_CS not set; skipped");
        return;
    }
    let exe = std::env::current_exe().unwrap();
    for k in 0..KINDS.len() as u32 {
        for m in 0..MODES.len() as u32 {
            let mut child = std::process::Command::new(&exe)
                .args(["--ignored", "--exact", "emit::null_kind_tests::null_kind_one", "--nocapture"])
                .env("XNW_NK", format!("{k},{m}"))
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .spawn()
                .unwrap();
            let start = std::time::Instant::now();
            let status = loop {
                if let Some(s) = child.try_wait().unwrap() {
                    break Some(s);
                }
                if start.elapsed().as_secs() > 20 {
                    let _ = child.kill();
                    break None;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            };
            let mut text = String::new();
            let _ = std::io::Read::read_to_string(child.stdout.as_mut().unwrap(), &mut text);
            let line = match status {
                None => "hung (killed after 20 s)".to_string(),
                Some(st) => text.lines().find(|l| l.starts_with("NK ")).map_or_else(|| format!("exit {:#x}", st.code().unwrap_or(-1) as u32), |l| l.splitn(4, ' ').nth(3).unwrap_or("").to_string()),
            };
            let line = match text.lines().find(|l| l.starts_with("NKAV")) {
                Some(av) => format!("{line}; {av}"),
                None => line,
            };
            println!("{:<40} | {:<20} | {line}", KINDS[k as usize], MODES[m as usize]);
        }
    }
}
