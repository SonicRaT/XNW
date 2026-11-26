//! Whether the host runs a compute group's wave operations as the
//! console's waves need. Translated programs declare the console's
//! `WaveSize` and run its whole-wave sections as host wave operations where
//! only some lanes run (the console's `exec`). Each thread of the groups
//! records its index, lane, wave base and lane count, and inside a branch
//! the first K lanes take (or after the others return) its wave's
//! active-lane count, a wave sum of 1.0 and lane 0's index. Needs
//! `XNW_TEST_GPU=1`, the Agility SDK (`XNW_AGILITY`) and DXC (Windows SDK,
//! or `XNW_DXC`); prints SKIPPED and passes without them. The table prints
//! with `--nocapture`; `XNW_WAVE_RUNS`, `XNW_WAVE_PROBE_GROUPS`,
//! `XNW_WAVE_PROBE_LANES` (K, default 10) and `XNW_WAVE_DETAIL` vary it.
//!
//! WARP 1.0.22 (2026-11-25, V): with `WaveSize(32)` or `WaveSize(64)`
//! (waves emulated over its native 8 lanes) the operations inside the
//! branch count 0 or garbage active lanes (525, 82, 68) and sum no lane,
//! differently run to run: in groups of more than one wave, and in groups
//! of one wave when the other lanes return first. Its native waves (no
//! `WaveSize`, 8 lanes) are right in every case. The hardware adapter must
//! be right in every case (asserted); on the software adapter only the
//! native rows are. This is sink 490 of pk8a f1027 (report).

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::process::Command;

use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;
use xnw_gpu::Host;

const SRC: &str = r#"
RWByteAddressBuffer o : register(u0);
[RootSignature("UAV(u0)")]
#if WS
[WaveSize(WS)]
#endif
[numthreads(NT, 1, 1)]
void main(uint t : SV_GroupIndex, uint3 g : SV_GroupID) {
    uint at = (g.x * NT + t) * 32;
    uint lane = WaveGetLaneIndex();
    o.Store4(at, uint4(t, lane, WaveReadLaneFirst(t), WaveGetLaneCount()));
    uint4 r = uint4(WaveActiveCountBits(true), 0, 0, 0);
#if EXIT
    if (lane >= K) {
        o.Store4(at + 16, r);
        return;
    }
#else
    if (lane < K)
#endif
    {
        r.y = WaveActiveCountBits(true);
        r.z = (uint)WaveActiveSum(1.0f);
        r.w = WaveReadLaneAt(t, 0);
    }
    o.Store4(at + 16, r);
}
"#;

/// Lanes taking the branch (`XNW_WAVE_PROBE_LANES`, default 10: sink 490's).
fn k() -> u32 {
    std::env::var("XNW_WAVE_PROBE_LANES").ok().and_then(|v| v.parse().ok()).unwrap_or(10)
}

/// Groups per dispatch (`XNW_WAVE_PROBE_GROUPS`, default 8).
fn groups() -> u32 {
    std::env::var("XNW_WAVE_PROBE_GROUPS").ok().and_then(|v| v.parse().ok()).unwrap_or(8)
}

fn dxc() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("XNW_DXC") {
        return Some(PathBuf::from(d)).filter(|p| p.is_file());
    }
    let kits = Path::new(r"C:\Program Files (x86)\Windows Kits\10\bin");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(kits).ok()?.flatten().map(|e| e.path().join("x64")).collect();
    dirs.sort();
    dirs.into_iter().rev().map(|d| d.join("dxc.exe")).find(|p| p.is_file())
}

fn compile(dxc: &Path, threads: u32, ws: u32, exit: bool) -> Vec<u8> {
    compile_src(dxc, "w", SRC, threads, ws, exit)
}

fn compile_src(dxc: &Path, name: &str, text: &str, threads: u32, ws: u32, exit: bool) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!("xnw-wave-packing-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (src, out) = (dir.join(format!("{name}.hlsl")), dir.join(format!("{name}{threads}-{ws}-{}-{}.dxil", u8::from(exit), k())));
    std::fs::write(&src, text).unwrap();
    let run = Command::new(dxc).args(["-nologo", "-O3", "-T", "cs_6_6", "-E", "main"]).arg(format!("-DWS={ws}")).arg(format!("-DNT={threads}")).arg(format!("-DEXIT={}", u8::from(exit))).arg(format!("-DK={}", k())).arg("-Fo").arg(&out).arg(&src).output().unwrap();
    assert!(run.status.success(), "dxc: {}", String::from_utf8_lossy(&run.stderr));
    std::fs::read(out).unwrap()
}

/// Runs a probe over [`groups`] groups of `threads`: 8 words per thread.
fn run(host: &Host, cs: &[u8], threads: u32) -> windows::core::Result<Vec<u32>> {
    let n = u64::from(groups() * threads) * 32;
    let dev = &host.device;
    // SAFETY: plain D3D12 calls; descriptions outlive each call, the
    // mapping is read after the fence says the GPU is done.
    unsafe {
        let root: ID3D12RootSignature = dev.CreateRootSignature(0, cs)?;
        let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: std::mem::transmute_copy(&root), CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr().cast(), BytecodeLength: cs.len() }, ..Default::default() })?;
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: n, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, ..Default::default() };
        let mut buf: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, None, &mut buf)?;
        let buf = buf.ok_or_else(windows::core::Error::empty)?;
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso)?;
        list.SetComputeRootSignature(&root);
        list.SetComputeRootUnorderedAccessView(0, buf.GetGPUVirtualAddress());
        list.Dispatch(groups(), 1, 1);
        list.Close()?;
        host.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        host.queue.Signal(&fence, 1)?;
        let event: HANDLE = CreateEventW(None, false, false, None)?;
        fence.SetEventOnCompletion(1, event)?;
        WaitForSingleObject(event, INFINITE);
        let _ = CloseHandle(event);
        dev.GetDeviceRemovedReason()?;
        let mut p: *mut c_void = std::ptr::null_mut();
        buf.Map(0, None, Some(&mut p))?;
        let words = std::slice::from_raw_parts(p.cast::<u32>(), (n / 4) as usize).to_vec();
        buf.Unmap(0, None);
        Ok(words)
    }
}

/// Waves (by index in the group) whose threads are not in console order
/// or whose branch saw the wrong lanes, and the line for the table.
fn judge(w: &[u32], threads: u32, ws: u32) -> (Vec<u32>, String) {
    let mut bad = std::collections::BTreeSet::new();
    let mut lanes = 0;
    for r in w.chunks_exact(8) {
        let (t, lane, first, count) = (r[0], r[1], r[2], r[3]);
        lanes = count;
        let base = first;
        let in_order = ws == 0 || (lane == t % ws && base == t - t % ws && count == ws);
        let kk = k().min(count.min(threads));
        let branch = lane >= k() || (r[5] == kk && r[6] == kk && r[7] == base);
        if !(in_order && r[4] == count.min(threads) && branch) {
            if bad.insert(t / count.max(1)) && std::env::var_os("XNW_WAVE_DETAIL").is_some() {
                println!("  t {t} lane {lane} first {first} lanes {count} active {} branch count {} sum {} lane0 {}", r[4], r[5], r[6], r[7]);
            }
        }
    }
    let bad: Vec<u32> = bad.into_iter().collect();
    (bad.clone(), format!("threads {threads:4} WaveSize({ws:2}) lanes {lanes:3}: waves wrong {bad:?}"))
}

#[test]
fn wave_packing() {
    let agility = std::env::var_os("XNW_AGILITY").map(PathBuf::from).filter(|d| !d.as_os_str().is_empty());
    let host = match Host::new(agility.as_deref()) {
        Ok(h) if h.caps.shader_model >= (6, 6) => h,
        Ok(h) => return println!("SKIPPED: shader model {:?} < 6.6", h.caps.shader_model),
        Err(e) => return println!("SKIPPED: no device ({e})"),
    };
    let Some(dxc) = dxc() else { return println!("SKIPPED: no dxc") };
    let soft = host.caps.wave_lanes.0 < 32;
    let mut wrong = Vec::new();
    for (threads, exit) in [64u32, 128, 192, 256, 512, 1024].into_iter().flat_map(|t| [(t, false), (t, true)]) {
        for ws in [64u32, 32, 0] {
            if ws > 0 && !(host.caps.wave_lanes.0..=host.caps.wave_lanes.1).contains(&ws) {
                continue;
            }
            let cs = compile(&dxc, threads, ws, exit);
            let mut bad = Vec::new();
            for _ in 0..std::env::var("XNW_WAVE_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(1) {
                let w = run(&host, &cs, threads).expect("dispatch");
                let (b, line) = judge(&w, threads, ws);
                println!("{line}{}", if exit { " (other lanes return)" } else { "" });
                bad.extend(b);
            }
            if !bad.is_empty() {
                wrong.push((threads, ws, exit));
            }
        }
    }
    println!("wrong: {wrong:?}");
    assert!(wrong.iter().all(|&(_, ws, _)| ws != 0), "native waves ran wrong: {wrong:?}");
    if !soft {
        assert!(wrong.is_empty(), "the hardware adapter ran waves wrong: {wrong:?}");
    }
}

/// `WaveReadLaneAt` with a lane index each lane computes, every lane
/// running: what a translated `ds_swizzle_b32` (rotate, FFT order) and DPP
/// rotation become. Lane i holds i + 1 and reads lane (i + 1) % n, i ^ 3,
/// i ^ n/2, the bit-reversed i, and lane 7 (the same index in every lane).
const READS: &str = r#"
RWByteAddressBuffer o : register(u0);
[RootSignature("UAV(u0)")]
#if WS
[WaveSize(WS)]
#endif
[numthreads(NT, 1, 1)]
void main(uint t : SV_GroupIndex, uint3 g : SV_GroupID) {
    uint at = (g.x * NT + t) * 32;
    uint lane = WaveGetLaneIndex();
    uint n = WaveGetLaneCount();
    uint x = lane + 1;
    uint rev = reversebits(lane) >> (32 - firstbithigh(n));
    o.Store4(at, uint4(t, lane, n, WaveReadLaneAt(x, (lane + 1) % n)));
    o.Store4(at + 16, uint4(WaveReadLaneAt(x, lane ^ 3), WaveReadLaneAt(x, lane ^ (n >> 1)), WaveReadLaneAt(x, rev), WaveReadLaneAt(x, 7)));
}
"#;

/// The reads of [`READS`] that came back wrong, by name, with the first
/// wrong lane of each: (lane, got, want).
fn judge_reads(w: &[u32]) -> Vec<(&'static str, u32, u32, u32)> {
    let mut bad: Vec<(&'static str, u32, u32, u32)> = Vec::new();
    for r in w.chunks_exact(8) {
        let (lane, n) = (r[1], r[2]);
        let rev = lane.reverse_bits() >> (32 - n.trailing_zeros());
        let want = [("rotate", (lane + 1) % n), ("xor 3", lane ^ 3), ("xor half", lane ^ (n >> 1)), ("bit-reversed", rev), ("lane 7", 7)];
        for (k, (name, src)) in want.into_iter().enumerate() {
            if r[3 + k] != src + 1 && !bad.iter().any(|b| b.0 == name) {
                bad.push((name, lane, r[3 + k], src + 1));
            }
        }
    }
    bad
}

/// WARP 1.0.22 (2026-11-25, V on both PCs; this is the minimal case): with
/// `WaveSize(32)` or `WaveSize(64)` (waves emulated over its native 8
/// lanes) a `WaveReadLaneAt` whose lane index differs between lanes reads
/// wrong lanes: rotate, lane 6 reads 16 (lane 7 holds 8) and lane 7 reads 1
/// (lane 8 holds 9); bit-reversed, lane 1 reads 1 (want 17, or 33 of 64).
/// The same index in every lane (`7`) and `lane ^ 3`, `lane ^ n/2` are
/// right, and every read on its native waves (no `WaveSize`) is right.
/// Every lane runs: this is not the partial-exec case of `wave_packing`.
/// The translated `ds_swizzle_b32` gives the same lists, which is why
/// `xnw-shader`'s `ds_swizzle_rotate_mode`, `_fft_mode` and
/// `_masked_rotations` differ on that adapter (not asserted there).
#[test]
fn lane_reads() {
    let agility = std::env::var_os("XNW_AGILITY").map(PathBuf::from).filter(|d| !d.as_os_str().is_empty());
    let host = match Host::new(agility.as_deref()) {
        Ok(h) if h.caps.shader_model >= (6, 6) => h,
        Ok(h) => return println!("SKIPPED: shader model {:?} < 6.6", h.caps.shader_model),
        Err(e) => return println!("SKIPPED: no device ({e})"),
    };
    let Some(dxc) = dxc() else { return println!("SKIPPED: no dxc") };
    let soft = host.caps.wave_lanes.0 < 32;
    let mut wrong = Vec::new();
    for ws in [64u32, 32, 0] {
        if ws > 0 && !(host.caps.wave_lanes.0..=host.caps.wave_lanes.1).contains(&ws) {
            continue;
        }
        let threads = ws.max(64);
        let cs = compile_src(&dxc, "r", READS, threads, ws, false);
        let w = run(&host, &cs, threads).expect("dispatch");
        let bad = judge_reads(&w);
        println!("WaveSize({ws:2}) lanes {:3}: reads wrong (name, lane, got, want) {bad:?}", w[2]);
        if !bad.is_empty() {
            wrong.push(ws);
        }
    }
    assert!(!wrong.contains(&0), "native waves read lanes wrong");
    if !soft {
        assert!(wrong.is_empty(), "the hardware adapter read lanes wrong: WaveSize {wrong:?}");
    }
}
