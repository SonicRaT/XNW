//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.
//!
//! Ported whole from VIctory-base (branch `xnw/base` @ 5d0516e = VIctory
//! d2a4bae + 26 PRs; none touches this file), `crates/victory-gpu/src/
//! indirect.rs` 1-217 with its test, GPL-3.0-or-later. Changed: `tracing`
//! -> `xnw_core`; the test skips without `XNW_AGILITY`, as our other GPU
//! tests do. The records are read from the title's host copy and
//! written, clamped, with the clamped count, into a room of our own that the
//! execution reads; the copy is never transitioned (it may be a vertex
//! buffer in the same list).

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use super::host::{HostList, host};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1), UAV(u2), UAV(u3)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint counted; };
// The title's records and count (host copies; only read here).
RWByteAddressBuffer args : register(u0);
RWByteAddressBuffer count : register(u1);
// Our room: the clamped records, then the clamped count.
RWByteAddressBuffer outp : register(u2);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u3);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// Copies the record at `at` into our room, clamped there; its cost.
uint clamped(uint at) {
    for (uint i = 0; i < stride; i += 4) {
        outp.Store(at + i, args.Load(at + i));
    }
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args.Load(at), 1u << 24);
        uint instances = min(args.Load(at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        outp.Store2(at, uint2(count, instances));
    } else if (kind == 2) {
        uint3 g = min(args.Load3(at), uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        outp.Store3(at, g);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = args.Load3(at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        outp.Store3(at, g);
    }
    return cost;
}
// A record past the budget (no cost reaches it).
static const uint PAST = 0xffffffffu;
groupshared uint costs[64];
// VIctory PR #1510 (4ab20fc, GPL-3.0-or-later): one group walks the
// records in tiles of 64: every lane clamps its own record, then the first
// lane charges the tile's records in order, so rejected records charge
// nothing and the budget word never wraps (a shared add let garbage
// records through again once enough were rejected).
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    if (lane == 0 && counted != 0) {
        outp.Store(records * stride, min(count.Load(0), records));
    }
    uint left = 0;
    if (lane == 0) {
        left = BUDGET - min(budget.Load(0), BUDGET);
    }
    for (uint tile = 0; tile < records / 64 + min(records % 64, 1u); tile++) {
        uint record = tile * 64 + lane;
        uint at = record * stride;
        uint cost = 0;
        if (record < records) {
            cost = clamped(at);
        }
        costs[lane] = cost;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0) {
            for (uint i = 0; i < 64; i++) {
                if (costs[i] > left) {
                    // The first record that does not fit takes what is
                    // left, so those after it run empty too.
                    costs[i] = PAST;
                    left = 0;
                } else {
                    left -= costs[i];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (record < records && costs[lane] == PAST) {
            // Draws keep their layout with no vertices; groups become zero.
            outp.Store(at, 0);
        }
    }
    if (lane == 0) {
        budget.Store(0, BUDGET - left);
    }
}"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                xnw_core::warn!("d3d12x: no argument clamp: indirect execution dropped ({e})");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            xnw_core::warn!("d3d12x: argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` (and
/// their count at host address `count`, if any) into our room at host
/// address `out`: the records, then the count (`records * stride` on).
/// The title's copy is only read, never transitioned (a host copy that
/// also serves as a vertex buffer in the list was moved from unordered
/// access to indirect argument and back, against the state it was
/// promoted to; validation ids 527 and 538).
/// Replaces the list's compute root signature and pipeline (callers bind
/// theirs after). False if the clamp is not available.
pub(crate) fn bound(list: &HostList, gpu: u64, count: Option<u64>, (kind, stride, records): (u32, u32, u32), out: u64) -> bool {
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return false;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) || count.is_some_and(|c| !c.is_multiple_of(4)) {
        return false;
    }
    let l = &list.list;
    list.crumbs().note(|| format!("argument clamp: kind {kind}, stride {stride}, {records} records from host {gpu:#x}, count {}, into our room {out:#x}", count.map_or("none".into(), |c| format!("{c:#x}"))));
    let constants = [kind, stride, records, u32::from(count.is_some())];
    // SAFETY: an open list; our root signature, pipeline and buffers the
    // caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, count.unwrap_or(gpu));
        l.SetComputeRootUnorderedAccessView(3, out);
        l.SetComputeRootUnorderedAccessView(4, budget);
        let _crumb = super::crumbs::begin_compute(list, || format!("indirect arguments pass {records} records"));
        // One group walks every record (VIctory).
        l.Dispatch(1, 1, 1);
        l.ResourceBarrier(&[uav_barrier()]);
    }
    true
}
fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

#[cfg(test)]
mod tests {
    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if std::env::var_os("XNW_AGILITY").is_none_or(|d| d.is_empty()) {
            println!("SKIPPED: set XNW_AGILITY to the Agility SDK folder to run on the host GPU");
            return;
        }
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }
}
