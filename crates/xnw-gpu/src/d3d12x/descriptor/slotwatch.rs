//! `XNW_SLOT_WATCH=<slot>[,<slot>...]`: what the host makes in
//! chosen view-heap slots, and what is there when a chosen program runs.
//!
//! Every title run on WARP faulted reading title view slot 327,966 from
//! InstanceCulling:CS_Prop_BatchCull 0451d340 (REF1-REF3). This names the
//! descriptor in such a slot three times:
//!
//! * at each write into it (the gated `Create*View` and copies,
//!   [`super::gate`]): the kind (CBV, SRV, UAV, copy), the view's format,
//!   dimension, elements and stride, and the resource behind it;
//! * at each dispatch of a program whose name holds one of
//!   `XNW_SLOT_WATCH_PIPE` (default `0451d340`): the title view the slot was
//!   made from, the record in title memory there, the runtime stride-table
//!   entry the translated code picks its read by (stride 0: the program's
//!   heap-picked scalar loads read the slot as a constant buffer, else
//!   through a raw view), and on WARP the descriptor's own words (kind
//!   nibble, read pointers);
//! * at an access violation outside every module, for each register that
//!   points into a view heap (CPU or GPU range): the slot, those words, and
//!   the slot's last watched write.
//!
//! Off (no slot named) it costs one load per descriptor write.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;

/// The watched slots (`XNW_SLOT_WATCH`), sorted.
pub(crate) fn slots() -> &'static [u32] {
    static SLOTS: OnceLock<Vec<u32>> = OnceLock::new();
    SLOTS.get_or_init(|| {
        let mut v: Vec<u32> = std::env::var("XNW_SLOT_WATCH").unwrap_or_default().split(',').filter_map(|s| s.trim().parse().ok()).collect();
        v.sort_unstable();
        v.dedup();
        v
    })
}

/// Whether any slot is watched.
pub(crate) fn on() -> bool {
    !slots().is_empty()
}

/// Whether the pipeline named `name` is one whose dispatches report the
/// watched slots (`XNW_SLOT_WATCH_PIPE`, comma-separated name parts;
/// default `0451d340`).
pub(crate) fn pipe_watched(name: &str) -> bool {
    static PIPES: OnceLock<Vec<String>> = OnceLock::new();
    let pipes = PIPES.get_or_init(|| {
        let v = std::env::var("XNW_SLOT_WATCH_PIPE").unwrap_or_else(|_| "0451d340".into());
        v.split(',').map(|s| s.trim().to_ascii_lowercase()).filter(|s| !s.is_empty()).collect()
    });
    on() && pipes.iter().any(|p| name.contains(p.as_str()))
}

/// A view heap as the watch knows it: CPU and GPU starts, slots (title
/// slots first), bytes per slot, title slots, the title heap's address.
#[derive(Clone, Copy)]
struct Heap {
    cpu: usize,
    gpu: u64,
    slots: u32,
    step: u32,
    title: u32,
    title_start: u64,
}

static HEAPS: Mutex<Vec<Heap>> = Mutex::new(Vec::new());

/// The last watched write per (heap CPU start, slot), for the crash report.
static LAST: Mutex<Vec<(usize, u32, String)>> = Mutex::new(Vec::new());

fn heaps() -> std::sync::MutexGuard<'static, Vec<Heap>> {
    HEAPS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Notes view heap `cpu` (see [`Heap`]); always, so the crash report can
/// name registers pointing into heaps whether or not slots are watched.
pub(crate) fn register(cpu: usize, gpu: u64, slots: u32, step: u32, title: u32, title_start: u64) {
    heaps().push(Heap { cpu, gpu, slots, step, title, title_start });
}

pub(crate) fn forget(cpu: usize) {
    heaps().retain(|h| h.cpu != cpu);
}

/// The view heap and slot of CPU handle `ptr`, when that slot is watched.
fn watched_slot(ptr: usize) -> Option<(Heap, u32)> {
    let h = *heaps().iter().find(|h| ptr >= h.cpu && ptr < h.cpu + h.slots as usize * h.step as usize)?;
    let slot = ((ptr - h.cpu) / h.step as usize) as u32;
    slots().binary_search(&slot).ok().map(|_| (h, slot))
}

/// On WARP, a descriptor's 64 bytes at `at` (CPU memory there): the kind
/// nibble (0 never written, 1 UAV, 2 SRV, 3 CBV) and qwords 0-3,
/// which the JIT reads through (qword 1 + 0x388 was the fault).
pub(crate) fn warp_words(at: usize, step: u32) -> String {
    if !xnw_device::warp() || step < 64 || at == 0 {
        return "words: not WARP".into();
    }
    // SAFETY: a slot of a live WARP heap (CPU memory); a racing writer
    // only makes the report approximate.
    let q: [u64; 8] = std::array::from_fn(|k| unsafe { std::ptr::read_volatile((at + 8 * k) as *const u64) });
    let kind = ["never written", "UAV", "SRV", "CBV"].get((q[7] >> 60) as usize).copied().unwrap_or("?");
    format!("WARP kind {} ({kind}), q0 {:#x} q1 {:#x} q2 {:#x} q3 {:#x} q4 {:#x} q7 {:#x}", q[7] >> 60, q[0], q[1], q[2], q[3], q[4], q[7])
}

/// A resource in a few words: dimension, size, format, flags.
pub(crate) fn resource_text(r: Option<&ID3D12Resource>) -> String {
    let Some(r) = r else { return "no resource (null view)".into() };
    // SAFETY: a live resource.
    let d = unsafe { r.GetDesc() };
    // SAFETY: as above.
    let va = unsafe { r.GetGPUVirtualAddress() };
    let dim = ["unknown", "buffer", "tex1d", "tex2d", "tex3d"].get(d.Dimension.0 as usize).copied().unwrap_or("?");
    format!("resource {:#x} {dim} {}x{}x{} mips {} format {} flags {:#x} va {va:#x}", windows::core::Interface::as_raw(r) as usize, d.Width, d.Height, d.DepthOrArraySize, d.MipLevels, d.Format.0, d.Flags.0)
}

/// An SRV description in a few words.
pub(crate) fn srv_text(d: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>) -> String {
    let Some(d) = d else { return "default view".into() };
    let mut t = format!("view dimension {} format {}", d.ViewDimension.0, d.Format.0);
    if d.ViewDimension == D3D12_SRV_DIMENSION_BUFFER {
        // SAFETY: the buffer arm, by the dimension.
        let b = unsafe { d.Anonymous.Buffer };
        let kind = if b.Flags.0 & 1 != 0 { "raw" } else if b.StructureByteStride > 0 { "structured" } else { "typed" };
        t += &format!(" buffer {kind} first {} elements {} stride {}", b.FirstElement, b.NumElements, b.StructureByteStride);
    } else if d.ViewDimension == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE {
        t += " acceleration structure";
    } else {
        // SAFETY: every texture arm starts with MostDetailedMip, MipLevels.
        let m = unsafe { d.Anonymous.Texture2D };
        t += &format!(" texture mips {}+{}", m.MostDetailedMip, m.MipLevels);
    }
    t
}

/// A UAV description in a few words.
pub(crate) fn uav_text(d: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>) -> String {
    let Some(d) = d else { return "default view".into() };
    let mut t = format!("view dimension {} format {}", d.ViewDimension.0, d.Format.0);
    if d.ViewDimension == D3D12_UAV_DIMENSION_BUFFER {
        // SAFETY: the buffer arm, by the dimension.
        let b = unsafe { d.Anonymous.Buffer };
        let kind = if b.Flags.0 & 1 != 0 { "raw" } else if b.StructureByteStride > 0 { "structured" } else { "typed" };
        t += &format!(" buffer {kind} first {} elements {} stride {} counter offset {}", b.FirstElement, b.NumElements, b.StructureByteStride, b.CounterOffsetInBytes);
    } else {
        // SAFETY: the texture arms start with MipSlice.
        t += &format!(" texture mip {}", unsafe { d.Anonymous.Texture2D.MipSlice });
    }
    t
}

/// Writes at watched slots logged so far (each with a short stack while
/// few).
static WRITES: AtomicU64 = AtomicU64::new(0);

/// A gated write of `n` slots at `at` of kind `what`, described by
/// `text` (given the watched slot's place in the write): logged when it
/// covers a watched slot.
pub(crate) fn write(at: usize, n: u32, what: &str, text: impl FnOnce(u32) -> String) {
    if !on() {
        return;
    }
    let step = step_of(at);
    let hit = (0..n.max(1)).find_map(|k| watched_slot(at + k as usize * step).map(|w| (w, k)));
    let Some(((h, slot), k)) = hit else { return };
    let text = text(k);
    let k = WRITES.fetch_add(1, Ordering::Relaxed) + 1;
    let place = if slot < h.title { "title" } else { "dynamic" };
    xnw_core::info!(
        "slot watch: write {k} {what} at view heap {:#x} ({} slots) {place} slot {slot}{}: {text}; frame {}",
        h.cpu,
        h.slots,
        if n > 1 { format!(" (in {n})") } else { String::new() },
        super::super::queue::frames_presented()
    );
    if k <= 8 || k.is_power_of_two() {
        xnw_core::info!("slot watch: write {k} from {}", caller());
    }
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    match last.iter_mut().find(|l| l.0 == h.cpu && l.1 == slot) {
        Some(l) => l.2 = format!("write {k} {what}: {text}"),
        None => last.push((h.cpu, slot, format!("write {k} {what}: {text}"))),
    }
}

/// Bytes per slot of the heap holding `at` (1 when none does).
fn step_of(at: usize) -> usize {
    heaps().iter().find(|h| at >= h.cpu && at < h.cpu + h.slots as usize * h.step as usize).map_or(1, |h| h.step as usize)
}

/// The xnw functions on the stack, outermost last, without this module's
/// and the gate's.
fn caller() -> String {
    let bt = std::backtrace::Backtrace::force_capture().to_string();
    let names: Vec<&str> = bt
        .lines()
        .map(str::trim)
        .filter_map(|l| l.split_once(": ").map(|(_, f)| f))
        .filter(|f| f.starts_with("xnw") && !f.contains("slotwatch") && !f.contains("::gate::"))
        .take(6)
        .collect();
    if names.is_empty() { "?".into() } else { names.join(" < ") }
}

/// Dispatches reported so far (all of the first few, then powers of two).
static DISPATCHES: AtomicU32 = AtomicU32::new(0);

/// Whether this dispatch of a watched program is reported.
pub(crate) fn dispatch_reported() -> Option<u32> {
    let k = DISPATCHES.fetch_add(1, Ordering::Relaxed) + 1;
    (k <= 4 || k.is_power_of_two()).then_some(k)
}

/// The crash report: for each of `regs` (name, value) pointing into a view
/// heap's CPU or GPU range, or equal to a watched slot number, what is
/// there. Called at the first access violation outside every module.
pub fn on_fault(regs: &[(&str, u64)]) {
    for line in fault_lines(regs) {
        xnw_core::warn!("  {line}");
    }
}

/// [`on_fault`]'s lines.
fn fault_lines(regs: &[(&str, u64)]) -> Vec<String> {
    let heaps = heaps().clone();
    let mut out = Vec::new();
    let lanes: Vec<String> = regs.iter().filter(|r| r.0.starts_with("xmm")).map(|&(n, v)| format!("{n} {v:#x}")).collect();
    if !lanes.is_empty() {
        out.push(format!("slot watch: vector lanes {}", lanes.join(" ")));
    }
    for &(name, v) in regs {
        // A vector lane below a title heap's slot count: a slot index (the
        // JIT's per-lane descriptor index), described in the largest heap.
        if name.starts_with("xmm") {
            let Some(h) = heaps.iter().filter(|h| v < u64::from(h.title)).max_by_key(|h| h.title) else { continue };
            let slot = v as u32;
            let at = h.cpu + slot as usize * h.step as usize;
            let last = LAST.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|l| l.0 == h.cpu && l.1 == slot).map(|l| l.2.clone());
            out.push(format!(
                "slot watch: {name} {v:#x} as a slot of view heap {:#x}: {}; title record {}; last watched write: {}",
                h.cpu,
                warp_words(at, h.step),
                title_record(h.title_start, slot, h.title),
                last.as_deref().unwrap_or("-")
            ));
            continue;
        }
        for h in &heaps {
            let bytes = u64::from(h.slots) * u64::from(h.step);
            let (base, kind) = if v >= h.cpu as u64 && v < h.cpu as u64 + bytes {
                (h.cpu as u64, "CPU")
            } else if v >= h.gpu && v < h.gpu + bytes {
                (h.gpu, "GPU")
            } else {
                continue;
            };
            let slot = ((v - base) / u64::from(h.step)) as u32;
            let at = h.cpu + slot as usize * h.step as usize;
            let last = LAST.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|l| l.0 == h.cpu && l.1 == slot).map(|l| l.2.clone());
            out.push(format!(
                "slot watch: {name} {v:#x} is in view heap {:#x} ({} slots, {} title) {kind} at slot {slot} +{:#x}: {}; title record {}; last watched write: {}",
                h.cpu,
                h.slots,
                h.title,
                (v - base) % u64::from(h.step),
                warp_words(at, h.step),
                title_record(h.title_start, slot, h.title),
                last.as_deref().unwrap_or("none (slot not watched or not written since)")
            ));
        }
    }
    out
}

/// The record the title wrote at title slot `slot` of the heap at
/// `title_start` (32 bytes a view on the console), in a few words.
pub(crate) fn title_record(title_start: u64, slot: u32, title: u32) -> String {
    if title_start == 0 || slot >= title {
        return "-".into();
    }
    let at = title_start + u64::from(slot) * 32;
    if !super::super::shadow::in_buffer(at, 32) && !super::host::committed(at, 32) {
        return format!("{at:#x}: not readable");
    }
    // SAFETY: committed title memory, 32 bytes.
    let w = unsafe { (at as *const [u32; 8]).read_unaligned() };
    // SAFETY: as above.
    let id = unsafe { super::record_id(at) };
    let view = id.and_then(|(i, g)| super::view_at(i, g)).map(|v| format!("{v:?}").chars().take(160).collect::<String>());
    format!("{at:#x} words {:08x?} view {}", w, view.as_deref().unwrap_or("none (no tag)"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::d3d12x;
    use crate::d3d12x::descriptor::gate;
    use crate::d3d12x::descriptor::host::HostHeap;
    use windows::Win32::Graphics::Dxgi::Common::*;

    /// The instrument on the software adapter: slot 3 of a host view heap
    /// written as a CBV, a texture SRV, a raw buffer UAV and a null SRV
    /// through the gated writes; each write is noted with its kind, WARP's
    /// kind nibble follows it, and a register pointing at the slot (GPU
    /// range, as the JIT's was) is named with the last write. Run:
    /// `XNW_TEST_GPU=1 XNW_ADAPTER=warp XNW_WARP=<dir> XNW_AGILITY=<dir>
    /// cargo test -p xnw-gpu --lib slotwatch -- --nocapture`.
    #[test]
    fn a_watched_slot_is_named_by_kind() {
        if std::env::var_os("XNW_AGILITY").is_none_or(|d| d.is_empty()) {
            println!("SKIPPED: set XNW_AGILITY to the Agility SDK folder to run on the host GPU");
            return;
        }
        // SAFETY: set before any thread of this test reads it (run alone).
        unsafe { std::env::set_var("XNW_SLOT_WATCH", "3") };
        assert_eq!(slots(), [3], "XNW_SLOT_WATCH read before this test set it: run the test alone");
        let host = d3d12x::host::host().expect("a host GPU");
        let device = &host.device;
        let heap = HostHeap::new(false, 8, 0).expect("a view heap");
        let h = *heaps().iter().find(|h| h.slots >= 8 && h.title == 8).expect("registered");
        let at = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: h.cpu + 3 * h.step as usize };
        let make = |texture: bool| {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: if texture { D3D12_RESOURCE_DIMENSION_TEXTURE2D } else { D3D12_RESOURCE_DIMENSION_BUFFER },
                Width: if texture { 16 } else { 4096 },
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: if texture { DXGI_FORMAT_R8G8B8A8_UNORM } else { DXGI_FORMAT_UNKNOWN },
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: if texture { D3D12_TEXTURE_LAYOUT_UNKNOWN } else { D3D12_TEXTURE_LAYOUT_ROW_MAJOR },
                Flags: if texture { D3D12_RESOURCE_FLAG_NONE } else { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS },
                ..Default::default()
            };
            let mut r: Option<ID3D12Resource> = None;
            // SAFETY: plain creation.
            unsafe { device.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap() };
            r.unwrap()
        };
        let (texture, buffer) = (make(true), make(false));
        let last = || LAST.lock().unwrap().iter().find(|l| l.0 == h.cpu && l.1 == 3).map(|l| l.2.clone()).unwrap_or_default();
        let fault = || fault_lines(&[("rbx", 3), ("rcx", h.gpu + 3 * u64::from(h.step) + 8)]).join(" | ");
        // SAFETY: slots of a live heap, live resources, descriptions on the stack.
        unsafe {
            gate::cbv(device, Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: buffer.GetGPUVirtualAddress(), SizeInBytes: 256 }), at);
            println!("cbv: {} | {}", last(), warp_words(at.ptr, h.step));
            assert!(last().contains("CBV: location"));
            gate::srv(device, &texture, None, at);
            println!("srv: {} | {}", last(), warp_words(at.ptr, h.step));
            assert!(last().contains("SRV: default view, resource") && last().contains("tex2d 16x1x1"));
            let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: D3D12_BUFFER_UAV { FirstElement: 0, NumElements: 1024, StructureByteStride: 0, CounterOffsetInBytes: 0, Flags: D3D12_BUFFER_UAV_FLAG_RAW } },
            };
            gate::uav(device, &buffer, None::<&ID3D12Resource>, Some(&uav), at);
            println!("uav: {} | {}", last(), warp_words(at.ptr, h.step));
            assert!(last().contains("buffer raw first 0 elements 1024"));
            let null = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: 0x1688,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } },
            };
            gate::srv(device, None::<&ID3D12Resource>, Some(&null), at);
            println!("null srv: {} | {}", last(), warp_words(at.ptr, h.step));
            assert!(last().contains("no resource (null view)"));
        }
        let f = fault();
        println!("fault: {f}");
        assert!(f.contains("at slot 3 +0x8") && f.contains("last watched write: write 4 SRV"), "{f}");
        if xnw_device::warp() {
            assert!(warp_words(at.ptr, h.step).starts_with("WARP kind 2"));
        }
        drop(heap);
    }
}
