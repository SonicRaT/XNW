//! `XNW_LOOP_COUNTS=1`: shader loop iteration counts in a buffer of
//! process memory that survives a device removal, so the failure report
//! names the programs whose loops ran longest. The runtime half; the
//! translator's half is the contract below. Idea and report shape from
//! VIctory (open, `crates/victory-gpu/src/breadcrumb.rs`,
//! GPL-3.0-or-later): their slots of 8 words, cleared before each
//! submission, top loops by count with the pipelines that use the slot;
//! ours sits over our own memory (as the crumbs, `crumbs::process_buffer`)
//! and is reached through a reserved view heap slot.
//!
//! The contract for the translator (xnw-shader), with the switch on (it
//! must be part of the shader cache key):
//! - The program declares one more user-data dword, of root type
//!   `RootType::Other(ROOT)` ([`ROOT`] = 0x4C43, "LC"). The runtime puts
//!   there the index, in the bound CBV/SRV/UAV heap (SM 6.6
//!   `ResourceDescriptorHeap[]`), of a `RWByteAddressBuffer` of
//!   [`SLOTS`] x [`LOOPS`] u32 words (`State::user_value`, draw.rs).
//! - The program's slot is its id: the first 8 hex digits of its 128-bit
//!   hash as the runtime logs it (`compute 2a900415...`: 0x2a900415),
//!   modulo [`SLOTS`]. Loop `k` is the k-th loop in program order; loops
//!   past the 8th use word 7.
//! - At each loop's back edge, every 64 iterations and at its exit, the
//!   first active lane (`WaveIsFirstLane`) does
//!   `InterlockedMax(word (slot * 8 + k) * 4, iterations so far)`, with
//!   the wave's largest count (`WaveActiveMax`). One atomic per wave per
//!   64 iterations; nothing else changes.
//!
//! Cost unset: none (one cached test per heap made and per submission).

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS;

/// The user-data root type the translator asks the heap index with.
pub(crate) const ROOT: u32 = 0x4C43;
/// Programs (slots) and loops per program.
pub(crate) const SLOTS: u32 = 1024;
pub(crate) const LOOPS: u32 = 8;
const BYTES: usize = (SLOTS * LOOPS * 4) as usize;
/// How many loops the report names.
const REPORTED: usize = 20;

/// `XNW_LOOP_COUNTS=1`.
pub(crate) fn on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        let on = std::env::var("XNW_LOOP_COUNTS").is_ok_and(|v| v.trim() == "1");
        if on {
            xnw_core::info!("d3d12x: shader loop counts on (XNW_LOOP_COUNTS=1): {SLOTS} programs x {LOOPS} loops, user-data root type {ROOT:#x}");
        }
        on
    });
    *ON
}

struct Buffer {
    memory: *mut u32,
    buffer: ID3D12Resource,
    _heap: ID3D12Heap,
}
// SAFETY: D3D12 objects are free-threaded; the memory is ours for the run.
unsafe impl Send for Buffer {}
// SAFETY: as above.
unsafe impl Sync for Buffer {}

fn buffer() -> Option<&'static Buffer> {
    static BUFFER: OnceLock<Option<Buffer>> = OnceLock::new();
    BUFFER
        .get_or_init(|| {
            let device = &super::host::host()?.device;
            match super::crumbs::process_buffer(device, BYTES, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {
                Ok((memory, _, heap, buffer)) => Some(Buffer { memory: memory.cast(), buffer, _heap: heap }),
                Err(e) => {
                    xnw_core::warn!("d3d12x: no loop count buffer: {e}");
                    None
                }
            }
        })
        .as_ref()
}

/// Makes the buffer's raw UAV at `at` (a view heap's reserved slot).
pub(crate) fn describe(device: &ID3D12Device, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(b) = buffer() else { return };
    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV { NumElements: (BYTES / 4) as u32, Flags: D3D12_BUFFER_UAV_FLAG_RAW, ..Default::default() },
        },
    };
    // SAFETY: a slot of a live heap and a live buffer.
    unsafe { device.CreateUnorderedAccessView(&b.buffer, None, Some(&desc), at) };
}

/// Zeroes the counts before a submission (every one is waited for).
pub(crate) fn clear() {
    if !on() {
        return;
    }
    if let Some(b) = buffer() {
        // SAFETY: our memory; no submission is running.
        unsafe { std::ptr::write_bytes(b.memory, 0, BYTES / 4) };
    }
}

/// Program slots by the pipelines naming them.
static NAMES: Mutex<BTreeMap<u32, Vec<String>>> = Mutex::new(BTreeMap::new());

/// The slot of a program by its logged 128-bit hash (32 hex digits).
pub(crate) fn slot_of(hash: &str) -> Option<u32> {
    (hash.len() == 32 && hash.bytes().all(|b| b.is_ascii_hexdigit())).then(|| u32::from_str_radix(&hash[..8], 16).ok()).flatten().map(|v| v % SLOTS)
}

/// Remembers which programs pipeline `name` uses (crumbs/loop counts on).
pub(crate) fn note_pipeline(name: &str) {
    if !on() {
        return;
    }
    let mut names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
    for word in name.split(|c: char| !c.is_ascii_hexdigit()) {
        if let Some(slot) = slot_of(word) {
            let list = names.entry(slot).or_default();
            if !list.iter().any(|n| n == name) && list.len() < 4 {
                list.push(name.to_owned());
            }
        }
    }
}

/// The loops that counted most, as report lines (most first).
fn summarize(words: &[u32], names: &BTreeMap<u32, Vec<String>>) -> Vec<String> {
    let mut loops: Vec<(u32, u32, u32)> = words.iter().enumerate().filter(|&(_, &c)| c > 0).map(|(i, &c)| (c, i as u32 / LOOPS, i as u32 % LOOPS)).collect();
    loops.sort_unstable_by(|a, b| b.cmp(a));
    loops
        .into_iter()
        .take(REPORTED)
        .map(|(count, slot, k)| {
            let who = names.get(&slot).map_or_else(|| "a program not seen bound".to_owned(), |n| n.join(" | "));
            format!("loop {k} of program slot {slot} ran {count} iterations ({who})")
        })
        .collect()
}

/// On a failure: the loops that counted most in the last submission.
pub(crate) fn report() {
    if !on() {
        return;
    }
    let Some(b) = buffer() else { return };
    // SAFETY: our memory; the device is gone or idle.
    let words: Vec<u32> = (0..BYTES / 4).map(|i| unsafe { b.memory.add(i).read_volatile() }).collect();
    let lines = summarize(&words, &NAMES.lock().unwrap_or_else(|e| e.into_inner()));
    if lines.is_empty() {
        xnw_core::warn!("loop counts: no loop counted in the last submission (translated programs emit them only with the translator's half)");
    }
    for line in lines {
        xnw_core::warn!("loop counts: {line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_longest_loops_are_named_with_their_pipelines() {
        let mut words = vec![0u32; (SLOTS * LOOPS) as usize];
        let at = |slot: u32, k: u32| (slot * LOOPS + k) as usize;
        let slot = slot_of("2a900415a7de75d26b87efe1039dc774").unwrap();
        assert_eq!(slot, 0x2a90_0415 % SLOTS);
        words[at(slot, 0)] = 17;
        words[at(slot, 2)] = 1 << 20;
        words[at(900, 7)] = 5;
        let mut names = BTreeMap::new();
        names.insert(slot, vec!["compute 2a900415a7de75d26b87efe1039dc774".to_owned()]);
        let lines = summarize(&words, &names);
        assert_eq!(lines.len(), 3);
        assert!(lines[0].starts_with(&format!("loop 2 of program slot {slot} ran 1048576 iterations (compute 2a900415")), "{}", lines[0]);
        assert!(lines[2].contains("a program not seen bound"));
        let many: Vec<u32> = (1..=40).collect();
        assert_eq!(summarize(&many, &BTreeMap::new()).len(), REPORTED);
        assert_eq!(slot_of("not a hash"), None);
    }
}
