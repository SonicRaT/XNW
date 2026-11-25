//! The address map translated shaders read title memory through when they
//! compute its address (a pointer they load, the BVH a ray-tracing pass
//! walks): see `xnw_shader::emit::address_map` for its layout. It lists
//! the title's buffers and the copies of title memory made so far
//! ([`super::shadow`]), each through raw views in dynamic slots of the view
//! heap a list binds.
//!
//! Ported from VIctory (`crates/victory-gpu/src/address_map.rs`,
//! commit fe078c6, GPL-3.0-or-later). Changed (XNW):
//! - with `XNW_SHARED=2` a buffer whose range resolves in a shared window
//!   is read there in place (what in-place views write is there, not in a
//!   copy), and buffers without a copy are listed when they resolve;
//! - the map (pieces, entries, words, and its views in a CPU-only heap) is
//!   made once per change of the buffers, copies or windows
//!   ([`super::shadow::generation`], [`crate::shared::remaps`]); a list
//!   then copies the views into its slots in one call and the words into
//!   its upload memory, with the view numbers moved to those slots;
//! - its views and copies are held by the list, its counters on a frame
//!   statistics line.
//!
//! shortcut (VIctory's): every copy in the map is brought up to date (the
//! pages the title wrote since) once per list that binds it, and what
//! shaders write through a copy stays there (no copy-back); writes through
//! an in-place window are title memory.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS;
use windows::core::Interface;
use xnw_shader::emit::address_map::{ENTRY, HEADER, Range, encode};

use super::descriptor::host::HostHeap;
use super::host::HostList;
use super::shadow::Shadow;

/// The most bytes one raw view holds: D3D12 buffer views have at most 2^27
/// elements.
const VIEW_BYTES: u64 = 4 << 27;

/// Pieces of title memory the spans `(start, len)` cover, sorted and not
/// overlapping, each with the span that holds it: where spans overlap, the
/// one starting last (the innermost buffer, as [`super::shadow::find`]
/// picks) holds the piece.
///
/// Changed from VIctory (same result): VIctory sweeps start and end
/// points through an ordered set of the open spans, 4.6 ms for 20,000
/// spans (`a_map_of_twenty_thousand_buffers_is_made_quickly`). Here the
/// spans in start order go on a stack: its top is the open span starting
/// last; spans that ended are dropped when they come to the top.
pub(crate) fn pieces(spans: &[(u64, u64)]) -> Vec<(u64, u64, usize)> {
    let end = |i: usize| spans[i].0.saturating_add(spans[i].1);
    let mut order: Vec<usize> = (0..spans.len()).filter(|&i| spans[i].1 > 0).collect();
    order.sort_unstable_by_key(|&i| (spans[i].0, i));
    let mut stack: Vec<usize> = Vec::new();
    let mut out: Vec<(u64, u64, usize)> = Vec::new();
    let (mut at, mut next) = (0u64, 0usize);
    loop {
        while stack.last().is_some_and(|&t| end(t) <= at) {
            stack.pop();
        }
        let x = match (order.get(next).map(|&i| spans[i].0), stack.last().map(|&t| end(t))) {
            (None, None) => break,
            (Some(s), None) => s,
            (None, Some(e)) => e,
            (Some(s), Some(e)) => s.min(e),
        };
        if let Some(&holder) = stack.last()
            && x > at
        {
            match out.last_mut() {
                Some(last) if last.2 == holder && last.0 + last.1 == at => last.1 += x - at,
                _ => out.push((at, x - at, holder)),
            }
        }
        at = x;
        while let Some(&i) = order.get(next).filter(|&&i| spans[i].0 == x) {
            stack.push(i);
            next += 1;
        }
    }
    out
}

/// The map's entries for `pieces` of the spans: each piece cut where its
/// holder's views ([`VIEW_BYTES`] each) end, with the view (holder, view
/// number within it) and the piece's offset in that view. `bases`: per
/// span, its holder and the byte in the holder its first byte is at.
/// (VIctory: the span is its own holder, at byte 0.)
fn entries(spans: &[(u64, u64)], bases: &[(usize, u64)], pieces: &[(u64, u64, usize)]) -> Vec<(u64, u32, (usize, u64), u32)> {
    let mut out = Vec::new();
    for &(start, len, span) in pieces {
        let (holder, base) = bases[span];
        let mut at = start;
        let end = start + len;
        while at < end {
            let rel = base + (at - spans[span].0);
            let view = rel / VIEW_BYTES;
            let view_end = at + ((view + 1) * VIEW_BYTES - rel);
            let next = end.min(view_end);
            out.push((at, (next - at) as u32, (holder, view), (rel - view * VIEW_BYTES) as u32));
            at = next;
        }
    }
    out
}

/// What holds a span's bytes on the host.
enum Holder {
    /// A host copy of title memory.
    Copy(Arc<Shadow>),
    /// A shared window over title memory (`XNW_SHARED=2`).
    Window(ID3D12Resource),
}

impl Holder {
    fn buffer(&self) -> &ID3D12Resource {
        match self {
            Holder::Copy(c) => &c.buffer,
            Holder::Window(b) => b,
        }
    }
}

/// A map made for one state of the buffers, copies and windows.
struct Built {
    /// [`key`] when it was made.
    key: Key,
    holders: Vec<Holder>,
    /// The raw views, one per (holder, view number), made in `cpu`.
    views: usize,
    cpu: Option<ID3D12DescriptorHeap>,
    /// The map's words, view numbers counted from 0.
    words: Vec<u32>,
    entries: usize,
}

// SAFETY: D3D12 objects are free-threaded; the rest is plain data.
unsafe impl Send for Built {}
// SAFETY: as above.
unsafe impl Sync for Built {}

/// What a map made now would be made from.
/// Buffers without a copy are listed only in place, so without
/// `XNW_SHARED=2` their coming and going does not count.
pub(crate) fn key() -> Key {
    let (copies, buffers) = super::shadow::generation();
    let in_place = crate::shared::in_place();
    if in_place { (copies, buffers, crate::shared::remaps(), true) } else { (copies, 0, 0, false) }
}

/// What a map was made from: copies, buffers and windows (their change
/// counts), whether in place.
pub(crate) type Key = (u64, u64, u64, bool);

static BUILT: Mutex<Option<Arc<Built>>> = Mutex::new(None);

/// Maps made (rebuilds), the time they took (us), the largest; maps bound
/// to lists; entries of the last.
static BUILDS: AtomicU64 = AtomicU64::new(0);
static BUILD_US: AtomicU64 = AtomicU64::new(0);
static BUILD_MAX_US: AtomicU64 = AtomicU64::new(0);
static BOUND: AtomicU64 = AtomicU64::new(0);
/// Time spent giving lists their map (views copied, words written, copies
/// queued for refresh), us.
static BIND_US: AtomicU64 = AtomicU64::new(0);
static ENTRIES: AtomicU64 = AtomicU64::new(0);
static IN_PLACE: AtomicU64 = AtomicU64::new(0);
/// Lists the map could not be bound to (no dynamic slots, no upload memory).
static FAILED: AtomicU64 = AtomicU64::new(0);

/// The current map, made again when what it lists changed.
fn built() -> Option<Arc<Built>> {
    let now = key();
    let mut b = BUILT.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(b) = b.as_ref().filter(|b| b.key == now) {
        return Some(b.clone());
    }
    let started = std::time::Instant::now();
    let made = Arc::new(build(now)?);
    let us = started.elapsed().as_micros() as u64;
    BUILDS.fetch_add(1, Ordering::Relaxed);
    BUILD_US.fetch_add(us, Ordering::Relaxed);
    BUILD_MAX_US.fetch_max(us, Ordering::Relaxed);
    ENTRIES.store(made.entries as u64, Ordering::Relaxed);
    *b = Some(made.clone());
    Some(made)
}

fn build(key: Key) -> Option<Built> {
    let in_place = key.3;
    let mut spans: Vec<(u64, u64)> = Vec::new();
    let mut bases: Vec<(usize, u64)> = Vec::new();
    let mut holders: Vec<Holder> = Vec::new();
    let mut windows: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut placed = 0;
    for (start, len, copy) in super::shadow::all() {
        if len == 0 {
            continue;
        }
        let window = if in_place { crate::shared::lookup(start, len) } else { None };
        let base = match (window, copy) {
            (Some(r), _) => {
                placed += 1;
                let n = holders.len();
                let h = *windows.entry(r.buffer.as_raw() as usize).or_insert(n);
                if h == n {
                    holders.push(Holder::Window(r.buffer));
                }
                (h, r.offset)
            }
            (None, Some(c)) => {
                holders.push(Holder::Copy(c));
                (holders.len() - 1, 0)
            }
            (None, None) => continue,
        };
        spans.push((start, len));
        bases.push(base);
    }
    IN_PLACE.store(placed, Ordering::Relaxed);
    let pieces = pieces(&spans);
    let entries = entries(&spans, &bases, &pieces);
    let mut views: Vec<(usize, u64)> = entries.iter().map(|e| e.2).collect();
    views.sort_unstable();
    views.dedup();
    let cpu = if views.is_empty() { None } else { Some(raw_views(&holders, &views)?) };
    let ranges: Vec<Range> = entries
        .iter()
        .map(|&(start, len, view, view_offset)| Range { start, len, view: views.binary_search(&view).expect("listed") as u32, view_offset })
        .collect();
    Some(Built { key, holders, views: views.len(), cpu, words: encode(&ranges), entries: ranges.len() })
}

/// A CPU-only heap with a raw unordered access view per (holder, view
/// number): `VIEW_BYTES` of the holder's buffer from that view's start on.
fn raw_views(holders: &[Holder], views: &[(usize, u64)]) -> Option<ID3D12DescriptorHeap> {
    let host = super::host::host()?;
    // SAFETY: plain heap creation and views of live buffers into it.
    unsafe {
        let heap: ID3D12DescriptorHeap = host
            .device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: views.len() as u32, ..Default::default() })
            .ok()?;
        let start = heap.GetCPUDescriptorHandleForHeapStart().ptr;
        let step = host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        for (k, &(holder, view)) in views.iter().enumerate() {
            let buffer = holders[holder].buffer();
            let width = buffer.GetDesc().Width;
            let offset = view * VIEW_BYTES;
            let bytes = width.saturating_sub(offset).min(VIEW_BYTES);
            let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV { FirstElement: offset / 4, NumElements: (bytes / 4) as u32, Flags: D3D12_BUFFER_UAV_FLAG_RAW, ..Default::default() },
                },
            };
            host.device.CreateUnorderedAccessView(buffer, None, Some(&desc), D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start + k * step });
        }
        Some(heap)
    }
}

/// The address map for draws of `list` while `heap` is its view heap: its
/// host GPU address, and the dynamic slots its views take (to give back
/// when the list is reset). Each copy is brought up to date when the list
/// is submitted.
pub(crate) fn make(list: &HostList, heap: &HostHeap) -> Option<(u64, (u32, u32))> {
    let b = built();
    let started = std::time::Instant::now();
    let made = (|| {
        let b = b?;
        let n = b.views as u32;
        let first = if n == 0 { 0 } else { heap.dynamic(n)? };
        if let Some(cpu) = &b.cpu {
            // SAFETY: a live heap of ours.
            heap.copy_views_in(first, n, unsafe { cpu.GetCPUDescriptorHandleForHeapStart() });
            list.keep(cpu);
        }
        for h in &b.holders {
            match h {
                Holder::Copy(c) => {
                    if list.first_use(Arc::as_ptr(c) as usize ^ MAP_KEY) {
                        let (start, len) = c.span();
                        c.refresh(list, start, len);
                    }
                }
                Holder::Window(w) => list.keep(w),
            }
        }
        // Room for one entry even when there is none: the shader reads it.
        let bytes = u64::from(HEADER) + u64::from(ENTRY) * (b.entries.max(1) as u64);
        let (gpu, to) = list.scratch(bytes)?;
        // SAFETY: `bytes` of the list's upload memory, which outlives the
        // list's run; the words fit.
        unsafe {
            std::ptr::write_bytes(to, 0, bytes as usize);
            std::ptr::copy_nonoverlapping(b.words.as_ptr().cast::<u8>(), to, b.words.len() * 4);
            // The view numbers, moved to the list's slots.
            let words = to.cast::<u32>();
            for i in 0..b.entries {
                *words.add((HEADER / 4) as usize + i * (ENTRY / 4) as usize + 3) += first;
            }
        }
        Some((gpu, (first, n)))
    })();
    BIND_US.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
    if made.is_some() {
        BOUND.fetch_add(1, Ordering::Relaxed);
    } else {
        FAILED.fetch_add(1, Ordering::Relaxed);
    }
    made
}

/// Keeps the map's refreshes apart from other first uses the list asks
/// about (texture keys are addresses too).
const MAP_KEY: usize = 0x5;

/// The frame statistics part (totals since start).
pub(crate) fn stats_text() -> String {
    let builds = BUILDS.load(Ordering::Relaxed);
    format!(
        "{builds} maps made ({} us each on average, {} us at most), {} entries in the last ({} buffers in place), bound to {} lists ({} us each on average), {} not bound",
        BUILD_US.load(Ordering::Relaxed) / builds.max(1),
        BUILD_MAX_US.load(Ordering::Relaxed),
        ENTRIES.load(Ordering::Relaxed),
        IN_PLACE.load(Ordering::Relaxed),
        BOUND.load(Ordering::Relaxed),
        BIND_US.load(Ordering::Relaxed) / BOUND.load(Ordering::Relaxed).max(1),
        FAILED.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn own(spans: &[(u64, u64)]) -> Vec<(usize, u64)> {
        (0..spans.len()).map(|i| (i, 0)).collect()
    }

    #[test]
    fn inner_buffers_win_and_pieces_do_not_overlap() {
        // A heap-sized buffer with two placed inside it, one past its end,
        // and an empty one.
        let spans = [(0x1000, 0x1000), (0x1200, 0x100), (0x1800, 0x100), (0x3000, 0x40), (0x5000, 0)];
        assert_eq!(pieces(&spans), [(0x1000, 0x200, 0), (0x1200, 0x100, 1), (0x1300, 0x500, 0), (0x1800, 0x100, 2), (0x1900, 0x700, 0), (0x3000, 0x40, 3),]);
    }

    /// The stack sweep gives what VIctory's rule gives (each stretch held by
    /// the covering span starting last, ties by later index) on spans that
    /// nest, overlap in part, repeat and are empty.
    #[test]
    fn the_sweep_agrees_with_the_rule_everywhere() {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = |m: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % m
        };
        for _ in 0..200 {
            let spans: Vec<(u64, u64)> = (0..rnd(12) + 1).map(|_| (rnd(64), rnd(24))).collect();
            let mut want: Vec<(u64, u64, usize)> = Vec::new();
            for x in 0..96u64 {
                let holder = spans.iter().enumerate().filter(|&(_, &(s, l))| s <= x && x < s + l).max_by_key(|&(i, &(s, _))| (s, i)).map(|(i, _)| i);
                if let Some(h) = holder {
                    match want.last_mut() {
                        Some(last) if last.2 == h && last.0 + last.1 == x => last.1 += 1,
                        _ => want.push((x, 1, h)),
                    }
                }
            }
            assert_eq!(pieces(&spans), want, "{spans:?}");
        }
    }

    #[test]
    fn the_same_span_twice_is_one_piece() {
        assert_eq!(pieces(&[(0x100, 0x100), (0x100, 0x100)]), [(0x100, 0x100, 1)]);
    }

    #[test]
    fn pieces_are_cut_where_views_end() {
        let spans = [(0x10, 2 * VIEW_BYTES + 8)];
        let e = entries(&spans, &own(&spans), &pieces(&spans));
        assert_eq!(e, [(0x10, VIEW_BYTES as u32, (0, 0), 0), (0x10 + VIEW_BYTES, VIEW_BYTES as u32, (0, 1), 0), (0x10 + 2 * VIEW_BYTES, 8, (0, 2), 0),]);
    }

    /// XNW: spans in one window share its views, at their offset in it; a
    /// span crossing a view end of its window is cut there.
    #[test]
    fn spans_in_a_window_are_read_at_their_offset() {
        let spans = [(0x1_0000, 0x100), (0x9_0000, 0x200)];
        let bases = [(0, 0x40), (0, VIEW_BYTES - 0x80)];
        let e = entries(&spans, &bases, &pieces(&spans));
        assert_eq!(e, [(0x1_0000, 0x100, (0, 0), 0x40), (0x9_0000, 0x80, (0, 0), (VIEW_BYTES - 0x80) as u32), (0x9_0080, 0x180, (0, 1), 0)]);
    }

    /// The cost question: the CPU part of a map (pieces, entries, words)
    /// for 20,000 buffers, a few placed inside larger ones.
    #[test]
    fn a_map_of_twenty_thousand_buffers_is_made_quickly() {
        let spans: Vec<(u64, u64)> = (0..20_000u64).map(|i| if i % 10 == 0 { (i * 0x10000, 0x30000) } else { (i * 0x10000 + 0x100, 0x800) }).collect();
        let bases = own(&spans);
        let started = std::time::Instant::now();
        let p = pieces(&spans);
        let t1 = started.elapsed().as_micros();
        let e = entries(&spans, &bases, &p);
        let t2 = started.elapsed().as_micros();
        let ranges: Vec<Range> = e.iter().map(|&(start, len, v, o)| Range { start, len, view: v.0 as u32, view_offset: o }).collect();
        let words = encode(&ranges);
        let us = started.elapsed().as_micros();
        eprintln!("address map: 20000 spans -> {} entries, {} words in {us} us (pieces {t1}, entries {})", ranges.len(), words.len(), t2 - t1);
        assert!(ranges.windows(2).all(|w| w[0].start + u64::from(w[0].len) <= w[1].start));
    }
}
