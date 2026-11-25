//! The page table (`XNW_PAGETABLE=1`): how translated programs turn
//! a console address into (raw view, byte offset) over the shared windows
//! title memory is read in place through ([`crate::shared`]). Layout and
//! rules (the translator's interface): a top table of 4,096 words (one per 4 GiB span: its
//! block, or `u32::MAX`), then blocks of 65,536 entries of 8 bytes (one
//! per 64 KiB page: the heap index of a raw SRV, the raw UAV at index + 1,
//! and the page's byte offset in that view; `u32::MAX` unmapped). Bound
//! as a raw root SRV at `t0` in [`xnw_shader::emit::PAGE_TABLE_SPACE`].
//!
//! Filled incrementally, never rebuilt: a page is resolved when the
//! runtime first meets its address (any in-place lookup of a range over
//! it, the title buffers when the table is made, a mapping event of up to
//! [`EVENT_PAGES`] pages), again when the title maps, unmaps or remaps it
//! ([`remapped`]), and cleared when its window's heap is released
//! ([`window_gone`]). Queued pages are resolved when a list binds the
//! table, before its submission. Pages that resolve to no window (regions
//! of 4 GiB or more, memory in no section view, images, stacks: "unknown")
//! read zero and are counted by why.
//!
//! One live table: submissions run to completion before the next is
//! recorded today, so an entry changed while recording is not read by a
//! submission still running. Asynchronous submission would need versioned
//! blocks (copy-on-write per block, freed when the last submission reading
//! the old version has run).
//!
//! The views: one SRV/UAV pair per (window buffer, 256 MiB step), each over
//! 512 MiB of the buffer (cut to its end), so a page and the 16 bytes past
//! it lie in one view where the buffer has them (spec section 3). Pairs
//! take two dynamic slots of the bound view heap each, for the run.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use super::descriptor::host::HostHeap;
use super::host::HostList;

/// Bytes per page.
const PAGE: u64 = 64 << 10;
/// Top-table words, and bytes before the first block.
const TOP: usize = 4096;
const TOP_BYTES: usize = TOP * 4;
/// Entries per block (pages per 4 GiB span), and words per block.
const BLOCK_PAGES: usize = 65_536;
const BLOCK_WORDS: usize = BLOCK_PAGES * 2;
/// Blocks the table holds (4 GiB spans with a mapped page): 32 MiB.
const BLOCKS: usize = 64;
const NONE: u32 = u32::MAX;
/// One view per this step of a window buffer, over [`VIEW_BYTES`].
const VIEW_STEP: u64 = 256 << 20;
const VIEW_BYTES: u64 = 512 << 20;
/// A mapping event queues its pages when it covers at most this many (a
/// larger one, a reservation of gigabytes, is left to first touches).
const EVENT_PAGES: u64 = 1024;

/// Why a page reads nothing: the [`crate::shared::Miss`] kinds, then
/// these.
const NO_BLOCK: usize = crate::shared::MISS_KINDS;
const NO_SLOTS: usize = crate::shared::MISS_KINDS + 1;
const KINDS: usize = crate::shared::MISS_KINDS + 2;

struct Table {
    /// The view heap whose slots the entries name.
    heap: Arc<HostHeap>,
    buffer: ID3D12Resource,
    words: *mut u32,
    gpu: u64,
    blocks: usize,
    /// View pairs: (window id, step) -> SRV slot.
    pairs: HashMap<(u32, u64), u32>,
    /// Pages resolved to a window, by window id, and the other way.
    by_window: HashMap<u32, BTreeSet<u64>>,
    window_of: HashMap<u64, u32>,
    /// Pages that read nothing, and why.
    unresolved: BTreeMap<u64, u8>,
}

// SAFETY: the mapping is written under the state's lock only; D3D12
// objects are free-threaded.
unsafe impl Send for Table {}

#[derive(Default)]
struct Queue {
    /// Pages met so far (resolved, unresolved or queued).
    known: BTreeSet<u64>,
    /// Pages to resolve at the next bind.
    pending: Vec<u64>,
}

static TABLE: Mutex<Option<Table>> = Mutex::new(None);
static QUEUE: Mutex<Option<Queue>> = Mutex::new(None);

/// Counters (totals since start): entries written, pages resolved and
/// left unresolved, binds, microseconds resolving, mapping events,
/// windows gone, pages pinned per kind of the last count.
static WRITTEN: AtomicU64 = AtomicU64::new(0);
static BINDS: AtomicU64 = AtomicU64::new(0);
static RESOLVE_US: AtomicU64 = AtomicU64::new(0);
static EVENTS: AtomicU64 = AtomicU64::new(0);
static GONE: AtomicU64 = AtomicU64::new(0);
/// Entries written since the last frame line.
static WRITTEN_WINDOW: AtomicU64 = AtomicU64::new(0);

fn queue<R>(f: impl FnOnce(&mut Queue) -> R) -> R {
    let mut q = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    f(q.get_or_insert_with(Queue::default))
}

/// The runtime met title memory `[address, address + len)`: its pages not
/// met before are resolved at the next bind.
pub(crate) fn touch(address: u64, len: u64) {
    if !crate::shared::page_table_on() || len == 0 {
        return;
    }
    let (first, last) = (address / PAGE, (address.saturating_add(len) - 1) / PAGE);
    queue(|q| {
        for p in first..=last {
            if q.known.insert(p) {
                q.pending.push(p);
            }
        }
    });
}

/// The title mapped, unmapped or remapped `[at, at + len)`: its pages met
/// before are resolved again (unmapped ones then read nothing); a small
/// new mapping is queued whole.
pub(crate) fn remapped(at: u64, len: u64) {
    if !crate::shared::page_table_on() || len == 0 {
        return;
    }
    EVENTS.fetch_add(1, Ordering::Relaxed);
    let (first, last) = (at / PAGE, (at.saturating_add(len) - 1) / PAGE);
    queue(|q| {
        if last - first < EVENT_PAGES {
            for p in first..=last {
                q.known.insert(p);
                q.pending.push(p);
            }
        } else {
            let met: Vec<u64> = q.known.range(first..=last).copied().collect();
            q.pending.extend(met);
        }
    });
}

/// Window `id`'s heap was released: its pages read nothing, its views are
/// made null, and the pages are resolved again when next met.
pub(crate) fn window_gone(id: u32) {
    if !crate::shared::page_table_on() {
        return;
    }
    let pages = {
        let mut t = TABLE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(t) = t.as_mut() else { return };
        GONE.fetch_add(1, Ordering::Relaxed);
        let pages = t.by_window.remove(&id).unwrap_or_default();
        for &p in &pages {
            t.set(p, NONE, 0);
            t.window_of.remove(&p);
        }
        let steps: Vec<(u32, u64)> = t.pairs.keys().filter(|k| k.0 == id).copied().collect();
        for k in steps {
            // The pair's slots become raw nulls of their kinds (an entry
            // a running submission still reads reads zero) and are never
            // handed out again; the entries above read unmapped already.
            if let Some(slot) = t.pairs.remove(&k) {
                t.heap.raw_null_pair(slot);
            }
        }
        pages
    };
    queue(|q| {
        for p in pages {
            q.known.remove(&p);
        }
    });
}

impl Table {
    fn new(heap: Arc<HostHeap>) -> Option<Table> {
        let bytes = (TOP_BYTES + BLOCKS * BLOCK_WORDS * 4) as u64;
        let (buffer, data) = super::host::host()?.shared_buffer(bytes)?;
        super::host::name(&buffer, || "page table".to_owned());
        let words = data.cast::<u32>();
        // SAFETY: the top table of the fresh mapping; a live buffer.
        let gpu = unsafe {
            std::slice::from_raw_parts_mut(words, TOP).fill(NONE);
            buffer.GetGPUVirtualAddress()
        };
        xnw_core::info!("page table: made ({} MiB, {BLOCKS} blocks of 4 GiB spans)", bytes >> 20);
        Some(Table { heap, buffer, words, gpu, blocks: 0, pairs: HashMap::new(), by_window: HashMap::new(), window_of: HashMap::new(), unresolved: BTreeMap::new() })
    }

    /// Page `p`'s entry; false if its span has no block and none is left.
    fn set(&mut self, p: u64, index: u32, offset: u32) -> bool {
        let span = (p >> 16) as usize;
        if span >= TOP {
            return false;
        }
        // SAFETY: word `span` of the top table.
        let mut b = unsafe { *self.words.add(span) };
        if b == NONE {
            if index == NONE {
                return true;
            }
            if self.blocks == BLOCKS {
                return false;
            }
            b = self.blocks as u32;
            self.blocks += 1;
            // SAFETY: the new block, inside the mapping.
            unsafe {
                let block = std::slice::from_raw_parts_mut(self.words.add(TOP + b as usize * BLOCK_WORDS), BLOCK_WORDS);
                for e in block.chunks_exact_mut(2) {
                    e[0] = NONE;
                    e[1] = 0;
                }
                *self.words.add(span) = b;
            }
        }
        let at = TOP + b as usize * BLOCK_WORDS + (p as usize & (BLOCK_PAGES - 1)) * 2;
        // SAFETY: entry `p` of block `b`, inside the mapping.
        unsafe {
            *self.words.add(at) = index;
            *self.words.add(at + 1) = offset;
        }
        WRITTEN.fetch_add(1, Ordering::Relaxed);
        WRITTEN_WINDOW.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// The SRV slot of the view pair over step `step` of `buffer`.
    fn pair(&mut self, buffer: &ID3D12Resource, id: u32, step: u64) -> Option<u32> {
        if let Some(&s) = self.pairs.get(&(id, step)) {
            return Some(s);
        }
        let slot = self.heap.dynamic(2)?;
        // SAFETY: a live buffer.
        let width = unsafe { buffer.GetDesc() }.Width;
        let first = step * VIEW_STEP;
        let bytes = (width.saturating_sub(first)).min(VIEW_BYTES) & !3;
        self.heap.raw_pair_at(slot, buffer, first, bytes);
        self.pairs.insert((id, step), slot);
        Some(slot)
    }

    /// Page `p` is no longer listed under its window.
    fn forget(&mut self, p: u64) {
        if let Some(id) = self.window_of.remove(&p)
            && let Some(pages) = self.by_window.get_mut(&id)
        {
            pages.remove(&p);
        }
    }

    /// Resolves page `p`: its entry written. False if it must wait (its
    /// window is being opened).
    fn resolve(&mut self, p: u64) -> bool {
        let address = p * PAGE;
        let why = match crate::shared::page(address, PAGE) {
            Ok(r) => {
                let id = crate::shared::window_id(&r.buffer);
                let step = r.offset / VIEW_STEP;
                match self.pair(&r.buffer, id, step) {
                    Some(slot) => {
                        if self.set(p, slot, (r.offset - step * VIEW_STEP) as u32) {
                            self.unresolved.remove(&p);
                            self.forget(p);
                            self.window_of.insert(p, id);
                            self.by_window.entry(id).or_default().insert(p);
                            return true;
                        }
                        NO_BLOCK
                    }
                    None => NO_SLOTS,
                }
            }
            Err(crate::shared::Miss::Opening) => return false,
            Err(m) => m as usize,
        };
        self.set(p, NONE, 0);
        self.forget(p);
        self.unresolved.insert(p, why as u8);
        true
    }
}

/// The page table's host GPU address for a list bound with view heap
/// `heap`, after resolving the pages queued since; the table is held by
/// the list. Made on first use (then the title buffers' pages are queued);
/// made again for another heap (tests: GPU-side descriptors have one).
pub(crate) fn bind(list: &HostList, heap: &Arc<HostHeap>) -> Option<u64> {
    let mut t = TABLE.lock().unwrap_or_else(|e| e.into_inner());
    if t.as_ref().is_none_or(|t| !Arc::ptr_eq(&t.heap, heap)) {
        *t = Some(Table::new(heap.clone())?);
        let mut all: Vec<u64> = Vec::new();
        for (start, len, _) in super::shadow::all() {
            if len > 0 {
                all.extend(start / PAGE..=(start + len - 1) / PAGE);
            }
        }
        // Every page met so far is resolved again, for the new table.
        queue(|q| {
            q.known.extend(all);
            q.pending = q.known.iter().copied().collect();
        });
    }
    let t = t.as_mut()?;
    let pending = queue(|q| std::mem::take(&mut q.pending));
    if !pending.is_empty() {
        let started = std::time::Instant::now();
        let mut later = Vec::new();
        for p in pending {
            if !t.resolve(p) {
                later.push(p);
            }
        }
        RESOLVE_US.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        if !later.is_empty() {
            queue(|q| q.pending.extend(later));
        }
    }
    BINDS.fetch_add(1, Ordering::Relaxed);
    list.keep(&t.buffer);
    Some(t.gpu)
}

/// The "frame page table" line's text: entries written (in all, since the
/// last line), mapping events, resolve time, binds, blocks, pairs, pages
/// unresolved by why, pinned MiB.
pub(crate) fn stats_text() -> String {
    let t = TABLE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(t) = t.as_ref() else {
        return "not made".into();
    };
    let mut kinds = [0u64; KINDS];
    for &k in t.unresolved.values() {
        kinds[k as usize] += 1;
    }
    let why: Vec<String> = kinds
        .iter()
        .enumerate()
        .filter(|&(_, &n)| n > 0)
        .map(|(k, n)| {
            let name = match k {
                NO_BLOCK => "no block left",
                NO_SLOTS => "no view slots",
                k => crate::shared::miss_name(k),
            };
            format!("{name} {n}")
        })
        .collect();
    let mapped: usize = t.by_window.values().map(BTreeSet::len).sum();
    format!(
        "{mapped} pages mapped over {} windows ({} view pairs, {} blocks); entries written {} ({} since the last line); mapping events {}; windows gone {}; resolving {} us in all over {} binds; pages unresolved: {}; pinned {} MiB ({} heaps refused for the cap)",
        t.by_window.len(),
        t.pairs.len(),
        t.blocks,
        WRITTEN.load(Ordering::Relaxed),
        WRITTEN_WINDOW.swap(0, Ordering::Relaxed),
        EVENTS.load(Ordering::Relaxed),
        GONE.load(Ordering::Relaxed),
        RESOLVE_US.load(Ordering::Relaxed),
        BINDS.load(Ordering::Relaxed),
        if why.is_empty() { "none".into() } else { why.join(", ") },
        crate::shared::pinned_mib(),
        crate::shared::PIN_REFUSED.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{Backing, Miss};
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D12::*;
    use xnw_core::win;

    /// Title regions the test's xmem stand-in knows: (base, bytes).
    static REGIONS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

    fn locate(address: usize, len: usize) -> Result<Backing, Miss> {
        for &(base, size) in REGIONS.lock().unwrap().iter() {
            if (base..base + size).contains(&address) {
                return if address + len <= base + size { Ok(Backing::Region { base, size }) } else { Err(Miss::PastRegion) };
            }
        }
        Err(Miss::Unknown)
    }

    fn window(_: usize, _: usize) -> Option<(usize, usize)> {
        None
    }

    /// The lookup of spec section 2, in HLSL: reads five console addresses
    /// through the table, writes each value + 100 to five more, and writes
    /// 7 at the fifth (unresolved) address.
    const PROBE: &str = r#"
#define RS "RootConstants(num32BitConstants=20, b0), SRV(t0, space=4), DescriptorTable(SRV(t0, space=1, numDescriptors=64)), DescriptorTable(UAV(u0, space=2, numDescriptors=64))"
cbuffer K : register(b0) { uint4 q[5]; };
ByteAddressBuffer pt : register(t0, space4);
ByteAddressBuffer srvs[64] : register(t0, space1);
RWByteAddressBuffer uavs[64] : register(u0, space2);
// Real branches, never `?:` (SM5 evaluates both sides of a select: a read
// through heap index 0xffffffff hung the card twice, TDR) and the index
// clamped to the table (a root SRV has no bounds either).
uint2 entry(uint lo, uint hi) {
    uint2 e = uint2(0xffffffff, 0);
    [branch] if (hi < 4096) {
        uint b = pt.Load(hi * 4);
        [branch] if (b < 64) e = pt.Load2(16384 + b * 524288 + (lo >> 16) * 8);
    }
    return e;
}
uint ld(uint lo, uint hi) {
    uint2 e = entry(lo, hi);
    uint v = 0;
    [branch] if (e.x < 62) v = srvs[NonUniformResourceIndex(e.x)].Load(e.y + (lo & 0xffff));
    return v;
}
void st(uint lo, uint hi, uint v) {
    uint2 e = entry(lo, hi);
    [branch] if (e.x < 62) uavs[NonUniformResourceIndex(e.x + 1)].Store(e.y + (lo & 0xffff), v);
}
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    for (uint i = 0; i < 5; i++) st(q[i].z, q[i].w, ld(q[i].x, q[i].y) + 100);
    st(q[4].x, q[4].y, 7);
}
"#;

    /// A compute program reads and writes title memory through the
    /// table across a page boundary (one region) and a window boundary (two
    /// regions back to back, two heaps); an address in no region reads zero
    /// and its write goes nowhere.
    #[test]
    fn a_program_reads_and_writes_title_memory_through_the_page_table() {
        let Some(host) = super::super::host::host() else { return };
        crate::shared::PAGE_TABLE_TEST.store(true, Ordering::Relaxed);
        crate::shared::use_memory(locate, window);
        if crate::shared::shared().is_none() {
            eprintln!("shared memory was made without the test's stand-in: skipped");
            return;
        }
        const MIB: usize = 1 << 20;
        // Two 1 MiB allocations back to back (each its own heap), and a
        // hole after them.
        // SAFETY: plain reservation and allocation of our own memory.
        let base = unsafe {
            let at = win::VirtualAlloc(std::ptr::null(), 8 * MIB, win::MEM_RESERVE, win::PAGE_READWRITE) as usize;
            assert_ne!(at, 0);
            win::VirtualFree(at as *mut _, 0, win::MEM_RELEASE);
            let a = win::VirtualAlloc(at as *const _, MIB, win::MEM_RESERVE | win::MEM_COMMIT, win::PAGE_READWRITE) as usize;
            let b = win::VirtualAlloc((at + MIB) as *const _, MIB, win::MEM_RESERVE | win::MEM_COMMIT, win::PAGE_READWRITE) as usize;
            assert_eq!((a, b), (at, at + MIB), "two regions back to back");
            at
        };
        REGIONS.lock().unwrap().extend([(base, MIB), (base + MIB, MIB)]);
        let ins = [base + 0xFFFC, base + 0x1_0000, base + MIB - 4, base + MIB, base + 4 * MIB];
        let outs: Vec<usize> = (0..5).map(|i| base + MIB + 0x2000 + 4 * i).collect();
        // SAFETY: inside the two regions.
        unsafe {
            for (k, &a) in ins[..4].iter().enumerate() {
                (a as *mut u32).write(11 + k as u32);
            }
        }
        touch(base as u64, 2 * MIB as u64);
        touch(ins[4] as u64, 4);
        let heap = Arc::new(HostHeap::new(false, 16, 0).unwrap());
        let list = host.new_list().unwrap();
        let va = bind(&list, &heap).expect("a page table");
        let mut consts = [0u32; 20];
        for i in 0..5 {
            consts[4 * i..4 * i + 4].copy_from_slice(&[ins[i] as u32, (ins[i] as u64 >> 32) as u32, outs[i] as u32, (outs[i] as u64 >> 32) as u32]);
        }
        let mut code: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;
        // SAFETY: the source and names live through the calls; an open list
        // with what the probe reads set.
        unsafe {
            let mut errors: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;
            let made = D3DCompile(PROBE.as_ptr().cast(), PROBE.len(), None, None, None, windows::core::s!("main"), windows::core::s!("cs_5_1"), 1 << 20, 0, &mut code, Some(&mut errors)); // D3DCOMPILE_ENABLE_UNBOUNDED_DESCRIPTOR_TABLES
            if let Some(e) = &errors {
                eprintln!("{}", String::from_utf8_lossy(std::slice::from_raw_parts(e.GetBufferPointer().cast::<u8>(), e.GetBufferSize())));
            }
            made.expect("compiles");
            let code = code.unwrap();
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: std::mem::transmute_copy(&root), CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() }, ..Default::default() })
                .unwrap();
            let l = &list.list;
            l.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
            l.SetComputeRootSignature(&root);
            l.SetPipelineState(&pso);
            l.SetComputeRoot32BitConstants(0, 20, consts.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, va);
            let start = heap.heap.GetGPUDescriptorHandleForHeapStart();
            l.SetComputeRootDescriptorTable(2, start);
            l.SetComputeRootDescriptorTable(3, start);
            l.Dispatch(1, 1, 1);
            list.keep(&root);
            list.keep(&pso);
        }
        list.close();
        host.execute(&[&list]);
        list.reset();
        eprintln!("page table test: {}", stats_text());
        // SAFETY: inside the second region; the GPU has finished.
        let got: Vec<u32> = outs.iter().map(|&o| unsafe { (o as *const u32).read_volatile() }).collect();
        assert_eq!(got, [111, 112, 113, 114, 100], "page end, next page, region end, next region, no region");
        assert!(stats_text().contains("unknown"), "the page in no region counted: {}", stats_text());
        crate::shared::PAGE_TABLE_TEST.store(false, Ordering::Relaxed);
    }

    /// Bookkeeping without the GPU: a remapping queues the pages met before;
    /// a large one only those, a small one all of its pages.
    #[test]
    fn mapping_events_queue_pages() {
        crate::shared::PAGE_TABLE_TEST.store(true, Ordering::Relaxed);
        let at = 0x7_1234_0000u64;
        queue(|q| {
            q.known.clear();
            q.pending.clear();
        });
        touch(at, 1);
        remapped(at, 64 << 30);
        remapped(at + (1 << 30), 2 * PAGE);
        let pending = queue(|q| std::mem::take(&mut q.pending));
        assert_eq!(pending, [at / PAGE, at / PAGE, (at >> 16) + (1 << 14), (at >> 16) + (1 << 14) + 1]);
        crate::shared::PAGE_TABLE_TEST.store(false, Ordering::Relaxed);
    }
}
