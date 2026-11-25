//! Title memory the GPU reads in place: the runtime layer ([`SharedMemory`],
//! first phase) and the probe that measured it.
//!
//! `xnw gpu-check --shared-memory`: can the GPU read the title's memory in
//! place? Opens D3D12 heaps over process memory
//! (`ID3D12Device3::OpenExistingHeapFromAddress`, `...FromFileMapping`),
//! places buffers on them and reads them as vertex, constant and raw
//! shader-resource buffers with no copy; then lifetime (decommit, unmap,
//! remap, read-only), cost, and textures.
//!
//! Flags as five-star-recomp found them (src/FiveStar.Loader/Shims/D3D12X/
//! XboxGameMemoryBuffers.cs, XboxPlacedResources.cs, a19995d, GPL-3.0):
//! buffers on an opened heap carry ALLOW_CROSS_ADAPTER, state COMMON.

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::time::Instant;

use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{Interface, PCSTR, Result};
use xnw_core::win;

use crate::Host;
use crate::host::feature;

const KIB: usize = 1 << 10;
const MIB: usize = 1 << 20;
const GIB: usize = 1 << 30;

// ---- the runtime layer (first phase) ---------------------------------------
//
// Title memory the GPU reads in place, as the probe below measured it.
// Physical pages: the runtime's own view of each page section, in windows of
// `WINDOW` bytes mapped and committed by xmem at first use, each opened as a
// heap (pinned for the process) with one buffer over it; a title address
// resolves to (window buffer, offset) through the section offset, so it
// stays valid while the title maps and unmaps its own views (probe step 6).
// Plain title VirtualAlloc regions (all committed, under 4 GiB): one heap
// over the whole allocation, released before the title frees or decommits
// any of it (a pinned range cannot be freed: probe step 3).
// Phase 1 only resolves and counts (`XNW_SHARED=1`); no draw binds these yet.

/// Bytes per window of a page section the runtime maps and opens as a heap.
/// Windows overlap: window `w` starts at `w * WINDOW_STRIDE` and spans
/// `WINDOW` (twice the stride), so a range of up to one stride lies wholly
/// in the window its start falls in. With one window per 64 MiB, 382
/// unordered access views a heavy frame straddled two and stayed on host
/// copies with snapshots and copy-back.
/// The pages are the same physical pages: no more memory, only a second
/// view of each.
pub const WINDOW: usize = 128 << 20;
/// Bytes between window starts.
pub const WINDOW_STRIDE: usize = 64 << 20;
/// The largest heap `OpenExistingHeapFromAddress` takes (probe 6e: 4095 MiB
/// opens, 4096 does not).
const HEAP_LIMIT: usize = 4 << 30;

/// What title memory is behind a range, as xmem knows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backing {
    /// Physical pages, contiguous in one section: the section (an opaque
    /// key) and the range's byte offset in it.
    Physical { section: usize, offset: u64 },
    /// A title VirtualAlloc allocation, all committed: its base and size.
    Region { base: usize, size: usize },
}

/// Why a range cannot be read in place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Miss {
    /// Neither physical pages nor a title region (or not all mapped).
    Unknown = 0,
    /// Physical pages, but not consecutive in one section.
    NotContiguous,
    /// Physical pages spanning two windows.
    CrossesWindow,
    /// A region not all committed.
    Uncommitted,
    /// A region of 4 GiB or more.
    TooLarge,
    /// The host refused the heap or buffer.
    Failed,
    /// Physical pages, then a page of the range not mapped (split from
    /// not contiguous).
    PartlyMapped,
    /// Inside a title region but running past its end (split from
    /// unknown).
    PastRegion,
    /// A page window whose heap is being opened off the frame; read
    /// through a host copy meanwhile.
    Opening,
}
const MISS_NAMES: [&str; 9] = ["unknown", "not contiguous", "crosses window", "uncommitted", "too large", "failed", "partly mapped", "past a region end", "window opening"];

/// xmem's side: what is behind `[address, address + len)`; and the
/// runtime's view of window `index` of a section, `(address, len)`, mapped
/// and committed on first use.
pub type Locate = fn(usize, usize) -> std::result::Result<Backing, Miss>;
pub type WindowView = fn(usize, usize) -> Option<(usize, usize)>;

/// A title range read in place: `len` bytes at `offset` of `buffer`. GPU
/// virtual address for vertex, index and constant buffer views: the
/// buffer's plus `offset`. The buffer stays in state COMMON; a user states
/// its transitions explicitly (VIctory: implicit ones hung NVIDIA).
pub struct SharedRange {
    pub buffer: ID3D12Resource,
    pub offset: u64,
    pub len: usize,
}

/// Heaps over title memory with one buffer each, opened at first use.
pub struct SharedMemory {
    device: ID3D12Device,
    locate: Locate,
    window: WindowView,
    /// (section, window index) or (region base, `usize::MAX`) -> heap, buffer, bytes.
    heaps: std::sync::Mutex<std::collections::HashMap<(usize, usize), (ID3D12Heap, ID3D12Resource, usize)>>,
    /// Page windows are opened on a thread of their own (the runtime's
    /// instance; tests open inline).
    background: bool,
}

// SAFETY: D3D12 objects are free-threaded; the map is behind a mutex.
unsafe impl Send for SharedMemory {}
// SAFETY: as above.
unsafe impl Sync for SharedMemory {}

impl SharedMemory {
    pub fn new(device: ID3D12Device, locate: Locate, window: WindowView) -> SharedMemory {
        SharedMemory { device, locate, window, heaps: Default::default(), background: false }
    }

    /// Where a range falls, without opening anything: the heap key and the
    /// offset in that heap's buffer.
    fn resolve(&self, address: usize, len: usize) -> std::result::Result<((usize, usize), u64), Miss> {
        match (self.locate)(address, len.max(1))? {
            Backing::Physical { section, offset } => {
                let w = offset as usize / WINDOW_STRIDE;
                let within = offset as usize - w * WINDOW_STRIDE;
                if within + len.max(1) > WINDOW {
                    return Err(Miss::CrossesWindow);
                }
                Ok(((section, w), within as u64))
            }
            Backing::Region { base, size } => {
                if size >= HEAP_LIMIT {
                    return Err(Miss::TooLarge);
                }
                Ok(((base, usize::MAX), (address - base) as u64))
            }
        }
    }

    /// The buffer over `[address, address + len)`, opening its heap on
    /// first use. A view must stay inside `[offset, offset + len)`: one past
    /// a run's end removed the device in five-star-recomp (their error #31),
    /// and a range that does not fit one buffer is refused here.
    pub fn buffer(&self, address: usize, len: usize) -> std::result::Result<SharedRange, Miss> {
        let (key, offset) = resolve_cached(address, len, || self.resolve(address, len))?;
        let fits = |bytes: usize| offset as usize + len <= bytes;
        {
            let heaps = self.heaps.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((_, buf, bytes)) = heaps.get(&key) {
                return if fits(*bytes) { Ok(SharedRange { buffer: buf.clone(), offset, len }) } else { Err(Miss::CrossesWindow) };
            }
        }
        // A page window's heap is opened off the frame. Opened inline,
        // 42 windows of 128 MiB took 2.56 s over the first 11 world frames,
        // inside bindless plans ('plan make: in-place lookups' 230 ms a
        // frame). The range resolves
        // (it lies in the window), so a host copy reads it meanwhile.
        // Only where a title buffer's host copy can read the range
        // meanwhile. A raw view of memory no buffer covers (the title's
        // command-list memory: culling constants) has nothing to fall
        // back to and is opened inline, as before.
        if key.1 != usize::MAX && self.background && crate::d3d12x::has_host_copy(address as u64, len as u64) {
            open_later(key);
            return Err(Miss::Opening);
        }
        let (at, bytes) = if key.1 == usize::MAX {
            let Ok(Backing::Region { base, size }) = (self.locate)(address, len.max(1)) else { return Err(Miss::Unknown) };
            (base, size)
        } else {
            (self.window)(key.0, key.1).ok_or(Miss::Failed)?
        };
        if !fits(bytes) {
            return Err(Miss::CrossesWindow);
        }
        // A page window opens outside the heaps' lock here too (region
        // heaps stay under it: `before_free` releases them under it).
        let gone = (key.1 != usize::MAX).then(|| SECTIONS_GONE.load(std::sync::atomic::Ordering::Acquire));
        let buffer = self.open(key, at, bytes, gone)?;
        Ok(SharedRange { buffer, offset, len })
    }

    /// Opens the heap of key over [at, at + bytes) (or finds it open).
    /// gone (background opens): the section-gone count when it was asked
    /// for; a section let go since may be this one, so the heap is not kept.
    fn open(&self, key: (usize, usize), at: usize, bytes: usize, gone: Option<u64>) -> std::result::Result<ID3D12Resource, Miss> {
        // Region heaps open under the lock, as efore_free releases them
        // under it; page windows (background) open outside it, so lookups
        // of open windows do not wait 60 ms.
        let mut held = Some(self.heaps.lock().unwrap_or_else(|e| e.into_inner()));
        if let Some((_, buf, _)) = held.as_ref().and_then(|h| h.get(&key)) {
            return Ok(buf.clone());
        }
        if gone.is_some() {
            held = None;
        }
        // With the page table every region a frame touches is opened;
        // the pinned total is capped.
        if let Some(cap) = pin_cap()
            && PINNED.load(std::sync::atomic::Ordering::Relaxed) + bytes as u64 > cap
        {
            PIN_REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                xnw_core::warn!(
                    "shared memory: pin cap reached: {} MiB pinned, a heap of {} MiB over {at:#x} not opened (XNW_PAGETABLE_PIN_MB={}); its pages read zero through the page table (counted as failed)",
                    PINNED.load(std::sync::atomic::Ordering::Relaxed) >> 20,
                    bytes >> 20,
                    cap >> 20
                )
            });
            return Err(Miss::Failed);
        }
        let started = std::time::Instant::now();
        let opened = (|| -> Result<(ID3D12Heap, ID3D12Resource)> {
            let d3: ID3D12Device3 = self.device.cast()?;
            // SAFETY: committed memory of ours or the title's, at an allocation base.
            let heap: ID3D12Heap = unsafe { d3.OpenExistingHeapFromAddress(at as *const c_void)? };
            let mut out: Option<ID3D12Resource> = None;
            // SAFETY: the description outlives the call.
            unsafe { self.device.CreatePlacedResource(&heap, 0, &buffer_desc(bytes as u64, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_COMMON, None, &mut out)? };
            Ok((heap, out.expect("resource on success")))
        })();
        let mut heaps = match held {
            Some(h) => h,
            None => self.heaps.lock().unwrap_or_else(|e| e.into_inner()),
        };
        match opened {
            Ok((heap, buf)) => {
                OPENED[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                OPENED[1].fetch_add(started.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
                if let Some(g) = gone
                    && SECTIONS_GONE.load(std::sync::atomic::Ordering::Acquire) != g
                {
                    return Err(Miss::Failed);
                }
                if let Some((_, b, _)) = heaps.get(&key) {
                    return Ok(b.clone());
                }
                HEAPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                PINNED.fetch_add(bytes as u64, std::sync::atomic::Ordering::Relaxed);
                xnw_core::info!("shared memory: heap over {at:#x} ({} MiB, {})", bytes >> 20, if key.1 == usize::MAX { "title region" } else { "page window" });
                heaps.insert(key, (heap, buf.clone(), bytes));
                Ok(buf)
            }
            Err(e) => {
                xnw_core::syscalls::creation_failed(xnw_core::syscalls::Made::Opened, &e, &format_args!("heap over {at:#x} ({} MiB)", bytes >> 20));
                Err(Miss::Failed)
            }
        }
    }
    /// Releases the heaps of a section (`window` None: all its windows) or
    /// of the region at `base` (`region`). Their memory is unpinned; the
    /// caller must know the GPU no longer reads them (phase 1: nothing does).
    pub fn release(&self, key: usize, region: bool) {
        let mut gone = Vec::new();
        let mut gone_ids = Vec::new();
        let mut heaps = self.heaps.lock().unwrap_or_else(|e| e.into_inner());
        heaps.retain(|&(k, w), (_, buf, bytes)| {
            let hit = k == key && (w == usize::MAX) == region;
            if hit {
                HEAPS.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                PINNED.fetch_sub(*bytes as u64, std::sync::atomic::Ordering::Relaxed);
                // The address map holds views of released heaps' buffers.
                RELEASES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // SAFETY: a plain query on a live buffer.
                gone.push((unsafe { buf.GetGPUVirtualAddress() }, *bytes as u64));
                gone_ids.push(window_id(buf));
            }
            !hit
        });
        drop(heaps);
        // Page-table entries over these windows read nothing from here on.
        for id in gone_ids {
            crate::d3d12x::page_table_window_gone(id);
        }
        // Slots made in place there are made null before the
        // next submission; the memory is the title's to free from here on.
        for (gpu, bytes) in gone {
            crate::d3d12x::shared_window_gone(gpu, bytes);
        }
    }

    /// Releases region heaps overlapping `[address, address + len)` (`len`
    /// 0: the region at `address`): before the title frees or decommits.
    pub fn before_free(&self, address: usize, len: usize) {
        let bases: Vec<usize> = {
            let heaps = self.heaps.lock().unwrap_or_else(|e| e.into_inner());
            heaps
                .iter()
                .filter(|&(&(base, w), (_, _, size))| w == usize::MAX && address < base + size && (len == 0 && address == base || address + len > base))
                .map(|(&(base, _), _)| base)
                .collect()
        };
        for base in bases {
            self.release(base, true);
        }
    }
}

/// Resolved ranges (address, bytes) -> (heap key, offset) or why not, until
/// title memory is remapped (xmem's resolve walks the views of a range
/// page by page under its lock; 1,600 buffer slots made a frame cost 116 ms
/// in it, 'plan make: in-place lookups').
type Resolved = std::result::Result<((usize, usize), u64), Miss>;

/// The resolve cache, and its index by 2 MiB region: a remapping forgets
/// only the ranges over its regions. Forgetting all of it at every
/// remapping (the title remaps hundreds of times a frame) left it nearly
/// always empty ('plan make: in-place lookups' 49 ms a frame in a test run).
struct ResolveCache {
    ranges: std::collections::BTreeMap<(usize, usize), Resolved>,
    by_region: std::collections::BTreeMap<usize, Vec<(usize, usize)>>,
}
const REGION: usize = 2 << 20;
static RESOLVED: std::sync::Mutex<ResolveCache> =
    std::sync::Mutex::new(ResolveCache { ranges: std::collections::BTreeMap::new(), by_region: std::collections::BTreeMap::new() });
static REMAPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Resolves answered from the cache, and asked of xmem (statistics).
pub static RESOLVE_COUNTS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Title memory `[at, at + len)` was mapped, remapped or unmapped: the
/// resolved ranges over its regions go.
pub fn remapped(at: usize, len: usize) {
    REMAPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    crate::d3d12x::page_table_remapped(at as u64, len as u64);
    let end = at.saturating_add(len.max(1));
    let mut c = RESOLVED.lock().unwrap_or_else(|e| e.into_inner());
    let regions: Vec<usize> = c.by_region.range(at / REGION..end.div_ceil(REGION)).map(|(&r, _)| r).collect();
    for r in regions {
        if let Some(keys) = c.by_region.remove(&r) {
            for k in keys {
                c.ranges.remove(&k);
            }
        }
    }
}

/// How often a heap over title memory was released (the address map's
/// in-place entries are made again then). Not at every remapping: the
/// title remaps hundreds of times a frame, which would make the map
/// again at nearly every bind. shortcut: a remapping that moves a window
/// under a listed buffer is seen only at the map's next rebuild; the page
/// table follows remappings.
pub fn remaps() -> u64 {
    RELEASES.load(std::sync::atomic::Ordering::Relaxed)
}
static RELEASES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn resolve_cached(address: usize, len: usize, resolve: impl FnOnce() -> Resolved) -> Resolved {
    use std::sync::atomic::Ordering::Relaxed;
    let epoch = REMAPS.load(Relaxed);
    if let Some(&r) = RESOLVED.lock().unwrap_or_else(|e| e.into_inner()).ranges.get(&(address, len)) {
        RESOLVE_COUNTS[0].fetch_add(1, Relaxed);
        return r;
    }
    RESOLVE_COUNTS[1].fetch_add(1, Relaxed);
    let r = resolve();
    let mut c = RESOLVED.lock().unwrap_or_else(|e| e.into_inner());
    // A remapping meanwhile may have made the answer old: not kept.
    if REMAPS.load(Relaxed) == epoch {
        if c.ranges.len() > 1 << 16 {
            c.ranges.clear();
            c.by_region.clear();
        }
        c.ranges.insert((address, len), r);
        for region in address / REGION..(address + len.max(1)).div_ceil(REGION) {
            c.by_region.entry(region).or_default().push((address, len));
        }
    }
    r
}

static HEAPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PINNED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static MEMORY: std::sync::OnceLock<(Locate, WindowView)> = std::sync::OnceLock::new();

/// xmem hands over its lookups (at device creation, as `use_page_mapper`).
pub fn use_memory(locate: Locate, window: WindowView) {
    let _ = MEMORY.set((locate, window));
}

/// The title region (base, bytes) holding `address`, if xmem knows one
/// Buffer roles key buffers by region size and offset, which repeat
/// run to run where region bases do not).
pub fn region_of(address: usize) -> Option<(usize, usize)> {
    let &(locate, _) = MEMORY.get()?;
    match locate(address, 1) {
        Ok(Backing::Region { base, size }) => Some((base, size)),
        _ => None,
    }
}

/// Whether xmem knows `[address, address + len)` as mapped title
/// memory (physical pages, contiguous in one section, or a committed title
/// region): then a plain copy reads it without faulting. False without
/// xmem's lookups.
pub fn mapped(address: usize, len: usize) -> bool {
    MEMORY.get().is_some_and(|&(locate, _)| locate(address, len.max(1)).is_ok())
}

/// The process's shared memory over the one host device, once xmem's
/// lookups are in.
pub fn shared() -> Option<&'static SharedMemory> {
    // Not kept as missing when asked before xmem's lookups are in (a
    // test asking first made the page-table test's stand-in unusable).
    static S: std::sync::OnceLock<SharedMemory> = std::sync::OnceLock::new();
    if let Some(s) = S.get() {
        return Some(s);
    }
    let &(locate, window) = MEMORY.get()?;
    let device = crate::d3d12x::host_device()?.device.clone();
    Some(S.get_or_init(|| {
        let mut s = SharedMemory::new(device, locate, window);
        // `XNW_SHARED_OPEN_INLINE=1`: page windows opened where first
        // asked for, as before.
        s.background = std::env::var("XNW_SHARED_OPEN_INLINE").map_or(true, |v| v.trim() != "1");
        s
    }))
}

/// Before the title frees or decommits `[address, address + len)` of its
/// own memory (xmem's `VirtualFree`): heaps over it go first. Cheap when
/// none is open.
pub fn before_free(address: usize, len: usize) {
    if HEAPS.load(std::sync::atomic::Ordering::Relaxed) > 0
        && let Some(s) = shared()
    {
        s.before_free(address, len);
    }
}

/// Sections let go so far: a window opened in the background across
/// one is not kept (its section may be the one gone).
static SECTIONS_GONE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Page windows waiting for their heap, and the thread that opens them.
static OPENING: std::sync::Mutex<std::collections::BTreeSet<(usize, usize)>> = std::sync::Mutex::new(std::collections::BTreeSet::new());
static OPENER: std::sync::OnceLock<std::sync::Mutex<std::sync::mpsc::Sender<((usize, usize), u64)>>> = std::sync::OnceLock::new();

/// Asks the opener thread for window key's heap (once while waiting).
fn open_later(key: (usize, usize)) {
    if !OPENING.lock().unwrap_or_else(|e| e.into_inner()).insert(key) {
        return;
    }
    let gone = SECTIONS_GONE.load(std::sync::atomic::Ordering::Acquire);
    let sender = OPENER.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<((usize, usize), u64)>();
        let _ = std::thread::Builder::new().name("xnw shared heaps".into()).spawn(move || {
            for (key, gone) in rx {
                if let Some(s) = shared()
                    && let Some((at, bytes)) = (s.window)(key.0, key.1)
                {
                    if s.open(key, at, bytes, Some(gone)).is_ok() {
                        // Bindless slots left null meanwhile are looked
                        // at again; those on host copies move in place when
                        // the title next writes them.
                        crate::d3d12x::bindless_viewed_changed();
                    }
                }
                OPENING.lock().unwrap_or_else(|e| e.into_inner()).remove(&key);
            }
        });
        std::sync::Mutex::new(tx)
    });
    let _ = sender.lock().unwrap_or_else(|e| e.into_inner()).send((key, gone));
}

/// A page section is gone (all its pages freed): its window heaps go.
pub fn section_gone(section: usize) {
    SECTIONS_GONE.fetch_add(1, std::sync::atomic::Ordering::Release);
    if HEAPS.load(std::sync::atomic::Ordering::Relaxed) > 0
        && let Some(s) = shared()
    {
        s.release(section, false);
    }
}

// The count of `XNW_SHARED=1`: of the title-memory copies made today, how
// many (and bytes) would resolve to a shared buffer, by kind, and why the
// others would not. Resolving opens nothing.

/// Copy kinds, as `d3d12x::host::COPY_KINDS` counts them (fill, merged
/// fill, buffer copy, refresh, bindless read, bounce, skipped bounce).
pub const KIND_NAMES: [&str; 7] = ["fills", "merged", "buffer copies", "refreshes", "bindless reads", "bounces", "skipped"];
static WOULD: [[std::sync::atomic::AtomicU64; 4]; 7] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 4] }; 7];
static MISSES: [std::sync::atomic::AtomicU64; 9] = [const { std::sync::atomic::AtomicU64::new(0) }; 9];

fn mode() -> u8 {
    static MODE: std::sync::LazyLock<u8> = std::sync::LazyLock::new(|| std::env::var("XNW_SHARED").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
    *MODE
}

/// Whether `XNW_SHARED` is 1 (count only) or 2 (read in place, and count
/// what is still copied).
pub fn counting() -> bool {
    mode() >= 1
}

/// Whether `XNW_SHARED=2`: buffer views, vertex, index and root views of
/// title memory that resolve read it in place.
pub fn in_place() -> bool {
    mode() >= 2
}

/// `XNW_PAGETABLE=1`: translated programs that ask read title
/// memory through the page table over the shared windows.
pub fn page_table_on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_PAGETABLE").is_ok_and(|v| v == "1"));
    *ON || PAGE_TABLE_TEST.load(std::sync::atomic::Ordering::Relaxed)
}

/// Tests turn the page table on without the environment.
pub(crate) static PAGE_TABLE_TEST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The most bytes of title memory heaps may pin with the page table on
/// (`XNW_PAGETABLE_PIN_MB`, default 24,576 MiB); none without it.
fn pin_cap() -> Option<u64> {
    static CAP: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| std::env::var("XNW_PAGETABLE_PIN_MB").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(24_576) << 20);
    page_table_on().then_some(*CAP)
}

/// Heaps not opened for the pin cap.
pub static PIN_REFUSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Bytes of title memory pinned by open heaps, in MiB.
pub fn pinned_mib() -> u64 {
    PINNED.load(std::sync::atomic::Ordering::Relaxed) >> 20
}

/// The page table's resolve of one title page: the shared buffer over
/// `[address, address + len)` (opening its heap as views do), not counted
/// as a view, without the one-copy rule (pages the GPU wrote through a host
/// copy are a host-copy matter the page table does not see; roles off).
pub fn page(address: u64, len: u64) -> std::result::Result<SharedRange, Miss> {
    shared().ok_or(Miss::Unknown)?.buffer(address as usize, len as usize)
}

/// [`Miss`] names, by number.
pub fn miss_name(m: usize) -> &'static str {
    MISS_NAMES.get(m).copied().unwrap_or("?")
}
/// Number of [`Miss`] kinds.
pub const MISS_KINDS: usize = MISS_NAMES.len();

/// Heaps opened (count, microseconds), for the frame statistics.
static OPENED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Kinds of views made in place, for the frame statistics.
pub const VIEW_NAMES: [&str; 9] = ["buffer srv", "buffer uav", "cbv", "table cbv", "vertex", "index", "root", "indirect args", "buffer copy"];
pub const VIEW_SRV: usize = 0;
pub const VIEW_UAV: usize = 1;
pub const VIEW_CBV: usize = 2;
pub const VIEW_TABLE: usize = 3;
pub const VIEW_VERTEX: usize = 4;
pub const VIEW_INDEX: usize = 5;
pub const VIEW_ROOT: usize = 6;
pub const VIEW_ARGS: usize = 7;
pub const VIEW_COPY: usize = 8;
/// Per kind: in place, left on host copies.
static VIEWS: [[std::sync::atomic::AtomicU64; 2]; 9] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 2] }; 9];
/// Why a view of each kind was left on a host copy: the [`Miss`] reasons,
/// then the two below.
pub const WHY_MISALIGNED: usize = 9;
pub const WHY_HOST: usize = 10;
const WHY_NAMES: [&str; 11] = ["unknown", "not contiguous", "crosses window", "uncommitted", "too large", "failed", "partly mapped", "past a region end", "window opening", "misaligned", "host copy needed"];
static WHY: [[std::sync::atomic::AtomicU64; 11]; 9] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 11] }; 9];
/// Bytes of the views that resolved to no shared buffer, per view kind.
static MISS_BYTES: [std::sync::atomic::AtomicU64; 9] = [const { std::sync::atomic::AtomicU64::new(0) }; 9];

/// Notes a view of kind `kind` left on a host copy for `why` (a `WHY_*`).
pub fn left_because(kind: usize, why: usize) {
    if in_place() {
        VIEWS[kind][1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        WHY[kind][why].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Notes a view of kind `kind` that resolved but lands off the binding's
/// alignment (counted in place by [`range`] already): left on a copy.
pub fn misaligned(kind: usize) {
    fell_back(kind);
    WHY[kind][WHY_MISALIGNED].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// A small stable number for the shared buffer `buffer` (a window or
/// region heap's), given on first ask: GPU-side descriptors put a raw view
/// of each in a fixed slot. Numbers are not reused.
pub fn window_id(buffer: &ID3D12Resource) -> u32 {
    static IDS: std::sync::Mutex<std::collections::BTreeMap<usize, u32>> = std::sync::Mutex::new(std::collections::BTreeMap::new());
    let mut ids = IDS.lock().unwrap_or_else(|e| e.into_inner());
    let n = ids.len() as u32;
    *ids.entry(buffer.as_raw() as usize).or_insert(n)
}

/// With `XNW_SHARED=2`, the shared buffer over `[address, address + len)`,
/// not counted (for asking again what a view was made on).
pub fn lookup(address: u64, len: u64) -> Option<SharedRange> {
    crate::d3d12x::page_table_touch(address, len);
    if !in_place() || len == 0 || one_copy(address, len) {
        return None;
    }
    shared()?.buffer(address as usize, len as usize).ok()
}

/// A test run's GPU hang: with GPU writes going back to title memory only
/// when the CPU touches them (`XNW_ROLES=on`), a range the GPU wrote through
/// a host copy and then read through an in-place view read stale title
/// memory (a GPU read never faults). So a range over a host copy the GPU
/// writes is read through that copy too (one copy, no copy-back between
/// two views; sticky per buffer), and an in-place read of a range with GPU
/// writes still owed gets them first.
fn one_copy(address: u64, len: u64) -> bool {
    if crate::d3d12x::gpu_written_copy(address, len) {
        ONE_COPY.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return true;
    }
    crate::d3d12x::kernel_will_read(address as usize, len as usize);
    false
}

/// In-place views refused because their range is on a GPU-written host
/// copy, for the statistics.
pub static ONE_COPY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// With `XNW_SHARED=2`, the shared buffer over `[address, address + len)`
/// (none for an empty range); counted under view kind `kind`.
pub fn range(kind: usize, address: u64, len: u64) -> Option<SharedRange> {
    crate::d3d12x::page_table_touch(address, len);
    if !in_place() || len == 0 {
        return None;
    }
    if one_copy(address, len) {
        VIEWS[kind][1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        WHY[kind][WHY_HOST].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    let r = shared()?.buffer(address as usize, len as usize);
    VIEWS[kind][usize::from(r.is_err())].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if let Err(m) = r {
        WHY[kind][m as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        MISS_BYTES[kind].fetch_add(len, std::sync::atomic::Ordering::Relaxed);
    }
    r.ok()
}

/// Counts a view of kind `kind` made in place or not (for views resolved
/// with [`lookup`]).
pub fn count_view(kind: usize, in_place: bool) {
    if self::in_place() {
        VIEWS[kind][usize::from(!in_place)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Notes a view of kind `kind` left on a host copy though its range resolved
/// (its first element off the shared buffer's elements, say).
pub fn fell_back(kind: usize) {
    VIEWS[kind][0].fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    VIEWS[kind][1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

// ---- Could every copied page be read in place? ----------------
//
// Swing B reads all buffer memory in place
// through a 64 KiB page table. Before anyone builds it, every range copied
// or refreshed today is split into 64 KiB pages and each page classified:
// in an open shared window; in memory a window could be opened for; in a
// title region of 4 GiB or more; in memory xmem has no section or region
// for (image, stack, heap); unmapped. Counted in pages and bytes, per copy
// kind, on sampled frames (every 4th presented frame: a page asks xmem
// through the resolve cache).

/// Page classes.
pub const ZC_OPEN: usize = 0;
pub const ZC_OPENABLE: usize = 1;
pub const ZC_LARGE: usize = 2;
pub const ZC_UNKNOWN: usize = 3;
pub const ZC_UNMAPPED: usize = 4;
const ZC_NAMES: [&str; 5] = ["open window", "openable", "region of 4 GiB or more", "no section or region", "unmapped"];
/// Copy kinds as [KIND_NAMES], then root views and indirect arguments
/// through host copies (also counted in refreshes).
pub const ZC_ROOT: usize = 7;
pub const ZC_ARGS: usize = 8;
const ZC_KINDS: usize = 9;
const ZC_PAGE: usize = 64 << 10;
/// Per kind and class: pages, bytes; per kind: bytes of ranges all of whose
/// pages are open or openable, and all bytes.
static ZC_PAGES: [[std::sync::atomic::AtomicU64; 5]; ZC_KINDS] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 5] }; ZC_KINDS];
static ZC_BYTES: [[std::sync::atomic::AtomicU64; 5]; ZC_KINDS] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 5] }; ZC_KINDS];
static ZC_WHOLE: [[std::sync::atomic::AtomicU64; 2]; ZC_KINDS] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 2] }; ZC_KINDS];
/// Windows and regions a sampled frame touched that are not open: key ->
/// bytes it would pin; and the running maximum of pinned + those.
static ZC_WOULD_OPEN: std::sync::Mutex<std::collections::BTreeMap<(usize, usize), usize>> = std::sync::Mutex::new(std::collections::BTreeMap::new());
static ZC_PIN_PEAK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Table root arguments and root views by address: memory in an open or
/// openable window, not; table records read by draws: ours ('vVIE'), raw
/// console, zero.
pub static ZC_ROOTS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
pub static ZC_RECORDS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Whether this frame is sampled.
fn zc_sampled() -> bool {
    crate::d3d12x::frames_presented().is_multiple_of(4)
}

/// The class of the 64 KiB page at `page` (page-aligned), and the key and
/// bytes a window for it would pin when it is openable.
fn zc_page(s: &SharedMemory, page: usize) -> (usize, Option<((usize, usize), usize)>) {
    let r = resolve_cached(page, ZC_PAGE, || s.resolve(page, ZC_PAGE));
    let open = r.is_ok_and(|(key, _)| s.heaps.lock().unwrap_or_else(|e| e.into_inner()).contains_key(&key));
    let class = zc_class(r, open);
    let would = match r {
        Ok((key, _)) if class == ZC_OPENABLE => Some((key, if key.1 == usize::MAX { region_of(page).map_or(0, |(_, size)| size) } else { WINDOW })),
        _ => None,
    };
    (class, would)
}

/// The class of a page that resolved as `r`, its window `open` or not.
fn zc_class(r: Resolved, open: bool) -> usize {
    match r {
        Ok(_) if open => ZC_OPEN,
        Ok(_) => ZC_OPENABLE,
        Err(Miss::TooLarge) => ZC_LARGE,
        Err(Miss::PartlyMapped | Miss::Uncommitted) => ZC_UNMAPPED,
        Err(Miss::Opening) => ZC_OPEN,
        Err(_) => ZC_UNKNOWN,
    }
}

/// The 64 KiB pages of `[address, address + len)` and the bytes of the
/// range in each.
fn zc_pages(address: usize, len: usize) -> Vec<(usize, usize)> {
    let end = address.saturating_add(len);
    let mut out = Vec::new();
    if len == 0 {
        return out;
    }
    let mut page = address & !(ZC_PAGE - 1);
    while page < end {
        out.push((page, (page + ZC_PAGE).min(end) - page.max(address)));
        page += ZC_PAGE;
    }
    out
}

/// Classifies `[address, address + len)` page by page under copy kind
/// `kind` (sampled frames only).
pub fn zero_copy_note(kind: usize, address: usize, len: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    if len == 0 || kind >= ZC_KINDS || !zc_sampled() {
        return;
    }
    let Some(s) = shared() else { return };
    let mut whole = true;
    for (page, bytes) in zc_pages(address, len) {
        let (class, open) = zc_page(s, page);
        ZC_PAGES[kind][class].fetch_add(1, Relaxed);
        ZC_BYTES[kind][class].fetch_add(bytes as u64, Relaxed);
        whole &= class <= ZC_OPENABLE;
        let by = alloc_class(page);
        ALLOC_PAGES[by].fetch_add(1, Relaxed);
        ALLOC_BYTES[kind.min(ZC_KINDS - 1)][by].fetch_add(bytes as u64, Relaxed);
        if by == AL_ELSEWHERE {
            let region = region_of(page).map_or(page & !(WINDOW - 1), |(b, _)| b);
            *ALLOC_ELSEWHERE.lock().unwrap_or_else(|e| e.into_inner()).entry(region).or_default() += bytes as u64;
        }
        if let Some((key, b)) = open {
            ZC_WOULD_OPEN.lock().unwrap_or_else(|e| e.into_inner()).insert(key, b);
        }
    }
    ZC_WHOLE[kind][0].fetch_add(if whole { len as u64 } else { 0 }, Relaxed);
    ZC_WHOLE[kind][1].fetch_add(len as u64, Relaxed);
}

// ---- Copied bytes by the title's allocator ---------------------------
//
// All per-frame CPU-to-GPU
// traffic comes from an 8 MiB page pool (32 pages of 256 KiB, page records
// at `xmmword_1475DF1B8`), six per-frame arenas (`qword_1477AE310[0..5]`)
// and placed resources. Their addresses are found at run time from those
// globals (layouts unknown, so by shape: the pool is where the records'
// 256 KiB-aligned pointers fall within 8 MiB; an arena is the xmem region a
// slot's pointer lies in) and logged once, with the raw words, so a run
// confirms them.

pub const AL_POOL: usize = 0;
pub const AL_ARENA: usize = 1;
pub const AL_BUFFER: usize = 2;
pub const AL_ELSEWHERE: usize = 3;
const AL_NAMES: [&str; 4] = ["page pool", "frame arenas", "title buffer objects", "elsewhere"];
static ALLOC_PAGES: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
static ALLOC_BYTES: [[std::sync::atomic::AtomicU64; 4]; ZC_KINDS] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 4] }; ZC_KINDS];
/// Bytes copied "elsewhere" by xmem region (or 128 MiB span), for the
/// biggest few on the line.
static ALLOC_ELSEWHERE: std::sync::Mutex<std::collections::BTreeMap<usize, u64>> = std::sync::Mutex::new(std::collections::BTreeMap::new());

/// The pool range and arena regions, once found.
struct Allocators {
    pool: Option<(usize, usize)>,
    arenas: Vec<(usize, usize)>,
}
static ALLOCATORS: std::sync::Mutex<Option<Allocators>> = std::sync::Mutex::new(None);

/// Reads the title's allocator globals (from presented frame 64 on, when
/// they are set up); None before that or without the title.
fn allocators() -> Option<std::sync::MutexGuard<'static, Option<Allocators>>> {
    let mut a = ALLOCATORS.lock().unwrap_or_else(|e| e.into_inner());
    if a.is_some() {
        return Some(a);
    }
    let base = crate::d3d12x::title_image_base();
    if base == 0 || crate::d3d12x::frames_presented() < 64 {
        return None;
    }
    let read = |rva: usize, n: usize| -> Vec<u64> {
        let mut v = vec![0u64; n];
        crate::d3d12x::read_title(base + rva, v.as_mut_ptr().cast(), n * 8);
        v
    };
    // 32 records of 88 bytes.
    let records = read(0x75D_F1B8, 32 * 88 / 8);
    let pages: Vec<usize> = records.iter().map(|&v| v as usize).filter(|&v| v > 0x1_0000 && v.is_multiple_of(256 << 10)).collect();
    let pool = pages.iter().copied().min().and_then(|lo| {
        let inside = pages.iter().filter(|&&v| v < lo + (8 << 20)).collect::<std::collections::BTreeSet<_>>().len();
        (inside >= 16).then_some((lo, 8 << 20))
    });
    let slots = read(0x77A_E310, 6);
    let mut arenas: Vec<(usize, usize)> = slots.iter().filter_map(|&v| region_of(v as usize)).collect();
    // A slot may point at an arena object rather than its memory: its first
    // words then hold the memory's address.
    for &v in &slots {
        if region_of(v as usize).is_none() && v > 0x1_0000 {
            let inner = {
                let mut w = [0u64; 4];
                crate::d3d12x::read_title(v as usize, w.as_mut_ptr().cast(), 32);
                w
            };
            arenas.extend(inner.iter().filter_map(|&x| region_of(x as usize)));
        }
    }
    arenas.sort_unstable();
    arenas.dedup();
    // Not found yet: asked again on later frames (until frame 2048).
    if pool.is_none() && crate::d3d12x::frames_presented() < 2048 {
        return None;
    }
    if let Some((p, _)) = pool {
        arenas.retain(|&(b, s)| !(b <= p && p < b + s && s <= 8 << 20));
    }
    xnw_core::info!(
        "allocators: page pool {}; frame arenas {:?}; arena slots {:x?}; first page records {:x?}",
        pool.map_or("not found".into(), |(b, l)| format!("{b:#x}+{l:#x}")),
        arenas.iter().map(|&(b, s)| format!("{b:#x}+{s:#x}")).collect::<Vec<_>>(),
        slots,
        &records[..records.len().min(22)]
    );
    *a = Some(Allocators { pool, arenas });
    Some(a)
}

/// Which allocator `address` belongs to.
fn alloc_class(address: usize) -> usize {
    if let Some(a) = allocators()
        && let Some(a) = a.as_ref()
    {
        if a.pool.is_some_and(|(b, l)| (b..b + l).contains(&address)) {
            return AL_POOL;
        }
        if a.arenas.iter().any(|&(b, l)| (b..b + l).contains(&address)) {
            return AL_ARENA;
        }
    }
    if crate::d3d12x::has_host_copy_object(address as u64) { AL_BUFFER } else { AL_ELSEWHERE }
}

/// `XNW_SYNC=pools` (off by default): ranges wholly in the page pool or
/// the frame arenas are copied whole at each refresh, plainly: no write
/// watch, no guard, no page-by-page check.
pub fn sync_pools() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_SYNC").is_ok_and(|v| v.trim() == "pools"));
    *ON
}

/// Whether `[address, address + len)` lies wholly in the page pool or one
/// frame arena (then mapped title memory a plain copy may read).
pub fn in_pools(address: usize, len: usize) -> bool {
    let end = address.saturating_add(len);
    allocators().and_then(|a| {
        let a = a.as_ref()?;
        Some(a.pool.is_some_and(|(b, l)| b <= address && end <= b + l) || a.arenas.iter().any(|&(b, l)| b <= address && end <= b + l))
    }) == Some(true)
}

/// Pool and arena copies made plainly under `XNW_SYNC=pools` (count, bytes).
pub static SYNC_POOLS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The allocator part of the zero-copy line, per sampled frame, zeroing.
fn alloc_text(sampled: u64) -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let kinds: Vec<String> = (0..ZC_KINDS)
        .filter_map(|k| {
            let b: Vec<u64> = ALLOC_BYTES[k].iter().map(|c| c.swap(0, Relaxed) / sampled).collect();
            (b.iter().sum::<u64>() > 0).then(|| {
                let name = if k < KIND_NAMES.len() { KIND_NAMES[k] } else if k == ZC_ROOT { "root views" } else { "indirect arguments" };
                format!("{name} [{}]", AL_NAMES.iter().zip(&b).map(|(c, b)| format!("{c} {} KiB", b >> 10)).collect::<Vec<_>>().join(", "))
            })
        })
        .collect();
    let pages: Vec<u64> = ALLOC_PAGES.iter().map(|c| c.swap(0, Relaxed) / sampled).collect();
    let mut top: Vec<(usize, u64)> = std::mem::take(&mut *ALLOC_ELSEWHERE.lock().unwrap_or_else(|e| e.into_inner())).into_iter().collect();
    top.sort_unstable_by_key(|&(_, b)| std::cmp::Reverse(b));
    let top: Vec<String> = top.iter().take(6).map(|&(r, b)| format!("{r:#x} {} KiB", b / sampled >> 10)).collect();
    format!(
        "by allocator: {}; 64 KiB pages {} pool, {} arenas, {} buffer objects, {} elsewhere; biggest elsewhere [{}]; XNW_SYNC=pools plain copies {} ({} KiB)",
        if kinds.is_empty() { "nothing".into() } else { kinds.join("; ") },
        pages[0], pages[1], pages[2], pages[3],
        top.join(", "),
        SYNC_POOLS[0].swap(0, Relaxed) / sampled.max(1),
        SYNC_POOLS[1].swap(0, Relaxed) / sampled.max(1) >> 10
    )
}

/// Classifies the memory at `address` of a table root argument or a root
/// view: in an open or openable window, or not (sampled frames only).
pub fn zero_copy_root(table: bool, address: usize) {
    if !zc_sampled() {
        return;
    }
    let Some(s) = shared() else { return };
    let (class, _) = zc_page(s, address & !(ZC_PAGE - 1));
    let i = usize::from(!table) * 2 + usize::from(class > ZC_OPENABLE);
    ZC_ROOTS[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// The "frame zero-copy" line: per sampled frame over `n` frames, zeroing.
pub fn zero_copy_text(n: u64) -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let sampled = n.div_ceil(4).max(1);
    let (mut ok, mut all) = (0u64, 0u64);
    let kinds: Vec<String> = (0..ZC_KINDS)
        .filter_map(|k| {
            let pages: Vec<u64> = ZC_PAGES[k].iter().map(|c| c.swap(0, Relaxed) / sampled).collect();
            let bytes: Vec<u64> = ZC_BYTES[k].iter().map(|c| c.swap(0, Relaxed) / sampled).collect();
            let (w, a) = (ZC_WHOLE[k][0].swap(0, Relaxed) / sampled, ZC_WHOLE[k][1].swap(0, Relaxed) / sampled);
            if k < ZC_ROOT {
                (ok, all) = (ok + w, all + a);
            }
            (pages.iter().sum::<u64>() > 0).then(|| {
                let name = if k < KIND_NAMES.len() { KIND_NAMES[k] } else if k == ZC_ROOT { "root views (in refreshes)" } else { "indirect arguments (in refreshes)" };
                let classes: Vec<String> = ZC_NAMES.iter().zip(pages.iter().zip(&bytes)).map(|(c, (p, b))| format!("{c} {p}/{} KiB", b >> 10)).collect();
                format!("{name}: {} [{}], {}% of bytes in ranges wholly open or openable", a >> 10, classes.join(", "), if a == 0 { 0 } else { w * 100 / a })
            })
        })
        .collect();
    let would: usize = {
        let mut m = ZC_WOULD_OPEN.lock().unwrap_or_else(|e| e.into_inner());
        let v = m.values().sum();
        m.clear();
        v
    };
    let pinned = PINNED.load(Relaxed);
    let peak = ZC_PIN_PEAK.fetch_max(pinned + would as u64, Relaxed).max(pinned + would as u64);
    let r: Vec<u64> = ZC_ROOTS.iter().map(|c| c.swap(0, Relaxed) / sampled).collect();
    let rec: Vec<u64> = ZC_RECORDS.iter().map(|c| c.swap(0, Relaxed) / n.max(1)).collect();
    format!(
        "{}% of copied bytes in ranges whose every 64 KiB page is in an open or openable window ({} of {} KiB per sampled frame); {}; {}; pinned now {} MiB, with every openable window touched opened {} MiB (most so far {} MiB); table roots {} in a window, {} not; root views by address {} in a window, {} not; table records read by draws per frame: {} ours (vVIE), {} raw console, {} zero",
        if all == 0 { 0 } else { ok * 100 / all },
        ok >> 10,
        all >> 10,
        if kinds.is_empty() { "nothing copied".into() } else { kinds.join("; ") },
        alloc_text(sampled),
        pinned >> 20,
        (pinned + would as u64) >> 20,
        peak >> 20,
        r[0], r[1], r[2], r[3],
        rec[0], rec[1], rec[2]
    )
}

/// Notes one copy of kind `kind` of title memory `[address, address + len)`.
pub fn would_share(kind: usize, address: usize, len: usize) {
    zero_copy_note(kind, address, len);
    use std::sync::atomic::Ordering::Relaxed;
    if !counting() || kind >= WOULD.len() {
        return;
    }
    let Some(s) = shared() else { return };
    let w = &WOULD[kind];
    match s.resolve(address, len) {
        Ok(_) => {
            w[0].fetch_add(1, Relaxed);
            w[1].fetch_add(len as u64, Relaxed);
        }
        Err(m) => {
            w[2].fetch_add(1, Relaxed);
            w[3].fetch_add(len as u64, Relaxed);
            MISSES[m as usize].fetch_add(1, Relaxed);
        }
    }
}

/// Why a fill of title memory at `address` was not read in place:
/// 0 resolves but off the 256-byte alignment of constants (misaligned
/// constant table), 1 not resolved, 2 crosses a window, 3 other.
pub fn fill_cause(address: usize, len: usize) -> usize {
    match shared().map(|s| s.resolve(address, len)) {
        Some(Ok((_, offset))) => if offset % 256 != 0 { 0 } else { 3 },
        Some(Err(Miss::CrossesWindow)) => 2,
        _ => 1,
    }
}

/// Bytes of the views left on host copies by reason, buffer srv and
/// buffer uav only (`XNW_SHARED=2`), read without zeroing.
pub fn view_misses_text(n: u64) -> String {
    use std::sync::atomic::Ordering::Relaxed;
    if !in_place() {
        return "needs XNW_SHARED=2".into();
    }
    let n = n.max(1);
    let parts: Vec<String> = [VIEW_SRV, VIEW_UAV]
        .iter()
        .map(|&k| {
            let why: Vec<String> = WHY_NAMES.iter().zip(&WHY[k]).filter_map(|(w, a)| Some(a.load(Relaxed) / n).filter(|&v| v > 0).map(|v| format!("{w} {v}"))).collect();
            format!("{} in place {} of {} [{}], {} KiB left", VIEW_NAMES[k], VIEWS[k][0].load(Relaxed) / n, (VIEWS[k][0].load(Relaxed) + VIEWS[k][1].load(Relaxed)) / n, if why.is_empty() { "none".into() } else { why.join(", ") }, MISS_BYTES[k].load(Relaxed) / n / 1024)
        })
        .collect();
    parts.join("; ")
}

/// The frame-statistics part, per frame over `n` frames, zeroing the
/// counts; empty unless `XNW_SHARED=1`.
pub fn stats_text(n: u64) -> String {
    use std::sync::atomic::Ordering::Relaxed;
    if !counting() {
        return String::new();
    }
    let n = n.max(1);
    let kinds: Vec<String> = KIND_NAMES
        .iter()
        .zip(&WOULD)
        .filter_map(|(name, c)| {
            let v: Vec<u64> = c.iter().map(|a| a.swap(0, Relaxed) / n).collect();
            (v[0] + v[2] > 0).then(|| format!("{name} {} ({} KiB) of {} ({} KiB)", v[0], v[1] >> 10, v[0] + v[2], (v[1] + v[3]) >> 10))
        })
        .collect();
    let misses: Vec<String> = MISS_NAMES.iter().zip(&MISSES).filter_map(|(name, c)| Some(c.swap(0, Relaxed) / n).filter(|&v| v > 0).map(|v| format!("{name} {v}"))).collect();
    let views: Vec<String> = VIEW_NAMES
        .iter()
        .zip(&VIEWS)
        .filter_map(|(name, [yes, no])| {
            let (yes, no) = (yes.swap(0, Relaxed) / n, no.swap(0, Relaxed) / n);
            (yes + no > 0).then(|| format!("{name} {yes}/{}", yes + no))
        })
        .collect();
    // Per kind, why views stayed on host copies.
    let whys: Vec<String> = VIEW_NAMES
        .iter()
        .zip(&WHY)
        .filter_map(|(name, c)| {
            let v: Vec<String> = WHY_NAMES.iter().zip(c).filter_map(|(w, a)| Some(a.swap(0, Relaxed) / n).filter(|&v| v > 0).map(|v| format!("{w} {v}"))).collect();
            (!v.is_empty()).then(|| format!("{name} [{}]", v.join(", ")))
        })
        .collect();
    MISS_BYTES.iter().for_each(|b| b.store(0, Relaxed));
    let opened = OPENED[0].swap(0, Relaxed);
    let opened_us = OPENED[1].swap(0, Relaxed);
    format!(
        "; shared ({}, per frame): {}; misses: {}; in place: {}; left on copies: {}; heaps {} (opened {} in {} us over the {n} frames), pinned {} MiB",
        if in_place() { "still copied that would read in place" } else { "would read in place" },
        if kinds.is_empty() { "none".into() } else { kinds.join(", ") },
        if misses.is_empty() { "none".into() } else { misses.join(", ") },
        if views.is_empty() { "none".into() } else { views.join(", ") },
        if whys.is_empty() { "none".into() } else { whys.join("; ") },
        HEAPS.load(Relaxed),
        opened,
        opened_us,
        PINNED.load(Relaxed) >> 20
    )
}

/// Test helper: the GPU writes `data` at `offset` of `buffer` (a copy from
/// an upload buffer).
pub fn write_through(host: &Host, buffer: &ID3D12Resource, offset: u64, data: &[u8]) -> Result<()> {
    let up = host.buffer(data.len() as u64, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
    // SAFETY: COM calls on live objects; the mapped range is `data.len()` long.
    unsafe {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        up.Map(0, None, Some(&mut ptr))?;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.cast::<u8>(), data.len());
        up.Unmap(0, None);
        let alloc: ID3D12CommandAllocator = host.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
        let list: ID3D12GraphicsCommandList = host.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
        list.CopyBufferRegion(buffer, offset, &up, 0, data.len() as u64);
        list.Close()?;
        host.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
        let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        host.queue.Signal(&fence, 1)?;
        let event = CreateEventW(None, false, false, None)?;
        fence.SetEventOnCompletion(1, event)?;
        WaitForSingleObject(event, 10_000);
        let _ = CloseHandle(event);
        host.device.GetDeviceRemovedReason()
    }
}

/// Test helper: copies `len` bytes at `offset` of `buffer` to the CPU
/// through the GPU (a copy reads the buffer's memory on the GPU).
pub fn read_back(host: &Host, buffer: &ID3D12Resource, offset: u64, len: usize) -> Result<Vec<u8>> {
    let back = host.buffer(len as u64, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
    // SAFETY: COM calls on live objects; the mapped range is `len` long.
    unsafe {
        let alloc: ID3D12CommandAllocator = host.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
        let list: ID3D12GraphicsCommandList = host.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
        list.CopyBufferRegion(&back, 0, buffer, offset, len as u64);
        list.Close()?;
        host.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
        let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        host.queue.Signal(&fence, 1)?;
        let event = CreateEventW(None, false, false, None)?;
        fence.SetEventOnCompletion(1, event)?;
        WaitForSingleObject(event, 10_000);
        let _ = CloseHandle(event);
        host.device.GetDeviceRemovedReason()?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        back.Map(0, None, Some(&mut ptr))?;
        let out = std::slice::from_raw_parts(ptr.cast::<u8>(), len).to_vec();
        back.Unmap(0, None);
        Ok(out)
    }
}

// ---- the probe (`xnw gpu-check --shared-memory`) ---------------------------
/// Output words: 0 sum, 4..8 the constant buffer's first 16 bytes, 8.. vertices.
const OUT_BYTES: u64 = 4096;
const VERTS: u32 = 64;

/// DXIL the caller compiled (see `SHADERS`).
pub struct Shaders {
    pub cs: Vec<u8>,
    pub vs: Vec<u8>,
    pub tex: Vec<u8>,
}

/// One source, three entry points chosen by `VS` / `TEX` (entry `main`).
pub const SHADERS: &str = r#"
cbuffer C : register(b0) { uint4 c0; };
cbuffer K : register(b1) { uint n4; uint threads; uint pad0; uint pad1; };
RWByteAddressBuffer o : register(u0);
SamplerState pt : register(s0);
#if TEX
Texture2D<float4> tex : register(t0);
[numthreads(1, 1, 1)]
void main() {
    float4 t = tex.SampleLevel(pt, float2(1.5 / 1024.0, 0.5 / 1024.0), 0);
    o.Store4(16, uint4(round(t * 255.0)));
}
#elif VS
struct VI { uint v : V; };
float4 main(VI i, uint id : SV_VertexID) : SV_Position {
    o.Store(32 + id * 4, i.v);
    return float4(0, 0, 0, 1);
}
#else
ByteAddressBuffer src : register(t0);
[numthreads(256, 1, 1)]
void main(uint3 t : SV_DispatchThreadID) {
    uint s = 0;
    for (uint i = t.x; i < n4; i += threads) {
        uint4 v = src.Load4(i * 16);
        s += v.x + v.y + v.z + v.w;
    }
    o.InterlockedAdd(0, s);
    if (t.x == 0) o.Store4(16, c0);
}
#endif
"#;

fn hr(e: &windows::core::Error) -> String {
    format!("hr {:#010x}", e.code().0 as u32)
}

fn line(ok: bool, what: &str, detail: &str) {
    println!("{}  {what}: {detail}", if ok { "PASS" } else { "FAIL" });
}

fn info(what: &str, detail: &str) {
    println!("info  {what}: {detail}");
}

fn pattern(k: usize, seed: u32) -> u32 {
    (k as u32).wrapping_mul(2_654_435_761).wrapping_add(seed)
}

/// Writes the pattern over `bytes` at `p`; word k of the range holds `pattern(k, seed)`.
fn fill(p: *mut u8, bytes: usize, seed: u32) {
    // SAFETY: callers pass committed, writable memory of `bytes`.
    let w = unsafe { std::slice::from_raw_parts_mut(p.cast::<u32>(), bytes / 4) };
    for (k, v) in w.iter_mut().enumerate() {
        *v = pattern(k, seed);
    }
}

fn expected_sum(k0: usize, bytes: usize, seed: u32) -> u32 {
    (k0..k0 + bytes / 4).fold(0u32, |s, k| s.wrapping_add(pattern(k, seed)))
}

// ---- process memory ------------------------------------------------------

struct Block {
    p: *mut u8,
}

impl Block {
    fn new(size: usize, kind: u32) -> Option<Block> {
        // SAFETY: a fresh allocation.
        let p = unsafe { win::VirtualAlloc(std::ptr::null(), size, win::MEM_COMMIT | win::MEM_RESERVE | kind, win::PAGE_READWRITE) };
        (!p.is_null()).then_some(Block { p: p.cast() })
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        if !self.p.is_null() {
            // SAFETY: our own allocation.
            unsafe { win::VirtualFree(self.p.cast(), 0, win::MEM_RELEASE) };
        }
    }
}

struct Section(win::HANDLE);

impl Section {
    fn new(size: usize, sec: u32) -> std::result::Result<Section, String> {
        // SAFETY: plain section creation, as xnw-hle xmem does.
        let h = unsafe { win::CreateFileMappingW(win::INVALID_HANDLE_VALUE, std::ptr::null(), win::PAGE_READWRITE | sec, (size as u64 >> 32) as u32, size as u32, std::ptr::null()) };
        if h.is_null() { Err(format!("CreateFileMappingW: {}", win::last_error())) } else { Ok(Section(h)) }
    }

    fn handle(&self) -> HANDLE {
        HANDLE(self.0)
    }

    /// The whole section, mapped at an address of the system's choosing.
    fn view(&self, size: usize) -> Option<View> {
        // SAFETY: a live section handle.
        let p = unsafe { win::MapViewOfFile(self.0, win::FILE_MAP_WRITE, 0, 0, size) };
        (!p.is_null()).then_some(View { p: p.cast(), placeholder: false })
    }

    /// `size` bytes at section `offset`, mapped into a fresh placeholder
    /// (MapViewOfFile3, as xmem maps physical pages).
    fn slice(&self, offset: usize, size: usize) -> Option<View> {
        let p = placeholder(size)?;
        map_into(self, p, offset, size).then_some(View { p, placeholder: true })
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        // SAFETY: our own handle.
        unsafe { win::CloseHandle(self.0) };
    }
}

fn placeholder(size: usize) -> Option<*mut u8> {
    // SAFETY: a fresh reservation.
    let p = unsafe {
        win::virtual_alloc2()(win::GetCurrentProcess(), std::ptr::null(), size, win::MEM_RESERVE | win::MEM_RESERVE_PLACEHOLDER, win::PAGE_NOACCESS, std::ptr::null_mut(), 0)
    };
    (!p.is_null()).then_some(p.cast())
}

fn map_into(section: &Section, at: *mut u8, offset: usize, size: usize) -> bool {
    // SAFETY: `at` is a placeholder of exactly `size` bytes.
    let p = unsafe {
        win::map_view_of_file3()(section.0, win::GetCurrentProcess(), at.cast(), offset as u64, size, win::MEM_REPLACE_PLACEHOLDER, win::PAGE_READWRITE, std::ptr::null_mut(), 0)
    };
    p == at.cast()
}

/// Unmaps a placeholder view, keeping the placeholder.
fn unmap_keep(at: *mut u8) -> bool {
    // SAFETY: a view we mapped into a placeholder.
    unsafe { win::unmap_view_of_file2()(win::GetCurrentProcess(), at.cast(), win::MEM_PRESERVE_PLACEHOLDER) != 0 }
}

struct View {
    p: *mut u8,
    placeholder: bool,
}

impl Drop for View {
    fn drop(&mut self) {
        // SAFETY: our own view (a failed unmap of an already unmapped view is harmless).
        unsafe {
            win::UnmapViewOfFile(self.p.cast());
            if self.placeholder {
                win::VirtualFree(self.p.cast(), 0, win::MEM_RELEASE);
            }
        }
    }
}

fn commit(p: *mut u8, size: usize) -> bool {
    // SAFETY: commits pages of a reserved range we own.
    unsafe { !win::VirtualAlloc(p.cast(), size, win::MEM_COMMIT, win::PAGE_READWRITE).is_null() }
}

fn protect(p: *mut u8, size: usize, prot: u32) -> bool {
    let mut old = 0;
    // SAFETY: a range we own.
    unsafe { win::VirtualProtect(p.cast(), size, prot, &mut old) != 0 }
}

// ---- GPU -----------------------------------------------------------------

struct Readout {
    sum: u32,
    c0: [u32; 4],
    vb: Vec<u32>,
    seconds: f64,
}

struct Rig {
    host: Host,
    root: ID3D12RootSignature,
    cs: ID3D12PipelineState,
    vs: ID3D12PipelineState,
    tex: ID3D12PipelineState,
    heap: ID3D12DescriptorHeap,
    out: ID3D12Resource,
    out_ptr: *mut u32,
    queries: ID3D12QueryHeap,
    stamps: ID3D12Resource,
    fence: ID3D12Fence,
    fence_value: std::cell::Cell<u64>,
}

fn device3(host: &Host) -> Result<ID3D12Device3> {
    host.device.cast()
}

fn buffer_desc(bytes: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

fn heap_desc(h: &ID3D12Heap) -> String {
    // SAFETY: a live heap.
    let d = unsafe { h.GetDesc() };
    format!(
        "size {} KiB, type {}, cpu page {}, pool {}, align {:#x}, flags {:#x}",
        d.SizeInBytes / KIB as u64,
        d.Properties.Type.0,
        d.Properties.CPUPageProperty.0,
        d.Properties.MemoryPoolPreference.0,
        d.Alignment,
        d.Flags.0
    )
}

impl Rig {
    fn new(agility: Option<&Path>, sh: &Shaders) -> Result<Rig> {
        let host = Host::new(agility)?;
        // SAFETY: COM calls; every pointer passed is to a local that outlives the call.
        unsafe {
            let d = &host.device;
            let range = D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: 1,
                BaseShaderRegister: 0,
                RegisterSpace: 0,
                OffsetInDescriptorsFromTableStart: 0,
            };
            let all = D3D12_SHADER_VISIBILITY_ALL;
            let params = [
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                    Anonymous: D3D12_ROOT_PARAMETER_0 { DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &range } },
                    ShaderVisibility: all,
                },
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
                    Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } },
                    ShaderVisibility: all,
                },
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                    Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } },
                    ShaderVisibility: all,
                },
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 1, RegisterSpace: 0, Num32BitValues: 4 } },
                    ShaderVisibility: all,
                },
            ];
            let sampler = D3D12_STATIC_SAMPLER_DESC {
                Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
                AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                MaxLOD: f32::MAX,
                ShaderVisibility: all,
                ..Default::default()
            };
            let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: 1,
                pStaticSamplers: &sampler,
                Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
            };
            let mut blob: Option<ID3DBlob> = None;
            D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &mut blob, None)?;
            let blob = blob.expect("blob on success");
            let root: ID3D12RootSignature = d.CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()))?;

            let compute = |dxil: &[u8]| -> Result<ID3D12PipelineState> {
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: ManuallyDrop::new(Some(root.clone())),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr().cast(), BytecodeLength: dxil.len() },
                    ..Default::default()
                };
                let pso = d.CreateComputePipelineState(&desc);
                drop(ManuallyDrop::into_inner(desc.pRootSignature));
                pso
            };
            let cs = compute(&sh.cs)?;
            let tex = compute(&sh.tex)?;
            let element = D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(c"V".as_ptr().cast()),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32_UINT,
                InputSlot: 0,
                AlignedByteOffset: 0,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            };
            let gdesc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(root.clone())),
                VS: D3D12_SHADER_BYTECODE { pShaderBytecode: sh.vs.as_ptr().cast(), BytecodeLength: sh.vs.len() },
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() },
                InputLayout: D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: &element, NumElements: 1 },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let vs = d.CreateGraphicsPipelineState(&gdesc);
            drop(ManuallyDrop::into_inner(gdesc.pRootSignature));
            let vs = vs?;

            let heap: ID3D12DescriptorHeap = d.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?;
            // Output in CPU-visible write-back memory: zeroed and read with no copy.
            let props = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let mut out: Option<ID3D12Resource> = None;
            d.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &buffer_desc(OUT_BYTES, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_COMMON, None, &mut out)?;
            let out = out.expect("resource on success");
            let mut ptr: *mut c_void = std::ptr::null_mut();
            out.Map(0, None, Some(&mut ptr))?;
            let mut queries: Option<ID3D12QueryHeap> = None;
            d.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2, NodeMask: 0 }, &mut queries)?;
            let stamps = host.buffer(16, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
            let fence = d.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            Ok(Rig {
                host,
                root,
                cs,
                vs,
                tex,
                heap,
                out,
                out_ptr: ptr.cast(),
                queries: queries.expect("query heap on success"),
                stamps,
                fence,
                fence_value: std::cell::Cell::new(0),
            })
        }
    }

    fn d3(&self) -> Result<ID3D12Device3> {
        device3(&self.host)
    }

    fn open_address(&self, p: *const u8) -> Result<ID3D12Heap> {
        // SAFETY: the address is passed to the runtime, which validates it.
        unsafe { self.d3()?.OpenExistingHeapFromAddress(p.cast()) }
    }

    fn place(&self, heap: &ID3D12Heap, offset: u64, bytes: u64, flags: D3D12_RESOURCE_FLAGS) -> Result<ID3D12Resource> {
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: the description outlives the call.
        unsafe { self.host.device.CreatePlacedResource(heap, offset, &buffer_desc(bytes, flags), D3D12_RESOURCE_STATE_COMMON, None, &mut out)? };
        Ok(out.expect("resource on success"))
    }

    fn removed(&self) -> String {
        // SAFETY: a live device.
        match unsafe { self.host.device.GetDeviceRemovedReason() } {
            Ok(()) => "device ok".into(),
            Err(e) => format!("DEVICE REMOVED {}", hr(&e)),
        }
    }

    fn list(&self, pso: &ID3D12PipelineState) -> Result<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)> {
        // SAFETY: plain COM calls.
        unsafe {
            let alloc: ID3D12CommandAllocator = self.host.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.host.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, pso)?;
            Ok((alloc, list))
        }
    }

    /// Closes, submits and waits (10 s at most) for `list`.
    fn submit(&self, list: &ID3D12GraphicsCommandList) -> Result<()> {
        // SAFETY: plain COM calls; the event is closed before return.
        unsafe {
            list.Close()?;
            self.host.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
            let v = self.fence_value.get() + 1;
            self.fence_value.set(v);
            self.host.queue.Signal(&self.fence, v)?;
            let event = CreateEventW(None, false, false, None)?;
            self.fence.SetEventOnCompletion(v, event)?;
            let waited = WaitForSingleObject(event, 10_000);
            let _ = CloseHandle(event);
            self.host.device.GetDeviceRemovedReason()?;
            if waited != WAIT_OBJECT_0 {
                return Err(windows::core::Error::from_hresult(windows::core::HRESULT(0x8000_0005u32 as i32)));
            }
            Ok(())
        }
    }

    fn copy(&self, dst: &ID3D12Resource, src: &ID3D12Resource, bytes: u64) -> Result<()> {
        let (_alloc, list) = self.list(&self.cs)?;
        // SAFETY: both buffers are at least `bytes` long.
        unsafe { list.CopyBufferRegion(dst, 0, src, 0, bytes) };
        self.submit(&list)
    }

    /// Reads `bytes` of `src` from byte `first`: summed by the compute shader
    /// through a raw SRV, the first 16 bytes as a root CBV, and (if `vb`) the
    /// first `VERTS` words as a vertex buffer. `groups` of 256 threads.
    fn read(&self, src: &ID3D12Resource, first: u64, bytes: u64, vb: bool, groups: u32) -> Result<Readout> {
        self.read_to(src, first, bytes, vb, groups, None)
    }

    /// As [`Rig::read`], the shader's output (sum at byte 0, the constant
    /// buffer's first 16 bytes at 16) going to GPU address `uav` instead of
    /// the probe's own buffer (whose readout is then zero).
    fn read_to(&self, src: &ID3D12Resource, first: u64, bytes: u64, vb: bool, groups: u32, uav: Option<u64>) -> Result<Readout> {
        // SAFETY: COM calls; `out_ptr` maps OUT_BYTES; locals outlive each call.
        unsafe {
            std::ptr::write_bytes(self.out_ptr.cast::<u8>(), 0, OUT_BYTES as usize);
            let addr = src.GetGPUVirtualAddress() + first;
            let words = (bytes / 4).min(1 << 27);
            let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV { FirstElement: first / 4, NumElements: words as u32, StructureByteStride: 0, Flags: D3D12_BUFFER_SRV_FLAG_RAW },
                },
            };
            self.host.device.CreateShaderResourceView(src, Some(&srv), self.heap.GetCPUDescriptorHandleForHeapStart());
            let consts = [(words / 4) as u32, groups * 256, 0, 0];
            let out_va = uav.unwrap_or_else(|| self.out.GetGPUVirtualAddress());
            let (_alloc, list) = self.list(&self.cs)?;
            list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            let table = self.heap.GetGPUDescriptorHandleForHeapStart();
            if vb {
                list.SetPipelineState(&self.vs);
                list.SetGraphicsRootSignature(&self.root);
                list.SetGraphicsRootDescriptorTable(0, table);
                list.SetGraphicsRootConstantBufferView(1, addr);
                list.SetGraphicsRootUnorderedAccessView(2, out_va);
                list.SetGraphicsRoot32BitConstants(3, 4, consts.as_ptr().cast(), 0);
                list.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
                list.IASetVertexBuffers(0, Some(&[D3D12_VERTEX_BUFFER_VIEW { BufferLocation: addr, SizeInBytes: VERTS * 4, StrideInBytes: 4 }]));
                list.DrawInstanced(VERTS, 1, 0, 0);
                list.SetPipelineState(&self.cs);
            }
            list.SetComputeRootSignature(&self.root);
            list.SetComputeRootDescriptorTable(0, table);
            list.SetComputeRootConstantBufferView(1, addr);
            list.SetComputeRootUnorderedAccessView(2, out_va);
            list.SetComputeRoot32BitConstants(3, 4, consts.as_ptr().cast(), 0);
            list.EndQuery(&self.queries, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            list.Dispatch(groups, 1, 1);
            list.EndQuery(&self.queries, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            list.ResolveQueryData(&self.queries, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.stamps, 0);
            self.submit(&list)?;
            let o = std::slice::from_raw_parts(self.out_ptr, (OUT_BYTES / 4) as usize);
            let mut ptr: *mut c_void = std::ptr::null_mut();
            self.stamps.Map(0, None, Some(&mut ptr))?;
            let t = std::slice::from_raw_parts(ptr.cast::<u64>(), 2);
            let ticks = t[1].wrapping_sub(t[0]);
            self.stamps.Unmap(0, None);
            let freq = self.host.queue.GetTimestampFrequency()?;
            Ok(Readout {
                sum: o[0],
                c0: [o[4], o[5], o[6], o[7]],
                vb: if vb { o[8..8 + VERTS as usize].to_vec() } else { Vec::new() },
                seconds: ticks as f64 / freq as f64,
            })
        }
    }

    /// Reads `bytes` from byte `first` of `src` and compares with the pattern
    /// `seed` that the CPU wrote there (word 0 of the range = pattern(k0)).
    /// Prints one line and returns whether every binding matched.
    #[allow(clippy::too_many_arguments)]
    fn verify(&self, what: &str, src: &ID3D12Resource, first: u64, bytes: usize, vb: bool, seed: u32, k0: usize) -> bool {
        match self.read(src, first, bytes as u64, vb, 64) {
            Err(e) => {
                line(false, what, &format!("submit failed {} ({})", hr(&e), self.removed()));
                false
            }
            Ok(r) => {
                let sum_ok = r.sum == expected_sum(k0, bytes, seed);
                let cb_ok = (0..4).all(|k| r.c0[k] == pattern(k0 + k, seed));
                let vb_ok = !vb || (0..VERTS as usize).all(|k| r.vb[k] == pattern(k0 + k, seed));
                let stale = |s: u32| r.sum == expected_sum(k0, bytes, s);
                let seen = if sum_ok {
                    "new data".to_string()
                } else if r.sum == 0 {
                    "zeros".to_string()
                } else if let Some(s) = (1..=4).find(|&s| s != seed && stale(s)) {
                    format!("STALE data (pattern {s})")
                } else {
                    format!("other data (sum {:#x})", r.sum)
                };
                let mut parts = vec![format!("SRV sum {}", if sum_ok { "ok" } else { "wrong" }), format!("CBV {}", if cb_ok { "ok" } else { "wrong" })];
                if vb {
                    parts.push(format!("VB {}", if vb_ok { "ok" } else { "wrong" }));
                }
                let ok = sum_ok && cb_ok && vb_ok;
                line(ok, what, &format!("{}; GPU sees {seen}; {}", parts.join(", "), self.removed()));
                ok
            }
        }
    }
}

fn texture_desc(format: DXGI_FORMAT, mips: u16, layout: D3D12_TEXTURE_LAYOUT, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 1024,
        Height: 1024,
        DepthOrArraySize: 1,
        MipLevels: mips,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: layout,
        Flags: flags,
        ..Default::default()
    }
}

/// Opens a heap over `p` and places one buffer of `bytes` (cross-adapter).
fn open_buffer(rig: &Rig, p: *const u8, bytes: usize) -> Result<(ID3D12Heap, ID3D12Resource)> {
    let heap = rig.open_address(p)?;
    let buf = rig.place(&heap, 0, bytes as u64, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER)?;
    Ok((heap, buf))
}

fn fresh(agility: Option<&Path>, sh: &Shaders, step: &str) -> Option<Rig> {
    match Rig::new(agility, sh) {
        Ok(r) => Some(r),
        Err(e) => {
            line(false, step, &format!("no device: {}", hr(&e)));
            None
        }
    }
}

pub fn run(agility: Option<&Path>, sh: &Shaders) -> Result<()> {
    println!("\nshared memory probe");
    let rig = Rig::new(agility, sh)?;
    caps(&rig);
    step1(&rig);
    step2(&rig);
    drop(rig);
    step3(agility, sh);
    if let Some(rig) = fresh(agility, sh, "4 cost") {
        step4(&rig);
    }
    if let Some(rig) = fresh(agility, sh, "5 textures") {
        step5(&rig);
    }
    if let Some(rig) = fresh(agility, sh, "6 runtime view") {
        step6(&rig);
    }
    Ok(())
}

fn caps(rig: &Rig) {
    let d = &rig.host.device;
    let existing = feature(d, D3D12_FEATURE_EXISTING_HEAPS, D3D12_FEATURE_DATA_EXISTING_HEAPS::default());
    let arch = feature(d, D3D12_FEATURE_ARCHITECTURE1, D3D12_FEATURE_DATA_ARCHITECTURE1::default());
    let o0 = feature(d, D3D12_FEATURE_D3D12_OPTIONS, D3D12_FEATURE_DATA_D3D12_OPTIONS::default());
    let o16 = feature(d, D3D12_FEATURE_D3D12_OPTIONS16, D3D12_FEATURE_DATA_D3D12_OPTIONS16::default());
    info("caps", &format!("EXISTING_HEAPS.Supported {:?}", existing.map(|e| e.Supported.as_bool())));
    info("caps", &format!("ARCHITECTURE1 UMA {:?} CacheCoherentUMA {:?} IsolatedMMU {:?}", arch.map(|a| a.UMA.as_bool()), arch.map(|a| a.CacheCoherentUMA.as_bool()), arch.map(|a| a.IsolatedMMU.as_bool())));
    info("caps", &format!("OPTIONS16 GPUUploadHeapSupported {:?}", o16.map(|o| o.GPUUploadHeapSupported.as_bool())));
    info("caps", &format!("OPTIONS CrossAdapterRowMajorTextureSupported {:?}", o0.map(|o| o.CrossAdapterRowMajorTextureSupported.as_bool())));
    info("caps", &format!("ID3D12Device3 {}  ID3D12Device13 {}", rig.d3().is_ok(), d.cast::<ID3D12Device13>().is_ok()));
}

fn step1(rig: &Rig) {
    println!("\n-- 1 VirtualAlloc block (64 MiB, MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE)");
    let size = 64 * MIB;
    let Some(blk) = Block::new(size, 0) else {
        line(false, "1 VirtualAlloc", &win::last_error().to_string());
        return;
    };
    fill(blk.p, size, 1);
    let heap = match rig.open_address(blk.p) {
        Ok(h) => {
            line(true, "1 OpenExistingHeapFromAddress(base)", &heap_desc(&h));
            h
        }
        Err(e) => return line(false, "1 OpenExistingHeapFromAddress(base)", &hr(&e)),
    };
    // SAFETY: inside the block.
    match rig.open_address(unsafe { blk.p.add(64 * KIB) }) {
        Ok(h) => info("1 OpenExistingHeapFromAddress(base+64K)", &format!("accepted: {}", heap_desc(&h))),
        Err(e) => info("1 OpenExistingHeapFromAddress(base+64K)", &format!("refused {} (only an allocation base opens)", hr(&e))),
    }
    let mut buf = None;
    for (name, flags) in [("NONE", D3D12_RESOURCE_FLAG_NONE), ("ALLOW_CROSS_ADAPTER", D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER)] {
        match rig.place(&heap, 0, size as u64, flags) {
            Ok(b) => {
                line(true, &format!("1 CreatePlacedResource buffer 64 MiB flags {name}"), "ok");
                buf = Some(b);
            }
            Err(e) => line(false, &format!("1 CreatePlacedResource buffer 64 MiB flags {name}"), &hr(&e)),
        }
    }
    match rig.place(&heap, 0, size as u64, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {
        Ok(_) => info("1 CreatePlacedResource flags ALLOW_CROSS_ADAPTER|UAV", "ok"),
        Err(e) => info("1 CreatePlacedResource flags ALLOW_CROSS_ADAPTER|UAV", &hr(&e)),
    }
    let Some(buf) = buf else { return };
    rig.verify("1 GPU reads CPU writes (pattern 1)", &buf, 0, size, true, 1, 0);
    fill(blk.p, size, 2);
    rig.verify("1 GPU reads CPU rewrite after creation, no copy (pattern 2)", &buf, 0, size, true, 2, 0);
}

fn step2(rig: &Rig) {
    println!("\n-- 2 pagefile sections (xmem physical pages)");
    let d3 = match rig.d3() {
        Ok(d) => d,
        Err(e) => return line(false, "2 ID3D12Device3", &hr(&e)),
    };
    for (sec_name, sec, size) in [
        ("SEC_COMMIT", win::SEC_COMMIT, 64 * MIB),
        ("SEC_COMMIT", win::SEC_COMMIT, GIB),
        ("SEC_RESERVE", win::SEC_RESERVE, 64 * MIB),
        ("SEC_RESERVE", win::SEC_RESERVE, GIB),
        ("SEC_RESERVE", win::SEC_RESERVE, 4 * GIB),
        ("SEC_RESERVE", win::SEC_RESERVE, 16 * GIB),
    ] {
        let what = format!("2 OpenExistingHeapFromFileMapping {sec_name} {} MiB", size / MIB);
        let s = match Section::new(size, sec) {
            Ok(s) => s,
            Err(e) => {
                line(false, &what, &e);
                continue;
            }
        };
        // SAFETY: a live section handle.
        match unsafe { d3.OpenExistingHeapFromFileMapping::<ID3D12Heap>(s.handle()) } {
            Ok(h) => {
                line(true, &what, &heap_desc(&h));
                if size == 64 * MIB {
                    section_alias(rig, &s, &h, sec == win::SEC_RESERVE, &format!("2 {sec_name} file-mapping heap"));
                }
            }
            Err(e) => line(false, &what, &hr(&e)),
        }
    }

    // The five-star-recomp route: a whole-section view opened by address.
    for (sec_name, sec, commit_view) in [("SEC_COMMIT", win::SEC_COMMIT, false), ("SEC_RESERVE uncommitted", win::SEC_RESERVE, false), ("SEC_RESERVE committed", win::SEC_RESERVE, true)] {
        let size = 64 * MIB;
        let what = format!("2 OpenExistingHeapFromAddress(whole-section view) {sec_name} 64 MiB");
        let Ok(s) = Section::new(size, sec) else { continue };
        let Some(v) = s.view(size) else {
            line(false, &what, &format!("MapViewOfFile: {}", win::last_error()));
            continue;
        };
        if commit_view && !commit(v.p, size) {
            line(false, &what, &format!("commit view: {}", win::last_error()));
            continue;
        }
        match rig.open_address(v.p) {
            Ok(h) => {
                line(true, &what, &heap_desc(&h));
                section_alias(rig, &s, &h, sec == win::SEC_RESERVE && !commit_view, &format!("2 {sec_name} address heap"));
            }
            Err(e) => line(false, &what, &hr(&e)),
        }
    }

    // As xmem: one 16 GiB SEC_RESERVE section; a 64 MiB window view at 1 GiB,
    // committed and opened by address; the title's page alias is a placeholder view.
    if let Ok(s) = Section::new(16 * GIB, win::SEC_RESERVE) {
        let what = "2 16 GiB SEC_RESERVE section: OpenExistingHeapFromAddress(64 MiB view at 1 GiB, committed)";
        // SAFETY: a live section; the view is unmapped by hand below.
        let v = unsafe { win::MapViewOfFile(s.0, win::FILE_MAP_WRITE, 0, GIB as u32, 64 * MIB) }.cast::<u8>();
        if v.is_null() || !commit(v, 64 * MIB) {
            line(false, what, &win::last_error().to_string());
        } else {
            let v = View { p: v, placeholder: false };
            match rig.open_address(v.p) {
                Ok(h) => {
                    line(true, what, &heap_desc(&h));
                    if let Ok(buf) = rig.place(&h, MIB as u64, 64 * KIB as u64, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER)
                        && let Some(p) = s.slice(GIB + MIB, 64 * KIB)
                    {
                        fill(p.p, 64 * KIB, 3);
                        rig.verify("2 16 GiB section: GPU reads writes through the page's placeholder alias", &buf, 0, 64 * KIB, true, 3, 0);
                    }
                }
                Err(e) => line(false, what, &hr(&e)),
            }
        }
    }

    // A heap over one 64 KiB placeholder view (how xmem maps a title page).
    let Ok(s) = Section::new(64 * MIB, win::SEC_COMMIT) else { return };
    let Some(p) = s.slice(MIB, 64 * KIB) else {
        return line(false, "2 MapViewOfFile3 into placeholder", &win::last_error().to_string());
    };
    match rig.open_address(p.p) {
        Ok(h) => line(true, "2 OpenExistingHeapFromAddress(64 KiB placeholder view)", &heap_desc(&h)),
        Err(e) => line(false, "2 OpenExistingHeapFromAddress(64 KiB placeholder view)", &hr(&e)),
    }
    match rig.host.device.cast::<ID3D12Device13>() {
        // SAFETY: the address is validated by the runtime.
        Ok(d13) => match unsafe { d13.OpenExistingHeapFromAddress1::<ID3D12Heap>(p.p.cast(), 64 * KIB) } {
            Ok(h) => line(true, "2 OpenExistingHeapFromAddress1(64 KiB placeholder view, 64 KiB)", &heap_desc(&h)),
            Err(e) => line(false, "2 OpenExistingHeapFromAddress1(64 KiB placeholder view, 64 KiB)", &hr(&e)),
        },
        Err(e) => line(false, "2 ID3D12Device13", &hr(&e)),
    }
}

/// With a heap over the whole section: map the slice at 1 MiB into a fresh
/// placeholder (as xmem maps a title page), write through that alias, and
/// read it on the GPU from a buffer placed at the slice's section offset.
fn section_alias(rig: &Rig, s: &Section, heap: &ID3D12Heap, needs_commit: bool, what: &str) {
    let buf = match rig.place(heap, MIB as u64, 64 * KIB as u64, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER) {
        Ok(b) => b,
        Err(e) => return line(false, &format!("{what}: CreatePlacedResource at 1 MiB"), &hr(&e)),
    };
    let Some(p) = s.slice(MIB, 64 * KIB) else {
        return line(false, &format!("{what}: MapViewOfFile3"), &win::last_error().to_string());
    };
    if needs_commit && !commit(p.p, 64 * KIB) {
        return line(false, &format!("{what}: commit slice"), &win::last_error().to_string());
    }
    fill(p.p, 64 * KIB, 3);
    rig.verify(&format!("{what}: GPU reads writes through a placeholder alias view"), &buf, 0, 64 * KIB, true, 3, 0);
}

fn step3(agility: Option<&Path>, sh: &Shaders) {
    println!("\n-- 3 lifetime and remap (fresh device each)");
    let size = 4 * MIB;
    let r = (MIB, MIB); // the range acted on: [1 MiB, 2 MiB)

    // a. decommit part of a VirtualAlloc block under a heap.
    if let Some(rig) = fresh(agility, sh, "3a") {
        if let Some(blk) = Block::new(size, 0) {
            fill(blk.p, size, 1);
            match open_buffer(&rig, blk.p, size) {
                Ok((_h, buf)) => {
                    rig.verify("3a before decommit", &buf, 0, size, false, 1, 0);
                    // SAFETY: inside our block.
                    let at = unsafe { blk.p.add(r.0) };
                    // SAFETY: decommits pages of our block.
                    let ok = unsafe { win::VirtualFree(at.cast(), r.1, win::MEM_DECOMMIT) } != 0;
                    info("3a VirtualFree(MEM_DECOMMIT 1 MiB) under the heap", &if ok { "succeeded".to_string() } else { format!("refused: {}", win::last_error()) });
                    rig.verify("3a GPU read of decommitted range (expects pattern 1)", &buf, r.0 as u64, r.1, false, 1, r.0 / 4);
                    if ok && commit(at, r.1) {
                        fill(at, r.1, 4);
                        rig.verify("3a recommit + CPU write pattern 4: GPU sees it", &buf, r.0 as u64, r.1, false, 4, 0);
                    }
                    drop(buf);
                    drop(_h);
                    // SAFETY: decommits pages of our block.
                    let ok = unsafe { win::VirtualFree(at.cast(), r.1, win::MEM_DECOMMIT) } != 0;
                    info("3a control: the same decommit after the heap is released", if ok { "succeeded" } else { "refused" });
                }
                Err(e) => line(false, "3a open", &hr(&e)),
            }
        }
    }

    // b. read-only (the write watch) before and after opening.
    if let Some(rig) = fresh(agility, sh, "3b") {
        if let Some(blk) = Block::new(size, 0) {
            fill(blk.p, size, 1);
            match open_buffer(&rig, blk.p, size) {
                Ok((_h, buf)) => {
                    let ok = protect(blk.p, size, win::PAGE_READONLY);
                    info("3b VirtualProtect(PAGE_READONLY) under the heap", if ok { "succeeded" } else { "refused" });
                    rig.verify("3b GPU read while read-only", &buf, 0, size, true, 1, 0);
                    protect(blk.p, size, win::PAGE_READWRITE);
                    fill(blk.p, size, 2);
                    rig.verify("3b back to read-write, CPU writes pattern 2", &buf, 0, size, true, 2, 0);
                    let one = protect(blk.p, 4 * KIB, win::PAGE_READONLY);
                    info("3b VirtualProtect(PAGE_READONLY) of one 4 KiB page under the heap", if one { "succeeded" } else { "refused" });
                    drop(buf);
                    drop(_h);
                    let ok = protect(blk.p, size, win::PAGE_READONLY);
                    info("3b control: the same VirtualProtect after the heap is released", if ok { "succeeded" } else { "refused" });
                    protect(blk.p, size, win::PAGE_READWRITE);
                }
                Err(e) => line(false, "3b open", &hr(&e)),
            }
        }
        if let Some(blk) = Block::new(size, 0) {
            fill(blk.p, size, 1);
            protect(blk.p, size, win::PAGE_READONLY);
            match open_buffer(&rig, blk.p, size) {
                Ok((_h, buf)) => {
                    line(true, "3b OpenExistingHeapFromAddress on a PAGE_READONLY block", "ok");
                    rig.verify("3b GPU read of a block opened read-only", &buf, 0, size, false, 1, 0);
                }
                Err(e) => line(false, "3b OpenExistingHeapFromAddress on a PAGE_READONLY block", &hr(&e)),
            }
            protect(blk.p, size, win::PAGE_READWRITE);
        }
        if let Some(blk) = Block::new(size, win::MEM_WRITE_WATCH) {
            fill(blk.p, size, 1);
            match open_buffer(&rig, blk.p, size) {
                Ok((_h, buf)) => {
                    line(true, "3b OpenExistingHeapFromAddress on a MEM_WRITE_WATCH block", "ok");
                    rig.verify("3b GPU read of a MEM_WRITE_WATCH block", &buf, 0, size, false, 1, 0);
                }
                Err(e) => line(false, "3b OpenExistingHeapFromAddress on a MEM_WRITE_WATCH block", &hr(&e)),
            }
        }
    }

    // c. heap over a whole-section view; title page aliases unmapped/remapped.
    if let Some(rig) = fresh(agility, sh, "3c") {
        let size = 64 * MIB;
        if let Ok(s) = Section::new(size, win::SEC_COMMIT)
            && let Some(v) = s.view(size)
        {
            match open_buffer(&rig, v.p, size) {
                Ok((_h, buf)) => {
                    let (a, b) = (MIB, 2 * MIB);
                    if let Some(p) = s.slice(a, 64 * KIB) {
                        fill(p.p, 64 * KIB, 1);
                        rig.verify("3c alias of slice A written: GPU reads A", &buf, a as u64, 64 * KIB, false, 1, 0);
                        let ok = unmap_keep(p.p);
                        info("3c UnmapViewOfFile2(alias, MEM_PRESERVE_PLACEHOLDER)", if ok { "ok" } else { "failed" });
                        rig.verify("3c after unmap of alias: GPU reads A (section pages live)", &buf, a as u64, 64 * KIB, false, 1, 0);
                        if ok && map_into(&s, p.p, b, 64 * KIB) {
                            fill(p.p, 64 * KIB, 2);
                            rig.verify("3c alias remapped to slice B, written: GPU reads B at B's offset", &buf, b as u64, 64 * KIB, false, 2, 0);
                            rig.verify("3c slice A unchanged", &buf, a as u64, 64 * KIB, false, 1, 0);
                        }
                    }
                }
                Err(e) => line(false, "3c open", &hr(&e)),
            }
        }
    }

    // d. heap over the placeholder view itself (only if that opens).
    if let Some(rig) = fresh(agility, sh, "3d")
        && let Ok(s) = Section::new(64 * MIB, win::SEC_COMMIT)
        && let Some(p) = s.slice(MIB, 64 * KIB)
    {
        fill(p.p, 64 * KIB, 1);
        match open_buffer(&rig, p.p, 64 * KIB) {
            Ok((_h, buf)) => {
                rig.verify("3d heap over a placeholder view: GPU reads it", &buf, 0, 64 * KIB, false, 1, 0);
                let ok = unmap_keep(p.p);
                info("3d UnmapViewOfFile2 under the heap", if ok { "ok" } else { "failed" });
                rig.verify("3d after unmap (expects old pattern 1)", &buf, 0, 64 * KIB, false, 1, 0);
                if ok && map_into(&s, p.p, 2 * MIB, 64 * KIB) {
                    fill(p.p, 64 * KIB, 2);
                    rig.verify("3d remapped to another slice, written: GPU sees new", &buf, 0, 64 * KIB, false, 2, 0);
                }
                drop(buf);
                drop(_h);
                let ok = unmap_keep(p.p);
                info("3d control: the same unmap after the heap is released", if ok { "ok" } else { "failed" });
            }
            Err(e) => info("3d heap over a placeholder view", &format!("not opened {} (see 2)", hr(&e))),
        }
    }

    // e. release the memory while the heap lives.
    if let Some(rig) = fresh(agility, sh, "3e")
        && let Some(mut blk) = Block::new(size, 0)
    {
        fill(blk.p, size, 1);
        match open_buffer(&rig, blk.p, size) {
            Ok((h, buf)) => {
                // SAFETY: releases our block; the Block no longer owns it.
                let ok = unsafe { win::VirtualFree(blk.p.cast(), 0, win::MEM_RELEASE) } != 0;
                if ok {
                    blk.p = std::ptr::null_mut();
                }
                info("3e VirtualFree(MEM_RELEASE) under the heap", &if ok { "succeeded".into() } else { format!("refused: {}", win::last_error()) });
                rig.verify("3e GPU read after release (expects pattern 1)", &buf, 0, size, false, 1, 0);
                drop(buf);
                drop(h);
                info("3e heap released", &rig.removed());
                if !ok {
                    // SAFETY: releases our block.
                    let again = unsafe { win::VirtualFree(blk.p.cast(), 0, win::MEM_RELEASE) } != 0;
                    if again {
                        blk.p = std::ptr::null_mut();
                    }
                    info("3e control: the same release after the heap is released", if again { "succeeded" } else { "refused" });
                }
            }
            Err(e) => line(false, "3e open", &hr(&e)),
        }
    }
}

fn step4(rig: &Rig) {
    println!("\n-- 4 cost");
    for (size, touch) in [(MIB, true), (64 * MIB, true), (GIB, true), (GIB, false)] {
        let Some(blk) = Block::new(size, 0) else {
            line(false, "4 VirtualAlloc", &win::last_error().to_string());
            continue;
        };
        if touch {
            fill(blk.p, size, 1);
        }
        let ws0 = working_set();
        let t0 = Instant::now();
        let heap = rig.open_address(blk.p);
        let t1 = Instant::now();
        let ws = (working_set() as i64 - ws0 as i64) / MIB as i64;
        let what = format!("4 open + place {} MiB ({})", size / MIB, if touch { "written" } else { "never touched" });
        match heap {
            Ok(h) => match rig.place(&h, 0, size as u64, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER) {
                Ok(b) => {
                    let t2 = Instant::now();
                    drop(b);
                    drop(h);
                    let t3 = Instant::now();
                    line(true, &what, &format!("open {:.2} ms (working set {ws:+} MiB), place {:.3} ms, release {:.2} ms", ms(t1 - t0), ms(t2 - t1), ms(t3 - t2)));
                }
                Err(e) => line(false, &what, &format!("place {}", hr(&e))),
            },
            Err(e) => line(false, &what, &format!("open {}", hr(&e))),
        }
    }

    let size = 256 * MIB;
    let Some(blk) = Block::new(size, 0) else { return };
    fill(blk.p, size, 1);
    let shared = match open_buffer(rig, blk.p, size) {
        Ok(s) => s,
        Err(e) => return line(false, "4 shared 256 MiB", &hr(&e)),
    };
    let built = (|| -> Result<(ID3D12Resource, ID3D12Resource)> {
        let default = rig.host.buffer(size as u64, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON)?;
        rig.copy(&default, &shared.1, size as u64)?;
        let upload = rig.host.buffer(size as u64, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: the upload buffer maps `size` bytes.
        unsafe {
            upload.Map(0, None, Some(&mut ptr))?;
            std::ptr::copy_nonoverlapping(blk.p, ptr.cast::<u8>(), size);
            upload.Unmap(0, None);
        }
        Ok((default, upload))
    })();
    let (default, upload) = match built {
        Ok(b) => b,
        Err(e) => return line(false, "4 default/upload buffers", &hr(&e)),
    };
    let want = expected_sum(0, size, 1);
    for round in 1..=2 {
        let mut parts = Vec::new();
        let mut ok = true;
        for (name, buf) in [("shared (system memory, opened heap)", &shared.1), ("default heap (VRAM)", &default), ("upload heap", &upload)] {
            let _ = rig.read(buf, 0, size as u64, false, 1024); // warm
            match rig.read(buf, 0, size as u64, false, 1024) {
                Ok(r) => {
                    ok &= r.sum == want;
                    parts.push(format!("{name} {:.0} MB/s{}", size as f64 / 1e6 / r.seconds, if r.sum == want { "" } else { " (WRONG SUM)" }));
                }
                Err(e) => {
                    ok = false;
                    parts.push(format!("{name} failed {}", hr(&e)));
                }
            }
        }
        line(ok, &format!("4 compute read of 256 MiB, round {round}"), &parts.join("; "));
    }
}

fn working_set() -> usize {
    let mut c = win::ProcessMemoryCounters { cb: size_of::<win::ProcessMemoryCounters>() as u32, ..Default::default() };
    // SAFETY: `c` is the structure and its size is passed.
    unsafe { win::K32GetProcessMemoryInfo(win::GetCurrentProcess(), &mut c, c.cb) };
    c.working_set_size
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn step5(rig: &Rig) {
    println!("\n-- 5 textures on an opened heap (VirtualAlloc 64 MiB)");
    let size = 64 * MIB;
    let Some(blk) = Block::new(size, 0) else { return };
    // SAFETY: zeroing our block.
    unsafe { std::ptr::write_bytes(blk.p, 0, size) };
    let heap = match rig.open_address(blk.p) {
        Ok(h) => h,
        Err(e) => return line(false, "5 open", &hr(&e)),
    };
    let unknown = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    let row = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    let cross = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
    let none = D3D12_RESOURCE_FLAG_NONE;
    let mut rgba_row = None;
    for (name, format, mips, layout, flags) in [
        ("RGBA8 1024^2 11 mips, layout UNKNOWN", DXGI_FORMAT_R8G8B8A8_UNORM, 11, unknown, none),
        ("RGBA8 1024^2 11 mips, layout UNKNOWN, CROSS_ADAPTER", DXGI_FORMAT_R8G8B8A8_UNORM, 11, unknown, cross),
        ("RGBA8 1024^2 11 mips, ROW_MAJOR, CROSS_ADAPTER", DXGI_FORMAT_R8G8B8A8_UNORM, 11, row, cross),
        ("RGBA8 1024^2 1 mip, ROW_MAJOR, CROSS_ADAPTER", DXGI_FORMAT_R8G8B8A8_UNORM, 1, row, cross),
        ("BC1 1024^2 11 mips, layout UNKNOWN", DXGI_FORMAT_BC1_UNORM, 11, unknown, none),
        ("BC1 1024^2 11 mips, layout UNKNOWN, CROSS_ADAPTER", DXGI_FORMAT_BC1_UNORM, 11, unknown, cross),
        ("BC1 1024^2 1 mip, ROW_MAJOR, CROSS_ADAPTER", DXGI_FORMAT_BC1_UNORM, 1, row, cross),
    ] {
        let mut out: Option<ID3D12Resource> = None;
        let desc = texture_desc(format, mips, layout, flags);
        // SAFETY: the description outlives the call.
        let r = unsafe { rig.host.device.CreatePlacedResource(&heap, 0, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut out) };
        match r {
            Ok(()) => {
                line(true, &format!("5 CreatePlacedResource {name}"), "ok");
                if format == DXGI_FORMAT_R8G8B8A8_UNORM && mips == 1 {
                    rgba_row = out;
                }
            }
            Err(e) => line(false, &format!("5 CreatePlacedResource {name}"), &hr(&e)),
        }
    }
    let Some(tex) = rgba_row else { return };
    // Texel (1, 0) of a row-major texture: byte 4.
    // SAFETY: inside the block.
    unsafe { blk.p.add(4).cast::<[u8; 4]>().write([10, 20, 30, 40]) };
    let sampled = (|| -> Result<[u32; 4]> {
        // SAFETY: COM calls; `out_ptr` maps OUT_BYTES.
        unsafe {
            std::ptr::write_bytes(rig.out_ptr.cast::<u8>(), 0, OUT_BYTES as usize);
            let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1, PlaneSlice: 0, ResourceMinLODClamp: 0.0 } },
            };
            rig.host.device.CreateShaderResourceView(&tex, Some(&srv), rig.heap.GetCPUDescriptorHandleForHeapStart());
            let (_alloc, list) = rig.list(&rig.tex)?;
            list.SetDescriptorHeaps(&[Some(rig.heap.clone())]);
            list.SetComputeRootSignature(&rig.root);
            list.SetComputeRootDescriptorTable(0, rig.heap.GetGPUDescriptorHandleForHeapStart());
            list.SetComputeRootConstantBufferView(1, rig.out.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2, rig.out.GetGPUVirtualAddress());
            list.Dispatch(1, 1, 1);
            rig.submit(&list)?;
            let o = std::slice::from_raw_parts(rig.out_ptr, 8);
            Ok([o[4], o[5], o[6], o[7]])
        }
    })();
    match sampled {
        Ok(t) => line(t == [10, 20, 30, 40], "5 sample RGBA8 row-major texel (1,0) written by the CPU", &format!("{t:?} (want [10, 20, 30, 40]); {}", rig.removed())),
        Err(e) => line(false, "5 sample RGBA8 row-major texture", &format!("{} ({})", hr(&e), rig.removed())),
    }
}

/// Maps `size` bytes of `s` at section `offset` at an address of the system's choosing.
fn window(s: &Section, offset: usize, size: usize) -> Option<View> {
    // SAFETY: a live section handle.
    let p = unsafe { win::MapViewOfFile(s.0, win::FILE_MAP_WRITE, (offset as u64 >> 32) as u32, offset as u32, size) };
    (!p.is_null()).then_some(View { p: p.cast(), placeholder: false })
}

/// The design candidate: one runtime-owned view A of the physical-page
/// section with heaps over it; the title's alias views B change freely.
fn step6(rig: &Rig) {
    println!("
-- 6 runtime-owned view A with a heap; title alias B of the same pages (16 GiB SEC_RESERVE section)");
    let s = match Section::new(16 * GIB, win::SEC_RESERVE) {
        Ok(s) => s,
        Err(e) => return line(false, "6 section", &e),
    };
    let win_size = 64 * MIB;
    let (a_off, c_off) = (GIB + MIB, GIB + 2 * MIB);
    let Some(a) = window(&s, GIB, win_size) else { return line(false, "6 view A", &win::last_error().to_string()) };
    if !commit(a.p, win_size) {
        return line(false, "6 commit A", &win::last_error().to_string());
    }
    let heap = match rig.open_address(a.p) {
        Ok(h) => h,
        Err(e) => return line(false, "6 heap over A", &hr(&e)),
    };
    let place = |off: usize| rig.place(&heap, (off - GIB) as u64, 64 * KIB as u64, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER);
    let (Ok(buf_b), Ok(buf_c)) = (place(a_off), place(c_off)) else { return line(false, "6 place", "failed") };
    let Some(b) = s.slice(a_off, 64 * KIB) else { return line(false, "6 alias B", &win::last_error().to_string()) };
    fill(b.p, 64 * KIB, 1);
    rig.verify("6 write through B, GPU reads through A at once", &buf_b, 0, 64 * KIB, true, 1, 0);
    let ro = protect(b.p, 64 * KIB, win::PAGE_READONLY);
    line(ro, "6a VirtualProtect(B, PAGE_READONLY) while the heap over A lives", if ro { "ok" } else { "refused" });
    rig.verify("6a GPU read through A while B is read-only", &buf_b, 0, 64 * KIB, true, 1, 0);
    let rw = protect(b.p, 64 * KIB, win::PAGE_READWRITE);
    line(rw, "6a VirtualProtect(B, PAGE_READWRITE) back", if rw { "ok" } else { "refused" });
    fill(b.p, 64 * KIB, 2);
    rig.verify("6a write through B after re-protect, GPU sees it through A", &buf_b, 0, 64 * KIB, true, 2, 0);
    let un = unmap_keep(b.p);
    line(un, "6b UnmapViewOfFile2(B, MEM_PRESERVE_PLACEHOLDER)", if un { "ok" } else { "failed" });
    rig.verify("6b after unmap of B, GPU still reads the slice through A", &buf_b, 0, 64 * KIB, true, 2, 0);
    let re = un && map_into(&s, b.p, c_off, 64 * KIB);
    line(re, "6b MapViewOfFile3(B -> another slice)", if re { "ok" } else { "failed" });
    if re {
        fill(b.p, 64 * KIB, 3);
        rig.verify("6b write through remapped B, GPU reads new slice through A", &buf_c, 0, 64 * KIB, true, 3, 0);
        rig.verify("6b old slice unchanged through A", &buf_b, 0, 64 * KIB, true, 2, 0);
    }
    // GPU writes: a UAV buffer on the same heap, written by a compute
    // shader; the title's alias B (now on slice C) reads the result.
    let what = "6f CreatePlacedResource ALLOW_UNORDERED_ACCESS|ALLOW_CROSS_ADAPTER on the window heap";
    match rig.place(&heap, (c_off - GIB) as u64, 64 * KIB as u64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER) {
        Ok(uav) => {
            line(true, what, "ok");
            if re {
                // SAFETY: a live buffer.
                let va = unsafe { uav.GetGPUVirtualAddress() };
                match rig.read_to(&buf_b, 0, 64 * KIB as u64, false, 64, Some(va)) {
                    Ok(_) => {
                        // B shows slice C: word 0 = slice C's pattern-3 word 0 + the sum, 4..8 = slice A's first words.
                        // SAFETY: B is mapped read-write.
                        let w = unsafe { std::slice::from_raw_parts(b.p.cast::<u32>(), 8) };
                        let want_sum = pattern(0, 3).wrapping_add(expected_sum(0, 64 * KIB, 2));
                        let want_c0 = [pattern(0, 2), pattern(1, 2), pattern(2, 2), pattern(3, 2)];
                        let ok = w[0] == want_sum && w[4..8] == want_c0;
                        line(ok, "6f compute shader writes through the UAV land in title memory (read through alias B)", &format!("word0 {:#x} want {want_sum:#x}, words 4..8 {:x?} want {want_c0:x?}; {}", w[0], &w[4..8], rig.removed()));
                    }
                    Err(e) => line(false, "6f compute write to the shared UAV", &format!("{} ({})", hr(&e), rig.removed())),
                }
            }
        }
        Err(e) => line(false, what, &hr(&e)),
    }
    drop((buf_b, buf_c, heap));

    // Commit cost: a window half committed.
    if let Some(v) = window(&s, 2 * GIB, win_size) {
        commit(v.p, win_size / 2);
        match rig.open_address(v.p) {
            Ok(h) => line(true, "6c heap over a half-committed SEC_RESERVE window", &heap_desc(&h)),
            Err(e) => line(false, "6c heap over a half-committed SEC_RESERVE window", &hr(&e)),
        }
    }

    // Adjacent windows, one heap each.
    let (w0, w1) = (window(&s, 3 * GIB, win_size), window(&s, 3 * GIB + win_size, win_size));
    if let (Some(w0), Some(w1)) = (w0, w1)
        && commit(w0.p, win_size)
        && commit(w1.p, win_size)
    {
        fill(w0.p, win_size, 1);
        fill(w1.p, win_size, 2);
        match (open_buffer(rig, w0.p, win_size), open_buffer(rig, w1.p, win_size)) {
            (Ok((_h0, b0)), Ok((_h1, b1))) => {
                let ok0 = rig.verify("6d adjacent window 0 heap", &b0, 0, win_size, false, 1, 0);
                let ok1 = rig.verify("6d adjacent window 1 heap", &b1, 0, win_size, false, 2, 0);
                line(ok0 && ok1, "6d two heaps over adjacent 64 MiB windows", "both read");
            }
            (r0, r1) => line(false, "6d adjacent windows", &format!("{:?} {:?}", r0.err().map(|e| hr(&e)), r1.err().map(|e| hr(&e)))),
        }
    }

    // Largest window: committed (charge only), then opened; working set shows residency.
    for size in [GIB, 2 * GIB, 4 * GIB - 64 * KIB, 4 * GIB, 8 * GIB] {
        let what = format!("6e heap over a committed {} MiB window", size / MIB);
        let Some(v) = window(&s, 4 * GIB, size) else {
            line(false, &what, &format!("MapViewOfFile: {}", win::last_error()));
            continue;
        };
        if !commit(v.p, size) {
            line(false, &what, &format!("commit: {}", win::last_error()));
            continue;
        }
        let ws0 = working_set();
        let t0 = Instant::now();
        let r = rig.open_address(v.p);
        let t = ms(t0.elapsed());
        let ws = (working_set() as i64 - ws0 as i64) / MIB as i64;
        match r {
            Ok(h) => {
                let placed = rig.place(&h, 0, size as u64, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER).map(|_| ()).map_err(|e| hr(&e));
                line(true, &what, &format!("open {t:.0} ms, working set {ws:+} MiB, one buffer over all of it: {placed:?}"));
            }
            Err(e) => line(false, &what, &format!("{} after {t:.0} ms", hr(&e))),
        }
        // Decommit is not possible on section views; the view goes with `v`.
    }
}

#[cfg(test)]
mod zero_copy_tests {
    use super::*;

    /// A copied range splits into its 64 KiB pages with its bytes in
    /// each (partial pages at both ends).
    #[test]
    fn ranges_split_into_pages() {
        assert_eq!(zc_pages(0x1_0000, 0x1_0000), vec![(0x1_0000, 0x1_0000)]);
        assert_eq!(zc_pages(0x1_fff0, 0x20), vec![(0x1_0000, 0x10), (0x2_0000, 0x10)]);
        let p = zc_pages(0x3_8000, 0x2_0000);
        assert_eq!(p, vec![(0x3_0000, 0x8000), (0x4_0000, 0x1_0000), (0x5_0000, 0x8000)]);
        assert_eq!(p.iter().map(|x| x.1).sum::<usize>(), 0x2_0000, "every byte in one page");
        assert!(zc_pages(0x1000, 0).is_empty());
    }

    /// Ranges wholly in the page pool or an arena are the pools'; a
    /// range reaching past either is not; pages classify by allocator.
    #[test]
    fn pool_and_arena_ranges_classify() {
        *ALLOCATORS.lock().unwrap() = Some(Allocators { pool: Some((0x1000_0000, 8 << 20)), arenas: vec![(0x2000_0000, 1 << 20)] });
        assert!(in_pools(0x1000_0000, 256 << 10));
        assert!(in_pools(0x1000_0000 + (8 << 20) - 16, 16));
        assert!(!in_pools(0x1000_0000 + (8 << 20) - 16, 32), "past the pool's end");
        assert!(in_pools(0x2000_0100, 0x100));
        assert!(!in_pools(0x3000_0000, 16));
        assert_eq!(alloc_class(0x1040_0000), AL_POOL);
        assert_eq!(alloc_class(0x2000_8000), AL_ARENA);
        *ALLOCATORS.lock().unwrap() = None;
    }

    /// A page's class from how it resolves.
    #[test]
    fn pages_classify_by_resolution() {
        let ok: Resolved = Ok(((1, 2), 0));
        assert_eq!(zc_class(ok, true), ZC_OPEN);
        assert_eq!(zc_class(ok, false), ZC_OPENABLE);
        assert_eq!(zc_class(Err(Miss::Opening), false), ZC_OPEN, "opening: open once its thread is done");
        assert_eq!(zc_class(Err(Miss::TooLarge), false), ZC_LARGE);
        assert_eq!(zc_class(Err(Miss::PartlyMapped), false), ZC_UNMAPPED);
        assert_eq!(zc_class(Err(Miss::Uncommitted), false), ZC_UNMAPPED);
        for m in [Miss::Unknown, Miss::PastRegion, Miss::NotContiguous, Miss::Failed] {
            assert_eq!(zc_class(Err(m), false), ZC_UNKNOWN);
        }
    }
}