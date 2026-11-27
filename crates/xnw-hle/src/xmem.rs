//! `xmem.dll`: console memory, served from host memory.
//!
//! Adapted from VIctory, `crates/victory-hle/src/modules/xmem.rs` at d2a4bae
//! (GPL-3.0-or-later). The GPU side works from CPU addresses (unified
//! memory), so memory types and bandwidth classes change nothing here; what
//! constrains addresses (alignment, 2 MiB pages, address-bit limits) is
//! honoured.

use std::alloc::{Layout, alloc, dealloc};
use std::ffi::c_void;
use std::sync::OnceLock;

use xnw_core::win::*;

const MIB: usize = 1 << 20;

// Console additions to the host's allocation flags.
/// 2 MiB pages; the host would read this bit as large pages.
const PAGES_2M: u32 = 0x2000_0000;
/// With `PAGES_2M`: 64 KiB pages; the host would read it as AWE memory.
const PAGES_64K_BIT: u32 = 0x40_0000;
const PAGES_64K: u32 = PAGES_2M | PAGES_64K_BIT;
/// The host's protection constants and `PAGE_GUARD`; above them the console
/// adds GPU access bits.
const HOST_PROTECTION: u32 = 0x1ff;
/// XMem flags of memory both processors reach: all of ours.
const GRAPHICS: u64 = 3;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for
/// 32-bit memory; else from 4 GiB up to 46 down to 40 bits (field at bit
/// 12), or 40 bits without one, where the console keeps title memory.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field,
        _ => return (0, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

pub extern "system" fn XMemVirtualAlloc(base: *mut c_void, size: usize, kind: u32, flags: u64, protect: u32) -> *mut c_void {
    let asked = protect;
    let (lowest, highest) = address_range(flags);
    let mut requirements = MemAddressRequirements {
        lowest,
        highest,
        alignment: if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M { 2 * MIB } else { 0 },
    };
    let mut protect = match protect & HOST_PROTECTION {
        0 => PAGE_NOACCESS,
        p => p,
    };
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if placeholder {
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            // Part of a placeholder the title split: it stays one.
            return base;
        }
        protect = PAGE_NOACCESS;
    }
    let mut host_kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    if placeholder {
        host_kind |= MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
    }
    let mut parameter = MemExtendedParameter { kind: 1, pointer: &mut requirements };
    // Address requirements apply to reservations at an address of our choice.
    let constrained = base.is_null();
    if constrained {
        host_kind |= MEM_RESERVE;
        // Lets the GPU side copy only pages written since its last copy;
        // placeholders cannot be watched.
        if host_kind & MEM_RESERVE_PLACEHOLDER == 0 {
            host_kind |= MEM_WRITE_WATCH;
        }
    }
    let started = xnw_core::perf::start();
    // SAFETY: the title's own arguments, cleaned of console bits; the
    // parameter points at `requirements`, which outlives the call.
    let mut call = |len: usize| unsafe {
        virtual_alloc2()(
            GetCurrentProcess(),
            base,
            len,
            host_kind,
            protect,
            if constrained { &mut parameter } else { std::ptr::null_mut() },
            constrained.into(),
        )
    };
    // GRP1: placement and pairing of reservations at an address of our
    // choice, one at a time (the pair's two calls must meet nothing between).
    let _placing = constrained.then(|| lock(&PLACING));
    // A commit inside one of the title's regions is counted there.
    let at = if constrained {
        place(size, kind, asked, placeholder, &mut call)
    } else {
        tracked(base as usize, size, || call(size))
    };
    if !at.is_null() && !placeholder && constrained {
        let committed = if host_kind & MEM_COMMIT != 0 { committed_in(at as usize, at as usize + size) } else { 0 };
        lock(&REGIONS).insert(at as usize, (size, committed));
        COMMITTED.fetch_add(committed, Ordering::Relaxed);
    }
    xnw_core::perf::since(xnw_core::perf::Area::XmemAlloc, started, size as u64);
    if base.is_null() {
        pair_reservation(at as usize, size, kind, asked, placeholder);
    }
    if base.is_null() || at.is_null() {
        xnw_core::info!(
            "XMemVirtualAlloc({base:p}, {size:#x}, kind {kind:#x}, flags {flags:#x}, protect {asked:#x}) -> {at:p}{}",
            if at.is_null() { format!("  FAILED: {}", last_error()) } else { String::new() }
        );
    }
    at
}

/// A run's reservation made again at the address its log names (the
/// burst's stand-in for title memory, ZC1): as [`XMemVirtualAlloc`] with
/// that base, then kept as a reservation of our choosing is (a region, or a
/// placeholder that may pair). False if the address is taken.
pub fn replay_reservation(at: usize, size: usize, kind: u32, flags: u64, protect: u32) -> bool {
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if XMemVirtualAlloc(at as *mut c_void, size, kind | MEM_RESERVE, flags, protect) as usize != at {
        return false;
    }
    if !placeholder {
        let committed = if kind & MEM_COMMIT != 0 { committed_in(at, at + size) } else { 0 };
        lock(&REGIONS).insert(at, (size, committed));
        COMMITTED.fetch_add(committed, Ordering::Relaxed);
    }
    pair_reservation(at, size, kind, protect, placeholder);
    true
}

/// `MEMORY_BASIC_INFORMATION`, then the XMem flags and tag.
#[repr(C)]
pub struct BasicInformation {
    host: MemoryBasicInformation,
    flags: u64,
    tag_code: u32,
    tag: [u8; 4],
}

// shortcut (VIctory): every committed page reports graphics memory and no tag.
pub extern "system" fn XMemVirtualQuery(process: HANDLE, address: *const c_void, out: *mut BasicInformation) -> i32 {
    // SAFETY: the title passes room for the whole structure.
    unsafe {
        if xnw_core::syscalls::timed(xnw_core::syscalls::Call::QueryTitle, || VirtualQueryEx(process, address, out.cast(), size_of::<MemoryBasicInformation>())) == 0 {
            return 0;
        }
        let info = &mut *out;
        info.flags = if info.host.state == MEM_COMMIT { GRAPHICS } else { 0 };
        info.tag_code = 0;
        info.tag = [0; 4];
    }
    1
}

// `XMemAlloc` attributes: alignment as log2 (5 bits at 24), then the memory
// type, then bit 33: a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_32BIT: u64 = 1 << 33;
/// Our block header: the block's size and alignment, just below the pointer
/// handed out.
const HEADER: usize = 2 * size_of::<usize>();

fn layout(size: usize, attributes: u64) -> Option<Layout> {
    // An alignment of at least 16 leaves room for the header.
    let align = 1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f).max(4);
    Layout::from_size_align(size.checked_add(align)?, align).ok()
}

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The routines the title's exe exports as `__XMemAllocationHooks__`:
/// `XMemAlloc` and `XMemFree` call them (for the title's own calls and its
/// middleware's); the `...Default` exports do not.
fn hooks() -> [usize; 2] {
    static AT: OnceLock<usize> = OnceLock::new();
    let at = *AT.get_or_init(|| {
        let base = crate::title_base();
        if base == 0 {
            return 0;
        }
        // SAFETY: a mapped image and a NUL-terminated name.
        let at = unsafe { GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast()) } as usize;
        xnw_core::info!("xmem allocation hooks: {}", if at != 0 { "exported by the title" } else { "none" });
        at
    });
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    match hooks()[0] {
        0 => XMemAllocDefault(size, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let f: AllocRoutine = unsafe { std::mem::transmute(hook) };
            f(size, attributes)
        }
    }
}

pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let f: FreeRoutine = unsafe { std::mem::transmute(hook) };
            f(address, attributes)
        }
    }
}

/// The system allocator behind [`XMemAlloc`].
pub extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    if attributes & ALLOC_32BIT != 0 {
        return XMemVirtualAlloc(std::ptr::null_mut(), size, MEM_RESERVE | MEM_COMMIT, GRAPHICS | ADDRESS_32BIT, PAGE_READWRITE);
    }
    let Some(layout) = layout(size, attributes) else { return std::ptr::null_mut() };
    // shortcut (VIctory): host heap addresses lie above 2^40, unlike the
    // console's; nothing seen needs these blocks below it.
    // SAFETY: the layout has a non-zero size.
    let block = unsafe { alloc(layout) };
    if block.is_null() {
        return block.cast();
    }
    HEAP_USED.fetch_add(layout.size(), Ordering::Relaxed);
    // SAFETY: the block holds the alignment (at least the header), then `size` bytes.
    unsafe {
        let user = block.add(layout.align());
        user.sub(HEADER).cast::<[usize; 2]>().write([layout.size(), layout.align()]);
        user.cast()
    }
}

/// Frees what [`XMemAllocDefault`] returned, given the same attributes.
pub extern "system" fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    // SAFETY: the address came from `XMemAllocDefault` with these
    // attributes: a whole reservation, or a header-prefixed block.
    unsafe {
        if attributes & ALLOC_32BIT != 0 {
            release(address, 0, MEM_RELEASE);
            return;
        }
        let user = address.cast::<u8>();
        let [size, align] = user.sub(HEADER).cast::<[usize; 2]>().read();
        HEAP_USED.fetch_sub(size, Ordering::Relaxed);
        dealloc(user.sub(align), Layout::from_size_align_unchecked(size, align));
    }
}

/// A retail console has no tools memory.
pub extern "system" fn XMemIsToolsMemAvailable() -> i32 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_blocks_are_aligned_and_freed() {
        for log2 in [0u64, 4, 6, 12, 16] {
            let attributes = log2 << ALIGNMENT_SHIFT;
            let p = XMemAllocDefault(100, attributes);
            assert!(!p.is_null());
            assert_eq!(p as usize % (1usize << log2.max(4)), 0);
            // SAFETY: 100 bytes were asked for.
            unsafe { std::ptr::write_bytes(p.cast::<u8>(), 0xAB, 100) };
            XMemFreeDefault(p, attributes);
        }
    }

    #[test]
    fn reservations_honour_the_address_limits() {
        // 64 KiB-page reservation: a placeholder below 2^40, above 4 GiB.
        let p = XMemVirtualAlloc(std::ptr::null_mut(), 64 << 20, MEM_RESERVE | PAGES_64K, 0x21, PAGE_READWRITE) as usize;
        assert!(p >= 1 << 32 && p < 1 << 40, "{p:#x}");
        // 32-bit graphics memory, committed.
        let q = XMemVirtualAlloc(std::ptr::null_mut(), 1 << 16, MEM_RESERVE | MEM_COMMIT, GRAPHICS | ADDRESS_32BIT, PAGE_READWRITE) as usize;
        assert!(q != 0 && q <= u32::MAX as usize, "{q:#x}");
        // SAFETY: committed read-write memory of our own.
        unsafe { (q as *mut u32).write(7) };
    }
}

// Physical pages: memory titles map and remap at addresses they reserved
// (a 64 KiB-page placeholder, see `XMemVirtualAlloc`). An allocation is a
// pagefile-backed section, its pages are 64 KiB slices of it, and a mapping
// is a view of consecutive slices in place of a host placeholder. Ported
// from VIctory `modules/xmem.rs` (d2a4bae, GPL-3.0-or-later).

const PAGE: usize = 64 << 10;
/// Page numbers count 4 KiB frames, so a 64 KiB page's is a multiple of 16:
/// the title keeps them as 21-bit `number >> 4` and rebuilds them with `<< 4`.
const FRAMES: usize = PAGE >> 12;
const ERROR_COMMITMENT_LIMIT: i32 = 1455;

/// A section and the page number of its first page (its pages' numbers
/// follow in steps of [`FRAMES`]).
struct Section(HANDLE, usize);

// SAFETY: a section handle may be used from any thread.
unsafe impl Send for Section {}
// SAFETY: as above.
unsafe impl Sync for Section {}

impl Drop for Section {
    fn drop(&mut self) {
        // SAFETY: our handle; views keep the section alive on their own.
        unsafe { CloseHandle(self.0) };
    }
}

/// Where a page lives, or what a view starts with: a section and a page
/// index in it.
#[derive(Clone)]
struct Slice {
    section: std::sync::Arc<Section>,
    index: usize,
}

struct Physical {
    next_page: usize,
    pages: std::collections::HashMap<usize, Slice>,
    /// Mapped views by address: their length and what they show.
    views: std::collections::BTreeMap<usize, (usize, Slice)>,
    pair: Option<Pair>,
    /// Bytes mapped in the pair's CPU view and in its GPU view.
    mapped: [usize; 2],
    /// Physical-page mappings repeated in the GPU view.
    mirrored: usize,
    /// Live sections by key (`Arc` address): the section, its size and its
    /// pages not yet freed.
    sections: std::collections::HashMap<usize, (std::sync::Arc<Section>, usize, usize)>,
    /// The runtime's own views of sections, by (section key, window index),
    /// for the GPU to read in place (`xnw_gpu::shared`); kept while the
    /// section lives.
    windows: std::collections::HashMap<(usize, usize), View>,
}

static PHYSICAL: std::sync::Mutex<Option<Physical>> = std::sync::Mutex::new(None);

fn with_physical<R>(f: impl FnOnce(&mut Physical) -> R) -> R {
    let mut guard = PHYSICAL.lock().unwrap_or_else(|e| e.into_inner());
    // Page numbers are opaque to titles; 0 stays unused.
    let physical = guard.get_or_insert_with(|| Physical {
        next_page: 0x10_0000,
        pages: Default::default(),
        views: Default::default(),
        pair: None,
        mapped: [0; 2],
        mirrored: 0,
        sections: Default::default(),
        windows: Default::default(),
    });
    f(physical)
}

/// `count` is in and out (pages wanted, pages given); all or nothing here.
pub extern "system" fn XMemAllocatePhysicalPages(flags: u64, count: *mut usize, pages: *mut usize) -> i32 {
    if count.is_null() || pages.is_null() {
        return 0;
    }
    // SAFETY: checked non-null.
    let n = unsafe { *count };
    let Some(size) = n.checked_mul(PAGE).filter(|&s| s > 0) else {
        return 0;
    };
    xnw_core::perf::time(xnw_core::perf::Area::XmemPhysical, size as u64, || allocate_physical(flags, n, size, count, pages))
}

fn allocate_physical(flags: u64, n: usize, size: usize, count: *mut usize, pages: *mut usize) -> i32 {
    // Reserved, committed when first mapped or written (`commit`): the title
    // asks for several GiB at once, more than a busy host may commit.
    // SAFETY: plain section creation.
    let handle = unsafe {
        CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE | SEC_RESERVE, (size as u64 >> 32) as u32, size as u32, std::ptr::null())
    };
    if handle.is_null() {
        xnw_core::warn!("XMemAllocatePhysicalPages({n} pages, flags {flags:#x}) failed: {}", last_error());
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
        return 0;
    }
    xnw_core::info!("XMemAllocatePhysicalPages({n} pages = {} MiB, flags {flags:#x})", size / MIB);
    PHYSICAL_BYTES.fetch_add(size, Ordering::Relaxed);
    with_physical(|physical| {
        let first = physical.next_page;
        let section = std::sync::Arc::new(Section(handle, first));
        physical.sections.insert(section_key(&section), (section.clone(), size, n));
        physical.next_page += n * FRAMES;
        for index in 0..n {
            let number = first + index * FRAMES;
            physical.pages.insert(number, Slice { section: section.clone(), index });
            // SAFETY: the title passes room for `n` page numbers.
            unsafe { pages.add(index).write(number) };
        }
    });
    1
}

/// Pages still mapped stay readable through their views until unmapped.
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> i32 {
    if pages.is_null() {
        return 0;
    }
    // Sections left without a page, and their windows: let go outside the
    // lock (a heap release may wait for a frame of the world pass, whose
    // lookups take this lock: ZC2).
    let mut sections_gone = Vec::new();
    let mut windows_gone = Vec::new();
    let freed = with_physical(|physical| {
        let mut freed = 0;
        for i in 0..n {
            // SAFETY: the title passes `n` page numbers.
            let Some(slice) = physical.pages.remove(&unsafe { *pages.add(i) }) else { continue };
            freed += 1;
            let key = section_key(&slice.section);
            let Some(live) = physical.sections.get_mut(&key).map(|s| &mut s.2) else { continue };
            *live -= 1;
            if *live == 0 {
                // No page left: the GPU's heaps over it, then our windows go.
                physical.sections.remove(&key);
                sections_gone.push(key);
                let gone: Vec<(usize, usize)> = physical.windows.keys().filter(|k| k.0 == key).copied().collect();
                for k in gone {
                    windows_gone.extend(physical.windows.remove(&k));
                }
            }
        }
        freed
    });
    for key in sections_gone {
        xnw_gpu::shared::section_gone(key);
    }
    drop(windows_gone);
    // SAFETY: as above.
    xnw_formats::origin::registry().forget_frames(unsafe { std::slice::from_raw_parts(pages, n) });
    PHYSICAL_BYTES.fetch_sub(freed * PAGE, Ordering::Relaxed);
    1
}

// SAFETY: a view is plain address space of ours; any thread may unmap it.
unsafe impl Send for View {}

/// A view of our own on consecutive slices of one section, committed and
/// writable, unmapped when dropped.
struct View {
    at: *mut c_void,
    len: usize,
}

impl View {
    fn map(slice: &Slice, n: usize) -> Option<View> {
        let (offset, len) = ((slice.index * PAGE) as u64, n * PAGE);
        // SAFETY: slices of a section we own; the view is ours until dropped.
        let at = unsafe { MapViewOfFile(slice.section.0, FILE_MAP_WRITE, (offset >> 32) as u32, offset as u32, len) };
        if at.is_null() {
            return None;
        }
        let view = View { at, len };
        commit(at as usize, len).then_some(view)
    }

    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: the view holds `len` committed read-write bytes.
        unsafe { std::slice::from_raw_parts_mut(self.at.cast(), self.len) }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(self.at) };
    }
}

/// The pages behind bytes `[offset, offset + len)` of `pages` (64 KiB each),
/// as runs of consecutive slices of one section: each run's first slice and
/// its length in pages. `None` if a page is unknown or `pages` ends early.
fn runs(pages: &[usize], offset: usize, len: usize) -> Option<Vec<(Slice, usize)>> {
    let end = offset.checked_add(len)?;
    let pages = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    with_physical(|physical| {
        let mut runs: Vec<(Slice, usize)> = Vec::new();
        for page in pages {
            let slice = physical.pages.get(page)?;
            match runs.last_mut() {
                Some((first, n)) if std::sync::Arc::ptr_eq(&first.section, &slice.section) && slice.index == first.index + *n => *n += 1,
                _ => runs.push((slice.clone(), 1)),
            }
        }
        Some(runs)
    })
}

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations).
/// False if a page is unknown or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    xnw_core::perf::time(xnw_core::perf::Area::DsPages, data.len() as u64, || write_runs(pages, offset, data))
}

fn write_runs(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    let Some(runs) = runs(pages, offset, data.len()) else {
        return false;
    };
    let (mut within, mut done) = (offset % PAGE, 0);
    for (slice, n) in &runs {
        let Some(mut view) = View::map(slice, *n) else {
            return false;
        };
        let k = (n * PAGE - within).min(data.len() - done);
        view.bytes()[within..within + k].copy_from_slice(&data[done..done + k]);
        (within, done) = (0, done + k);
    }
    true
}

fn section_key(section: &std::sync::Arc<Section>) -> usize {
    std::sync::Arc::as_ptr(section) as usize
}

/// What is behind `[address, address + len)` for the GPU to read in place
/// (`xnw_gpu::shared`): physical pages contiguous in one section (by
/// section offset, which holds while the title remaps its views), or one of
/// the title's regions, all committed.
pub fn backing(address: usize, len: usize) -> Result<xnw_gpu::shared::Backing, xnw_gpu::shared::Miss> {
    use xnw_gpu::shared::{Backing, Miss};
    let end = address.checked_add(len).ok_or(Miss::Unknown)?;
    let physical = with_physical(|physical| {
        let (&base, (size, slice)) = physical.views.range(..=address).next_back().filter(|(b, (size, _))| address < *b + size)?;
        let key = section_key(&slice.section);
        let offset = slice.index * PAGE + (address - base);
        // Following views must continue the same section where this one ends.
        let (mut at, mut next) = (base + size, slice.index * PAGE + size);
        while at < end {
            match physical.views.get(&at) {
                Some((size, s)) if section_key(&s.section) == key && s.index * PAGE == next => (at, next) = (at + size, next + size),
                Some(_) => return Some(Err(Miss::NotContiguous)),
                None => return Some(Err(Miss::PartlyMapped)),
            }
        }
        Some(Ok(Backing::Physical { section: key, offset: offset as u64 }))
    });
    if let Some(r) = physical {
        return r;
    }
    let regions = lock(&REGIONS);
    let (&base, &(size, committed)) = match regions.range(..=address).next_back() {
        Some((b, (size, _))) if address < b + size && end > b + size => return Err(Miss::PastRegion),
        Some(r) if end <= r.0 + r.1.0 => r,
        _ => return Err(Miss::Unknown),
    };
    if committed < size {
        return Err(Miss::Uncommitted);
    }
    Ok(Backing::Region { base, size })
}

/// The runtime's own view of window `index` (`xnw_gpu::shared::WINDOW`
/// bytes) of section `key`: `(address, len)`, mapped and committed on first
/// use and kept while the section lives. None if the section is gone.
pub fn physical_window(key: usize, index: usize) -> Option<(usize, usize)> {
    // Windows overlap: window `index` starts at `index * stride`.
    const WINDOW: usize = xnw_gpu::shared::WINDOW;
    const STRIDE: usize = xnw_gpu::shared::WINDOW_STRIDE;
    with_physical(|physical| {
        if let Some(v) = physical.windows.get(&(key, index)) {
            return Some((v.at as usize, v.len));
        }
        let (section, size, _) = physical.sections.get(&key)?;
        let from = index * STRIDE;
        let len = size.checked_sub(from).filter(|&l| l > 0)?.min(WINDOW);
        let view = View::map(&Slice { section: section.clone(), index: from / PAGE }, len / PAGE)?;
        let out = (view.at as usize, view.len);
        physical.windows.insert((key, index), view);
        let n = physical.windows.len();
        if n.is_power_of_two() {
            let mib: usize = physical.windows.values().map(|v| v.len).sum::<usize>() / MIB;
            xnw_core::info!("shared memory: {n} page windows mapped ({mib} MiB committed)");
        }
        Some(out)
    })
}

/// Maps `n` pages at `address`, in place of what is mapped there; null
/// `pages` unmaps. Returns `address`, or null if a page or the host fails.
/// In the CPU view of a paired reservation the same happens at the same
/// offset in its GPU view.
pub extern "system" fn XMemMapPhysicalPages(address: *mut c_void, n: usize, pages: *const usize) -> *mut c_void {
    // SAFETY: the title passes `n` page numbers unless null.
    let pages = (!pages.is_null()).then(|| unsafe { std::slice::from_raw_parts(pages, n) });
    xnw_core::perf::time(xnw_core::perf::Area::XmemMap, (n * PAGE) as u64, || map_physical(address, n, pages))
}

fn map_physical(address: *mut c_void, n: usize, pages: Option<&[usize]>) -> *mut c_void {
    let base = address as usize;
    // `XNW_ZEROCOPY` (ZC2): the windows over the pages mapped, asked of the
    // opener thread outside the lock, so a frame finds their heaps open.
    let mut wanted: Vec<(usize, usize)> = Vec::new();
    let out = with_physical(|physical| {
        if xnw_gpu::shared::world_on()
            && let Some(pages) = pages
        {
            for page in pages {
                if let Some(s) = physical.pages.get(page) {
                    let w = (section_key(&s.section), s.index * PAGE / xnw_gpu::shared::WINDOW_STRIDE);
                    if wanted.last() != Some(&w) && !wanted.contains(&w) {
                        wanted.push(w);
                    }
                }
            }
        }
        let done = map_pages(physical, base, n, pages);
        if let Some(gpu) = physical.pair.and_then(|p| p.gpu_of(base, n * PAGE)) {
            // A failed CPU mapping leaves both views empty there.
            if !map_pages(physical, gpu, n, pages.filter(|_| done)) {
                xnw_core::warn!("paired reservation: mirroring {n} pages at {gpu:#x} failed");
            }
            physical.mirrored += 1;
            if physical.mirrored.is_power_of_two() {
                let [cpu, gpu] = physical.mapped.map(|b| b / PAGE);
                xnw_core::info!("paired reservation: {} mappings mirrored; pages mapped: cpu {cpu}, gpu {gpu}", physical.mirrored);
            }
        }
        if done { address } else { std::ptr::null_mut() }
    });
    if !out.is_null() {
        for (section, w) in wanted {
            xnw_gpu::shared::world_mapped(section, w * xnw_gpu::shared::WINDOW_STRIDE, 1);
        }
    }
    out
}

fn map_pages(physical: &mut Physical, base: usize, n: usize, pages: Option<&[usize]>) -> bool {
    unmap(physical, base, n * PAGE);
    let Some(pages) = pages else {
        return true;
    };
    let mut i = 0;
    while i < n {
        let Some(first) = physical.pages.get(&pages[i]).cloned() else {
            xnw_core::warn!("XMemMapPhysicalPages: unknown page {:#x}", pages[i]);
            return false;
        };
        // One view per run of consecutive slices of one section.
        let run = 1 + pages[i + 1..]
            .iter()
            .zip(1..)
            .take_while(|&(page, k)| {
                physical.pages.get(page).is_some_and(|s| std::sync::Arc::ptr_eq(&s.section, &first.section) && s.index == first.index + k)
            })
            .count();
        if !map(physical, base + i * PAGE, run * PAGE, first) {
            return false;
        }
        i += run;
    }
    true
}

// The paired reservation. Before WinMain the title reserves 24 GiB of
// 64 KiB pages twice in a row, same size and kind: read-write (XMem flags
// 0x21), then no-access with the GPU protection bit (flags 0x3). On the
// console the second is the GPU view of the first: the title maps physical
// pages into the first only and places GPU resources (the movie's planes)
// at the same offsets in the second. Recognised by position and protection,
// as Xbox2Windows `src/xmem_mapped_pages.cpp` (f464f34, MIT) does in its
// `XMemVirtualAlloc` and `local_mirror`; another runtime `9bd7635` (idea only) agrees.
// No flag names the pairing.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Pair {
    cpu: usize,
    gpu: usize,
    len: usize,
}

impl Pair {
    /// The GPU-view address of `[base, base + len)` if it lies in the CPU view.
    fn gpu_of(&self, base: usize, len: usize) -> Option<usize> {
        (base >= self.cpu && base.checked_add(len)? <= self.cpu + self.len).then(|| self.gpu + (base - self.cpu))
    }

    /// Which view `[base, base + len)` lies in: 0 CPU, 1 GPU.
    fn side(&self, base: usize, len: usize) -> Option<usize> {
        let inside = |start: usize| base >= start && base + len <= start + self.len;
        if inside(self.cpu) {
            Some(0)
        } else if inside(self.gpu) {
            Some(1)
        } else {
            None
        }
    }
}

/// The last reservation at an address of our choice, if a CPU-view
/// candidate: a 64 KiB-page placeholder the title asked to be accessible.
static LAST_RESERVATION: std::sync::Mutex<Option<(usize, usize, u32)>> = std::sync::Mutex::new(None);
/// Held across one reservation at an address of our choice ([`place`] and
/// [`pair_reservation`]).
static PLACING: std::sync::Mutex<()> = std::sync::Mutex::new(());
/// The placeholder reserved right above the last CPU-view candidate, same
/// size and kind: (address, size, kind). Taken by the GPU-view request that
/// follows the candidate, released by any other reservation.
static PENDING_GPU: std::sync::Mutex<Option<(usize, usize, u32)>> = std::sync::Mutex::new(None);

/// A reservation at an address of our choice, `call(len)` the host call.
/// GRP1: the pair's two views were placed one call apart, each at the
/// lowest free range; a host mapping made by another thread in between
/// `paired reservation 0x100000000 <-> 0x667200000`, a test run
/// `<-> 0x8df800000`, the next reservation then at 0x500000000) moved the
/// GPU view off `cpu + len`, where the GPU side folded it (then a fixed
/// 0x4_0000_0000 in xnw-frame and xnw-world, now `xnw_core::pair`): every bridged row lost its
/// group count for the whole run. So the first CPU-view candidate takes
/// twice its size in one call, the upper half split off and held for the
/// GPU view.
fn place(size: usize, kind: u32, protect: u32, placeholder: bool, call: &mut impl FnMut(usize) -> *mut c_void) -> *mut c_void {
    let accessible = protect & 0xff != PAGE_NOACCESS;
    let pending = lock(&PENDING_GPU).take();
    if let Some((at, len, k)) = pending {
        if placeholder && !accessible && len == size && k == kind {
            return at as *mut c_void;
        }
        // SAFETY: our own placeholder, never handed out.
        unsafe { VirtualFree(at as *mut c_void, 0, MEM_RELEASE) };
    }
    if !(placeholder && accessible) || with_physical(|p| p.pair.is_some()) {
        return call(size);
    }
    let Some(twice) = size.checked_mul(2) else {
        return call(size);
    };
    let at = call(twice);
    if at.is_null() {
        xnw_core::warn!("paired reservation: {twice:#x} bytes in one range failed ({}); the GPU view goes where the host puts it", last_error());
        return call(size);
    }
    let upper = at as usize + size;
    // SAFETY: splits the placeholder just reserved in two.
    if unsafe { VirtualFree(upper as *mut c_void, size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) } == 0 {
        xnw_core::warn!("paired reservation: splitting {at:p} failed ({}); the GPU view goes where the host puts it", last_error());
        // SAFETY: the placeholder just reserved, whole.
        unsafe { VirtualFree(at, 0, MEM_RELEASE) };
        return call(size);
    }
    *lock(&PENDING_GPU) = Some((upper, size, kind));
    at
}

/// Called for every reservation at an address of our choice, `at` null if
/// it failed; `protect` as the title asked.
fn pair_reservation(at: usize, size: usize, kind: u32, protect: u32, placeholder: bool) {
    let accessible = protect & 0xff != PAGE_NOACCESS;
    let mut last = lock(&LAST_RESERVATION);
    let previous = last.take();
    if at == 0 || !placeholder {
        return;
    }
    if accessible {
        *last = Some((at, size, kind));
        return;
    }
    let Some((cpu, len, cpu_kind)) = previous.filter(|&(_, len, k)| len == size && k == kind) else {
        return;
    };
    with_physical(|physical| {
        if physical.pair.is_none() {
            physical.pair = Some(Pair { cpu, gpu: at, len });
            xnw_gpu::d3d12x::use_pair(cpu, at, len);
            xnw_core::info!("paired reservation {cpu:#x} <-> {at:#x} ({len:#x} bytes, kind {cpu_kind:#x})");
            if at != cpu + len {
                xnw_core::warn!("paired reservation: the GPU view is not right above the CPU view (GRP1); the GPU side folds by the pair published");
            }
            // ALIAS1: the GPU side's folds (xnw-frame, xnw-world) read it from here.
            if at > cpu && !xnw_core::pair::publish(xnw_core::pair::Pair { cpu: cpu as u64, gpu: at as u64, len: len as u64 }) {
                xnw_core::warn!("paired reservation: a pair {:x?} was published before; it stays", xnw_core::pair::live());
            }
        }
    });
}

/// One page at each address; null `pages`, or a page number 0, unmaps (the
/// title passes 0 for addresses to clear).
pub extern "system" fn XMemMapPhysicalPagesScatter(addresses: *const *mut c_void, n: usize, pages: *const usize) -> i32 {
    if addresses.is_null() {
        return 0;
    }
    for i in 0..n {
        // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
        let (address, page) = unsafe { (*addresses.add(i), if pages.is_null() || *pages.add(i) == 0 { std::ptr::null() } else { pages.add(i) }) };
        if XMemMapPhysicalPages(address, 1, page).is_null() {
            return 0;
        }
    }
    1
}

/// Maps the physical pages shown at `[from, from + len)` at `to` as well,
/// in place of what `to` shows; `from` 0 unmaps. The GPU's page-mapping
/// copies (queue `CopyPageMappingsBatchX`, through
/// `d3d12x::use_page_mapper`): the title maps pages into a pool view, then
/// has the GPU map them into its tiled heap. Page by page; false if a
/// source page is not mapped physical memory. From VIctory-base
/// (`xnw/base` @ 5d0516e) `crates/victory-hle/src/modules/xmem.rs`
/// `map_alias` 808-840, GPL-3.0-or-later. A destination in the CPU view of
/// the paired reservation is mirrored in its GPU view, as the title's own
/// mappings are.
pub fn map_alias(to: u64, from: u64, len: u64) -> bool {
    let (to, from, len) = (to as usize, from as usize, len as usize);
    with_physical(|physical| {
        let done = alias(physical, to, from, len);
        if let Some(gpu) = physical.pair.and_then(|p| p.gpu_of(to, len)) {
            let source = if done { from } else { 0 };
            if !alias(physical, gpu, source, len) {
                xnw_core::warn!("paired reservation: mirroring a page mapping of {len:#x} bytes at {gpu:#x} failed");
            }
        }
        done
    })
}

fn alias(physical: &mut Physical, to: usize, from: usize, len: usize) -> bool {
    unmap(physical, to, len);
    for k in (0..len).step_by(PAGE) {
        if from == 0 {
            break;
        }
        let at = from + k;
        let Some(slice) = physical
            .views
            .range(..=at)
            .next_back()
            .filter(|(base, (size, _))| at < *base + size)
            .map(|(base, (_, s))| Slice { index: s.index + (at - base) / PAGE, section: s.section.clone() })
        else {
            return false;
        };
        if !map(physical, to + k, PAGE, slice) {
            return false;
        }
    }
    true
}

/// Maps `slice` and the slices after it over `[base, base + len)`, which
/// must lie in one placeholder.
fn map(physical: &mut Physical, base: usize, len: usize, slice: Slice) -> bool {
    // SAFETY: splits the placeholder around the range (failing harmlessly
    // when the range is the whole placeholder), then maps the section there;
    // both act only on placeholders.
    let view = unsafe {
        VirtualFree(base as *mut c_void, len, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
        map_view_of_file3()(
            slice.section.0,
            GetCurrentProcess(),
            base as *const c_void,
            (slice.index * PAGE) as u64,
            len,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    };
    if view as usize != base {
        xnw_core::warn!("mapping physical pages at {base:#x} ({len:#x} bytes) failed (not a placeholder?): {}", last_error());
        return false;
    }
    if !commit(base, len) {
        // SAFETY: the view just mapped, back to a placeholder.
        unsafe { unmap_view_of_file2()(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
        return false;
    }
    // New contents, read-write: the GPU side's copies no longer guard it
    // (VIctory-base victory-hle xmem.rs 875).
    xnw_gpu::d3d12x::title_memory_remapped(base, len);
    // The texture origin registry follows title addresses to frames.
    xnw_formats::origin::registry().map_va(base, len, (slice.section.1 + slice.index * FRAMES) as u64);
    physical.views.insert(base, (len, slice));
    count(physical, base, len, true);
    true
}

/// Keeps the pair's mapped-byte counts as views come and go.
fn count(physical: &mut Physical, base: usize, len: usize, added: bool) {
    if let Some(side) = physical.pair.and_then(|p| p.side(base, len)) {
        let n = &mut physical.mapped[side];
        *n = if added { *n + len } else { n.saturating_sub(len) };
    }
}

/// Commits the section pages a view shows (once per page, whichever view).
fn commit(base: usize, len: usize) -> bool {
    // SAFETY: `[base, base + len)` is a view of a reserved section.
    if !unsafe { VirtualAlloc(base as *const c_void, len, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
        return true;
    }
    let error = last_error();
    if error.raw_os_error() == Some(ERROR_COMMITMENT_LIMIT) {
        xnw_core::warn!("Windows commit limit reached committing {len} bytes of physical pages: enlarge the paging file");
    } else {
        xnw_core::warn!("committing {len} bytes of physical pages failed: {error}");
    }
    false
}

/// Returns `[base, base + len)` to placeholders; views reaching past it are
/// mapped again outside it.
fn unmap(physical: &mut Physical, base: usize, len: usize) {
    // No bytes: no view to return. The search below would still find the
    // view holding `base` and cut it in two there, and its range of views
    // starting inside would run backwards (which `BTreeMap` panics on).
    // (VIctory 02fcf72.)
    if len == 0 {
        return;
    }
    let end = base + len;
    let first = physical.views.range(..=base).next_back().filter(|(at, (size, _))| *at + size > base).map(|(at, _)| *at);
    let hit: Vec<usize> = first.into_iter().chain(physical.views.range(base + 1..end).map(|(at, _)| *at)).collect();
    for at in hit {
        let Some((size, slice)) = physical.views.remove(&at) else {
            continue;
        };
        count(physical, at, size, false);
        // VIctory-base victory-hle xmem.rs 946.
        xnw_gpu::d3d12x::title_memory_remapped(at, size);
        xnw_formats::origin::registry().unmap_va(at, size);
        // SAFETY: a view we mapped; it becomes a placeholder again.
        unsafe { unmap_view_of_file2()(GetCurrentProcess(), at as *const c_void, MEM_PRESERVE_PLACEHOLDER) };
        if at < base {
            map(physical, at, base - at, slice.clone());
        }
        if at + size > end {
            let index = slice.index + (end - at) / PAGE;
            map(physical, end, at + size - end, Slice { index, ..slice });
        }
    }
}

/// The oldest layout we know; newer titles read the same leading fields.
#[repr(C)]
#[derive(Debug, Default)]
pub struct WorkingSet {
    pub title_limit: usize,
    pub title_used: usize,
    pub tools_limit: usize,
    pub tools_used: usize,
    pub stack_used: u32,
    pub physical_pages: u32,
    pub gpu_optimal_limit: usize,
    pub gpu_optimal_used: usize,
    pub tools_gpu_optimal_limit: usize,
    pub tools_gpu_optimal_used: usize,
    pub kernel_pool_used: u32,
    pub kernel_pool_limit: u32,
}

/// Title memory of the chosen console: (GPU-optimal, standard) bytes
/// (Series S: no GPU-optimal class, 7,776 MiB standard; VIctory `console.rs`).
fn title_memory() -> (usize, usize) {
    xnw_gpu::d3d12x::console::current().title_memory()
}

use std::sync::atomic::Ordering;

fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Bytes in live `XMemAllocDefault` heap blocks.
static HEAP_USED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// The title's own `XMemVirtualAlloc` reservations at addresses of our
/// choice (not placeholders, which physical pages fill): base -> (size,
/// committed bytes). Kept where the title commits, decommits and releases
/// (`XMemVirtualAlloc`, and its `VirtualAlloc`/`VirtualFree`, [`title`]),
/// so the working set is read without walking them.
static REGIONS: std::sync::Mutex<std::collections::BTreeMap<usize, (usize, usize)>> = std::sync::Mutex::new(std::collections::BTreeMap::new());
/// Committed bytes over all `REGIONS`.
static COMMITTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Bytes of live physical pages.
static PHYSICAL_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

const HOST_PAGE: usize = 4096;

/// Committed bytes in `[from, to)`, asked of the host for that range only.
fn committed_in(from: usize, to: usize) -> usize {
    let (mut at, mut total) = (from, 0);
    while at < to {
        // SAFETY: plain data out.
        let mut info: MemoryBasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: our process, our structure.
        if xnw_core::syscalls::timed(xnw_core::syscalls::Call::QueryXmem, || unsafe { VirtualQueryEx(GetCurrentProcess(), at as *const c_void, &mut info, size_of::<MemoryBasicInformation>()) }) == 0 || info.region_size == 0 {
            break;
        }
        let next = (info.base + info.region_size).min(to);
        if info.state == MEM_COMMIT {
            total += next - at;
        }
        at = next;
    }
    total
}

/// Runs `call`, a host call that may commit or decommit `[at, at + len)`
/// (`len` 0: to the end of the allocation), and if that lies in one of the
/// title's regions keeps the region's committed count: the range is asked
/// before and after.
fn tracked<R>(at: usize, len: usize, call: impl FnOnce() -> R) -> R {
    let mut regions = lock(&REGIONS);
    let Some((base, (size, _))) = regions.range(..=at).next_back().map(|(&b, &r)| (b, r)).filter(|&(b, (size, _))| at < b + size) else {
        drop(regions);
        return call();
    };
    let from = at & !(HOST_PAGE - 1);
    let to = if len == 0 { base + size } else { at.saturating_add(len).next_multiple_of(HOST_PAGE).min(base + size) };
    let before = committed_in(from, to);
    let r = call();
    let after = committed_in(from, to);
    if let Some((_, committed)) = regions.get_mut(&base) {
        *committed = (*committed + after).saturating_sub(before);
    }
    if after >= before {
        COMMITTED.fetch_add(after - before, Ordering::Relaxed);
    } else {
        COMMITTED.fetch_sub(before - after, Ordering::Relaxed);
    }
    r
}

/// `VirtualFree` with `MEM_RELEASE`; a region of the title's released
/// stops counting.
fn release(address: *mut c_void, size: usize, kind: u32) -> i32 {
    let mut regions = lock(&REGIONS);
    // SAFETY: the caller's own arguments.
    let done = unsafe { VirtualFree(address, size, kind) };
    if done != 0
        && let Some((_, committed)) = regions.remove(&(address as usize))
    {
        COMMITTED.fetch_sub(committed, Ordering::Relaxed);
    }
    done
}

/// The title's own `VirtualAlloc` and `VirtualFree` (it imports them from
/// `api-ms-win-core-memory`): passed to the host, commits and releases in
/// its XMem regions counted on the way ([`REGIONS`]).
pub mod title {
    use super::*;

    pub extern "system" fn VirtualAlloc(address: *const c_void, size: usize, kind: u32, protect: u32) -> *mut c_void {
        // SAFETY: the title's own arguments.
        let call = || unsafe { xnw_core::win::VirtualAlloc(address, size, kind, protect) };
        if address.is_null() || kind & MEM_COMMIT == 0 {
            return call();
        }
        xnw_core::perf::time(xnw_core::perf::Area::XmemFree, size as u64, || tracked(address as usize, size, call))
    }

    pub extern "system" fn VirtualFree(address: *mut c_void, size: usize, kind: u32) -> i32 {
        // A heap over the memory pins it: it goes first.
        xnw_gpu::shared::before_free(address as usize, if kind & MEM_RELEASE != 0 { 0 } else { size });
        xnw_core::perf::time(xnw_core::perf::Area::XmemFree, size as u64, || {
            if kind & MEM_RELEASE != 0 {
                release(address, size, kind)
            } else {
                // SAFETY: the title's own arguments.
                tracked(address as usize, size, || unsafe { xnw_core::win::VirtualFree(address, size, kind) })
            }
        })
    }
}

/// Standard memory the title has handed to the system.
static TRANSFERRED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The title's budget and what the title itself holds: live
/// `XMemAllocDefault` blocks, committed pages of its `XMemVirtualAlloc`
/// reservations, and its physical pages. Host allocations (D3D12, shader
/// work, our own) are not the title's (after VIctory, b7d3263,
/// GPL-3.0-or-later; it counted process commit before, as we did). No
/// tools memory: a retail console. The counts are kept as memory comes and
/// goes; nothing is walked here (upstream walks the reservations per call:
/// about 12 ms a call in this title).
// shortcut: all fields are filled whatever `which` asks for, and use is not
// split by bandwidth class.
pub extern "system" fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> i32 {
    if out.is_null() {
        return 0;
    }
    let started = xnw_core::perf::start();
    let physical = PHYSICAL_BYTES.load(Ordering::Relaxed);
    let used = HEAP_USED.load(Ordering::Relaxed) + COMMITTED.load(Ordering::Relaxed) + physical;
    let stats = WorkingSet {
        title_limit: title_memory().1 - TRANSFERRED.load(std::sync::atomic::Ordering::Relaxed),
        title_used: used,
        gpu_optimal_limit: title_memory().0,
        ..Default::default()
    };
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        xnw_core::info!("XMemGetWorkingSetStatistics: title uses {} MiB ({} MiB physical pages) of {} MiB", used / MIB, physical / MIB, stats.title_limit / MIB);
    }
    // SAFETY: checked non-null; the title passes room for the structure.
    unsafe { out.write(stats) };
    xnw_core::perf::since(xnw_core::perf::Area::XmemStats, started, 0);
    1
}

/// Hands title memory (standard, in 2 MiB pages) to the system. The other
/// transfers involve tools memory, which a retail console has none of.
pub extern "system" fn XMemTransferMemory(kind: u32, size: usize) -> i32 {
    const TITLE_TO_SYSTEM: u32 = 0;
    xnw_core::info!("XMemTransferMemory(kind {kind}, {size:#x} bytes)");
    if size == 0 {
        return 1;
    }
    let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
        return 0;
    };
    let moved = kind == TITLE_TO_SYSTEM
        && TRANSFERRED
            .fetch_update(std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed, |t| {
                t.checked_add(size).filter(|&t| t <= title_memory().1)
            })
            .is_ok();
    moved.into()
}

#[cfg(test)]
mod physical_tests {
    use super::*;

    #[test]
    fn physical_pages_map_alias_and_unmap() {
        let mut n = 2usize;
        let mut pages = [0usize; 2];
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(pages[1], pages[0] + FRAMES);
        // Two 64 KiB-page reservations; the same pages shown in both.
        let a = XMemVirtualAlloc(std::ptr::null_mut(), 2 * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
        let b = XMemVirtualAlloc(std::ptr::null_mut(), 2 * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
        assert!(!a.is_null() && !b.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        assert_eq!(XMemMapPhysicalPages(b, 2, pages.as_ptr()), b);
        // SAFETY: both views are mapped read-write.
        unsafe {
            a.cast::<u8>().add(PAGE + 5).write(42);
            assert_eq!(b.cast::<u8>().add(PAGE + 5).read(), 42);
        }
        assert!(write_physical(&pages, PAGE - 1, &[7, 8]));
        // SAFETY: as above.
        unsafe { assert_eq!((a.cast::<u8>().add(PAGE - 1).read(), b.cast::<u8>().add(PAGE).read()), (7, 8)) };
        assert_eq!(XMemMapPhysicalPages(a, 2, std::ptr::null()), a);
        assert_eq!(XMemMapPhysicalPagesScatter([b, b.wrapping_byte_add(PAGE)].as_ptr(), 2, std::ptr::null()), 1);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// The region's own count, exact whatever other tests do alongside.
    fn committed(region: *mut c_void) -> Option<usize> {
        lock(&REGIONS).get(&(region as usize)).map(|r| r.1)
    }

    /// Only what the title holds counts, kept as it commits, decommits and
    /// releases: a reservation's committed pages, through `XMemVirtualAlloc`
    /// and the title's own `VirtualAlloc`/`VirtualFree`.
    #[test]
    fn working_set_counts_commits_where_they_happen() {
        let region = XMemVirtualAlloc(std::ptr::null_mut(), 16 * PAGE, MEM_RESERVE, 0, PAGE_READWRITE);
        assert!(!region.is_null());
        assert_eq!(committed(region), Some(0));
        // Commit 2 pages through XMem, then 1 more (and 1 again) through the host.
        let at = region.wrapping_byte_add(PAGE);
        assert_eq!(XMemVirtualAlloc(at, 2 * PAGE, MEM_COMMIT, 0, PAGE_READWRITE), at);
        assert_eq!(committed(region), Some(2 * PAGE));
        let more = region.wrapping_byte_add(2 * PAGE);
        assert_eq!(title::VirtualAlloc(more, 2 * PAGE, MEM_COMMIT, PAGE_READWRITE), more);
        assert_eq!(committed(region), Some(3 * PAGE));
        // Decommit one page and a half: whole host pages go.
        assert_ne!(title::VirtualFree(at, PAGE + 100, MEM_DECOMMIT), 0);
        assert_eq!(committed(region), Some(2 * PAGE - HOST_PAGE));
        // The statistics read the counters; a release drops the region.
        let mut ws = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0, &mut ws), 1);
        assert!(ws.title_limit > 0 && ws.title_used >= 2 * PAGE - HOST_PAGE);
        assert_ne!(title::VirtualFree(region, 0, MEM_RELEASE), 0);
        assert_eq!(committed(region), None);
        // Committed at reservation; released by XMemFreeDefault.
        let low = XMemAllocDefault(3 * PAGE, ALLOC_32BIT);
        assert_eq!(committed(low), Some(3 * PAGE));
        XMemFreeDefault(low, ALLOC_32BIT);
        assert_eq!(committed(low), None);
        // Memory outside the title's regions is not counted.
        let host = title::VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        assert!(!host.is_null() && committed(host).is_none());
        assert_ne!(title::VirtualFree(host, 0, MEM_RELEASE), 0);
    }

    /// The statistics cost a few loads: far below the 12 ms a walk of the
    /// title's reservations cost.
    #[test]
    fn working_set_statistics_are_cheap() {
        let regions: Vec<_> = (0..64).map(|_| XMemVirtualAlloc(std::ptr::null_mut(), 4 * PAGE, MEM_RESERVE | MEM_COMMIT, 0, PAGE_READWRITE)).collect();
        let mut ws = WorkingSet::default();
        let started = std::time::Instant::now();
        for _ in 0..1000 {
            XMemGetWorkingSetStatistics(0, &mut ws);
        }
        assert!(started.elapsed() < std::time::Duration::from_millis(50), "{:?}", started.elapsed());
        for r in regions {
            title::VirtualFree(r, 0, MEM_RELEASE);
        }
    }

    /// VIctory 02fcf72: mapping or unmapping no pages changes nothing.
    #[test]
    fn mapping_no_pages_changes_nothing() {
        let mut n = 2usize;
        let mut pages = [0usize; 2];
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let a = XMemVirtualAlloc(std::ptr::null_mut(), 2 * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: both pages are mapped read-write.
        unsafe { a.cast::<u8>().write(5) };
        let second = a.wrapping_byte_add(PAGE);
        // A count of 0 at the start of a view, inside it and past it: with
        // pages, without (an unmap), and as the GPU's page-mapping copy.
        for at in [a, second, a.wrapping_byte_add(2 * PAGE)] {
            assert_eq!(XMemMapPhysicalPages(at, 0, pages.as_ptr()), at);
            assert_eq!(XMemMapPhysicalPages(at, 0, std::ptr::null()), at);
            assert!(map_alias(at as u64, a as u64, 0));
            assert!(map_alias(at as u64, 0, 0));
        }
        // The view is still the one view it was.
        assert_eq!(with_physical(|p| p.views.get(&(a as usize)).map(|v| v.0)), Some(2 * PAGE));
        // SAFETY: still mapped.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 5);
        // A count of 1 still unmaps that page alone.
        assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
        assert_eq!(with_physical(|p| p.views.get(&(a as usize)).map(|v| v.0)), Some(PAGE));
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// The title's pair: pages mapped at `cpu + o` read at `gpu + o`, and
    /// unmapping at the CPU view unmaps both.
    #[test]
    fn paired_reservation_mirrors_cpu_pages() {
        const GPU_NO_ACCESS: u32 = 0x800 | PAGE_NOACCESS;
        let size = 64 * PAGE;
        // Other tests reserve alongside and may come between the two calls.
        let pair = (0..8)
            .find_map(|_| {
                let cpu = XMemVirtualAlloc(std::ptr::null_mut(), size, MEM_RESERVE | PAGES_64K, 0x21, PAGE_READWRITE) as usize;
                // GRP1: a host reservation by another thread between the
                // two, lowest fit in the same range: before, it took
                // `cpu + size` and the GPU view went above it.
                let mut req = MemAddressRequirements { lowest: 1 << 32, highest: (1 << 40) - 1, alignment: 0 };
                let mut par = MemExtendedParameter { kind: 1, pointer: &mut req };
                // SAFETY: a plain reservation; the parameter outlives the call.
                let host = unsafe { virtual_alloc2()(GetCurrentProcess(), std::ptr::null_mut(), size, MEM_RESERVE, PAGE_NOACCESS, &mut par, 1) };
                let gpu = XMemVirtualAlloc(std::ptr::null_mut(), size, MEM_RESERVE | PAGES_64K, GRAPHICS, GPU_NO_ACCESS) as usize;
                // SAFETY: the reservation just made.
                unsafe { VirtualFree(host, 0, MEM_RELEASE) };
                let pair = Pair { cpu, gpu, len: size };
                (with_physical(|p| p.pair) == Some(pair)).then_some(pair)
            })
            .expect("the pair is recognised");
        // GRP1: the GPU view right above the CPU view, whatever else the
        // process maps meanwhile.
        assert_eq!(pair.gpu, pair.cpu + size);
        let mut n = 3usize;
        let mut pages = [0usize; 3];
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let o = 5 * PAGE;
        let at = (pair.cpu + o) as *mut c_void;
        assert_eq!(XMemMapPhysicalPages(at, 2, pages.as_ptr()), at);
        // One page by scatter, after a gap.
        let lone = (pair.cpu + 9 * PAGE) as *mut c_void;
        assert_eq!(XMemMapPhysicalPagesScatter([lone].as_ptr(), 1, pages[2..].as_ptr()), 1);
        assert_eq!(with_physical(|p| p.mapped), [3 * PAGE, 3 * PAGE]);
        // SAFETY: both views are mapped read-write.
        unsafe {
            (at as *mut u8).add(PAGE + 3).write(0x5a);
            (lone as *mut u8).write(0xa5);
            assert_eq!(((pair.gpu + o + PAGE + 3) as *const u8).read(), 0x5a);
            assert_eq!(((pair.gpu + 9 * PAGE) as *const u8).read(), 0xa5);
            // And back: the GPU view is the same memory.
            ((pair.gpu + o) as *mut u8).write(0x11);
            assert_eq!((at as *const u8).read(), 0x11);
        }
        // Unmapping one page of the run in the CPU view unmaps it in both.
        assert_eq!(XMemMapPhysicalPages(at, 1, std::ptr::null()), at);
        assert_eq!(with_physical(|p| p.mapped), [2 * PAGE, 2 * PAGE]);
        assert_eq!(XMemMapPhysicalPagesScatter([lone, (pair.cpu + o + PAGE) as *mut c_void].as_ptr(), 2, [0usize, 0].as_ptr()), 1);
        assert_eq!(with_physical(|p| p.mapped), [0, 0]);
        let mut info: MemoryBasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: our process, our structure.
        unsafe { VirtualQueryEx(GetCurrentProcess(), (pair.gpu + o) as *const c_void, &mut info, size_of::<MemoryBasicInformation>()) };
        assert_ne!(info.state, MEM_COMMIT);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    /// The GPU reads title pages in place, through the runtime's view
    /// of their section, while the title remaps its own view; and a title
    /// region in place, under the write watch, released before it is freed.
    /// Skipped without a GPU.
    #[test]
    fn gpu_reads_title_memory_in_place() {
        use xnw_gpu::shared::{Miss, SharedMemory, read_back};
        let _ = xnw_gpu::shared::in_place();
        let agility = std::env::var_os("XNW_AGILITY").map(std::path::PathBuf::from).filter(|p| !p.as_os_str().is_empty());
        let Ok(host) = xnw_gpu::Host::new(agility.as_deref()) else {
            eprintln!("no host GPU: skipped");
            return;
        };
        let pattern = |seed: u8, len: usize| -> Vec<u8> { (0..len).map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed)).collect() };
        let shared = SharedMemory::new(host.device.clone(), backing, physical_window);

        let mut n = 4usize;
        let mut pages = [0usize; 4];
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let b = XMemVirtualAlloc(std::ptr::null_mut(), 2 * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
        assert!(!b.is_null());
        let write = |seed: u8| {
            // SAFETY: two pages are mapped read-write at `b`.
            unsafe { std::slice::from_raw_parts_mut(b.cast::<u8>(), 2 * PAGE).copy_from_slice(&pattern(seed, 2 * PAGE)) }
        };

        // Pages 0 and 1 at B, written through B, read on the GPU.
        assert_eq!(XMemMapPhysicalPages(b, 2, pages.as_ptr()), b);
        write(1);
        let first = shared.buffer(b as usize, 2 * PAGE).expect("pages 0-1 resolve");
        assert_eq!(read_back(&host, &first.buffer, first.offset, 2 * PAGE).unwrap(), pattern(1, 2 * PAGE));

        // The write watch's read-only guard on B is allowed under the heap.
        let mut old = 0;
        // SAFETY: B is ours.
        assert_ne!(unsafe { VirtualProtect(b, PAGE, PAGE_READONLY, &mut old) }, 0);
        // SAFETY: as above.
        assert_ne!(unsafe { VirtualProtect(b, PAGE, PAGE_READWRITE, &mut old) }, 0);

        // B remapped to pages 2 and 3: the same buffer, the new offset.
        assert_eq!(XMemMapPhysicalPages(b, 2, pages[2..].as_ptr()), b);
        write(2);
        let second = shared.buffer(b as usize, 2 * PAGE).expect("pages 2-3 resolve");
        assert_eq!(second.offset, first.offset + 2 * PAGE as u64);
        assert_eq!(read_back(&host, &second.buffer, second.offset, 2 * PAGE).unwrap(), pattern(2, 2 * PAGE));
        assert_eq!(read_back(&host, &first.buffer, first.offset, 2 * PAGE).unwrap(), pattern(1, 2 * PAGE));

        // The GPU writes through the shared buffer; the title reads it at B.
        xnw_gpu::shared::write_through(&host, &second.buffer, second.offset + 16, &[1, 2, 3, 4]).unwrap();
        // SAFETY: B is mapped read-write.
        assert_eq!(unsafe { std::slice::from_raw_parts(b.cast::<u8>().add(16), 4) }, &[1, 2, 3, 4]);

        // Pages out of section order: one page resolves, both do not.
        assert_eq!(XMemMapPhysicalPagesScatter([b, b.wrapping_byte_add(PAGE)].as_ptr(), 2, [pages[3], pages[0]].as_ptr()), 1);
        assert!(shared.buffer(b as usize, PAGE).is_ok());
        assert_eq!(shared.buffer(b as usize, 2 * PAGE).err(), Some(Miss::NotContiguous));
        assert_eq!(XMemMapPhysicalPages(b, 2, std::ptr::null()), b);

        // A title region, all committed (write-watched): read in place; the
        // watch still sees CPU writes under the heap; pinned until released.
        let size = 4 * MIB;
        let r = XMemVirtualAlloc(std::ptr::null_mut(), size, MEM_COMMIT | MEM_RESERVE, 0, PAGE_READWRITE);
        assert!(!r.is_null());
        let at = r.wrapping_byte_add(MIB).cast::<u8>();
        // SAFETY: committed read-write memory of the region.
        unsafe { std::slice::from_raw_parts_mut(at, PAGE).copy_from_slice(&pattern(3, PAGE)) };
        let watched = || {
            let mut found = [std::ptr::null_mut::<c_void>(); 1024];
            let (mut count, mut granularity) = (found.len(), 0u32);
            // SAFETY: our region and our arrays.
            unsafe { GetWriteWatch(WRITE_WATCH_FLAG_RESET, r, size, found.as_mut_ptr(), &mut count, &mut granularity) };
            count
        };
        watched();
        // SAFETY: as above.
        unsafe { at.write(9) };
        let before = watched();
        let region = shared.buffer(at as usize, PAGE).expect("region resolves");
        assert_eq!((region.offset, region.len), (MIB as u64, PAGE));
        // SAFETY: as above.
        unsafe { at.write(pattern(3, 1)[0]) };
        assert_eq!(read_back(&host, &region.buffer, region.offset, PAGE).unwrap(), pattern(3, PAGE));
        let opened = watched();
        let quiet = watched();
        // SAFETY: as above.
        unsafe { at.add(2 * PAGE).write(9) };
        let fresh = watched();
        // SAFETY: as above.
        unsafe { at.write(10) };
        let again = watched();
        // SAFETY: as above.
        unsafe { at.add(2 * PAGE).write(11) };
        let again2 = watched();
        eprintln!(
            "write watch: one write before the heap {before} pages; at opening {opened}; then {quiet}; under the heap, a write to an untouched page {fresh}, to a written page {again}, to the first again {again2}"
        );
        assert_eq!((fresh, again, again2), (1, 1, 1), "GetWriteWatch sees each CPU write under the heap");
        assert_eq!(title::VirtualFree(at.cast(), PAGE, MEM_DECOMMIT), 0, "pinned while the heap lives");
        // The heap goes with its last buffer reference.
        drop(region);
        shared.before_free(r as usize, 0);
        assert_eq!(title::VirtualFree(r, 0, MEM_RELEASE), 1);

        drop((first, second, shared));
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
    }
}
