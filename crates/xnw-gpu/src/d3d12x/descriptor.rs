//! Descriptor heaps: zeroed memory, with handles that are plain addresses
//! (CPU and GPU alike, as on the console).
//!
//! Descriptor sizes: views 32 bytes and samplers 16, the size of RDNA 2's
//! image and sampler descriptors, so a title that copies descriptors itself
//! moves whole ones. Render target and depth views are ours alone; 128 bytes
//! leaves room for whatever stride a title assumes.
//!
//! What a slot holds is ours, not the console's descriptor bits: a tag, an
//! index into the table of distinct views and that entry's generation
//! ([`record`]); render target and depth slots hold the resource and the
//! view description ([`RtvRecord`]). Shader-visible view and sampler heaps
//! get a host heap with the same descriptor order ([`host`]), whose slots
//! are made from these records.
//!
//! Ported from VIctory-base (VIctory `d2a4bae` + open pull requests, branch
//! `xnw/base` @ 5d0516e, GPL-3.0-or-later) `crates/victory-gpu/src/
//! descriptor.rs` 1-827, whole (upstream adds `host_arc`, `forget`'s
//! bindless epoch). Changed: our object tables (`SLOTS`, `LEN`) and the
//! heap description method returning its out pointer (as before);
//! `tracing` -> `xnw_core` (heap creation is an info line); no
//! `cfg(windows)` (the crate is Windows-only); descriptor traffic counters
//! ([`traffic`]) are ours.
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]

use std::sync::atomic::{AtomicU64, Ordering};

use super::object::{self, DESCRIPTOR_HEAP, Done, E_INVALIDARG, HRESULT, Iid, Object, Slot, Table, This, guid};
use super::resource::{Memory, PAGEABLE_IIDS};

pub(crate) mod gate;
pub mod slotwatch;
pub mod host;

const HEAP_IIDS: [Iid; 4] = [
    guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HeapDesc {
    pub kind: u32,
    pub count: u32,
    pub flags: u32,
    pub nodes: u32,
}

/// Newer titles (the target title among them) pass the description followed by
/// memory of their own for the descriptors: an address and its size in
/// bytes, or a null address to have the heap allocate (inferred from the
/// title's call sites; older titles pass the plain description).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Placement {
    address: u64,
    size: u64,
}

#[repr(C)]
pub(crate) struct Heap {
    object: Object,
    desc: HeapDesc,
    /// The first descriptor: in `memory`, or in the title's own memory.
    pub(crate) start: u64,
    _memory: Option<Memory>,
    /// XNW: with GPU-side descriptors, a shader-visible heap in our own
    /// memory lives in a GPU-readable buffer (its records read in place).
    _gpu_memory: Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    /// Shader-visible view and sampler heaps get a host heap with the same
    /// descriptor order when a draw first uses them (VIctory decision
    /// 0007, step 3); slots written before wait in `pending`.
    host: std::sync::OnceLock<Option<std::sync::Arc<host::HostHeap>>>,
    pending: std::sync::Mutex<Vec<u32>>,
}

pub(crate) fn heap<'a>(this: This) -> &'a Heap {
    // SAFETY: called from descriptor heap methods only.
    unsafe { &*this.cast::<Heap>() }
}

impl Heap {
    /// Whether this is a sampler heap.
    pub(crate) fn samplers(&self) -> bool {
        self.desc.kind == SAMPLERS
    }

    /// The host heap, made on first use from what the title wrote so far.
    /// Made late because a host view heap the size of the title's
    /// costs over 100 MB of commit, which its boot needs for its page pool.
    pub(crate) fn host(&self) -> Option<&host::HostHeap> {
        self.host_arc_ref().map(std::sync::Arc::as_ref)
    }

    /// Shared reference to the host heap for submission-time bindless sync.
    pub(crate) fn host_arc(&self) -> Option<std::sync::Arc<host::HostHeap>> {
        self.host_arc_ref().cloned()
    }

    fn host_arc_ref(&self) -> Option<&std::sync::Arc<host::HostHeap>> {
        if self.desc.flags & SHADER_VISIBLE == 0 || !matches!(self.desc.kind, VIEWS_KIND | SAMPLERS) {
            return None;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // A recorder can take the fast path as soon as the OnceLock is
        // initialized. Publish only after replaying the queued writes
        // (VIctory a4e7f1d, GPL-3.0-or-later).
        self.host
            .get_or_init(|| {
                let step = u64::from(increment(self.desc.kind)?);
                let h = host::HostHeap::new(self.samplers(), self.desc.count, self.start)?;
                for i in pending.drain(..) {
                    h.mirror(i, self.start + u64::from(i) * step, false);
                }
                Some(std::sync::Arc::new(h))
            })
            .as_ref()
    }

    /// The descriptor index of `handle` in this heap, if it lies in it.
    /// XNW: the first byte past the heap's descriptors.
    pub(crate) fn end(&self) -> u64 {
        self.start + u64::from(increment(self.desc.kind).unwrap_or(0)) * u64::from(self.desc.count)
    }

    pub(crate) fn index(&self, handle: u64) -> Option<u32> {
        let step = u64::from(increment(self.desc.kind)?);
        let at = handle.checked_sub(self.start)?;
        let index = at / step;
        (index < u64::from(self.desc.count)).then_some(index as u32)
    }
}

const VIEWS_KIND: u32 = 0;
const SAMPLERS: u32 = 1;
const SHADER_VISIBLE: u32 = 1;

/// Shader-visible heaps by address range, for mirroring descriptor writes:
/// start, end, heap.
static SHADER_HEAPS: std::sync::Mutex<Vec<(u64, u64, usize)>> = std::sync::Mutex::new(Vec::new());

/// The shader-visible heap holding `handle`, and the descriptor's index.
fn shader_slot<'a>(handle: u64) -> Option<(&'a Heap, u32)> {
    let heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let &(_, _, at) = heaps.iter().rev().find(|&&(start, end, _)| (start..end).contains(&handle))?;
    // SAFETY: registered heaps are live (removed when dropped).
    let heap = unsafe { &*(at as *const Heap) };
    Some((heap, heap.index(handle)?))
}

/// Makes the host descriptor for what the title wrote at `handle`, if it
/// is in a shader-visible heap.
fn mirror(handle: u64) {
    let Some((heap, index)) = shader_slot(handle) else {
        return;
    };
    TRAFFIC[MIRRORED].fetch_add(1, Ordering::Relaxed);
    let mut pending = heap.pending.lock().unwrap_or_else(|e| e.into_inner());
    match heap.host.get() {
        None => pending.push(index),
        Some(host) => {
            drop(pending);
            if let Some(host) = host {
                host.mirror(index, handle, false);
            }
        }
    }
}

/// Host heaps of title heaps the title released, kept for the run:
/// draws borrow a host heap from its title heap without holding it, so one
/// freed while a list was recorded on another thread would be used freed (a
/// descriptor handle into a released heap is what the driver faulted on in
/// Title heaps are made a handful of times per run.
static RETIRED: std::sync::Mutex<Vec<std::sync::Arc<host::HostHeap>>> = std::sync::Mutex::new(Vec::new());

impl Drop for Heap {
    fn drop(&mut self) {
        if let Some(Some(host)) = self.host.get() {
            host.detach_title();
            RETIRED.lock().unwrap_or_else(|e| e.into_inner()).push(host.clone());
        }
        let at = std::ptr::from_ref::<Heap>(self) as usize;
        let mut heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
        heaps.retain(|&(_, _, h)| h != at);
    }
}

/// Bytes per descriptor, by heap type (views, samplers, render targets,
/// depth).
fn increment(kind: u32) -> Option<u32> {
    [32, 16, 128, 128].get(kind as usize).copied()
}

/// Device method 0x78.
pub(crate) extern "system" fn increment_size(_: This, kind: u32) -> u32 {
    increment(kind).unwrap_or(0)
}

/// What our render target views hold: the resource and the view
/// description, turned into a host view when a command uses them.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RtvRecord {
    tag: u32,
    described: u32,
    pub(crate) resource: This,
    desc: [u32; 6],
}
const RTV_TAG: u32 = u32::from_le_bytes(*b"vRTV");
const _: () = assert!(size_of::<RtvRecord>() <= 112);

impl RtvRecord {
    /// The view description in the host's type (the same layout).
    pub(crate) fn desc(&self) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_RENDER_TARGET_VIEW_DESC> {
        const TEXTURE_2D: u32 = 4;
        self.host_desc(TEXTURE_2D)
    }

    /// The depth view description in the host's type (the same layout).
    pub(crate) fn depth_desc(&self) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCIL_VIEW_DESC> {
        const TEXTURE_2D: u32 = 3;
        self.host_desc(TEXTURE_2D)
    }

    /// The description words as a host type, the format as the host has it.
    /// Without one, a typeless host copy gets a 2D view (`dimension`) in
    /// the title's format, which the host needs spelled out.
    fn host_desc<T>(&self, dimension: u32) -> Option<T> {
        assert!(size_of::<T>() <= size_of::<[u32; 6]>());
        let mut words = if self.described != 0 { self.desc } else { [super::resource::view_format(self.resource)?, dimension, 0, 0, 0, 0] };
        // A described view without a format takes the title's own too.
        if words[0] == 0 {
            words[0] = super::resource::view_format(self.resource).unwrap_or(0);
        }
        words[0] = super::resource::pc_format(words[0]);
        // SAFETY: both host view descriptions are 24 bytes or less in this layout.
        Some(unsafe { (&raw const words).cast::<T>().read_unaligned() })
    }
}

/// Device method 0xa0: records the view at `handle` (descriptor memory).
pub(crate) extern "system" fn create_rtv(_: This, resource: This, desc: *const [u32; 6], handle: u64) {
    if handle == 0 {
        return;
    }
    TRAFFIC[TARGETS].fetch_add(1, Ordering::Relaxed);
    // SAFETY: the title passes a description or null.
    let described = unsafe { desc.as_ref() };
    let record = RtvRecord { tag: RTV_TAG, described: u32::from(described.is_some()), resource, desc: described.copied().unwrap_or_default() };
    // SAFETY: the handle is a descriptor slot, at least 112 bytes.
    unsafe { (handle as *mut RtvRecord).write_unaligned(record) };
}

/// Device method 0xa8: a depth view, recorded as render target views are.
pub(crate) extern "system" fn create_dsv(this: This, resource: This, desc: *const [u32; 6], handle: u64) {
    create_rtv(this, resource, desc, handle);
    if handle != 0 {
        // SAFETY: just written by `create_rtv`.
        unsafe { (handle as *mut u32).write_unaligned(DSV_TAG) };
    }
}
const DSV_TAG: u32 = u32::from_le_bytes(*b"vDSV");

/// A shader view or sampler as the title described it (descriptions in the
/// host's layout, kept as words), for the host to build its own from.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum View {
    /// Constant buffer: GPU address and size.
    Constant(u64, u32),
    /// Shader resource: resource, description (none: the resource's own).
    Shader(usize, Option<[u32; 10]>),
    /// Unordered access: resource, counter resource, description.
    Unordered(usize, usize, Option<[u32; 10]>),
    Sampler([u32; 13]),
    /// The console's own sampler description: flags, filters per axis,
    /// addressing, LOD and anisotropy settings, border color, then
    /// console-only tuning words.
    ConsoleSampler([u32; 27]),
    /// A view of memory without a resource object: whether it is unordered
    /// access, the resource description, then the view's (the plain one
    /// followed by the memory address and console fields).
    Placed(bool, [u32; 14], [u32; 16]),
}

impl View {
    /// Whether the view reads buffer memory (constant buffers, buffer views
    /// of resources or of memory) or memory the title placed a texture in:
    /// such views get host slots only for draws.
    fn is_buffer(&self) -> bool {
        match self {
            View::Constant(..) | View::Placed(..) => true,
            View::Shader(r, _) | View::Unordered(r, _, _) => super::resource::buffer_bytes(*r as This).is_some(),
            View::Sampler(_) | View::ConsoleSampler(_) => false,
        }
    }
}

/// Every distinct view in use; a descriptor holds a tag, an index into it
/// and the entry's generation, which survive titles copying descriptors as
/// bytes. A view of a resource lives as long as the resource: when the
/// title releases it, its entries are freed for reuse under a new
/// generation, so descriptors (and host slots made from them) naming the
/// old one go stale instead of naming whatever is made at the same address
/// next.
/// shortcut: views without a resource object (constant buffers, memory
/// views, samplers) are kept for the run; they are bounded by the title's
/// memory and sampler states.
struct Views {
    /// Each entry's view and generation.
    list: Vec<(View, u32)>,
    index: std::collections::BTreeMap<View, u32>,
    /// Entries naming each resource.
    by_resource: std::collections::BTreeMap<usize, Vec<u32>>,
    free: Vec<u32>,
    /// Sampler entries' GPU-mode sampler heap indices.
    samplers: std::collections::BTreeMap<u32, u32>,
    /// Each entry's record kind word (dword 4), for nulling its table slot.
    kinds: Vec<u32>,
}

static VIEWS: std::sync::Mutex<Views> = std::sync::Mutex::new(Views {
    list: Vec::new(),
    index: std::collections::BTreeMap::new(),
    by_resource: std::collections::BTreeMap::new(),
    free: Vec::new(),
    samplers: std::collections::BTreeMap::new(),
    kinds: Vec::new(),
});
pub(crate) const VIEW_TAG: u32 = u32::from_le_bytes(*b"vVIE");

impl View {
    /// The resource objects the view names.
    fn resources(&self) -> [usize; 2] {
        match *self {
            View::Shader(r, _) => [r, 0],
            View::Unordered(r, counter, _) => [r, counter],
            _ => [0, 0],
        }
    }
}

fn record(handle: u64, view: View) {
    if handle == 0 {
        return;
    }
    TRAFFIC[RECORDED].fetch_add(1, Ordering::Relaxed);
    let kind = kind_word(&view);
    let sampler = matches!(view, View::Sampler(_) | View::ConsoleSampler(_));
    let for_table = host::gpu_mode().then(|| view.clone());
    // Read in place: dword 4 bit 16, dwords 5-7 the raw window view, byte
    // offset, bytes (GPU-side descriptors only; `u32::MAX` otherwise).
    let (kind, raw) = match for_table.as_ref().filter(|_| !sampler).and_then(host::table_in_place) {
        Some(raw) => (kind | IN_PLACE, raw),
        None => (kind, [u32::MAX; 3]),
    };
    let mut views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let i = match views.index.get(&view) {
        Some(&i) => i,
        None => {
            let i = match views.free.pop() {
                Some(i) => {
                    views.list[i as usize].0 = view.clone();
                    i
                }
                None => {
                    views.list.push((view.clone(), 1));
                    views.list.len() as u32 - 1
                }
            };
            for r in view.resources().into_iter().filter(|&r| r != 0) {
                views.by_resource.entry(r).or_default().push(i);
            }
            views.index.insert(view, i);
            i
        }
    };
    let generation = views.list[i as usize].1;
    let distinct = views.index.len() as u64;
    if views.kinds.len() <= i as usize {
        views.kinds.resize(i as usize + 1, 0);
    }
    views.kinds[i as usize] = kind;
    let index = if sampler { sampler_index(&mut views, i) } else { host::table_index(i, kind) };
    if !sampler && index < host::TABLE_NULLS {
        NULL_RECORDS[(kind & 0xf) as usize].fetch_add(1, Ordering::Relaxed);
    }
    // SAFETY: the handle is a descriptor slot, at least 16 bytes (32 for
    // views): the record layout for GPU-side descriptors.
    let v_sharp = for_table.as_ref().filter(|_| !sampler && crate::shared::page_table_on()).and_then(|v| v_sharp(v, index, kind, i, generation));
    unsafe {
        if let Some(w) = v_sharp {
            (handle as *mut [u32; 8]).write_unaligned(w);
            V_SHARPS.fetch_add(1, Ordering::Relaxed);
        } else {
            (handle as *mut [u32; 4]).write_unaligned([index, i, generation, VIEW_TAG]);
            if !sampler {
                ((handle + 16) as *mut [u32; 4]).write_unaligned([kind, raw[0], raw[1], raw[2]]);
            }
        }
    }
    drop(views);
    TRAFFIC[DISTINCT].store(distinct, Ordering::Relaxed);
    if let Some(view) = &for_table {
        host::table_made(index, view);
    }
    mirror(handle);
    log_traffic();
}

/// XNW: view records written with a null's index (entry past the view
/// table), by kind.
static NULL_RECORDS: [std::sync::atomic::AtomicU64; 16] = [const { std::sync::atomic::AtomicU64::new(0) }; 16];

/// [`NULL_RECORDS`] as text ("none" when all are 0).
pub(crate) fn null_records_text() -> String {
    let parts: Vec<String> = NULL_RECORDS
        .iter()
        .enumerate()
        .filter_map(|(k, n)| {
            let n = n.load(Ordering::Relaxed);
            (n > 0).then(|| format!("kind {k} {n}"))
        })
        .collect();
    if parts.is_empty() { "none".into() } else { parts.join(", ") }
}

/// Record kinds (dword 4, bits 0-7) of the layout for GPU-side descriptors;
/// host slot `k` of the GPU-mode heap is a null view of kind `k`.
pub(crate) const KIND_CBV: u32 = 1;
pub(crate) const KIND_SRV_RAW: u32 = 2;
pub(crate) const KIND_SRV_TYPED: u32 = 3;
pub(crate) const KIND_SRV_STRUCTURED: u32 = 4;
pub(crate) const KIND_SRV_TEXTURE: u32 = 5;
pub(crate) const KIND_UAV_RAW: u32 = 6;
pub(crate) const KIND_UAV_TYPED: u32 = 7;
pub(crate) const KIND_UAV_STRUCTURED: u32 = 8;
pub(crate) const KIND_UAV_TEXTURE: u32 = 9;
pub(crate) const KIND_SAMPLER: u32 = 10;
/// Dword 4 bit 16: the buffer is read and written in place.
pub(crate) const IN_PLACE: u32 = 1 << 16;

/// Dword 4 of a view's record: its kind, and for textures the D3D12 view
/// dimension in bits 8-15.
pub(crate) fn kind_word(view: &View) -> u32 {
    const BUFFER: u32 = 1;
    // SRV words: format, dimension, mapping, pad, first (2), count, stride, flags.
    // UAV words: format, dimension, first (2), count, stride, counter (2), flags.
    let buffer_kind = |raw: bool, stride: u32, base: u32| base + if raw { 0 } else if stride > 0 { 2 } else { 1 };
    let srv = |d: &Option<[u32; 10]>, buffer: bool| match d {
        Some(d) if d[1] == BUFFER => buffer_kind(d[8] & 1 != 0, d[7], KIND_SRV_RAW),
        Some(d) => KIND_SRV_TEXTURE | (d[1] & 0xff) << 8,
        None if buffer => KIND_SRV_TYPED,
        None => KIND_SRV_TEXTURE,
    };
    let uav = |d: &Option<[u32; 10]>, buffer: bool| match d {
        Some(d) if d[1] == BUFFER => buffer_kind(d[8] & 1 != 0, d[5], KIND_UAV_RAW),
        Some(d) => KIND_UAV_TEXTURE | (d[1] & 0xff) << 8,
        None if buffer => KIND_UAV_TYPED,
        None => KIND_UAV_TEXTURE,
    };
    // Without a description only the resource tells buffer from texture:
    // asked only where the kind is read (`XNW_DESC=gpu`).
    let buffer = |r: usize, d: &Option<[u32; 10]>| d.is_none() && host::gpu_mode() && super::resource::buffer_bytes(r as This).is_some();
    match view {
        View::Constant(..) => KIND_CBV,
        View::Shader(r, d) => srv(d, buffer(*r, d)),
        View::Unordered(r, _, d) => uav(d, buffer(*r, d)),
        View::Placed(unordered, resource, desc) => {
            let d = Some(std::array::from_fn::<u32, 10, _>(|k| desc[k]));
            let buffer = resource[0] == BUFFER;
            if *unordered { uav(&d, buffer) } else { srv(&d, buffer) }
        }
        View::Sampler(_) | View::ConsoleSampler(_) => KIND_SAMPLER,
    }
}

/// Host samplers of the GPU-mode sampler heap: at most this many, as D3D12
/// allows in a shader-visible sampler heap.
pub(crate) const SAMPLER_TABLE: u32 = 2048;

/// The GPU-mode sampler heap index of sampler entry `i`, given on first
/// use and kept for the run (samplers are); 0 past [`SAMPLER_TABLE`].
fn sampler_index(views: &mut Views, i: u32) -> u32 {
    let n = views.samplers.len() as u32;
    *views.samplers.entry(i).or_insert(if n < SAMPLER_TABLE { n } else { 0 })
}

/// The title released `resource`: the views naming it go stale.
pub fn forget(resource: usize) {
    let mut views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(entries) = views.by_resource.remove(&resource) else {
        return;
    };
    host::bindless_viewed_changed();
    let mut freed = Vec::new();
    let mut released = Vec::new();
    for i in entries {
        let (view, generation) = views.list[i as usize].clone();
        // Entries freed through a view's other resource may hold another
        // view by now.
        if views.index.get(&view) != Some(&i) || !view.resources().contains(&resource) {
            continue;
        }
        views.index.remove(&view);
        views.list[i as usize].1 = generation.wrapping_add(1).max(1);
        // The view's other resource (a counter) keeps a stale entry, which
        // `forget` of that resource skips.
        views.free.push(i);
        freed.push((i, views.kinds.get(i as usize).copied().unwrap_or(0)));
        released.push((i, generation));
    }
    if !released.is_empty() {
        let mut log = RELEASED_LOG.lock().unwrap_or_else(|e| e.into_inner());
        for (&(i, generation), &(_, kind)) in released.iter().zip(&freed) {
            log.1.push_back((i, generation, kind));
        }
        let over = log.1.len().saturating_sub(RELEASED_KEPT);
        log.1.drain(..over);
        log.0 += over as u64;
        RELEASED.fetch_add(released.len() as u64, Ordering::Release);
    }
    drop(views);
    // Their GPU-mode table slots read nothing until reused.
    for (i, kind) in freed {
        host::table_null(host::table_index(i, kind), kind);
    }
}

/// View entries released so far (their generation moved on): heaps
/// null the slots made from them before the next submission.
static RELEASED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The last [`RELEASED_KEPT`] released entries (entry, generation it
/// had, kind word), oldest first, and how many were cut before them.
static RELEASED_LOG: std::sync::Mutex<(u64, std::collections::VecDeque<(u32, u32, u32)>)> =
    std::sync::Mutex::new((0, std::collections::VecDeque::new()));
const RELEASED_KEPT: usize = 1 << 16;

/// The entries released since `from` (a count [`RELEASED`] had) and
/// the count now; `None` inside when the log no longer reaches back to
/// `from` (look at every slot), nothing when none was released since.
#[allow(clippy::type_complexity)]
pub(crate) fn released_since(from: u64) -> Option<(Option<Vec<(u32, u32, u32)>>, u64)> {
    if RELEASED.load(Ordering::Acquire) == from {
        return None;
    }
    let log = RELEASED_LOG.lock().unwrap_or_else(|e| e.into_inner());
    let upto = log.0 + log.1.len() as u64;
    if from < log.0 {
        return Some((None, upto));
    }
    Some((Some(log.1.iter().skip((from - log.0) as usize).copied().collect()), upto))
}

/// Each view entry's generation and record kind word, one look.
pub(crate) fn generations() -> Vec<(u32, u32)> {
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    views.list.iter().enumerate().map(|(i, (_, g))| (*g, views.kinds.get(i).copied().unwrap_or(0))).collect()
}

/// `XNW_GPU_CHECK`: the view entry `i` holds now, its generation and
/// kind word.
pub fn entry(i: u32) -> Option<(View, u32, u32)> {
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let (view, g) = views.list.get(i as usize)?;
    Some((view.clone(), *g, views.kinds.get(i as usize).copied().unwrap_or(0)))
}

/// Of `entries` (slot, (view entry, generation)), those whose view went
/// stale: one look at the views for all.
fn stale_views(entries: &[(u32, (u32, u32))]) -> Vec<(u32, (u32, u32))> {
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    entries
        .iter()
        .filter(|(_, (i, generation))| views.list.get(*i as usize).is_none_or(|(_, current)| current != generation))
        .copied()
        .collect()
}

/// The view of entry `i` if `generation` is still its own.
fn view_at(i: u32, generation: u32) -> Option<View> {
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let (view, current) = views.list.get(i as usize)?;
    (*current == generation).then(|| view.clone())
}

/// The view recorded at `handle`, if one is (and its resource is alive).
pub(crate) fn view(handle: u64) -> Option<View> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let (i, generation) = unsafe { record_id(handle) }?;
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let (view, current) = views.list.get(i as usize)?;
    (*current == generation).then(|| view.clone())
}

/// The tag of a buffer-view record written as a
/// GFX10 V# (dword 3 bits 20-31; bits 0-19 the host view's heap index).
pub(crate) const PAGE_TAG: u32 = xnw_shader::emit::PAGE_TAG;

/// The view entry and generation of the record at `handle`: dwords 1-2
/// under the `vVIE` tag in dword 3; for a V# record, dwords 5-6 under
/// the tag in dword 7.
///
/// # Safety
/// `handle` is a descriptor slot of a live heap (32 bytes for views).
pub unsafe fn record_id(handle: u64) -> Option<(u32, u32)> {
    // SAFETY: the caller's.
    let w = unsafe { (handle as *const [u32; 4]).read_unaligned() };
    if w[3] == VIEW_TAG {
        return Some((w[1], w[2]));
    }
    if w[3] >> 20 != PAGE_TAG {
        return None;
    }
    // SAFETY: a V# record is a view's, 32 bytes.
    let hi = unsafe { ((handle + 16) as *const [u32; 4]).read_unaligned() };
    (hi[3] == VIEW_TAG).then_some((hi[1], hi[2]))
}

/// With `XNW_DESC=gpu` and `XNW_PAGETABLE=1`, a buffer view's record
/// as the page-table design has it: dwords 0-3 a GFX10 V# (base, stride and
/// records, the tag over the host view's heap index), dwords 4-7 the kind
/// and the entry, generation and `vVIE` (where dwords 1-3 hold them in
/// other records). None for other views.
fn v_sharp(view: &View, index: u32, kind: u32, i: u32, generation: u32) -> Option<[u32; 8]> {
    // Raw views count bytes (stride 0); structured and typed ones elements.
    let raw = matches!(kind & 0xff, KIND_CBV | KIND_SRV_RAW | KIND_UAV_RAW);
    let (address, len, stride) = match view {
        View::Constant(address, size) => (*address, u64::from(*size), 0),
        View::Sampler(_) | View::ConsoleSampler(_) => return None,
        // A whole buffer without a description: all of it.
        View::Shader(r, None) | View::Unordered(r, _, None) => {
            let (data, len) = super::resource::buffer_bytes(*r as This)?;
            (data as u64, len, 0)
        }
        v => {
            let (address, len) = host::buffer_range(v)?;
            (address, len, if raw { 0 } else { host::view_stride_of(v) })
        }
    };
    super::page_table::touch(address, len);
    let records = if stride > 0 { len / u64::from(stride) } else { len };
    // Past 20 bits: the null raw SRV (slot 2), as the spec says.
    let host_view = if index < 1 << 20 { index } else { KIND_SRV_RAW };
    Some([
        address as u32,
        (address >> 32) as u32 & 0xFFFF | (stride & 0x3FFF) << 16,
        records.min(u64::from(u32::MAX)) as u32,
        PAGE_TAG << 20 | host_view,
        kind,
        i,
        generation,
        VIEW_TAG,
    ])
}

/// V# records written (buffer views), for the page-table line.
pub(crate) static V_SHARPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A description of `N` words, if the title passes one.
fn words<const N: usize>(desc: *const [u32; N]) -> Option<[u32; N]> {
    // SAFETY: the title passes a description or null.
    (!desc.is_null()).then(|| unsafe { desc.read_unaligned() })
}

// View descriptions as the title passes them hold stack garbage in their
// padding and unused union words, so equal views compared different: each
// write made a new view entry, and each placed view a new placed texture
// (VIctory, e7b2182, `canonical_*`; GPL-3.0-or-later). The layout
// checks are theirs.
const _: () = {
    use std::mem::{offset_of, size_of};
    use windows::Win32::Graphics::Direct3D12::*;
    assert!(size_of::<D3D12_SHADER_RESOURCE_VIEW_DESC>() == 40);
    assert!(offset_of!(D3D12_SHADER_RESOURCE_VIEW_DESC, Anonymous) == 16);
    assert!(size_of::<D3D12_BUFFER_SRV>() == 24);
    assert!(size_of::<D3D12_TEX2D_SRV>() == 16);
    assert!(size_of::<D3D12_TEX2D_ARRAY_SRV>() == 24);
    assert!(size_of::<D3D12_UNORDERED_ACCESS_VIEW_DESC>() == 40);
    assert!(offset_of!(D3D12_UNORDERED_ACCESS_VIEW_DESC, Anonymous) == 8);
    assert!(size_of::<D3D12_BUFFER_UAV>() == 32);
    assert!(size_of::<D3D12_TEX2D_UAV>() == 8);
    assert!(size_of::<D3D12_TEX2D_ARRAY_UAV>() == 16);
};

/// Zeroes union alignment padding (`d[3]`) and unused union tail words of a
/// raw `D3D12_SHADER_RESOURCE_VIEW_DESC` word array.
fn canonical_srv_desc(mut d: [u32; 10]) -> [u32; 10] {
    d[3] = 0;
    let keep = match d[1] {
        1 => 9,  // Buffer (d[9] is trailing struct padding)
        2 => 7,  // Texture1D
        3 => 9,  // Texture1DArray
        4 => 8,  // Texture2D
        5 => 10, // Texture2DArray
        6 => 4,  // Texture2DMS (UnusedField_NothingToDefine only)
        7 => 6,  // Texture2DMSArray
        8 => 7,  // Texture3D
        9 => 7,  // TextureCube
        10 => 9, // TextureCubeArray
        11 => 6, // RaytracingAccelerationStructure
        _ => 10,
    };
    d[keep..].fill(0);
    d
}

/// Zeroes unused union tail words (and `Buffer`'s trailing padding word) of a
/// raw `D3D12_UNORDERED_ACCESS_VIEW_DESC` word array. The union begins at
/// offset 8 (`d[2]`) with no leading padding.
fn canonical_uav_desc(mut d: [u32; 10]) -> [u32; 10] {
    let keep = match d[1] {
        1 => 9, // Buffer (d[9] is trailing struct padding)
        2 => 3, // Texture1D
        3 => 5, // Texture1DArray
        4 => 4, // Texture2D
        5 => 6, // Texture2DArray
        6 => 2, // Texture2DMS (UnusedField_NothingToDefine only)
        7 => 4, // Texture2DMSArray
        8 => 5, // Texture3D
        _ => 10,
    };
    d[keep..].fill(0);
    d
}

/// A placed view description with its leading view description made canonical.
fn canonical_placed_desc(mut desc: [u32; 16], canonical: fn([u32; 10]) -> [u32; 10]) -> [u32; 16] {
    let head = canonical(std::array::from_fn(|i| desc[i]));
    desc[..10].copy_from_slice(&head);
    desc
}

#[test]
fn canonical_views_drop_padding() {
    let mut srv = [0x55u32; 10];
    srv[1] = 4; // Texture2D: words 0-7 count, 3 is padding
    let c = canonical_srv_desc(srv);
    assert_eq!((c[3], c[8], c[9], c[2], c[7]), (0, 0, 0, 0x55, 0x55));
    let mut uav = [0x55u32; 10];
    uav[1] = 4;
    assert_eq!(canonical_uav_desc(uav)[4..], [0; 6]);
    let r = super::resource::canonical_resource_desc([7; 14]);
    assert_eq!((r[0], r[1], r[2], r[12], r[13]), (7, 0, 7, 7, 0));
}

/// Device method 0x88.
pub(crate) extern "system" fn create_cbv(_: This, desc: *const [u32; 4], handle: u64) {
    if let Some([lo, hi, size, _]) = words(desc) {
        let address = u64::from(lo) | u64::from(hi) << 32;
        record(handle, View::Constant(address, size));
    }
}

/// Device method 0x90.
pub extern "system" fn create_srv(_: This, resource: This, desc: *const [u32; 10], handle: u64) {
    super::texture::view_made(resource as usize);
    record(handle, View::Shader(resource as usize, words(desc).map(canonical_srv_desc)));
}

/// Device method 0x98.
pub(crate) extern "system" fn create_uav(_: This, resource: This, counter: This, desc: *const [u32; 10], handle: u64) {
    super::texture::view_made(resource as usize);
    let view = View::Unordered(resource as usize, counter as usize, words(desc).map(canonical_uav_desc));
    record(handle, view);
}

/// Device methods 0x230 and 0x238: shader resource and unordered access
/// views of memory the title placed itself, `(resource desc, view desc,
/// handle)`.
pub extern "system" fn create_placed_srv(_: This, resource: *const [u32; 14], desc: *const [u32; 16], handle: u64) {
    if let (Some(resource), Some(desc)) = (words(resource), words(desc)) {
        let resource = super::resource::canonical_resource_desc(resource);
        let desc = canonical_placed_desc(desc, canonical_srv_desc);
        record(handle, View::Placed(false, resource, desc));
    }
}

pub(crate) extern "system" fn create_placed_uav(_: This, resource: *const [u32; 14], desc: *const [u32; 16], handle: u64) {
    if let (Some(resource), Some(desc)) = (words(resource), words(desc)) {
        let resource = super::resource::canonical_resource_desc(resource);
        let desc = canonical_placed_desc(desc, canonical_uav_desc);
        record(handle, View::Placed(true, resource, desc));
    }
}

/// Device method 0xb0.
pub(crate) extern "system" fn create_sampler(_: This, desc: *const [u32; 13], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::Sampler(desc));
    }
}

/// Device method 0x1e8.
pub(crate) extern "system" fn create_sampler_x(_: This, desc: *const [u32; 27], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::ConsoleSampler(desc));
    }
}

/// Device method 0xc0: `count` descriptors of `kind` from `src` to `dst`.
pub(crate) extern "system" fn copy_simple(_: This, count: u32, dst: u64, src: u64, kind: u32) {
    let step = u64::from(increment(kind).unwrap_or(0));
    let len = u64::from(count) * step;
    if dst != 0 && src != 0 {
        TRAFFIC[COPIED].fetch_add(u64::from(count), Ordering::Relaxed);
        // SAFETY: both ranges are descriptor slots the title owns.
        unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len as usize) };
        for i in 0..u64::from(count) {
            mirror(dst + i * step);
        }
        log_traffic();
    }
}

/// Device method 0xb8: ranges of descriptors (start and count each)
/// gathered from the sources and scattered to the destinations, in order.
pub(crate) extern "system" fn copy(
    _: This,
    dst_ranges: u32,
    dst_starts: *const u64,
    dst_sizes: *const u32,
    src_ranges: u32,
    src_starts: *const u64,
    src_sizes: *const u32,
    kind: u32,
) {
    let Some(step) = increment(kind) else {
        return;
    };
    // Each range as (start, count); a null size array means one descriptor
    // per range.
    let ranges = |n: u32, starts: *const u64, sizes: *const u32| -> Vec<(u64, u32)> {
        (0..n as usize)
            // SAFETY: the title passes `n` starts and sizes (or null sizes).
            .map(|i| unsafe { (*starts.add(i), if sizes.is_null() { 1 } else { *sizes.add(i) }) })
            .collect()
    };
    if dst_starts.is_null() || src_starts.is_null() {
        return;
    }
    let src = ranges(src_ranges, src_starts, src_sizes);
    let mut dst = ranges(dst_ranges, dst_starts, dst_sizes).into_iter();
    let (mut to, mut room) = dst.next().unwrap_or((0, 0));
    for (mut from, mut left) in src {
        while left > 0 {
            while room == 0 {
                let Some(next) = dst.next() else { return };
                (to, room) = next;
            }
            let n = left.min(room);
            copy_simple(std::ptr::null(), n, to, from, kind);
            let bytes = u64::from(n) * u64::from(step);
            (to, from, room, left) = (to + bytes, from + bytes, room - n, left - n);
        }
    }
}

/// The view recorded at `handle`, if one is.
pub(crate) fn rtv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, RTV_TAG)
}

/// The depth view recorded at `handle`, if one is.
pub(crate) fn dsv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, DSV_TAG)
}

fn tagged_record(handle: u64, tag: u32) -> Option<RtvRecord> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let record = unsafe { (handle as *const RtvRecord).read_unaligned() };
    (record.tag == tag).then_some(record)
}

/// Device method 0x70.
pub(crate) extern "system" fn create_heap(_: This, desc_ptr: *const HeapDesc, iid: *const Iid, out: *mut This) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc_ptr.as_ref() }) else {
        return E_INVALIDARG;
    };
    let Some(size) = increment(desc.kind).map(|i| u64::from(i) * u64::from(desc.count.max(1))) else {
        return E_INVALIDARG;
    };
    // SAFETY: read only as a hint: a title that passes the plain
    // description has its own stack or data after it.
    let placement = unsafe { desc_ptr.add(1).cast::<Placement>().read_unaligned() };
    // shortcut: a placement is recognised by its size matching the count,
    // since the plain and extended descriptions share the slot and the
    // interface version; find the flag that tells them apart if a title
    // trips over this.
    let placed = placement.address != 0 && placement.size == size;
    let visible = desc.flags & SHADER_VISIBLE != 0 && matches!(desc.kind, VIEWS_KIND | SAMPLERS);
    let gpu_memory = (!placed && visible).then(|| host::own_window(size)).flatten();
    let (start, memory) = if placed {
        (placement.address, None)
    } else if let Some((address, _)) = &gpu_memory {
        (*address, None)
    } else {
        match Memory::zeroed(size, 64) {
            Ok(memory) => (memory.address(), Some(memory)),
            Err(hr) => return hr,
        }
    };
    let gpu_memory = gpu_memory.map(|(_, r)| r);
    xnw_core::info!(
        "d3d12x: descriptor heap kind {} count {} flags {:#x} at {start:#x}{}{}",
        desc.kind,
        desc.count,
        desc.flags,
        if placed { " (title memory)" } else { "" },
        if visible { " shader-visible" } else { "" }
    );
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
        start,
        _memory: memory,
        _gpu_memory: gpu_memory,
        host: std::sync::OnceLock::new(),
        pending: Default::default(),
    });
    if visible {
        let at = std::ptr::from_ref::<Heap>(&heap) as usize;
        let mut heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
        heaps.push((start, start + size, at));
    }
    object::hand_out(DESCRIPTOR_HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(DESCRIPTOR_HEAP, iid, object::is(iid, &HEAP_IIDS).then_some(this), out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}
extern "system" fn desc(this: This, out: *mut HeapDesc) -> *mut HeapDesc {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(heap(this).desc) };
    }
    out
}
/// The same address for the CPU and the GPU start, returned in a register
/// (as the console's heaps do).
extern "system" fn start(this: This) -> u64 {
    heap(this).start
}

/// IUnknown 3, object 4, device child 1, heap 3.
pub(crate) const SLOTS: &[Slot] = &[
    (0, query as *const (), Done, "descriptor.rs:694"),
    (2, release as *const (), Done, "descriptor.rs:695"),
    (7, object::parent_device as *const (), Done, "descriptor.rs:696"),
    (8, desc as *const (), Done, "descriptor.rs:697"),
    (9, start as *const (), Done, "descriptor.rs:698"),
    (10, start as *const (), Done, "descriptor.rs:699"),
];
pub(crate) const LEN: usize = 24;
static FUNCTIONS: Table<LEN> = object::table::<DESCRIPTOR_HEAP, LEN>(true, SLOTS);

// Descriptor traffic (ours): what the title writes, for the run log.
const RECORDED: usize = 0;
const COPIED: usize = 1;
const MIRRORED: usize = 2;
const TARGETS: usize = 3;
const DISTINCT: usize = 4;
static TRAFFIC: [AtomicU64; 5] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

/// Views recorded, descriptors copied, of both those in shader-visible
/// heaps, target views recorded, distinct views alive.
pub(crate) fn traffic() -> [u64; 5] {
    std::array::from_fn(|i| TRAFFIC[i].load(Ordering::Relaxed))
}

/// One line each time views recorded plus descriptors copied pass a power
/// of two from 1024 on: the title's descriptor traffic without a line per
/// write.
fn log_traffic() {
    static NEXT: AtomicU64 = AtomicU64::new(1024);
    let [recorded, copied, mirrored, targets, distinct] = traffic();
    let total = recorded + copied;
    let next = NEXT.load(Ordering::Relaxed);
    if total >= next && NEXT.compare_exchange(next, next * 2, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
        let (made, reused) = host::plan_counts();
        let extended = host::plans_extended();
        let [srv, uav, rtv, dsv, refused] = std::array::from_fn::<u64, 5, _>(|i| TYPELESS[i].load(Ordering::Relaxed));
        xnw_core::info!(
            "d3d12x: descriptors: {recorded} views recorded ({distinct} distinct), {copied} copied, {mirrored} into shader-visible heaps, {targets} target views; bindless plans {made} made, {extended} extended, {reused} reused; views remapped from typeless: srv {srv}, uav {uav}, rtv {rtv}, dsv {dsv}; views refused {refused}"
        );
    }
}

/// Host views remapped from a typeless format (or none) to a concrete
/// one, by kind ([`TYPELESS_SRV`]...), and views refused (a format of
/// another family than the resource's, or no format the host can take).
pub(crate) static TYPELESS: [AtomicU64; 5] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
pub(crate) const TYPELESS_SRV: usize = 0;
pub(crate) const TYPELESS_UAV: usize = 1;
pub(crate) const TYPELESS_RTV: usize = 2;
pub(crate) const TYPELESS_DSV: usize = 3;
pub(crate) const VIEWS_REFUSED: usize = 4;

/// Counts one view in [`TYPELESS`] slot `kind`.
pub(crate) fn count_typeless(kind: usize) {
    TYPELESS[kind].fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::d3d12x::object::S_OK;

    /// The base's safe `slot` (ours is `unsafe`): tests call our objects only.
    fn slot<F: Copy>(object: This, offset: usize) -> F {
        // SAFETY: tests pass objects of ours with the slot's signature.
        unsafe { crate::d3d12x::object::slot(object, offset) }
    }

    /// What newer titles pass; tests use it throughout, so that reading the
    /// placement stays inside the value.
    #[repr(C)]
    struct Extended(HeapDesc, Placement);

    fn create(desc: HeapDesc, placement: Placement, out: &mut This) -> HRESULT {
        let extended = Extended(desc, placement);
        create_heap(std::ptr::null(), &extended.0, &HEAP_IIDS[0], out)
    }

    #[test]
    fn heaps_hold_count_descriptors_from_their_start() {
        let null = std::ptr::null();
        let desc = HeapDesc { kind: 0, count: 1000, flags: 1, nodes: 0 };
        let none = Placement { address: 0, size: 0 };
        let mut h = null;
        assert_eq!(create(desc, none, &mut h), S_OK);
        let cpu = slot::<extern "system" fn(This) -> u64>(h, 0x48)(h);
        assert_eq!(cpu % 64, 0);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), cpu);
        assert_eq!(increment_size(null, 0), 32);
        // SAFETY: the heap holds 1000 descriptors of 32 bytes.
        unsafe { (cpu as *mut u8).add(999 * 32 + 31).write(1) };
        assert_eq!(increment_size(null, 9), 0);
        let bad = HeapDesc { kind: 9, ..desc };
        assert_eq!(create(bad, none, &mut h), E_INVALIDARG);
    }

    #[test]
    fn views_survive_copies() {
        let null = std::ptr::null();
        let mut heap = vec![0u64; 4 * 8]; // four 32-byte slots
        let at = |i: usize| heap.as_ptr() as u64 + 32 * i as u64;
        let (s0, s1, s2, s3) = (at(0), at(1), at(2), at(3));
        create_cbv(null, &[0x1000, 1, 256, 0], s0);
        assert_eq!(view(s0), Some(View::Constant(0x1_0000_1000, 256)));
        create_srv(null, null, std::ptr::null(), s1);
        assert_eq!(view(s1), Some(View::Shader(0, None)));
        // The same view twice is one entry.
        create_srv(null, null, std::ptr::null(), s2);
        // SAFETY: both slots are in `heap`.
        assert_eq!(unsafe { *(s1 as *const [u32; 8]) }, unsafe { *(s2 as *const [u32; 8]) });
        // Two ranges of one into one range of two.
        let (dst, dst_n, src, src_n) = ([s2], [2u32], [s0, s1], [1u32, 1]);
        copy(null, 1, dst.as_ptr(), dst_n.as_ptr(), 2, src.as_ptr(), src_n.as_ptr(), 0);
        assert_eq!(view(s2), view(s0));
        assert_eq!(view(s3), view(s1));
        assert_eq!(view(at(0) + 8), None);
        heap[0] = 0;
    }

    /// The record layout for GPU-side descriptors:
    /// dword 0 the host heap index, 1-2 entry and generation, 3 the tag, 4
    /// the kind; copies carry it; samplers fill 16 bytes.
    #[test]
    fn records_carry_host_index_and_kind() {
        let null = std::ptr::null();
        let slots = [0u32; 8 * 3];
        let at = |i: usize| slots.as_ptr() as u64 + 32 * i as u64;
        create_cbv(null, &[0x2000, 1, 256, 0], at(0));
        // SAFETY: slots of `slots`.
        let r = unsafe { *(at(0) as *const [u32; 8]) };
        assert_eq!(r[3], VIEW_TAG, "tag in dword 3: the title tests dword 1 itself");
        assert_eq!(r[0], host::TABLE_NULLS + r[1], "index = nulls + entry (dword 1)");
        assert_eq!(r[4], KIND_CBV);
        assert_eq!(r[5..], [u32::MAX; 3], "not in place");
        // A typed buffer view of memory, as a placed SRV (Buffer dimension).
        let mut desc = [0u32; 16];
        desc[1] = 1;
        desc[6] = 4; // elements
        desc[10] = 0x3000;
        let mut resource = [0u32; 14];
        resource[0] = 1;
        create_placed_srv(null, &resource, &desc, at(1));
        // SAFETY: as above.
        assert_eq!(unsafe { *((at(1) + 16) as *const u32) }, KIND_SRV_TYPED);
        desc[8] = 1; // raw
        create_placed_srv(null, &resource, &desc, at(2));
        // SAFETY: as above.
        assert_eq!(unsafe { *((at(2) + 16) as *const u32) }, KIND_SRV_RAW);
        // A byte copy keeps all of it.
        copy_simple(null, 1, at(2), at(0), 0);
        // SAFETY: as above.
        assert_eq!(unsafe { *(at(2) as *const [u32; 8]) }, r);
        assert_eq!(host::table_index(host::VIEW_TABLE + 5, KIND_UAV_RAW), KIND_UAV_RAW, "past the table: the null of its kind");
    }

    /// A released resource's views go stale; the same view of a new
    /// resource at the same address is a new record.
    #[test]
    fn views_die_with_their_resource() {
        let null = std::ptr::null();
        let slots = [0u64; 2 * 4];
        let (a, b) = (slots.as_ptr() as u64, slots.as_ptr() as u64 + 32);
        let resource = 0x7777_0000usize as This;
        create_srv(null, resource, std::ptr::null(), a);
        assert_eq!(view(a), Some(View::Shader(0x7777_0000, None)));
        forget(0x7777_0000);
        assert_eq!(view(a), None, "stale after release");
        create_srv(null, resource, std::ptr::null(), b);
        assert_eq!(view(b), Some(View::Shader(0x7777_0000, None)));
        // SAFETY: both slots are in `slots`.
        let (old, new) = unsafe { (*(a as *const [u32; 4]), *(b as *const [u32; 4])) };
        assert_ne!(old, new, "a new generation");
    }

    /// The title places its shader-visible heap in its own memory
    /// and checks that the heap starts there.
    #[test]
    fn placed_heaps_start_at_the_title_memory() {
        let mut memory = vec![0u8; 64 * 32];
        let address = memory.as_mut_ptr() as u64;
        let desc = HeapDesc { kind: 0, count: 64, flags: 1, nodes: 1 };
        let mut h = std::ptr::null();
        assert_eq!(create(desc, Placement { address, size: 64 * 32 }, &mut h), S_OK);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), address);
        // A size that does not match the count is not a placement.
        assert_eq!(create(desc, Placement { address, size: 7 }, &mut h), S_OK);
        assert_ne!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
    }

    /// Ours: render target and depth views are records of their own kind,
    /// read back with their description.
    #[test]
    fn target_views_are_recorded_by_kind() {
        let null = std::ptr::null();
        let slots = vec![0u8; 2 * 128];
        let (rtv, dsv) = (slots.as_ptr() as u64, slots.as_ptr() as u64 + 128);
        let resource = 0x5555_0000usize as This;
        create_rtv(null, resource, std::ptr::null(), rtv);
        create_dsv(null, resource, &[45, 3, 0, 0, 0, 0], dsv);
        assert_eq!(rtv_record(rtv).map(|r| r.resource), Some(resource));
        assert!(dsv_record(rtv).is_none() && rtv_record(dsv).is_none());
        let depth = dsv_record(dsv).and_then(|r| r.depth_desc()).unwrap();
        assert_eq!((depth.Format.0, depth.ViewDimension.0), (45, 3));
    }
}
