//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.
//!
//! Ported from VIctory-base (branch `xnw/base` @ 5d0516e = VIctory d2a4bae
//! + PRs #135 #413 #552 #581 #638 #681 and fix-up 5d0516e),
//! `crates/victory-gpu/src/draw.rs` 1-1324, GPL-3.0-or-later.
//! Changed: `tracing` -> `xnw_core`; the `diagnostics` feature's code
//! (draw traces, op texts, producers, captures) left out; `dispatch_mesh`
//! runs the compute fallback, indirect mesh dispatches with their group
//! counts on the GPU (`dispatch_mesh_gpu`, after VIctory @ 13d3718, see
//! `crate::mesh`; `XNW_MESH_INDIRECT=cpu` reads them when recorded);
//! `VICTORY_DROP_UNBOUND_UAV` read as `XNW_DROP_UNBOUND_UAV`.

use std::sync::atomic::{AtomicU64, Ordering};

use xnw_shader::{ResourceType, RootType, UsedResource};

use super::{descriptor, pipeline, resource, shadow, texture};
use descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use super::host::HostList;
use super::object::This;
use pipeline::host::HostPipeline;

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: usize,
    /// `XNW_DEPTH_LOG`: the depth view handle, first viewport and
    /// scissor the title set.
    depth_handle: u64,
    viewport: D3D12_VIEWPORT,
    scissor: RECT,
    /// A draw ran into the bound targets since they were bound (for the
    /// target capture).
    drew: bool,
    /// The first bound colour target's size (for the world detection).
    rt_size: (u32, u32),
    /// (pipeline, whether it writes depth) for the world detection.
    depth_write: (usize, bool),
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: (table, what it holds) â†’ first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u64), u32>,
    /// The dynamic slots the list holds: title heap, first, count; given
    /// back when the list is reset or released.
    dynamic: Vec<(std::sync::Arc<descriptor::host::HostHeap>, u32, u32)>,
    /// The bytes behind each root SRV of the last compute bind, in the
    /// program's order (a mesh fallback program's lengths).
    root_lengths: Vec<u32>,
    /// Root SRV tables made in this list: (host heap, title address) to
    /// the dynamic slot holding the view.
    root_srv_slots: std::collections::HashMap<(usize, u64), u32>,
    /// Address maps made in this list (VIctory): host view
    /// heap, what the map was made from, its host GPU address; and the
    /// dynamic slots their views hold (given back as `dynamic`'s).
    maps: Vec<(usize, super::address_map::Key, u64)>,
}

impl Drop for State {
    fn drop(&mut self) {
        // VIctory's case: given back to the host heap they
        // came from, held here, not looked up by the title heap's address,
        // which a heap made after the title released this one may reuse
        // (its allocator then took back slots it never gave out).
        for (host, first, n) in &self.dynamic {
            host.release(*first, *n);
        }
    }
}

/// Counters (totals since start): root UAVs bound, draws and dispatches
/// bound with the address map, with the descriptor words.
static ROOT_UAVS: AtomicU64 = AtomicU64::new(0);
static WITH_MAP: AtomicU64 = AtomicU64::new(0);
static WITH_WORDS: AtomicU64 = AtomicU64::new(0);
/// Draws and dispatches bound with the page table; buffer views of
/// page-table programs left without a host-copy refresh.
static PAGE_TABLE_BINDS: AtomicU64 = AtomicU64::new(0);
static PAGED_VIEWS: AtomicU64 = AtomicU64::new(0);

/// The "frame page table" line's text.
pub(crate) fn page_table_text() -> String {
    format!(
        "programs bound {} (gated {}), draws/dispatches bound {}, buffer views read through it only {}, V# records {}; {}",
        pipeline::host::PAGE_TABLE_PROGRAMS.load(Ordering::Relaxed),
        pipeline::host::GATED[3].load(Ordering::Relaxed),
        PAGE_TABLE_BINDS.load(Ordering::Relaxed),
        PAGED_VIEWS.load(Ordering::Relaxed),
        descriptor::V_SHARPS.load(Ordering::Relaxed),
        super::page_table::stats_text()
    )
}

/// The frame statistics line's text.
pub(crate) fn addresses_text() -> String {
    format!(
        "gates: {}; root UAVs bound {}; draws/dispatches with the address map {}, with descriptor words {}; maps: {}",
        pipeline::host::gate_text(),
        ROOT_UAVS.load(Ordering::Relaxed),
        WITH_MAP.load(Ordering::Relaxed),
        WITH_WORDS.load(Ordering::Relaxed),
        super::address_map::stats_text()
    )
}

/// Counts a draw or dispatch of title pipeline `pipeline` dropped for want
/// of a host pipeline, by gate when it was gated.
fn count_gated(pipeline: usize) {
    if let Some(g) = pipeline::gate_of(pipeline as This) {
        pipeline::host::GATED_DROPS[g].fetch_add(1, Ordering::Relaxed);
    }
}

/// The host address of a root UAV at title address `address`: in place
/// where it resolves (`XNW_SHARED=2`), else its buffer's host copy, whose
/// range is then declared written by shaders (copied back after the
/// submission, not refreshed over meanwhile), as table UAVs are.
/// shortcut: as [`root_view`], a root view has no size: 1 MiB, cut at its
/// buffer's end.
fn root_uav(list: &HostList, address: u64) -> Option<u64> {
    let shadow = shadow::find(address, 1)?;
    let len = shadow.left(address).min(1 << 20);
    if let Some(r) = crate::shared::range(crate::shared::VIEW_ROOT, address, len) {
        list.keep(&r.buffer);
        return Some(descriptor::host::shared_va(&r));
    }
    shadow.refresh(list, address, len);
    shadow.gpu_writes(list, address, len);
    Some(shadow.gpu(address))
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// `XNW_SHOW=rt<i>[:<w>x<h>]` or `ds[:<w>x<h>]`: present the last
/// render target bound at slot `i` (or the depth target) of that size in
/// place of the title's planes: the scene before lighting and composite
/// passes, for a world those passes do not draw yet.
fn show_spec() -> Option<(bool, usize, Option<(u64, u32)>)> {
    static SPEC: std::sync::LazyLock<Option<(bool, usize, Option<(u64, u32)>)>> = std::sync::LazyLock::new(|| {
        let v = std::env::var("XNW_SHOW").ok()?;
        let (slot, size) = match v.trim().split_once(':') {
            Some((s, z)) => (s.to_owned(), Some(z.to_owned())),
            None => (v.trim().to_owned(), None),
        };
        let (depth, index) = if slot == "ds" { (true, 0) } else { (false, slot.strip_prefix("rt")?.parse().ok()?) };
        let size = match size {
            Some(z) => {
                let (w, h) = z.split_once('x')?;
                Some((w.parse().ok()?, h.parse().ok()?))
            }
            None => None,
        };
        xnw_core::info!("d3d12x: XNW_SHOW: presenting {} {index} {size:?} in place of the title's planes", if depth { "depth" } else { "render target" });
        Some((depth, index, size))
    });
    *SPEC
}

/// The title resource `XNW_SHOW` presents (0: none yet).
static SHOWN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn show_candidate(depth: bool, index: usize, resource: usize, texture: &ID3D12Resource) {
    let Some((d, i, size)) = show_spec() else { return };
    if d != depth || i != index {
        return;
    }
    if let Some((w, h)) = size {
        // SAFETY: a live texture.
        let desc = unsafe { texture.GetDesc() };
        if (desc.Width, desc.Height) != (w, h) {
            return;
        }
    }
    SHOWN.store(resource, Ordering::Relaxed);
}

/// The title resource to present instead of the planes (`XNW_SHOW`).
// shortcut: a debug switch; the title's render targets live as long as
// the world, and a released one would be read here.
pub(crate) fn shown_target() -> Option<This> {
    show_spec()?;
    let r = SHOWN.load(Ordering::Relaxed);
    (r != 0).then_some(r as This)
}

thread_local! {
    /// Texture UAVs bound for the dispatch being prepared, in a target
    /// capture frame.
    static UAV_TEXTURES: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Indirect rectangle-list draws dropped.
static RECT_INDIRECT: AtomicU64 = AtomicU64::new(0);
/// Rectangle-list draws with and without the rectangle geometry shader.
static RECT_GS: AtomicU64 = AtomicU64::new(0);
static RECT_NO_GS: AtomicU64 = AtomicU64::new(0);
/// Rectangle variants bound, and the frame of the first (a world marker
/// for `XNW_CAPTURE_TARGETS=rect+<k>`).
static RECT_VARIANT_DRAWS: AtomicU64 = AtomicU64::new(0);
pub(crate) static RECT_VARIANT_FRAME: AtomicU64 = AtomicU64::new(u64::MAX);

/// Record bytes one mesh draw may use; groups past it are cut.
const MESH_RECORDS_PER_DRAW: u64 = 64 << 20;

/// The key of a list's mesh arena in its first-use set.
const ARENA_KEY: usize = usize::MAX - 0x6d65;
/// The key of the indirect mesh ring's emptying in a list's first-use set.
const RING_KEY: usize = usize::MAX - 0x7269;

/// The most groups one indirect mesh dispatch runs (a hard ceiling; a
/// count the culling pass left as garbage cannot run for seconds). Counted
/// on the GPU as clamped, and logged with the frame statistics.
const MESH_INDIRECT_MAX_GROUPS: u32 = 4096;

/// Indirect mesh dispatches run with their counts on the GPU, and read on
/// the processor (`XNW_MESH_INDIRECT=cpu`), since the last frame line.
static MESH_INDIRECT: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Indirect mesh records and counts traced in frame `MESH_TRACED_FRAME`
/// (debug line), and indirect mesh dispatches dropped for want of memory
/// since the last line.
static MESH_TRACED: AtomicU64 = AtomicU64::new(0);
static MESH_TRACED_FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
static MESH_NO_MEMORY: AtomicU64 = AtomicU64::new(0);

/// `XNW_MESH_INDIRECT=cpu`: indirect mesh dispatches read their group
/// counts when recorded (the older way).
fn mesh_indirect_cpu() -> bool {
    static CPU: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_MESH_INDIRECT").is_ok_and(|v| v.eq_ignore_ascii_case("cpu")));
    *CPU
}

static WRITER: std::sync::OnceLock<Option<crate::mesh::IndirectWriter>> = std::sync::OnceLock::new();

/// The indirect mesh writer and its ring on the host device, made once.
fn mesh_writer() -> Option<&'static crate::mesh::IndirectWriter> {
    WRITER
        .get_or_init(|| match crate::mesh::IndirectWriter::new(super::host::host_device()?, crate::mesh::RING_BYTES) {
            Ok(w) => Some(w),
            Err(e) => {
                xnw_core::warn!("d3d12x: no indirect mesh writer: indirect mesh dispatches dropped ({e})");
                None
            }
        })
        .as_ref()
}

/// The indirect mesh part of the frame's calls line, per frame over `n`
/// frames: dispatches with GPU counts (and, as the writer counted them on
/// the GPU, those it ran, clamped, left empty, the groups, and those the
/// ring had no room for), and those read on the processor.
pub(crate) fn mesh_indirect_text(n: u64) -> String {
    static LAST: std::sync::Mutex<[u32; 5]> = std::sync::Mutex::new([0; 5]);
    let n = n.max(1);
    let per = |c: &AtomicU64| (c.swap(0, Ordering::Relaxed) + n / 2) / n;
    let (gpu, cpu) = (per(&MESH_INDIRECT[0]), per(&MESH_INDIRECT[1]));
    // Not made here: only once a dispatch asked for it.
    let now = WRITER.get().and_then(Option::as_ref).map_or([0; 5], crate::mesh::IndirectWriter::stats);
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let d: Vec<u64> = now.iter().zip(last.iter()).map(|(a, b)| (u64::from(a.wrapping_sub(*b)) + n / 2) / n).collect();
    *last = now;
    let (max, split) = (MESH_MAX_GROUPS.swap(0, Ordering::Relaxed), per(&MESH_SPLIT));
    let (null, x, oom) = (per(&MESH_NULL_VIEWS), per(&MESH_X_INDIRECT), per(&MESH_NO_MEMORY));
    if d[1] > 0 {
        xnw_core::warn!("d3d12x: indirect mesh dispatches cut to {MESH_INDIRECT_MAX_GROUPS} groups (or the ring's room): {} a frame", d[1]);
    }
    format!(
        "indirect mesh dispatches {gpu} with GPU counts (writer: {} run, {} clamped, {} empty, {} groups, {} without ring room; {oom} dropped for memory; {null} with a null descriptor in the program's tables; {x} through DispatchMeshX), {cpu} read on the processor; direct mesh dispatches: most groups {max}, {split} split into pieces",
        d[0], d[1], d[2], d[3], d[4]
    )
}

/// Indirect mesh dispatches whose program's tables name a slot with no
/// view (VIctory), and `DispatchMeshX` calls with indirect arguments,
/// since the last frame line.
static MESH_NULL_VIEWS: AtomicU64 = AtomicU64::new(0);
static MESH_X_INDIRECT: AtomicU64 = AtomicU64::new(0);

/// Direct mesh dispatches (and indirect ones read on the processor) since
/// the last frame line: the most groups one asked for, and how many ran in
/// pieces.
static MESH_MAX_GROUPS: AtomicU64 = AtomicU64::new(0);
static MESH_SPLIT: AtomicU64 = AtomicU64::new(0);

/// `bytes` of argument room in the list's arena, writable.
fn arg_room(list: &HostList, bytes: u64) -> Option<crate::mesh::ArgsSlot> {
    let gpu = super::host::host_device()?;
    let reset = list.first_use(ARENA_KEY);
    let key = std::ptr::from_ref(list) as usize;
    crate::mesh::with_arena(gpu, key, reset, |a| a.args_room(gpu, &list.list, bytes).ok())
}

/// After our pass filled `room`: readable as arguments; its buffer, kept
/// by the list.
fn arg_written(list: &HostList, room: crate::mesh::ArgsSlot) -> windows::Win32::Graphics::Direct3D12::ID3D12Resource {
    let key = std::ptr::from_ref(list) as usize;
    let gpu = super::host::host_device().expect("an arena was made on it");
    let b = crate::mesh::with_arena(gpu, key, false, |a| a.args_written(&list.list, room));
    list.keep(&b);
    b
}
/// The host address of a dword holding `first`, the first console group of
/// a mesh fallback piece (compute `t0`): for 0 (every dispatch not split,
/// and every indirect one) a zero made once for the process (nothing
/// allocated per draw); for a later piece of a split dispatch, the list's
/// scratch memory.
fn first_group(list: &HostList, first: u32) -> Option<u64> {
    if first == 0 {
        struct Zero(windows::Win32::Graphics::Direct3D12::ID3D12Resource);
        // SAFETY: D3D12 resources are free-threaded.
        unsafe impl Send for Zero {}
        // SAFETY: as above.
        unsafe impl Sync for Zero {}
        static ZERO: std::sync::OnceLock<Option<Zero>> = std::sync::OnceLock::new();
        let zero = ZERO.get_or_init(|| {
            let (buffer, data) = super::host::host()?.upload_buffer(256)?;
            // SAFETY: the buffer's own mapping, 256 bytes.
            unsafe { std::ptr::write_bytes(data, 0, 256) };
            Some(Zero(buffer))
        });
        // SAFETY: a live buffer.
        return zero.as_ref().map(|z| unsafe { z.0.GetGPUVirtualAddress() });
    }
    let Some((gpu, data)) = list.scratch(4) else {
        xnw_core::warn!("d3d12x: no scratch memory for a mesh piece's first group: mesh draw dropped");
        return None;
    };
    // SAFETY: four bytes of the list's mapped scratch memory, ours.
    unsafe { data.cast::<u32>().write(first) };
    Some(gpu)
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Draws and dispatches dropped for want of a host pipeline, by kind.
pub(crate) const NO_PIPELINE_KINDS: [&str; 5] = ["draw", "indexed", "indirect", "mesh", "dispatch"];
const NP_DRAW: usize = 0;
const NP_INDEXED: usize = 1;
const NP_INDIRECT: usize = 2;
const NP_MESH: usize = 3;
const NP_DISPATCH: usize = 4;
pub(crate) static NO_PIPELINE: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// A drop of kind `kind`: counted if the pipeline has no host pipeline
/// made (rather than another reason). Does not wait.
fn no_pipeline_if(pipeline: usize, kind: usize) {
    if !pipeline::host_pipeline_made(pipeline as This) {
        NO_PIPELINE[kind].fetch_add(1, Ordering::Relaxed);
    }
}

fn no_pipeline(kind: usize) {
    NO_PIPELINE[kind].fetch_add(1, Ordering::Relaxed);
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory.
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB,
/// other views as 1 MiB (both cut at their buffer's end).
fn root_view(list: &HostList, address: u64, srv: bool) -> Option<(u64, u64, &'static str)> {
    crate::shared::zero_copy_root(false, address as usize);
    // On the console a root view is any GPU address. Outside the
    // title's buffer objects (1,911 of 1,913 root views a frame)
    // the copy is of the title memory around it
    // (`shadow::find`'s aligned windows, refreshed at submission), now
    // asked for the whole span bound (1 MiB / 64 KiB), not 1 byte: a
    // window ending before it left the shader reading past its copy.
    let want = if srv { 1 << 20 } else { 1 << 16 };
    let found = if shadow::in_buffer(address, 1) { shadow::find(address, 1) } else { shadow::find(address, want) };
    let Some(shadow) = found else {
        // Nothing to read there: a zeroed buffer of our own.
        set_backing(zero_buffer().map(|b| (b, 0, ZERO_VIEW)));
        return zero_view().map(|g| (g, ZERO_VIEW, "zeroed dummy"));
    };
    let len = shadow.left(address).min(if srv { 1 << 20 } else { 1 << 16 });
    // In place where the range resolves (and a constant buffer lands
    // 256-byte aligned, which it does where the title's address is).
    if let Some(r) = crate::shared::range(crate::shared::VIEW_ROOT, address, len) {
        let gpu = descriptor::host::shared_va(&r);
        if srv || gpu.is_multiple_of(256) {
            list.keep(&r.buffer);
            // SAFETY: a live buffer.
            let left = unsafe { r.buffer.GetDesc() }.Width.saturating_sub(r.offset);
            set_backing(Some((r.buffer.clone(), r.offset, left)));
            return Some((gpu, left, "in place"));
        }
        crate::shared::misaligned(crate::shared::VIEW_ROOT);
    }
    let gpu = shadow.gpu(address);
    if srv || gpu % 256 == 0 {
        crate::shared::zero_copy_note(crate::shared::ZC_ROOT, address as usize, if srv { shadow.left(address).min(16 << 20) } else { len } as usize);
        // A root SRV's table views the copy to its end: up to 16 MiB
        // of it brought up to date (vertex streams reach past 1 MiB).
        shadow.refresh(list, address, if srv { shadow.left(address).min(16 << 20) } else { len });
        set_backing(Some((shadow.buffer.clone(), shadow.offset(address), shadow.left(address))));
        return Some((gpu, shadow.left(address), "host copy"));
    }
    let (gpu, to) = list.scratch(len)?;
    list.defer_fill(to, address as *const u8, len as usize);
    Some((gpu, len, "upload copy"))
}

thread_local! {
    /// What the last [`root_view`] bound (buffer, offset, bytes to its
    /// end), for a root SRV made a one-view table.
    static BACKING: std::cell::RefCell<Option<(ID3D12Resource, u64, u64)>> = const { std::cell::RefCell::new(None) };
}

fn set_backing(b: Option<(ID3D12Resource, u64, u64)>) {
    BACKING.with(|c| *c.borrow_mut() = b);
}

/// The backing of the last root view bound on this thread (its host GPU
/// address `gpu` is checked against it).
fn backing_of(gpu: u64) -> Option<(ID3D12Resource, u64, u64)> {
    let b = BACKING.with(|c| c.borrow_mut().take())?;
    // SAFETY: a live buffer.
    (unsafe { b.0.GetGPUVirtualAddress() } + b.1 == gpu).then_some(b)
}

/// The zeroed buffer of [`ZERO_VIEW`] bytes as a resource.
fn zero_buffer() -> Option<ID3D12Resource> {
    zero_view()?;
    ZERO.get().and_then(|z| z.as_ref()).map(|z| z.0.clone())
}

/// Root SRVs bound as one-view tables: views of their backing, null views
/// (not on 16 bytes, or no backing), and draws dropped (no view heap or
/// slot), since the last frame line.
static ROOT_TABLES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Bytes of the zeroed buffer a root view with nothing behind it reads.
const ZERO_VIEW: u64 = 64 << 10;

struct Zero(ID3D12Resource);
// SAFETY: D3D12 resources are free-threaded.
unsafe impl Send for Zero {}
// SAFETY: as above.
unsafe impl Sync for Zero {}
static ZERO: std::sync::OnceLock<Option<Zero>> = std::sync::OnceLock::new();

/// The zeroed buffer of [`ZERO_VIEW`] bytes, made once.
fn zero_view() -> Option<u64> {
    let z = ZERO.get_or_init(|| {
        let (buffer, data) = super::host::host()?.upload_buffer(ZERO_VIEW)?;
        // SAFETY: the buffer's own mapping.
        unsafe { std::ptr::write_bytes(data, 0, ZERO_VIEW as usize) };
        Some(Zero(buffer))
    });
    // SAFETY: a live buffer.
    z.as_ref().map(|z| unsafe { z.0.GetGPUVirtualAddress() })
}

/// Root views bound, outside every buffer the title made, with less than
/// 64 KiB behind them, and bound to the zeroed dummy, since the last frame
/// line.
static ROOT_VIEWS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// `XNW_ROOT_LOG=1`: each root view bound (the first 4096) with what is
/// behind it.
fn root_log() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_ROOT_LOG").is_ok_and(|v| v == "1"));
    *ON
}

/// The root view part of the frame's calls line, per frame over `n`.
pub(crate) fn root_views_text(n: u64) -> String {
    let n = n.max(1);
    let v: Vec<u64> = ROOT_VIEWS.iter().map(|c| (c.swap(0, Ordering::Relaxed) + n / 2) / n).collect();
    let c: Vec<u64> = ROOT_CLASSES.iter().map(|c| (c.swap(0, Ordering::Relaxed) + n / 2) / n).collect();
    let w: Vec<u64> = ROOT_TABLES.iter().map(|c| (c.swap(0, Ordering::Relaxed) + n / 2) / n).collect();
    format!(
        "root views {} bound ({} outside the title's buffers, {} with less than 64 KiB behind them, {} on the zeroed dummy); by resolution: in a title buffer {} in place, {} host copy, {} upload copy; outside buffers {} in place, {} host copy of its 4 MiB window, {} upload copy; outside buffers by memory: {} paired CPU view, {} paired GPU view, {} title region, {} other; root SRV tables ({}): {} sized views, {} null (not on 16 bytes), {} draws dropped (no view heap or slot)",
        v[0], v[1], v[2], v[3], c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], if pipeline::host::bounded_root_srvs() { "on" } else { "off" }, w[0], w[1], w[2]
    )
}

/// Root views by how they resolved (in a title buffer: in place, host
/// copy, upload copy; outside: the same three), and those outside buffers
/// by the memory they are in (the paired reservation's CPU or GPU view, a
/// title region xmem knows, other).
static ROOT_CLASSES: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

/// Counts how a root view at `address` resolved (`how` from
/// [`root_view`]); the first 16 outside the title's buffers are logged.
fn root_class(address: u64, outside: bool, how: &str) {
    let base = if outside { 3 } else { 0 };
    let i = match how {
        "in place" => Some(base),
        "host copy" => Some(base + 1),
        "upload copy" => Some(base + 2),
        _ => None,
    };
    if let Some(i) = i {
        ROOT_CLASSES[i].fetch_add(1, Ordering::Relaxed);
    }
    if !outside {
        return;
    }
    let at = address as usize;
    let (k, kind) = if super::watch::gpu_alias(at).is_some() {
        (6, "the paired reservation's CPU view")
    } else if super::watch::paired(at) {
        (7, "the paired reservation's GPU view")
    } else if crate::shared::region_of(at).is_some() {
        (8, "a title region")
    } else {
        (9, "memory xmem does not know")
    };
    ROOT_CLASSES[k].fetch_add(1, Ordering::Relaxed);
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if LOGGED.fetch_add(1, Ordering::Relaxed) < 16 {
        xnw_core::info!("d3d12x: root view at {address:#x} outside the title's buffers, in {kind}: {how}");
    }
}
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| xnw_core::warn!("d3d12x: {}", format_args!($($arg)*)));
    }};
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    /// The bound pipeline's name if XNW_LATE_LOG=1 and it is a
    /// culling program (see [super::late]).
    fn late_name(&self) -> Option<String> {
        if !super::late::on() {
            return None;
        }
        pipeline::host_pipeline(self.pipeline as This).map(|p| p.name.to_string()).filter(|n| super::late::wanted(n))
    }

    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`super::primitive`]).
    /// A value D3D12 does not define is never forwarded (five-star:
    /// the console's 17 passed through hung the GPU at the next list's start,
    /// with no validation message); draws under it are dropped and counted
    /// ([`State::topology_fits`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        let Some(host) = host_topology(topology) else {
            once!("console topology {topology} has no host equivalent: its draws dropped");
            return;
        };
        if matches!(topology, TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP) {
            once!("console topology {topology} drawn as host topology {host}");
        }
        // SAFETY: an open list; a topology D3D12 defines.
        unsafe { list.list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32)) };
    }

    /// Whether a draw with pipeline `p` may run under the title's
    /// topology: one D3D12 defines, of the pipeline's topology type. Counts
    /// the draw by console topology, and the drop when not.
    fn topology_fits(&self, p: &HostPipeline) -> bool {
        TOPOLOGY_DRAWS[(self.topology as usize).min(TOPOLOGY_DRAWS.len() - 1)].fetch_add(1, Ordering::Relaxed);
        let Some(host) = host_topology(self.topology) else {
            TOPOLOGY_DROPPED[0].fetch_add(1, Ordering::Relaxed);
            return false;
        };
        let kind = topology_type(host);
        if (1..=4).contains(&p.topology) && kind != p.topology as u32 {
            TOPOLOGY_DROPPED[1].fetch_add(1, Ordering::Relaxed);
            once!("a draw of topology {} (type {kind}) with a pipeline of topology type {}: dropped ({})", self.topology, p.topology, p.name);
            return false;
        }
        true
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute { &mut self.compute } else { &mut self.graphics }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        let _call = super::counters::call(super::counters::ROOT_VIEW);
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let _call = super::counters::call(super::counters::ROOT_CONSTANTS);
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self.vertex.iter_mut().skip(start as usize).take(count as usize) {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = super::host::host() else {
            return;
        };
        self.snapshot_targets(list);
        self.targets.clear();
        self.rt_size = (0, 0);
        list.crumbs().note(|| format!("targets: {} colour, depth {}", handles.len(), if depth == 0 { "none".to_string() } else { format!("{depth:#x}") }));
        let mut rtvs = Vec::new();
        for &h in handles {
            let Some(record) = descriptor::rtv_record(h) else {
                once!("render target without a recorded view (handle {h:#x})");
                continue;
            };
            let Some(texture) = resource::host_resource(record.resource) else {
                continue;
            };
            if self.targets.is_empty() {
                // SAFETY: a live texture.
                let d = unsafe { texture.GetDesc() };
                self.rt_size = (d.Width as u32, d.Height);
            }
            self.targets.push(record.resource as usize);
            show_candidate(false, self.targets.len() - 1, record.resource as usize, texture);
            rtvs.push(host.rtv(texture, record.desc().as_ref()));
        }
        self.depth = 0;
        self.depth_handle = depth;
        let dsv = descriptor::dsv_record(depth).and_then(|record| {
            let texture = resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            show_candidate(true, 0, record.resource as usize, texture);
            Some(host.dsv(texture, record.depth_desc().as_ref()))
        });
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// Whether the current draw writes depth: a depth target bound and the
    /// pipeline's depth writes on (cached per pipeline).
    fn depth_writes(&mut self) -> bool {
        if self.depth == 0 {
            return false;
        }
        if self.depth_write.0 != self.pipeline {
            let on = pipeline::graphics_state(self.pipeline as This).is_some_and(|g| g.depth.DepthEnable.as_bool() && g.depth.DepthWriteMask.0 != 0);
            self.depth_write = (self.pipeline, on);
        }
        self.depth_write.1
    }

    /// `XNW_CULL_DUMP`: a culling dispatch's bindings, as recorded and
    /// (later) as executed; `indirect` its argument address.
    fn cull_dump(&self, list: &HostList, p: &HostPipeline, indirect: Option<u64>) {
        let Some(stage) = p.stages.first() else { return };
        let Some(name) = stage.shader.pack_name() else { return };
        if !name.contains("Cull") {
            return;
        }
        let used = &stage.translated().used;
        let constants: Vec<(u8, Vec<u32>)> = self
            .compute
            .iter()
            .enumerate()
            .filter_map(|(i, r)| match r {
                Root::Constants(c) => Some((i as u8, c.clone())),
                _ => None,
            })
            .collect();
        let how = indirect.map_or_else(|| "direct".to_owned(), |a| format!("indirect, arguments at {a:#x}"));
        let Some(g) = super::capture::cull_begin(&name, &how, &constants) else { return };
        if let Some(a) = indirect {
            super::capture::cull_bytes(g, "args", a, 12);
        }
        for (n, u) in used.iter().enumerate() {
            let kind = match u.kind {
                ResourceType::Cbv => "cbv",
                ResourceType::Uav => "uav",
                _ => "srv",
            };
            let label = format!("{kind}{n} slot {}", u.slot);
            match (u.root, self.compute.get(usize::from(u.slot))) {
                (RootType::Table, Some(&Root::Table(table))) => {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    let Some(view) = descriptor::view(handle) else { continue };
                    if let Some((address, len)) = descriptor::host::buffer_range(&view) {
                        super::capture::cull_bytes(g, &label, address, len);
                    } else if let View::Shader(resource, _) | View::Unordered(resource, _, _) = view
                        && let Some(t) = resource::host_resource(resource as This)
                    {
                        super::capture::cull_texture(list, g, resource as This, t, &label);
                    }
                }
                (_, Some(&Root::View(address))) => super::capture::cull_bytes(g, &label, address, if u.kind == ResourceType::Cbv { 576 } else { 256 }),
                _ => {}
            }
        }
    }

    /// With `XNW_CAPTURE_TARGETS` in its frame: copies of the bound targets
    /// if a draw went into them.
    pub(crate) fn snapshot_targets(&mut self, list: &HostList) {
        if !std::mem::replace(&mut self.drew, false) || !super::capture::targets_wanted() {
            return;
        }
        let bound: Vec<(usize, String)> = self.targets.iter().enumerate().map(|(i, &r)| (r, format!("rt{i}"))).chain((self.depth != 0).then(|| (self.depth, "ds".to_owned()))).collect();
        let pass = pipeline::host_pipeline(self.pipeline as This).map_or_else(|| "?".to_owned(), |p| p.name.clone());
        for (r, label) in bound {
            if let Some(t) = resource::host_resource(r as This) {
                super::capture::snapshot(list, r as This, t, &label, &pass);
            }
        }
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!("descriptor table outside the bound heaps (handle {handle:#x})");
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => c.get(dword as usize).copied().unwrap_or(0),
            // Runtime-provided: a root descriptor's console
            // address, dword 0 its low half, 1 its high half; its pages are
            // met for the page table (to its buffer's end, at most 16 MiB,
            // or one page outside buffers).
            (Some(&Root::View(address)), RootType::Descriptor) => {
                if dword == 0 {
                    let len = shadow::buffer_left(address).unwrap_or(1).min(16 << 20);
                    super::page_table::touch(address, len);
                }
                if dword & 1 == 0 { address as u32 } else { (address >> 32) as u32 }
            }
            // The loop counts' heap index (`loopcount.rs`'s contract).
            (_, RootType::Other(super::loopcount::ROOT)) => {
                (self.heaps[0] != 0).then(|| descriptor::heap(self.heaps[0] as This).host().and_then(|h| h.loop_slot())).flatten().unwrap_or(0)
            }
            _ => {
                once!("a shader reads a root argument the title did not set (slot {slot}, {root:?})");
                0
            }
        }
    }

    /// XNW: draws and dispatches bound with GPU-side descriptors, and
    /// those dropped because a table was not in a shared window.
    pub(crate) fn gpu_counts() -> (u64, u64, u64) {
        (GPU_DRAWS.load(Ordering::Relaxed), NO_WINDOW.load(Ordering::Relaxed), GPU_REMADE.load(Ordering::Relaxed))
    }

    /// XNW: for a table root argument (or the bound view or sampler
    /// heap), the shared-window raw view and byte offset of its records, to
    /// the end of the heap holding them.
    fn window_value(&self, compute: bool, slot: u8, root: RootType) -> Option<[u32; 2]> {
        let (start, heap) = match (root, self.roots_ref(compute).get(usize::from(slot))) {
            (RootType::Table, Some(Root::Table(handle))) => (*handle, self.heap_of(*handle)?),
            (RootType::ResourceHeap, _) => (descriptor::heap(self.heaps[0] as This).start, descriptor::heap(self.heaps[0] as This)),
            (RootType::SamplerHeap, _) => (descriptor::heap(self.heaps[1] as This).start, descriptor::heap(self.heaps[1] as This)),
            _ => return None,
        };
        if matches!(root, RootType::ResourceHeap | RootType::SamplerHeap) && self.heaps[usize::from(root == RootType::SamplerHeap)] == 0 {
            return None;
        }
        // The table's first record must lie in a window; the shader's raw
        // view is the whole window buffer, so records after it read there too.
        descriptor::host::window_of(start, heap.end().checked_sub(start)?.min(32))
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static descriptor::Heap> {
        self.heaps.iter().filter(|&&h| h != 0).map(|&h| descriptor::heap(h as This)).find(|h| h.index(handle).is_some())
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        let Some(p) = pipeline::host_pipeline(self.pipeline as This) else {
            count_gated(self.pipeline);
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!("a draw or mesh dispatch with another kind of pipeline (mesh {mesh})");
            return false;
        }
        // Mesh draws set their own triangle list.
        if !mesh && !self.topology_fits(p) {
            return false;
        }
        if !self.bind(list, p, false) {
            return false;
        }
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            if let Some(r) = crate::shared::range(crate::shared::VIEW_VERTEX, v.address, u64::from(v.size)) {
                list.keep(&r.buffer);
                *view = D3D12_VERTEX_BUFFER_VIEW { BufferLocation: descriptor::host::shared_va(&r), SizeInBytes: v.size, StrideInBytes: v.stride };
                continue;
            }
            let Some(shadow) = shadow::find(v.address, u64::from(v.size)) else {
                // No host memory there: the view stays null (reads 0).
                VIEWS_UNTRANSLATED[0].fetch_add(1, Ordering::Relaxed);
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            *view = D3D12_VERTEX_BUFFER_VIEW { BufferLocation: shadow.gpu(v.address), SizeInBytes: v.size, StrideInBytes: v.stride };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            if v.address == 0 || v.size == 0 {
                // No index buffer bound: nothing to translate.
                VIEWS_UNTRANSLATED[1].fetch_add(1, Ordering::Relaxed);
                return false;
            }
            let location = match crate::shared::range(crate::shared::VIEW_INDEX, v.address, u64::from(v.size)) {
                Some(r) => {
                    list.keep(&r.buffer);
                    descriptor::host::shared_va(&r)
                }
                None => {
                    let Some(shadow) = shadow::find(v.address, u64::from(v.size)) else {
                        VIEWS_UNTRANSLATED[1].fetch_add(1, Ordering::Relaxed);
                        return false;
                    };
                    shadow.refresh(list, v.address, u64::from(v.size));
                    shadow.gpu(v.address)
                }
            };
            let view = D3D12_INDEX_BUFFER_VIEW { BufferLocation: location, SizeInBytes: v.size, Format: DXGI_FORMAT(v.format as i32) };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        const DEPTH_WRITE: u32 = 0x10;
        for &t in &self.targets {
            super::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            super::list::ensure_state(list, self.depth as This, DEPTH_WRITE);
        }
        self.depth_log(p);
        true
    }

    pub(crate) fn set_viewports(&mut self, list: &HostList, items: &[D3D12_VIEWPORT]) {
        self.viewport = items.first().copied().unwrap_or_default();
        viewports(list, items);
    }

    pub(crate) fn set_scissors(&mut self, list: &HostList, rects: &[RECT]) {
        self.scissor = rects.first().copied().unwrap_or_default();
        scissors(list, rects);
    }

    /// `XNW_DEPTH_LOG=1`: in the first 3 world frames and every 64th
    /// after (`capture::depth_log_frame`), the first 20 draws into colour
    /// targets: programs, the depth view (title resource, host texture
    /// format and size, view format, dimension, flags), the pipeline's depth
    /// state and format, depth clip, viewport and scissor.
    fn depth_log(&self, p: &HostPipeline) {
        static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_DEPTH_LOG").is_ok_and(|v| v == "1"));
        static SEEN: std::sync::Mutex<(u64, u32)> = std::sync::Mutex::new((u64::MAX, 0));
        if !*ON || self.targets.is_empty() {
            return;
        }
        // The first 3 world frames (by the capture's world start), then
        // every 64th.
        let frame = super::queue::presented();
        if !super::capture::depth_log_frame(frame) {
            return;
        }
        let n = {
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            if seen.0 != frame {
                *seen = (frame, 0);
            }
            seen.1 += 1;
            seen.1
        };
        if n > 20 {
            return;
        }
        let ds = match (descriptor::dsv_record(self.depth_handle), resource::host_resource(self.depth as This)) {
            (Some(record), Some(texture)) => {
                // SAFETY: a live texture.
                let t = unsafe { texture.GetDesc() };
                let asked = record.depth_desc();
                let used = super::host::depth_view(&t, asked);
                let view = |d: Option<D3D12_DEPTH_STENCIL_VIEW_DESC>| d.map_or_else(|| "none".to_owned(), |d| format!("f{} dim {} flags {:#x}", d.Format.0, d.ViewDimension.0, d.Flags.0));
                format!("{:#x} host f{} {}x{}x{} flags {:#x}; view asked {}, used {}", self.depth, t.Format.0, t.Width, t.Height, t.DepthOrArraySize, t.Flags.0, view(asked), view(used))
            }
            _ if self.depth_handle == 0 => "none bound".to_owned(),
            _ => format!("handle {:#x} without a recorded view or host texture", self.depth_handle),
        };
        let state = match pipeline::graphics_state(self.pipeline as This) {
            Some(g) => {
                let d = &g.depth;
                let f = &d.FrontFace;
                format!(
                    "depth {} write {} func {} stencil {} (front {}/{}/{}/{}) dsv f{} clip {} cull {} bias {}",
                    d.DepthEnable.as_bool(),
                    d.DepthWriteMask.0,
                    d.DepthFunc.0,
                    d.StencilEnable.as_bool(),
                    f.StencilFailOp.0,
                    f.StencilDepthFailOp.0,
                    f.StencilPassOp.0,
                    f.StencilFunc.0,
                    g.depth_format.0,
                    g.rasterizer.DepthClipEnable.as_bool(),
                    g.rasterizer.CullMode.0,
                    g.rasterizer.DepthBias
                )
            }
            None => "no graphics state".to_owned(),
        };
        let (v, s) = (&self.viewport, &self.scissor);
        xnw_core::info!(
            "depth log: frame {frame} draw {n} ({}{} targets): {}; ds {ds}; {state}; viewport {}x{} at {},{} depth {}..{}; scissor {},{}..{},{}",
            if p.mesh { "mesh, " } else { "" },
            self.targets.len(),
            p.name,
            v.Width,
            v.Height,
            v.TopLeftX,
            v.TopLeftY,
            v.MinDepth,
            v.MaxDepth,
            s.left,
            s.top,
            s.right,
            s.bottom
        );
    }

    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        let l = &list.list;
        list.crumbs().note(|| format!("bind {} {} (root signature, heaps, root arguments)", if compute { "compute" } else { "graphics" }, p.name));
        if compute {
            self.root_lengths.clear();
        }
        // XNW: with XNW_DESC=gpu every program reads the title's
        // descriptor records and indexes the one global view and sampler
        // heap; no per-title host heaps, tables or bindless records.
        let gpu = descriptor::host::gpu_mode();
        if gpu {
            const GLOBAL: [usize; 2] = [usize::MAX; 2];
            if self.host_heaps != GLOBAL {
                let Some((views, samplers)) = descriptor::host::gpu_heaps() else {
                    once!("XNW_DESC=gpu without its global heaps: draws dropped");
                    return false;
                };
                let heaps = [Some(views.heap.clone()), Some(samplers.heap.clone())];
                heaps.iter().flatten().for_each(|h| list.keep(h));
                // SAFETY: an open list and shader-visible heaps.
                unsafe { l.SetDescriptorHeaps(&heaps) };
                self.host_heaps = GLOBAL;
                self.host_roots = [0; 2];
            }
        } else {
            // VIctory's check: compared by the host heaps
            // themselves, not the title's heap pointers: a heap released
            // and another made at its address kept the old host heap bound.
            let heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            let mut ids = [0usize; 2];
            for (i, h) in heaps.iter().flatten().enumerate().take(2) {
                ids[i] = windows::core::Interface::as_raw(h) as usize;
            }
            if self.host_heaps != ids {
                heaps.iter().flatten().for_each(|h| list.keep(h));
                // SAFETY: an open list and shader-visible heaps.
                unsafe { l.SetDescriptorHeaps(&heaps) };
                self.host_heaps = ids;
                self.host_roots = [0; 2];
            }
        }
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(&p.pso);
            list.keep(&p.root);
        }
        if !gpu
            && p.stages.iter().any(|s| s.translated().resource_heap)
            && self.heaps[0] != 0
            && let Some(host_heap) = descriptor::heap(self.heaps[0] as This).host_arc()
        {
            host_heap.prepare_bindless_record(list);
            let writes = p.stages.iter().any(|s| s.translated().writes_memory);
            list.defer_bindless(&host_heap, writes);
        }
        for stage in &p.stages {
            let t = stage.translated();
            let made = if gpu {
                // XNW: the shader indexes the global heap itself, but
                // what a table's views read must still be current: textures
                // the title wrote uploaded, buffer copies refreshed, states
                // made readable (as `tables` does through `table_entry`).
                self.gpu_readable(list, &t.used, compute, stage.host_copies);
                Some(Vec::new())
            } else {
                self.tables(list, &t.used, compute)
            };
            let Some(made) = made else {
                // Without its fresh table the shader would read the title's
                // slots, which for constant buffer tables hold no constant
                // view: a texture or null descriptor read as a buffer gives a
                // wild address on AMD and the device is removed (VIctory
                // 3c81420). The draw or dispatch is dropped instead.
                static DROPPED_TABLES: AtomicU64 = AtomicU64::new(0);
                let n = DROPPED_TABLES.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    xnw_core::warn!("d3d12x: a draw or dispatch dropped: no dynamic slots for its tables ({n} so far)");
                }
                return false;
            };
            if let Some((param, dwords)) = stage.user_data {
                let mut values = vec![0u32; dwords as usize];
                for u in &t.host.user_data {
                    if u.dword >= xnw_shader::emit::WINDOW_DWORD {
                        // The window view and byte offset of the table's
                        // records (GPU-side descriptors).
                        let Some(w) = self.window_value(compute, u.slot, u.root) else {
                            let n = NO_WINDOW.fetch_add(1, Ordering::Relaxed) + 1;
                            if n.is_power_of_two() {
                                xnw_core::warn!("d3d12x: a draw or dispatch dropped: its descriptor table is not in a shared window ({n} so far)");
                            }
                            return false;
                        };
                        values[u.index as usize] = w[(u.dword - xnw_shader::emit::WINDOW_DWORD) as usize & 1];
                        continue;
                    }
                    let fresh = made.iter().find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None => self.user_value(compute, u.slot, u.root, u.dword),
                    };
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            for (views, srv) in [(&stage.root_cbvs, false), (&stage.root_srvs, true)] {
                for &(slot, param) in views {
                    let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) else {
                        once!("a shader reads a root view the title did not set (slot {slot}, srv {srv})");
                        return false;
                    };
                    if let Some(name) = self.late_name() {
                        list.note_late(super::late::recorded(if srv { "root SRV" } else { "root CBV" }, &name, address, 256));
                    }
                    let Some((gpu, left, how)) = root_view(list, address, srv) else {
                        once!("a shader reads a root view outside title buffers (slot {slot}, srv {srv}, address {address:#x})");
                        return false;
                    };
                    list.crumbs().note(|| format!("root {} slot {slot}: title {address:#x} -> {how}, GPU {gpu:#x}, {left} bytes behind", if srv { "SRV" } else { "CBV" }));
                    // What is behind the view, counted (and logged).
                    let outside = !shadow::in_buffer(address, 1);
                    ROOT_VIEWS[0].fetch_add(1, Ordering::Relaxed);
                    ROOT_VIEWS[1].fetch_add(u64::from(outside), Ordering::Relaxed);
                    ROOT_VIEWS[2].fetch_add(u64::from(left < 64 << 10), Ordering::Relaxed);
                    ROOT_VIEWS[3].fetch_add(u64::from(how == "zeroed dummy"), Ordering::Relaxed);
                    root_class(address, outside, how);
                    if root_log() {
                        static LOGGED: AtomicU64 = AtomicU64::new(0);
                        if LOGGED.fetch_add(1, Ordering::Relaxed) < 4096 {
                            xnw_core::info!("d3d12x: root {} slot {slot} at {address:#x}: {how}, {left} bytes behind it{} ({})", if srv { "srv" } else { "cbv" }, if outside { ", outside the title's buffers" } else { "" }, p.name);
                        }
                    }
                    if srv && compute {
                        self.root_lengths.push(left.min(u64::from(u32::MAX)) as u32);
                    }
                    if srv && pipeline::host::bounded_root_srvs() {
                        if !self.root_srv_table(list, address, param, compute, gpu) {
                            return false;
                        }
                        continue;
                    }
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, srv) {
                            (false, false) => l.SetGraphicsRootConstantBufferView(param, gpu),
                            (false, true) => l.SetGraphicsRootShaderResourceView(param, gpu),
                            (true, false) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, true) => l.SetComputeRootShaderResourceView(param, gpu),
                        }
                    }
                }
            }
            // XNW: the runtime stride table of the bound view heap
            // (VIctory draw.rs).
            if let Some(param) = stage.stride_table {
                // XNW: with GPU-side descriptors, the global heap's table.
                let heap = if gpu {
                    descriptor::host::gpu_heaps().map(|h| h.0)
                } else {
                    (self.heaps[0] != 0).then(|| descriptor::heap(self.heaps[0] as This).host_arc()).flatten()
                };
                let gpu = heap.as_ref().and_then(|h| h.stride_table()).map(|(table, gpu)| {
                    list.keep(table);
                    gpu
                });
                let Some(gpu) = gpu else {
                    once!("a shader reads the runtime stride table without a host view heap");
                    return false;
                };
                // SAFETY: an open list; the table lives as long as the heap.
                unsafe {
                    if compute {
                        l.SetComputeRootShaderResourceView(param, gpu);
                    } else {
                        l.SetGraphicsRootShaderResourceView(param, gpu);
                    }
                }
            }
            if !self.bind_addresses(list, stage, compute, gpu) {
                return false;
            }
        }
        true
    }

    /// A stage's root UAVs, address map and descriptor words; false
    /// if one cannot be bound (the draw or dispatch is dropped then: the
    /// shader would read or write through an unbound root parameter).
    fn bind_addresses(&mut self, list: &HostList, stage: &pipeline::host::Stage, compute: bool, gpu: bool) -> bool {
        let l = &list.list;
        for &(slot, param) in &stage.root_uavs {
            let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) else {
                once!("a shader writes a root UAV the title did not set (slot {slot})");
                return false;
            };
            let Some(va) = root_uav(list, *address) else {
                once!("a shader writes a root UAV outside title buffers (slot {slot}, address {address:#x})");
                return false;
            };
            ROOT_UAVS.fetch_add(1, Ordering::Relaxed);
            // SAFETY: an open list; the address is in a live buffer the list holds.
            unsafe {
                if compute {
                    l.SetComputeRootUnorderedAccessView(param, va);
                } else {
                    l.SetGraphicsRootUnorderedAccessView(param, va);
                }
            }
        }
        // The bound heaps' sizes and guard slots; a
        // heap that is not there reads as 0 slots, so every index is guarded.
        if let Some(param) = stage.heap_bounds {
            let (views, samplers) = if gpu {
                descriptor::host::gpu_heaps().map_or((None, None), |(v, s)| (Some(v), Some(s)))
            } else {
                let host = |h: usize| (h != 0).then(|| descriptor::heap(h as This).host_arc()).flatten();
                (host(self.heaps[0]), host(self.heaps[1]))
            };
            let w = descriptor::host::bounds_words(views.as_deref(), samplers.as_deref());
            // SAFETY: an open list; 8 constants for an 8-dword parameter.
            unsafe {
                if compute {
                    l.SetComputeRoot32BitConstants(param, 8, w.as_ptr().cast(), 0);
                } else {
                    l.SetGraphicsRoot32BitConstants(param, 8, w.as_ptr().cast(), 0);
                }
            }
        }
        if stage.address_map.is_none() && stage.descriptor_words.is_none() && stage.page_table.is_none() {
            return true;
        }
        // The address map and the descriptor words belong to the bound view
        // heap (VIctory draw.rs): the global one with XNW_DESC=gpu.
        let heap = if gpu {
            descriptor::host::gpu_heaps().map(|h| h.0)
        } else {
            (self.heaps[0] != 0).then(|| descriptor::heap(self.heaps[0] as This).host_arc()).flatten()
        };
        let Some(heap) = heap else {
            once!("a draw or dispatch dropped: it reads the address map or descriptor words without a host view heap");
            return false;
        };
        let mut extra = Vec::new();
        // The page table, its entries naming slots of this heap.
        if let Some(param) = stage.page_table {
            let Some(va) = super::page_table::bind(list, &heap) else {
                once!("a draw or dispatch dropped: no page table");
                return false;
            };
            PAGE_TABLE_BINDS.fetch_add(1, Ordering::Relaxed);
            extra.push((param, va));
        }
        if let Some(param) = stage.address_map {
            let at = std::sync::Arc::as_ptr(&heap) as usize;
            let key = super::address_map::key();
            let made = self.maps.iter().find(|m| m.0 == at && m.1 == key).map(|m| m.2);
            let va = match made {
                Some(va) => Some(va),
                None => super::address_map::make(list, &heap).map(|(va, (first, n))| {
                    if n > 0 {
                        self.dynamic.push((heap.clone(), first, n));
                    }
                    self.maps.push((at, key, va));
                    va
                }),
            };
            let Some(va) = va else {
                once!("a draw or dispatch dropped: no address map for its view heap");
                return false;
            };
            WITH_MAP.fetch_add(1, Ordering::Relaxed);
            extra.push((param, va));
        }
        if let Some(param) = stage.descriptor_words {
            // XNW_DESC=gpu: records hold host heap indices, not the console's
            // words; the gate keeps such programs out (no host pipeline).
            let Some(va) = heap.words().filter(|_| !gpu) else {
                once!("a draw or dispatch dropped: no descriptor words for its view heap");
                return false;
            };
            // (The words live as long as the heap, which lives for the run.)
            WITH_WORDS.fetch_add(1, Ordering::Relaxed);
            extra.push((param, va));
        }
        for (param, va) in extra {
            // SAFETY: an open list; the buffers live as long as it runs.
            unsafe {
                if compute {
                    l.SetComputeRootShaderResourceView(param, va);
                } else {
                    l.SetGraphicsRootShaderResourceView(param, va);
                }
            }
        }
        true
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        let Some(p) = pipeline::host_pipeline(self.pipeline as This) else {
            once!("a dispatch without a host pipeline dropped (pipeline {:#x})", self.pipeline);
            DROPPED.fetch_add(1, Ordering::Relaxed);
            no_pipeline(NP_DISPATCH);
            count_gated(self.pipeline);
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        if self.drop_unbound_uav(p) {
            once!("a compute dispatch with unbound UAVs dropped (pipeline {:#x})", self.pipeline);
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        if !self.bind(list, p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        // SAFETY: an open list with everything set.
        let _crumb = super::crumbs::begin_compute(list, || format!("dispatch {x}x{y}x{z}"));
        unsafe { list.list.Dispatch(x, y, z) };
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
        if super::capture::cull_dump_frame() {
            self.cull_dump(list, p, None);
        }
        // In target-capture frames, the textures the dispatch writes.
        let written = UAV_TEXTURES.with(|u| std::mem::take(&mut *u.borrow_mut()));
        if !written.is_empty() && super::capture::targets_wanted() {
            for (i, r) in written.into_iter().enumerate() {
                if let Some(t) = resource::host_resource(r as This) {
                    super::capture::snapshot(list, r as This, t, &format!("uav{i}"), &p.name);
                }
            }
        }
    }

    /// Whether any UAV table entry required by `p` is unbound or has no view.
    fn has_unbound_uav(&self, p: &HostPipeline) -> bool {
        p.stages.iter().any(|stage| {
            stage.translated().used.iter().any(|u| {
                if u.kind == ResourceType::Uav
                    && u.root == RootType::Table
                    && let Some(Root::Table(table)) = self.roots_ref(true).get(usize::from(u.slot))
                {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    return descriptor::view(handle).is_none();
                }
                false
            })
        })
    }

    /// Binds a root SRV as a one-view table
    /// ([`pipeline::host::bounded_root_srvs`]): a raw view, in a dynamic slot
    /// of the bound view heap, from the root view's host address `gpu` (what
    /// [`root_view`] chose: the in-place window, the host copy, the zeroed
    /// dummy) to the end of that buffer; a null view (zeros) where the
    /// address is not on 16 bytes, as raw views must start. One slot per
    /// address per list, given back with the list's others. False (the draw
    /// is dropped) when no view heap is bound or no slot is free.
    /// Ported from victory-fork @ 4e17478 `crates/victory-gpu/src/draw.rs`
    /// `root_srv_table` (1177-1250), GPL-3.0-or-later; changed: the backing
    /// is found by its host address among ours (window, copy, dummy).
    fn root_srv_table(&mut self, list: &HostList, address: u64, param: u32, compute: bool, gpu: u64) -> bool {
        let heap = if descriptor::host::gpu_mode() {
            descriptor::host::gpu_heaps().map(|h| h.0)
        } else {
            (self.heaps[0] != 0).then(|| descriptor::heap(self.heaps[0] as This).host_arc()).flatten()
        };
        let Some(heap) = heap else {
            once!("a root SRV without a bound view heap: the draw is dropped");
            ROOT_TABLES[2].fetch_add(1, Ordering::Relaxed);
            return false;
        };
        let key = (std::sync::Arc::as_ptr(&heap) as usize, address);
        let index = match self.root_srv_slots.get(&key) {
            Some(&i) => i,
            None => {
                let Some(index) = heap.dynamic(1) else {
                    ROOT_TABLES[2].fetch_add(1, Ordering::Relaxed);
                    return false;
                };
                self.dynamic.push((heap.clone(), index, 1));
                self.root_srv_slots.insert(key, index);
                match backing_of(gpu) {
                    Some((buffer, offset, bytes)) if offset.is_multiple_of(16) => {
                        list.keep(&buffer);
                        heap.raw_at(index, &buffer, offset, bytes);
                        ROOT_TABLES[0].fetch_add(1, Ordering::Relaxed);
                    }
                    _ => {
                        once!("a root SRV not on 16 bytes of its buffer (or of no buffer of ours) reads zeros ({address:#x})");
                        heap.null_raw_at(index);
                        ROOT_TABLES[1].fetch_add(1, Ordering::Relaxed);
                    }
                }
                index
            }
        };
        let handle = heap.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }
    /// After the compute half's bind, the bytes behind each of its root
    /// SRVs, where the program reads them.
    fn set_root_lengths(&self, list: &HostList, f: &pipeline::host::MeshFallback) {
        let Some((param, n)) = f.lengths else {
            return;
        };
        let mut v = self.root_lengths.clone();
        v.resize(n as usize, 0);
        // SAFETY: an open list with the compute half's root signature; `v`
        // covers the parameter.
        unsafe { list.list.SetComputeRoot32BitConstants(param, n, v.as_ptr().cast(), 0) };
    }

    /// Whether any table entry `p`'s stages read (any kind) has no view, by
    /// the compute roots (VIctory: mesh programs given
    /// valid-looking arguments spun on null descriptors).
    fn has_null_view(&self, p: &HostPipeline) -> bool {
        p.stages.iter().any(|stage| {
            stage.translated().used.iter().any(|u| {
                u.root == RootType::Table && matches!(self.roots_ref(true).get(usize::from(u.slot)), Some(Root::Table(table)) if descriptor::view(table + u64::from(u.table_offset_dwords) * 4).is_none())
            })
        })
    }

    /// Whether a compute dispatch should be dropped due to unbound UAVs.
    fn drop_unbound_uav(&self, p: &HostPipeline) -> bool {
        static DROP_UNBOUND_UAV: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::env::var("XNW_DROP_UNBOUND_UAV").is_ok_and(|v| v == "1" || v == "true"));
        *DROP_UNBOUND_UAV && self.has_unbound_uav(p)
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    ///
    /// This card has no mesh shaders, so in
    /// place of the base's `DispatchMesh` (base draw.rs 636-658) the mesh
    /// program runs as compute over the `x` groups, bound from the title's
    /// graphics root arguments, into a segment of the list's record and
    /// index memory ([`crate::mesh::Arena`]); a barrier; then one
    /// `DrawIndexedInstanced(x * max_prims * 3)` with the pass-through
    /// vertex shader and the title's pixel shader, under the title's state.
    // shortcut: each mesh draw has its own barrier pair (compute -> draw,
    // and back before the next write into the same chunk): batching
    // consecutive mesh draws behind one barrier needs the list to defer
    // their draws until a non-mesh command, a hook in every list method.
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32, indirect: bool) {
        if !indirect {
            self.tap(xnw_formats::framepack::op::MESH, x, [0; 4]);
        }
        let Some(p) = pipeline::host_pipeline(self.pipeline as This) else {
            once!("a mesh dispatch without a host pipeline dropped (pipeline {:#x})", self.pipeline);
            DROPPED.fetch_add(1, Ordering::Relaxed);
            no_pipeline(NP_MESH);
            count_gated(self.pipeline);
            return;
        };
        let Some(f) = p.fallback.as_deref() else {
            once!("a mesh dispatch with another kind of pipeline");
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        };
        if x == 0 {
            return;
        }
        // `XNW_MESH` (crate::mesh::Switches) drops or cuts mesh draws: to
        // tell a fault of the fallback from one elsewhere in the frame and
        // find the program behind it.
        let sw = crate::mesh::switches();
        if !sw.allows(&p.name) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let lay = &f.layout;
        // Record memory per piece is bounded: a stale or garbage group
        // count read for an indirect dispatch would otherwise allocate and
        // run gigabytes of records. A direct dispatch of more groups
        // (the title's `directmesh` batches: meshlets times instances, past
        // 65,535) runs in pieces of that size, each told its first group.
        let cap = (MESH_RECORDS_PER_DRAW / u64::from(lay.record_bytes().max(1))).clamp(1, 65535) as u32;
        let total = sw.max_groups.map_or(x, |m| x.min(m));
        MESH_MAX_GROUPS.fetch_max(u64::from(total), Ordering::Relaxed);
        let pieces = total.div_ceil(cap);
        if pieces > 1 {
            MESH_SPLIT.fetch_add(1, Ordering::Relaxed);
            once!("mesh dispatch of {total} groups run in {pieces} pieces of up to {cap} ({})", p.name);
        }
        // The list's first mesh draw since its reset frees its arena.
        let reset = list.first_use(ARENA_KEY);
        for k in 0..pieces {
            let first = k * cap;
            self.mesh_piece(list, (p, f), (total - first).min(cap), first, (reset && k == 0, indirect));
        }
    }

    /// One piece of a direct mesh dispatch: `x` groups from console group
    /// `first` on, through the compute fallback into a segment of the
    /// list's arena, then the indexed draw.
    fn mesh_piece(&mut self, list: &HostList, (p, f): (&HostPipeline, &pipeline::host::MeshFallback), x: u32, first: u32, (reset, indirect): (bool, bool)) {
        let Some(gpu) = super::host::host_device() else {
            return;
        };
        let sw = crate::mesh::switches();
        let lay = &f.layout;
        let (records, indices) = (u64::from(x) * u64::from(lay.record_bytes()), u64::from(x) * u64::from(lay.index_bytes()));
        let key = std::ptr::from_ref(list) as usize;
        let Some(seg) = crate::mesh::with_arena(gpu, key, reset, |a| a.segment(gpu, &list.list, records, indices).ok()) else {
            once!("mesh record memory not allocated ({records} bytes)");
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let Some(base) = first_group(list, first) else {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        };
        // The compute half reads the graphics root arguments (the mesh
        // stage's) through the compute binding path.
        std::mem::swap(&mut self.compute, &mut self.graphics);
        let bound = self.bind(list, &f.cs, true);
        std::mem::swap(&mut self.compute, &mut self.graphics);
        if !bound {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        self.set_root_lengths(list, f);
        let l = &list.list;
        let timed = crate::mesh::with_arena(gpu, key, false, |a| a.time_begin(gpu, l));
        // SAFETY: an open list with the compute half bound; the segment is
        // in the list's arena, alive until the list's next reset; the first
        // group in its scratch memory.
        unsafe {
            l.SetComputeRootUnorderedAccessView(f.records, seg.records);
            l.SetComputeRootUnorderedAccessView(f.indices, seg.indices);
            l.SetComputeRootShaderResourceView(f.base, base);
            let _crumb = super::crumbs::begin_compute(list, || format!("mesh records dispatch {x} groups"));
            l.Dispatch(x, 1, 1);
        }
        crate::mesh::with_arena(gpu, key, false, |a| {
            if let Some(q) = timed {
                a.time_mark(l, q, 1, &p.name, x);
            }
            a.written(l, seg);
        });
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
        let n = crate::mesh::MESH_DRAWS.fetch_add(1, Ordering::Relaxed);
        crate::mesh::MESH_GROUPS.fetch_add(u64::from(x), Ordering::Relaxed);
        if n < sw.log {
            let how = if indirect { "indirect" } else { "direct" };
            xnw_core::info!("d3d12x: mesh draw {n}: {x} groups from {first} ({how}) through the compute fallback, {records} B records, {indices} B indices ({})", p.name);
        }
        let drawn = !sw.compute_only && self.prepare_for(list, false, true);
        if drawn {
            let view = D3D12_INDEX_BUFFER_VIEW { BufferLocation: seg.indices, SizeInBytes: indices as u32, Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT };
            // SAFETY: as above, with the draw half bound.
            unsafe {
                l.SetGraphicsRootShaderResourceView(f.vertex_records, seg.records);
                l.IASetIndexBuffer(Some(&view));
                l.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                let _crumb = super::crumbs::begin(list, || format!("mesh draw {} indices rt {:?}", x * lay.max_prims * 3, self.rt_size));
                l.DrawIndexedInstanced(x * lay.max_prims * 3, 1, 0, 0, 0);
            }
            // Later draws keep the title's topology.
            let topology = self.topology;
            self.set_topology(list, topology);
            DRAWN.fetch_add(1, Ordering::Relaxed);
            self.drew = true;
            {
                let dw = self.depth_writes();
                super::capture::note_draw(self.rt_size, dw);
            }
        } else {
            DROPPED.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(q) = timed {
            crate::mesh::with_arena(gpu, key, false, |a| a.time_mark(l, q, 2, &p.name, x));
        }
    }

    /// An indirect mesh dispatch whose arguments (`x, y, z` at title address
    /// `arguments`) are read when the list is recorded: right when the
    /// title's processor wrote them, stale when its GPU work in the same
    /// submission does (the world's culled geometry). Only with
    /// `XNW_MESH_INDIRECT=cpu`; [`State::dispatch_mesh_gpu`] otherwise.
    /// `DispatchMeshX` with indirect arguments (one record of 12 bytes at
    /// title address `arguments`): on the GPU as `ExecuteIndirect`'s
    /// (VIctory drops these, list.rs 1074-1087 @ ffe310e).
    pub(crate) fn dispatch_mesh_x_indirect(&mut self, list: &HostList, arguments: u64) {
        self.tap(xnw_formats::framepack::op::MESH_X, 0, [0; 4]);
        MESH_X_INDIRECT.fetch_add(1, Ordering::Relaxed);
        once!("DispatchMeshX with indirect arguments seen ({arguments:#x})");
        if mesh_indirect_cpu() {
            self.dispatch_mesh_indirect(list, arguments);
        } else if !self.dispatch_mesh_gpu(list, (12, 1), arguments, None) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn dispatch_mesh_indirect(&mut self, list: &HostList, arguments: u64) -> bool {
        if shadow::find(arguments, 12).is_none() {
            once!("indirect mesh arguments outside title buffers ({arguments:#x})");
            return false;
        }
        // SAFETY: title memory inside a buffer the title made.
        let [x, y, z] = unsafe { [0, 1, 2].map(|i| (arguments as *const u32).add(i).read_volatile()) };
        once!("indirect mesh dispatches: group counts read when recorded (first {x} {y} {z})");
        MESH_INDIRECT[1].fetch_add(1, Ordering::Relaxed);
        if (y, z) != (1, 1) && x != 0 {
            once!("indirect mesh dispatch of {x} x {y} x {z} groups: run as {x}");
        }
        self.dispatch_mesh(list, x.min(65535), true);
        true
    }

    /// An indirect mesh dispatch whose group counts stay on the GPU
    /// (`records` records `stride` bytes apart at title address
    /// `arguments`, their count at `count` if the title gives one). Per
    /// record: the writer pass ([`crate::mesh::IndirectWriter`]) turns the
    /// record into a clamped compute command and draw command over room it
    /// takes in the writer's ring (one ring, sized once, emptied before
    /// each submission), then the program and the draw run through
    /// `ExecuteIndirect`. The record is read from the host resource the
    /// title's passes wrote: the in-place view of title memory where the
    /// range resolves (`XNW_SHARED=2`), else the buffer's host copy, brought
    /// up to date at submission (bytes the processor wrote after recording
    /// count; bytes shaders of the submission write are not copied over).
    // shortcut: with `XNW_SHARED=2` a record whose range resolves in place
    // is taken as written in place; a culling view left on a host copy over
    // the same range (a window crossing) would be read stale.
    fn dispatch_mesh_gpu(&mut self, list: &HostList, (stride, max): (u32, u32), arguments: u64, count: Option<u64>) -> bool {
        let Some(p) = pipeline::host_pipeline(self.pipeline as This) else {
            once!("an indirect mesh dispatch without a host pipeline dropped (pipeline {:#x})", self.pipeline);
            DROPPED.fetch_add(1, Ordering::Relaxed);
            no_pipeline(NP_MESH);
            return true;
        };
        let Some(f) = p.fallback.as_deref() else {
            once!("an indirect mesh dispatch with another kind of pipeline");
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return true;
        };
        let sw = crate::mesh::switches();
        if !sw.allows(&p.name) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        let (Some(gpu), Some(writer)) = (super::host::host_device(), mesh_writer()) else {
            return false;
        };
        let roots = crate::mesh::CommandRoots { records: f.records, indices: f.indices, vertex_records: f.vertex_records };
        let signatures = match writer.signatures(gpu, (&f.cs.root, &p.root), roots) {
            Ok(s) => s,
            Err(e) => {
                once!("indirect mesh command signatures refused ({e}): indirect mesh dispatches dropped");
                return false;
            }
        };
        // Debug: the first 8 of a frame, every 64th frame.
        let frame = super::queue::presented();
        if MESH_TRACED_FRAME.swap(frame, Ordering::Relaxed) != frame {
            MESH_TRACED.store(0, Ordering::Relaxed);
        }
        let traced = |n: u64| frame.is_multiple_of(64) && n < 8;
        // The title's records and count as host addresses.
        let source = |address: u64, len: u64, what: &str| -> Option<u64> {
            let (at, how) = if let Some(r) = crate::shared::lookup(address, len) {
                crate::shared::count_view(crate::shared::VIEW_ARGS, true);
                list.keep(&r.buffer);
                (descriptor::host::shared_va(&r), format!("in place (window {} + {:#x})", crate::shared::window_id(&r.buffer), r.offset))
            } else {
                let s = shadow::find(address, len)?;
                crate::shared::left_because(crate::shared::VIEW_ARGS, crate::shared::WHY_HOST);
                s.refresh(list, address, len);
                let (declared, bindless) = list.writes_declared(&s, address as usize, len as usize);
                let start = address - s.offset(address);
                let how = format!("host copy of {start:#x}+{:#x} at {:#x}; GPU write declared earlier in the list {declared}, writing bindless draws in the list {bindless}", s.offset(address) + s.left(address), s.offset(address));
                (s.gpu(address), how)
            };
            let n = MESH_TRACED.fetch_add(1, Ordering::Relaxed);
            if traced(n) {
                // SAFETY: title memory inside a buffer (found above).
                let now = unsafe { [0, 1, 2].map(|i| (address as *const u32).add(i).read_volatile()) };
                xnw_core::info!("d3d12x: indirect mesh {what} {n} (frame {frame}): title {address:#x}, {how}; processor sees {now:?} now ({})", p.name);
            }
            at.is_multiple_of(4).then_some(at)
        };
        let counter = match count {
            Some(at) => match source(at, 4, "count") {
                Some(c) => Some(c),
                None => {
                    once!("indirect mesh count outside title buffers or misaligned ({at:#x})");
                    return false;
                }
            },
            None => None,
        };
        let lay = &f.layout;
        let capacity = sw.max_groups.map_or(MESH_INDIRECT_MAX_GROUPS, |m| m.clamp(1, MESH_INDIRECT_MAX_GROUPS));
        // At most 32 MiB of output per dispatch (VIctory).
        let per_group = u64::from(lay.record_bytes() + lay.index_bytes()).max(1);
        let capacity = capacity.min((crate::mesh::DISPATCH_BYTES / per_group).max(1) as u32);
        let reset = list.first_use(ARENA_KEY);
        if list.first_use(RING_KEY) {
            // The ring is the submission's: empty before it runs (submissions
            // run one at a time, each waited for).
            list.defer_host(move || writer.ring_reset());
        }
        let key = std::ptr::from_ref(list) as usize;
        let l = &list.list;
        let budget = list.budget();
        // Every record's groups start at console group 0.
        let Some(zero) = first_group(list, 0) else {
            return false;
        };
        list.keep(&writer.ring);
        let mut ran = false;
        for r in 0..max.min(64) {
            let address = arguments + u64::from(r) * u64::from(stride);
            let Some(args) = source(address, 12, "record") else {
                once!("indirect mesh arguments outside title buffers or misaligned ({address:#x})");
                return ran;
            };
            let Some(slot) = crate::mesh::with_arena(gpu, key, reset && r == 0, |a| a.args_slot(gpu, l).ok()) else {
                once!("indirect mesh argument slot not allocated: dropped");
                MESH_NO_MEMORY.fetch_add(1, Ordering::Relaxed);
                DROPPED.fetch_add(1, Ordering::Relaxed);
                return ran;
            };
            let plan = crate::mesh::Plan { record_bytes: lay.record_bytes(), index_bytes: lay.index_bytes(), corners: lay.max_prims * 3, record: r, capacity };
            {
                let _crumb = super::crumbs::begin_compute(list, || "indirect mesh writer pass".to_string());
                writer.record(l, plan, args, counter, slot.gpu, budget);
            }
            let slots = crate::mesh::with_arena(gpu, key, false, |a| a.args_written(l, slot));
            list.keep(&slots);
            // The writer replaced the compute root signature and pipeline.
            self.host_pipeline = 0;
            self.host_roots[1] = 0;
            std::mem::swap(&mut self.compute, &mut self.graphics);
            if self.has_null_view(&f.cs) {
                MESH_NULL_VIEWS.fetch_add(1, Ordering::Relaxed);
            }
            let bound = self.bind(list, &f.cs, true);
            std::mem::swap(&mut self.compute, &mut self.graphics);
            if !bound {
                DROPPED.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            self.set_root_lengths(list, f);
            let timed = crate::mesh::with_arena(gpu, key, false, |a| a.time_begin(gpu, l));
            // SAFETY: an open list with the compute half bound; the slot is in
            // the list's arena (alive until its reset), the ring the writer's.
            unsafe {
                l.SetComputeRootShaderResourceView(f.base, zero);
            }
            writer.ring_writable(l);
            // SAFETY: as above.
            unsafe {
                let _crumb = super::crumbs::begin_compute(list, || "indirect mesh records dispatch".to_string());
                l.ExecuteIndirect(&signatures.compute, 1, &slots, slot.offset, None, 0);
            }
            writer.ring_readable(l);
            if let Some(q) = timed {
                crate::mesh::with_arena(gpu, key, false, |a| a.time_mark(l, q, 1, &p.name, 0));
            }
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
            let n = crate::mesh::MESH_DRAWS.fetch_add(1, Ordering::Relaxed);
            MESH_INDIRECT[0].fetch_add(1, Ordering::Relaxed);
            if n < sw.log {
                xnw_core::info!("d3d12x: mesh draw {n}: up to {capacity} groups (indirect, counted on the GPU) through the compute fallback, room from the ring ({})", p.name);
            }
            if !sw.compute_only && self.prepare_for(list, false, true) {
                // SAFETY: a live buffer.
                let ring = unsafe { writer.ring.GetGPUVirtualAddress() };
                let view = D3D12_INDEX_BUFFER_VIEW { BufferLocation: ring, SizeInBytes: writer.ring_bytes().min(u64::from(u32::MAX)) as u32, Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT };
                // SAFETY: as above, with the draw half bound.
                unsafe {
                    l.IASetIndexBuffer(Some(&view));
                    l.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    let _crumb = super::crumbs::begin(list, || format!("indirect mesh draw rt {:?}", self.rt_size));
                    l.ExecuteIndirect(&signatures.draw, 1, &slots, slot.offset + crate::mesh::DRAW_COMMAND, None, 0);
                }
                let topology = self.topology;
                self.set_topology(list, topology);
                DRAWN.fetch_add(1, Ordering::Relaxed);
                self.drew = true;
                {
                let dw = self.depth_writes();
                super::capture::note_draw(self.rt_size, dw);
            }
            } else {
                DROPPED.fetch_add(1, Ordering::Relaxed);
            }
            writer.ring_released(l);
            // The commands set root views of both halves.
            self.host_roots = [0; 2];
            if let Some(q) = timed {
                crate::mesh::with_arena(gpu, key, false, |a| a.time_mark(l, q, 2, &p.name, 0));
            }
            ran = true;
        }
        ran
    }

    /// Hands this draw's binding to the frame pack's tap and the second
    /// draw consumer (tap.rs), each if registered (null mode included:
    /// before any host check).
    fn tap(&self, op: u8, count: u32, args: [u32; 4]) {
        if let Some(f) = super::tap::get() {
            self.tap_roots(f, op, count, [0; 2], args, &self.graphics);
        }
        if let Some((want, f)) = super::tap::draw_tap()
            && want(self.pipeline)
        {
            self.tap_roots(f, op, count, [0; 2], args, &self.graphics);
        }
    }

    /// A direct dispatch of a pipeline the pack asked for
    /// (`tap::set_dispatch_want`), with the compute roots; before any host
    /// or keep check, as the draws.
    pub(crate) fn tap_dispatch(&self, x: u32, y: u32, z: u32) {
        if super::tap::dispatch_want().is_some_and(|want| want(self.pipeline))
            && let Some(f) = super::tap::get()
        {
            self.tap_roots(f, xnw_formats::framepack::op::DISPATCH, x, [y, z], [0; 4], &self.compute);
        }
        if let Some((want, f)) = super::tap::dispatch_tap()
            && want(self.pipeline)
        {
            self.tap_roots(f, xnw_formats::framepack::op::DISPATCH, x, [y, z], [0; 4], &self.compute);
        }
    }

    fn tap_roots(&self, f: super::tap::Tap, op: u8, count: u32, yz: [u32; 2], args: [u32; 4], set: &[Root]) {
        use super::tap::{TABLE_SLOTS, TapDraw, TapRoot};
        let mut roots = [TapRoot::None; 32];
        let n = set.len().min(roots.len());
        for (out, r) in roots.iter_mut().zip(set) {
            *out = match r {
                Root::None => TapRoot::None,
                Root::Table(h) => match self.heap_of(*h) {
                    Some(heap) => TapRoot::Table { handle: *h, heap_start: heap.start, slots: super::tap::slots_in(heap, *h, TABLE_SLOTS) },
                    None => TapRoot::Table { handle: *h, heap_start: 0, slots: 0 },
                },
                Root::Constants(c) => TapRoot::Constants(c),
                Root::View(a) => TapRoot::View(*a),
            };
        }
        f(&TapDraw { op, count, yz, args, pipeline: self.pipeline, view_heap: self.heaps[0], sampler_heap: self.heaps[1], roots: &roots[..n] });
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute { &self.compute } else { &self.graphics }
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// XNW: GPU-side descriptors: makes what the views of `used`
    /// read current for the list (see [`readable`]), and counts draws.
    fn gpu_readable(&self, list: &HostList, used: &[UsedResource], compute: bool, host_copies: Option<u32>) {
        let n = GPU_DRAWS.fetch_add(1, Ordering::Relaxed);
        // A program translated with the page table
        // reads buffers in place through it unless its host-copy bits name
        // typed buffers or descriptors its code builds: its buffer views
        // then need no host copy brought up to date.
        use xnw_shader::emit::host_copies::{BUILT_DESCRIPTORS, TYPED_BUFFERS};
        let paged = host_copies.is_some_and(|b| b & (TYPED_BUFFERS | BUILT_DESCRIPTORS) == 0);
        // XNW_DESC_TRACE=<n>: the first n stages bound, what they read.
        static TRACE: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| std::env::var("XNW_DESC_TRACE").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
        static TRACED: AtomicU64 = AtomicU64::new(0);
        let textures = used.iter().any(|u| u.kind == ResourceType::Srv && u.dwords == 8);
        if textures && TRACED.fetch_add(1, Ordering::Relaxed) < *TRACE {
            for u in used {
                let root = self.roots_ref(compute).get(usize::from(u.slot));
                let text = match (u.root, root) {
                    (RootType::Table, Some(&Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        // SAFETY: a descriptor record in title memory.
                        let w = unsafe { (handle as *const [u32; 8]).read_unaligned() };
                        let view = descriptor::view(handle).map(|v| format!("{v:?}").chars().take(60).collect::<String>());
                        format!("record {w:x?} view {view:?} {}", descriptor::host::table_slot_text(w[0]))
                    }
                    (r, root) => format!("{r:?} {root:?}"),
                };
                xnw_core::info!("gpu desc #{n} (compute {compute}): {:?} slot {} +{} dwords {}: {text}", u.kind, u.slot, u.table_offset_dwords, u.dwords);
            }
        }
        for u in used.iter().filter(|u| u.root == RootType::Table) {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let handle = table + u64::from(u.table_offset_dwords) * 4;
            if let Some(view) = descriptor::view(handle) {
                if paged && (matches!(view, View::Constant(..)) || descriptor::host::buffer_range(&view).is_some()) {
                    PAGED_VIEWS.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                readable(list, u, &view);
                // SAFETY: a descriptor record in title memory.
                let [index, _, _, tag] = unsafe { (handle as *const [u32; 4]).read_unaligned() };
                if tag == u32::from_le_bytes(*b"vVIE") && descriptor::host::table_refresh(index, &view) {
                    GPU_REMADE.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some(view) = host.mirror(index, handle, true) {
            readable(list, u, &view);
        } else if u.kind == ResourceType::Uav {
            host.make_null_uav(index, u.dwords == 4, u.stride);
        } else {
            host.make_null_kind(index, u.kind, u.dwords);
        }
    }

    /// The descriptors a shader reads through tables. Tables holding a
    /// constant buffer are made afresh for the draw in dynamic slots, with
    /// the constant data copied to upload memory when the list is
    /// submitted: the host wants constant buffers 256-byte aligned, the
    /// console only 16. Returns those tables' new byte offsets by root
    /// parameter; the other tables' slots are made from the title's.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once: its constant data is copied at submission either way, and the
    /// big tables the world binds would otherwise take hundreds of slots per
    /// draw, wrapping the dynamic slots onto lists not yet submitted.
    fn tables(&mut self, list: &HostList, used: &[UsedResource], compute: bool) -> Option<Vec<(u8, u32)>> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = used.iter().filter(tables).filter(|u| u.kind == ResourceType::Cbv).map(|u| u.slot).collect();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let end = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(u.slot);
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used.iter().filter(tables).filter(|u| u.slot == slot).collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let Some(host) = heap.host() else {
                continue;
            };
            let n = entries.iter().map(|u| u.table_offset_dwords / 8 + 1).max().unwrap_or(1);
            // The table and what its entries hold (record words).
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    // SAFETY: a descriptor slot in title memory.
                    // Index, tag, entry, generation (record layout).
                    unsafe { (at as *const [u32; 4]).read_unaligned() }.hash(&mut h);
                }
                (table, h.finish())
            };
            if let Some(&first) = self.fresh_tables.get(&key) {
                for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = descriptor::view(handle) {
                        readable(list, u, &view);
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            // The table's memory, and its records by kind.
            crate::shared::zero_copy_root(true, table as usize);
            for k in 0..u64::from(n) {
                // SAFETY: a descriptor slot in title memory (the table's).
                let words = unsafe { ((table + k * 32) as *const [u32; 4]).read_unaligned() };
                let kind = if words[3] == descriptor::VIEW_TAG { 0 } else if words == [0; 4] { 2 } else { 1 };
                crate::shared::ZC_RECORDS[kind].fetch_add(1, Ordering::Relaxed);
            }
            let first = host.dynamic(n)?;
            self.fresh_tables.insert(key, first);
            if let Some(arc) = heap.host_arc() {
                self.dynamic.push((arc, first, n));
            }
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match descriptor::view(table + u64::from(k) * 32) {
                    Some(View::Constant(address, size)) => {
                        if let Some(name) = self.late_name() {
                            list.note_late(super::late::recorded("table constants", &name, address, u64::from(size)));
                        }
                        // Constants read in place, not filled into upload
                        // memory: the shared buffer's address (not the title's)
                        // must be 256-byte aligned, and the view (size rounded
                        // up) stays inside the shared buffer.
                        if let Some(r) = crate::shared::range(crate::shared::VIEW_TABLE, address, u64::from(size).next_multiple_of(256)) {
                            let gpu = descriptor::host::shared_va(&r);
                            if gpu.is_multiple_of(256) {
                                list.keep(&r.buffer);
                                host.constant_at(index, gpu, size);
                                continue;
                            }
                            crate::shared::misaligned(crate::shared::VIEW_TABLE);
                        }
                        let Some((gpu, to)) = list.scratch(u64::from(size)) else {
                            host.make_null_kind(index, ResourceType::Cbv, 4);
                            continue;
                        };
                        list.defer_fill(to, address as *const u8, size as usize);
                        host.constant_at(index, gpu, size);
                    }
                    Some(view) => {
                        host.make_at(index, &view);
                        if let Some(u) = used {
                            readable(list, u, &view);
                        }
                    }
                    None => {
                        if let Some(u) = used
                            && u.kind == ResourceType::Uav
                        {
                            host.make_null_uav(index, u.dwords == 4, u.stride);
                            once!("a table entry without a view (slot {slot})");
                        } else if let Some(u) = used {
                            host.make_null_kind(index, u.kind, u.dwords);
                            once!("a table entry without a view (slot {slot})");
                        } else {
                            host.make_null(index);
                        }
                    }
                }
            }
            made.push((slot, first * 32));
        }
        Some(made)
    }

    /// Whether rectangles drawn now go through the pipeline's rectangle
    /// geometry shader: then they draw as a plain triangle list.
    fn rect_gs(&self) -> bool {
        if self.topology != RECTANGLES {
            return false;
        }
        let p = pipeline::host_pipeline(self.pipeline as This);
        // The first rectangle draw on a pipeline not typed for rectangles
        // marks the world (`XNW_CAPTURE_TARGETS=rect+<k>`).
        if !p.is_some_and(|p| p.rect) {
            let _ = RECT_VARIANT_FRAME.compare_exchange(u64::MAX, super::queue::presented(), Ordering::Relaxed, Ordering::Relaxed);
        }
        let gs = p.is_some_and(|p| p.rect || p.rect_pso().is_some());
        let _call = super::counters::call(super::counters::DRAW_RECT);
        let n = if gs { &RECT_GS } else { &RECT_NO_GS };
        let n = n.fetch_add(1, Ordering::Relaxed);
        if n.is_power_of_two() {
            xnw_core::info!("d3d12x: rectangle-list draws through the geometry shader {}, without {}", RECT_GS.load(Ordering::Relaxed), RECT_NO_GS.load(Ordering::Relaxed));
        }
        gs
    }

    /// After a rectangle draw's pipeline is bound: its rectangle variant in
    /// its place, when the pipeline was not made for rectangles.
    /// `false` when the variant is still being made (off this thread):
    /// the draw drops, counted under `kind` for want of a pipeline.
    fn bind_rect_variant(&mut self, list: &HostList, kind: usize) -> bool {
        if self.topology != RECTANGLES {
            return true;
        }
        let Some(p) = pipeline::host_pipeline(self.pipeline as This) else { return true };
        if p.rect {
            return true;
        }
        let pso = match pipeline::rect_variant(p, self.pipeline as This) {
            Ok(Some(pso)) => pso,
            Ok(None) => return true,
            Err(()) => {
                no_pipeline(kind);
                return false;
            }
        };
        if self.host_pipeline != pso.as_raw() as usize {
            // SAFETY: an open list; the variant has the pipeline's root signature.
            unsafe { list.list.SetPipelineState(pso) };
            self.host_pipeline = pso.as_raw() as usize;
            list.keep(pso);
            RECT_VARIANT_DRAWS.fetch_add(1, Ordering::Relaxed);
        }
        true
    }

    pub(crate) fn draw(&mut self, list: &HostList, vertices: u32, instances: u32, first: u32, first_instance: u32) {
        self.tap(xnw_formats::framepack::op::DRAW, vertices, [instances, first, 0, first_instance]);
        let rects = self.topology == RECTANGLES && !self.rect_gs();
        let ran = self.prepare(list, false) && (!rects || self.rectangles(list, vertices, first));
        if ran && !rects && !self.bind_rect_variant(list, NP_DRAW) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects { (vertices / 3 * 6, 0) } else { (vertices, first) };
            // SAFETY: an open list with everything set.
            let _crumb = super::crumbs::begin(list, || format!("draw {count} vertices x{instances} from {from} rt {:?}", self.rt_size));
            unsafe { list.list.DrawInstanced(count, instances, from, first_instance) };
            DRAWN.fetch_add(1, Ordering::Relaxed);
            self.drew = true;
            {
                let dw = self.depth_writes();
                super::capture::note_draw(self.rt_size, dw);
            }
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            no_pipeline_if(self.pipeline, NP_DRAW);
        }
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p.inputs.iter().filter(|e| e.slot == slot && !e.per_instance).map(|e| (e.offset, e.format)).collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0 || stride == 0 || u64::from(first + count) * stride > u64::from(v.size) || shadow::find(from, rects * 3 * stride).is_none() {
                once!("rectangle vertices outside their buffer (slot {slot})");
                return false;
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (std::slice::from_raw_parts(from as *const u8, n * 3 * stride), std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride))
                };
                super::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW { BufferLocation: gpu, SizeInBytes: (rects * 6 * stride as u64) as u32, StrideInBytes: v.stride };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(&mut self, list: &HostList, indices: u32, instances: u32, first: u32, base: i32, first_instance: u32) {
        self.tap(xnw_formats::framepack::op::INDEXED, indices, [instances, first, base as u32, first_instance]);
        let rects = self.topology == RECTANGLES && !self.rect_gs();
        if rects {
            once!("indexed rectangles without the geometry shader not drawn");
        }
        let ran = !rects && self.prepare(list, true);
        if ran && !self.bind_rect_variant(list, NP_INDEXED) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        if ran {
            // SAFETY: an open list with everything set.
            let _crumb = super::crumbs::begin(list, || format!("draw indexed {indices} indices x{instances} from {first} base {base} rt {:?}", self.rt_size));
            unsafe { list.list.DrawIndexedInstanced(indices, instances, first, base, first_instance) };
            DRAWN.fetch_add(1, Ordering::Relaxed);
            self.drew = true;
            {
                let dw = self.depth_writes();
                super::capture::note_draw(self.rt_size, dw);
            }
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            no_pipeline_if(self.pipeline, NP_INDEXED);
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(&mut self, list: &HostList, (kind, stride, max): (u32, u32, u32), arguments: u64, count: Option<u64>) -> bool {
        // The GPU meshlet-cull path's draws arrive here (one capture: 3,523 of
        // 4,149 draws a frame); kind 10 mesh, else draws/dispatches.
        self.tap(if kind == 10 { xnw_formats::framepack::op::INDIRECT_MESH } else { xnw_formats::framepack::op::INDIRECT }, max, [0; 4]);
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        // The base names the reason only in diagnostics builds.
        let why = |_: &str| {};
        // Mesh dispatches: group counts stay on the GPU, where the
        // title's culling passes write them; `XNW_MESH_INDIRECT=cpu` reads
        // them when recorded instead, each a direct fallback dispatch.
        if kind == MESH && !mesh_indirect_cpu() {
            return self.dispatch_mesh_gpu(list, (stride, max), arguments, count);
        }
        if kind == MESH {
            let records = match count {
                // SAFETY: a title buffer, checked first.
                Some(at) if shadow::find(at, 4).is_some() => unsafe { (at as *const u32).read_volatile() }.min(max),
                Some(_) => return false,
                None => max,
            };
            let mut ran = false;
            for r in 0..records.min(4096) {
                ran |= self.dispatch_mesh_indirect(list, arguments + u64::from(r) * u64::from(stride));
            }
            return ran;
        }
        let len = u64::from(stride) * u64::from(max.max(1));
        if let Some(name) = self.late_name() {
            list.note_late(super::late::recorded("indirect arguments", &name, arguments, len));
            if let Some(at) = count {
                list.note_late(super::late::recorded("indirect count", &name, at, 4));
            }
        }
        let Some(args) = shadow::find(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match shadow::find(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH).then(|| host_signature(kind, stride)).flatten() else {
            why("no host signature");
            return false;
        };
        // Read from the host copy (the clamp copies the records out of it).
        crate::shared::left_because(crate::shared::VIEW_ARGS, crate::shared::WHY_HOST);
        crate::shared::zero_copy_note(crate::shared::ZC_ARGS, arguments as usize, len as usize);
        args.refresh(list, arguments, len);
        if let Some((s, at)) = &counter {
            s.refresh(list, *at, 4);
        }
        // What would run is known first: a refused or pipeline-less
        // execution records nothing, our clamp pass included (it ran for
        // refused dispatches before).
        let runnable = match (kind, pipeline::host_pipeline(self.pipeline as This)) {
            (DRAW | INDEXED, _) if self.topology == RECTANGLES && !self.rect_gs() => {
                // Not drawn on the host; counted (VIctory drops
                // about 246 a frame, the likely full-screen passes).
                let n = RECT_INDIRECT.fetch_add(1, Ordering::Relaxed);
                if n.is_power_of_two() {
                    xnw_core::info!("d3d12x: indirect rectangle-list draws not drawn: {}", n + 1);
                }
                false
            }
            (DRAW | INDEXED, Some(p)) => !p.compute && !p.mesh,
            (DISPATCH, Some(p)) => p.compute,
            _ => false,
        };
        if !runnable {
            return false;
        }
        // XNW_INDIRECT_CLAMP: where the GPU reads the arguments.
        let mode = clamp_mode();
        let source: (ID3D12Resource, u64, Option<(ID3D12Resource, u64)>) = match mode {
            ClampMode::Cpu if kind == DISPATCH && max == 1 && counter.is_none() => {
                // Upload memory the processor fills at submission from title
                // memory, clamped there; no pass of ours, nothing the GPU wrote.
                let Some(buffer) = cpu_arguments(list, arguments) else {
                    why("no upload memory");
                    return false;
                };
                CLAMP_MODES[2].fetch_add(1, Ordering::Relaxed);
                (buffer, 0, None)
            }
            ClampMode::Off | ClampMode::Cpu => {
                // The title's records as the culling pass left them, in the
                // host copy, moved to indirect argument for the execution
                // and back (a UAV barrier alone leaves the command processor
                // reading stale bytes: the unit test reads 0 groups). Diagnostic:
                // the copy may be bound otherwise in the list (ids 527/538 in one capture).
                off_moves(list, &args.buffer, counter.as_ref().map(|(s, _)| &s.buffer), true);
                list.keep(&args.buffer);
                CLAMP_MODES[1].fetch_add(1, Ordering::Relaxed);
                (args.buffer.clone(), args.offset(arguments), counter.as_ref().map(|(s, at)| (s.buffer.clone(), s.offset(*at))))
            }
            ClampMode::Gpu => {
                // Bounded first, into our own room (the title's copy is only
                // read): the clamp binds a compute pipeline of its own.
                let room_bytes = len + 4;
                if room_bytes > 16 << 20 {
                    why("records past 16 MiB");
                    return false;
                }
                let Some(room) = arg_room(list, room_bytes) else {
                    why("no argument room");
                    return false;
                };
                let count_at = counter.as_ref().map(|(s, at)| s.gpu(*at));
                if !super::indirect::bound(list, args.gpu(arguments), count_at, (kind, stride, max), room.gpu) {
                    why("not bounded (no clamp, or arguments not 4-byte aligned)");
                    return false;
                }
                let room_buffer = arg_written(list, room);
                self.host_pipeline = 0;
                self.host_roots[1] = 0;
                CLAMP_MODES[0].fetch_add(1, Ordering::Relaxed);
                let count = counter.is_some().then(|| (room_buffer.clone(), room.offset + u64::from(stride) * u64::from(max)));
                (room_buffer, room.offset, count)
            }
        };
        let ran = match kind {
            DRAW | INDEXED => self.prepare(list, kind == INDEXED),
            DISPATCH => match pipeline::host_pipeline(self.pipeline as This) {
                Some(p) if p.compute => self.bind(list, p, true),
                _ => false,
            },
            _ => false,
        };
        if !ran {
            no_pipeline_if(self.pipeline, if kind == MESH { NP_MESH } else { NP_INDIRECT });
            return false;
        }
        if matches!(kind, DRAW | INDEXED) && !self.bind_rect_variant(list, NP_INDIRECT) {
            return false;
        }
        let _crumb = super::crumbs::begin_kind(list, kind == DISPATCH, || format!("execute indirect kind {kind} up to {max} rt {:?}", self.rt_size));
        // SAFETY: an open list with everything set; the argument buffers are
        // kept by the list (room, host copy, upload memory), the signature is
        // cached.
        unsafe {
            list.list.ExecuteIndirect(&signature, max, &source.0, source.1, source.2.as_ref().map(|c| &c.0), source.2.as_ref().map_or(0, |c| c.1))
        };        if matches!(mode, ClampMode::Off | ClampMode::Cpu) && source.0.as_raw() == args.buffer.as_raw() {
            off_moves(list, &args.buffer, counter.as_ref().map(|(s, _)| &s.buffer), false);
        }
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
            if super::capture::cull_dump_frame()
                && let Some(p) = pipeline::host_pipeline(self.pipeline as This)
            {
                self.cull_dump(list, p, Some(arguments));
            }
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
            self.drew = true;
            {
                let dw = self.depth_writes();
                super::capture::note_draw(self.rt_size, dw);
            }
        }
        true
    }
}

/// Where indirect executions (not mesh dispatches) read their
/// arguments, `XNW_INDIRECT_CLAMP`: unset, our clamp pass copies them into
/// a room of ours; `off`, the title's host copy as the culling pass left
/// it, no pass of ours; `cpu`, for one-record dispatches without a count,
/// upload memory the processor fills from title memory at submission and
/// clamps (right only for arguments the processor or an earlier submission
/// wrote; a diagnostic), else as `off`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClampMode {
    Gpu,
    Off,
    Cpu,
}

fn clamp_mode() -> ClampMode {
    static MODE: std::sync::LazyLock<ClampMode> = std::sync::LazyLock::new(|| match std::env::var("XNW_INDIRECT_CLAMP").as_deref().map(str::trim) {
        Ok("off") => ClampMode::Off,
        Ok("cpu") => ClampMode::Cpu,
        _ => ClampMode::Gpu,
    });
    *MODE
}

/// `XNW_INDIRECT_CLAMP=off`: the title's copy (and its count's) from
/// unordered access to indirect argument (`to`) or back.
fn off_moves(list: &HostList, args: &ID3D12Resource, count: Option<&ID3D12Resource>, to: bool) {
    let (a, b) = (D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    let (before, after) = if to { (a, b) } else { (b, a) };
    let mut v = vec![super::host::transition(args, before, after)];
    if let Some(c) = count.filter(|c| c.as_raw() != args.as_raw()) {
        v.push(super::host::transition(c, before, after));
    }
    // SAFETY: an open list; live buffers the list keeps.
    unsafe { list.list.ResourceBarrier(&v) };
}
/// `XNW_INDIRECT_CLAMP=cpu`: upload memory the processor fills at
/// submission with the 12 bytes of a dispatch at title address `at`,
/// clamped as the GPU clamp does (65,535 per dimension, 2^20 groups).
fn cpu_arguments(list: &HostList, at: u64) -> Option<ID3D12Resource> {
    let (buffer, data) = list.upload(super::host::host()?, 256)?;
    list.defer_fill(data, at as *const u8, 12);
    let to = data as usize;
    list.defer_host(move || {
        // SAFETY: the list's own upload memory, 12 bytes filled before.
        unsafe {
            let g = to as *mut [u32; 3];
            let mut v = g.read_unaligned();
            for x in &mut v {
                *x = (*x).min(65535);
            }
            if u64::from(v[0]) * u64::from(v[1]) * u64::from(v[2]) > 1 << 20 {
                v = [0; 3];
            }
            g.write_unaligned(v);
        }
    });
    Some(buffer)
}
/// Indirect executions by where their arguments came from: our clamp
/// pass, the title's host copy, processor-filled upload memory.
static CLAMP_MODES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// The clamp part of the frame's calls line, per frame over `n`.
pub(crate) fn clamp_text(n: u64) -> String {
    let n = n.max(1);
    let v: Vec<u64> = CLAMP_MODES.iter().map(|c| (c.swap(0, Ordering::Relaxed) + n / 2) / n).collect();
    format!("indirect arguments ({:?}): {} through our clamp, {} from the title's copy, {} filled by the processor", clamp_mode(), v[0], v[1], v[2])
}

#[cfg(test)]
fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }) },
    }
}
/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> = std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32), ..Default::default() };
    let desc = D3D12_COMMAND_SIGNATURE_DESC { ByteStride: stride, NumArgumentDescs: 1, pArgumentDescs: &argument, NodeMask: 0 };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe { super::host::host()?.device.CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut signature) };
    if let Err(e) = result {
        once!("host command signature refused (kind {kind}, stride {stride}): {e}");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need.
/// XNW: see [`State::gpu_counts`].
static GPU_DRAWS: AtomicU64 = AtomicU64::new(0);
static NO_WINDOW: AtomicU64 = AtomicU64::new(0);
static GPU_REMADE: AtomicU64 = AtomicU64::new(0);

fn readable(list: &HostList, u: &UsedResource, view: &View) {
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = descriptor::host::placed_texture(view);
    let view = placed.as_ref().map_or(view, |(v, _)| v);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    // A view made in place reads and writes title memory itself.
    // VIctory 86a5ed6: in the copy the host view reads.
    if let Some((shadow, address, len)) = descriptor::host::view_copy(view)
        && len > 0
        && descriptor::host::view_in_place(view).is_none()
    {
        shadow.refresh(list, address, len);
        if u.kind == ResourceType::Uav {
            shadow.gpu_writes(list, address, len);
        }
    }
    match (view, u.kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            if let Some(t) = resource::host_resource(*resource as This)
                && texture::title_written(t)
            {
                texture::upload(list, *resource as This, t);
            }
            super::list::ensure_state(list, *resource as This, SHADER_RESOURCE);
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            super::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
            if super::capture::targets_wanted() {
                UAV_TEXTURES.with(|u| {
                    let mut u = u.borrow_mut();
                    if !u.contains(&(*resource as usize)) {
                        u.push(*resource as usize);
                    }
                });
            }
        }
        _ => {}
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// The host topology a console one is drawn as: the console's own
/// rectangle, quad and fan lists as triangle lists, its line loop as a line
/// strip (the closing segment lost), the values D3D12 defines as they are
/// (point, line and triangle lists and strips, their adjacency forms, patch
/// lists); `None` for any other.
pub(crate) fn host_topology(t: u32) -> Option<u32> {
    match t {
        RECTANGLES | TRIANGLE_FAN | QUADS | QUAD_STRIP => Some(4),
        LINE_LOOP => Some(3),
        1..=5 | 10..=13 | 33..=64 => Some(t),
        _ => None,
    }
}

/// The D3D12 topology type of a host topology (1 point, 2 line, 3
/// triangle, 4 patch).
pub(crate) fn topology_type(host: u32) -> u32 {
    match host {
        1 => 1,
        2 | 3 | 10 | 11 => 2,
        4 | 5 | 12 | 13 => 3,
        _ => 4,
    }
}

/// Draws by console topology (65 and past in the last entry), and draws
/// dropped for a topology with no host equivalent and for one of another
/// type than the pipeline's, since the last frame line.
static TOPOLOGY_DRAWS: [AtomicU64; 66] = [const { AtomicU64::new(0) }; 66];
static TOPOLOGY_DROPPED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Vertex buffer views left null (no host memory behind the title's
/// address) and draws dropped for an index buffer view with none, since the
/// last frame line.
static VIEWS_UNTRANSLATED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The topology and vertex-view part of the frame's calls line, per frame
/// over `n` frames.
pub(crate) fn topology_text(n: u64) -> String {
    let n = n.max(1);
    let per = |c: &AtomicU64| (c.swap(0, Ordering::Relaxed) + n / 2) / n;
    let by: Vec<String> = TOPOLOGY_DRAWS
        .iter()
        .enumerate()
        .filter_map(|(t, c)| {
            let v = per(c);
            (v > 0).then(|| format!("{t}: {v}"))
        })
        .collect();
    format!(
        "draws by console topology [{}], dropped for an unknown topology {}, for a topology of another type than the pipeline's {}; vertex views left null (untranslated) {}, draws dropped for an untranslated index view {}",
        by.join(", "),
        per(&TOPOLOGY_DROPPED[0]),
        per(&TOPOLOGY_DROPPED[1]),
        per(&VIEWS_UNTRANSLATED[0]),
        per(&VIEWS_UNTRANSLATED[1])
    )
}

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    list.crumbs().note(|| format!("viewports {:?}", viewports.iter().map(|v| (v.TopLeftX, v.TopLeftY, v.Width, v.Height)).collect::<Vec<_>>()));
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

/// Device tests of the bindings the translator's classes need, with the
/// shapes of the translator's per-class tests (`xnw_shader::emit::s7_tests`, hand-encoded
/// GFX10, no title code) translated here and run through a host pipeline.
#[cfg(test)]
mod a2_tests {
    use super::*;
    use crate::d3d12x::descriptor::host::tests::{make_shader_heap, make_title_buffer};
    use xnw_shader::native::{ComputeIo, StageIo};
    use xnw_shader::{HwStage, NativeProgram, RootParam};

    fn param(root: RootType, sgprs: u32, user_data: u8, slot: u8) -> RootParam {
        RootParam { root, table_offset_dwords: 0, sgprs, base_register: 0, user_data, slot }
    }

    fn used(kind: ResourceType, root: RootType, stride: u32, user_data: u8, slot: u8) -> UsedResource {
        UsedResource { kind, root, table_offset_dwords: 0, lower_bound: 0, stride, dwords: 4, user_data, slot }
    }

    /// A host compute pipeline for `words`, a wave32 program of 32 threads.
    fn pipeline_for(words: &[u32], layout: Vec<RootParam>, used: Vec<UsedResource>, rsrc2: u32, spilled: bool) -> HostPipeline {
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout,
            used: used.clone(),
            io: StageIo::Cs(ComputeIo { rsrc2, threads: [32, 1, 1] }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = xnw_shader::ir::build_with(&native.code, false, 8, spilled).unwrap();
        let res = xnw_shader::resource::track(&p, &native.layout, native.stage).unwrap();
        let io = xnw_shader::io::raise(&p, &native).unwrap();
        let st = xnw_shader::structure::structure(&p).unwrap();
        // A container of no parts: a compute program has no signatures.
        let mut empty = b"DXBC".to_vec();
        empty.extend([0; 16]);
        empty.extend(1u16.to_le_bytes());
        empty.extend(0u16.to_le_bytes());
        empty.extend(32u32.to_le_bytes());
        empty.extend(0u32.to_le_bytes());
        let c = xnw_shader::Container::parse(&empty).unwrap();
        let host = xnw_shader::emit::emit(&c, &native, &p, &st, &io, &res).unwrap();
        let shader = pipeline::translated_for_test("compute", host, used);
        HostPipeline::new(&[shader], None).expect("a host pipeline")
    }

    /// Runs `p` once (one group) with `state`'s arguments.
    fn run(state: &mut State, p: &HostPipeline) {
        let host = super::super::host::host().unwrap();
        let list = host.new_list().unwrap();
        assert!(state.bind(&list, p, true), "bound");
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(1, 1, 1) };
        list.close();
        host.execute(&[&list]);
        list.reset();
        state.reset(std::ptr::null());
    }

    /// Class "an address from a root parameter that is no address": a
    /// pointer in root constants, read (scalar and vector) and written
    /// through the address map. The map must send it to the right buffer
    /// of two, at the right offset: `*p = *p + *p` there, nothing next door.
    #[test]
    fn a_program_reads_and_writes_the_right_buffer_through_the_address_map() {
        if super::super::host::host().is_none() {
            return;
        }
        let words = [
            0xF400_0100, 0xFA00_0000, // s_load_dword s4, s[0:1], 0x0
            0x7E04_0200, // v_mov_b32 v2, s0
            0x7E06_0201, // v_mov_b32 v3, s1
            0xDC30_8000, 0x017D_0002, // global_load_dword v1, v[2:3], off
            0xBF8C_C07F,
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0x4A02_0204, // v_add_nc_u32 v1, s4, v1
            0xDC70_8000, 0x007D_0102, // global_store_dword v[2:3], v1, off
            0xBF81_0000,
        ];
        let p = pipeline_for(&words, vec![param(RootType::Constants, 2, 0, 0)], vec![used(ResourceType::Cbv, RootType::Constants, 0, 0, 0)], 2 << 1, false);
        assert!(p.stages[0].address_map.is_some(), "the program asks for the map");
        let mem = super::super::resource::Memory::zeroed(256 << 10, 64 << 10).unwrap();
        let (a, b) = (mem.address(), mem.address() + 0x1_0000);
        let ra = make_title_buffer(a, 0x100, true);
        let rb = make_title_buffer(b, 0x100, true);
        // SAFETY: inside `mem`.
        unsafe {
            (a as *mut [u32; 4]).write([0x11, 0x12, 0x13, 0x14]);
            (b as *mut [u32; 4]).write([0x21, 0x22, 0x23, 0x24]);
        }
        let (copy_a, copy_b) = (shadow::find(a, 1).unwrap(), shadow::find(b, 1).unwrap());
        let (heap, _) = make_shader_heap(16);
        let mut state = State::default();
        state.set_heaps(&[heap]);
        let target = b + 4;
        state.set_constants(true, 0, &[target as u32, (target >> 32) as u32], 0);
        run(&mut state, &p);
        let word = |bytes: Vec<u8>| u32::from_le_bytes(bytes.try_into().unwrap());
        // Through a copy, so the result is in the copy (no copy-back).
        assert_eq!(word(copy_b.copy_bytes(b + 4, 4)), 0x44, "*p + *p at p");
        assert_eq!(word(copy_b.copy_bytes(b, 4)), 0x21, "the word before untouched");
        assert_eq!(word(copy_b.copy_bytes(b + 8, 4)), 0x23, "the word after untouched");
        assert_eq!(word(copy_a.copy_bytes(a + 4, 4)), 0x12, "the other buffer untouched");
        eprintln!("map test: {}", addresses_text());
        release(ra);
        release(rb);
        release(heap);
    }

    /// Class "a root UAV": the root arguments in the spill table, a raw
    /// descriptor built around the root UAV's address, stored through at
    /// 16 bytes per lane: the stores land in title memory from the address
    /// the title set, and nowhere else.
    #[test]
    fn a_root_uav_write_lands_in_its_range() {
        if super::super::host::host().is_none() {
            return;
        }
        let words = [
            0x8703_FF01, 0x0000_00FF, // s_and_b32 s3, s1, 0xff
            0xBE82_0300, // s_mov_b32 s2, s0
            0xF410_0201, 0xFA00_0010, // s_load_dwordx16 s[8:23], s[2:3], 0x10
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xF404_0001, 0xFA00_0008, // s_load_dwordx2 s[0:1], s[2:3], 0x8
            0x3408_0084, // v_lshlrev_b32 v4, 4, v0
            0x7E00_020A, // v_mov_b32 v0, s10
            0xBE82_03C1, // s_mov_b32 s2, -1
            0xBE83_03FF, 0x1101_4FAC, // s_mov_b32 s3, 0x11014fac
            0xBF8C_C07F,
            0xBE81_1D90, // s_bitset1_b32 s1, 16
            0xE070_1000, 0x8000_0004, // buffer_store_dword v0, v4, s[0:3], 0 offen
            0xBF81_0000, // s_endpgm
        ];
        let layout = vec![param(RootType::Descriptor, 2, 16, 0), param(RootType::Descriptor, 2, 18, 1), param(RootType::Constants, 16, 20, 2)];
        let p = pipeline_for(&words, layout, vec![used(ResourceType::Uav, RootType::Descriptor, 1, 18, 1)], 2 << 1, true);
        assert_eq!(p.stages[0].root_uavs.len(), 1, "one root UAV parameter");
        let mem = super::super::resource::Memory::zeroed(256 << 10, 64 << 10).unwrap();
        let u = mem.address();
        let ru = make_title_buffer(u, 0x400, true);
        let (heap, _) = make_shader_heap(16);
        let mut state = State::default();
        state.set_heaps(&[heap]);
        state.set_view(true, 0, u);
        state.set_view(true, 1, u + 0x100);
        let mut constants = [0u32; 16];
        constants[2] = 0xABCD;
        state.set_constants(true, 2, &constants, 0);
        run(&mut state, &p);
        // SAFETY: inside `mem`; copied back after the submission.
        let title = unsafe { std::slice::from_raw_parts(u as *const u32, 0x100) };
        for (i, &w) in title.iter().enumerate() {
            let lane = i.checked_sub(0x40).filter(|d| d % 4 == 0 && d / 4 < 32);
            assert_eq!(w, if lane.is_some() { 0xABCD } else { 0 }, "dword {i:#x}");
        }
        eprintln!("root UAV test: {}", addresses_text());
        release(ru);
        release(heap);
    }

    /// The gate: pieces held back by `XNW_GATE`, and the descriptor words
    /// with `XNW_DESC=gpu`, are not bound; the rest are.
    #[test]
    fn pieces_held_back_are_not_bound() {
        use pipeline::host::piece_bound_with as bound;
        let held = |k: &[&str]| k.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        assert!((0..3).all(|g| bound(g, &held(&[]), false)));
        assert!(!bound(0, &held(&["uav"]), false) && bound(1, &held(&["uav"]), false));
        assert!((0..3).all(|g| !bound(g, &held(&["all"]), false)));
        assert!(bound(1, &held(&[]), true) && !bound(2, &held(&[]), true));
    }

    /// Releases one of our objects.
    fn release(o: This) {
        // SAFETY: our object's Release slot.
        let f: extern "system" fn(This) -> u32 = unsafe { crate::d3d12x::object::slot(o, 0x10) };
        f(o);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Tracked;

    /// A compute pipeline from HLSL with its root signature inside (tests).
    fn hlsl(source: &str) -> (ID3D12RootSignature, ID3D12PipelineState) {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        let mut code: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;
        // SAFETY: the source and names live through the call.
        unsafe { D3DCompile(source.as_ptr().cast(), source.len(), None, None, None, windows::core::s!("main"), windows::core::s!("cs_5_1"), 0, 0, &mut code, None) }.expect("compiles");
        let code = code.unwrap();
        let device = &super::super::host::host().unwrap().device;
        // SAFETY: the blob's own bytes.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).unwrap();
            let pso = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        }
    }

    /// VIctory's test: rejected records do not wrap the
    /// list's budget: of 128 garbage draws (2^26 vertex instances each once
    /// clamped) two run, of 600 garbage mesh dispatches sixteen.
    #[test]
    fn rejected_records_do_not_wrap_the_budget() {
        if std::env::var_os("XNW_AGILITY").is_none_or(|d| d.is_empty()) {
            println!("SKIPPED: set XNW_AGILITY to the Agility SDK folder to run on the host GPU");
            return;
        }
        let (Some(host), Some(gpu)) = (super::super::host::host(), super::super::host::host_device()) else { panic!("no host GPU") };
        let run = |kind: u32, per: u32, records: u32| -> usize {
            let bytes = u64::from(per * 4 * records);
            let (src, data) = host.shared_buffer(bytes).unwrap();
            // SAFETY: the mapping.
            unsafe { std::ptr::write_bytes(data, 0xff, bytes as usize) };
            let list = host.new_list().unwrap();
            let room = arg_room(&list, bytes + 4).unwrap();
            // SAFETY: a live buffer.
            assert!(super::super::indirect::bound(&list, unsafe { src.GetGPUVirtualAddress() }, None, (kind, per * 4, records), room.gpu));
            let buffer = arg_written(&list, room);
            let back = gpu.buffer(bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST).unwrap();
            // SAFETY: an open list; live buffers.
            unsafe {
                list.list.ResourceBarrier(&[super::super::host::transition(&buffer, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                list.list.CopyBufferRegion(&back, 0, &buffer, room.offset, bytes);
                list.list.ResourceBarrier(&[super::super::host::transition(&buffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)]);
            }
            list.close();
            host.execute(&[&list]);
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
            // SAFETY: a readback buffer the GPU is done with.
            let ran = unsafe {
                back.Map(0, None, Some(&mut p)).unwrap();
                let w = std::slice::from_raw_parts(p.cast::<u32>(), (bytes / 4) as usize);
                let n = (0..records as usize).filter(|&r| w[r * per as usize] != 0).count();
                back.Unmap(0, None);
                n
            };
            list.reset();
            ran
        };
        assert_eq!(run(0, 4, 128), 2);
        assert_eq!(run(10, 3, 600), 16);
    }
    /// A root SRV bound as a one-view table of 64 KiB: reads inside
    /// see the buffer, reads past it (16 MiB on) read zeros, and the device
    /// stays up.
    #[test]
    fn root_srv_tables_read_zeros_past_their_view() {
        if std::env::var_os("XNW_AGILITY").is_none_or(|d| d.is_empty()) {
            println!("SKIPPED: set XNW_AGILITY to the Agility SDK folder to run on the host GPU");
            return;
        }
        let (Some(host), Some(gpu)) = (super::super::host::host(), super::super::host::host_device()) else { panic!("no host GPU") };
        let (root, pso) = hlsl(concat!(
            "#define RS \"DescriptorTable(SRV(t3, space=1)), UAV(u0)\"\n",
            "ByteAddressBuffer v : register(t3, space1);\nRWByteAddressBuffer o : register(u0);\n",
            "[RootSignature(RS)][numthreads(1,1,1)] void main() { o.Store4(0, uint4(v.Load(0), v.Load(4096), v.Load(65532), v.Load(65536))); o.Store(16, v.Load(16 << 20)); }"
        ));
        let heap = descriptor::host::HostHeap::new(false, 16, 0).expect("a heap");
        let (src, data) = host.shared_buffer(1 << 20).unwrap();
        // SAFETY: the mapping, 1 MiB.
        unsafe {
            for i in 0..(1u32 << 18) {
                data.cast::<u32>().add(i as usize).write(i + 1);
            }
        }
        // A 64 KiB view at 64 KiB into the buffer: dword i of the view holds 16385 + i.
        heap.raw_at(0, &src, 64 << 10, 64 << 10);
        let out = gpu.buffer(256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS).unwrap();
        let back = gpu.buffer(256, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST).unwrap();
        let list = host.new_list().unwrap();
        let l = &list.list;
        // SAFETY: an open list; live heap, buffers and pipeline.
        unsafe {
            l.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
            l.SetComputeRootSignature(&root);
            l.SetPipelineState(&pso);
            l.SetComputeRootDescriptorTable(0, heap.gpu_handle(0));
            l.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
            l.Dispatch(1, 1, 1);
            l.ResourceBarrier(&[super::super::host::transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            l.CopyBufferRegion(&back, 0, &out, 0, 20);
        }
        list.close();
        host.execute(&[&list]);
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: a readback buffer the GPU is done with.
        let v = unsafe {
            back.Map(0, None, Some(&mut p)).unwrap();
            let v = std::slice::from_raw_parts(p.cast::<u32>(), 5).to_vec();
            back.Unmap(0, None);
            v
        };
        assert_eq!(v, vec![16385, 16385 + 1024, 16385 + 16383, 0, 0]);
        // SAFETY: a plain query.
        assert!(unsafe { gpu.device.GetDeviceRemovedReason() }.is_ok());
        list.reset();
    }
    /// An indirect dispatch whose arguments a compute pass in the same
    /// list wrote runs with their count through our clamp (default) and from
    /// the title's copy (`off`); with arguments the processor wrote, from
    /// upload memory filled at submission (`cpu`).
    #[test]
    fn indirect_dispatch_counts_in_each_clamp_mode() {
        if std::env::var_os("XNW_AGILITY").is_none_or(|d| d.is_empty()) {
            println!("SKIPPED: set XNW_AGILITY to the Agility SDK folder to run on the host GPU");
            return;
        }
        let (Some(host), Some(gpu)) = (super::super::host::host(), super::super::host::host_device()) else { panic!("no host GPU") };
        let (wroot, wpso) = hlsl("#define RS \"UAV(u0)\"\nRWByteAddressBuffer a : register(u0);\n[RootSignature(RS)][numthreads(1,1,1)] void main() { a.Store3(0, uint3(6, 1, 1)); }");
        let (croot, cpso) = hlsl("#define RS \"UAV(u0)\"\nRWByteAddressBuffer n : register(u0);\n[RootSignature(RS)][numthreads(1,1,1)] void main() { uint o; n.InterlockedAdd(0, 1, o); }");
        let signature = host_signature(2, 12).unwrap();
        for mode in [ClampMode::Gpu, ClampMode::Off, ClampMode::Cpu] {
            // The title's copy (shared memory, as host copies are) and the
            // processor's title memory.
            let (title, _) = host.shared_buffer(256).unwrap();
            let cpu_args: Box<[u32; 3]> = Box::new([10, 1, 1]);
            let counter = gpu.buffer(256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS).unwrap();
            let back = gpu.buffer(256, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST).unwrap();
            let list = host.new_list().unwrap();
            let l = &list.list;
            // SAFETY: an open list; live buffers and pipelines.
            unsafe {
                // The culling-like pass writes {6, 1, 1} into the title's copy.
                l.SetComputeRootSignature(&wroot);
                l.SetPipelineState(&wpso);
                l.SetComputeRootUnorderedAccessView(0, title.GetGPUVirtualAddress());
                l.Dispatch(1, 1, 1);
                l.ResourceBarrier(&[uav_barrier()]);
            }
            let source = match mode {
                ClampMode::Gpu => {
                    let room = arg_room(&list, 64).unwrap();
                    // SAFETY: a live buffer.
                    assert!(super::super::indirect::bound(&list, unsafe { title.GetGPUVirtualAddress() }, None, (2, 12, 1), room.gpu));
                    (arg_written(&list, room), room.offset)
                }
                ClampMode::Off => {
                    off_moves(&list, &title, None, true);
                    (title.clone(), 0)
                }
                ClampMode::Cpu => (cpu_arguments(&list, cpu_args.as_ptr() as u64).unwrap(), 0),
            };
            // SAFETY: as above.
            unsafe {
                l.SetComputeRootSignature(&croot);
                l.SetPipelineState(&cpso);
                l.SetComputeRootUnorderedAccessView(0, counter.GetGPUVirtualAddress());
                l.ExecuteIndirect(&signature, 1, &source.0, source.1, None, 0);
                l.ResourceBarrier(&[super::super::host::transition(&counter, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                l.CopyBufferRegion(&back, 0, &counter, 0, 4);
            }
            list.close();
            host.execute(&[&list]);
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
            // SAFETY: a readback buffer the GPU is done with.
            let groups = unsafe {
                back.Map(0, None, Some(&mut p)).unwrap();
                let v = p.cast::<u32>().read();
                back.Unmap(0, None);
                v
            };
            assert_eq!(groups, if mode == ClampMode::Cpu { 10 } else { 6 }, "{mode:?}");
            list.reset();
        }
    }
    /// Every console topology becomes one D3D12 defines, of the type
    /// a pipeline must have for it, or none (its draws are dropped).
    #[test]
    fn console_topologies_map_to_defined_host_ones() {
        let defined = |h: u32| matches!(h, 1..=5 | 10..=13 | 33..=64);
        for t in 0..=255u32 {
            if let Some(h) = host_topology(t) {
                assert!(defined(h), "console {t} -> host {h}");
            }
        }
        assert_eq!(host_topology(RECTANGLES), Some(4));
        assert_eq!(host_topology(LINE_LOOP), Some(3));
        for t in [0, 6 + 1, 8, 9, 14, 16, 21, 32, 65, 0x80] {
            assert_eq!(host_topology(t), None, "console {t}");
        }
        assert_eq!([1, 2, 3, 4, 5, 10, 13, 33].map(topology_type), [1, 2, 2, 3, 3, 2, 3, 4]);
        assert_eq!(topology_type(host_topology(RECTANGLES).unwrap()), 3, "drawn with triangle pipelines");
    }
    /// An indirect draw's records are clamped from the title's host copy
    /// into our own room, the copy itself untouched and never transitioned;
    /// the room's chunk ends readable as arguments, explicitly. With the
    /// validation layer on (`XNW_GPU_DEBUG=layer`), the submission adds no
    /// error.
    #[test]
    fn indirect_records_are_clamped_into_our_room() {
        if std::env::var_os("XNW_AGILITY").is_none_or(|d| d.is_empty()) {
            println!("SKIPPED: set XNW_AGILITY to the Agility SDK folder to run on the host GPU");
            return;
        }
        let (Some(host), Some(gpu)) = (super::super::host::host(), super::super::host::host_device()) else { panic!("no host GPU") };
        let errors = || -> u64 {
            use windows::core::Interface;
            // SAFETY: plain queries.
            gpu.device.cast::<ID3D12InfoQueue>().map_or(0, |q| unsafe { q.GetNumStoredMessages() })
        };
        let before = errors();
        let (title, data) = host.shared_buffer(256).expect("a host copy");
        let words = data.cast::<u32>();
        // SAFETY: the buffer's own mapping, 256 bytes.
        unsafe {
            for (i, w) in [u32::MAX, u32::MAX, 7, 8, 9].into_iter().enumerate() {
                words.add(i).write(w);
            }
            words.add(16).write(5);
        }
        // SAFETY: a live buffer.
        let at = unsafe { title.GetGPUVirtualAddress() };
        let list = host.new_list().unwrap();
        let room = arg_room(&list, 64).expect("room");
        assert!(super::super::indirect::bound(&list, at, Some(at + 64), (1, 20, 1), room.gpu));
        let buffer = arg_written(&list, room);
        let key = std::ptr::from_ref(&list) as usize;
        assert_eq!(crate::mesh::with_arena(gpu, key, false, |a| a.args_states()), vec![Tracked::Readable]);
        let back = gpu.buffer(256, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST).unwrap();
        // SAFETY: an open list; live buffers; the room is 64 bytes at its offset.
        unsafe {
            list.list.ResourceBarrier(&[super::super::host::transition(&buffer, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            list.list.CopyBufferRegion(&back, 0, &buffer, room.offset, 64);
            list.list.ResourceBarrier(&[super::super::host::transition(&buffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)]);
        }
        list.close();
        host.execute(&[&list]);
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: a readback buffer of 256 bytes the GPU is done with.
        let out = unsafe {
            back.Map(0, None, Some(&mut p)).unwrap();
            let v = std::slice::from_raw_parts(p.cast::<u32>(), 6).to_vec();
            back.Unmap(0, None);
            v
        };
        // Index count to 2^24, instances to 2^26 / count; the rest copied;
        // the count (5) cut to the one record.
        assert_eq!(out, vec![1 << 24, 4, 7, 8, 9, 1]);
        // SAFETY: the mapping, read after the submission.
        assert_eq!(unsafe { [words.read(), words.add(1).read()] }, [u32::MAX; 2], "the title's copy is only read");
        assert_eq!(errors(), before, "the validation layer reported messages");
        list.reset();
    }
}

#[cfg(test)]
mod root_tests {
    use super::{root_view, shadow};

    /// A root SRV at title memory no buffer object covers reads the
    /// title's bytes (through a host copy of the memory around it, brought
    /// up to date at submission: written after the bind, before it).
    #[test]
    fn root_view_outside_buffers_reads_title_memory() {
        if std::env::var_os("XNW_AGILITY").is_none_or(|d| d.is_empty()) {
            println!("SKIPPED: set XNW_AGILITY to the Agility SDK folder to run on the host GPU");
            return;
        }
        let Some(host) = super::super::host::host() else { panic!("no host submission queue") };
        const LEN: usize = 2 << 20;
        // SAFETY: a fresh allocation of our own.
        let mem = unsafe { xnw_core::win::VirtualAlloc(std::ptr::null(), LEN, xnw_core::win::MEM_COMMIT | xnw_core::win::MEM_RESERVE, xnw_core::win::PAGE_READWRITE) }.cast::<u8>();
        assert!(!mem.is_null());
        let address = mem as u64 + 0x1_2340;
        assert!(!shadow::in_buffer(address, 1), "no buffer object there");
        let list = host.new_list().unwrap();
        let (_, left, how) = root_view(&list, address, true).expect("a root view outside buffers resolves");
        assert_eq!(how, "host copy");
        assert!(left >= 1 << 20, "the whole bound span is copied ({left} bytes)");
        let pattern: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(13).wrapping_add(5)).collect();
        // SAFETY: inside our allocation; written after the bind.
        unsafe { std::ptr::copy_nonoverlapping(pattern.as_ptr(), address as *mut u8, pattern.len()) };
        list.close();
        host.execute(&[&list]);
        // The copy root_view used: asked for the span it binds (a 1-byte ask
        // names another window where the span crosses a 4 MiB boundary).
        let copy = shadow::find(address, 1 << 20).unwrap();
        assert_eq!(copy.bytes_at(address, pattern.len()), pattern, "the host copy holds the title's bytes");
        list.reset();
        // As the runtime does when title memory goes: its copies go first
        // (a later allocation at the same address must not find this one).
        drop(copy);
        shadow::evict_loose(mem as usize, LEN);
        // SAFETY: our own allocation.
        unsafe { xnw_core::win::VirtualFree(mem.cast(), 0, xnw_core::win::MEM_RELEASE) };
    }
}
