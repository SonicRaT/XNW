// With VIctory (4322b61 and 42538cd), GPL-3.0-or-later: the runtime stride table (no raw-view pool).
// With VIctory main 86a5ed6 and 4051149 (GPL-3.0-or-later): buffer views off their copy's elements get their own copy (`view_copy`, `shadow::for_view`); views empty or past their copy are null (`in_buffer`).
//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed, or when a bindless
//! `ResourceDescriptorHeap[]` draw or dispatch materializes and refreshes
//! deferred buffer, constant-buffer, and placed-texture slots.
//!
//! Ported from VIctory-base (VIctory `d2a4bae` + open pull requests #135,
//! #651, #581, #552, #413 merged by hand; branch `xnw/base` @ 5d0516e,
//! GPL-3.0-or-later) `crates/victory-gpu/src/descriptor/host.rs` 1-3163,
//! whole. Changed: `tracing` -> `xnw_core` log lines; `victory_shader` ->
//! `xnw_shader`; the `diagnostics` feature's copy statistics dropped and
//! `describe_slot` kept unconditionally; base paths `crate::{host, shadow,
//! watch, texture, resource, list, queue}` are `crate::d3d12x::*`; two tests' `HostHeap::new(false, 4)` (a merge slip: #552's
//! tests against #413's three-argument `new`) take the title start 0;
//! Plan counters ([`plan_counts`]) are ours. Within a presented frame a
//! plan is extended with the slots made since instead of rebuilt from all
//! slots (once per frame per heap).

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use xnw_shader::emit::StrideEntry;

use crate::d3d12x;
use super::{VIEW_TAG, View, gate, stale_views, view_at};
use d3d12x::host::host;

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more.
    count: u32,
    /// Dynamic slots past `count` (none in sampler heaps): [`DYNAMIC`]
    /// where the host takes them, else [`MIN_DYNAMIC`].
    dynamic: u32,
    sampler: bool,
    /// The record (view index and generation) each slot was last made from.
    made: Mutex<Made>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    kept: Mutex<Vec<Option<ID3D12Resource>>>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Vec<bool>)>,
    /// Populated buffer, constant-buffer, and placed-texture slots (`< count`)
    /// tracked for bindless `ResourceDescriptorHeap[]` materialization.
    bindless: Mutex<Bindless>,
    /// A recording thread is preparing the placed-texture slots (others
    /// skip it rather than wait).
    recording: std::sync::atomic::AtomicBool,
    /// Placed-texture slots are waiting to be prepared at recording.
    record_hint: std::sync::atomic::AtomicBool,
    /// The placed-texture epoch, and the frame (plus one), recording last
    /// looked for evicted textures at.
    record_looked: [AtomicU64; 2],
    /// The frame the oldest uploaded texture is due a look again in.
    recheck_at: AtomicU64,
    /// [`TEXTURES_READY`] as recording last looked, and whether slots wait
    /// for the texture thread there.
    record_ready: AtomicU64,
    record_waits: std::sync::atomic::AtomicBool,
    /// Per slot, the stride of the buffer view there, for shaders that read
    /// a buffer whose stride translation could not recover (VIctory's
    /// entry layout); none for sampler heaps.
    strides: Option<StrideTable>,
    /// View heaps: the console words of each slot's descriptor, 8 dwords
    /// per slot, for shaders that read descriptors as numbers (`t0` in
    /// [`xnw_shader::emit::DESCRIPTOR_WORDS_SPACE`]): the buffer, its
    /// mapping and its host GPU address. VIctory (fe078c6,
    /// GPL-3.0-or-later); XNW: made on first ask, not with the heap
    /// (32 bytes per slot, 40 MiB for a heap of a million).
    words: std::sync::OnceLock<Option<(ID3D12Resource, usize, u64)>>,
}

/// View heaps keep this many permanent null
/// slots, one per kind (texture SRV 2D, raw buffer SRV, raw buffer UAV,
/// texture UAV 2D, CBV, in that order), just below the sink (the last
/// dynamic slot): translated code that bounds its heap indices reads them
/// in place of an index out of range.
pub(crate) const GUARDS: u32 = 5;

/// The guard slots of a view heap of `count` title slots and `dynamic`
/// more: the [`GUARDS`] slots below the last.
pub(crate) fn guard_slots(count: u32, dynamic: u32) -> Option<[u32; GUARDS as usize]> {
    (dynamic > GUARDS + 1).then(|| {
        let base = count + dynamic - 1 - GUARDS;
        std::array::from_fn(|k| base + k as u32)
    })
}

/// The heap-bounds root constants for view heap
/// `views` and sampler heap `samplers` bound: view slots, the five guards,
/// sampler slots, the sampler guard (slot 0: every sampler slot holds a
/// valid sampler, made point samplers at creation). A missing heap gives 0
/// slots: every index is then out of range and guarded.
pub(crate) fn bounds_words(views: Option<&HostHeap>, samplers: Option<&HostHeap>) -> [u32; 8] {
    let mut w = [0u32; 8];
    if let Some(v) = views
        && let Some(g) = v.guards()
    {
        w[0] = v.slots();
        w[1..6].copy_from_slice(&g);
    }
    w[6] = samplers.map_or(0, HostHeap::slots);
    w
}

/// A GFX10 buffer descriptor's dword 3 for raw access: X Y Z W selects, a
/// 32-bit float format, raw out-of-range checking, resource level 1. The
/// address and size in dwords 0 to 2 are what programs read. VIctory
/// (fe078c6, GPL-3.0-or-later).
const RAW_DWORD3: u32 = 0x3101_6FAC;

/// The console words of a descriptor for `view`, as far as programs read
/// them as numbers: a buffer descriptor's address, stride and record count
/// for buffer views, constant buffers and acceleration structures (whose
/// descriptor is a buffer at the structure); zero otherwise. VIctory
/// (fe078c6, GPL-3.0-or-later); changed: the stride from
/// [`view_stride`] (VIctory reads dword 7 for both kinds, which in an
/// unordered access description is the counter offset's high half).
pub(crate) fn console_words(view: &View) -> [u32; 8] {
    let (address, len, stride) = match view {
        View::Constant(address, size) => (*address, u64::from(*size), 0),
        // An acceleration structure (dimension 11) names no resource: its
        // location is the union's first quadword.
        View::Shader(0, Some(d)) if d[1] == 11 => (u64::from(d[4]) | u64::from(d[5]) << 32, u64::from(u32::MAX), 0),
        v => match buffer_range(v) {
            Some((address, len)) => (address, len, view_stride(v)),
            None => return [0; 8],
        },
    };
    let records = if stride > 0 { len / u64::from(stride) } else { len };
    [address as u32, (address >> 32) as u32 & 0xFFFF | (stride & 0x3FFF) << 16, records.min(u64::from(u32::MAX)) as u32, RAW_DWORD3, 0, 0, 0, 0]
}

/// The runtime stride table: a mapped upload buffer of one
/// [`StrideEntry`] per view-heap slot (VIctory 4322b61,
/// 42538cd, GPL-3.0-or-later; the shader reads it as `t0` in
/// [`xnw_shader::emit::STRIDE_SPACE`]).
/// Changed from VIctory: no raw-view pool; every entry names the
/// slot itself, so the raw handle reads the title's own structured or typed
/// view. VIctory measured that right on AMD and WARP, wrong on
/// NVIDIA; this host is an RX 5700 XT.
struct StrideTable {
    resource: ID3D12Resource,
    ptr: usize,
    gpu: u64,
    len: u32,
}

impl StrideTable {
    fn new(device: &ID3D12Device, len: u32) -> Option<Self> {
        let bytes = (u64::from(len) * u64::from(StrideEntry::BYTES)).max(8);
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut resource: Option<ID3D12Resource> = None;
        let mut p = std::ptr::null_mut();
        // SAFETY: plain committed upload buffer creation and persistent CPU map.
        let gpu = unsafe {
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut resource,
                )
                .ok()?;
            let res = resource.as_ref()?;
            res.Map(0, None, Some(&mut p)).ok()?;
            std::ptr::write_bytes(p.cast::<u8>(), 0, bytes as usize);
            res.GetGPUVirtualAddress()
        };
        Some(Self { resource: resource?, ptr: p as usize, gpu, len })
    }
}

// SAFETY: D3D12 descriptor heaps and resources are free-threaded; mutable
// slot tracking is synchronized behind mutexes and atomics.
unsafe impl Send for HostHeap {}
// SAFETY: as above.
unsafe impl Sync for HostHeap {}

const BOUNCE_CHUNK: u64 = 256 << 10;

/// Persistent 256-byte-aligned bounce slice for a misaligned bindless CBV.
struct BounceSlot {
    gpu: u64,
    cpu: usize,
    cap: u32,
    resource: ID3D12Resource,
    active: bool,
}

#[derive(Default)]
struct Bindless {
    /// Start address of the title's shader-visible descriptor memory (0 when
    /// detached or in standalone unit tests without a title heap).
    title_start: u64,
    /// Last observed `(generation << 32 | view_index)` key per title slot (0
    /// if no valid tag), so raw byte copies/overwrites into the title heap
    /// that bypass `mirror()` are detected on bindless sync.
    seen: Vec<u64>,
    /// Last `watch::writes` asking number scanned for byte-copied slots (0 =
    /// initial full scan not yet done).
    scan_asked: u64,
    /// Active `View::is_buffer()` slots: `slot_index -> (view_index, generation)`.
    slots: BTreeMap<u32, (u32, u32)>,
    /// Of them, those viewing placed textures.
    placed_set: BTreeMap<u32, (u32, u32)>,
    /// Placed-texture slots changed since draws last prepared them.
    record_pending: BTreeSet<u32>,
    /// Persistent 256-byte-aligned bounce slices by `slot_index`.
    cbv_bounce: HashMap<u32, BounceSlot>,
    /// Backing 256 KiB shared buffers `(resource, cpu_ptr, gpu_va, used_bytes)`.
    bounce_chunks: Vec<(ID3D12Resource, usize, u64, u64)>,
    /// Slots written since the plan was made: only they are made again.
    dirty: BTreeSet<u32>,
    /// What each slot made for bindless access needs at submission.
    uses: BTreeMap<u32, SlotUse>,
    /// The released-view and placed-texture epochs `uses` was checked at.
    checked: (u64, u64),
    /// What submissions do for the slots, until they or what they view change.
    plan: Option<Arc<BindlessPlan>>,
    /// Placed-texture slots as last prepared at recording: view key,
    /// texture.
    recorded: HashMap<u32, (u64, usize)>,
    /// Placed textures reached through slots, by resource object: the host
    /// texture (held, so its address names it alone) and the frame last
    /// uploaded.
    uploaded: HashMap<usize, (ID3D12Resource, u64)>,
    /// Uploaded textures by frame, oldest first, to look at again.
    recheck: VecDeque<(u64, usize)>,
    /// A frame and the uploads of known textures done in it.
    rechecks: (u64, u32),
    /// When the slot-making budget was last refilled, and the microseconds
    /// left: plans at submission, placed textures at recording.
    budget: Option<(std::time::Instant, [u64; 2])>,
    /// The frame (plus one) released views and evicted textures were last
    /// looked for in: at submission, at recording. Once per frame: the title
    /// releases and evicts all the time, and each look is over all slots.
    looked: [u64; 2],
    /// The last remapping ([`bindless_remapped`]) looked at.
    remap_seen: u64,
    /// Slots waiting for the texture thread: at submission, at recording;
    /// and [`TEXTURES_READY`] as last looked at by each.
    waiting: BTreeSet<u32>,
    record_waiting: BTreeSet<u32>,
    ready_seen: [u64; 2],
    /// Slots made in place, and the GPU address each was made at.
    in_place: HashMap<u32, u64>,
}

/// Time for making changed slots, at submission and at recording: an eighth
/// of the time passing (4 ms of a 30 fps frame), banked up to this many
/// microseconds; what is left over waits.
///
/// shortcut: while the title streams thousands of slots in at once, the
/// slots not made yet read nothing (null views) for a few frames, as
/// streaming pops in; making all of them at once stalled frames for
/// seconds. Tests make every slot at once: they share the device and the
/// CPU with each other, and on a busy machine 8 ms of wall clock left
/// slots (a UAV, a copied SRV) unmade at their first submission.
const SLOT_BUDGET_US: u64 = if cfg!(test) { 1 << 40 } else { 8_000 };
/// Slots taken to make at once (the rest wait without being looked at).
const SLOT_CHUNK: usize = 256;

impl Bindless {
    /// The slot-making budget, refilled for the time passed since.
    fn budget(&mut self) -> &mut [u64; 2] {
        let now = std::time::Instant::now();
        let (at, left) = self
            .budget
            .get_or_insert((now, [SLOT_BUDGET_US; 2]));
        let add = now.duration_since(*at).as_micros() as u64 / 8;
        if add > 0 {
            *at = now;
            for l in left.iter_mut() {
                *l = (*l + add).min(SLOT_BUDGET_US);
            }
        }
        left
    }
}

/// What one bindless slot needs at submission.
#[derive(Default)]
struct SlotUse {
    /// Title memory it reads (host copy, address, bytes).
    read: Option<(Arc<d3d12x::shadow::Shadow>, usize, usize)>,
    /// What shaders may write through it.
    write: Option<(Arc<d3d12x::shadow::Shadow>, usize, usize)>,
    /// A misaligned constant buffer: bounce slice, title address, bytes.
    bounce: Option<(usize, u64, u32)>,
    /// The host resource it views.
    retained: Option<ID3D12Resource>,
    /// It views a placed texture: the texture's object (0 for none), made
    /// again when the texture is evicted.
    placed: Option<usize>,
    /// It views unmapped memory (looked at again when memory is remapped).
    unmapped: bool,
    /// It reads title memory in place (address, bytes): nothing to
    /// refresh or copy back, made again when that memory is remapped.
    shared: Option<(u64, u64)>,
}

/// Placed textures reached only through bindless slots are uploaded when a
/// slot first reaches them, and then looked at again (their memory hashed)
/// at most this often, in frames, ...
///
/// shortcut: a texture the title rewrites in place without writing a slot
/// shows its new contents up to this many frames late through bindless
/// access (tables still upload what their draws read). Hashing all of them
/// at each submission cost seconds per frame while the world streams in.
const RECHECK_FRAMES: u64 = 60;
/// ... and at most this many of them per frame.
const RECHECKS: u32 = 8;

/// Bumped when views go stale or title memory is mapped or unmapped: what
/// bindless slots view may have changed though no slot was written (their
/// views' generations are checked, all at once, at the next submission).
static VIEWED: AtomicU64 = AtomicU64::new(0);

/// The released-view epoch ([`super::RELEASED`]) each heap (by
/// address) last nulled its slots at.
static SWEPT: Mutex<BTreeMap<usize, u64>> = Mutex::new(BTreeMap::new());

/// `XNW_GPU_CHECK` only: the path that last made each slot, by heap
/// address and slot.
static MADE_BY: Mutex<BTreeMap<(usize, u32), &'static str>> = Mutex::new(BTreeMap::new());

/// Each slot's made record, and the slots made from each view entry,
/// so slots of a released entry are found without walking the heap.
#[derive(Default)]
pub(crate) struct Made {
    keys: Vec<u64>,
    /// Slots by view entry they were made from (some since made again from
    /// another: checked against `keys` when used), and how many links.
    by_view: HashMap<u32, Vec<u32>>,
    links: usize,
}

impl std::ops::Deref for Made {
    type Target = Vec<u64>;
    fn deref(&self) -> &Vec<u64> {
        &self.keys
    }
}
impl std::ops::DerefMut for Made {
    fn deref_mut(&mut self) -> &mut Vec<u64> {
        &mut self.keys
    }
}

impl Made {
    /// Links slot `index` to the view entry its key now names; links gone
    /// stale are dropped when they outnumber the slots twice over.
    fn link(&mut self, index: u32) {
        let key = self.keys[index as usize];
        if key == 0 {
            return;
        }
        if self.links > 2 * self.keys.len() + 65_536 {
            let mut by_view: HashMap<u32, Vec<u32>> = HashMap::new();
            for (i, &k) in self.keys.iter().enumerate().filter(|&(_, &k)| k != 0) {
                by_view.entry(k as u32).or_default().push(i as u32);
            }
            self.links = by_view.values().map(Vec::len).sum();
            self.by_view = by_view;
        }
        self.by_view.entry(key as u32).or_default().push(index);
        self.links += 1;
    }
}

/// GPU ranges (start, end) of shared windows (heaps over title memory)
/// released, numbered, newest last: the memory goes back to the title (freed,
/// decommitted, unmapped) and may be anyone's next. Slots made in place there
/// are made null before the next submission ([`HostHeap::null_gone_windows`]).
static WINDOWS_GONE: Mutex<(u64, VecDeque<(u64, u64, u64)>)> = Mutex::new((0, VecDeque::new()));
/// The last window release each heap (by address) has nulled after.
static WINDOWS_SEEN: Mutex<BTreeMap<usize, u64>> = Mutex::new(BTreeMap::new());

/// The shared window at GPU `[gpu, gpu + bytes)` is released (its title
/// memory freed, decommitted or its section gone).
pub(crate) fn shared_window_gone(gpu: u64, bytes: u64) {
    let mut w = WINDOWS_GONE.lock().unwrap_or_else(|e| e.into_inner());
    w.0 += 1;
    let n = w.0;
    w.1.push_back((n, gpu, gpu.saturating_add(bytes)));
    if w.1.len() > 4096 {
        w.1.pop_front();
    }
}

/// A record kind word as text.
fn kind_name(kind: u32) -> &'static str {
    use super::{KIND_CBV, KIND_SAMPLER, KIND_SRV_RAW, KIND_SRV_STRUCTURED, KIND_SRV_TEXTURE, KIND_SRV_TYPED, KIND_UAV_RAW, KIND_UAV_STRUCTURED, KIND_UAV_TEXTURE, KIND_UAV_TYPED};
    match kind & 0xff {
        KIND_CBV => "constant buffer",
        KIND_SRV_RAW | KIND_SRV_TYPED | KIND_SRV_STRUCTURED => "buffer shader view",
        KIND_SRV_TEXTURE => "texture shader view",
        KIND_UAV_RAW | KIND_UAV_TYPED | KIND_UAV_STRUCTURED => "buffer unordered view",
        KIND_UAV_TEXTURE => "texture unordered view",
        KIND_SAMPLER => "sampler",
        _ => "unknown",
    }
}

/// `a / b / c` of three counters per frame over `n`, zeroing them.
fn three(c: &[AtomicU64; 3], n: u64) -> String {
    let v: Vec<u64> = c.iter().map(|x| x.swap(0, Ordering::Relaxed) / n.max(1)).collect();
    format!("{} / {} / {}", v[0], v[1], v[2])
}

/// Bindless plans made, reused and extended at submissions (ours, for the
/// run log).
static PLANS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

/// Bindless plans made anew and reused so far.
pub(crate) fn plan_counts() -> (u64, u64) {
    (PLANS[0].load(Ordering::Relaxed), PLANS[1].load(Ordering::Relaxed))
}

/// Bindless plans extended with the slots made since (not rebuilt) so far.
pub(crate) fn plans_extended() -> u64 {
    PLANS[2].load(Ordering::Relaxed)
}

/// Descriptor writes asked for slots past their host heap, sent to its sink
/// slot instead; for the statistics, never zeroed.
pub(crate) static SLOTS_OUT: AtomicU64 = AtomicU64::new(0);

/// Bindless counters for the frame statistics, zeroed by each line.
static STATS: [AtomicU64; STAT_NAMES.len()] = [const { AtomicU64::new(0) }; STAT_NAMES.len()];
const STAT_NAMES: [&str; 45] = [
    "reads examined",
    "reads skipped (unwritten)",
    "title slots examined",
    "title slots changed",
    "slots made",
    "heap scans whole",
    "us plan scan",
    "us plan look",
    "us plan make",
    "us plan build",
    "memory lookups cached",
    "memory lookups asked",
    "slots of remapped memory",
    "record calls with work",
    "record slots prepared",
    "record textures uploaded",
    "bounces examined",
    "bounces skipped (unwritten)",
    "reads refreshed: first",
    "reads refreshed: block open",
    "reads refreshed: written since",
    "KiB in reads",
    "us plan make in placed textures",
    "placed slots made in plans",
    "us plan look: live textures",
    "us plan look: slots walked",
    "us plan look: stale views",
    "heap blocks scanned",
    "us plan make: memory checks",
    "us plan make: in-place lookups",
    "us plan make: host slots",
    "us plan make: host copies found",
    "record slots put back (over budget)",
    "placed textures queued (not ready)",
    "placed textures made off-thread",
    "plans full (over the lines' frames, not per frame)",
    "plans extended (over the frames)",
    "plans reused (over the frames)",
    "plan parts at submissions (over the frames)",
    "us plan look: lock waits",
    "us plan look: remaps and checks",
    "us plan total (the parts above are in it)",
    "us plan lock waits (scan start, build)",
    "us plan old plan let go",
    "bounces copied plainly (memory mapped)",
];
const READS_EXAMINED: usize = 0;
const READS_SKIPPED: usize = 1;
const SLOTS_EXAMINED: usize = 2;
const SLOTS_CHANGED: usize = 3;
const SLOTS_MADE: usize = 4;
const WHOLE_SCANS: usize = 5;
const US_SCAN: usize = 6;
const US_LOOK: usize = 7;
const US_MAKE: usize = 8;
const US_BUILD: usize = 9;
const LOOKUPS_CACHED: usize = 10;
const LOOKUPS_ASKED: usize = 11;
const SLOTS_REMAPPED: usize = 12;
const RECORD_CALLS: usize = 13;
const RECORD_SLOTS: usize = 14;
const RECORD_UPLOADS: usize = 15;
const BOUNCES_EXAMINED: usize = 16;
const BOUNCES_SKIPPED: usize = 17;
const READS_NEVER: usize = 18;
const READS_OPEN: usize = 19;
const READS_NEWER: usize = 20;
const READ_KIB: usize = 21;
const US_MAKE_PLACED: usize = 22;
const PLACED_MADE: usize = 23;
const US_LOOK_LIVE: usize = 24;
const US_LOOK_WALK: usize = 25;
const US_LOOK_STALE: usize = 26;
const BLOCKS_SCANNED: usize = 27;
const US_MAKE_CHECK: usize = 28;
const US_MAKE_IN_PLACE: usize = 29;
const US_MAKE_SLOT: usize = 30;
const US_MAKE_COPY: usize = 31;
const RECORD_BACK: usize = 32;
const TEXTURES_QUEUED: usize = 33;
const TEXTURES_MADE_OFF: usize = 34;
const PLANS_FULL: usize = 35;
const PLANS_EXTENDED: usize = 36;
const PLANS_REUSED: usize = 37;
const PLAN_PARTS: usize = 38;
const US_LOOK_LOCK: usize = 39;
const US_LOOK_CHECKS: usize = 40;
const US_PLAN_TOTAL: usize = 41;
const US_PLAN_LOCKS: usize = 42;
const US_PLAN_DROP: usize = 43;
const BOUNCES_PLAIN: usize = 44;

/// VIctory: tables dropped for want of dynamic slots:
/// dynamic slots taken, the most in use at once, tables not made for want
/// of room (frame statistics); and the slots in use now.
static DYNAMIC_COUNTS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static DYNAMIC_IN_USE: AtomicU64 = AtomicU64::new(0);

/// The title heap's areas, by slot. Five-star-recomp measured the
/// same 1,245,184-slot heap (five-star-recomp 196-205):
/// persistent views 0-327,679 (written through the device's view calls),
/// then a gap, and from 360,448 the title's per-draw ring (tables the title
/// copies records into, mixed with other data). Ours has the same slot
/// count (log: "host view heap: 1245184 title slots"); the split of changes
/// by sixteenth (`CHANGED_SIXTEENTHS`) checks the boundaries in a run.
const PERSISTENT_END: u32 = 327_680;
const RING_START: u32 = 360_448;

fn area(index: u32) -> usize {
    if index < PERSISTENT_END { 0 } else if index < RING_START { 1 } else { 2 }
}

/// Adds the slots of `[first, last)` to `counts` by area.
fn area_count(counts: &[AtomicU64; 3], first: u32, last: u32) {
    for (i, (a, e)) in [(0, PERSISTENT_END), (PERSISTENT_END, RING_START), (RING_START, u32::MAX)].into_iter().enumerate() {
        let (lo, hi) = (first.max(a), last.min(e));
        if lo < hi {
            counts[i].fetch_add(u64::from(hi - lo), Ordering::Relaxed);
        }
    }
}

/// Per area (persistent, gap, ring): title slots examined, changed, made in
/// plans; changes by sixteenth of the heap (zeroed by each line).
static AREA_EXAMINED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static AREA_CHANGED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static AREA_MADE: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static CHANGED_SIXTEENTHS: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];

/// `XNW_PLAN_SKIP=ring` (off by default): plans neither scan nor make slots
/// of the ring area.
pub(crate) fn plan_skip_ring() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_PLAN_SKIP").is_ok_and(|v| v.trim() == "ring"));
    *ON
}

/// Draws and dispatches of programs that index the view heap directly
/// (their reads may land anywhere, the ring too) while `XNW_PLAN_SKIP=ring`
/// leaves ring slots unmade: the risk, counted.
pub(crate) static INDEXED_WHILE_SKIPPING: AtomicU64 = AtomicU64::new(0);

/// A plan reused at a submission (per frame in the statistics too).
fn plan_reused() {
    PLANS[1].fetch_add(1, Ordering::Relaxed);
    count(PLANS_REUSED, 1);
}

fn count_us(i: usize, since: std::time::Instant) {
    count(i, since.elapsed().as_micros() as u64);
}

/// [`count_us`], returning now (the next part's start).
fn count_us_next(i: usize, since: std::time::Instant) -> std::time::Instant {
    let now = std::time::Instant::now();
    count(i, (now - since).as_micros() as u64);
    now
}

/// Counts the microseconds from its start to its drop into counter `.0`.
struct Timed(usize, std::time::Instant);
impl Drop for Timed {
    fn drop(&mut self) {
        count_us(self.0, self.1);
    }
}

fn count(i: usize, n: u64) {
    STATS[i].fetch_add(n, Ordering::Relaxed);
}

/// The bindless part of the frame statistics, per frame over `n` frames,
/// zeroing the counts.
pub(crate) fn stats_text(n: u64) -> String {
    let parts: Vec<String> = STAT_NAMES
        .iter()
        .zip(&STATS)
        .enumerate()
        // Plan counts are a few per line during loads; per frame they
        // read 0.
        .map(|(i, (name, c))| format!("{} {name}", c.swap(0, Ordering::Relaxed) / if (PLANS_FULL..=PLAN_PARTS).contains(&i) { 1 } else { n }))
        .collect();
    let plans: Vec<u64> = PLANS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let snapshot: Vec<u64> = d3d12x::shadow::SNAPSHOT.iter().map(|c| c.swap(0, Ordering::Relaxed) / n).collect();
    let copied: Vec<String> = crate::shared::KIND_NAMES.iter().zip(&d3d12x::host::COPY_BYTES).map(|(k, c)| format!("{k} {}", c.swap(0, Ordering::Relaxed) / n >> 10)).collect();
    let resolves: Vec<u64> = crate::shared::RESOLVE_COUNTS.iter().map(|c| c.swap(0, Ordering::Relaxed) / n).collect();
    format!(
        "{}; by area (persistent 0-327,679 / gap / ring from 360,448) per frame: examined {}, changed {}, made {}; changes by sixteenth of the heap [{}]; heap-indexing draws while the ring is skipped {}; dynamic slots: {} taken per frame, at most {} in use at once (of {} a heap), {} tables not made for want of room; copied per frame by kind (KiB): {}; blocks unguarded now {}; in-place memory lookups {} cached, {} asked of xmem; write snapshots {} KiB in {} pages; plans so far {} made, {} extended, {} reused; slots out of range so far {}; blocks left unguarded (written every frame) so far {}",
        parts.join(", "),
        three(&AREA_EXAMINED, n),
        three(&AREA_CHANGED, n),
        three(&AREA_MADE, n),
        CHANGED_SIXTEENTHS.iter().map(|c| (c.swap(0, Ordering::Relaxed) / n).to_string()).collect::<Vec<_>>().join(" "),
        INDEXED_WHILE_SKIPPING.swap(0, Ordering::Relaxed) / n,
        DYNAMIC_COUNTS[0].swap(0, Ordering::Relaxed) / n,
        DYNAMIC_COUNTS[1].swap(DYNAMIC_IN_USE.load(Ordering::Relaxed), Ordering::Relaxed),
        DYNAMIC,
        DYNAMIC_COUNTS[2].swap(0, Ordering::Relaxed),
        copied.join(", "),
        d3d12x::watch::hot_now(),
        resolves[0],
        resolves[1],
        snapshot[0] >> 10,
        snapshot[1],
        plans[0],
        plans[2],
        plans[1],
        SLOTS_OUT.load(Ordering::Relaxed),
        d3d12x::watch::HOT_BLOCKS.load(Ordering::Relaxed)
    )
}

/// Bindless slots must be checked again: a resource their views name was
/// released, or title memory was mapped or unmapped.
pub(crate) fn bindless_viewed_changed() {
    VIEWED.fetch_add(1, Ordering::Release);
}

/// Title memory mapped, remapped or unmapped, numbered, newest last:
/// at their next look, heaps make again the slots that read such memory
/// outside buffer objects (their windows were let go), which then read
/// the new memory or, unmapped, nothing. Also ends [`committed`]'s answers.
static REMAPPED: Mutex<(u64, VecDeque<(u64, u64, u64)>)> = Mutex::new((0, VecDeque::new()));
/// Remappings kept for heaps to look at; a heap that missed older ones
/// looks at all its slots of loose memory.
const REMAPPED_KEEP: usize = 1 << 16;

/// The title remapped `[at, at + len)`.
pub(crate) fn bindless_remapped(at: u64, len: u64) {
    {
        let mut r = REMAPPED.lock().unwrap_or_else(|e| e.into_inner());
        r.0 += 1;
        let n = r.0;
        r.1.push_back((n, at, at.saturating_add(len)));
        if r.1.len() > REMAPPED_KEEP {
            r.1.pop_front();
        }
    }
    forget_committed(at, at.saturating_add(len));
    bindless_viewed_changed();
}

/// What was remapped since remapping `seen` (moved to the newest).
enum Remapped {
    /// Sorted, merged ranges.
    Ranges(Vec<(u64, u64)>),
    /// More than is kept.
    All,
}

fn remapped_since(seen: &mut u64) -> Remapped {
    let r = REMAPPED.lock().unwrap_or_else(|e| e.into_inner());
    let last = std::mem::replace(seen, r.0);
    if r.0 == last {
        return Remapped::Ranges(Vec::new());
    }
    if r.1.front().is_none_or(|&(n, ..)| n > last + 1) {
        return Remapped::All;
    }
    let mut ranges: Vec<(u64, u64)> = r.1.iter().filter(|&&(n, ..)| n > last).map(|&(_, a, e)| (a, e)).collect();
    drop(r);
    ranges.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::new();
    for (a, e) in ranges {
        match out.last_mut() {
            Some((_, end)) if a <= *end => *end = (*end).max(e),
            _ => out.push((a, e)),
        }
    }
    Remapped::Ranges(out)
}

impl Remapped {
    /// Whether slot use `u` reads loose memory that was remapped.
    fn hits(&self, u: &SlotUse) -> bool {
        let in_place = u.shared.is_some_and(|(a, l)| match self {
            Remapped::All => true,
            Remapped::Ranges(r) => {
                let at = r.partition_point(|&(_, e)| e <= a);
                r.get(at).is_some_and(|&(s, _)| s < a + l)
            }
        });
        let ranges = u.read.iter().map(|(_, a, l)| (*a as u64, *l as u64)).chain(u.bounce.iter().map(|&(_, a, l)| (a, u64::from(l))));
        let mut hit = in_place;
        for (a, l) in ranges {
            let moved = match self {
                Remapped::All => true,
                Remapped::Ranges(r) => {
                    let at = r.partition_point(|&(_, e)| e <= a);
                    r.get(at).is_some_and(|&(s, _)| s < a + l)
                }
            };
            hit |= moved && !d3d12x::shadow::in_buffer(a, l);
        }
        hit
    }
}

/// Title memory found committed, by region (start, end), until title
/// memory over them is remapped: a slot's memory was asked of the kernel
/// each time it was made (`VirtualQuery`, milliseconds in the big pools).
/// shortcut: memory decommitted without a remapping is still taken as
/// committed; its window copy then reads zeros, as a null view would.
///
/// A remapping forgets only the regions it overlaps (it cleared them
/// all, and the title remaps all the time: 240 asked of the kernel a frame
/// against 839 answered here). The number
/// counts forgettings, so an answer asked across one is not kept.
static COMMITTED: Mutex<(u64, BTreeMap<u64, u64>)> = Mutex::new((0, BTreeMap::new()));

/// The title remapped `[at, end)`: committed regions over it are forgotten.
fn forget_committed(at: u64, end: u64) {
    let mut c = COMMITTED.lock().unwrap_or_else(|e| e.into_inner());
    c.0 += 1;
    let gone: Vec<u64> = c.1.range(..end).filter(|&(_, &e)| e > at).map(|(&s, _)| s).collect();
    for s in gone {
        c.1.remove(&s);
    }
}

/// Whether `[address, address + len)` is committed, from [`COMMITTED`]
/// where it can.
pub(super) fn committed(address: u64, len: u64) -> bool {
    let end = address.saturating_add(len.max(1));
    let epoch;
    {
        let c = COMMITTED.lock().unwrap_or_else(|e| e.into_inner());
        epoch = c.0;
        if c.1.range(..=address).next_back().is_some_and(|(_, &e)| end <= e) {
            count(LOOKUPS_CACHED, 1);
            return true;
        }
    }
    count(LOOKUPS_ASKED, 1);
    let mut found = Vec::new();
    let mut p = address;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: xnw_core::win::MemoryBasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = xnw_core::syscalls::timed(xnw_core::syscalls::Call::QuerySlot, || unsafe { xnw_core::win::VirtualQueryEx(xnw_core::win::GetCurrentProcess(), p as *const _, &mut info, size_of_val(&info)) });
        if n == 0 || info.state != xnw_core::win::MEM_COMMIT {
            return false;
        }
        found.push((info.base as u64, (info.base + info.region_size) as u64));
        p = (info.base + info.region_size) as u64;
    }
    let mut c = COMMITTED.lock().unwrap_or_else(|e| e.into_inner());
    if c.0 == epoch {
        c.1.extend(found);
    }
    true
}

/// Bindless unordered access ranges larger than this are not copied back.
///
/// shortcut: copy-back finds what shaders wrote by keeping a range's bytes
/// before each submission; a bindless slot viewing a title's big shared heap
/// would cost a copy and a compare of all of it per submission whether or
/// not a shader writes there. Such ranges stay on the GPU's side.
const WRITE_CAP: usize = 8 << 20;

/// Gap below which two ranges of one host copy are refreshed as one: the
/// write-watch answers by 64 KiB page, and fewer, larger askings cost less.
/// One page, not 1 MiB. Ranges closer than a page touch no page the two
/// do not, so a range skipped as unwritten is no harder to skip; 1 MiB
/// merged up to 15 pages nobody reads, any write to which refreshed it all.
const READ_GAP: usize = d3d12x::watch::PAGE - 1;

/// What a submission does for a heap's bindless slots, made again only when
/// the slots or what they view change ([`HostHeap::bindless_plan`]).
///
/// A chain of parts, the frame's full plan first and then what each
/// later submission's slots added (only what the parts before do not do).
/// An extension cloned and merged the whole plan again (reads, writes,
/// bounces, held resources): 82 ms a frame over 22 plans in a test run.
pub(crate) struct BindlessPlan {
    parts: Vec<Arc<PlanPart>>,
    /// The frame (plus one) its first part was built from all slots in:
    /// later plans of that frame extend it with the slots made since.
    frame: u64,
}

/// One part of a [`BindlessPlan`].
struct PlanPart {
    /// Title memory the slots read, merged per host copy (address, bytes):
    /// refreshed at each submission using the heap.
    reads: Vec<(Arc<d3d12x::shadow::Shadow>, usize, usize)>,
    /// The unordered access slots' ranges (up to [`WRITE_CAP`] each), merged:
    /// copied back after a submission whose shaders may write memory.
    writes: Vec<(Arc<d3d12x::shadow::Shadow>, usize, usize)>,
    /// Misaligned constant buffers: bounce slice, title address, bytes.
    bounces: Vec<(usize, u64, u32)>,
    /// Per bounce, the write-watch asking it was last copied after (0:
    /// never), carried over from the plan before for the same bounce.
    bounce_asked: Mutex<Vec<u64>>,
    /// Per read, the write-watch asking noted before it was last brought
    /// up to date (0: never), carried over as `bounce_asked` is.
    read_asked: Mutex<Vec<u64>>,
    /// The host resources the slots view, held while a submission runs,
    /// each once.
    _retained: Vec<ID3D12Resource>,
    /// Of them, the raw pointers.
    held: std::collections::HashSet<usize>,
    /// The bounces by slice: (title address, bytes).
    bounce_at: HashMap<usize, (u64, u32)>,
    /// Title memory the slots read in place (start, end), merged: GPU
    /// writes still owed there go back before a submission using the heap
    ///
    in_place: Vec<(usize, usize)>,
}

/// `spans` (start, end) sorted and merged where they overlap or touch.
fn merged_spans(mut spans: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    spans.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (a, e) in spans {
        match out.last_mut() {
            Some((_, end)) if a <= *end => *end = (*end).max(e),
            _ => out.push((a, e)),
        }
    }
    out
}

// SAFETY: D3D12 resources are free-threaded; the rest is plain data.
unsafe impl Send for PlanPart {}
// SAFETY: as above.
unsafe impl Sync for PlanPart {}

impl BindlessPlan {
    /// The host resources the slots view (made resident at submission).
    pub(crate) fn retained(&self) -> impl Iterator<Item = &ID3D12Resource> {
        self.parts.iter().flat_map(|p| p._retained.iter())
    }

    /// The unordered access ranges (host copy, address, bytes), per part
    /// merged: copied back after a submission whose shaders may write
    /// memory.
    pub(crate) fn writes(&self) -> impl Iterator<Item = &(Arc<d3d12x::shadow::Shadow>, usize, usize)> {
        self.parts.iter().flat_map(|p| p.writes.iter())
    }

    /// Whether bounce `k` is the one the plan copies into its slice (the
    /// newest part bouncing that slice has it).
    fn bounces(&self, &(cpu, a, l): &(usize, u64, u32)) -> bool {
        self.parts.iter().rev().find_map(|p| p.bounce_at.get(&cpu)).is_some_and(|&had| had == (a, l))
    }

    /// For tests: all parts' reads, in order.
    #[cfg(test)]
    fn flat_reads(&self) -> Vec<(Arc<d3d12x::shadow::Shadow>, usize, usize)> {
        self.parts.iter().flat_map(|p| p.reads.iter().cloned()).collect()
    }

    /// For tests: all parts' bounces, in order.
    #[cfg(test)]
    fn flat_bounces(&self) -> Vec<(usize, u64, u32)> {
        self.parts.iter().flat_map(|p| p.bounces.iter().copied()).collect()
    }

    /// Classifies the plan's reads and bounces page by page as a
    /// submission would copy them, without copying.
    pub(crate) fn zero_copy_dry(&self) {
        for p in &self.parts {
            for (_, address, len) in &p.reads {
                crate::shared::zero_copy_note(d3d12x::host::COPY_BINDLESS, *address, *len);
            }
            for &(_, address, len) in &p.bounces {
                crate::shared::zero_copy_note(d3d12x::host::COPY_BOUNCE, address as usize, len as usize);
            }
        }
    }

    /// The parts in the chain (1: a full plan alone).
    pub(crate) fn parts(&self) -> usize {
        self.parts.len()
    }

    /// Before a submission using the heap: GPU writes still owed over what
    /// the slots read in place go back first (a GPU read of
    /// title memory never faults, so it read stale bytes). A lookup in the
    /// traps per range, nothing when none is set.
    pub(crate) fn settle_in_place(&self) {
        for p in &self.parts {
            for &(a, e) in &p.in_place {
                d3d12x::watch::reading(a, e - a);
            }
        }
    }

    /// Whether the plan already does all slot use `u` needs: its read and
    /// write inside ranges of the same host copies, no bounce, its resource
    /// held, its in-place range read (each in some part).
    fn covers(&self, u: &SlotUse) -> bool {
        u.bounce.is_none_or(|k| self.bounces(&k))
            && u.shared.as_ref().is_none_or(|s| self.parts.iter().any(|p| p.covers_in_place(s)))
            && u.retained.as_ref().is_none_or(|r| self.parts.iter().any(|p| p.holds(r)))
            && self.parts.iter().any(|p| inside(&p.reads, &u.read))
            && self.parts.iter().any(|p| inside(&p.writes, &u.write))
    }

    /// Brings the slots' reads up to date through `update`, but for those
    /// whose memory nobody wrote since this plan (or one before it with the
    /// same range) last did: one look at the write watch for all of them,
    /// no kernel call ([`d3d12x::watch::unchanged`]). The reads were each
    /// asked about at every submission: 50,888 a frame, most unwritten.
    pub(crate) fn refresh_reads(&self, mut update: impl FnMut(&Arc<d3d12x::shadow::Shadow>, usize, usize)) {
        for p in &self.parts {
            p.refresh_reads(&mut update);
        }
    }

    /// Copies the misaligned constant buffers into their bounce slices,
    /// those whose memory the title wrote since they were last copied (all,
    /// where it is not write-watched). A slice a later part bounces again
    /// is left to it: newest parts first, each slice once.
    ///
    /// The title's constants are in the paired reservation (section views),
    /// which the write watch answers for only when asked to guard it: asked
    /// without, every bounce was copied (9,127 a frame, 326 skipped).
    /// Bounces unwritten since are first
    /// found all at once without the kernel ([`d3d12x::watch::unchanged`]).
    pub(crate) fn copy_constants(&self) {
        let mut later = std::collections::HashSet::new();
        for (i, p) in self.parts.iter().enumerate().rev() {
            p.copy_constants(&later);
            if i > 0 {
                later.extend(p.bounces.iter().map(|b| b.0));
            }
        }
    }
}

/// Whether `r` lies inside one range of the same host copy in `list`.
fn inside(list: &[(Arc<d3d12x::shadow::Shadow>, usize, usize)], r: &Option<(Arc<d3d12x::shadow::Shadow>, usize, usize)>) -> bool {
    let Some((s, a, l)) = r else { return true };
    let key = (Arc::as_ptr(s) as usize, *a);
    // Sorted by (copy, address) and disjoint per copy (`merged`).
    let at = list.partition_point(|(t, b, _)| (Arc::as_ptr(t) as usize, *b) <= key);
    at > 0 && {
        let (t, b, m) = &list[at - 1];
        Arc::ptr_eq(t, s) && *b <= *a && a + l <= b + m
    }
}

impl PlanPart {
    fn covers_in_place(&self, &(a, l): &(u64, u64)) -> bool {
        let (a, e) = (a as usize, (a + l) as usize);
        let at = self.in_place.partition_point(|&(s, _)| s <= a);
        at > 0 && self.in_place[at - 1].1 >= e
    }

    fn holds(&self, r: &ID3D12Resource) -> bool {
        self.held.contains(&(windows::core::Interface::as_raw(r) as usize))
    }

    fn refresh_reads(&self, update: &mut impl FnMut(&Arc<d3d12x::shadow::Shadow>, usize, usize)) {
        let mut asked = self.read_asked.lock().unwrap_or_else(|e| e.into_inner());
        let ranges: Vec<(usize, usize, u64)> =
            self.reads.iter().zip(asked.iter()).map(|(&(_, a, l), &since)| (a, a + l, since)).collect();
        let unchanged = d3d12x::watch::unchanged(&ranges);
        // Noted before any update: a write found by one is journalled after
        // it, so the range is brought up to date once more, never missed.
        let before = d3d12x::watch::asked();
        let mut why = [0u64; 4];
        for ((shadow, address, len), (last, answer)) in self.reads.iter().zip(asked.iter_mut().zip(unchanged)) {
            why[answer as usize] += 1;
            if answer == d3d12x::watch::Unchanged::Yes {
                continue;
            }
            update(shadow, *address, *len);
            *last = before;
        }
        count(READS_EXAMINED, self.reads.len() as u64);
        count(READS_SKIPPED, why[0]);
        count(READS_NEVER, why[1]);
        count(READS_OPEN, why[2]);
        count(READS_NEWER, why[3]);
        let bytes: usize = self.reads.iter().map(|r| r.2).sum();
        count(READ_KIB, bytes as u64 >> 10);
    }

    /// [`BindlessPlan::copy_constants`] for this part, but for the bounce
    /// slices in `later` (bounced by a newer part).
    fn copy_constants(&self, later: &std::collections::HashSet<usize>) {
        use d3d12x::host::{COPY_BOUNCE, COPY_BOUNCE_SKIPPED, COPY_KINDS};
        let mut asked = self.bounce_asked.lock().unwrap_or_else(|e| e.into_inner());
        let ranges: Vec<(usize, usize, u64)> = self
            .bounces
            .iter()
            .zip(asked.iter())
            .map(|(&(_, a, l), &since)| (a as usize, a as usize + l as usize, since))
            .collect();
        let unchanged = d3d12x::watch::unchanged(&ranges);
        count(BOUNCES_EXAMINED, self.bounces.len() as u64);
        for ((&(cpu, address, len), last), same) in self.bounces.iter().zip(asked.iter_mut()).zip(unchanged) {
            if later.contains(&cpu) {
                continue;
            }
            let (from, to) = (address as usize, address as usize + len as usize);
            if same == d3d12x::watch::Unchanged::Yes {
                count(BOUNCES_SKIPPED, 1);
                COPY_KINDS[COPY_BOUNCE_SKIPPED].fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if let Some((now, written)) = d3d12x::watch::writes(from, to, true) {
                if *last != 0 && written.iter().all(|&(_, w)| w <= *last) {
                    count(BOUNCES_SKIPPED, 1);
                    COPY_KINDS[COPY_BOUNCE_SKIPPED].fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                *last = now;
            }
            COPY_KINDS[COPY_BOUNCE].fetch_add(1, Ordering::Relaxed);
            crate::shared::would_share(COPY_BOUNCE, address as usize, len as usize);
            let started = std::time::Instant::now();
            // A plain copy where xmem knows the memory mapped (a
            // kernel copy each cost ~16 us with its write-watch ask, 30k
            // bounces a frame in a test run); the kernel's otherwise.
            if crate::shared::mapped(address as usize, len as usize) {
                // SAFETY: mapped title memory (xmem) into our bounce slice.
                unsafe { std::ptr::copy_nonoverlapping(address as *const u8, cpu as *mut u8, len as usize) };
                count(BOUNCES_PLAIN, 1);
            } else {
                d3d12x::host::copy_mapped(cpu as *mut u8, address as *const u8, len as usize, false);
            }
            let us = started.elapsed().as_micros() as u64;
            d3d12x::host::PREPARING[0].fetch_add(us, Ordering::Relaxed);
            d3d12x::host::PREPARING[1].fetch_add(u64::from(len), Ordering::Relaxed);
            d3d12x::host::stage(d3d12x::host::STAGE_COPY, started);
        }
    }
}

/// `ranges` (host copy, address, bytes) sorted and merged per host copy
/// where they overlap or lie within `gap` of each other.
fn merged(
    mut ranges: Vec<(Arc<d3d12x::shadow::Shadow>, usize, usize)>,
    gap: usize,
) -> Vec<(Arc<d3d12x::shadow::Shadow>, usize, usize)> {
    ranges.sort_unstable_by_key(|(s, a, _)| (Arc::as_ptr(s) as usize, *a));
    let mut out: Vec<(Arc<d3d12x::shadow::Shadow>, usize, usize)> = Vec::new();
    for (shadow, address, len) in ranges {
        if let Some((s, a, l)) = out.last_mut()
            && Arc::ptr_eq(s, &shadow)
            && address <= *a + *l + gap
        {
            *l = (*l).max(address + len - *a);
            continue;
        }
        out.push((shadow, address, len));
    }
    out
}

impl Drop for HostHeap {
    fn drop(&mut self) {
        SCAN_HEAPS.lock().unwrap_or_else(|e| e.into_inner()).retain(|h| h.0 != self.cpu);
        gate::forget(self.cpu);
        super::slotwatch::forget(self.cpu);
        if let Ok(mut kept) = self.kept.lock() {
            for res in kept.drain(..).flatten() {
                d3d12x::host::retire(res);
            }
        }
    }
}

/// View heaps for [`warp_scan`]: (CPU start, slots, bytes per slot, title slots).
static SCAN_HEAPS: Mutex<Vec<(usize, u32, u32, u32)>> = Mutex::new(Vec::new());

/// `XNW_WARP_SCAN=1` (report) or `fix`, on the software adapter only: WARP's
/// shader-visible descriptors are CPU memory, 64 bytes each (`xnw-shader` test `null_kinds` with `XNW_NK_BYTES=1`): the top nibble of
/// the last qword is the kind (0 never written, 1 UAV, 2 SRV, 3 CBV) and the
/// third qword the pointer a shader-resource read goes through (null for a
/// slot never written and for a null UAV; a null SRV points it at zeros). A
/// shader reading such a slot as a resource reads address 0 + offset and
/// WARP removes the device (the title's `reading address 0x388`). The scan
/// counts those slots per heap; `fix` makes them null SRVs (which read zeros
/// and drop writes under every kind).
fn warp_scan_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("XNW_WARP_SCAN").as_deref().map(str::trim) {
        Ok("1") if xnw_device::warp() => 1,
        Ok("fix") if xnw_device::warp() => 2,
        _ => 0,
    })
}

/// Slots made null SRVs by `XNW_WARP_SCAN=fix`.
static SCAN_FIXED: AtomicU64 = AtomicU64::new(0);

/// See [`warp_scan_mode`]: one line per view heap (`why` names the moment).
pub(crate) fn warp_scan(why: &str) {
    let mode = warp_scan_mode();
    if mode == 0 {
        return;
    }
    let heaps = SCAN_HEAPS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for (cpu, slots, step, title) in heaps {
        if step < 64 {
            continue;
        }
        let started = std::time::Instant::now();
        // never written, null UAV, SRV without a read pointer, CBV; title or dynamic.
        let mut n = [[0u32; 2]; 4];
        let mut first: Vec<String> = Vec::new();
        let mut fixed = 0u64;
        for i in 0..slots {
            let at = cpu + i as usize * step as usize;
            // SAFETY: a slot of a live heap (WARP: CPU memory); racing writers
            // only make the report approximate.
            let (q0, q2, q7) = unsafe { (std::ptr::read_volatile(at as *const u64), std::ptr::read_volatile((at + 16) as *const u64), std::ptr::read_volatile((at + 56) as *const u64)) };
            let kind = (q7 >> 60) as usize;
            let class = match kind {
                0 if q2 == 0 => 0,
                1 if q2 == 0 => 1,
                2 if q2 == 0 => 2,
                3 => 3,
                _ => continue,
            };
            n[class][usize::from(i >= title)] += 1;
            if class < 3 && first.len() < 12 {
                first.push(format!("{i}:{}{}", ["unwritten", "null-uav", "srv-no-ptr"][class], if q0 != 0 { "(live)" } else { "" }));
            }
            if mode == 2 && class < 2 && q0 == 0 {
                null_view(D3D12_CPU_DESCRIPTOR_HANDLE { ptr: at });
                fixed += 1;
            }
        }
        SCAN_FIXED.fetch_add(fixed, Ordering::Relaxed);
        xnw_core::info!(
            "warp scan ({why}): heap of {slots} slots ({title} title): never written {}+{} (title+dynamic), null UAV {}+{}, SRV without a read pointer {}+{}, CBV {}+{}; first {}; made null SRV {fixed} (total {}); {} ms",
            n[0][0], n[0][1], n[1][0], n[1][1], n[2][0], n[2][1], n[3][0], n[3][1],
            if first.is_empty() { "-".to_string() } else { first.join(" ") },
            SCAN_FIXED.load(Ordering::Relaxed),
            started.elapsed().as_millis()
        );
    }
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
#[allow(dead_code)] // the DRED report (base host.rs 718), dropped with DRED
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!(
                "{:#x} past its end (slot {})",
                va - start - bytes,
                (va - start) / step
            )
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
const DYNAMIC: u32 = 1 << 18;

/// The dynamic slots of a heap whose host would not take [`DYNAMIC`] more
/// (a capped heap, NVIDIA): the title's slots keep their room first.
const MIN_DYNAMIC: u32 = 1 << 16;

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32, title_start: u64) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        // The larger dynamic range only where the host takes it on top of
        // every title slot; else the smaller one, and the cap after that
        // (VIctory main 3c81420 + e8d3b64, GPL-3.0-or-later).
        let (heap, count, extra) = if sampler {
            match make(count) {
                Ok(heap) => (heap, count, 0),
                Err(e) => {
                    xnw_core::warn!("d3d12x: no host descriptor heap (count {count}, sampler {sampler}): {e}");
                    return None;
                }
            }
        } else if let Ok(heap) = make(count + DYNAMIC) {
            (heap, count, DYNAMIC)
        } else {
            match make(count + MIN_DYNAMIC) {
                Ok(heap) => (heap, count, MIN_DYNAMIC),
                Err(e) if count + MIN_DYNAMIC > PORTABLE_VIEWS => {
                    // shortcut: descriptors past the host's limit are left
                    // out; a title that reaches them needs remapping.
                    xnw_core::warn!("d3d12x: host view heap of {count} capped at {PORTABLE_VIEWS}: {e}");
                    (make(PORTABLE_VIEWS).ok()?, PORTABLE_VIEWS - MIN_DYNAMIC, MIN_DYNAMIC)
                }
                Err(e) => {
                    xnw_core::warn!("d3d12x: no host descriptor heap (count {count}, sampler {sampler}): {e}");
                    return None;
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(count + extra) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let h = HostHeap {
            heap,
            cpu,
            step,
            count,
            sampler,
            made: Mutex::new(Made { keys: vec![0; count as usize], ..Made::default() }),
            kept: Mutex::new(vec![None; (count + extra) as usize]),
            dynamic: extra,
            next: Mutex::new((0, vec![false; extra as usize])),
            bindless: Mutex::new(Bindless {
                title_start: if sampler { 0 } else { title_start },
                seen: if sampler {
                    Vec::new()
                } else {
                    vec![0; count as usize]
                },
                // Remappings before the heap concern none of its slots.
                remap_seen: REMAPPED.lock().unwrap_or_else(|e| e.into_inner()).0,
                ..Bindless::default()
            }),
            recording: std::sync::atomic::AtomicBool::new(false),
            record_hint: std::sync::atomic::AtomicBool::new(false),
            record_looked: [AtomicU64::new(0), AtomicU64::new(0)],
            recheck_at: AtomicU64::new(u64::MAX),
            record_ready: AtomicU64::new(0),
            record_waits: std::sync::atomic::AtomicBool::new(false),
            strides: if sampler { None } else { StrideTable::new(&host.device, count + extra) },
            words: std::sync::OnceLock::new(),
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, count + extra) {
            xnw_core::warn!("d3d12x: host descriptor heap not made null: {e}");
        }
        h.make_guards();
        // From here on the heap may be read by submissions: writes into it
        // while one runs are staged (`gate`).
        gate::register(&h.heap, kind, count + extra, step);
        if !sampler {
            SCAN_HEAPS.lock().unwrap_or_else(|e| e.into_inner()).push((h.cpu, count + extra, h.step, count));
            super::slotwatch::register(h.cpu, gpu, count + extra, h.step, count, title_start);
        }
        xnw_core::info!(
            "d3d12x: host {} heap: {count} title slots + {extra} dynamic, made null in {} ms",
            if sampler { "sampler" } else { "view" },
            started.elapsed().as_millis()
        );
        // `XNW_LOOP_COUNTS=1`: a dynamic slot kept for the loop
        // counts' buffer, never handed out.
        if let Some(slot) = h.loop_slot() {
            h.next.lock().unwrap_or_else(|e| e.into_inner()).1[(slot - h.count) as usize] = true;
            d3d12x::loopcount::describe(&host.device, h.cpu(slot));
        }
        Some(h)
    }

    /// The slot of the loop counts' UAV (the dynamic slot before the
    /// sink), with `XNW_LOOP_COUNTS=1`.
    pub(crate) fn loop_slot(&self) -> Option<u32> {
        (d3d12x::loopcount::on() && !self.sampler && self.dynamic >= 2).then(|| self.count + self.dynamic - 2)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: BLOCK,
                        ..Default::default()
                    })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device
                    .CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device
                    .CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { gate::sampler(&host.device, &desc, at) };
}

impl HostHeap {
    /// The microseconds of slot making (`which`: 0 plans, 1 recording) left.
    fn budget_left(&self, which: usize) -> u64 {
        self.bindless
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .budget()[which]
    }

    /// Notes `us` microseconds of slot making spent.
    fn budget_spend(&self, which: usize, us: u64) {
        let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
        let left = &mut b.budget()[which];
        *left = left.saturating_sub(us);
    }

    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    /// The view heap's guard slots, by kind
    /// (see [`GUARDS`]); none for sampler heaps.
    pub(crate) fn guards(&self) -> Option<[u32; GUARDS as usize]> {
        if self.sampler { None } else { guard_slots(self.count, self.dynamic) }
    }

    /// All slots of the heap (title slots, then dynamic ones).
    pub(crate) fn slots(&self) -> u32 {
        self.count + self.dynamic
    }

    /// Writes the guard slots: permanent nulls of each kind.
    fn make_guards(&self) {
        let Some(g) = self.guards() else { return };
        null_view(self.cpu(g[0]));
        null_buffer_view(self.cpu(g[1]));
        null_uav_buffer_view(self.cpu(g[2]), 0);
        null_uav_view(self.cpu(g[3]));
        if let Some(host) = host() {
            // SAFETY: a slot of our heap; a zero location makes a null CBV.
            unsafe { gate::cbv(&host.device, Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC::default()), self.cpu(g[4])) };
        }
    }

    /// Slots `index` and `index + 1` a raw null SRV and a raw null UAV
    /// (a released window's page-table pair; never handed out again).
    pub(crate) fn raw_null_pair(&self, index: u32) {
        null_buffer_view(self.cpu(index));
        null_uav_buffer_view(self.cpu(index + 1), 0);
        for i in [index, index + 1] {
            self.keep(i, None);
            self.set_stride(i, 0);
            self.write_words(i, [0; 8]);
        }
    }

    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        // The last dynamic slot is the sink for slots past the heap (`cpu`).
        // And below it the guard slots (`guard_slots`), never handed out.
        let cap = (self.dynamic as usize).saturating_sub(1 + GUARDS as usize);
        if cap == 0 {
            return None;
        }
        let n = n as usize;
        // Search from `next` on, then from the start.
        let starts = (*next as usize..cap).chain(0..*next as usize);
        let mut run = 0;
        let mut found = None;
        let mut last = usize::MAX;
        for at in starts {
            // A run cannot wrap past the end.
            if at != last.wrapping_add(1) {
                run = 0;
            }
            last = at;
            run = if taken[at] { 0 } else { run + 1 };
            if run == n {
                found = Some(at + 1 - n);
                break;
            }
        }
        let Some(first) = found else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| xnw_core::warn!("d3d12x: no {n} free dynamic descriptor slots of {cap}: a table is not made"));
            DYNAMIC_COUNTS[2].fetch_add(1, Ordering::Relaxed);
            return None;
        };
        taken[first..first + n].fill(true);
        DYNAMIC_COUNTS[0].fetch_add(n as u64, Ordering::Relaxed);
        let now = DYNAMIC_IN_USE.fetch_add(n as u64, Ordering::Relaxed) + n as u64;
        DYNAMIC_COUNTS[1].fetch_max(now, Ordering::Relaxed);
        *next = ((first + n) % cap) as u32;
        Some(self.count + first as u32)
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = (at + n).min(self.dynamic) as usize;
        if let Some(slots) = guard.1.get_mut(at as usize..end) {
            let given = slots.iter().filter(|&&t| t).count() as u64;
            slots.fill(false);
            let _ = DYNAMIC_IN_USE.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| Some(v.saturating_sub(given)));
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let kept = make(view, self.cpu(index));
        if kept.is_none() {
            null_for(view, self.cpu(index));
        }
        self.keep(index, kept.flatten());
        self.set_stride(index, view_stride(view));
        self.write_words(index, console_words(view));
    }

    /// The host GPU address of the heap's descriptor words (view heaps),
    /// made on first ask from the slots made so far (VIctory).
    pub(crate) fn words(&self) -> Option<u64> {
        let w = self.words.get_or_init(|| {
            if self.sampler {
                return None;
            }
            let bytes = u64::from(self.count + self.dynamic) * 32;
            let (buffer, data) = host()?.shared_buffer(bytes)?;
            d3d12x::host::name(&buffer, || format!("descriptor words {}+{}", self.count, self.dynamic));
            // SAFETY: a fresh mapping of `bytes`; a live buffer.
            let gpu = unsafe {
                std::ptr::write_bytes(data, 0, bytes as usize);
                buffer.GetGPUVirtualAddress()
            };
            // The title slots made before: their records' words.
            let made: Vec<(u32, u64)> = self.made.lock().unwrap_or_else(|e| e.into_inner()).iter().enumerate().filter(|&(_, &k)| k != 0).map(|(i, &k)| (i as u32, k)).collect();
            for (index, key) in made {
                if let Some(view) = view_at(key as u32, (key >> 32) as u32) {
                    // SAFETY: slot `index` of the fresh mapping.
                    unsafe { std::ptr::copy_nonoverlapping(console_words(&view).as_ptr(), (data as *mut u32).add(index as usize * 8), 8) };
                }
            }
            Some((buffer, data as usize, gpu))
        });
        w.as_ref().map(|w| w.2)
    }

    /// Slot `index`'s console words, once the words exist.
    fn write_words(&self, index: u32, words: [u32; 8]) {
        let Some(Some((_, data, _))) = self.words.get() else {
            return;
        };
        if index >= self.count + self.dynamic {
            return;
        }
        // SAFETY: slot `index` of the mapping, 32 bytes each.
        unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), (*data as *mut u32).add(index as usize * 8), 8) };
    }

    /// Slot `index` a raw SRV and `index + 1` a raw
    /// UAV over `bytes` of `buffer` from byte `first` on (both multiples of
    /// 4), each holding the buffer while it does.
    pub(crate) fn raw_pair_at(&self, index: u32, buffer: &ID3D12Resource, first: u64, bytes: u64) {
        let Some(host) = host() else { return };
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Buffer: D3D12_BUFFER_SRV { FirstElement: first / 4, NumElements: (bytes / 4) as u32, StructureByteStride: 0, Flags: D3D12_BUFFER_SRV_FLAG_RAW } },
        };
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: D3D12_BUFFER_UAV { FirstElement: first / 4, NumElements: (bytes / 4) as u32, Flags: D3D12_BUFFER_UAV_FLAG_RAW, ..Default::default() } },
        };
        // SAFETY: two slots of our heap and a live buffer.
        unsafe {
            gate::srv(&host.device, buffer, Some(&srv), self.cpu(index));
            gate::uav(&host.device, buffer, None, Some(&uav), self.cpu(index + 1));
        }
        for i in [index, index + 1] {
            self.keep(i, Some(buffer.clone()));
            self.set_stride(i, 0);
            self.write_words(i, [0; 8]);
        }
    }

    /// Copies `n` views from `src` (a CPU-only heap of ours) into slots
    /// `first..` (dynamic ones), which then read nothing as words (the
    /// address map's raw views, made once per map, copied per list).
    pub(crate) fn copy_views_in(&self, first: u32, n: u32, src: D3D12_CPU_DESCRIPTOR_HANDLE) {
        let Some(host) = host() else { return };
        if n == 0 || first + n > self.count + self.dynamic - 1 {
            return;
        }
        // SAFETY: `n` slots of our heap from `first` and `n` views at `src`.
        unsafe { gate::copy(&host.device, n, self.cpu(first), src, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };
        for i in first..first + n {
            self.keep(i, None);
            self.write_words(i, [0; 8]);
        }
    }

    /// Slot `index` a raw view of `bytes` of `buffer` from `offset`
    /// (16-byte aligned), for a root SRV bound as a one-view table; the
    /// buffer held while the slot is. Ported from victory-fork @ 4e17478
    /// (`descriptor/host.rs` `raw_at`), GPL-3.0-or-later.
    pub(crate) fn raw_at(&self, index: u32, buffer: &ID3D12Resource, offset: u64, bytes: u64) {
        let Some(host) = host() else { return };
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV { FirstElement: offset / 4, NumElements: (bytes / 4).min(u64::from(u32::MAX)) as u32, StructureByteStride: 0, Flags: D3D12_BUFFER_SRV_FLAG_RAW },
            },
        };
        // SAFETY: a slot of our heap; a live buffer covering the range.
        unsafe { gate::srv(&host.device, buffer, Some(&desc), self.cpu(index)) };
        self.keep(index, Some(buffer.clone()));
        self.set_stride(index, 0);
    }

    /// Slot `index` a null raw buffer view (reads zeros).
    pub(crate) fn null_raw_at(&self, index: u32) {
        self.make_null_kind(index, xnw_shader::ResourceType::Srv, 4);
    }

    /// The shader-visible handle of slot `index`.
    pub(crate) fn gpu_handle(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a live heap.
        let start = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        D3D12_GPU_DESCRIPTOR_HANDLE { ptr: start.ptr + u64::from(index) * u64::from(self.step) }
    }
    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
        self.set_stride(index, 0);
        self.write_words(index, [0; 8]);
    }

    /// Makes slot `index` a null view matching the resource kind a shader
    /// expects there (CBV, buffer SRV/UAV, or 2D texture).
    pub(crate) fn make_null_kind(&self, index: u32, kind: xnw_shader::ResourceType, dwords: u8) {
        if self.sampler {
            self.null_sampler(index);
            self.keep(index, None);
            return;
        }
        let at = self.cpu(index);
        match kind {
            xnw_shader::ResourceType::Cbv => {
                if let Some(host) = host() {
                    let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
                    // SAFETY: a slot of our heap; zero location makes a null CBV.
                    unsafe { gate::cbv(&host.device, Some(&desc), at) };
                }
            }
            xnw_shader::ResourceType::Srv if dwords == 4 => null_buffer_view(at),
            xnw_shader::ResourceType::Uav if null_srv() => {
                NULL_UAV_AS_SRV.fetch_add(1, Ordering::Relaxed);
                if dwords == 4 { null_buffer_view(at) } else { null_view(at) }
            }
            xnw_shader::ResourceType::Uav => {
                if let Some(host) = host() {
                    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                        Format: DXGI_FORMAT_R32_UINT,
                        ViewDimension: if dwords == 4 {
                            D3D12_UAV_DIMENSION_BUFFER
                        } else {
                            D3D12_UAV_DIMENSION_TEXTURE2D
                        },
                        ..Default::default()
                    };
                    // SAFETY: a slot of our heap; null resource makes a null UAV.
                    unsafe {
                        gate::uav(&host.device,
                            None::<&ID3D12Resource>,
                            None::<&ID3D12Resource>,
                            Some(&desc),
                            at,
                        )
                    };
                }
            }
            _ => null_view(at),
        }
        self.keep(index, None);
        self.set_stride(index, 0);
        self.write_words(index, [0; 8]);
    }

    /// Makes slot `index` a null unordered access view (raw buffer or 2D texture)
    /// so that shader UAV writes are safely dropped by hardware rather than faulting.
    pub(crate) fn make_null_uav(&self, index: u32, buffer: bool, stride: u32) {
        if buffer {
            null_uav_buffer_view(self.cpu(index), stride);
        } else {
            null_uav_view(self.cpu(index));
        }
        self.keep(index, None);
        self.set_stride(index, stride);
        self.write_words(index, [0; 8]);
    }

    /// Slot `index` now views `resource` (or nothing of the host's). A
    /// previously viewed resource is retired rather than dropped inline so an
    /// in-flight GPU submission reading that slot cannot fault on freed memory.
    fn keep(&self, index: u32, resource: Option<ID3D12Resource>) {
        let old = {
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            kept.get_mut(index as usize)
                .and_then(|slot| std::mem::replace(slot, resource))
        };
        if let Some(old) = old {
            d3d12x::host::retire(old);
        }
    }

    /// `XNW_GPU_CHECK=1`, before each submission: the first slot made
    /// from a view whose title resource is freed, or made from a released
    /// view and not made since; with the number of such slots.
    /// `null`: the stale slots found are made null.
    pub(crate) fn check_live(&self, null: bool) -> Option<(String, usize)> {
        let made = self.made.lock().unwrap_or_else(|e| e.into_inner()).keys.clone();
        let mut stale = Vec::new();
        let mut first = None;
        let mut n = 0;
        for (index, &key) in made.iter().enumerate().take(self.count as usize) {
            if key == 0 || self.sampler {
                continue;
            }
            let (vi, generation) = (key as u32, (key >> 32) as u32);
            let why = match view_at(vi, generation) {
                None => Some(format!("view {vi} generation {generation} released")),
                Some(View::Shader(r, _) | View::Unordered(r, _, _)) if r != 0 && !d3d12x::resource::is_live(r) => {
                    Some(format!("view {vi} names freed resource {r:#x}"))
                }
                _ => None,
            };
            if let Some(why) = why {
                n += 1;
                first.get_or_insert_with(|| format!("slot {index}: {why}; {}", self.slot_text(index as u32, vi)));
                if null {
                    stale.push(index as u32);
                }
            }
        }
        for index in stale {
            self.clear_stale_slot(index);
            d3d12x::resource::HELD_COUNTS[d3d12x::resource::HC_CHECK_NULLED].fetch_add(1, Ordering::Relaxed);
        }
        first.map(|f| (f, n))
    }

    /// `XNW_GPU_CHECK`: what slot `index`, made from view entry `vi`,
    /// is: the path that made it, the entry's kind and resource now, and
    /// the host resource the slot holds (its COM count, whether the target
    /// free pool has it).
    fn slot_text(&self, index: u32, vi: u32) -> String {
        use windows::core::Interface;
        let path = MADE_BY.lock().unwrap_or_else(|e| e.into_inner()).get(&(std::ptr::from_ref(self) as usize, index)).copied().unwrap_or("unknown");
        let (owner, kind) = match super::entry(vi) {
            Some((view, g, kind)) => {
                let r = match view {
                    View::Shader(r, _) | View::Unordered(r, _, _) => format!("resource {r:#x} ({})", if d3d12x::resource::is_live(r) { "alive" } else { "freed" }),
                    View::Placed(..) => "placed view".into(),
                    View::Constant(..) => "constant view".into(),
                    _ => "sampler".into(),
                };
                (format!("{r}, entry generation now {g}"), kind_name(kind))
            }
            None => ("no entry".into(), "?"),
        };
        let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner()).get(index as usize).cloned().flatten();
        let host = match kept {
            Some(k) => format!("host resource {:p} (count {}, {})", k.as_raw(), d3d12x::resource::com_count(&k) - 1, if d3d12x::resource::in_free_pool(&k) { "in the target free pool" } else { "not pooled" }),
            None => "no host resource held".into(),
        };
        format!("made by {path}, kind {kind}, {owner}, {host}")
    }

    /// Slots made from view entries released since this heap last
    /// looked are made null (of their kind), before the submission that
    /// would carry them: a released view's host texture may go back to the
    /// target free pool and be handed to another target. Returns how many.
    pub(crate) fn null_released(&self) -> usize {
        if self.sampler || self.count == 0 {
            return 0;
        }
        let at = std::time::Instant::now();
        let me = std::ptr::from_ref(self) as usize;
        let mut swept = SWEPT.lock().unwrap_or_else(|e| e.into_inner());
        let from = swept.get(&me).copied().unwrap_or(0);
        let Some((released, upto)) = super::released_since(from) else {
            return 0;
        };
        swept.insert(me, upto);
        drop(swept);
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let mut n = 0;
        match released {
            // Proportional to the releases: the slots linked to each entry.
            Some(released) => {
                for (vi, generation, kind) in released {
                    let key = u64::from(generation) << 32 | u64::from(vi);
                    for index in made.by_view.remove(&vi).unwrap_or_default() {
                        if made.keys.get(index as usize) == Some(&key) {
                            made.keys[index as usize] = 0;
                            self.null_of_kind(index, kind);
                            n += 1;
                        }
                    }
                }
            }
            // The release log was cut past this heap's mark: all slots.
            None => {
                let generations = super::generations();
                for index in 0..made.keys.len() {
                    let key = made.keys[index];
                    let (vi, generation) = (key as u32, (key >> 32) as u32);
                    if key != 0 && generations.get(vi as usize).is_some_and(|&(now, _)| now != generation) {
                        made.keys[index] = 0;
                        self.null_of_kind(index as u32, generations[vi as usize].1);
                        n += 1;
                    }
                }
            }
        }
        drop(made);
        d3d12x::resource::HELD_COUNTS[d3d12x::resource::HC_SWEEP_US].fetch_add(at.elapsed().as_micros() as u64, Ordering::Relaxed);
        d3d12x::resource::HELD_COUNTS[d3d12x::resource::HC_SLOTS_NULLED].fetch_add(n as u64, Ordering::Relaxed);
        n
    }

    /// Slots made in place over a shared window released since this
    /// heap last looked are made null and made again by the plan (which
    /// starts whole: older parts read and settle their ranges). Returns how
    /// many. Window releases are rare (a heap over a region the title frees).
    pub(crate) fn null_gone_windows(&self) -> usize {
        if self.sampler || self.count == 0 {
            return 0;
        }
        let me = std::ptr::from_ref(self) as usize;
        let ranges: Vec<(u64, u64)> = {
            let w = WINDOWS_GONE.lock().unwrap_or_else(|e| e.into_inner());
            let mut seen = WINDOWS_SEEN.lock().unwrap_or_else(|e| e.into_inner());
            let last = seen.insert(me, w.0).unwrap_or(0);
            if last == w.0 {
                return 0;
            }
            w.1.iter().filter(|&&(n, ..)| n > last).map(|&(_, a, e)| (a, e)).collect()
        };
        let gone: Vec<u32> = {
            let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
            let b = &mut *b;
            let gone: Vec<u32> = b.in_place.iter().filter(|&(_, &gpu)| ranges.iter().any(|&(a, e)| a <= gpu && gpu < e)).map(|(&i, _)| i).collect();
            for i in &gone {
                b.in_place.remove(i);
                b.uses.remove(i);
                b.dirty.insert(*i);
            }
            if !gone.is_empty() {
                b.plan = None;
            }
            gone
        };
        for &i in &gone {
            self.clear_stale_slot(i);
        }
        d3d12x::resource::HELD_COUNTS[d3d12x::resource::HC_WINDOW_NULLED].fetch_add(gone.len() as u64, Ordering::Relaxed);
        gone.len()
    }

    /// `XNW_GPU_CHECK` only: notes the path that made slot `index`.
    fn made_by(&self, index: u32, path: &'static str) {
        if d3d12x::host::gpu_check_on() {
            MADE_BY.lock().unwrap_or_else(|e| e.into_inner()).insert((std::ptr::from_ref(self) as usize, index), path);
        }
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory (the list's own, which it holds).
    pub(crate) fn constant_at(&self, index: u32, gpu: u64, size: u32) {
        self.keep(index, None);
        self.set_stride(index, 0);
        self.write_words(index, [0; 8]);
        let Some(host) = host() else { return };
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: size.min(65_536).next_multiple_of(256),
        };
        // SAFETY: a slot of our heap; the memory is the caller's.
        unsafe {
            gate::cbv(&host.device,
                Some(&desc), self.cpu(index))
        };
    }

    /// Writes slot `index`'s runtime stride table entry: `stride` bytes per
    /// element (0: no buffer view), the slot itself as the raw view.
    fn set_stride(&self, index: u32, stride: u32) {
        if let Some(t) = &self.strides
            && index < t.len
        {
            let entry = StrideEntry { stride, ..StrideEntry::default() };
            // SAFETY: `t.ptr` is a live mapped upload buffer of `t.len`
            // entries of two dwords.
            unsafe { (t.ptr as *mut [u32; 2]).add(index as usize).write(entry.words()) };
        }
    }

    /// The runtime stride table (view heaps): its buffer and GPU address.
    pub(crate) fn stride_table(&self) -> Option<(&ID3D12Resource, u64)> {
        self.strides.as_ref().map(|t| (&t.resource, t.gpu))
    }

    /// The CPU handle of slot `index`. A slot past the heap (title slots
    /// past the host's, dynamic slots past the dynamic range) is never
    /// written: it is counted and the heap's last slot, a sink nothing
    /// reads (never handed out by [`HostHeap::dynamic`]), is written
    /// instead (a test run crashed in the driver's CreateUnorderedAccessView
    /// on a handle it did not know, called from `sync_title_slots`).
    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slots = self.count + self.dynamic;
        // View heaps: the last dynamic slot is the sink. Sampler heaps have
        // no spare slot: their last slot takes it (never seen: their
        // indices are checked against the count).
        let usable = if self.dynamic > 0 { slots - 1 } else { slots };
        let index = if index < usable || slots == 0 {
            index
        } else {
            debug_assert!(cfg!(test), "descriptor slot {index} past a host heap of {slots}");
            SLOTS_OUT.fetch_add(1, Ordering::Relaxed);
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| xnw_core::warn!("d3d12x: descriptor slot {index} past a host heap of {slots} slots: not written (counted)"));
            slots - 1
        };
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, Ordering::Relaxed) < index {
                xnw_core::warn!(
                    "d3d12x: title slot {index} past the host heap of {} (for a draw: {for_draw})",
                    self.count
                );
            }
            return None;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let (i, generation, tag) = match unsafe { super::record_id(handle) } { Some((i, g)) => (i, g, VIEW_TAG), None => (0, 0, 0) };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            {
                let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
                if made[index as usize] != 0 {
                    made[index as usize] = 0;
                    self.make_null(index);
                }
            }
            if !self.sampler {
                self.untrack_bindless(index, 0);
                self.write_words(index, [0; 8]);
            }
            return None;
        };
        // Even buffer views made only for draws have their words now
        // (VIctory).
        if !self.sampler {
            self.write_words(index, console_words(&view));
        }
        let is_buf = !self.sampler && view.is_buffer();
        if is_buf && for_draw && self.in_place_changed(index, view_in_place(&view).as_ref().map(shared_va)) {
            self.unmake(index);
        }
        {
            let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            if made[index as usize] != key {
                let later = !for_draw && is_buf;
                made[index as usize] = if later { 0 } else { key };
                made.link(index);
                self.made_by(index, "title descriptor write (mirror)");
                let kept = if later {
                    None
                } else {
                    make(&view, self.cpu(index))
                };
                if kept.is_none() && !self.sampler {
                    null_for(&view, self.cpu(index));
                }
                self.keep(index, kept.flatten());
                self.set_stride(index, view_stride(&view));
            }
        }
        if !self.sampler {
            if is_buf {
                let placed = matches!(&view, View::Placed(false, r, _) if r[0] != 1);
                self.track_bindless(index, (i, generation), placed);
            } else {
                self.untrack_bindless(index, key);
            }
        }
        Some(view)
    }

    /// Detaches the title's descriptor memory when its `Heap` is dropped so
    /// later bindless scans never read freed heap memory.
    pub(crate) fn detach_title(&self) {
        let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
        b.title_start = 0;
    }

    fn track_bindless(&self, index: u32, entry: (u32, u32), placed: bool) {
        let key = u64::from(entry.1) << 32 | u64::from(entry.0);
        let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(seen) = b.seen.get_mut(index as usize) {
            *seen = key;
        }
        if b.slots.insert(index, entry) != Some(entry) {
            b.dirty.insert(index);
            if let Some(bounce) = b.cbv_bounce.get_mut(&index) {
                bounce.active = false;
            }
        }
        let changed = if placed {
            b.placed_set.insert(index, entry) != Some(entry)
        } else {
            b.placed_set.remove(&index).is_some()
        };
        if changed {
            b.record_pending.insert(index);
            self.record_hint.store(true, Ordering::Release);
        }
    }

    fn untrack_bindless(&self, index: u32, seen_key: u64) {
        let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(seen) = b.seen.get_mut(index as usize) {
            *seen = seen_key;
        }
        if b.slots.remove(&index).is_some() {
            b.dirty.insert(index);
            if let Some(bounce) = b.cbv_bounce.get_mut(&index) {
                bounce.active = false;
            }
        }
        if b.placed_set.remove(&index).is_some() {
            b.recorded.remove(&index);
        }
    }

    /// Scans title descriptor memory for slots written or overwritten as raw
    /// bytes (bypassing `mirror()`), using write-watch when the heap's memory
    /// supports it and scanning all slots otherwise.
    fn sync_title_slots(&self) {
        if self.sampler || self.count == 0 {
            return;
        }
        let start = {
            let at = std::time::Instant::now();
            let b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
            count_us(US_PLAN_LOCKS, at);
            b.title_start
        };
        if start == 0 {
            return;
        }
        const TITLE_SLOT_BYTES: usize = 32;
        let started = std::time::Instant::now();
        let len = self.count as usize * TITLE_SLOT_BYTES;
        let end = start as usize + len;
        // Guarded where the heap is view memory, as the title's
        // (a placeholder reservation with pages mapped in, never
        // write-watched): asked without, the answer was none and every plan
        // scanned all 1,245,184 slots (22.8 million a frame, 18 whole scans).
        // Writes of ours (view creation,
        // descriptor copies) and the title's own stores open a block once
        // per plan; only open blocks are scanned.
        let watch = d3d12x::watch::writes_exact(start as usize, end);
        let (mut examined, mut changed) = (0u64, 0u64);

        let mut cleared = Vec::new();
        let mut direct = Vec::new();
        {
            let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
            if b.title_start != start {
                return;
            }
            let b = &mut *b;
            let mut check_range = |first: u32, last: u32| {
                // `XNW_PLAN_SKIP=ring`: the title's per-draw ring
                // is not scanned (its tables are made where draws bind them).
                let last = if plan_skip_ring() { last.min(RING_START) } else { last };
                if first >= last {
                    return;
                }
                examined += u64::from(last - first);
                area_count(&AREA_EXAMINED, first, last);
                for index in first..last {
                    let handle = start + u64::from(index) * TITLE_SLOT_BYTES as u64;
                    // SAFETY: `index < self.count` lies in the live title heap
                    // (`detach_title` clears `b.title_start` under this lock).
                    let (vi, generation, tag) = match unsafe { super::record_id(handle) } { Some((i, g)) => (i, g, VIEW_TAG), None => (0, 0, 0) };
                    let key = if tag == VIEW_TAG && generation != 0 {
                        u64::from(generation) << 32 | u64::from(vi)
                    } else {
                        0
                    };
                    let Some(seen) = b.seen.get_mut(index as usize) else {
                        continue;
                    };
                    if *seen == key {
                        continue;
                    }
                    changed += 1;
                    AREA_CHANGED[area(index)].fetch_add(1, Ordering::Relaxed);
                    CHANGED_SIXTEENTHS[(u64::from(index) * 16 / u64::from(self.count.max(1))) as usize % 16].fetch_add(1, Ordering::Relaxed);
                    *seen = key;
                    if key == 0 {
                        if b.slots.remove(&index).is_some() {
                            b.dirty.insert(index);
                            if let Some(bounce) = b.cbv_bounce.get_mut(&index) {
                                bounce.active = false;
                            }
                        }
                        if b.placed_set.remove(&index).is_some() {
                            b.recorded.remove(&index);
                        }
                        cleared.push(index);
                        continue;
                    }
                    let Some(view) = view_at(vi, generation) else {
                        if b.slots.remove(&index).is_some() {
                            b.dirty.insert(index);
                            if let Some(bounce) = b.cbv_bounce.get_mut(&index) {
                                bounce.active = false;
                            }
                        }
                        if b.placed_set.remove(&index).is_some() {
                            b.recorded.remove(&index);
                        }
                        cleared.push(index);
                        continue;
                    };
                    if view.is_buffer() {
                        let entry = (vi, generation);
                        if b.slots.insert(index, entry) != Some(entry) {
                            b.dirty.insert(index);
                            if let Some(bounce) = b.cbv_bounce.get_mut(&index) {
                                bounce.active = false;
                            }
                        }
                        let placed = matches!(&view, View::Placed(false, r, _) if r[0] != 1);
                        let changed = if placed {
                            b.placed_set.insert(index, entry) != Some(entry)
                        } else {
                            b.placed_set.remove(&index).is_some()
                        };
                        if changed {
                            b.record_pending.insert(index);
                            self.record_hint.store(true, Ordering::Release);
                        }
                    } else {
                        if b.slots.remove(&index).is_some() {
                            b.dirty.insert(index);
                            if let Some(bounce) = b.cbv_bounce.get_mut(&index) {
                                bounce.active = false;
                            }
                        }
                        if b.placed_set.remove(&index).is_some() {
                            b.recorded.remove(&index);
                        }
                        direct.push((index, key, view));
                    }
                }
            };

            match watch {
                Some((now, pages)) if b.scan_asked != 0 => {
                    let base = start as usize;
                    for (page, written_at) in pages {
                        if written_at <= b.scan_asked {
                            continue;
                        }
                        let page_start = page * d3d12x::watch::PAGE;
                        let page_end = page_start + d3d12x::watch::PAGE;
                        let lo = page_start.saturating_sub(base) / TITLE_SLOT_BYTES;
                        let hi = page_end
                            .saturating_sub(base)
                            .div_ceil(TITLE_SLOT_BYTES)
                            .min(self.count as usize);
                        if lo < hi {
                            count(BLOCKS_SCANNED, 1);
                            check_range(lo as u32, hi as u32);
                        }
                    }
                    b.scan_asked = b.scan_asked.max(now);
                }
                Some((now, _)) => {
                    check_range(0, self.count);
                    b.scan_asked = b.scan_asked.max(now);
                    count(WHOLE_SCANS, 1);
                }
                None => {
                    check_range(0, self.count);
                    count(WHOLE_SCANS, 1);
                }
            }
        }
        count(SLOTS_EXAMINED, examined);
        count(SLOTS_CHANGED, changed);
        for index in cleared {
            self.clear_stale_slot(index);
        }
        for (index, key, view) in direct {
            let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            if made[index as usize] != key {
                let kept = make(&view, self.cpu(index));
                if kept.is_none() {
                    null_for(&view, self.cpu(index));
                    made[index as usize] = 0;
                } else {
                    made[index as usize] = key;
                    made.link(index);
                    self.made_by(index, "title slot scan (sync)");
                }
                self.keep(index, kept.flatten());
            }
        }
        count_us(US_SCAN, started);
    }

    /// Slot `index` (viewing `entry`) is no longer one for bindless access.
    fn drop_bindless(&self, b: &mut Bindless, index: u32, entry: Option<(u32, u32)>) {
        b.uses.remove(&index);
        b.recorded.remove(&index);
        if entry.is_none_or(|e| b.slots.get(&index) == Some(&e)) {
            b.slots.remove(&index);
            if let Some(bounce) = b.cbv_bounce.get_mut(&index) {
                bounce.active = false;
            }
            b.placed_set.remove(&index);
        }
    }

    fn clear_stale_slot(&self, index: u32) {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if made[index as usize] != 0 {
            made[index as usize] = 0;
            self.make_null(index);
        }
    }

    fn set_unmapped_null(&self, index: u32, view: &View) {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        made[index as usize] = 0;
        if !self.sampler {
            null_for(view, self.cpu(index));
        }
        self.keep(index, None);
    }

    fn ensure_placed_slot(
        &self,
        index: u32,
        key: u64,
        shader_view: &View,
        raw_tex: *mut std::ffi::c_void,
    ) {
        use windows::core::Interface;
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let same_tex = self
            .kept
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)
            .and_then(Option::as_ref)
            .is_some_and(|r| r.as_raw() == raw_tex);
        if made[index as usize] != key || !same_tex {
            let kept = make(shader_view, self.cpu(index));
            if kept.is_none() {
                null_for(shader_view, self.cpu(index));
                made[index as usize] = 0;
            } else {
                made[index as usize] = key;
                made.link(index);
                self.made_by(index, "placed slot (plan or record)");
            }
            self.keep(index, kept.flatten());
        }
    }

    /// Notes where slot `index` reads in place now (`None`: on its host
    /// copy) and whether that changed, so the slot must be made again (the
    /// same view, its memory remapped: a host copy follows the address, a
    /// view in place does not).
    fn in_place_changed(&self, index: u32, now: Option<u64>) -> bool {
        if !crate::shared::in_place() {
            return false;
        }
        let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
        let before = match now {
            Some(gpu) => b.in_place.insert(index, gpu),
            None => b.in_place.remove(&index),
        };
        before != now
    }

    /// Forgets the slot's made record, so the next use makes it again.
    fn unmake(&self, index: u32) {
        self.made.lock().unwrap_or_else(|e| e.into_inner())[index as usize] = 0;
    }

    fn ensure_buffer_slot(&self, index: u32, key: u64, view: &View) -> Option<ID3D12Resource> {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if made[index as usize] != key {
            let kept = make(view, self.cpu(index));
            let res = kept.flatten();
            if res.is_none() {
                null_for(view, self.cpu(index));
                made[index as usize] = 0;
            } else {
                made[index as usize] = key;
                made.link(index);
                self.made_by(index, "plan buffer");
            }
            self.keep(index, res.clone());
            return res;
        }
        self.kept
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)
            .and_then(Option::clone)
    }

    fn ensure_cbv_bounce(
        &self,
        index: u32,
        aligned: u32,
    ) -> Option<(u64, usize, ID3D12Resource, bool)> {
        let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = b.cbv_bounce.get_mut(&index)
            && slot.cap >= aligned
        {
            let was_active = std::mem::replace(&mut slot.active, true);
            return Some((slot.gpu, slot.cpu, slot.resource.clone(), was_active));
        }
        let needed = u64::from(aligned);
        let need_chunk = b
            .bounce_chunks
            .last()
            .is_none_or(|&(_, _, _, used)| used + needed > BOUNCE_CHUNK);
        if need_chunk {
            let chunk_size = BOUNCE_CHUNK.max(needed);
            let (res, data) = host()?.shared_buffer(chunk_size)?;
            d3d12x::host::name(&res, || "bindless cbv bounce".into());
            // SAFETY: a live buffer.
            let gpu = unsafe { res.GetGPUVirtualAddress() };
            b.bounce_chunks.push((res, data as usize, gpu, 0));
        }
        let last = b.bounce_chunks.last_mut()?;
        let at = last.3;
        last.3 += needed;
        let (gpu, cpu, res) = (last.2 + at, last.1 + at as usize, last.0.clone());
        b.cbv_bounce.insert(
            index,
            BounceSlot {
                gpu,
                cpu,
                cap: aligned,
                resource: res.clone(),
                active: true,
            },
        );
        Some((gpu, cpu, res, false))
    }

    fn materialize_bindless_cbv(
        &self,
        index: u32,
        key: u64,
        address: u64,
        capped: u32,
        in_place: Option<crate::shared::SharedRange>,
        reads: &mut Vec<(Arc<d3d12x::shadow::Shadow>, usize, usize)>,
        bounces: &mut Vec<(usize, u64, u32)>,
    ) -> Option<ID3D12Resource> {
        let host = host()?;
        let aligned = capped.next_multiple_of(256);
        if let Some(r) = in_place {
            // The constants read in place, no copy and no bounce.
            let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
            let bounce_active = b.cbv_bounce.get_mut(&index).map(|s| std::mem::replace(&mut s.active, false)).unwrap_or(false);
            drop(b);
            let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            if made[index as usize] != key || bounce_active {
                let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: shared_va(&r), SizeInBytes: aligned };
                // SAFETY: a slot of our heap; the range lies in the shared buffer.
                unsafe { gate::cbv(&host.device, Some(&desc), self.cpu(index)) };
                made[index as usize] = key;
                made.link(index);
                self.made_by(index, "plan constant");
                self.keep(index, Some(r.buffer.clone()));
            }
            crate::shared::count_view(crate::shared::VIEW_CBV, true);
            return Some(r.buffer);
        }
        crate::shared::count_view(crate::shared::VIEW_CBV, false);
        if address.is_multiple_of(256)
            && let Some(shadow) = d3d12x::shadow::find(address, u64::from(capped))
            && shadow.gpu(address).is_multiple_of(256)
        {
            let bounce_active = self
                .bindless
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .cbv_bounce
                .get(&index)
                .is_some_and(|s| s.active);
            {
                let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
                if made[index as usize] != key || bounce_active {
                    let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                        BufferLocation: shadow.gpu(address),
                        SizeInBytes: aligned,
                    };
                    // SAFETY: a slot of our heap; `shadow` covers the range.
                    unsafe {
                        gate::cbv(&host.device,
                            Some(&desc), self.cpu(index))
                    };
                    made[index as usize] = key;
                    made.link(index);
                    self.made_by(index, "plan constant");
                    self.keep(index, Some(shadow.buffer.clone()));
                }
            }
            if bounce_active
                && let Some(slot) = self
                    .bindless
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .cbv_bounce
                    .get_mut(&index)
            {
                slot.active = false;
            }
            let len = u64::from(capped).min(shadow.left(address)) as usize;
            if len > 0 {
                reads.push((shadow.clone(), address as usize, len));
            }
            return Some(shadow.buffer.clone());
        }

        let (gpu, cpu, res, was_active) = self.ensure_cbv_bounce(index, aligned)?;
        {
            let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            if made[index as usize] != key || !was_active {
                let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                    BufferLocation: gpu,
                    SizeInBytes: aligned,
                };
                // SAFETY: a slot of our heap; `gpu` is 256-byte aligned in `res`.
                unsafe {
                    gate::cbv(&host.device,
                        Some(&desc), self.cpu(index))
                };
                made[index as usize] = key;
                made.link(index);
                self.made_by(index, "plan constant");
                self.keep(index, Some(res.clone()));
            }
        }
        bounces.push((cpu, address, capped));
        Some(res)
    }

    /// At draw/dispatch recording time: makes the placed-texture slots
    /// changed since (or whose textures were evicted), and records on `list`
    /// the upload of the textures they newly reach and of a few known ones
    /// due a look ([`RECHECK_FRAMES`]), each moved to the shader resource
    /// state. Buffer and constant-buffer views are materialized and refreshed
    /// at submission from [`HostHeap::bindless_plan`].
    ///
    /// shortcut: a texture is moved to the shader resource state in the list
    /// that uploads it and keeps that state after; one another list moves
    /// (a copy into it, say) is read in that state through bindless slots.
    pub(crate) fn prepare_bindless_record(&self, list: &d3d12x::host::HostList) {
        if self.sampler {
            return;
        }
        let epoch = d3d12x::resource::placed_epoch();
        let frame = d3d12x::queue::presented();
        // Most draws find nothing to do: a few atomics, no lock.
        let evictions = self.record_looked[0].load(Ordering::Acquire) != epoch
            && self.record_looked[1].load(Ordering::Acquire) != frame + 1;
        let ready = TEXTURES_READY.load(Ordering::Acquire);
        if !evictions
            && !self.record_hint.load(Ordering::Acquire)
            && self.recheck_at.load(Ordering::Acquire) > frame
            && (ready == self.record_ready.load(Ordering::Acquire) || !self.record_waits.load(Ordering::Acquire))
        {
            return;
        }
        // One recording thread prepares at a time; the others go on.
        if self.recording.swap(true, Ordering::Acquire) {
            return;
        }
        struct Done<'a>(&'a std::sync::atomic::AtomicBool);
        impl Drop for Done<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }
        let _done = Done(&self.recording);
        let started = std::time::Instant::now();
        // Asked before taking the heap's lock: the placed textures' lock is
        // taken first where both are held.
        let live = evictions.then(d3d12x::resource::placed_live);
        let (todo, due) = {
            // The submission may hold the heap's lock for a while: the work
            // waits for a later draw rather than this one waiting.
            let mut b = match self.bindless.try_lock() {
                Ok(b) => b,
                Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) => return,
            };
            let b = &mut *b;
            // Textures made since: their slots are prepared again.
            if b.ready_seen[1] != ready {
                b.ready_seen[1] = ready;
                self.record_ready.store(ready, Ordering::Release);
                let waiting = std::mem::take(&mut b.record_waiting);
                b.record_pending.extend(waiting);
                self.record_waits.store(false, Ordering::Release);
            }
            if let Some(live) = &live {
                b.uploaded.retain(|t, _| live.contains(t));
                for (&slot, &(_, t)) in &b.recorded {
                    if !live.contains(&t) {
                        b.record_pending.insert(slot);
                    }
                }
                b.recorded.retain(|_, (_, t)| live.contains(t));
                self.record_looked[0].store(epoch, Ordering::Release);
                self.record_looked[1].store(frame + 1, Ordering::Release);
            }
            let mut todo: Vec<(u32, Option<(u32, u32)>)> = Vec::new();
            if b.budget()[1] > 0 {
                while todo.len() < SLOT_CHUNK
                    && let Some(i) = b.record_pending.pop_first()
                {
                    todo.push((i, b.placed_set.get(&i).copied()));
                }
            }
            let mut due = Vec::new();
            while due.len() < RECHECKS as usize
                && (b.rechecks.0 != frame || b.rechecks.1 < RECHECKS)
                && let Some(&(f, t)) = b.recheck.front()
                && f + RECHECK_FRAMES <= frame
            {
                b.recheck.pop_front();
                due.push(t as crate::d3d12x::object::This);
            }
            self.recheck_at.store(
                b.recheck.front().map_or(u64::MAX, |&(f, _)| f + RECHECK_FRAMES),
                Ordering::Release,
            );
            self.record_hint
                .store(!b.record_pending.is_empty(), Ordering::Release);
            (todo, due)
        };
        if todo.is_empty() && due.is_empty() {
            return;
        }
        count(RECORD_CALLS, 1);
        let taken = todo.len();
        let mut stale = Vec::new();
        let mut learnt = Vec::new();
        let mut reached = Vec::new();
        let left = self.budget_left(1);
        let mut todo = todo.into_iter();
        let mut over = Vec::new();
        for (index, entry) in todo.by_ref() {
            let Some((vi, generation)) = entry else {
                continue;
            };
            if started.elapsed().as_micros() as u64 > left {
                // Over budget: the rest waits (this one too).
                over.push(index);
                break;
            }
            match self.resolve_placed(index, vi, generation) {
                Ok(Some(held)) => {
                    let key = u64::from(generation) << 32 | u64::from(vi);
                    learnt.push((index, (key, held.as_this() as usize)));
                    reached.push(held);
                }
                Ok(None) => {}
                Err(()) => stale.push(index),
            }
        }
        over.extend(todo.map(|(i, _)| i));
        // Slots were counted as taken, and most went back over budget
        // to be taken again (225,126 "prepared" a frame in 880 calls): what
        // is prepared and what goes back are counted apart.
        count(RECORD_SLOTS, (taken - over.len()) as u64);
        count(RECORD_BACK, over.len() as u64);
        reached.extend(due.into_iter().filter_map(d3d12x::resource::hold_placed));
        for held in reached {
            let texture = held.as_this();
            let Some(host_tex) = d3d12x::resource::host_resource(texture) else {
                continue;
            };
            // A texture whose last upload was put off or partial is
            // asked again every frame until complete (otherwise only every
            // 60 frames, 8 a frame: a deferred upload was lost and the
            // loading-screen artwork stayed black).
            if d3d12x::texture::title_written(host_tex)
                && (d3d12x::texture::pending(texture) || self.upload_due(texture, host_tex, frame))
            {
                count(RECORD_UPLOADS, 1);
                d3d12x::texture::upload(list, texture, host_tex);
            }
            const SHADER_RESOURCE: u32 = 0x40 | 0x80;
            d3d12x::list::ensure_state(list, texture, SHADER_RESOURCE);
        }
        {
            let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
            b.recorded.extend(learnt);
            for idx in stale {
                self.drop_bindless(&mut b, idx, None);
            }
            b.record_pending.extend(over);
            self.record_hint
                .store(!b.record_pending.is_empty(), Ordering::Release);
        }
        let us = started.elapsed().as_micros() as u64;
        self.budget_spend(1, us);
        d3d12x::host::BINDLESS_US[1].fetch_add(us, Ordering::Relaxed);
        d3d12x::host::stage(d3d12x::host::STAGE_RECORD, started);
    }

    /// The placed texture slot `index` views, made in the slot: `Err` if the
    /// view went stale, `None` if it is not a placed texture or cannot be made
    /// (the slot is then null). Returns a held reference so eviction cannot
    /// free it before the caller finishes uploading/transitioning.
    fn resolve_placed(
        &self,
        index: u32,
        vi: u32,
        generation: u32,
    ) -> Result<Option<d3d12x::resource::HeldTexture>, ()> {
        use windows::core::Interface;
        let Some(view) = view_at(vi, generation) else {
            self.clear_stale_slot(index);
            return Err(());
        };
        let View::Placed(false, resource, desc) = &view else {
            return Ok(None);
        };
        const BUFFER: u32 = 1;
        if resource[0] == BUFFER {
            return Ok(None);
        }
        let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
        if !plausible(address, 1) {
            self.set_unmapped_null(index, &view);
            return Ok(None);
        }
        if !placed_ready(&view) {
            // Prepared again once the texture thread has made it.
            self.set_unmapped_null(index, &view);
            self.bindless.lock().unwrap_or_else(|e| e.into_inner()).record_waiting.insert(index);
            self.record_waits.store(true, Ordering::Release);
            return Ok(None);
        }
        let Some((shader_view, held)) = placed_texture(&view) else {
            self.set_unmapped_null(index, &view);
            return Ok(None);
        };
        let tex_this = held.as_this();
        let Some(host_tex) = d3d12x::resource::host_resource(tex_this) else {
            self.set_unmapped_null(index, &view);
            return Ok(None);
        };
        let key = u64::from(generation) << 32 | u64::from(vi);
        self.ensure_placed_slot(index, key, &shader_view, host_tex.as_raw());
        Ok(Some(held))
    }

    /// Whether to upload placed texture `texture` (reached through a slot)
    /// in a list recorded in `frame`: the first time, then as
    /// [`RECHECK_FRAMES`] allows.
    fn upload_due(&self, texture: crate::d3d12x::object::This, host: &ID3D12Resource, frame: u64) -> bool {
        use windows::core::Interface;
        let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
        let b = &mut *b;
        let due = match b.uploaded.get(&(texture as usize)) {
            None => true,
            // Another texture at the object's address.
            Some((h, _)) if h.as_raw() != host.as_raw() => true,
            Some((_, last)) if frame >= last + RECHECK_FRAMES => {
                if b.rechecks.0 != frame {
                    b.rechecks = (frame, 0);
                }
                b.rechecks.1 += 1;
                b.rechecks.1 <= RECHECKS
            }
            Some(_) => false,
        };
        if due {
            b.uploaded
                .insert(texture as usize, (host.clone(), frame));
            b.recheck.push_back((frame, texture as usize));
            self.recheck_at.store(
                b.recheck.front().map_or(u64::MAX, |&(f, _)| f + RECHECK_FRAMES),
                Ordering::Release,
            );
        }
        due
    }

    /// At submission: what the heap's bindless slots need. Only slots
    /// written since the last plan are made again; released views are found
    /// by one look at all slots' generations, and slots of evicted placed
    /// textures, of unmapped memory once remapped, and of loose memory
    /// remapped since ([`bindless_remapped`]) are made again. With nothing
    /// changed, or nothing new needed by the slots made, it is the last
    /// plan, whose reads the caller refreshes where they were written.
    /// The heap's last plan, if one was made (read only: nothing is
    /// synced or made).
    pub(crate) fn last_plan(&self) -> Option<Arc<BindlessPlan>> {
        self.bindless.lock().unwrap_or_else(|e| e.into_inner()).plan.clone()
    }

    pub(crate) fn bindless_plan(&self) -> Option<Arc<BindlessPlan>> {
        if self.sampler {
            return None;
        }
        // The whole plan, against which its parts are read
        // (scan + look + make + build were 240 of 613 ms a frame).
        let _total = Timed(US_PLAN_TOTAL, std::time::Instant::now());
        self.sync_title_slots();
        let looking = std::time::Instant::now();
        let epochs = (
            VIEWED.load(Ordering::Acquire),
            d3d12x::resource::placed_epoch(),
        );
        // Asked before taking the heap's lock: the placed textures' lock is
        // taken first where both are held.
        let frame = d3d12x::queue::presented() + 1;
        let evictions = {
            let at = std::time::Instant::now();
            let b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
            count_us(US_LOOK_LOCK, at);
            b.checked.1 != epochs.1 && b.looked[0] != frame
        };
        let at = std::time::Instant::now();
        let live = evictions.then(d3d12x::resource::placed_live);
        count_us(US_LOOK_LIVE, at);
        let (mut todo, check) = {
            let at = std::time::Instant::now();
            let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
            count_us(US_LOOK_LOCK, at);
            let b = &mut *b;
            // Textures made since by the texture thread: slots waiting on one
            // are made again.
            let ready = TEXTURES_READY.load(Ordering::Acquire);
            if b.ready_seen[0] != ready {
                b.ready_seen[0] = ready;
                let waiting = std::mem::take(&mut b.waiting);
                b.dirty.extend(waiting);
            }
            // Released views and evicted textures are looked for once a frame.
            let look = b.looked[0] != frame && b.checked != epochs;
            if b.dirty.is_empty()
                && !look
                && let Some(plan) = &b.plan
            {
                plan_reused();
                return Some(plan.clone());
            }
            if !look
                && b.budget()[0] == 0
                && let Some(plan) = &b.plan
            {
                plan_reused();
                return Some(plan.clone());
            }
            let viewed = look && b.checked.0 != epochs.0;
            let live = if look { live.as_ref() } else { None };
            // Loose memory remapped since the last look.
            let checking = std::time::Instant::now();
            let remapped = if viewed { remapped_since(&mut b.remap_seen) } else { Remapped::Ranges(Vec::new()) };
            count_us(US_LOOK_CHECKS, checking);
            if look {
                b.checked = epochs;
                b.looked[0] = frame;
            }
            let mut dirty = BTreeSet::new();
            while dirty.len() < 4 * SLOT_CHUNK
                && let Some(i) = b.dirty.pop_first()
            {
                dirty.insert(i);
            }
            // All slots are walked only at the frame's look: with nothing to
            // look for, every plan walked them all for nothing.
            let walking = std::time::Instant::now();
            let mut hit = 0;
            if live.is_some() || viewed {
                for (&i, u) in &b.uses {
                    let evicted = live
                        .zip(u.placed)
                        .is_some_and(|(l, t)| !l.contains(&t));
                    let moved = remapped.hits(u);
                    hit += u64::from(moved);
                    if evicted || viewed && u.unmapped || moved {
                        dirty.insert(i);
                    }
                }
            }
            count(SLOTS_REMAPPED, hit);
            count_us(US_LOOK_WALK, walking);
            let checking = std::time::Instant::now();
            let check: Vec<(u32, (u32, u32))> = if viewed {
                b.slots
                    .iter()
                    .filter(|(i, _)| !dirty.contains(i))
                    .map(|(&i, &v)| (i, v))
                    .collect()
            } else {
                Vec::new()
            };
            count_us(US_LOOK_CHECKS, checking);
            let todo: Vec<(u32, Option<(u32, u32)>)> = dirty
                .into_iter()
                .map(|i| (i, b.slots.get(&i).copied()))
                .collect();
            (todo, check)
        };
        let at = std::time::Instant::now();
        todo.extend(stale_views(&check).into_iter().map(|(i, v)| (i, Some(v))));
        count_us(US_LOOK_STALE, at);
        count_us(US_LOOK, looking);
        let (started, left) = (std::time::Instant::now(), self.budget_left(0));
        let mut made_now = Vec::new();
        let mut todo = todo.into_iter();
        for (index, entry) in todo.by_ref() {
            if started.elapsed().as_micros() as u64 > left {
                // Over budget: the rest waits for the next submission.
                self.bindless
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .dirty
                    .insert(index);
                break;
            }
            made_now.push(index);
            AREA_MADE[area(index)].fetch_add(1, Ordering::Relaxed);
            let made = entry.map(|e| (e, self.use_slot(index, e)));
            let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
            match made {
                None => {
                    b.uses.remove(&index);
                }
                Some((_, Some(u))) => {
                    b.uses.insert(index, u);
                }
                Some((e, None)) => self.drop_bindless(&mut b, index, Some(e)),
            }
        }
        self.budget_spend(0, started.elapsed().as_micros() as u64);
        count_us(US_MAKE, started);
        count(SLOTS_MADE, made_now.len() as u64);
        let building = std::time::Instant::now();
        let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
        count_us(US_PLAN_LOCKS, building);
        b.dirty.extend(todo.map(|(i, _)| i));
        if made_now.is_empty() && let Some(plan) = &b.plan {
            plan_reused();
            return Some(plan.clone());
        }
        // In the frame of the last full plan: that plan and the slots made
        // now. The title rewrites slots at every submission (513 plans made,
        // 0 reused by 99.6 s), and a full plan looks
        // at every slot. What changed or went keeps its old reads, writes
        // and resources until the frame's next full plan: an extra refresh,
        // a range copied back only where the GPU wrote it, a resource held a
        // frame longer; a slice bounced again is copied by its newest part
        // only.
        let base = b.plan.as_ref().filter(|p| p.frame == frame).cloned();
        // The slots made now need nothing the plan does not do already (the
        // title rewrote them with views of memory the plan reads): it is
        // the same plan. Rewritten slots made every plan new (23 made,
        // 445 extended, 0 reused by 129 s).
        if let Some(p) = &base
            && made_now.iter().filter_map(|i| b.uses.get(i)).all(|u| p.covers(u))
        {
            plan_reused();
            count_us(US_BUILD, building);
            return Some(p.clone());
        }
        // An extension is a new part holding only what the plan does
        // not do already; the parts before are shared, not copied.
        let (mut reads, mut writes, mut bounces, mut retained, mut held) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), std::collections::HashSet::new());
        let mut in_place: Vec<(usize, usize)> = Vec::new();
        let mut add = |u: &SlotUse| {
            let p = base.as_deref();
            if let Some((a, l)) = u.shared
                && p.is_none_or(|p| !p.parts.iter().any(|q| q.covers_in_place(&(a, l))))
            {
                in_place.push((a as usize, (a + l) as usize));
            }
            if p.is_none_or(|p| !p.parts.iter().any(|q| inside(&q.reads, &u.read))) {
                reads.extend(u.read.clone());
            }
            if p.is_none_or(|p| !p.parts.iter().any(|q| inside(&q.writes, &u.write))) {
                writes.extend(u.write.clone());
            }
            if let Some(k) = u.bounce
                && p.is_none_or(|p| !p.bounces(&k))
            {
                bounces.push(k);
            }
            // Once each: most slots view the same few host copies.
            if let Some(r) = &u.retained
                && p.is_none_or(|p| !p.parts.iter().any(|q| q.holds(r)))
                && held.insert(windows::core::Interface::as_raw(r) as usize)
            {
                retained.push(r.clone());
            }
        };
        match &base {
            Some(_) => made_now.iter().filter_map(|i| b.uses.get(i)).for_each(&mut add),
            None => b.uses.values().for_each(&mut add),
        }
        // One bounce per slice, the newest: an old one left in place could
        // copy its address over the new one's (bounces are skipped when
        // their memory is unwritten).
        let mut bounce_at = HashMap::new();
        let mut bounces: Vec<(usize, u64, u32)> =
            bounces.into_iter().rev().filter(|&(cpu, a, l)| bounce_at.insert(cpu, (a, l)).is_none()).collect();
        bounces.reverse();
        let reads = merged(reads, READ_GAP);
        // A full plan carries over the old plan's askings: a bounce keeps its
        // own; a read the old reads cover keeps their oldest, a part never
        // read since makes it new. Keyed by exact range, a range grown by one
        // slot was read again from scratch at each extension (51,195
        // bindless reads a frame, 3,536 skipped). The old plan holds its
        // copies, so a pointer still names the same copy. An extension's
        // reads and bounces are new (what the plan did already is left out).
        let old_plan = if base.is_none() { b.plan.clone() } else { None };
        let asked: HashMap<(usize, u64, u32), u64> = old_plan
            .iter()
            .flat_map(|p| p.parts.iter())
            .flat_map(|p| {
                let asked = p.bounce_asked.lock().unwrap_or_else(|e| e.into_inner());
                p.bounces.iter().copied().zip(asked.iter().copied()).collect::<Vec<_>>()
            })
            .collect();
        let bounce_asked = Mutex::new(bounces.iter().map(|k| asked.get(k).copied().unwrap_or(0)).collect());
        let mut old: Vec<(usize, usize, usize, u64)> = old_plan
            .iter()
            .flat_map(|p| p.parts.iter())
            .flat_map(|p| {
                let asked = p.read_asked.lock().unwrap_or_else(|e| e.into_inner());
                p.reads.iter().zip(asked.iter()).map(|((s, a, l), &w)| (Arc::as_ptr(s) as usize, *a, a + l, w)).collect::<Vec<_>>()
            })
            .collect();
        // Parts may overlap: merged per copy, the oldest asking kept.
        old.sort_unstable();
        let old = old.into_iter().fold(Vec::<(usize, usize, usize, u64)>::new(), |mut out, (p, s, e, w)| {
            match out.last_mut() {
                Some(last) if last.0 == p && s <= last.2 => (last.2, last.3) = (last.2.max(e), last.3.min(w)),
                _ => out.push((p, s, e, w)),
            }
            out
        });
        let read_asked = Mutex::new(
            reads
                .iter()
                .map(|(s, a, l)| {
                    let (key, end) = (Arc::as_ptr(s) as usize, a + l);
                    // Old reads are sorted by (copy, address) and disjoint per copy.
                    let mut at = old.partition_point(|&(p, _, e, _)| (p, e) <= (key, *a));
                    let (mut cur, mut since) = (*a, u64::MAX);
                    while cur < end
                        && let Some(&(p, s0, e0, w)) = old.get(at)
                        && p == key
                        && s0 <= cur
                    {
                        (cur, since) = (cur.max(e0), since.min(w));
                        at += 1;
                    }
                    if cur >= end { since } else { 0 }
                })
                .collect(),
        );
        let part = Arc::new(PlanPart {
            reads,
            writes: merged(writes, 0),
            bounces,
            bounce_at,
            bounce_asked,
            read_asked,
            _retained: retained,
            held,
            in_place: merged_spans(in_place),
        });
        let mut parts = base.as_ref().map(|p| p.parts.clone()).unwrap_or_default();
        parts.push(part);
        let plan = Arc::new(BindlessPlan { parts, frame });
        let old = b.plan.replace(plan.clone());
        drop(b);
        // The old plan let go outside the lock, timed: a full plan's
        // predecessor holds a frame of parts and their resources.
        let dropping = std::time::Instant::now();
        drop(old);
        count_us(US_PLAN_DROP, dropping);
        PLANS[if base.is_some() { 2 } else { 0 }].fetch_add(1, Ordering::Relaxed);
        count(if base.is_some() { PLANS_EXTENDED } else { PLANS_FULL }, 1);
        count(PLAN_PARTS, plan.parts() as u64);
        count_us(US_BUILD, building);
        Some(plan)
    }

    /// Materializes bindless slot `index` from its view `(vi, generation)`
    /// and finds what it reads and writes; `None` if the slot is no longer
    /// one for bindless access (its view went stale, or is not of a buffer).
    fn use_slot(&self, index: u32, (vi, generation): (u32, u32)) -> Option<SlotUse> {
        use windows::core::Interface;
        let Some(view) = view_at(vi, generation) else {
            self.clear_stale_slot(index);
            return None;
        };
        if !view.is_buffer() {
            return None;
        }
        let unmapped = |view: &View| {
            self.set_unmapped_null(index, view);
            Some(SlotUse {
                unmapped: true,
                ..SlotUse::default()
            })
        };
        let key = u64::from(generation) << 32 | u64::from(vi);
        let mut slot = SlotUse::default();
        match &view {
            &View::Constant(address, size) => {
                let capped = size.min(65_536);
                let at = std::time::Instant::now();
                let valid = valid_title_address(address, u64::from(capped));
                let at = count_us_next(US_MAKE_CHECK, at);
                if !valid {
                    return unmapped(&view);
                }
                let (mut reads, mut bounces) = (Vec::new(), Vec::new());
                let in_place = view_in_place(&view);
                if self.in_place_changed(index, in_place.as_ref().map(shared_va)) {
                    self.unmake(index);
                }
                if in_place.is_some() {
                    slot.shared = Some((address, u64::from(capped)));
                }
                let at = count_us_next(US_MAKE_IN_PLACE, at);
                let _slot_time = Timed(US_MAKE_SLOT, at);
                match self.materialize_bindless_cbv(
                    index,
                    key,
                    address,
                    capped,
                    in_place,
                    &mut reads,
                    &mut bounces,
                ) {
                    Some(res) => slot.retained = Some(res),
                    None => return unmapped(&view),
                }
                slot.read = reads.pop();
                slot.bounce = bounces.pop();
            }
            View::Placed(false, resource, desc) if resource[0] != 1 => {
                slot.placed = Some(0);
                let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                if !plausible(address, 1) {
                    self.set_unmapped_null(index, &view);
                    return Some(slot);
                }
                // Timed apart: the texture is made here on first reach
                // (CreateCommittedResource), which is most of plan making.
                let at = std::time::Instant::now();
                count(PLACED_MADE, 1);
                if !placed_ready(&view) {
                    // A null view until the texture thread has made it.
                    self.set_unmapped_null(index, &view);
                    self.bindless.lock().unwrap_or_else(|e| e.into_inner()).waiting.insert(index);
                    count_us(US_MAKE_PLACED, at);
                    return Some(slot);
                }
                if let Some((shader_view, held)) = placed_texture(&view)
                    && let Some(host_tex) = d3d12x::resource::host_resource(held.as_this())
                {
                    self.ensure_placed_slot(index, key, &shader_view, host_tex.as_raw());
                    slot.placed = Some(held.as_this() as usize);
                    slot.retained = Some(host_tex.clone());
                } else {
                    self.set_unmapped_null(index, &view);
                }
                count_us(US_MAKE_PLACED, at);
            }
            _ => {
                let Some((address, len)) = buffer_range(&view) else {
                    self.set_unmapped_null(index, &view);
                    return Some(slot);
                };
                let at = std::time::Instant::now();
                let valid = valid_title_address(address, len);
                let at = count_us_next(US_MAKE_CHECK, at);
                if !valid {
                    return unmapped(&view);
                }
                let in_place = view_in_place(&view);
                if self.in_place_changed(index, in_place.as_ref().map(shared_va)) {
                    self.unmake(index);
                }
                let at = count_us_next(US_MAKE_IN_PLACE, at);
                slot.retained = self.ensure_buffer_slot(index, key, &view);
                let at = count_us_next(US_MAKE_SLOT, at);
                let _copy_time = Timed(US_MAKE_COPY, at);
                if let Some(r) = &in_place
                    && slot.retained.as_ref().is_some_and(|k| k.as_raw() == r.buffer.as_raw())
                {
                    // Read (and written) in place: nothing to refresh or copy back.
                    slot.shared = Some((address, len));
                } else if let Some((shadow, _, _)) = view_copy(&view) {
                    let clamped = len.min(shadow.left(address)) as usize;
                    if clamped > 0 {
                        slot.read = Some((shadow.clone(), address as usize, clamped));
                        if matches!(view, View::Unordered(..) | View::Placed(true, ..)) {
                            if clamped <= WRITE_CAP {
                                slot.write = Some((shadow, address as usize, clamped));
                            } else {
                                static ONCE: std::sync::Once = std::sync::Once::new();
                                ONCE.call_once(|| {
                                    xnw_core::warn!(
                                        "d3d12x: a bindless unordered access view too big to copy back ({clamped} bytes)"
                                    );
                                });
                            }
                        }
                    }
                }
            }
        }
        Some(slot)
    }

    /// `XNW_SLOT_WATCH`: at dispatch `k` of the watched program
    /// `name`, each watched slot of this heap: the view it was made from,
    /// the title's record there, the runtime stride-table entry (stride 0:
    /// heap-picked scalar loads read the slot as a constant buffer), and on
    /// WARP the descriptor's words.
    pub(crate) fn watch_report(&self, name: &str, k: u32) {
        for &slot in super::slotwatch::slots() {
            if slot >= self.count + self.dynamic {
                continue;
            }
            let stride = self.strides.as_ref().filter(|t| slot < t.len).map_or_else(
                || "none".to_string(),
                // SAFETY: `t.ptr` is a live mapped upload buffer of `t.len` entries of two dwords.
                |t| format!("{:08x?}", unsafe { (t.ptr as *const [u32; 2]).add(slot as usize).read() }),
            );
            let title_start = self.bindless.lock().unwrap_or_else(|e| e.into_inner()).title_start;
            xnw_core::info!(
                "slot watch: dispatch {k} of {name}: slot {slot} made from {}; stride entry {stride}; {}; title record {}",
                self.describe_slot(slot).unwrap_or_else(|| "- (dynamic)".into()),
                super::slotwatch::warp_words(self.cpu(slot).ptr, self.step),
                super::slotwatch::title_record(title_start, slot, self.count)
            );
        }
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = *self
            .made
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)?;
        if key == 0 {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = d3d12x::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Whether `[address, address + len)` may be title memory: inside the
/// title's address range, not null.
fn plausible(address: u64, len: u64) -> bool {
    address >= 0x10000
        && len > 0
        && address
            .checked_add(len)
            .is_some_and(|end| end <= (1u64 << 48))
}

/// Whether `[address, address + len)` is title memory: plausible, and in a
/// title buffer or committed. The kernel is asked only outside buffers:
/// in the title's big pools each asking costs milliseconds.
fn valid_title_address(address: u64, len: u64) -> bool {
    plausible(address, len)
        && (d3d12x::shadow::in_buffer(address, len)
            || committed(address, 1))
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| xnw_core::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = d3d12x::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = d3d12x::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
/// Bytes per element of the buffer view `view` (0: not a buffer view, or
/// one whose element size is unknown). VIctory (42538cd).
/// [`view_stride`] for the record writer.
pub(crate) fn view_stride_of(view: &View) -> u32 {
    view_stride(view)
}

fn view_stride(view: &View) -> u32 {
    let uav = |w: &[u32; 10]| {
        let desc = uav_desc(w);
        // SAFETY: a buffer view's union member.
        let buffer = unsafe { desc.Anonymous.Buffer };
        if buffer.StructureByteStride > 0 {
            u64::from(buffer.StructureByteStride)
        } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
            4
        } else {
            d3d12x::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
        }
    };
    let bytes = match view {
        View::Shader(_, Some(desc)) => element_size(&srv_desc(desc)),
        View::Placed(false, resource, desc) if resource[0] == 1 => {
            desc[..10].try_into().ok().map_or(0, |w: &[u32; 10]| element_size(&srv_desc(w)))
        }
        View::Unordered(_, _, Some(desc)) => uav(desc),
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            desc[..10].try_into().ok().map_or(0, |w: &[u32; 10]| uav(w))
        }
        _ => 0,
    };
    u32::try_from(bytes).unwrap_or(0)
}

fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        d3d12x::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            d3d12x::resource::buffer_bytes(*r as crate::d3d12x::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = d3d12x::resource::buffer_bytes(*r as crate::d3d12x::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let first_off = buffer.FirstElement.checked_mul(element)?;
    let byte_len = u64::from(buffer.NumElements).checked_mul(element)?;
    Some((base.checked_add(first_off)?, byte_len))
}

/// The host copy a buffer view's bytes are in (the one its host view reads,
/// [`d3d12x::shadow::for_view`]) and the view's title range. VIctory
/// 86a5ed6 (`view_copy`).
pub(crate) fn view_copy(view: &View) -> Option<(Arc<d3d12x::shadow::Shadow>, u64, u64)> {
    let (address, len) = buffer_range(view)?;
    let shadow = d3d12x::shadow::for_view(address, len, u64::from(view_stride(view)))?;
    Some((shadow, address, len))
}

/// Whether `count` elements of `element` bytes from element `first` lie in
/// `buffer`. The console takes a view that is empty or runs past its
/// memory; the host removes the device for one. VIctory 4051149.
fn in_buffer(buffer: &ID3D12Resource, first: u64, count: u32, element: u64) -> bool {
    // SAFETY: a live buffer.
    let width = unsafe { buffer.GetDesc() }.Width;
    count > 0 && (first + u64::from(count)).saturating_mul(element) <= width
}

/// Buffer views past their host copy made null.
static PAST_COPY: AtomicU64 = AtomicU64::new(0);

/// The bindless line: buffer views off their copy's elements (asked,
/// own copies made) and views past their copy made null, since the last.
pub(crate) fn view_fix_text() -> String {
    let [asked, made] = [0, 1].map(|i| d3d12x::shadow::OFF_ELEMENT[i].swap(0, Ordering::Relaxed));
    format!("buffer views off their copy's elements {asked} (own copies made {made}), views past their copy made null {}", PAST_COPY.swap(0, Ordering::Relaxed))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        d3d12x::resource::element(desc.Format.0 as u32)?.0
    };
    let first_off = buffer.FirstElement.checked_mul(element)?;
    let byte_len = u64::from(buffer.NumElements).checked_mul(element)?;
    Some((base.checked_add(first_off)?, byte_len))
}

/// The host resource a slot made by [`make`] views, if any.
type Kept = Option<ID3D12Resource>;

/// The GPU address of a range of title memory read in place.
pub(crate) fn shared_va(r: &crate::shared::SharedRange) -> u64 {
    // SAFETY: a live buffer.
    unsafe { r.buffer.GetGPUVirtualAddress() + r.offset }
}

/// `XNW_SHARED=2`: the shared buffer `len` bytes of title memory at
/// `start` are read through in place, if they resolve and land on an
/// element (`element` bytes) of it; counted under `kind` when given.
fn in_place_at(kind: Option<usize>, start: u64, len: u64, element: u64) -> Option<crate::shared::SharedRange> {
    if element == 0 || len == 0 {
        return None;
    }
    let r = match kind {
        Some(k) => crate::shared::range(k, start, len)?,
        None => crate::shared::lookup(start, len)?,
    };
    if !r.offset.is_multiple_of(element) {
        if let Some(k) = kind {
            crate::shared::fell_back(k);
        }
        return None;
    }
    Some(r)
}

/// Where [`make`] makes a buffer or constant buffer view: in place on a
/// shared buffer (`XNW_SHARED=2`, the range resolves), or `None` for its
/// host copy. Asked again by those who would refresh or copy back the copy.
pub(crate) fn view_in_place(view: &View) -> Option<crate::shared::SharedRange> {
    if !crate::shared::in_place() {
        return None;
    }
    let element = match view {
        &View::Constant(address, size) => {
            if !address.is_multiple_of(256) || address == 0 || size == 0 {
                return None;
            }
            return in_place_at(None, address, u64::from(size.min(65_536).next_multiple_of(256)), 256);
        }
        View::Shader(_, Some(desc)) => element_size(&srv_desc(desc)),
        View::Placed(false, resource, desc) if resource[0] == 1 => element_size(&srv_desc(desc[..10].try_into().ok()?)),
        View::Unordered(_, _, Some(desc)) => uav_element(desc)?,
        View::Placed(true, resource, desc) if resource[0] == 1 => uav_element(desc[..10].try_into().ok()?)?,
        _ => return None,
    };
    let (start, len) = buffer_range(view)?;
    in_place_at(None, start, len, element)
}

/// The element size of an unordered access buffer view.
fn uav_element(words: &[u32; 10]) -> Option<u64> {
    uav_desc_element(&uav_desc(words))
}

fn uav_desc_element(desc: &D3D12_UNORDERED_ACCESS_VIEW_DESC) -> Option<u64> {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some(if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        d3d12x::resource::element(desc.Format.0 as u32)?.0
    })
}

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    {
        let element = element_size(&desc);
        // SAFETY: a buffer view's union member.
        let buffer = unsafe { &mut desc.Anonymous.Buffer };
        if let Some(first) = buffer.FirstElement.checked_mul(element)
            && let Some(r) = in_place_at(Some(crate::shared::VIEW_SRV), address + first, u64::from(buffer.NumElements) * element, element)
        {
            buffer.FirstElement = r.offset / element;
            // SAFETY: a live buffer and a slot of our heap; the view lies in it.
            unsafe { gate::srv(&host()?.device, &r.buffer, Some(&desc), at) };
            return Some(Some(r.buffer));
        }
    }
    let _ = len;
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    if element == 0 {
        once!("d3d12x: buffer view of elements without a size");
        return None;
    }
    if buffer.NumElements == 0 {
        once!("d3d12x: buffer view of no elements made null");
        return None;
    }
    // VIctory 86a5ed6: the copy the view's own bytes are in (its own copy
    // if it is off the buffer copy's elements), its first element counted
    // from there.
    let start = address + buffer.FirstElement * element;
    let shadow = d3d12x::shadow::for_view(start, u64::from(buffer.NumElements) * element, element)?;
    buffer.FirstElement = shadow.offset(start) / element;
    if !in_buffer(&shadow.buffer, buffer.FirstElement, buffer.NumElements, element) {
        PAST_COPY.fetch_add(1, Ordering::Relaxed);
        once!("d3d12x: buffer view empty or past its host copy made null (first {}, count {}, element {element})", buffer.FirstElement, buffer.NumElements);
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        gate::srv(&host.device,
            &shadow.buffer, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    {
        let element = uav_desc_element(&desc).unwrap_or(0);
        // SAFETY: a buffer view's union member.
        let buffer = unsafe { &mut desc.Anonymous.Buffer };
        if buffer.NumElements > 0
            && let Some(first) = buffer.FirstElement.checked_mul(element)
            && let Some(r) = in_place_at(Some(crate::shared::VIEW_UAV), address + first, u64::from(buffer.NumElements) * element, element)
        {
            // The shaders' writes land in title memory (probe 6f).
            buffer.FirstElement = r.offset / element;
            buffer.CounterOffsetInBytes = 0;
            // SAFETY: a live buffer and a slot of our heap; the view lies in it.
            unsafe { gate::uav(&host()?.device, &r.buffer, None::<&ID3D12Resource>, Some(&desc), at) };
            return Some(Some(r.buffer));
        }
    }
    let _ = len;
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        d3d12x::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    if element == 0 {
        once!("d3d12x: unordered buffer view of elements without a size");
        return None;
    }
    // A view of no elements is an invalid call that removes the host
    // device (validation id 345, run of 2026-11-24); the caller makes a
    // null view instead.
    if buffer.NumElements == 0 {
        once!("d3d12x: unordered buffer view of no elements made null");
        return None;
    }
    // VIctory 86a5ed6, as for shader views.
    let start = address + buffer.FirstElement * element;
    let shadow = d3d12x::shadow::for_view(start, u64::from(buffer.NumElements) * element, element)?;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("d3d12x: unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    buffer.FirstElement = shadow.offset(start) / element;
    // VIctory 4051149: a view past its copy removes the host device.
    if !in_buffer(&shadow.buffer, buffer.FirstElement, buffer.NumElements, element) {
        PAST_COPY.fetch_add(1, Ordering::Relaxed);
        once!("d3d12x: unordered buffer view empty or past its host copy made null (first {}, count {}, element {element})", buffer.FirstElement, buffer.NumElements);
        return None;
    }
    buffer.CounterOffsetInBytes = 0;
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        gate::uav(&host.device,
            &shadow.buffer,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
    Some(Some(shadow.buffer.clone()))
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in another format than a
/// typed resource's. A view of a typed texture in another format of its
/// family takes the texture's format.
/// shortcut: such a view (sRGB of UNORM, say) reads without its
/// conversion; cast with castable formats if colors come out wrong.
fn fits(d: &D3D12_RESOURCE_DESC, mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use d3d12x::resource::{DEPTH_FORMATS, family};
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!("d3d12x: view the host does not allow (unordered {unordered}, format {own}, flags {:#x})", d.Flags.0);
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = family(own) == own && family(own + 1) == own;
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = d3d12x::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    // Packed 16-bit color textures stored as R16_TYPELESS (53) on NVIDIA.
    // A typeless view of one too, as its render target view writes
    // (host.rs `target_view`); VIctory's typeless-view rule would read R16_FLOAT.
    if own == 53 && matches!(view, 0 | 53 | 86 | 115) {
        if let Some(format) = format {
            format.0 = if unordered { 57 } else { 56 }; // R16_UINT / R16_UNORM
            if matches!(view, 0 | 53) {
                super::count_typeless(if unordered { super::TYPELESS_UAV } else { super::TYPELESS_SRV });
            }
            return true;
        }
        super::count_typeless(super::VIEWS_REFUSED);
        return false;
    }
    if view != 0 && family(view) != family(own) {
        once!("d3d12x: view format {view} of another family than the texture's {own}");
        super::count_typeless(super::VIEWS_REFUSED);
        return false;
    }
    // VIctory `adapt_view_format`: a typeless (or unnamed) view of
    // a typeless texture takes the family's concrete format.
    if typeless && (view == 0 || d3d12x::resource::is_typeless(view)) {
        let Some(format) = format else {
            super::count_typeless(super::VIEWS_REFUSED);
            return false;
        };
        format.0 = typeless_default(d) as i32;
        super::count_typeless(if unordered { super::TYPELESS_UAV } else { super::TYPELESS_SRV });
        return true;
    }
    if let Some(format) = format
        && !typeless
    {
        *format = d.Format;
    }
    true
}

/// The concrete view format of a typeless texture (VIctory
/// `concrete_format_for`): a depth texture's depth aspect, else the
/// family's first typed format.
fn typeless_default(d: &D3D12_RESOURCE_DESC) -> u32 {
    let own = d.Format.0 as u32;
    let depth = d.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    depth.then(|| d3d12x::resource::depth_readable(own)).flatten().unwrap_or_else(|| d3d12x::resource::first_concrete(own))
}

/// A whole-resource shader view of `resource` in `format` (VIctory
/// `default_srv_desc`; XNW: multisampled textures get their own dimension).
fn default_srv_desc(resource: &D3D12_RESOURCE_DESC, format: DXGI_FORMAT) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut desc = D3D12_SHADER_RESOURCE_VIEW_DESC { Format: format, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, ..Default::default() };
    let layers = u32::from(resource.DepthOrArraySize);
    let (one, many) = (layers <= 1, layers > 1);
    match resource.Dimension {
        D3D12_RESOURCE_DIMENSION_TEXTURE1D if many => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_SRV { MipLevels: u32::MAX, ArraySize: layers, ..Default::default() };
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE1D => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
            desc.Anonymous.Texture1D = D3D12_TEX1D_SRV { MipLevels: u32::MAX, ..Default::default() };
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE3D => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_SRV { MipLevels: u32::MAX, ..Default::default() };
        }
        _ if resource.SampleDesc.Count > 1 && one => desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS,
        _ if resource.SampleDesc.Count > 1 => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
            desc.Anonymous.Texture2DMSArray = D3D12_TEX2DMS_ARRAY_SRV { FirstArraySlice: 0, ArraySize: layers };
        }
        _ if many => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_SRV { MipLevels: u32::MAX, ArraySize: layers, ..Default::default() };
        }
        _ => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            desc.Anonymous.Texture2D = D3D12_TEX2D_SRV { MipLevels: u32::MAX, ..Default::default() };
        }
    }
    desc
}

/// A whole-resource unordered access view of `resource` in `format`
/// (VIctory `default_uav_desc`), mip 0.
fn default_uav_desc(resource: &D3D12_RESOURCE_DESC, format: DXGI_FORMAT) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: format, ..Default::default() };
    let layers = u32::from(resource.DepthOrArraySize);
    match resource.Dimension {
        D3D12_RESOURCE_DIMENSION_TEXTURE1D if layers > 1 => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV { ArraySize: layers, ..Default::default() };
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE1D => desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D,
        D3D12_RESOURCE_DIMENSION_TEXTURE3D => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV { WSize: u32::MAX, ..Default::default() };
        }
        _ if layers > 1 => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV { ArraySize: layers, ..Default::default() };
        }
        _ => desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D,
    }
    desc
}

/// The format a view of a typeless host copy names when the title named
/// none: the title texture's own (an sRGB texture stored typeless for
/// unordered access reads as sRGB, a depth one its depth aspect), else 0
/// (the family default in [`fits`]). XNW, not upstream: VIctory's typeless-view rule takes the
/// family's first format, which would read such a texture as UNORM.
fn title_format(resource: crate::d3d12x::object::This) -> DXGI_FORMAT {
    DXGI_FORMAT(d3d12x::resource::view_format(resource).unwrap_or(0) as i32)
}

/// A shader view of a typeless depth texture names the readable aspect.
/// For a 2D view D3D12 also requires the plane index to match that format.
fn texture_srv_desc(
    resource: &D3D12_RESOURCE_DESC,
    words: &[u32; 10],
) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    depth_aspect(resource, srv_desc(words))
}

/// The depth aspect of a shader view of a typeless depth texture.
fn depth_aspect(resource: &D3D12_RESOURCE_DESC, mut view: D3D12_SHADER_RESOURCE_VIEW_DESC) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    if !resource
        .Flags
        .contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
    {
        return view;
    }
    let (dsv, depth, stencil) = match resource.Format {
        DXGI_FORMAT_R32G8X24_TYPELESS => (
            DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
            DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
            Some(DXGI_FORMAT_X32_TYPELESS_G8X24_UINT),
        ),
        DXGI_FORMAT_R24G8_TYPELESS => (
            DXGI_FORMAT_D24_UNORM_S8_UINT,
            DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
            Some(DXGI_FORMAT_X24_TYPELESS_G8_UINT),
        ),
        DXGI_FORMAT_R32_TYPELESS => (DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R32_FLOAT, None),
        DXGI_FORMAT_R16_TYPELESS => (DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_R16_UNORM, None),
        _ => return view,
    };
    let plane = if Some(view.Format) == stencil {
        1
    } else if [DXGI_FORMAT_UNKNOWN, resource.Format, dsv, depth].contains(&view.Format) {
        view.Format = depth;
        0
    } else {
        return view;
    };
    match view.ViewDimension {
        D3D12_SRV_DIMENSION_TEXTURE2D => view.Anonymous.Texture2D.PlaneSlice = plane,
        D3D12_SRV_DIMENSION_TEXTURE2DARRAY => view.Anonymous.Texture2DArray.PlaneSlice = plane,
        _ => {}
    }
    view
}

/// Stand-in UAV scratch texture for a depth-stencil texture (PC strips
/// `ALLOW_UNORDERED_ACCESS` from depth targets; NVIDIA removes the device
/// if the shader then accesses a null descriptor in that slot).
fn depth_scratch(
    device: &ID3D12Device,
    texture: &ID3D12Resource,
    desc: Option<D3D12_UNORDERED_ACCESS_VIEW_DESC>,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    use windows::core::Interface;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    if d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 == 0 {
        return None;
    }
    let (res_fmt, uav_fmt) = match d.Format.0 as u32 {
        53 | 55 => (DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_UINT),
        _ => (DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_UINT),
    };
    static SCRATCH: std::sync::OnceLock<Mutex<std::collections::HashMap<usize, ID3D12Resource>>> =
        std::sync::OnceLock::new();
    let key = texture.as_raw() as usize;
    let scratch = {
        let mut map = SCRATCH
            .get_or_init(|| Mutex::new(std::collections::HashMap::new()))
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(existing) = map.get(&key) {
            existing.clone()
        } else {
            let mut sd = d;
            sd.Format = res_fmt;
            sd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let mut made: Option<ID3D12Resource> = None;
            // SAFETY: valid heap and resource description.
            let ok = unsafe {
                device.CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &sd,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    None,
                    &mut made,
                )
            };
            if let Err(e) = ok {
                once!("d3d12x: depth UAV scratch creation failed (format {}): {e}", d.Format.0);
                return None;
            }
            let res = made?;
            d3d12x::host::name(&res, || {
                format!("depth UAV scratch {}x{} f{}", d.Width, d.Height, res_fmt.0)
            });
            once!("d3d12x: depth UAV: scratch stand-in ({}x{} format {})", d.Width, d.Height, d.Format.0);
            map.insert(key, res.clone());
            res
        }
    };
    let mut uav = desc.unwrap_or_else(|| D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: uav_fmt,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    });
    uav.Format = uav_fmt;
    // Non-planar scratch formats (R16_TYPELESS / R32_TYPELESS) only allow PlaneSlice 0.
    match uav.ViewDimension {
        D3D12_UAV_DIMENSION_TEXTURE2D => uav.Anonymous.Texture2D.PlaneSlice = 0,
        D3D12_UAV_DIMENSION_TEXTURE2DARRAY => uav.Anonymous.Texture2DArray.PlaneSlice = 0,
        _ => {}
    }
    // SAFETY: a live scratch texture and a slot of our heap.
    unsafe {
        gate::uav(device, &scratch, None::<&ID3D12Resource>, Some(&uav), at);
    }
    Some(Some(scratch))
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    // The title resources the view names, held while their host view
    // is made (the title may release them meanwhile); a freed one: the
    // caller makes the slot null.
    let names = match *view {
        View::Shader(r, _) => [r, 0],
        View::Unordered(r, counter, _) => [r, counter],
        _ => [0, 0],
    };
    let mut _held = Vec::new();
    for r in names.into_iter().filter(|&r| r != 0) {
        let Some(held) = d3d12x::resource::hold_live(r as crate::d3d12x::object::This) else {
            d3d12x::resource::HELD_COUNTS[d3d12x::resource::HC_STALE_NULLED].fetch_add(1, Ordering::Relaxed);
            return None;
        };
        _held.push(held);
    }
    match view {
        &View::Constant(address, size) => {
            if address == 0 || size == 0 {
                return None;
            }
            let capped = size.min(65_536);
            if let Some(r) = view_in_place(view) {
                let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: shared_va(&r), SizeInBytes: capped.next_multiple_of(256) };
                // SAFETY: a slot of our heap; the range lies in the shared buffer.
                unsafe { gate::cbv(device, Some(&desc), at) };
                return Some(Some(r.buffer));
            }
            let shadow = d3d12x::shadow::find(address, u64::from(capped))?;
            let gpu = shadow.gpu(address);
            if gpu % 256 != 0 {
                once!("d3d12x: constant buffer {address:#x} not 256-byte aligned on the host");
                return None;
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: capped.next_multiple_of(256),
            };
            // SAFETY: a slot of our heap; the range lies in the host copy.
            unsafe { gate::cbv(device, Some(&desc), at) };
            Some(Some(shadow.buffer.clone()))
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::d3d12x::object::This;
            if let Some(texture) = d3d12x::resource::host_resource(resource) {
                // SAFETY: a live host texture.
                let info = unsafe { texture.GetDesc() };
                // A typeless host copy needs the view spelled out, in the
                // title's format where it named none.
                let typeless = d3d12x::resource::is_typeless(info.Format.0 as u32);
                let mut desc = match desc {
                    Some(words) if typeless && words[0] == 0 => {
                        let mut words = *words;
                        words[0] = title_format(resource).0 as u32;
                        Some(texture_srv_desc(&info, &words))
                    }
                    Some(words) => Some(texture_srv_desc(&info, words)),
                    None if typeless => Some(depth_aspect(&info, default_srv_desc(&info, title_format(resource)))),
                    None => None,
                };
                if !fits(&info, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // A host copy without its top mips (`XNW_MIP_DROP`).
                if let Some(d) = desc.as_mut() {
                    d3d12x::texture::shift_view_mips(resource, &info, d);
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    gate::srv(device,
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some(texture.clone()))
            } else if let (Some((data, len)), Some(desc)) =
                (d3d12x::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::d3d12x::object::This;
            let Some(texture) = d3d12x::resource::host_resource(resource) else {
                return match (d3d12x::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            let mut desc = desc.as_ref().map(uav_desc);
            // SAFETY: a live host texture.
            let info = unsafe { texture.GetDesc() };
            // As for shader views above.
            if d3d12x::resource::is_typeless(info.Format.0 as u32) {
                let d = desc.get_or_insert_with(|| default_uav_desc(&info, DXGI_FORMAT(0)));
                if d.Format.0 == 0 {
                    d.Format = title_format(resource);
                }
            }
            if !fits(&info, desc.as_mut().map(|d| &mut d.Format), true) {
                return depth_scratch(device, texture, desc, at);
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                gate::uav(device,
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some(texture.clone()))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { gate::sampler(device, &desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::d3d12x::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { gate::sampler(device, &desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some((view, _held)) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // shortcut: unordered access to a texture the title placed
                // itself is not mirrored; give it a resource as shader
                // views of such textures get when a dispatch writes one.
                once!("d3d12x: placed unordered texture view not mirrored (dimension {})", resource[0]);
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// A shader or unordered access view of a texture the title placed in memory
/// itself, as a view of the resource made for that memory (on first use),
/// together with a held reference so concurrent eviction cannot free the
/// resource while in use.
// ---- GPU-side descriptors (`XNW_DESC=gpu`) ----------------------------
//
// Record layout:
// dword 0 of each record the device writes is the host heap index of the
// view in one global host heap, so the title's byte copies carry it and a
// translated shader indexes `ResourceDescriptorHeap` with it after reading
// the record in place. Slots [0, TABLE_NULLS) hold one null view per record
// kind; slot TABLE_NULLS + entry holds the view of entry `entry` of the
// views table, made when the view is created and nulled when it goes
// stale. Samplers: their own global heap, one slot per distinct sampler.
// Until the translator reads records, draws bind the per-title-heap
// host heaps as before; the tables are only kept with `XNW_DESC=gpu`.

/// Host slots before the view table: the null view of each record kind.
pub(crate) const TABLE_NULLS: u32 = 16;
/// View-table entries the GPU-mode heap holds; a view entry past it gets
/// the null of its kind as its index.
pub(crate) const VIEW_TABLE: u32 = 1 << 20;

/// Whether descriptors are kept for the GPU to read (`XNW_DESC=gpu`).
pub(crate) fn gpu_mode() -> bool {
    xnw_shader::emit::gpu_descriptors()
}

/// The host heap index a view record carries for view entry `i` of kind
/// word `kind`.
pub(crate) fn table_index(i: u32, kind: u32) -> u32 {
    if i < VIEW_TABLE { TABLE_NULLS + i } else { kind & 0xff }
}

/// The GPU-mode heaps (views, samplers), made on first use; and the view
/// slots waiting for a placed texture, by slot.
struct Tables {
    views: Option<Arc<HostHeap>>,
    samplers: Option<Arc<HostHeap>>,
    waiting: Mutex<BTreeMap<u32, View>>,
    /// Window raw views made, by window number.
    windows: Mutex<BTreeSet<u32>>,
}

static TABLES: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();

fn tables() -> Option<&'static Tables> {
    if !gpu_mode() {
        return None;
    }
    Some(TABLES.get_or_init(|| {
        let views = HostHeap::new(false, TABLE_NULLS + VIEW_TABLE + WINDOW_TABLE, 0).map(Arc::new);
        if let Some(h) = &views {
            for kind in 0..TABLE_NULLS {
                h.null_of_kind(kind, kind);
            }
            xnw_core::info!("d3d12x: GPU-side descriptors: view table of {VIEW_TABLE} (+{TABLE_NULLS} nulls), samplers {}", super::SAMPLER_TABLE);
        }
        let samplers = HostHeap::new(true, super::SAMPLER_TABLE, 0).map(Arc::new);
        Tables { views, samplers, waiting: Mutex::default(), windows: Mutex::default() }
    }))
}

/// The GPU-mode heaps to bind (views, samplers) for draws whose shaders
/// read descriptor records themselves; `None` unless `XNW_DESC=gpu`.
pub(crate) fn gpu_heaps() -> Option<(Arc<HostHeap>, Arc<HostHeap>)> {
    let t = tables()?;
    Some((t.views.clone()?, t.samplers.clone()?))
}

/// The view at record index `index` was created: its table slot is made
/// (a placed texture not made yet: a null of its kind until it is).
pub(crate) fn table_made(index: u32, view: &View) {
    let Some(t) = tables() else { return };
    if matches!(view, View::Sampler(_) | View::ConsoleSampler(_)) {
        if let Some(s) = &t.samplers {
            s.make_at(index, view);
        }
        return;
    }
    let Some(h) = &t.views else { return };
    if index < TABLE_NULLS {
        return;
    }
    // XNW: a constant buffer is read through a raw view of the shared
    // window at its own address: the console aligns constants to 16 bytes,
    // a D3D12 CBV wants 256, a raw view 16.
    if let &View::Constant(address, size) = view {
        h.raw_constants_at(index, address, size);
        return;
    }
    if !placed_ready(view) {
        h.null_of_kind(index, super::kind_word(view));
        t.waiting.lock().unwrap_or_else(|e| e.into_inner()).insert(index, view.clone());
        return;
    }
    h.make_at(index, view);
}

/// Raw views of the shared windows (a window or region heap's buffer) after
/// the view table: slot `TABLE_NULLS + VIEW_TABLE + window number`.
pub(crate) const WINDOW_TABLE: u32 = 4096;

/// For a view read in place, dwords 5-7 of its record: the host heap index
/// of a raw view over its shared window, the view's first byte in it, its
/// bytes; the raw view is made on first use. `None` unless `XNW_DESC=gpu`
/// and the view resolves in place.
pub(crate) fn table_in_place(view: &View) -> Option<[u32; 3]> {
    let t = tables()?;
    let h = t.views.as_ref()?;
    let r = view_in_place(view)?;
    let index = window_view(t, h, &r.buffer)?;
    Some([index, r.offset as u32, r.len as u32])
}

/// XNW: for title memory `[address, address + len)` read in place,
/// the host heap index of its window's raw view and the byte offset in it
/// (what a shader reading a descriptor table there needs); `None` unless
/// `XNW_DESC=gpu` and the range resolves in a shared window.
pub(crate) fn window_of(address: u64, len: u64) -> Option<[u32; 2]> {
    let t = tables()?;
    let h = t.views.as_ref()?;
    for &(start, bytes, index) in OWN_WINDOWS.lock().unwrap_or_else(|e| e.into_inner()).iter() {
        if address >= start && address + len <= start + bytes {
            return Some([index, (address - start) as u32]);
        }
    }
    let r = crate::shared::lookup(address, len)?;
    let index = window_view(t, h, &r.buffer)?;
    Some([index, u32::try_from(r.offset).ok()?])
}

/// XNW: our own shader-visible descriptor heaps' memory in gpu mode:
/// (address, bytes, global heap index of its raw view), from the top of the
/// window slots down.
static OWN_WINDOWS: Mutex<Vec<(u64, u64, u32)>> = Mutex::new(Vec::new());

/// XNW: `bytes` of zeroed memory the CPU reads and writes cached and
/// the GPU reads in place (custom heap, write-back, L0), with a raw view of
/// it in the global view heap so shaders read descriptor records there;
/// `None` unless `XNW_DESC=gpu`. Returns its address and the buffer.
pub(crate) fn own_window(bytes: u64) -> Option<(u64, ID3D12Resource)> {
    let t = tables()?;
    let h = t.views.as_ref()?;
    let host = host()?;
    let bytes = bytes.next_multiple_of(256);
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_CUSTOM,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
        MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
        ..Default::default()
    };
    let mut resource: Option<ID3D12Resource> = None;
    let mut p = std::ptr::null_mut();
    // SAFETY: plain buffer creation and a persistent map.
    unsafe {
        host.device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut resource).ok()?;
        resource.as_ref()?.Map(0, None, Some(&mut p)).ok()?;
        std::ptr::write_bytes(p.cast::<u8>(), 0, bytes as usize);
    }
    let r = resource?;
    let mut own = OWN_WINDOWS.lock().unwrap_or_else(|e| e.into_inner());
    let index = (TABLE_NULLS + VIEW_TABLE + WINDOW_TABLE).checked_sub(1 + own.len() as u32)?;
    let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV { FirstElement: 0, NumElements: (bytes / 4) as u32, StructureByteStride: 0, Flags: D3D12_BUFFER_SRV_FLAG_RAW },
        },
    };
    // SAFETY: a live buffer and a slot of our heap.
    unsafe { gate::srv(&host.device, &r, Some(&srv), h.cpu(index)) };
    h.keep(index, Some(r.clone()));
    own.push((p as u64, bytes, index));
    xnw_core::info!("d3d12x: GPU-side descriptors: heap memory of {bytes} bytes at {:#x} read in place (slot {index})", p as u64);
    Some((p as u64, r))
}

/// The heap index of the raw view over shared window `buffer`, made on
/// first use.
fn window_view(t: &Tables, h: &HostHeap, buffer: &ID3D12Resource) -> Option<u32> {
    let id = crate::shared::window_id(buffer);
    if id >= WINDOW_TABLE {
        return None;
    }
    let index = TABLE_NULLS + VIEW_TABLE + id;
    if t.windows.lock().unwrap_or_else(|e| e.into_inner()).insert(id) {
        let r = buffer;
        // SAFETY: a live buffer.
        let bytes = unsafe { r.GetDesc() }.Width;
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV { FirstElement: 0, NumElements: (bytes / 4) as u32, StructureByteStride: 0, Flags: D3D12_BUFFER_SRV_FLAG_RAW },
            },
        };
        if let Some(host) = host() {
            // SAFETY: a live buffer and a slot of our heap.
            unsafe { gate::srv(&host.device, r, Some(&desc), h.cpu(index)) };
        }
        h.keep(index, Some(r.clone()));
    }
    Some(index)
}
/// Table slot `index` (kind word `kind`) no longer has a view.
pub(crate) fn table_null(index: u32, kind: u32) {
    if let Some(t) = tables()
        && let Some(h) = &t.views
        && index >= TABLE_NULLS
    {
        t.waiting.lock().unwrap_or_else(|e| e.into_inner()).remove(&index);
        h.null_of_kind(index, kind);
    }
}

/// XNW: what global view slot `index` holds, for traces.
pub(crate) fn table_slot_text(index: u32) -> String {
    let Some(t) = tables() else { return "no tables".into() };
    let Some(h) = &t.views else { return "no view heap".into() };
    let kept = h.kept.lock().unwrap_or_else(|e| e.into_inner()).get(index as usize).cloned();
    let what = match kept {
        Some(Some(r)) => {
            // SAFETY: a live resource.
            let d = unsafe { r.GetDesc() };
            format!("{}x{}x{} format {} mips {}", d.Width, d.Height, d.DepthOrArraySize, d.Format.0, d.MipLevels)
        }
        Some(None) => "no resource".into(),
        None => "past the heap".into(),
    };
    let waiting = t.waiting.lock().unwrap_or_else(|e| e.into_inner()).contains_key(&index);
    format!("slot {index}: {what}, waiting {waiting}")
}

/// XNW: global view slot `index` (from a record a draw reads) made
/// from `view` again if it holds no host resource: a view created before
/// its host texture existed was made null. Returns whether it was remade.
pub(crate) fn table_refresh(index: u32, view: &View) -> bool {
    let Some(t) = tables() else { return false };
    let Some(h) = &t.views else { return false };
    if !(TABLE_NULLS..TABLE_NULLS + VIEW_TABLE).contains(&index) || matches!(view, View::Sampler(_) | View::ConsoleSampler(_)) {
        return false;
    }
    let empty = h.kept.lock().unwrap_or_else(|e| e.into_inner()).get(index as usize).is_some_and(Option::is_none);
    if let &View::Constant(address, size) = view {
        if empty {
            h.raw_constants_at(index, address, size);
        }
        return empty;
    }
    if !placed_ready(view) {
        return false;
    }
    // The host resource the view reads now (a placed texture's, or the
    // title resource's host copy, which the texture code may replace).
    let now: Option<ID3D12Resource> = match view {
        View::Placed(..) => placed_texture(view).and_then(|(v, _)| match v {
            View::Shader(r, _) | View::Unordered(r, _, _) => d3d12x::resource::host_resource(r as crate::d3d12x::object::This).cloned(),
            _ => None,
        }),
        // Held while read (the title may release it meanwhile).
        View::Shader(r, _) | View::Unordered(r, _, _) => d3d12x::resource::hold_live(*r as crate::d3d12x::object::This)
            .and_then(|_held| d3d12x::resource::host_resource(*r as crate::d3d12x::object::This).cloned()),
        _ => None,
    };
    let same = {
        let kept = h.kept.lock().unwrap_or_else(|e| e.into_inner());
        match (kept.get(index as usize), &now) {
            (Some(Some(k)), Some(n)) => windows::core::Interface::as_raw(k) == windows::core::Interface::as_raw(n),
            (Some(Some(_)), None) => true,
            _ => false,
        }
    };
    if same && !empty {
        return false;
    }
    t.waiting.lock().unwrap_or_else(|e| e.into_inner()).remove(&index);
    h.make_at(index, view);
    true
}

/// Table slots that waited for placed textures the texture thread has made
/// since: made now.
pub(crate) fn table_catch_up() {
    let Some(t) = tables() else { return };
    let Some(h) = &t.views else { return };
    let ready: Vec<(u32, View)> = {
        let w = t.waiting.lock().unwrap_or_else(|e| e.into_inner());
        w.iter().filter(|(_, v)| placed_ready(v)).map(|(&i, v)| (i, v.clone())).collect()
    };
    for (i, v) in ready {
        h.make_at(i, &v);
        t.waiting.lock().unwrap_or_else(|e| e.into_inner()).remove(&i);
    }
}

/// XNW: constant records made as raw views over a window, and those
/// left null (not in a window, or misaligned).
pub(crate) static RAW_CONSTANTS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

impl HostHeap {
    /// XNW: slot `index` a raw view of the shared window at the
    /// constants' `[address, address + size)`, else the null raw view.
    fn raw_constants_at(&self, index: u32, address: u64, size: u32) {
        let at = self.cpu(index);
        let made = crate::shared::lookup(address, u64::from(size.max(16)))
            .filter(|r| r.offset.is_multiple_of(16))
            .and_then(|r| {
                let host = host()?;
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            FirstElement: r.offset / 4,
                            NumElements: size.max(16).div_ceil(4),
                            StructureByteStride: 0,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        },
                    },
                };
                // SAFETY: a live window buffer and a slot of our heap.
                unsafe { gate::srv(&host.device, &r.buffer, Some(&desc), at) };
                Some(r.buffer)
            });
        RAW_CONSTANTS[usize::from(made.is_none())].fetch_add(1, Ordering::Relaxed);
        if made.is_none() {
            null_buffer_view(at);
        }
        self.keep(index, made);
    }

    /// Makes slot `index` the null view of record kind word `kind`.
    fn null_of_kind(&self, index: u32, kind: u32) {
        use super::{KIND_CBV, KIND_SAMPLER, KIND_SRV_RAW, KIND_SRV_STRUCTURED, KIND_SRV_TYPED, KIND_UAV_RAW, KIND_UAV_STRUCTURED, KIND_UAV_TEXTURE, KIND_UAV_TYPED};
        let at = self.cpu(index);
        match kind & 0xff {
            KIND_CBV => {
                if let Some(host) = host() {
                    // SAFETY: a slot of our heap; a zero location is a null CBV.
                    unsafe { gate::cbv(&host.device, Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC::default()), at) };
                }
            }
            0 | KIND_SRV_RAW | KIND_SRV_TYPED | KIND_SRV_STRUCTURED => null_buffer_view(at),
            KIND_UAV_RAW | KIND_UAV_TYPED | KIND_UAV_STRUCTURED => null_uav_buffer_view(at, 0),
            KIND_UAV_TEXTURE => null_uav_view(at),
            KIND_SAMPLER if self.sampler => self.null_sampler(index),
            _ => null_view(at),
        }
        self.keep(index, None);
    }
}

/// Placed textures queued for making off the submission and recording
/// threads, the keys queued, and textures made so far (heaps waiting
/// on one look again when it moves).
///
/// Making a host texture (CreateCommittedResource) inside plan making cost
/// 120 of its 241 ms a frame. A slot whose
/// texture is not made yet reads a null view until it is: a frame or so,
/// as streaming pops in. `XNW_PLACED_SYNC=1` makes them where needed, as
/// before; tests do.
static TEXTURE_QUEUE: (Mutex<(VecDeque<(u64, [u32; 14])>, BTreeSet<(u64, [u32; 14])>)>, std::sync::Condvar) =
    (Mutex::new((VecDeque::new(), BTreeSet::new())), std::sync::Condvar::new());
static TEXTURES_READY: AtomicU64 = AtomicU64::new(0);

/// Whether the placed texture `view` names exists (or is to be made where
/// needed); if not, it is queued for the texture thread.
fn placed_ready(view: &View) -> bool {
    static SYNC: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| cfg!(test) || std::env::var("XNW_PLACED_SYNC").is_ok_and(|v| v == "1"));
    let View::Placed(_, resource, desc) = view else { return true };
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    if *SYNC || resource[0] == 1 || d3d12x::resource::has_placed_texture(address, resource) {
        return true;
    }
    let (queue, wake) = &TEXTURE_QUEUE;
    let mut q = queue.lock().unwrap_or_else(|e| e.into_inner());
    if q.1.insert((address, *resource)) {
        q.0.push_back((address, *resource));
        count(TEXTURES_QUEUED, 1);
        wake.notify_one();
    }
    drop(q);
    static THREAD: std::sync::Once = std::sync::Once::new();
    THREAD.call_once(|| {
        let _ = std::thread::Builder::new().name("xnw placed textures".into()).spawn(|| {
            let (queue, wake) = &TEXTURE_QUEUE;
            loop {
                let (address, desc) = {
                    let mut q = queue.lock().unwrap_or_else(|e| e.into_inner());
                    loop {
                        if let Some(next) = q.0.pop_front() {
                            break next;
                        }
                        q = wake.wait(q).unwrap_or_else(|e| e.into_inner());
                    }
                };
                // The cache keeps its own reference; ours goes at once.
                drop(d3d12x::resource::texture_at(address, &desc));
                count(TEXTURES_MADE_OFF, 1);
                queue.lock().unwrap_or_else(|e| e.into_inner()).1.remove(&(address, desc));
                TEXTURES_READY.fetch_add(1, Ordering::Release);
            }
        });
    });
    false
}

pub fn placed_texture(view: &View) -> Option<(View, d3d12x::resource::HeldTexture)> {
    const BUFFER: u32 = 1;
    let (unordered, resource, desc) = match view {
        View::Placed(unordered, resource, desc) => (*unordered, resource, desc),
        _ => return None,
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = d3d12x::resource::texture_at(address, resource)?;
    let this = texture.as_this() as usize;
    let view = if unordered {
        View::Unordered(this, 0, desc[..10].try_into().ok())
    } else {
        View::Shader(this, desc[..10].try_into().ok())
    };
    Some((view, texture))
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or texture of matching dimension)
/// or a shader view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    // A freed resource is read only while held.
    let is_buffer = |r: &usize| d3d12x::resource::hold_live(*r as crate::d3d12x::object::This).is_some_and(|_held| d3d12x::resource::buffer_bytes(*r as crate::d3d12x::object::This).is_some());
    match view {
        View::Unordered(r, _, Some(desc)) => {
            if is_buffer(r) || desc[1] == 1 {
                return null_uav_buffer_view(at, 0);
            }
            null_uav_like(desc, at);
        }
        View::Unordered(r, _, None) => {
            if is_buffer(r) {
                null_uav_buffer_view(at, 0);
            } else {
                null_uav_view(at);
            }
        }
        View::Placed(true, resource, desc) => {
            if resource[0] == 1 {
                null_uav_buffer_view(at, 0);
            } else {
                let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
                null_uav_like(words, at);
            }
        }
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => null_buffer_view(at),
        View::Shader(_, Some(desc)) => null_like(desc, at),
        View::Placed(false, _, desc) => {
            null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::Constant(..) => {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of our heap; a zero location makes a null view.
            unsafe { gate::cbv(&host.device, Some(&desc), at) };
        }
        _ => null_view(at),
    }
}

/// The dimension of the null UAV for a title view description's dimension
/// word (the host's `D3D12_UAV_DIMENSION` values: 3D is 8, 6 and 7 are
/// multisampled). Multisampled UAVs need a capability, and unknown words
/// name no dimension: those get a 2D null view, as before.
fn null_uav_dimension(word: u32) -> D3D12_UAV_DIMENSION {
    match D3D12_UAV_DIMENSION(word as i32) {
        d @ (D3D12_UAV_DIMENSION_BUFFER
        | D3D12_UAV_DIMENSION_TEXTURE1D
        | D3D12_UAV_DIMENSION_TEXTURE1DARRAY
        | D3D12_UAV_DIMENSION_TEXTURE2DARRAY
        | D3D12_UAV_DIMENSION_TEXTURE3D) => d,
        _ => D3D12_UAV_DIMENSION_TEXTURE2D,
    }
}

/// An unordered access view that safely drops writes and reads zeros, matching
/// the title's dimension from `words` (1D, 2D, arrays, 3D, buffer).
fn null_uav_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    if null_srv() {
        NULL_UAV_AS_SRV.fetch_add(1, Ordering::Relaxed);
        return null_like(words, at);
    }
    let Some(host) = host() else { return };
    let dim = null_uav_dimension(words[1]);
    let desc = match dim {
        D3D12_UAV_DIMENSION_BUFFER => return null_uav_buffer_view(at, 0),
        D3D12_UAV_DIMENSION_TEXTURE1D => D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE1D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture1D: D3D12_TEX1D_UAV { MipSlice: 0 },
            },
        },
        D3D12_UAV_DIMENSION_TEXTURE1DARRAY => D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE1DARRAY,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture1DArray: D3D12_TEX1D_ARRAY_UAV {
                    MipSlice: 0,
                    FirstArraySlice: 0,
                    ArraySize: 1,
                },
            },
        },
        D3D12_UAV_DIMENSION_TEXTURE2DARRAY => D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_UAV {
                    MipSlice: 0,
                    FirstArraySlice: 0,
                    ArraySize: 1,
                    PlaneSlice: 0,
                },
            },
        },
        D3D12_UAV_DIMENSION_TEXTURE3D => D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE3D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture3D: D3D12_TEX3D_UAV {
                    MipSlice: 0,
                    FirstWSlice: 0,
                    // All slices, as VIctory main (cbb50b8) makes it.
                    WSize: u32::MAX,
                },
            },
        },
        _ => D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_UAV {
                    MipSlice: 0,
                    PlaneSlice: 0,
                },
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        gate::uav(&host.device,
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// `XNW_NULL_SRV=1`: every null view we make for a view or unordered-access
/// slot is a shader-resource null. WARP 1.0.22 removes the device when a
/// shader reads a null unordered-access view (or a constant buffer view, or
/// a slot never written) as a resource, while a shader-resource null of any
/// dimension reads zeros and drops stores and atomics under every view kind
/// (`xnw-shader` test `null_kinds`, 13 slot kinds x 10 reads on
/// another machine). The title reads one descriptor under several kinds, as
/// the console's descriptors have one format. Off by default.
pub(crate) fn null_srv() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XNW_NULL_SRV").is_ok_and(|v| matches!(v.trim(), "1" | "dummy"));
        if on {
            xnw_core::info!("d3d12x: null views are shader-resource nulls (XNW_NULL_SRV=1)");
        }
        on
    })
}

/// `XNW_NULL_SRV=dummy`: every null shader-resource view (and so every null
/// unordered-access view, [`null_srv`]) is instead a raw view of a zeroed
/// 64 KiB buffer of ours. WARP reads a resource through two fields of a
/// descriptor (the JIT code at a run's fault: a flag behind
/// the fourth qword, then a byte at +0x388 behind the second); a slot
/// rewritten from a live view to a null one while a list reading it runs
/// can be read half old, half new: the live flag, then the null's zero
/// pointer, and four WARP threads fault reading 0x388. A live view over
/// zeros has no null pointer to read. What a shader writes through such a
/// slot lands in the dummy buffer.
fn null_dummy() -> Option<&'static ID3D12Resource> {
    struct Dummy(Option<ID3D12Resource>);
    // SAFETY: D3D12 resources are free-threaded.
    unsafe impl Send for Dummy {}
    // SAFETY: as above.
    unsafe impl Sync for Dummy {}
    static DUMMY: std::sync::OnceLock<Dummy> = std::sync::OnceLock::new();
    DUMMY
        .get_or_init(|| {
            let on = std::env::var("XNW_NULL_SRV").is_ok_and(|v| v.trim() == "dummy");
            let Some(host) = host().filter(|_| on) else { return Dummy(None) };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: NULL_DUMMY_BYTES,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut r: Option<ID3D12Resource> = None;
            // SAFETY: a plain committed buffer (default heaps start zeroed).
            let made = unsafe { host.device.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r) };
            match made {
                Ok(()) => xnw_core::info!("d3d12x: null views are views of a zeroed {NULL_DUMMY_BYTES}-byte buffer (XNW_NULL_SRV=dummy)"),
                Err(e) => xnw_core::warn!("d3d12x: XNW_NULL_SRV=dummy: no dummy buffer ({e}); null views stay null"),
            }
            Dummy(r)
        })
        .0
        .as_ref()
}

/// Bytes of the [`null_dummy`] buffer.
const NULL_DUMMY_BYTES: u64 = 1 << 16;

/// Writes the [`null_dummy`] view at `at`; false without the dummy.
fn dummy_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) -> bool {
    let (Some(host), Some(dummy)) = (host(), null_dummy()) else { return false };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV { NumElements: (NULL_DUMMY_BYTES / 4) as u32, Flags: D3D12_BUFFER_SRV_FLAG_RAW, ..Default::default() },
        },
    };
    // SAFETY: a live buffer of ours and a slot of our heap.
    unsafe { gate::srv(&host.device, dummy, Some(&desc), at) };
    true
}

/// Null unordered-access views made as shader-resource nulls ([`null_srv`]).
pub(crate) static NULL_UAV_AS_SRV: AtomicU64 = AtomicU64::new(0);

/// A buffer UAV that safely drops writes (raw, or structured if `stride` > 0).
fn null_uav_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE, stride: u32) {
    if null_srv() {
        NULL_UAV_AS_SRV.fetch_add(1, Ordering::Relaxed);
        return null_buffer_view(at);
    }
    let Some(host) = host() else { return };
    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: if stride > 0 {
            DXGI_FORMAT_UNKNOWN
        } else {
            DXGI_FORMAT_R32_TYPELESS
        },
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                NumElements: 1,
                StructureByteStride: stride,
                Flags: if stride > 0 {
                    D3D12_BUFFER_UAV_FLAG_NONE
                } else {
                    D3D12_BUFFER_UAV_FLAG_RAW
                },
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        gate::uav(&host.device,
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A 2D texture UAV that safely drops writes.
fn null_uav_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    if null_srv() {
        NULL_UAV_AS_SRV.fetch_add(1, Ordering::Relaxed);
        return null_view(at);
    }
    let Some(host) = host() else { return };
    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_UAV {
                MipSlice: 0,
                PlaneSlice: 0,
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        gate::uav(&host.device,
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    if dummy_view(at) {
        return;
    }
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        gate::srv(&host.device,
            None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    if dummy_view(at) {
        return;
    }
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        gate::srv(&host.device,
            None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    if dummy_view(at) {
        return;
    }
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        gate::srv(&host.device,
            None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::d3d12x::descriptor::{
        HeapDesc, copy_simple, create_cbv, create_heap, create_placed_srv, create_srv, create_uav,
        heap,
    };
    use crate::d3d12x::object::{Iid, S_OK, This, guid};

    /// The base's safe `slot` (ours is `unsafe`): tests call our objects only.
    fn slot<F: Copy>(object: This, offset: usize) -> F {
        // SAFETY: tests pass objects of ours with the slot's signature.
        unsafe { crate::d3d12x::object::slot(object, offset) }
    }
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::core::s;

    const HEAP_IID: Iid = guid(
        0x8efb471d,
        0x616c,
        0x4f49,
        [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
    );
    const RESOURCE_IID: Iid = guid(
        0x696442be,
        0xa72e,
        0x4059,
        [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
    );

    #[repr(C)]
    struct ExtendedHeapDesc {
        desc: HeapDesc,
        addr: u64,
        size: u64,
    }

    pub(crate) fn make_shader_heap(count: u32) -> (This, u64) {
        let ext = ExtendedHeapDesc {
            desc: HeapDesc {
                kind: 0,
                count,
                flags: 1,
                nodes: 0,
            },
            addr: 0,
            size: 0,
        };
        let mut out = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &ext.desc, &HEAP_IID, &mut out),
            S_OK
        );
        let start = slot::<extern "system" fn(This) -> u64>(out, 0x48)(out);
        (out, start)
    }

    pub(crate) fn make_title_buffer(address: u64, size: u64, uav: bool) -> This {
        let words: [u32; 14] = [
            1, // BUFFER
            0,
            0,
            0,
            size as u32,
            (size >> 32) as u32,
            1,
            0x0001_0001,
            0,
            1,
            0,
            1, // ROW_MAJOR
            if uav { 4 } else { 0 },
            0,
        ];
        let mut out = std::ptr::null();
        assert_eq!(
            d3d12x::resource::create_at_address(
                std::ptr::null(),
                address,
                words.as_ptr().cast(),
                0,
                std::ptr::null(),
                &RESOURCE_IID,
                &mut out,
            ),
            S_OK
        );
        out
    }

    fn compile_probe_cs(device: &ID3D12Device) -> (ID3D12RootSignature, ID3D12PipelineState) {
        const PROBE_HLSL: &str = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=1)), \
            DescriptorTable(CBV(b0, numDescriptors=1)), \
            DescriptorTable(SRV(t1, numDescriptors=1)), \
            DescriptorTable(UAV(u0, numDescriptors=1)), \
            DescriptorTable(SRV(t2, numDescriptors=1)), \
            DescriptorTable(CBV(b1, numDescriptors=1))"
ByteAddressBuffer srv_buf : register(t0);
cbuffer Cbv16 : register(b0) { uint4 cb_vals; };
Texture2D<float4> placed_tex : register(t1);
RWByteAddressBuffer uav_buf : register(u0);
ByteAddressBuffer copied_srv : register(t2);
cbuffer NullCbv : register(b1) { uint4 null_vals; };

[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint v_srv = srv_buf.Load(0);
    uint v_cbv = cb_vals.x + cb_vals.y + cb_vals.z + cb_vals.w;
    float4 px = placed_tex.Load(int3(0, 0, 0));
    uint v_tex = (uint(round(px.x * 255.0)) << 0)
               | (uint(round(px.y * 255.0)) << 8)
               | (uint(round(px.z * 255.0)) << 16)
               | (uint(round(px.w * 255.0)) << 24);
    uint v_copied = copied_srv.Load(0);
    uint v_null = null_vals.x;
    uav_buf.Store(0, v_srv);
    uav_buf.Store(4, v_cbv);
    uav_buf.Store(8, v_tex);
    uav_buf.Store(12, v_copied);
    uav_buf.Store(16, v_null);
}
"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: static source and entry point strings; out pointer ours.
        unsafe {
            D3DCompile(
                PROBE_HLSL.as_ptr().cast(),
                PROBE_HLSL.len(),
                s!("bindless_probe"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
            .unwrap();
            let code = code.unwrap();
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).unwrap();
            let pso = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        }
    }

    #[test]
    fn depth_srv_uses_the_requested_aspect() {
        for (own, request, expected, plane) in [
            (19, 20, 21, 0),
            (19, 22, 22, 1),
            (44, 45, 46, 0),
            (44, 47, 47, 1),
            (44, 120, 47, 1),
            (44, 119, 46, 0),
            (39, 40, 41, 0),
            (53, 55, 56, 0),
            (19, 0, 21, 0),
            (44, 44, 46, 0),
        ] {
            let resource = D3D12_RESOURCE_DESC {
                Format: DXGI_FORMAT(own),
                Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                ..Default::default()
            };
            for dim in [
                D3D12_SRV_DIMENSION_TEXTURE2D,
                D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            ] {
                let mut words = [0; 10];
                words[0] = request;
                words[1] = dim.0 as u32;
                words[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                words[4] = 2; // MostDetailedMip
                words[5] = 3; // MipLevels
                if dim == D3D12_SRV_DIMENSION_TEXTURE2DARRAY {
                    words[6] = 1; // FirstArraySlice
                    words[7] = 2; // ArraySize
                }
                let plane_word = if dim == D3D12_SRV_DIMENSION_TEXTURE2DARRAY {
                    8
                } else {
                    6
                };
                words[plane_word] = 1 - plane;
                let view = texture_srv_desc(&resource, &words);
                assert_eq!(view.Format.0, expected);
                // SAFETY: read the member selected by the input view dimension.
                unsafe {
                    if dim == D3D12_SRV_DIMENSION_TEXTURE2DARRAY {
                        let a = view.Anonymous.Texture2DArray;
                        assert_eq!(
                            (
                                a.MostDetailedMip,
                                a.MipLevels,
                                a.FirstArraySlice,
                                a.ArraySize,
                                a.PlaneSlice
                            ),
                            (2, 3, 1, 2, plane)
                        );
                    } else {
                        let a = view.Anonymous.Texture2D;
                        assert_eq!(
                            (a.MostDetailedMip, a.MipLevels, a.PlaneSlice),
                            (2, 3, plane)
                        );
                    }
                }
            }
        }
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4, 0) else {
            return; // no host GPU
        };
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        // The last dynamic slot is the sink for slots past the heap.
        assert_eq!(heap.dynamic(39), Some(4 + heap.dynamic - 40));
    }

    /// A slot past the heap is never written: counted, and the
    /// sink slot (never handed out) takes the write.
    #[test]
    fn slots_past_the_heap_are_not_written() {
        let Some(heap) = HostHeap::new(false, 4, 0) else {
            return; // no host GPU
        };
        let before = SLOTS_OUT.load(Ordering::Relaxed);
        let past = heap.count + heap.dynamic + 1000;
        heap.make_null(past);
        heap.make_null_uav(past, false, 0);
        heap.make_null_kind(u32::MAX, xnw_shader::ResourceType::Cbv, 4);
        assert!(SLOTS_OUT.load(Ordering::Relaxed) >= before + 3);
        assert_eq!(heap.cpu(past).ptr, heap.cpu(heap.count + heap.dynamic - 1).ptr, "the sink");
        assert_eq!(heap.cpu(3).ptr, heap.cpu + 3 * heap.step as usize);
    }

    /// Null UAV descriptors for textures and raw/structured buffers can be made on host heaps.
    #[test]
    fn null_uav_descriptors_created_on_host_heap() {
        let Some(heap) = HostHeap::new(false, 4, 0) else {
            return; // no host GPU
        };
        // Texture UAV null descriptor
        heap.make_null_uav(0, false, 0);
        // Raw buffer UAV null descriptor
        heap.make_null_uav(1, true, 0);
        // Structured buffer UAV null descriptor (stride 16)
        heap.make_null_uav(2, true, 16);
    }

    /// null_for creates valid null descriptors matching UAV dimensions without faulting.
    #[test]
    fn null_for_creates_uav_matching_dimensions() {
        let Some(heap) = HostHeap::new(false, 4, 0) else {
            return; // no host GPU
        };
        // Unordered access with no desc (fallback 2D)
        null_for(&View::Unordered(0, 0, None), heap.cpu(0));
        // Unordered access buffer
        null_for(
            &View::Unordered(0, 0, Some([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])),
            heap.cpu(1),
        );
        // Unordered access Texture3D (dim 8)
        null_for(
            &View::Unordered(0, 0, Some([0, 8, 0, 0, 0, 0, 0, 0, 0, 0])),
            heap.cpu(2),
        );
        // Placed UAV on a buffer (resource[0] == 1)
        null_for(&View::Placed(true, [1; 14], [0; 16]), heap.cpu(3));
    }

    #[test]
    fn null_uavs_follow_the_hosts_dimension_numbers() {
        let cases = [
            (1, D3D12_UAV_DIMENSION_BUFFER),
            (2, D3D12_UAV_DIMENSION_TEXTURE1D),
            (3, D3D12_UAV_DIMENSION_TEXTURE1DARRAY),
            (4, D3D12_UAV_DIMENSION_TEXTURE2D),
            (5, D3D12_UAV_DIMENSION_TEXTURE2DARRAY),
            (6, D3D12_UAV_DIMENSION_TEXTURE2D),
            (7, D3D12_UAV_DIMENSION_TEXTURE2D),
            (8, D3D12_UAV_DIMENSION_TEXTURE3D),
            (0, D3D12_UAV_DIMENSION_TEXTURE2D),
            (99, D3D12_UAV_DIMENSION_TEXTURE2D),
        ];
        for (word, want) in cases {
            assert_eq!(null_uav_dimension(word), want, "word {word}");
        }
    }

    /// End-to-end GPU execution test proving that buffer SRVs, 16-byte-aligned
    /// CBVs, copied buffer SRVs, `View::Placed` streamed textures, and buffer
    /// UAVs in a shader-visible heap materialize non-null host descriptors,
    /// refresh modified title memory across submissions, copy bindless UAV
    /// writes back to title memory, and map null/invalid addresses safely to 0.
    #[test]
    fn bindless_slots_materialize_refresh_and_copy_back_on_gpu() {
        let Some(host) = host() else {
            return;
        };
        let (title_heap, base) = make_shader_heap(16);
        let slot_handle = |i: u64| base + i * 32;

        // Backing title memory (64 KiB aligned).
        let mem = d3d12x::resource::Memory::zeroed(256 << 10, 64 << 10).unwrap();
        let mem_base = mem.address();
        let srv_addr = mem_base;
        let cbv_16b_addr = mem_base + 0x30; // 16B-aligned, NOT 256B-aligned!
        let copy_srv_addr = mem_base + 0x1000;
        let uav_addr = mem_base + 0x2000;
        let placed_tex_addr = mem_base + (64 << 10);

        let srv_res = make_title_buffer(srv_addr, 256, false);
        let copy_srv_res = make_title_buffer(copy_srv_addr, 256, false);
        let uav_res = make_title_buffer(uav_addr, 256, true);

        // Initial CPU data in title memory.
        // SAFETY: offsets lie within the 256 KiB allocation `mem`.
        unsafe {
            (srv_addr as *mut u32).write(13);
            (cbv_16b_addr as *mut [u32; 4]).write([10, 20, 30, 40]);
            (copy_srv_addr as *mut u32).write(0xCAFE_1234);
            (uav_addr as *mut [u32; 8]).write([0; 8]);
            // First pixel of linear 4x4 RGBA8 placed texture at `placed_tex_addr`.
            (placed_tex_addr as *mut [u8; 4]).write([0x12, 0x34, 0x56, 0x78]);
        }

        // Raw ByteAddressBuffer SRV/UAV descriptions in host D3D12 layout.
        let srv_desc_host = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: 16,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        let uav_desc_host = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: 0,
                    NumElements: 16,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                },
            },
        };
        // SAFETY: both D3D12 view descriptions are 40-byte POD values (`[u32; 10]`).
        let (raw_srv_words, raw_uav_words): ([u32; 10], [u32; 10]) = unsafe {
            (
                std::ptr::from_ref(&srv_desc_host)
                    .cast::<[u32; 10]>()
                    .read(),
                std::ptr::from_ref(&uav_desc_host)
                    .cast::<[u32; 10]>()
                    .read(),
            )
        };

        // Slot 0: Buffer SRV.
        create_srv(std::ptr::null(), srv_res, &raw_srv_words, slot_handle(0));
        // Slot 1: 16B-aligned CBV (size 64 bytes, address % 256 == 0x30).
        create_cbv(
            std::ptr::null(),
            &[cbv_16b_addr as u32, (cbv_16b_addr >> 32) as u32, 64, 0],
            slot_handle(1),
        );
        // Slot 2: Placed 4x4 R8G8B8A8_UNORM linear texture (`layout = 0x100`).
        let placed_res_words: [u32; 14] = [
            3, // TEXTURE2D
            0,
            0,
            0,
            4,
            0,
            4,
            0x0001_0001,
            DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32,
            1,
            0,
            0x100, // console LINEAR layout
            0,
            0,
        ];
        let tex_srv_host = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        let mut placed_view_words = [0u32; 16];
        // SAFETY: `D3D12_SHADER_RESOURCE_VIEW_DESC` is 40 bytes (`[u32; 10]`).
        placed_view_words[..10].copy_from_slice(&unsafe {
            std::ptr::from_ref(&tex_srv_host).cast::<[u32; 10]>().read()
        });
        placed_view_words[10] = placed_tex_addr as u32;
        placed_view_words[11] = (placed_tex_addr >> 32) as u32;
        create_placed_srv(
            std::ptr::null(),
            &placed_res_words,
            &placed_view_words,
            slot_handle(2),
        );
        // Slot 3: Buffer UAV.
        create_uav(
            std::ptr::null(),
            uav_res,
            std::ptr::null(),
            &raw_uav_words,
            slot_handle(3),
        );
        // Slot 6 -> copied to Slot 4 via `copy_simple`.
        create_srv(
            std::ptr::null(),
            copy_srv_res,
            &raw_srv_words,
            slot_handle(6),
        );
        copy_simple(std::ptr::null(), 1, slot_handle(4), slot_handle(6), 0);
        // Slot 5: Unmapped/null CBV (address = 0) -> must read 0 safely.
        create_cbv(std::ptr::null(), &[0, 0, 256, 0], slot_handle(5));

        let host_heap = heap(title_heap).host_arc().unwrap();
        // Before bindless preparation, `mirror(..., false)` deferred all
        // `View::is_buffer()` slots (`made[idx] == 0`).
        for idx in 0..=5 {
            assert_eq!(
                host_heap.made.lock().unwrap()[idx],
                0,
                "slot {idx} starts deferred"
            );
        }

        let (root, pso) = compile_probe_cs(&host.device);
        let list = host.new_list().unwrap();

        let record_probe = |list: &d3d12x::host::HostList| {
            host_heap.prepare_bindless_record(list);
            list.defer_bindless(&host_heap, true);
            // SAFETY: open command list, shader-visible heap, and matching root/PSO.
            unsafe {
                let gpu_start = host_heap.heap.GetGPUDescriptorHandleForHeapStart().ptr;
                let gpu_at = |slot_idx: u32| D3D12_GPU_DESCRIPTOR_HANDLE {
                    ptr: gpu_start + u64::from(slot_idx) * u64::from(host_heap.step),
                };
                list.list
                    .SetDescriptorHeaps(&[Some(host_heap.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                list.list.SetComputeRootDescriptorTable(0, gpu_at(0));
                list.list.SetComputeRootDescriptorTable(1, gpu_at(1));
                list.list.SetComputeRootDescriptorTable(2, gpu_at(2));
                list.list.SetComputeRootDescriptorTable(3, gpu_at(3));
                list.list.SetComputeRootDescriptorTable(4, gpu_at(4));
                list.list.SetComputeRootDescriptorTable(5, gpu_at(5));
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
        };

        record_probe(&list);
        host.execute(&[&list]);
        list.reset();

        // SAFETY: GPU finished and `Host::execute` copied UAV writes back to `uav_addr`.
        let out1 = unsafe { (uav_addr as *const [u32; 5]).read_volatile() };
        assert_eq!(
            out1[0], 13,
            "bindless buffer SRV materialized and refreshed"
        );
        assert_eq!(
            out1[1], 100,
            "16B-aligned bindless CBV materialized via bounce"
        );
        assert_eq!(
            out1[2], 0x7856_3412,
            "placed streamed texture uploaded and sampled"
        );
        assert_eq!(out1[3], 0xCAFE_1234, "copied bindless SRV materialized");
        assert_eq!(out1[4], 0, "null-address CBV safely reads zero");

        // Modify CPU memory for both the SRV buffer and the 16B-aligned CBV,
        // overwrite slot 5 with a `u64::MAX` garbage address CBV, and release
        // `copy_srv_res` so slot 4 goes stale -> both slots 4 and 5 read 0!
        create_cbv(
            std::ptr::null(),
            &[u32::MAX, u32::MAX, 256, 0],
            slot_handle(5),
        );
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(copy_srv_res), 0);
        // SAFETY: offsets lie within `mem`.
        unsafe {
            (srv_addr as *mut u32).write_volatile(500);
            (cbv_16b_addr as *mut [u32; 4]).write_volatile([100, 200, 300, 400]);
        }

        // Released views are looked for once a frame.
        d3d12x::queue::pass_frame();
        record_probe(&list);
        host.execute(&[&list]);
        list.reset();

        // SAFETY: GPU finished and copied back.
        let out2 = unsafe { (uav_addr as *const [u32; 5]).read_volatile() };
        assert_eq!(
            out2[0], 500,
            "dirty SRV title memory refreshed on second submit"
        );
        assert_eq!(
            out2[1], 1000,
            "dirty 16B-aligned CBV refreshed on second submit"
        );
        assert_eq!(out2[3], 0, "released resource slot pruned and reads zero");
        assert_eq!(
            out2[4], 0,
            "overflowing u64::MAX garbage address safely reads zero"
        );

        assert_eq!(release(srv_res), 0);
        assert_eq!(release(uav_res), 0);
        assert_eq!(release(title_heap), 0);
    }

    /// A shared window released (its title memory freed) makes the
    /// slots made in place over it null before the next submission.
    #[test]
    fn slots_in_place_over_a_released_window_are_made_null() {
        if host().is_none() {
            return;
        }
        let (title_heap, _) = make_shader_heap(8);
        let host_heap = heap(title_heap).host_arc().unwrap();
        // Slot 5 made in place at a GPU address of a shared window, slot 6
        // elsewhere (a region the title frees).
        const WINDOW: u64 = 0x7_0000_0000;
        {
            let mut made = host_heap.made.lock().unwrap();
            made[5] = 1 << 32 | 77;
            made[6] = 1 << 32 | 78;
        }
        {
            let mut b = host_heap.bindless.lock().unwrap();
            b.in_place.insert(5, WINDOW + 0x100);
            b.in_place.insert(6, WINDOW + (64 << 20));
        }
        // Earlier releases (other tests) are seen first.
        host_heap.null_gone_windows();
        shared_window_gone(WINDOW, 64 << 20);
        assert_eq!(host_heap.null_gone_windows(), 1, "the slot over the window");
        assert_eq!(host_heap.made.lock().unwrap()[5], 0, "made null");
        assert_ne!(host_heap.made.lock().unwrap()[6], 0, "the other slot stays");
        let b = host_heap.bindless.lock().unwrap();
        assert!(!b.in_place.contains_key(&5) && b.dirty.contains(&5), "made again by the next plan");
        assert!(b.plan.is_none(), "the next plan starts whole");
        drop(b);
        assert_eq!(host_heap.null_gone_windows(), 0, "once");
    }

    /// Submissions with nothing changed reuse a heap's bindless plan (no
    /// slot is made or range found again); writing a slot makes a new one,
    /// whose reads of one host copy are merged into one range.
    #[test]
    fn bindless_plan_is_remade_only_when_slots_change() {
        if host().is_none() {
            return;
        }
        let (title_heap, base) = make_shader_heap(8);
        let mem = d3d12x::resource::Memory::zeroed(256 << 10, 64 << 10).unwrap();
        let cbv = |at: u64, slot: u64| {
            create_cbv(
                std::ptr::null(),
                &[at as u32, (at >> 32) as u32, 256, 0],
                base + slot * 32,
            );
        };
        cbv(mem.address(), 0);
        let host_heap = heap(title_heap).host_arc().unwrap();
        let first = host_heap.bindless_plan().unwrap();
        assert_eq!(first.flat_reads().len(), 1);
        assert_eq!(first.flat_reads()[0].2, 256);
        assert!(first.writes().next().is_none(), "constant buffers are not written");
        // Other tests may release resources meanwhile, which remakes plans.
        let reused = (0..100).any(|_| {
            let a = host_heap.bindless_plan().unwrap();
            Arc::ptr_eq(&a, &host_heap.bindless_plan().unwrap())
        });
        assert!(reused, "unchanged slots reuse the plan");

        // Slot 0 marked unmade: were it made again, it would be made anew.
        host_heap.made.lock().unwrap()[0] = 0;
        cbv(mem.address() + 0x1000, 1);
        let second = host_heap.bindless_plan().unwrap();
        assert!(
            !Arc::ptr_eq(&first, &second),
            "a written slot remakes the plan"
        );
        // In the same frame the plan is extended by a part holding only
        // the new read (the frame's next full plan merges them).
        assert_eq!(second.parts(), 2, "an extension adds a part");
        let reads = second.flat_reads();
        assert_eq!(reads.len(), 2);
        assert_eq!((reads[1].1, reads[1].2), (mem.address() as usize + 0x1000, 256));
        assert_eq!(
            host_heap.made.lock().unwrap()[0],
            0,
            "only the written slot is made again"
        );
        assert_ne!(host_heap.made.lock().unwrap()[1], 0);

        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(title_heap), 0);
    }

    /// Placed textures reached through slots upload the first time, then
    /// only after [`RECHECK_FRAMES`] (a few per frame), or when another
    /// texture takes the object's address.
    #[test]
    fn bindless_texture_uploads_are_throttled() {
        let Some(host) = host() else {
            return;
        };
        let Some(heap) = HostHeap::new(false, 4, 0) else {
            return;
        };
        let (a, _) = host.shared_buffer(1 << 16).unwrap();
        let (b, _) = host.shared_buffer(1 << 16).unwrap();
        let tex = 0x1000 as crate::d3d12x::object::This;
        assert!(heap.upload_due(tex, &a, 10), "first reach uploads");
        assert!(!heap.upload_due(tex, &a, 10));
        assert!(!heap.upload_due(tex, &a, 10 + RECHECK_FRAMES - 1));
        assert!(heap.upload_due(tex, &b, 11), "a new texture at the address");
        assert!(heap.upload_due(tex, &b, 11 + RECHECK_FRAMES), "looked at again");
        // At most `RECHECKS` known textures per frame.
        let frame = 1000;
        let many: Vec<_> = (0..2 * RECHECKS as usize)
            .map(|i| {
                let t = (0x2000 + i) as crate::d3d12x::object::This;
                assert!(heap.upload_due(t, &a, 0));
                t
            })
            .collect();
        let again = many.iter().filter(|&&t| heap.upload_due(t, &a, frame)).count();
        assert_eq!(again, RECHECKS as usize);
    }

    /// A list queues each heap once, keeping whether any draw may write.
    #[test]
    fn bindless_heaps_queue_once_per_list() {
        let Some(host) = host() else {
            return;
        };
        let (title_heap, _) = make_shader_heap(8);
        let host_heap = heap(title_heap).host_arc().unwrap();
        let list = host.new_list().unwrap();
        list.defer_bindless(&host_heap, false);
        list.defer_bindless(&host_heap, true);
        list.defer_bindless(&host_heap, false);
        assert_eq!(list.bindless_queued(), vec![true]);
        list.close();
        host.execute(&[&list]);
        list.reset();
        assert!(list.bindless_queued().is_empty());

        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(title_heap), 0);
    }

    /// Verifies `PLACED_TEXTURES` entries overlapping a remapped virtual range
    /// are evicted on `title_memory_remapped` while non-overlapping entries stay.
    #[test]
    fn placed_textures_evict_on_title_memory_remapped() {
        if host().is_none() {
            return;
        }
        let mem = d3d12x::resource::Memory::zeroed(256 << 10, 64 << 10).unwrap();
        let addr_a = mem.address();
        let addr_b = mem.address() + (128 << 10);
        let desc: [u32; 14] = [
            3, // TEXTURE2D
            0,
            0,
            0,
            16,
            0,
            16,
            0x0001_0001,
            DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32,
            1,
            0,
            0x100, // LINEAR
            0,
            0,
        ];

        assert!(d3d12x::resource::texture_at(addr_a, &desc).is_some());
        assert!(d3d12x::resource::texture_at(addr_b, &desc).is_some());
        assert!(d3d12x::resource::has_placed_texture(addr_a, &desc));
        assert!(d3d12x::resource::has_placed_texture(addr_b, &desc));

        let epoch_before = d3d12x::resource::placed_epoch();
        // Remap 256 bytes inside `addr_a`'s footprint (not even starting at `addr_a`!).
        d3d12x::watch::title_memory_remapped((addr_a + 128) as usize, 256);

        assert!(
            !d3d12x::resource::has_placed_texture(addr_a, &desc),
            "overlapping placed texture must be evicted"
        );
        assert!(
            d3d12x::resource::has_placed_texture(addr_b, &desc),
            "non-overlapping placed texture must remain cached"
        );
        assert!(d3d12x::resource::placed_epoch() > epoch_before);

        // Clean up `addr_b` too.
        d3d12x::watch::title_memory_remapped(addr_b as usize, 64 << 10);
        assert!(!d3d12x::resource::has_placed_texture(addr_b, &desc));
    }

    /// A structured view placed off its copy's elements views a copy
    /// of its own bytes; one on an element views the shared copy.
    #[test]
    fn structured_views_off_their_copys_elements_get_their_own_copy() {
        // VIctory 86a5ed6's test, on a placed view of memory outside buffers.
        let Some(host) = host() else { return };
        let mem = d3d12x::resource::Memory::zeroed(1 << 16, 1 << 16).unwrap();
        let start = mem.address();
        // SAFETY: a plain CPU-side heap of one view.
        let cpu: ID3D12DescriptorHeap = unsafe {
            host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 1, ..Default::default() })
        }
        .unwrap();
        // SAFETY: a live heap.
        let at = unsafe { cpu.GetCPUDescriptorHandleForHeapStart() };
        // Four 456-byte elements at `address`, as the title passes them.
        let view = |address: u64| {
            let resource: [u32; 14] = [1, 0, 0, 0, 4 * 456, 0, 1, 1 | 1 << 16, 0, 1, 0, 1, 0, 0];
            let mut desc = [0u32; 16];
            desc[..10].copy_from_slice(&[0, 1, 5768, 0, 0, 0, 4, 456, 0, 0]);
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            View::Placed(false, resource, desc)
        };
        let copy = d3d12x::shadow::find(start, 1).unwrap();
        let on = start - copy.offset(start) % 456 + 456;
        let off = if (on + 100 - start) < (1 << 16) - 4 * 456 { on + 100 } else { on - 100 };
        let width = |address: u64| {
            let made = make(&view(address), at).flatten().expect("a view");
            // SAFETY: a live buffer.
            unsafe { made.GetDesc() }.Width
        };
        assert!(width(on) >= 1 << 16, "on an element: the window's copy");
        assert_eq!(width(off), (4 * 456u64).next_multiple_of(256), "off: its own bytes");
        let (c, address, len) = view_copy(&view(off)).unwrap();
        assert_eq!((c.offset(off), address, len), (0, off, 4 * 456));
    }

    /// A `HeldTexture` returned by `texture_at` keeps the placed resource
    /// alive across concurrent eviction by `title_memory_remapped`.
    #[test]
    fn placed_texture_handle_survives_eviction() {
        if host().is_none() {
            return;
        }
        let mem = d3d12x::resource::Memory::zeroed(128 << 10, 64 << 10).unwrap();
        let addr = mem.address();
        let desc: [u32; 14] = [
            3, // TEXTURE2D
            0,
            0,
            0,
            16,
            0,
            16,
            0x0001_0001,
            DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32,
            1,
            0,
            0x100, // LINEAR
            0,
            0,
        ];

        let held = d3d12x::resource::texture_at(addr, &desc).unwrap();
        assert!(d3d12x::resource::has_placed_texture(addr, &desc));

        // Evict while the caller still holds `held`.
        d3d12x::watch::title_memory_remapped(addr as usize, 64 << 10);
        assert!(!d3d12x::resource::has_placed_texture(addr, &desc));

        // The caller's handle still points to a live resource and host texture.
        let host_tex = d3d12x::resource::host_resource(held.as_this()).unwrap();
        // SAFETY: `held` keeps the resource (and its host texture) alive.
        let host_desc = unsafe { host_tex.GetDesc() };
        assert_eq!((host_desc.Width, host_desc.Height), (16, 16));
        drop(held);
    }

    /// Slots filled, overwritten, or cleared by raw byte copies into the
    /// title heap (bypassing `mirror()`) are detected by `bindless_plan`.
    #[test]
    fn bindless_slots_detect_byte_copies_and_overwrites() {
        if host().is_none() {
            return;
        }
        let (title_heap, base) = make_shader_heap(8);
        let mem = d3d12x::resource::Memory::zeroed(256 << 10, 64 << 10).unwrap();
        let addr_a = mem.address();
        let addr_b = mem.address() + (128 << 10);
        let mut staging = [0u64; 8]; // two 32-byte staging slots outside the shader heap
        let stage_a = staging.as_mut_ptr() as u64;
        let stage_b = stage_a + 32;
        create_cbv(
            std::ptr::null(),
            &[addr_a as u32, (addr_a >> 32) as u32, 256, 0],
            stage_a,
        );
        create_cbv(
            std::ptr::null(),
            &[addr_b as u32, (addr_b >> 32) as u32, 256, 0],
            stage_b,
        );

        let host_heap = heap(title_heap).host_arc().unwrap();
        let slot1 = base + 32;

        // Raw byte copy into an empty shader-visible slot (no API mirror call).
        // SAFETY: both `stage_a` and `slot1` are valid 32-byte descriptor slots.
        unsafe { std::ptr::copy_nonoverlapping(stage_a as *const u8, slot1 as *mut u8, 32) };
        let p1 = host_heap.bindless_plan().unwrap();
        assert_eq!(p1.flat_reads().len(), 1);
        assert_eq!(p1.flat_reads()[0].1, addr_a as usize);
        assert_ne!(host_heap.made.lock().unwrap()[1], 0);

        // Raw byte overwrite of an already-tracked slot with another view.
        // SAFETY: both `stage_b` and `slot1` are valid 32-byte descriptor slots.
        unsafe { std::ptr::copy_nonoverlapping(stage_b as *const u8, slot1 as *mut u8, 32) };
        let p2 = host_heap.bindless_plan().unwrap();
        assert!(!Arc::ptr_eq(&p1, &p2));
        // In the same frame the plan is extended (the old read stays until
        // the frame's next full plan).
        assert!(p2.flat_reads().iter().any(|r| r.1 <= addr_b as usize && addr_b as usize + 256 <= r.1 + r.2));

        // Raw byte zeroing removes the tracked slot and clears its host
        // descriptor; in a new frame the plan is made from all slots.
        d3d12x::queue::pass_frame();
        // SAFETY: `slot1` is a valid 32-byte descriptor slot.
        unsafe { (slot1 as *mut u8).write_bytes(0, 32) };
        let p3 = host_heap.bindless_plan().unwrap();
        assert!(p3.flat_reads().is_empty());
        assert_eq!(host_heap.made.lock().unwrap()[1], 0);

        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(title_heap), 0);
    }

    /// A heap in write-watched title memory is looked at only where
    /// written since the last plan: a slot written as bytes is made, a page
    /// nobody wrote is not looked at.
    #[test]
    fn bindless_scan_looks_only_at_written_pages() {
        use xnw_core::win;
        if host().is_none() {
            return;
        }
        let size = SCAN_COUNT as usize * 32;
        // SAFETY: a fresh write-watched allocation, kept for the run (the
        // journal keeps its pages).
        let at = unsafe { win::VirtualAlloc(std::ptr::null(), size, win::MEM_RESERVE | win::MEM_COMMIT | win::MEM_WRITE_WATCH, win::PAGE_READWRITE) } as u64;
        assert_ne!(at, 0);
        scan_only_written_pages(at);
    }

    /// The same for a heap in view memory (the title's: pages
    /// mapped into a placeholder, never write-watched), which is guarded;
    /// asked without guarding it was scanned whole at every plan.
    #[test]
    fn bindless_scan_of_view_memory_looks_only_at_written_pages() {
        use xnw_core::win;
        if host().is_none() {
            return;
        }
        let size = SCAN_COUNT as usize * 32;
        // SAFETY: plain section creation and mapping; kept for the run.
        let at = unsafe {
            let section = win::CreateFileMappingW(win::INVALID_HANDLE_VALUE, std::ptr::null(), win::PAGE_READWRITE, 0, size as u32, std::ptr::null());
            assert!(!section.is_null());
            let v = win::MapViewOfFile(section, win::FILE_MAP_WRITE, 0, 0, size) as u64;
            win::CloseHandle(section);
            v
        };
        assert_ne!(at, 0);
        // A whole scan would find slot 0 changed again (`seen[0]`).
        scan_only_written_pages(at);
    }

    /// Slots in the scan tests: two 64 KiB pages.
    const SCAN_COUNT: u32 = 4096;

    /// A heap of [`SCAN_COUNT`] slots placed at `at`: written as bytes, a
    /// slot is made; a page nobody wrote is not looked at.
    fn scan_only_written_pages(at: u64) {
        const COUNT: u32 = SCAN_COUNT;
        let size = COUNT as usize * 32;
        // Touched first, as the title's long-used heap is: a fresh page the
        // first scan reads is filled then and reported written by the
        // next asking.
        // SAFETY: the memory the caller made.
        unsafe { (at as *mut u8).write_bytes(0, size) };
        let ext = ExtendedHeapDesc { desc: HeapDesc { kind: 0, count: COUNT, flags: 1, nodes: 0 }, addr: at, size: size as u64 };
        let mut title_heap = std::ptr::null();
        assert_eq!(create_heap(std::ptr::null(), &ext.desc, &HEAP_IID, &mut title_heap), S_OK);
        let mem = d3d12x::resource::Memory::zeroed(256 << 10, 64 << 10).unwrap();
        let staging = [0u64; 4];
        let stage = staging.as_ptr() as u64;
        // A view recorded outside the heap, copied in as bytes (no mirror).
        let put = |slot: u64, address: u64| {
            create_cbv(std::ptr::null(), &[address as u32, (address >> 32) as u32, 256, 0], stage);
            // SAFETY: a 32-byte record into a slot of the heap.
            unsafe { std::ptr::copy_nonoverlapping(stage as *const u8, (at + slot * 32) as *mut u8, 32) };
        };
        put(0, mem.address());
        let host_heap = heap(title_heap).host_arc().unwrap();
        host_heap.bindless_plan().unwrap();
        assert_ne!(host_heap.made.lock().unwrap()[0], 0, "a slot written as bytes is made");
        // Were page 0 looked at again, slot 0 would be found changed.
        host_heap.bindless.lock().unwrap().seen[0] = 0;
        put(2048, mem.address() + 0x1000);
        let plan = host_heap.bindless_plan().unwrap();
        assert_ne!(host_heap.made.lock().unwrap()[2048], 0, "the written slot is made");
        assert!(plan.parts.iter().flat_map(|p| p.reads.iter()).any(|r| r.1 <= mem.address() as usize + 0x1000 && mem.address() as usize + 0x1100 <= r.1 + r.2));
        assert_eq!(host_heap.bindless.lock().unwrap().seen[0], 0, "the unwritten page is not looked at");
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(title_heap), 0);
    }

    /// A slot viewing memory outside buffer objects is made again at
    /// the next frame's look after the memory is unmapped, and reads nothing.
    #[test]
    fn bindless_slots_of_unmapped_memory_are_dropped() {
        use xnw_core::win;
        if host().is_none() {
            return;
        }
        let (title_heap, base) = make_shader_heap(8);
        // SAFETY: a fresh allocation, released below.
        let at = unsafe { win::VirtualAlloc(std::ptr::null(), 64 << 10, win::MEM_RESERVE | win::MEM_COMMIT, win::PAGE_READWRITE) } as u64;
        assert_ne!(at, 0);
        create_cbv(std::ptr::null(), &[at as u32, (at >> 32) as u32, 256, 0], base + 3 * 32);
        let host_heap = heap(title_heap).host_arc().unwrap();
        let before = host_heap.bindless_plan().unwrap();
        assert_ne!(host_heap.made.lock().unwrap()[3], 0, "made while mapped");
        assert!(before.flat_reads().iter().any(|r| r.1 == at as usize));
        // Decommitted, still reserved: a parallel test's allocation cannot
        // take the address meanwhile.
        // SAFETY: our allocation, nothing refers to it but the slot.
        unsafe { win::VirtualFree(at as *mut _, 64 << 10, win::MEM_DECOMMIT) };
        d3d12x::watch::title_memory_remapped(at as usize, 64 << 10);
        d3d12x::queue::pass_frame();
        let after = host_heap.bindless_plan().unwrap();
        assert_eq!(host_heap.made.lock().unwrap()[3], 0, "dropped once unmapped");
        assert!(!after.flat_reads().iter().any(|r| r.1 == at as usize), "and no longer read");
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(title_heap), 0);
        // SAFETY: our reservation.
        unsafe { win::VirtualFree(at as *mut _, 0, win::MEM_RELEASE) };
    }

    /// A plan's reads of guarded memory are brought up to date until
    /// one finds nothing written, then skipped until the title writes.
    #[test]
    fn bindless_reads_skip_unwritten_memory() {
        use xnw_core::win;
        if host().is_none() {
            return;
        }
        // A pagefile section mapped once: view memory, which is guarded.
        // SAFETY: plain section creation and mapping.
        let view = unsafe {
            let section = win::CreateFileMappingW(win::INVALID_HANDLE_VALUE, std::ptr::null(), win::PAGE_READWRITE, 0, 64 << 10, std::ptr::null());
            assert!(!section.is_null());
            let v = win::MapViewOfFile(section, win::FILE_MAP_WRITE, 0, 0, 64 << 10) as u64;
            win::CloseHandle(section);
            v
        };
        assert_ne!(view, 0);
        let (title_heap, base) = make_shader_heap(8);
        create_cbv(std::ptr::null(), &[view as u32, (view >> 32) as u32, 256, 0], base);
        let host_heap = heap(title_heap).host_arc().unwrap();
        let plan = host_heap.bindless_plan().unwrap();
        let refreshed = || {
            let mut n = 0;
            plan.refresh_reads(|s, a, l| {
                n += 1;
                s.update(a, l);
            });
            n
        };
        // The first refresh guards the block; the second finds it written
        // by nobody (its own asking is journalled after the first's note).
        assert_eq!(refreshed(), 1);
        assert_eq!(refreshed(), 1);
        assert_eq!(refreshed(), 0, "unwritten: skipped");
        assert_eq!(refreshed(), 0);
        // SAFETY: a write into the view: the handler opens the block.
        unsafe { ((view + 16) as *mut u32).write(7) };
        assert_eq!(refreshed(), 1, "written: brought up to date");
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(title_heap), 0);
        drop(plan);
        // SAFETY: our view.
        unsafe { win::UnmapViewOfFile(view as *const _) };
        d3d12x::watch::title_memory_remapped(view as usize, 64 << 10);
    }

    /// Ours, runnable before the device layer: a misaligned constant buffer in a
    /// shader-visible heap is made through a bounce slice (no host copy
    /// needed), the plan is reused while nothing changes and made again
    /// when a slot is written, and only the written slot is made again.
    /// Sampler heaps mirror console samplers when the host heap is made.
    #[test]
    fn bindless_bounce_plan_is_reused_until_a_slot_changes() {
        if std::env::var_os("XNW_AGILITY").is_none_or(|d| d.is_empty()) || host().is_none() {
            println!("SKIPPED: set XNW_AGILITY to the Agility SDK folder");
            return;
        }
        let (title_heap, base) = make_shader_heap(8);
        let mem = d3d12x::resource::Memory::zeroed(64 << 10, 64 << 10).unwrap();
        let at = mem.address() + 0x30; // 16-byte aligned, not 256
        // SAFETY: inside `mem`.
        unsafe { (at as *mut [u32; 4]).write([1, 2, 3, 4]) };
        let cbv = |address: u64, slot: u64| {
            create_cbv(std::ptr::null(), &[address as u32, (address >> 32) as u32, 64, 0], base + slot * 32);
        };
        cbv(at, 0);
        let host_heap = heap(title_heap).host_arc().unwrap();
        let first = host_heap.bindless_plan().unwrap();
        assert_eq!(first.flat_bounces().len(), 1, "the misaligned buffer bounces");
        assert_ne!(host_heap.made.lock().unwrap()[0], 0, "slot 0 made");
        first.copy_constants();
        let (cpu, _, len) = first.flat_bounces()[0];
        // SAFETY: the bounce slice holds `len` bytes.
        assert_eq!(unsafe { std::slice::from_raw_parts(cpu as *const u32, len as usize / 4) }, &[1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let reused = (0..100).any(|_| {
            let a = host_heap.bindless_plan().unwrap();
            Arc::ptr_eq(&a, &host_heap.bindless_plan().unwrap())
        });
        assert!(reused, "unchanged slots reuse the plan");
        host_heap.made.lock().unwrap()[0] = 0;
        cbv(at + 0x100, 1);
        let second = host_heap.bindless_plan().unwrap();
        assert!(!Arc::ptr_eq(&first, &second), "a written slot remakes the plan");
        assert_eq!(second.flat_bounces().len(), 2);
        assert_eq!(host_heap.made.lock().unwrap()[0], 0, "only the written slot is made again");
        assert_ne!(host_heap.made.lock().unwrap()[1], 0);
        let (made, reused) = plan_counts();
        assert!(made + plans_extended() >= 2 && reused >= 1);

        // A shader-visible sampler heap: slots written before the host heap
        // exists are made when it is.
        let ext = ExtendedHeapDesc { desc: HeapDesc { kind: 1, count: 4, flags: 1, nodes: 0 }, addr: 0, size: 0 };
        let mut samplers = std::ptr::null();
        assert_eq!(create_heap(std::ptr::null(), &ext.desc, &HEAP_IID, &mut samplers), S_OK);
        let start = slot::<extern "system" fn(This) -> u64>(samplers, 0x48)(samplers);
        let mut words = [0u32; 27];
        words[1] = 1;
        // Wrap: address mode 0 is not a host mode, and a host sampler made
        // with it removes the device (seen when later tests used the GPU).
        words[6..9].copy_from_slice(&[1, 1, 1]);
        words[10] = 1000.0f32.to_bits();
        crate::d3d12x::descriptor::create_sampler_x(std::ptr::null(), &words, start + 16);
        let sampler_heap = heap(samplers).host_arc().unwrap();
        assert!(sampler_heap.sampler);
        assert_ne!(sampler_heap.made.lock().unwrap()[1], 0, "pending sampler slot made");
        assert_eq!(sampler_heap.made.lock().unwrap()[0], 0);
        assert!(sampler_heap.bindless_plan().is_none(), "sampler heaps have no bindless plan");

        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(samplers), 0);
        assert_eq!(release(title_heap), 0);
    }

    // Typeless views. The first two tests are VIctory's (GPL-3.0-or-later)
    // VIctory tests (Xvania7, branch fix/issue-850-typeless-view, 2be93af,
    // `typeless_view_on_typeless_resource_remaps_to_concrete_format` and
    // `typeless_resource_with_none_or_zero_view_format_maps_to_concrete`) on
    // our `fits`; where we differ the assertion says why.

    fn fits_of(own: i32, flags: D3D12_RESOURCE_FLAGS, view: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
        fits(&D3D12_RESOURCE_DESC { Format: DXGI_FORMAT(own), Flags: flags, ..Default::default() }, view, unordered)
    }

    #[test]
    fn typeless_view_on_typeless_resource_remaps_to_concrete_format() {
        let none = D3D12_RESOURCE_FLAG_NONE;
        for (own, view, expected) in [
            (27, 27, 28), // R8G8B8A8_TYPELESS view -> R8G8B8A8_UNORM
            (90, 90, 87), // B8G8R8A8_TYPELESS -> B8G8R8A8_UNORM
            (39, 39, 41), // R32_TYPELESS -> R32_FLOAT
            (27, 0, 28),  // unnamed view on a typeless texture
            (27, 29, 29), // a typed view of a typeless texture stays
            (28, 27, 28), // a typeless view of a typed texture takes its format
            (97, 97, 98), // BC7_TYPELESS -> BC7_UNORM
            (94, 94, 95), // BC6H_TYPELESS -> BC6H_UF16
        ] {
            let mut format = DXGI_FORMAT(view);
            assert!(fits_of(own, none, Some(&mut format), false), "{own} {view}");
            assert_eq!(format.0, expected, "{own} {view}");
        }
        // Ours: a typeless texture without a view description is refused
        // here; `make` always spells one out for such a texture.
        assert!(!fits_of(27, none, None, false));
        // Another family is refused.
        let mut format = DXGI_FORMAT(10);
        assert!(!fits_of(27, none, Some(&mut format), false));
        // Unordered access: sRGB becomes linear.
        let mut format = DXGI_FORMAT(29);
        assert!(fits_of(27, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, Some(&mut format), true));
        assert_eq!(format.0, 28);
        let mut format = DXGI_FORMAT(27);
        assert!(fits_of(27, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, Some(&mut format), true));
        assert_eq!(format.0, 28);
    }

    #[test]
    fn typeless_depth_and_r16_views_take_a_readable_format() {
        let depth = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        // VIctory's depth rule rather than its typeless-view rule (which refuses planar depth without a
        // plane format): the depth aspect, as `texture_srv_desc` does.
        for (own, expected) in [(19, 21), (44, 46), (39, 41), (53, 56)] {
            for view in [0, own] {
                let mut format = DXGI_FORMAT(view);
                assert!(fits_of(own, depth, Some(&mut format), false), "{own} {view}");
                assert_eq!(format.0, expected, "{own} {view}");
            }
        }
        // Non-depth R16_TYPELESS is packed 16-bit colour here: R16_UNORM
        // (VIctory's typeless-view rule would read R16_FLOAT).
        let mut format = DXGI_FORMAT(53);
        assert!(fits_of(53, D3D12_RESOURCE_FLAG_NONE, Some(&mut format), false));
        assert_eq!(format.0, 56);
        assert_eq!(crate::d3d12x::resource::first_concrete(27), 28);
        assert!(!crate::d3d12x::resource::is_typeless(28) && !crate::d3d12x::resource::is_typeless(87));
        assert!(crate::d3d12x::resource::is_typeless(90) && crate::d3d12x::resource::is_typeless(92));
    }

    /// Render and depth target views of typeless textures, made through the
    /// title's entry points with the view formats the title uses, write;
    /// shader views of them in a typeless or no format read it back.
    #[test]
    fn typeless_targets_render_and_read_back() {
        use crate::d3d12x::descriptor::{create_dsv, create_rtv, dsv_record, rtv_record};
        use crate::d3d12x::resource::{HeapProperties, RESOURCE_IIDS, ResourceDesc, create_committed, host_resource};
        if std::env::var_os("XNW_AGILITY").is_none_or(|d| d.is_empty()) {
            println!("SKIPPED: set XNW_AGILITY to the Agility SDK folder to run on the host GPU");
            return;
        }
        let host = d3d12x::host::host().expect("a host GPU");
        let device = &host.device;
        const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=1)), DescriptorTable(UAV(u0, numDescriptors=1))"
Texture2D<float4> tex : register(t0);
RWByteAddressBuffer result : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() { result.Store4(0, asuint(tex.Load(int3(1, 1, 0)))); }
"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: static source and names; out pointer ours.
        let (root, pso) = unsafe {
            D3DCompile(SOURCE.as_ptr().cast(), SOURCE.len(), s!("typeless_probe"), None, None, s!("main"), s!("cs_5_1"), 0, 0, &mut code, None).unwrap();
            let code = code.unwrap();
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        };
        let heap = HostHeap::new(false, 4, 0).unwrap();
        // SAFETY: a plain query.
        let increment = unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) } as u64;
        // SAFETY: a live heap.
        let gpu_start = unsafe { heap.heap.GetGPUDescriptorHandleForHeapStart() };
        let gpu = |i: u64| D3D12_GPU_DESCRIPTOR_HANDLE { ptr: gpu_start.ptr + i * increment };
        let buffer = |kind: D3D12_HEAP_TYPE, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES| {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let mut r: Option<ID3D12Resource> = None;
            // SAFETY: plain creation.
            unsafe { device.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r).unwrap() };
            r.unwrap()
        };
        // Errors the validation layer stored (`XNW_GPU_DEBUG=layer`).
        let errors = || {
            use windows::core::Interface;
            let Ok(queue) = device.cast::<ID3D12InfoQueue>() else { return 0 };
            let mut n = 0;
            // SAFETY: plain queries into buffers of the size the queue asked for.
            unsafe {
                for i in 0..queue.GetNumStoredMessages() {
                    let mut len = 0usize;
                    if queue.GetMessage(i, None, &mut len).is_err() || len == 0 {
                        continue;
                    }
                    let mut buf = vec![0u64; len.div_ceil(8)];
                    let m = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                    if queue.GetMessage(i, Some(m), &mut len).is_ok() && (*m).Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 {
                        let text = std::slice::from_raw_parts((*m).pDescription, (*m).DescriptionByteLength.saturating_sub(1));
                        println!("validation: {}", String::from_utf8_lossy(text));
                        n += 1;
                    }
                }
            }
            n
        };
        let srgb = |c: f32| if c <= 0.003_130_8 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
        const C: [f32; 4] = [0.25, 0.5, 0.75, 1.0];
        const RT: (u32, u32) = (1, 4); // flag, state
        const DEPTH: (u32, u32) = (2, 0x10);
        // Title format, target, target view format (None: no description),
        // shader view format, expected texel (components compared), tolerance.
        let encoded = [srgb(0.25), srgb(0.5), srgb(0.75), 1.0];
        let cases: Vec<(u32, (u32, u32), Option<u32>, Option<u32>, [f32; 4], usize, f32)> = vec![
            (27, RT, None, None, C, 4, 1.5 / 255.0),
            (27, RT, Some(0), Some(0), C, 4, 1.5 / 255.0),
            (27, RT, Some(27), Some(27), C, 4, 1.5 / 255.0),
            (27, RT, Some(28), Some(0), C, 4, 1.5 / 255.0),
            (27, RT, Some(29), Some(27), encoded, 4, 1.5 / 255.0),
            (27, RT, Some(29), Some(29), C, 4, 1.5 / 255.0),
            (53, RT, None, Some(53), C, 1, 1e-4),
            (53, RT, Some(53), None, C, 1, 1e-4),
            (86, RT, None, None, C, 1, 1e-4),
            (45, DEPTH, None, None, C, 1, 1e-5),
            (45, DEPTH, Some(0), Some(0), C, 1, 1e-5),
            (45, DEPTH, Some(44), Some(44), C, 1, 1e-5),
            (45, DEPTH, Some(45), Some(45), C, 1, 1e-5),
            (118, DEPTH, Some(118), Some(119), C, 1, 1e-5),
            (40, DEPTH, None, Some(39), C, 1, 0.0),
            (40, DEPTH, Some(39), None, C, 1, 0.0),
            (39, DEPTH, None, None, C, 1, 0.0),
            (55, DEPTH, Some(0), Some(53), C, 1, 1e-4),
            (55, DEPTH, None, None, C, 1, 1e-4),
        ];
        let before = errors();
        for (format, (flag, state), target, view, expected, compared, tolerance) in cases {
            let what = format!("title format {format}, target view {target:?}, shader view {view:?}");
            let desc = ResourceDesc { dimension: 3, alignment: 0, width: 8, height: 8, depth_or_array_size: 1, mip_levels: 1, format, sample_count: 1, sample_quality: 0, layout: 0, flags: flag };
            let properties = HeapProperties { kind: 1, cpu_pages: 0, pool: 0, creation_nodes: 0, visible_nodes: 0 };
            let mut this = std::ptr::null();
            assert_eq!(create_committed(std::ptr::null(), &properties, 0, &desc, state, std::ptr::null(), &RESOURCE_IIDS[0], &mut this), S_OK, "{what}");
            let texture = host_resource(this).expect("a host texture").clone();
            // SAFETY: a live texture.
            let host_format = unsafe { texture.GetDesc() }.Format.0;
            assert!(crate::d3d12x::resource::is_typeless(host_format as u32), "{what}: host copy {host_format} is typeless");
            let mut record = [0u64; 16];
            let words = target.map(|f| [f, if flag == 2 { 3 } else { 4 }, 0, 0, 0, 0]);
            let words_ptr = words.as_ref().map_or(std::ptr::null(), std::ptr::from_ref);
            let at = record.as_mut_ptr() as u64;
            let list = host.new_list().unwrap();
            let result = buffer(D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            let readback = buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            let state = D3D12_RESOURCE_STATES(state as i32);
            // SAFETY: an open list; live resources; slots of our heaps.
            unsafe {
                if flag == 2 {
                    create_dsv(std::ptr::null(), this, words_ptr, at);
                    let handle = host.dsv(&texture, dsv_record(at).unwrap().depth_desc().as_ref());
                    list.list.ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH, C[0], 0, None);
                } else {
                    create_rtv(std::ptr::null(), this, words_ptr, at);
                    let handle = host.rtv(&texture, rtv_record(at).unwrap().desc().as_ref());
                    list.list.ClearRenderTargetView(handle, &C, None);
                }
                list.list.ResourceBarrier(&[d3d12x::host::transition(&texture, state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);
                let srv = view.map(|f| [f, 4, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, 0, 0, 1, 0, 0, 0, 0]);
                assert!(make(&View::Shader(this as usize, srv), heap.cpu(0)).is_some(), "{what}: shader view made");
                let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: D3D12_BUFFER_UAV { NumElements: 4, Flags: D3D12_BUFFER_UAV_FLAG_RAW, ..Default::default() } },
                };
                device.CreateUnorderedAccessView(&result, None, Some(&uav), heap.cpu(1));
                list.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                list.list.SetComputeRootDescriptorTable(0, gpu(0));
                list.list.SetComputeRootDescriptorTable(1, gpu(1));
                list.list.Dispatch(1, 1, 1);
                list.list.ResourceBarrier(&[d3d12x::host::transition(&result, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                list.list.ResourceBarrier(&[d3d12x::host::transition(&texture, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, state)]);
                list.list.CopyBufferRegion(&readback, 0, &result, 0, 16);
            }
            list.close();
            host.execute(&[&list]);
            let mut mapped = std::ptr::null_mut();
            // SAFETY: the finished copy's 16 bytes.
            let got: [f32; 4] = unsafe {
                readback.Map(0, None, Some(&mut mapped)).unwrap();
                mapped.cast::<[f32; 4]>().read()
            };
            for i in 0..compared {
                assert!((got[i] - expected[i]).abs() <= tolerance, "{what}: read {got:?}, expected {expected:?}");
            }
            let release = slot::<extern "system" fn(This) -> u32>(this, 0x10);
            release(this);
        }
        assert_eq!(errors(), before, "no new validation errors");
        let counts: Vec<u64> = crate::d3d12x::descriptor::TYPELESS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        println!("typeless views remapped srv/uav/rtv/dsv, refused: {counts:?}");
        assert!(counts[0] > 0 && counts[2] > 0 && counts[3] > 0);
    }
}

/// The heap-bounds arithmetic (CPU only, no device).
#[cfg(test)]
mod guard_tests {
    use super::*;

    #[test]
    fn guard_slots_sit_below_the_sink_and_out_of_dynamic_reach() {
        let g = guard_slots(16, DYNAMIC).unwrap();
        let last = 16 + DYNAMIC - 1;
        assert_eq!(g, [last - 5, last - 4, last - 3, last - 2, last - 1], "five guards, then the sink");
        // `dynamic` searches slots below `dynamic - 1 - GUARDS` only.
        assert!(g.iter().all(|&s| s - 16 >= DYNAMIC - 1 - GUARDS));
        assert_eq!(guard_slots(16, GUARDS + 1), None, "no room: no guards");
    }

    #[test]
    fn missing_heaps_bound_every_index() {
        assert_eq!(bounds_words(None, None), [0; 8], "0 slots: every index out of range, guarded");
    }
}
