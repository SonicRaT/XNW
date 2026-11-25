//! The frame pack's list sink. The title
//! binds a draw's tables, root constants and pipeline in its flush after
//! the geometry-draw hook, so the binding is only current where the list
//! records the draw. A registered tap sees each draw there (also in null
//! mode, before the host pipeline check) with the title's pipeline, heaps
//! and root arguments, and resolves descriptor slots to the views the
//! title made. Off (no tap registered): one relaxed load per draw.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::descriptor::{self, View};
use super::object::This;
use super::{pipeline, resource};

/// A root argument as the title set it, at a draw.
#[derive(Clone, Copy, Debug)]
pub enum TapRoot<'a> {
    None,
    /// A descriptor table: its first slot's handle, the heap's first slot,
    /// and how many 32-byte slots from `handle` lie in the heap (capped).
    Table { handle: u64, heap_start: u64, slots: u32 },
    Constants(&'a [u32]),
    /// A root view's GPU address.
    View(u64),
}

/// One draw at the list sink.
pub struct TapDraw<'a> {
    /// `xnw_formats::framepack::op`.
    pub op: u8,
    /// Mesh groups, indices or vertices; a dispatch's x groups.
    pub count: u32,
    /// A dispatch's y and z groups (draws: 0).
    pub yz: [u32; 2],
    /// Draw and DrawIndexed: instances, first vertex (or index), base
    /// vertex (indexed, as i32), first instance; other ops 0.
    pub args: [u32; 4],
    /// The title's pipeline object (ours).
    pub pipeline: usize,
    /// The bound view heap object (0: none).
    pub view_heap: usize,
    /// The bound sampler heap object (0: none).
    pub sampler_heap: usize,
    pub roots: &'a [TapRoot<'a>],
}

pub type Tap = fn(&TapDraw);

static TAP: AtomicUsize = AtomicUsize::new(0);

/// Registers (or with `None` removes) the tap.
pub fn set_tap(f: Option<Tap>) {
    TAP.store(f.map_or(0, |f| f as usize), Ordering::Release);
}

pub(crate) fn get() -> Option<Tap> {
    let f = TAP.load(Ordering::Acquire);
    // SAFETY: only `set_tap` stores, a `Tap` or 0.
    (f != 0).then(|| unsafe { std::mem::transmute::<usize, Tap>(f) })
}

/// Which pipelines' direct dispatches go to the tap (compute roots,
/// op DISPATCH). None registered: one relaxed load per dispatch.
pub type DispatchWant = fn(pipeline: usize) -> bool;

static DISPATCH: AtomicUsize = AtomicUsize::new(0);

pub fn set_dispatch_want(f: Option<DispatchWant>) {
    DISPATCH.store(f.map_or(0, |f| f as usize), Ordering::Release);
}

pub(crate) fn dispatch_want() -> Option<DispatchWant> {
    let f = DISPATCH.load(Ordering::Relaxed);
    // SAFETY: only `set_dispatch_want` stores, a `DispatchWant` or 0.
    (f != 0).then(|| unsafe { std::mem::transmute::<usize, DispatchWant>(f) })
}

/// A second consumer of direct dispatches only (no draw reaches it): the
/// pipelines it wants and its tap. None registered: one relaxed load per
/// dispatch.
static DISPATCH_TAP: [AtomicUsize; 2] = [AtomicUsize::new(0), AtomicUsize::new(0)];

pub fn set_dispatch_tap(f: Option<(DispatchWant, Tap)>) {
    DISPATCH_TAP[1].store(f.map_or(0, |f| f.1 as usize), Ordering::Release);
    DISPATCH_TAP[0].store(f.map_or(0, |f| f.0 as usize), Ordering::Release);
}

pub(crate) fn dispatch_tap() -> Option<(DispatchWant, Tap)> {
    let w = DISPATCH_TAP[0].load(Ordering::Acquire);
    let t = DISPATCH_TAP[1].load(Ordering::Acquire);
    // SAFETY: only `set_dispatch_tap` stores, a `DispatchWant` and a `Tap`, or 0.
    (w != 0 && t != 0).then(|| unsafe { (std::mem::transmute::<usize, DispatchWant>(w), std::mem::transmute::<usize, Tap>(t)) })
}

/// A second consumer of draws (the live pass's instanced vehicle reads):
/// the pipelines it wants and its tap. None registered: one relaxed load
/// per draw.
static DRAW_TAP: [AtomicUsize; 2] = [AtomicUsize::new(0), AtomicUsize::new(0)];

pub fn set_draw_tap(f: Option<(DispatchWant, Tap)>) {
    DRAW_TAP[1].store(f.map_or(0, |f| f.1 as usize), Ordering::Release);
    DRAW_TAP[0].store(f.map_or(0, |f| f.0 as usize), Ordering::Release);
}

pub(crate) fn draw_tap() -> Option<(DispatchWant, Tap)> {
    let w = DRAW_TAP[0].load(Ordering::Acquire);
    if w == 0 {
        return None;
    }
    let t = DRAW_TAP[1].load(Ordering::Acquire);
    // SAFETY: only `set_draw_tap` stores, a `DispatchWant` and a `Tap`, or 0.
    (t != 0).then(|| unsafe { (std::mem::transmute::<usize, DispatchWant>(w), std::mem::transmute::<usize, Tap>(t)) })
}

/// Slots a table root reaches at most (T8 is 0xB40 bytes = 90 slots).
pub const TABLE_SLOTS: u32 = 96;
/// A view heap's slot size (bl[k] << 5).
pub const SLOT: u64 = 32;

/// How many slots from `handle` lie in `heap` (at most `max`).
pub(crate) fn slots_in(h: &descriptor::Heap, handle: u64, max: u32) -> u32 {
    (0..max).take_while(|k| h.index(handle + SLOT * u64::from(*k)).is_some()).count() as u32
}

/// The first slot of a view heap object, and the handle of its slot
/// `index` when that lies in the heap.
pub fn heap_slot(view_heap: usize, index: u32) -> Option<(u64, u64)> {
    if view_heap == 0 {
        return None;
    }
    let h = descriptor::heap(view_heap as This);
    let handle = h.start.checked_add(SLOT * u64::from(index))?;
    (h.index(handle) == Some(index)).then_some((h.start, handle))
}

/// A sampler heap's slot size (16-byte entries).
pub const SAMPLER_SLOT: u64 = 16;

/// A sampler heap object's first slot and slot count (None: no heap, or
/// not a sampler heap).
pub fn sampler_heap(heap: usize) -> Option<(u64, u32)> {
    if heap == 0 {
        return None;
    }
    let h = descriptor::heap(heap as This);
    h.samplers().then(|| (h.start, (h.end().saturating_sub(h.start) / SAMPLER_SLOT).min(u64::from(u32::MAX)) as u32))
}

/// The sampler description behind a sampler heap slot's 16 bytes, when
/// they are a record this device wrote: (13, the D3D12 description's
/// words) or (27, the console description's words), padded to 27.
pub fn sampler(raw: [u32; 4]) -> Option<(u32, [u32; 27])> {
    // The record reader may look at a second 16 bytes: give it zeros.
    let rec = [raw[0], raw[1], raw[2], raw[3], 0, 0, 0, 0];
    match descriptor::view(rec.as_ptr() as u64)? {
        View::Sampler(w) => {
            let mut out = [0; 27];
            out[..13].copy_from_slice(&w);
            Some((13, out))
        }
        View::ConsoleSampler(w) => Some((27, w)),
        _ => None,
    }
}

/// One stage of a pipeline: its code's identity.
#[derive(Clone, Debug)]
pub struct TapStage {
    pub name: &'static str,
    /// A DXBC container (else a cache code block).
    pub container: bool,
    /// `xnw_core::hash128` of the bytes the title passed (PackIndex::by_hash).
    pub hash: u128,
    pub size: usize,
    pub prefix: [u8; 24],
}

/// The stages of the title's pipeline object `p` (ours).
pub fn stages(p: usize) -> Vec<TapStage> {
    let Some(d) = pipeline::desc(p as This) else { return Vec::new() };
    d.stages
        .iter()
        .map(|s| {
            let bytes = s.bytes();
            let hash = s.shader.as_ref().map_or_else(|| xnw_core::hash128(&[bytes]), |sh| sh.hash);
            let mut prefix = [0u8; 24];
            let n = bytes.len().min(24);
            prefix[..n].copy_from_slice(&bytes[..n]);
            TapStage { name: s.name, container: s.form == pipeline::Form::Container, hash, size: bytes.len(), prefix }
        })
        .collect()
}

/// Stage `i`'s bytes as the title passed them.
pub fn stage_bytes(p: usize, i: usize) -> Option<Vec<u8>> {
    pipeline::desc(p as This)?.stages.get(i).map(|s| s.bytes().to_vec())
}

/// A resolved descriptor slot (what the title described there).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TapView {
    /// `xnw_formats::framepack::view`.
    pub kind: u32,
    /// Constant buffer or texel address.
    pub addr: u64,
    /// Constant buffer size, or the texture's bytes in console memory.
    pub bytes: u64,
    /// Resource object (0 for placed views and constants).
    pub resource: u64,
    pub width: u64,
    pub height: u32,
    pub depth: u16,
    pub mips: u16,
    pub format: u32,
    pub layout: u32,
    /// An unordered access view (the GPU writes it).
    pub unordered: bool,
    /// Placed views: resource description (14) then view description (16).
    pub words: [u32; 30],
}

/// The view at descriptor slot `handle`, which the caller took from
/// [`TapRoot::Table`] (within its `slots`) or [`heap_slot`].
pub fn view(handle: u64) -> TapView {
    use xnw_formats::framepack::view as k;
    let texture = |kind: u32, addr: u64, d: &resource::ResourceDesc, res: u64| TapView {
        kind,
        addr,
        bytes: resource::footprint_bytes(d),
        resource: res,
        width: d.width,
        height: d.height,
        depth: d.depth_or_array_size,
        mips: d.mip_levels,
        format: d.format,
        layout: d.layout,
        unordered: false,
        words: [0; 30],
    };
    match descriptor::view(handle) {
        None => TapView::default(),
        Some(View::Constant(addr, size)) => TapView { kind: k::CONSTANT, addr, bytes: u64::from(size), ..TapView::default() },
        Some(v @ (View::Shader(r, _) | View::Unordered(r, _, _))) => {
            if let Some((data, d)) = resource::texture_memory(r as This) {
                texture(k::TEXTURE, data as u64, &d, r as u64)
            } else if let Some((data, n)) = resource::buffer_bytes(r as This) {
                // The view's own range: D3D12 buffer view
                // descriptions as words; SRV: format 0, dimension 1 (1 =
                // buffer), FirstElement 4-5, NumElements 6, stride 7; UAV:
                // FirstElement 2-3, NumElements 4, stride 5. Element size:
                // the stride, else the format's, else 4 (raw).
                let (unordered, desc) = match &v {
                    View::Shader(_, d) => (false, *d),
                    View::Unordered(_, _, d) => (true, *d),
                    _ => (false, None),
                };
                let mut out = TapView { kind: k::BUFFER, addr: data as u64, bytes: n, resource: r as u64, unordered, ..TapView::default() };
                if let Some(w) = desc.filter(|w| w[1] == 1) {
                    let (first, count, stride) = if unordered { (u64::from(w[2]) | u64::from(w[3]) << 32, w[4], w[5]) } else { (u64::from(w[4]) | u64::from(w[5]) << 32, w[6], w[7]) };
                    let elem = if stride != 0 { u64::from(stride) } else { resource::element(w[0]).map_or(4, |e| e.0) };
                    let start = first.saturating_mul(elem).min(n);
                    out.addr = data as u64 + start;
                    out.bytes = u64::from(count).saturating_mul(elem).min(n - start);
                    out.format = w[0];
                    out.width = elem;
                    out.words[14..24].copy_from_slice(&w);
                }
                out
            } else {
                TapView { kind: k::OTHER, resource: r as u64, ..TapView::default() }
            }
        }
        Some(View::Placed(_, res, desc)) => {
            // SAFETY: 56 bytes in the description's layout (as `texture_at`).
            let d = unsafe { res.as_ptr().cast::<resource::ResourceDesc>().read_unaligned() };
            let addr = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let mut v = texture(if res[0] == 1 { k::PLACED_BUFFER } else { k::PLACED }, addr, &d, 0);
            v.words[..14].copy_from_slice(&res);
            v.words[14..].copy_from_slice(&desc);
            v
        }
        Some(View::Sampler(_) | View::ConsoleSampler(_)) => TapView { kind: k::SAMPLER, ..TapView::default() },
    }
}

/// Bytes from `address` to the end of the title buffer holding it (the
/// root view rule of draw.rs `user_value`), if a buffer does.
pub fn buffer_left(address: u64) -> Option<u64> {
    super::shadow::buffer_left(address)
}
