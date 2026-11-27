//! The frame pack's list sink. The title
//! binds a draw's tables, root constants and pipeline in its flush after
//! the geometry-draw hook, so the binding is only current where the list
//! records the draw. A registered tap sees each draw there (also in null
//! mode, before the host pipeline check) with the title's pipeline, heaps
//! and root arguments, and resolves descriptor slots to the views the
//! title made. Off (no tap registered): one relaxed load per draw.
//!
//! A burst replay (`xnw burst`, BURST2) drives the same consumers without
//! the title: [`serve`] puts a captured frame's objects (pipelines' stages,
//! heaps, resolved slots) in place of our device's, and [`replay`] hands a
//! sink rebuilt from the capture to the registered consumers as the list
//! sink does. Off: one relaxed load per query.

use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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

/// HAIR6: which pipelines' graphics draws go to the tap (None registered:
/// every draw, as before). The live hair path asks for the strand draws only.
static TAP_WANT: AtomicUsize = AtomicUsize::new(0);

pub fn set_tap_want(f: Option<DispatchWant>) {
    TAP_WANT.store(f.map_or(0, |f| f as usize), Ordering::Release);
}

pub(crate) fn tap_want() -> Option<DispatchWant> {
    let f = TAP_WANT.load(Ordering::Acquire);
    // SAFETY: only `set_tap_want` stores, a `DispatchWant` or 0.
    (f != 0).then(|| unsafe { std::mem::transmute::<usize, DispatchWant>(f) })
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

/// The list sink's second consumers for a sink rebuilt from a capture
/// (draw.rs `tap` / `tap_dispatch` without the pack's tap): a dispatch to
/// the dispatch tap, a draw to the draw tap, each when it wants the
/// pipeline.
pub fn replay(d: &TapDraw) {
    let tap = if d.op == xnw_formats::framepack::op::DISPATCH { dispatch_tap() } else { draw_tap() };
    if let Some((want, f)) = tap
        && want(d.pipeline)
    {
        f(d);
    }
}

/// LIVE1: the frame pack's own tap for a sink rebuilt from a capture, as
/// draw.rs `tap` / `tap_dispatch` hand it (a draw through the tap's draw
/// filter with its IA views; a direct dispatch when the pack wants the
/// pipeline; an indirect dispatch without a draw filter), with the title's (phase, stage)
/// the sink was recorded in ([`title_stage`] while it runs). True when handed.
pub fn replay_pack(d: &TapDraw, ia: Option<TapIa>, stage: Option<(u32, u32)>) -> bool {
    use xnw_formats::framepack::op;
    let Some(f) = get() else { return false };
    let wanted = match d.op {
        op::INDIRECT_DISPATCH => tap_want().is_none(),
        op::DISPATCH => dispatch_want().is_some_and(|want| want(d.pipeline)),
        _ => tap_want().is_none_or(|want| want(d.pipeline)),
    };
    if !wanted {
        return false;
    }
    set_ia(ia.filter(|_| matches!(d.op, op::DRAW | op::INDEXED)));
    STAGE.with(|c| c.set(stage));
    f(d);
    STAGE.with(|c| c.set(None));
    set_ia(None);
    true
}

/// A captured frame's objects, served in place of our device's while a
/// burst runs: pipelines by their object number (the capture's), each
/// stage with its bytes; view and sampler heaps by number (first slot,
/// slot count); resolved slots by handle; title buffers (start, bytes)
/// for root views.
#[derive(Default)]
pub struct Served {
    pub pipelines: HashMap<usize, Vec<(TapStage, Vec<u8>)>>,
    pub targets: HashMap<usize, (Vec<u32>, u32)>,
    pub view_heaps: HashMap<usize, (u64, u32)>,
    pub sampler_heaps: HashMap<usize, (u64, u32)>,
    pub views: HashMap<u64, TapView>,
    pub buffers: std::collections::BTreeMap<u64, u64>,
    /// LIVE1: graphics pipelines' state bytes (the capture's GSTATE records).
    pub gstates: HashMap<usize, Vec<u8>>,
    /// LIVE1: sampler records as the capture resolved them (raw words -> form, words).
    pub samplers: HashMap<[u32; 4], (u32, [u32; 27])>,
}

static SERVING: AtomicBool = AtomicBool::new(false);
static SERVED: RwLock<Option<Served>> = RwLock::new(None);

/// Serves `s` (None: our device's objects again).
pub fn serve(s: Option<Served>) {
    let on = s.is_some();
    if let Ok(mut g) = SERVED.write() {
        *g = s;
    }
    SERVING.store(on, Ordering::Release);
}

/// `f` over the served objects while a burst serves them (None: not serving).
fn served<T>(f: impl FnOnce(&Served) -> T) -> Option<T> {
    if !SERVING.load(Ordering::Relaxed) {
        return None;
    }
    let g = SERVED.read().ok()?;
    g.as_ref().map(f)
}

/// Slots a table root reaches at most (the lighting programs read the
/// frame table up to slot 181; the pack keeps 96 of other tables).
pub const TABLE_SLOTS: u32 = 192;
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
    if let Some(got) = served(|s| s.view_heaps.get(&view_heap).and_then(|&(start, count)| (index < count).then(|| (start, start + SLOT * u64::from(index))))) {
        return got;
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
    if let Some(got) = served(|s| s.sampler_heaps.get(&heap).copied()) {
        return got;
    }
    let h = descriptor::heap(heap as This);
    h.samplers().then(|| (h.start, (h.end().saturating_sub(h.start) / SAMPLER_SLOT).min(u64::from(u32::MAX)) as u32))
}

/// The sampler description behind a sampler heap slot's 16 bytes, when
/// they are a record this device wrote: (13, the D3D12 description's
/// words) or (27, the console description's words), padded to 27.
pub fn sampler(raw: [u32; 4]) -> Option<(u32, [u32; 27])> {
    if let Some(got) = served(|s| s.samplers.get(&raw).copied()) {
        return got;
    }
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

/// UI1: the input assembler's views at a Draw or DrawIndexed sink: the
/// index view (address, bytes, DXGI format) and the vertex views of the
/// slots the pipeline's input layout reads (address, bytes, stride; slot
/// k of `vertex` is IA slot k, `slots` the mask of slots the layout reads).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TapIa {
    pub index: (u64, u32, u32),
    pub vertex: [(u64, u32, u32); IA_SLOTS],
    pub slots: u32,
    /// The title's primitive topology (D3D numbering; console-only kinds as the title set them).
    pub topology: u32,
}

/// Vertex buffer slots [`TapIa`] carries.
pub const IA_SLOTS: usize = 4;

thread_local! {
    static IA: std::cell::Cell<Option<TapIa>> = const { std::cell::Cell::new(None) };
    /// The (phase, stage) of a sink [`replay_pack`] hands (None: the title's TLS).
    static STAGE: std::cell::Cell<Option<(u32, u32)>> = const { std::cell::Cell::new(None) };
}

/// Sets (draw.rs, around a Draw or DrawIndexed tap) the IA views the
/// tapped draw reads; None after it.
pub(crate) fn set_ia(ia: Option<TapIa>) {
    IA.with(|c| c.set(ia));
}

/// The IA views of the Draw or DrawIndexed sink being tapped on this
/// thread (None for other ops and outside a tap).
pub fn ia() -> Option<TapIa> {
    IA.with(|c| c.get())
}

/// UI1: the title's (pass phase, stage) on this thread, read from its TLS
/// as `ui=keep` reads it (uikeep.rs; phase u32::MAX = outside a pass).
pub fn title_stage() -> Option<(u32, u32)> {
    if let Some(st) = STAGE.with(|c| c.get()) {
        return Some(st);
    }
    super::uikeep::here()
}

/// UI1: a graphics pipeline's host state as the warm cache encodes it
/// (blend, rasterizer, depth, topology, targets, input layout; decode with
/// [`PipeState::decode`]); None for compute pipelines and unknown objects
/// (a burst serves the capture's).
pub fn graphics_state(p: usize) -> Option<Vec<u8>> {
    if let Some(got) = served(|s| s.gstates.get(&p).cloned()) {
        return got;
    }
    pipeline::graphics_state(p as This).map(|g| g.encode())
}

/// One input element of a pipeline's layout: semantic, index, DXGI
/// format, IA slot, byte offset (worked out), per instance, step rate.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PipeInput {
    pub name: String,
    pub index: u32,
    pub format: u32,
    pub slot: u32,
    pub offset: u32,
    pub per_instance: bool,
    pub step: u32,
}

/// What a replay needs of a pipeline's [`graphics_state`] bytes: render
/// target 0's blend (enable, source, destination, op, source alpha,
/// destination alpha, alpha op, write mask; D3D12 numbering), the cull
/// mode, the topology type, the colour formats and the input layout.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PipeState {
    pub blend: [u32; 8],
    pub cull: u32,
    pub topology: u32,
    pub targets: Vec<u32>,
    pub depth_format: u32,
    pub inputs: Vec<PipeInput>,
}

impl PipeState {
    pub fn decode(bytes: &[u8]) -> Option<PipeState> {
        let g = pipeline::host::GraphicsState::decode(bytes)?;
        let t = &g.blend.RenderTarget[0];
        let n = (g.targets.NumRenderTargets as usize).min(8);
        Some(PipeState {
            blend: [t.BlendEnable.0 as u32, t.SrcBlend.0 as u32, t.DestBlend.0 as u32, t.BlendOp.0 as u32, t.SrcBlendAlpha.0 as u32, t.DestBlendAlpha.0 as u32, t.BlendOpAlpha.0 as u32, u32::from(t.RenderTargetWriteMask)],
            cull: g.rasterizer.CullMode.0 as u32,
            topology: g.topology.0 as u32,
            targets: g.targets.RTFormats[..n].iter().map(|f| f.0 as u32).collect(),
            depth_format: g.depth_format.0 as u32,
            inputs: g.input_list().into_iter().map(|(name, index, format, slot, offset, per_instance, step)| PipeInput { name, index, format, slot, offset, per_instance, step }).collect(),
        })
    }
}

/// The stages of the title's pipeline object `p` (ours).
pub fn stages(p: usize) -> Vec<TapStage> {
    if let Some(got) = served(|s| s.pipelines.get(&p).map(|v| v.iter().map(|x| x.0.clone()).collect()).unwrap_or_default()) {
        return got;
    }
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

/// A pipeline's render-target formats and depth format (DXGI), if known.
pub fn targets(p: usize) -> Option<(Vec<u32>, u32)> {
    if let Some(got) = served(|s| s.targets.get(&p).cloned()) {
        return got;
    }
    pipeline::desc(p as This).map(|d| (d.targets.clone(), d.depth))
}

/// Stage `i`'s bytes as the title passed them.
pub fn stage_bytes(p: usize, i: usize) -> Option<Vec<u8>> {
    if let Some(got) = served(|s| s.pipelines.get(&p).and_then(|v| v.get(i)).map(|x| x.1.clone())) {
        return got;
    }
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
    if let Some(got) = served(|s| s.views.get(&handle).copied().unwrap_or_default()) {
        return got;
    }
    match descriptor::view(handle) {
        None => TapView::default(),
        Some(View::Constant(addr, size)) => TapView { kind: k::CONSTANT, addr, bytes: u64::from(size), ..TapView::default() },
        Some(v @ (View::Shader(r, _) | View::Unordered(r, _, _))) => {
            if let Some((data, d)) = resource::texture_memory(r as This) {
                // The resource's description and the view's own (None: the
                // resource's default view, zeros) in a placed view's places,
                // so a reader finds the dimensions where it does for those:
                // words[0] the resource's, words[15] the view's (4 2D, 5 2D
                // array, 8 3D, 9 cube, 10 cube array). Before BUG3 a
                // resource view's words were zeros and it was taken as 2D.
                let mut out = texture(k::TEXTURE, data as u64, &d, r as u64);
                out.words[..14].copy_from_slice(&desc_words(&d));
                if let View::Shader(_, Some(w)) | View::Unordered(_, _, Some(w)) = &v {
                    out.words[14..24].copy_from_slice(w);
                }
                out
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

/// A resource description as the 14 words a placed view carries it in
/// (`ResourceDesc`'s 56 bytes, padding as zeros).
fn desc_words(d: &resource::ResourceDesc) -> [u32; 14] {
    [d.dimension, 0, d.alignment as u32, (d.alignment >> 32) as u32, d.width as u32, (d.width >> 32) as u32, d.height, u32::from(d.depth_or_array_size) | u32::from(d.mip_levels) << 16, d.format, d.sample_count, d.sample_quality, d.layout, d.flags, 0]
}

/// Bytes from `address` to the end of the title buffer holding it (the
/// root view rule of draw.rs `user_value`), if a buffer does.
pub fn buffer_left(address: u64) -> Option<u64> {
    if let Some(got) = served(|s| s.buffers.range(..=address).next_back().and_then(|(&at, &n)| (address < at + n).then(|| at + n - address))) {
        return got;
    }
    super::shadow::buffer_left(address)
}

/// A burst's served objects answer the queries, and the replayed sink
/// reaches the draw tap only for a pipeline it wants.
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    static SEEN: AtomicU32 = AtomicU32::new(0);

    fn want(p: usize) -> bool {
        p == 7
    }

    fn got(d: &TapDraw) {
        if let Some(TapRoot::Table { handle, .. }) = d.roots.first() {
            SEEN.fetch_add(view(*handle + SLOT).kind, Ordering::Relaxed);
        }
    }

    #[test]
    fn served_objects_and_replay() {
        let mut s = Served::default();
        let st = TapStage { name: "pixel", container: true, hash: 5, size: 3, prefix: [0; 24] };
        s.pipelines.insert(7, vec![(st, vec![1, 2, 3])]);
        s.view_heaps.insert(9, (0x1000, 4));
        s.views.insert(0x1020, TapView { kind: 3, addr: 0x5000, bytes: 64, ..TapView::default() });
        s.buffers.insert(0x8000, 0x100);
        s.sampler_heaps.insert(11, (0x3000, 2));
        serve(Some(s));
        assert_eq!(stages(7).len(), 1);
        assert_eq!(stage_bytes(7, 0), Some(vec![1, 2, 3]));
        assert!(stages(8).is_empty(), "a pipeline the capture does not name has no stages");
        assert_eq!(heap_slot(9, 1), Some((0x1000, 0x1020)));
        assert_eq!(heap_slot(9, 4), None, "past the heap");
        assert_eq!(view(0x1020).addr, 0x5000);
        assert_eq!(view(0x1040).kind, 0, "a slot the capture did not resolve is empty");
        assert_eq!(buffer_left(0x8010), Some(0xF0));
        assert_eq!(buffer_left(0x8100), None);
        assert_eq!(sampler_heap(11), Some((0x3000, 2)));
        set_draw_tap(Some((want, got)));
        let roots = [TapRoot::Table { handle: 0x1000, heap_start: 0x1000, slots: 4 }];
        let d = |pipeline| TapDraw { op: xnw_formats::framepack::op::MESH, count: 1, yz: [0; 2], args: [0; 4], pipeline, view_heap: 9, sampler_heap: 0, roots: &roots };
        replay(&d(7));
        replay(&d(8));
        set_draw_tap(None);
        serve(None);
        assert_eq!(SEEN.load(Ordering::Relaxed), 3, "the wanted pipeline's sink only, its slot 1 served");
    }
}
