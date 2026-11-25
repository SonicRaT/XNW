//! `xnw burst <capture dir> --adapter hw|warp [--from N] [--frames K]
//! [--switches a,b=c] [--png DIR] [--png-at N,..]`: a captured burst of
//! consecutive frames through the live list and pass functions, at full
//! speed, without the title (BURST1).
//!
//! Per frame, as the live path does after the hook:
//! - the list's geometry step is the live `geomread::take` (asks, copies,
//!   decode; `geomasync=on` hands keys to the same reader thread,
//!   `geomcopy=direct` takes the same copy) over the frame's rows;
//! - the coherence line is the live `coherence`;
//! - the pass is `xnw_world::screen_pass` / `screen_draw`, the functions the
//!   live screen path calls: the screen budgets, the cut watch and hold,
//!   asynchronous pipelines, the live log lines (cut, slow frame, every 60).
//!
//! Title memory is served from the capture (`read`, consulted by the
//! probe's `bytes` and `dcopy` while a burst runs): a table of captured
//! objects at their title addresses. The capture keeps a geometry's raw
//! vertex buffer and its expanded indices (blobs `k-<mesh>-<hash>.{vb,idx}`);
//! the meshlets are made again (`geomread::meshlets`) and served whole at
//! the mesh address. A read the table does not cover is unreadable, as a
//! fault would make it, and counted.
//!
//! The list sink's consumers (BURST2) run as live from a v2 capture's sink
//! records: `taps_of` rebuilds each sink's `TapDraw` (roots from kinds 1, 2,
//! 4; args 51; dispatch y, z 44) and serves the objects it names in place
//! of our device's (`tap::serve`: pipelines' stages from programs.tsv,
//! resolved slots by handle, the view and sampler heaps, root-view buffers)
//! and the bytes behind them (kinds 3, 4, 7, 17, 46, 48, 49, 53, sampler
//! words) in a per-frame table; `drive` hands the sinks to the registered
//! consumers (`tap::replay`, as the list sink does: wheels, PreSkinD join,
//! bridged rows, pre-skin dispatches), each sink's hook draw standing for
//! "this thread's last hooked draw" (`attach`); `apply` puts what they gave
//! on the list as the live list build does (`place_wheel`, `frame_for`,
//! `bridged::Frame`) and prints the live lines. `gpucull=on` rebuilds the
//! batches as the replay does (`xnw_frame::gpucull::fill`).
//!
//! Not driven: the hook (`capture_light` reads the TLS block, the render
//! context and the geometry headers, which no capture holds: spec on the
//! feed); live textures
//! (`TexMode::Live` resolves through our layer's resource objects).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use xnw_world::GeomKey;

use super::{Draw, burst_mem, coherence, geomread};

static SERVING: AtomicBool = AtomicBool::new(false);

/// Whether reads of title memory are served from a burst's table.
pub(super) fn serving() -> bool {
    SERVING.load(Ordering::Relaxed)
}

struct Entry {
    bytes: Arc<Vec<u8>>,
    /// A read at the entry's start gets all of it, whatever length it asks
    /// (meshlets made again need not be the title's length).
    whole: bool,
}

/// Captured objects by title address.
static MEM: RwLock<BTreeMap<u64, Entry>> = RwLock::new(BTreeMap::new());
/// The frame's pack objects by title address (BURST2: the sink consumers'
/// reads), replaced every frame.
static FRAME_MEM: RwLock<BTreeMap<u64, Arc<Vec<u8>>>> = RwLock::new(BTreeMap::new());
/// Reads served, bytes served, reads not covered.
static COUNTS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// `len` bytes of title memory at `addr` from the tables (None: not
/// captured): the frame's pack objects first, then the geometries. Moved by
/// `copy(to, from)`, the live read's own copy (the kernel's, `XNW_PLAINCOPY`'s
/// in place or the fault-fixed one), so a burst pays what the live read pays
/// for the same bytes.
pub(super) fn read(addr: u64, len: usize, copy: fn(&mut [u8], *const u8) -> bool) -> Option<Vec<u8>> {
    let moved = |from: &[u8]| {
        let mut v = vec![0u8; from.len()];
        copy(&mut v, from.as_ptr()).then_some(v)
    };
    let got = FRAME_MEM.read().ok().and_then(|m| covering(&m, addr, len).and_then(moved)).or_else(|| {
        MEM.read().ok().and_then(|m| {
            let (start, e) = m.range(..=addr).next_back()?;
            let off = (addr - start) as usize;
            let from = if e.whole && off == 0 { &e.bytes[..] } else { e.bytes.get(off..off.checked_add(len)?)? };
            moved(from)
        })
    });
    match &got {
        Some(b) => {
            COUNTS[0].fetch_add(1, Ordering::Relaxed);
            COUNTS[1].fetch_add(b.len() as u64, Ordering::Relaxed);
            burst_mem::note(addr, b.len());
        }
        None => {
            COUNTS[2].fetch_add(1, Ordering::Relaxed);
        }
    }
    got
}

/// The bytes of the nearest entry at or below `addr` that holds all of
/// `[addr, addr + len)` (a short object inside a long one does not hide it;
/// at most 64 entries back).
fn covering(m: &BTreeMap<u64, Arc<Vec<u8>>>, addr: u64, len: usize) -> Option<&[u8]> {
    let end = addr.checked_add(len as u64)?;
    m.range(..=addr).rev().take(64).find(|(at, b)| **at + b.len() as u64 >= end).map(|(at, b)| &b[(addr - at) as usize..(end - at) as usize])
}

fn place(at: u64, bytes: Vec<u8>, whole: bool) {
    if let Ok(mut m) = MEM.write() {
        m.insert(at, Entry { bytes: Arc::new(bytes), whole });
    }
}

/// A geometry's captured bytes at its title addresses: the vertex buffer,
/// and the meshlets made again (or the classic index buffer at its width).
/// False when the indices cannot be put back in the title's form.
fn place_geometry(key: &GeomKey, vb: Vec<u8>, idx: &[u32]) -> bool {
    let mesh = if key.mesh_len != 0 {
        // The meshlet form holds vertex indices below 65,536 (u16 vertex lists).
        if idx.iter().any(|&i| i > 0xFFFF) {
            return false;
        }
        geomread::meshlets(idx)
    } else {
        match key.ib & 0xFF {
            2 if idx.iter().all(|&i| i <= 0xFFFF) => idx.iter().flat_map(|&i| (i as u16).to_le_bytes()).collect(),
            4 => idx.iter().flat_map(|i| i.to_le_bytes()).collect(),
            _ => return false,
        }
    };
    if burst_mem::mirrored() {
        burst_mem::put(key.vb_at, &vb);
        burst_mem::put(key.mesh_at, &mesh);
    }
    place(key.vb_at, vb, false);
    place(key.mesh_at, mesh, key.mesh_len != 0);
    true
}

/// The hook's record of a row, as far as the geometry step reads it.
fn draw_of(k: &GeomKey) -> Draw {
    Draw { ghash: k.hash, vb_at: k.vb_at, vb_len: k.vb_len, mesh_at: k.mesh_at, mesh_len: k.mesh_len, ib_len: if k.mesh_len == 0 { k.ib } else { 0 }, ..Default::default() }
}

/// The packs' texel providers, opened ahead in frame order by `PACK_THREADS`
/// loader threads (thread t takes frames t, t + T, ...; each holds two ready).
const PACK_THREADS: usize = 4;
type PackRx = std::sync::mpsc::Receiver<Result<Option<xnw_world::textures::Provider>, String>>;
fn prefetch_packs(frames: &[(u64, PathBuf)], counts: Arc<[AtomicU64; 3]>) -> Vec<PackRx> {
    (0..PACK_THREADS.min(frames.len()))
        .map(|t| {
            let (tx, rx) = std::sync::mpsc::sync_channel(2);
            let dirs: Vec<PathBuf> = frames.iter().skip(t).step_by(PACK_THREADS).map(|f| f.1.clone()).collect();
            let counts = counts.clone();
            std::thread::spawn(move || {
                for d in dirs {
                    let got = xnw_frame::pack::Pack::open(&d).map(|p| p.map(|p| p.texel_provider(counts.clone())));
                    if tx.send(got).is_err() {
                        return;
                    }
                }
            });
            rx
        })
        .collect()
}

/// p50, p95 and the largest of `v` (ms).
fn spread(v: &[f64]) -> String {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let at = |q: f64| s.get(((s.len() as f64 - 1.0) * q).round() as usize).copied().unwrap_or(0.0);
    format!("p50 {:.1} ms, p95 {:.1} ms, max {:.1} ms", at(0.5), at(0.95), at(1.0))
}

// ---- BURST2: the list sink's consumers driven from the capture's sinks ----

/// What the sink consumers gave one hook draw: the live `Draw`'s sink
/// fields, with what the consumers ask of the hook's record (skinned, the
/// vertex stream and meshlet buffer) and the hook's range word (the join
/// line's second key).
#[derive(Default)]
pub(super) struct Given {
    pub skinned: bool,
    pub streams: (u64, u64, u64),
    pub range: Option<u32>,
    pub wheel: Option<super::skinhook::WheelInst>,
    pub bridged: Option<Box<xnw_world::bridged::Owned>>,
    pub sink_range: Option<u32>,
}

/// The burst's frame (`frame_key` while serving).
static FRAME_N: AtomicU64 = AtomicU64::new(0);
/// The hook draw the sink being handed belongs to ("this thread's last
/// hooked draw"; None: the sink had none).
static CUR: Mutex<Option<u32>> = Mutex::new(None);
/// The frame's hook draws by v1 draw number.
static GIVEN: Mutex<Option<HashMap<u32, Given>>> = Mutex::new(None);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

pub(super) fn frame() -> u64 {
    FRAME_N.load(Ordering::Relaxed)
}

/// The current sink's hook draw.
pub(super) struct Current {
    pub streams: (u64, u64, u64),
}

pub(super) fn current() -> Option<Current> {
    let d = (*lock(&CUR))?;
    lock(&GIVEN).as_ref()?.get(&d).map(|g| Current { streams: g.streams })
}

/// Gives the current sink's hook draw what `f` sets (false: no hook draw,
/// or `f` refused it), as the live `attach_*` do.
pub(super) fn attach(f: impl FnOnce(&mut Given) -> bool) -> bool {
    let Some(d) = *lock(&CUR) else { return false };
    lock(&GIVEN).as_mut().and_then(|m| m.get_mut(&d)).is_some_and(f)
}

/// A root argument rebuilt from the sink's records.
enum Root {
    None,
    Table(u64),
    Constants(Vec<u32>),
    View(u64),
}

/// One sink rebuilt from the capture: the `TapDraw` the list sink made.
struct SinkDraw {
    op: u8,
    count: u32,
    yz: [u32; 2],
    args: [u32; 4],
    pipeline: usize,
    view_heap: usize,
    sampler_heap: usize,
    hook: Option<u32>,
    roots: Vec<Root>,
}

/// A frame's sinks, the objects they name and the bytes behind them.
struct Taps {
    served: xnw_gpu::d3d12x::tap::Served,
    mem: BTreeMap<u64, Arc<Vec<u8>>>,
    sinks: Vec<SinkDraw>,
    given: HashMap<u32, Given>,
    /// list row -> v1 draw.
    rows: Vec<u32>,
}

fn stage_name(n: &str) -> &'static str {
    match n {
        "vertex" => "vertex",
        "pixel" => "pixel",
        "mesh" => "mesh",
        "amplification" => "amplification",
        "compute" => "compute",
        "geometry" => "geometry",
        "hull" => "hull",
        "domain" => "domain",
        _ => "other",
    }
}

fn tap_view(v: &xnw_formats::framepack::SlotView) -> xnw_gpu::d3d12x::tap::TapView {
    use xnw_formats::framepack::view;
    let unordered = v.kind == view::BUFFER && v.layout == view::UNORDERED;
    xnw_gpu::d3d12x::tap::TapView { kind: v.kind, addr: v.addr, bytes: v.bytes, resource: v.resource, width: v.width, height: v.height, depth: v.depth, mips: v.mips, format: v.format, layout: if unordered { 0 } else { v.layout }, unordered, words: v.words }
}

fn u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Decoded slot blobs by hash, shared by the frames of a burst.
type SlotCache = HashMap<u128, Arc<Vec<xnw_formats::framepack::SlotView>>>;

/// The frame's sinks, objects and bytes from its pack and draws.tsv.
fn taps_of(pack: &xnw_frame::pack::Pack, fdir: &std::path::Path, slots: &mut SlotCache) -> Result<Taps, String> {
    use xnw_formats::framepack::{FRAME_SCOPE, kind, op, reason};
    use xnw_gpu::d3d12x::tap;
    let mut served = tap::Served::default();
    let mut mem: BTreeMap<u64, Arc<Vec<u8>>> = BTreeMap::new();
    let mut blobs: HashMap<u128, Arc<Vec<u8>>> = HashMap::new();
    let mut blob = |h: u128| -> Option<Arc<Vec<u8>>> {
        if let Some(b) = blobs.get(&h) {
            return Some(b.clone());
        }
        let b = Arc::new(pack.blob(h)?);
        blobs.insert(h, b.clone());
        Some(b)
    };
    let put = |mem: &mut BTreeMap<u64, Arc<Vec<u8>>>, at: u64, b: Arc<Vec<u8>>| {
        if at != 0 && !b.is_empty() && mem.get(&at).is_none_or(|o| o.len() < b.len()) {
            mem.insert(at, b);
        }
    };
    // Pipelines: their stages and bytes; render targets.
    for p in &pack.data.programs {
        let bytes = blob(p.container).map(|b| b.to_vec()).unwrap_or_default();
        let st = tap::TapStage { name: stage_name(&p.stage), container: p.form == 1, hash: p.container, size: p.size as usize, prefix: p.prefix };
        served.pipelines.entry(p.pipeline as usize).or_default().push((st, bytes));
    }
    let mut view_heap = 0u64;
    let mut slots_of = |r: &xnw_formats::framepack::Ref, blob: &mut dyn FnMut(u128) -> Option<Arc<Vec<u8>>>| -> Arc<Vec<xnw_formats::framepack::SlotView>> {
        if let Some(v) = slots.get(&r.hash) {
            return v.clone();
        }
        let v = Arc::new(blob(r.hash).map(|b| xnw_formats::framepack::decode_slots(&b)).unwrap_or_default());
        slots.insert(r.hash, v.clone());
        v
    };
    let mut seen: std::collections::HashSet<(u16, u64, u128)> = std::collections::HashSet::new();
    for r in &pack.data.refs {
        if !reason::has_bytes(r.reason) || !seen.insert((r.kind, r.addr, r.hash)) {
            continue;
        }
        match r.kind {
            // Resolved slots at handle + 32 * slot.
            kind::TABLE | kind::HANDLE_VIEW => {
                for v in slots_of(r, &mut blob).iter() {
                    served.views.insert(r.addr + 32 * u64::from(v.slot), tap_view(v));
                }
            }
            kind::HEAP_SLOT if r.owner == FRAME_SCOPE && r.addr != 0 => {
                view_heap = r.addr;
                for v in slots_of(r, &mut blob).iter() {
                    served.views.insert(r.addr + 32 * u64::from(v.slot), tap_view(v));
                }
            }
            kind::DRAW_WORDS if r.slot == 1 && r.addr != 0 => {
                view_heap = r.addr;
                for v in slots_of(r, &mut blob).iter() {
                    served.views.insert(r.addr + 32 * u64::from(v.slot), tap_view(v));
                }
            }
            // One view at the handle, its slot field the heap index.
            kind::HEAP_VIEW | kind::DISP_MAP | kind::STRAND_VIEW => {
                if let Some(v) = slots_of(r, &mut blob).first() {
                    served.views.insert(r.addr, tap_view(v));
                    if view_heap == 0 {
                        view_heap = r.addr - 32 * u64::from(v.slot);
                    }
                }
            }
            kind::TARGETS => {
                if let Some(b) = blob(r.hash) {
                    let w = u32s(&b);
                    if w.len() >= 10 {
                        served.targets.insert(r.addr as usize, (w[1..1 + (w[0] as usize).min(8)].to_vec(), w[9]));
                    }
                }
            }
            kind::SAMPLERS => {
                if let Some(e) = blob(r.hash).and_then(|b| xnw_formats::framepack::decode_samplers(&b)) {
                    let n = e.iter().map(|x| x.index + 1).max().unwrap_or(0);
                    let mut raw = vec![0u8; 16 * n as usize];
                    for x in &e {
                        for (k, w) in x.raw.iter().enumerate() {
                            let at = 16 * x.index as usize + 4 * k;
                            raw[at..at + 4].copy_from_slice(&w.to_le_bytes());
                        }
                    }
                    served.sampler_heaps.insert(r.addr as usize, (r.addr, n));
                    put(&mut mem, r.addr, Arc::new(raw));
                }
            }
            kind::ROOT_VIEW => {
                if let Some(b) = blob(r.hash) {
                    served.buffers.insert(r.addr, b.len() as u64);
                    put(&mut mem, r.addr, b);
                }
            }
            kind::CBV | kind::BUFFER | kind::FRAME_CB | kind::HEAP_BYTES | kind::PAL_WINDOW | kind::STRAND_BYTES => {
                if let Some(b) = blob(r.hash) {
                    put(&mut mem, r.addr, b);
                }
            }
            kind::DRAW_WORDS if r.slot == 0 => {
                if let Some(b) = blob(r.hash) {
                    put(&mut mem, r.addr, b);
                }
            }
            _ => {}
        }
    }
    // The descriptor slots themselves (a consumer reads a slot before it
    // asks what it holds); their bytes are not the title's, only readable.
    let slot = Arc::new(vec![0u8; tap::SLOT as usize]);
    for &h in served.views.keys() {
        mem.entry(h).or_insert_with(|| slot.clone());
    }
    if view_heap != 0 {
        served.view_heaps.insert(view_heap as usize, (view_heap, 1 << 24));
    }
    // Sinks: roots from their records, in root order.
    let mut sinks = Vec::with_capacity(pack.data.sinks.len());
    for sk in &pack.data.sinks {
        let mut roots: Vec<Root> = Vec::new();
        let (mut yz, mut args, mut sampler_heap) = ([0u32; 2], [0u32; 4], 0usize);
        let set = |roots: &mut Vec<Root>, at: u16, r: Root| {
            let at = usize::from(at);
            if roots.len() <= at {
                roots.resize_with(at + 1, || Root::None);
            }
            roots[at] = r;
        };
        for r in pack.data.refs_of(sk.index) {
            match r.kind {
                kind::ROOTS => set(&mut roots, r.slot, Root::Constants(blob(r.hash).map(|b| u32s(&b)).unwrap_or_default())),
                kind::TABLE => set(&mut roots, r.slot, Root::Table(r.addr)),
                kind::ROOT_VIEW => set(&mut roots, r.slot, Root::View(r.addr)),
                kind::DRAW_ARGS => {
                    if let Some(w) = blob(r.hash).map(|b| u32s(&b)).filter(|w| w.len() >= 4) {
                        args = [w[0], w[1], w[2], w[3]];
                    }
                }
                kind::DISPATCH | kind::GROUPS => {
                    if let Some(w) = blob(r.hash).map(|b| u32s(&b)).filter(|w| w.len() >= 3) {
                        yz = [w[1], w[2]];
                    }
                }
                kind::SAMPLERS if r.addr != 0 => sampler_heap = r.addr as usize,
                _ => {}
            }
        }
        let hook = if op::is_dispatch(sk.op) { None } else { sk.hook_draw };
        sinks.push(SinkDraw { op: sk.op, count: sk.count, yz, args, pipeline: sk.pipeline as usize, view_heap: view_heap as usize, sampler_heap, hook, roots });
    }
    // The hook draws: what the consumers ask of the hook's record.
    let tsv = std::fs::read_to_string(fdir.join("draws.tsv")).map_err(|e| format!("draws.tsv: {e}"))?;
    let mut lines = tsv.lines();
    let head: Vec<&str> = lines.next().unwrap_or_default().split('\t').collect();
    let col = |n: &str| head.iter().position(|h| *h == n);
    let hex = |s: Option<&&str>| s.and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok());
    let (cd, cs, cv, cvn, cm, cr) = (col("draw"), col("model_skinned"), col("vb"), col("vb_bytes"), col("mesh_data"), col("range"));
    let mut given = HashMap::new();
    for l in lines {
        let f: Vec<&str> = l.split('\t').collect();
        let Some(d) = cd.and_then(|c| f.get(c)?.parse::<u32>().ok()) else { continue };
        let at = |c: Option<usize>| hex(c.and_then(|c| f.get(c))).unwrap_or(0);
        let vb_len = cvn.and_then(|c| f.get(c)?.parse().ok()).unwrap_or(0);
        let range = hex(cr.and_then(|c| f.get(c))).filter(|&w| w != 0).map(|w| w as u32);
        given.insert(d, Given { skinned: cs.and_then(|c| f.get(c)).is_some_and(|v| *v == "1"), streams: (at(cv), vb_len, at(cm)), range, ..Default::default() });
    }
    let rows = pack.list_draws().unwrap_or_default();
    Ok(Taps { served, mem, sinks, given, rows })
}

/// Hands the frame's sinks to the registered consumers as the list sink
/// does; returns (sinks handed, ms).
fn drive(taps: &Taps) -> (u32, f64) {
    use xnw_gpu::d3d12x::tap::{self, TapDraw, TapRoot};
    let t0 = Instant::now();
    let mut n = 0u32;
    for s in &taps.sinks {
        let roots: Vec<TapRoot> = s
            .roots
            .iter()
            .map(|r| match r {
                Root::None => TapRoot::None,
                Root::Table(h) => TapRoot::Table { handle: *h, heap_start: s.view_heap as u64, slots: tap::TABLE_SLOTS },
                Root::Constants(c) => TapRoot::Constants(c),
                Root::View(a) => TapRoot::View(*a),
            })
            .collect();
        *lock(&CUR) = s.hook;
        tap::replay(&TapDraw { op: s.op, count: s.count, yz: s.yz, args: s.args, pipeline: s.pipeline, view_heap: s.view_heap, sampler_heap: s.sampler_heap, roots: &roots });
        n += 1;
    }
    *lock(&CUR) = None;
    (n, t0.elapsed().as_secs_f64() * 1e3)
}

/// The bridged rows' pipelines of this frame, made before the frame (the
/// live pass makes them on a thread of their own the first time it sees
/// the pipeline; a burst's few frames would draw none of them).
fn warm_bridged(served: &xnw_gpu::d3d12x::tap::Served, host: &xnw_gpu::host::Host) -> usize {
    let mut n = 0;
    for v in served.pipelines.values() {
        let stages: Vec<_> = v.iter().map(|x| x.0.clone()).collect();
        let Some(key) = super::bridgeps::key_of(&stages, super::bridgeps::wanted()) else { continue };
        if xnw_world::bridged::cache().asked(key) {
            continue;
        }
        let bytes = |name: &str| v.iter().find(|x| x.0.name == name).map(|x| x.1.clone());
        if let (Some(msc), Some(psc)) = (bytes("mesh"), bytes("pixel")) {
            xnw_world::bridged::cache().ask(key, msc, psc, &host.device, &host.caps, true);
            n += 1;
        }
    }
    n
}

/// What the consumers gave, onto the list as the live list build puts it
/// (wheels, the pre-skin join, the bridged rows); the live lines.
fn apply(list: &mut xnw_world::FrameList, taps: &mut Taps, n: u64) -> Vec<String> {
    let (wheels, preskin_on, bridge) = super::sinks_on();
    let mut lines = Vec::new();
    let given = lock(&GIVEN).take().unwrap_or_default();
    let of = |row: usize| taps.rows.get(row).and_then(|d| given.get(d));
    if wheels {
        let (mut pal, mut left, mut placed) = (None, [0u32; 4], 0u32);
        for row in 0..list.draws.len() {
            let Some(w) = of(row).and_then(|g| g.wheel.as_ref()) else { continue };
            let d = &list.draws[row];
            let mut r = super::WheelRow { world: d.world, palette: d.palette, bone_first: d.bone_first, bone_count: d.bone_count, remap: d.remap, inst_first: 0, inst_count: 0 };
            super::place_wheel(list, &mut pal, w, &mut r, &mut left, &mut placed);
            let d = &mut list.draws[row];
            if r.inst_count > 0 {
                (d.world, d.palette, d.bone_first, d.bone_count, d.remap, d.inst_first, d.inst_count) = (r.world, r.palette, r.bone_first, r.bone_count, r.remap, r.inst_first, r.inst_count);
            }
        }
        lines.push(super::skinhook::wheel_line(n, placed, left));
    }
    if preskin_on {
        list.preskin = super::preskin::frame_for(list.frame, list, |row| of(row).and_then(|g| g.sink_range).unwrap_or(u32::MAX));
        let words: Vec<(Option<u32>, Option<u32>)> = (0..list.draws.len()).filter_map(of).map(|g| (g.sink_range, g.range)).collect();
        lines.push(super::preskin::join_line(n, &words, list.preskin.draws.len()));
    }
    if bridge {
        let rows: Vec<(u32, xnw_world::bridged::Owned)> = (0..list.draws.len()).filter_map(|row| of(row)?.bridged.as_deref().map(|o| (row as u32, o.clone()))).collect();
        let (not_ready, unread) = super::bridgeps::take_counts();
        let mut f = xnw_world::bridged::Frame::new(rows);
        (f.not_ready, f.unread) = (not_ready, unread);
        lines.push(super::bridgeps::reads_line(n, f.rows.len()));
        list.bridged = Some(Arc::new(f));
    }
    taps.sinks.clear();
    lines
}

pub fn run(args: &[String]) -> Result<(), String> {
    let usage = "usage: xnw burst <capture dir> --adapter hw|warp [--from N] [--frames K] [--switches a,b=c] [--png DIR] [--png-at N,..]";
    let dir = PathBuf::from(args.first().filter(|a| !a.starts_with("--")).ok_or(usage)?);
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let hw = match opt("--adapter").as_deref() {
        Some("hw") => true,
        Some("warp") => false,
        _ => return Err(usage.into()),
    };
    // The live switches, read once by the pass and the probe (`screen` first).
    let switches = opt("--switches").unwrap_or_default();
    let render = if switches.is_empty() { "screen".to_string() } else { format!("screen,{switches}") };
    // SAFETY: set before any thread of ours reads the environment.
    unsafe {
        std::env::set_var("XNW_WORLD_RENDER", &render);
        std::env::set_var("XNW_ADAPTER", if hw { "hw" } else { "warp" });
    }
    let mut frames: Vec<(u64, PathBuf)> = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?.flatten().filter_map(|e| Some((e.file_name().to_str()?.strip_prefix("frame-")?.parse().ok()?, e.path()))).filter(|f: &(u64, PathBuf)| f.1.join("list.tsv").exists()).collect();
    frames.sort();
    let from: u64 = opt("--from").and_then(|v| v.parse().ok()).unwrap_or(0);
    let count: usize = opt("--frames").and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    frames.retain(|(n, _)| *n >= from);
    frames.truncate(count);
    if frames.is_empty() {
        return Err(format!("{}: no frame-N with a list.tsv", dir.display()));
    }
    let gaps = frames.windows(2).filter(|w| w[1].0 != w[0].0 + 1).count();
    let png_dir = opt("--png").map(PathBuf::from);
    let png_at: HashSet<u64> = opt("--png-at").map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
    if let Some(d) = &png_dir {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }

    let t_setup = Instant::now();
    // The live pass's device (the bridged rows' pipelines are made on it).
    let host = xnw_gpu::d3d12x::host::host_device().ok_or("no host device")?;
    if host.caps.software == hw {
        return Err(format!("asked for the {} adapter, got {}", if hw { "hardware" } else { "software" }, host.caps.adapter));
    }
    if hw {
        xnw_frame::watchdog_start(std::time::Duration::from_millis(opt("--watchdog-ms").and_then(|v| v.parse().ok()).unwrap_or(xnw_frame::WATCHDOG_MS)));
    }
    let mut pass = xnw_world::screen_pass(host)?;
    // Textures as the replay draws them: the pack's texels, else checkers
    // (`Live` would read title memory the capture does not hold).
    let packed = xnw_frame::pack::Pack::open(&frames[0].1)?.is_some();
    pass.tex.opts.mode = if packed { xnw_world::TexMode::Provided } else { xnw_world::TexMode::Checker };
    // `--steady`: no time budgets (geometry uploads, pipelines, texture
    // reads, cut holds), so two runs draw the same picture whatever their
    // timing (ZC1's before/after check).
    if args.iter().any(|a| a == "--steady") {
        pass.upload_ms_budget = None;
        pass.pso_async = false;
        pass.pso_budget = usize::MAX;
        pass.cuts.hold = false;
        pass.tex.opts.frame_ms = 0.0;
    }
    pass.terrain.table.opts.mode = xnw_world::TexMode::Checker;
    // The list sink's consumers the switches ask for, driven from the
    // capture's sink records (a v2 capture; BURST2).
    if packed {
        super::install_sinks(true);
    }
    let gpucull = render.split(',').any(|w| w == "gpucull=on");
    println!(
        "burst: {} frames {}..{} of {} ({gaps} gaps), {} on {}; XNW_WORLD_RENDER={render}; geometry reads {} {}; textures {:?}; setup {:.0} ms",
        frames.len(),
        frames[0].0,
        frames[frames.len() - 1].0,
        dir.display(),
        if hw { "card" } else { "software" },
        host.caps.adapter,
        if geomread::asynchronous() { "on the reader thread" } else { "in the frame" },
        if geomread::direct() { "(direct copy)" } else { "(ReadProcessMemory path)" },
        pass.tex.opts.mode,
        t_setup.elapsed().as_secs_f64() * 1e3
    );

    // The run's reservations: the census of what the burst reads, and with
    // `XNW_ZEROCOPY=on` the memory made again for in-place reads (ZC1).
    match burst_mem::load(&dir, opt("--xmem-log").as_deref().map(std::path::Path::new)) {
        Ok(n) => {
            println!("burst: title memory census by the run's {n} reservations");
            if xnw_gpu::shared::world_on() {
                println!("burst: XNW_ZEROCOPY=on: {}", burst_mem::mirror_start()?);
            }
        }
        Err(e) => println!("burst: no title memory census ({e})"),
    }
    SERVING.store(true, Ordering::Relaxed);
    // The copy counts from here on (setup's taken and dropped).
    let _ = (xnw_core::syscalls::text(1), xnw_core::plaincopy::text(1));
    let mut have: HashSet<GeomKey> = HashSet::new();
    let mut unplaced = 0u32;
    let mut cache = xnw_world::GeometryCache::<()>::default();
    let (mut list_ms, mut pass_ms, mut load_ms, mut pack_ms) = (Vec::new(), Vec::new(), 0.0f64, 0.0f64);
    let (mut slow_list, mut slow_pass, mut held) = (0u32, 0u32, 0u32);
    let t_run = Instant::now();
    let counts = Arc::new([const { AtomicU64::new(0) }; 3]);
    let mut slot_cache = SlotCache::new();
    let mut warmed = 0usize;
    let packs = if packed { prefetch_packs(&frames, counts.clone()) } else { Vec::new() };
    let result = (|| -> Result<(), String> {
        for (k, (n, fdir)) in frames.iter().enumerate() {
            let n = *n;
            // Load: the frame's list and the bytes of geometries not seen before (not timed as the frame).
            let tl = Instant::now();
            let (mut list, mut geoms) = xnw_frame::load::load_list_with(fdir, &|k| have.contains(k))?;
            list.frame = n;
            xnw_frame::audit::skin_args(args, fdir, &mut list, &mut geoms)?;
            xnw_frame::preskin::preskin_arg(args, fdir, &mut list, &mut geoms)?;
            xnw_frame::list_options(args, &mut list);
            for (k, (vb, idx)) in geoms {
                if have.insert(k) && !place_geometry(&k, vb, &idx) {
                    unplaced += 1;
                }
            }
            load_ms += tl.elapsed().as_secs_f64() * 1e3;
            // The pack's texels (opened ahead on the loader threads; the wait is counted).
            let tk = Instant::now();
            if let Some(rx) = packs.get(k % packs.len().max(1)) {
                pass.tex.set_provider(rx.recv().map_err(|_| "pack loader gone".to_string())??);
            }
            pack_ms += tk.elapsed().as_secs_f64() * 1e3;
            // The sinks' objects and bytes (loading, not timed as the frame).
            let tl = Instant::now();
            let mut taps = match (packed, xnw_frame::pack::Pack::open(fdir)?) {
                (true, Some(p)) => Some(taps_of(&p, fdir, &mut slot_cache)?),
                _ => None,
            };
            if let Some(t) = taps.as_mut() {
                warmed += warm_bridged(&t.served, host);
                xnw_gpu::d3d12x::tap::serve(Some(std::mem::take(&mut t.served)));
                if let Ok(mut m) = FRAME_MEM.write() {
                    *m = std::mem::take(&mut t.mem);
                    if burst_mem::mirrored() {
                        for (at, b) in m.iter() {
                            burst_mem::put(*at, b);
                        }
                    }
                }
                *lock(&GIVEN) = Some(std::mem::take(&mut t.given));
            }
            let mut cull_rep = None;
            if gpucull {
                cull_rep = Some(xnw_frame::gpucull::fill(fdir, &mut list, true)?);
            }
            load_ms += tl.elapsed().as_secs_f64() * 1e3;
            // The list sink's consumers, as the title's threads run them at its list sink.
            FRAME_N.store(n, Ordering::Relaxed);
            let on = super::sinks_on();
            let (sinks, taps_ms) = taps.as_ref().filter(|_| on.0 || on.1 || on.2).map_or((0, 0.0), drive);
            let tap_lines = taps.as_mut().map(|t| apply(&mut list, t, n)).unwrap_or_default();
            for l in &tap_lines {
                println!("{l}");
            }
            if let Some(r) = &cull_rep {
                for l in xnw_frame::gpucull::lines(r, &list.cull) {
                    println!("burst frame {n}: {l}");
                }
            }
            let (cull_rows, preskin_frame, bridged_frame) = (list.cull.rows.len(), list.preskin.clone(), list.bridged.clone());

            // The list side, as render_frame does it after the hook.
            let t0 = Instant::now();
            let draws: Vec<Draw> = list.draws.iter().map(|r| draw_of(&r.geom)).collect();
            let mut why: HashMap<GeomKey, u8> = HashMap::new();
            burst_mem::GEOMETRY.with(|g| g.set(true));
            let taken = geomread::take(&draws, n, &mut cache, &mut list.new_geoms, &mut why, false, geomread::asynchronous(), geomread::direct());
            burst_mem::GEOMETRY.with(|g| g.set(false));
            let geom_ms = t0.elapsed().as_secs_f64() * 1e3;
            if geomread::asynchronous() && n % 60 == 0 {
                let (p, flight) = geomread::stats();
                xnw_core::info!("world render: frame {n}: geometry reader (60 frames): {}; in flight {flight}", p.line());
            }
            coherence(&list, HashMap::new());
            cache.evict(n, 600);
            xnw_world::geometry::known().sweep(n, 600);
            let new = list.new_geoms.len();
            let lm = t0.elapsed().as_secs_f64() * 1e3;
            if lm > 50.0 {
                slow_list += 1;
                xnw_core::warn!(
                    "world render: slow frame {n}: hook - (burst: not driven), list {lm:.1} ms off them ({new} new geometries copied, {} KiB; geometry reads {geom_ms:.1} ms ({}{}))",
                    taken.copied >> 10,
                    taken.parts.line(),
                    if taken.sent + taken.arrived > 0 { format!("; reader: sent {}, arrived {}, waited {} frames at most", taken.sent, taken.arrived, taken.waited) } else { String::new() }
                );
            }

            // The pass, as the live screen path draws a list.
            let rows = list.draws.len();
            let png = png_dir.is_some() && (png_at.is_empty() || png_at.contains(&n));
            let tp = Instant::now();
            if hw {
                xnw_frame::watchdog_arm();
            }
            let st = xnw_world::screen_draw(&mut pass, list, png);
            let pm = tp.elapsed().as_secs_f64() * 1e3;
            if hw {
                xnw_frame::watchdog_disarm();
            }
            let st = st?;
            if png && let Some(d) = &png_dir {
                let rgba = pass.read_rgba()?;
                xnw_frame::write_png(&d.join(format!("seq-{n:05}.png")), pass.width, pass.height, &rgba).map_err(|e| e.to_string())?;
            }
            // The pass's lines for this frame (live prints them every 60:
            // then the pass has printed them already).
            let every = n % 60 != 0;
            if every && cull_rows > 0 {
                println!("{}", xnw_world::gpucull::line(n, &pass.cull_stats()));
            }
            if every && (!preskin_frame.items.is_empty() || preskin_frame.unreadable > 0) {
                println!("{}", xnw_world::preskin_gpu::line(n, &preskin_frame, &pass.preskin_stats()));
            }
            if every && let Some(f) = &bridged_frame {
                println!("{}", xnw_world::bridged::line(n, f, &pass.bridge_stats));
            }
            slow_pass += u32::from(pm > 50.0);
            held += u32::from(st.held);
            println!(
                "burst frame {n}: rows {rows}, taps {taps_ms:.1} ms ({sinks} sinks), list {lm:.1} ms (geometry reads {geom_ms:.1} ms: {}; {new} new, {} failed, {} not ready{}), pass {pm:.1} ms (upload {:.1} ms: {} geometries, {} KiB, {} pending; pipelines {} new; textures {:.1} ms; record {:.1} ms; GPU {:.1} ms), drawn {} ({} without geometry), coverage {:.2}{}",
                taken.parts.line(),
                taken.failed,
                taken.not_ready,
                if taken.sent + taken.arrived > 0 { format!("; reader sent {}, arrived {}, waited {} frames", taken.sent, taken.arrived, taken.waited) } else { String::new() },
                st.upload_ms,
                st.uploaded_geometries,
                st.uploaded_bytes >> 10,
                st.pending,
                st.new_psos,
                st.tex_ms,
                (st.record_ms - st.upload_ms - st.pso_ms - st.tex_ms).max(0.0),
                st.gpu_ms,
                st.draws,
                st.skipped,
                st.coverage,
                if st.held { " held" } else { "" }
            );
            println!("burst frame {n}: title memory read: {}", burst_mem::text());
            println!("burst frame {n}: device memory: {}; geometries in place {} (dropped for a window gone or pages remapped {})", xnw_gpu::d3d12x::resource::video_memory_text(), st.in_place_geoms, st.in_place_dropped);
            if xnw_gpu::shared::world_on() {
                println!("burst frame {n}: XNW_ZEROCOPY: {}; mirror {}", xnw_gpu::shared::world_text(), burst_mem::mirror_text());
            }
            list_ms.push(lm);
            pass_ms.push(pm);
        }
        Ok(())
    })();
    SERVING.store(false, Ordering::Relaxed);
    xnw_gpu::d3d12x::tap::serve(None);
    if warmed > 0 {
        println!("burst: bridged pipelines made before their frame: {warmed}");
    }
    xnw_frame::layer_report(&pass);
    let wall = t_run.elapsed().as_secs_f64();
    let c: Vec<u64> = COUNTS.iter().map(|a| a.load(Ordering::Relaxed)).collect();
    println!(
        "burst: {} frames in {wall:.1} s (loading lists {:.1} s, waits for packs {:.1} s of it); list {}; pass {}; slow frames (over 50 ms): list {slow_list}, pass {slow_pass}; held {held}; title memory from the capture: {} reads served ({} MiB), {} not covered; geometries {} ({unplaced} not put back in the title's form)",
        list_ms.len(),
        load_ms / 1e3,
        pack_ms / 1e3,
        spread(&list_ms),
        spread(&pass_ms),
        c[0],
        c[1] >> 20,
        c[2],
        have.len()
    );
    let n = list_ms.len().max(1) as u64;
    println!("burst: copies per frame: memory system calls (calls/us): {}{}", xnw_core::syscalls::text(n), xnw_core::plaincopy::text(n));
    result
}

/// The burst's table serves what it holds and nothing else.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_serves_captured_ranges_only() {
        // Through the live read's copy (the kernel's, or in place with XNW_PLAINCOPY=on).
        let read = |at, len| read(at, len, |to, from| crate::watch::guarded_read(from as usize, to));
        let k = GeomKey { mesh_at: 0x7_0000_1000, mesh_len: 64, vb_at: 0x7_0000_0000, vb_len: 32, ib: 0, hash: 1 };
        let idx: Vec<u32> = (0..30).map(|i| i % 7).collect();
        assert!(place_geometry(&k, (0..32u8).collect(), &idx));
        assert_eq!(read(0x7_0000_0004, 8), Some((4..12u8).collect()), "inside the vertex buffer");
        assert_eq!(read(0x7_0000_001C, 8), None, "past its end");
        assert_eq!(read(0x6_FFFF_FFF0, 8), None, "before any entry");
        let mesh = read(k.mesh_at, 4).expect("the meshlets, whole");
        assert_eq!(super::super::expand_meshlets(&mesh).unwrap(), idx);
        // Classic u16 indices at their width; a u16 buffer cannot hold 70,000.
        let c = GeomKey { mesh_at: 0x7_0000_2000, mesh_len: 0, vb_at: 0x7_0000_3000, vb_len: 4, ib: 3 << 8 | 2, hash: 2 };
        assert!(place_geometry(&c, vec![0; 4], &[1, 2, 3]));
        assert_eq!(read(c.mesh_at, 6), Some(vec![1, 0, 2, 0, 3, 0]));
        assert!(!place_geometry(&GeomKey { hash: 3, ..c }, vec![0; 4], &[70_000, 1, 2]));
    }
}
