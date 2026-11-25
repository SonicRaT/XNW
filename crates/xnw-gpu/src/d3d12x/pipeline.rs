//! Pipeline states: what the title creates through device slots 10, 11,
//! 55, 57, 62, 67 and 86, kept as a description ([`Desc`]) shared with the
//! lists that bind it; and the census of the pipelines the title actually
//! draws or dispatches with.
//!
//! Layouts: slots 10/11/86 as on PC (VIctory `crates/victory-gpu/src/
//! pipeline.rs` at xnw/base 5d0516e, GPL-3.0-or-later: `GraphicsHead`,
//! `ComputeHead`, `StreamDesc`, `subobject` sizes 465-491); slots 55, 57,
//! 62, 67 from five-star-recomp `docs/d3d12x-pipelines.md` (GPL-3.0):
//! 57 takes a 0x108-byte description of the title's pipeline cache (root
//! signature, state metadata, outputs-and-inputs metadata, seven shader
//! slots of `{code, size, metadata, size}`: 0 pixel, 1 vertex, 2 mesh, 6
//! fetch), 67 a 0x48-byte one (one slot), 55 derives from a base pipeline
//! with override descriptions.
//!
//! Host pipelines: nothing is translated at creation
//! (the title creates ~70,000 pipelines and draws with a handful). The
//! first time a list binds a pipeline, its shaders are deduplicated by
//! content hash and the pipeline is queued to the workers
//! (`pipeline/make.rs`), which translate each shader once (through the disk
//! cache, `shader_cache.rs`) and make the host pipeline (`pipeline/host.rs`,
//! kept in the host's pipeline library, `pipeline/library.rs`). A draw
//! asks [`host_pipeline`]: ready, or `None` (the stand-in: the draw is
//! dropped) while it is still being made or if it was refused. Shader
//! dedup and translation-on-first-use follow VIctory base `pipeline.rs`
//! 57-176 (`Shader`, `translated`); cache code blocks (slots 57/67) are
//! translated from the pack container their 24-byte prefix names
//! (`pack.rs`).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use xnw_shader::emit::{HostShader, Options};

pub(crate) mod host;
mod library;

/// The pipeline library written now (the GPU deadline's last act).
pub(crate) fn save_library_now() {
    library::save(true);
}
mod compile;
mod make;
mod prebuild;
pub(crate) mod warm;

pub(crate) use host::HostPipeline;
use host::{GraphicsState, Override};
use make::Pending;
pub use prebuild::note_phase;

use super::object::{self, Done, E_INVALIDARG, HRESULT, Iid, Object, PIPELINE, Slot, Table, This, guid};
use super::resource::PAGEABLE_IIDS;

pub(crate) const PIPELINE_IIDS: [Iid; 4] = [
    guid(0x765a30f3, 0xf624, 0x4c6f, [0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45]),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

/// Where a stage's code came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Form {
    /// A DXBC container as the pack's SGD5 records hold it (slots 10, 11, 62, 86).
    Container,
    /// A bare code block of the title's pipeline cache (slots 57, 67).
    CacheCode,
}

/// One stage of a pipeline: its code as the title passed it (address and
/// size; cache code stays in the title's loaded pipeline cache), and for
/// cache code its metadata entry. Containers are copied at creation (the
/// base's rule: streams live only for the call), as a shared [`Shader`].
#[derive(Clone)]
pub(crate) struct Stage {
    pub name: &'static str,
    pub form: Form,
    pub code: usize,
    pub size: usize,
    pub meta: usize,
    pub meta_size: usize,
    pub shader: Option<Arc<Shader>>,
}

impl Stage {
    /// The code bytes.
    pub(crate) fn bytes(&self) -> &[u8] {
        if let Some(s) = &self.shader {
            return s.bytes();
        }
        if self.code == 0 || self.size == 0 {
            return &[];
        }
        // SAFETY: the title passed `size` bytes at `code` and keeps them.
        unsafe { std::slice::from_raw_parts(self.code as *const u8, self.size) }
    }

    /// The shared shader of this stage (made on first use for cache code).
    fn shader(&self) -> Arc<Shader> {
        match &self.shader {
            Some(s) => s.clone(),
            None => shader(self.name, self.form, ShaderCode::Title(self.code, self.size)),
        }
    }
}

/// What a pipeline was created from.
#[derive(Default)]
pub(crate) struct Desc {
    /// The creating device slot.
    pub slot: u8,
    pub stages: Vec<Stage>,
    /// The title's root signature object.
    pub root: usize,
    pub targets: Vec<u32>,
    pub depth: u32,
    pub topology: u32,
    /// Slot 55: the pipeline derived from, the override kinds, and the
    /// overrides themselves.
    pub base: Option<Arc<Desc>>,
    pub overrides: Vec<u32>,
    override_descs: Vec<Override>,
    /// Slot 86: the stream's graphics state.
    graphics: Option<Box<GraphicsState>>,
    /// Slot 57: the state and the outputs-and-inputs entries of the cache,
    /// copied at creation (the title reuses that memory: run 0002 read
    /// another pipeline's bytes there at first bind).
    state_entry: Box<[u8]>,
    outputs_entry: Box<[u8]>,
    /// Drawn or dispatched at least once.
    pub drawn: AtomicBool,
    /// The host pipeline's making, set up on first bind.
    host: OnceLock<Arc<Pending>>,
    /// Its last program is unnamed or of the UI class (uikeep.rs).
    ui_class: OnceLock<bool>,
}

impl Desc {
    /// The graphics state of a graphics pipeline (`None`: compute, or a
    /// plain description, whose state is not kept).
    fn graphics_state(&self) -> Option<GraphicsState> {
        match self.slot {
            86 => self.graphics.as_deref().cloned(),
            // SAFETY: the entries lie in the title's loaded pipeline cache.
            57 => Some(GraphicsState::from_cache(&self.state_entry, &self.outputs_entry)),
            55 => {
                let mut g = self.base.as_ref()?.graphics_state()?;
                g.apply_overrides(&self.override_descs);
                Some(g)
            }
            _ => None,
        }
    }
}

#[repr(C)]
struct Pipeline {
    object: Object,
    /// The title reads +0x10..+0x2F of a pipeline (five-star: 0x14248c700).
    packet: [u32; 8],
    desc: Arc<Desc>,
}
const _: () = assert!(std::mem::offset_of!(Pipeline, packet) == 0x10);

fn pipeline<'a>(this: This) -> &'a Pipeline {
    // SAFETY: the title passes one of our pipelines.
    unsafe { &*this.cast::<Pipeline>() }
}

/// The description behind a pipeline the title passes.
pub(crate) fn desc(this: This) -> Option<Arc<Desc>> {
    (!this.is_null()).then(|| pipeline(this).desc.clone())
}

/// The graphics state a pipeline's host pipeline is made with.
pub(crate) fn graphics_state(this: This) -> Option<GraphicsState> {
    desc(this)?.graphics_state()
}

/// Pipelines created, by slot: 10, 11, 55, 57, 62, 67, 86.
pub(crate) const SLOTS_COUNTED: [u8; 7] = [10, 11, 55, 57, 62, 67, 86];
static CREATED: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

fn slot_index(slot: u8) -> usize {
    SLOTS_COUNTED.iter().position(|&s| s == slot).unwrap_or(0)
}

pub(crate) fn created_text() -> String {
    SLOTS_COUNTED
        .iter()
        .zip(&CREATED)
        .map(|(s, n)| format!("{s}:{}", n.load(Ordering::Relaxed)))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Pipelines created so far, all slots.
/// Host pipelines made after the null point.
pub(crate) fn made_in_null() -> u64 {
    make::MADE_IN_NULL.load(Ordering::Relaxed)
}

pub(crate) fn created() -> u64 {
    CREATED.iter().map(|n| n.load(Ordering::Relaxed)).sum()
}

fn hand_out(desc: Desc, iid: *const Iid, out: *mut This) -> HRESULT {
    let n = CREATED[slot_index(desc.slot)].fetch_add(1, Ordering::Relaxed) + 1;
    if n == 1 {
        xnw_core::info!("d3d12x: first pipeline from device slot {} ({} stages)", desc.slot, desc.stages.len());
    }
    let total = created();
    if total.is_multiple_of(10_000) {
        xnw_core::info!("d3d12x: {total} pipelines created, by slot {}", created_text());
    }
    if desc.stages.iter().any(|s| s.form == Form::CacheCode) {
        // Cache code is translated from the pack: open its index early,
        // off the title's threads (2.5 s to build, 0.01 s once saved).
        static OPEN: std::sync::Once = std::sync::Once::new();
        OPEN.call_once(|| {
            let _ = std::thread::Builder::new().name("xnw-pack-index".into()).spawn(|| {
                pack();
            });
        });
    }
    let desc = Arc::new(desc);
    prebuild::offer(&desc);
    let p = Box::new(Pipeline { object: Object::new(&FUNCTIONS), packet: [0; 8], desc });
    object::hand_out(PIPELINE, p, iid, &PIPELINE_IIDS, out)
}

#[repr(C)]
struct Bytecode {
    code: *const u8,
    size: usize,
}

const GRAPHICS_STAGES: [&str; 5] = ["vertex", "pixel", "domain", "hull", "geometry"];

fn container(name: &'static str, b: &Bytecode) -> Option<Stage> {
    if b.code.is_null() || b.size == 0 {
        return None;
    }
    // SAFETY: the title passes `size` bytes of shader code.
    let bytes = unsafe { std::slice::from_raw_parts(b.code, b.size) };
    let shader = shader(name, Form::Container, ShaderCode::Owned(bytes.to_vec()));
    Some(Stage { name, form: Form::Container, code: b.code as usize, size: b.size, meta: 0, meta_size: 0, shader: Some(shader) })
}

fn read<T: Copy>(at: *const u8, offset: usize) -> T {
    // SAFETY: callers read inside a description the title passed.
    unsafe { at.add(offset).cast::<T>().read_unaligned() }
}

/// Slot 10: PC's `D3D12_GRAPHICS_PIPELINE_STATE_DESC`.
pub(crate) extern "system" fn create_graphics(_: This, d: *const u8, iid: *const Iid, out: *mut This) -> HRESULT {
    if d.is_null() {
        return E_INVALIDARG;
    }
    let mut desc = Desc { slot: 10, root: read(d, 0), ..Desc::default() };
    for (i, name) in GRAPHICS_STAGES.iter().enumerate() {
        desc.stages.extend(container(name, &read_bytecode(d, 8 + i * 16)));
    }
    desc.topology = read(d, 0x23c);
    let count = read::<u32>(d, 0x240).min(8) as usize;
    desc.targets = (0..count).map(|i| read(d, 0x244 + i * 4)).collect();
    desc.depth = read(d, 0x264);
    hand_out(desc, iid, out)
}

fn read_bytecode(d: *const u8, at: usize) -> Bytecode {
    Bytecode { code: read::<usize>(d, at) as *const u8, size: read(d, at + 8) }
}

fn compute(slot: u8, d: *const u8, iid: *const Iid, out: *mut This) -> HRESULT {
    if d.is_null() {
        return E_INVALIDARG;
    }
    let mut desc = Desc { slot, root: read(d, 0), ..Desc::default() };
    desc.stages.extend(container("compute", &read_bytecode(d, 8)));
    hand_out(desc, iid, out)
}

/// Slot 11: PC's `D3D12_COMPUTE_PIPELINE_STATE_DESC`.
pub(crate) extern "system" fn create_compute(_: This, d: *const u8, iid: *const Iid, out: *mut This) -> HRESULT {
    compute(11, d, iid, out)
}

/// Slot 62: the compute description plus console extensions (always 0, 0).
pub(crate) extern "system" fn create_compute_x(_: This, d: *const u8, _count: u32, _ext: *const u8, iid: *const Iid, out: *mut This) -> HRESULT {
    compute(62, d, iid, out)
}

/// Slot 55: `(base pipeline, override count, overrides, iid, out)`; each
/// override is 0x14c bytes, a kind (0 depth-stencil, 1 rasterizer, 2 blend,
/// 5 unknown) then the description.
pub(crate) extern "system" fn create_derived(_: This, base: This, count: u32, overrides: *const u8, iid: *const Iid, out: *mut This) -> HRESULT {
    // The title uses the out pointer whatever the result: always answer.
    let b = desc(base).unwrap_or_default();
    let n = if overrides.is_null() { 0 } else { count.min(16) as usize };
    // SAFETY (read): `count` entries of 0x14c bytes at `overrides`.
    let kinds = (0..n).map(|i| read(overrides, i * 0x14c)).collect();
    // SAFETY: as above; the entries live only for the call, so copied.
    let override_descs = (0..n).filter_map(|i| Override::read(unsafe { overrides.add(i * 0x14c) })).collect();
    let desc = Desc {
        slot: 55,
        stages: b.stages.clone(),
        root: b.root,
        targets: b.targets.clone(),
        depth: b.depth,
        topology: b.topology,
        base: Some(b),
        overrides: kinds,
        override_descs,
        ..Desc::default()
    };
    hand_out(desc, iid, out)
}

/// A copy of a cache metadata entry `{pointer, size}` (empty if absent or
/// implausibly large).
fn entry(at: usize, size: usize) -> Box<[u8]> {
    if at == 0 || size == 0 || size > 64 << 10 {
        return Box::default();
    }
    // SAFETY: the title's description names `size` bytes at `at`.
    unsafe { std::slice::from_raw_parts(at as *const u8, size) }.into()
}

/// An empty pipeline from slot 55, 57 or 67 (`XNW_PIPELINE_X=empty`,
/// pipeline_x.rs): no stages, never drawn with.
pub(crate) fn create_empty(slot: u8, iid: *const Iid, out: *mut This) -> HRESULT {
    hand_out(Desc { slot, ..Desc::default() }, iid, out)
}

/// The stage of each graphics slot of the title's pipeline cache.
const CACHE_STAGES: [&str; 7] = ["pixel", "vertex", "mesh", "slot3", "slot4", "slot5", "fetch"];

fn cache_stage(name: &'static str, d: *const u8, at: usize) -> Option<Stage> {
    let code: usize = read(d, at);
    let size: usize = read(d, at + 8);
    (code != 0 && size != 0).then(|| Stage { name, form: Form::CacheCode, code, size, meta: read(d, at + 16), meta_size: read(d, at + 24), shader: None })
}

/// Slot 57: a graphics pipeline of the title's pipeline cache (0x108 bytes).
pub(crate) extern "system" fn create_cached_graphics(_: This, d: *const u8, iid: *const Iid, out: *mut This) -> HRESULT {
    if d.is_null() {
        return E_INVALIDARG;
    }
    let mut desc = Desc { slot: 57, root: read(d, 0), ..Desc::default() };
    for (i, name) in CACHE_STAGES.iter().enumerate() {
        desc.stages.extend(cache_stage(name, d, 0x28 + i * 32));
    }
    // Outputs and inputs (pipeline-cache.md "Pipeline outputs"): formats
    // at +0x10 (bytes), depth +0x24, count +0x25, topology +0x26.
    desc.state_entry = entry(read(d, 0x08), read(d, 0x10));
    let outputs: usize = read(d, 0x18);
    let outputs_size: usize = read(d, 0x20);
    desc.outputs_entry = entry(outputs, outputs_size);
    if outputs != 0 && outputs_size >= 0x27 {
        let o = outputs as *const u8;
        let count = usize::from(read::<u8>(o, 0x25)).min(8);
        desc.targets = (0..count).map(|i| u32::from(read::<u8>(o, 0x10 + i))).collect();
        desc.depth = u32::from(read::<u8>(o, 0x24));
        desc.topology = u32::from(read::<u8>(o, 0x26) & 15);
    }
    hand_out(desc, iid, out)
}

/// Slot 67: a compute pipeline of the title's pipeline cache (0x48 bytes).
pub(crate) extern "system" fn create_cached_compute(_: This, d: *const u8, iid: *const Iid, out: *mut This) -> HRESULT {
    if d.is_null() {
        return E_INVALIDARG;
    }
    let mut desc = Desc { slot: 67, root: read(d, 0), ..Desc::default() };
    desc.stages.extend(cache_stage("compute", d, 0x28));
    hand_out(desc, iid, out)
}

/// A pipeline described as a stream: its size and the subobjects.
#[repr(C)]
pub(crate) struct StreamDesc {
    size: usize,
    stream: *const u8,
}

/// A subobject type's size (type included), and the stage it is if it is a
/// shader; console numbering (amplification 23, mesh 24). VIctory base
/// pipeline.rs 465-491.
fn subobject(kind: u32) -> Option<(usize, Option<&'static str>)> {
    let stage = |name| Some((24, Some(name)));
    match kind {
        1 => stage("vertex"),
        2 => stage("pixel"),
        3 => stage("domain"),
        4 => stage("hull"),
        5 => stage("geometry"),
        6 => stage("compute"),
        23 => stage("amplification"),
        24 => stage("mesh"),
        0 => Some((16, None)),
        12 | 19 => Some((24, None)),
        7 | 15 => Some((40, None)),
        8 => Some((336, None)),
        10 => Some((48, None)),
        11 => Some((56, None)),
        17 => Some((16, None)),
        21 => Some((64, None)),
        22 => Some((32, None)),
        9 | 13 | 14 | 16 | 18 | 20 => Some((8, None)),
        _ => None,
    }
}

/// Slot 86: a pipeline from a stream.
pub(crate) extern "system" fn create_stream(_: This, d: *const StreamDesc, iid: *const Iid, out: *mut This) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(d) = (unsafe { d.as_ref() }) else {
        return E_INVALIDARG;
    };
    if d.stream.is_null() {
        return E_INVALIDARG;
    }
    let mut desc = Desc { slot: 86, ..Desc::default() };
    let mut graphics = GraphicsState::default();
    let s = d.stream;
    let mut at = 0;
    while at + 8 <= d.size {
        let kind: u32 = read(s, at);
        let Some((size, stage)) = subobject(kind) else {
            warn_once_kind(kind, at);
            break;
        };
        if let Some(name) = stage {
            desc.stages.extend(container(name, &read_bytecode(s, at + 8)));
        }
        match kind {
            0 => desc.root = read(s, at + 8),
            14 => desc.topology = read(s, at + 4),
            15 => {
                let count = read::<u32>(s, at + 0x24).min(8) as usize;
                desc.targets = (0..count).map(|i| read(s, at + 4 + i * 4)).collect();
            }
            16 => desc.depth = read(s, at + 4),
            _ => {}
        }
        // SAFETY: a whole subobject of this kind lies at `at`.
        unsafe { graphics.read(kind, s.add(at)) };
        at += size;
    }
    let compute = desc.stages.iter().any(|st| st.name == "compute");
    desc.graphics = (!compute).then(|| Box::new(graphics));
    hand_out(desc, iid, out)
}

fn warn_once_kind(kind: u32, at: usize) {
    static SEEN: Mutex<Vec<u32>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&kind) {
        seen.push(kind);
        xnw_core::warn!("d3d12x: pipeline subobject {kind} at {at:#x} of unknown size; the rest of the stream is ignored");
    }
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(PIPELINE, iid, object::is(iid, &PIPELINE_IIDS).then_some(this), out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Pipeline>(this)
}

// ---- What lists bind, and the census of drawn pipelines ----

/// The pipeline each list has set, by list address.
static BOUND: Mutex<Option<HashMap<usize, Arc<Desc>>>> = Mutex::new(None);

/// List method 25 (`SetPipelineState`), and the pipeline a reset passes:
/// kept for the list's draws; the first bind of a pipeline queues its host
/// pipeline (no wait).
pub(crate) fn bind(list: This, pipeline: This) {
    let d = desc(pipeline);
    if let Some(d) = &d
        && enabled()
    {
        make::begin(pending(d));
    }
    let mut bound = BOUND.lock().unwrap_or_else(|e| e.into_inner());
    let map = bound.get_or_insert_with(HashMap::new);
    match d {
        Some(d) => {
            map.insert(list as usize, d);
        }
        None => {
            map.remove(&(list as usize));
        }
    }
}

/// The pipeline `list` has set.
pub(crate) fn bound(list: This) -> Option<Arc<Desc>> {
    BOUND.lock().unwrap_or_else(|e| e.into_inner()).as_ref()?.get(&(list as usize)).cloned()
}

/// Distinct pipelines drawn or dispatched, by slot (same order as [`SLOTS_COUNTED`]).
static DRAWN: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
static DRAWN_ALL: AtomicU64 = AtomicU64::new(0);
static UNBOUND_DRAWS: AtomicU64 = AtomicU64::new(0);

/// How many first-drawn pipelines the census details.
const CENSUS: u64 = 64;

/// A draw (`dispatch` false) or dispatch on `list`: the census of the
/// first pipelines used.
pub(crate) fn used(list: This, what: &str) {
    let Some(d) = bound(list) else {
        if UNBOUND_DRAWS.fetch_add(1, Ordering::Relaxed) == 0 {
            xnw_core::info!("census: first {what} with no pipeline set");
        }
        return;
    };
    // What the draw will do: the host pipeline, waited for within the
    // frame's budget, else the stand-in (the draw is dropped).
    let counter = if host_pipeline_of(&d).is_some() { &USED_READY } else { &USED_STANDIN };
    counter.fetch_add(1, Ordering::Relaxed);
    summary_now_and_then();
    // The early list is saved from the workers after a make; with the
    // prebuild there may be none, so also every 256th draw.
    static DRAWS: AtomicU64 = AtomicU64::new(0);
    if DRAWS.fetch_add(1, Ordering::Relaxed).is_multiple_of(256) {
        library::save(false);
    }
    if d.drawn.swap(true, Ordering::Relaxed) {
        return;
    }
    DRAWN[slot_index(d.slot)].fetch_add(1, Ordering::Relaxed);
    let n = DRAWN_ALL.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= CENSUS {
        census(n, &d, what);
    }
    if n <= CENSUS || n.is_power_of_two() {
        let drawn = SLOTS_COUNTED.iter().zip(&DRAWN).map(|(s, c)| format!("{s}:{}", c.load(Ordering::Relaxed))).collect::<Vec<_>>().join(" ");
        xnw_core::info!(
            "census: {n} distinct pipelines used (frame {}); used by slot {drawn}; created by slot {}",
            super::presented(),
            created_text()
        );
    }
}

fn census(n: u64, d: &Desc, what: &str) {
    let targets = d.targets.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(",");
    let base = d.base.as_ref().map(|b| format!(" base slot {} overrides {:?}", b.slot, d.overrides)).unwrap_or_default();
    xnw_core::info!(
        "census #{n}: {what} frame {} slot {}{base} targets [{targets}] depth {} topology {} root {}",
        super::presented(),
        d.slot,
        d.depth,
        d.topology,
        super::root::shape(d.root as This)
    );
    // The cache entries the state comes from (slot 57, or a 55's base).
    let cached = if d.slot == 57 { Some(d) } else { d.base.as_deref().filter(|b| b.slot == 57) };
    if let (Some(c), Some(dir)) = (cached, xnw_core::run_dir()) {
        let dir = dir.join("census");
        let _ = std::fs::create_dir_all(&dir);
        for (name, bytes) in [("state", &c.state_entry), ("outputs", &c.outputs_entry)] {
            if !bytes.is_empty() {
                let _ = std::fs::write(dir.join(format!("{n:02}-{name}.entry")), bytes);
            }
        }
    }
    for s in &d.stages {
        let bytes = s.bytes();
        let hash = xnw_core::hash128(&[bytes]);
        let prefix: String = bytes.iter().take(24).map(|b| format!("{b:02x}")).collect();
        xnw_core::info!("census #{n}:   {} {:?} {} bytes hash {hash:032x} meta {} prefix {prefix}", s.name, s.form, s.size, s.meta_size);
        if let Some(dir) = xnw_core::run_dir() {
            let dir = dir.join("census");
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::fs::write(dir.join(format!("{n:02}-{}-{hash:032x}.bin", s.name)), bytes);
            if s.meta != 0 && s.meta_size != 0 && s.meta_size < 1 << 20 {
                // SAFETY: the title's metadata entry, `meta_size` bytes.
                let meta = unsafe { std::slice::from_raw_parts(s.meta as *const u8, s.meta_size) };
                let _ = std::fs::write(dir.join(format!("{n:02}-{}-{hash:032x}.meta", s.name)), meta);
            }
        }
    }
}

// ---- Shaders, host pipelines and their counts ----

/// A shader's code: copied (containers) or the title's (cache code, which
/// stays loaded with the title's pipeline cache).
pub(crate) enum ShaderCode {
    Owned(Vec<u8>),
    Title(usize, usize),
    /// The console container itself, read from the pack by the prebuild
    /// without the title (for cache code it stands in for the code block).
    Packed(Vec<u8>),
}

/// A shader the title passed, shared by every pipeline with the same bytes.
pub(crate) struct Shader {
    pub(crate) stage: &'static str,
    pub(crate) form: Form,
    /// `xnw_core::hash128` of the bytes, as logged.
    pub(crate) hash: u128,
    code: ShaderCode,
    translated: OnceLock<Result<Translated, String>>,
    /// The hash of the console container translated (for cache code,
    /// the pack container), once known: the shader cache's key and the
    /// pipeline list's recipe name it.
    container: OnceLock<u128>,
}

// SAFETY: title code is read-only once created; the rest is owned.
unsafe impl Send for Shader {}
// SAFETY: as above.
unsafe impl Sync for Shader {}

/// A translated shader and the console resources its code reads (base
/// pipeline.rs 98-106).
pub(crate) struct Translated {
    pub(crate) host: HostShader,
    pub(crate) used: Vec<xnw_shader::UsedResource>,
    /// Whether the shader indexes `ResourceDescriptorHeap[]` directly.
    pub(crate) resource_heap: bool,
    /// Whether the shader may write memory (buffer, image, or global stores
    /// and atomics); `true` when its code is not known.
    pub(crate) writes_memory: bool,
    /// A mesh program's record layout (XNW: it runs as compute).
    pub(crate) mesh: Option<xnw_shader::emit::mesh::Layout>,
}

/// A shader already translated, for device tests that make host pipelines
/// from hand-made programs.
#[cfg(test)]
pub(crate) fn translated_for_test(stage: &'static str, host: HostShader, used: Vec<xnw_shader::UsedResource>) -> Arc<Shader> {
    let s = Shader { stage, form: Form::Container, hash: 0, code: ShaderCode::Owned(Vec::new()), translated: OnceLock::new(), container: OnceLock::new() };
    let _ = s.translated.set(Ok(Translated { host, used, resource_heap: false, writes_memory: true, mesh: None }));
    Arc::new(s)
}

impl Shader {
    /// The shader's name in the pack (`effect:Program_hash_Wrapped`), if
    /// the pack index has it (the culling dump picks programs by name).
    pub(crate) fn pack_name(&self) -> Option<String> {
        let pack = pack()?;
        let place = match self.form {
            Form::Container => pack.by_hash(self.hash),
            Form::CacheCode => pack.by_code(self.bytes()),
        }?;
        Some(place.name.clone())
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        match &self.code {
            ShaderCode::Owned(v) | ShaderCode::Packed(v) => v,
            // SAFETY: the title's bytes, kept loaded.
            ShaderCode::Title(at, n) => unsafe { std::slice::from_raw_parts(*at as *const u8, *n) },
        }
    }

    /// A shader the prebuild read from the pack (`container`), named by
    /// the hash the title's bytes had (`hash`); not shared with the title's.
    pub(crate) fn packed(stage: &'static str, form: Form, hash: u128, container: Vec<u8>) -> Arc<Shader> {
        let c = OnceLock::new();
        let _ = c.set(xnw_core::hash128(&[&container]));
        Arc::new(Shader { stage, form, hash, code: ShaderCode::Packed(container), translated: OnceLock::new(), container: c })
    }

    /// The hash of the container translated, once the translation ran.
    pub(crate) fn container_hash(&self) -> Option<u128> {
        self.container.get().copied()
    }

    /// The options a translation of this shader takes in this process.
    pub(crate) fn options(&self) -> Options {
        options_for(self.stage, self.hash)
    }

    /// The translation (made on first use, on a pipeline worker).
    pub(crate) fn translated(&self) -> Option<&Translated> {
        self.translation().ok()
    }

    pub(crate) fn translation(&self) -> Result<&Translated, &String> {
        self.translated.get_or_init(|| self.translate()).as_ref()
    }

    fn translate(&self) -> Result<Translated, String> {
        let started = std::time::Instant::now();
        let opts = self.options();
        let form = self.form as u8;
        let bytes = self.bytes();
        let hits = super::shader_cache::HITS.load(Ordering::Relaxed);
        // The container first (cache code: from the pack), read once;
        // the cache key names it.
        let container: Result<std::borrow::Cow<[u8]>, String> = match (&self.code, self.form) {
            (ShaderCode::Packed(c), _) => Ok(c.as_slice().into()),
            (_, Form::Container) => Ok(bytes.into()),
            (_, Form::CacheCode) => (|| {
                let pack = pack().ok_or("no shader pack index (XNW_GAME_DRIVE or XNW_SHADER_PACK)")?;
                let place = pack.by_code(bytes).ok_or("cache code with no pack record of the same 24-byte prefix")?;
                Ok(pack.container(place).ok_or("pack record unreadable")?.into())
            })(),
        };
        let container = match container {
            Ok(c) => c,
            Err(e) => {
                super::shader_cache::FAILED.fetch_add(1, Ordering::Relaxed);
                note_failure(&format!("{} does not translate: {e}", self.stage), &format!("{:032x}", self.hash));
                return Err(e);
            }
        };
        let container_hash = *self.container.get_or_init(|| if self.form == Form::Container { self.hash } else { xnw_core::hash128(&[&container]) });
        let key = super::shader_cache::key(form, container_hash, opts);
        // Code the pack does not hold is kept for the prebuild.
        if self.form == Form::Container
            && !matches!(self.code, ShaderCode::Packed(_))
            && pack().is_none_or(|p| p.by_hash(container_hash).is_none())
            && let Some(dir) = super::shader_cache::dir()
        {
            super::shader_cache::keep_container(dir, container_hash, &container);
        }
        // `XNW_SHADER_DUMP=<hash prefix>,...` writes the console container of
        // those shaders to `<run>/shaders/`, for `xnw shader-info`.
        static DUMP: std::sync::LazyLock<Vec<String>> =
            std::sync::LazyLock::new(|| std::env::var("XNW_SHADER_DUMP").unwrap_or_default().split(',').map(|s| s.trim().to_ascii_lowercase()).filter(|s| !s.is_empty()).collect());
        let hex = format!("{:032x}", self.hash);
        if DUMP.iter().any(|p| hex.starts_with(p.as_str()))
            && let Some(dir) = xnw_core::run_dir()
        {
            let dir = dir.join("shaders");
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::fs::write(dir.join(format!("{}-{hex}.dxbc", self.stage)), &container);
        }
        // XNW: XNW_REFUSE, XNW_REFUSE_NEW, XNW_REFUSE_KIND.
        let refuse = super::refuse::switches();
        if refuse.any()
            && let Some(switch) = refuse.why(&hex, || super::shader_cache::dir().and_then(|d| super::shader_cache::first_version(d, form, container_hash, bytes, opts)), || Some(container.to_vec()))
        {
            note_failure(&format!("{} refused by switch {switch}", self.stage), &hex);
            return Err(format!("refused by switch {switch}"));
        }
        let result = super::shader_cache::cached(super::shader_cache::dir(), key, || {
            note_wave64(self, &container);
            xnw_shader::emit::translate_with(&container, opts).map_err(|e| e.to_string())
        });
        // The prebuild has no title bytes for cache code (the record's
        // older-key fallback needs them): the run records those.
        let title_bytes = !(matches!(self.code, ShaderCode::Packed(_)) && self.form == Form::CacheCode);
        if result.is_ok()
            && title_bytes
            && let Some(dir) = super::shader_cache::dir()
        {
            super::shader_cache::note_first(dir, form, container_hash, bytes, opts);
        }
        let how = if super::shader_cache::HITS.load(Ordering::Relaxed) > hits { "from the cache" } else { "translated" };
        match &result {
            Ok(h) => {
                let n = SHADERS_LOGGED.fetch_add(1, Ordering::Relaxed);
                if n < 64 {
                    xnw_core::info!(
                        "d3d12x: shader {} {:032x} {how} ({:?} fma, {} bytes DXIL, {} user dwords, {:.1} ms)",
                        self.stage,
                        self.hash,
                        opts.fma,
                        h.dxil.len(),
                        h.user_data_dwords(),
                        started.elapsed().as_secs_f64() * 1e3
                    );
                }
            }
            Err(e) => note_failure(&format!("{} does not translate: {e}", self.stage), &format!("{:032x}", self.hash)),
        }
        let host = result?;
        // The native programs' resources and stores (base pipeline.rs
        // 146-176), read from the container also on a cache hit.
        let (used, resource_heap, writes_memory) = native_facts(&container);
        let mesh = if self.stage == "mesh" { xnw_shader::emit::mesh::record_layout(&container).ok() } else { None };
        Ok(Translated { host, used, resource_heap, writes_memory, mesh })
    }
}

/// Base pipeline.rs 146-176: the first VS/PS/MS/CS program's used
/// resources, whether it indexes the resource heap, and whether any
/// program may write memory.
fn native_facts(container: &[u8]) -> (Vec<xnw_shader::UsedResource>, bool, bool) {
    use xnw_shader::{Container, FourCC, HwStage, RootType};
    let programs = Container::parse(container)
        .ok()
        .and_then(|c| c.part(FourCC::DXIL).map(|p| p.data))
        .and_then(|d| xnw_shader::native_programs(d).ok())
        .unwrap_or_default();
    let writes_memory = programs.is_empty() || programs.iter().any(writes_memory);
    let (used, resource_heap) = programs
        .into_iter()
        .find(|p| matches!(p.stage, HwStage::Vs | HwStage::Ps | HwStage::Ms | HwStage::Cs))
        .map(|p| {
            let rh = p.layout.iter().any(|rp| rp.root == RootType::ResourceHeap) || p.used.iter().any(|u| u.root == RootType::ResourceHeap);
            (p.used, rh)
        })
        .unwrap_or_default();
    (used, resource_heap, writes_memory)
}

/// Base pipeline.rs 182-200: whether a native program may write memory: it
/// has a buffer, image, flat, or global store or atomic, or code that is
/// not all known (an instruction outside our table, an indirect jump, or
/// code that does not decode).
fn writes_memory(program: &xnw_shader::NativeProgram) -> bool {
    use xnw_shader::gfx10::{Encoding, decode};
    let Ok(code) = decode(&program.code) else {
        return true;
    };
    code.iter().any(|i| {
        let name = i.name();
        match i.encoding {
            Encoding::Mubuf | Encoding::Mtbuf | Encoding::Mimg | Encoding::Flat | Encoding::Global => name.is_none_or(stores),
            Encoding::Sop1 => name.is_none_or(|n| matches!(n, "s_setpc_b64" | "s_swappc_b64")),
            _ => false,
        }
    })
}

/// Whether a memory instruction (by name) writes memory (base pipeline.rs 204).
fn stores(name: &str) -> bool {
    name.contains("store") || name.contains("atomic")
}

/// The options a translation of a `stage` shader whose bytes hash to
/// `hash` takes in this process.
/// The mesh form, descriptor mode, multiply-add mode and page table are
/// `xnw_shader::emit::runtime_options` (the shader gate uses the same).
pub(crate) fn options_for(stage: &str, hash: u128) -> Options {
    // XNW: `XNW_LOOP_PROBE=<hash prefix>,...`.
    let probe = super::refuse::probed(&format!("{hash:032x}"));
    // XNW: `XNW_LOOP_COUNTS=1`: the program's slot
    // (`loopcount::slot_of`: first 8 hex digits of its hash mod 1024).
    let loop_counts = if super::loopcount::on() { super::loopcount::slot_of(&format!("{hash:032x}")) } else { None };
    xnw_shader::emit::runtime_options(stage, hash, probe, loop_counts)
}

static SHADERS_LOGGED: AtomicU64 = AtomicU64::new(0);

/// A wave64 vertex or pixel program runs at the host's wave size (no
/// `WaveSize` for those stages in SM 6.6): logged once per shader, for the
/// may-differ policy.
fn note_wave64(s: &Shader, container: &[u8]) {
    if !matches!(s.stage, "vertex" | "pixel") {
        return;
    }
    let wave64 = xnw_shader::Container::parse(container)
        .ok()
        .and_then(|c| c.part(xnw_shader::FourCC::DXIL).map(|p| p.data))
        .and_then(|d| xnw_shader::native_programs(d).ok())
        .is_some_and(|ps| ps.iter().any(|p| p.wave64));
    if wave64 {
        WAVE64_GRAPHICS.fetch_add(1, Ordering::Relaxed);
        xnw_core::info!("d3d12x: wave64 {} shader {:032x} runs at the host's wave size (may differ in cross-lane code)", s.stage, s.hash);
    }
}
static WAVE64_GRAPHICS: AtomicU64 = AtomicU64::new(0);

/// Every shader seen, by (form, content hash).
static SHADERS: Mutex<Option<HashMap<(u8, u128), Arc<Shader>>>> = Mutex::new(None);

/// The shared shader for these bytes.
fn shader(stage: &'static str, form: Form, code: ShaderCode) -> Arc<Shader> {
    let bytes = match &code {
        ShaderCode::Owned(v) | ShaderCode::Packed(v) => &v[..],
        // SAFETY: the title's bytes.
        ShaderCode::Title(at, n) => unsafe { std::slice::from_raw_parts(*at as *const u8, *n) },
    };
    let hash = xnw_core::hash128(&[bytes]);
    let mut shaders = SHADERS.lock().unwrap_or_else(|e| e.into_inner());
    shaders
        .get_or_insert_with(HashMap::new)
        .entry((form as u8, hash))
        .or_insert_with(|| Arc::new(Shader { stage, form, hash, code, translated: OnceLock::new(), container: OnceLock::new() }))
        .clone()
}

/// The pack `xnw prebuild` was given (else [`super::pack::default_pack`]).
static PACK_PATH: OnceLock<std::path::PathBuf> = OnceLock::new();

/// The shader pack's index, opened on first need (on a worker).
fn pack() -> Option<&'static super::pack::PackIndex> {
    static PACK: OnceLock<Option<super::pack::PackIndex>> = OnceLock::new();
    PACK.get_or_init(|| {
        let path = PACK_PATH.get().cloned().or_else(super::pack::default_pack)?;
        let started = std::time::Instant::now();
        let saved = super::shader_cache::dir().map(|d| d.join("pack-index.bin"));
        match super::pack::PackIndex::open(&path, saved.as_deref()) {
            Ok(p) => {
                xnw_core::info!("d3d12x: shader pack index: {} containers ({:.2} s, {})", p.len(), started.elapsed().as_secs_f64(), path.display());
                Some(p)
            }
            Err(e) => {
                xnw_core::warn!("d3d12x: no shader pack index from {}: {e}", path.display());
                None
            }
        }
    })
    .as_ref()
}

/// The host pipeline's making for `d`, set up once.
fn pending(d: &Arc<Desc>) -> &Arc<Pending> {
    d.host.get_or_init(|| {
        init_cache_dir();
        Arc::new(Pending::new(shaders_of(d), d.graphics_state().map(Box::new)))
    })
}

/// The pipeline library and early list live in `<cache>/pipelines` (set once).
fn init_cache_dir() {
    static DIR: std::sync::Once = std::sync::Once::new();
    DIR.call_once(|| {
        if let Some(dir) = super::shader_cache::dir() {
            library::use_pipeline_cache(dir.join("pipelines"));
            compile::use_dir(dir.join("pipelines"));
        }
    });
}

fn shaders_of(d: &Desc) -> Vec<Arc<Shader>> {
    d.stages.iter().filter(|s| s.name != "fetch").map(Stage::shader).collect()
}

/// A frame statistics line: the prebuild counters.
pub(crate) fn prebuild_text() -> String {
    prebuild::stats_text()
}

/// The run is failing: the pipeline library goes to disk now.
pub(crate) fn flush_library(why: &str) {
    // Compiles still running are named, and the slowest at exit.
    compile::report(why);
    library::flush(why);
}

/// The pipelines past the compile deadline in an earlier run.
pub(crate) fn slow_pipelines() -> Vec<(String, String)> {
    init_cache_dir();
    compile::slow_list()
}

/// The host device is lost: host pipelines compiling now are recorded
/// as suspects (`suspect.txt`).
pub(crate) fn note_device_lost(why: &str) {
    compile::device_lost(why);
}

/// A mesh draw was recorded (the world draws with mesh programs).
pub(crate) fn note_mesh_draw() {
    prebuild::note_mesh_draw();
    // The world's first frame, for the cold-cache warning.
    if WORLD_FRAME.load(Ordering::Relaxed) == 0 && library::STORY_MS.load(Ordering::Relaxed) != 0 {
        let _ = WORLD_FRAME.compare_exchange(0, super::presented().max(1), Ordering::Relaxed, Ordering::Relaxed);
    }
}

/// The entry: the host pipeline of a title pipeline when it is made (or
/// made within the frame's wait budget); `None` is the stand-in: the draw
/// or dispatch is dropped (still being made, or refused: see the failure
/// counts).
pub(crate) fn host_pipeline_of(d: &Arc<Desc>) -> Option<&HostPipeline> {
    // In null mode no draw or dispatch is ever executed: none is
    // prepared (no pipeline asked for or compiled, no root argument or
    // descriptor resolved, no fill); the list keeps its memory operations.
    if !enabled() {
        return None;
    }
    if super::frameskip::null_now() {
        return kept_pipeline(d);
    }
    make::ready(pending(d))
}

/// `ui=keep`: in null mode, a kept command's pipeline if it is made
/// already (never compiled or waited for now), and, inside a stage-30 pass,
/// only for a program of the UI class.
fn kept_pipeline(d: &Arc<Desc>) -> Option<&HostPipeline> {
    if !super::uikeep::keep_now() {
        return None;
    }
    let Some(made) = make::made_now(pending(d)) else {
        super::uikeep::NOT_MADE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    };
    if super::uikeep::stage_in_pass() == Some(30) && !*d.ui_class.get_or_init(|| made.stages.last().and_then(|s| s.shader.pack_name()).is_none_or(|n| super::uikeep::ui_name(&n))) {
        super::uikeep::NOT_UI.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    Some(made)
}

/// Base pipeline.rs 274 (the form draw.rs calls): the host pipeline of the
/// title pipeline `this`, through its description ([`host_pipeline_of`]).
pub(crate) fn host_pipeline<'a>(this: This) -> Option<&'a HostPipeline> {
    if this.is_null() {
        return None;
    }
    host_pipeline_of(&pipeline(this).desc)
}

/// The gate (`host::GATES`) the host pipeline of title pipeline `this`
/// was refused at, if it was; for counting the draws and dispatches dropped
/// there.
pub(crate) fn gate_of(this: This) -> Option<usize> {
    if this.is_null() {
        return None;
    }
    pipeline(this).desc.host.get()?.gate()
}

/// Whether the title pipeline `this` has a host pipeline made now (no
/// queueing, no waiting).
pub(crate) fn host_pipeline_made(this: This) -> bool {
    !this.is_null() && pipeline(this).desc.host.get().is_some_and(|p| make::is_made(p))
}

/// The rectangle variant of `this`'s host pipeline when it is needed:
/// `Ok(Some)` made, `Ok(None)` not needed (or refused), `Err(())` still being
/// made (queued now on the first ask; the draw drops). Never compiles on
/// the caller's thread unless `XNW_PIPELINE_WAIT=1`.
pub(crate) fn rect_variant<'a>(p: &'a HostPipeline, this: This) -> Result<Option<&'a windows::Win32::Graphics::Direct3D12::ID3D12PipelineState>, ()> {
    if make::blocking() || !p.rect_pending() {
        return Ok(p.rect_pso());
    }
    if p.rect_first_ask()
        && let Some(d) = desc(this)
    {
        static TX: OnceLock<Option<std::sync::mpsc::Sender<Arc<Desc>>>> = OnceLock::new();
        let tx = TX.get_or_init(|| {
            let (tx, rx) = std::sync::mpsc::channel::<Arc<Desc>>();
            let spawned = std::thread::Builder::new().name("xnw-rect-variants".into()).spawn(move || {
                for d in rx {
                    // The description is held: the pipeline outlives the making.
                    if let Some(p) = d.host.get().and_then(|p| make::made(p)) {
                        p.make_rect_pso();
                    }
                }
            });
            spawned.is_ok().then_some(tx)
        });
        if let Some(tx) = tx {
            let _ = tx.send(d);
        }
    }
    Err(())
}

/// The pipelines part of the frame lines: listed, cache hits, made,
/// queued, and the draws and dispatches dropped for want of a pipeline
/// since the last call (by kind, totals over the window).
pub(crate) fn pipelines_text() -> String {
    let dropped: Vec<String> = super::draw::NO_PIPELINE_KINDS
        .iter()
        .zip(&super::draw::NO_PIPELINE)
        .map(|(k, c)| format!("{k} {}", c.swap(0, Ordering::Relaxed)))
        .collect();
    format!(
        "pipelines listed {}, cache hits {} (translated {}, failed {}), made {} (loaded {}), queued {}; dropped for want of a pipeline: {}",
        library::listed(),
        super::shader_cache::HITS.load(Ordering::Relaxed),
        super::shader_cache::TRANSLATED.load(Ordering::Relaxed),
        super::shader_cache::FAILED.load(Ordering::Relaxed),
        READY.load(Ordering::Relaxed),
        library::LOADED.load(Ordering::Relaxed),
        make::waiting(),
        dropped.join(", ")
    )
}

/// From every present: every 64 frames the pipelines line (always, not
/// only with the frame statistics); and ten frames after the first mesh
/// draw after the story load started (the world), one WARN if more than 50
/// pipeline builds are still queued.
pub(crate) fn note_present(frame: u64) {
    static WARNED: AtomicBool = AtomicBool::new(false);
    if frame.is_multiple_of(64) {
        xnw_core::info!("frame pipelines to {frame}: {}", pipelines_text());
    }
    let world = WORLD_FRAME.load(Ordering::Relaxed);
    if world != 0 && frame >= world + 10 && !WARNED.swap(true, Ordering::Relaxed) {
        let queued = make::waiting();
        if queued > 50 {
            xnw_core::warn!("cold shader cache: {queued} builds queued at frame {frame}, 10 frames into the world (run `xnw prebuild` before the next run)");
        }
    }
}

/// The frame of the first mesh draw after the story load started (0: not yet).
static WORLD_FRAME: AtomicU64 = AtomicU64::new(0);

/// `XNW_PIPELINES=off` makes no host pipelines (every draw gets the
/// stand-in): the baseline for start-up time and memory.
fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XNW_PIPELINES").map_or(true, |v| v != "off");
        if !on {
            xnw_core::warn!("d3d12x: XNW_PIPELINES=off: no host pipelines");
        }
        on
    })
}

/// The process's commit now and at its peak, in MiB.
fn commit_mib() -> (usize, usize) {
    use xnw_core::win::{GetCurrentProcess, K32GetProcessMemoryInfo, ProcessMemoryCounters};
    let mut c = ProcessMemoryCounters { cb: size_of::<ProcessMemoryCounters>() as u32, ..Default::default() };
    // SAFETY: a plain query into our own struct.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    (c.pagefile_usage >> 20, c.peak_pagefile_usage >> 20)
}

/// Mesh pipelines asked for (no host pipeline until the fallback).
pub(crate) static MESH: AtomicU64 = AtomicU64::new(0);
static USED_READY: AtomicU64 = AtomicU64::new(0);
static USED_STANDIN: AtomicU64 = AtomicU64::new(0);
static READY: AtomicU64 = AtomicU64::new(0);

/// Failures by reason: count and the first pipeline or shader.
static FAILURES: Mutex<Option<HashMap<String, (u64, String)>>> = Mutex::new(None);

/// Counts a failure; the first of each reason is logged.
pub(crate) fn note_failure(reason: &str, what: &str) {
    let mut f = FAILURES.lock().unwrap_or_else(|e| e.into_inner());
    let entry = f.get_or_insert_with(HashMap::new).entry(reason.to_owned()).or_insert_with(|| (0, what.to_owned()));
    entry.0 += 1;
    if entry.0 == 1 {
        xnw_core::warn!("d3d12x: no host pipeline: {reason} ({what}; logged once per reason)");
    }
}

/// A host pipeline was made.
pub(crate) fn note_ready(p: &HostPipeline) {
    let n = READY.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 64 {
        xnw_core::info!("d3d12x: pipeline ready #{n}: {}", p.name);
    }
}

/// The counts, logged at frames 1, 2, 4, ... (from the draw path).
fn summary_now_and_then() {
    static LAST: AtomicU64 = AtomicU64::new(0);
    let frame = super::presented();
    let last = LAST.load(Ordering::Relaxed);
    if frame > last && frame.is_power_of_two() && LAST.compare_exchange(last, frame, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
        xnw_core::info!("d3d12x: pipelines at frame {frame}: {}", summary());
    }
}

/// Shaders, pipelines, draws and failures so far, as one line.
pub(crate) fn summary() -> String {
    use super::shader_cache::{FAILED, HITS, TRANSLATED};
    let failures = FAILURES.lock().unwrap_or_else(|e| e.into_inner());
    let mut f: Vec<_> = failures.iter().flatten().map(|(r, (n, w))| format!("{n} x {r} (first {w})")).collect();
    f.sort();
    let (commit, peak) = commit_mib();
    format!(
        "commit {commit} MiB (peak {peak}); created {} (by slot {}); shaders translated {} cache hits {} failed {}; host pipelines ready {} waiting {} mesh stubs {}; wave64 vs/ps {}; draws with a host pipeline {} with the stand-in {}; failures [{}]",
        created(),
        created_text(),
        TRANSLATED.load(Ordering::Relaxed),
        HITS.load(Ordering::Relaxed),
        FAILED.load(Ordering::Relaxed),
        READY.load(Ordering::Relaxed),
        make::waiting(),
        MESH.load(Ordering::Relaxed),
        WAVE64_GRAPHICS.load(Ordering::Relaxed),
        USED_READY.load(Ordering::Relaxed),
        USED_STANDIN.load(Ordering::Relaxed),
        f.join("; ")
    )
}

/// IUnknown 3, object 4, device child 1.
pub(crate) const SLOTS: &[Slot] = &[
    (0, query as *const (), Done, "pipeline.rs:544"),
    (2, release as *const (), Done, "pipeline.rs:548"),
    (7, object::parent_device as *const (), Done, "pipeline.rs:557"),
];
pub(crate) const LEN: usize = 24;
static FUNCTIONS: Table<LEN> = object::table::<PIPELINE, LEN>(true, SLOTS);

#[cfg(test)]
mod tests {
    use super::*;

    fn stage_name(s: &str) -> Option<&'static str> {
        ["vertex", "pixel", "compute", "mesh", "fetch", "domain", "hull", "geometry"].into_iter().find(|&n| n == s)
    }

    /// Offline check of the host pipelines: host pipelines for the first-draw shaders a
    /// run's census dumped. Needs `XNW_CENSUS=<run folder>\census`,
    /// `XNW_AGILITY`, and the pack (`XNW_SHADER_PACK` or `XNW_GAME_DRIVE`);
    /// skips without them. State comes from the dumped cache entries (a
    /// slot 55's overrides are not dumped: its base's state is used), so
    /// take the census of an `XNW_BASE_SLOTS=0` run (run 0003); a slot-86
    /// stream's state is not dumped: such a pipeline is checked only if it
    /// has no vertex inputs (census #1, format 28, depth 20), else skipped.
    #[test]
    fn first_draw_pipelines_from_a_census() {
        let (Some(dir), Some(_)) = (std::env::var_os("XNW_CENSUS"), std::env::var_os("XNW_AGILITY")) else {
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let mut groups: std::collections::BTreeMap<String, Vec<std::path::PathBuf>> = Default::default();
        for e in std::fs::read_dir(&dir).expect("census folder").flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            groups.entry(name[..2].to_owned()).or_default().push(e.path());
        }
        let mut made = 0;
        for (n, files) in &groups {
            let mut shaders = Vec::new();
            let (mut state, mut outputs) = (Vec::new(), Vec::new());
            for f in files {
                let name = f.file_name().unwrap().to_string_lossy().into_owned();
                let bytes = std::fs::read(f).unwrap();
                if name.ends_with("-state.entry") {
                    state = bytes;
                } else if name.ends_with("-outputs.entry") {
                    outputs = bytes;
                } else if name.ends_with(".bin") {
                    let stage = stage_name(name.split('-').nth(1).unwrap()).unwrap();
                    let form = if bytes.starts_with(b"DXBC") { Form::Container } else { Form::CacheCode };
                    shaders.push(shader(stage, form, ShaderCode::Owned(bytes)));
                }
            }
            shaders.retain(|s| s.stage != "fetch");
            // Vertex before pixel, as pipelines list them.
            shaders.sort_by_key(|s| s.stage != "vertex");
            let compute = shaders.iter().any(|s| s.stage == "compute");
            let g = (!compute).then(|| {
                if outputs.is_empty() {
                    let mut g = GraphicsState::default();
                    g.targets.NumRenderTargets = 1;
                    g.targets.RTFormats[0].0 = 28;
                    g.depth_format.0 = 20;
                    g.topology.0 = 3;
                    g
                } else {
                    GraphicsState::from_cache(&state, &outputs)
                }
            });
            if outputs.is_empty() && !compute && *n != "01" {
                eprintln!("census #{n}: skipped (stream state not dumped)");
                made += 1;
                continue;
            }
            let p = HostPipeline::new(&shaders, g.as_ref());
            eprintln!("census #{n}: {}", p.as_ref().map_or_else(|| "REFUSED (see the failure log)".into(), |p| format!("host pipeline: {}", p.name)));
            made += usize::from(p.is_some());
        }
        assert_eq!(made, groups.len(), "every first-draw pipeline gets a host pipeline");
    }
}
