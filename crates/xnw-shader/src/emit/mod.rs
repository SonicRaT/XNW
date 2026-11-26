// Ported from VIctory crates/victory-shader/src/emit/mod.rs, commit d2a4bae1f49533ef5cb512fd1b1ba1513f2e1989, GPL-3.0-or-later.
// With VIctory open pull requests (GPL-3.0-or-later) where they touch this file: (8ae232f, d112889), (4c9a922, db20637, 3e8f2d3), (815178d), (fa5189d).
// With VIctory (head 42538cd, on upstream 4322b61; GPL-3.0-or-later): unknown-stride structured buffers read through the runtime stride table (shader side).
// With victory-fork (a26859f, GPL-3.0-or-later): the retry of a program refused for a first half of an interpolation.
// With victory-fork (578d459, emit/mod.rs, GPL-3.0-or-later): the retry of a refused program with threaded joins (`threading`; UP6).
//! DXIL emission (decision 0003, step 3): a console vertex, pixel, compute
//! or mesh shader as a PC DXIL container (SM 6.6) that host D3D12 accepts.
//! Compute shaders keep the console's thread-group size; one that depends
//! on the lane layout (cross-lane operations) declares the console's wave
//! size (`WaveSize`), so the host device must support it.
//!
//! What the host binds for a translated shader ([`HostShader`]):
//!
//! * **User data** at `b0, space 0` (32-bit root constants or a constant
//!   buffer, per stage): dword `index` holds a root parameter's value, as
//!   [`UserData`] lists. A root-constant dword is the constant. A
//!   descriptor table is the byte offset of its first descriptor in its
//!   heap: the descriptor index times 32 for CBV/SRV/UAV tables, times 16
//!   for sampler tables (the console's descriptor sizes; the code adds
//!   offsets in those units). The heap bases of dynamic resources are 0
//!   and not listed.
//! * **Descriptors** through SM 6.6 `ResourceDescriptorHeap[]` and
//!   `SamplerDescriptorHeap[]`, at the index the console address gives, so
//!   the host heaps keep the title's descriptor order. The root signature
//!   needs `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` and
//!   `SAMPLER_HEAP_DIRECTLY_INDEXED`.
//! * **Immediate samplers** as static samplers `s<n>, space 0`, `n` the
//!   position in [`HostShader::samplers`] (the console sampler words) and
//!   [`HostShader::static_samplers`] (the same decoded into a D3D12 static
//!   sampler description). Comparison sampling (added 2026-11-23): an
//!   immediate sampler it uses has a comparison filter there (bit 0x80);
//!   a heap sampler it uses must be a comparison sampler on the host.
//! * **Root CBVs and root SRVs** (raw buffers) as root descriptors
//!   `b<i>` / `t<i>` in [`ROOT_CBV_SPACE`], `i` the title's root parameter
//!   index ([`HostShader::root_cbvs`], [`HostShader::root_srvs`]); the
//!   address the title sets is the buffer's.
//! * **Vertex fetch** (added 2026-11-23). the title's mesh and
//!   vertex programs fetch vertices themselves with typed-format loads
//!   (`tbuffer_load_format_*`): the format (8_8_8_8 unorm, 16_16 snorm,
//!   2_10_10_10 unorm, 8_8_8_8 and 16_16_16_16 uint...) is in the
//!   instruction, so the translation reads raw bytes and converts them
//!   itself. Their descriptors are built around a root SRV (stride 1, no
//!   record limit): the host binds the root SRV `t<slot>` above with the
//!   address the title sets, nothing more. The vertex index, the stride
//!   and the attribute offset are the shader's own arithmetic on data in
//!   that buffer (per-stream headers), not descriptor fields.
//! * **Buffer views in the heap** are declared by how the shader reads
//!   them: raw (ByteAddressBuffer) for dword loads and stores, atomics and
//!   typed-format loads without an index; structured with the used-resource
//!   entry's stride when indexed; typed with four 32-bit float channels for
//!   `buffer_load_format_*` and `buffer_store_format_*`. The host's view at
//!   that index must be of that kind and start at the console descriptor's
//!   base address (raw: `R32_TYPELESS` with the raw flag), whatever kind the
//!   title's view description says. Typed-format loads from heap
//!   descriptors (1,024 world compute programs, ray-tracing material
//!   resolves) read raw views this way.
//!
//! * **Texture atomics** (added 2026-11-23): a texture UAV the shader uses
//!   atomically is declared a 32-bit unsigned single-channel view; the
//!   host's view at that index must be an R32 integer view (64-bit atomics
//!   stop the translation). Global loads from a root SRV read it raw.
//!
//! Entry points: [`translate`] (container bytes as the title passes them
//! at pipeline creation → [`HostShader`]), [`emit`] (the same with the
//! earlier stages already run). The root constants of each stage must
//! cover [`HostShader::user_data_dwords`]. Stability: these types and the
//! binding scheme are what pipeline creation builds on; changes are
//! announced.
//! * **Signatures** as the console container has them (same semantics,
//!   registers and masks), so the input layout and stage linking are the
//!   title's; pixel outputs are plain 32-bit values.
//! * **Mesh shaders** ([`HwStage::Ms`], `ms_6_6`): user data and
//!   descriptors as above (the root constants visible to the mesh stage);
//!   the pipeline is a mesh pipeline (pipeline-state stream, no input
//!   layout) with the title's pixel shader. The thread group is the
//!   console's in lanes (`[numthreads]` in the DXIL), the vertex and
//!   primitive maxima its own, triangles only, no wave size required. The
//!   console numbers groups in x only: the draw path passes the title's
//!   count as `DispatchMesh(x, 1, 1)`. Per-primitive outputs (`PSG1`) are
//!   the console's, without `SV_PrimitiveID` and `SV_CullPrimitive` (the
//!   console never exports them).

pub mod address_map;
mod bitcode;
#[cfg(test)]
mod barycentric_tests;
#[cfg(test)]
mod first_half_tests;
#[cfg(test)]
mod s7_tests;
mod container;
#[cfg(all(windows, any(test, feature = "host")))]
pub mod host;
#[cfg(all(test, windows))]
mod image_atomic_tests;
mod lower;
pub mod mesh;
pub mod rect;
#[cfg(all(test, windows))]
mod mesh_tests;
mod sampler;
#[cfg(all(test, windows))]
mod sampler_words_tests;
mod shader;
mod threading;
#[cfg(windows)]
pub mod validator;
#[cfg(all(test, windows))]
mod valu_tests;
mod wave;
#[cfg(all(test, windows))]
mod world_tests;
#[cfg(all(test, windows))]
mod null_kind_tests;
// XNW: the non-uniform heap index on the software adapter.
#[cfg(all(test, windows))]
mod heap_uniform_tests;

use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, RootType};
use crate::{ShaderError, io, ir, resource, structure};

pub use sampler::StaticSampler;
pub use shader::{ADDRESS_MAP_SPACE, DESCRIPTOR_WORDS_SPACE, HEAP_BOUNDS_SPACE, PAGE_TABLE_SPACE, ROOT_CBV_SPACE, ROOT_LEN_SPACE, RawWindow, SAMPLER_SPACE, STRIDE_SPACE, StrideEntry, USER_DATA_SPACE};

/// XNW (page table): bits 20-31 of dword 3 of a buffer-view record
/// that is a GFX10 V# over console memory, read through the page table
/// (`XNW_PAGETABLE=1`); bits 0-19 hold the heap index of the host view
/// typed (format) accesses still use.
pub const PAGE_TAG: u32 = 0x5A7;

/// XNW: what a program translated with the page table on still
/// reads or writes through host copies or host views, by kind
/// ([`HostShader::host_copies`]); the runtime keeps those for it.
pub mod host_copies {
    /// Typed (format) buffer loads or stores.
    pub const TYPED_BUFFERS: u32 = 1;
    /// Buffers through root descriptors (root SRV/CBV/UAV by address).
    pub const ROOT_VIEWS: u32 = 2;
    /// Buffer descriptors the code builds or holds as constants.
    pub const BUILT_DESCRIPTORS: u32 = 4;
    /// Textures and images (host textures in every mode).
    pub const TEXTURES: u32 = 8;
}

/// The translator's version (XNW addition). Which counter to raise (
/// the shader cache's rule, `xnw-gpu` `shader_cache.rs`):
///
/// - [`VERSION`]: a change that makes refused programs translate (a new
///   instruction, a lifted refusal) and leaves the output of programs that
///   already translated alone. Programs whose translation FAILED under an
///   older `VERSION` are retried; programs that translated keep their
///   cached output.
/// - [`VERSION_BREAKING`]: a change to the DXIL or a [`HostShader`] field
///   of programs that already translated (a miscompile fix, a binding
///   layout). Every program is translated again (`xnw prebuild` or the
///   launch prebuild does it before the title starts).
/// - [`MESH_VERSION`]: as `VERSION_BREAKING`, for mesh programs only.
///
/// When unsure, raise `VERSION_BREAKING`. History of `VERSION`:
/// 4: (stride table at t0 space 3, compute derivatives) changed
/// emitted DXIL under 3; caches from before served stale translations.
/// 11: (gpu descriptor mode: window dwords after each program's own
/// root parameters) and (waterfall descriptors per lane) merged.
/// 12: mesh fallback programs add the piece's first group (`t0`).
/// 13: (pixel programs: spilled masks, CBV dwords on dead join edges,
/// packed input halves, SGPRs above the user data).
/// 14: (64-bit image atomics, barycentrics, compute mask data).
/// 15: (one loop budget per invocation; pixel budget 4,096): breaking
/// for every program with a loop.
/// 16: (VIctory mask dwords at joins, s_bcnt1 SCC;
/// compare-and-swap image atomics).
/// 17: (victory-fork: heap-picked buffers read raw in scalar loads).
/// 18: (native mesh: primitive indices masked to 9 bits, from upstream).
/// 19: (loop budget 256 per invocation in every stage): breaking for
/// every program with a loop.
/// 20: (victory-fork unspill: parked scalars as virtual registers):
/// breaking for every program that parks scalars in VGPR lanes.
/// 21: (root UAVs, the address map in space 5 and the
/// descriptor words in space 6, ray intersection, LDS and buffer bytes and
/// halves, spill-table copies read at entry; [`HostShader`] gains
/// root_uavs, address_map, descriptor_words: `VERSION_BREAKING` raised too).
/// 22: (every heap index bounded by the heap-bound constants, b0 space
/// 7): breaking for every program that indexes a descriptor heap.
/// 23: sampler dwords held as data (a waterfall loop over a sampler's
/// dwords reads them as their sampler heap offsets; a sampler heap dword
/// on a dead join edge is dropped).
/// 24: computed barycentrics on a host that has them (victory-fork
/// 578d459); also retries the earlier lifts (sampler phi, wave64 s_mov mask).
/// 25: first halves of interpolations read as data, on a retry (
/// victory-fork a26859f).
/// 26: (every heap index that may differ between lanes, or that a
/// block only some lanes run reads, is declared non-uniform): breaking for
/// every program with such an index.
/// 27: (victory-fork: 16-bit halves moved through a view of a
/// table go as unsigned integers, [`HostShader::half_views`]): breaking for
/// the programs that move halves so.
/// 28: victory-fork 578d459's lane-mask phi joins (`phi_masks`) where an
/// edge's mask has no entry of its own: retries the refusals (no program
/// that translated changes).
pub const VERSION: u32 = 28;

/// Part of every shader-cache key: raise it when programs that
/// already translated must be translated again (see [`VERSION`]).
/// 1: at `VERSION` 13. 2: merged at `VERSION` 16 (the output of every
/// program with a loop changed; the mask dwords at joins).
/// 3: merged at `VERSION` 19 (raw scalar heap-buffer reads, loop
/// budget 256 in every stage). 4: merged at `VERSION` 20 (parked
/// scalars as virtual registers). 5: merged at `VERSION` 21 (the
/// cache record gains root UAVs and the address-map and words flags). 6: (heap indices
/// bounded). 7: merged at `VERSION` 26 (non-uniform heap indices).
/// 8: merged at `VERSION` 27 (half views, the cache record gains them;
/// a comparison sampler replaces the reduction).
pub const VERSION_BREAKING: u32 = 8;

/// The mesh fallback's own translator version, part of the cache key of
/// mesh programs only (a `VERSION` change retranslates every program,
/// thousands, on the next load). 1: the piece's first group (`t0`).
/// 2: root SRV reads bounded by lengths in `b0` of [`ROOT_LEN_SPACE`].
/// 3: primitive indices masked to 9 bits (VIctory).
/// 4: loop budget 256 per invocation in every stage.
pub const MESH_VERSION: u32 = 4;

/// The loop budget without an override (2^16; the base's 2^20). XNW
/// 256 per invocation in every stage, after victory-fork's finding
/// (its README): a single submission over the 2 s driver limit
/// removes the device; their `VICTORY_SHADER_LOOP_LIMIT=256` was stable,
/// 32 gave a wrong picture, 1,024 hung after about 7 minutes. Theirs
/// counts per loop and entry (lower.rs count_iteration), ours all loops of
/// an invocation together, so ours is the tighter of the two for nested
/// loops. Cut short at this: PS_IntegrateEnvBRDF's 1,024-sample loops (a
/// lookup-table build) and long BVH traversals.
pub const DEFAULT_LOOP_LIMIT: u32 = lower::LOOP_LIMIT;

/// XNW: the pixel programs' budget, the same as the others now
/// (it was 4,096 after a survey: largest constant pixel loop 1,024,
/// 2,048 per invocation).
pub const DEFAULT_PIXEL_LOOP_LIMIT: u32 = 256;

/// The back edges one invocation (lane) may take in all its loops together
/// before it leaves the program (pixel: discarded), by stage: pixel
/// [`DEFAULT_PIXEL_LOOP_LIMIT`], others [`DEFAULT_LOOP_LIMIT`].
/// `XNW_LOOP_LIMIT_PIXEL`, `_COMPUTE` (compute and mesh programs in their
/// compute form), `_MESH`, `_VERTEX` (decimal or `0x` hex, at least 1) set
/// them. A limit is part of the translation, so the shader cache keys on
/// [`loop_limit_key`].
pub fn loop_limit_for(stage: HwStage, fallback: bool) -> u32 {
    limits()[limit_slot(stage, fallback)]
}

/// What the cache key holds for the loop limits: nothing while all are
/// the defaults (XNW_LOOP_LIMIT_* unset), else every limit.
pub fn loop_limit_key() -> Vec<u8> {
    let l = limits();
    if l == defaults() { Vec::new() } else { l.iter().flat_map(|v| v.to_le_bytes()).collect() }
}

/// The clean-up passes run over a translated program before it is written
/// ported from victory-fork `shader_opt`; `bitcode::opt`): a set of
/// pass bits, OFF by default (victory-fork: with the fork's passes on,
/// black blocks and streaks in the prologue; off renders clean).
/// `XNW_SHADER_OPT`: unset, `0` or `none`: no pass; `all`: every pass; a
/// list of names (`dead,fold,reuse,facts,loads`, separated by `,`, `+` or
/// blanks) or a number (decimal or `0x` hex: 1 dead, 2 fold, 4 reuse,
/// 8 facts, 16 loads), so a bad pass can be bisected. Part of the cache key
/// ([`shader_opt_key`]).
pub fn shader_opt() -> u32 {
    static PASSES: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *PASSES.get_or_init(|| parse_shader_opt(std::env::var("XNW_SHADER_OPT").ok().as_deref()))
}

/// `XNW_SHADER_OPT`'s value; no pass when it is unset or unreadable.
fn parse_shader_opt(value: Option<&str>) -> u32 {
    use bitcode::opt::{DEAD, FACTS, FOLD, LOADS, MODULE, REUSE};
    let Some(v) = value.map(str::trim) else { return 0 };
    if let Some(n) = v.strip_prefix("0x").map_or_else(|| v.parse().ok(), |h| u32::from_str_radix(h, 16).ok()) {
        return n & MODULE;
    }
    let mut passes = 0;
    for name in v.split([',', '+', ' ']).filter(|n| !n.is_empty()) {
        passes |= match name.to_ascii_lowercase().as_str() {
            "all" => MODULE,
            "dead" => DEAD,
            "fold" => FOLD,
            "reuse" => REUSE,
            "facts" => FACTS,
            "loads" => LOADS,
            _ => return 0,
        };
    }
    passes
}

/// What the cache key holds for the clean-up passes: nothing while none
/// runs (the default), else the pass bits.
pub fn shader_opt_key() -> Vec<u8> {
    let p = shader_opt();
    let mut k = if p == 0 { Vec::new() } else { [b"opt".as_slice(), &p.to_le_bytes()].concat() };
    if wave_by_need() {
        k.extend(b"wave");
    }
    k
}

/// `XNW_WAVE_SIZE=need`: a compute program declares the console's wave
/// size only where its lanes are observable (lane reads, DPP, swizzles,
/// permutes, `v_mbcnt`, masks read as numbers or held as lane sets) or
/// where it is one console wave sharing LDS; the others take any host wave
/// size. Off (the default): every compute program declares it. Part
/// of the cache key ([`shader_opt_key`]).
pub fn wave_by_need() -> bool {
    #[cfg(test)]
    if let Some(on) = WAVE_BY_NEED.with(std::cell::Cell::get) {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XNW_WAVE_SIZE").is_ok_and(|v| v.trim() == "need"))
}

#[cfg(test)]
thread_local! {
    /// [`wave_by_need`] for this test thread, in place of the switch.
    pub(crate) static WAVE_BY_NEED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// The wave size a compute program declares (see [`wave_by_need`]):
/// `wave` lanes (the console's), or None for any. `console_lanes`: its
/// lanes are observable; `one_wave`: its group is one console wave.
pub(crate) fn compute_wave_size(wave: u32, console_lanes: bool, one_wave: bool, lds_dwords: u32, by_need: bool) -> Option<u32> {
    (!by_need || console_lanes || one_wave && lds_dwords > 0).then_some(wave)
}

/// Slot in [`limits`]: pixel, compute, mesh, vertex (and the others).
fn limit_slot(stage: HwStage, fallback: bool) -> usize {
    match stage {
        HwStage::Ps => 0,
        HwStage::Cs => 1,
        HwStage::Ms if fallback => 1,
        HwStage::Ms => 2,
        _ => 3,
    }
}

fn defaults() -> [u32; 4] {
    [DEFAULT_PIXEL_LOOP_LIMIT, DEFAULT_LOOP_LIMIT, DEFAULT_LOOP_LIMIT, DEFAULT_LOOP_LIMIT]
}

fn limits() -> [u32; 4] {
    static LIMITS: std::sync::OnceLock<[u32; 4]> = std::sync::OnceLock::new();
    *LIMITS.get_or_init(|| {
        let d = defaults();
        let var = |k: &str, d: u32| parse_loop_limit(std::env::var(k).ok().as_deref(), d);
        [var("XNW_LOOP_LIMIT_PIXEL", d[0]), var("XNW_LOOP_LIMIT_COMPUTE", d[1]), var("XNW_LOOP_LIMIT_MESH", d[2]), var("XNW_LOOP_LIMIT_VERTEX", d[3])]
    })
}

/// A limit switch's value; `default` when it is unset, unreadable or 0.
/// Pure, for testing.
fn parse_loop_limit(value: Option<&str>, default: u32) -> u32 {
    let parsed = value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        });
    match parsed {
        Some(n) if n > 0 => n,
        _ => default,
    }
}

/// How the console's fused f32 multiply-adds are written (XNW addition).
/// Part of a translation's identity: a shader cache must key on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Fma {
    /// Bit-exact up to double rounding: double `fma`, rounded to f32 (the
    /// RX 5700 XT fuses only doubles).
    #[default]
    Exact,
    /// `mad` (DXIL `FMad`), which the host may or may not fuse.
    Fast,
}

/// Choices made per translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Options {
    pub fma: Fma,
    /// How a mesh program is emitted (XNW).
    pub mesh: MeshForm,
    /// Where descriptors come from (XNW).
    pub descriptors: Descriptors,
    /// XNW: `XNW_LOOP_PROBE`): a pixel invocation that runs out of its
    /// loop budget writes magenta to its targets instead of discarding.
    pub probe: bool,
    /// XNW: `XNW_LOOP_COUNTS=1`): the program's slot in the
    /// runtime's loop-count buffer (its hash's first 8 hex digits mod 1024):
    /// each loop reports its count there. None: nothing emitted.
    pub loop_counts: Option<u32>,
    /// XNW: `XNW_PAGETABLE=1`, with GPU-side descriptors): buffer
    /// records tagged [`PAGE_TAG`] are read through the page table.
    pub pagetable: bool,
    /// XNW: the host has SM 6.1 barycentrics (OPTIONS3
    /// BarycentricsSupported): an interpolation with computed barycentrics
    /// reads the attribute's vertices (`attributeAtVertex`) and the
    /// barycentrics (`SV_Barycentrics`), as victory-fork 578d459 does.
    /// Off: refused, as before (the RX 5700 XT reports 0; WARP 1.0.22, 1).
    pub barycentrics: bool,
    /// XNW (`XNW_HEAP_NONUNIFORM=<hash prefix>,...|all`): every
    /// descriptor heap index of the program is declared non-uniform. Our
    /// "scalar" indices are per-lane values that lanes outside the
    /// executing mask never computed; WARP 1.0.22 reads a uniform-declared
    /// index from lane 0 whether or not it runs (ref4b: CS_Prop_BatchCull
    /// 0451d340, lane 0 slot 0, a null, lane 1 slot 729: AV at 0x388).
    pub nonuniform_heap: bool,
    /// XNW (`XNW_HEAP_UNIFORMITY=off`): every heap index keeps
    /// the form the lowering gave it; the uniformity pass
    /// (`bitcode::uniform`), which declares non-uniform each index that
    /// may differ between lanes, is skipped. For bisecting only.
    pub keep_uniform_heap: bool,
    /// XNW: the host reads a uniform-declared heap index from
    /// lane 0 whether lane 0 runs or not (WARP 1.0.22): an index
    /// read where only some lanes run is declared non-uniform too
    /// ("masked" in `bitcode::uniform`). Off: only indices that vary
    /// between lanes (the hardware adapter: its compiler reads the first
    /// active lane). `XNW_ADAPTER=warp`, or `XNW_HEAP_UNIFORMITY=masked`.
    pub masked_lanes: bool,
}

thread_local! {
    /// [`Options::nonuniform_heap`] of the translation on this thread.
    pub(crate) static NONUNIFORM_HEAP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// [`Options::keep_uniform_heap`] and [`Options::masked_lanes`] of the
    /// translation on this thread.
    pub(crate) static HEAP_UNIFORMITY: std::cell::Cell<(bool, bool)> = const { std::cell::Cell::new((false, false)) };
}

/// `XNW_HEAP_UNIFORMITY`: `off` ([`Options::keep_uniform_heap`]), `masked`
/// ([`Options::masked_lanes`] on any adapter); else the masked rule with
/// `XNW_ADAPTER=warp` only (read each time: replay sets it when it picks
/// the adapter). Returns (keep, masked).
pub fn heap_uniformity() -> (bool, bool) {
    let v = std::env::var("XNW_HEAP_UNIFORMITY").unwrap_or_default();
    let warp = std::env::var("XNW_ADAPTER").is_ok_and(|a| a.trim().eq_ignore_ascii_case("warp"));
    (v.trim() == "off", v.trim() == "masked" || warp)
}

/// Whether `XNW_HEAP_NONUNIFORM` names the program with container hash `hash`.
pub fn nonuniform_heap_for(hash: u128) -> bool {
    static LIST: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let list = LIST.get_or_init(|| std::env::var("XNW_HEAP_NONUNIFORM").unwrap_or_default().split(',').map(|s| s.trim().to_ascii_lowercase()).filter(|s| !s.is_empty()).collect());
    let hex = format!("{hash:032x}");
    list.iter().any(|p| p == "all" || hex.starts_with(p.as_str()))
}

/// Where a translated program finds its descriptors (XNW; the record
/// layout is "Record layout for GPU-side descriptors").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Descriptors {
    /// Traced to root parameters at translation; the host makes the slots.
    #[default]
    Plan,
    /// Read by the program from the title's records (`XNW_DESC=gpu`): a
    /// table's records are loaded from the shared window through user-data
    /// dwords [`WINDOW_DWORD`] and [`WINDOW_DWORD`]` + 1` of the table's
    /// parameter, and every use indexes the global heaps with the record's
    /// dword 0 (a typed null unless dword 3 is the tag).
    Gpu,
}

/// [`UserData::dword`] of a table or heap parameter's shared-window raw
/// view (host heap index); the next is the table's byte offset in it.
pub const WINDOW_DWORD: u32 = 0x100;
/// Dword 3 of a record the device wrote ('vVIE').
pub const RECORD_TAG: u32 = 0x4549_5676;
/// Host view heap slots in `XNW_DESC=gpu` (16 nulls, the view table, the
/// window views); indices past it read the null of the expected kind.
pub const GPU_VIEW_SLOTS: u32 = 16 + (1 << 20) + 4096;
/// Host sampler heap slots in `XNW_DESC=gpu`.
pub const GPU_SAMPLER_SLOTS: u32 = 2048;

/// The form of an emitted mesh program (XNW).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MeshForm {
    /// A mesh shader (`ms_6_6`), for hosts with mesh shaders.
    #[default]
    Native,
    /// A compute shader (`cs_6_6`, the console's wave size) that writes one
    /// record per group and the group's index triples ([`mesh`]); drawn by
    /// [`mesh::passthrough_vs`] and the title's pixel shader.
    Fallback,
}

/// A translated shader and what the host binds for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostShader {
    pub stage: HwStage,
    /// The DXIL container, hashed as D3D12 expects.
    pub dxil: Vec<u8>,
    /// The user-data dwords the shader reads.
    pub user_data: Vec<UserData>,
    /// Immediate samplers: the four sampler words of `s<n>`.
    pub samplers: Vec<[u32; 4]>,
    /// The same samplers decoded, `s<n>` at position n: the static sampler
    /// descriptions the host's root signature declares.
    pub static_samplers: Vec<StaticSampler>,
    /// Root CBVs the shader reads: root parameter indices, each bound as
    /// `b<index>` in [`ROOT_CBV_SPACE`].
    pub root_cbvs: Vec<u8>,
    /// Root SRVs (raw buffers) the shader reads: `t<index>` in
    /// [`ROOT_CBV_SPACE`].
    pub root_srvs: Vec<u8>,
    /// Root UAVs (raw buffers) the shader reads or writes: `u<index>` in
    /// [`ROOT_CBV_SPACE`] (XNW: the BVH builder's root UAVs).
    pub root_uavs: Vec<u8>,
    /// The shader reads title memory at addresses it computes (pointers
    /// it loads, a BVH it walks): the host binds the address map as a
    /// root SRV, `t0` in [`ADDRESS_MAP_SPACE`]. See [`address_map`] for
    /// its layout (VIctory, 1a2a3a1, GPL-3.0-or-later).
    pub address_map: bool,
    /// The shader reads descriptors as numbers (a buffer's address, say):
    /// the host binds the console words of the view heap's descriptors, 32
    /// bytes per slot as the console lays them out, as a root SRV, `t0` in
    /// [`DESCRIPTOR_WORDS_SPACE`]. A slot whose words the host cannot give
    /// reads zero (VIctory).
    pub descriptor_words: bool,
    /// Table descriptors (root parameter, dword offset in the table) whose
    /// buffer the shader reads or writes sixteen bits a channel
    /// (`buffer_load_format_d16_*`, `buffer_store_format_d16_*`), in
    /// order. On the console each channel's 16 bits move as they are
    /// through a 16-bit float view and through a 16-bit integer one alike
    /// (the title keeps half floats, and indices beside them, in
    /// `R16G16B16A16_UINT` and `R16_UINT` buffers). The DXIL moves them as
    /// unsigned integers, which an unsigned 16-bit view carries unchanged;
    /// the host shows the shader a 16-bit float or signed buffer view at
    /// these places as the unsigned format of the same layout.
    /// shortcut: a view in a normalised 16-bit format, whose values the
    /// console would convert to halves, is read as its integers; a
    /// descriptor picked at run time keeps the float path (its slot, and so
    /// its format, is known only then). (victory-fork open pull
    /// request, emit/mod.rs, head 5a55384, GPL-3.0-or-later.)
    pub half_views: Vec<(u8, u32)>,
}

impl HostShader {
    /// XNW: with the page table on, what the program still needs host
    /// copies for ([`host_copies`] bits, from the container's `PRIV`
    /// part); `None`: translated without the page table.
    pub fn host_copies(&self) -> Option<u32> {
        let c = Container::parse(&self.dxil).ok()?;
        let d = c.part(FourCC(*b"PRIV"))?.data;
        (d.get(..4) == Some(b"XNWH")).then(|| d.get(4..8).map(|b| u32::from_le_bytes(b.try_into().unwrap_or_default())))?
    }

    /// XNW: whether the program reads the page table (`t0` in
    /// [`PAGE_TABLE_SPACE`]): its PSV0 lists that root SRV.
    pub fn reads_page_table(&self) -> bool {
        self.psv_lists(PAGE_TABLE_SPACE)
    }

    /// XNW: whether the program reads the heap bounds (`b0` in
    /// [`HEAP_BOUNDS_SPACE`]): its PSV0 lists that CBV.
    pub fn reads_heap_bounds(&self) -> bool {
        self.psv_lists_kind(HEAP_BOUNDS_SPACE, 2)
    }

    /// Whether `PSV0` lists a raw SRV at `t0` in `space`.
    fn psv_lists(&self, space: u32) -> bool {
        self.psv_lists_kind(space, 4)
    }

    /// Whether `PSV0` lists register 0 of PSV resource type `kind` (2 CBV,
    /// 4 raw SRV) in `space`.
    fn psv_lists_kind(&self, space: u32, kind: u32) -> bool {
        let Ok(c) = Container::parse(&self.dxil) else { return false };
        let Some(psv) = c.part(FourCC::PSV0) else { return false };
        let d = psv.data;
        let word = |at: usize| d.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap_or_default()));
        let Some(info) = word(0) else { return false };
        let at = 4 + info as usize;
        let (Some(n), Some(size)) = (word(at), word(at + 4)) else { return false };
        (0..n as usize).any(|k| {
            let e = at + 8 + k * size as usize;
            word(e + 4) == Some(space) && word(e) == Some(kind) && word(e + 8) == Some(0)
        })
    }

    /// Whether the shader reads the runtime stride table (`t0` in
    /// [`STRIDE_SPACE`], VIctory): its `PSV0` lists that
    /// binding. Changed from VIctory (a `stride_table` field): read from
    /// the container so the host side and its shader cache need no change.
    pub fn reads_stride_table(&self) -> bool {
        self.psv_lists(STRIDE_SPACE)
    }

    /// Size of the user-data buffer the shader declares, in dwords (a
    /// multiple of 4; 0 when it reads none). Root constants must cover it.
    pub fn user_data_dwords(&self) -> u32 {
        let n = self
            .user_data
            .iter()
            .map(|u| u.index + 1)
            .max()
            .unwrap_or(0);
        n.next_multiple_of(4)
    }
}

/// One dword of user data and the root parameter that fills it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserData {
    /// Dword in the user-data buffer (the console's user-data index).
    pub index: u32,
    /// Root parameter index in the title's root signature.
    pub slot: u8,
    /// `Constants` or `Table` (see the module notes for the value).
    pub root: RootType,
    /// Dword within the parameter.
    pub dword: u32,
}

/// XNW: the options the runtime translates a program with; xnw-gpu
/// (`pipeline::options_for`) and the shader gate (`shader-gate`) both call
/// this, so the gate checks what the title gets. `stage` as the runtime
/// names it (`"mesh"`: the compute form, this host has no mesh shaders),
/// `hash` the container's 128-bit hash. Switches, read once:
/// `XNW_DESC=gpu` ([`gpu_descriptors`]), `XNW_FMA` / `XNW_FMA_EXACT`
/// ([`fma_for`]), `XNW_PAGETABLE=1`. `probe` and `loop_counts` are the
/// runtime's instruments (`XNW_LOOP_PROBE`, `XNW_LOOP_COUNTS`).
pub fn runtime_options(stage: &str, hash: u128, probe: bool, loop_counts: Option<u32>) -> Options {
    static PAGETABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let pagetable = *PAGETABLE.get_or_init(|| std::env::var("XNW_PAGETABLE").is_ok_and(|v| v.trim() == "1"));
    let uniformity = heap_uniformity();
    Options {
        fma: fma_for(hash),
        mesh: if stage == "mesh" { MeshForm::Fallback } else { MeshForm::Native },
        descriptors: if gpu_descriptors() { Descriptors::Gpu } else { Descriptors::Plan },
        probe,
        loop_counts,
        pagetable,
        // XNW: a host with barycentrics; xnw-gpu's `options_for`
        // puts the device's answer here.
        barycentrics: true,
        nonuniform_heap: nonuniform_heap_for(hash),
        keep_uniform_heap: uniformity.0,
        masked_lanes: uniformity.1,
    }
}

/// XNW: whether programs read their descriptor records themselves
/// (`XNW_DESC=gpu`).
pub fn gpu_descriptors() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XNW_DESC").is_ok_and(|v| v == "gpu"))
}

/// XNW: the multiply-add mode of the program with container hash `hash`:
/// `fast` by default, `XNW_FMA=exact` for all, `XNW_FMA_EXACT=<hash>,...`
/// for the listed ones.
pub fn fma_for(hash: u128) -> Fma {
    static ALL: std::sync::OnceLock<Fma> = std::sync::OnceLock::new();
    static LISTED: std::sync::OnceLock<Vec<u128>> = std::sync::OnceLock::new();
    let all = *ALL.get_or_init(|| match std::env::var("XNW_FMA").as_deref() {
        Ok("exact") => Fma::Exact,
        _ => Fma::Fast,
    });
    let listed = LISTED.get_or_init(|| std::env::var("XNW_FMA_EXACT").unwrap_or_default().split(',').filter_map(|h| u128::from_str_radix(h.trim(), 16).ok()).collect());
    if listed.contains(&hash) { Fma::Exact } else { all }
}

/// XNW: the instruction count of the last program this thread
/// translated (after any clean-up pass; 0 before the first).
pub fn last_insts() -> usize {
    shader::INSTS.with(std::cell::Cell::get)
}

/// XNW: of the last program this thread translated, the heap
/// handles the uniformity pass declared non-uniform, those of them whose
/// index varies between lanes (the rest: read where only some lanes run),
/// and all heap handles (0, 0, 0 with the pass off).
pub fn last_nonuniform() -> (usize, usize, usize) {
    shader::NONUNIFORM_MARKED.with(std::cell::Cell::get)
}

thread_local! {
    /// XNW: milliseconds and counts per emit step of this thread's
    /// translations since the last [`take_laps`] (shader-gate `--stages`).
    static LAPS: std::cell::RefCell<Vec<(&'static str, f64, u32)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// XNW: adds the time since `since` to the emit step `name`.
pub(crate) fn lap(name: &'static str, since: std::time::Instant) {
    let ms = since.elapsed().as_secs_f64() * 1e3;
    LAPS.with(|l| {
        let mut l = l.borrow_mut();
        match l.iter_mut().find(|e| e.0 == name) {
            Some(e) => (e.1, e.2) = (e.1 + ms, e.2 + 1),
            None => l.push((name, ms, 1)),
        }
    });
}

/// XNW: the emit steps timed on this thread since the last call:
/// (name, milliseconds, times entered).
pub fn take_laps() -> Vec<(&'static str, f64, u32)> {
    LAPS.with(|l| std::mem::take(&mut *l.borrow_mut()))
}

/// Translates the vertex, pixel, compute or mesh program of a console
/// shader container.
pub fn translate(container: &[u8]) -> Result<HostShader, ShaderError> {
    translate_with(container, Options::default())
}

/// [`translate`] with options (XNW addition: the multiply-add mode).
///
/// victory-fork a26859f: a program refused because the first half of an
/// interpolation is read as data is translated once more with first halves
/// evaluated as the pair they start (`lower::with_first_half_pairs`), and
/// only such a program: with that on for every program, 8,551 pixel
/// programs of the title that already translated came out changed (an extra
/// evaluation in each), for the sake of 9 that did not translate.
pub fn translate_with(container: &[u8], opts: Options) -> Result<HostShader, ShaderError> {
    translate_patched(container, opts, &[])
}

/// `translate_with` over code with `words` (byte offset, instruction word)
/// written over the program's code after its export patches: the debug
/// export of `world-render --lighting --ps-patch` (XNW LIT3). Same-size
/// replacements only; nothing else calls it with words.
pub fn translate_patched(container: &[u8], opts: Options, words: &[(u32, u32)]) -> Result<HostShader, ShaderError> {
    match translate_plain(container, opts, words) {
        Err(ShaderError::Unsupported { what, .. }) if what == lower::FIRST_HALF => {
            lower::with_first_half_pairs(|| translate_plain(container, opts, words))
        }
        other => other,
    }
}

fn translate_plain(container: &[u8], opts: Options, words: &[(u32, u32)]) -> Result<HostShader, ShaderError> {
    let c = Container::parse(container)?;
    let dxil = c
        .part(FourCC::DXIL)
        .ok_or(ShaderError::Malformed("no DXIL part"))?;
    let native = crate::native_programs(dxil.data)?
        .into_iter()
        .find(|p| {
            matches!(
                p.stage,
                HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms
            )
        })
        .ok_or(ShaderError::Unsupported {
            offset: 0,
            what: "a container without a vertex, pixel, compute or mesh program",
        })?;
    // A compute program whose root arguments all sit past the 16 user SGPRs
    // finds them through the spill table `s[0:1]` points to.
    let spilled = native.stage == HwStage::Cs
        && !native.layout.is_empty()
        && native
            .layout
            .iter()
            .all(|r| native.stage.user_sgpr(r.user_data) >= 16);
    let t = std::time::Instant::now();
    let mut code = native.patched_code();
    for &(off, word) in words {
        let at = off as usize;
        let slot = code.get_mut(at..at + 4).ok_or(ShaderError::Malformed("debug patch outside the code"))?;
        slot.copy_from_slice(&word.to_le_bytes());
    }
    let p = ir::build_with(&code, native.wave64, native.vgprs, spilled);
    lap("ir", t);
    let p = p?;
    let t = std::time::Instant::now();
    let res = resource::track_with(&p, &native.layout, native.stage, opts.descriptors == Descriptors::Gpu);
    let io = res.and_then(|res| Ok((res, io::raise(&p, &native)?)));
    lap("resources+io", t);
    let (res, io) = io?;
    let st = structure::structure(&p)?;
    let original = emit_with(&c, &native, &p, &st, &io, &res, opts);
    // victory-fork 578d459 (emit/mod.rs, GPL-3.0-or-later): refused
    // programs only, again with the joins of uniform diamonds whose next
    // branch one arm's flag decides copied for that arm (`threading`):
    // registers recycled on that arm then never reach the other successor.
    // (Not for spilled user SGPRs, whose entry reads renaming would lose.)
    if !matches!(original, Err(ShaderError::Unsupported { .. })) || spilled {
        return original;
    }
    let edges = threading::constant_edges(&p, &st);
    if edges.is_empty() {
        return original;
    }
    let p = ir::thread_edges(&p, &edges)?;
    let st = structure::structure(&p)?;
    let res = resource::track_with(&p, &native.layout, native.stage, opts.descriptors == Descriptors::Gpu)?;
    let io = io::raise(&p, &native)?;
    emit_with(&c, &native, &p, &st, &io, &res, opts)
}

/// Emits a program whose earlier stages already ran, with default options.
pub fn emit(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
) -> Result<HostShader, ShaderError> {
    emit_with(c, native, p, st, io, res, Options::default())
}

/// Emits a program whose earlier stages already ran.
pub fn emit_with(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
    opts: Options,
) -> Result<HostShader, ShaderError> {
    let sig = |fourcc| -> Result<Vec<container::SigElement>, ShaderError> {
        match c.part(fourcc) {
            Some(part) => signature(part.data),
            None => Ok(Vec::new()),
        }
    };
    let pixel = native.stage == HwStage::Ps;
    let mut inputs = sig(FourCC::ISG1)?;
    let mut outputs = sig(FourCC::OSG1)?;
    // Interpolation as DXC declares it by default; pixel inputs that are
    // read get the mode their barycentrics say. Conservative depth outputs
    // (`SV_DepthGreaterEqual`/`SV_DepthLessEqual`) require centroid or sample
    // interpolation on `SV_Position`.
    let conservative_depth = pixel && outputs.iter().any(|o| matches!(o.kind, 18 | 19));
    let psv_modes = c
        .part(FourCC::PSV0)
        .and_then(|p| psv_interp(p.data, pixel));
    let interpolated = if pixel { &mut inputs } else { &mut outputs };
    for e in interpolated.iter_mut() {
        e.interp = match (e.kind, e.comp) {
            (3, _) => if conservative_depth { 5 } else { 4 },
            // Constant, as DXC declares it (the AMD compiler crashed on a
            // front face without a mode).
            (13, _) => 1,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
        if let Some(modes) = &psv_modes
            && let Some(&(_, _, _, m)) = modes
                .iter()
                .find(|&&(row, col, kind, m)| row == e.row && col == e.col && kind == e.kind && m != 0)
        {
            e.interp = if e.kind == 3 && conservative_depth && m == 4 {
                5
            } else {
                m
            };
        }
    }
    // Per-primitive outputs are constant. The console never exports a
    // primitive id or cull flag a mesh shader writes (it culls with the
    // primitive export), so those rows are left out.
    let mut prims = sig(FourCC::PSG1)?;
    prims.retain(|e| !matches!(e.kind, 10 | 30));
    for e in &mut prims {
        e.interp = 1;
    }
    // A mesh group's id arrives multiplied by max(threads, vertices,
    // primitives) (the code divides by that); the thread count is in PSV0
    // from runtime info version 2 on.
    // shortcut: older containers fall back to the group's lane count, right
    // when the HLSL group fills its waves (the title's do).
    let mesh_stride = match &native.io {
        crate::native::StageIo::Ms(ms) => {
            let psv = c.part(FourCC::PSV0).map(|p| p.data);
            let threads = psv
                .filter(|d| crate::dxbc::u32_at(d, 0).is_ok_and(|n| n >= 48))
                .and_then(|d| crate::dxbc::u32_at(d, 4 + 36).ok());
            threads.map_or(ms.threads, |t| {
                t.max(ms.max_vertices).max(ms.max_primitives)
            })
        }
        _ => 0,
    };
    NONUNIFORM_HEAP.with(|f| f.set(opts.nonuniform_heap));
    HEAP_UNIFORMITY.with(|f| f.set((opts.keep_uniform_heap, opts.masked_lanes)));
    let l = lower::lower(&lower::Parts {
        native,
        p,
        st,
        io,
        res,
        inputs,
        outputs,
        prims,
        mesh_stride,
        fma: opts.fma,
        fallback: native.stage == HwStage::Ms && opts.mesh == MeshForm::Fallback,
        gpu_desc: opts.descriptors == Descriptors::Gpu,
        probe: opts.probe,
        loop_counts: opts.loop_counts,
        pagetable: opts.pagetable,
        barycentrics: opts.barycentrics,
    });
    NONUNIFORM_HEAP.with(|f| f.set(false));
    HEAP_UNIFORMITY.with(|f| f.set((false, false)));
    let l = l?;
    Ok(HostShader {
        stage: native.stage,
        dxil: l.dxil,
        user_data: l.user_data,
        static_samplers: l
            .samplers
            .iter()
            .enumerate()
            .map(|(n, &w)| {
                let mut s = StaticSampler::from_words(w);
                if l.cmp_samplers >> n & 1 != 0 {
                    s.make_comparison();
                }
                s
            })
            .collect(),
        samplers: l.samplers,
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        half_views: l.half_views,
        address_map: l.address_map,
        descriptor_words: l.descriptor_words,
    })
}

/// Interpolation modes of the input (for pixel) or output (for vertex/mesh)
/// signature elements carried in a container's `PSV0` part: `(row, col, kind,
/// interp)`.
fn psv_interp(part: &[u8], pixel: bool) -> Option<Vec<(Option<u32>, u8, u8, u8)>> {
    use crate::dxbc::u32_at;
    let info_sz = u32_at(part, 0).ok()? as usize;
    if info_sz < 32 {
        return None;
    }
    let info = part.get(4..4 + info_sz)?;
    let (n_in, n_out, n_prim) = (info[28] as usize, info[29] as usize, info[30] as usize);
    let mut pos = 4 + info_sz;
    let n_bind = u32_at(part, pos).ok()? as usize;
    pos += 4;
    if n_bind > 0 {
        let bsz = u32_at(part, pos).ok()? as usize;
        pos = pos.checked_add(4)?.checked_add(n_bind.checked_mul(bsz)?)?;
    }
    let str_sz = u32_at(part, pos).ok()? as usize;
    pos = pos.checked_add(4)?.checked_add(str_sz)?;
    let idx_cnt = u32_at(part, pos).ok()? as usize;
    pos = pos.checked_add(4)?.checked_add(idx_cnt.checked_mul(4)?)?;
    if n_in + n_out + n_prim == 0 {
        return Some(Vec::new());
    }
    let esz = u32_at(part, pos).ok()? as usize;
    pos += 4;
    if esz < 14 {
        return None;
    }
    let (start_i, count) = if pixel { (0, n_in) } else { (n_in, n_out) };
    let mut out = Vec::with_capacity(count);
    for i in start_i..start_i + count {
        let off = pos.checked_add(i.checked_mul(esz)?)?;
        let e = part.get(off..off + esz)?;
        let alloc = e[10] & 0x40 != 0;
        let row = alloc.then_some(u32::from(e[9]));
        let col = if alloc { (e[10] >> 4) & 3 } else { 0 };
        out.push((row, col, e[11], e[13]));
    }
    Some(out)
}

/// The rows of an `ISG1`/`OSG1` part.
fn signature(part: &[u8]) -> Result<Vec<container::SigElement>, ShaderError> {
    use crate::dxbc::u32_at;
    let count = u32_at(part, 0)? as usize;
    let base = u32_at(part, 4)? as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = base + 32 * i;
        let name_at = u32_at(part, at + 4)? as usize;
        let name = part
            .get(name_at..)
            .and_then(|s| s.split(|&b| b == 0).next())
            .ok_or(ShaderError::Malformed("signature name out of bounds"))?;
        let mask = *part
            .get(at + 24)
            .ok_or(ShaderError::Malformed("truncated"))?;
        if u32_at(part, at + 28)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a minimum-precision signature element",
            });
        }
        if u32_at(part, at)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a signature element of another stream",
            });
        }
        let kind = match u32_at(part, at + 12)? {
            0 => 0,
            1 => 3,
            2 => 6,
            3 => 7,
            4 => 4,
            5 => 5,
            6 => 1,
            7 => 10,
            8 => 2,
            9 => 13,
            10 => 12,
            64 => 16,
            65 => 17,
            66 => 14,
            67 => 19,
            68 => 18,
            69 => 20,
            70 => 15,
            23 => 28, // barycentrics
            25 => 30, // cull primitive
            _ => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: "a system value in the signature",
                });
            }
        };
        // 16-bit elements are declared 32-bit; reading or writing one
        // converts (16-bit values travel packed on the console).
        let raw = u32_at(part, at + 16)?;
        let comp = match (kind, raw) {
            // DXIL declares the front face a uint (the validator insists).
            (13, _) => 5,
            (_, 1 | 4) => 5,
            (_, 2 | 5) => 4,
            (_, 3 | 6) => 9,
            (_, c) => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: match c {
                        0 => "a signature element of unknown type",
                        _ => "a 64-bit signature element",
                    },
                });
            }
        };
        let register = u32_at(part, at + 20)?;
        out.push(container::SigElement {
            name: String::from_utf8_lossy(name).into_owned(),
            index: u32_at(part, at + 8)?,
            kind,
            comp,
            interp: 0,
            row: (register != u32::MAX).then_some(register),
            col: mask.trailing_zeros().min(3) as u8,
            cols: mask.count_ones().max(1) as u8,
            usage: 0,
            half: (4..=6).contains(&raw),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::bitcode::Attr;

    #[test]
    fn shader_opt_switch() {
        use super::bitcode::opt::{DEAD, FOLD, LOADS, MODULE};
        use super::parse_shader_opt as p;
        assert_eq!(p(None), 0);
        assert_eq!(p(Some("0")), 0);
        assert_eq!(p(Some("none")), 0);
        assert_eq!(p(Some("all")), MODULE);
        assert_eq!(p(Some(" 1 ")), DEAD);
        assert_eq!(p(Some("0x12")), FOLD | LOADS);
        assert_eq!(p(Some("dead,fold+loads")), DEAD | FOLD | LOADS);
        assert_eq!(p(Some("dead,bogus")), 0);
        assert_eq!(p(Some("0xffffffff")), MODULE);
    }
    use super::lower::LOOP_LIMIT;
    use super::parse_loop_limit;

    #[test]
    fn the_loop_limit_switch_parses_or_falls_back() {
        assert_eq!(parse_loop_limit(None, LOOP_LIMIT), LOOP_LIMIT);
        assert_eq!(parse_loop_limit(Some("4096"), 7), 4096);
        assert_eq!(parse_loop_limit(Some(" 0x1000 "), 7), 4096);
        assert_eq!(parse_loop_limit(Some("1"), 7), 1);
        for bad in ["", "0", "-5", "lots", "0x", "4294967296"] {
            assert_eq!(parse_loop_limit(Some(bad), LOOP_LIMIT), LOOP_LIMIT, "{bad:?}");
        }
    }
    use super::container::SigElement;
    use super::shader::Shader;
    use super::*;
    use crate::native::{ComputeIo, StageIo};

    /// A pixel shader writing SV_StencilRef (channel y of `exp mrtz`).
    /// DXC sets ShaderFlags.StencilRef (bit 11) for a pixel shader whose
    /// output signature has SV_StencilRef, and the feature bit
    /// D3D_SHADER_FEATURE_STENCIL_REF (0x200) in SFI0; the validator checks
    /// both against the module.
    #[test]
    fn a_stencil_export_declares_the_stencil_ref_feature() {
        let words = [
            0x7E00_02F2u32, // v_mov_b32 v0, 1.0
            0x7E02_0285,    // v_mov_b32 v1, 5
            0xF800_1883,
            0x0000_0100, // exp mrtz v0, v1, off, off done vm
            0xBF81_0000,
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let el = |name: &str, kind, comp| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp,
            interp: 0,
            row: None,
            col: 0,
            cols: 1,
            usage: 1,
            half: false,
        };
        let osg1 =
            container::signature(&[el("SV_Depth", 17, 9), el("SV_StencilRef", 20, 5)], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let out = emit(&c, &native, &p, &st, &io, &res).unwrap();
        // dxil.dll, where there is one, checks the flags against the module.
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut out.dxil.clone()).unwrap();
        }
        let host = Container::parse(&out.dxil).unwrap();
        let sfi0 = host.part(FourCC::SFI0).unwrap().data;
        let features = u64::from_le_bytes(sfi0.try_into().unwrap());
        assert_ne!(features & 0x200, 0, "SFI0 {features:#x}: no StencilRef");
    }

    /// A wave64 compute program that writes only EXEC's high dword
    /// (`s_mov_b32 exec_hi, 0`) before a vector instruction. The new mask was
    /// opaque and keyed by the unchanged low dword, so building its lane bit
    /// needed the mask's own value: lowering recursed until the stack
    /// overflowed, which takes the whole process down (found by
    /// random-input testing). Now that low dword, seen with two high
    /// dwords, has no mask, and the instruction fails like any other that
    /// cannot be expressed: its result is missing, and only a side effect
    /// that uses it would stop the translation.
    #[test]
    fn an_exec_high_write_does_not_overflow_the_stack() {
        // s_mov_b32 exec_hi, 0; v_mov_b32 v1, v0; s_endpgm
        let words = [0xBEFF_0380u32, 0x7E02_0300, 0xBF81_0000];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        if let Err(e) = emit(&c, &native, &p, &st, &io, &res) {
            assert!(e.to_string().contains("its own value"), "{e}");
        }
    }

    /// A wave64 compute program: `body`, then a store of v1 to a UAV.
    fn stores_v1(body: &[u32]) -> Result<(), ShaderError> {
        use crate::native::RootParam;
        use crate::{ResourceType, RootType, UsedResource};
        // s_mov_b32 s1, 0; s_load_dwordx4 s[8:11], s[0:1], 0
        let mut words = vec![0xBE81_0380, 0xF408_0200, 0xFA00_0000];
        words.extend_from_slice(body);
        // v_lshlrev_b32 v3, 2, v0; buffer_store_dword v1, v3, s[8:11], 0 offen;
        // s_endpgm
        words.extend_from_slice(&[0x3406_0082, 0xE070_1000, 0x8002_0103, 0xBF81_0000]);
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 4,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 2,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        emit(&c, &native, &p, &st, &io, &res).map(|_| ())
    }

    /// The high dword of a wave64 mask rewritten alone (`s_mov_b32 vcc_hi`
    /// or `exec_hi`): lanes 32-63 took the mask the low dword was first
    /// written with (the compare's, EXEC's on entry). Now the store refuses;
    /// without the write it translates.
    #[test]
    fn a_mask_whose_high_dword_is_rewritten_is_refused() {
        // v_cmp_eq_u32 vcc, 0, v0; v_mov_b32 v2, 11; v_cndmask_b32 v1, 0, v2, vcc
        let select = [0x7D84_0080, 0x7E04_028B, 0x0202_0480];
        // The same with v_add_co_ci_u32 v1, vcc, 0, v2, vcc.
        let carry = [0x7D84_0080, 0x7E04_028B, 0x5002_0480];
        // v_mov_b32 v1, 7
        let mov = [0x7E02_0287];
        for body in [&select[..], &carry, &mov] {
            stores_v1(body).unwrap();
        }
        // s_mov_b32 vcc_hi, 0 after the compare.
        let select_hi = [0x7D84_0080, 0xBEEB_0380, 0x7E04_028B, 0x0202_0480];
        let carry_hi = [0x7D84_0080, 0xBEEB_0380, 0x7E04_028B, 0x5002_0480];
        // s_mov_b32 exec_hi, 0; v_mov_b32 v1, 7, then the same after
        // s_mov_b64 s[2:3], exec.
        let exec_hi = [0xBEFF_0380, 0x7E02_0287];
        let exec_hi_saved = [0xBE82_047E, 0xBEFF_0380, 0x7E02_0287];
        for body in [&select_hi[..], &carry_hi, &exec_hi, &exec_hi_saved] {
            assert!(stores_v1(body).is_err(), "{body:x?}");
        }
    }

    /// A carry-in reading a join of compares after a divergent branch. A
    /// lane that skipped the inner compare has VCC clear there (VOPC clears
    /// inactive lanes), not the first compare's bit, so the join is no
    /// per-lane phi: lowering refuses it.
    #[test]
    fn a_carry_in_from_a_divergent_join_is_refused() {
        let body = [
            0x7E04_028B, // v_mov_b32 v2, 11
            0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
            0xBE82_376A, // s_andn1_saveexec_b64 s[2:3], vcc
            0xBF88_0001, // s_cbranch_execz 1
            0x7D84_0081, // v_cmp_eq_u32 vcc, 1, v0
            0x88FE_027E, // s_or_b64 exec, exec, s[2:3]
            0x5002_0480, // v_add_co_ci_u32 v1, vcc, 0, v2, vcc
        ];
        let e = stores_v1(&body).unwrap_err().to_string();
        // With VIctory the untracked mask is read as lane bits, and
        // this join has no data form: still refused, under that reason.
        assert!(e.contains("a carry-in from an untracked mask") || e.contains("a lane mask read as data"), "{e}");
    }

    /// A pixel shader writing a constant colour.
    #[cfg(windows)]
    fn constant_color() -> Vec<u8> {
        let mut s = Shader::new(true);
        s.outputs.push(SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        });
        let b = s.entry;
        for (c, v) in [1.0f32, 0.5, 0.25, 1.0].into_iter().enumerate() {
            let args = [
                s.m.i32(0),
                s.m.i32(0),
                s.m.i8(c as u8),
                s.m.f32(v.to_bits()),
            ];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        s.m.terminate(b, super::bitcode::Term::Ret);
        s.finish()
    }

    #[test]
    #[cfg(windows)]
    fn stub_pipeline_is_created() {
        let Some(host) = super::host::Host::for_tests() else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        let ps = super::HostShader {
            stage: crate::HwStage::Ps,
            dxil: constant_color(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            half_views: Vec::new(),
            address_map: false,
            descriptor_words: false,
        };
        let file = |var: &str, stage| {
            std::env::var(var).ok().map(|p| super::HostShader {
                stage,
                dxil: std::fs::read(p).unwrap(),
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                half_views: Vec::new(),
                address_map: false,
                descriptor_words: false,
            })
        };
        let ps = file("XNW_REF_PS", crate::HwStage::Ps).unwrap_or(ps);
        let vs = file("XNW_REF_VS", crate::HwStage::Vs)
            .unwrap_or_else(|| super::host::stub_vs(&ps).unwrap());
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut vs.dxil.clone()).unwrap();
        }
        host.pipeline(&vs, &ps).unwrap();
        // The runtime checks the container hash: no hash or a wrong one is
        // refused. (Recent runtimes also take sixteen 0x01 bytes, a bypass
        // value; ours works with older ones too.)
        let mut wrong = ps.dxil[4..20].to_vec();
        wrong[0] ^= 1;
        for digest in [&[0u8; 16][..], &wrong] {
            let mut other = ps.clone();
            other.dxil[4..20].copy_from_slice(digest);
            assert!(
                host.pipeline(&vs, &other).is_err(),
                "digest {digest:02x?} accepted"
            );
        }
    }

    #[test]
    #[cfg(windows)]
    fn constant_color_validates() {
        let mut c = constant_color();
        let path = super::validator::Validator::default_path();
        let Ok(mut v) = super::validator::Validator::load(path.as_deref()) else {
            eprintln!("no dxil.dll; skipped");
            return;
        };
        let ours = c[4..20].to_vec();
        v.validate(&mut c).unwrap();
        assert_eq!(c[4..20], ours[..], "our hash differs from the validator's");
    }

    /// A pixel shader interpolating at sample positions (PERSP_SAMPLE
    /// barycentrics, `SPI_PS_INPUT_ADDR` bit 0) runs at sample frequency:
    /// DXC's PSV0 says so in PSInfo.SampleFrequency (runtime info byte 1)
    /// whenever an input is interpolated per sample.
    #[test]
    fn sample_interpolated_inputs_set_psv_sample_frequency() {
        let words = [
            0xC808_0000u32, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,    // v_interp_p2_f32 v2, v1, attr0.x
            0xF800_180F,
            0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
            0xBF81_0000,
        ];
        let psv_of = |input_addr: u32| {
            let native = super::NativeProgram {
                stage: super::HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: crate::native::StageIo::Ps {
                    inputs: vec![crate::native::PsInput {
                        usage: crate::native::Usage::Generic,
                        row: 0,
                        mask: 0b1111,
                        attr: 0,
                        flat: false,
                        fp16: false,
                        pack2: false,
                        row2: 0,
                    }],
                    input_addr,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let el = |name: &str, kind, usage| SigElement {
                name: name.into(),
                index: 0,
                kind,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage,
                half: false,
            };
            let isg1 = super::container::signature(&[el("TEXCOORD", 0, 0)], true);
            let osg1 = super::container::signature(&[el("SV_Target", 16, 15)], false);
            let bytes =
                crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
            let c = super::Container::parse(&bytes).unwrap();
            let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let res = crate::resource::track(&p, &native.layout, native.stage).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let out = super::emit(&c, &native, &p, &st, &io, &res).unwrap();
            // dxil.dll, where there is one, checks PSV0 against the module.
            #[cfg(windows)]
            if let Ok(mut v) = super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut out.dxil.clone()).unwrap();
            }
            let host = super::Container::parse(&out.dxil).unwrap();
            host.part(super::FourCC::PSV0).unwrap().data.to_vec()
        };
        // Runtime info starts after its size; PSInfo is {DepthOutput, SampleFrequency}.
        let center = psv_of(1 << 1);
        assert_eq!(center[4 + 1], 0, "center interpolation is per pixel");
        let sample = psv_of(1 << 0);
        assert_eq!(sample[4 + 1], 1, "PSInfo.SampleFrequency");
    }

    /// A pixel shader reading channel z of an input whose element starts at
    /// w. Looking the channel up subtracted the element's first channel
    /// before checking the channel lies in it, so in a build with overflow
    /// checks (tests, debug) any read below an element that starts at z or
    /// w panicked, whichever row it was in (found by random-input testing).
    #[test]
    fn reading_a_channel_below_an_input_element_does_not_underflow() {
        // v_interp_mov_f32 v3, p0, attr0.z; s_endpgm
        let words = [0xC80E_0202u32, 0xBF81_0000];
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs: vec![crate::native::PsInput {
                    usage: crate::native::Usage::Generic,
                    row: 0,
                    mask: 0b1000,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let w_only = SigElement {
            name: "TEXCOORD".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 3,
            cols: 1,
            usage: 0,
            half: false,
        };
        let isg1 = super::container::signature(&[w_only], true);
        let bytes = crate::dxbc::build(&[(super::FourCC::ISG1, &isg1)]);
        let c = super::Container::parse(&bytes).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        // Translated or refused with a reason; it must not panic.
        let _ = super::emit(&c, &native, &p, &st, &io, &res);
    }

    #[test]
    fn valu_div_f32_sequence_executes_on_gpu() {
        let words: &[u32] = &[
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
            0xF408_0302,
            0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
            0x340E_0083, // v_lshlrev_b32 v7, 3, v0
            0xE034_1000,
            0x8002_0107, // buffer_load_dwordx2 v[1:2], v7, s[8:11], 0 offen
            0xD56D_6A03,
            0x0406_0502, // v_div_scale_f32 v3, vcc_lo, v2, v2, v1
            0xD56D_1008,
            0x0406_0501, // v_div_scale_f32 v8, s16, v1, v2, v1
            0x7E08_5503, // v_rcp_f32 v4, v3
            0xD54B_0005,
            0x23CA_0903, // v_fma_f32 v5, -v3, v4, 1.0
            0xD54B_0004,
            0x0412_0905, // v_fma_f32 v4, v5, v4, v4
            0x100A_0908, // v_mul_f32 v5, v8, v4
            0xD54B_0006,
            0x2422_0B03, // v_fma_f32 v6, -v3, v5, v8
            0xD56F_0004,
            0x0416_0906, // v_div_fmas_f32 v4, v6, v4, v5
            0xD55F_0001,
            0x0406_0504, // v_div_fixup_f32 v1, v4, v2, v1
            0x340E_0082, // v_lshlrev_b32 v7, 2, v0
            0xE070_1000,
            0x8003_0107, // buffer_store_dword v1, v7, s[12:15], 0 offen
            0xBF81_0000, // s_endpgm
        ];
        let native = super::NativeProgram {
            stage: super::HwStage::Cs,
            wave64: false,
            vgprs: 9,
            layout: vec![
                crate::RootParam {
                    root: crate::RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs: 4,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                crate::RootParam {
                    root: crate::RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 4,
                    slot: 1,
                },
            ],
            used: vec![
                crate::UsedResource {
                    kind: crate::ResourceType::Srv,
                    root: crate::RootType::Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 4,
                    dwords: 4,
                    user_data: 4,
                    slot: 1,
                },
                crate::UsedResource {
                    kind: crate::ResourceType::Uav,
                    root: crate::RootType::Table,
                    table_offset_dwords: 8,
                    lower_bound: 0,
                    stride: 4,
                    dwords: 4,
                    user_data: 4,
                    slot: 1,
                },
            ],
            io: crate::native::StageIo::Cs(crate::native::ComputeIo {
                rsrc2: 5 << 1,
                threads: [7, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let l = super::lower::lower(&super::lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            fallback: false,
            fma: super::Fma::Exact,
            gpu_desc: false,
            probe: false,
            loop_counts: None,
            pagetable: false,
            barycentrics: false,
        })
        .unwrap();
        let mut dxil = l.dxil;
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut dxil).unwrap();
        }
        #[cfg(windows)]
        if let Some(host) = super::host::Host::for_tests() {
            let cs = super::HostShader {
                stage: super::HwStage::Cs,
                dxil,
                user_data: l.user_data,
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: l.root_cbvs,
                root_srvs: l.root_srvs,
                root_uavs: l.root_uavs,
                half_views: l.half_views,
                address_map: l.address_map,
                descriptor_words: l.descriptor_words,
            };
            let (root, pso) = host.compute_pipeline(&cs).unwrap();
            let inputs: Vec<u32> = [
                7.0f32, 3.0, -1.0, 1e-40, 1.0, 1e35, 1e-35, 1e-36, 0.0, 0.0, -1.0, 0.0, -1e35, 1e35,
            ]
            .into_iter()
            .map(f32::to_bits)
            .collect();
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &super::host::Dispatch {
                        roots: super::host::Host::roots(&cs, vec![0, 0, 0, 0, 0], 2, 0),
                        views: &[
                            super::host::View::Buffer {
                                data: &inputs,
                                stride: super::host::Stride::Raw,
                            },
                            super::host::View::RawUav(64),
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            let got: Vec<f32> = out[0][..7].iter().copied().map(f32::from_bits).collect();
            assert!((got[0] - (7.0f32 / 3.0)).abs() < 1e-6, "got {:?}", got);
            assert_eq!(out[0][1], (-1.0f32 / 1e-40).to_bits());
            assert!(
                (got[2] - 1e-35).abs() <= 1e-40,
                "lane 2 got {:?} ({:#x})",
                got[2],
                out[0][2]
            );
            assert!(
                (got[3] - 10.0).abs() < 1e-5,
                "lane 3 got {:?} ({:#x})",
                got[3],
                out[0][3]
            );
            assert_eq!(out[0][4], 0xFFC0_0000);
            assert_eq!(out[0][5], (-1.0f32 / 0.0).to_bits());
            assert!(
                (got[6] - (-1.0)).abs() < 1e-6,
                "lane 6 got {:?} ({:#x})",
                got[6],
                out[0][6]
            );
        }
    }
}
