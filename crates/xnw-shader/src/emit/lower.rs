// Ported from VIctory crates/victory-shader/src/emit/lower.rs, commit d2a4bae1f49533ef5cb512fd1b1ba1513f2e1989, GPL-3.0-or-later.
// With VIctory open pull requests (GPL-3.0-or-later) where they touch this file: (8ae232f, d112889), (4c9a922, db20637, 3e8f2d3), (815178d), (fa5189d).
// With VIctory (head 42538cd, on upstream 4322b61; GPL-3.0-or-later): unknown-stride structured buffers read through the runtime stride table (shader side).
// With VIctory (499bd98, GPL-3.0-or-later): multi-term, scaled, masked and phi-joined addresses in resource tracking.
// With VIctory (afbe339, f800f1c; GPL-3.0-or-later): untracked masks in selects and carry-ins via lane bits; descriptor-only GetDimensions typed by metadata.
// With VIctory (c78056c, 5ff639e; GPL-3.0-or-later): quad gradients for compute samples, twin descriptor loads, descriptor dwords at joins.
// With VIctory (35bf153, GPL-3.0-or-later): second rows of packed 16-bit pixel inputs.
// With VIctory (permadiego/VIctory shader/mask-dword-joins 7ab94ab) and its commit 1661a3e, GPL-3.0-or-later: mask dwords at data joins; compare-and-swap image atomics.
// With VIctory-Belobug-AMD (fork of VIctory at 8e79c80; read at 4cbe9d7), GPL-3.0-or-later: pixel quads read through quad reads, helper lanes' v_readfirstlane (03629f3, f520772; upstream 0601955).
//! IR to DXIL, one machine instruction at a time (decision 0003).
//!
//! Every register dword is an `i32` (SCC an `i1`); float operations bitcast
//! their operands. Lane masks exist only as the per-lane booleans of
//! structurisation; branches and guards are built from them. A value the
//! emitter cannot express is `Missing` with a reason, and only fails the
//! program when something with an effect (an export, a branch) needs it.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::UserData;
use super::bitcode::{Attr, Bin, Cast, Term, V, pred};
use super::container::SigElement;
use super::shader::{Shader, flags};
use crate::ShaderError;
use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::io::{Barycentric, Export, Input, Interp, Io, Target};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, Value};
use crate::native::{HwStage, NativeProgram, ResourceType, RootType, StageIo, Usage, UsedResource};
use crate::resource::{Address, CODE, RelOffset, Resources, Role, Site, Source};
use crate::structure::{Atom, Bool, LaneExit, Structured};

type Why = &'static str;

// VIctory (1a2a3a1, GPL-3.0-or-later): title memory at computed
// addresses and the console's ray intersection (XNW).
mod bvh;
mod global;

/// The high dword `s_getpc_b64` gives in translated code (XNW): above
/// every 40-bit title address, so a computed address with it reads the
/// program's own code ([`Lower::code_dword`]).
const CODE_TAG: u32 = 0xC0DE_0000;

#[derive(Debug, Clone, Copy)]
enum Val {
    I32(V),
    Bit(V),
    Missing(Why),
}

pub(crate) struct Lowered {
    pub dxil: Vec<u8>,
    /// Bit n: immediate sampler n is used for comparisons.
    pub cmp_samplers: u64,
    pub root_cbvs: Vec<u8>,
    pub root_srvs: Vec<u8>,
    pub root_uavs: Vec<u8>,
    /// Reads title memory through the address map.
    pub address_map: bool,
    /// Reads descriptor words.
    pub descriptor_words: bool,
    pub user_data: Vec<UserData>,
    pub samplers: Vec<[u32; 4]>,
    /// The names of the emitted functions (which dx.ops ran); tests only.
    #[cfg(test)]
    pub op_names: Vec<String>,
    /// The emitted instructions, one line each; tests only.
    #[cfg(test)]
    pub insts: Vec<String>,
}

pub(crate) struct Parts<'a> {
    pub native: &'a NativeProgram,
    pub p: &'a Program,
    pub st: &'a Structured,
    pub io: &'a Io,
    pub res: &'a Resources,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    pub prims: Vec<SigElement>,
    /// Mesh: what the group id arrives multiplied by.
    pub mesh_stride: u32,
    /// How fused f32 multiply-adds are written (XNW addition).
    pub fma: super::Fma,
    /// Mesh: emit as compute writing records (XNW: `mesh_lower`).
    pub fallback: bool,
    /// Descriptors are read from the title's records (XNW).
    pub gpu_desc: bool,
    /// XNW: [`super::Options::probe`].
    pub probe: bool,
    /// XNW: [`super::Options::loop_counts`].
    pub loop_counts: Option<u32>,
    /// XNW: [`super::Options::pagetable`].
    pub pagetable: bool,
}

/// XNW: the user-data root type the runtime answers with the
/// heap index of its loop-count buffer (xnw-gpu `loopcount::ROOT`, "LC").
pub(crate) const LOOP_COUNT_ROOT: u32 = 0x4C43;
/// Words per program slot in that buffer (`loopcount::LOOPS`).
const LOOP_COUNT_LOOPS: u32 = 8;

enum Stop {
    /// Phis whose incoming values have no DXIL form were read: redo with
    /// them missing.
    Retry(Vec<(Value, Why)>),
    Fail(ShaderError),
}

pub(crate) fn lower(parts: &Parts) -> Result<Lowered, ShaderError> {
    let mut bad = HashMap::new();
    loop {
        let t = std::time::Instant::now();
        let r = Lower::new(parts, &bad).run();
        super::lap("lower", t);
        match r {
            Ok(l) => return Ok(l),
            Err(Stop::Retry(phis)) => {
                for (v, why) in phis {
                    if bad.insert(v, why).is_some() {
                        return Err(ShaderError::Unsupported {
                            offset: 0,
                            what: "a phi that does not settle",
                        });
                    }
                }
            }
            Err(Stop::Fail(e)) => return Err(e),
        }
    }
}

// XNW: the mesh program as compute writing records.
#[path = "mesh_lower.rs"]
mod mesh_lower;

const TWO_PI: f32 = std::f32::consts::TAU;
/// `s_sendmsg` id of a mesh or NGG stage's output allocation request.
const GS_ALLOC_REQ: u16 = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Numeric {
    Unorm,
    Snorm,
    Uscaled,
    Sscaled,
    Uint,
    Sint,
    Float,
}

/// A GFX10 buffer format (the unified format of typed buffer loads): the
/// channel widths from the lowest bits up, and how the channels convert.
/// Formats are numbered in families (`8`, `16`, `8_8`, ...: AMD names
/// list channels from the highest bits), each with its numeric formats in
/// the order unorm, snorm, uscaled, sscaled, uint, sint, float.
fn buffer_format(format: u8) -> Option<(&'static [u32], Numeric)> {
    use Numeric::*;
    const NORM: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint];
    const ALL: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Float];
    const WIDE: &[Numeric] = &[Uint, Sint, Float];
    // Numbers and order from LLVM's `UfmtGFX10` (`SIDefines.h`); the GFX10
    // buffer formats have no 5_9_9_9. Channel order from Mesa
    // (`ac_formats.c`): R11G11B10_FLOAT is 10_11_11, and 10_10_10_2 has
    // its 2-bit channel in the lowest bits.
    const FAMILIES: [(&[u32], &[Numeric]); 14] = [
        (&[8], NORM),
        (&[16], ALL),
        (&[8, 8], NORM),
        (&[32], WIDE),
        (&[16, 16], ALL),
        (&[11, 11, 10], ALL),     // 10_11_11
        (&[10, 11, 11], ALL),     // 11_11_10
        (&[2, 10, 10, 10], NORM), // 10_10_10_2
        (&[10, 10, 10, 2], NORM), // 2_10_10_10
        (&[8, 8, 8, 8], NORM),
        (&[32, 32], WIDE),
        (&[16, 16, 16, 16], ALL),
        (&[32, 32, 32], WIDE),
        (&[32, 32, 32, 32], WIDE),
    ];
    let mut n = usize::from(format.checked_sub(1)?);
    for (bits, numerics) in FAMILIES {
        if n < numerics.len() {
            return Some((bits, numerics[n]));
        }
        n -= numerics.len();
    }
    None
}

/// How far a float channel of `width` bits (16, or the unsigned 11- and
/// 10-bit floats of the packed formats) shifts up to be a half:
/// the small floats are a half's exponent (5 bits, biased 15) and the top
/// of its mantissa, without the sign, so `legacyF16ToF32` widens them
/// exactly, denormals, infinities and NaNs included.
fn half_shift(width: u32) -> u32 {
    debug_assert!(matches!(width, 10 | 11 | 16), "{width}-bit float");
    if width == 16 { 0 } else { 15 - width }
}

struct Lower<'a> {
    s: Shader,
    x: &'a Parts<'a>,
    bad: &'a HashMap<Value, Why>,
    lane_fail: HashMap<Value, Why>,
    vals: HashMap<Value, Val>,
    floats: HashMap<Value, V>,
    /// DXIL block per IR block (unreachable: none).
    block: Vec<Option<usize>>,
    cur: usize,
    bools: HashMap<(usize, Bool), V>,
    lane_bits: HashMap<Value, V>,
    /// IR phi location by value.
    phi_at: HashMap<Value, (usize, usize)>,
    mask_phis: HashMap<Value, V>,
    pending_masks: Vec<(V, usize, usize)>,
    data_phis: Vec<(V, usize, usize)>,
    read_phis: HashSet<Value>,
    /// Relative-offset DXIL phis for address phis, keyed by `(phi_value, base_value)`.
    addr_phis: HashMap<(Value, Value), V>,
    /// Address phis waiting for incoming values in `fill_phis`: `(phi, ir_block, phi_idx, base)`.
    pending_addr_phis: Vec<(V, usize, usize, Value)>,
    /// Defining instruction `(block, inst_index)` for each non-phi, non-input IR value.
    def_site: HashMap<Value, Site>,
    /// XNW: per scalar load, the first resource use of its dwords and the
    /// first such use by an image or buffer instruction (`descriptor_data`;
    /// built once instead of a walk of every descriptor per load).
    load_uses: HashMap<Site, (Option<(Site, u8)>, Option<(Site, u8)>)>,
    /// Whether the program walks a BVH ([`Self::walks_bvh`], once).
    bvh_walk: bool,
    params: HashMap<Value, (usize, u32)>,
    user_data: Option<V>,
    user_reads: BTreeMap<u32, UserData>,
    /// GPU-side descriptors: per table or heap parameter, the user-data
    /// index of its window view and byte offset (XNW).
    windows: HashMap<usize, u32>,
    /// The buffer instruction being lowered (its record names the stride
    /// table slot with GPU-side descriptors).
    buffer_site: Option<Site>,
    samplers: Vec<[u32; 4]>,
    descriptors: HashMap<(Site, bool), &'a Source>,
    scalar: HashMap<Site, Address>,
    exports: HashMap<Site, &'a Export>,
    interps: HashMap<Site, &'a Interp>,
    handles: HashMap<(usize, Site, u8, bool, u32), V>,
    /// Runtime stride table entries read in a block, by the descriptor's
    /// load: the raw slot, the stride and the bias.
    entries: HashMap<(usize, Site, u8), [V; 3]>,
    /// Blocks splitting the edge from one IR block to another.
    edges: HashMap<(usize, usize), usize>,
    /// Per IR block: a `done` export is behind every lane entering it.
    exported: Vec<bool>,
    /// Reverse-postorder index per IR block.
    rpo: Vec<usize>,
    /// DXIL block each IR block ends in (guarded stores split blocks).
    tail: Vec<Option<usize>>,
    /// The program writes UAVs: barriers fence them too.
    writes_uavs: bool,
    /// src0 of the current instruction as DPP fetched it.
    src0: Option<V>,
    /// Compute code that reads other lanes: thread ids follow the host
    /// lanes, so lane n is console lane n.
    console_lanes: bool,
    /// Mesh: the thread index in the group (made in the entry block).
    flat: Option<V>,
    /// Mesh fallback: the first console group of this dispatch, read
    /// from `t0` in the record space (a split direct dispatch runs its
    /// groups in pieces).
    group_base: Option<V>,
    /// Mesh: IR blocks whose `s_barrier` is left out.
    no_barrier: HashSet<usize>,
    /// Vertex and mesh outputs holding 16-bit values: row, whether two per
    /// dword, and the row packed after it.
    half_outputs: HashMap<u8, (bool, Option<u8>)>,
    /// How lane masks become data.
    mask_data: MaskData,
    /// The instruction being lowered may read lane masks as data.
    mask_ok: bool,
    /// Ballots of masks read as data, by low dword: (low, high).
    ballots: HashMap<Value, (V, V)>,
    /// Masks whose ballot is being built: a mask whose lane bit needs its
    /// own value as a number (an opaque mask keyed by a dword it shares
    /// with another) would otherwise recurse until the stack overflows.
    balloting: HashSet<Value>,
    /// Block writing each value.
    def_block: HashMap<Value, usize>,
    /// IR block being lowered.
    cur_ir: usize,
    /// Whole-wave sections by block and start index.
    sections: HashMap<usize, HashMap<usize, super::wave::Section>>,
    /// Values written in a section that code outside it reads.
    section_out: HashSet<Value>,
    /// The program's constant table (the code from the first byte read
    /// through `s_getpc_b64` on): global, first byte, dwords.
    code_table: Option<(V, u32, u32)>,
    /// XNW: the invocation's loop budget (see [`LOOP_LIMIT`]): back
    /// edges taken so far, as a phi at every IR block but the entry, by IR
    /// block, with its incoming (value, DXIL block) pairs.
    counters: HashMap<usize, (V, Vec<(V, usize)>)>,
    /// The budget's value in the block being lowered.
    budget: Option<V>,
    /// XNW: the loop-count buffer's heap index, once read.
    loop_count_index: Option<V>,
    /// XNW: the stride the program declares for the buffer of the
    /// access being lowered (Lower::page_address takes it).
    declared_stride: Option<u32>,
    /// The block a loop stopped by [`LOOP_LIMIT`] leaves the program from.
    abort: Option<usize>,
}

/// Iterations after which a loop is left, unless [`super::loop_limit_for`]
/// says otherwise. XNW: one budget per invocation for all its loops
/// (back edges taken), not per loop and entry: an outer loop's limit times
/// an inner one's bounded nothing. The console runs a loop as long as
/// its data says; data the host has not mirrored yet (a null view reads
/// zeros) can make that four billion iterations, which hangs the host GPU
/// until the driver resets it. A lane that reaches the limit leaves the
/// program as killed (a pixel discards).
// shortcut: a genuine loop of more iterations than this is cut short too.
// XNW: 2^16, not the base's 2^20. In the first mission world four
// mesh-fallback pixel shaders (with programs 248b38ff, 62df4a93, 0a86963f,
// 841b5259) run a loop to the limit: 131 ms per draw of 1 to 7 groups at
// 2^20, 8-12 ms at 2^16; several per submission led the driver to reset
// the device (0x887A0006). five-star-recomp b12a952 bounds at 65,536 too.
// XNW: 256, after victory-fork's stable value (see
// [super::DEFAULT_LOOP_LIMIT]).
pub(super) const LOOP_LIMIT: u32 = 256;

/// Lane masks read as numbers (`s_bcnt1`, `s_ff1`, `v_mbcnt`, scalar
/// compares, bit tests) are ballots of the lanes' booleans, taken where the
/// mask is written: structurisation's `active` masks hold only lanes that
/// reach that block, so the ballot misses none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaskData {
    /// Compute: host lane n is console lane n (the program declares the
    /// console's wave size), so ballots are the console's masks bit for bit
    /// and any use is exact.
    Console,
    /// Pixel and vertex programs: ballots over the host's wave, whatever its
    /// size. Only uses that mean the same on any wave are allowed: counts,
    /// the first lane, the count below a lane, equality (the HLSL wave
    /// operations the console compiler expands). Lane numbers stay host
    /// lanes throughout (`v_mbcnt`, `v_readlane` at a computed lane).
    Host,
    /// Mesh programs: their lanes and waves (LDS per console wave) are the
    /// console's but no host wave size can be declared.
    None,
}

/// What a value holding a lane mask read as a number stops with.
const LANE_MASK: Why = "a lane mask read as data";

/// Instructions that read a lane mask (not EXEC implicitly, not a lane's
/// bit) as a number.
/// Samples whose level comes from derivatives across each quad of lanes.
fn implicit_derivatives(name: &str) -> bool {
    matches!(name, "image_sample" | "image_sample_b")
}

fn reads_mask_as_data(st: &Structured, i: &Inst) -> bool {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let ops: Vec<&Operand> = match &i.fields {
        // A mask operation (its result a mask) is no data use.
        Fields::Sop { .. }
            if i.writes
                .iter()
                .any(|&(r, w)| matches!(r, Reg::S(_)) && is_mask(w)) =>
        {
            return false;
        }
        Fields::Sop { src, .. } => src.iter().collect(),
        Fields::Valu(v)
            if !matches!(i.name(), "v_cndmask_b32") && !i.name().contains("_co_ci_") =>
        {
            v.src.iter().collect()
        }
        _ => return false,
    };
    ops.into_iter().any(|op| match *op {
        Operand::Scalar { code, .. } if code != NULL => i.read(Reg::S(code)).is_some_and(is_mask),
        _ => false,
    })
}

/// Where half `h` (dword h / 2, the high half when odd) of a parameter
/// packing 16-bit rows `a` and `b` two per dword comes from: a row and
/// channel of `elements`, none for padding. Channel c of `a` is half c;
/// `b` starts at the dword after `a`'s last channel.
fn packed_half(elements: &[SigElement], a: u8, b: Option<u8>, h: u8) -> Option<(u8, u8)> {
    let extent = |r: u8| {
        elements
            .iter()
            .filter(|e| e.row == Some(u32::from(r)))
            .map(|e| e.col + e.cols)
            .max()
    };
    let na = extent(a)?;
    let a_halves = na.div_ceil(2) * 2;
    if h < a_halves {
        return (h < na).then_some((a, h));
    }
    let b = b?;
    let c = h - a_halves;
    (c < extent(b)?).then_some((b, c))
}

/// Pixel inputs packing two 16-bit rows per dword whose second row the
/// metadata leaves out (`row2` = `row`): as for outputs
/// ([`half_outputs`]), the next 16-bit row of the signature that no input
/// entry names, taken in row order (rows the metadata does give are taken
/// first). VIctory (35bf153, GPL-3.0-or-later), verbatim.
fn half_input_partners(io: &StageIo, inputs: &[SigElement]) -> HashMap<u8, u8> {
    let StageIo::Ps {
        inputs: entries, ..
    } = io
    else {
        return HashMap::new();
    };
    let packed = || entries.iter().filter(|i| i.fp16 && i.pack2);
    let mut free: Vec<u8> = inputs
        .iter()
        .filter(|e| e.half && e.kind == 0)
        .filter_map(|e| u8::try_from(e.row?).ok())
        .filter(|&r| !entries.iter().any(|i| i.row == r))
        .filter(|&r| !packed().any(|i| i.row2 != i.row && i.row2 == r))
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut missing: Vec<_> = packed().filter(|i| i.row2 == i.row).collect();
    missing.sort_by_key(|i| i.row);
    let mut partners = HashMap::new();
    for i in missing {
        if let Some(k) = free.iter().position(|&r| r > i.row) {
            partners.insert(i.row, free.remove(k));
        }
    }
    partners
}
/// Vertex or mesh outputs of 16-bit values (see [`Lower::half_outputs`]):
/// a packed row's partner is the next 16-bit row of the signature that no
/// output entry names, taken in row order.
fn half_outputs(io: &StageIo, outputs: &[SigElement]) -> HashMap<u8, (bool, Option<u8>)> {
    let entries = match io {
        StageIo::Vs { outputs, .. } => outputs,
        StageIo::Ms(ms) => &ms.outputs,
        _ => return HashMap::new(),
    };
    let per_vertex = || entries.iter().filter(|o| o.param & 0x80 == 0);
    let mut free: Vec<u8> = outputs
        .iter()
        .filter(|e| e.half && e.kind == 0)
        .filter_map(|e| u8::try_from(e.row?).ok())
        .filter(|&r| !per_vertex().any(|o| o.row == r))
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut halves: Vec<_> = per_vertex().filter(|o| o.fp16).collect();
    halves.sort_by_key(|o| o.row);
    halves
        .into_iter()
        .map(|o| {
            let b = o
                .pack2
                .then(|| free.iter().position(|&r| r > o.row))
                .flatten();
            (o.row, (o.pack2, b.map(|i| free.remove(i))))
        })
        .collect()
}

fn unsupported(offset: usize, what: Why) -> Stop {
    Stop::Fail(ShaderError::Unsupported { offset, what })
}

/// The image atomics made by [`Lower::atomic_loop`], in place of a DXIL
/// atomic operation code (VIctory, 1661a3e).
const ATOMIC_LOOP: u32 = u32::MAX - 1;

impl<'a> Lower<'a> {
    fn new(x: &'a Parts<'a>, bad: &'a HashMap<Value, Why>) -> Self {
        let p = x.p;
        let mut phi_at = HashMap::new();
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, phi) in block.phis.iter().enumerate() {
                phi_at.insert(phi.value, (b, i));
            }
        }
        let mut s = Shader::new(x.native.stage == HwStage::Ps);
        s.inputs = x.inputs.clone();
        // XNW: the host-copy bits, written with the page table on.
        if x.pagetable && x.gpu_desc {
            s.host_copies = Some(0);
        }
        s.outputs = x.outputs.clone();
        let insts = || p.blocks.iter().flat_map(|b| &b.insts);
        let writes_uavs = insts().any(|i| {
            matches!(i.fields, Fields::Image { .. } | Fields::Buffer { .. })
                && (i.name().contains("store") || i.name().contains("atomic"))
        });
        // Code reading other lanes needs the console's wave size and lane
        // order; so do lane masks read as numbers, and LDS shared within one
        // wave, where the console compiler leaves out barriers. (Also for
        // mesh programs in the compute fallback.)
        let uses_console_lanes = || {
            insts().any(|i| {
                let n = i.name();
                n.starts_with("ds_swizzle")
                    || n.contains("permute")
                    || n.starts_with("v_mbcnt")
                    || matches!(n, "v_readlane_b32" | "v_writelane_b32")
                    || implicit_derivatives(n)
                    || matches!(&i.fields, Fields::Valu(v) if v.dpp.is_some())
                    || reads_mask_as_data(x.st, i)
            }) || {
                // Masks held as numbers or constant lane sets: the lane
                // tests its own bit by its console lane.
                let whens = x.st.exits.iter().filter_map(|e| match *e {
                    LaneExit::Branch { when, .. } => Some(when),
                    _ => None,
                });
                let mut all: Vec<Bool> =
                    x.st.guards
                        .iter()
                        .flatten()
                        .flatten()
                        .copied()
                        .chain(whens)
                        .collect();
                all.sort_unstable();
                all.dedup();
                all.into_iter().any(|g| {
                    x.st.bools
                        .support(g)
                        .iter()
                        .any(|a| matches!(a, Atom::Opaque(_) | Atom::Constant(_)))
                }) || insts().any(|i| {
                    matches!(
                        i.name(),
                        "v_cndmask_b32"
                            | "v_add_co_ci_u32"
                            | "v_sub_co_ci_u32"
                            | "v_subrev_co_ci_u32"
                    ) && match &i.fields {
                        Fields::Valu(v) => match v.src.get(2) {
                            Some(&Operand::Scalar { code, .. }) => i
                                .read(Reg::S(code))
                                .is_none_or(|m| !x.st.masks.contains_key(&m)),
                            _ => false,
                        },
                        _ => false,
                    }
                })
            }
        };
        let mut console_lanes = false;
        if let StageIo::Cs(cs) = x.native.io {
            s.threads = Some(cs.threads);
            s.lds_dwords = cs.lds_bytes() / 4;
            console_lanes = uses_console_lanes();
            let wave = if x.native.wave64 { 64 } else { 32 };
            let one_wave = cs.threads.iter().product::<u32>() <= wave;
            // XNW: every compute program declares the console's wave
            // size, so the host never runs a wave64 program in 32 lanes;
            // with `XNW_WAVE_SIZE=need`, only where lanes are observable or
            // one wave shares LDS (the rule of VIctory and its fork).
            s.wave_size = super::compute_wave_size(wave, console_lanes, one_wave, s.lds_dwords, super::wave_by_need());
        }
        let mut no_barrier = HashSet::new();
        if let StageIo::Ms(ms) = &x.native.io {
            s.prims = x.prims.clone();
            s.threads = Some([ms.threads, 1, 1]);
            s.lds_dwords = ms.lds_bytes.div_ceil(4);
            s.mesh = Some(super::shader::Mesh {
                max_vertices: ms.max_vertices,
                max_primitives: ms.max_primitives,
                topology: ms.topology,
            });
            if x.fallback {
                // XNW: compute with the console's wave, so lanes and
                // masks are the console's as for compute programs.
                s.fallback = true;
                s.wave_size = Some(if x.native.wave64 { 64 } else { 32 });
                console_lanes = uses_console_lanes();
            }
            // Barriers order the lanes' LDS accesses; the console's
            // allocation request needs them too, but the host does not. One
            // lane sends that request, and what it dominates (the path for
            // counts out of range) runs in that lane alone, so its barrier
            // would hang the host.
            let order = crate::ir::reverse_postorder(&p.blocks);
            let mut index = vec![usize::MAX; p.blocks.len()];
            for (n, &b) in order.iter().enumerate() {
                index[b] = n;
            }
            let idom = crate::ir::idoms(&p.blocks, &order, &index);
            let alloc: Vec<usize> = (0..p.blocks.len())
                .filter(|&b| {
                    p.blocks[b].insts.iter().any(|i| {
                        i.name() == "s_sendmsg"
                            && matches!(i.fields, Fields::Sopp { imm } if imm & 0xF == GS_ALLOC_REQ)
                    })
                })
                .collect();
            for &b in &order {
                let mut d = b;
                let dominated = loop {
                    if alloc.contains(&d) {
                        break true;
                    }
                    if idom[d] == d || idom[d] == usize::MAX {
                        break false;
                    }
                    d = idom[d];
                };
                if ms.lds_bytes == 0 || dominated {
                    no_barrier.insert(b);
                }
            }
        }
        Lower {
            s,
            x,
            bad,
            lane_fail: HashMap::new(),
            vals: HashMap::new(),
            floats: HashMap::new(),
            block: vec![None; p.blocks.len()],
            cur: 0,
            bools: HashMap::new(),
            lane_bits: HashMap::new(),
            phi_at,
            mask_phis: HashMap::new(),
            pending_masks: Vec::new(),
            data_phis: Vec::new(),
            read_phis: HashSet::new(),
            addr_phis: HashMap::new(),
            pending_addr_phis: Vec::new(),
            load_uses: {
                let mut m: HashMap<Site, (Option<(Site, u8)>, Option<(Site, u8)>)> = HashMap::new();
                for d in x.res.descriptors.iter().filter(|d| d.role == Role::Resource) {
                    let ui = &p.blocks[d.site.0].insts[d.site.1];
                    let rich = matches!(ui.fields, Fields::Image { .. } | Fields::Buffer { .. });
                    for l in d.source.leaves() {
                        if let Source::Loaded { load, first } = *l {
                            let e = m.entry(load).or_default();
                            e.0.get_or_insert((d.site, first));
                            if rich {
                                e.1.get_or_insert((d.site, first));
                            }
                        }
                    }
                }
                m
            },
            bvh_walk: p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name().starts_with("image_bvh")),
            def_site: p
                .blocks
                .iter()
                .enumerate()
                .flat_map(|(b, block)| {
                    block
                        .insts
                        .iter()
                        .enumerate()
                        .flat_map(move |(i, inst)| inst.writes.iter().map(move |w| (w.1, (b, i))))
                })
                .collect(),
            params: x.res.params.iter().map(|&(v, p, d)| (v, (p, d))).collect(),
            user_data: None,
            user_reads: BTreeMap::new(),
            windows: HashMap::new(),
            buffer_site: None,
            samplers: Vec::new(),
            descriptors: x
                .res
                .descriptors
                .iter()
                .map(|d| ((d.site, d.role == Role::Sampler), &d.source))
                .collect(),
            scalar: x.res.scalar.iter().copied().collect(),
            exports: x.io.exports.iter().map(|e| (e.site, e)).collect(),
            interps: x.io.interps.iter().map(|(s, i)| (*s, i)).collect(),
            handles: HashMap::new(),
            entries: HashMap::new(),
            edges: HashMap::new(),
            exported: vec![false; p.blocks.len()],
            rpo: vec![usize::MAX; p.blocks.len()],
            tail: vec![None; p.blocks.len()],
            writes_uavs: writes_uavs || x.fallback,
            src0: None,
            console_lanes,
            flat: None,
            group_base: None,
            no_barrier,
            half_outputs: half_outputs(&x.native.io, &x.outputs),
            mask_data: match x.native.stage {
                HwStage::Cs => MaskData::Console,
                HwStage::Ms if x.fallback => MaskData::Console,
                HwStage::Ps | HwStage::Vs => MaskData::Host,
                _ => MaskData::None,
            },
            mask_ok: false,
            ballots: HashMap::new(),
            balloting: HashSet::new(),
            def_block: p
                .blocks
                .iter()
                .enumerate()
                .flat_map(|(b, block)| {
                    block
                        .insts
                        .iter()
                        .flat_map(move |i| i.writes.iter().map(move |w| (w.1, b)))
                        .chain(block.phis.iter().map(move |phi| (phi.value, b)))
                })
                .collect(),
            cur_ir: 0,
            sections: HashMap::new(),
            section_out: HashSet::new(),
            code_table: None,
            counters: HashMap::new(),
            budget: None,
            loop_count_index: None,
            declared_stride: None,
            abort: None,
        }
    }

    /// The dword of the program's code at byte `byte` (a constant table
    /// the compiler appends to the code), as an internal constant array.
    // shortcut: an index outside the table reads its last dword (the
    // console would read whatever memory follows the code).
    fn code_dword(&mut self, byte: V) -> Result<V, Why> {
        let (g, base, len) = match self.code_table {
            Some(t) => t,
            None => {
                let first = self
                    .x
                    .res
                    .scalar
                    .iter()
                    .map(|s| s.1)
                    .chain(self.x.res.descriptors.iter().flat_map(|d| {
                        d.source.leaves().into_iter().filter_map(|l| match l {
                            Source::Built { address, .. } => Some(*address),
                            _ => None,
                        })
                    }))
                    .filter(|a| a.param == CODE)
                    .map(|a| a.offset & !3)
                    .min()
                    // XNW: code read only through `s_getpc_b64` values
                    // (no tracked code address): the whole code.
                    .unwrap_or(0);
                let code = &self.x.native.code;
                let words: Vec<u32> = code
                    .get(first as usize..)
                    .ok_or("a code table past the code")?
                    .chunks(4)
                    .map(|c| {
                        let mut w = [0u8; 4];
                        w[..c.len()].copy_from_slice(c);
                        u32::from_le_bytes(w)
                    })
                    .collect();
                if words.is_empty() {
                    return Err("a code table past the code");
                }
                let elems: Vec<V> = words.iter().map(|&w| self.s.m.i32(w)).collect();
                let arr = self.s.m.array(words.len() as u32, self.s.t.i32);
                let init = self.s.m.aggregate(arr, &elems);
                let g = self.s.m.constant_global("code", init);
                let t = (g, first, words.len() as u32);
                self.code_table = Some(t);
                t
            }
        };
        let c = self.cur;
        let k = self.s.m.i32(base);
        let rel = self.s.m.bin(c, Bin::Sub, byte, k);
        let two = self.s.m.i32(2);
        let index = self.s.m.bin(c, Bin::LShr, rel, two);
        let last = self.s.m.i32(len - 1);
        let over = self.s.m.cmp(c, pred::IUGT, index, last);
        let index = self.s.m.select(c, over, last, index);
        let at = self.s.m.element(c, g, index);
        Ok(self.s.m.load(c, at))
    }

    /// Whether the program walks a BVH (`image_bvh*_intersect_ray`): its
    /// descriptors' console words are read (XNW).
    fn walks_bvh(&self) -> bool {
        self.bvh_walk
    }

    /// A global memory load (`global_load_*`): from the program's constant
    /// table, or from a root SRV at the address the title sets.
    fn flat(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat { dst, .. } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        // VIctory: title memory at an address the program computes.
        if self.is_global((b, i)) {
            return if name.starts_with("global_load_") {
                self.global_load(inst, b, i)
            } else if name.starts_with("global_store_") {
                self.global_store(inst, b, i)
            } else if name.starts_with("global_atomic_") {
                self.global_atomic(inst, b, i)
            } else {
                Err("a global access to title memory")
            };
        }
        let rest = name
            .strip_prefix("global_load_")
            .ok_or("a global memory write")?;
        let (n, sub) = match rest {
            "dword" => (1, None),
            "dwordx2" => (2, None),
            "dwordx3" => (3, None),
            "dwordx4" => (4, None),
            "ubyte" => (1, Some((8, false))),
            "sbyte" => (1, Some((8, true))),
            "ushort" => (1, Some((16, false))),
            "sshort" => (1, Some((16, true))),
            _ => return Err(name),
        };
        let a = *self
            .scalar
            .get(&(b, i))
            .ok_or("an untracked global address")?;
        let guard = self.guard(b, i)?;
        let Some(Operand::Vector { reg, .. }) = *dst else {
            return Err("a global load result");
        };
        let c = self.cur;
        let off = self.address_offset(a)?;
        let h = if a.param == CODE {
            None
        } else {
            let src = Source::Built {
                address: a,
                bits: 0,
                words: [0, 0],
            };
            Some(self.handle((b, i), &src, false, (11, 0))?)
        };
        let dword = |l: &mut Self, byte: V| -> Result<V, Why> {
            match h {
                None => l.code_dword(byte),
                Some(h) => {
                    let c = l.cur;
                    let (u, mask, al) = (l.s.m.undef(l.s.t.i32), l.s.m.i8(1), l.s.m.i32(4));
                    let r = l.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        l.s.t.res_ret,
                        139,
                        &[h, byte, u, mask, al],
                    );
                    let x = l.s.m.extract(c, r, 0);
                    Ok(l.as_int(x))
                }
            }
        };
        for k in 0..n {
            let x = match sub {
                None => {
                    let four = self.s.m.i32(4 * k);
                    let at = self.s.m.bin(c, Bin::Add, off, four);
                    dword(self, at)?
                }
                Some((width, signed)) => {
                    // shortcut: the field is taken not to cross a dword.
                    let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
                    let at = self.s.m.bin(c, Bin::And, off, not3);
                    let w = dword(self, at)?;
                    let byte = self.s.m.bin(c, Bin::And, off, three);
                    let sh = self.s.m.bin(c, Bin::Shl, byte, three);
                    let x = self.s.m.bin(c, Bin::LShr, w, sh);
                    let left = self.s.m.i32(32 - width);
                    let x = self.s.m.bin(c, Bin::Shl, x, left);
                    self.s
                        .m
                        .bin(c, if signed { Bin::AShr } else { Bin::LShr }, x, left)
                }
            };
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// Finds the whole-wave sections and the values they leave.
    fn find_sections(&mut self) {
        let p = self.x.p;
        let mut inside: HashMap<Value, (usize, usize, usize)> = HashMap::new();
        for b in 0..p.blocks.len() {
            let found = super::wave::sections(p, b);
            for (&start, sec) in &found {
                for i in start + 1..sec.end {
                    for &(_, w) in &p.blocks[b].insts[i].writes {
                        inside.insert(w, (b, start, sec.end));
                    }
                }
            }
            if !found.is_empty() {
                self.sections.insert(b, found);
            }
        }
        if inside.is_empty() {
            return;
        }
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                self.section_out
                    .extend(phi.args.iter().filter(|a| inside.contains_key(a)));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, r) in &inst.reads {
                    if let Some(&(sb, start, end)) = inside.get(&r)
                        && !(sb == b && i > start && i < end)
                    {
                        self.section_out.insert(r);
                    }
                }
            }
        }
    }

    /// An instruction inside a whole-wave section: the values that leave
    /// the section are made from what the simulation found them to be.
    fn section_inst(&mut self, b: usize, i: usize, start: usize) {
        const UNKNOWN: Why = "a whole-wave section step the simulation does not follow";
        let inst = &self.x.p.blocks[b].insts[i];
        for &(_, w) in &inst.writes {
            let val = if self.section_out.contains(&w) {
                let shape = self.sections[&b][&start].shapes.get(&w).copied();
                match shape.ok_or(UNKNOWN).and_then(|sh| self.shape(sh)) {
                    Ok(x) => Val::I32(x),
                    Err(why) => Val::Missing(why),
                }
            } else {
                Val::Missing(UNKNOWN)
            };
            self.vals.insert(w, val);
        }
    }

    /// A value a whole-wave section leaves, as DXIL wave operations over the
    /// host's active lanes: the console's result with console lanes; on
    /// another wave size, what the HLSL wave operation the console compiler
    /// expanded gives on that wave.
    fn shape(&mut self, shape: super::wave::Shape) -> Result<V, Why> {
        use super::wave::{Op, Shape};
        let (op, src, mask, ident, prefix) = match shape {
            Shape::K(k) => return Ok(self.s.m.i32(k)),
            Shape::Value(v) => return self.i32_of(v),
            Shape::Active(op, src, mask, ident) => (op, src, mask, ident, None),
            Shape::Masked(src, mask, ident) => {
                let c = self.cur;
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&mask)
                    .ok_or("a wave reduction's untracked mask")?;
                let on = self.boolean(m, c)?;
                let x = self.i32_of(src)?;
                let id = self.s.m.i32(ident);
                return Ok(self.s.m.select(c, on, x, id));
            }
            Shape::Prefix(op, src, mask, ident, inclusive) => {
                (op, src, mask, ident, Some(inclusive))
            }
        };
        if self.mask_data == MaskData::None {
            return Err("a wave reduction in a mesh shader");
        }
        let c = self.cur;
        let &m = self
            .x
            .st
            .masks
            .get(&mask)
            .ok_or("a wave reduction's untracked mask")?;
        let on = self.boolean(m, c)?;
        let x = self.i32_of(src)?;
        let id = self.s.m.i32(ident);
        let x = self.s.m.select(c, on, x, id);
        self.s.flags |= flags::WAVE_OPS;
        let float = matches!(op, Op::AddF | Op::MinF | Op::MaxF);
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let arg = if float { self.as_float(x) } else { x };
        let ty = if float { f32t } else { i32t };
        let sfx = if float { "f32" } else { "i32" };
        let r = match (op, prefix) {
            (Op::AddU | Op::AddF, Some(inclusive)) => {
                let (k0, k1) = (self.s.m.i8(0), self.s.m.i8(1));
                let r = self.s.op(
                    c,
                    &format!("wavePrefixOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    121,
                    &[arg, k0, k1],
                );
                if inclusive {
                    self.s.m.bin(c, Bin::Add, r, arg)
                } else {
                    r
                }
            }
            (_, Some(_)) => return Err("a wave prefix of an operation other than a sum"),
            (Op::Or | Op::And | Op::Xor, None) => {
                let k = self.s.m.i8(match op {
                    Op::And => 0,
                    Op::Or => 1,
                    _ => 2,
                });
                self.s
                    .op(c, "waveActiveBit.i32", Attr::NoUnwind, i32t, 120, &[arg, k])
            }
            (_, None) => {
                let (code, unsigned) = match op {
                    Op::AddU => (0, 1),
                    Op::AddF => (0, 0),
                    Op::MinU => (2, 1),
                    Op::MaxU => (3, 1),
                    Op::MinI | Op::MinF => (2, 0),
                    _ => (3, 0),
                };
                let (k, u) = (self.s.m.i8(code), self.s.m.i8(unsigned));
                self.s.op(
                    c,
                    &format!("waveActiveOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    119,
                    &[arg, k, u],
                )
            }
        };
        Ok(if float { self.as_int(r) } else { r })
    }

    fn run(mut self) -> Result<Lowered, Stop> {
        let t = std::time::Instant::now();
        let p = self.x.p;
        let order = crate::ir::reverse_postorder(&p.blocks);
        for (n, &b) in order.iter().enumerate() {
            self.rpo[b] = n;
        }
        for (n, &b) in order.iter().enumerate() {
            self.block[b] = Some(if n == 0 {
                self.s.entry
            } else {
                self.s.m.new_block()
            });
        }
        // The loop budget (`LOOP_LIMIT`) flows through every block; the
        // entry starts it at 0 (unless a back edge reaches it).
        for (n, &b) in order.iter().enumerate() {
            let header = p.blocks[b]
                .preds
                .iter()
                .any(|&q| self.block[q].is_some() && self.rpo[b] <= self.rpo[q]);
            if header || n > 0 {
                let blk = self.block[b].expect("reachable");
                let phi = self.s.m.phi(blk, self.s.t.i32);
                self.counters.insert(b, (phi, Vec::new()));
            }
        }
        self.cur = self.s.entry;
        self.find_sections();
        self.inputs()?;
        if let Some(mesh) = self.s.mesh {
            let r = if self.s.fallback {
                self.fallback_prologue(mesh)
            } else {
                self.mesh_prologue(mesh)
            };
            r.map_err(|w| unsupported(0, w))?;
            self.block[order[0]] = Some(self.cur);
        }
        for &b in &order {
            self.cur = self.block[b].expect("reachable");
            self.cur_ir = b;
            self.budget = Some(match self.counters.get(&b) {
                Some(&(phi, _)) => phi,
                None => self.s.m.i32(0),
            });
            let preds: Vec<usize> = p.blocks[b]
                .preds
                .iter()
                .copied()
                .filter(|&q| self.block[q].is_some())
                .collect();
            self.exported[b] = !preds.is_empty() && preds.iter().all(|&q| self.exported_out(q));
            for (i, phi) in p.blocks[b].phis.iter().enumerate() {
                let val = match phi.reg {
                    Reg::S(c) if c == EXEC || c == EXEC + 1 => Val::Missing("a lane mask"),
                    _ if self.bad.contains_key(&phi.value) => Val::Missing(self.bad[&phi.value]),
                    Reg::Scc => {
                        let v = self.s.m.phi(self.cur, self.s.t.i1);
                        self.data_phis.push((v, b, i));
                        Val::Bit(v)
                    }
                    _ => {
                        let v = self.s.m.phi(self.cur, self.s.t.i32);
                        self.data_phis.push((v, b, i));
                        Val::I32(v)
                    }
                };
                self.vals.insert(phi.value, val);
            }
            let mut i = 0;
            while i < p.blocks[b].insts.len() {
                self.inst(b, i)?;
                if let Some(end) = self.sections.get(&b).and_then(|s| s.get(&i)).map(|s| s.end) {
                    for j in i + 1..end {
                        self.section_inst(b, j, i);
                    }
                    i = end;
                    continue;
                }
                i += 1;
            }
            self.mask_ok = self.mask_data == MaskData::Console;
            self.tail[b] = Some(self.cur);
            self.exit(b)?;
        }
        super::lap("lower.walk", t);
        let t = std::time::Instant::now();
        self.fill_phis()?;
        super::lap("lower.phis", t);
        let user_data = self.user_reads.values().copied().collect();
        self.s.samplers = self.samplers.len() as u32;
        let root_cbvs = self.s.root_cbvs.iter().map(|r| r.0).collect();
        let root_srvs = self.s.root_srvs.iter().map(|r| r.0).collect();
        let root_uavs = self.s.root_uavs.iter().map(|r| r.0).collect();
        let address_map = self.s.address_map.is_some();
        let descriptor_words = self.s.descriptor_words.is_some();
        let cmp_samplers = self.s.cmp_samplers;
        #[cfg(test)]
        let op_names = self.s.m.func_names().map(str::to_owned).collect();
        #[cfg(test)]
        let insts = self.s.m.inst_lines();
        Ok(Lowered {
            cmp_samplers,
            root_cbvs,
            root_srvs,
            root_uavs,
            address_map,
            descriptor_words,
            dxil: {
                let t = std::time::Instant::now();
                let d = self.s.finish();
                super::lap("finish", t);
                d
            },
            user_data,
            samplers: self.samplers,
            #[cfg(test)]
            op_names,
            #[cfg(test)]
            insts,
        })
    }

    fn exported_out(&self, b: usize) -> bool {
        self.exported[b] || self.has_done_export(b)
    }

    fn has_done_export(&self, b: usize) -> bool {
        (0..self.x.p.blocks[b].insts.len())
            .any(|i| self.exports.get(&(b, i)).is_some_and(|e| e.done))
    }

    /// Values of the program's input registers.
    fn inputs(&mut self) -> Result<(), Stop> {
        let io_inputs: HashMap<Value, Input> = self.x.io.inputs.iter().copied().collect();
        for &(reg, v) in &self.x.p.inputs {
            let val = match reg {
                Reg::S(_) if self.params.contains_key(&v) => continue, // loaded on first use
                Reg::S(n) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Above the SGPRs the hardware initializes, a register
                    // holds what the last wave left, so no program depends
                    // on it: a read the paths only seem to allow (a branch
                    // past the write, taken only with one past the read)
                    // reads zero, as VGPRs do.
                    None if n <= 105 && self.x.io.initialized_sgprs.is_some_and(|k| n >= k) => {
                        Val::I32(self.s.m.i32(0))
                    }
                    None => Val::Missing("a hardware SGPR input"),
                },
                Reg::Scc => Val::Missing("SCC on entry"),
                Reg::V(_) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Registers no one set read as zero (the validator
                    // refuses reads of undef).
                    None => Val::I32(self.s.m.i32(0)),
                },
            };
            self.vals.insert(v, val);
        }
        Ok(())
    }

    fn input(&mut self, input: Input) -> Val {
        let (kind, col) = match input {
            Input::VertexId => (1, 0),
            Input::InstanceId => (2, 0),
            Input::Position(c) => (3, c),
            Input::Attribute { row, channel } => {
                return match self.find_input(row, channel) {
                    Some((id, col)) => self.load_input(id, col),
                    None => Val::Missing("a vertex input the signature lacks"),
                };
            }
            Input::Barycentric { .. } => return self.barycentric(),
            Input::PrimitiveId => return Val::Missing("the primitive id"),
            // The console code tests it against 0 (`v_cmp_ne_u32`), as the
            // host's value can be.
            Input::FrontFace => (13, 0),
            Input::ThreadIdInGroup(c) => return self.thread_id(c),
            Input::GroupId(c) => return self.console_group(c),
            Input::GroupInfo => return Val::Missing("the thread-group info SGPR"),
            Input::MeshGroup => {
                let Val::I32(g) = self.console_group(0) else {
                    unreachable!()
                };
                let n = self.s.m.i32(self.x.mesh_stride);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Mul, g, n));
            }
            Input::MergedWaveInfo => {
                // shortcut: only the wave index (bits 24-27) is set; the
                // console's other bits (lane counts) are read by no program.
                let f = self.mesh_thread();
                let shift = self.s.m.i32(self.wave().trailing_zeros());
                let w = self.s.m.bin(self.s.entry, Bin::LShr, f, shift);
                let k24 = self.s.m.i32(24);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Shl, w, k24));
            }
            _ => return Val::Missing("a pixel system input"),
        };
        let free = self
            .s
            .inputs
            .iter()
            .filter_map(|e| e.row)
            .map(|r| r + 1)
            .max()
            .unwrap_or(0);
        let id = match self.s.inputs.iter().position(|e| e.kind == kind) {
            Some(id) => {
                if kind != 3 {
                    self.s.inputs[id].row.get_or_insert(free);
                }
                id
            }
            None if kind >= 3 => return Val::Missing("a pixel system input the signature lacks"),
            None => {
                // The console signature leaves out vertex and instance ids; PC
                // gives them a row after the others.
                let row = self
                    .s
                    .inputs
                    .iter()
                    .filter_map(|e| e.row)
                    .map(|r| r + 1)
                    .max()
                    .unwrap_or(0);
                self.s.inputs.push(SigElement {
                    name: ["", "SV_VertexID", "SV_InstanceID"][kind as usize].into(),
                    index: 0,
                    kind,
                    comp: 5,
                    interp: 0,
                    row: Some(row),
                    col: 0,
                    cols: 1,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        self.load_input(id, col)
    }

    /// XNW: a barycentric VGPR read as data (not by `v_interp`) in a
    /// program whose signature declares no `SV_Barycentrics`: the source
    /// never asked for them, so the read is of a VGPR the compiler reuses
    /// before writing it whole, and no defined result depends on it (V:
    /// 7e040f4d shifts v1's bits out at 0x12c-0x14c; 83e0fdc4 keeps the
    /// high half of v1 at 0x518, then shifts it out at 0x528). It reads 0.
    /// The host's SV_Barycentrics is no option here: the RX 5700 XT reports
    /// BarycentricsSupported 0 (OPTIONS3).
    fn barycentric(&mut self) -> Val {
        if self.s.inputs.iter().any(|e| e.kind == 28) {
            return Val::Missing("barycentrics read directly");
        }
        Val::I32(self.s.m.i32(0))
    }

    /// A component of a compute thread or group id, read in the entry block.
    fn compute_id(&mut self, name: &str, opcode: u32, c: u8) -> Val {
        let (entry, c) = (self.s.entry, self.s.m.i32(u32::from(c)));
        Val::I32(
            self.s
                .op(entry, name, Attr::ReadNone, self.s.t.i32, opcode, &[c]),
        )
    }

    fn find_input(&self, row: u8, channel: u8) -> Option<(usize, u8)> {
        self.s.inputs.iter().enumerate().find_map(|(id, e)| {
            let r = e.row?;
            // `then`, not `then_some`: the channel is only past `col` when
            // the condition holds.
            (r == u32::from(row) && channel >= e.col && channel < e.col + e.cols)
                .then(|| (id, channel - e.col))
        })
    }

    fn load_input(&mut self, id: usize, col: u8) -> Val {
        if self.s.inputs[id].half {
            // shortcut: 16-bit vertex inputs (the fetch routine's) are not
            // unpacked; pixel inputs are (`interp`).
            return Val::Missing("a 16-bit input");
        }
        self.load_any_input(id, col)
    }

    /// An input element's channel as declared (16-bit ones as 32-bit floats).
    fn load_any_input(&mut self, id: usize, col: u8) -> Val {
        let e = &mut self.s.inputs[id];
        e.usage |= 1 << col;
        let (int, bit) = (e.comp != 9, e.comp == 1);
        let ty = if bit {
            self.s.t.i1
        } else if int {
            self.s.t.i32
        } else {
            self.s.t.f32
        };
        let suffix = if bit {
            ".i1"
        } else if int {
            ".i32"
        } else {
            ".f32"
        };
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.undef(self.s.t.i32),
        ];
        let v = self.s.op(
            entry,
            &format!("loadInput{suffix}"),
            Attr::ReadNone,
            ty,
            4,
            &args,
        );
        if bit {
            let z = self.s.m.cast(entry, Cast::ZExt, v, self.s.t.i32);
            Val::I32(z)
        } else if int {
            Val::I32(v)
        } else {
            Val::I32(self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.i32))
        }
    }

    /// The DXIL form of an IR value.
    fn get(&mut self, v: Value) -> Result<Val, Why> {
        if self.phi_at.contains_key(&v) {
            self.read_phis.insert(v);
        }
        let is_mask = self.x.st.masks.contains_key(&v) || self.x.st.highs.contains_key(&v);
        match self.vals.get(&v) {
            Some(&Val::Missing(_)) | None if self.mask_ok && is_mask => {
                return self.mask_dword(v).map(Val::I32);
            }
            Some(&Val::Missing(w)) => return Err(w),
            Some(&val) => return Ok(val),
            None => {}
        }
        if let Some(&(param, dword)) = self.params.get(&v) {
            let val = self.user(param, dword);
            self.vals.insert(v, val);
            return Ok(val);
        }
        Err("a value defined after its use")
    }

    fn i32_of(&mut self, v: Value) -> Result<V, Why> {
        match self.get(v)? {
            Val::I32(x) => Ok(x),
            Val::Bit(x) => Ok(self.s.m.cast(self.cur, Cast::ZExt, x, self.s.t.i32)),
            Val::Missing(w) => Err(w),
        }
    }

    /// A user-data dword: a root parameter's value.
    fn user(&mut self, param: usize, dword: u32) -> Val {
        let rp = self.x.native.layout[param];
        match rp.root {
            RootType::ResourceHeap | RootType::SamplerHeap => return Val::I32(self.s.m.i32(0)),
            RootType::Constants | RootType::Table => {}
            _ => return Val::Missing("a root descriptor"),
        }
        let index = u32::from(rp.user_data) + dword;
        if self.x.gpu_desc && index >= self.window_base() {
            return Val::Missing("a user-data dword where GPU-side descriptors put window dwords");
        }
        self.user_reads.insert(
            index,
            UserData {
                index,
                slot: rp.slot,
                root: rp.root,
                dword,
            },
        );
        self.s.user_data = self.s.user_data.max(index + 1);
        let h = match self.user_data {
            Some(h) => h,
            None => {
                let h = self.s.user_data_handle();
                self.user_data = Some(h);
                h
            }
        };
        let entry = self.s.entry;
        let row = self.s.m.i32(index / 4);
        let r = self.s.op(
            entry,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, row],
        );
        Val::I32(self.s.m.extract(entry, r, index % 4))
    }

    /// A dword of a lane mask as a number: the ballot of the lanes' bits,
    /// taken in the block that writes the mask (see [`MaskData`]).
    fn mask_dword(&mut self, v: Value) -> Result<V, Why> {
        let (lo, high) = match self.x.st.highs.get(&v) {
            Some(&lo) => (lo, true),
            None => (v, false),
        };
        let &m = self.x.st.masks.get(&lo).ok_or(LANE_MASK)?;
        if !self.x.st.active.contains(&lo) {
            return Err("a lane mask holding lanes elsewhere read as data");
        }
        let (x, y) = match self.ballots.get(&lo) {
            Some(&xy) => xy,
            None => {
                let at = match self.def_block.get(&lo) {
                    // Program inputs (EXEC on entry).
                    None => self.s.entry,
                    Some(&b) if b == self.cur_ir => self.cur,
                    Some(&b) => self.tail[b].ok_or("a lane mask written after its use")?,
                };
                if !self.balloting.insert(lo) {
                    return Err("a lane mask whose lanes depend on its own value");
                }
                let bit = self.boolean(m, at);
                self.balloting.remove(&lo);
                let bit = bit?;
                let i32t = self.s.t.i32;
                let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
                let r = self
                    .s
                    .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
                self.s.flags |= flags::WAVE_OPS;
                let xy = (self.s.m.extract(at, r, 0), self.s.m.extract(at, r, 1));
                self.ballots.insert(lo, xy);
                xy
            }
        };
        Ok(if high {
            y
        } else if self.mask_data == MaskData::Host && !self.x.native.wave64 {
            // A wave32 program's host wave may still have 64 lanes, whose
            // upper half is a second console wave: its bits are the ballot's
            // second dword.
            let lane = self.lane_index();
            let c = self.cur;
            let k32 = self.s.m.i32(32);
            let upper = self.s.m.cmp(c, pred::IUGE, lane, k32);
            self.s.flags |= flags::WAVE_OPS;
            self.s.m.select(c, upper, y, x)
        } else {
            x
        })
    }

    // ---- operands ----

    fn operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Ok(self.s.m.i32(0)),
            Operand::Scalar { code, .. } => {
                let v = inst.read(Reg::S(code)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            Operand::Vector { reg, .. } => {
                let v = inst.read(Reg::V(reg)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            Operand::Const(c) => crate::resource::inline(c)
                .map(|x| self.s.m.i32(x))
                .ok_or("a special operand (vccz, execz, scc, lds_direct)"),
            Operand::Literal(x) => Ok(self.s.m.i32(x)),
            Operand::Attr { .. } => Err("an attribute operand"),
        }
    }

    /// GFX10 puts an inline f16 constant in the low half of the operand.
    fn f16_operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Const(c @ 240..=248) => {
                let h = [
                    0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118,
                ][usize::from(c - 240)];
                Ok(self.s.m.i32(h))
            }
            _ => self.operand(inst, op),
        }
    }

    fn as_float(&mut self, x: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, x, self.s.t.f32)
    }

    fn as_int(&mut self, f: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, f, self.s.t.i32)
    }

    /// The SDWA operand selection of source `k`: a byte or word of the
    /// dword, zero- or sign-extended (none: the whole dword).
    fn sdwa_src(inst: &Inst, k: usize) -> Option<(u8, bool)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        let sel = *s.src_sel.get(k)?;
        let sext = s.sext >> k & 1 != 0;
        (sel < 6).then_some((sel, sext))
    }

    /// Bits `sel` (0-3 a byte, 4-5 a word) of `x`, extended.
    fn select_bits(&mut self, x: V, sel: u8, sext: bool) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8, 8 * sel)
        } else {
            (16, 16 * (sel - 4))
        };
        let left = self.s.m.i32(u32::from(32 - width - at));
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - u32::from(width));
        let op = if sext { Bin::AShr } else { Bin::LShr };
        self.s.m.bin(c, op, t, right)
    }

    /// A float source with its abs/neg modifiers.
    fn fsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let selected = Self::sdwa_src(inst, k);
        let mut f = match op {
            _ if selected.is_some() => {
                let (sel, sext) = selected.expect("checked");
                let x = self.operand(inst, &op)?;
                let x = self.select_bits(x, sel, sext);
                self.as_float(x)
            }
            _ if k == 0 && self.src0.is_some() => {
                let x = self.src0.expect("checked");
                self.as_float(x)
            }
            Operand::Vector { reg, .. } | Operand::Scalar { code: reg, .. }
                if !matches!(op, Operand::Scalar { code: NULL, .. }) =>
            {
                let r = if matches!(op, Operand::Vector { .. }) {
                    Reg::V(reg)
                } else {
                    Reg::S(reg)
                };
                let val = inst.read(r).ok_or("an unread operand")?;
                match self.floats.get(&val) {
                    Some(&f) => f,
                    None => {
                        let x = self.i32_of(val)?;
                        self.as_float(x)
                    }
                }
            }
            _ => {
                let x = self.operand(inst, &op)?;
                self.as_float(x)
            }
        };
        if v.abs >> k & 1 != 0 {
            f = self.unary(6, f);
        }
        if v.neg >> k & 1 != 0 {
            let z = self.s.m.f32(0x8000_0000);
            f = self.s.m.bin(self.cur, Bin::Sub, z, f);
        }
        Ok(f)
    }

    fn src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 {
            return Err("integer operand modifiers");
        }
        match self.src0 {
            Some(x) if k == 0 => Ok(x),
            _ => {
                let x = self.operand(inst, &op)?;
                Ok(match Self::sdwa_src(inst, k) {
                    Some((sel, sext)) => self.select_bits(x, sel, sext),
                    None => x,
                })
            }
        }
    }

    fn unary(&mut self, op: u32, f: V) -> V {
        self.s.op(
            self.cur,
            "unary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[f],
        )
    }

    fn binary(&mut self, op: u32, a: V, b: V) -> V {
        self.s.op(
            self.cur,
            "binary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[a, b],
        )
    }

    /// A fused f32 multiply-add (`v_fma_f32`, `v_fmac_f32`, `v_fma_mix_f32`).
    /// XNW addition: [`super::Fma::Exact`] computes it as a double `fma`
    /// rounded to f32 (the host's only fused route); the f32 product is
    /// exact in a double, so only the final rounding can differ from a
    /// single fused step, by double rounding. [`super::Fma::Fast`] is
    /// VIctory's `FMad`, which this driver does not fuse.
    fn fused(&mut self, a: V, b: V, c: V) -> V {
        if self.x.fma == super::Fma::Fast {
            return self.fmad(a, b, c);
        }
        let blk = self.cur;
        let f64t = self.s.m.double();
        let f32t = self.s.t.f32;
        let [a, b, c] = [a, b, c].map(|v| self.s.m.cast(blk, Cast::FpExt, v, f64t));
        self.s.flags |= super::shader::flags::DOUBLES | super::shader::flags::DOUBLE_EXTENSIONS;
        let r = self.s.op(blk, "tertiary.f64", Attr::ReadNone, f64t, 47, &[a, b, c]);
        self.s.m.cast(blk, Cast::FpTrunc, r, f32t)
    }

    /// The hardware's maximum (`max`) or minimum of two floats: as DXIL's
    /// FMax/FMin, but -0 orders below +0 (RDNA2 ISA, `v_max_f32`), where
    /// the compare and select DXIL specifies gives either zero.
    fn max_min(&mut self, max: bool, a: V, b: V) -> V {
        let c = self.cur;
        let r = self.binary(if max { 35 } else { 36 }, a, b);
        let (ia, ib) = (self.as_int(a), self.as_int(b));
        let either = self.s.m.bin(c, Bin::Or, ia, ib);
        let magnitude = self.s.m.i32(0x7FFF_FFFF);
        let either = self.s.m.bin(c, Bin::And, either, magnitude);
        let zero = self.s.m.i32(0);
        let zeros = self.s.m.cmp(c, pred::EQ, either, zero);
        // Of two zeros, +0 unless both are -0 (max), -0 unless neither (min).
        let op = if max { Bin::And } else { Bin::Or };
        let sign = self.s.m.bin(c, op, ia, ib);
        let signed = self.as_float(sign);
        self.s.m.select(c, zeros, signed, r)
    }

    fn fmad(&mut self, a: V, b: V, c: V) -> V {
        self.s.op(
            self.cur,
            "tertiary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            46,
            &[a, b, c],
        )
    }

    /// `f` as an f16 in the low bits, rounded toward zero: the nearest-even
    /// conversion, one step back toward zero where it rounded away (which
    /// also turns an overflow to infinity into the largest finite half).
    fn f16_toward_zero(&mut self, f: V) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let h = self
            .s
            .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
        let back = self
            .s
            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h]);
        let (fa, ba) = (self.unary(6, f), self.unary(6, back));
        // Ordered: never for NaN.
        let away = self.s.m.cmp(c, pred::OGT, ba, fa);
        let one = self.s.m.i32(1);
        let stepped = self.s.m.bin(c, Bin::Sub, h, one);
        self.s.m.select(c, away, stepped, h)
    }

    /// `(a >> off[4:0]) & ((1 << width) - 1)`, sign-extended from the field's
    /// top bit when `signed`; widths from 32 on take every bit. A signed
    /// field that runs past bit 31 extends from bit 31, as the hardware
    /// does: the shift is arithmetic then.
    fn bfe(&mut self, a: V, off: V, width: V, signed: bool) -> V {
        let c = self.cur;
        let (k31, k32, zero, one, ones) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(0),
            self.s.m.i32(1),
            self.s.m.i32(u32::MAX),
        );
        let off = self.s.m.bin(c, Bin::And, off, k31);
        let shift = if signed { Bin::AShr } else { Bin::LShr };
        let shifted = self.s.m.bin(c, shift, a, off);
        let full = self.s.m.cmp(c, pred::IUGE, width, k32);
        let w = self.s.m.bin(c, Bin::And, width, k31);
        let mask = self.s.m.bin(c, Bin::Shl, one, w);
        let mask = self.s.m.bin(c, Bin::Sub, mask, one);
        let mask = self.s.m.select(c, full, ones, mask);
        let r = self.s.m.bin(c, Bin::And, shifted, mask);
        if !signed {
            return r;
        }
        // Sign-extend a field narrower than 32 bits (none for width 0).
        let left = self.s.m.bin(c, Bin::Sub, k32, w);
        let left = self.s.m.bin(c, Bin::And, left, k31);
        let t = self.s.m.bin(c, Bin::Shl, r, left);
        let t = self.s.m.bin(c, Bin::AShr, t, left);
        let is0 = self.s.m.cmp(c, pred::EQ, w, zero);
        let keep = self.s.m.bin(c, Bin::Or, full, is0);
        self.s.m.select(c, keep, r, t)
    }

    /// The median of three. With a NaN among them the hardware gives the
    /// minimum of the three instead (the host's minimum skips NaNs, so that
    /// is the smallest number among them).
    fn med3(&mut self, a: V, b: V, third: V) -> V {
        let c = self.cur;
        let lo = self.binary(36, a, b);
        let hi = self.binary(35, a, b);
        let t = self.binary(36, hi, third);
        let median = self.binary(35, lo, t);
        let least = self.binary(36, lo, third);
        let ab = self.s.m.cmp(c, pred::UNO, a, b);
        let nan3 = self.s.m.cmp(c, pred::UNO, third, third);
        let nan = self.s.m.bin(c, Bin::Or, ab, nan3);
        self.s.m.select(c, nan, least, median)
    }

    /// `a - floor(a)`, which on the hardware never reaches 1.0: where it
    /// would round to it (a tiny negative `a`), the result is `below_one`,
    /// the largest value under 1.0 in the operation's width. Hosts differ
    /// here (some give 1.0). Comparing against a `threshold` distinct from
    /// `below_one` prevents drivers (observed on NVIDIA) from folding
    /// `r > k ? k : r` into `minNum` (`FMNMX`) and swallowing NaN.
    fn fract(&mut self, a: V, threshold: f32, below_one: f32) -> V {
        let c = self.cur;
        let r = self.unary(22, a);
        let limit = self.s.m.f32(threshold.to_bits());
        let k = self.s.m.f32(below_one.to_bits());
        let over = self.s.m.cmp(c, pred::OGE, r, limit);
        self.s.m.select(c, over, k, r)
    }

    /// Sine, or cosine, of an angle in turns, as the hardware takes it. The
    /// whole turns are taken off first (`x - round_even(x)`, exact, within
    /// half a turn), so a large angle keeps its fraction; scaling to radians
    /// first rounds it away. A zero passes unreduced (the sine of -0 is -0).
    /// The sine of a half turn and the cosine of a quarter turn are exactly
    /// +0 on the hardware, where the host's function of the rounded angle
    /// gives about 1e-7. Infinities and NaN give NaN.
    /// shortcut: below 2^-12 turns the hardware scales by a slightly
    /// different constant than 2π (a last-place difference).
    fn trig_turns(&mut self, x: V, cosine: bool) -> V {
        let c = self.cur;
        let zero = self.s.m.f32(0);
        let whole = self.unary(26, x);
        let part = self.s.m.bin(c, Bin::Sub, x, whole);
        let is_zero = self.s.m.cmp(c, pred::OEQ, x, zero);
        let part = self.s.m.select(c, is_zero, x, part);
        let k = self.s.m.f32(TWO_PI.to_bits());
        let angle = self.s.m.bin(c, Bin::Mul, part, k);
        let r = self.unary(if cosine { 12 } else { 13 }, angle);
        let at = if cosine { 0.25f32 } else { 0.5 };
        let at = self.s.m.f32(at.to_bits());
        let size = self.unary(6, part);
        let exact = self.s.m.cmp(c, pred::OEQ, size, at);
        self.s.m.select(c, exact, zero, r)
    }

    /// The high dword of a 32 × 32-bit product.
    fn mul_hi(&mut self, a: V, b: V, signed: bool) -> V {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let ext = if signed { Cast::SExt } else { Cast::ZExt };
        let a = self.s.m.cast(c, ext, a, i64t);
        let b = self.s.m.cast(c, ext, b, i64t);
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let k32 = self.s.m.int_const(i64t, 32);
        let hi = self.s.m.bin(c, Bin::LShr, m, k32);
        self.s.flags |= flags::INT64;
        self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32)
    }

    /// `0 × anything = 0`, as in D3D9.
    fn legacy_mul(&mut self, a: V, b: V) -> V {
        let c = self.cur;
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let zero = self.s.m.f32(0);
        let z = self.legacy_zero(a, b);
        self.s.m.select(c, z, zero, m)
    }

    /// Whether a legacy multiply's product is zero: a factor with zero
    /// exponent bits (zero or denormal). XNW, after VIctory's
    /// description (not its code): denormals count as zero; a compare with
    /// 0.0 would miss them where the host keeps denormals.
    fn legacy_zero(&mut self, a: V, b: V) -> V {
        let c = self.cur;
        let (ai, bi) = (self.as_int(a), self.as_int(b));
        let exp = self.s.m.i32(0x7F80_0000);
        let zero = self.s.m.i32(0);
        let ae = self.s.m.bin(c, Bin::And, ai, exp);
        let be = self.s.m.bin(c, Bin::And, bi, exp);
        let az = self.s.m.cmp(c, pred::EQ, ae, zero);
        let bz = self.s.m.cmp(c, pred::EQ, be, zero);
        self.s.m.bin(c, Bin::Or, az, bz)
    }

    /// `a × b + addend` fused, with `0 × anything = 0` as in D3D9
    /// (`v_fma_legacy_f32`).
    fn fma_legacy(&mut self, a: V, b: V, addend: V) -> V {
        let c = self.cur;
        let zero = self.s.m.f32(0);
        let z = self.legacy_zero(a, b);
        // `+0 + addend`: the addend, except -0 gives +0. XNW: written as a
        // select, because the host compiler folds `0.0 + x` to `x` (measured:
        // -0 came back); XCL reports +0 on the console (I).
        let ai = self.as_int(addend);
        let neg0 = self.s.m.i32(0x8000_0000);
        let is_neg0 = self.s.m.cmp(c, pred::EQ, ai, neg0);
        let product_zero = self.s.m.select(c, is_neg0, zero, addend);
        // XNW: through the `--fma` switch, as `v_fma_f32` (VIctory used FMad).
        let fused = self.fused(a, b, addend);
        self.s.m.select(c, z, product_zero, fused)
    }

    // ---- lane booleans ----

    fn boolean(&mut self, b: Bool, at: usize) -> Result<V, Why> {
        if b == Bool::TRUE || b == Bool::FALSE {
            return Ok(self.s.m.bool(b == Bool::TRUE));
        }
        if let Some(&v) = self.bools.get(&(at, b)) {
            return Ok(v);
        }
        let (atom, lo, hi) = self.x.st.bools.node(b).expect("not a constant");
        let a = self.atom(atom, at)?;
        let m = &mut self.s.m;
        let v = match (lo, hi) {
            (Bool::FALSE, Bool::TRUE) => a,
            (Bool::TRUE, Bool::FALSE) => {
                let t = m.bool(true);
                m.bin(at, Bin::Xor, a, t)
            }
            (Bool::FALSE, h) => {
                let h = self.boolean(h, at)?;
                self.s.m.bin(at, Bin::And, a, h)
            }
            (l, Bool::TRUE) => {
                let l = self.boolean(l, at)?;
                self.s.m.bin(at, Bin::Or, a, l)
            }
            (l, h) => {
                let l = self.boolean(l, at)?;
                let h = self.boolean(h, at)?;
                self.s.m.select(at, a, h, l)
            }
        };
        self.bools.insert((at, b), v);
        Ok(v)
    }

    fn atom(&mut self, atom: Atom, at: usize) -> Result<V, Why> {
        match atom {
            Atom::Lane(v) => self.lane_bits.get(&v).copied().ok_or_else(|| {
                self.lane_fail
                    .get(&v)
                    .copied()
                    .unwrap_or("a lane mask from a carry")
            }),
            Atom::Uniform(v) => match self.get(v)? {
                Val::Bit(x) => Ok(x),
                Val::I32(x) => {
                    let z = self.s.m.i32(0);
                    Ok(self.s.m.cmp(at, pred::NE, x, z))
                }
                Val::Missing(w) => Err(w),
            },
            Atom::Phi(id) => {
                if let Some(&v) = self.mask_phis.get(&id) {
                    return Ok(v);
                }
                let &(b, i) = self
                    .phi_at
                    .get(&id)
                    .ok_or("a lane-mask join without a phi")?;
                let blk = self.block[b].ok_or("an unreachable join")?;
                let v = self.s.m.phi(blk, self.s.t.i1);
                self.mask_phis.insert(id, v);
                self.pending_masks.push((v, b, i));
                Ok(v)
            }
            // shortcut: "some lane of the quad is set" is taken as true: the
            // lanes it adds only feed derivatives, and exports keep their own
            // masks. Wrong if such a section had a side effect.
            Atom::Quad(_) => Ok(self.s.m.bool(true)),
            // With console lanes, a mask held as a number is the console's:
            // the lane tests its own bit.
            Atom::Opaque(v) if self.console_lanes => {
                let hi = match self.pair_high(v) {
                    Some(h) => Some(self.get(h)?),
                    None => None,
                };
                let lo = self.get(v)?;
                let word = |w: Val| match w {
                    Val::I32(x) => Ok(x),
                    _ => Err("an opaque lane mask"),
                };
                let lo = word(lo)?;
                let hi = hi.map(word).transpose()?;
                Ok(self.lane_bit(at, lo, hi))
            }
            Atom::Constant(k) if self.console_lanes => {
                let lo = self.s.m.i32(k as u32);
                let hi = self.s.m.i32((k >> 32) as u32);
                Ok(self.lane_bit(at, lo, Some(hi)))
            }
            Atom::Opaque(_) => Err("an opaque lane mask"),
            Atom::Constant(_) => Err("a constant lane mask"),
        }
    }

    /// This lane's bit from a mask operand (`v_cndmask_b32` / `*_ci_u32`):
    /// uses the tracked boolean when `structure` followed it, handles inline
    /// `0`/`-1` constants, or tests this console lane's bit of the scalar
    /// mask register (pair in wave64) when console lanes are active.
    fn mask_cond(
        &mut self,
        inst: &Inst,
        op: Option<&Operand>,
        at: usize,
        err: Why,
    ) -> Result<V, Why> {
        match op {
            Some(&Operand::Const(128)) => Ok(self.s.m.bool(false)),
            Some(&Operand::Const(193)) => Ok(self.s.m.bool(true)),
            Some(&Operand::Scalar { code, .. }) => {
                if let Some(m) = inst
                    .read(Reg::S(code))
                    .and_then(|v| self.x.st.masks.get(&v).copied())
                {
                    return self.boolean(m, at);
                }
                if !self.console_lanes {
                    return Err(err);
                }
                let lo = inst.read(Reg::S(code)).ok_or(err)?;
                let lo = self.i32_of(lo)?;
                let hi = if self.x.native.wave64 {
                    match inst.read(Reg::S(code + 1)) {
                        Some(v) => Some(self.i32_of(v)?),
                        None => Some(self.s.m.i32(0)),
                    }
                } else {
                    None
                };
                self.s.flags |= flags::WAVE_OPS;
                Ok(self.lane_bit(at, lo, hi))
            }
            _ => Err(err),
        }
    }

    /// This lane's bit of the mask `lo`, `hi` (wave64).
    fn lane_bit(&mut self, at: usize, lo: V, hi: Option<V>) -> V {
        let lane = self.s.op(
            at,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        );
        let (k31, k32, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let word = match hi {
            Some(hi) => {
                let low = self.s.m.cmp(at, pred::IULT, lane, k32);
                self.s.m.select(at, low, lo, hi)
            }
            None => lo,
        };
        let sh = self.s.m.bin(at, Bin::And, lane, k31);
        let t = self.s.m.bin(at, Bin::LShr, word, sh);
        let t = self.s.m.bin(at, Bin::And, t, one);
        self.s.m.cmp(at, pred::NE, t, z)
    }

    /// The value of the high dword of the wave64 register pair whose low
    /// dword holds `lo` (none in wave32).
    fn pair_high(&self, lo: Value) -> Option<Value> {
        if !self.x.native.wave64 {
            return None;
        }
        let p = self.x.p;
        let pick = |reg: Reg| match reg {
            Reg::S(c) => Some(Reg::S(c + 1)),
            _ => None,
        };
        if let Some(&(reg, _)) = p.inputs.iter().find(|i| i.1 == lo) {
            let h = pick(reg)?;
            return p.inputs.iter().find(|i| i.0 == h).map(|i| i.1);
        }
        if let Some(&(b, i)) = self.phi_at.get(&lo) {
            let h = pick(p.blocks[b].phis[i].reg)?;
            return p.blocks[b]
                .phis
                .iter()
                .find(|q| q.reg == h)
                .map(|q| q.value);
        }
        let b = *self.def_block.get(&lo)?;
        let inst = p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == lo))?;
        let &(reg, _) = inst.writes.iter().find(|w| w.1 == lo)?;
        inst.write(pick(reg)?)
    }

    fn guard(&mut self, b: usize, i: usize) -> Result<Option<V>, Why> {
        match self.x.st.guards[b][i] {
            None => Ok(None),
            Some(g) => Ok(Some(self.boolean(g, self.cur)?)),
        }
    }

    // ---- results ----

    fn set(&mut self, v: Option<Value>, val: Val) {
        if let Some(v) = v {
            self.vals.insert(v, val);
        }
    }

    /// Writes VGPR `reg` (or the SGPR of a lane read), under `guard`.
    fn write(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let x = match (Self::sdwa_dst(inst), reg) {
            (Some((sel, unused)), Reg::V(_)) => self.place_bits(inst, reg, x, sel, unused),
            _ => x,
        };
        self.write_placed(inst, reg, x, guard);
    }

    /// [`Self::write`] of a dword already placed (SDWA applied).
    fn write_placed(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let Some(new) = inst.write(reg) else { return };
        let x = match (guard, reg) {
            (Some(g), Reg::V(_)) => {
                let old = match inst.read(reg).map(|o| self.i32_of(o)) {
                    Some(Ok(o)) => o,
                    // The validator refuses reads of undef.
                    _ => self.s.m.i32(0),
                };
                self.s.m.select(self.cur, g, x, old)
            }
            _ => x,
        };
        self.vals.insert(new, Val::I32(x));
    }

    /// The SDWA destination selection: a byte or word, and what the other
    /// bits get (0 zeros, 1 the sign above the field, 2 the old bits).
    fn sdwa_dst(inst: &Inst) -> Option<(u8, u8)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        (s.dst_sel < 6).then_some((s.dst_sel, s.dst_unused))
    }

    /// The low byte or word of `x` placed at `sel` in VGPR `reg`.
    fn place_bits(&mut self, inst: &Inst, reg: Reg, x: V, sel: u8, unused: u8) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8u32, 8 * u32::from(sel))
        } else {
            (16, 16 * u32::from(sel - 4))
        };
        let left = self.s.m.i32(32 - width);
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - width - at);
        if unused == 1 {
            // Bits below the field shift in as zeros.
            return self.s.m.bin(c, Bin::AShr, t, right);
        }
        let field = self.s.m.bin(c, Bin::LShr, t, right);
        if unused != 2 {
            return field;
        }
        let old = match inst.read(reg).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let keep = self.s.m.i32(!((((1u64 << width) - 1) as u32) << at));
        let old = self.s.m.bin(c, Bin::And, old, keep);
        self.s.m.bin(c, Bin::Or, old, field)
    }

    fn write_f(&mut self, inst: &Inst, f: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let mut f = f;
        match v.omod {
            0 => {}
            k => {
                let factor = [1.0f32, 2.0, 4.0, 0.5][usize::from(k)];
                let c = self.s.m.f32(factor.to_bits());
                f = self.s.m.bin(self.cur, Bin::Mul, f, c);
            }
        }
        if v.clamp {
            f = self.unary(7, f);
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a float result outside a VGPR");
        };
        let x = self.as_int(f);
        self.write(inst, Reg::V(reg), x, guard);
        if guard.is_none()
            && Self::sdwa_dst(inst).is_none()
            && let Some(new) = inst.write(Reg::V(reg))
        {
            self.floats.insert(new, f);
        }
        Ok(())
    }

    fn write_i(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.clamp || v.omod != 0 {
            return Err("integer output modifiers");
        }
        match v.dst {
            Some(Operand::Vector { reg, .. }) => self.write(inst, Reg::V(reg), x, guard),
            Some(Operand::Scalar { code, .. }) => self.write(inst, Reg::S(code), x, None),
            _ => return Err("a result without a register"),
        }
        Ok(())
    }

    /// Starts code only the lanes where `guard` holds run (a store under a
    /// lane guard): returns where it started and the join block.
    fn begin_guarded(&mut self, guard: Option<V>) -> Option<(usize, usize)> {
        let g = guard?;
        let (then, join) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(self.cur, Term::CondBr(g, then, join));
        let head = self.cur;
        self.cur = then;
        Some((head, join))
    }

    /// Ends guarded code; each (value inside, value outside) becomes a phi.
    fn end_guarded(&mut self, at: Option<(usize, usize)>, values: &[(V, V)]) -> Vec<V> {
        let Some((head, join)) = at else {
            return values.iter().map(|v| v.0).collect();
        };
        let then = self.cur;
        self.s.m.terminate(then, Term::Br(join));
        self.cur = join;
        values
            .iter()
            .map(|&(inside, outside)| {
                let ty = self.s.m.type_of(inside);
                let phi = self.s.m.phi(join, ty);
                self.s.m.set_phi(phi, vec![(inside, then), (outside, head)]);
                phi
            })
            .collect()
    }

    /// Code that depends on the lane layout: wave operations, and in compute
    /// the console's wave size (lanes are thread ids in group order there).
    /// Pixel lanes match only within a quad.
    fn cross_lane(&mut self, quad: bool) -> Result<(), Why> {
        if self.s.mesh.is_some() && !self.s.fallback {
            // shortcut: no wave size for mesh shaders in SM 6.6, so host
            // lanes cannot follow the console's; emulate through group-shared
            // memory if mesh programs turn out to need these.
            return Err("a cross-lane operation in a mesh shader");
        }
        if self.s.threads.is_none() && !(quad && self.s.pixel) {
            return Err("a cross-lane operation outside compute");
        }
        debug_assert!(self.s.threads.is_none() || self.console_lanes);
        self.s.flags |= flags::WAVE_OPS;
        Ok(())
    }

    /// Component `c` of the thread id in the group. With console lanes, the
    /// thread is relabelled so that its host lane is its console lane: the
    /// flat id is the wave's first lane's, rounded down to the wave size,
    /// plus the lane index. (A bijection while each host wave holds a
    /// wave-aligned run of flat ids, as the console's waves do.)
    fn thread_id(&mut self, c: u8) -> Val {
        if !self.console_lanes {
            return self.compute_id("threadIdInGroup.i32", 95, c);
        }
        let [x, y, _] = self.s.threads.expect("compute");
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let flat = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let first = self.s.op(
            e,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[flat],
        );
        let wave = self.s.wave_size.expect("set with console lanes");
        let mask = self.s.m.i32(!(wave - 1));
        let base = self.s.m.bin(e, Bin::And, first, mask);
        let lane = self.lane_index();
        let f = self.s.m.bin(e, Bin::Add, base, lane);
        self.s.flags |= flags::WAVE_OPS;
        let (xk, yk, xy) = (self.s.m.i32(x), self.s.m.i32(y), self.s.m.i32(x * y));
        Val::I32(match c {
            0 => self.s.m.bin(e, Bin::URem, f, xk),
            1 => {
                let t = self.s.m.bin(e, Bin::UDiv, f, xk);
                self.s.m.bin(e, Bin::URem, t, yk)
            }
            _ => self.s.m.bin(e, Bin::UDiv, f, xy),
        })
    }

    /// Mesh: the counts and the primitives no lane exports. The console's
    /// counts come from one lane's allocation request, which DXIL cannot
    /// express (one call, dominating every output write).
    // shortcut: the counts are the declared maxima and each lane first culls
    // its primitive (all indices 0, zero area), so primitives past the
    // program's count are culled and vertices past it unused; this costs
    // the host the unused vertex and primitive slots. Pass the program's own
    // counts if that shows in frame time.
    fn mesh_prologue(&mut self, mesh: super::shader::Mesh) -> Result<(), Why> {
        let threads = self.s.threads.expect("mesh")[0];
        if mesh.topology != 2 {
            return Err("a mesh shader not drawing triangles");
        }
        if threads > 128 || mesh.max_vertices > 256 || mesh.max_primitives > 256 {
            return Err("a mesh shader beyond the host's limits");
        }
        let e = self.s.entry;
        let void = self.s.t.void;
        let (v, p) = (
            self.s.m.i32(mesh.max_vertices),
            self.s.m.i32(mesh.max_primitives),
        );
        self.s
            .op(e, "setMeshOutputCounts", Attr::NoUnwind, void, 168, &[v, p]);
        let flat = self.mesh_thread();
        let guard = (threads > mesh.max_primitives).then(|| self.s.m.cmp(e, pred::IULT, flat, p));
        let at = self.begin_guarded(guard);
        let z = self.s.m.i32(0);
        self.s.op(
            self.cur,
            "emitIndices",
            Attr::NoUnwind,
            void,
            169,
            &[flat, z, z, z],
        );
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// The group id the console program sees, component `c`. Mesh
    /// fallback: the host dispatch's x plus the first group of the
    /// piece (`t0` of the record space, dword 0); the records stay indexed
    /// by the host's id. A direct dispatch of more groups than one segment
    /// holds runs in pieces.
    fn console_group(&mut self, c: u8) -> Val {
        let id = self.compute_id("groupId.i32", 94, c);
        let (true, 0, Val::I32(g)) = (self.s.fallback, c, id) else {
            return id;
        };
        let base = match self.group_base {
            Some(b) => b,
            None => {
                let cur = std::mem::replace(&mut self.cur, self.s.entry);
                let h = self.s.raw_view(false, crate::emit::mesh::RECORD_REGISTER);
                let z = self.s.m.i32(0);
                let b = self.raw_dword(h, z);
                self.cur = cur;
                self.group_base = Some(b);
                b
            }
        };
        Val::I32(self.s.m.bin(self.s.entry, Bin::Add, g, base))
    }

    /// Mesh: the thread's index in the group, which is its vertex and
    /// primitive index.
    fn mesh_thread(&mut self) -> V {
        if let Some(f) = self.flat {
            return f;
        }
        let e = self.s.entry;
        let f = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            self.s.t.i32,
            96,
            &[],
        );
        self.flat = Some(f);
        f
    }

    /// The console wave size.
    fn wave(&self) -> u32 {
        if self.x.native.wave64 { 64 } else { 32 }
    }

    fn lane_index(&mut self) -> V {
        if self.s.mesh.is_some() {
            // The console's lane: mesh code reads no other lanes, so host
            // waves need not match the console's.
            let f = self.mesh_thread();
            let m = self.s.m.i32(self.wave() - 1);
            return self.s.m.bin(self.s.entry, Bin::And, f, m);
        }
        self.s.op(
            self.cur,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        )
    }

    /// `x` of lane `src` (in this lane's quad) through quad reads, whose
    /// lane must be a constant.
    fn quad_pick(&mut self, x: V, src: V) -> V {
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        let three = self.s.m.i32(3);
        let q = self.s.m.bin(c, Bin::And, src, three);
        let mut r = x;
        for n in 0..4 {
            let kn = self.s.m.i32(n);
            let read = self.s.op(
                c,
                "quadReadLaneAt.i32",
                Attr::NoUnwind,
                self.s.t.i32,
                122,
                &[x, kn],
            );
            let is = self.s.m.cmp(c, pred::EQ, q, kn);
            r = self.s.m.select(c, is, read, r);
        }
        r
    }

    /// Whether a read of another lane of this lane's quad must reach helper
    /// lanes: in a pixel shader, always. At a triangle's edge the quad's
    /// other pixels are helpers (and so are discarded ones), which only quad
    /// reads see: wave reads give 0 from them on AMD, so a derivative taken
    /// by hand (`quad_perm`) was the whole value there, the texture read
    /// that used it took its smallest level, and the triangles' edges showed
    /// as lines.
    fn helpers_read(&self, quad: bool) -> bool {
        quad && self.s.pixel
    }

    /// `v_readfirstlane_b32`'s result `first` of `own` in a pixel shader:
    /// a helper lane takes its own value instead.
    ///
    /// On the console, whole-quad mode puts helper lanes in EXEC, so a
    /// waterfall loop (read the first lane's value, run the lanes with that
    /// value, take them out of EXEC) serves helper lanes too. On the host,
    /// wave operations leave helper lanes out: once only helper lanes are
    /// left in such a loop, `WaveReadLaneFirst` reads no lane, no lane
    /// matches, and the loop runs to its limit in every such lane, with
    /// whatever descriptor the read gave. With its own value a helper lane
    /// matches in the first pass and leaves; what it computes only feeds
    /// derivatives (those of a texture read's dependent coordinates, at
    /// triangle edges), and its stores are dropped.
    fn helper_reads_own(&mut self, own: V, first: V) -> V {
        if !self.s.pixel {
            return first;
        }
        let c = self.cur;
        let helper = self
            .s
            .op(c, "isHelperLane.i1", Attr::ReadOnly, self.s.t.i1, 221, &[]);
        self.s.m.select(c, helper, own, first)
    }

    fn read_lane(&mut self, x: V, lane: V) -> V {
        self.s.op(
            self.cur,
            "waveReadLaneAt.i32",
            Attr::NoUnwind,
            self.s.t.i32,
            117,
            &[x, lane],
        )
    }

    /// Coarse derivatives of `coords` across this lane's quad, as DXIL
    /// gradients (x for up to three coordinates, then y), each times
    /// `scale` if given.
    fn quad_gradients(&mut self, coords: &[V], scale: Option<V>) -> Vec<V> {
        let c = self.cur;
        let lane = self.lane_index();
        let top = self.s.m.i32(!3);
        let base = self.s.m.bin(c, Bin::And, lane, top);
        let lanes: Vec<V> = (1..3u32)
            .map(|k| {
                let k = self.s.m.i32(k);
                self.s.m.bin(c, Bin::Or, base, k)
            })
            .collect();
        let fu = self.s.m.undef(self.s.t.f32);
        let mut g = vec![fu; 6];
        for (k, &x) in coords.iter().enumerate().take(3) {
            let xi = self.as_int(x);
            let at = |l: &mut Self, lane: V| {
                let r = l.read_lane(xi, lane);
                l.as_float(r)
            };
            let origin = at(self, base);
            for (d, &lane) in lanes.iter().enumerate() {
                let there = at(self, lane);
                let mut diff = self.s.m.bin(c, Bin::Sub, there, origin);
                if let Some(s) = scale {
                    diff = self.s.m.bin(c, Bin::Mul, diff, s);
                }
                g[3 * d + k] = diff;
            }
        }
        g
    }

    /// The float value of a register value.
    fn float_of(&mut self, v: Value) -> Result<V, Why> {
        if let Some(&f) = self.floats.get(&v) {
            return Ok(f);
        }
        let x = self.i32_of(v)?;
        Ok(self.as_float(x))
    }

    /// The values of VGPRs `reg..reg + n` as `inst` reads them.
    fn vgprs(&mut self, inst: &Inst, op: &Operand, n: u8) -> Result<Vec<V>, Why> {
        let Operand::Vector { reg, .. } = *op else {
            return Err("a vector operand");
        };
        (0..n)
            .map(|k| {
                let v = inst.read(Reg::V(reg + k)).ok_or("an unread operand")?;
                self.i32_of(v)
            })
            .collect()
    }

    /// LDS (thread-group shared memory) and `ds_swizzle`.
    fn ds(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Ds {
            dst,
            addr,
            data,
            offset0,
            offset1,
            gds,
        } = &inst.fields
        else {
            unreachable!()
        };
        if *gds {
            return Err("GDS");
        }
        let name = inst.name();
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let offset = u32::from(*offset0) | u32::from(*offset1) << 8;
        let dst_reg = match dst {
            Some(Operand::Vector { reg, .. }) => Some(*reg),
            _ => None,
        };
        if name == "ds_swizzle_b32" {
            // From 0xE000: an FFT order; from 0xC000: a rotation. Below, bit
            // 15 set: each lane of a quad picks a lane of it; else lanes of
            // each group of 32 combine their index with and/or/xor masks.
            let (fft, rotate) = (offset >= 0xE000, (0xC000..0xE000).contains(&offset));
            let quad = !(fft || rotate)
                && (offset & 0x8000 != 0
                    || offset & 0x1C == 0x1C && offset & 0x380 == 0 && offset >> 10 & 0x1F < 4);
            // Mesh: host quads are threads 4n..4n+3 of a 1D group, as the
            // console's, so quad-local swizzles read through quad operations.
            let mesh_quad = self.s.mesh.is_some() && quad;
            if !mesh_quad {
                self.cross_lane(quad)?;
            }
            // The source sits in the address field.
            let x = self.vgprs(inst, addr.as_ref().ok_or("a swizzle source")?, 1)?[0];
            let lane = self.lane_index();
            let k = |l: &mut Self, v: u32| l.s.m.i32(v);
            let src = if fft || rotate {
                // Both stay within the lane's group of 32.
                let (low, high) = (k(self, 0x1F), k(self, !0x1F));
                let mask = offset & 0x1F;
                let i = self.s.m.bin(c, Bin::And, lane, low);
                let j = if fft {
                    // The 5 index bits reversed, shifted down by the number
                    // of mask bits; the mask's bits stay the lane's own.
                    let (top, m, i32t) = (k(self, 27), k(self, mask), self.s.t.i32);
                    let down = k(self, mask.count_ones());
                    let j = self.s.op(c, "unary.i32", Attr::ReadNone, i32t, 30, &[i]);
                    let j = self.s.m.bin(c, Bin::LShr, j, top);
                    let j = self.s.m.bin(c, Bin::LShr, j, down);
                    let own = self.s.m.bin(c, Bin::And, i, m);
                    self.s.m.bin(c, Bin::Or, j, own)
                } else {
                    // Left by bits 5-9 lanes (bit 10: right); the mask's
                    // bits stay the lane's own. This is the ISA's
                    // the code, which RDNA2 hardware follows; its
                    // examples for mask 1 list other lanes.
                    let n = offset >> 5 & 0x1F;
                    let n = if offset & 0x400 != 0 {
                        n.wrapping_neg()
                    } else {
                        n
                    };
                    let (n, m, rest) = (k(self, n), k(self, mask), k(self, !mask & 0x1F));
                    let j = self.s.m.bin(c, Bin::Add, i, n);
                    let j = self.s.m.bin(c, Bin::And, j, rest);
                    let own = self.s.m.bin(c, Bin::And, i, m);
                    self.s.m.bin(c, Bin::Or, j, own)
                };
                let group = self.s.m.bin(c, Bin::And, lane, high);
                self.s.m.bin(c, Bin::Or, group, j)
            } else if offset & 0x8000 != 0 {
                let (three, one) = (k(self, 3), k(self, 1));
                let sel = self.s.m.bin(c, Bin::And, lane, three);
                let sh = self.s.m.bin(c, Bin::Shl, sel, one);
                let pattern = k(self, offset & 0xFF);
                let pick = self.s.m.bin(c, Bin::LShr, pattern, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, three);
                let not3 = k(self, !3);
                let base = self.s.m.bin(c, Bin::And, lane, not3);
                self.s.m.bin(c, Bin::Or, base, pick)
            } else {
                let and = k(self, offset & 0x1F | !0x1F);
                let or = k(self, offset >> 5 & 0x1F);
                let xor = k(self, offset >> 10 & 0x1F);
                let t = self.s.m.bin(c, Bin::And, lane, and);
                let t = self.s.m.bin(c, Bin::Or, t, or);
                self.s.m.bin(c, Bin::Xor, t, xor)
            };
            let r = if mesh_quad || self.helpers_read(quad) {
                self.quad_pick(x, src)
            } else {
                self.read_lane(x, src)
            };
            let reg = dst_reg.ok_or("a swizzle result")?;
            self.write(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        if self.s.lds_dwords == 0 {
            return Err("LDS the program does not allocate");
        }
        let a = self.vgprs(inst, addr.as_ref().ok_or("an LDS address")?, 1)?[0];
        // shortcut: LDS addresses are taken as dword-aligned and in bounds
        // (out-of-bounds console accesses read 0 and drop writes).
        let dword = |l: &mut Self, bytes: u32| -> V {
            let c = l.cur;
            let o = l.s.m.i32(bytes);
            let at = l.s.m.bin(c, Bin::Add, a, o);
            let two = l.s.m.i32(2);
            let index = l.s.m.bin(c, Bin::LShr, at, two);
            let lds = l.s.lds();
            l.s.m.element(c, lds, index)
        };
        let width = |n: &str| match n.rsplit('_').next() {
            Some("b64") => 2,
            Some("b96") => 3,
            Some("b128") => 4,
            _ => 1,
        };
        // (byte offset, dwords) per element; read2/write2 scale their two
        // offsets by the element size (×64 for st64).
        let pair = |n: &str| -> Vec<(u32, u32)> {
            let w = width(n);
            let scale = 4 * w * if n.contains("st64") { 64 } else { 1 };
            vec![
                (u32::from(*offset0) * scale, w),
                (u32::from(*offset1) * scale, w),
            ]
        };
        // XNW: a backward permute: each lane reads data0 of the lane its
        // address names (byte address over 4, within the wave); no LDS.
        if name == "ds_bpermute_b32" {
            let lanes = if self.x.native.wave64 { 64 } else { 32 };
            let x = self.vgprs(inst, data.first().ok_or("a permute's data")?, 1)?[0];
            let (o, two, m) = (self.s.m.i32(offset), self.s.m.i32(2), self.s.m.i32(lanes - 1));
            let at = self.s.m.bin(c, Bin::Add, a, o);
            let lane = self.s.m.bin(c, Bin::LShr, at, two);
            let lane = self.s.m.bin(c, Bin::And, lane, m);
            let r = self.read_lane(x, lane);
            let reg = dst_reg.ok_or("a permute result")?;
            self.write(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        // VIctory (b6af8b1, GPL-3.0-or-later), XNW: bytes and
        // halves: (bits, signed) of `ds_read_{u,i}{8,16}` and
        // `ds_write_b{8,16}`, read and written within their dword.
        // shortcut: a half at byte 3 of a dword (crossing into the next) is
        // taken as within it.
        // `_d16`: the value is a 16-bit half of the VGPR (`_hi`: the high
        // one); a read keeps the other half.
        let base = name.trim_end_matches("_hi").trim_end_matches("_d16");
        let (d16, hi) = (name.contains("_d16"), name.ends_with("_hi"));
        let narrow = match base {
            "ds_read_u8" | "ds_write_b8" => Some((8u32, false)),
            "ds_read_i8" => Some((8, true)),
            "ds_read_u16" | "ds_write_b16" => Some((16, false)),
            "ds_read_i16" => Some((16, true)),
            _ => None,
        };
        if let Some((bits, signed)) = narrow {
            let k = |l: &mut Self, v: u32| l.s.m.i32(v);
            let (o, three) = (k(self, offset), k(self, 3));
            let at = self.s.m.bin(c, Bin::Add, a, o);
            let byte = self.s.m.bin(c, Bin::And, at, three);
            let shift = self.s.m.bin(c, Bin::Shl, byte, three);
            let p = dword(self, offset);
            let mask = k(self, (1u32 << bits) - 1);
            if name.starts_with("ds_read") {
                let reg = dst_reg.ok_or("an LDS read result")?;
                let w = self.s.m.load(c, p);
                let x = self.s.m.bin(c, Bin::LShr, w, shift);
                let x = if signed {
                    // Up to the top, then arithmetically back down.
                    let up = k(self, 32 - bits);
                    let t = self.s.m.bin(c, Bin::Shl, x, up);
                    self.s.m.bin(c, Bin::AShr, t, up)
                } else {
                    self.s.m.bin(c, Bin::And, x, mask)
                };
                let x = if !d16 {
                    x
                } else if !hi {
                    self.low_half(inst, reg, x)
                } else {
                    let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
                        Some(Ok(o)) => o,
                        _ => k(self, 0),
                    };
                    let (low, sixteen) = (k(self, 0xFFFF), k(self, 16));
                    let old = self.s.m.bin(c, Bin::And, old, low);
                    let x = self.s.m.bin(c, Bin::Shl, x, sixteen);
                    self.s.m.bin(c, Bin::Or, old, x)
                };
                self.write(inst, Reg::V(reg), x, guard);
            } else {
                let mut v = self.vgprs(inst, data.first().ok_or("an LDS write")?, 1)?[0];
                if hi {
                    let sixteen = k(self, 16);
                    v = self.s.m.bin(c, Bin::LShr, v, sixteen);
                }
                let v = self.s.m.bin(c, Bin::And, v, mask);
                let v = self.s.m.bin(c, Bin::Shl, v, shift);
                let hole = self.s.m.bin(c, Bin::Shl, mask, shift);
                let all = k(self, u32::MAX);
                let keep = self.s.m.bin(c, Bin::Xor, hole, all);
                // Two atomics: lanes writing other bytes of the same dword
                // keep theirs.
                use super::bitcode::Rmw;
                let at = self.begin_guarded(guard);
                self.s.m.atomic(self.cur, Rmw::And, p, keep);
                self.s.m.atomic(self.cur, Rmw::Or, p, v);
                self.end_guarded(at, &[]);
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("ds_read") {
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                return Err("an LDS read of bytes or halves");
            };
            let reg = dst_reg.ok_or("an LDS read result")?;
            let mut n = 0;
            for (off, w) in elements {
                for k in 0..w {
                    let p = dword(self, off + 4 * k);
                    let x = self.s.m.load(self.cur, p);
                    self.write(inst, Reg::V(reg + n), x, guard);
                    n += 1;
                }
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("ds_write") {
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                return Err("an LDS write of bytes or halves");
            };
            let mut stores = Vec::new();
            for (e, (off, w)) in elements.into_iter().enumerate() {
                let vals = self.vgprs(inst, data.get(e).ok_or("an LDS write")?, w as u8)?;
                for (k, x) in vals.into_iter().enumerate() {
                    stores.push((off + 4 * k as u32, x));
                }
            }
            let at = self.begin_guarded(guard);
            for (off, x) in stores {
                let p = dword(self, off);
                self.s.m.store(self.cur, p, x);
            }
            self.end_guarded(at, &[]);
            return Ok(());
        }
        // Atomics: the old value returns with `_rtn`.
        let op = name
            .strip_prefix("ds_")
            .map(|n| n.replace("_rtn", ""))
            .ok_or("an LDS operation")?;
        use super::bitcode::Rmw;
        let rmw = match op.as_str() {
            "add_u32" => Some(Rmw::Add),
            "sub_u32" => Some(Rmw::Sub),
            "and_b32" => Some(Rmw::And),
            "or_b32" => Some(Rmw::Or),
            "xor_b32" => Some(Rmw::Xor),
            "min_i32" => Some(Rmw::Min),
            "max_i32" => Some(Rmw::Max),
            "min_u32" => Some(Rmw::UMin),
            "max_u32" => Some(Rmw::UMax),
            "wrxchg_b32" => Some(Rmw::Xchg),
            "cmpst_b32" => None,
            _ => return Err("an LDS operation"),
        };
        let v0 = self.vgprs(inst, data.first().ok_or("an LDS atomic")?, 1)?[0];
        let v1 = match data.get(1) {
            Some(d) => Some(self.vgprs(inst, d, 1)?[0]),
            None => None,
        };
        let old_reg = dst_reg.and_then(|r| inst.read(Reg::V(r)).map(|v| (r, v)));
        let outside = match old_reg {
            Some((_, v)) => self.i32_of(v).unwrap_or(self.s.m.i32(0)),
            None => self.s.m.i32(0),
        };
        let p = dword(self, offset);
        let at = self.begin_guarded(guard);
        let r = match rmw {
            Some(op) => self.s.m.atomic(self.cur, op, p, v0),
            // Compare with data0, store data1.
            None => self
                .s
                .m
                .cmpxchg(self.cur, p, v0, v1.ok_or("an LDS compare")?),
        };
        let r = self.end_guarded(at, &[(r, outside)])[0];
        if let Some(reg) = dst_reg {
            self.write(inst, Reg::V(reg), r, None);
        }
        Ok(())
    }

    // ---- instructions ----

    fn inst(&mut self, b: usize, i: usize) -> Result<(), Stop> {
        let inst: &'a Inst = &self.x.p.blocks[b].insts[i];
        let offset = inst.machine.offset;
        let name = inst.name();
        self.mask_ok = match self.mask_data {
            MaskData::Console => true,
            MaskData::Host => {
                name.starts_with("s_bcnt1_i32_")
                    || name.starts_with("s_ff1_i32_")
                    || name.starts_with("v_mbcnt_")
                    || matches!(
                        name,
                        "s_cmp_eq_u32" | "s_cmp_lg_u32" | "s_cmp_eq_u64" | "s_cmp_lg_u64"
                    )
            }
            MaskData::None => false,
        };
        let effect = matches!(inst.fields, Fields::Exp { .. })
            || matches!(inst.fields, Fields::Flat { .. }) && !name.starts_with("global_load_")
            || name.contains("store")
            || name.contains("atomic")
            || name == "s_barrier"
            || self.s.fallback && name == "s_sendmsg"
            || matches!(inst.fields, Fields::Ds { .. })
                && !(name.starts_with("ds_read")
                    || name.starts_with("ds_swizzle")
                    || name.contains("permute"));
        let r = match &inst.fields {
            Fields::Sop { .. } => self.sop(inst),
            Fields::Sopp { .. } if name == "s_barrier" && self.no_barrier.contains(&b) => Ok(()),
            Fields::Sopp { .. } if name == "s_barrier" => {
                // Thread-group sync with a shared-memory fence, and a UAV
                // fence when the program writes UAVs (the waits before an
                // s_barrier make those writes visible on the console).
                let mode = if self.writes_uavs { 1 | 2 | 8 } else { 1 | 8 };
                let mode = self.s.m.i32(mode);
                let void = self.s.t.void;
                self.s
                    .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
                Ok(())
            }
            Fields::Sopp { imm }
                if self.s.fallback
                    && name == "s_sendmsg"
                    && imm & 0xF == GS_ALLOC_REQ =>
            {
                self.fallback_alloc(inst, b, i)
            }
            Fields::Sopp { .. } => Ok(()),
            Fields::Smem { .. } => self.smem(inst, (b, i)),
            Fields::Valu(_) if name.starts_with("v_interp_") => self.interp(inst, (b, i)),
            Fields::Valu(_) => self.valu(inst, b, i),
            Fields::Vintrp { .. } => self.interp(inst, (b, i)),
            Fields::Image { .. } if name.starts_with("image_bvh") => self.bvh(inst, b, i),
            Fields::Image { .. } => self.image(inst, b, i),
            Fields::Buffer { .. } => self.buffer(inst, b, i),
            Fields::Exp { .. } => self.export(b, i),
            Fields::Ds { .. } => self.ds(inst, b, i),
            Fields::Flat { .. } if name.starts_with("global_") => self.flat(inst, b, i),
            Fields::Flat { .. } => Err("flat memory"),
        };
        match r {
            Ok(()) => Ok(()),
            Err(why) if effect => Err(unsupported(offset, why)),
            Err(why) => {
                for &(_, v) in &inst.writes {
                    self.vals.insert(v, Val::Missing(why));
                }
                Ok(())
            }
        }
    }

    fn is_mask(&self, v: Option<Value>) -> bool {
        v.is_some_and(|v| self.x.st.masks.contains_key(&v))
    }

    fn sop(&mut self, inst: &Inst) -> Result<(), Why> {
        let Fields::Sop { dst, src, imm } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let scc = inst.write(Reg::Scc);
        let dst_code = match dst {
            Some(Operand::Scalar { code, .. }) => Some(*code),
            _ => None,
        };
        let lo = dst_code.and_then(|c| inst.write(Reg::S(c)));
        if inst.write(Reg::S(EXEC)).is_some() || self.is_mask(lo) {
            return Err(LANE_MASK);
        }
        if matches!(
            name,
            "s_setreg_imm32_b32" | "s_setreg_b32" | "s_swappc_b64" | "s_version"
        ) || name.starts_with("s_waitcnt")
        {
            // shortcut: the float MODE register is not modelled; s_swappc_b64 is the
            // vertex fetch call, whose results arrive as inputs.
            return Err("a register the emitter does not model");
        }
        let c = self.cur;
        let src32 = |l: &mut Self, k: usize| -> Result<V, Why> {
            // SOPK arithmetic reads its destination as the first source.
            if k == 0 && matches!(name, "s_addk_i32" | "s_mulk_i32") {
                let d = dst.as_ref().ok_or("a missing operand")?;
                return l.operand(inst, d);
            }
            let op = src.get(k).ok_or("a missing operand")?;
            l.operand(inst, op)
        };
        let sext16 = |l: &mut Self| l.s.m.i32(i32::from(imm.unwrap_or(0) as i16) as u32);
        // Dword `h` of 64-bit source `k` (inline constants sign-extend).
        let half = |l: &mut Self, k: usize, h: u8| -> Result<V, Why> {
            match src.get(k) {
                Some(&Operand::Scalar { code, .. }) if code != NULL => {
                    let v = inst.read(Reg::S(code + h)).ok_or("an unread operand")?;
                    l.i32_of(v)
                }
                Some(op) if h == 0 => l.operand(inst, op),
                Some(Operand::Const(k)) => {
                    let x = crate::resource::inline(*k).ok_or("a special operand")?;
                    Ok(l.s.m.i32(if (x as i32) < 0 { u32::MAX } else { 0 }))
                }
                _ => Ok(l.s.m.i32(0)),
            }
        };
        if let Some(op) = name.strip_prefix("s_bitcmp") {
            // SCC is whether bit src1 (mod 32, or mod 64 for the 64-bit form)
            // of src0 is clear (bitcmp0) or set (bitcmp1).
            let (bit, k31) = (src32(self, 1)?, self.s.m.i32(31));
            let idx = self.s.m.bin(c, Bin::And, bit, k31);
            let word = if op.ends_with("_b64") {
                let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                let k32 = self.s.m.i32(32);
                let upper = self.s.m.bin(c, Bin::And, bit, k32);
                let z = self.s.m.i32(0);
                let upper = self.s.m.cmp(c, pred::NE, upper, z);
                self.s.m.select(c, upper, b, a)
            } else {
                src32(self, 0)?
            };
            let shifted = self.s.m.bin(c, Bin::LShr, word, idx);
            let one = self.s.m.i32(1);
            let tested = self.s.m.bin(c, Bin::And, shifted, one);
            let want = self.s.m.i32(u32::from(op.starts_with('1')));
            let r = self.s.m.cmp(c, pred::EQ, tested, want);
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").filter(|o| o.ends_with("_u64")) {
            // eq and lg only: both dwords.
            let mut r = None;
            for h in 0..2 {
                let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                let e = self.s.m.cmp(c, pred::EQ, a, b);
                r = Some(match r {
                    Some(x) => self.s.m.bin(c, Bin::And, x, e),
                    None => e,
                });
            }
            let mut r = r.expect("two dwords");
            if op.starts_with("lg") {
                let t = self.s.m.bool(true);
                r = self.s.m.bin(c, Bin::Xor, r, t);
            }
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").or(name.strip_prefix("s_cmpk_")) {
            let (a, b) = if name.starts_with("s_cmpk_") {
                let b = if op.ends_with("_i32") {
                    sext16(self)
                } else {
                    self.s.m.i32(u32::from(imm.unwrap_or(0)))
                };
                (src32(self, 0)?, b)
            } else {
                if op.ends_with("64") {
                    return Err("a 64-bit scalar compare");
                }
                (src32(self, 0)?, src32(self, 1)?)
            };
            let signed = op.ends_with("_i32");
            let p = match &op[..2] {
                "eq" => pred::EQ,
                "lg" => pred::NE,
                "gt" if signed => pred::SGT,
                "ge" if signed => pred::SGE,
                "lt" if signed => pred::SLT,
                "le" if signed => pred::SLE,
                "gt" => pred::IUGT,
                "ge" => pred::IUGE,
                "lt" => pred::IULT,
                "le" => pred::IULE,
                _ => return Err("a scalar compare"),
            };
            let r = self.s.m.cmp(c, p, a, b);
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        let wide = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
        if wide {
            let hi = dst_code.and_then(|c| inst.write(Reg::S(c + 1)));
            let nonzero = |l: &mut Self, a: V, b: V| {
                let o = l.s.m.bin(c, Bin::Or, a, b);
                let z = l.s.m.i32(0);
                l.s.m.cmp(c, pred::NE, o, z)
            };
            return match name {
                "s_bcnt1_i32_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let i32t = self.s.t.i32;
                    let na = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[a]);
                    let nb = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[b]);
                    let r = self.s.m.bin(c, Bin::Add, na, nb);
                    self.set(lo, Val::I32(r));
                    let z = self.s.m.i32(0);
                    let s = self.s.m.cmp(c, pred::NE, r, z);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_ff1_i32_b64" => {
                    // The lowest set bit; -1 for none (as DXIL's first bit low).
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let i32t = self.s.t.i32;
                    let fa = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[a]);
                    let fb = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[b]);
                    let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
                    let fb32 = self.s.m.bin(c, Bin::Add, fb, k32);
                    let none = self.s.m.cmp(c, pred::EQ, b, z);
                    let fb = self.s.m.select(c, none, fb, fb32);
                    let low = self.s.m.cmp(c, pred::NE, a, z);
                    let r = self.s.m.select(c, low, fa, fb);
                    self.set(lo, Val::I32(r));
                    Ok(())
                }
                "s_and_b64" | "s_or_b64" | "s_xor_b64" | "s_andn2_b64" | "s_orn2_b64"
                | "s_nand_b64" | "s_nor_b64" | "s_xnor_b64" | "s_not_b64" => {
                    let ones = self.s.m.i32(u32::MAX);
                    let mut r = Vec::new();
                    for h in 0..2u8 {
                        let a = half(self, 0, h)?;
                        let x = if name == "s_not_b64" {
                            self.s.m.bin(c, Bin::Xor, a, ones)
                        } else {
                            let mut b = half(self, 1, h)?;
                            if name.contains("n2") {
                                b = self.s.m.bin(c, Bin::Xor, b, ones);
                            }
                            let op = match &name[2..5] {
                                "and" | "nan" => Bin::And,
                                "or_" | "orn" | "nor" => Bin::Or,
                                _ => Bin::Xor,
                            };
                            let x = self.s.m.bin(c, op, a, b);
                            if matches!(name, "s_nand_b64" | "s_nor_b64" | "s_xnor_b64") {
                                self.s.m.bin(c, Bin::Xor, x, ones)
                            } else {
                                x
                            }
                        };
                        r.push(x);
                    }
                    self.set(lo, Val::I32(r[0]));
                    self.set(hi, Val::I32(r[1]));
                    let s = nonzero(self, r[0], r[1]);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let n = src32(self, 1)?;
                    let i64t = self.s.m.int(64);
                    self.s.flags |= flags::INT64;
                    let a = self.s.m.cast(c, Cast::ZExt, a, i64t);
                    let b = self.s.m.cast(c, Cast::ZExt, b, i64t);
                    let k32 = self.s.m.int_const(i64t, 32);
                    let b = self.s.m.bin(c, Bin::Shl, b, k32);
                    let x = self.s.m.bin(c, Bin::Or, a, b);
                    let k63 = self.s.m.i32(63);
                    let n = self.s.m.bin(c, Bin::And, n, k63);
                    let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
                    let op = match name {
                        "s_lshl_b64" => Bin::Shl,
                        "s_lshr_b64" => Bin::LShr,
                        _ => Bin::AShr,
                    };
                    let r = self.s.m.bin(c, op, x, n);
                    let rl = self.s.m.cast(c, Cast::Trunc, r, self.s.t.i32);
                    let rh = self.s.m.bin(c, Bin::LShr, r, k32);
                    let rh = self.s.m.cast(c, Cast::Trunc, rh, self.s.t.i32);
                    self.set(lo, Val::I32(rl));
                    self.set(hi, Val::I32(rh));
                    let s = nonzero(self, rl, rh);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                // XNW: the program counter as a code address: the next
                // instruction's byte offset, high dword [`CODE_TAG`]. Only
                // addresses the tracer could not resolve read it (a phi of
                // two code tables); title-memory reads see the tag.
                "s_getpc_b64" => {
                    let a = self.s.m.i32(inst.machine.offset as u32 + 4);
                    let b = self.s.m.i32(CODE_TAG);
                    self.set(lo, Val::I32(a));
                    self.set(hi, Val::I32(b));
                    Ok(())
                }
                "s_mov_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    self.set(lo, Val::I32(a));
                    self.set(hi, Val::I32(b));
                    Ok(())
                }
                "s_cselect_b64" => {
                    let cond = self.scc_in(inst)?;
                    for (h, v) in [(0, lo), (1, hi)] {
                        let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                        let r = self.s.m.select(c, cond, a, b);
                        self.set(v, Val::I32(r));
                    }
                    Ok(())
                }
                _ => Err(name),
            };
        }
        let m = &mut self.s.m;
        let _ = m;
        let (r, carry): (V, Option<V>) = match name {
            "s_mov_b32" => (src32(self, 0)?, None),
            "s_movk_i32" => (sext16(self), None),
            "s_cselect_b32" => {
                let cond = self.scc_in(inst)?;
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.s.m.select(c, cond, a, b), None)
            }
            "s_brev_b32" => {
                let a = src32(self, 0)?;
                let r = self
                    .s
                    .op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a]);
                (r, None)
            }
            "s_not_b32" => {
                let a = src32(self, 0)?;
                let ones = self.s.m.i32(u32::MAX);
                (self.s.m.bin(c, Bin::Xor, a, ones), None)
            }
            "s_add_u32" | "s_add_i32" | "s_addk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_addk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                let r = self.s.m.bin(c, Bin::Add, a, b);
                let carry = if name == "s_add_u32" {
                    Some(self.s.m.cmp(c, pred::IULT, r, a))
                } else {
                    None
                };
                (r, carry)
            }
            "s_sub_u32" | "s_sub_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let r = self.s.m.bin(c, Bin::Sub, a, b);
                let carry = (name == "s_sub_u32").then(|| self.s.m.cmp(c, pred::IULT, a, b));
                (r, carry)
            }
            "s_addc_u32" | "s_subb_u32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let cin = self.scc_in(inst)?;
                let (r, carry) = self.add_sub_carry(c, name == "s_addc_u32", a, b, Some(cin));
                (r, Some(carry))
            }
            "s_mul_i32" | "s_mulk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_mulk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                (self.s.m.bin(c, Bin::Mul, a, b), None)
            }
            "s_mul_hi_u32" | "s_mul_hi_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.mul_hi(a, b, name.ends_with("i32")), None)
            }
            "s_flbit_i32_b32" | "s_flbit_i32" => {
                // The first bit from the top (differing from the sign for
                // the signed form), -1 for none: DXIL's first-bit-high.
                let a = src32(self, 0)?;
                let op = if name == "s_flbit_i32_b32" { 33 } else { 34 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                (r, None)
            }
            "s_ff1_i32_b32" | "s_bcnt1_i32_b32" => {
                // First set bit (-1 for zero, as DXIL's), or the bit count.
                let a = src32(self, 0)?;
                let op = if name == "s_ff1_i32_b32" { 32 } else { 31 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                (r, None)
            }
            "s_bfe_u32" | "s_bfe_i32" => {
                // src1: offset in bits 0-4, width in bits 16-22.
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let (k16, k127) = (self.s.m.i32(16), self.s.m.i32(0x7F));
                let w = self.s.m.bin(c, Bin::LShr, b, k16);
                let w = self.s.m.bin(c, Bin::And, w, k127);
                (self.bfe(a, b, w, name == "s_bfe_i32"), None)
            }
            "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_andn2_b32" | "s_orn2_b32"
            | "s_nand_b32" | "s_nor_b32" | "s_xnor_b32" => {
                let (a, mut b) = (src32(self, 0)?, src32(self, 1)?);
                let ones = self.s.m.i32(u32::MAX);
                if name.contains("n2") {
                    b = self.s.m.bin(c, Bin::Xor, b, ones);
                }
                let op = match &name[2..5] {
                    "and" | "nan" => Bin::And,
                    "or_" | "orn" | "nor" => Bin::Or,
                    _ => Bin::Xor,
                };
                let mut r = self.s.m.bin(c, op, a, b);
                if matches!(name, "s_nand_b32" | "s_nor_b32" | "s_xnor_b32") {
                    r = self.s.m.bin(c, Bin::Xor, r, ones);
                }
                (r, None)
            }
            "s_lshl_b32" | "s_lshr_b32" | "s_ashr_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let op = match name {
                    "s_lshl_b32" => Bin::Shl,
                    "s_lshr_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                (self.s.m.bin(c, op, a, b), None)
            }
            "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k = self.s.m.i32(u32::from(name.as_bytes()[6] - b'0'));
                let t = self.s.m.bin(c, Bin::Shl, a, k);
                (self.s.m.bin(c, Bin::Add, t, b), None)
            }
            "s_min_u32" | "s_max_u32" | "s_min_i32" | "s_max_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let p = match name {
                    "s_min_u32" => pred::IULT,
                    "s_max_u32" => pred::IUGT,
                    "s_min_i32" => pred::SLT,
                    _ => pred::SGT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                (self.s.m.select(c, t, a, b), None)
            }
            "s_abs_i32" => {
                let a = src32(self, 0)?;
                let z = self.s.m.i32(0);
                let neg = self.s.m.cmp(c, pred::SLT, a, z);
                let n = self.s.m.bin(c, Bin::Sub, z, a);
                (self.s.m.select(c, neg, n, a), None)
            }
            "s_bitset0_b32" | "s_bitset1_b32" => {
                // Sets (clears) bit src[4:0] of the destination it reads.
                let old = self.operand(inst, dst.as_ref().ok_or("a missing operand")?)?;
                let a = src32(self, 0)?;
                let (k31, one) = (self.s.m.i32(31), self.s.m.i32(1));
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let bit = self.s.m.bin(c, Bin::Shl, one, a);
                if name == "s_bitset1_b32" {
                    (self.s.m.bin(c, Bin::Or, old, bit), None)
                } else {
                    let ones = self.s.m.i32(u32::MAX);
                    let nb = self.s.m.bin(c, Bin::Xor, bit, ones);
                    (self.s.m.bin(c, Bin::And, old, nb), None)
                }
            }
            "s_bfm_b32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let one = self.s.m.i32(1);
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                (self.s.m.bin(c, Bin::Shl, m, b), None)
            }
            _ => return Err(name),
        };
        self.set(lo, Val::I32(r));
        if let Some(scc) = scc {
            let val = match carry {
                Some(k) => Val::Bit(k),
                None if matches!(
                    name,
                    "s_mov_b32"
                        | "s_movk_i32"
                        | "s_cselect_b32"
                        | "s_brev_b32"
                        | "s_mul_i32"
                        | "s_mulk_i32"
                ) =>
                {
                    return Ok(());
                }
                None if name.starts_with("s_add") || name.starts_with("s_sub") => {
                    Val::Missing("the carry of a scalar operation")
                }
                None => {
                    let z = self.s.m.i32(0);
                    Val::Bit(self.s.m.cmp(c, pred::NE, r, z))
                }
            };
            self.vals.insert(scc, val);
        }
        Ok(())
    }

    /// `a + b (+ cin)` or `a - b (- cin)` with the carry (borrow) out, as the
    /// scalar and vector add-with-carry instructions set it.
    fn add_sub_carry(&mut self, c: usize, add: bool, a: V, b: V, cin: Option<V>) -> (V, V) {
        let op = if add { Bin::Add } else { Bin::Sub };
        let mut r = self.s.m.bin(c, op, a, b);
        // Without a carry in: a + b wraps below a; a - b borrows when a < b.
        let mut carry = if add {
            self.s.m.cmp(c, pred::IULT, r, a)
        } else {
            self.s.m.cmp(c, pred::IULT, a, b)
        };
        if let Some(cin) = cin {
            let one = self.s.m.cast(c, Cast::ZExt, cin, self.s.t.i32);
            let t = r;
            r = self.s.m.bin(c, op, t, one);
            // The carry in carries on when the partial result is all ones
            // (add) or zero (sub).
            let edge = self.s.m.i32(if add { u32::MAX } else { 0 });
            let at_edge = self.s.m.cmp(c, pred::EQ, t, edge);
            let more = self.s.m.bin(c, Bin::And, cin, at_edge);
            carry = self.s.m.bin(c, Bin::Or, carry, more);
        }
        (r, carry)
    }

    fn scc_in(&mut self, inst: &Inst) -> Result<V, Why> {
        let v = inst.read(Reg::Scc).ok_or("an unread SCC")?;
        match self.get(v)? {
            Val::Bit(x) => Ok(x),
            _ => Err("SCC without a value"),
        }
    }

    /// XNW: the user-data index of table or heap parameter `param`'s
    /// window view; its byte offset is the next dword. Two dwords per
    /// parameter right after the program's root parameters
    /// ([`super::WINDOW_DWORD`]).
    fn window_index(&mut self, param: usize) -> Result<u32, Why> {
        let n = self.windows.len() as u32;
        let base = self.window_base();
        let index = *self.windows.entry(param).or_insert(base + 2 * n);
        if index + 2 > 64 {
            return Err("more table parameters than root constants hold");
        }
        Ok(index)
    }

    /// The first user-data dword past every root parameter of the program
    /// (the window dwords go there, so a stage's root constants stay small:
    /// D3D12 allows 64 dwords for the whole root signature).
    fn window_base(&self) -> u32 {
        self.x
            .native
            .layout
            .iter()
            .filter(|r| matches!(r.root, RootType::Constants | RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap))
            .map(|r| u32::from(r.user_data) + r.sgprs.max(1))
            .max()
            .unwrap_or(0)
    }

    /// User-data dword `index` holding `dword` of parameter `param` (as
    /// [`Lower::user`], for the dwords the host adds).
    fn user_extra(&mut self, param: usize, index: u32, dword: u32) -> V {
        let rp = self.x.native.layout[param];
        self.user_reads.insert(index, UserData { index, slot: rp.slot, root: rp.root, dword });
        self.s.user_data = self.s.user_data.max(index + 1);
        let h = match self.user_data {
            Some(h) => h,
            None => {
                let h = self.s.user_data_handle();
                self.user_data = Some(h);
                h
            }
        };
        let entry = self.s.entry;
        let row = self.s.m.i32(index / 4);
        let r = self.s.op(entry, "cbufferLoadLegacy.i32", Attr::ReadOnly, self.s.t.cbuf_ret, 59, &[h, row]);
        self.s.m.extract(entry, r, index % 4)
    }

    /// XNW: a scalar load from a table or heap parameter's memory read
    /// in place: raw loads from the shared window's view the host names in
    /// user data, at the table's byte offset plus the load's.
    fn window_smem(&mut self, inst: &Inst, a: Address) -> Result<(), Why> {
        let Fields::Smem { data: Some(Operand::Scalar { code, .. }), .. } = &inst.fields else {
            return Err("a scalar load");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar load"),
        };
        let index = self.window_index(a.param)?;
        let view = self.user_extra(a.param, index, super::WINDOW_DWORD);
        let base = self.user_extra(a.param, index + 1, super::WINDOW_DWORD + 1);
        let c = self.cur;
        let h = self.s.heap_handle(c, view, false, (11, 0));
        let off = self.s.m.i32(a.offset);
        let mut byte = self.s.m.bin(c, Bin::Add, base, off);
        if let Some(d) = a.dynamic {
            let d = self.i32_of(d)?;
            byte = self.s.m.bin(c, Bin::Add, byte, d);
        }
        for k in 0..count {
            let four = self.s.m.i32(4 * k);
            let at = self.s.m.bin(c, Bin::Add, byte, four);
            let (u, mask, al) = (self.s.m.undef(self.s.t.i32), self.s.m.i8(1), self.s.m.i32(4));
            let r = self.s.op(c, "rawBufferLoad.f32", Attr::ReadOnly, self.s.t.res_ret, 139, &[h, at, u, mask, al]);
            let x = self.s.m.extract(c, r, 0);
            let x = self.as_int(x);
            self.write(inst, Reg::S(*code + k as u8), x, None);
        }
        Ok(())
    }

    /// XNW: the handle of the descriptor record the instruction at
    /// `site` reads (its resource, or with `sampler` its sampler): dword 0
    /// indexes the global heap when dword 3 is the device's tag, else the
    /// typed null of the kind `props` expects (host slots 0-15); an index
    /// past the heap reads that null too.
    fn record_handle(&mut self, site: Site, sampler: bool, props: (u32, u32)) -> Result<V, Why> {
        let idx = self.record_index(site, sampler, props)?;
        let c = self.cur;
        Ok(self.s.heap_handle(c, idx, sampler, props))
    }

    /// The global heap index [`Lower::record_handle`] takes.
    fn record_index(&mut self, site: Site, sampler: bool, props: (u32, u32)) -> Result<V, Why> {
        let inst: &Inst = &self.x.p.blocks[site.0].insts[site.1];
        let op = match &inst.fields {
            Fields::Image { rsrc, sampler: s, .. } => {
                if sampler {
                    s.as_ref().ok_or("an image without its sampler")?
                } else {
                    rsrc
                }
            }
            Fields::Buffer { rsrc, .. } => rsrc,
            Fields::Smem { base, .. } => base,
            _ => return Err("a descriptor record read by this instruction"),
        };
        let Operand::Scalar { code, .. } = *op else {
            return Err("a descriptor record not in SGPRs");
        };
        let (w0, w3) = (inst.read(Reg::S(code)), inst.read(Reg::S(code + 3)));
        let (w0, w3) = (w0.ok_or("a descriptor record")?, w3.ok_or("a descriptor record")?);
        let (mut w0, mut w3) = (self.i32_of(w0)?, self.i32_of(w3)?);
        if self.x.pagetable {
            // XNW: a page-table record (PAGE_TAG in bits 20-31 of
            // dword 3) names the host view in bits 0-19, for what still
            // reads through one (typed accesses).
            let c = self.cur;
            let (k20, m20, tag, rt) = (self.s.m.i32(20), self.s.m.i32(0xF_FFFF), self.s.m.i32(super::PAGE_TAG), self.s.m.i32(super::RECORD_TAG));
            let high = self.s.m.bin(c, Bin::LShr, w3, k20);
            let paged = self.s.m.cmp(c, pred::EQ, high, tag);
            let idx = self.s.m.bin(c, Bin::And, w3, m20);
            w0 = self.s.m.select(c, paged, idx, w0);
            w3 = self.s.m.select(c, paged, rt, w3);
        }
        let uav = props.0 & 0x1000 != 0;
        let null = if sampler {
            0
        } else {
            match props.0 & 0xff {
                13 => 1,
                11 => if uav { 6 } else { 2 },
                10 => if uav { 7 } else { 3 },
                12 => if uav { 8 } else { 4 },
                _ => if uav { 9 } else { 5 },
            }
        };
        let slots = if sampler { super::GPU_SAMPLER_SLOTS } else { super::GPU_VIEW_SLOTS };
        let c = self.cur;
        let (tag, null, last) = (self.s.m.i32(super::RECORD_TAG), self.s.m.i32(null), self.s.m.i32(slots - 1));
        let ours = self.s.m.cmp(c, pred::EQ, w3, tag);
        let inside = self.s.m.cmp(c, pred::IULE, w0, last);
        let ok = self.s.m.bin(c, Bin::And, ours, inside);
        Ok(self.s.m.select(c, ok, w0, null))
    }

    fn smem(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Fields::Smem {
            data,
            offset,
            soffset,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if let Some(a) = self.scalar.get(&site).copied().filter(|a| a.param == CODE) {
            return self.code_smem(inst, a);
        }
        if self.is_global(site) {
            return self.global_smem(inst);
        }
        if self.x.gpu_desc
            && name.starts_with("s_load_dword")
            && let Some(a) = self.scalar.get(&site).copied()
            && matches!(
                self.x.native.layout.get(a.param).map(|r| r.root),
                Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap)
            )
        {
            return self.window_smem(inst, a);
        }
        let Some(rest) = name.strip_prefix("s_buffer_load_dword") else {
            return self.descriptor_data(inst, site);
        };
        let count = match rest {
            "" => 1,
            "x2" => 2,
            "x4" => 4,
            "x8" => 8,
            "x16" => 16,
            _ => return Err("a scalar buffer load"),
        };
        let Some(Operand::Scalar { code, .. }) = *data else {
            return Err("a scalar buffer load");
        };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked descriptor")?;
        let kind = self.used_kind(src);
        let c = self.cur;
        let (base, extra) = self.built_offset(src)?;
        let offset = &((*offset as u32).wrapping_add(base) as i32);
        let mut addr = self.s.m.i32(*offset as u32);
        if let Some(x) = extra {
            addr = self.s.m.bin(c, Bin::Add, addr, x);
        }
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(inst, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        if matches!(src, Source::Built { address, .. } if address.param == CODE) {
            // A descriptor built around a table in the code.
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let x = self.code_dword(a)?;
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        let root = self.root_view_base(src);
        if !self.paged(src) && root.is_none() {
            self.note_unpaged(src, false);
        }
        if self.paged(src) || root.is_some() {
            // XNW: stage c): constants and scalar buffer reads raw at
            // base + offset through the page table, the only path (no host
            // constant buffer view, no 256-byte alignment).
            for (k, x) in self.page_load(site, root, None, addr, count, true)?.into_iter().enumerate() {
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        if let Some(u) = kind
            && matches!(u.kind, ResourceType::Srv | ResourceType::Uav)
        {
            let u_bit = if u.kind == ResourceType::Uav { 0x1000 } else { 0 };
            let structured = u.stride >= 4 && u.stride.is_multiple_of(4);
            let h = self.handle(
                site,
                src,
                false,
                if structured {
                    (12 | u_bit, u.stride)
                } else {
                    (11 | u_bit, 0)
                },
            )?;
            let st = structured.then(|| self.s.m.i32(u.stride));
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let (idx, elem_off) = match st {
                    Some(st) => (
                        self.s.m.bin(c, Bin::UDiv, a, st),
                        self.s.m.bin(c, Bin::URem, a, st),
                    ),
                    None => (a, self.s.m.undef(self.s.t.i32)),
                };
                let mask = self.s.m.i8(1);
                let al = self.s.m.i32(4);
                let r = self.s.op(
                    c,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    self.s.t.res_ret,
                    139,
                    &[h, idx, elem_off, mask, al],
                );
                let x = self.s.m.extract(c, r, 0);
                let x = self.as_int(x);
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        if self.x.gpu_desc {
            // XNW: constants at the console's 16-byte alignment read
            // through the raw view the host makes for a constant record
            // (a CBV needs 256 bytes).
            let h = self.handle(site, src, false, (11, 0))?;
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let x = self.raw_dword(h, a);
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        // victory-fork (fxceless/victory-fork 578d459, crates/victory-shader
        // src/emit/lower.rs 5344-5395, GPL-3.0-or-later): a buffer the
        // program picks from the view heap by an index it computes
        // (`ResourceHeap`) has no used-resource entry, and the heap holds a
        // buffer view there, not a constant buffer; read as one, the loads
        // gave nothing: the title's meshlet culling reads its batch records,
        // mesh headers and transforms this way, so every indirect mesh draw
        // ran empty. Changed from the fork: the slot's stride comes from our
        // runtime stride table (`stride_entry`, VIctory/704) and the
        // bytes through the raw view that table names, not a table of its
        // own. A stride of 0 (no buffer view) is still read as a constant
        // buffer; only the load that fits the slot runs.
        if kind.is_none() && self.heap_picked(src) {
            let [raw, stride, bias] = self.stride_entry(src)?;
            let zero = self.s.m.i32(0);
            let is_buffer = self.s.m.cmp(self.cur, pred::NE, stride, zero);
            let at = self.begin_guarded(Some(is_buffer));
            let h = self.s.heap_handle(self.cur, raw, false, (11, 0));
            let mut bytes = Vec::new();
            for k in 0..count {
                let b = self.cur;
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(b, Bin::Add, addr, off);
                let a = self.s.m.bin(b, Bin::Add, a, bias);
                bytes.push((self.raw_dword(h, a), zero));
            }
            let bytes = self.end_guarded(at, &bytes);
            let other = self.s.m.cmp(self.cur, pred::EQ, stride, zero);
            let at = self.begin_guarded(Some(other));
            let h = self.handle(site, src, false, (13, 0x1_0000))?;
            let mut both = Vec::new();
            for (k, &r) in bytes.iter().enumerate() {
                let b = self.cur;
                let off = self.s.m.i32(4 * k as u32);
                let a = self.s.m.bin(b, Bin::Add, addr, off);
                both.push((self.cbuffer_dword(h, a), r));
            }
            let both = self.end_guarded(at, &both);
            for (k, x) in both.into_iter().enumerate() {
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        let h = self.handle(site, src, false, (13, 0x1_0000))?;
        let constant = match addr {
            V::Const(_)
                if matches!(soffset, Operand::Scalar { code: NULL, .. }) && extra.is_none() =>
            {
                Some(*offset as u32)
            }
            _ => None,
        };
        let mut rows: HashMap<u32, V> = HashMap::new();
        for k in 0..count {
            let x = match constant {
                Some(base) => {
                    let byte = base + 4 * k;
                    let row = match rows.get(&(byte >> 4)) {
                        Some(&r) => r,
                        None => {
                            let ri = self.s.m.i32(byte >> 4);
                            let r = self.s.op(
                                c,
                                "cbufferLoadLegacy.i32",
                                Attr::ReadOnly,
                                self.s.t.cbuf_ret,
                                59,
                                &[h, ri],
                            );
                            rows.insert(byte >> 4, r);
                            r
                        }
                    };
                    self.s.m.extract(c, row, (byte >> 2) & 3)
                }
                None => {
                    let off = self.s.m.i32(4 * k);
                    let a = self.s.m.bin(c, Bin::Add, addr, off);
                    self.cbuffer_dword(h, a)
                }
            };
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// A descriptor load (`s_load_dword*` from a table or heap) whose dwords
    /// the code also reads as numbers: the console compiler computes sizes
    /// from descriptor bits (an image's width, height, depth, levels; a
    /// buffer's record count) instead of querying them. Those fields are
    /// rebuilt from the host view's dimensions in the GFX10 layout (image
    /// dword 1 bits 30-31 and dword 2 bits 0-13 width - 1, dword 2 bits
    /// 14-29 height - 1, dword 3 identity swizzle, last level at 16, type at
    /// 28, dword 4 depth or layers - 1; buffer dword 2 the record count).
    // shortcut: the base level reads 0 (the host view starts at its own
    // level 0, so sizes the code shifts by it come out right) and the other
    // dwords (addresses, formats, tiling) have no host value: reading them
    // stops the program.
    /// Two scalar loads of the same dwords: same opcode and immediate
    /// offset, base read from the same values, and the same or an
    /// equivalent register offset.
    fn twin_loads(&self, a: Site, b: Site) -> bool {
        let (ia, ib): (&Inst, &Inst) = (
            &self.x.p.blocks[a.0].insts[a.1],
            &self.x.p.blocks[b.0].insts[b.1],
        );
        let (
            Fields::Smem {
                base: Operand::Scalar { code: ba, .. },
                offset: oa,
                soffset: sa,
                ..
            },
            Fields::Smem {
                base: Operand::Scalar { code: bb, .. },
                offset: ob,
                soffset: sb,
                ..
            },
        ) = (&ia.fields, &ib.fields)
        else {
            return false;
        };
        let same_base = (0..2).all(|k| {
            let va = ia.read(Reg::S(ba + k));
            va.is_some() && va == ib.read(Reg::S(bb + k))
        });
        let same_soffset = match (sa, sb) {
            (Operand::Scalar { code: NULL, .. }, Operand::Scalar { code: NULL, .. }) => true,
            (Operand::Scalar { code: ca, .. }, Operand::Scalar { code: cb, .. }) => {
                match (ia.read(Reg::S(*ca)), ib.read(Reg::S(*cb))) {
                    (Some(x), Some(y)) => self.same_scalar(x, y),
                    _ => false,
                }
            }
            _ => false,
        };
        ia.name() == ib.name() && oa == ob && same_base && same_soffset
    }

    /// The same value, or two `v_readfirstlane_b32` of the same VGPR value
    /// (each waterfall loop reads the index anew).
    fn same_scalar(&self, x: Value, y: Value) -> bool {
        if x == y {
            return true;
        }
        let first_lane_of = |v: Value| {
            self.def_site
                .get(&v)
                .map(|&(b, i)| &self.x.p.blocks[b].insts[i])
                .filter(|i| i.name() == "v_readfirstlane_b32")
                .and_then(|i| {
                    i.reads
                        .iter()
                        .find(|r| matches!(r.0, Reg::V(_)))
                        .map(|r| r.1)
                })
        };
        matches!((first_lane_of(x), first_lane_of(y)), (Some(a), Some(b)) if a == b)
    }

    fn descriptor_data(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        const OTHER: Why = "a descriptor's address or format read as data";
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar memory read as data");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar memory read as data"),
        };
        // XNW: a sampler a waterfall loop walks. Its dwords are only copied
        // per lane, read from the first lane and compared for equality, so
        // each stands as its own byte offset in the sampler heap (16 bytes
        // an entry, as the console's): lanes compare equal when they hold
        // the same entry, and each samples with its own.
        if self.x.res.lane_samplers.contains(&site) {
            for k in 0..count {
                let x = self.heap_byte(site, k as u8)?;
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        // How the program uses the descriptor tells the view's kind. Prefer an
        // image or vector buffer use when a descriptor has both scalar and
        // vector uses, since those carry format and indexing cues.
        // The first use by an image or buffer instruction, else the first.
        let mut use_at = self.load_uses.get(&site).and_then(|&(first, rich)| rich.or(first));
        // VIctory (5ff639e): else the uses of a twin load of the
        // same dwords (a non-uniform index reloads the descriptor in each
        // waterfall loop, one for the size bits, another for the access).
        if use_at.is_none() {
            use_at = self
                .x
                .res
                .descriptors
                .iter()
                .filter(|d| d.role == Role::Resource)
                .find_map(|d| {
                    d.source.leaves().into_iter().find_map(|l| match *l {
                        Source::Loaded { load, first } if self.twin_loads(site, load) => {
                            Some((d.site, first))
                        }
                        _ => None,
                    })
                });
        }
        let (src, uav, props, image) = if let Some((at, first)) = use_at {
            let src = Source::Loaded { load: site, first };
            let used = self.used_kind(&src);
            let ui: &Inst = &self.x.p.blocks[at.0].insts[at.1];
            let un = ui.name();
            let uav = un.contains("store")
                || un.contains("atomic")
                || used.is_some_and(|u| u.kind == ResourceType::Uav);
            let u = if uav { 0x1000 } else { 0 };
            let (props, image) = match &ui.fields {
                Fields::Image { dim, .. } => {
                    let kind = match dim {
                        0 => 1,
                        1 => 2,
                        2 => 4,
                        4 => 6,
                        5 => 7,
                        _ => return Err("an image dimension (cube or MSAA)"),
                    };
                    ((kind | u, 0x409), Some(kind))
                }
                Fields::Buffer { idxen, format, .. } => {
                    let stride = used.map_or(0, |u| u.stride);
                    let props = if un.starts_with("buffer_load_format")
                        || un.starts_with("buffer_store_format")
                    {
                        (10 | u, 0x409)
                    } else if *idxen && format.is_none() && stride >= 4 && stride.is_multiple_of(4) {
                        (12 | u, stride)
                    } else {
                        (11 | u, 0)
                    };
                    if used.is_some_and(|u| u.kind == ResourceType::Cbv) {
                        return Err("a constant buffer's descriptor read as data");
                    }
                    (props, None)
                }
                Fields::Smem { .. } if un.starts_with("s_buffer_") => {
                    if used.is_some_and(|u| u.kind == ResourceType::Cbv) {
                        return Err("a constant buffer's descriptor read as data");
                    }
                    let stride = used.map_or(0, |u| u.stride);
                    let props = if stride >= 4 && stride.is_multiple_of(4) {
                        (12 | u, stride)
                    } else {
                        (11 | u, 0)
                    };
                    (props, None)
                }
                _ => return Err("a scalar memory read as data"),
            };
            (src, uav, props, image)
        } else {
            // VIctory (f800f1c): a descriptor read only for its size
            // (GetDimensions) is typed by the program's resource metadata.
            let src = Source::Loaded {
                load: site,
                first: 0,
            };
            let Some(used) = self.used_kind(&src) else {
                // VIctory: no access names it, so the console words
                // are all there is.
                let words = self.descriptor_words(site, count)?;
                for (k, x) in words.into_iter().enumerate() {
                    self.write(inst, Reg::S(code + k as u8), x, None);
                }
                return Ok(());
            };
            if used.kind == ResourceType::Cbv {
                return Err("a constant buffer's descriptor read as data");
            }
            if !matches!(used.kind, ResourceType::Srv | ResourceType::Uav) {
                return Err("a descriptor read only as data");
            }
            let uav = used.kind == ResourceType::Uav;
            let u = if uav { 0x1000 } else { 0 };
            let (props, image) = if used.dwords == 4 {
                let stride = used.stride;
                let props = if stride >= 4 && stride.is_multiple_of(4) {
                    (12 | u, stride)
                } else {
                    (11 | u, 0)
                };
                (props, None)
            } else {
                ((2 | u, 0x409), Some(2))
            };
            (src, uav, props, image)
        };
        let Source::Loaded { first, .. } = src else {
            unreachable!()
        };
        let h = self.handle(site, &src, false, props)?;
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let mip = if image.is_some() && !uav {
            self.s.m.i32(0)
        } else {
            self.s.m.undef(i32t)
        };
        let d = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
        let field = |l: &mut Self, k: u32| l.s.m.extract(c, d, k);
        let words: Vec<(u32, V)> = match image {
            Some(kind) => {
                let one = self.s.m.i32(1);
                let w = field(self, 0);
                let (height, depth) = match kind {
                    1 => (one, one),
                    2 => (field(self, 1), one),
                    6 => (one, field(self, 1)),
                    _ => (field(self, 1), field(self, 2)),
                };
                let levels = if uav { one } else { field(self, 3) };
                let less = |l: &mut Self, x: V| l.s.m.bin(c, Bin::Sub, x, one);
                let (w, height, depth, levels) = (
                    less(self, w),
                    less(self, height),
                    less(self, depth),
                    less(self, levels),
                );
                let k = |l: &mut Self, v: u32| l.s.m.i32(v);
                let (k30, k2, k14, k16, mask) = (
                    k(self, 30),
                    k(self, 2),
                    k(self, 14),
                    k(self, 16),
                    k(self, 0x3FFF),
                );
                let d1 = self.s.m.bin(c, Bin::Shl, w, k30);
                let hi = self.s.m.bin(c, Bin::LShr, w, k2);
                let hi = self.s.m.bin(c, Bin::And, hi, mask);
                let ht = self.s.m.bin(c, Bin::Shl, height, k14);
                let d2 = self.s.m.bin(c, Bin::Or, hi, ht);
                let ty = [0, 8, 9, 0, 10, 0, 12, 13][kind as usize];
                let base = k(self, 0xFAC | ty << 28);
                let last = self.s.m.bin(c, Bin::Shl, levels, k16);
                let d3 = self.s.m.bin(c, Bin::Or, base, last);
                vec![(1, d1), (2, d2), (3, d3), (4, depth)]
            }
            None => vec![(2, field(self, 0))],
        };
        // VIctory: fields the host view gives are rebuilt from it;
        // the console's other words (addresses, formats) come from the
        // descriptor words. Changed from VIctory (always): only in programs
        // that walk a BVH, so programs that translate today bind nothing new.
        let missing = (0..count).any(|k| !words.iter().any(|w| w.0 == k.wrapping_sub(u32::from(first))));
        let console = if missing && self.walks_bvh() { self.descriptor_words(site, count).ok() } else { None };
        for k in 0..count {
            let rel = k.wrapping_sub(u32::from(first));
            match (words.iter().find(|w| w.0 == rel), &console) {
                (Some(&(_, x)), _) => self.write(inst, Reg::S(code + k as u8), x, None),
                (None, Some(c)) => self.write(inst, Reg::S(code + k as u8), c[k as usize], None),
                (None, None) => {
                    if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                        self.vals.insert(v, Val::Missing(OTHER));
                    }
                }
            }
        }
        Ok(())
    }

    /// `s_load_dword*` from the program's constant table.
    fn code_smem(&mut self, inst: &Inst, a: Address) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar load");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar load from the code"),
        };
        let c = self.cur;
        let addr = self.address_offset(a)?;
        for k in 0..count {
            let off = self.s.m.i32(4 * k);
            let at = self.s.m.bin(c, Bin::Add, addr, off);
            let x = self.code_dword(at)?;
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// The dword at a computed byte address of a constant buffer.
    /// The dword at byte `a` of raw buffer `h` (XNW).
    fn raw_dword(&mut self, h: V, a: V) -> V {
        let c = self.cur;
        let (u, mask, al) = (self.s.m.undef(self.s.t.i32), self.s.m.i8(1), self.s.m.i32(4));
        let r = self.s.op(c, "rawBufferLoad.f32", Attr::ReadOnly, self.s.t.res_ret, 139, &[h, a, u, mask, al]);
        let x = self.s.m.extract(c, r, 0);
        self.as_int(x)
    }

    fn cbuffer_dword(&mut self, h: V, a: V) -> V {
        let c = self.cur;
        let four = self.s.m.i32(4);
        let ri = self.s.m.bin(c, Bin::LShr, a, four);
        let r = self.s.op(
            c,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, ri],
        );
        let two = self.s.m.i32(2);
        let three = self.s.m.i32(3);
        let comp = self.s.m.bin(c, Bin::LShr, a, two);
        let comp = self.s.m.bin(c, Bin::And, comp, three);
        let mut x = self.s.m.extract(c, r, 0);
        for j in 1..4 {
            let e = self.s.m.extract(c, r, j);
            let jc = self.s.m.i32(j);
            let is = self.s.m.cmp(c, pred::EQ, comp, jc);
            x = self.s.m.select(c, is, e, x);
        }
        x
    }

    /// `channels` channels of the element at byte `off` of raw buffer `h`,
    /// converted from the GFX10 buffer format `format` as the hardware's
    /// typed loads do (channels past the format's read 0, 0, 0, 1).
    // shortcut: the descriptor's channel swizzle is taken as the identity
    // (it is for raw views and the descriptors code builds), elements of 4
    // bytes or more as dword-aligned. Norm formats divide, which the AMD
    // driver does 1 ULP off the hardware's correctly rounded conversion for
    // some values; correct it with an integer check if a title compares
    // converted values exactly.
    fn format_load(&mut self, h: V, off: V, format: u8, channels: u32) -> Result<Vec<V>, Why> {
        let (bits, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let total: u32 = bits.iter().sum();
        let (u, align) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
        let load = |l: &mut Self, at: V, n: u32| -> Vec<V> {
            let mask = l.s.m.i8((1u8 << n) - 1);
            let r = l.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                l.s.t.res_ret,
                139,
                &[h, at, u, mask, align],
            );
            (0..n)
                .map(|k| {
                    let x = l.s.m.extract(c, r, k);
                    l.as_int(x)
                })
                .collect()
        };
        let words = if total < 32 {
            // A byte or half inside the dword that holds it.
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            let w = load(self, at, 1)[0];
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            vec![self.s.m.bin(c, Bin::LShr, w, sh)]
        } else {
            load(self, off, total / 32)
        };
        let float = |l: &mut Self, f: V| l.as_int(f);
        let mut out = Vec::new();
        let mut at = 0;
        for &w in bits {
            let word = words[(at / 32) as usize];
            let field = if w == 32 {
                word
            } else {
                let sh = self.s.m.i32(at % 32);
                let t = self.s.m.bin(c, Bin::LShr, word, sh);
                let m = self.s.m.i32((1 << w) - 1);
                self.s.m.bin(c, Bin::And, t, m)
            };
            at += w;
            // Sign-extended from the field's top bit.
            let signed = |l: &mut Self| {
                if w == 32 {
                    return field;
                }
                let k = l.s.m.i32(32 - w);
                let t = l.s.m.bin(c, Bin::Shl, field, k);
                l.s.m.bin(c, Bin::AShr, t, k)
            };
            let f32t = self.s.t.f32;
            let x = match numeric {
                Numeric::Unorm => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    let max = self.s.m.f32((((1u64 << w) - 1) as f32).to_bits());
                    let q = self.s.m.bin(c, Bin::SDiv, f, max);
                    float(self, q)
                }
                Numeric::Snorm => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    let max = self.s.m.f32((((1u64 << (w - 1)) - 1) as f32).to_bits());
                    let q = self.s.m.bin(c, Bin::SDiv, f, max);
                    let lo = self.s.m.f32((-1.0f32).to_bits());
                    let q = self.binary(35, q, lo);
                    float(self, q)
                }
                Numeric::Uscaled => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    float(self, f)
                }
                Numeric::Sscaled => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    float(self, f)
                }
                Numeric::Uint => field,
                Numeric::Sint => signed(self),
                Numeric::Float if w == 32 => field,
                Numeric::Float => {
                    let h = match half_shift(w) {
                        0 => field,
                        k => {
                            let sh = self.s.m.i32(k);
                            self.s.m.bin(c, Bin::Shl, field, sh)
                        }
                    };
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    float(self, f)
                }
            };
            out.push(x);
        }
        let one = match numeric {
            Numeric::Uint | Numeric::Sint => 1,
            _ => 1.0f32.to_bits(),
        };
        for k in out.len() as u32..channels {
            out.push(self.s.m.i32(if k == 3 { one } else { 0 }));
        }
        out.truncate(channels as usize);
        Ok(out)
    }

    /// The byte offset a buffer descriptor built around a root descriptor adds
    /// (constant and computed parts); none for other descriptors.
    fn built_offset(&mut self, src: &Source) -> Result<(u32, Option<V>), Why> {
        match src {
            Source::Built { address, .. } => {
                if address.rel.is_some() {
                    Ok((0, Some(self.address_offset(*address)?)))
                } else {
                    let x = match address.dynamic {
                        Some(v) => Some(self.i32_of(v)?),
                        None => None,
                    };
                    Ok((address.offset, x))
                }
            }
            _ => Ok((0, None)),
        }
    }

    /// Computes the byte offset `a` adds to its root parameter (or code base).
    fn address_offset(&mut self, a: Address) -> Result<V, Why> {
        let c = self.cur;
        if let Some(RelOffset {
            value,
            base,
            extra_dynamic,
        }) = a.rel
        {
            let mut local = HashMap::new();
            let mut off = self.rel_offset(value, base, &mut local)?;
            if a.offset != 0 {
                let k = self.s.m.i32(a.offset);
                off = self.s.m.bin(c, Bin::Add, off, k);
            }
            if let Some(d) = extra_dynamic {
                let d = self.i32_of(d)?;
                off = self.s.m.bin(c, Bin::Add, off, d);
            }
            return Ok(off);
        }
        let mut off = self.s.m.i32(a.offset);
        if let Some(d) = a.dynamic {
            let d = self.i32_of(d)?;
            off = self.s.m.bin(c, Bin::Add, off, d);
        }
        Ok(off)
    }

    fn op_value(inst: &Inst, op: &Operand) -> Option<Value> {
        match *op {
            Operand::Scalar { code, .. } if code != NULL => inst.read(Reg::S(code)),
            Operand::Vector { reg, .. } => inst.read(Reg::V(reg)),
            _ => None,
        }
    }

    /// Follows unmodified, unguarded `v_mov_b32` chains back to an SGPR value.
    fn uniform_vgpr_src(&self, mut inst: &'a Inst, mut op: &'a Operand) -> Option<Value> {
        loop {
            match *op {
                Operand::Scalar { code, .. } if code != NULL => return inst.read(Reg::S(code)),
                Operand::Vector { reg, .. } => {
                    let v = inst.read(Reg::V(reg))?;
                    let &(b, i) = self.def_site.get(&v)?;
                    if self.x.st.guards[b][i].is_some() {
                        return None;
                    }
                    let next = &self.x.p.blocks[b].insts[i];
                    if next.name() != "v_mov_b32" {
                        return None;
                    }
                    let Fields::Valu(f) = &next.fields else {
                        return None;
                    };
                    if f.abs != 0
                        || f.neg != 0
                        || f.sdwa.is_some()
                        || f.dpp.is_some()
                        || f.clamp
                        || f.omod != 0
                    {
                        return None;
                    }
                    inst = next;
                    op = f.src.first()?;
                }
                _ => return None,
            }
        }
    }

    /// Resolves a constant-lane `v_readlane_b32` reading an SGPR spilled via `v_writelane_b32`.
    fn spilled_value(&self, inst: &Inst, src: &Operand, lane: &Operand) -> Option<Value> {
        let lane_idx = match *lane {
            Operand::Const(c) => crate::resource::inline(c)?,
            Operand::Literal(x) => x,
            _ => return None,
        };
        let Operand::Vector { reg, .. } = *src else {
            return None;
        };
        let v = inst.read(Reg::V(reg))?;
        self.spilled_in_vgpr(v, lane_idx, &mut Vec::new())
            .ok()
            .flatten()
    }

    fn spilled_in_vgpr(
        &self,
        v: Value,
        lane: u32,
        seen: &mut Vec<Value>,
    ) -> Result<Option<Value>, ()> {
        if let Some(&(b, i)) = self.phi_at.get(&v) {
            if seen.contains(&v) {
                return Ok(None);
            }
            seen.push(v);
            let mut found = None;
            for &a in &self.x.p.blocks[b].phis[i].args {
                if let Some(m) = self.spilled_in_vgpr(a, lane, seen)? {
                    if found.is_some_and(|f| f != m) {
                        return Err(());
                    }
                    found = Some(m);
                }
            }
            seen.pop();
            return Ok(found);
        }
        let &(b, i) = self.def_site.get(&v).ok_or(())?;
        let inst = &self.x.p.blocks[b].insts[i];
        if inst.name() != "v_writelane_b32" {
            return Err(());
        }
        let Fields::Valu(w) = &inst.fields else {
            return Err(());
        };
        let (Some(Operand::Vector { reg, .. }), [value, at]) = (w.dst, &w.src[..]) else {
            return Err(());
        };
        let at_lane = match *at {
            Operand::Const(c) => crate::resource::inline(c).ok_or(())?,
            Operand::Literal(x) => x,
            _ => return Err(()),
        };
        if at_lane != lane {
            return self.spilled_in_vgpr(inst.read(Reg::V(reg)).ok_or(())?, lane, seen);
        }
        match *value {
            Operand::Scalar { code, .. } if code != NULL => Ok(inst.read(Reg::S(code))),
            _ => Err(()),
        }
    }

    fn op_reaches_base(&self, inst: &Inst, op: &Operand, base: Value) -> bool {
        Self::op_value(inst, op).is_some_and(|v| self.reaches_base(v, base))
    }

    fn reaches_base(&self, v: Value, base: Value) -> bool {
        fn rec(
            l: &Lower<'_>,
            v: Value,
            base: Value,
            visiting: &mut Vec<Value>,
            memo: &mut HashMap<Value, bool>,
        ) -> bool {
            if v == base {
                return true;
            }
            if let Some(&hit) = memo.get(&v) {
                return hit;
            }
            if visiting.contains(&v) {
                return false;
            }
            let r = if let Some(&(b, i)) = l.phi_at.get(&v) {
                visiting.push(v);
                let hit = l.x.p.blocks[b].phis[i]
                    .args
                    .iter()
                    .any(|&a| rec(l, a, base, visiting, memo));
                visiting.pop();
                hit
            } else if let Some(&(b, i)) = l.def_site.get(&v) {
                let inst = &l.x.p.blocks[b].insts[i];
                let name = inst.name();
                let mut op_reaches = |op: &Operand| {
                    Lower::op_value(inst, op).is_some_and(|x| rec(l, x, base, visiting, memo))
                };
                match &inst.fields {
                    Fields::Sop { dst, src, .. } => match name {
                        "s_mov_b32" => src.first().is_some_and(&mut op_reaches),
                        "s_mov_b64" => {
                            let k = inst.writes.iter().position(|w| w.1 == v);
                            match (k, src.first()) {
                                (Some(k), Some(&Operand::Scalar { code, .. })) if code != NULL => {
                                    inst.read(Reg::S(code + k as u8))
                                        .is_some_and(|x| rec(l, x, base, visiting, memo))
                                }
                                _ => false,
                            }
                        }
                        "s_addk_i32" => match dst {
                            Some(Operand::Scalar { code, .. }) if *code != NULL => inst
                                .read(Reg::S(*code))
                                .is_some_and(|x| rec(l, x, base, visiting, memo)),
                            _ => false,
                        },
                        "s_add_u32" | "s_add_i32" | "s_or_b32" => {
                            src.first().is_some_and(&mut op_reaches)
                                || src.get(1).is_some_and(&mut op_reaches)
                        }
                        "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32"
                        | "s_lshl4_add_u32" => src.get(1).is_some_and(&mut op_reaches),
                        "s_sub_u32" | "s_sub_i32" => src.first().is_some_and(&mut op_reaches),
                        _ => false,
                    },
                    Fields::Valu(f)
                        if f.abs == 0
                            && f.neg == 0
                            && f.sdwa.is_none()
                            && f.dpp.is_none()
                            && !f.clamp
                            && f.omod == 0 =>
                    {
                        match name {
                            "v_mov_b32" | "v_readfirstlane_b32" => {
                                f.src.first().is_some_and(&mut op_reaches)
                            }
                            "v_readlane_b32" => match (f.src.first(), f.src.get(1)) {
                                (Some(op0), Some(op1)) => {
                                    if let Some(sv) = l.spilled_value(inst, op0, op1) {
                                        rec(l, sv, base, visiting, memo)
                                    } else {
                                        op_reaches(op0)
                                    }
                                }
                                _ => false,
                            },
                            "v_add_co_u32" | "v_add_nc_u32" | "v_add_nc_i32" | "v_or_b32" => {
                                f.src.first().is_some_and(&mut op_reaches)
                                    || f.src.get(1).is_some_and(&mut op_reaches)
                            }
                            "v_lshl_add_u32" => f.src.get(2).is_some_and(&mut op_reaches),
                            "v_add3_u32" => f.src.iter().take(3).any(&mut op_reaches),
                            "v_sub_co_u32" | "v_sub_nc_u32" => {
                                f.src.first().is_some_and(&mut op_reaches)
                            }
                            "v_subrev_co_u32" | "v_subrev_nc_u32" => {
                                f.src.get(1).is_some_and(&mut op_reaches)
                            }
                            _ => false,
                        }
                    }
                    _ => false,
                }
            } else {
                false
            };
            if visiting.is_empty() {
                memo.insert(v, r);
            }
            r
        }
        rec(self, v, base, &mut Vec::new(), &mut HashMap::new())
    }

    fn rel_offset_op(
        &mut self,
        inst: &Inst,
        op: &Operand,
        base: Value,
        local: &mut HashMap<Value, V>,
    ) -> Result<V, Why> {
        let v = Self::op_value(inst, op).ok_or("an untracked relative address")?;
        self.rel_offset(v, base, local)
    }

    /// Emits `v - base` in `self.cur`, caching non-phi nodes only in `local`
    /// (per block / edge) and phi nodes globally in `self.addr_phis`.
    fn rel_offset(
        &mut self,
        v: Value,
        base: Value,
        local: &mut HashMap<Value, V>,
    ) -> Result<V, Why> {
        if v == base {
            return Ok(self.s.m.i32(0));
        }
        if let Some(&x) = local.get(&v) {
            return Ok(x);
        }
        if let Some(&(phi_b, phi_idx)) = self.phi_at.get(&v) {
            if let Some(&phi) = self.addr_phis.get(&(v, base)) {
                return Ok(phi);
            }
            let blk = self.block[phi_b].ok_or("an unreachable address phi")?;
            let phi = self.s.m.phi(blk, self.s.t.i32);
            self.addr_phis.insert((v, base), phi);
            self.pending_addr_phis.push((phi, phi_b, phi_idx, base));
            return Ok(phi);
        }
        let &(ib, ii) = self
            .def_site
            .get(&v)
            .ok_or("an untracked relative address")?;
        let inst: &'a Inst = &self.x.p.blocks[ib].insts[ii];
        let name = inst.name();
        let c = self.cur;
        let x = match &inst.fields {
            Fields::Sop { dst, src, imm } => match name {
                "s_mov_b32" => {
                    let op0 = src.first().ok_or("a missing operand")?;
                    self.rel_offset_op(inst, op0, base, local)?
                }
                "s_mov_b64" => {
                    let k = inst
                        .writes
                        .iter()
                        .position(|w| w.1 == v)
                        .ok_or("an untracked relative address")?;
                    match src.first() {
                        Some(&Operand::Scalar { code, .. }) if code != NULL => {
                            let prev = inst
                                .read(Reg::S(code + k as u8))
                                .ok_or("an unread operand")?;
                            self.rel_offset(prev, base, local)?
                        }
                        _ => return Err("an untracked relative address"),
                    }
                }
                "s_addk_i32" => {
                    let Some(Operand::Scalar { code, .. }) = dst else {
                        return Err("a missing operand");
                    };
                    let prev = inst.read(Reg::S(*code)).ok_or("an unread operand")?;
                    let a = self.rel_offset(prev, base, local)?;
                    let b = self.s.m.i32(i32::from(imm.unwrap_or(0) as i16) as u32);
                    self.s.m.bin(c, Bin::Add, a, b)
                }
                "s_add_u32" | "s_add_i32" | "s_or_b32" => {
                    let (op0, op1) = (
                        src.first().ok_or("a missing operand")?,
                        src.get(1).ok_or("a missing operand")?,
                    );
                    match (
                        self.op_reaches_base(inst, op0, base),
                        self.op_reaches_base(inst, op1, base),
                    ) {
                        (true, false) => {
                            let a = self.rel_offset_op(inst, op0, base, local)?;
                            let b = self.operand(inst, op1)?;
                            self.s.m.bin(c, Bin::Add, a, b)
                        }
                        (false, true) => {
                            let a = self.rel_offset_op(inst, op1, base, local)?;
                            let b = self.operand(inst, op0)?;
                            self.s.m.bin(c, Bin::Add, a, b)
                        }
                        _ => return Err("an untracked relative address"),
                    }
                }
                "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
                    let (op0, op1) = (
                        src.first().ok_or("a missing operand")?,
                        src.get(1).ok_or("a missing operand")?,
                    );
                    if self.op_reaches_base(inst, op0, base)
                        || !self.op_reaches_base(inst, op1, base)
                    {
                        return Err("an untracked relative address");
                    }
                    let a = self.operand(inst, op0)?;
                    let k = self.s.m.i32(u32::from(name.as_bytes()[6] - b'0'));
                    let sh = self.s.m.bin(c, Bin::Shl, a, k);
                    let b = self.rel_offset_op(inst, op1, base, local)?;
                    self.s.m.bin(c, Bin::Add, sh, b)
                }
                "s_sub_u32" | "s_sub_i32" => {
                    let (op0, op1) = (
                        src.first().ok_or("a missing operand")?,
                        src.get(1).ok_or("a missing operand")?,
                    );
                    if !self.op_reaches_base(inst, op0, base)
                        || self.op_reaches_base(inst, op1, base)
                    {
                        return Err("an untracked relative address");
                    }
                    let a = self.rel_offset_op(inst, op0, base, local)?;
                    let b = self.operand(inst, op1)?;
                    self.s.m.bin(c, Bin::Sub, a, b)
                }
                _ => return Err("an untracked relative address"),
            },
            Fields::Valu(f)
                if f.abs == 0
                    && f.neg == 0
                    && f.sdwa.is_none()
                    && f.dpp.is_none()
                    && !f.clamp
                    && f.omod == 0 =>
            {
                let mut raw = match name {
                    "v_mov_b32" => {
                        let op0 = f.src.first().ok_or("a missing operand")?;
                        self.rel_offset_op(inst, op0, base, local)?
                    }
                    "v_readfirstlane_b32" => {
                        let op0 = f.src.first().ok_or("a missing operand")?;
                        if let Some(sv) = self.uniform_vgpr_src(inst, op0) {
                            self.rel_offset(sv, base, local)?
                        } else {
                            let a = self.rel_offset_op(inst, op0, base, local)?;
                            self.s.flags |= flags::WAVE_OPS;
                            self.s.op(
                                c,
                                "waveReadLaneFirst.i32",
                                Attr::NoUnwind,
                                self.s.t.i32,
                                118,
                                &[a],
                            )
                        }
                    }
                    "v_readlane_b32" => {
                        let (op0, op1) = (
                            f.src.first().ok_or("a missing operand")?,
                            f.src.get(1).ok_or("a missing operand")?,
                        );
                        if let Some(sv) = self
                            .spilled_value(inst, op0, op1)
                            .or_else(|| self.uniform_vgpr_src(inst, op0))
                        {
                            self.rel_offset(sv, base, local)?
                        } else {
                            let a = self.rel_offset_op(inst, op0, base, local)?;
                            let lane = self.operand(inst, op1)?;
                            if self.mask_data == MaskData::Host && !matches!(lane, V::Const(_)) {
                                self.s.flags |= flags::WAVE_OPS;
                            } else {
                                self.cross_lane(false)?;
                            }
                            self.read_lane(a, lane)
                        }
                    }
                    "v_add_co_u32" | "v_add_nc_u32" | "v_add_nc_i32" | "v_or_b32" => {
                        let (op0, op1) = (
                            f.src.first().ok_or("a missing operand")?,
                            f.src.get(1).ok_or("a missing operand")?,
                        );
                        match (
                            self.op_reaches_base(inst, op0, base),
                            self.op_reaches_base(inst, op1, base),
                        ) {
                            (true, false) => {
                                let a = self.rel_offset_op(inst, op0, base, local)?;
                                let b = self.operand(inst, op1)?;
                                self.s.m.bin(c, Bin::Add, a, b)
                            }
                            (false, true) => {
                                let a = self.rel_offset_op(inst, op1, base, local)?;
                                let b = self.operand(inst, op0)?;
                                self.s.m.bin(c, Bin::Add, a, b)
                            }
                            _ => return Err("an untracked relative address"),
                        }
                    }
                    "v_lshl_add_u32" => {
                        let (op0, op1, op2) = (
                            f.src.first().ok_or("a missing operand")?,
                            f.src.get(1).ok_or("a missing operand")?,
                            f.src.get(2).ok_or("a missing operand")?,
                        );
                        if self.op_reaches_base(inst, op0, base)
                            || self.op_reaches_base(inst, op1, base)
                            || !self.op_reaches_base(inst, op2, base)
                        {
                            return Err("an untracked relative address");
                        }
                        let a = self.operand(inst, op0)?;
                        let sh = self.operand(inst, op1)?;
                        let k31 = self.s.m.i32(31);
                        let sh = self.s.m.bin(c, Bin::And, sh, k31);
                        let t = self.s.m.bin(c, Bin::Shl, a, sh);
                        let b = self.rel_offset_op(inst, op2, base, local)?;
                        self.s.m.bin(c, Bin::Add, t, b)
                    }
                    "v_add3_u32" => {
                        let (op0, op1, op2) = (
                            f.src.first().ok_or("a missing operand")?,
                            f.src.get(1).ok_or("a missing operand")?,
                            f.src.get(2).ok_or("a missing operand")?,
                        );
                        let ops = [op0, op1, op2];
                        let reaches: Vec<usize> = (0..3)
                            .filter(|&k| self.op_reaches_base(inst, ops[k], base))
                            .collect();
                        let [base_idx] = reaches[..] else {
                            return Err("an untracked relative address");
                        };
                        let mut acc = self.rel_offset_op(inst, ops[base_idx], base, local)?;
                        for (k, op) in ops.into_iter().enumerate() {
                            if k != base_idx {
                                let term = self.operand(inst, op)?;
                                acc = self.s.m.bin(c, Bin::Add, acc, term);
                            }
                        }
                        acc
                    }
                    "v_sub_co_u32" | "v_sub_nc_u32" => {
                        let (op0, op1) = (
                            f.src.first().ok_or("a missing operand")?,
                            f.src.get(1).ok_or("a missing operand")?,
                        );
                        if !self.op_reaches_base(inst, op0, base)
                            || self.op_reaches_base(inst, op1, base)
                        {
                            return Err("an untracked relative address");
                        }
                        let a = self.rel_offset_op(inst, op0, base, local)?;
                        let b = self.operand(inst, op1)?;
                        self.s.m.bin(c, Bin::Sub, a, b)
                    }
                    "v_subrev_co_u32" | "v_subrev_nc_u32" => {
                        let (op0, op1) = (
                            f.src.first().ok_or("a missing operand")?,
                            f.src.get(1).ok_or("a missing operand")?,
                        );
                        if self.op_reaches_base(inst, op0, base)
                            || !self.op_reaches_base(inst, op1, base)
                        {
                            return Err("an untracked relative address");
                        }
                        let a = self.operand(inst, op0)?;
                        let b = self.rel_offset_op(inst, op1, base, local)?;
                        self.s.m.bin(c, Bin::Sub, b, a)
                    }
                    _ => return Err("an untracked relative address"),
                };
                if let Some(Operand::Vector { reg, .. }) = f.dst
                    && let Some(g) = self.x.st.guards[ib][ii]
                    && let Some(old) = inst.read(Reg::V(reg))
                    && self.reaches_base(old, base)
                {
                    let cond = self.boolean(g, c)?;
                    let fallback = self.rel_offset(old, base, local)?;
                    raw = self.s.m.select(c, cond, raw, fallback);
                }
                raw
            }
            _ => return Err("an untracked relative address"),
        };
        local.insert(v, x);
        Ok(x)
    }

    /// The used-resource entry a descriptor loaded from a table falls in.
    fn used_kind(&self, src: &Source) -> Option<&'a UsedResource> {
        if let Source::Phi { args, .. } = src {
            return self.used_kind(args.first()?);
        }
        if let Source::Built { address, .. } = src {
            let rp = self.x.native.layout.get(address.param)?;
            return self
                .x
                .native
                .used
                .iter()
                .find(|u| u.root == RootType::Descriptor && u.user_data == rp.user_data);
        }
        let Source::Loaded { load, first } = src else {
            return None;
        };
        let a = self.scalar.get(load)?;
        let rp = self.x.native.layout.get(a.param)?;
        if rp.root != RootType::Table {
            return None;
        }
        let dword = (a.offset / 4) + u32::from(*first);
        self.x
            .native
            .used
            .iter()
            .filter(|u| {
                u.root == RootType::Table
                    && u.user_data == rp.user_data
                    && u.table_offset_dwords <= dword
            })
            .max_by_key(|u| u.table_offset_dwords)
    }

    /// A handle for the descriptor `src` read at `site`.
    fn handle(
        &mut self,
        site: Site,
        src: &Source,
        sampler: bool,
        props: (u32, u32),
    ) -> Result<V, Why> {
        if self.x.gpu_desc && !matches!(src, Source::Built { .. } | Source::Constant(_)) {
            return self.record_handle(site, sampler, props);
        }
        match src {
            Source::Runtime => Err("a descriptor resolved at run time"),
            Source::Loaded { load, first } => {
                let key = (self.cur, *load, *first, sampler, props.0);
                if let Some(&h) = self.handles.get(&key) {
                    return Ok(h);
                }
                let index = self.heap_index(*load, *first, sampler)?;
                let h = self.s.heap_handle(self.cur, index, sampler, props);
                self.handles.insert(key, h);
                Ok(h)
            }
            Source::Constant(words) if sampler => {
                let words: [u32; 4] = words
                    .get(..4)
                    .and_then(|w| w.try_into().ok())
                    .ok_or("a sampler constant")?;
                let n = match self.samplers.iter().position(|w| *w == words) {
                    Some(n) => n,
                    None => {
                        self.samplers.push(words);
                        self.samplers.len() - 1
                    }
                };
                if props.0 & 0x8000 != 0 {
                    // shortcut: one sampler used with and without comparison
                    // would need two static samplers; it becomes a comparison one.
                    self.s.cmp_samplers |= 1 << n;
                }
                Ok(self.s.sampler_handle(self.cur, n as u32))
            }
            Source::Constant(_) => Err("a constant resource descriptor"),
            Source::Built { address, .. } => {
                let rp = *self
                    .x
                    .native
                    .layout
                    .get(address.param)
                    .ok_or("a descriptor built around a table in the code")?;
                let kind = self
                    .x
                    .native
                    .used
                    .iter()
                    .find(|u| u.user_data == rp.user_data)
                    .map(|u| u.kind);
                match (rp.root, kind, props.0) {
                    (RootType::Descriptor, Some(ResourceType::Cbv) | None, 13) => {
                        Ok(self.s.root_cbv_handle(rp.slot))
                    }
                    (RootType::Descriptor, Some(ResourceType::Srv), 11) => {
                        Ok(self.s.root_srv_handle(rp.slot))
                    }
                    // VIctory (bffaefd, GPL-3.0-or-later): root UAVs.
                    (RootType::Descriptor, Some(ResourceType::Uav) | None, p)
                        if p == 11 | 0x1000 =>
                    {
                        Ok(self.s.root_uav_handle(rp.slot))
                    }
                    _ => Err("a typed root view"),
                }
            }
            Source::Phi { block, args } => {
                // Each predecessor computes its heap index at its end; the
                // join picks one with a phi.
                let _ = site;
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let (Some(tail), Source::Loaded { load, first }) = (self.tail[q], arg) else {
                        return Err("a descriptor chosen by control flow");
                    };
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    let here = std::mem::replace(&mut self.cur, at);
                    let index = self.heap_index(*load, *first, sampler);
                    self.cur = here;
                    incoming.push((index?, at));
                }
                let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok(self.s.heap_handle(self.cur, phi, sampler, props))
            }
        }
    }

    /// The view-heap slot of `src` when it comes from a descriptor table or
    /// heap (`Loaded` or `Phi` of `Loaded`).
    /// victory-fork (578d459, `slot_stride`'s test, GPL-3.0-or-later):
    /// whether the program picks `src` from the view heap at an index it
    /// computes (root type `ResourceHeap`).
    fn heap_picked(&self, src: &Source) -> bool {
        let Source::Loaded { load, .. } = src else {
            return false;
        };
        self.scalar
            .get(load)
            .and_then(|a| self.x.native.layout.get(a.param))
            .is_some_and(|p| p.root == RootType::ResourceHeap)
    }

    fn heap_slot(&mut self, src: &Source, sampler: bool) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => self.heap_index(*load, *first, sampler),
            Source::Phi { block, args } => {
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let (Some(tail), Source::Loaded { load, first }) = (self.tail[q], arg) else {
                        return Err("a descriptor chosen by control flow");
                    };
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    let here = std::mem::replace(&mut self.cur, at);
                    let index = self.heap_index(*load, *first, sampler);
                    self.cur = here;
                    incoming.push((index?, at));
                }
                let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok(phi)
            }
            _ => Err("a descriptor outside the heaps"),
        }
    }

    /// The entry of the host's runtime stride table (`t0`,
    /// [`super::STRIDE_SPACE`]) for the view-heap slot of `src`, read in the
    /// current block: the slot holding a raw view of the buffer's bytes
    /// (the slot itself when the table names none), the stride and the bias
    /// ([`super::StrideEntry`]).
    fn stride_entry(&mut self, src: &Source) -> Result<[V; 3], Why> {
        let key = match src {
            // XNW: with GPU-side descriptors the slot is the record's.
            _ if self.x.gpu_desc => None,
            Source::Loaded { load, first } => Some((self.cur, *load, *first)),
            _ => None,
        };
        if let Some(e) = key.and_then(|k| self.entries.get(&k)) {
            return Ok(*e);
        }
        let slot = match self.buffer_site {
            // XNW: the global heap's stride table, by the record's slot.
            Some(site) if self.x.gpu_desc => self.record_index(site, false, (11, 0))?,
            _ => self.heap_slot(src, false).map_err(|_| "a structured buffer of unknown stride")?,
        };
        let c = self.cur;
        let table = self.s.stride_table_handle();
        let bytes = self.s.m.i32(super::StrideEntry::BYTES);
        let addr = self.s.m.bin(c, Bin::Mul, slot, bytes);
        let iu = self.s.m.undef(self.s.t.i32);
        let mask = self.s.m.i8(3);
        let align = self.s.m.i32(4);
        let res = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[table, addr, iu, mask, align],
        );
        let stride = self.s.m.extract(c, res, 0);
        let stride = self.as_int(stride);
        let word = self.s.m.extract(c, res, 1);
        let word = self.as_int(word);
        // The raw slot plus one under the bias; 0: the slot itself.
        let low = self.s.m.i32(super::StrideEntry::RAW_SLOTS);
        let named = self.s.m.bin(c, Bin::And, word, low);
        let zero = self.s.m.i32(0);
        let own = self.s.m.cmp(c, pred::EQ, named, zero);
        let one = self.s.m.i32(1);
        let named = self.s.m.bin(c, Bin::Sub, named, one);
        let raw = self.s.m.select(c, own, slot, named);
        let k = self.s.m.i32(28);
        let bias = self.s.m.bin(c, Bin::LShr, word, k);
        let e = [raw, stride, bias];
        if let Some(k) = key {
            self.entries.insert(k, e);
        }
        Ok(e)
    }

    /// The raw handle (`props`: 11, with the UAV bit for one written) and
    /// the byte offset of `off` in element `index` of a buffer whose stride
    /// was not recovered statically. Both come from the host's runtime
    /// stride table: the handle is of the slot the table names, which holds
    /// a raw view of the buffer's bytes whatever view the title made (raw,
    /// structured or typed), and the byte is `index * stride + off` past
    /// that view's bias. The raw view ends with the title's view, so an
    /// index past its elements reads zeros (and its writes are dropped).
    /// shortcut: the console reads zeros (and drops writes) where `off` and
    /// the access pass the element's end; here they reach the next element,
    /// as the statically known strides' structured accesses may.
    fn runtime_view(
        &mut self,
        src: &Source,
        index: V,
        off: V,
        props: u32,
    ) -> Result<(V, V), Why> {
        let [raw, stride, bias] = self.stride_entry(src)?;
        let c = self.cur;
        let element = self.s.m.bin(c, Bin::Mul, index, stride);
        let byte = self.s.m.bin(c, Bin::Add, element, off);
        let byte = self.s.m.bin(c, Bin::Add, byte, bias);
        Ok((self.s.heap_handle(c, raw, false, (props, 0)), byte))
    }

    /// The heap index of the descriptor at dword `first` of what the scalar
    /// load at `load` reads, computed in the current block.
    fn heap_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        let addr = self.heap_byte(load, first)?;
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        Ok(self.s.m.bin(self.cur, Bin::LShr, addr, shift))
    }

    /// The byte offset in its heap of dword `first` of what the scalar load
    /// at `load` reads (the console's entry sizes), computed in the current
    /// block.
    fn heap_byte(&mut self, load: Site, first: u8) -> Result<V, Why> {
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        let Fields::Smem {
            base,
            offset,
            soffset,
            ..
        } = &li.fields
        else {
            return Err("a descriptor load");
        };
        let Operand::Scalar { code, .. } = *base else {
            return Err("a descriptor load");
        };
        let c = self.cur;
        let base = li.read(Reg::S(code)).ok_or("a descriptor load")?;
        let mut addr = self.i32_of(base)?;
        let k = self
            .s
            .m
            .i32((*offset as u32).wrapping_add(4 * u32::from(first)));
        addr = self.s.m.bin(c, Bin::Add, addr, k);
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(li, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        Ok(addr)
    }

    fn valu(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        if let Some(rest) = name.strip_prefix("v_cmp_").or(name.strip_prefix("v_cmpx_")) {
            let r = if v.dpp.is_some() {
                Err("a compare with DPP")
            } else {
                self.compare(inst, rest)
            };
            if let Err(why) = r {
                for &(_, w) in &inst.writes {
                    self.lane_fail.insert(w, why);
                }
            }
            return r;
        }
        let mut guard = self.guard(b, i)?;
        if let Some(d) = v.dpp {
            let (x, enable) = self.dpp(inst, d)?;
            guard = match (guard, enable) {
                (Some(g), Some(e)) => Some(self.s.m.bin(self.cur, Bin::And, g, e)),
                (g, e) => g.or(e),
            };
            self.src0 = Some(x);
        }
        let r = self.valu_op(inst, guard);
        self.src0 = None;
        r
    }

    /// DPP: src0 as read from another lane, and which lanes write their
    /// result (none: all).
    fn dpp(&mut self, inst: &Inst, dpp: Dpp) -> Result<(V, Option<V>), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(&Operand::Vector { reg, .. }) = v.src.first() else {
            return Err("a DPP operand");
        };
        let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("a DPP operand")?)?;
        // Quad permutations and DPP8 stay in groups the pixel layout shares.
        let quad = matches!(dpp, Dpp::Row { ctrl: 0..=0xFF, .. });
        let mesh_quad = self.s.mesh.is_some() && quad;
        if !mesh_quad {
            self.cross_lane(quad)?;
        }
        let c = self.cur;
        let lane = self.lane_index();
        let k = |l: &mut Self, n: u32| l.s.m.i32(n);
        let (sixteen_mask, low) = (k(self, !15), k(self, 15));
        let row = self.s.m.bin(c, Bin::And, lane, sixteen_mask);
        let pos = self.s.m.bin(c, Bin::And, lane, low);
        // Source lane, and whether it lies in the row.
        let (src, inside, masks, zero) = match dpp {
            Dpp::Lanes { sel, .. } => {
                let (seven, not7, three) = (k(self, 7), k(self, !7), k(self, 3));
                let p = self.s.m.bin(c, Bin::And, lane, seven);
                let sh = self.s.m.bin(c, Bin::Mul, p, three);
                let sel = k(self, sel);
                let pick = self.s.m.bin(c, Bin::LShr, sel, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, seven);
                let base = self.s.m.bin(c, Bin::And, lane, not7);
                (self.s.m.bin(c, Bin::Or, base, pick), None, None, false)
            }
            Dpp::Row {
                ctrl,
                row_mask,
                bank_mask,
                bound_ctrl,
                ..
            } => {
                let n = u32::from(ctrl & 15);
                let (src, inside) = match ctrl {
                    0..=0xFF => {
                        let (three, one, not3) = (k(self, 3), k(self, 1), k(self, !3));
                        let p = self.s.m.bin(c, Bin::And, lane, three);
                        let sh = self.s.m.bin(c, Bin::Shl, p, one);
                        let perm = k(self, u32::from(ctrl));
                        let pick = self.s.m.bin(c, Bin::LShr, perm, sh);
                        let pick = self.s.m.bin(c, Bin::And, pick, three);
                        let base = self.s.m.bin(c, Bin::And, lane, not3);
                        (self.s.m.bin(c, Bin::Or, base, pick), None)
                    }
                    0x101..=0x10F | 0x111..=0x11F => {
                        let nn = k(self, n);
                        let left = ctrl < 0x110;
                        let (op, p) = if left {
                            (Bin::Add, pred::IULT)
                        } else {
                            (Bin::Sub, pred::IUGE)
                        };
                        let s = self.s.m.bin(c, op, lane, nn);
                        // Left: pos + n < 16; right: pos >= n.
                        let bound = if left { k(self, 16 - n) } else { nn };
                        (s, Some(self.s.m.cmp(c, p, pos, bound)))
                    }
                    0x121..=0x12F => {
                        let nn = k(self, n);
                        let t = self.s.m.bin(c, Bin::Sub, pos, nn);
                        let t = self.s.m.bin(c, Bin::And, t, low);
                        (self.s.m.bin(c, Bin::Or, row, t), None)
                    }
                    0x140 | 0x141 => {
                        let m = k(self, if ctrl == 0x140 { 15 } else { 7 });
                        let t = self.s.m.bin(c, Bin::Xor, lane, m);
                        (t, None)
                    }
                    0x150..=0x15F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Or, row, nn), None)
                    }
                    0x160..=0x16F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Xor, lane, nn), None)
                    }
                    _ => return Err("a DPP lane pattern"),
                };
                let masks = (row_mask != 0xF || bank_mask != 0xF).then_some((row_mask, bank_mask));
                (src, inside, masks, bound_ctrl)
            }
        };
        let mut r = if mesh_quad || self.helpers_read(quad) {
            self.quad_pick(x, src)
        } else {
            self.read_lane(x, src)
        };
        let mut enable = None;
        if let Some(inside) = inside {
            if zero {
                let z = k(self, 0);
                r = self.s.m.select(c, inside, r, z);
            } else {
                enable = Some(inside);
            }
        }
        if let Some((rows, banks)) = masks {
            // Row (16 lanes) and bank (4 lanes of a row) write masks.
            let (four, two, three) = (k(self, 4), k(self, 2), k(self, 3));
            let ri = self.s.m.bin(c, Bin::LShr, lane, four);
            let ri = self.s.m.bin(c, Bin::And, ri, three);
            let bi = self.s.m.bin(c, Bin::LShr, lane, two);
            let bi = self.s.m.bin(c, Bin::And, bi, three);
            let (rm, bm) = (k(self, u32::from(rows)), k(self, u32::from(banks)));
            let rb = self.s.m.bin(c, Bin::LShr, rm, ri);
            let bb = self.s.m.bin(c, Bin::LShr, bm, bi);
            let both = self.s.m.bin(c, Bin::And, rb, bb);
            let (one, z) = (k(self, 1), k(self, 0));
            let both = self.s.m.bin(c, Bin::And, both, one);
            let on = self.s.m.cmp(c, pred::NE, both, z);
            enable = Some(match enable {
                Some(e) => self.s.m.bin(c, Bin::And, e, on),
                None => on,
            });
        }
        Ok((r, enable))
    }

    fn valu_op(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let c = self.cur;
        let f = |l: &mut Self, k| l.fsrc(inst, k);
        let s = |l: &mut Self, k| l.src(inst, k);
        let float = |l: &mut Self, r: V| l.write_f(inst, r, guard);
        let int = |l: &mut Self, r: V| l.write_i(inst, r, guard);
        if let Some(r) = self.valu16(inst, name, guard) {
            return r;
        }
        if let Some(op) = name.strip_prefix("v_pk_") {
            return self.packed16(inst, op, guard);
        }
        match name {
            "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64" => {
                let n = s(self, 0)?;
                let x = self.src64(inst, 1)?;
                let i64t = self.s.m.int(64);
                let k63 = self.s.m.i32(63);
                let n = self.s.m.bin(c, Bin::And, n, k63);
                let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
                let op = match name {
                    "v_lshlrev_b64" => Bin::Shl,
                    "v_lshrrev_b64" => Bin::LShr,
                    _ => Bin::AShr,
                };
                let r = self.s.m.bin(c, op, x, n);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a 64-bit result");
                };
                let (lo, hi) = self.split64(r);
                self.write(inst, Reg::V(reg), lo, guard);
                self.write(inst, Reg::V(reg + 1), hi, guard);
                Ok(())
            }
            "v_cubeid_f32" | "v_cubesc_f32" | "v_cubetc_f32" | "v_cubema_f32" => {
                // Cube-map face coordinates (AMD's ISA): the major axis is z
                // when |z| >= |x| and |z| >= |y|, else y when |y| >= |x|,
                // else x.
                let (x, y, z) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let (ax, ay, az) = (self.unary(6, x), self.unary(6, y), self.unary(6, z));
                let zx = self.s.m.cmp(c, pred::OGE, az, ax);
                let zy = self.s.m.cmp(c, pred::OGE, az, ay);
                let zmaj = self.s.m.bin(c, Bin::And, zx, zy);
                let ymaj = self.s.m.cmp(c, pred::OGE, ay, ax);
                let zero = self.s.m.f32(0);
                let neg = |l: &mut Self, v: V| l.s.m.cmp(c, pred::OLT, v, zero);
                let minus = |l: &mut Self, v: V| {
                    let z = l.s.m.f32(0x8000_0000);
                    l.s.m.bin(c, Bin::Sub, z, v)
                };
                let pick = |l: &mut Self, zv: V, yv: V, xv: V| {
                    let t = l.s.m.select(c, ymaj, yv, xv);
                    l.s.m.select(c, zmaj, zv, t)
                };
                let r = match name {
                    "v_cubeid_f32" => {
                        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
                        let (nz, ny, nx) = (neg(self, z), neg(self, y), neg(self, x));
                        let (f5, f4, f3, f2, f1, f0) = (
                            k(self, 5.0),
                            k(self, 4.0),
                            k(self, 3.0),
                            k(self, 2.0),
                            k(self, 1.0),
                            k(self, 0.0),
                        );
                        let zf = self.s.m.select(c, nz, f5, f4);
                        let yf = self.s.m.select(c, ny, f3, f2);
                        let xf = self.s.m.select(c, nx, f1, f0);
                        pick(self, zf, yf, xf)
                    }
                    "v_cubesc_f32" => {
                        let (nz, nx) = (neg(self, z), neg(self, x));
                        let (mx, mz) = (minus(self, x), minus(self, z));
                        let zf = self.s.m.select(c, nz, mx, x);
                        let xf = self.s.m.select(c, nx, z, mz);
                        pick(self, zf, x, xf)
                    }
                    "v_cubetc_f32" => {
                        let ny = neg(self, y);
                        let (my, mz) = (minus(self, y), minus(self, z));
                        let yf = self.s.m.select(c, ny, mz, z);
                        pick(self, my, yf, my)
                    }
                    _ => {
                        let t = pick(self, z, y, x);
                        let two = self.s.m.f32(2.0f32.to_bits());
                        self.s.m.bin(c, Bin::Mul, t, two)
                    }
                };
                float(self, r)
            }
            "v_movrels_b32" | "v_movreld_b32" | "v_movrelsd_b32" => {
                // M0 indexes the source (movrels), the destination (movreld)
                // or both, from the encoded register up to the program's
                // VGPR count (the registers the instruction reads or writes).
                let m0 = self.i32_of(inst.read(Reg::S(crate::ir::M0)).ok_or("M0")?)?;
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("an M0-relative move");
                };
                let indexed_src = name != "v_movreld_b32";
                let value = if indexed_src {
                    let mut r = self.s.m.i32(0);
                    for k in 0..=255u8.saturating_sub(sreg) {
                        let Some(x) = inst.read(Reg::V(sreg + k)) else {
                            break;
                        };
                        let x = self.i32_of(x)?;
                        let kk = self.s.m.i32(u32::from(k));
                        let is = self.s.m.cmp(c, pred::EQ, m0, kk);
                        r = self.s.m.select(c, is, x, r);
                    }
                    r
                } else {
                    s(self, 0)?
                };
                if name == "v_movrels_b32" {
                    return int(self, value);
                }
                for k in 0..=255u8.saturating_sub(dreg) {
                    let Some(_) = inst.write(Reg::V(dreg + k)) else {
                        break;
                    };
                    let old = match inst.read(Reg::V(dreg + k)) {
                        Some(o) => self.i32_of(o)?,
                        None => self.s.m.i32(0),
                    };
                    let kk = self.s.m.i32(u32::from(k));
                    let is = self.s.m.cmp(c, pred::EQ, m0, kk);
                    let x = self.s.m.select(c, is, value, old);
                    self.write_placed(inst, Reg::V(dreg + k), x, guard);
                }
                Ok(())
            }
            "v_swap_b32" => {
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("v_swap_b32");
                };
                let a = self.i32_of(inst.read(Reg::V(dreg)).ok_or("v_swap_b32")?)?;
                let b = self.i32_of(inst.read(Reg::V(sreg)).ok_or("v_swap_b32")?)?;
                self.write_placed(inst, Reg::V(dreg), b, guard);
                if dreg != sreg {
                    self.write_placed(inst, Reg::V(sreg), a, guard);
                }
                Ok(())
            }
            "v_ffbh_u32" | "v_ffbh_i32" => {
                // Counted from the top bit, -1 when none (DXIL's too).
                let a = s(self, 0)?;
                let op = if name == "v_ffbh_u32" { 33 } else { 34 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                int(self, r)
            }
            "v_mov_b32" => {
                if v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0 {
                    let a = f(self, 0)?;
                    return float(self, a);
                }
                let a = s(self, 0)?;
                int(self, a)
            }
            "v_add_f32" | "v_sub_f32" | "v_subrev_f32" | "v_mul_f32" | "v_mul_legacy_f32"
            | "v_max_f32" | "v_min_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let r = match name {
                    "v_add_f32" => self.s.m.bin(c, Bin::Add, a, bb),
                    "v_sub_f32" => self.s.m.bin(c, Bin::Sub, a, bb),
                    "v_subrev_f32" => self.s.m.bin(c, Bin::Sub, bb, a),
                    "v_mul_f32" => self.s.m.bin(c, Bin::Mul, a, bb),
                    "v_mul_legacy_f32" => self.legacy_mul(a, bb),
                    "v_max_f32" => self.max_min(true, a, bb),
                    _ => self.max_min(false, a, bb),
                };
                float(self, r)
            }
            "v_fma_f32" | "v_fmaak_f32" | "v_fmamk_f32" => {
                // shortcut: DXIL has no fused f32 multiply-add; FMad lets the driver fuse.
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = self.fused(a, bb, cc);
                float(self, r)
            }
            "v_div_scale_f32" => {
                // RDNA2 ISA: src0 = value to scale, src1 = denominator, src2 = numerator.
                let (a, den, num) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let (ai, ni, di) = (self.as_int(a), self.as_int(num), self.as_int(den));
                let (k23, k255, kabs) = (
                    self.s.m.i32(23),
                    self.s.m.i32(0xFF),
                    self.s.m.i32(0x7FFF_FFFF),
                );
                let (zero, k23_exp, k232_exp, k96) = (
                    self.s.m.i32(0),
                    self.s.m.i32(23),
                    self.s.m.i32(232),
                    self.s.m.i32(96),
                );
                let exp_of = |l: &mut Self, x: V| {
                    let s = l.s.m.bin(c, Bin::LShr, x, k23);
                    l.s.m.bin(c, Bin::And, s, k255)
                };
                let (e1, e2) = (exp_of(self, ni), exp_of(self, di));
                let n_abs = self.s.m.bin(c, Bin::And, ni, kabs);
                let d_abs = self.s.m.bin(c, Bin::And, di, kabs);
                let z1 = self.s.m.cmp(c, pred::EQ, n_abs, zero);
                let z2 = self.s.m.cmp(c, pred::EQ, d_abs, zero);
                let any_zero = self.s.m.bin(c, Bin::Or, z1, z2);
                let special = {
                    let s1 = self.s.m.cmp(c, pred::EQ, e1, k255);
                    let s2 = self.s.m.cmp(c, pred::EQ, e2, k255);
                    let a12 = self.s.m.bin(c, Bin::Or, s1, s2);
                    self.s.m.bin(c, Bin::Or, a12, any_zero)
                };
                let is_den = self.s.m.cmp(c, pred::EQ, ai, di);
                let diff = self.s.m.bin(c, Bin::Sub, e2, e1);
                let den_tiny = self.s.m.cmp(c, pred::IULE, e2, k23_exp);
                let den_huge = self.s.m.cmp(c, pred::IUGE, e2, k232_exp);
                let num_tiny_exp = self.s.m.cmp(c, pred::IULE, e1, k23_exp);
                let num_huge = self.s.m.cmp(c, pred::IUGE, e1, k232_exp);
                let num_gap = self.s.m.cmp(c, pred::SGE, diff, k96);
                let not_den_huge = {
                    let t = self.s.m.bool(true);
                    self.s.m.bin(c, Bin::Xor, den_huge, t)
                };
                let num_gap = self.s.m.bin(c, Bin::And, num_gap, not_den_huge);
                let num_tiny = self.s.m.bin(c, Bin::Or, num_tiny_exp, num_gap);
                let up_raw = self.s.m.select(c, is_den, den_tiny, num_tiny);
                let dn_raw = self.s.m.select(c, is_den, den_huge, num_huge);
                let not_special = {
                    let t = self.s.m.bool(true);
                    self.s.m.bin(c, Bin::Xor, special, t)
                };
                let up = self.s.m.bin(c, Bin::And, up_raw, not_special);
                let dn = self.s.m.bin(c, Bin::And, dn_raw, not_special);
                let uncancelled = {
                    let x_tiny = self.s.m.bin(c, Bin::Xor, den_tiny, num_tiny);
                    let x_huge = self.s.m.bin(c, Bin::Xor, den_huge, num_huge);
                    self.s.m.bin(c, Bin::Or, x_tiny, x_huge)
                };
                let flag_raw = self.s.m.select(c, is_den, uncancelled, up_raw);
                let flag = self.s.m.bin(c, Bin::And, flag_raw, not_special);
                let k_up = self.s.m.f32(0x5F80_0000);
                let k_dn = self.s.m.f32(0x1F80_0000);
                let a_up = self.s.m.bin(c, Bin::Mul, a, k_up);
                let a_dn = self.s.m.bin(c, Bin::Mul, a, k_dn);
                let r = self.s.m.select(c, dn, a_dn, a);
                let r = self.s.m.select(c, up, a_up, r);
                let qnan = self.s.m.f32(0x7FC0_0000);
                let r = self.s.m.select(c, any_zero, qnan, r);
                float(self, r)?;
                if let Some(Operand::Scalar { code, .. }) = v.carry
                    && let Some(w) = inst.write(Reg::S(code))
                {
                    self.lane_bits.insert(w, flag);
                }
                for &(_, w) in &inst.writes {
                    self.vals
                        .entry(w)
                        .or_insert(Val::Missing("a lane mask read as data"));
                }
                Ok(())
            }
            "v_div_fmas_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let m = inst
                    .read(Reg::S(crate::ir::VCC))
                    .and_then(|v| self.x.st.masks.get(&v).copied())
                    .ok_or("v_div_fmas_f32 without VCC")?;
                let vcc = self.boolean(m, c)?;
                let tmp = self.fused(a, bb, cc);
                let ci = self.as_int(cc);
                let (k23, k255, k128) = (self.s.m.i32(23), self.s.m.i32(0xFF), self.s.m.i32(128));
                let sh = self.s.m.bin(c, Bin::LShr, ci, k23);
                let exp = self.s.m.bin(c, Bin::And, sh, k255);
                let exp_ge = self.s.m.cmp(c, pred::IUGE, exp, k128);
                let (k_up, k_dn) = (self.s.m.f32(0x5F80_0000), self.s.m.f32(0x1F80_0000));
                let scale = self.s.m.select(c, exp_ge, k_up, k_dn);
                let scaled = self.s.m.bin(c, Bin::Mul, tmp, scale);
                let r = self.s.m.select(c, vcc, scaled, tmp);
                float(self, r)
            }
            "v_div_fixup_f32" => {
                let (q, den, num) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let (qi, di, ni) = (self.as_int(q), self.as_int(den), self.as_int(num));
                let (ksign, kabs, kinf) = (
                    self.s.m.i32(0x8000_0000),
                    self.s.m.i32(0x7FFF_FFFF),
                    self.s.m.i32(0x7F80_0000),
                );
                let (kqnan_bit, kneg_qnan, zero) = (
                    self.s.m.i32(0x0040_0000),
                    self.s.m.i32(0xFFC0_0000),
                    self.s.m.i32(0),
                );
                let sign = {
                    let x = self.s.m.bin(c, Bin::Xor, ni, di);
                    self.s.m.bin(c, Bin::And, x, ksign)
                };
                let q_abs = self.s.m.bin(c, Bin::And, qi, kabs);
                let d_abs = self.s.m.bin(c, Bin::And, di, kabs);
                let n_abs = self.s.m.bin(c, Bin::And, ni, kabs);
                let q_nan = self.s.m.cmp(c, pred::IUGT, q_abs, kinf);
                let d_nan = self.s.m.cmp(c, pred::IUGT, d_abs, kinf);
                let n_nan = self.s.m.cmp(c, pred::IUGT, n_abs, kinf);
                let d_inf = self.s.m.cmp(c, pred::EQ, d_abs, kinf);
                let n_inf = self.s.m.cmp(c, pred::EQ, n_abs, kinf);
                let d_zero = self.s.m.cmp(c, pred::EQ, d_abs, zero);
                let n_zero = self.s.m.cmp(c, pred::EQ, n_abs, zero);
                let both_inf = self.s.m.bin(c, Bin::And, n_inf, d_inf);
                let both_zero = self.s.m.bin(c, Bin::And, n_zero, d_zero);
                let invalid = self.s.m.bin(c, Bin::Or, both_inf, both_zero);
                let make_inf = {
                    let t = self.s.m.bin(c, Bin::Or, n_inf, d_zero);
                    self.s.m.bin(c, Bin::Or, t, q_nan)
                };
                let make_zero = self.s.m.bin(c, Bin::Or, d_inf, n_zero);
                let signed_q = self.s.m.bin(c, Bin::Or, sign, q_abs);
                let signed_inf = self.s.m.bin(c, Bin::Or, sign, kinf);
                let n_qnan = self.s.m.bin(c, Bin::Or, ni, kqnan_bit);
                let d_qnan = self.s.m.bin(c, Bin::Or, di, kqnan_bit);
                let r = self.s.m.select(c, make_zero, sign, signed_q);
                let r = self.s.m.select(c, make_inf, signed_inf, r);
                let r = self.s.m.select(c, invalid, kneg_qnan, r);
                let r = self.s.m.select(c, d_nan, d_qnan, r);
                let r = self.s.m.select(c, n_nan, n_qnan, r);
                let rf = self.as_float(r);
                float(self, rf)
            }
            "v_fmac_f32" | "v_mac_f32" | "v_fmac_legacy_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("an accumulator");
                };
                let acc = inst.read(Reg::V(reg)).ok_or("an accumulator")?;
                let acc = match self.floats.get(&acc) {
                    Some(&x) => x,
                    None => {
                        let x = self.i32_of(acc)?;
                        self.as_float(x)
                    }
                };
                let r = match name {
                    "v_fmac_f32" => self.fused(a, bb, acc),
                    "v_mac_f32" => {
                        let m = self.s.m.bin(c, Bin::Mul, a, bb);
                        self.s.m.bin(c, Bin::Add, m, acc)
                    }
                    _ => self.fma_legacy(a, bb, acc),
                };
                float(self, r)
            }
            "v_fma_legacy_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = self.fma_legacy(a, bb, cc);
                float(self, r)
            }
            "v_mad_f32" | "v_madak_f32" | "v_madmk_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let m = self.s.m.bin(c, Bin::Mul, a, bb);
                let r = self.s.m.bin(c, Bin::Add, m, cc);
                float(self, r)
            }
            "v_max3_f32" | "v_min3_f32" | "v_med3_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = match name {
                    "v_max3_f32" => {
                        let t = self.max_min(true, a, bb);
                        self.max_min(true, t, cc)
                    }
                    "v_min3_f32" => {
                        let t = self.max_min(false, a, bb);
                        self.max_min(false, t, cc)
                    }
                    _ => self.med3(a, bb, cc),
                };
                float(self, r)
            }
            "v_rcp_f32" | "v_rcp_iflag_f32" => {
                // shortcut: the hardware reciprocal is approximate (1 ULP); this divides.
                let a = f(self, 0)?;
                let one = self.s.m.f32(1.0f32.to_bits());
                let r = self.s.m.bin(c, Bin::SDiv, one, a);
                float(self, r)
            }
            "v_rsq_f32" | "v_sqrt_f32" | "v_log_f32" | "v_exp_f32" | "v_fract_f32"
            | "v_trunc_f32" | "v_floor_f32" | "v_ceil_f32" | "v_rndne_f32" | "v_sin_f32"
            | "v_cos_f32" => {
                let a = f(self, 0)?;
                let op = match name {
                    "v_rsq_f32" => 25,
                    "v_sqrt_f32" => 24,
                    "v_log_f32" => 23,
                    "v_exp_f32" => 21,
                    "v_fract_f32" => 22,
                    "v_trunc_f32" => 29,
                    "v_floor_f32" => 27,
                    "v_ceil_f32" => 28,
                    "v_rndne_f32" => 26,
                    "v_sin_f32" => 13,
                    _ => 12,
                };
                let r = match op {
                    12 | 13 => self.trig_turns(a, op == 12),
                    22 => self.fract(a, 1.0, f32::from_bits(0x3F7F_FFFF)),
                    _ => self.unary(op, a),
                };
                float(self, r)
            }
            "v_cvt_i32_f32" | "v_cvt_u32_f32" | "v_cvt_flr_i32_f32" | "v_cvt_rpi_i32_f32" => {
                let mut a = f(self, 0)?;
                if name == "v_cvt_rpi_i32_f32" {
                    let h = self.s.m.f32(0.5f32.to_bits());
                    a = self.s.m.bin(c, Bin::Add, a, h);
                }
                if name != "v_cvt_i32_f32" && name != "v_cvt_u32_f32" {
                    a = self.unary(27, a);
                }
                // D3D's conversions saturate and give 0 for NaN, as the
                // hardware does (checked on the RX 6700 XT and WARP).
                let op = if name == "v_cvt_u32_f32" {
                    Cast::FpToUi
                } else {
                    Cast::FpToSi
                };
                let r = self.s.m.cast(c, op, a, self.s.t.i32);
                int(self, r)
            }
            "v_cvt_f32_i32" | "v_cvt_f32_u32" | "v_cvt_f32_ubyte0" | "v_cvt_f32_ubyte1"
            | "v_cvt_f32_ubyte2" | "v_cvt_f32_ubyte3" => {
                let mut a = s(self, 0)?;
                if let Some(n) = name.strip_prefix("v_cvt_f32_ubyte") {
                    let sh = self.s.m.i32(8 * u32::from(n.as_bytes()[0] - b'0'));
                    let ff = self.s.m.i32(0xff);
                    a = self.s.m.bin(c, Bin::LShr, a, sh);
                    a = self.s.m.bin(c, Bin::And, a, ff);
                }
                let op = if name == "v_cvt_f32_i32" {
                    Cast::SiToFp
                } else {
                    Cast::UiToFp
                };
                let r = self.s.m.cast(c, op, a, self.s.t.f32);
                float(self, r)
            }
            "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => {
                // Each source is f32, or with op_sel_hi the low (op_sel: high)
                // f16 half; neg_hi is the absolute value here.
                let mut xs = Vec::new();
                for i in 0..3 {
                    let op = v.src.get(i).ok_or("a missing operand")?;
                    let x = self.operand(inst, op)?;
                    let mut x = if v.opsel_hi >> i & 1 != 0 {
                        let h = if v.opsel >> i & 1 != 0 {
                            let k16 = self.s.m.i32(16);
                            self.s.m.bin(c, Bin::LShr, x, k16)
                        } else {
                            x
                        };
                        self.s
                            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h])
                    } else {
                        self.as_float(x)
                    };
                    if v.neg_hi >> i & 1 != 0 {
                        x = self.unary(6, x);
                    }
                    if v.neg >> i & 1 != 0 {
                        let z = self.s.m.f32(0x8000_0000);
                        x = self.s.m.bin(c, Bin::Sub, z, x);
                    }
                    xs.push(x);
                }
                // f16 sources are exact in f32 (an f16 product fits), so
                // only the f32 form needs the exact path; it is fused.
                let mut r = self.fused(xs[0], xs[1], xs[2]);
                if name == "v_fma_mix_f32" {
                    return float(self, r);
                }
                if v.clamp {
                    r = self.unary(7, r);
                }
                // The f16 result goes to one half; the other half stays.
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a mixed-precision result");
                };
                let old = match inst.read(Reg::V(reg)) {
                    Some(o) => self.i32_of(o)?,
                    None => self.s.m.i32(0),
                };
                let h = self
                    .s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[r]);
                let (keep, h) = if name == "v_fma_mixhi_f16" {
                    let k16 = self.s.m.i32(16);
                    (0xFFFF, self.s.m.bin(c, Bin::Shl, h, k16))
                } else {
                    (0xFFFF_0000, h)
                };
                let keep = self.s.m.i32(keep);
                let old = self.s.m.bin(c, Bin::And, old, keep);
                let x = self.s.m.bin(c, Bin::Or, old, h);
                self.write(inst, Reg::V(reg), x, guard);
                Ok(())
            }
            "v_cvt_pkrtz_f16_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let lo = self.f16_toward_zero(a);
                let hi = self.f16_toward_zero(bb);
                let sh = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, hi, sh);
                let r = self.s.m.bin(c, Bin::Or, lo, hi);
                int(self, r)
            }
            "v_cvt_f16_f32" if v.opsel == 0 => {
                // Rounded to nearest even (the MODE default) into the low half.
                let a = f(self, 0)?;
                let r = self
                    .s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[a]);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a 16-bit result outside a VGPR");
                };
                let r = self.low_half(inst, reg, r);
                int(self, r)
            }
            "v_pack_b32_f16" => {
                // Two f16 sources (op_sel: their high halves; abs and neg on
                // the sign bit) into the low and high half.
                let mut halves = Vec::new();
                for k in 0..2 {
                    let op = v.src.get(k).ok_or("a missing operand")?;
                    let mut x = self.operand(inst, op)?;
                    if v.opsel >> k & 1 != 0 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let mask = if v.abs >> k & 1 != 0 { 0x7FFF } else { 0xFFFF };
                    let mask = self.s.m.i32(mask);
                    x = self.s.m.bin(c, Bin::And, x, mask);
                    if v.neg >> k & 1 != 0 {
                        let sign = self.s.m.i32(0x8000);
                        x = self.s.m.bin(c, Bin::Xor, x, sign);
                    }
                    halves.push(x);
                }
                let k16 = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
                let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
                int(self, r)
            }
            "v_cvt_f32_f16" if v.opsel == 0 => {
                let a = s(self, 0)?;
                let r = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[a]);
                float(self, r)
            }
            "v_cndmask_b32" => {
                let cond =
                    self.mask_cond(inst, v.src.get(2), c, "a select on an untracked mask")?;
                if v.abs != 0 || v.neg != 0 {
                    let (a, bb) = (f(self, 0)?, f(self, 1)?);
                    let r = self.s.m.select(c, cond, bb, a);
                    return float(self, r);
                }
                let (a, bb) = (s(self, 0)?, s(self, 1)?);
                let r = self.s.m.select(c, cond, bb, a);
                int(self, r)
            }
            "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" | "v_add_co_ci_u32"
            | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32" => {
                // The carry (borrow) out is a lane mask: this lane's bit.
                let (mut a, mut bb) = (s(self, 0)?, s(self, 1)?);
                if name.contains("subrev") {
                    std::mem::swap(&mut a, &mut bb);
                }
                let cin = if name.ends_with("_ci_u32") {
                    Some(self.mask_cond(
                        inst,
                        v.src.get(2),
                        c,
                        "a carry-in from an untracked mask",
                    )?)
                } else {
                    None
                };
                let (r, carry) = self.add_sub_carry(c, name.starts_with("v_add"), a, bb, cin);
                int(self, r)?;
                if let Some(Operand::Scalar { code, .. }) = v.carry
                    && let Some(w) = inst.write(Reg::S(code))
                {
                    self.lane_bits.insert(w, carry);
                }
                for &(_, w) in &inst.writes {
                    self.vals
                        .entry(w)
                        .or_insert(Val::Missing("a lane mask read as data"));
                }
                Ok(())
            }
            "v_readlane_b32" => {
                // XNW: a constant lane of a VGPR the program spilled an
                // SGPR to (`v_writelane_b32`) reads the SGPR spilled there.
                // (Outside compute only: compute reads lanes for real, and
                // the search through joins is costly on its large programs.)
                if self.s.threads.is_none()
                    && let (Some(src), Some(l)) = (v.src.first(), v.src.get(1))
                    && let Some(sv) = self.spilled_value(inst, src, l)
                {
                    let r = self.i32_of(sv)?;
                    return int(self, r);
                }
                let (a, lane) = (s(self, 0)?, s(self, 1)?);
                // A computed lane in a pixel or vertex program is a host lane
                // (from a mask or `v_mbcnt`); a constant one is the console's.
                if self.mask_data == MaskData::Host && !matches!(lane, V::Const(_)) {
                    self.s.flags |= flags::WAVE_OPS;
                } else {
                    self.cross_lane(false)?;
                }
                let r = self.read_lane(a, lane);
                int(self, r)
            }
            "v_writelane_b32" => {
                // XNW: outside compute, a lane write is only an SGPR
                // spill, which `v_readlane_b32` reads back as the SGPR; the
                // VGPR itself is left without a value.
                if self.s.threads.is_none() {
                    return Err("a lane write outside compute");
                }
                // The lane given takes the scalar; the others keep their value.
                let (a, lane) = (s(self, 0)?, s(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a lane write");
                };
                let old = match inst.read(Reg::V(reg)) {
                    Some(o) => self.i32_of(o)?,
                    None => self.s.m.i32(0),
                };
                self.cross_lane(false)?;
                let me = self.lane_index();
                let is = self.s.m.cmp(c, pred::EQ, me, lane);
                let r = self.s.m.select(c, is, a, old);
                int(self, r)
            }
            "v_readfirstlane_b32" => {
                let a = s(self, 0)?;
                self.s.flags |= flags::WAVE_OPS;
                let r = self.s.op(
                    c,
                    "waveReadLaneFirst.i32",
                    Attr::NoUnwind,
                    self.s.t.i32,
                    118,
                    &[a],
                );
                let r = self.helper_reads_own(a, r);
                int(self, r)
            }
            _ => {
                let r = self.integer(inst, name)?;
                int(self, r)
            }
        }
    }

    /// A 64-bit source (a register pair; inline constants sign-extend).
    fn src64(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 || v.sdwa.is_some() || v.dpp.is_some() {
            return Err("64-bit operand modifiers");
        }
        let c = self.cur;
        let i64t = self.s.m.int(64);
        self.s.flags |= flags::INT64;
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let pair = |l: &mut Self, r: Reg, r1: Reg| -> Result<V, Why> {
            let lo = l.i32_of(inst.read(r).ok_or("an unread operand")?)?;
            let hi = l.i32_of(inst.read(r1).ok_or("an unread operand")?)?;
            let lo = l.s.m.cast(c, Cast::ZExt, lo, i64t);
            let hi = l.s.m.cast(c, Cast::ZExt, hi, i64t);
            let k32 = l.s.m.int_const(i64t, 32);
            let hi = l.s.m.bin(c, Bin::Shl, hi, k32);
            Ok(l.s.m.bin(c, Bin::Or, lo, hi))
        };
        match op {
            Operand::Vector { reg, .. } => pair(self, Reg::V(reg), Reg::V(reg + 1)),
            Operand::Scalar { code, .. } if code != NULL => {
                pair(self, Reg::S(code), Reg::S(code + 1))
            }
            Operand::Const(k @ 128..=208) => {
                let x = crate::resource::inline(k).ok_or("a special operand")?;
                Ok(self.s.m.int_const(i64t, i64::from(x as i32)))
            }
            _ => {
                // Literals and the null register: zero-extended.
                let x = self.operand(inst, &op)?;
                Ok(self.s.m.cast(c, Cast::ZExt, x, i64t))
            }
        }
    }

    /// The low and high dwords of an i64.
    fn split64(&mut self, x: V) -> (V, V) {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let k32 = self.s.m.int_const(i64t, 32);
        let lo = self.s.m.cast(c, Cast::Trunc, x, self.s.t.i32);
        let hi = self.s.m.bin(c, Bin::LShr, x, k32);
        let hi = self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32);
        (lo, hi)
    }

    /// Source `k` of a 16-bit operation in the low 16 bits (the rest 0):
    /// op_sel (or an SDWA word select) picks the high half; inline float
    /// constants are f16 values here.
    fn half_src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let mut high = v.opsel >> k & 1 != 0;
        if let Some(sd) = v.sdwa {
            match sd.src_sel.get(k).copied().unwrap_or(6) {
                4 => {}
                5 => high = true,
                6 => {}
                _ => return Err("a byte select on a 16-bit operation"),
            }
        }
        let x = match self.src0 {
            Some(x) if k == 0 => x,
            _ => self.f16_operand(inst, &op)?,
        };
        let c = self.cur;
        let x = if high {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// An f16 source as an f32 (exact), with its abs/neg modifiers.
    fn half_fsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let h = self.half_src(inst, k)?;
        let f32t = self.s.t.f32;
        let mut f = self
            .s
            .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
        if v.abs >> k & 1 != 0 {
            f = self.unary(6, f);
        }
        if v.neg >> k & 1 != 0 {
            let z = self.s.m.f32(0x8000_0000);
            f = self.s.m.bin(self.cur, Bin::Sub, z, f);
        }
        Ok(f)
    }

    /// Writes a 16-bit result (low bits of `x`) into the destination's low
    /// half, or its high half with op_sel bit 3 (or an SDWA WORD_1 select);
    /// the other half is kept (GFX10).
    fn write_half(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a 16-bit result outside a VGPR");
        };
        let mut high = v.opsel >> 3 & 1 != 0;
        // SDWA: the word written, the other one kept (preserve) or zero
        // (pad; a sign extension above the high word adds nothing).
        let mut keep = true;
        if let Some(sd) = v.sdwa {
            match (sd.dst_sel, sd.dst_unused) {
                (4, 2) => {}
                (5, 2) => high = true,
                (4 | 6, _) => keep = false,
                (5, _) => (high, keep) = (true, false),
                _ => return Err("an SDWA destination of a 16-bit operation"),
            }
        }
        let c = self.cur;
        let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
            Some(Ok(o)) if keep => o,
            _ => self.s.m.i32(0),
        };
        let (k16, low) = (self.s.m.i32(16), self.s.m.i32(0xFFFF));
        let x = self.s.m.bin(c, Bin::And, x, low);
        let r = if high {
            let o = self.s.m.bin(c, Bin::And, old, low);
            let x = self.s.m.bin(c, Bin::Shl, x, k16);
            self.s.m.bin(c, Bin::Or, o, x)
        } else {
            let keep = self.s.m.i32(0xFFFF_0000);
            let o = self.s.m.bin(c, Bin::And, old, keep);
            self.s.m.bin(c, Bin::Or, o, x)
        };
        self.write_placed(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    /// 16-bit operations, computed in f32 or i32: f16 sums, differences and
    /// products are exact there and round once to f16 (f32 has the 2 x 11 +
    /// 2 bits that make double rounding harmless); `None` if `name` is none.
    fn valu16(&mut self, inst: &Inst, name: &str, guard: Option<V>) -> Option<Result<(), Why>> {
        const FLOAT2: &[&str] = &[
            "v_add_f16",
            "v_sub_f16",
            "v_subrev_f16",
            "v_mul_f16",
            "v_max_f16",
            "v_min_f16",
        ];
        const FLOAT3: &[&str] = &[
            "v_fma_f16",
            "v_mad_f16",
            "v_fmac_f16",
            "v_mac_f16",
            "v_max3_f16",
            "v_min3_f16",
            "v_med3_f16",
        ];
        const FLOAT1: &[&str] = &[
            "v_rcp_f16",
            "v_sqrt_f16",
            "v_rsq_f16",
            "v_log_f16",
            "v_exp_f16",
            "v_fract_f16",
            "v_trunc_f16",
            "v_floor_f16",
            "v_ceil_f16",
            "v_rndne_f16",
            "v_sin_f16",
            "v_cos_f16",
            "v_cvt_f16_u16",
            "v_cvt_f16_i16",
        ];
        const INT: &[&str] = &[
            "v_add_nc_u16",
            "v_sub_nc_u16",
            "v_subrev_nc_u16",
            "v_mul_lo_u16",
            "v_lshlrev_b16",
            "v_lshrrev_b16",
            "v_ashrrev_i16",
            "v_max_u16",
            "v_max_i16",
            "v_min_u16",
            "v_min_i16",
            "v_cvt_u16_f16",
            "v_cvt_i16_f16",
            "v_cvt_norm_i16_f16",
            "v_cvt_norm_u16_f16",
        ];
        let all = [FLOAT1, FLOAT2, FLOAT3, INT];
        if !all.iter().any(|l| l.contains(&name))
            && !matches!(name, "v_cvt_f16_f32" | "v_cvt_f32_f16")
        {
            return None;
        }
        Some(self.valu16_op(inst, name, guard, FLOAT1, FLOAT2, FLOAT3))
    }

    #[allow(clippy::too_many_arguments)]
    fn valu16_op(
        &mut self,
        inst: &Inst,
        name: &str,
        guard: Option<V>,
        float1: &[&str],
        float2: &[&str],
        float3: &[&str],
    ) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let i32t = self.s.t.i32;
        let f32t = self.s.t.f32;
        if name == "v_cvt_f32_f16" {
            let f = self.half_fsrc(inst, 0)?;
            return self.write_f(inst, f, guard);
        }
        let float_result = |l: &mut Self, mut f: V| -> Result<(), Why> {
            if v.omod != 0 {
                return Err("an output modifier on a 16-bit operation");
            }
            if v.clamp {
                f = l.unary(7, f);
            }
            let h =
                l.s.op(l.cur, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
            l.write_half(inst, h, guard)
        };
        if name == "v_cvt_f16_f32" {
            let f = self.fsrc(inst, 0)?;
            return float_result(self, f);
        }
        if float2.contains(&name) {
            let (a, b) = (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?);
            let r = match name {
                "v_add_f16" => self.s.m.bin(c, Bin::Add, a, b),
                "v_sub_f16" => self.s.m.bin(c, Bin::Sub, a, b),
                "v_subrev_f16" => self.s.m.bin(c, Bin::Sub, b, a),
                "v_mul_f16" => self.s.m.bin(c, Bin::Mul, a, b),
                "v_max_f16" => self.max_min(true, a, b),
                _ => self.max_min(false, a, b),
            };
            return float_result(self, r);
        }
        if float3.contains(&name) {
            let (a, b) = (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?);
            let acc = if matches!(name, "v_fmac_f16" | "v_mac_f16") {
                // The accumulator is the destination's half that is written.
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("an accumulator");
                };
                let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("an accumulator")?)?;
                let x = if v.opsel >> 3 & 1 != 0 {
                    let k16 = self.s.m.i32(16);
                    self.s.m.bin(c, Bin::LShr, x, k16)
                } else {
                    x
                };
                let m = self.s.m.i32(0xFFFF);
                let x = self.s.m.bin(c, Bin::And, x, m);
                self.s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
            } else {
                self.half_fsrc(inst, 2)?
            };
            let r = match name {
                // shortcut: fused f16 multiply-adds as FMad in f32 (the
                // product is exact; the sum rounds twice).
                "v_fma_f16" | "v_fmac_f16" | "v_mad_f16" | "v_mac_f16" => self.fmad(a, b, acc),
                "v_max3_f16" => {
                    let t = self.max_min(true, a, b);
                    self.max_min(true, t, acc)
                }
                "v_min3_f16" => {
                    let t = self.max_min(false, a, b);
                    self.max_min(false, t, acc)
                }
                _ => self.med3(a, b, acc),
            };
            return float_result(self, r);
        }
        if float1.contains(&name) {
            let r = match name {
                "v_cvt_f16_u16" | "v_cvt_f16_i16" => {
                    let mut x = self.half_src(inst, 0)?;
                    let op = if name == "v_cvt_f16_i16" {
                        let k16 = self.s.m.i32(16);
                        let t = self.s.m.bin(c, Bin::Shl, x, k16);
                        x = self.s.m.bin(c, Bin::AShr, t, k16);
                        Cast::SiToFp
                    } else {
                        Cast::UiToFp
                    };
                    self.s.m.cast(c, op, x, f32t)
                }
                "v_rcp_f16" => {
                    let a = self.half_fsrc(inst, 0)?;
                    let one = self.s.m.f32(1.0f32.to_bits());
                    self.s.m.bin(c, Bin::SDiv, one, a)
                }
                _ => {
                    let a = self.half_fsrc(inst, 0)?;
                    let op = match name {
                        "v_sqrt_f16" => 24,
                        "v_rsq_f16" => 25,
                        "v_log_f16" => 23,
                        "v_exp_f16" => 21,
                        "v_fract_f16" => 22,
                        "v_trunc_f16" => 29,
                        "v_floor_f16" => 27,
                        "v_ceil_f16" => 28,
                        "v_rndne_f16" => 26,
                        "v_sin_f16" => 13,
                        _ => 12,
                    };
                    match op {
                        12 | 13 => self.trig_turns(a, op == 12),
                        // Any f16 in (-2^-11, 0) has fract >= 1 - 2^-12; clamp to 0x3bff (1 - 2^-11).
                        22 => self.fract(a, 1.0 - 1.0 / 4096.0, 1.0 - 1.0 / 2048.0),
                        _ => self.unary(op, a),
                    }
                }
            };
            return float_result(self, r);
        }
        if matches!(name, "v_cvt_norm_i16_f16" | "v_cvt_norm_u16_f16") {
            if v.clamp || v.omod != 0 {
                return Err("modifiers on a 16-bit integer operation");
            }
            let r = self.cvt_norm16(inst, name == "v_cvt_norm_i16_f16")?;
            return self.write_half(inst, r, guard);
        }
        // 16-bit integers: computed on the low halves, the low 16 bits kept.
        if v.clamp || v.omod != 0 || v.abs != 0 || v.neg != 0 {
            return Err("modifiers on a 16-bit integer operation");
        }
        let r = match name {
            "v_cvt_u16_f16" | "v_cvt_i16_f16" => {
                // Toward zero, saturating at the 16-bit range (infinities
                // included); NaN gives 0. Clamped in f32 first, where every
                // f16 is exact, so the conversion never leaves i32's range.
                let a = self.half_fsrc(inst, 0)?;
                let (lo, hi) = if name == "v_cvt_u16_f16" {
                    (0.0f32, 65535.0f32)
                } else {
                    (-32768.0, 32767.0)
                };
                let (lo, hi) = (self.s.m.f32(lo.to_bits()), self.s.m.f32(hi.to_bits()));
                let t = self.binary(35, a, lo);
                let t = self.binary(36, t, hi);
                let r = self.s.m.cast(c, Cast::FpToSi, t, i32t);
                let nan = self.s.m.cmp(c, pred::UNO, a, a);
                let z = self.s.m.i32(0);
                self.s.m.select(c, nan, z, r)
            }
            _ => {
                let (mut a, mut b) = (self.half_src(inst, 0)?, self.half_src(inst, 1)?);
                let k16 = self.s.m.i32(16);
                let sext = |l: &mut Self, x: V| {
                    let t = l.s.m.bin(c, Bin::Shl, x, k16);
                    l.s.m.bin(c, Bin::AShr, t, k16)
                };
                if name.ends_with("_i16") {
                    a = sext(self, a);
                    b = sext(self, b);
                }
                let k15 = self.s.m.i32(15);
                match name {
                    "v_add_nc_u16" => self.s.m.bin(c, Bin::Add, a, b),
                    "v_sub_nc_u16" => self.s.m.bin(c, Bin::Sub, a, b),
                    "v_subrev_nc_u16" => self.s.m.bin(c, Bin::Sub, b, a),
                    "v_mul_lo_u16" => self.s.m.bin(c, Bin::Mul, a, b),
                    "v_lshlrev_b16" | "v_lshrrev_b16" | "v_ashrrev_i16" => {
                        // src0 is the shift (4 bits), src1 the value.
                        let sh = self.s.m.bin(c, Bin::And, a, k15);
                        let x = if name == "v_ashrrev_i16" {
                            sext(self, b)
                        } else {
                            b
                        };
                        let op = match name {
                            "v_lshlrev_b16" => Bin::Shl,
                            "v_lshrrev_b16" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        self.s.m.bin(c, op, x, sh)
                    }
                    _ => {
                        let p = match name {
                            "v_max_u16" => pred::IUGT,
                            "v_min_u16" => pred::IULT,
                            "v_max_i16" => pred::SGT,
                            _ => pred::SLT,
                        };
                        let t = self.s.m.cmp(c, p, a, b);
                        self.s.m.select(c, t, a, b)
                    }
                }
            }
        };
        self.write_half(inst, r, guard)
    }

    /// `v_cvt_norm_i16_f16` / `v_cvt_norm_u16_f16`: the f16 source (with its
    /// abs/neg modifiers) clamped to [-1, 1] or [0, 1], times 32767 or 65535,
    /// rounded to nearest even once from the exact product; NaN gives 0.
    /// Worked on the f16's bits in integers, because in f32 the product
    /// (up to 26 significant bits) would round before the rounding to an
    /// integer.
    fn cvt_norm16(&mut self, inst: &Inst, signed: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let mut h = self.half_src(inst, 0)?;
        if v.abs & 1 != 0 {
            let m = k(self, 0x7FFF);
            h = self.s.m.bin(c, Bin::And, h, m);
        }
        if v.neg & 1 != 0 {
            let m = k(self, 0x8000);
            h = self.s.m.bin(c, Bin::Xor, h, m);
        }
        let scale = if signed { 0x7FFF } else { 0xFFFF };
        let (k0, k1, k10) = (k(self, 0), k(self, 1), k(self, 10));
        let mag_mask = k(self, 0x7FFF);
        let mag = self.s.m.bin(c, Bin::And, h, mag_mask);
        let k15 = k(self, 15);
        let neg = self.s.m.bin(c, Bin::LShr, h, k15);
        let neg = self.s.m.cmp(c, pred::NE, neg, k0);
        // |x| < 1: x = m * 2^-s with m < 2^11 and 11 <= s <= 24.
        let exp = self.s.m.bin(c, Bin::LShr, mag, k10);
        let man_mask = k(self, 0x3FF);
        let man = self.s.m.bin(c, Bin::And, mag, man_mask);
        let implicit = k(self, 0x400);
        let normal = self.s.m.bin(c, Bin::Or, man, implicit);
        let sub = self.s.m.cmp(c, pred::EQ, exp, k0);
        let m = self.s.m.select(c, sub, man, normal);
        let k25 = k(self, 25);
        let s_norm = self.s.m.bin(c, Bin::Sub, k25, exp);
        let k24 = k(self, 24);
        let sh = self.s.m.select(c, sub, k24, s_norm);
        let ks = k(self, scale);
        let p = self.s.m.bin(c, Bin::Mul, m, ks);
        let q = self.s.m.bin(c, Bin::LShr, p, sh);
        let one_s = self.s.m.bin(c, Bin::Shl, k1, sh);
        let low_mask = self.s.m.bin(c, Bin::Sub, one_s, k1);
        let rem = self.s.m.bin(c, Bin::And, p, low_mask);
        let half = self.s.m.bin(c, Bin::LShr, one_s, k1);
        let above = self.s.m.cmp(c, pred::IUGT, rem, half);
        let tie = self.s.m.cmp(c, pred::EQ, rem, half);
        let odd = self.s.m.bin(c, Bin::And, q, k1);
        let odd = self.s.m.cmp(c, pred::NE, odd, k0);
        let tie_up = self.s.m.bin(c, Bin::And, tie, odd);
        let up = self.s.m.bin(c, Bin::Or, above, tie_up);
        let r = self.s.m.select(c, up, k1, k0);
        let r = self.s.m.bin(c, Bin::Add, q, r);
        // |x| >= 1 (and infinity) saturates; NaN gives 0.
        let one_h = k(self, 0x3C00);
        let big = self.s.m.cmp(c, pred::IUGE, mag, one_h);
        let r = self.s.m.select(c, big, ks, r);
        let inf = k(self, 0x7C00);
        let nan = self.s.m.cmp(c, pred::IUGT, mag, inf);
        let r = self.s.m.select(c, nan, k0, r);
        let r = if signed {
            let n = self.s.m.bin(c, Bin::Sub, k0, r);
            self.s.m.select(c, neg, n, r)
        } else {
            self.s.m.select(c, neg, k0, r)
        };
        let m16 = k(self, 0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, r, m16))
    }

    /// Half `high` of source `k` of a packed (VOP3P) operation, in the low
    /// 16 bits: op_sel picks the half the low result reads, op_sel_hi the
    /// half the high result reads. Packed integer instructions see the f32
    /// bits of an inline float constant.
    fn packed_src(&mut self, inst: &Inst, k: usize, high: bool, float: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let sel = if high { v.opsel_hi } else { v.opsel } >> k & 1 != 0;
        let x = if float {
            self.f16_operand(inst, &op)?
        } else {
            self.operand(inst, &op)?
        };
        let c = self.cur;
        let x = if sel {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// Packed 16-bit operations (`v_pk_*`): each half computed as the
    /// 16-bit operation does, f16 through f32 as in [`Self::valu16`].
    fn packed16(&mut self, inst: &Inst, op: &str, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let float = op.ends_with("_f16");
        let n = match op {
            "fma_f16" | "mad_u16" | "mad_i16" => 3,
            "add_f16" | "mul_f16" | "min_f16" | "max_f16" | "add_u16" | "sub_u16" | "add_i16"
            | "sub_i16" | "mul_lo_u16" | "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16"
            | "max_u16" | "min_u16" | "max_i16" | "min_i16" => 2,
            _ => return Err("a packed 16-bit operation"),
        };
        if v.omod != 0 || !float && (v.clamp || v.neg != 0 || v.neg_hi != 0) {
            return Err("modifiers on a packed integer operation");
        }
        let c = self.cur;
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let mut halves = Vec::new();
        for high in [false, true] {
            let mut xs = Vec::new();
            for k in 0..n {
                let h = self.packed_src(inst, k, high, float)?;
                if float {
                    let mut f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    let neg = if high { v.neg_hi } else { v.neg };
                    if neg >> k & 1 != 0 {
                        let z = self.s.m.f32(0x8000_0000);
                        f = self.s.m.bin(c, Bin::Sub, z, f);
                    }
                    xs.push(f);
                } else {
                    xs.push(h);
                }
            }
            let r = if float {
                let mut f = match op {
                    "add_f16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "mul_f16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "min_f16" => self.max_min(false, xs[0], xs[1]),
                    "max_f16" => self.max_min(true, xs[0], xs[1]),
                    // shortcut: FMad in f32, as v_fma_f16.
                    _ => self.fmad(xs[0], xs[1], xs[2]),
                };
                if v.clamp {
                    f = self.unary(7, f);
                }
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
            } else {
                let k16 = self.s.m.i32(16);
                let sext = |l: &mut Self, x: V| {
                    let t = l.s.m.bin(c, Bin::Shl, x, k16);
                    l.s.m.bin(c, Bin::AShr, t, k16)
                };
                if op.ends_with("_i16") {
                    for x in &mut xs {
                        *x = sext(self, *x);
                    }
                }
                let k15 = self.s.m.i32(15);
                match op {
                    "add_u16" | "add_i16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "sub_u16" | "sub_i16" => self.s.m.bin(c, Bin::Sub, xs[0], xs[1]),
                    "mul_lo_u16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "mad_u16" | "mad_i16" => {
                        let m = self.s.m.bin(c, Bin::Mul, xs[0], xs[1]);
                        self.s.m.bin(c, Bin::Add, m, xs[2])
                    }
                    "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16" => {
                        let sh = self.s.m.bin(c, Bin::And, xs[0], k15);
                        let o = match op {
                            "lshlrev_b16" => Bin::Shl,
                            "lshrrev_b16" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        let x = if op == "ashrrev_i16" {
                            sext(self, xs[1])
                        } else {
                            xs[1]
                        };
                        self.s.m.bin(c, o, x, sh)
                    }
                    _ => {
                        let p = match op {
                            "max_u16" => pred::IUGT,
                            "min_u16" => pred::IULT,
                            "max_i16" => pred::SGT,
                            _ => pred::SLT,
                        };
                        let t = self.s.m.cmp(c, p, xs[0], xs[1]);
                        self.s.m.select(c, t, xs[0], xs[1])
                    }
                }
            };
            let m = self.s.m.i32(0xFFFF);
            halves.push(self.s.m.bin(c, Bin::And, r, m));
        }
        let k16 = self.s.m.i32(16);
        let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
        let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a packed result outside a VGPR");
        };
        self.write(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    fn integer(&mut self, inst: &Inst, name: &'static str) -> Result<V, Why> {
        let c = self.cur;
        let s = |l: &mut Self, k| l.src(inst, k);
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let m31 = |l: &mut Self, x: V| {
            let k31 = l.s.m.i32(31);
            l.s.m.bin(c, Bin::And, x, k31)
        };
        Ok(match name {
            "v_and_b32" | "v_or_b32" | "v_xor_b32" | "v_add_nc_u32" | "v_add_nc_i32"
            | "v_sub_nc_u32" | "v_subrev_nc_u32" | "v_mul_lo_u32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let (op, a, b) = match name {
                    "v_and_b32" => (Bin::And, a, b),
                    "v_or_b32" => (Bin::Or, a, b),
                    "v_xor_b32" => (Bin::Xor, a, b),
                    "v_add_nc_u32" | "v_add_nc_i32" => (Bin::Add, a, b),
                    "v_sub_nc_u32" => (Bin::Sub, a, b),
                    "v_subrev_nc_u32" => (Bin::Sub, b, a),
                    _ => (Bin::Mul, a, b),
                };
                self.s.m.bin(c, op, a, b)
            }
            "v_xnor_b32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let x = self.s.m.bin(c, Bin::Xor, a, b);
                let ones = k(self, u32::MAX);
                self.s.m.bin(c, Bin::Xor, x, ones)
            }
            "v_not_b32" => {
                let a = s(self, 0)?;
                let ones = k(self, u32::MAX);
                self.s.m.bin(c, Bin::Xor, a, ones)
            }
            "v_lshlrev_b32" | "v_lshrrev_b32" | "v_ashrrev_i32" => {
                let (sh, a) = (s(self, 0)?, s(self, 1)?);
                let sh = m31(self, sh);
                let op = match name {
                    "v_lshlrev_b32" => Bin::Shl,
                    "v_lshrrev_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                self.s.m.bin(c, op, a, sh)
            }
            "v_lshl_add_u32" | "v_lshl_or_b32" | "v_add_lshl_u32" | "v_and_or_b32"
            | "v_or3_b32" | "v_xad_u32" | "v_add3_u32" | "v_xor3_b32" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                match name {
                    "v_lshl_add_u32" | "v_lshl_or_b32" => {
                        let b = m31(self, b);
                        let t = self.s.m.bin(c, Bin::Shl, a, b);
                        let op = if name == "v_lshl_add_u32" {
                            Bin::Add
                        } else {
                            Bin::Or
                        };
                        self.s.m.bin(c, op, t, d)
                    }
                    "v_add_lshl_u32" => {
                        let t = self.s.m.bin(c, Bin::Add, a, b);
                        let d = m31(self, d);
                        self.s.m.bin(c, Bin::Shl, t, d)
                    }
                    _ => {
                        let (o1, o2) = match name {
                            "v_and_or_b32" => (Bin::And, Bin::Or),
                            "v_or3_b32" => (Bin::Or, Bin::Or),
                            "v_xad_u32" => (Bin::Xor, Bin::Add),
                            "v_add3_u32" => (Bin::Add, Bin::Add),
                            _ => (Bin::Xor, Bin::Xor),
                        };
                        let t = self.s.m.bin(c, o1, a, b);
                        self.s.m.bin(c, o2, t, d)
                    }
                }
            }
            "v_mul_u32_u24" | "v_mul_i32_i24" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let (sh, op) = (
                    k(self, 8),
                    if name == "v_mul_i32_i24" {
                        Bin::AShr
                    } else {
                        Bin::LShr
                    },
                );
                let a = self.s.m.bin(c, Bin::Shl, a, sh);
                let a = self.s.m.bin(c, op, a, sh);
                let b = self.s.m.bin(c, Bin::Shl, b, sh);
                let b = self.s.m.bin(c, op, b, sh);
                self.s.m.bin(c, Bin::Mul, a, b)
            }
            "v_bfe_u32" | "v_bfe_i32" => {
                let (a, off, w) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let w = m31(self, w);
                self.bfe(a, off, w, name == "v_bfe_i32")
            }
            "v_bfi_b32" => {
                let (m, a, b) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let ones = k(self, u32::MAX);
                let nm = self.s.m.bin(c, Bin::Xor, m, ones);
                let x = self.s.m.bin(c, Bin::And, m, a);
                let y = self.s.m.bin(c, Bin::And, nm, b);
                self.s.m.bin(c, Bin::Or, x, y)
            }
            "v_max_i32" | "v_min_i32" | "v_max_u32" | "v_min_u32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let p = match name {
                    "v_max_i32" => pred::SGT,
                    "v_min_i32" => pred::SLT,
                    "v_max_u32" => pred::IUGT,
                    _ => pred::IULT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                self.s.m.select(c, t, a, b)
            }
            "v_mul_hi_u32" | "v_mul_hi_i32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                self.mul_hi(a, b, name == "v_mul_hi_i32")
            }
            "v_alignbit_b32" | "v_alignbyte_b32" => {
                // The low dword of {a, b} >> shift.
                let (a, b, sh) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let sh = if name == "v_alignbyte_b32" {
                    let three = k(self, 3);
                    let t = self.s.m.bin(c, Bin::And, sh, three);
                    let k3 = k(self, 3);
                    self.s.m.bin(c, Bin::Shl, t, k3)
                } else {
                    m31(self, sh)
                };
                let k32 = k(self, 32);
                let lo = self.s.m.bin(c, Bin::LShr, b, sh);
                let left = self.s.m.bin(c, Bin::Sub, k32, sh);
                let left = m31(self, left);
                let hi = self.s.m.bin(c, Bin::Shl, a, left);
                let zero = k(self, 0);
                let none = self.s.m.cmp(c, pred::EQ, sh, zero);
                let hi = self.s.m.select(c, none, zero, hi);
                self.s.m.bin(c, Bin::Or, lo, hi)
            }
            "v_bfrev_b32" => {
                let a = s(self, 0)?;
                self.s
                    .op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a])
            }
            "v_bcnt_u32_b32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let n = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 31, &[a]);
                self.s.m.bin(c, Bin::Add, n, b)
            }
            "v_mad_u32_u24" | "v_mad_i32_i24" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let (sh, op) = (
                    k(self, 8),
                    if name == "v_mad_i32_i24" {
                        Bin::AShr
                    } else {
                        Bin::LShr
                    },
                );
                let a = self.s.m.bin(c, Bin::Shl, a, sh);
                let a = self.s.m.bin(c, op, a, sh);
                let b = self.s.m.bin(c, Bin::Shl, b, sh);
                let b = self.s.m.bin(c, op, b, sh);
                let m = self.s.m.bin(c, Bin::Mul, a, b);
                self.s.m.bin(c, Bin::Add, m, d)
            }
            "v_min3_i32" | "v_max3_i32" | "v_min3_u32" | "v_max3_u32" | "v_med3_i32"
            | "v_med3_u32" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let signed = name.ends_with("i32");
                let (lt, gt) = if signed {
                    (pred::SLT, pred::SGT)
                } else {
                    (pred::IULT, pred::IUGT)
                };
                let pick = |l: &mut Self, p, x, y| {
                    let t = l.s.m.cmp(c, p, x, y);
                    l.s.m.select(c, t, x, y)
                };
                match &name[2..6] {
                    "min3" => {
                        let t = pick(self, lt, a, b);
                        pick(self, lt, t, d)
                    }
                    "max3" => {
                        let t = pick(self, gt, a, b);
                        pick(self, gt, t, d)
                    }
                    _ => {
                        let lo = pick(self, lt, a, b);
                        let hi = pick(self, gt, a, b);
                        let t = pick(self, lt, hi, d);
                        pick(self, gt, lo, t)
                    }
                }
            }
            "v_perm_b32" => {
                // Byte i of the result is byte S of {src0, src1} (src1 the
                // low dword) for selector byte S below 8; 8-11 repeat the
                // sign of bytes 1 and 3 of src1, then of src0; 12 is 0x00,
                // higher 0xFF.
                let Fields::Valu(v) = &inst.fields else {
                    unreachable!()
                };
                let sel = match v.src.get(2) {
                    Some(&Operand::Literal(x)) => x,
                    Some(&Operand::Const(k)) => crate::resource::inline(k).ok_or(name)?,
                    _ => {
                        // A computed selector: each byte picked at run time.
                        let (a, b, sv) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                        let mut r = k(self, 0);
                        for i in 0..4u32 {
                            let (sh, ff) = (k(self, 8 * i), k(self, 0xFF));
                            let sel = self.s.m.bin(c, Bin::LShr, sv, sh);
                            let sel = self.s.m.bin(c, Bin::And, sel, ff);
                            // Select a dword and a byte within it.
                            // Later selects handle codes 8 and above.
                            let four = k(self, 4);
                            let zero = k(self, 0);
                            let high = self.s.m.bin(c, Bin::And, sel, four);
                            let high = self.s.m.cmp(c, pred::NE, high, zero);
                            let word = self.s.m.select(c, high, a, b);
                            let three = k(self, 3);
                            let index = self.s.m.bin(c, Bin::And, sel, three);
                            let bits = self.s.m.bin(c, Bin::Shl, index, three);
                            let x = self.s.m.bin(c, Bin::LShr, word, bits);
                            let mut byte = self.s.m.bin(c, Bin::And, x, ff);
                            // 8-11: the sign of bytes 1, 3 of src1, then of src0.
                            for (n, src, bit) in [(8, b, 15), (9, b, 31), (10, a, 15), (11, a, 31)]
                            {
                                let left = k(self, 31 - bit);
                                let t = self.s.m.bin(c, Bin::Shl, src, left);
                                let k31 = k(self, 31);
                                let t = self.s.m.bin(c, Bin::AShr, t, k31);
                                let t = self.s.m.bin(c, Bin::And, t, ff);
                                let kn = k(self, n);
                                let is = self.s.m.cmp(c, pred::EQ, sel, kn);
                                byte = self.s.m.select(c, is, t, byte);
                            }
                            let (k12, z) = (k(self, 12), k(self, 0));
                            let is12 = self.s.m.cmp(c, pred::EQ, sel, k12);
                            byte = self.s.m.select(c, is12, z, byte);
                            let above = self.s.m.cmp(c, pred::IUGT, sel, k12);
                            byte = self.s.m.select(c, above, ff, byte);
                            let at = k(self, 8 * i);
                            let byte = self.s.m.bin(c, Bin::Shl, byte, at);
                            r = self.s.m.bin(c, Bin::Or, r, byte);
                        }
                        return Ok(r);
                    }
                };
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let mut r = k(self, 0);
                for i in 0..4 {
                    let pick = sel >> (8 * i) & 0xFF;
                    let byte = match pick {
                        0..=7 => {
                            let (src, n) = if pick < 4 { (b, pick) } else { (a, pick - 4) };
                            let sh = k(self, 8 * n);
                            let t = self.s.m.bin(c, Bin::LShr, src, sh);
                            let ff = k(self, 0xFF);
                            self.s.m.bin(c, Bin::And, t, ff)
                        }
                        8..=11 => {
                            let src = if pick < 10 { b } else { a };
                            let bit = if pick % 2 == 0 { 15 } else { 31 };
                            let left = k(self, 31 - bit);
                            let t = self.s.m.bin(c, Bin::Shl, src, left);
                            let k31 = k(self, 31);
                            let t = self.s.m.bin(c, Bin::AShr, t, k31);
                            let ff = k(self, 0xFF);
                            self.s.m.bin(c, Bin::And, t, ff)
                        }
                        12 => continue,
                        _ => k(self, 0xFF),
                    };
                    let at = k(self, 8 * i);
                    let byte = self.s.m.bin(c, Bin::Shl, byte, at);
                    r = self.s.m.bin(c, Bin::Or, r, byte);
                }
                r
            }
            "v_bfm_b32" => {
                // ((1 << src0[4:0]) - 1) << src1[4:0]
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let one = k(self, 1);
                let (a, b) = (m31(self, a), m31(self, b));
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                self.s.m.bin(c, Bin::Shl, m, b)
            }
            "v_cvt_pk_u8_f32" => {
                // src0 rounded to nearest even and saturated to a byte, into
                // byte src1[1:0] of src2.
                let f = self.fsrc(inst, 0)?;
                let (b, d) = (s(self, 1)?, s(self, 2)?);
                let f = self.unary(26, f);
                let (zero, max) = (self.s.m.f32(0), self.s.m.f32(255.0f32.to_bits()));
                let f = self.binary(35, f, zero);
                let f = self.binary(36, f, max);
                let x = self.s.m.cast(c, Cast::FpToUi, f, self.s.t.i32);
                let (three, k3) = (k(self, 3), k(self, 3));
                let sh = self.s.m.bin(c, Bin::And, b, three);
                let sh = self.s.m.bin(c, Bin::Shl, sh, k3);
                let x = self.s.m.bin(c, Bin::Shl, x, sh);
                let ff = k(self, 0xFF);
                let hole = self.s.m.bin(c, Bin::Shl, ff, sh);
                let ones = k(self, u32::MAX);
                let hole = self.s.m.bin(c, Bin::Xor, hole, ones);
                let d = self.s.m.bin(c, Bin::And, d, hole);
                self.s.m.bin(c, Bin::Or, d, x)
            }
            "v_ffbl_b32" => {
                // -1 for zero, as DXIL's first-bit-low.
                let a = s(self, 0)?;
                self.s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 32, &[a])
            }
            "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
                // Bits of the mask's low (high) half below this lane, plus src1:
                // with an all-ones mask, the lane index.
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                // Mesh lanes are the console's by construction; pixel and
                // vertex lanes are the host's, as their masks are.
                if self.mask_data == MaskData::Host {
                    // Host lanes: the mask must be one (a ballot) or all lanes.
                    let Fields::Valu(v) = &inst.fields else {
                        unreachable!()
                    };
                    let host = match v.src.first() {
                        Some(&Operand::Const(193)) => true,
                        Some(&Operand::Scalar { code, .. }) => {
                            inst.read(Reg::S(code)).is_some_and(|m| {
                                self.x.st.masks.contains_key(&m) || self.x.st.highs.contains_key(&m)
                            })
                        }
                        _ => false,
                    };
                    if !host {
                        return Err("v_mbcnt of a mask that is no ballot (console lane layout)");
                    }
                    self.s.flags |= flags::WAVE_OPS;
                } else if self.s.mesh.is_none() {
                    self.cross_lane(false)?;
                }
                // On a host wave wider than the console's the upper half is a
                // second console wave: `lo` counts bits of the lane's own
                // half's mask below its lane in it, and a wave32 mask's high
                // half is empty. The count is a lane number, and lane numbers
                // stay host lanes: a second wave's rank takes its 32-lane
                // base, so `v_mbcnt_lo(-1)` is still the lane index.
                let host_halves = self.mask_data == MaskData::Host && !self.x.native.wave64;
                let lane = self.lane_index();
                let below = if name == "v_mbcnt_lo_u32_b32" {
                    if host_halves {
                        let k31 = k(self, 31);
                        self.s.m.bin(c, Bin::And, lane, k31)
                    } else {
                        lane
                    }
                } else {
                    let (k32, z) = (k(self, 32), k(self, 0));
                    let t = self.s.m.bin(c, Bin::Sub, lane, k32);
                    let low = self.s.m.cmp(c, pred::IULT, lane, k32);
                    self.s.m.select(c, low, z, t)
                };
                // (1 << below) - 1, all ones from 32 on.
                let (one, k32, ones) = (k(self, 1), k(self, 32), k(self, u32::MAX));
                let sh = m31(self, below);
                let m = self.s.m.bin(c, Bin::Shl, one, sh);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                let full = self.s.m.cmp(c, pred::IUGE, below, k32);
                let m = self.s.m.select(c, full, ones, m);
                let x = if host_halves && name == "v_mbcnt_hi_u32_b32" {
                    k(self, 0)
                } else {
                    self.s.m.bin(c, Bin::And, a, m)
                };
                let n = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 31, &[x]);
                let n = if host_halves && name == "v_mbcnt_lo_u32_b32" {
                    let base = self.s.m.bin(c, Bin::And, lane, k32);
                    self.s.m.bin(c, Bin::Add, n, base)
                } else {
                    n
                };
                self.s.m.bin(c, Bin::Add, n, b)
            }
            _ => return Err(name),
        })
    }

    fn compare(&mut self, inst: &Inst, rest: &str) -> Result<(), Why> {
        let (op, ty) = rest.rsplit_once('_').ok_or("a compare")?;
        let c = self.cur;
        let r = match ty {
            "f32" | "f16" if op == "class" => {
                // Bit k of src1 tests class k: signalling NaN, quiet NaN,
                // -inf, -normal, -denormal, -0, +0, +denormal, +normal, +inf.
                let f16 = ty == "f16";
                let x = if f16 {
                    let Fields::Valu(v) = &inst.fields else {
                        unreachable!()
                    };
                    let mut h = self.half_src(inst, 0)?;
                    if v.abs & 1 != 0 {
                        let m = self.s.m.i32(0x7FFF);
                        h = self.s.m.bin(c, Bin::And, h, m);
                    }
                    if v.neg & 1 != 0 {
                        let s = self.s.m.i32(0x8000);
                        h = self.s.m.bin(c, Bin::Xor, h, s);
                    }
                    h
                } else {
                    let x = self.fsrc(inst, 0)?;
                    self.as_int(x)
                };
                let mask = self.src(inst, 1)?;
                let k = |l: &mut Self, v: u32| l.s.m.i32(v);
                let (exp_sh, sign_sh, exp_mask, mant_mask, quiet_bit) = if f16 {
                    (10, 15, 0x1F, 0x03FF, 0x0200)
                } else {
                    (23, 31, 0xFF, 0x7F_FFFF, 0x40_0000)
                };
                let (k_exp, k_sign, k_emax, mant, quiet) = (
                    k(self, exp_sh),
                    k(self, sign_sh),
                    k(self, exp_mask),
                    k(self, mant_mask),
                    k(self, quiet_bit),
                );
                let e = self.s.m.bin(c, Bin::LShr, x, k_exp);
                let e = self.s.m.bin(c, Bin::And, e, k_emax);
                let m = self.s.m.bin(c, Bin::And, x, mant);
                let neg = self.s.m.bin(c, Bin::LShr, x, k_sign);
                let zero = k(self, 0);
                let (e0, emax) = (
                    self.s.m.cmp(c, pred::EQ, e, zero),
                    self.s.m.cmp(c, pred::EQ, e, k_emax),
                );
                let m0 = self.s.m.cmp(c, pred::EQ, m, zero);
                let q = self.s.m.bin(c, Bin::And, x, quiet);
                let q = self.s.m.cmp(c, pred::NE, q, zero);
                // Positive classes 6-9; negative ones are 11 minus those.
                let (normal, inf, zero_c, denorm) =
                    (k(self, 8), k(self, 9), k(self, 6), k(self, 7));
                let t = self.s.m.select(c, m0, zero_c, denorm);
                let t = self.s.m.select(c, e0, t, normal);
                let t = self.s.m.select(c, emax, inf, t);
                let eleven = k(self, 11);
                let mirrored = self.s.m.bin(c, Bin::Sub, eleven, t);
                let is_neg = self.s.m.cmp(c, pred::NE, neg, zero);
                let t = self.s.m.select(c, is_neg, mirrored, t);
                let (one, snan) = (k(self, 1), k(self, 0));
                let nan_class = self.s.m.select(c, q, one, snan);
                let yes = self.s.m.bool(true);
                let nan = self.s.m.bin(c, Bin::Xor, m0, yes);
                let nan = self.s.m.bin(c, Bin::And, nan, emax);
                let t = self.s.m.select(c, nan, nan_class, t);
                let bit = self.s.m.bin(c, Bin::LShr, mask, t);
                let bit = self.s.m.bin(c, Bin::And, bit, one);
                self.s.m.cmp(c, pred::NE, bit, zero)
            }
            "f32" | "f16" => {
                let (a, b) = if ty == "f16" {
                    (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?)
                } else {
                    (self.fsrc(inst, 0)?, self.fsrc(inst, 1)?)
                };
                if op == "lg" {
                    // Ordered and not equal, false for a NaN, as `olt | ogt`.
                    // `fcmp one` means the same, but the AMD D3D12 driver
                    // runs it as unordered (true for a NaN) where NVIDIA
                    // and WARP do not; olt and ogt agree on all three.
                    let lt = self.s.m.cmp(c, pred::OLT, a, b);
                    let gt = self.s.m.cmp(c, pred::OGT, a, b);
                    self.s.m.bin(c, Bin::Or, lt, gt)
                } else {
                    let p = match op {
                        "f" => pred::FALSE,
                        "lt" => pred::OLT,
                        "eq" => pred::OEQ,
                        "le" => pred::OLE,
                        "gt" => pred::OGT,
                        "ge" => pred::OGE,
                        "o" => pred::ORD,
                        "u" => pred::UNO,
                        "nge" => pred::ULT,
                        "nlg" => pred::UEQ,
                        "ngt" => pred::ULE,
                        "nle" => pred::UGT,
                        "neq" => pred::UNE,
                        "nlt" => pred::UGE,
                        "tru" => pred::TRUE,
                        _ => return Err(inst.name()),
                    };
                    self.s.m.cmp(c, p, a, b)
                }
            }
            "u64" | "i64" => {
                let (a, b) = (self.src64(inst, 0)?, self.src64(inst, 1)?);
                let signed = ty == "i64";
                let p = match op {
                    "eq" => pred::EQ,
                    "ne" | "lg" => pred::NE,
                    "lt" if signed => pred::SLT,
                    "le" if signed => pred::SLE,
                    "gt" if signed => pred::SGT,
                    "ge" if signed => pred::SGE,
                    "lt" => pred::IULT,
                    "le" => pred::IULE,
                    "gt" => pred::IUGT,
                    "ge" => pred::IUGE,
                    _ => return Err("a 64-bit compare"),
                };
                self.s.m.cmp(c, p, a, b)
            }
            "u32" | "i32" | "u16" | "i16" => {
                let (a, b) = if ty.ends_with("16") {
                    let (mut a, mut b) = (self.half_src(inst, 0)?, self.half_src(inst, 1)?);
                    if ty == "i16" {
                        let k16 = self.s.m.i32(16);
                        for x in [&mut a, &mut b] {
                            let t = self.s.m.bin(c, Bin::Shl, *x, k16);
                            *x = self.s.m.bin(c, Bin::AShr, t, k16);
                        }
                    }
                    (a, b)
                } else {
                    (self.src(inst, 0)?, self.src(inst, 1)?)
                };
                let signed = ty == "i32" || ty == "i16";
                let p = match op {
                    "eq" => pred::EQ,
                    "ne" | "lg" => pred::NE,
                    "lt" if signed => pred::SLT,
                    "le" if signed => pred::SLE,
                    "gt" if signed => pred::SGT,
                    "ge" if signed => pred::SGE,
                    "lt" => pred::IULT,
                    "le" => pred::IULE,
                    "gt" => pred::IUGT,
                    "ge" => pred::IUGE,
                    "f" | "t" => {
                        self.lane(inst, Some(op == "t"));
                        return Ok(());
                    }
                    _ => return Err("an integer compare"),
                };
                self.s.m.cmp(c, p, a, b)
            }
            _ => return Err(inst.name()),
        };
        self.lane_bits_for(inst, r);
        Ok(())
    }

    fn lane(&mut self, inst: &Inst, value: Option<bool>) {
        let b = self.s.m.bool(value == Some(true));
        self.lane_bits_for(inst, b);
    }

    fn lane_bits_for(&mut self, inst: &Inst, r: V) {
        let Fields::Valu(v) = &inst.fields else {
            return;
        };
        let code = if inst.name().starts_with("v_cmpx") {
            Some(EXEC)
        } else {
            match v.dst {
                Some(Operand::Scalar { code, .. }) => Some(code),
                _ => None,
            }
        };
        if let Some(lo) = code.and_then(|c| inst.write(Reg::S(c))) {
            self.lane_bits.insert(lo, r);
        }
        for &(_, w) in &inst.writes {
            self.vals
                .entry(w)
                .or_insert(Val::Missing("a lane mask read as data"));
        }
    }

    fn interp(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Some(&interp) = self.interps.get(&site) else {
            return Err("the first half of an interpolation");
        };
        let (attribute, mode) = match interp {
            Interp::Interpolated {
                attribute,
                barycentric,
                ..
            } => (
                attribute,
                match barycentric {
                    Barycentric::PerspectiveCenter => 2,
                    Barycentric::PerspectiveCentroid => 3,
                    Barycentric::PerspectiveSample => 6,
                    Barycentric::LinearCenter => 4,
                    Barycentric::LinearCentroid => 5,
                    Barycentric::LinearSample => 7,
                    Barycentric::PerspectivePullModel => return Err("pull-model interpolation"),
                },
            ),
            // P0, the provoking vertex: a flat (nointerpolation) input.
            Interp::Vertex {
                attribute,
                vertex: 2,
            } => (attribute, 1),
            Interp::Vertex { .. } => return Err("a non-provoking vertex value"),
        };
        let dst = match &inst.fields {
            Fields::Vintrp { dst, .. } => *dst,
            Fields::Valu(v) => v.dst.ok_or("an interpolation")?,
            _ => unreachable!(),
        };
        let Operand::Vector { reg, .. } = dst else {
            return Err("an interpolation");
        };
        // A channel of an input element, interpolated as `mode` says.
        let channel = |l: &mut Self, row: u8, ch: u8| -> Result<V, Why> {
            let (id, col) = l
                .find_input(row, ch)
                .ok_or("an input the signature lacks")?;
            let e = &mut l.s.inputs[id];
            if e.interp != mode && e.usage != 0 {
                return Err("one input interpolated two ways");
            }
            e.interp = mode;
            match l.load_any_input(id, col) {
                Val::I32(x) => Ok(x),
                _ => Err("an input"),
            }
        };
        let packing = match &self.x.native.io {
            StageIo::Ps { inputs, .. } => inputs
                .iter()
                .find(|i| i.row == attribute.row && i.fp16)
                .map(|i| {
                    let b = if i.row2 != i.row {
                        Some(i.row2)
                    } else if i.pack2 {
                        half_input_partners(&self.x.native.io, &self.s.inputs)
                            .get(&i.row)
                            .copied()
                    } else {
                        None
                    };
                    (i.pack2, b)
                }),
            _ => None,
        };
        let f16 = inst.name().ends_with("_f16");
        let x = match packing {
            None if f16 || attribute.high => return Err("a 16-bit read of a 32-bit input"),
            None => channel(self, attribute.row, attribute.channel)?,
            Some((pack2, b)) => {
                // 16-bit values: one in the low half of each dword, or two
                // per dword (`packed_half`); the host's are 32-bit floats.
                let half = |l: &mut Self, h: u8| -> Result<Option<V>, Why> {
                    let src = if pack2 {
                        packed_half(&l.s.inputs, attribute.row, b, h)
                    } else {
                        h.is_multiple_of(2).then_some((attribute.row, h / 2))
                    };
                    let Some((row, ch)) = src else {
                        return Ok(None);
                    };
                    let x = channel(l, row, ch)?;
                    let f = l.as_float(x);
                    let i32t = l.s.t.i32;
                    Ok(Some(l.s.op(
                        l.cur,
                        "legacyF32ToF16",
                        Attr::ReadNone,
                        i32t,
                        130,
                        &[f],
                    )))
                };
                let h = 2 * attribute.channel;
                let c = self.cur;
                if f16 {
                    // XNW: a half no element holds (padding of the
                    // parameter's dwords) reads 0, as for flat inputs below.
                    // V: f5c7197d reads attr3.w, half 6 of rows 6 (four
                    // halves) and 7 (two).
                    let lo = match half(self, h + u8::from(attribute.high))? {
                        Some(lo) => lo,
                        None => self.s.m.i32(0),
                    };
                    self.low_half(inst, reg, lo)
                } else if mode == 1 {
                    // The whole dword of a flat input: both halves.
                    let z = self.s.m.i32(0);
                    let lo = half(self, h)?.unwrap_or(z);
                    let hi = half(self, h + 1)?.unwrap_or(z);
                    let k16 = self.s.m.i32(16);
                    let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                    self.s.m.bin(c, Bin::Or, lo, hi)
                } else {
                    return Err("32-bit interpolation of a 16-bit input");
                }
            }
        };
        // A pixel interpolates in every lane; guards only hide inactive ones.
        self.write(inst, Reg::V(reg), x, None);
        Ok(())
    }

    /// `x` (an f16 in the low bits) as the low half of VGPR `reg`: 16-bit
    /// results keep the register's high half (GFX10).
    fn low_half(&mut self, inst: &Inst, reg: u8, x: V) -> V {
        let c = self.cur;
        let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let (high, low) = (self.s.m.i32(0xFFFF_0000), self.s.m.i32(0xFFFF));
        let old = self.s.m.bin(c, Bin::And, old, high);
        let x = self.s.m.bin(c, Bin::And, x, low);
        self.s.m.bin(c, Bin::Or, old, x)
    }

    /// An image atomic DXIL has no operation for (`inc`, `dec`, `fcmpswap`,
    /// `fmin`, `fmax`), on the dword at `coords` of the 32-bit integer view
    /// `h`: a compare-and-swap loop from a first guess of 0, until the
    /// exchange finds the value the new one was made from (a null view reads
    /// 0, so it leaves at once). `vals`: the data (fcmpswap: new, compare).
    /// Returns the old value. VIctory (1661a3e, GPL-3.0-or-later).
    /// XNW: each pass spends the invocation's loop budget, and the loop
    /// also leaves when the budget is gone (the exchange then not made).
    fn atomic_loop(&mut self, h: V, coords: &[V], op: &str, vals: &[V]) -> V {
        let i32t = self.s.t.i32;
        let entry = self.cur;
        let (head, done) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(entry, Term::Br(head));
        self.cur = head;
        let old = self.s.m.phi(head, i32t);
        let budget = self.s.m.phi(head, i32t);
        let (data, zero, one) = (vals[0], self.s.m.i32(0), self.s.m.i32(1));
        let new = match op {
            // old >= data ? 0 : old + 1
            "inc" => {
                let up = self.s.m.bin(head, Bin::Add, old, one);
                let wrap = self.s.m.cmp(head, pred::IUGE, old, data);
                self.s.m.select(head, wrap, zero, up)
            }
            // old == 0 || old > data ? data : old - 1
            "dec" => {
                let down = self.s.m.bin(head, Bin::Sub, old, one);
                let empty = self.s.m.cmp(head, pred::EQ, old, zero);
                let over = self.s.m.cmp(head, pred::IUGT, old, data);
                let reload = self.s.m.bin(head, Bin::Or, empty, over);
                self.s.m.select(head, reload, data, down)
            }
            // A float compare: -0 matches +0, NaN matches nothing.
            "fcmpswap" => {
                let (o, c) = (self.as_float(old), self.as_float(vals[1]));
                let eq = self.s.m.cmp(head, pred::OEQ, o, c);
                self.s.m.select(head, eq, data, old)
            }
            // As v_min_f32 and v_max_f32.
            _ => {
                let (d, o) = (self.as_float(data), self.as_float(old));
                let m = self.binary(if op == "fmax" { 35 } else { 36 }, d, o);
                self.as_int(m)
            }
        };
        let mut args = vec![h];
        args.extend_from_slice(coords);
        args.extend([old, new]);
        let prev = self.s.op(
            head,
            "atomicCompareExchange.i32",
            Attr::NoUnwind,
            i32t,
            79,
            &args,
        );
        let found = self.s.m.cmp(head, pred::EQ, prev, old);
        let next = self.s.m.bin(head, Bin::Add, budget, one);
        let limit = self.s.m.i32(super::loop_limit_for(self.x.native.stage, self.x.fallback));
        let spent = self.s.m.cmp(head, pred::IUGE, next, limit);
        let leave = self.s.m.bin(head, Bin::Or, found, spent);
        self.s.m.terminate(head, Term::CondBr(leave, done, head));
        let before = self.budget.expect("block started");
        self.s.m.set_phi(old, vec![(zero, entry), (prev, head)]);
        self.s.m.set_phi(budget, vec![(before, entry), (next, head)]);
        self.cur = done;
        self.budget = Some(next);
        prev
    }

    /// XNW: a 64-bit image atomic (`dmask` 0x3, cmpswap 0xF): the
    /// value is the VGPR pair (low dword first), the image an R32G32_UINT
    /// texture the host reads as `uint64_t` (DXIL `atomicBinOp.i64`,
    /// `atomicCompareExchange.i64`). With glc the old value returns in the
    /// pair. Seen: the alpha prepasses' `image_atomic_umax v[3:4]`.
    #[allow(clippy::too_many_arguments)]
    fn image_atomic64(
        &mut self,
        inst: &Inst,
        site: Site,
        src: &Source,
        kind: u32,
        code: u32,
        negate: bool,
        cs: Vec<V>,
        data: &Operand,
        glc: bool,
        guard: Option<V>,
    ) -> Result<(), Why> {
        let Operand::Vector { reg, .. } = *data else {
            return Err("image atomic data");
        };
        let cmpswap = code == u32::MAX;
        let vals = self.vgprs(inst, data, if cmpswap { 4 } else { 2 })?;
        let i64t = self.s.m.int(64);
        let c = self.cur;
        let k32 = self.s.m.int_const(i64t, 32);
        let wide = |l: &mut Self, lo: V, hi: V| {
            let lo = l.s.m.cast(c, Cast::ZExt, lo, i64t);
            let hi = l.s.m.cast(c, Cast::ZExt, hi, i64t);
            let hi = l.s.m.bin(c, Bin::Shl, hi, k32);
            l.s.m.bin(c, Bin::Or, lo, hi)
        };
        let mut value = wide(self, vals[0], vals[1]);
        if negate {
            let z = self.s.m.int_const(i64t, 0);
            value = self.s.m.bin(c, Bin::Sub, z, value);
        }
        let compare = if cmpswap { Some(wide(self, vals[2], vals[3])) } else { None };
        let old = if glc {
            let lo = inst.read(Reg::V(reg)).map(|v| self.i32_of(v)).transpose()?;
            let hi = inst.read(Reg::V(reg + 1)).map(|v| self.i32_of(v)).transpose()?;
            let z = self.s.m.i32(0);
            wide(self, lo.unwrap_or(z), hi.unwrap_or(z))
        } else {
            self.s.m.int_const(i64t, 0)
        };
        // one component.
        let h = self.handle(site, src, false, (kind | 0x1000, 0x107))?;
        self.s.flags |= flags::INT64 | flags::ATOMIC_INT64_TYPED | flags::ATOMIC_INT64_HEAP;
        let at = self.begin_guarded(guard);
        let mut args = vec![h];
        let r = match compare {
            Some(cmp) => {
                // cmpswap: data is (new, compare).
                args.extend(cs);
                args.extend([cmp, value]);
                self.s.op(self.cur, "atomicCompareExchange.i64", Attr::NoUnwind, i64t, 79, &args)
            }
            None => {
                args.push(self.s.m.i32(code));
                args.extend(cs);
                args.push(value);
                self.s.op(self.cur, "atomicBinOp.i64", Attr::NoUnwind, i64t, 78, &args)
            }
        };
        let r = self.end_guarded(at, &[(r, old)])[0];
        if glc {
            let c = self.cur;
            let i32t = self.s.t.i32;
            let lo = self.s.m.cast(c, Cast::Trunc, r, i32t);
            let hi = self.s.m.bin(c, Bin::LShr, r, k32);
            let hi = self.s.m.cast(c, Cast::Trunc, hi, i32t);
            self.write(inst, Reg::V(reg), lo, None);
            self.write(inst, Reg::V(reg + 1), hi, None);
        }
        Ok(())
    }

    fn image(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        if let Some(h) = self.s.host_copies.as_mut() {
            *h |= super::host_copies::TEXTURES;
        }
        let Fields::Image {
            data,
            addr,
            dmask,
            dim,
            unorm,
            a16,
            d16,
            tfe,
            lwe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        if *tfe || *lwe {
            return Err("an image access with tfe or lwe");
        }
        let name = inst.name();
        // A load or sample with no channel enabled returns the first.
        let dmask = if *dmask == 0 { &1 } else { dmask };
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let (kind, coords) = match dim {
            0 => (1, 1),
            1 => (2, 2),
            2 => (4, 3),
            // Sampled (face coordinates and face, see `cube_direction`) or
            // sized only.
            3 if name.starts_with("image_sample")
                || name.starts_with("image_gather")
                || name == "image_get_resinfo" =>
            {
                (5, 3)
            }
            4 => (6, 2),
            5 => (7, 3),
            _ => return Err("an image dimension (cube loads or MSAA)"),
        };
        // Address VGPRs in order.
        let mut regs = Vec::new();
        for a in addr {
            if let Operand::Vector { reg, dwords } = *a {
                regs.extend((0..dwords).map(|k| reg + k));
            }
        }
        // Full 32-bit address VGPR (offset, bias, z-compare are always 32-bit
        // even with `a16`).
        let addr_dword = |l: &mut Self, w: usize| -> Result<V, Why> {
            let r = *regs.get(w).ok_or("an image address")?;
            let v = inst.read(Reg::V(r)).ok_or("an image address")?;
            l.i32_of(v)
        };
        // Packed component `k` starting at VGPR `base` (with `a16`, two 16-bit
        // components per VGPR, low half first).
        let addr_packed = |l: &mut Self, base: usize, k: usize| -> Result<V, Why> {
            let w = base + if *a16 { k / 2 } else { k };
            let x = addr_dword(l, w)?;
            if !*a16 {
                return Ok(x);
            }
            let c = l.cur;
            let (sh, mask) = (l.s.m.i32(16 * (k as u32 % 2)), l.s.m.i32(0xFFFF));
            let x = l.s.m.bin(c, Bin::LShr, x, sh);
            Ok(l.s.m.bin(c, Bin::And, x, mask))
        };
        let addr_val = |l: &mut Self, k: usize| addr_packed(l, 0, k);
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked image")?;
        let store = name.starts_with("image_store");
        let uav = store
            || self
                .used_kind(src)
                .is_some_and(|u| u.kind == ResourceType::Uav);
        let props = (kind | if uav { 0x1000 } else { 0 }, 0x409);
        if let Some(op) = name.strip_prefix("image_atomic_") {
            // DXIL's atomic operations on a 32-bit integer texture UAV; the
            // old value returns with glc. Sub is the add of the negation.
            let code = match op {
                "add" | "sub" => 0,
                "and" => 1,
                "or" => 2,
                "xor" => 3,
                "smin" => 4,
                "smax" => 5,
                "umin" => 6,
                "umax" => 7,
                "swap" => 8,
                "cmpswap" => u32::MAX,
                // VIctory (1661a3e): no DXIL operation, a
                // compare-and-swap loop (`atomic_loop`).
                "inc" | "dec" | "fcmpswap" | "fmin" | "fmax" => ATOMIC_LOOP,
                _ => return Err(name),
            };
            let Fields::Image { cache, .. } = &inst.fields else {
                unreachable!()
            };
            if *d16 || *a16 {
                return Err("a 16-bit image atomic");
            }
            // dmask: the data dwords (the compare-and-swaps: twice the value's).
            let pair = matches!(op, "cmpswap" | "fcmpswap");
            let (one, two) = if pair { (3, 0xF) } else { (1, 3) };
            if *dmask != one && *dmask != two {
                return Err("an image atomic of some channels");
            }
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            if *dmask == two && code == ATOMIC_LOOP {
                return Err("a 64-bit image atomic of a compare-and-swap kind");
            }
            if *dmask == two {
                // XNW: two dwords per value, a 64-bit atomic on an
                // R32G32_UINT image, with the host's typed 64-bit atomics
                // (SM 6.6; the RX 5700 XT reports AtomicInt64OnTypedResource
                // and AtomicInt64OnDescriptorHeapResource, OPTIONS9/11).
                return self.image_atomic64(inst, site, src, kind, code, op == "sub", cs, data, cache.glc, guard);
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image atomic data");
            };
            let vals = self.vgprs(inst, data, if pair { 2 } else { 1 })?;
            let mut value = vals[0];
            if op == "sub" {
                let z = self.s.m.i32(0);
                value = self.s.m.bin(c, Bin::Sub, z, value);
            }
            let old = match inst.read(Reg::V(reg)) {
                Some(v) if cache.glc => self.i32_of(v)?,
                _ => self.s.m.i32(0),
            };
            let h = self.handle(site, src, false, (kind | 0x1000, 0x105))?;
            let before = self.budget.expect("block started");
            let at = self.begin_guarded(guard);
            let i32t = self.s.t.i32;
            let mut args = vec![h];
            let r = if code == ATOMIC_LOOP {
                self.atomic_loop(h, &cs, op, &vals)
            } else if code == u32::MAX {
                // cmpswap: data is (new, compare).
                args.extend(cs);
                args.extend([vals[1], vals[0]]);
                self.s.op(
                    self.cur,
                    "atomicCompareExchange.i32",
                    Attr::NoUnwind,
                    i32t,
                    79,
                    &args,
                )
            } else {
                args.push(self.s.m.i32(code));
                args.extend(cs);
                args.push(value);
                self.s
                    .op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &args)
            };
            // XNW: the budget a compare-and-swap loop spent joins too.
            let after = self.budget.expect("block started");
            let joined = self.end_guarded(at, &[(r, old), (after, before)]);
            let r = joined[0];
            self.budget = Some(joined[1]);
            if cache.glc {
                self.write(inst, Reg::V(reg), r, None);
            }
            return Ok(());
        }
        if store {
            if name != "image_store" {
                return Err("an image store to a mip level");
            }
            // shortcut: stores are taken to write all four channels of a
            // float format (the console converts by the view's format).
            if *dmask != 0xF {
                return Err("an image store of some channels");
            }
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image store data");
            };
            let mut vals = Vec::new();
            for k in 0..4u8 {
                // d16: two f16 channels per VGPR, low half first.
                let r = if *d16 { k / 2 } else { k };
                let v = inst.read(Reg::V(reg + r)).ok_or("image store data")?;
                vals.push(if *d16 {
                    let mut x = self.i32_of(v)?;
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(self.cur, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    self.s
                        .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
                } else {
                    self.float_of(v)?
                });
            }
            let h = self.handle(site, src, false, props)?;
            let at = self.begin_guarded(guard);
            let mut args = vec![h];
            args.extend(cs);
            args.extend(vals);
            args.push(self.s.m.i8(0xF));
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                "textureStore.f32",
                Attr::NoUnwind,
                void,
                67,
                &args,
            );
            self.end_guarded(at, &[]);
            return Ok(());
        }
        let (op, sample) = match name {
            "image_sample" | "image_sample_cl" | "image_sample_o" => (60, true),
            "image_sample_lz" | "image_sample_l" => (62, true),
            "image_sample_b" | "image_sample_b_cl" => (61, true),
            "image_sample_c_lz" => (65, true),
            "image_sample_d" | "image_sample_d_cl" => (63, true),
            "image_get_lod" if self.s.pixel => (81, true),
            // Implicit derivatives: pixel shaders only.
            "image_sample_c" | "image_sample_c_cl" if self.s.pixel => (64, true),
            "image_gather4"
            | "image_gather4_lz"
            | "image_gather4_lz_o"
            | "image_gather4_l"
            | "image_gather4_l_o" => (73, true),
            "image_load" | "image_load_mip" => (66, false),
            "image_get_resinfo" => (72, false),
            _ => return Err(name),
        };
        if op == 72 {
            // Width, height, depth or layers, levels (per dmask) of the level
            // in the first address: DXIL's dimensions; a channel DXIL leaves
            // out reads 1 (UAVs: one level).
            let h = self.handle(site, src, false, props)?;
            let mip = if uav {
                self.s.m.undef(self.s.t.i32)
            } else {
                addr_val(self, 0)?
            };
            let i32t = self.s.t.i32;
            let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
            let d = self
                .s
                .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
            let one = self.s.m.i32(1);
            let present = match kind {
                1 => [true, false, false, true],
                2 | 5 | 6 => [true, true, false, true],
                _ => [true, true, true, true],
            };
            let Operand::Vector { reg, .. } = *data else {
                return Err("an image result");
            };
            for (k, ch) in (0..4u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                let x = if ch == 3 && uav || !present[ch as usize] {
                    one
                } else {
                    self.s.m.extract(c, d, ch)
                };
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        if uav && op != 66 {
            return Err("sampling a UAV");
        }
        let h = self.handle(site, src, false, props)?;
        let fu = self.s.m.undef(self.s.t.f32);
        let iu = self.s.m.undef(self.s.t.i32);
        let r = if sample {
            let ss = self
                .descriptors
                .get(&(site, true))
                .copied()
                .ok_or("an untracked sampler")?;
            // Comparison: the reference value is the first address after
            // the bias, and the sampler is a comparison sampler.
            let cmp = matches!(op, 64 | 65);
            if cmp && dmask.count_ones() != 1 {
                return Err("a comparison sample of several channels");
            }
            let sh = self.handle(site, ss, true, (14 | if cmp { 0x8000 } else { 0 }, 0))?;
            // Texel offsets (`_o`), LOD bias (`_b`), and comparison reference
            // (`_c`) come first as full 32-bit VGPRs (even when `a16` is set).
            let offset_word = name.ends_with("_o");
            let has_bias = op == 61;
            let has_clamp = name.ends_with("_cl");
            let whole = usize::from(offset_word) + usize::from(has_bias) + usize::from(cmp);
            // Coordinates the size divides (not the array slice); explicit
            // gradients (x then y) come before the coordinates, each direction
            // packed into `spatial.div_ceil(2)` VGPRs when `a16` is set.
            let spatial = coords - usize::from(kind == 6 || kind == 7);
            // A cube's gradients are two per direction, in face space (s, t).
            let grad_spatial = if kind == 5 { 2 } else { spatial };
            let grad_dir_words = if op == 63 {
                if *a16 { grad_spatial.div_ceil(2) } else { grad_spatial }
            } else {
                0
            };
            let coord_base = whole + 2 * grad_dir_words;
            let mut cs = vec![fu; 4];
            // With a16, sampling coordinates, gradients, LOD and clamp are f16.
            let to_float = |l: &mut Self, x: V| {
                if *a16 {
                    let f = l.s.t.f32;
                    l.s.op(l.cur, "legacyF16ToF32", Attr::ReadNone, f, 131, &[x])
                } else {
                    l.as_float(x)
                }
            };
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                let x = addr_packed(self, coord_base, k)?;
                *slot = to_float(self, x);
            }
            // XNW: a cube sample with gradients (`image_sample_d` on
            // a cube) gets direction gradients: the direction is affine in
            // (s, t) on a face, so the difference of directions is exact
            // (five-star-recomp ImageSampleBuilder.cs 274-282, 346cf65, GPL-3).
            let mut cube_grads = None;
            if kind == 5 {
                if *unorm || op == 81 {
                    return Err("a cube sample with texel coordinates or a level query");
                }
                let dir = self.cube_direction(cs[0], cs[1], cs[2]);
                if op == 63 {
                    let mut g = [fu; 6];
                    for d in 0..2 {
                        let ds = addr_packed(self, whole + d * grad_dir_words, 0)?;
                        let ds = to_float(self, ds);
                        let dt = addr_packed(self, whole + d * grad_dir_words, 1)?;
                        let dt = to_float(self, dt);
                        let c = self.cur;
                        let s1 = self.s.m.bin(c, Bin::Add, cs[0], ds);
                        let t1 = self.s.m.bin(c, Bin::Add, cs[1], dt);
                        let moved = self.cube_direction(s1, t1, cs[2]);
                        for k in 0..3 {
                            g[3 * d + k] = self.s.m.bin(c, Bin::Sub, moved[k], dir[k]);
                        }
                    }
                    cube_grads = Some(g);
                }
                cs[..3].copy_from_slice(&dir);
            }
            if *unorm {
                // Texel coordinates: divided by the size of level 0.
                let i32t = self.s.t.i32;
                let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                let z = self.s.m.i32(0);
                let d = self
                    .s
                    .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, z]);
                for (k, slot) in cs.iter_mut().enumerate().take(spatial) {
                    let n = self.s.m.extract(c, d, k as u32);
                    let n = self.s.m.cast(c, Cast::UiToFp, n, self.s.t.f32);
                    *slot = self.s.m.bin(c, Bin::SDiv, *slot, n);
                }
            }
            if op == 81 {
                // The clamped then the unclamped level of detail.
                let Operand::Vector { reg, .. } = *data else {
                    return Err("an image result");
                };
                let f32t = self.s.t.f32;
                for (k, ch) in (0..2u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                    let clamped = self.s.m.bool(ch == 0);
                    let lod = self.s.op(
                        c,
                        "calculateLOD.f32",
                        Attr::ReadOnly,
                        f32t,
                        81,
                        &[
                            h,
                            sh,
                            cs[0],
                            if spatial > 1 { cs[1] } else { fu },
                            if spatial > 2 { cs[2] } else { fu },
                            clamped,
                        ],
                    );
                    let x = self.as_int(lod);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let offsets = if kind == 5 {
                0
            } else {
                (coords - usize::from(kind == 6 || kind == 7)).min(3)
            };
            let zero = self.s.m.i32(0);
            let mut os = vec![iu; 3];
            for o in os.iter_mut().take(offsets) {
                *o = zero;
            }
            let mut args = vec![h, sh];
            args.extend(&cs);
            let clamp_val = if has_clamp {
                let x = addr_packed(self, coord_base, coords)?;
                to_float(self, x)
            } else {
                fu
            };
            match op {
                // VIctory (c78056c): implicit derivatives outside
                // pixel shaders.
                // Changed from VIctory (`!pixel`): compute only. Our
                // mesh programs validated with implicit samples, and 15 of
                // them failed through this arm (pack_listing).
                60 | 61 if self.x.native.stage == HwStage::Cs => {
                    // Outside pixel shaders the console still takes the level
                    // from each quad of lanes (lane 1 minus lane 0 across,
                    // lane 2 minus lane 0 down); DXIL before validator 1.8
                    // has no implicit derivatives there, so the quad's
                    // differences become explicit gradients. A bias scales
                    // them by 2^bias, the same level for isotropic filtering.
                    self.cross_lane(true)?;
                    if offset_word {
                        let w = addr_dword(self, 0)?;
                        let w = self
                            .s
                            .m
                            .int_value(w)
                            .ok_or("a sample offset that is not constant")?
                            as u32;
                        for (k, o) in os.iter_mut().enumerate().take(offsets) {
                            let x = (w >> (8 * k) & 0x3F) as i32;
                            let x = if x >= 32 { x - 64 } else { x };
                            if !(-8..=7).contains(&x) {
                                return Err("a sample offset past -8..7");
                            }
                            *o = self.s.m.i32(x as u32);
                        }
                    }
                    let scale = if op == 61 {
                        let bias = addr_dword(self, usize::from(offset_word))?;
                        let bias = self.as_float(bias);
                        Some(self.unary(21, bias))
                    } else {
                        None
                    };
                    let g = self.quad_gradients(&cs[..spatial], scale);
                    args.extend(&os);
                    args.extend(g);
                    args.push(clamp_val);
                    self.s.op(
                        c,
                        "sampleGrad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        63,
                        &args,
                    )
                }
                60 => {
                    if offset_word {
                        // DXIL takes immediate offsets in -8..=7; the
                        // console's are 6-bit and may be computed.
                        let w = addr_dword(self, 0)?;
                        let w = self
                            .s
                            .m
                            .int_value(w)
                            .ok_or("a sample offset that is not constant")?
                            as u32;
                        for (k, o) in os.iter_mut().enumerate().take(offsets) {
                            let x = (w >> (8 * k) & 0x3F) as i32;
                            let x = if x >= 32 { x - 64 } else { x };
                            if !(-8..=7).contains(&x) {
                                return Err("a sample offset past -8..7");
                            }
                            *o = self.s.m.i32(x as u32);
                        }
                    }
                    args.extend(&os);
                    args.push(clamp_val);
                    self.s
                        .op(c, "sample.f32", Attr::ReadOnly, self.s.t.res_ret, 60, &args)
                }
                61 => {
                    let bias = addr_dword(self, usize::from(offset_word))?;
                    let bias = self.as_float(bias);
                    args.extend(&os);
                    args.extend([bias, clamp_val]);
                    self.s.op(
                        c,
                        "sampleBias.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        61,
                        &args,
                    )
                }
                63 if cube_grads.is_some() => {
                    args.extend(&os);
                    args.extend(cube_grads.unwrap_or([fu; 6]));
                    args.push(clamp_val);
                    self.s.op(
                        c,
                        "sampleGrad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        63,
                        &args,
                    )
                }
                63 => {
                    let mut g = vec![fu; 6];
                    for k in 0..spatial {
                        let x = addr_packed(self, whole, k)?;
                        g[k] = to_float(self, x);
                        let y = addr_packed(self, whole + grad_dir_words, k)?;
                        g[3 + k] = to_float(self, y);
                    }
                    args.extend(&os);
                    args.extend(g);
                    args.push(clamp_val);
                    self.s.op(
                        c,
                        "sampleGrad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        63,
                        &args,
                    )
                }
                64 | 65 => {
                    let reference = addr_dword(self, whole - 1)?;
                    let reference = self.as_float(reference);
                    args.extend(&os);
                    args.push(reference);
                    let name = if op == 64 {
                        args.push(clamp_val);
                        "sampleCmp.f32"
                    } else {
                        "sampleCmpLevelZero.f32"
                    };
                    self.s
                        .op(c, name, Attr::ReadOnly, self.s.t.res_ret, op, &args)
                }
                62 => {
                    let lod = if name == "image_sample_l" {
                        let x = addr_packed(self, coord_base, coords)?;
                        to_float(self, x)
                    } else {
                        self.s.m.f32(0)
                    };
                    args.extend(&os);
                    args.push(lod);
                    self.s.op(
                        c,
                        "sampleLevel.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        62,
                        &args,
                    )
                }
                _ => {
                    if name == "image_gather4" {
                        return Err("a gather with implicit level");
                    }
                    if dmask.count_ones() != 1 {
                        return Err("a gather of several channels");
                    }
                    if offset_word {
                        let w = addr_dword(self, 0)?;
                        for (k, at) in [(0, 26), (1, 18)] {
                            // Bits at..at+5 sign-extended: shift up, then down.
                            let left = self.s.m.i32(at);
                            let t = self.s.m.bin(c, Bin::Shl, w, left);
                            let k26 = self.s.m.i32(26);
                            os[k] = self.s.m.bin(c, Bin::AShr, t, k26);
                        }
                    }
                    if matches!(name, "image_gather4_l" | "image_gather4_l_o") {
                        // Console-only Gather at an explicit mip level: DXIL's
                        // textureGather has no LOD argument, so emulate with
                        // four clamped textureLoad calls at that mip in HLSL
                        // gather order: (-,+), (+,+), (+,-), (-,-).
                        if kind != 2 {
                            return Err("a gather with explicit level on a non-2D texture");
                        }
                        let lod = addr_packed(self, coord_base, coords)?;
                        let lod = to_float(self, lod);
                        let i32t = self.s.t.i32;
                        let f32t = self.s.t.f32;
                        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                        let d0 = self
                            .s
                            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, zero]);
                        let levels = self.s.m.extract(c, d0, 3);
                        let one = self.s.m.i32(1);
                        let max_mip = self.s.m.bin(c, Bin::Sub, levels, one);
                        let fzero = self.s.m.f32(0);
                        let lod = self.binary(35, lod, fzero);
                        let mip = self.s.m.cast(c, Cast::FpToUi, lod, i32t);
                        let over = self.s.m.cmp(c, pred::IUGT, mip, max_mip);
                        let mip = self.s.m.select(c, over, max_mip, mip);
                        let dm = self
                            .s
                            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
                        let w = self.s.m.extract(c, dm, 0);
                        let ht = self.s.m.extract(c, dm, 1);
                        let half = self.s.m.f32(0.5f32.to_bits());
                        let clamp_pair = |l: &mut Self, coord: V, dim: V, off: V| {
                            let df = l.s.m.cast(c, Cast::UiToFp, dim, f32t);
                            let p = l.s.m.bin(c, Bin::Mul, coord, df);
                            let p = l.s.m.bin(c, Bin::Sub, p, half);
                            let fl = l.s.op(c, "unary.f32", Attr::ReadNone, f32t, 27, &[p]);
                            let t0 = l.s.m.cast(c, Cast::FpToSi, fl, i32t);
                            let t0 = l.s.m.bin(c, Bin::Add, t0, off);
                            let t1 = l.s.m.bin(c, Bin::Add, t0, one);
                            let hi = l.s.m.bin(c, Bin::Sub, dim, one);
                            let mut cl = |x: V| {
                                let lo_c = l.s.m.cmp(c, pred::SLT, x, zero);
                                let x = l.s.m.select(c, lo_c, zero, x);
                                let hi_c = l.s.m.cmp(c, pred::SGT, x, hi);
                                l.s.m.select(c, hi_c, hi, x)
                            };
                            (cl(t0), cl(t1))
                        };
                        let (x0, x1) = clamp_pair(self, cs[0], w, os[0]);
                        let (y0, y1) = clamp_pair(self, cs[1], ht, os[1]);
                        let ch = dmask.trailing_zeros();
                        let Operand::Vector { reg, .. } = *data else {
                            return Err("an image result");
                        };
                        let mut xs = Vec::with_capacity(4);
                        for (xi, yi) in [(x0, y1), (x1, y1), (x1, y0), (x0, y0)] {
                            let tl = self.s.op(
                                c,
                                "textureLoad.f32",
                                Attr::ReadOnly,
                                self.s.t.res_ret,
                                66,
                                &[h, mip, xi, yi, iu, iu, iu, iu],
                            );
                            let x = self.s.m.extract(c, tl, ch);
                            xs.push(if *d16 {
                                self.s
                                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[x])
                            } else {
                                self.as_int(x)
                            });
                        }
                        if *d16 {
                            let k16 = self.s.m.i32(16);
                            xs = xs
                                .chunks(2)
                                .map(|pair| {
                                    let lo = pair[0];
                                    match pair.get(1) {
                                        Some(&hi) => {
                                            let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                                            self.s.m.bin(c, Bin::Or, lo, hi)
                                        }
                                        None => lo,
                                    }
                                })
                                .collect();
                        }
                        for (k, x) in xs.into_iter().enumerate() {
                            self.write(inst, Reg::V(reg + k as u8), x, guard);
                        }
                        return Ok(());
                    }
                    args.extend(&os[..2]);
                    args.push(self.s.m.i32(dmask.trailing_zeros()));
                    self.s.op(
                        c,
                        "textureGather.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        73,
                        &args,
                    )
                }
            }
        } else {
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let mip = if name == "image_load_mip" {
                addr_val(self, coords)?
            } else if uav {
                // Our declarations have four channels.
                self.s.flags |= flags::TYPED_UAV_LOADS;
                iu
            } else {
                self.s.m.i32(0)
            };
            let mut args = vec![h, mip];
            args.extend(&cs);
            args.extend([iu, iu, iu]);
            self.s.op(
                c,
                "textureLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                66,
                &args,
            )
        };
        let Operand::Vector { reg, .. } = *data else {
            return Err("an image result");
        };
        let channels: Vec<u32> = if op == 73 {
            vec![0, 1, 2, 3]
        } else {
            (0..4).filter(|ch| dmask >> ch & 1 != 0).collect()
        };
        let mut xs = Vec::new();
        for ch in channels {
            let x = self.s.m.extract(c, r, ch);
            xs.push(if *d16 {
                // shortcut: d16 results are taken as floats (rounded to
                // nearest even); integer formats would keep 16 bits.
                let i32t = self.s.t.i32;
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[x])
            } else {
                self.as_int(x)
            });
        }
        if *d16 {
            // Two channels per VGPR, low half first (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            xs = xs
                .chunks(2)
                .map(|p| match *p {
                    [lo, hi] => {
                        let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                        self.s.m.bin(c, Bin::Or, lo, hi)
                    }
                    _ => p[0],
                })
                .collect();
        }
        for (k, x) in xs.into_iter().enumerate() {
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// The direction a cube sample's face coordinates stand for: the
    /// console takes s and t in [1, 2] (`v_cubesc`/`v_cubetc` over
    /// `|v_cubema|`, plus 1.5) and the face id (`v_cubeid`), which the
    /// host's cube sampling turns back into the same texel.
    // shortcut: the view is taken as a cube, not a cube array (whose face
    // operand adds 8 × the slice).
    fn cube_direction(&mut self, s: V, t: V, face: V) -> [V; 3] {
        let c = self.cur;
        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
        let (two, three) = (k(self, 2.0), k(self, 3.0));
        let u = self.s.m.bin(c, Bin::Mul, s, two);
        let u = self.s.m.bin(c, Bin::Sub, u, three);
        let v = self.s.m.bin(c, Bin::Mul, t, two);
        let v = self.s.m.bin(c, Bin::Sub, v, three);
        let z = self.s.m.f32(0x8000_0000);
        let (nu, nv) = (
            self.s.m.bin(c, Bin::Sub, z, u),
            self.s.m.bin(c, Bin::Sub, z, v),
        );
        let (one, mone) = (k(self, 1.0), k(self, -1.0));
        let f = self.s.m.cast(c, Cast::FpToUi, face, self.s.t.i32);
        // Per face +x, -x, +y, -y, +z, -z.
        let table = [
            [one, mone, u, u, u, nu],
            [nv, nv, one, mone, nv, nv],
            [nu, u, v, nv, one, mone],
        ];
        table.map(|row| {
            let mut r = row[0];
            for (n, &x) in row.iter().enumerate().skip(1) {
                let kn = self.s.m.i32(n as u32);
                let is = self.s.m.cmp(c, pred::EQ, f, kn);
                r = self.s.m.select(c, is, x, r);
            }
            r
        })
    }

    fn buffer(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        self.buffer_site = Some((b, i));
        let Fields::Buffer {
            data,
            addr,
            soffset,
            offset,
            offen,
            idxen,
            format,
            lds,
            tfe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if *lds || *tfe {
            return Err("a buffer access to LDS or with tfe");
        }
        if format.is_some() && !name.starts_with("tbuffer_load_format_") {
            return Err(name);
        }
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let store = name.starts_with("buffer_store_");
        // DXIL atomic operations: add, and, or, xor, signed and unsigned
        // min and max, exchange; compare-exchange apart.
        let atomic = match name.strip_prefix("buffer_atomic_") {
            None => None,
            Some("add") => Some(0),
            Some("and") => Some(1),
            Some("or") => Some(2),
            Some("xor") => Some(3),
            Some("smin") => Some(4),
            Some("smax") => Some(5),
            Some("umin") => Some(6),
            Some("umax") => Some(7),
            Some("swap") => Some(8),
            Some("cmpswap") => Some(u32::MAX),
            Some(_) => return Err(name),
        };
        // d16: two 16-bit channels per VGPR, low half first.
        let d16 = name.starts_with("buffer_") && name.contains("_format_d16_");
        let plain = name.replace("_format_d16_", "_format_");
        // XNW: bytes and halves a load reads within their dword
        // (bits, signed), as `global_load_*` does (VIctory).
        let narrow = match name {
            "buffer_store_byte" => Some((8u32, false)),
            "buffer_store_short" => Some((16, false)),
            "buffer_load_ubyte" => Some((8u32, false)),
            "buffer_load_sbyte" => Some((8, true)),
            "buffer_load_ushort" => Some((16, false)),
            "buffer_load_sshort" => Some((16, true)),
            _ => None,
        };
        let (typed, count) = match plain
            .strip_prefix("buffer_load_")
            .or(plain.strip_prefix("buffer_store_"))
            .or(plain.strip_prefix("tbuffer_load_"))
        {
            _ if atomic.is_some() => (false, 1),
            _ if narrow.is_some() => (false, 1),
            Some("dword") => (false, 1),
            Some("dwordx2") => (false, 2),
            Some("dwordx3") => (false, 3),
            Some("dwordx4") => (false, 4),
            Some("format_x") => (true, 1),
            Some("format_xy") => (true, 2),
            Some("format_xyz") => (true, 3),
            Some("format_xyzw") => (true, 4),
            _ => return Err(name),
        };
        let Operand::Vector { reg: areg, .. } = *addr else {
            return Err("a buffer address");
        };
        let vaddr = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a buffer address")?;
            l.i32_of(v)
        };
        let index = if *idxen { Some(vaddr(self, 0)?) } else { None };
        let mut off = self.s.m.i32(u32::from(*offset));
        if *offen {
            let o = vaddr(self, u8::from(*idxen))?;
            off = self.s.m.bin(c, Bin::Add, off, o);
        }
        if !matches!(
            soffset,
            Operand::Scalar { code: NULL, .. } | Operand::Const(128)
        ) {
            let so = self.operand(inst, soffset)?;
            off = self.s.m.bin(c, Bin::Add, off, so);
        }
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked buffer")?;
        let used = self.used_kind(src);
        let (base, extra) = self.built_offset(src)?;
        if base != 0 {
            let b = self.s.m.i32(base);
            off = self.s.m.bin(c, Bin::Add, off, b);
        }
        if let Some(x) = extra {
            off = self.s.m.bin(c, Bin::Add, off, x);
        }
        // stage b: a root view by address (not by index: its stride is the
        // code's, not a record's).
        let root = if typed || index.is_some() { None } else { self.root_view_base(src) };
        let paged = !typed && (self.paged(src) || root.is_some());
        if !paged {
            self.note_unpaged(src, typed);
        }
        if paged {
            self.declared_stride = used.map(|u| u.stride);
            // XNW: stages a and d): the page table is the only path
            // for raw loads, stores and atomics through a record.
            let uniform = !*offen && !*idxen;
            // A constant buffer indexed per lane: 16-byte rows (as below).
            let (index, off) = match index {
                Some(i) if used.is_some_and(|u| u.kind == ResourceType::Cbv) => {
                    let sixteen = self.s.m.i32(16);
                    let rows = self.s.m.bin(c, Bin::Mul, i, sixteen);
                    (None, self.s.m.bin(c, Bin::Add, rows, off))
                }
                other => (other, off),
            };
            if let Some(op) = atomic {
                let d = data.as_ref().ok_or("buffer atomic data")?;
                let vals = self.vgprs(inst, d, if op == u32::MAX { 2 } else { 1 })?;
                let Operand::Vector { reg, .. } = *d else {
                    return Err("buffer atomic data");
                };
                let returns = matches!(inst.fields, Fields::Buffer { cache, .. } if cache.glc);
                let old = match inst.read(Reg::V(reg)) {
                    Some(v) if returns => self.i32_of(v)?,
                    _ => self.s.m.i32(0),
                };
                let r = self.page_atomic(site, root, index, off, op, &vals, old, uniform, guard)?;
                if returns {
                    self.write(inst, Reg::V(reg), r, None);
                }
                return Ok(());
            }
            if store {
                let vals = self.vgprs(inst, data.as_ref().ok_or("buffer store data")?, count as u8)?;
                return self.page_store(site, root, index, off, &vals, uniform, guard);
            }
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            for (k, x) in self.page_load(site, root, index, off, count, uniform)?.into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        // XNW: a structured access through a root descriptor (the BVH
        // builder's root UAVs, stride in the program's metadata) is a raw
        // access at index × stride: D3D12 root views take no stride.
        let mut index = index;
        if let (Some(i), Source::Built { address, .. }, Some(u)) = (index, src, used)
            && !typed
            && u.stride >= 4
            && u.stride.is_multiple_of(4)
            && self.x.native.layout.get(address.param).is_some_and(|r| r.root == RootType::Descriptor)
            && u.kind != ResourceType::Cbv
        {
            let stride = self.s.m.i32(u.stride);
            let at = self.s.m.bin(c, Bin::Mul, i, stride);
            off = self.s.m.bin(c, Bin::Add, at, off);
            index = None;
        }
        // XNW: a byte or half: the raw dword holding it, shifted down.
        // shortcut: the field is taken not to cross a dword; an indexed one
        // needs the stride the program's metadata names.
        if let Some((bits, signed)) = narrow {
            if let Some(i) = index {
                let stride = used.map_or(0, |u| u.stride);
                if stride == 0 {
                    return Err("a byte or half of a structured buffer of unknown stride");
                }
                let k = self.s.m.i32(stride);
                let at = self.s.m.bin(c, Bin::Mul, i, k);
                off = self.s.m.bin(c, Bin::Add, at, off);
            }
            let u = if store || used.is_some_and(|u| u.kind == ResourceType::Uav) { 0x1000 } else { 0 };
            let h = self.handle(site, src, false, (11 | u, 0))?;
            let (three, not3) = (self.s.m.i32(3), self.s.m.i32(!3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            if store {
                // Two atomics, `and` then `or`, so lanes writing the other
                // bytes of the dword keep theirs (as LDS bytes, VIctory).
                let v = self.vgprs(inst, data.as_ref().ok_or("buffer store data")?, 1)?[0];
                let c = self.cur;
                let byte = self.s.m.bin(c, Bin::And, off, three);
                let sh = self.s.m.bin(c, Bin::Shl, byte, three);
                let mask = self.s.m.i32((1u32 << bits) - 1);
                let v = self.s.m.bin(c, Bin::And, v, mask);
                let v = self.s.m.bin(c, Bin::Shl, v, sh);
                let hole = self.s.m.bin(c, Bin::Shl, mask, sh);
                let all = self.s.m.i32(u32::MAX);
                let keep = self.s.m.bin(c, Bin::Xor, hole, all);
                let (iu, i32t) = (self.s.m.undef(self.s.t.i32), self.s.t.i32);
                let (and, or) = (self.s.m.i32(1), self.s.m.i32(2));
                let g = self.begin_guarded(guard);
                self.s.op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &[h, and, at, iu, iu, keep]);
                self.s.op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &[h, or, at, iu, iu, v]);
                self.end_guarded(g, &[]);
                return Ok(());
            }
            let w = self.raw_dword(h, at);
            let c = self.cur;
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            let x = self.s.m.bin(c, Bin::LShr, w, sh);
            let left = self.s.m.i32(32 - bits);
            let x = self.s.m.bin(c, Bin::Shl, x, left);
            let x = self.s.m.bin(c, if signed { Bin::AShr } else { Bin::LShr }, x, left);
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            self.write(inst, Reg::V(reg), x, guard);
            return Ok(());
        }
        if used.is_some_and(|u| u.kind == ResourceType::Cbv) && !typed {
            // A constant buffer read per lane (a dynamically indexed array).
            // shortcut: the console's CBV descriptor is taken to have a 16-byte
            // stride, as the vector loads of whole rows suggest.
            let gpu = self.x.gpu_desc;
            let h = self.handle(site, src, false, if gpu { (11, 0) } else { (13, 0x1_0000) })?;
            let byte = match index {
                Some(i) => {
                    let sixteen = self.s.m.i32(16);
                    let rows = self.s.m.bin(c, Bin::Mul, i, sixteen);
                    self.s.m.bin(c, Bin::Add, rows, off)
                }
                None => off,
            };
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            for k in 0..count {
                let four = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, byte, four);
                let x = if gpu { self.raw_dword(h, a) } else { self.cbuffer_dword(h, a) };
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let uav = store || atomic.is_some() || used.is_some_and(|u| u.kind == ResourceType::Uav);
        let u = if uav { 0x1000 } else { 0 };
        let iu = self.s.m.undef(self.s.t.i32);
        if let Some(op) = atomic {
            let d = data.as_ref().ok_or("buffer atomic data")?;
            let vals = self.vgprs(inst, d, if op == u32::MAX { 2 } else { 1 })?;
            let (h, coords) = match index {
                Some(index) => {
                    let stride = used.map_or(0, |u| u.stride);
                    if stride != 0 && (stride < 4 || !stride.is_multiple_of(4)) {
                        return Err("a structured buffer of unknown stride");
                    }
                    if stride >= 4 && stride.is_multiple_of(4) {
                        (
                            self.handle(site, src, false, (12 | u, stride))?,
                            [index, off, iu],
                        )
                    } else {
                        let (h, byte) = self.runtime_view(src, index, off, 11 | u)?;
                        (h, [byte, iu, iu])
                    }
                }
                None => (self.handle(site, src, false, (11 | u, 0))?, [off, iu, iu]),
            };
            let Operand::Vector { reg, .. } = *d else {
                return Err("buffer atomic data");
            };
            let returns = matches!(inst.fields, Fields::Buffer { cache, .. } if cache.glc);
            let old = match inst.read(Reg::V(reg)) {
                Some(v) if returns => self.i32_of(v)?,
                _ => self.s.m.i32(0),
            };
            let at = self.begin_guarded(guard);
            let i32t = self.s.t.i32;
            let r = if op == u32::MAX {
                // cmpswap: data is (new, compare).
                let mut args = vec![h];
                args.extend(coords);
                args.extend([vals[1], vals[0]]);
                self.s.op(
                    self.cur,
                    "atomicCompareExchange.i32",
                    Attr::NoUnwind,
                    i32t,
                    79,
                    &args,
                )
            } else {
                let mut args = vec![h, self.s.m.i32(op)];
                args.extend(coords);
                args.push(vals[0]);
                self.s
                    .op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &args)
            };
            let r = self.end_guarded(at, &[(r, old)])[0];
            if returns {
                self.write(inst, Reg::V(reg), r, None);
            }
            return Ok(());
        }
        if store {
            let regs = if d16 { count.div_ceil(2) } else { count };
            let mut vals =
                self.vgprs(inst, data.as_ref().ok_or("buffer store data")?, regs as u8)?;
            if d16 {
                // shortcut: d16 data is taken as f16 (integer formats would
                // keep 16 bits), widened for the 32-bit float store.
                let mut wide = Vec::new();
                for k in 0..count {
                    let mut x = vals[(k / 2) as usize];
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                    wide.push(self.as_int(f));
                }
                vals = wide;
            }
            let mut args = Vec::new();
            let (op, name) = if typed {
                // shortcut: typed stores are taken to write all four channels
                // of a float format.
                if count != 4 || *offen || *offset != 0 {
                    return Err("a typed buffer store of some channels or at an offset");
                }
                args.push(self.handle(site, src, false, (10 | u, 0x409))?);
                args.extend([index.unwrap_or(self.s.m.i32(0)), iu]);
                for x in vals {
                    args.push(self.as_float(x));
                }
                (69, "bufferStore.f32")
            } else {
                match index {
                    Some(index) => {
                        let stride = used.map_or(0, |u| u.stride);
                        if stride >= 4 && stride.is_multiple_of(4) {
                            args.push(self.handle(site, src, false, (12 | u, stride))?);
                            args.extend([index, off]);
                        } else {
                            let (h, byte) = self.runtime_view(src, index, off, 11 | u)?;
                            args.push(h);
                            args.extend([byte, iu]);
                        }
                    }
                    None => {
                        args.push(self.handle(site, src, false, (11 | u, 0))?);
                        args.extend([off, iu]);
                    }
                }
                args.extend(&vals);
                args.resize(3 + 4, iu);
                (140, "rawBufferStore.i32")
            };
            args.push(self.s.m.i8(if typed { 0xF } else { (1u8 << count) - 1 }));
            if !typed {
                args.push(self.s.m.i32(4));
            }
            let at = self.begin_guarded(guard);
            let void = self.s.t.void;
            self.s.op(self.cur, name, Attr::NoUnwind, void, op, &args);
            self.end_guarded(at, &[]);
            return Ok(());
        }
        if let Some(f) = *format {
            if index.is_some() {
                return Err("a typed-format load by index");
            }
            let h = self.handle(site, src, false, (11 | u, 0))?;
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            for (k, x) in self.format_load(h, off, f, count)?.into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let r = if typed {
            if *offen || matches!(off, V::Inst(_)) || *offset != 0 {
                return Err("a typed buffer load at an offset");
            }
            if uav {
                self.s.flags |= flags::TYPED_UAV_LOADS;
            }
            let h = self.handle(site, src, false, (10 | u, 0x409))?;
            let index = index.unwrap_or(self.s.m.i32(0));
            self.s.op(
                c,
                "bufferLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                68,
                &[h, index, iu],
            )
        } else {
            let mask = self.s.m.i8((1u8 << count) - 1);
            let align = self.s.m.i32(4);
            match index {
                Some(index) => {
                    let stride = used.map_or(0, |u| u.stride);
                    if stride != 0 && (stride < 4 || !stride.is_multiple_of(4)) {
                        return Err("a structured buffer of unknown stride");
                    }
                    if stride >= 4 && stride.is_multiple_of(4) {
                        let h = self.handle(site, src, false, (12 | u, stride))?;
                        self.s.op(
                            c,
                            "rawBufferLoad.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            139,
                            &[h, index, off, mask, align],
                        )
                    } else {
                        let (h, byte) = self.runtime_view(src, index, off, 11 | u)?;
                        self.s.op(
                            c,
                            "rawBufferLoad.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            139,
                            &[h, byte, iu, mask, align],
                        )
                    }
                }
                None => {
                    let h = self.handle(site, src, false, (11 | u, 0))?;
                    self.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, off, iu, mask, align],
                    )
                }
            }
        };
        let Some(Operand::Vector { reg, .. }) = *data else {
            return Err("a buffer result");
        };
        if d16 {
            // Two f16 channels per VGPR (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            let i32t = self.s.t.i32;
            for k in 0..count.div_ceil(2) {
                let mut x = None;
                for ch in (2 * k..(2 * k + 2).min(count)).rev() {
                    let f = self.s.m.extract(c, r, ch);
                    let h = self
                        .s
                        .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
                    x = Some(match x {
                        Some(hi) => {
                            let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                            self.s.m.bin(c, Bin::Or, h, hi)
                        }
                        None => h,
                    });
                }
                self.write(inst, Reg::V(reg + k as u8), x.expect("a channel"), guard);
            }
            return Ok(());
        }
        for k in 0..count {
            let x = self.s.m.extract(c, r, k);
            let x = self.as_int(x);
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// XNW: with the page table on, an access through `src` (and
    /// `typed`) that does not go through it: its kind joins the program's
    /// host-copy bits ([`super::host_copies`]).
    fn note_unpaged(&mut self, src: &Source, typed: bool) {
        if !(self.x.pagetable && self.x.gpu_desc) {
            return;
        }
        use super::host_copies as h;
        let bit = if typed {
            h::TYPED_BUFFERS
        } else {
            match src {
                Source::Built { address, .. }
                    if self.x.native.layout.get(address.param).is_some_and(|r| r.root == RootType::Descriptor) =>
                {
                    h::ROOT_VIEWS
                }
                Source::Built { .. } | Source::Constant(_) => h::BUILT_DESCRIPTORS,
                _ => return,
            }
        };
        let b = self.s.host_copies.get_or_insert(0);
        *b |= bit;
    }

    /// XNW: whether accesses through `src` go through the page
    /// table only: the switch is on, descriptors are GPU-side, and `src` is
    /// a record (not a root descriptor or one the code builds).
    fn paged(&self, src: &Source) -> bool {
        self.x.pagetable && self.x.gpu_desc && !matches!(src, Source::Built { .. } | Source::Constant(_))
    }

    /// XNW: stage b): for a buffer the code builds around a root
    /// descriptor (root SRV/CBV/UAV by address) with the page table on, the
    /// view's console address, from the two user-data dwords the runtime
    /// fills for it (spec section 6: root type Descriptor, dword 0 = bits
    /// 0-31, dword 1 = bits 32-63).
    fn root_view_base(&mut self, src: &Source) -> Option<(V, V)> {
        if !(self.x.pagetable && self.x.gpu_desc) {
            return None;
        }
        let Source::Built { address, .. } = src else { return None };
        let param = address.param;
        let rp = *self.x.native.layout.get(param)?;
        if rp.root != RootType::Descriptor {
            return None;
        }
        let index = u32::from(rp.user_data);
        let lo = self.user_extra(param, index, 0);
        let hi = self.user_extra(param, index + 1, 1);
        Some((lo, hi))
    }

    /// XNW (page table): the console
    /// address of an access of `count` dwords through the record at `site`,
    /// a GFX10 V# tagged [`super::PAGE_TAG`]: base (dword 0, dword 1 bits
    /// 0-15) + index × stride (dword 1 bits 16-29) + `off` in 64 bits, and
    /// whether it is to run: the tag is there and the access in range by
    /// num_records (dword 2) as GFX10 checks it (structured: the index; raw:
    /// the bytes). A record without the tag runs nothing (reads 0).
    fn page_address(&mut self, site: Site, root: Option<(V, V)>, index: Option<V>, off: V, count: u32) -> Result<(V, V, V), Why> {
        if let Some((base_lo, base_hi)) = root {
            // A root view: base + offset, no record and no range (the runtime
            // maps the view's pages up to its buffer's end).
            let c = self.cur;
            let lo = self.s.m.bin(c, Bin::Add, base_lo, off);
            let carry = self.s.m.cmp(c, pred::IULT, lo, base_lo);
            let carry = self.s.m.cast(c, Cast::ZExt, carry, self.s.t.i32);
            let hi = self.s.m.bin(c, Bin::Add, base_hi, carry);
            let _ = (site, index, count);
            return Ok((lo, hi, self.s.m.bool(true)));
        }
        let [w0, w1, w2, w3] = self.record_words(site)?;
        let c = self.cur;
        let k = |l: &mut Self, n: u32| l.s.m.i32(n);
        let (tag, k20) = (k(self, super::PAGE_TAG), k(self, 20));
        let high = self.s.m.bin(c, Bin::LShr, w3, k20);
        let is_page = self.s.m.cmp(c, pred::EQ, high, tag);
        let (m16, k16, m14) = (k(self, 0xFFFF), k(self, 16), k(self, 0x3FFF));
        let base_hi = self.s.m.bin(c, Bin::And, w1, m16);
        // The program's declared stride where it has one (spec 4: a record
        // may carry stride 0 for strides of 256 or more, which would fail
        // the title's own dword-1 check), else the record's.
        let stride = match self.declared_stride.take() {
            Some(s) if s > 0 => k(self, s),
            _ => {
                let s = self.s.m.bin(c, Bin::LShr, w1, k16);
                self.s.m.bin(c, Bin::And, s, m14)
            }
        };
        let zero = k(self, 0);
        let end = k(self, 4 * count);
        let last = self.s.m.bin(c, Bin::Add, off, end);
        let by_bytes = self.s.m.cmp(c, pred::IULE, last, w2);
        let (byte, in_range) = match index {
            Some(i) => {
                let e = self.s.m.bin(c, Bin::Mul, i, stride);
                let byte = self.s.m.bin(c, Bin::Add, e, off);
                let structured = self.s.m.cmp(c, pred::NE, stride, zero);
                let by_index = self.s.m.cmp(c, pred::IULT, i, w2);
                (byte, self.s.m.select(c, structured, by_index, by_bytes))
            }
            None => (off, by_bytes),
        };
        let lo = self.s.m.bin(c, Bin::Add, w0, byte);
        let carry = self.s.m.cmp(c, pred::IULT, lo, w0);
        let carry = self.s.m.cast(c, Cast::ZExt, carry, self.s.t.i32);
        let hi = self.s.m.bin(c, Bin::Add, base_hi, carry);
        let go = self.s.m.bin(c, Bin::And, is_page, in_range);
        Ok((lo, hi, go))
    }

    /// XNW: the page table's entry for console address (lo, hi): the
    /// handle of the page's raw SRV (`uav`: its UAV, index + 1), the byte of
    /// the address in it, and whether the page is mapped (if not, the handle
    /// is the null raw view, host slot 2 or 6). One lookup at the access's
    /// first byte; `uniform`: the same address in every lane.
    fn page_entry(&mut self, lo: V, hi: V, uav: bool, uniform: bool) -> (V, V, V) {
        let c = self.cur;
        let t = self.s.page_table_handle();
        let k = |l: &mut Self, n: u32| l.s.m.i32(n);
        let (spans, none, four, zero) = (k(self, 4096), k(self, u32::MAX), k(self, 4), k(self, 0));
        let inside = self.s.m.cmp(c, pred::IULT, hi, spans);
        let span = self.s.m.select(c, inside, hi, zero);
        let at = self.s.m.bin(c, Bin::Mul, span, four);
        let block = self.raw_dword(t, at);
        let block = self.s.m.select(c, inside, block, none);
        let has = self.s.m.cmp(c, pred::NE, block, none);
        let blk = self.s.m.select(c, has, block, zero);
        let (k16, k3, top, size) = (k(self, 16), k(self, 3), k(self, 16384), k(self, 524_288));
        let page = self.s.m.bin(c, Bin::LShr, lo, k16);
        let page = self.s.m.bin(c, Bin::Shl, page, k3);
        let first = self.s.m.bin(c, Bin::Mul, blk, size);
        let e = self.s.m.bin(c, Bin::Add, first, page);
        let e = self.s.m.bin(c, Bin::Add, e, top);
        let mut index = self.raw_dword(t, e);
        let e4 = self.s.m.bin(c, Bin::Add, e, four);
        let offset = self.raw_dword(t, e4);
        let named = self.s.m.cmp(c, pred::NE, index, none);
        let mapped = self.s.m.bin(c, Bin::And, has, named);
        if uav {
            let one = k(self, 1);
            index = self.s.m.bin(c, Bin::Add, index, one);
        }
        let null = k(self, if uav { 6 } else { 2 });
        let index = self.s.m.select(c, mapped, index, null);
        let props = (if uav { 0x100B } else { 11 }, 0);
        let h = if uniform {
            self.s.heap_handle(c, index, false, props)
        } else {
            self.s.heap_handle_nonuniform(c, index, props)
        };
        let m16 = k(self, 0xFFFF);
        let low = self.s.m.bin(c, Bin::And, lo, m16);
        let byte = self.s.m.bin(c, Bin::Add, offset, low);
        (h, byte, mapped)
    }

    /// XNW: `count` dwords through the record at `site` and the
    /// page table, the only path (0 where the record has no tag, the access
    /// is out of range or the page is missing).
    fn page_load(&mut self, site: Site, root: Option<(V, V)>, index: Option<V>, off: V, count: u32, uniform: bool) -> Result<Vec<V>, Why> {
        let (lo, hi, go) = self.page_address(site, root, index, off, count)?;
        let zero = self.s.m.i32(0);
        let at = self.begin_guarded(Some(go));
        let (h, byte, mapped) = self.page_entry(lo, hi, false, uniform);
        let c = self.cur;
        let mut vals = Vec::new();
        for n in 0..count {
            let d = self.s.m.i32(4 * n);
            let a = self.s.m.bin(c, Bin::Add, byte, d);
            let x = self.raw_dword(h, a);
            vals.push((self.s.m.select(c, mapped, x, zero), zero));
        }
        Ok(self.end_guarded(at, &vals))
    }

    /// XNW: stage d): a store of `vals` through the record at `site`
    /// and the page table's UAV (index + 1); dropped where the record has no
    /// tag, the access is out of range, the page is missing or `guard` is off.
    #[allow(clippy::too_many_arguments)]
    fn page_store(&mut self, site: Site, root: Option<(V, V)>, index: Option<V>, off: V, vals: &[V], uniform: bool, guard: Option<V>) -> Result<(), Why> {
        let (lo, hi, mut go) = self.page_address(site, root, index, off, vals.len() as u32)?;
        if let Some(g) = guard {
            go = self.s.m.bin(self.cur, Bin::And, go, g);
        }
        let at = self.begin_guarded(Some(go));
        let (h, byte, mapped) = self.page_entry(lo, hi, true, uniform);
        let inner = self.begin_guarded(Some(mapped));
        let c = self.cur;
        let iu = self.s.m.undef(self.s.t.i32);
        let mut args = vec![h, byte, iu];
        args.extend(vals);
        args.resize(3 + 4, iu);
        args.push(self.s.m.i8((1u8 << vals.len()) - 1));
        args.push(self.s.m.i32(4));
        let void = self.s.t.void;
        self.s.op(c, "rawBufferStore.i32", Attr::NoUnwind, void, 140, &args);
        self.end_guarded(inner, &[]);
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// XNW: stage d): a buffer atomic through the record at `site`
    /// and the page table's UAV: DXIL atomic op `code` (`u32::MAX`:
    /// compare-exchange, `vals` = (new, compare)) on the raw dword. Returns
    /// the old value, `old` where it does not run.
    #[allow(clippy::too_many_arguments)]
    fn page_atomic(&mut self, site: Site, root: Option<(V, V)>, index: Option<V>, off: V, code: u32, vals: &[V], old: V, uniform: bool, guard: Option<V>) -> Result<V, Why> {
        let (lo, hi, mut go) = self.page_address(site, root, index, off, 1)?;
        if let Some(g) = guard {
            go = self.s.m.bin(self.cur, Bin::And, go, g);
        }
        let at = self.begin_guarded(Some(go));
        let (h, byte, mapped) = self.page_entry(lo, hi, true, uniform);
        let inner = self.begin_guarded(Some(mapped));
        let c = self.cur;
        let i32t = self.s.t.i32;
        let iu = self.s.m.undef(i32t);
        let r = if code == u32::MAX {
            self.s.op(c, "atomicCompareExchange.i32", Attr::NoUnwind, i32t, 79, &[h, byte, iu, iu, vals[1], vals[0]])
        } else {
            let op = self.s.m.i32(code);
            self.s.op(c, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &[h, op, byte, iu, iu, vals[0]])
        };
        // An atomic on a missing page returns 0 (GFX10: out of range).
        let zero = self.s.m.i32(0);
        let r = self.end_guarded(inner, &[(r, zero)])[0];
        Ok(self.end_guarded(at, &[(r, old)])[0])
    }
    /// The four dwords of the descriptor record the instruction at `site`
    /// names (GPU-side descriptors).
    fn record_words(&mut self, site: Site) -> Result<[V; 4], Why> {
        let inst: &Inst = &self.x.p.blocks[site.0].insts[site.1];
        let op = match &inst.fields {
            Fields::Buffer { rsrc, .. } => rsrc,
            Fields::Smem { base, .. } => base,
            _ => return Err("a descriptor record read by this instruction"),
        };
        let Operand::Scalar { code, .. } = *op else {
            return Err("a descriptor record not in SGPRs");
        };
        let mut w = [V::Const(0); 4];
        for (k, slot) in w.iter_mut().enumerate() {
            let v = inst.read(Reg::S(code + k as u8)).ok_or("a descriptor record")?;
            *slot = self.i32_of(v)?;
        }
        Ok(w)
    }

    fn export(&mut self, b: usize, i: usize) -> Result<(), Why> {
        let e = *self
            .exports
            .get(&(b, i))
            .ok_or("an export I/O raising did not see")?;
        if e.compressed {
            return Err("a compressed export");
        }
        let guard = self.guard(b, i)?;
        let mesh = self.s.mesh.is_some();
        let c = self.cur;
        if let Some(g) = guard
            && !mesh
        {
            if !(e.done && self.s.pixel) {
                return Err("a conditional export");
            }
            let t = self.s.m.bool(true);
            let kill = self.s.m.bin(c, Bin::Xor, g, t);
            self.s
                .op(c, "discard", Attr::NoUnwind, self.s.t.void, 82, &[kill]);
        }
        if mesh && e.target == Target::Primitive {
            if self.s.fallback {
                return self.fallback_indices(e, guard);
            }
            return self.emit_indices(e, guard);
        }
        if self.s.fallback && matches!(e.target, Target::PrimitiveParam { .. }) {
            return Err("a per-primitive output in the compute fallback");
        }
        let param_row = |usage: Usage, row: u8| match usage {
            // The 2020 compiler leaves a mesh program's first entry zeroed
            // (usage "position"): its row is the one no other entry names.
            Usage::Other(0) if mesh => self.unnamed_row(),
            _ => Some(row),
        };
        // (output element, channel) per exported channel; per-primitive
        // outputs are elements of `prims`.
        let mut stores = Vec::new();
        let primitive = matches!(e.target, Target::PrimitiveParam { .. });
        let outputs = if primitive {
            &self.s.prims
        } else {
            &self.s.outputs
        };
        let by_kind = |kind: u8| outputs.iter().position(|o| o.kind == kind);
        // 16-bit vertex outputs: the halves each dword carries, as (output
        // element, channel, high half) per channel.
        let halves = match e.target {
            Target::Param { usage, row } => {
                param_row(usage, row).and_then(|r| self.half_outputs.get(&r).map(|&h| (r, h)))
            }
            _ => None,
        };
        if let Some((row, (pack2, b))) = halves {
            for ch in 0..4u8 {
                let Some(value) = e.values[usize::from(ch)] else {
                    continue;
                };
                for high in [false, true] {
                    let h = 2 * ch + u8::from(high);
                    let src = if pack2 {
                        packed_half(outputs, row, b, h)
                    } else {
                        (!high).then_some((row, ch))
                    };
                    let found = src.and_then(|(r, c)| {
                        outputs.iter().position(|o| {
                            o.row == Some(u32::from(r)) && c >= o.col && c < o.col + o.cols
                        })
                    });
                    if let Some(id) = found {
                        let col = src.expect("found").1 - outputs[id].col;
                        stores.push((id, col, value, Some(high)));
                    }
                }
            }
        }
        for ch in 0..4u8 {
            let Some(value) = e.values[usize::from(ch)] else {
                continue;
            };
            if halves.is_some() {
                break;
            }
            let found = match e.target {
                Target::Color(n) => outputs
                    .iter()
                    .position(|o| o.kind == 16 && o.row == Some(u32::from(n)))
                    .map(|id| (id, ch)),
                Target::Position(0) => by_kind(3).map(|id| (id, ch)),
                Target::Depth => match ch {
                    0 => by_kind(17)
                        .or(by_kind(18))
                        .or(by_kind(19))
                        .map(|id| (id, 0)),
                    1 => by_kind(20).map(|id| (id, 0)),
                    2 => by_kind(14).map(|id| (id, 0)),
                    _ => None,
                },
                Target::Param { usage, row } | Target::PrimitiveParam { usage, row } => {
                    let row = param_row(usage, row);
                    outputs
                        .iter()
                        .position(|o| {
                            row.is_some_and(|r| o.row == Some(u32::from(r)))
                                && ch >= o.col
                                && ch < o.col + o.cols
                        })
                        .map(|id| (id, ch - outputs[id].col))
                }
                Target::Null | Target::Unread => None,
                Target::Position(n @ 1..=2) => {
                    // shortcut: without the misc vector (point size, render
                    // target or viewport index) pos1 and pos2 carry the clip
                    // then cull distances, in semantic order.
                    if outputs.iter().any(|o| matches!(o.kind, 4 | 5)) {
                        return Err("render-target or viewport index exports");
                    }
                    let mut dists: Vec<usize> = (0..outputs.len())
                        .filter(|&i| matches!(outputs[i].kind, 6 | 7))
                        .collect();
                    dists.sort_by_key(|&i| (outputs[i].kind, outputs[i].index));
                    let flat: Vec<(usize, u8)> = dists
                        .iter()
                        .flat_map(|&i| (0..outputs[i].cols).map(move |c| (i, c)))
                        .collect();
                    // Channels past the distances the signature declares are
                    // ones the clipper does not enable (`PA_CL_VS_OUT_CNTL`).
                    let k = usize::from(n - 1) * 4 + usize::from(ch);
                    flat.get(k).copied()
                }
                Target::Position(_) => return Err("clip, cull or point-size exports"),
                Target::Primitive => return Err("a primitive export"),
            };
            if let Some((id, col)) = found {
                stores.push((id, col, value, None));
            }
        }
        fn elements<'l>(l: &'l mut Lower, primitive: bool) -> &'l mut Vec<SigElement> {
            if primitive {
                &mut l.s.prims
            } else {
                &mut l.s.outputs
            }
        }
        if stores
            .iter()
            .any(|s| s.3.is_none() && elements(self, primitive)[s.0].half)
        {
            return Err("a 16-bit output the stage data does not mark");
        }
        // A mesh lane writes its own vertex or primitive, where it exports.
        let at = if mesh {
            self.begin_guarded(guard)
        } else {
            None
        };
        let (name, opcode) = match (mesh, primitive) {
            (false, _) => ("storeOutput", 5),
            (true, false) => ("storeVertexOutput", 171),
            (true, true) => ("storePrimitiveOutput", 172),
        };
        for (id, col, value, half) in stores {
            let mut x = self.i32_of(value)?;
            if let Some(high) = half {
                // An f16 half, declared a 32-bit float on the host.
                let c = self.cur;
                if high {
                    let k16 = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, k16);
                }
                let f32t = self.s.t.f32;
                let f = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                x = self.as_int(f);
            }
            if self.s.fallback {
                self.fallback_store(id, col, x)?;
                continue;
            }
            let el = &mut elements(self, primitive)[id];
            el.usage |= 1 << col;
            let (suffix, x) = match el.comp {
                9 if half.is_some() => (".f32", self.as_float(x)),
                9 => {
                    let f = match self.floats.get(&value) {
                        Some(&f) => f,
                        None => self.as_float(x),
                    };
                    (".f32", f)
                }
                _ => (".i32", x),
            };
            let mut args = vec![
                self.s.m.i32(id as u32),
                self.s.m.i32(0),
                self.s.m.i8(col),
                x,
            ];
            if mesh {
                args.push(self.mesh_thread());
            }
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                &format!("{name}{suffix}"),
                Attr::NoUnwind,
                void,
                opcode,
                &args,
            );
        }
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Mesh: `exp prim` as the lane's primitive. Bit 31 culls it (all
    /// indices 0: no area).
    fn emit_indices(&mut self, e: &Export, guard: Option<V>) -> Result<(), Why> {
        let v = e.values[0].ok_or("a primitive export without its indices")?;
        let x = self.i32_of(v)?;
        let at = self.begin_guarded(guard);
        let c = self.cur;
        let z = self.s.m.i32(0);
        let culled = self.s.m.cmp(c, pred::SLT, x, z);
        let mut args = vec![self.mesh_thread()];
        for k in 0..3 {
            // VIctory (c0228d9): a 9-bit index; bits 9, 19, 29 are
            // edge flags, no part of it.
            let (shift, mask) = (self.s.m.i32(10 * k), self.s.m.i32(0x1FF));
            let i = self.s.m.bin(c, Bin::LShr, x, shift);
            let i = self.s.m.bin(c, Bin::And, i, mask);
            args.push(self.s.m.select(c, culled, z, i));
        }
        let void = self.s.t.void;
        self.s
            .op(c, "emitIndices", Attr::NoUnwind, void, 169, &args);
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// The per-vertex arbitrary row no mesh output entry names but one
    /// (the 2020 compiler's zeroed entry).
    fn unnamed_row(&self) -> Option<u8> {
        let StageIo::Ms(ms) = &self.x.native.io else {
            return None;
        };
        let named = |r: u32| {
            ms.outputs
                .iter()
                .any(|o| o.param & 0x80 == 0 && o.usage != Usage::Other(0) && u32::from(o.row) == r)
        };
        let free: std::collections::BTreeSet<u32> = self
            .s
            .outputs
            .iter()
            .filter(|o| o.kind == 0)
            .filter_map(|o| o.row)
            .filter(|&r| !named(r))
            .collect();
        match free.into_iter().collect::<Vec<_>>()[..] {
            [r] => u8::try_from(r).ok(),
            _ => None,
        }
    }

    fn exit(&mut self, b: usize) -> Result<(), Stop> {
        let offset = self.x.p.blocks[b]
            .insts
            .last()
            .map_or(self.x.p.blocks[b].offset, |i| i.machine.offset);
        let term = match self.x.st.exits[b] {
            LaneExit::Jump(t) if self.counters.contains_key(&t) && self.rpo[t] <= self.rpo[b] => {
                let (next, ok) = self.count_iteration(pred::IULT);
                self.report_loop(t, next, None);
                self.flow(t, self.cur, next);
                let abort = self.abort_block();
                Term::CondBr(ok, self.block[t].expect("reachable"), abort)
            }
            LaneExit::Jump(t) => {
                let budget = self.budget.expect("block started");
                self.flow(t, self.cur, budget);
                Term::Br(self.block[t].expect("reachable"))
            }
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                // A negated condition branches the other way round.
                let (cond, taken, not_taken) = match self.x.st.bools.node(when) {
                    Some((atom, Bool::TRUE, Bool::FALSE)) => {
                        let a = self
                            .atom(atom, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (a, not_taken, taken)
                    }
                    _ => {
                        let c = self
                            .boolean(when, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (c, taken, not_taken)
                    }
                };
                // A back edge is taken only below the iteration limit.
                let back =
                    |l: &Self, to: usize| l.counters.contains_key(&to) && l.rpo[to] <= l.rpo[b];
                // (No `xor` of a compare: a loop exit testing one hung the
                // AMD driver.)
                let budget = self.budget.expect("block started");
                let (cond, next) = if back(self, taken) {
                    let (next, ok) = self.count_iteration(pred::IULT);
                    let no = self.s.m.bool(false);
                    let leaving = self.s.m.cmp(self.cur, pred::EQ, cond, no);
                    let c2 = self.s.m.bin(self.cur, Bin::And, cond, ok);
                    self.report_loop(taken, next, Some(leaving));
                    (c2, next)
                } else if back(self, not_taken) {
                    let (next, stop) = self.count_iteration(pred::IUGE);
                    let c2 = self.s.m.bin(self.cur, Bin::Or, cond, stop);
                    self.report_loop(not_taken, next, Some(cond));
                    (c2, next)
                } else {
                    (cond, budget)
                };
                // Critical edges get a block of their own, so loops have
                // dedicated exits and preheaders as DXC emits them.
                let (t, f) = (self.edge(b, taken), self.edge(b, not_taken));
                for (to, via) in [(taken, t), (not_taken, f)] {
                    // From an edge block, or from here.
                    let from = if Some(via) == self.block[to] {
                        self.cur
                    } else {
                        via
                    };
                    // The budget leaves counted (back edge) or as it came.
                    let value = if back(self, to) { next } else { budget };
                    self.flow(to, from, value);
                }
                Term::CondBr(cond, t, f)
            }
            LaneExit::End => {
                if self.s.pixel && !self.exported[b] && !self.has_done_export(b) {
                    // A lane leaving without its final export was killed.
                    let t = self.s.m.bool(true);
                    self.s
                        .op(self.cur, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
                }
                Term::Ret
            }
            LaneExit::Indirect => return Err(unsupported(offset, "an indirect jump")),
        };
        let _ = Exit::End;
        // XNW: a loop-count report (`report_loop`) may have moved on
        // to a block of its own; the block's values leave from there.
        self.tail[b] = Some(self.cur);
        self.s.m.terminate(self.cur, term);
        Ok(())
    }

    /// XNW: `XNW_LOOP_COUNTS=1`, contract in xnw-gpu
    /// `d3d12x/loopcount.rs`): at a back edge to loop `header`, every 64
    /// passes and where a lane leaves (`leaving`), the wave's largest count
    /// so far goes to word `slot * 8 + k` (k: the loop's place in program
    /// order, past the 8th: 7) of the runtime's buffer by an atomic max of
    /// its first active lane. The count is the invocation's loop budget
    /// spent so far (one budget for all its loops).
    fn report_loop(&mut self, header: usize, count: V, leaving: Option<V>) {
        let Some(slot) = self.x.loop_counts else { return };
        let p = self.x.p;
        let mut headers: Vec<usize> = (0..p.blocks.len())
            .filter(|&h| self.block[h].is_some() && p.blocks[h].preds.iter().any(|&q| self.block[q].is_some() && self.rpo[h] <= self.rpo[q]))
            .collect();
        headers.sort_by_key(|&h| p.blocks[h].offset);
        let k = headers.iter().position(|&h| h == header).unwrap_or(0).min(LOOP_COUNT_LOOPS as usize - 1) as u32;
        let c = self.cur;
        let (k63, zero) = (self.s.m.i32(63), self.s.m.i32(0));
        let low = self.s.m.bin(c, Bin::And, count, k63);
        let mut report = self.s.m.cmp(c, pred::EQ, low, zero);
        if let Some(l) = leaving {
            report = self.s.m.bin(c, Bin::Or, report, l);
        }
        let index = self.loop_count_buffer();
        let at = self.begin_guarded(Some(report));
        let c = self.cur;
        let i32t = self.s.t.i32;
        let (max, unsigned) = (self.s.m.i8(3), self.s.m.i8(1));
        let most = self.s.op(c, "waveActiveOp.i32", Attr::NoUnwind, i32t, 119, &[count, max, unsigned]);
        let i1 = self.s.t.i1;
        let first = self.s.op(c, "waveIsFirstLane", Attr::NoUnwind, i1, 110, &[]);
        self.s.flags |= flags::WAVE_OPS;
        let inner = self.begin_guarded(Some(first));
        let c = self.cur;
        let h = self.s.heap_handle(c, index, false, (11 | 0x1000, 0));
        let byte = self.s.m.i32((slot * LOOP_COUNT_LOOPS + k) * 4);
        let iu = self.s.m.undef(i32t);
        let umax = self.s.m.i32(7);
        self.s.op(c, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &[h, umax, byte, iu, iu, most]);
        self.end_guarded(inner, &[]);
        self.end_guarded(at, &[]);
    }

    /// XNW: the heap index of the loop-count buffer, from the
    /// user-data dword of root type `Other(LOOP_COUNT_ROOT)` the host fills:
    /// placed after every dword the program's parameters and their window
    /// dwords may take.
    fn loop_count_buffer(&mut self) -> V {
        if let Some(v) = self.loop_count_index {
            return v;
        }
        let tables = self
            .x
            .native
            .layout
            .iter()
            .filter(|r| matches!(r.root, RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap))
            .count() as u32;
        let index = self.window_base() + 2 * tables;
        self.user_reads.insert(index, UserData { index, slot: 0, root: RootType::Other(LOOP_COUNT_ROOT), dword: 0 });
        self.s.user_data = self.s.user_data.max(index + 1);
        let h = match self.user_data {
            Some(h) => h,
            None => {
                let h = self.s.user_data_handle();
                self.user_data = Some(h);
                h
            }
        };
        let entry = self.s.entry;
        let row = self.s.m.i32(index / 4);
        let r = self.s.op(entry, "cbufferLoadLegacy.i32", Attr::ReadOnly, self.s.t.cbuf_ret, 59, &[h, row]);
        let v = self.s.m.extract(entry, r, index % 4);
        self.loop_count_index = Some(v);
        v
    }

    /// The budget `value` arrives at IR block `to` from DXIL block `via`.
    fn flow(&mut self, to: usize, via: usize, value: V) {
        if let Some((_, incoming)) = self.counters.get_mut(&to) {
            incoming.push((value, via));
        }
    }

    /// A back edge (it keeps its block, no critical-edge block): the
    /// budget goes up by one; the comparison of it with the stage's limit by
    /// `p` (below: go on; at or above: stop).
    fn count_iteration(&mut self, p: u8) -> (V, V) {
        let c = self.cur;
        let budget = self.budget.expect("block started");
        let one = self.s.m.i32(1);
        let next = self.s.m.bin(c, Bin::Add, budget, one);
        let limit = self.s.m.i32(super::loop_limit_for(self.x.native.stage, self.x.fallback));
        (next, self.s.m.cmp(c, p, next, limit))
    }

    /// The block a lane stopped by [`LOOP_LIMIT`] leaves through.
    fn abort_block(&mut self) -> usize {
        if let Some(a) = self.abort {
            return a;
        }
        let a = self.s.m.new_block();
        if self.s.pixel && self.x.probe && !self.s.fallback {
            // XNW: `XNW_LOOP_PROBE`): magenta in every target, so one
            // run shows where the budget ran out.
            let targets: Vec<usize> = (0..self.s.outputs.len()).filter(|&i| self.s.outputs[i].kind == 16).collect();
            for id in targets {
                let e = &mut self.s.outputs[id];
                let (cols, float) = (e.cols, e.comp == 9);
                e.usage |= (1 << cols) - 1;
                for col in 0..cols {
                    let on = col != 1;
                    let (x, suffix) = if float {
                        (self.s.m.f32(if on { 1.0f32.to_bits() } else { 0 }), ".f32")
                    } else {
                        (self.s.m.i32(if on { u32::MAX } else { 0 }), ".i32")
                    };
                    let args = [self.s.m.i32(id as u32), self.s.m.i32(0), self.s.m.i8(col), x];
                    let void = self.s.t.void;
                    self.s.op(a, &format!("storeOutput{suffix}"), Attr::NoUnwind, void, 5, &args);
                }
            }
        } else if self.s.pixel {
            let t = self.s.m.bool(true);
            self.s
                .op(a, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
        }
        self.s.m.terminate(a, Term::Ret);
        self.abort = Some(a);
        a
    }

    /// The DXIL block a branch from IR block `from` to `to` goes to.
    fn edge(&mut self, from: usize, to: usize) -> usize {
        let target = self.block[to].expect("reachable");
        let preds = self.x.p.blocks[to]
            .preds
            .iter()
            .filter(|&&q| self.block[q].is_some())
            .count();
        // Back edges stay as they are: the exiting block is the latch.
        if preds < 2 || self.rpo[to] <= self.rpo[from] {
            return target;
        }
        let e = self.s.m.new_block();
        self.s.m.terminate(e, Term::Br(target));
        self.edges.insert((from, to), e);
        e
    }

    /// A dword a scalar load writes, where the load is a descriptor the
    /// program uses as a resource or sampler.
    fn descriptor_dword(&self, v: Value) -> bool {
        let Some(load) = self.x.p.blocks.iter().enumerate().find_map(|(b, block)| {
            block
                .insts
                .iter()
                .position(|i| i.writes.iter().any(|w| w.1 == v))
                .map(|i| (b, i))
        }) else {
            return false;
        };
        self.x.p.blocks[load.0].insts[load.1]
            .name()
            .starts_with("s_load_dword")
            && self.x.res.descriptors.iter().any(|d| {
                d.source
                    .leaves()
                    .iter()
                    .any(|l| matches!(l, Source::Loaded { load: at, .. } if *at == load))
            })
    }

    fn fill_phis(&mut self) -> Result<(), Stop> {
        let p = self.x.p;
        // A phi that only another read phi consumes is live too. Without this, a value with no DXIL form that
        // reaches a read through two joins is taken for dead at the first join and replaced by zero below, and the
        // lowering accepts what it refuses when the same value is read directly.
        let mut work: Vec<Value> = self.read_phis.iter().copied().collect();
        while let Some(v) = work.pop() {
            let Some(&(b, i)) = self.phi_at.get(&v) else {
                continue;
            };
            for &arg in &p.blocks[b].phis[i].args {
                if self.phi_at.contains_key(&arg) && self.read_phis.insert(arg) {
                    work.push(arg);
                }
            }
        }
        while let Some((phi, b, i, base)) = self.pending_addr_phis.pop() {
            let ir_phi = &p.blocks[b].phis[i];
            let offset = p.blocks[b].offset;
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let saved_cur = self.cur;
                let saved_ir = self.cur_ir;
                self.cur = qb;
                self.cur_ir = q;
                let mut local = HashMap::new();
                let x = self.rel_offset(ir_phi.args[n], base, &mut local);
                self.cur = saved_cur;
                self.cur_ir = saved_ir;
                let x = x.map_err(|w| unsupported(offset, w))?;
                incoming.push((x, qb));
            }
            self.s.m.set_phi(phi, incoming);
        }
        // XNW change: every phi found unreadable in one pass goes back at
        // once (was the first only: a program with 100 such phis was
        // lowered 100 times).
        let mut retry = Vec::new();
        for (v, b, i) in std::mem::take(&mut self.data_phis) {
            let phi = &p.blocks[b].phis[i];
            let bit = phi.reg == Reg::Scc;
            // A register initialized only on one branch may carry a dead
            // descriptor dword left over from earlier register reuse on the
            // other branch (guarded by a separate predicate). Only treat a
            // descriptor-dword `Missing` as fatal when no incoming edge carries
            // a real data value.
            let has_data = !bit
                && p.blocks[b]
                    .preds
                    .iter()
                    .enumerate()
                    .filter(|&(_, &q)| self.tail[q].is_some())
                    .any(|(n, _)| matches!(self.vals.get(&phi.args[n]), Some(Val::I32(_))));
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let arg = phi.args[n];
                let x = match (self.vals.get(&arg).copied(), bit) {
                    (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => x,
                    (Some(Val::Missing(why)), false)
                        if has_data
                            && matches!(
                                why,
                                "a descriptor's address or format read as data"
                                    | "a descriptor read only as data"
                                    // XNW: the same for a constant
                                    // buffer's (ped pixel programs, 957fc3f6
                                    // at 0x2ac8: s[20:27] is data on one
                                    // edge, a CBV descriptor on the other).
                                    | "a constant buffer's descriptor read as data"
                                    // XNW: and for a sampler heap entry's
                                    // (ped pixel programs: s[12:15] hold the
                                    // entry on one edge, constants on the other).
                                    | "a sampler descriptor read as data"
                            )
                            // XNW: an SGPR that held a lane mask and is
                            // then reused for data: the mask edge of the join
                            // is taken as dead (as the descriptor dwords above).
                            || has_data && why == LANE_MASK =>
                    {
                        self.s.m.i32(0)
                    }
                    (Some(Val::Bit(_)), false) if self.read_phis.contains(&phi.value) => {
                        retry.push((phi.value, "an SCC joined with SGPR values"));
                        break;
                    }
                    // VIctory (permadiego 7ab94ab): a lane mask's dword
                    // reaching a join that also takes data: the registers
                    // are reused once the mask is dead.
                    // shortcut: undefined on that edge, assuming the program
                    // reads the joined value only on paths from a data edge.
                    // Changed from upstream: only where another edge brings
                    // data (`has_data`); a join of masks on every edge (a
                    // carry-in from a divergent join) stays refused (test
                    // a_carry_in_from_a_divergent_join_is_refused).
                    (Some(Val::Missing(_)), false)
                        if has_data
                            && (self.x.st.masks.contains_key(&arg)
                                || self.x.st.highs.contains_key(&arg)) =>
                    {
                        self.s.m.undef(self.s.t.i32)
                    }
                    // A descriptor's dwords reaching a join that also takes
                    // data (the registers are reused after the last access).
                    // shortcut: undefined on that edge, assuming the program
                    // reads the joined value only after the data edge.
                    // Changed from VIctory: compute only; on mesh
                    // programs it turned 15 translations into lane-mask
                    // refusals (pack_listing).
                    (Some(Val::Missing(_)), _)
                        if self.x.native.stage == HwStage::Cs && self.descriptor_dword(arg) =>
                    {
                        if bit {
                            self.s.m.undef(self.s.t.i1)
                        } else {
                            self.s.m.undef(self.s.t.i32)
                        }
                    }
                    (Some(Val::Missing(why)), _) if self.read_phis.contains(&phi.value) => {
                        retry.push((phi.value, why));
                        break;
                    }
                    _ if self.read_phis.contains(&phi.value) => {
                        retry.push((phi.value, "a join of mismatched values"));
                        break;
                    }
                    _ if bit => self.s.m.bool(false),
                    _ => self.s.m.i32(0),
                };
                incoming.push((x, qb));
            }
            if !retry.is_empty() {
                continue;
            }
            self.s.m.set_phi(v, incoming);
        }
        if !retry.is_empty() {
            return Err(Stop::Retry(retry));
        }
        for (phi, incoming) in std::mem::take(&mut self.counters).into_values() {
            self.s.m.set_phi(phi, incoming);
        }
        while let Some((v, b, i)) = self.pending_masks.pop() {
            let phi = &p.blocks[b].phis[i];
            let offset = p.blocks[b].offset;
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let m =
                    *self.x.st.masks.get(&phi.args[n]).ok_or_else(|| {
                        unsupported(offset, "a lane-mask join of an untracked mask")
                    })?;
                let x = self.boolean(m, qb).map_err(|w| unsupported(offset, w))?;
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Lowered, Numeric, Parts, buffer_format, half_shift, lower};
    use crate::native::{ComputeIo, HwStage, NativeProgram, StageIo};
    use crate::{io, ir, resource, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;

    /// Lowers `words` as a `stage` program (`wave64`: the console's wave
    /// size) with no resources or inputs.
    fn lower_program(
        words: &[u32],
        stage: HwStage,
        wave64: bool,
        io: StageIo,
    ) -> Result<Lowered, crate::ShaderError> {
        lower_with(words, stage, wave64, io, vec![], vec![])
    }

    /// [`lower_program`] with a root layout and the resources it uses.
    fn lower_with(
        words: &[u32],
        stage: HwStage,
        wave64: bool,
        io: StageIo,
        layout: Vec<crate::RootParam>,
        used: Vec<crate::UsedResource>,
    ) -> Result<Lowered, crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage,
            wave64,
            vgprs: 256,
            layout,
            used,
            io,
            code,
        };
        let p = ir::build(&native.code, wave64, 256).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            fallback: false,
            fma: crate::emit::Fma::Exact,
            gpu_desc: false,
            probe: false,
            loop_counts: None,
            pagetable: false,
        })
    }

    /// Whether a synthetic compute program translates with these input registers.
    fn lowers(words: &[u32], rsrc2: u32) -> Result<(), crate::ShaderError> {
        lower_program(
            words,
            HwStage::Cs,
            false,
            StageIo::Cs(ComputeIo {
                rsrc2,
                threads: [1, 1, 1],
            }),
        )
        .map(|_| ())
    }

    #[test]
    fn a_scalar_carry_chain_reaches_the_select() {
        // The high dword of a 64-bit add or subtract, then a select on the
        // carry (borrow) the high half leaves in SCC, and a branch on it
        // (only code with an effect needs its operands).
        let add = [
            0xBE80_03C1, // s_mov_b32 s0, -1
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8000_8100, // s_add_u32 s0, s0, 1
            0x8201_8001, // s_addc_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];
        let sub = [
            0xBE80_0380, // s_mov_b32 s0, 0
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8080_8100, // s_sub_u32 s0, s0, 1
            0x8281_8001, // s_subb_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];
        for words in [add, sub] {
            lowers(&words, 0).unwrap();
        }
    }

    #[test]
    fn a_missing_value_stays_missing_through_two_live_joins() {
        let mut words = [
            0xF490_0180,
            0,           // s_memtime s[6:7]
            0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
            0xBF85_0001, // skip the first overwrite
            0xBE86_0380, // s_mov_b32 s6, 0
            0xBF06_8101, // s_cmp_eq_u32 s1, 1 (s1 = group id Y)
            0xBF85_0001, // skip the second overwrite
            0xBE86_0381, // s_mov_b32 s6, 1
            0xBF07_8006, // s_cmp_lg_u32 s6, 0
            0xBF85_0001, // use the result to branch
            0xBE83_0380,
            S_ENDPGM,
        ];
        let group_ids_xy = 3 << 7;
        // Compare to a direct read so future timestamp support works in both cases.
        let refusal = |code: &[u32]| match lowers(code, group_ids_xy) {
            Ok(()) => None,
            Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
            Err(error) => panic!("{error}"),
        };
        let direct = [&words[..2], &words[8..]].concat();
        assert_eq!(refusal(&words), refusal(&direct));
        words[8] = 0xBF07_8000; // read the group id instead; the timestamp is dead
        lowers(&words, group_ids_xy).expect("dead timestamp");
    }

    /// VIctory (c0228d9, GPL-3.0-or-later; its program, our check):
    /// a primitive export's corners are 9-bit indices, the edge flags
    /// above them (bits 9, 19, 29) no part of them: an edge-flagged
    /// triangle (0, 1, 2) masks each corner with 0x1FF, never 0x3FF.
    #[test]
    fn primitive_export_indices_drop_the_edge_flags() {
        const TRIANGLE: u32 = 1 << 10 | 2 << 20;
        const EDGE_FLAGS: u32 = 1 << 9 | 1 << 19 | 1 << 29;
        let words = [
            0xD765_0001u32, // v_mbcnt_lo_u32_b32 v1, -1, 0
            0x0001_00C1,
            0x3604_0281, // v_and_b32 v2, 1, v1
            0x3606_0282, // v_and_b32 v3, 2, v1
            0x7E04_0D02, // v_cvt_f32_u32 v2, v2
            0x7E06_0D03, // v_cvt_f32_u32 v3, v3
            0x1008_04F6, // v_mul_f32 v4, 4.0, v2
            0x0608_08F3, // v_add_f32 v4, -1.0, v4
            0x100A_06F4, // v_mul_f32 v5, 2.0, v3
            0x060A_0AF3, // v_add_f32 v5, -1.0, v5
            0x7E0C_02F0, // v_mov_b32 v6, 0.5
            0x7E0E_02F2, // v_mov_b32 v7, 1.0
            0x7E12_02FF, // v_mov_b32 v9, prim
            TRIANGLE | EDGE_FLAGS,
            0xF800_0941, // exp prim v9, off, off, off done
            0x0000_0009,
            0xF800_08CF, // exp pos0 v4, v5, v6, v7 done
            0x0706_0504,
            S_ENDPGM,
        ];
        let native = NativeProgram {
            stage: HwStage::Ms,
            wave64: false,
            vgprs: 10,
            layout: vec![],
            used: vec![],
            io: StageIo::Ms(crate::native::MeshIo {
                outputs: vec![],
                clip_cntl: 0,
                threads: 32,
                max_vertices: 32,
                max_primitives: 32,
                topology: 2,
                lds_bytes: 0,
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let position = super::SigElement {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0xF,
            half: false,
        };
        let p = ir::build(&native.code, false, 10).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let l = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![position],
            prims: vec![],
            mesh_stride: 0,
            fallback: false,
            fma: crate::emit::Fma::Exact,
            gpu_desc: false,
            probe: false,
            loop_counts: None,
            pagetable: false,
        })
        .unwrap();
        let masks = |k: &str| l.insts.iter().filter(|i| i.starts_with("bin:And") && i.ends_with(k)).count();
        assert_eq!((masks(" k511"), masks(" k1023")), (3, 0), "{:?}", l.insts);
    }

    /// A wave32 graphics program's host wave may have 64 lanes, whose upper
    /// half is a second console wave: a lane mask read as a number picks the
    /// reader's half of the ballot, and `v_mbcnt_lo` reports its rank as a
    /// host lane number.
    #[test]
    fn a_wave32_mask_read_takes_its_ballot_half() {
        let words = [
            0x7D8A_0480, // v_cmp_ne_u32 vcc, 0, v2
            0xBE84_0F6A, // s_bcnt1_i32_b32 s4, vcc_lo
            0xBF07_806A, // s_cmp_lg_u32 vcc_lo, 0
            0x7E08_0204, // v_mov_b32 v4, s4
            0xD765_0005,
            0x0001_006A, // v_mbcnt_lo_u32_b32 v5, vcc_lo, 0
            0xD766_0006,
            0x0002_0A6A, // v_mbcnt_hi_u32_b32 v6, vcc_lo, v5
            0xD765_0007,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v7, -1, 0
            0xBF85_0001, // s_cbranch_scc1 -> the export
            S_ENDPGM,    // no lane had its bit: done
            0xF800_180F,
            0x0706_0504, // exp mrt0, v4, v5, v6, v7 done vm
            S_ENDPGM,
        ];
        let lowered = lower_program(&words, HwStage::Ps, false, pixel_io()).unwrap();
        let f = |name| lowered.op_names.iter().position(|n| n == name);
        let (lane, ballot, bits) = (
            f("dx.op.waveGetLaneIndex").expect("no lane index for the pick"),
            f("dx.op.waveActiveBallot").expect("no ballot for the mask read"),
            f("dx.op.unaryBits.i32").expect("no bit count"),
        );
        let l = &lowered.insts;
        let lanes: Vec<usize> = l
            .iter()
            .enumerate()
            .filter(|(_, i)| **i == format!("call f{lane} [k111]"))
            .map(|(i, _)| i)
            .collect();
        assert!(!lanes.is_empty(), "no lane index for the pick");
        let i_ballot = l
            .iter()
            .position(|i| i.starts_with(&format!("call f{ballot} [k116, ")))
            .expect("no ballot");
        let (i_x, i_y) = (
            at(l, &format!("extract i{i_ballot} 0")),
            at(l, &format!("extract i{i_ballot} 1")),
        );
        // The pick: upper host lanes read the ballot's second dword.
        let i_cmp = l
            .iter()
            .position(|i| {
                i.starts_with("cmp:") && lanes.iter().any(|n| i.ends_with(&format!("i{n} k32")))
            })
            .expect("no `lane >= 32` compare");
        at(l, &format!("select i{i_cmp} i{i_y} i{i_x}"));
        // v_mbcnt_lo counts below the lane in its own half; a wave32 mask's
        // high half is empty, so v_mbcnt_hi's count is of no bits.
        assert!(
            lanes
                .iter()
                .any(|n| l.contains(&format!("bin:And i{n} k31"))),
            "no `lane & 31` in {l:?}"
        );
        at(l, &format!("call f{bits} [k31, k0]"));
        // The count is a lane number, so a second wave's rank takes its
        // 32-lane base: `v_mbcnt_lo(-1)` is the host lane index.
        let bases: Vec<usize> = l
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                s.starts_with("bin:And ")
                    && lanes.iter().any(|n| **s == format!("bin:And i{n} k32"))
            })
            .map(|(i, _)| i)
            .collect();
        assert!(!bases.is_empty(), "no `lane & 32` wave base in {l:?}");
        for base in &bases {
            assert!(
                l.iter().any(|s| {
                    s.starts_with("bin:Add ")
                        && s["bin:Add ".len()..]
                            .split(' ')
                            .any(|a| a == format!("i{base}"))
                }),
                "the wave base `i{base}` is never added in {l:?}"
            );
        }
    }

    /// A compute program declares the console's wave size always (the default), or
    /// with the wave rule by need, only where its lanes are observable.
    #[test]
    fn a_compute_wave_size_by_need() {
        let plain = [S_ENDPGM];
        let swizzle = [
            0xD8D4_801B,
            0x0400_0000, // ds_swizzle_b32 v4, v0 offset:swizzle(QDMODE, 3,2,1,0)
            S_ENDPGM,
        ];
        let wave = |words: &[u32], wave64: bool, need: bool| {
            crate::emit::WAVE_BY_NEED.with(|c| c.set(Some(need)));
            let l = lower_program(words, HwStage::Cs, wave64, compute_io()).unwrap();
            crate::emit::WAVE_BY_NEED.with(|c| c.set(None));
            crate::emit::container::declared_wave(&l.dxil)
        };
        assert_eq!(wave(&plain, false, false), Some(32));
        assert_eq!(wave(&plain, true, false), Some(64));
        assert_eq!(wave(&plain, false, true), None);
        assert_eq!(wave(&plain, true, true), None);
        assert_eq!(wave(&swizzle, false, true), Some(32));
        assert_eq!(wave(&swizzle, true, true), Some(64));
        // One console wave sharing LDS keeps it; more waves need not.
        let w = crate::emit::compute_wave_size;
        assert_eq!(w(32, false, true, 16, true), Some(32));
        assert_eq!(w(32, false, false, 16, true), None);
        assert_eq!(w(64, false, false, 0, false), Some(64));
    }

    /// A pixel program reads its quad (a DPP quad permutation, a quad-mode
    /// swizzle) through quad reads, which reach helper lanes; a helper
    /// lane's `v_readfirstlane_b32` takes its own value.
    #[test]
    fn a_pixel_quad_reads_reach_helper_lanes() {
        let words = [
            0x7E06_02FA,
            0xFF00_B102, // v_mov_b32 v3, v2 quad_perm:[1,0,3,2]
            0xD8D4_801B,
            0x0400_0002, // ds_swizzle_b32 v4, v2 offset:swizzle(QDMODE, 3,2,1,0)
            0x7E08_0502, // v_readfirstlane_b32 s4, v2
            0x7E0A_0204, // v_mov_b32 v5, s4
            0xF800_180F,
            0x0605_0403, // exp mrt0, v3, v4, v5, v6 done vm
            S_ENDPGM,
        ];
        let lowered = lower_program(&words, HwStage::Ps, false, pixel_io()).unwrap();
        let has = |name: &str| lowered.op_names.iter().any(|n| n == name);
        assert!(has("dx.op.quadReadLaneAt.i32"), "{:?}", lowered.op_names);
        assert!(!has("dx.op.waveReadLaneAt.i32"), "{:?}", lowered.op_names);
        assert!(has("dx.op.waveReadLaneFirst.i32"), "{:?}", lowered.op_names);
        let f = |name| lowered.op_names.iter().position(|n| n == name).unwrap();
        let (helper, first) = (f("dx.op.isHelperLane.i1"), f("dx.op.waveReadLaneFirst.i32"));
        let l = &lowered.insts;
        let i_helper = at(l, &format!("call f{helper} [k221]"));
        let i_first = l
            .iter()
            .position(|i| i.starts_with(&format!("call f{first} [k118, ")))
            .expect("no first-lane read");
        // The helper's own value (the read's operand) or the first lane's.
        let own = l[i_first].trim_end_matches(']').rsplit(", ").next().unwrap().to_owned();
        assert!(
            l.iter().any(|i| *i == format!("select i{i_helper} {own} i{i_first}")),
            "no helper select in {l:?}"
        );
    }

    /// The ballot-half pick is for wave32 graphics programs only: a wave64
    /// program's mask dwords are its own, and the console path (compute) is
    /// the whole host wave already, so neither reads the lane index for it.
    #[test]
    fn only_host_wave32_reads_take_a_ballot_half() {
        let words = [
            0x7D8A_0480, // v_cmp_ne_u32 vcc, 0, v2
            0xBE84_0F6A, // s_bcnt1_i32_b32 s4, vcc_lo
            0x7E08_0204, // v_mov_b32 v4, s4
            0xF800_180F,
            0x0404_0404, // exp mrt0, v4, v4, v4, v4 done vm
            S_ENDPGM,
        ];
        let l64 = lower_program(&words, HwStage::Ps, true, pixel_io()).unwrap();
        assert!(!l64.op_names.iter().any(|n| n == "dx.op.waveGetLaneIndex"));
        let words = [
            0x7D8A_0480, // v_cmp_ne_u32 vcc, 0, v0
            0xBE84_0F6A, // s_bcnt1_i32_b32 s4, vcc_lo
            0xBF07_8004, // s_cmp_lg_u32 s4, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE85_0380, // s_mov_b32 s5, 0
            S_ENDPGM,
        ];
        let cs = lower_program(&words, HwStage::Cs, false, compute_io()).unwrap();
        assert!(!cs.op_names.iter().any(|n| n == "dx.op.waveGetLaneIndex"));
    }

    /// XNW: class "an opaque lane mask" / "a lane-mask join of an
    /// untracked mask" (5c2592b0 PS_DeferredTextured at 0x4ba4, 6532f849,
    /// cb888a93): under SGPR pressure a pixel program spills its entry EXEC
    /// to VGPR lanes (`v_writelane_b32`) and reads it back
    /// (`v_readlane_b32`) to find helper lanes. The restored pair is the
    /// mask spilled, not an opaque number.
    #[test]
    fn a_mask_restored_from_its_spill_lanes_is_the_mask() {
        let words = [
            0xBE82_047E, // s_mov_b64 s[2:3], exec
            0xD761_0001,
            0x0001_0002, // v_writelane_b32 v1, s2, 0
            0xD761_0001,
            0x0001_0203, // v_writelane_b32 v1, s3, 1
            0xD760_0004,
            0x0001_0101, // v_readlane_b32 s4, v1, 0
            0xD760_0005,
            0x0001_0301, // v_readlane_b32 s5, v1, 1
            0x8A80_047E, // s_andn2_b64 s[0:1], exec, s[4:5]
            0xBF85_0004, // s_cbranch_scc1 -> s_endpgm (some helper lane)
            0xD501_0003,
            0x0001_0280, // v_cndmask_b32 v3, 0, 1, s[0:1]
            0xF800_180F,
            0x0303_0303, // exp mrt0, v3, v3, v3, v3 done vm
            S_ENDPGM,
        ];
        lower_program(&words, HwStage::Ps, true, pixel_io()).unwrap();
    }

    /// XNW: class "a constant buffer's descriptor read as data" (ped
    /// pixel programs, 957fc3f6 at 0x2ac8): a register holds a CBV
    /// descriptor's dword on one edge of a join and constant data on the
    /// other; only lanes that came by the data edge read it.
    #[test]
    fn a_cbv_descriptor_dword_on_a_dead_join_edge_is_dropped() {
        let words = [
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
            0xF420_0304,
            0xFA00_0000, // s_buffer_load_dword s12, s[8:11], 0x0
            0xBF06_800C, // s_cmp_eq_u32 s12, 0
            0xBF85_0002, // s_cbranch_scc1 -> the join (s9: descriptor dword)
            0xF420_0244,
            0xFA00_0004, // s_buffer_load_dword s9, s[8:11], 0x4
            0xBF06_800C, // s_cmp_eq_u32 s12, 0
            0xBF85_0005, // s_cbranch_scc1 -> s_endpgm
            0xBF06_8009, // s_cmp_eq_u32 s9, 0
            0xBF85_0003, // s_cbranch_scc1 -> s_endpgm
            0x7E00_0209, // v_mov_b32 v0, s9
            0xF800_180F,
            0x0000_0000, // exp mrt0, v0, v0, v0, v0 done vm
            S_ENDPGM,
        ];
        let param = |root, sgprs, user_data, slot| crate::RootParam { root, table_offset_dwords: 0, sgprs, base_register: 0, user_data, slot };
        let layout = vec![param(crate::RootType::Constants, 4, 0, 0), param(crate::RootType::Table, 1, 4, 1)];
        let used = vec![crate::UsedResource {
            kind: crate::ResourceType::Cbv,
            root: crate::RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }];
        lower_with(&words, HwStage::Ps, false, pixel_io(), layout, used).unwrap();
    }

    /// XNW: class "a hardware SGPR input" (8c2ff9cd, 28bf4456): a
    /// pixel program reads an SGPR above its user SGPRs (and the primitive
    /// mask) that no path writes first; it holds what the last wave left,
    /// read as zero as compute programs do.
    #[test]
    fn a_pixel_sgpr_above_the_user_data_reads_zero() {
        let words = [
            0xBF06_8028, // s_cmp_eq_u32 s40, 0
            0xBF85_0003, // s_cbranch_scc1 -> s_endpgm
            0x7E00_0228, // v_mov_b32 v0, s40
            0xF800_180F,
            0x0000_0000, // exp mrt0, v0, v0, v0, v0 done vm
            S_ENDPGM,
        ];
        let layout = vec![crate::RootParam {
            root: crate::RootType::Constants,
            table_offset_dwords: 0,
            sgprs: 4,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }];
        lower_with(&words, HwStage::Ps, false, pixel_io(), layout, vec![]).unwrap();
    }

    /// Class "a packed input half no element holds" (cbb83851 row 0, its
    /// `attr0.z` reading row 1). VIctory (35bf153, GPL-3.0+),
    /// verbatim; the last assertion is XNW.
    #[test]
    fn packed_pixel_inputs_find_their_second_row() {
        use super::super::container::SigElement;
        use super::half_input_partners;
        use crate::native::{PsInput, Usage};
        let entry = |row, row2, pack2| PsInput {
            usage: Usage::Generic,
            row,
            mask: 0xF,
            attr: row,
            flat: false,
            fp16: true,
            pack2,
            row2,
        };
        let element = |row| SigElement {
            name: "TEXCOORD".into(),
            index: row,
            kind: 0,
            comp: 9,
            interp: 2,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 0,
            half: true,
        };
        // Rows 0 and 3 pack and leave their second row out; 5 names its
        // own (6). Unnamed 16-bit rows: 1, 4, 6, 7.
        let io = StageIo::Ps {
            inputs: vec![entry(0, 0, true), entry(3, 3, true), entry(5, 6, true)],
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        };
        let sig: Vec<_> = [0, 1, 3, 4, 5, 6, 7].map(element).into();
        let p = half_input_partners(&io, &sig);
        assert_eq!(p.get(&0), Some(&1));
        assert_eq!(p.get(&3), Some(&4));
        assert_eq!(p.get(&5), None);
        // Halves 4, 5 (`attr0.z`, `attr0.z high`) are row 1's first two.
        assert_eq!(super::packed_half(&sig, 0, Some(1), 5), Some((1, 1)));
    }

    /// XNW: class "a 64-bit image atomic" (the alpha prepasses'
    /// `image_atomic_umax v[3:4]`, dmask 0x3): a typed 64-bit atomic on the
    /// R32G32_UINT image, value low dword first; the DXIL validates.
    #[test]
    fn a_two_dword_image_atomic_is_a_64_bit_atomic() {
        let words = [
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF40C_0202,
            0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0x0
            0xF05C_1308,
            0x0002_0002, // image_atomic_umax v[0:1], v[2:3], s[8:15] dmask:0x3 dim:2D unorm
            S_ENDPGM,
        ];
        let param = |root, sgprs, user_data, slot| crate::RootParam { root, table_offset_dwords: 0, sgprs, base_register: 0, user_data, slot };
        let layout = vec![param(crate::RootType::Constants, 4, 0, 0), param(crate::RootType::Table, 1, 4, 1)];
        let used = vec![crate::UsedResource {
            kind: crate::ResourceType::Uav,
            root: crate::RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 4,
            slot: 1,
        }];
        let l = lower_with(&words, HwStage::Ps, false, pixel_io(), layout, used).unwrap();
        assert!(l.op_names.iter().any(|n| n == "dx.op.atomicBinOp.i64"), "{:?}", l.op_names);
        if let Ok(mut v) = crate::emit::validator::Validator::load(crate::emit::validator::Validator::default_path().as_deref()) {
            v.validate(&mut l.dxil.clone()).unwrap();
        }
    }

    /// XNW: class "barycentrics read directly" (bloom c68f33cd, hair
    /// 7e040f4d): a program whose signature has no SV_Barycentrics reads a
    /// barycentric VGPR it reuses; the read is 0.
    #[test]
    fn an_undeclared_barycentric_read_is_zero() {
        let words = [
            0x7D8A_0080, // v_cmp_ne_u32 vcc, 0, v0 (perspective center i)
            0xBF87_0002, // s_cbranch_vccnz -> s_endpgm
            0xF800_180F,
            0x0303_0303, // exp mrt0, v3, v3, v3, v3 done vm
            S_ENDPGM,
        ];
        let io = StageIo::Ps {
            inputs: vec![],
            input_addr: 2,
            db_shader_control: 0,
            patches: vec![],
        };
        lower_program(&words, HwStage::Ps, false, io).unwrap();
    }

    /// XNW: nested loops share one budget per invocation: the inner
    /// loop's header takes the budget the outer loop brings (not a fresh 0),
    /// and each back edge compares the running total with the limit.
    #[test]
    fn nested_loops_share_one_budget() {
        let words = [
            0xBE80_0385, // s_mov_b32 s0, 5 (outer counter)
            0xBE81_0386, // s_mov_b32 s1, 6 (inner counter)
            0x8001_8101, // s_add_u32 s1, s1, 1
            0xBF07_FF01,
            0x0010_0000, // s_cmp_lg_u32 s1, 0x100000
            0xBF85_FFFC, // s_cbranch_scc1 L2
            0x8000_8100, // s_add_u32 s0, s0, 1
            0xBF07_FF00,
            0x0010_0000, // s_cmp_lg_u32 s0, 0x100000
            0xBF85_FFF8, // s_cbranch_scc1 L1
            S_ENDPGM,
        ];
        let l = lower_program(&words, HwStage::Cs, false, compute_io()).unwrap();
        // The budget's 0 is the first constant made; only the outer header
        // starts from it (per-loop counters started both at 0).
        let phis_from_zero = l.insts.iter().filter(|i| i.starts_with("Phi(") && i.contains("(Const(0),")).count();
        let limit = format!("k{}", super::super::DEFAULT_LOOP_LIMIT);
        let compares = l.insts.iter().filter(|i| i.starts_with("cmp:") && i.ends_with(&limit)).count();
        assert_eq!((phis_from_zero, compares), (1, 2), "{:?}", l.insts);
    }

    /// XNW: with loop counts on, each back edge reports the wave's
    /// count by one atomic max of its first lane into the program's slot of
    /// the buffer named by a user-data dword of root type `Other("LC")`; the
    /// DXIL validates.
    #[test]
    fn loop_counts_report_through_one_atomic_per_wave() {
        let words = [
            0xBE80_0385, // s_mov_b32 s0, 5
            0xBE81_0386, // s_mov_b32 s1, 6
            0x8001_8101, // s_add_u32 s1, s1, 1
            0xBF07_FF01,
            0x0010_0000, // s_cmp_lg_u32 s1, 0x100000
            0xBF85_FFFC, // s_cbranch_scc1 L2
            0x8000_8100, // s_add_u32 s0, s0, 1
            0xBF07_FF00,
            0x0010_0000, // s_cmp_lg_u32 s0, 0x100000
            0xBF85_FFF8, // s_cbranch_scc1 L1
            S_ENDPGM,
        ];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: compute_io(),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.code, false, 4).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let l = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            fallback: false,
            fma: crate::emit::Fma::Exact,
            gpu_desc: false,
            probe: false,
            loop_counts: Some(5),
            pagetable: false,
        })
        .unwrap();
        for op in ["dx.op.atomicBinOp.i32", "dx.op.waveIsFirstLane", "dx.op.waveActiveOp.i32"] {
            assert!(l.op_names.iter().any(|n| n == op), "{op} in {:?}", l.op_names);
        }
        let lc = l.user_data.iter().filter(|u| u.root == crate::RootType::Other(super::LOOP_COUNT_ROOT)).count();
        assert_eq!(lc, 1);
        if let Ok(mut v) = crate::emit::validator::Validator::load(crate::emit::validator::Validator::default_path().as_deref()) {
            v.validate(&mut l.dxil.clone()).unwrap();
        }
    }

    /// XNW: five-star-recomp 2781f9c's case (CS_Prop_BatchCull: a
    /// `v_cmp` into s[42:43] on one path, `s43 = s60 + 4` on another, then
    /// `s_add s32, s43, s32`): values are per write, so the integer read
    /// takes the integer, never the mask's high dword.
    #[test]
    fn an_sgpr_reused_after_a_mask_reads_its_own_value() {
        let words = [
            0xBF06_8000, // s_cmp_eq_u32 s0, 0
            0xBF85_0003, // s_cbranch_scc1 -> B
            0xD4C2_0002,
            0x0002_0080, // v_cmp_eq_u32 s[2:3], 0, v0
            S_ENDPGM,
            0x8103_8401, // B: s_add_i32 s3, s1, 4
            0x8104_0103, // s_add_i32 s4, s3, s1
            S_ENDPGM,
        ];
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = ir::build(&code, true, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let adds: Vec<&ir::Inst> = p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == "s_add_i32").collect();
        let int = adds[0].write(ir::Reg::S(3)).unwrap();
        assert_eq!(adds[1].read(ir::Reg::S(3)), Some(int));
        assert!(!st.highs.contains_key(&int) && !st.masks.contains_key(&int));
        lower_program(&words, HwStage::Cs, true, compute_io()).unwrap();
    }

    fn pixel_io() -> StageIo {
        StageIo::Ps {
            inputs: vec![],
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        }
    }

    fn compute_io() -> StageIo {
        StageIo::Cs(ComputeIo {
            rsrc2: 0,
            threads: [1, 1, 1],
        })
    }

    /// The index of the instruction `pat` names in `lines`.
    fn at(lines: &[String], pat: &str) -> usize {
        lines
            .iter()
            .position(|l| l == pat)
            .unwrap_or_else(|| panic!("no `{pat}` in {lines:?}"))
    }

    #[test]
    fn scalar_bit_tests_lower() {
        // Each tests one bit of a constant into SCC and branches on it (only
        // code with an effect needs its operands).
        let tail = [
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let bit_tests: [&[u32]; 4] = [
            &[0xBE81_0387, 0xBF0C_8301], // s_mov_b32 s1, 7; s_bitcmp0_b32 s1, 3
            &[0xBE81_0387, 0xBF0D_8301], // s_mov_b32 s1, 7; s_bitcmp1_b32 s1, 3
            &[
                0xBE80_0387, // s_mov_b32 s0, 7
                0xBE81_0380, // s_mov_b32 s1, 0
                0xBF0E_8300, // s_bitcmp0_b64 s[0:1], 3
            ],
            &[
                0xBE80_0387, // s_mov_b32 s0, 7
                0xBE81_0380, // s_mov_b32 s1, 0
                0xBF0F_8300, // s_bitcmp1_b64 s[0:1], 3
            ],
        ];
        for head in bit_tests {
            let words: Vec<u32> = head.iter().chain(&tail).copied().collect();
            lowers(&words, 0).unwrap();
        }
    }


    fn perm_features(selector: u32) -> u64 {
        // Synthetic compute shader:
        // v0 is the thread ID and can supply a dynamic selector.
        // Store the result in LDS to keep the permutation live.
        let words = [
            0x7E02_02FFu32,
            0x817F_34AB,
            0x7E04_02FF,
            0xFE01_56CD,
            0xD744_0003,
            257 | (258 << 9) | (selector << 18),
            0x3408_0082,
            0xD834_0000,
            0x0000_0304,
            S_ENDPGM,
        ];

        let code: Vec<u8> = words.iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();

        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15,
                threads: [64, 1, 1],
            }),
            code,
        };

        let p = ir::build(&native.code, false, 256).unwrap();

        assert!(p.blocks.iter()
            .flat_map(|b| &b.insts)
            .any(|i| i.name() == "v_perm_b32"));

        let res = resource::track(
            &p, &native.layout, native.stage
        ).unwrap();

        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();

        let emitted = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            fallback: false,
            fma: crate::emit::Fma::Exact,
            gpu_desc: false,
            probe: false,
            loop_counts: None,
            pagetable: false,
        }).unwrap();

        let container =
            crate::Container::parse(&emitted.dxil).unwrap();

        let features =
            container.part(crate::FourCC::SFI0).unwrap();

        u64::from_le_bytes(features.data.try_into().unwrap())
    }

    #[test]
    fn perm_constant_selector_needs_no_int64() {
        assert_eq!(perm_features(128) & 0x8000, 0);
    }

    #[test]
    fn perm_dynamic_selector_needs_no_int64() {
        assert_eq!(
            perm_features(256) & 0x8000,
            0,
            "dynamic v_perm_b32 must not require INT64"
        );
    }

    #[test]
    fn buffer_formats_by_number() {
        // The numbers the disassembler names 8_8_8_8_UNORM, 16_16_SNORM,
        // 2_10_10_10_UNORM, 16_FLOAT, 32_32_32_32_FLOAT.
        assert_eq!(buffer_format(56), Some((&[8, 8, 8, 8][..], Numeric::Unorm)));
        assert_eq!(buffer_format(24), Some((&[16, 16][..], Numeric::Snorm)));
        assert_eq!(
            buffer_format(50),
            Some((&[10, 10, 10, 2][..], Numeric::Unorm))
        );
        assert_eq!(buffer_format(13), Some((&[16][..], Numeric::Float)));
        assert_eq!(
            buffer_format(77),
            Some((&[32, 32, 32, 32][..], Numeric::Float))
        );
        // Invalid, past the table.
        assert_eq!(buffer_format(0), None);
        assert_eq!(buffer_format(78), None);
    }

    #[test]
    fn packed_buffer_formats_by_number() {
        // LLVM's `UfmtGFX10`: 10_11_11 is 30-36, 11_11_10 37-43,
        // 10_10_10_2 44-49. Channels from the lowest bits: R11G11B10_FLOAT
        // is 10_11_11 (Mesa), 10_10_10_2 has the 2-bit channel first.
        assert_eq!(buffer_format(30), Some((&[11, 11, 10][..], Numeric::Unorm)));
        assert_eq!(buffer_format(36), Some((&[11, 11, 10][..], Numeric::Float)));
        assert_eq!(buffer_format(37), Some((&[10, 11, 11][..], Numeric::Unorm)));
        assert_eq!(buffer_format(43), Some((&[10, 11, 11][..], Numeric::Float)));
        assert_eq!(
            buffer_format(44),
            Some((&[2, 10, 10, 10][..], Numeric::Unorm))
        );
        assert_eq!(
            buffer_format(45),
            Some((&[2, 10, 10, 10][..], Numeric::Snorm))
        );
        assert_eq!(
            buffer_format(46),
            Some((&[2, 10, 10, 10][..], Numeric::Uscaled))
        );
        assert_eq!(
            buffer_format(47),
            Some((&[2, 10, 10, 10][..], Numeric::Sscaled))
        );
        assert_eq!(
            buffer_format(48),
            Some((&[2, 10, 10, 10][..], Numeric::Uint))
        );
        assert_eq!(
            buffer_format(49),
            Some((&[2, 10, 10, 10][..], Numeric::Sint))
        );
    }

    /// A half's value (sign, 5-bit exponent biased 15, 10-bit mantissa), as
    /// `legacyF16ToF32` widens it.
    fn half(h: u32) -> f32 {
        let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 31, h & 1023);
        let v = match e {
            0 => m as f32 * 2f32.powi(-24),
            31 if m == 0 => f32::INFINITY,
            31 => f32::NAN,
            _ => (1024 + m) as f32 * 2f32.powi(e as i32 - 25),
        };
        if s == 1 { -v } else { v }
    }

    #[test]
    fn small_floats_widen_as_halves() {
        // Unsigned 11- and 10-bit floats: a 5-bit exponent biased 15 over a
        // 6- or 5-bit mantissa, no sign; values from the D3D/GL spec.
        let f11 = |x: u32| half(x << half_shift(11));
        let f10 = |x: u32| half(x << half_shift(10));
        assert_eq!(f11(0), 0.0);
        assert_eq!(f11(0x001), 2f32.powi(-20)); // smallest denormal
        assert_eq!(f11(0x03F), 63.0 * 2f32.powi(-20)); // largest denormal
        assert_eq!(f11(0x040), 2f32.powi(-14)); // smallest normal
        assert_eq!(f11(0x3C0), 1.0);
        assert_eq!(f11(0x3E0), 1.5);
        assert_eq!(f11(0x7BF), 65024.0); // largest
        assert_eq!(f11(0x7C0), f32::INFINITY);
        assert!(f11(0x7C1).is_nan());
        assert_eq!(f10(0x001), 2f32.powi(-19));
        assert_eq!(f10(0x020), 2f32.powi(-14));
        assert_eq!(f10(0x1E0), 1.0);
        assert_eq!(f10(0x3DF), 64512.0);
        assert_eq!(f10(0x3E0), f32::INFINITY);
        assert!(f10(0x3FF).is_nan());
        // A 16-bit float is a half already, sign included.
        assert_eq!(half(0xBC00 << half_shift(16)), -1.0);
    }
}
