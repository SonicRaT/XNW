// Ported from VIctory crates/victory-shader/src/emit/lower.rs, commit d2a4bae1f49533ef5cb512fd1b1ba1513f2e1989, GPL-3.0-or-later.
// With VIctory open pull requests (GPL-3.0-or-later) where they touch this file: (8ae232f, d112889), (4c9a922, db20637, 3e8f2d3), (815178d), (fa5189d).
// With VIctory (head 42538cd, on upstream 4322b61; GPL-3.0-or-later): unknown-stride structured buffers read through the runtime stride table (shader side).
// With VIctory (499bd98, GPL-3.0-or-later): multi-term, scaled, masked and phi-joined addresses in resource tracking.
// With VIctory (afbe339, f800f1c; GPL-3.0-or-later): untracked masks in selects and carry-ins via lane bits; descriptor-only GetDimensions typed by metadata.
// With VIctory (c78056c, 5ff639e; GPL-3.0-or-later): quad gradients for compute samples, twin descriptor loads, descriptor dwords at joins.
// With VIctory (35bf153, GPL-3.0-or-later): second rows of packed 16-bit pixel inputs.
// With VIctory (permadiego/VIctory shader/mask-dword-joins 7ab94ab) and its commit 1661a3e, GPL-3.0-or-later: mask dwords at data joins; compare-and-swap image atomics.
// With VIctory-Belobug-AMD (fork of VIctory at 8e79c80; read at 4cbe9d7), GPL-3.0-or-later: pixel quads read through quad reads, helper lanes' v_readfirstlane (03629f3, f520772; upstream 0601955).
// With DexNom-VIctory (ddb7332), GPL-3.0-or-later: a uniform branch choosing between two immediate samplers.
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

// XNW (BLD2): the methods of `Lower`, one module per stage, so an edit re-generates one stage's code.
mod setup;
mod values;
mod inst;
mod smem;
mod address;
mod valu;
mod valu16;
mod integer;
mod image;
mod buffer;
mod exit;

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

    /// DexNom-VIctory (ddb7332, GPL-3.0-or-later) test; XNW: our harness
    /// (compute stage, 512 bytes of LDS, user SGPRs s0-s1).
    #[test]
    fn a_uniform_branch_can_choose_between_immediate_samplers() {
        // A root constant selects wrap or clamp. Keep the sample live with an LDS store.
        let words = [
            0xBE84_0300, 0xBE85_0380, // s[4:5] = texture table
            0xF40C_0302, 0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0
            0xBE89_0380, 0xBE8A_0380, 0xBE8B_0380, // sampler words 1-3
            0xBF06_8001, // s_cmp_eq_u32 s1, 0
            0xBF85_0002, // branch to clamp
            0xBE88_0380, 0xBF82_0002, // wrap, skip clamp
            0xBE88_03FF, 0x0000_0092, // clamp all axes
            0xF09C_0F08, 0x0043_0C02, // image_sample_lz v[12:15], ..., s[8:11]
            0xD834_0000, 0x0000_0C00, // ds_write_b32 v0, v12
            S_ENDPGM,
        ];
        let param = |root, user_data, slot| crate::RootParam { root, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data, slot };
        let layout = vec![param(crate::RootType::Table, 0, 0), param(crate::RootType::Constants, 1, 1)];
        let used = vec![crate::UsedResource {
            kind: crate::ResourceType::Srv,
            root: crate::RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        }];
        let io = StageIo::Cs(ComputeIo {
            rsrc2: 1 << 15 | 2 << 1,
            threads: [1, 1, 1],
        });
        let l = lower_with(&words, HwStage::Cs, false, io, layout, used).unwrap();
        assert_eq!(l.samplers.len(), 2);
        assert!(l.samplers.contains(&[0, 0, 0, 0]));
        assert!(l.samplers.contains(&[0x92, 0, 0, 0]));
        if let Ok(mut v) = crate::emit::validator::Validator::load(crate::emit::validator::Validator::default_path().as_deref()) {
            v.validate(&mut l.dxil.clone()).unwrap();
        }
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
