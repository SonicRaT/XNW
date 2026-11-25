//! Pipelines from the title's pipeline library: device slots 57 (graphics
//! record), 67 (compute record) and 55 (a pipeline derived from one with
//! state overrides), decoded into one plain [`Description`] whether it comes
//! from the title's memory or from the pack's library offline
//! (`xnw pipeline-walk`), and handed to the pipeline layer as a D3D12
//! subobject stream ([`Stream`]) that the base's `GraphicsState::read`
//! takes as it is.
//!
//! Sources:
//! - five-star-recomp (GPL-3.0-or-later, commit 0bc9cd731e363041637504dac75e22f53ce34c58):
//!    11-17, 184-243;
//!   60-139; ported from `src/FiveStar.Shaders/Cache/GraphicsStateParser.cs`,
//!   `FixedFunctionDecoder.cs`, `PipelineTargetsParser.cs`,
//!   `InputLayoutParser.cs`, `ComputeStateParser.cs` and
//!   `src/FiveStar.Loader/Shims/D3D12X/Pipelines/CachedStateMapping.cs`,
//!   `PipelineLayout.cs`, `TopologyTypes.cs` (desktop rules of
//!   `DesktopStateRules.cs` as described in d3d12x-pipelines.md 242).
//! - Xbox2Windows (MIT, f464f34f): `src/startup_stubs/cb_blend.h` (the same
//!   CB_BLENDn_CONTROL fields, checked against Mesa gfx10.json there).
//! - The title: the 0x108-byte slot-57
//!   description is built at 0x14248f680 from a 0x70-byte record (ranges by
//!   0x14248feb0/0x14248fee0); the 0x48-byte slot-67 one in the same
//!   function; slot 55 is called at 0x14247a5c2 as `(base pipeline, count,
//!   entries of 0x14c bytes, iid, out)` with kinds 1, 0 (then 5), 2.
//!
//! Register fields are AMD's (RDNA2: DB_DEPTH_CONTROL, CB_BLENDn_CONTROL,
//! PA_SU_SC_MODE_CNTL, ...); D3D12 enum values are kept as plain numbers so
//! the decoder runs without a device.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_SAMPLE_DESC};

use super::object::{HRESULT, Iid, This};

// ---- The plain description ----

/// Where a stage's code is: in the title's memory (slots 57/67 pass the
/// loaded library's code block and its shader metadata), or in the pack's
/// library by id (offline).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Code {
    Title { code: usize, size: usize, meta: usize, meta_size: usize },
    Library { shader: u32, meta: u32 },
}

/// One stage: its library slot's name (`pixel`, `vertex`, `mesh`, `fetch`,
/// `compute`) and its code. A cache code block is found in the pack by its
/// first 24 bytes (five-star pipeline-cache.md 113-118).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageRef {
    pub stage: &'static str,
    pub code: Code,
}

/// One render target's blend, in D3D12 values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blend {
    pub enable: bool,
    pub logic_enable: bool,
    pub src: i32,
    pub dest: i32,
    pub op: i32,
    pub src_alpha: i32,
    pub dest_alpha: i32,
    pub op_alpha: i32,
    pub logic_op: i32,
    pub mask: u8,
}

pub const BLEND_ZERO: i32 = 1;
pub const BLEND_ONE: i32 = 2;
pub const BLEND_OP_ADD: i32 = 1;
const LOGIC_OP_NOOP: i32 = 4;

impl Default for Blend {
    fn default() -> Blend {
        Blend {
            enable: false,
            logic_enable: false,
            src: BLEND_ONE,
            dest: BLEND_ZERO,
            op: BLEND_OP_ADD,
            src_alpha: BLEND_ONE,
            dest_alpha: BLEND_ZERO,
            op_alpha: BLEND_OP_ADD,
            logic_op: LOGIC_OP_NOOP,
            mask: 15,
        }
    }
}

/// A stencil face, in D3D12 values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Face {
    pub fail: i32,
    pub depth_fail: i32,
    pub pass: i32,
    pub func: i32,
}

const STENCIL_KEEP: i32 = 1;
const COMPARE_LESS: i32 = 2;
const COMPARE_ALWAYS: i32 = 8;

impl Default for Face {
    fn default() -> Face {
        Face { fail: STENCIL_KEEP, depth_fail: STENCIL_KEEP, pass: STENCIL_KEEP, func: COMPARE_ALWAYS }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthStencil {
    pub depth: bool,
    pub write: bool,
    pub func: i32,
    /// Depth bounds test (DB_DEPTH_CONTROL bit 3); the base's state keeps
    /// the plain description, so this is reported, not built.
    pub bounds: bool,
    pub stencil: bool,
    pub read_mask: u8,
    pub write_mask: u8,
    pub front: Face,
    pub back: Face,
}

impl Default for DepthStencil {
    fn default() -> DepthStencil {
        DepthStencil { depth: true, write: true, func: COMPARE_LESS, bounds: false, stencil: false, read_mask: 0xff, write_mask: 0xff, front: Face::default(), back: Face::default() }
    }
}

pub const FILL_WIREFRAME: i32 = 2;
pub const FILL_SOLID: i32 = 3;
pub const CULL_NONE: i32 = 1;
pub const CULL_FRONT: i32 = 2;
pub const CULL_BACK: i32 = 3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rasterizer {
    pub fill: i32,
    pub cull: i32,
    pub front_ccw: bool,
    pub bias: i32,
    pub bias_clamp: f32,
    pub slope: f32,
    pub depth_clip: bool,
    pub multisample: bool,
    pub line_aa: bool,
    pub forced_samples: u32,
    pub conservative: i32,
}

impl Default for Rasterizer {
    fn default() -> Rasterizer {
        Rasterizer { fill: FILL_SOLID, cull: CULL_BACK, front_ccw: false, bias: 0, bias_clamp: 0.0, slope: 0.0, depth_clip: true, multisample: false, line_aa: false, forced_samples: 0, conservative: 0 }
    }
}

/// One vertex input element (the order of `D3D12_INPUT_ELEMENT_DESC`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Input {
    pub name: String,
    pub index: u32,
    pub format: u32,
    pub slot: u32,
    pub offset: u32,
    pub class: u32,
    pub step: u32,
}

/// The graphics record's outputs-and-inputs entry (metadata 0x0C): the
/// formats the pipeline is created with (console DXGI values).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Outputs {
    pub targets: Vec<u32>,
    pub depth: u32,
    /// Low 4 bits of byte 0x26 (D3D12 topology type); `topology_bit4` the
    /// unexplained bit 4.
    pub topology: u32,
    pub topology_bit4: bool,
    pub samples: u32,
    pub sample_mask: u32,
    pub inputs: Vec<Input>,
}

/// The context registers of a graphics state entry, as stored.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Registers {
    pub version: u32,
    pub mesh: bool,
    /// Dwords in the tail after the fixed part (low byte of 0x0C).
    pub tail: u32,
    pub gs_out_prim_type: u32,
    pub rsrc_vs: [u32; 2],
    pub rsrc_ps: [u32; 2],
    pub rsrc_gs: [u32; 2],
    pub col_format: u32,
    pub shader_mask: u32,
    pub clip: u32,
    pub mode: u32,
    pub poly_clamp: f32,
    pub poly_scale: f32,
    pub poly_offset: f32,
    pub depth: u32,
    pub stencil: u32,
    pub stencil_ref: u32,
    pub stencil_ref_back: u32,
    pub target_mask: u32,
    pub blend: [u32; 8],
    pub downconvert: u32,
    pub blend_opt_epsilon: u32,
    pub blend_opt_control: u32,
    pub mrt_blend_opt: [u32; 8],
    pub eqaa: u32,
}

/// A compute record's 28-byte state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ComputeState {
    pub threads: [u16; 3],
    pub wave32: bool,
    pub rsrc2: u32,
}

/// Fixed-function state, in D3D12 values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fixed {
    pub alpha_to_coverage: bool,
    pub independent: bool,
    pub dual_source: bool,
    pub blend: [Blend; 8],
    pub depth: DepthStencil,
    pub rasterizer: Rasterizer,
    /// From VGT_GS_OUT_PRIM_TYPE (1 point, 2 line, 3 triangle).
    pub topology: u32,
    pub sample_mask: u32,
}

impl Default for Fixed {
    fn default() -> Fixed {
        Fixed {
            alpha_to_coverage: false,
            independent: false,
            dual_source: false,
            blend: [Blend::default(); 8],
            depth: DepthStencil::default(),
            rasterizer: Rasterizer::default(),
            topology: 0,
            sample_mask: u32::MAX,
        }
    }
}

/// What a slot-55/57/67 pipeline (or a library record) is made of.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Description {
    /// 55, 57 or 67; 0 for a library record read offline.
    pub slot: u8,
    pub stages: Vec<StageRef>,
    /// The title's root signature object (slots 57/67 description +0).
    pub root: usize,
    pub registers: Option<Registers>,
    pub fixed: Fixed,
    pub outputs: Option<Outputs>,
    pub compute: Option<ComputeState>,
    /// Slot 55: the override kinds applied, in order.
    pub overrides: Vec<u32>,
    /// What could not be read or expressed (empty for a whole decode).
    pub problems: Vec<&'static str>,
}

impl Description {
    pub fn is_compute(&self) -> bool {
        self.stages.iter().any(|s| s.stage == "compute")
    }
    /// The stage combination, slot names joined by `+`.
    pub fn combination(&self) -> String {
        self.stages.iter().map(|s| s.stage).collect::<Vec<_>>().join("+")
    }
    /// The render-target formats the pipeline is created with: the
    /// outputs entry's (the registers can only guess a family).
    pub fn targets(&self) -> &[u32] {
        self.outputs.as_ref().map_or(&[], |o| &o.targets)
    }
    pub fn depth_format(&self) -> u32 {
        self.outputs.as_ref().map_or(0, |o| o.depth)
    }
    /// The D3D12 topology type: the outputs entry's, else the registers';
    /// mesh pipelines, 0, the console's 19 and anything else are triangles
    /// (five-star TopologyTypes.cs).
    pub fn topology(&self) -> u32 {
        let t = self.outputs.as_ref().map_or(self.fixed.topology, |o| o.topology);
        if self.stages.iter().any(|s| s.stage == "mesh") || !(1..=4).contains(&t) { 3 } else { t }
    }
    pub fn samples(&self) -> u32 {
        self.outputs.as_ref().map_or(1, |o| o.samples.max(1))
    }
}

// ---- Reading the state blocks ----

fn u32_at(b: &[u8], at: usize) -> u32 {
    b.get(at..at + 4).map_or(0, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}
fn f32_at(b: &[u8], at: usize) -> f32 {
    f32::from_bits(u32_at(b, at))
}

pub const STATE_VERSIONS: [u32; 3] = [0x4005_D800, 0x4005_D808, 0x4005_D810];
const VERTEX_KIND: u32 = 4;
const MESH_KIND: u32 = 6;
/// Bytes of the fixed part before the tail (vertex, mesh layout).
pub const VERTEX_FIXED: usize = 380;
pub const MESH_FIXED: usize = 464;
const MESH_SHIFT: usize = 0x54;

/// The registers of a graphics state entry (record metadata 0x08).
/// GraphicsStateParser.cs; the RSRC words and SX_MRTn_BLEND_OPT added from
/// pipeline-cache.md 113-139.
pub fn registers(state: &[u8]) -> Result<Registers, &'static str> {
    if state.len() < VERTEX_FIXED {
        return Err("state entry shorter than the vertex layout");
    }
    let version = u32_at(state, 0);
    if !STATE_VERSIONS.contains(&version) {
        return Err("state entry of an unknown version");
    }
    let mesh = match u32_at(state, 4) {
        VERTEX_KIND => false,
        MESH_KIND => true,
        _ => return Err("state entry of an unknown layout kind"),
    };
    if mesh && state.len() < MESH_FIXED {
        return Err("mesh state entry shorter than its layout");
    }
    let s = if mesh { MESH_SHIFT } else { 0 };
    let r = |at: usize| u32_at(state, at + s);
    let mut blend = [0; 8];
    let mut opt = [0; 8];
    for i in 0..8 {
        blend[i] = r(0xd0 + 4 * i);
        opt[i] = r(0x100 + 4 * i);
    }
    Ok(Registers {
        version,
        mesh,
        tail: u32_at(state, 0x0c) & 0xff,
        gs_out_prim_type: u32_at(state, 0x2c),
        rsrc_vs: if mesh { [0, 0] } else { [u32_at(state, 0x38), u32_at(state, 0x3c)] },
        rsrc_ps: [r(0x4c), r(0x50)],
        rsrc_gs: if mesh { [u32_at(state, 0x60), u32_at(state, 0x64)] } else { [0, 0] },
        col_format: r(0x70),
        shader_mask: r(0x78),
        clip: r(0x9c),
        mode: r(0xa0),
        poly_clamp: f32_at(state, 0xac + s),
        poly_scale: f32_at(state, 0xb0 + s),
        poly_offset: f32_at(state, 0xb4 + s),
        depth: r(0xb8),
        stencil: r(0xbc),
        stencil_ref: r(0xc0),
        stencil_ref_back: r(0xc4),
        target_mask: r(0xcc),
        blend,
        downconvert: r(0xf4),
        blend_opt_epsilon: r(0xf8),
        blend_opt_control: r(0xfc),
        mrt_blend_opt: opt,
        eqaa: r(0x120),
    })
}

fn field(v: u32, shift: u32, mask: u32) -> u32 {
    (v >> shift) & mask
}
fn bit(v: u32, b: u32) -> bool {
    (v >> b) & 1 != 0
}

/// gfx10 BlendOp -> D3D12_BLEND (0: none; 11/12 BOTH_(INV_)SRC_ALPHA).
const FACTORS: [i32; 21] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15, 16, 17, 18, 19, 20, 21];
/// CombFunc -> D3D12_BLEND_OP.
const COMBINE: [i32; 5] = [1, 2, 4, 5, 3];
/// gfx10 StencilOp -> D3D12_STENCIL_OP (ONES and REPLACE_OP have none).
const STENCIL_OPS: [i32; 10] = [1, 2, 0, 3, 0, 4, 5, 6, 7, 8];

fn lookup(table: &[i32], i: u32) -> i32 {
    table.get(i as usize).copied().unwrap_or(0)
}

/// On the alpha channel a colour factor reads its alpha twin; D3D12 only
/// takes the alpha forms there.
fn alpha_twin(f: i32) -> i32 {
    match f {
        3 => 5,
        4 => 6,
        9 => 7,
        10 => 8,
        16 => 18,
        17 => 19,
        other => other,
    }
}

/// True if a CB_BLENDn_CONTROL uses a SRC1 factor (gfx10 15-18).
fn second_source(c: u32) -> bool {
    bit(c, 30) && [0, 8, 16, 24].iter().any(|&s| (15..=18).contains(&field(c, s, 31)))
}

fn or(v: i32, fallback: i32) -> i32 {
    if v == 0 { fallback } else { v }
}

/// The D3D12 state of the registers (FixedFunctionDecoder.cs, then the
/// desktop mapping of CachedStateMapping.cs and DesktopStateRules). Forms
/// D3D12 cannot express are pushed onto `problems`.
pub fn decode(r: &Registers, problems: &mut Vec<&'static str>) -> Fixed {
    let mut f = Fixed { independent: true, ..Fixed::default() };
    for (i, t) in f.blend.iter_mut().enumerate() {
        let c = r.blend[i];
        t.mask = field(r.target_mask, 4 * i as u32, 15) as u8;
        t.enable = bit(c, 30);
        if !t.enable {
            continue;
        }
        let (cs, cd, cc) = (field(c, 0, 31), field(c, 8, 31), field(c, 5, 7));
        let (asrc, ad, ac) = if bit(c, 29) { (field(c, 16, 31), field(c, 24, 31), field(c, 21, 7)) } else { (cs, cd, cc) };
        let factors = [lookup(&FACTORS, cs), lookup(&FACTORS, cd), lookup(&FACTORS, asrc), lookup(&FACTORS, ad)];
        if factors.contains(&0) {
            problems.push("blend factor without a D3D12 form (BOTH_SRC_ALPHA or unknown)");
        }
        let (op, op_alpha) = (lookup(&COMBINE, cc), lookup(&COMBINE, ac));
        if op == 0 || op_alpha == 0 {
            problems.push("unknown blend combine function");
        }
        t.src = or(factors[0], BLEND_ONE);
        t.dest = or(factors[1], BLEND_ZERO);
        t.src_alpha = alpha_twin(or(factors[2], BLEND_ONE));
        t.dest_alpha = alpha_twin(or(factors[3], BLEND_ZERO));
        t.op = or(op, BLEND_OP_ADD);
        t.op_alpha = or(op_alpha, BLEND_OP_ADD);
    }
    f.dual_source = second_source(r.blend[0]);
    if f.dual_source {
        // D3D12 takes dual-source blending on target 0 only.
        f.independent = false;
        for t in &mut f.blend[1..] {
            *t = Blend::default();
        }
    }

    let (dc, sc) = (r.depth, r.stencil);
    let face = |ops: u32, func: u32| Face {
        fail: or(lookup(&STENCIL_OPS, field(ops, 0, 15)), STENCIL_KEEP),
        pass: or(lookup(&STENCIL_OPS, field(ops, 4, 15)), STENCIL_KEEP),
        depth_fail: or(lookup(&STENCIL_OPS, field(ops, 8, 15)), STENCIL_KEEP),
        func: func as i32 + 1,
    };
    for shift in [0, 4, 8, 12, 16, 20] {
        if matches!(field(sc, shift, 15), 2 | 4) && bit(dc, 0) {
            problems.push("stencil op ONES or REPLACE_OP");
        }
    }
    let front = face(sc, field(dc, 8, 7));
    f.depth = DepthStencil {
        depth: bit(dc, 1),
        write: bit(dc, 2),
        func: field(dc, 4, 7) as i32 + 1,
        bounds: bit(dc, 3),
        stencil: bit(dc, 0),
        read_mask: field(r.stencil_ref, 8, 0xff) as u8,
        write_mask: field(r.stencil_ref, 16, 0xff) as u8,
        front,
        back: if bit(dc, 7) { face(sc >> 12, field(dc, 20, 7)) } else { front },
    };

    let m = r.mode;
    let fill = if field(m, 3, 3) != 0 { if field(m, 5, 7) == 1 { FILL_WIREFRAME } else { FILL_SOLID } } else { FILL_SOLID };
    let cull = match (bit(m, 0), bit(m, 1)) {
        (true, true) => {
            problems.push("both faces culled");
            CULL_NONE
        }
        (true, false) => CULL_FRONT,
        (false, true) => CULL_BACK,
        (false, false) => CULL_NONE,
    };
    // Bias applies when POLY_OFFSET_FRONT_ENABLE (bit 11) is set.
    let biased = bit(m, 11);
    let clean = |v: f32| if v.is_nan() { 0.0 } else { v };
    let forced = 1u32 << field(r.eqaa, 8, 7);
    let forced = if forced == 1 { 0 } else { forced };
    f.rasterizer = Rasterizer {
        fill,
        cull,
        front_ccw: !bit(m, 2),
        bias: if biased { clean(r.poly_offset) as i32 } else { 0 },
        bias_clamp: if biased { clean(r.poly_clamp) } else { 0.0 },
        slope: if biased { clean(r.poly_scale) / 16.0 } else { 0.0 },
        depth_clip: r.clip & 0x0C00_0000 == 0,
        // D3D12 refuses a forced sample count with depth or stencil on.
        forced_samples: if f.depth.depth || f.depth.stencil { 0 } else { forced },
        ..Rasterizer::default()
    };
    f.topology = match r.gs_out_prim_type & 0x3f {
        0 => 1,
        1 => 2,
        2 => 3,
        _ => 0,
    };
    f
}

/// The outputs-and-inputs entry (record metadata 0x0C).
/// PipelineTargetsParser.cs, InputLayoutParser.cs.
pub fn outputs(meta: &[u8]) -> Result<Outputs, &'static str> {
    if meta.len() < 0x50 {
        return Err("outputs entry shorter than its header");
    }
    let count = meta[0x25] as usize;
    if count > 8 {
        return Err("render-target count above 8");
    }
    let n = u32_at(meta, 0x4c) as usize;
    if n > 64 {
        return Err("input element count out of range");
    }
    let mut inputs = Vec::with_capacity(n);
    let mut at = 0x50;
    for _ in 0..n {
        let len = u32_at(meta, at) as usize;
        let fields = at + 4 + len;
        if len == 0 || fields + 24 > meta.len() {
            return Err("input element runs past its entry");
        }
        let name = String::from_utf8_lossy(&meta[at + 4..at + 3 + len]).into_owned();
        let v = |i: usize| u32_at(meta, fields + 4 * i);
        inputs.push(Input { name, index: v(0), format: v(1), slot: v(2), offset: v(3), class: v(4), step: v(5) });
        at = fields + 24;
    }
    Ok(Outputs {
        targets: meta[0x10..0x10 + count].iter().map(|&f| u32::from(f)).collect(),
        depth: u32::from(meta[0x24]),
        topology: u32::from(meta[0x26] & 15),
        topology_bit4: meta[0x26] & 16 != 0,
        samples: u32::from(meta[0x28]),
        sample_mask: u32_at(meta, 0x1c),
        inputs,
    })
}

/// A compute record's state (ComputeStateParser.cs; pipeline-cache.md 86-101).
pub fn compute_state(state: &[u8]) -> Result<ComputeState, &'static str> {
    if state.len() < 0x1c {
        return Err("compute state shorter than 28 bytes");
    }
    if u32_at(state, 0) != 0x4005_CF00 {
        return Err("compute state of an unknown version");
    }
    let h = |at: usize| u16::from_le_bytes([state[at], state[at + 1]]);
    Ok(ComputeState { threads: [h(0x0c), h(0x0e), h(0x10)], wave32: h(0x12) & 16 != 0, rsrc2: u32_at(state, 0x18) })
}

/// The graphics description from a record's state and outputs entries
/// and its stages (what slot 57 reads from the title, and the walk from
/// the pack).
pub fn graphics(state: &[u8], outs: &[u8], stages: Vec<StageRef>) -> Description {
    let mut d = Description { slot: 57, stages, ..Description::default() };
    match registers(state) {
        Ok(r) => {
            let expected = if r.mesh { MESH_FIXED } else { VERTEX_FIXED } + 4 * r.tail as usize;
            if state.len() != expected {
                d.problems.push("state entry size is not fixed part + tail");
            }
            d.fixed = decode(&r, &mut d.problems);
            d.registers = Some(r);
        }
        Err(why) => d.problems.push(why),
    }
    if outs.is_empty() {
        d.problems.push("no outputs entry");
    } else {
        match outputs(outs) {
            Ok(o) => {
                d.fixed.sample_mask = o.sample_mask;
                d.outputs = Some(o);
            }
            Err(why) => d.problems.push(why),
        }
    }
    d
}

/// The compute description from a record's state entry and its stage.
pub fn compute(state: &[u8], stage: StageRef) -> Description {
    let mut d = Description { slot: 67, stages: vec![stage], ..Description::default() };
    match compute_state(state) {
        Ok(c) => d.compute = Some(c),
        Err(why) => d.problems.push(why),
    }
    d
}

// ---- Slot 55: overrides in desktop layouts ----

/// One override entry: a u32 kind, then the description (0x14c bytes in all).
pub const OVERRIDE_STRIDE: usize = 0x14c;
/// D3D12_DEPTH_STENCIL_DESC, D3D12_RASTERIZER_DESC, D3D12_BLEND_DESC sizes.
const DEPTH_DESC: usize = 0x34;
const RASTER_DESC: usize = 0x2c;
const BLEND_DESC: usize = 0x148;

fn read_face(b: &[u8], at: usize) -> Face {
    Face { fail: u32_at(b, at) as i32, depth_fail: u32_at(b, at + 4) as i32, pass: u32_at(b, at + 8) as i32, func: u32_at(b, at + 12) as i32 }
}

/// Applies one override entry (`entry` is the whole 0x14c bytes) to `d`.
/// Kinds: 0 depth-stencil, 1 rasterizer, 2 blend (desktop layouts; title
/// 0x142479f40 writes them in the order 1, 0, 5, 2); 5 (stencil masks
/// again, meaning unknown) and others are recorded and skipped.
pub fn apply_override(d: &mut Description, entry: &[u8]) {
    let kind = u32_at(entry, 0);
    d.overrides.push(kind);
    let b = &entry[4.min(entry.len())..];
    match kind {
        0 if b.len() >= DEPTH_DESC => {
            d.fixed.depth = DepthStencil {
                depth: u32_at(b, 0) != 0,
                write: u32_at(b, 4) != 0,
                func: u32_at(b, 8) as i32,
                bounds: false,
                stencil: u32_at(b, 12) != 0,
                read_mask: b[16],
                write_mask: b[17],
                front: read_face(b, 0x14),
                back: read_face(b, 0x24),
            };
        }
        1 if b.len() >= RASTER_DESC => {
            d.fixed.rasterizer = Rasterizer {
                fill: u32_at(b, 0) as i32,
                cull: u32_at(b, 4) as i32,
                front_ccw: u32_at(b, 8) != 0,
                bias: u32_at(b, 12) as i32,
                bias_clamp: f32_at(b, 16),
                slope: f32_at(b, 20),
                depth_clip: u32_at(b, 24) != 0,
                multisample: u32_at(b, 28) != 0,
                line_aa: u32_at(b, 32) != 0,
                forced_samples: u32_at(b, 36),
                conservative: u32_at(b, 40) as i32,
            };
        }
        2 if b.len() >= BLEND_DESC => {
            d.fixed.alpha_to_coverage = u32_at(b, 0) != 0;
            d.fixed.independent = u32_at(b, 4) != 0;
            for (i, t) in d.fixed.blend.iter_mut().enumerate() {
                let at = 8 + 40 * i;
                *t = Blend {
                    enable: u32_at(b, at) != 0,
                    logic_enable: u32_at(b, at + 4) != 0,
                    src: u32_at(b, at + 8) as i32,
                    dest: u32_at(b, at + 12) as i32,
                    op: u32_at(b, at + 16) as i32,
                    src_alpha: u32_at(b, at + 20) as i32,
                    dest_alpha: u32_at(b, at + 24) as i32,
                    op_alpha: u32_at(b, at + 28) as i32,
                    logic_op: u32_at(b, at + 32) as i32,
                    mask: b[at + 36],
                };
            }
            d.fixed.dual_source = d.fixed.blend[0].enable && [d.fixed.blend[0].src, d.fixed.blend[0].dest, d.fixed.blend[0].src_alpha, d.fixed.blend[0].dest_alpha].iter().any(|f| (16..=19).contains(f));
        }
        5 => {}
        _ => d.problems.push("unknown override kind"),
    }
}

/// Slot 55's description: the base's with the overrides applied.
pub fn derive(base: &Description, entries: &[u8]) -> Description {
    let mut d = Description { slot: 55, overrides: Vec::new(), ..base.clone() };
    for entry in entries.chunks_exact(OVERRIDE_STRIDE) {
        apply_override(&mut d, entry);
    }
    d
}

// ---- To the pipeline layer: a D3D12 subobject stream ----

/// A description as the subobjects of a pipeline-state stream, each `(type,
/// bytes)` with its payload where `GraphicsState::read` (VIctory base
/// `pipeline/host.rs` 92-184) looks: +4, input layout at +8.
pub struct Stream {
    pub subobjects: Vec<(u32, Vec<u64>)>,
    /// Input elements and their names, pointed at by the layout subobject.
    _elements: Vec<D3D12_INPUT_ELEMENT_DESC>,
    _names: Vec<CString>,
}

fn subobject<T: Copy>(kind: u32, offset: usize, value: &T) -> (u32, Vec<u64>) {
    let size = offset + size_of::<T>();
    let mut words = vec![0u64; size.div_ceil(8)];
    let p = words.as_mut_ptr().cast::<u8>();
    // SAFETY: `words` holds `size` bytes; the type and the value are copied in.
    unsafe {
        p.cast::<u32>().write(kind);
        p.add(offset).cast::<T>().write_unaligned(*value);
    }
    (kind, words)
}

impl Description {
    pub fn blend_desc(&self) -> D3D12_BLEND_DESC {
        let mut b = D3D12_BLEND_DESC { AlphaToCoverageEnable: self.fixed.alpha_to_coverage.into(), IndependentBlendEnable: self.fixed.independent.into(), ..Default::default() };
        for (t, s) in b.RenderTarget.iter_mut().zip(&self.fixed.blend) {
            *t = D3D12_RENDER_TARGET_BLEND_DESC {
                BlendEnable: s.enable.into(),
                LogicOpEnable: s.logic_enable.into(),
                SrcBlend: D3D12_BLEND(s.src),
                DestBlend: D3D12_BLEND(s.dest),
                BlendOp: D3D12_BLEND_OP(s.op),
                SrcBlendAlpha: D3D12_BLEND(s.src_alpha),
                DestBlendAlpha: D3D12_BLEND(s.dest_alpha),
                BlendOpAlpha: D3D12_BLEND_OP(s.op_alpha),
                LogicOp: D3D12_LOGIC_OP(s.logic_op),
                RenderTargetWriteMask: s.mask,
            };
        }
        b
    }
    pub fn rasterizer_desc(&self) -> D3D12_RASTERIZER_DESC {
        let r = &self.fixed.rasterizer;
        D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE(r.fill),
            CullMode: D3D12_CULL_MODE(r.cull),
            FrontCounterClockwise: r.front_ccw.into(),
            DepthBias: r.bias,
            DepthBiasClamp: r.bias_clamp,
            SlopeScaledDepthBias: r.slope,
            DepthClipEnable: r.depth_clip.into(),
            MultisampleEnable: r.multisample.into(),
            AntialiasedLineEnable: r.line_aa.into(),
            ForcedSampleCount: r.forced_samples,
            ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE(r.conservative),
        }
    }
    pub fn depth_desc(&self) -> D3D12_DEPTH_STENCIL_DESC {
        let d = &self.fixed.depth;
        let face = |f: &Face| D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP(f.fail),
            StencilDepthFailOp: D3D12_STENCIL_OP(f.depth_fail),
            StencilPassOp: D3D12_STENCIL_OP(f.pass),
            StencilFunc: D3D12_COMPARISON_FUNC(f.func),
        };
        D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: d.depth.into(),
            DepthWriteMask: if d.write { D3D12_DEPTH_WRITE_MASK_ALL } else { D3D12_DEPTH_WRITE_MASK_ZERO },
            DepthFunc: D3D12_COMPARISON_FUNC(or(d.func, COMPARE_ALWAYS)),
            StencilEnable: d.stencil.into(),
            StencilReadMask: d.read_mask,
            StencilWriteMask: d.write_mask,
            FrontFace: face(&d.front),
            BackFace: face(&d.back),
        }
    }

    /// The graphics state as a stream; console formats are left as they
    /// are (`GraphicsState::read` maps them with `pc_format`).
    pub fn stream(&self) -> Stream {
        let mut s = vec![
            subobject(8, 4, &self.blend_desc()),
            subobject(9, 4, &self.fixed.sample_mask),
            subobject(10, 4, &self.rasterizer_desc()),
            subobject(11, 4, &self.depth_desc()),
            subobject(14, 4, &D3D12_PRIMITIVE_TOPOLOGY_TYPE(self.topology() as i32)),
        ];
        let mut names = Vec::new();
        let mut elements = Vec::new();
        if let Some(o) = &self.outputs {
            let mut rt = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: o.targets.len() as u32, ..Default::default() };
            for (f, &t) in rt.RTFormats.iter_mut().zip(&o.targets) {
                *f = DXGI_FORMAT(t as i32);
            }
            s.push(subobject(15, 4, &rt));
            s.push(subobject(16, 4, &DXGI_FORMAT(o.depth as i32)));
            s.push(subobject(17, 4, &DXGI_SAMPLE_DESC { Count: self.samples(), Quality: 0 }));
            names = o.inputs.iter().map(|e| CString::new(e.name.as_bytes()).unwrap_or_default()).collect();
            elements = o
                .inputs
                .iter()
                .zip(&names)
                .map(|(e, n)| D3D12_INPUT_ELEMENT_DESC {
                    SemanticName: windows::core::PCSTR(n.as_ptr().cast()),
                    SemanticIndex: e.index,
                    Format: DXGI_FORMAT(e.format as i32),
                    InputSlot: e.slot,
                    AlignedByteOffset: e.offset,
                    InputSlotClass: D3D12_INPUT_CLASSIFICATION(e.class as i32),
                    InstanceDataStepRate: e.step,
                })
                .collect();
            let layout = D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: elements.as_ptr(), NumElements: elements.len() as u32 };
            s.push(subobject(12, 8, &layout));
        }
        Stream { subobjects: s, _elements: elements, _names: names }
    }
}

impl Stream {
    /// Each subobject's type and address, for `GraphicsState::read(kind, at)`.
    pub fn each(&self) -> impl Iterator<Item = (u32, *const u8)> + '_ {
        self.subobjects.iter().map(|(k, w)| (*k, w.as_ptr().cast::<u8>()))
    }
}

// ---- The device slots (title memory) ----

fn bytes<'a>(at: usize, size: usize) -> &'a [u8] {
    if at == 0 || size == 0 || size > 1 << 24 {
        return &[];
    }
    // SAFETY: the title passes `size` bytes at `at` (its loaded library).
    unsafe { std::slice::from_raw_parts(at as *const u8, size) }
}

fn word(d: *const u8, at: usize) -> usize {
    // SAFETY: inside the description the title passed.
    unsafe { d.add(at).cast::<usize>().read_unaligned() }
}

/// Library slot names in record order (pipeline-cache.md 60-75).
const SLOT_NAMES: [&str; 7] = ["pixel", "vertex", "mesh", "slot3", "slot4", "slot5", "fetch"];

fn title_stage(stage: &'static str, d: *const u8, at: usize) -> Option<StageRef> {
    let (code, size) = (word(d, at), word(d, at + 8));
    (code != 0 && size != 0).then(|| StageRef { stage, code: Code::Title { code, size, meta: word(d, at + 16), meta_size: word(d, at + 24) } })
}

/// Slot 57's 0x108-byte description: root signature +0, state {pointer,
/// size} +8, outputs +0x18, seven slots of {code, size, metadata, size}
/// from +0x28 (title 0x14248f680).
///
/// # Safety
/// `d` is a slot-57 description from the title.
pub unsafe fn read_cached_graphics(d: *const u8) -> Description {
    let stages = (0..7).filter_map(|i| title_stage(SLOT_NAMES[i], d, 0x28 + 32 * i)).collect();
    let mut x = graphics(bytes(word(d, 8), word(d, 16)), bytes(word(d, 0x18), word(d, 0x20)), stages);
    x.root = word(d, 0);
    x
}

/// Slot 67's 0x48-byte description: root +0, state +8, a second entry
/// +0x18, the stage +0x28 (title 0x14248f680, second loop).
///
/// # Safety
/// `d` is a slot-67 description from the title.
pub unsafe fn read_cached_compute(d: *const u8) -> Description {
    let stage = title_stage("compute", d, 0x28).unwrap_or(StageRef { stage: "compute", code: Code::Title { code: 0, size: 0, meta: 0, meta_size: 0 } });
    let mut x = compute(bytes(word(d, 8), word(d, 16)), stage);
    x.root = word(d, 0);
    if word(d, 0x28) == 0 {
        x.problems.push("compute description without its code");
    }
    x
}

/// Descriptions of the pipelines handed out, by object address (the
/// base of slot 55 is one of them). A later pipeline object will carry its
/// description itself; until then this table does.
static MADE: Mutex<Option<HashMap<usize, Arc<Description>>>> = Mutex::new(None);
static SEEN: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static PROBLEMS: AtomicU64 = AtomicU64::new(0);

/// The description behind a pipeline the title passes back.
pub(crate) fn description(pipeline: This) -> Option<Arc<Description>> {
    MADE.lock().unwrap_or_else(|e| e.into_inner()).as_ref()?.get(&(pipeline as usize)).cloned()
}

/// Blocks logged per slot as hex (`XNW_PIPELINE_X_DUMP=1`).
const DUMPED: u64 = 16;

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn note(index: usize, d: &Description, caller: &str, raw: &[(&str, &[u8])]) {
    let n = SEEN[index].fetch_add(1, Ordering::Relaxed) + 1;
    if !d.problems.is_empty() && PROBLEMS.fetch_add(1, Ordering::Relaxed) < 16 {
        xnw_core::warn!("pipeline x: slot {} from {caller}: {:?}", d.slot, d.problems);
    }
    if n <= DUMPED && std::env::var_os("XNW_PIPELINE_X_DUMP").is_some() {
        let f = &d.fixed;
        let b = &f.blend[0];
        xnw_core::info!(
            "pipeline x: slot {} #{n} from {caller}: {} root {:#x} targets {:?} depth {} topology {} samples {}; blend0 {} {}/{}/{} {}/{}/{} mask {:x}; depth {} write {} func {} stencil {}; cull {} fill {} ccw {} bias {}; compute {:?}; overrides {:?}",
            d.slot,
            d.combination(),
            d.root,
            d.targets(),
            d.depth_format(),
            d.topology(),
            d.samples(),
            b.enable,
            b.src,
            b.dest,
            b.op,
            b.src_alpha,
            b.dest_alpha,
            b.op_alpha,
            b.mask,
            f.depth.depth,
            f.depth.write,
            f.depth.func,
            f.depth.stencil,
            f.rasterizer.cull,
            f.rasterizer.fill,
            f.rasterizer.front_ccw,
            f.rasterizer.bias,
            d.compute,
            d.overrides
        );
        for (name, b) in raw {
            xnw_core::info!("pipeline x: slot {} #{n} {name} ({} bytes) {}", d.slot, b.len(), hex(b));
        }
    }
}

fn keep(out: *mut This, d: Description) {
    // SAFETY: `out` was checked and written by the creation.
    let Some(&p) = (unsafe { out.as_ref() }) else { return };
    if p.is_null() {
        return;
    }
    MADE.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).insert(p as usize, Arc::new(d));
}

/// `XNW_PIPELINE_X`: unset (the default) answers as the VIctory base does
/// (its device table leaves 55/57/67 unwritten: E_NOTIMPL, out cleared),
/// and the title then makes its pipelines through the
/// standard slots (60,744 in the first 40 s, V). `empty` answers with an
/// empty pipeline and S_OK, as before; `1` decodes and keeps each
/// description.
fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("XNW_PIPELINE_X").as_deref() {
        Ok("1") => 1,
        Ok("empty") => 0,
        _ => 2,
    })
}
fn on() -> bool {
    mode() == 1
}
fn refuse(out: *mut This) -> Option<HRESULT> {
    if mode() != 2 {
        return None;
    }
    // SAFETY: the title's out pointer (null tolerated).
    if let Some(o) = unsafe { out.as_mut() } {
        *o = std::ptr::null_mut();
    }
    Some(super::object::E_NOTIMPL)
}

/// Device slot 57 (offset 0x1c8): `(description 0x108, iid, out)`.
pub(crate) extern "system" fn create_cached_graphics(this: This, d: *const u8, iid: *const Iid, out: *mut This) -> HRESULT {
    if let Some(hr) = refuse(out) {
        return hr;
    }
    if d.is_null() || !on() {
        return super::pipeline::create_empty(57, iid, out);
    }
    // SAFETY: the title passes its slot-57 description.
    let x = unsafe { read_cached_graphics(d) };
    let caller = super::object::caller();
    note(1, &x, &caller, &[("desc", bytes(d as usize, 0x108)), ("state", bytes(word(d, 8), word(d, 16))), ("outputs", bytes(word(d, 0x18), word(d, 0x20)))]);
    // Cache code through the pack's 24-byte prefix, the state from
    // the copied entries (pipeline.rs; `x.stream()` is the next step).
    let hr = super::pipeline::create_cached_graphics(this, d, iid, out);
    keep(out, x);
    hr
}

/// Device slot 67 (offset 0x218): `(description 0x48, iid, out)`.
pub(crate) extern "system" fn create_cached_compute(this: This, d: *const u8, iid: *const Iid, out: *mut This) -> HRESULT {
    if let Some(hr) = refuse(out) {
        return hr;
    }
    if d.is_null() || !on() {
        return super::pipeline::create_empty(67, iid, out);
    }
    // SAFETY: the title passes its slot-67 description.
    let x = unsafe { read_cached_compute(d) };
    let caller = super::object::caller();
    note(2, &x, &caller, &[("desc", bytes(d as usize, 0x48)), ("state", bytes(word(d, 8), word(d, 16))), ("extra", bytes(word(d, 0x18), word(d, 0x20)))]);
    let hr = super::pipeline::create_cached_compute(this, d, iid, out);
    keep(out, x);
    hr
}

/// Device slot 55 (offset 0x1b8): `(base pipeline, count, entries of
/// 0x14c bytes, iid, out)`; the title uses `out` whatever the result
/// (`title+0x247a5c2`), so it always gets a pipeline.
pub(crate) extern "system" fn create_derived(this: This, base: This, count: u32, entries: *const u8, iid: *const Iid, out: *mut This) -> HRESULT {
    if let Some(hr) = refuse(out) {
        return hr;
    }
    if !on() {
        return super::pipeline::create_empty(55, iid, out);
    }
    let size = if entries.is_null() { 0 } else { count.min(16) as usize * OVERRIDE_STRIDE };
    let raw = bytes(entries as usize, size);
    let mut x = match description(base) {
        Some(b) => derive(&b, raw),
        None => {
            let mut d = derive(&Description::default(), raw);
            d.problems.push("base pipeline without a description");
            d
        }
    };
    x.slot = 55;
    let caller = super::object::caller();
    let base_word = [base as usize as u64];
    // SAFETY: a u64 viewed as its bytes.
    let base_bytes = unsafe { std::slice::from_raw_parts(base_word.as_ptr().cast::<u8>(), 8) };
    note(0, &x, &caller, &[("base", base_bytes), ("entries", raw)]);
    let hr = super::pipeline::create_derived(this, base, count, entries, iid, out);
    keep(out, x);
    hr
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put(b: &mut [u8], at: usize, v: u32) {
        b[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }

    /// A vertex-layout state block: reverse-Z GEQUAL depth with writes,
    /// back culling, target 0 premultiplied-alpha blend.
    fn vertex_state() -> Vec<u8> {
        let mut s = vec![0u8; VERTEX_FIXED];
        put(&mut s, 0, 0x4005_D800);
        put(&mut s, 4, 4);
        put(&mut s, 0x2c, 2);
        put(&mut s, 0xa0, 2); // CULL_BACK
        put(&mut s, 0xb8, (6 << 4) | 4 | 2); // GEQUAL, write, enable
        put(&mut s, 0xcc, 0xf);
        // ONE, ADD, INV_SRC_ALPHA (5)
        put(&mut s, 0xd0, (1 << 30) | (5 << 8) | 1);
        s
    }

    fn outputs_entry() -> Vec<u8> {
        let mut o = vec![0u8; 0x50];
        o[0x10] = 28;
        o[0x24] = 20;
        o[0x25] = 1;
        o[0x26] = 3;
        o[0x28] = 1;
        put(&mut o, 0x1c, u32::MAX);
        put(&mut o, 0x4c, 1);
        o.extend_from_slice(&9u32.to_le_bytes());
        o.extend_from_slice(b"POSITION\0");
        for v in [0u32, 6, 0, 0, 0, 0] {
            o.extend_from_slice(&v.to_le_bytes());
        }
        o
    }

    #[test]
    fn decodes_a_vertex_state() {
        let d = graphics(&vertex_state(), &outputs_entry(), vec![]);
        assert!(d.problems.is_empty(), "{:?}", d.problems);
        let f = &d.fixed;
        assert!(f.blend[0].enable);
        assert_eq!((f.blend[0].src, f.blend[0].dest, f.blend[0].op), (BLEND_ONE, 6, BLEND_OP_ADD));
        assert_eq!((f.blend[0].src_alpha, f.blend[0].dest_alpha), (BLEND_ONE, 6));
        assert_eq!(f.blend[0].mask, 15);
        assert_eq!(f.blend[1].mask, 0);
        assert!(f.depth.depth && f.depth.write);
        assert_eq!(f.depth.func, 7); // GREATER_EQUAL
        assert_eq!(f.rasterizer.cull, CULL_BACK);
        assert!(f.rasterizer.front_ccw && f.rasterizer.depth_clip);
        assert_eq!(d.targets(), &[28]);
        assert_eq!(d.depth_format(), 20);
        assert_eq!(d.topology(), 3);
        let o = d.outputs.as_ref().unwrap();
        assert_eq!(o.inputs[0].name, "POSITION");
        assert_eq!(o.inputs[0].format, 6);
    }

    #[test]
    fn mesh_layout_moves_the_pixel_block() {
        let v = vertex_state();
        let mut m = vec![0u8; MESH_FIXED];
        m[..0x58].copy_from_slice(&v[..0x58]);
        m[0x58 + MESH_SHIFT..VERTEX_FIXED + MESH_SHIFT].copy_from_slice(&v[0x58..]);
        put(&mut m, 4, 6);
        let d = graphics(&m, &outputs_entry(), vec![StageRef { stage: "mesh", code: Code::Library { shader: 0, meta: 0 } }]);
        assert!(d.problems.is_empty(), "{:?}", d.problems);
        assert_eq!(d.fixed.depth.func, 7);
        assert!(d.fixed.blend[0].enable);
        assert!(d.registers.unwrap().mesh);
    }

    #[test]
    fn separate_alpha_dual_source_stencil_and_bias() {
        let mut s = vertex_state();
        // colour SRC1_ALPHA(17)/INV_SRC1_ALPHA(18) MAX; alpha SRC_COLOR(2) -> SRC_ALPHA twin
        put(&mut s, 0xd0, (1 << 30) | (1 << 29) | (2 << 16) | (18 << 8) | (3 << 5) | 17);
        put(&mut s, 0xd4, 1 << 30);
        put(&mut s, 0xb8, 1 | (1 << 7) | (3 << 8) | (5 << 20));
        put(&mut s, 0xbc, 3 | (8 << 4) | (1 << 8) | (7 << 12));
        put(&mut s, 0xc0, (0x0f << 8) | (0xf0 << 16));
        put(&mut s, 0xa0, 1 | (1 << 11));
        s[0xb4..0xb8].copy_from_slice(&4.0f32.to_le_bytes());
        s[0xb0..0xb4].copy_from_slice(&32.0f32.to_le_bytes());
        put(&mut s, 0x9c, 0x0400_0000);
        let d = graphics(&s, &[], vec![]);
        assert_eq!(d.problems, vec!["no outputs entry"]);
        let f = &d.fixed;
        assert!(f.dual_source && !f.independent);
        assert_eq!((f.blend[0].src, f.blend[0].dest, f.blend[0].op), (18, 19, 5));
        assert_eq!(f.blend[0].src_alpha, 5);
        assert_eq!(f.blend[1], Blend::default());
        assert!(f.depth.stencil && !f.depth.depth);
        assert_eq!(f.depth.front, Face { fail: 3, pass: 7, depth_fail: 2, func: 4 });
        assert_eq!(f.depth.back.fail, 6);
        assert_eq!(f.depth.back.func, 6);
        assert_eq!((f.depth.read_mask, f.depth.write_mask), (0x0f, 0xf0));
        assert_eq!(f.rasterizer.cull, CULL_FRONT);
        assert_eq!(f.rasterizer.bias, 4);
        assert_eq!(f.rasterizer.slope, 2.0);
        assert!(!f.rasterizer.depth_clip);
    }

    #[test]
    fn refuses_unknown_blocks() {
        let mut s = vertex_state();
        put(&mut s, 0, 0x1234);
        assert!(registers(&s).is_err());
        assert!(registers(&s[..100]).is_err());
        let mut s = vertex_state();
        put(&mut s, 4, 5);
        assert!(registers(&s).is_err());
        let mut s = vertex_state();
        put(&mut s, 0x0c, 2);
        assert_eq!(graphics(&s, &outputs_entry(), vec![]).problems, vec!["state entry size is not fixed part + tail"]);
    }

    #[test]
    fn compute_state_reads_threads() {
        let mut c = vec![0u8; 0x1c];
        put(&mut c, 0, 0x4005_CF00);
        put(&mut c, 4, 0x6C09_0102);
        c[0x0c] = 8;
        c[0x0e] = 8;
        c[0x10] = 1;
        c[0x12] = 16;
        let d = compute(&c, StageRef { stage: "compute", code: Code::Library { shader: 1, meta: 2 } });
        assert!(d.problems.is_empty());
        assert_eq!(d.compute.unwrap().threads, [8, 8, 1]);
        assert!(d.compute.unwrap().wave32);
        assert!(d.is_compute());
    }

    #[test]
    fn derive_applies_desktop_overrides() {
        let base = graphics(&vertex_state(), &outputs_entry(), vec![]);
        let mut e = vec![0u8; 3 * OVERRIDE_STRIDE];
        // kind 1 rasterizer: wireframe, cull none
        put(&mut e, 0, 1);
        put(&mut e, 4, 2);
        put(&mut e, 8, 1);
        put(&mut e, 4 + 24, 1);
        // kind 0 depth: off, write zero, func ALWAYS
        let k = OVERRIDE_STRIDE;
        put(&mut e, k, 0);
        put(&mut e, k + 4 + 8, 8);
        // kind 2 blend: target 0 SRC_ALPHA / INV_SRC_ALPHA, mask 7
        let k = 2 * OVERRIDE_STRIDE;
        put(&mut e, k, 2);
        put(&mut e, k + 4 + 8, 1);
        put(&mut e, k + 4 + 16, 5);
        put(&mut e, k + 4 + 20, 6);
        put(&mut e, k + 4 + 24, 1);
        e[k + 4 + 8 + 36] = 7;
        let d = derive(&base, &e);
        assert_eq!(d.slot, 55);
        assert_eq!(d.overrides, vec![1, 0, 2]);
        assert_eq!(d.fixed.rasterizer.fill, FILL_WIREFRAME);
        assert_eq!(d.fixed.rasterizer.cull, CULL_NONE);
        assert!(!d.fixed.depth.depth);
        assert_eq!(d.fixed.depth.func, 8);
        assert!(d.fixed.blend[0].enable);
        assert_eq!((d.fixed.blend[0].src, d.fixed.blend[0].dest), (5, 6));
        assert_eq!(d.fixed.blend[0].mask, 7);
        assert_eq!(d.targets(), &[28]);
        assert!(d.problems.is_empty());
    }

    #[test]
    fn stream_carries_the_state() {
        let d = graphics(&vertex_state(), &outputs_entry(), vec![]);
        let s = d.stream();
        let kinds: Vec<u32> = s.each().map(|(k, _)| k).collect();
        assert_eq!(kinds, vec![8, 9, 10, 11, 14, 15, 16, 17, 12]);
        for (k, at) in s.each() {
            // SAFETY: the stream's own subobjects.
            unsafe {
                assert_eq!(at.cast::<u32>().read(), k);
                match k {
                    8 => {
                        let b = at.add(4).cast::<D3D12_BLEND_DESC>().read_unaligned();
                        assert!(b.RenderTarget[0].BlendEnable.as_bool());
                        assert_eq!(b.RenderTarget[0].DestBlend, D3D12_BLEND_INV_SRC_ALPHA);
                    }
                    11 => {
                        let z = at.add(4).cast::<D3D12_DEPTH_STENCIL_DESC>().read_unaligned();
                        assert_eq!(z.DepthFunc, D3D12_COMPARISON_FUNC_GREATER_EQUAL);
                    }
                    12 => {
                        let l = at.add(8).cast::<D3D12_INPUT_LAYOUT_DESC>().read_unaligned();
                        assert_eq!(l.NumElements, 1);
                        let e = &*l.pInputElementDescs;
                        assert_eq!(std::ffi::CStr::from_ptr(e.SemanticName.0.cast()).to_str().unwrap(), "POSITION");
                    }
                    15 => {
                        let rt = at.add(4).cast::<D3D12_RT_FORMAT_ARRAY>().read_unaligned();
                        assert_eq!((rt.NumRenderTargets, rt.RTFormats[0].0), (1, 28));
                    }
                    _ => {}
                }
            }
        }
    }
}
