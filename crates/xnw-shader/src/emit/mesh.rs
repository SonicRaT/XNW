//! Mesh programs on a host without mesh shaders (XNW).
//! The console mesh (NGG) program runs as a
//! compute shader (`cs_6_6`, the console's wave size, its own thread count)
//! that writes one fixed-size record per group plus the group's index
//! triples ([`super::MeshForm::Fallback`], lowered in `mesh_lower.rs`); one
//! `DrawIndexedInstanced(groups * max_prims * 3)` with
//! [`passthrough_vs`] and the title's pixel shader draws every group.
//!
//! Record of group `g` at `g * record_bytes` (all dwords raw bits):
//!
//! | bytes | holds |
//! |---|---|
//! | 0..16 | vertex count, primitive count (the `GS_ALLOC_REQ` request, 0 if none was made), M0 as sent, 0 |
//! | 16 + v * stride | vertex `v`: 16 bytes per output-signature row `0..rows`, channel `c` at `4 c` |
//! | prim_offset + p * 4 | primitive `p`: the exported word (indices at bits 0, 10, 20; bit 31 culls) |
//!
//! Index buffer of group `g` at `g * max_prims * 12`: three `u32` per
//! primitive slot, `g * max_vertices + index`, or `g * max_vertices` three
//! times for a primitive not exported, culled or out of bounds.
//!
//! Bindings beyond the program's own (user data, heaps, root views, static
//! samplers, all as for the mesh shader): compute `u0` record buffer and
//! `u1` index buffer, vertex shader `t0` record buffer, all raw buffers in
//! [`RECORD_SPACE`] (root descriptors on the host).
//!
//! The draw's vertex shader outputs exactly what the pixel shader reads:
//! its input signature copied (so D3D12 links them), each element filled
//! from the record row the console would link it to (see [`link`]), plus
//! `SV_Position` and the clip and cull distances. A vertex at or past the
//! header's vertex count gets a NaN position, so a group that requested
//! nothing (culled on the console) draws nothing.

use super::bitcode::{Attr, Bin, Cast, Term, V, pred};
use super::container::SigElement;
use super::shader::Shader;
use super::{HostShader, MeshForm, Options};
use crate::ShaderError;
use crate::dxbc::{Container, FourCC};

/// Register space of the record and index buffers.
pub const RECORD_SPACE: u32 = 2;
/// `u0` (compute) and `t0` (vertex shader): the record buffer.
pub const RECORD_REGISTER: u32 = 0;
/// `u1` (compute): the index buffer.
pub const INDEX_REGISTER: u32 = 1;
/// Bytes before the first vertex of a record.
pub const HEADER_BYTES: u32 = 16;

/// One output-signature element of the mesh program, where its record
/// row holds it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Element {
    pub name: String,
    pub index: u32,
    /// DXIL semantic kind (0 arbitrary, 3 position, 6 clip, 7 cull).
    pub kind: u8,
    /// DXIL component type (9 f32, 4 i32, 5 u32).
    pub comp: u8,
    pub row: u32,
    pub col: u8,
    pub cols: u8,
}

/// The fixed capacity of one group's record (per mesh program).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub max_vertices: u32,
    pub max_prims: u32,
    /// 16-byte rows per vertex: one past the highest signature row.
    pub rows: u32,
    pub elements: Vec<Element>,
}

impl Layout {
    pub(crate) fn new(outputs: &[SigElement], max_vertices: u32, max_prims: u32) -> Layout {
        let rows = outputs.iter().filter_map(|o| o.row).map(|r| r + 1).max().unwrap_or(1);
        let elements = outputs
            .iter()
            .filter_map(|o| {
                Some(Element { name: o.name.clone(), index: o.index, kind: o.kind, comp: o.comp, row: o.row?, col: o.col, cols: o.cols })
            })
            .collect();
        Layout { max_vertices, max_prims, rows, elements }
    }

    pub fn vertex_stride(&self) -> u32 {
        16 * self.rows
    }

    pub fn prim_offset(&self) -> u32 {
        HEADER_BYTES + self.max_vertices * self.vertex_stride()
    }

    pub fn record_bytes(&self) -> u32 {
        self.prim_offset() + self.max_prims * 4
    }

    /// Index-buffer bytes per group.
    pub fn index_bytes(&self) -> u32 {
        self.max_prims * 12
    }

    /// The record row holding `SV_Position`.
    pub fn position_row(&self) -> Option<u32> {
        self.elements.iter().find(|e| e.kind == 3).map(|e| e.row)
    }
}

/// The record layout of a console container's mesh program (what
/// [`super::MeshForm::Fallback`] emission writes).
pub fn record_layout(container: &[u8]) -> Result<Layout, ShaderError> {
    let c = Container::parse(container)?;
    let dxil = c.part(FourCC::DXIL).ok_or(ShaderError::Malformed("no DXIL part"))?;
    let native = crate::native_programs(dxil.data)?
        .into_iter()
        .find(|p| p.stage == crate::native::HwStage::Ms)
        .ok_or(ShaderError::Unsupported { offset: 0, what: "a container without a mesh program" })?;
    let crate::native::StageIo::Ms(ms) = &native.io else {
        return Err(ShaderError::Unsupported { offset: 0, what: "a mesh program without mesh stage data" });
    };
    let outputs = match c.part(FourCC::OSG1) {
        Some(p) => super::signature(p.data)?,
        None => Vec::new(),
    };
    Ok(Layout::new(&outputs, ms.max_vertices, ms.max_primitives))
}

/// The mesh program of `container` in compute form, and its record layout.
pub fn translate_fallback(container: &[u8], opts: Options) -> Result<(HostShader, Layout), ShaderError> {
    let shader = super::translate_with(container, Options { mesh: MeshForm::Fallback, ..opts })?;
    Ok((shader, record_layout(container)?))
}

/// Where a pixel-shader input comes from in the record: (row, channel of
/// the input's first column), or `None` (written as 0).
///
/// By semantic name and index first, as D3D12 links stages and as the
/// console's pipeline creation does from the containers' signatures; by
/// signature row (the attribute index) only where no output has the name.
/// On 200 sampled pairs (`xnw translate --mesh fallback --pipelines 200`),
/// 1,088 of 1,263 pixel inputs have a same-named mesh output, 851 of them at
/// the same row: the two rules disagree for 237, so the order matters. Another
/// runtime's notes pair by effect hash too and found
/// names that do not match; pairing by hash is only approximate, so which
/// rule the console follows for those is open.
pub fn link(layout: &Layout, input: (&str, u32, u8, u32, u8)) -> Option<(u32, u8)> {
    let (name, index, kind, row, col) = input;
    let same_kind = |e: &&Element| e.kind == kind;
    if kind == 0 {
        if let Some(e) = layout.elements.iter().filter(same_kind).find(|e| e.name.eq_ignore_ascii_case(name) && e.index == index) {
            return Some((e.row, e.col));
        }
        return layout.elements.iter().filter(same_kind).any(|e| e.row == row).then_some((row, col));
    }
    layout.elements.iter().filter(same_kind).find(|e| e.index == index).map(|e| (e.row, e.col))
}

/// How a pixel shader's arbitrary inputs (translated container `ps`) meet
/// `layout`'s rows: [inputs, a record element at the same row, one of the
/// same name and index, that one at the same row].
pub fn link_stats(layout: &Layout, ps: &[u8]) -> Result<[usize; 4], ShaderError> {
    let c = Container::parse(ps)?;
    let inputs = match c.part(FourCC::ISG1) {
        Some(p) => super::signature(p.data)?,
        None => Vec::new(),
    };
    let mut n = [0; 4];
    for e in inputs.iter().filter(|e| e.kind == 0) {
        let Some(row) = e.row else { continue };
        n[0] += 1;
        let arb = || layout.elements.iter().filter(|x| x.kind == 0);
        n[1] += usize::from(arb().any(|x| x.row == row));
        if let Some(x) = arb().find(|x| x.name.eq_ignore_ascii_case(&e.name) && x.index == e.index) {
            n[2] += 1;
            n[3] += usize::from(x.row == row);
        }
    }
    Ok(n)
}

/// The vertex shader that draws `layout`'s records for the pixel shader
/// whose translated container is `ps` (`None`: no pixel shader, position
/// and distances only). Input: `SV_VertexID` (`group * max_vertices +
/// vertex`, the index the compute pass wrote).
pub fn passthrough_vs(layout: &Layout, ps: Option<&[u8]>) -> Result<Vec<u8>, ShaderError> {
    let ps_inputs = match ps {
        Some(bytes) => {
            let c = Container::parse(bytes)?;
            match c.part(FourCC::ISG1) {
                Some(p) => super::signature(p.data)?,
                None => Vec::new(),
            }
        }
        None => Vec::new(),
    };
    // (output element, source row and first channel per element).
    let mut outs: Vec<(SigElement, Option<(u32, u8)>)> = Vec::new();
    for e in ps_inputs.iter().filter(|e| e.row.is_some() && matches!(e.kind, 0 | 3 | 6 | 7)) {
        let row = e.row.expect("filtered");
        let src = link(layout, (&e.name, e.index, e.kind, row, e.col));
        outs.push((e.clone(), src));
    }
    let mut free = outs.iter().filter_map(|o| o.0.row).map(|r| r + 1).max().unwrap_or(0);
    let element = |name: &str, index: u32, kind: u8, comp: u8, row: u32, col: u8, cols: u8| SigElement {
        name: name.to_owned(),
        index,
        kind,
        comp,
        interp: 0,
        row: Some(row),
        col,
        cols,
        usage: 0,
        half: false,
    };
    if !outs.iter().any(|o| o.0.kind == 3) {
        let src = layout.position_row().map(|r| (r, 0));
        outs.push((element("SV_Position", 0, 3, 9, free, 0, 4), src));
        free += 1;
    }
    for d in layout.elements.iter().filter(|e| matches!(e.kind, 6 | 7)) {
        if outs.iter().any(|o| o.0.kind == d.kind && o.0.index == d.index) {
            continue;
        }
        outs.push((element(&d.name, d.index, d.kind, d.comp, free, d.col, d.cols), Some((d.row, d.col))));
        free += 1;
    }
    for (e, _) in &mut outs {
        // Interpolation as emission declares vertex outputs.
        e.interp = match (e.kind, e.comp) {
            (3, _) => 4,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
        e.usage = (1u8 << e.cols) - 1;
        e.half = false;
    }

    let mut s = Shader::new(false);
    s.inputs = vec![SigElement { usage: 1, ..element("SV_VertexID", 0, 1, 5, 0, 0, 1) }];
    s.outputs = outs.iter().map(|o| o.0.clone()).collect();
    let e = s.entry;
    let i32t = s.t.i32;
    let void = s.t.void;
    let args = [s.m.i32(0), s.m.i32(0), s.m.i8(0), s.m.undef(i32t)];
    let vid = s.op(e, "loadInput.i32", Attr::ReadNone, i32t, 4, &args);
    let mv = s.m.i32(layout.max_vertices);
    let g = s.m.bin(e, Bin::UDiv, vid, mv);
    let v = s.m.bin(e, Bin::URem, vid, mv);
    let rb = s.m.i32(layout.record_bytes());
    let rec = s.m.bin(e, Bin::Mul, g, rb);
    let h = s.raw_view(false, RECORD_REGISTER);
    let load = |s: &mut Shader, at: V, mask: u8| {
        let (u, m, al) = (s.m.undef(i32t), s.m.i8(mask), s.m.i32(4));
        let res_ret = s.t.res_ret;
        s.op(e, "rawBufferLoad.f32", Attr::ReadOnly, res_ret, 139, &[h, at, u, m, al])
    };
    let hdr = load(&mut s, rec, 1);
    let count = s.m.extract(e, hdr, 0);
    let count = s.m.cast(e, Cast::Bitcast, count, i32t);
    let dead = s.m.cmp(e, pred::IUGE, v, count);
    let stride = s.m.i32(layout.vertex_stride());
    let vbase = s.m.bin(e, Bin::Mul, v, stride);
    let hb = s.m.i32(HEADER_BYTES);
    let vbase = s.m.bin(e, Bin::Add, vbase, hb);
    let vbase = s.m.bin(e, Bin::Add, rec, vbase);
    let mut rows: Vec<(u32, V)> = Vec::new();
    let nan = s.m.f32(f32::NAN.to_bits());
    let zero = s.m.f32(0);
    for (id, (el, src)) in outs.iter().enumerate() {
        for c in 0..el.cols {
            let value = match *src {
                Some((row, first)) if first + c < 4 => {
                    let r = match rows.iter().find(|r| r.0 == row) {
                        Some(&(_, r)) => r,
                        None => {
                            let k = s.m.i32(16 * row);
                            let at = s.m.bin(e, Bin::Add, vbase, k);
                            let r = load(&mut s, at, 0xF);
                            rows.push((row, r));
                            r
                        }
                    };
                    s.m.extract(e, r, u32::from(first + c))
                }
                _ => zero,
            };
            let value = if el.kind == 3 { s.m.select(e, dead, nan, value) } else { value };
            let (suffix, value) = if el.comp == 9 { (".f32", value) } else { (".i32", s.m.cast(e, Cast::Bitcast, value, i32t)) };
            let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(c), value];
            s.op(e, &format!("storeOutput{suffix}"), Attr::NoUnwind, void, 5, &args);
        }
    }
    s.m.terminate(e, Term::Ret);
    Ok(s.finish())
}

