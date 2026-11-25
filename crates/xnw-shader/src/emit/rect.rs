//! XNW: the console's rectangle lists on the host. A rectangle is three
//! vertices; the hardware makes the fourth from the vertex shader's outputs.
//! D3D12 has no such topology, so a pipeline drawing rectangles gets this
//! geometry shader (`gs_6_6`): a triangle in, a four-vertex strip out, every
//! output of the vertex shader passed through and made linearly for the
//! fourth vertex. Draws, indexed draws and indirect draws then run as
//! triangle lists with their arguments unchanged (on the GPU).
//!
//! The corner and the strip are TuxBox's (`runtime/d3d12/src/device/
//! rectangle.hlsl` and `rectangle.rs`, commit 11a278a, Apache-2.0): the
//! corner is the vertex at the right angle in screen space (the smallest
//! |dot| of its two edges), the fourth vertex `prev + next - corner` for
//! float outputs (integer outputs keep the corner's), the strip `corner,
//! next, prev, fourth`, which keeps the input winding. TuxBox writes HLSL and
//! compiles it with DXC at run time; here the same program is emitted as
//! DXIL directly, with no compiler.

use super::bitcode::{Attr, Bin, Term, V, pred};
use super::container::SigElement;
use super::shader::Shader;
use crate::ShaderError;
use crate::dxbc::{Container, FourCC};

/// The rectangle geometry shader for the translated vertex shader `vs` (a
/// host container): its outputs are this shader's inputs and outputs.
pub fn rect_gs(vs: &[u8]) -> Result<Vec<u8>, ShaderError> {
    let c = Container::parse(vs)?;
    let outputs = match c.part(FourCC::OSG1) {
        Some(p) => super::signature(p.data)?,
        None => Vec::new(),
    };
    if !outputs.iter().any(|o| o.kind == 3 && o.comp == 9 && o.col == 0 && o.cols == 4) {
        return Err(ShaderError::Unsupported { offset: 0, what: "a rectangle vertex shader without SV_Position" });
    }
    let elements: Vec<SigElement> = outputs
        .into_iter()
        .filter(|o| o.row.is_some())
        .map(|mut e| {
            // Interpolation as emission declares vertex outputs.
            e.interp = match (e.kind, e.comp) {
                (3, _) => 4,
                (0 | 6 | 7, 9) => 2,
                (0, _) => 1,
                _ => 0,
            };
            e.usage = (1u8 << e.cols) - 1;
            e.half = false;
            e
        })
        .collect();
    let mut s = Shader::new(false);
    s.geometry = true;
    s.inputs = elements.clone();
    s.outputs = elements.clone();
    let e = s.entry;
    let void = s.t.void;
    let (i32t, f32t) = (s.t.i32, s.t.f32);
    // loaded[element][column] = the three vertices' values.
    let mut loaded = Vec::new();
    for (id, el) in elements.iter().enumerate() {
        let float = el.comp == 9;
        let (suffix, ty) = if float { (".f32", f32t) } else { (".i32", i32t) };
        let mut cols = Vec::new();
        for col in 0..el.cols {
            let v: Vec<V> = (0..3)
                .map(|vertex| {
                    let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), s.m.i32(vertex)];
                    s.op(e, &format!("loadInput{suffix}"), Attr::ReadNone, ty, 4, &args)
                })
                .collect();
            cols.push(v);
        }
        loaded.push((float, suffix, cols));
    }
    // Screen positions p[i] = position.xy / position.w.
    let pos = elements.iter().position(|o| o.kind == 3).expect("checked");
    let (_, _, pc) = &loaded[pos];
    let p: Vec<[V; 2]> = (0..3)
        .map(|i| {
            let x = s.m.bin(e, Bin::SDiv, pc[0][i], pc[3][i]);
            let y = s.m.bin(e, Bin::SDiv, pc[1][i], pc[3][i]);
            [x, y]
        })
        .collect();
    let sub = |s: &mut Shader, a: [V; 2], b: [V; 2]| [s.m.bin(e, Bin::Sub, a[0], b[0]), s.m.bin(e, Bin::Sub, a[1], b[1])];
    let (e01, e12, e20) = (sub(&mut s, p[1], p[0]), sub(&mut s, p[2], p[1]), sub(&mut s, p[0], p[2]));
    let absdot = |s: &mut Shader, a: [V; 2], b: [V; 2]| {
        let x = s.m.bin(e, Bin::Mul, a[0], b[0]);
        let y = s.m.bin(e, Bin::Mul, a[1], b[1]);
        let d = s.m.bin(e, Bin::Add, x, y);
        s.op(e, "unary.f32", Attr::ReadNone, f32t, 6, &[d])
    };
    let (ax, ay, az) = (absdot(&mut s, e20, e01), absdot(&mut s, e01, e12), absdot(&mut s, e12, e20));
    // corner 0 if ax <= min(ay, az); else 1 if ay <= az; else 2.
    let xy = s.m.cmp(e, pred::OLE, ax, ay);
    let xz = s.m.cmp(e, pred::OLE, ax, az);
    let c0 = s.m.bin(e, Bin::And, xy, xz);
    let yz = s.m.cmp(e, pred::OLE, ay, az);
    let t = s.m.bool(true);
    let not0 = s.m.bin(e, Bin::Xor, c0, t);
    let c1 = s.m.bin(e, Bin::And, not0, yz);
    // pick(k): vertex (corner + k) % 3.
    let pick = |s: &mut Shader, v: &[V], k: usize| {
        let inner = s.m.select(e, c1, v[(1 + k) % 3], v[(2 + k) % 3]);
        s.m.select(e, c0, v[k % 3], inner)
    };
    // Per output column: corner, next, prev, fourth.
    let mut out = Vec::new();
    for (float, suffix, cols) in &loaded {
        let mut oc = Vec::new();
        for v in cols {
            let corner = pick(&mut s, v, 0);
            let next = pick(&mut s, v, 1);
            let prev = pick(&mut s, v, 2);
            let fourth = if *float {
                let sum = s.m.bin(e, Bin::Add, prev, next);
                s.m.bin(e, Bin::Sub, sum, corner)
            } else {
                corner
            };
            oc.push([corner, next, prev, fourth]);
        }
        out.push((*suffix, oc));
    }
    for vertex in 0..4 {
        for (id, (suffix, cols)) in out.iter().enumerate() {
            for (col, v) in cols.iter().enumerate() {
                let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col as u8), v[vertex]];
                s.op(e, &format!("storeOutput{suffix}"), Attr::NoUnwind, void, 5, &args);
            }
        }
        let stream = [s.m.i8(0)];
        s.op(e, "emitStream", Attr::NoUnwind, void, 97, &stream);
    }
    s.m.terminate(e, Term::Ret);
    Ok(s.finish())
}

#[cfg(test)]
mod tests {
    use super::super::mesh::{Element, Layout, passthrough_vs};

    /// A vertex shader with a position, a float4 and a uint output: its
    /// rectangle geometry shader passes the validator.
    #[test]
    fn rect_gs_validates() {
        let el = |name: &str, kind, comp, row| Element { name: name.into(), index: 0, kind, comp, row, col: 0, cols: 4 };
        let layout = Layout { max_vertices: 64, max_prims: 64, rows: 3, elements: vec![el("SV_Position", 3, 9, 0), el("TEXCOORD", 0, 9, 1), el("FLAGS", 0, 5, 2)] };
        // A pixel shader reading both attributes gives the VS those outputs.
        let ps = super::super::host::stub_ps();
        let vs = passthrough_vs(&layout, Some(&ps.dxil)).unwrap();
        let mut gs = super::rect_gs(&vs).unwrap();
        let Ok(mut v) = super::super::validator::Validator::load(super::super::validator::Validator::default_path().as_deref()) else {
            eprintln!("no validator: skipped");
            return;
        };
        v.validate(&mut gs).unwrap();
    }
}
