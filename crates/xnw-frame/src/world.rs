//! World space: a draw's transform source and world matrix, the scene OBJ
//! writer, bounds. Moved verbatim from apps/xnw/src/worldprobe.rs (59afddb).

use std::collections::HashMap;
use std::io::Write as _;

use crate::decode::{Mesh, transform};

/// Vertices per OBJ file at most.
pub const OBJ_MAX_VERTICES: usize = 24_000_000;

/// Where a draw's transform came from.
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub enum WorldSrc {
    #[default]
    None,
    /// srv[slot] with lazy 0x80: CPU matrices (0x144E3D960 and the three
    /// other setters store 128 B and the pointer) [V].
    Bound,
    /// The cache entry by the context's index [V 0x144E3D960].
    Cache,
    /// srv[slot] is a grc buffer object: its first 128 bytes [I].
    BoundBuffer,
    /// Offline only: another draw of the same geometry.
    Borrowed,
}

impl WorldSrc {
    pub fn name(self) -> &'static str {
        ["-", "bound", "cache", "bound-buffer", "borrowed"][self as usize]
    }
    pub fn parse(s: &str) -> WorldSrc {
        match s {
            "bound" => WorldSrc::Bound,
            "cache" => WorldSrc::Cache,
            "bound-buffer" => WorldSrc::BoundBuffer,
            "borrowed" => WorldSrc::Borrowed,
            _ => WorldSrc::None,
        }
    }
}

pub fn rigid_world(b: &[f32; 12], w: &[f32; 16]) -> [f32; 16] {
    let bt: [f32; 16] = [b[0], b[4], b[8], 0.0, b[1], b[5], b[9], 0.0, b[2], b[6], b[10], 0.0, b[3], b[7], b[11], 1.0];
    xnw_world::mul(&bt, w)
}

/// One draw for the world OBJ.
pub struct WorldDraw {
    pub index: usize,
    pub geom: u64,
    pub effect: u64,
    pub viewport: u64,
    /// Camera-relative transform (current, previous), if one was read.
    pub world: Option<[f32; 32]>,
    pub src: WorldSrc,
}

/// The camera position of a row-vector view matrix (`p_view = p R + T`):
/// `-T R^T`, component i = `-(T . row i)`.
pub fn camera_position(view: &[f32]) -> [f32; 3] {
    std::array::from_fn(|i| -(view[12] * view[4 * i] + view[13] * view[4 * i + 1] + view[14] * view[4 * i + 2]))
}

/// A draw's absolute world matrix: its camera-relative transform with the
/// draw viewport's camera position added to the translation row. In one capture
/// the same geometry drawn in several viewports lands on one absolute
/// position this way (e.g. 0x11b46f340 in 3 viewports at -241.14,
/// -194.58, 251.40) [V by the data].
pub fn absolute(world: &[f32], view: &[f32]) -> [f32; 16] {
    let c = camera_position(view);
    let mut m: [f32; 16] = world[..16].try_into().unwrap();
    for i in 0..3 {
        m[12 + i] += c[i];
    }
    m
}

/// A transform that can be an object's: finite, each 3x3 row of length
/// 1e-3..1e3, translation under 100 km. Matrices read by an
/// older probe from srv[] objects reached 6e8.
pub fn plausible(m: &[f32; 16]) -> bool {
    let row = |r: usize| (m[4 * r] * m[4 * r] + m[4 * r + 1] * m[4 * r + 1] + m[4 * r + 2] * m[4 * r + 2]).sqrt();
    m.iter().all(|v| v.is_finite()) && (0..3).all(|r| (1e-3..1e3).contains(&row(r))) && (0..3).all(|i| m[12 + i].abs() < 1e5)
}

/// The 3x3 part applied to a normal, renormalised.
pub fn rotate(m: &[f32], n: [f32; 3]) -> [f32; 3] {
    let r: [f32; 3] = std::array::from_fn(|j| n[0] * m[j] + n[1] * m[4 + j] + n[2] * m[8 + j]);
    let l = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    if l > 0.0 { r.map(|v| v / l) } else { r }
}

pub struct WorldStats {
    pub sources: String,
    pub borrowed: usize,
    pub missing: usize,
    pub skipped: usize,
    pub vertices: usize,
    pub bounds: Bounds,
}

/// Writes `scene.obj` (absolute world, one object `d<i>_v<viewport>_g<geom>_e<effect>`
/// per draw, with vt / vn) and `scene-local.obj` (one object per geometry,
/// untransformed). A draw without a transform borrows the absolute matrix
/// of another draw of the same geometry (`Borrowed`).
pub fn write_world(dir: &std::path::Path, frame: u64, draws: &mut [WorldDraw], meshes: &HashMap<u64, &Mesh>, views: &HashMap<u64, [f32; 16]>) -> std::io::Result<WorldStats> {
    let abs_of = |d: &WorldDraw| -> Option<[f32; 16]> { Some(absolute(&d.world?, views.get(&d.viewport)?)).filter(|m| plausible(m)) };
    let mut by_geom: HashMap<u64, [f32; 16]> = HashMap::new();
    for d in draws.iter() {
        if let Some(m) = abs_of(d) {
            by_geom.entry(d.geom).or_insert(m);
        }
    }
    let mut st = WorldStats { sources: String::new(), borrowed: 0, missing: 0, skipped: 0, vertices: 0, bounds: Bounds::new() };
    let mut count: HashMap<&'static str, usize> = HashMap::new();
    let mut out = std::io::BufWriter::new(std::fs::File::create(dir.join("scene.obj"))?);
    writeln!(out, "# XNW world probe, frame {frame}: absolute world (camera-relative transform + the draw viewport's camera position), one object per draw")?;
    let mut base = 0u64;
    for d in draws.iter_mut() {
        let Some(m) = meshes.get(&d.geom).filter(|m| !m.tris.is_empty()) else { continue };
        let w = match abs_of(d) {
            Some(w) => w,
            None => match by_geom.get(&d.geom) {
                Some(w) => {
                    d.src = WorldSrc::Borrowed;
                    st.borrowed += 1;
                    *w
                }
                None => {
                    st.missing += 1;
                    continue;
                }
            },
        };
        *count.entry(d.src.name()).or_default() += 1;
        if base as usize + m.pos.len() > OBJ_MAX_VERTICES {
            st.skipped += 1;
            continue;
        }
        let pos: Vec<[f32; 3]> = m.pos.iter().map(|p| transform(&w, *p)).collect();
        let nrm: Vec<[f32; 3]> = m.normal.iter().map(|n| rotate(&w, *n)).collect();
        for p in &pos {
            st.bounds.add(*p);
        }
        st.vertices += pos.len();
        write_obj_object(&mut out, &format!("d{}_v{:x}_g{:x}_e{:x}_s{}", d.index, d.viewport, d.geom, d.effect, d.src.name()), &pos, &m.uv, &nrm, &m.tris, &mut base)?;
    }
    out.flush()?;
    let mut c: Vec<_> = count.into_iter().collect();
    c.sort();
    st.sources = c.iter().map(|(k, n)| format!("{k} {n}")).collect::<Vec<_>>().join(", ");
    let mut out = std::io::BufWriter::new(std::fs::File::create(dir.join("scene-local.obj"))?);
    writeln!(out, "# XNW world probe, frame {frame}, untransformed, one object per geometry")?;
    let mut base = 0u64;
    let mut keys: Vec<&u64> = meshes.keys().collect();
    keys.sort();
    for g in keys {
        let m = meshes[g];
        if m.tris.is_empty() || base as usize + m.pos.len() > OBJ_MAX_VERTICES {
            continue;
        }
        write_obj_object(&mut out, &format!("g{g:x}"), &m.pos, &m.uv, &m.normal, &m.tris, &mut base)?;
    }
    out.flush()?;
    Ok(st)
}

/// One OBJ object: positions, optional uv and normals (same count), triangles
/// (0-based into `pos`); `base` counts the vertices written before it.
pub fn write_obj_object<W: std::io::Write>(out: &mut W, name: &str, pos: &[[f32; 3]], uv: &[[f32; 2]], normal: &[[f32; 3]], tris: &[[u32; 3]], base: &mut u64) -> std::io::Result<()> {
    writeln!(out, "o {name}")?;
    for p in pos {
        writeln!(out, "v {:.4} {:.4} {:.4}", p[0], p[1], p[2])?;
    }
    let (has_uv, has_n) = (!uv.is_empty() && uv.len() == pos.len(), !normal.is_empty() && normal.len() == pos.len());
    if has_uv {
        for t in uv {
            writeln!(out, "vt {:.5} {:.5}", t[0], t[1])?;
        }
    }
    if has_n {
        for n in normal {
            writeln!(out, "vn {:.4} {:.4} {:.4}", n[0], n[1], n[2])?;
        }
    }
    for t in tris {
        let [a, b, c] = t.map(|i| *base + u64::from(i) + 1);
        match (has_uv, has_n) {
            (true, true) => writeln!(out, "f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}")?,
            (true, false) => writeln!(out, "f {a}/{a} {b}/{b} {c}/{c}")?,
            (false, true) => writeln!(out, "f {a}//{a} {b}//{b} {c}//{c}")?,
            (false, false) => writeln!(out, "f {a} {b} {c}")?,
        }
    }
    *base += pos.len() as u64;
    Ok(())
}

#[derive(Clone, Copy)]
pub struct Bounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub n: u64,
}

impl Bounds {
    pub fn new() -> Bounds {
        Bounds { min: [f32::INFINITY; 3], max: [f32::NEG_INFINITY; 3], n: 0 }
    }
    pub fn add(&mut self, p: [f32; 3]) {
        if p.iter().all(|v| v.is_finite()) {
            for i in 0..3 {
                self.min[i] = self.min[i].min(p[i]);
                self.max[i] = self.max[i].max(p[i]);
            }
            self.n += 1;
        }
    }
    pub fn show(&self) -> String {
        if self.n == 0 {
            return "empty".into();
        }
        let s: [f32; 3] = std::array::from_fn(|i| self.max[i] - self.min[i]);
        format!("[{:.1} {:.1} {:.1}]..[{:.1} {:.1} {:.1}] size {:.1} x {:.1} x {:.1}", self.min[0], self.min[1], self.min[2], self.max[0], self.max[1], self.max[2], s[0], s[1], s[2])
    }
}
