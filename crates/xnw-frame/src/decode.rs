//! The geometry decoders: the vertex header (`Streams`), meshlet and indexed
//! geometry to `Mesh`, meshlets to vertex-buffer indices. Moved verbatim from
//! apps/xnw/src/worldprobe.rs (59afddb).

use std::collections::HashMap;

use crate::le32;

/// FNV-1a 64 (never 0, so 0 can mean "not hashed").
pub fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xCBF2_9CE4_8422_2325u64, |h, x| (h ^ u64::from(*x)).wrapping_mul(0x100_0000_01B3)).max(1)
}

/// A vertex buffer's [V over 8,765 programs and 3,449 blobs]
/// header is filled: word 0 = 0x4042 (Q24 positions, axis mode <= 2 and a
/// non-zero scale) or 0x3063 (float3 positions).
pub fn vb_ready(vb: &[u8]) -> bool {
    match le32(vb, 0) {
        0x4042 => le32(vb, 0x6C) <= 2 && (0..3).any(|k| f32::from_bits(le32(vb, 0x70 + 4 * k)) != 0.0),
        0x3063 => true,
        _ => false,
    }
}

/// What a vertex header (the first 0x80 bytes of the vertex buffer) says.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Streams {
    pub pos: Stream,
    pub normal: Stream,
    pub uv: Stream,
    pub bias: [f32; 3],
    pub scale: [f32; 3],
    /// Which component got 16 bits: 0 (x), 1 (y), else (z).
    pub mode: u32,
    pub uv_scale: f32,
    /// Blend weights (slots 6/7, UNORM8x4) and indices (slots 8/9,
    /// UINT8x4 or RGBA32_UINT) [V tables, real blobs].
    pub weights: [Stream; 2],
    pub indices: [Stream; 2],
}

/// A stream word: stride `(w >> 5 & 15) * 4`, offset `(w >> 9) * 4`, format `w & 15`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stream {
    pub stride: u32,
    pub offset: u32,
    pub format: u32,
}

impl Stream {
    pub fn parse(w: u32) -> Stream {
        Stream { stride: w >> 3 & 0x3C, offset: w >> 7 & 0x1FF_FFFC, format: w & 15 }
    }
    fn at(self, v: u32) -> usize {
        (self.stride * v + self.offset) as usize
    }
}

impl Streams {
    /// `MS_VS_TransformD_aa739e09`: dword 0 position, 8 normal, 0xC uv0;
    /// bias 0x60..0x6B, mode 0x6C, scale 0x70..0x7B, uv scale 0x7C.
    pub fn parse(vb: &[u8]) -> Option<Streams> {
        if vb.len() < 0x80 {
            return None;
        }
        let f = |at: usize| f32::from_bits(le32(vb, at));
        Some(Streams {
            pos: Stream::parse(le32(vb, 0)),
            normal: Stream::parse(le32(vb, 8)),
            uv: Stream::parse(le32(vb, 0xC)),
            bias: [f(0x60), f(0x64), f(0x68)],
            mode: le32(vb, 0x6C),
            scale: [f(0x70), f(0x74), f(0x78)],
            uv_scale: f(0x7C),
            weights: [Stream::parse(le32(vb, 0x18)), Stream::parse(le32(vb, 0x1C))],
            indices: [Stream::parse(le32(vb, 0x20)), Stream::parse(le32(vb, 0x24))],
        })
    }

    /// Position: two dwords A, B: c0 = A >> 8, c1 = A & 0xFF | (B & 0xFFFF) << 8,
    /// c2 = B >> 16, each `c * scale + bias`; mode 0 gives (c2, c1, c0), 1
    /// (c1, c2, c0), else (c0, c1, c2).
    pub fn position(&self, vb: &[u8], v: u32) -> Option<[f32; 3]> {
        let at = self.pos.at(v);
        // Format 3 is three floats (stride 12, the stream starting at
        // 0x60, so no scale/bias block): 36 of 537 geometries of one capture
        // frame 1700 (characters) decoded to 1e8 as packed [I from the data].
        if self.pos.format == 3 {
            let f = |k: usize| vb.get(at + 4 * k..at + 4 * k + 4).map(|s| f32::from_le_bytes(s.try_into().unwrap()));
            return Some([f(0)?, f(1)?, f(2)?]);
        }
        let (a, b) = (u32::from_le_bytes(vb.get(at..at + 4)?.try_into().ok()?), u32::from_le_bytes(vb.get(at + 4..at + 8)?.try_into().ok()?));
        let raw = [a >> 8, a & 0xFF | (b & 0xFFFF) << 8, b >> 16];
        let c: [f32; 3] = std::array::from_fn(|i| (raw[i] as f32).mul_add(self.scale[i], self.bias[i]));
        Some(match self.mode {
            0 => [c[2], c[1], c[0]],
            1 => [c[1], c[2], c[0]],
            _ => c,
        })
    }

    /// Normal: one dword. Code 7 (RGB10A2_UNORM): xyz * 2 - 1 [I the
    /// signed mapping]; else octahedral, sign bit set: 10:10 bits, else two bytes.
    pub fn normal(&self, vb: &[u8], v: u32) -> Option<[f32; 3]> {
        let at = self.normal.at(v);
        let w = u32::from_le_bytes(vb.get(at..at + 4)?.try_into().ok()?);
        // A format-3 normal is three floats [V].
        if self.normal.format == 3 {
            let f = |o: usize| vb.get(at + o..at + o + 4).map(|b| f32::from_le_bytes(b.try_into().unwrap()));
            let n = [f(0)?, f(4)?, f(8)?];
            let r = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-6).recip();
            return Some(n.map(|x| x * r));
        }
        if self.normal.format == 7 {
            let c = |s: u32| (w >> s & 0x3FF) as f32 / 1023.0 * 2.0 - 1.0;
            let n = [c(0), c(10), c(20)];
            let r = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-6).recip();
            return Some(n.map(|x| x * r));
        }
        let (ex, ey) = if w >> 31 != 0 {
            let k = f32::from_bits(0x3B00_2008);
            ((w & 0x3FF) as f32 * k - 1.0, (w >> 10 & 0x3FF) as f32 * k - 1.0)
        } else {
            let k = f32::from_bits(0x3C00_8081);
            ((w & 0xFF) as f32 * k - 1.0, (w >> 8 & 0xFF) as f32 * k - 1.0)
        };
        let z = 1.0 - ex.abs() - ey.abs();
        let t = (-z).clamp(0.0, 1.0);
        let x = ex + if ex >= 0.0 { -t } else { t };
        let y = ey + if ey >= 0.0 { -t } else { t };
        let r = (x * x + y * y + z * z).sqrt().recip();
        Some([x * r, y * r, z * r])
    }

    /// Up to 8 (palette index, weight) pairs: slots 8/9 indices, 6/7 weights
    /// (UNORM8x4: weights / 255). None without a first set.
    pub fn skin(&self, vb: &[u8], v: u32) -> Option<([u8; 8], [u8; 8])> {
        if self.weights[0].stride == 0 || self.indices[0].stride == 0 {
            return None;
        }
        let (mut idx, mut w) = ([0u8; 8], [0u8; 8]);
        for k in 0..2 {
            let (ws, is) = (self.weights[k], self.indices[k]);
            if ws.stride == 0 || is.stride == 0 {
                continue;
            }
            w[4 * k..4 * k + 4].copy_from_slice(vb.get(ws.at(v)..ws.at(v) + 4)?);
            let at = is.at(v);
            if is.format == 12 {
                for j in 0..4 {
                    idx[4 * k + j] = *vb.get(at + 4 * j)?;
                }
            } else {
                idx[4 * k..4 * k + 4].copy_from_slice(vb.get(at..at + 4)?);
            }
        }
        Some((idx, w))
    }

    /// UV0: format 2 two floats, else 16_16 snorm times the uv scale.
    pub fn uv(&self, vb: &[u8], v: u32) -> Option<[f32; 2]> {
        let at = self.uv.at(v);
        let a = u32::from_le_bytes(vb.get(at..at + 4)?.try_into().ok()?);
        let b = u32::from_le_bytes(vb.get(at + 4..at + 8).map_or([0; 4], |s| s.try_into().unwrap()));
        if self.uv.format == 2 || self.uv.format == 3 {
            return Some([f32::from_bits(a), f32::from_bits(b)]);
        }
        let sn = |h: u32| ((h as u16 as i16) as f32 / 32767.0).max(-1.0) * self.uv_scale;
        Some([sn(a), sn(a >> 16)])
    }
}

#[derive(Default, Debug)]
pub struct Mesh {
    pub pos: Vec<[f32; 3]>,
    pub normal: Vec<[f32; 3]>,
    pub uv: Vec<[f32; 2]>,
    pub tris: Vec<[u32; 3]>,
    pub meshlets: u32,
    pub culled: u32,
    /// Per vertex (palette indices, weights) when the buffer carries them.
    pub skin: Vec<([u8; 8], [u8; 8])>,
}

/// One 48-byte palette matrix (3 float4 rows: x' = row0 . (p, 1)) applied
/// to a point [I row order, 3x4 rows].
pub fn apply34(m: &[f32], p: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|r| m[4 * r] * p[0] + m[4 * r + 1] * p[1] + m[4 * r + 2] * p[2] + m[4 * r + 3])
}

impl Mesh {
    /// The skinned copy of this mesh: positions blended over the palette
    /// (48 B per bone) by the per-vertex weights; normals by the same
    /// matrices without translation. None without skin data.
    pub fn skinned(&self, palette: &[f32]) -> Option<Mesh> {
        if self.skin.len() != self.pos.len() || palette.len() < 12 {
            return None;
        }
        let bones = palette.len() / 12;
        let mut m = Mesh { normal: self.normal.clone(), uv: self.uv.clone(), tris: self.tris.clone(), meshlets: self.meshlets, culled: self.culled, ..Default::default() };
        for (i, p) in self.pos.iter().enumerate() {
            let (idx, w) = self.skin[i];
            let (mut acc, mut nacc, mut total) = ([0f32; 3], [0f32; 3], 0f32);
            for k in 0..8 {
                let (b, wt) = (usize::from(idx[k]), f32::from(w[k]) / 255.0);
                if wt == 0.0 || b >= bones {
                    continue;
                }
                let mat = &palette[12 * b..12 * b + 12];
                let q = apply34(mat, *p);
                let n = self.normal.get(i).map(|n| apply34(mat, *n)).unwrap_or_default();
                for j in 0..3 {
                    acc[j] += wt * q[j];
                    nacc[j] += wt * (n[j] - mat[4 * j + 3]);
                }
                total += wt;
            }
            if total <= 0.0 {
                m.pos.push(*p);
                continue;
            }
            m.pos.push(acc.map(|v| v / total));
            if let Some(n) = m.normal.get_mut(i) {
                let l = (nacc[0] * nacc[0] + nacc[1] * nacc[1] + nacc[2] * nacc[2]).sqrt().max(1e-6);
                *n = nacc.map(|v| v / l);
            }
        }
        Some(m)
    }
}

impl Mesh {
    /// Adds vertex `v` of the buffer once; its index in the mesh.
    fn vertex(&mut self, map: &mut HashMap<u32, u32>, s: &Streams, vb: &[u8], v: u32) -> Result<u32, String> {
        if let Some(i) = map.get(&v) {
            return Ok(*i);
        }
        let p = s.position(vb, v).ok_or_else(|| format!("vertex {v}: position past the buffer ({} B)", vb.len()))?;
        let i = self.pos.len() as u32;
        self.pos.push(p);
        if let Some(n) = s.normal(vb, v) {
            self.normal.push(n);
        }
        if let Some(t) = s.uv(vb, v) {
            self.uv.push(t);
        }
        if let Some(k) = s.skin(vb, v) {
            self.skin.push(k);
        }
        map.insert(v, i);
        Ok(i)
    }

    fn finish(mut self) -> Mesh {
        if self.normal.len() != self.pos.len() {
            self.normal.clear();
        }
        if self.uv.len() != self.pos.len() {
            self.uv.clear();
        }
        if self.skin.len() != self.pos.len() {
            self.skin.clear();
        }
        self
    }
}

/// The meshlet buffer as `MS_VS_TransformD_aa739e09` reads it: dword 0 the
/// meshlet count (& 0x1FFFF), dword 1 the primitive words in all; meshlet
/// `m`'s header at `12 + 16m`: h0 bits 0-6 vertices - 1, 7-13 primitives - 1,
/// bit 14 / 15 bit 16 of the first vertex / primitive, h1 low / high 16 bits
/// the first vertex / primitive. Primitive words at `12 + 16 count + 4 i`
/// (three 7-bit local indices; one at or past the vertex count culls it),
/// then 16-bit vertex indices (into the vertex buffer) at
/// `12 + 16 count + 4 prims + 2 i`.
pub fn decode_meshlets(blob: &[u8], vb: &[u8]) -> Result<Mesh, String> {
    let rd = |at: usize| blob.get(at..at + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap())).ok_or_else(|| format!("meshlet data: read at {at:#x} past {:#x}", blob.len()));
    let s = Streams::parse(vb).ok_or("vertex buffer shorter than its 0x80-byte header")?;
    let count = rd(0)? & 0x1_FFFF;
    let prims = rd(4)? as usize;
    let prim_base = 12 + 16 * count as usize;
    let index_base = prim_base + 4 * prims;
    let mut m = Mesh { meshlets: count, ..Default::default() };
    let mut map = HashMap::new();
    let mut local = [0u32; 128];
    for k in 0..count as usize {
        let (h0, h1) = (rd(12 + 16 * k)?, rd(16 + 16 * k)?);
        let nv = (h0 & 0x7F) + 1;
        let np = (h0 >> 7 & 0x7F) + 1;
        let first_v = (h1 & 0xFFFF | (h0 >> 14 & 1) << 16) as usize;
        let first_p = (h1 >> 16 | (h0 >> 15 & 1) << 16) as usize;
        for t in 0..nv as usize {
            let at = index_base + 2 * (first_v + t);
            let v = blob.get(at..at + 2).map(|s| u16::from_le_bytes(s.try_into().unwrap())).ok_or_else(|| format!("meshlet {k}: vertex index at {at:#x} past {:#x}", blob.len()))?;
            local[t] = m.vertex(&mut map, &s, vb, u32::from(v))?;
        }
        for p in 0..np as usize {
            let w = rd(prim_base + 4 * (first_p + p))?;
            let i = [w & 0x7F, w >> 7 & 0x7F, w >> 14 & 0x7F];
            if i.iter().any(|i| *i >= nv) {
                m.culled += 1;
                continue;
            }
            m.tris.push(i.map(|i| local[i as usize]));
        }
    }
    Ok(m.finish())
}

/// A classic indexed geometry (type-1 programs): indices of `elem` bytes
/// (2 or 4) into the same vertex layout [I: the layout is the mesh
/// programs'].
pub fn decode_indexed(ib: &[u8], elem: u32, vb: &[u8]) -> Result<Mesh, String> {
    let s = Streams::parse(vb).ok_or("vertex buffer shorter than its 0x80-byte header")?;
    let idx: Vec<u32> = match elem {
        2 => ib.chunks_exact(2).map(|c| u32::from(u16::from_le_bytes([c[0], c[1]]))).collect(),
        4 => ib.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect(),
        e => return Err(format!("index size {e}")),
    };
    let mut m = Mesh::default();
    let mut map = HashMap::new();
    for t in idx.chunks_exact(3) {
        let a = [m.vertex(&mut map, &s, vb, t[0])?, m.vertex(&mut map, &s, vb, t[1])?, m.vertex(&mut map, &s, vb, t[2])?];
        m.tris.push(a);
    }
    Ok(m.finish())
}

/// `p * M` with M's rows at `m[0..4]`, `m[4..8]`, ...: x row0 + y row1 + z row2 + row3.
pub fn transform(m: &[f32], p: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|j| p[0].mul_add(m[j], p[1].mul_add(m[4 + j], p[2].mul_add(m[8 + j], m[12 + j]))))
}

/// The meshlets of a blob as vertex-buffer indices, three per live
/// triangle (the layout `decode_meshlets` reads; culled triangles left out).
pub fn expand_meshlets(blob: &[u8]) -> Result<Vec<u32>, String> {
    let rd = |at: usize| blob.get(at..at + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap())).ok_or_else(|| format!("meshlet data: read at {at:#x} past {:#x}", blob.len()));
    let count = rd(0)? & 0x1_FFFF;
    let prims = rd(4)? as usize;
    let prim_base = 12 + 16 * count as usize;
    let index_base = prim_base + 4 * prims;
    let mut out = Vec::new();
    for k in 0..count as usize {
        let (h0, h1) = (rd(12 + 16 * k)?, rd(16 + 16 * k)?);
        let nv = (h0 & 0x7F) + 1;
        let np = (h0 >> 7 & 0x7F) + 1;
        let first_v = (h1 & 0xFFFF | (h0 >> 14 & 1) << 16) as usize;
        let first_p = (h1 >> 16 | (h0 >> 15 & 1) << 16) as usize;
        let local = |i: u32| -> Result<u32, String> {
            let at = index_base + 2 * (first_v + i as usize);
            blob.get(at..at + 2).map(|s| u32::from(u16::from_le_bytes(s.try_into().unwrap()))).ok_or_else(|| format!("meshlet {k}: vertex index at {at:#x} past {:#x}", blob.len()))
        };
        for p in 0..np as usize {
            let w = rd(prim_base + 4 * (first_p + p))?;
            let i = [w & 0x7F, w >> 7 & 0x7F, w >> 14 & 0x7F];
            if i.iter().any(|i| *i >= nv) {
                continue;
            }
            for x in i {
                out.push(local(x)?);
            }
        }
    }
    Ok(out)
}

/// The largest vertex count of one meshlet in a blob (upstream check).
pub fn max_meshlet_vertices(blob: &[u8]) -> u32 {
    let count = blob.get(0..4).map_or(0, |s| u32::from_le_bytes(s.try_into().unwrap()) & 0x1_FFFF) as usize;
    (0..count).filter_map(|k| blob.get(12 + 16 * k..16 + 16 * k)).map(|s| (u32::from_le_bytes(s.try_into().unwrap()) & 0x7F) + 1).max().unwrap_or(0)
}
