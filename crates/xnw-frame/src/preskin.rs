//! Whole peds in the replay from a capture's pre-skin dispatches. The
//! console skins a ped's vertices once a frame in a compute pass and its
//! PreSkinD draws read the result; a capture holds each dispatch's inputs
//! and each draw's range word, not the result. Here the CPU reference
//! (`xnw_world::preskin::cpu`) runs over each recorded
//! dispatch, its output range is decoded the way the PreSkinD mesh program
//! decodes it, and every joined list row gets a copy of its vertex buffer
//! whose position (and normal) streams are those floats and whose skin
//! streams are off, so the pass draws it without skinning it again.

use std::collections::HashMap;
use std::path::Path;

use xnw_formats::framepack::kind;
use xnw_world::preskin::{self as ps, PosFormat};
use xnw_world::{FrameList, GeomKey};

use crate::le32;
use crate::pack::{Pack, PreskinDispatch};

/// One dispatch's inputs as owned bytes ([`Loaded::inputs`] borrows them).
#[derive(Clone, Debug, Default)]
pub struct Loaded {
    pub b7: Vec<u8>,
    pub vb: Vec<u8>,
    pub remap: Vec<u32>,
    pub palette: Vec<u8>,
    pub t7: Vec<u8>,
    pub t71: Vec<u8>,
    pub t78: Vec<u8>,
    /// The second table's views a variant may read: constant buffers b5 and
    /// b6, buffers t73 to t76 (empty: not bound or not captured).
    pub b5: Vec<u8>,
    pub b6: Vec<u8>,
    pub t73: Vec<u8>,
    pub t74: Vec<u8>,
    pub t75: Vec<u8>,
    pub t76: Vec<u8>,
}

impl Loaded {
    pub fn inputs(&self) -> ps::Inputs<'_> {
        ps::Inputs { b7: &self.b7, vb: &self.vb, remap: &self.remap, palette: &self.palette, t7: &self.t7, t71: &self.t71, t78: &self.t78, b5: &self.b5, b6: &self.b6, t73: &self.t73, t74: &self.t74, t75: &self.t75, t76: &self.t76 }
    }
}

/// The palette a dispatch reads, cut to the rows a capture holds: `cur` and
/// `prev` are the rows from the constant buffer's two base rows (+0x28 this
/// frame, +0x2c the previous one). Returns the rows back to back and the
/// constant buffer with the bases pointing into them (0 and `cur`'s rows).
pub fn palette_rebase(b7: &[u8], cur: &[u8], prev: Option<&[u8]>) -> (Vec<u8>, Vec<u8>) {
    let mut b7 = b7.to_vec();
    b7.resize(b7.len().max(0x30), 0);
    let rows = (cur.len() / 48) as u32;
    let mut palette = cur[..rows as usize * 48].to_vec();
    b7[0x28..0x2c].copy_from_slice(&0u32.to_le_bytes());
    match prev {
        Some(p) => {
            palette.extend_from_slice(p);
            b7[0x2c..0x30].copy_from_slice(&rows.to_le_bytes());
        }
        None => b7[0x2c..0x30].copy_from_slice(&0u32.to_le_bytes()),
    }
    (b7, palette)
}

/// The bone remap's elements: 16-bit when the view says 2 bytes each (or
/// says nothing), else 32-bit.
pub fn remap_elements(bytes: &[u8], element: u64) -> Vec<u32> {
    if element == 4 {
        bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
    } else {
        bytes.chunks_exact(2).map(|c| u32::from(u16::from_le_bytes([c[0], c[1]]))).collect()
    }
}

/// One vertex of an output range as the PreSkinD mesh program decodes it:
/// 10:10:10 `q * 0x3bc0300c - 3`; 24:24:16 x and y `q * 0x34c00001 - 3`,
/// z `q * 0x38c000c0 - 3`; float3 as it is. A hidden vertex (all ones) is
/// NaN, so its triangles are not drawn.
pub fn range_position(b: &[u8], format: PosFormat) -> [f32; 3] {
    let f = |bits: u32| f32::from_bits(bits);
    match format {
        PosFormat::Float3 => {
            let d = [le32(b, 0), le32(b, 4), le32(b, 8)];
            if d == [u32::MAX; 3] { [f32::NAN; 3] } else { d.map(f32::from_bits) }
        }
        PosFormat::Packed10 => {
            let d = le32(b, 0);
            if d == u32::MAX {
                return [f32::NAN; 3];
            }
            [d & 0x3ff, d >> 10 & 0x3ff, d >> 20 & 0x3ff].map(|q| q as f32 * f(0x3bc0_300c) - 3.0)
        }
        PosFormat::Packed24 => {
            let (d0, d1) = (le32(b, 0), le32(b, 4));
            if d0 & d1 == u32::MAX {
                return [f32::NAN; 3];
            }
            let (x, y, z) = (d0 >> 8, d0 & 0xff | (d1 & 0xffff) << 8, d1 >> 16);
            [x as f32 * f(0x34c0_0001) - 3.0, y as f32 * f(0x34c0_0001) - 3.0, z as f32 * f(0x38c0_00c0) - 3.0]
        }
    }
}

/// The normal of an extra block's frame dword: octahedral 8:8 in bits
/// 0-15, or 10:10 in bits 0-19 when bit 31 is set.
pub fn range_normal(word: u32) -> [f32; 3] {
    let (x, y) = if word >> 31 != 0 {
        let u = |v: u32| v as f32 / 1023.0 * 2.0 - 1.0;
        (u(word & 0x3ff), u(word >> 10 & 0x3ff))
    } else {
        let u = |v: u32| v as f32 / 255.0 * 2.0 - 1.0;
        (u(word & 0xff), u(word >> 8 & 0xff))
    };
    let z = 1.0 - x.abs() - y.abs();
    let t = (-z).clamp(0.0, 1.0);
    let (x, y) = (x + if x >= 0.0 { -t } else { t }, y + if y >= 0.0 { -t } else { t });
    let l = (x * x + y * y + z * z).sqrt().max(1e-9);
    [x / l, y / l, z / l]
}

/// A written range (header, positions, extra blocks) decoded: positions,
/// and normals when the range is a primary one.
pub fn range_decode(bytes: &[u8], word: u32) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let (_, _, format, primary) = ps::range(word);
    let (ps_, es) = ps::strides(format, primary);
    let n = le32(bytes, 0);
    let at = |v: u32| (32 + v * ps_) as usize;
    let pos = (0..n).map(|v| range_position(bytes.get(at(v)..).unwrap_or(&[]), format)).collect();
    let normals = if primary { (0..n).map(|v| range_normal(le32(bytes, (32 + n * ps_ + v * es) as usize))).collect() } else { Vec::new() };
    (pos, normals)
}

/// A copy of vertex buffer `vb` whose position stream is `pos` as float3
/// (stream word 0: format 3, stride 12) appended to it, whose normal stream
/// is `normals` the same way when there are any (word 8), and whose skin
/// streams (words 0x18-0x24) are off. None: no stream header, or the
/// appended streams lie past what a stream word can address.
pub fn with_positions(vb: &[u8], pos: &[[f32; 3]], normals: &[[f32; 3]]) -> Option<Vec<u8>> {
    if vb.len() < 0x80 {
        return None;
    }
    let mut out = vb.to_vec();
    out.resize(out.len().div_ceil(4) * 4, 0);
    let put = |out: &mut Vec<u8>, at: usize, v: &[[f32; 3]]| -> Option<()> {
        let off = out.len() as u32;
        if u64::from(off) + 12 * v.len() as u64 > 0x1FF_FFFC {
            return None;
        }
        out.extend(v.iter().flatten().flat_map(|f| f.to_le_bytes()));
        out[at..at + 4].copy_from_slice(&(3 | 3 << 5 | (off / 4) << 9).to_le_bytes());
        Some(())
    };
    put(&mut out, 0, pos)?;
    if !normals.is_empty() {
        put(&mut out, 8, normals)?;
    }
    out[0x18..0x28].fill(0);
    Some(out)
}

/// `CS_PedStandardSkinning_<n>`'s `n` from a program name.
pub fn variant_of(name: &str) -> Option<u32> {
    let rest = name.split("CS_PedStandardSkinning_").nth(1)?;
    rest.split('_').next()?.parse().ok()
}

/// The inputs of dispatch `d` from the pack. The palette buffer itself is
/// larger than a capture stores; its rows come from the draw-time palette
/// copy that starts at the palette view's address + 48 x the base row.
pub fn load(pack: &Pack, d: &PreskinDispatch) -> Result<Loaded, String> {
    let heap = pack.heap_views(d.sink);
    let hv = |k: u16| heap.iter().find(|h| h.k == k).ok_or_else(|| format!("no heap view {k}"));
    let vb = hv(0)?.bytes.clone().map_err(|r| format!("vertex view: reason {r}"))?;
    let rv = hv(1)?;
    let remap_bytes = rv.bytes.clone().map_err(|r| format!("remap view: reason {r}"))?;
    let element = 2;
    let table = |slot: u32| pack.slot_bytes(d.sink, 0, slot).map(|x| x.1).unwrap_or_default();
    let pal_view = pack.tables(d.sink).into_iter().find(|t| t.0 == 0).and_then(|t| t.2.into_iter().find(|v| v.slot == 1)).ok_or("no palette view (root 0 slot 1)")?;
    let rows = |base: u32| -> Option<Vec<u8>> {
        let at = pal_view.addr + 48 * u64::from(base);
        let r = pack.data.refs.iter().filter(|r| r.kind == kind::PALETTE && r.addr <= at && at < r.addr + u64::from(r.len)).max_by_key(|r| r.addr + u64::from(r.len) - at)?;
        let b = pack.blob(r.hash)?;
        Some(b.get((at - r.addr) as usize..)?.to_vec())
    };
    let (cur_base, prev_base) = (le32(&d.rec.b7, 0x28), le32(&d.rec.b7, 0x2c));
    let cur = match pack.slot_bytes(d.sink, 0, 1) {
        Ok((_, whole)) => whole.get(48 * cur_base as usize..).map(<[u8]>::to_vec),
        Err(_) => rows(cur_base),
    }
    .ok_or_else(|| format!("no palette rows at {:#x} + 48 x {cur_base:#x}", pal_view.addr))?;
    let prev = (prev_base != cur_base).then(|| rows(prev_base)).flatten();
    let (b7, palette) = palette_rebase(&d.rec.b7, &cur, prev.as_deref());
    // HEAD1 (`headgraph=on`): t74/t75 from the head graph's table where it filled it (t75 = t74: one frame).
    let second = |slot: u32| match pack.slot_bytes(d.sink, 1, slot) {
        Ok((v, b)) => if matches!(slot, 0 | 3) { crate::head_graph::table_for(v.addr).unwrap_or(b) } else { b },
        Err(_) => Vec::new(),
    };
    Ok(Loaded { b7, vb, remap: remap_elements(&remap_bytes, element), palette, t7: table(9), t71: table(0), t78: table(6), b5: second(7), b6: second(9), t73: second(1), t74: second(0), t75: second(3), t76: second(5) })
}

/// What [`apply`] did.
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// List rows of a PreSkinD program; those joined to a dispatch.
    pub rows: usize,
    pub joined: usize,
    /// Rows now drawn from the pre-skin, by variant; rows left as they were, by reason.
    pub drawn: std::collections::BTreeMap<u32, usize>,
    /// The rows drawn from the pre-skin: (list row, variant).
    pub row_variants: Vec<(usize, u32)>,
    pub left: std::collections::BTreeMap<String, usize>,
    /// Dispatches whose position table is unfilled in the capture (a buffer
    /// the console fills on the GPU): run without the table step, so their
    /// rows show the bind shape under the bones.
    pub unfilled: usize,
    /// Dispatches run, their vertices and the reference's time.
    pub dispatches: usize,
    pub vertices: u64,
    pub cpu_ms: f64,
}

/// Which variants [`apply`] runs: the ones the reference covers, or also
/// the others through variant 0's maths (a diagnostic: wrong for them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    Covered,
    AllAsZero,
    /// Variants 8 and 9 (4 and 5 plus a map-driven displacement) through
    /// 4's and 5's maths, and variant 11 without its tables (the bind
    /// positions under the bones): a diagnostic, those steps are missing.
    Base,
}

/// Replaces the vertex data of every PreSkinD row of `list` that joins a
/// recorded dispatch (see the module text). `dump`: that many dispatches
/// print their stream descriptors and first vertices.
pub fn apply(dir: &Path, list: &mut FrameList, geoms: &mut HashMap<GeomKey, (Vec<u8>, Vec<u32>)>, which: Which, dump: usize) -> Result<Report, String> {
    let Some(pack) = Pack::open(dir)? else { return Err("pre-skin: no pack".into()) };
    let names = crate::audit::program_names(dir).ok_or_else(|| format!("pre-skin: no shader-names.tsv above {}", dir.display()))?;
    let v1 = pack.list_draws().filter(|v| v.len() == list.draws.len()).ok_or("pre-skin: no list-draws.tsv covering the list")?;
    let joined = pack.preskin_join(&|h| names.get(&h).is_some_and(|n| n.contains("PreSkinD")));
    let by_draw: HashMap<u32, (u32, Option<u32>)> = joined.iter().filter_map(|j| Some((j.draw?, (j.word?, j.dispatch)))).collect();
    let dispatches: HashMap<u32, PreskinDispatch> = pack.dispatches().into_iter().map(|d| (d.sink, d)).collect();
    let mut rep = Report::default();
    // Per dispatch: (variant, decoded positions and normals) or why not.
    let mut ran: HashMap<u32, Result<(u32, Vec<[f32; 3]>, Vec<[f32; 3]>), String>> = HashMap::new();
    let mut made: HashMap<(GeomKey, u32), GeomKey> = HashMap::new();
    let mut dumped = 0usize;
    for i in 0..list.draws.len() {
        let Some(&(word, dispatch)) = by_draw.get(&v1[i]) else { continue };
        rep.rows += 1;
        let Some(sink) = dispatch else {
            *rep.left.entry("no dispatch".into()).or_default() += 1;
            continue;
        };
        rep.joined += 1;
        let d = &dispatches[&sink];
        let res = ran.entry(sink).or_insert_with(|| {
            let variant = names.get(&d.program).and_then(|n| variant_of(n)).ok_or_else(|| format!("program {:032x} is no standard variant", d.program))?;
            let (rule, mut before) = match (which, ps::rule_of(variant, &d.rec.b7)) {
                (Which::Base, Some((r, _))) => (r, ps::Before::Nothing),
                (_, Some(r)) => r,
                (Which::AllAsZero, None) => (ps::Bones::Four, ps::Before::Nothing),
                (Which::Base, None) if ps::rule_more(variant).is_some() => ps::rule_more(variant).map_or((ps::Bones::Four, ps::Before::Nothing), |(r, _)| (r, ps::Before::Nothing)),
                (Which::Covered | Which::Base, None) => return Err(format!("variant {variant} not covered")),
            };
            let inp = load(&pack, d)?;
            if before == ps::Before::Tables && ps::table_unfilled(&inp.inputs()) {
                before = ps::Before::Nothing;
                rep.unfilled += 1;
            }
            let t = std::time::Instant::now();
            let w = ps::cpu_rule(&inp.inputs(), rule, before);
            rep.cpu_ms += t.elapsed().as_secs_f64() * 1e3;
            rep.dispatches += 1;
            rep.vertices += u64::from(d.rec.vertices);
            let (pos, normals) = range_decode(&w.bytes, d.rec.word);
            if dumped < dump {
                dumped += 1;
                describe(&pack, d, variant, &inp, &pos);
            }
            Ok((variant, pos, normals))
        });
        let (variant, pos, normals) = match res {
            Ok(x) => x,
            Err(e) => {
                *rep.left.entry(e.clone()).or_default() += 1;
                continue;
            }
        };
        let old = list.draws[i].geom;
        let key = match made.get(&(old, word & xnw_formats::framepack::RANGE_MASK)) {
            Some(k) => *k,
            None => {
                let Some((vb, idx)) = geoms.get(&old) else {
                    *rep.left.entry("no geometry".into()).or_default() += 1;
                    continue;
                };
                let top = idx.iter().max().map_or(0, |m| m + 1);
                if top as usize > pos.len() {
                    *rep.left.entry(format!("draw reads vertex {} of {}", top - 1, pos.len())).or_default() += 1;
                    continue;
                }
                // The draw's own range word says whether it reads the extra blocks.
                let n = if ps::range(word).3 { &normals[..] } else { &[] };
                let Some(new_vb) = with_positions(vb, pos, n) else {
                    *rep.left.entry("no stream header".into()).or_default() += 1;
                    continue;
                };
                let k = GeomKey { hash: old.hash ^ 0x5052_4553_4b00_0000 ^ u64::from(word), vb_len: new_vb.len() as u64, ..old };
                let idx = idx.clone();
                geoms.insert(k, (new_vb, idx));
                made.insert((old, word & xnw_formats::framepack::RANGE_MASK), k);
                k
            }
        };
        list.draws[i].geom = key;
        *rep.drawn.entry(*variant).or_default() += 1;
        rep.row_variants.push((i, *variant));
    }
    Ok(rep)
}

/// A dispatch's own inputs, printed: the stream descriptors, the views and
/// the first vertices in and out.
fn describe(pack: &Pack, d: &PreskinDispatch, variant: u32, inp: &Loaded, pos: &[[f32; 3]]) {
    let c = |at: usize| le32(&inp.b7, at);
    println!("pre-skin dispatch sink {} variant {variant}: {} vertices, range word {:#010x}", d.sink, d.rec.vertices, d.rec.word);
    println!("  b7 streams (stride | offset << 8): pos {:#x} frame {:#x} mask {:#x} idx {:#x} wgt {:#x} idx2 {:#x} wgt2 {:#x} +0x1c {:#x} +0x50 {:#x}; sets {:#x} {:#x}; bases cur {} prev {} remap {}; t7 row {}; +0x54 {:#x} +0x58 {:#x} +0x5c {:#x}", c(0), c(4), c(8), c(0xc), c(0x10), c(0x14), c(0x18), c(0x1c), c(0x50), c(0x20), c(0x24), le32(&d.rec.b7, 0x28), le32(&d.rec.b7, 0x2c), c(0x30), c(0x3c), c(0x54), c(0x58), c(0x5c));
    println!("  views: vb {} B (words {:#x} {:#x} {:#x} {:#x}; bias {:?} mode {} scale {:?}), remap {} elements (max {}), palette {} rows, t7 {} B, t71 {} B, t78 {} B", inp.vb.len(), le32(&inp.vb, 0), le32(&inp.vb, 4), le32(&inp.vb, 8), le32(&inp.vb, 0xc), [0x60, 0x64, 0x68].map(|a| f32::from_bits(le32(&inp.vb, a))), le32(&inp.vb, 0x6c), [0x70, 0x74, 0x78].map(|a| f32::from_bits(le32(&inp.vb, a))), inp.remap.len(), inp.remap.iter().max().copied().unwrap_or(0), inp.palette.len() / 48, inp.t7.len(), inp.t71.len(), inp.t78.len());
    println!("  second table: b5 {} B, b6 {} B (dwords {:#x} {:#x} {:#x} {:#x}), t73 {} B, t74 {} B, t75 {} B, t76 {} B (dwords {:#x} {:#x})", inp.b5.len(), inp.b6.len(), le32(&inp.b6, 0), le32(&inp.b6, 4), le32(&inp.b6, 8), le32(&inp.b6, 12), inp.t73.len(), inp.t74.len(), inp.t75.len(), inp.t76.len(), le32(&inp.t76, 0), le32(&inp.t76, 4));
    for (root, _, views) in pack.tables(d.sink) {
        let slots: Vec<String> = views
            .iter()
            .map(|v| {
                let held = match pack.slot_bytes(d.sink, root, v.slot) {
                    Ok((_, b)) => format!("{} B held, {} non-zero", b.len(), b.iter().filter(|x| **x != 0).count()),
                    Err(_) => match pack.texture(v) {
                        Some(t) => format!("texture {}x{} format {} texels {} B", v.width, v.height, v.format, t.texel_bytes),
                        None => "no bytes".into(),
                    },
                };
                format!("{}: kind {} at {:#x} {} B ({held})", v.slot, v.kind, v.addr, v.bytes)
            })
            .collect();
        println!("  root {root}: {}", slots.join("; "));
    }
    for (element, index, view) in pack.displacement_maps(d.sink) {
        match view {
            Ok(v) => println!("  displacement map of element {element}: heap slot {index}, kind {} {}x{} format {} {} B, texels held {} B", v.kind, v.width, v.height, v.format, v.bytes, pack.texture(&v).map_or(0, |t| t.texel_bytes)),
            Err(r) => println!("  displacement map of element {element}: heap slot {index}, no view (reason {r})"),
        }
    }
    if let Ok((_, t90)) = pack.slot_bytes(d.sink, 0, 8) {
        println!("  root 0 slot 8 records (144 B), dword 0x8c: {:x?}", t90.chunks_exact(144).take(12).map(|r| le32(r, 0x8c)).collect::<Vec<_>>());
    }
    let s = crate::decode::Streams::parse(&inp.vb);
    let stream = |desc: u32, v: u32| ((desc >> 8) + (desc & 0xff) * v) as usize;
    if c(0xc) & 0xff == 16 {
        // A 16-byte index stream: the fourth dword's top byte is a tag (0xb2: variant 4's extra block).
        let mut tags: std::collections::BTreeMap<u32, u32> = Default::default();
        for v in 0..d.rec.vertices {
            *tags.entry(le32(&inp.vb, stream(c(0xc), v) + 12) >> 24).or_default() += 1;
        }
        println!("  index dword 3 top bytes: {}", tags.iter().map(|(t, n)| format!("{t:#04x} x {n}")).collect::<Vec<_>>().join(", "));
    }
    if !inp.t74.is_empty() {
        let base = le32(&inp.b6, 0) as usize;
        let f = |at: usize| f32::from_bits(le32(&inp.t74, base + at));
        let entries = (le32(&inp.b6, 4) as usize).saturating_sub(28) / 8;
        println!("  t74 table at {base:#x}: bias {:?} order {} scale {:?}, {entries} entries", [f(0), f(4), f(8)], le32(&inp.t74, base + 12), [f(16), f(20), f(24)]);
        let nz = |b: &[u8]| b.iter().filter(|x| **x != 0).count();
        println!("    non-zero bytes: t74 {} of {}, t75 {} of {}, t76 {} of {}", nz(&inp.t74), inp.t74.len(), nz(&inp.t75), inp.t75.len(), nz(&inp.t76), inp.t76.len());
    }
    let bbox = |pts: &mut dyn Iterator<Item = [f32; 3]>| {
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for q in pts.filter(|q| q[0].is_finite()) {
            for k in 0..3 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
            }
        }
        (lo, hi)
    };
    println!("  box: bind {:?}, out {:?}", bbox(&mut (0..d.rec.vertices).filter_map(|v| s.as_ref().and_then(|s| s.position(&inp.vb, v)))), bbox(&mut pos.iter().copied()));
    for v in 0..d.rec.vertices.min(4) {
        let bind = s.as_ref().and_then(|s| s.position(&inp.vb, v));
        let (iw, ww) = (le32(&inp.vb, stream(c(0xc), v)), le32(&inp.vb, stream(c(0x10), v)));
        let bones: Vec<u32> = (0..4).map(|k| inp.remap.get(((iw >> (8 * k) & 0xff) + c(0x30)) as usize).copied().unwrap_or(u32::MAX)).collect();
        println!("  v{v}: bind {bind:?} idx {iw:#010x} -> bones {bones:?} wgt {ww:#010x} idx2 {:#010x} wgt2 {:#010x} -> out {:?}", le32(&inp.vb, stream(c(0x14), v)), le32(&inp.vb, stream(c(0x18), v)), pos.get(v as usize));
    }
}

/// `--preskin` (the covered variants), `--preskin=base` or `--preskin=all0` (diagnostics)
/// on a loaded list; `--preskin-dump N` prints N dispatches. Prints one
/// line. Without the option the list is untouched.
pub fn preskin_arg(args: &[String], dir: &Path, list: &mut FrameList, geoms: &mut HashMap<GeomKey, (Vec<u8>, Vec<u32>)>) -> Result<Option<Report>, String> {
    let which = match args.iter().find_map(|a| a.strip_prefix("--preskin").filter(|r| r.is_empty() || r.starts_with('='))) {
        Some("") => Which::Covered,
        Some("=all0") => Which::AllAsZero,
        Some("=base") => Which::Base,
        Some(o) => return Err(format!("--preskin{o}: use --preskin, --preskin=base or --preskin=all0")),
        None => return Ok(None),
    };
    let dump = args.iter().position(|a| a == "--preskin-dump").and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(0);
    let r = apply(dir, list, geoms, which, dump)?;
    let per = |m: &std::collections::BTreeMap<u32, usize>| m.iter().map(|(v, n)| format!("v{v} {n}")).collect::<Vec<_>>().join(", ");
    let left = r.left.iter().map(|(why, n)| format!("{n} {why}")).collect::<Vec<_>>().join("; ");
    println!("pre-skin: {} PreSkinD rows, {} joined to a dispatch; drawn from the pre-skin {} ({}); left as they were: {}; {} dispatches ({} without their unfilled position table), {} vertices, reference {:.2} ms", r.rows, r.joined, r.drawn.values().sum::<usize>(), per(&r.drawn), if left.is_empty() { "none".into() } else { left }, r.dispatches, r.unfilled, r.vertices, r.cpu_ms);
    if dump > 0 {
        println!("pre-skin rows (list row:variant): {}", r.row_variants.iter().map(|(i, v)| format!("{i}:{v}")).collect::<Vec<_>>().join(" "));
    }
    Ok(Some(r))
}

/// `preskin-oracle <frame dir> --pack <shader pack> [--index <pack index>]`:
/// every recorded dispatch of the frame through the title's own program,
/// translated, on the adapter `XNW_ADAPTER` names (the software one for
/// this), against the CPU reference: per dispatch the largest position
/// difference, the first vertex past `EPS` and how many extra blocks
/// differ; then a line per variant. Variants 8 and 9 go through 4's and
/// 5's maths, so their line is the size of what the reference lacks.
pub fn oracle(args: &[String]) -> Result<(), String> {
    const EPS: f32 = 2e-4;
    let dir = Path::new(args.first().ok_or("preskin-oracle: <frame dir> --pack <shader pack> [--index <pack index>]")?);
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1));
    let pack_path = Path::new(opt("--pack").ok_or("preskin-oracle: --pack <shader pack>")?);
    let index = opt("--index").map(Path::new);
    let Some(pack) = Pack::open(dir)? else { return Err("preskin-oracle: no pack".into()) };
    let names = crate::audit::program_names(dir).ok_or("preskin-oracle: no shader-names.tsv above the frame")?;
    let host = xnw_gpu::host::Host::new(crate::replay::hw_agility().as_deref()).map_err(|e| format!("preskin-oracle: device: {e}"))?;
    let mut programs: HashMap<u128, Result<(_, u32), String>> = HashMap::new();
    // Per variant: dispatches, vertices, vertices past EPS, largest difference, frame words differing, previous positions past EPS.
    let mut per: std::collections::BTreeMap<u32, (u32, u64, u64, f32, u64, u64)> = Default::default();
    for d in pack.dispatches() {
        let Some(variant) = names.get(&d.program).and_then(|n| variant_of(n)) else { continue };
        // Variants past the covered ones by their base variant's rule (7-10 without their maps).
        let (rule, before) = ps::rule_of(variant, &d.rec.b7).or_else(|| ps::rule_more(variant)).unwrap_or((ps::Bones::Four, ps::Before::Nothing));
        let line = |text: String| println!("oracle sink {} variant {variant} ({} vertices): {text}", d.sink, d.rec.vertices);
        let mut inp = match load(&pack, &d) {
            Ok(i) => i,
            Err(e) => {
                line(format!("not run: {e}"));
                continue;
            }
        };
        // The range moves to byte 0 of its buffer; the format bits stay.
        let word = le32(&inp.b7, 0x34) & 0xff00_0000;
        inp.b7[0x34..0x38].copy_from_slice(&word.to_le_bytes());
        let cpu = ps::cpu_rule(&inp.inputs(), rule, before);
        let program = programs.entry(d.program).or_insert_with(|| {
            let c = ps::container(pack_path, index, d.program).ok_or("program not in the shader pack")?;
            let wave = ps::wave_size(&c).ok_or("no wave size")?;
            if let Some(gap) = ps::adapter_gap(&host.caps, wave) {
                return Err(gap);
            }
            Ok((ps::translate(&c)?, wave))
        });
        let (shader, wave) = match program {
            Ok(p) => p,
            Err(e) => {
                line(format!("not run: {e}"));
                continue;
            }
        };
        let len = (cpu.bytes.len() as u32 + 0x40).next_multiple_of(16);
        let remap: Vec<u8> = inp.remap.iter().flat_map(|v| (*v as u16).to_le_bytes()).collect();
        let v = |kind, bytes| ps::View { kind, bytes };
        let mut second = vec![(8, v(ps::Kind::Cbv, &inp.b7[..]))];
        for (slot, kind, bytes) in [(0, ps::Kind::Raw, &inp.t74), (1, ps::Kind::Structured(16), &inp.t73), (3, ps::Kind::Raw, &inp.t75), (5, ps::Kind::Raw, &inp.t76), (7, ps::Kind::Cbv, &inp.b5), (9, ps::Kind::Cbv, &inp.b6)] {
            if !bytes.is_empty() {
                second.push((slot, v(kind, &bytes[..])));
            }
        }
        let dispatch = ps::Dispatch {
            groups: d.rec.groups,
            tables: vec![
                (4, vec![
                    (0, v(ps::Kind::Structured(416), &inp.t71[..])),
                    (1, v(ps::Kind::Structured(48), &inp.palette[..])),
                    (6, v(ps::Kind::Structured(1280), &inp.t78[..])),
                    (9, v(ps::Kind::Structured(48), &inp.t7[..])),
                    (24, v(ps::Kind::RawUav { len }, &[])),
                    (25, v(ps::Kind::RawUav { len }, &[])),
                ]),
                (5, second),
            ],
            heap: vec![(le32(&inp.b7, 0x40), v(ps::Kind::Raw, &inp.vb[..])), (le32(&inp.b7, 0x44), v(ps::Kind::Typed { format: 57, bytes: 2 }, &remap[..]))],
            constants: Vec::new(),
            texels: Vec::new(),
        };
        let outs = ps::bind(shader, &dispatch).and_then(|b| ps::run(&host, shader, *wave, &b, dispatch.groups));
        let (outs, seconds) = match outs {
            Ok(o) => o,
            Err(e) => {
                line(format!("not run: {e}"));
                continue;
            }
        };
        let Some(got) = outs.get(usize::from(cpu.buffer)).and_then(|o| o.get(..cpu.bytes.len())) else {
            line("no output buffer".into());
            continue;
        };
        let (want_pos, _) = range_decode(&cpu.bytes, word);
        let (got_pos, _) = range_decode(got, word);
        let (mut worst, mut first, mut past) = (0f32, None, 0u64);
        for (i, (a, b)) in want_pos.iter().zip(&got_pos).enumerate() {
            let diff = if a[0].is_nan() && b[0].is_nan() { 0.0 } else { (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, |m: f32, x| if x.is_nan() { f32::INFINITY } else { m.max(x) }) };
            worst = worst.max(diff);
            if diff > EPS {
                past += 1;
                first.get_or_insert((i, *a, *b));
            }
        }
        let (_, _, format, primary) = ps::range(word);
        let (ps_, es) = ps::strides(format, primary);
        let n = d.rec.vertices;
        let block = |b: &[u8], i: u32| b.get((32 + n * ps_ + i * es) as usize..(32 + n * ps_ + (i + 1) * es) as usize).map(<[u8]>::to_vec);
        // Extra blocks: frame dwords that differ (and by how many steps of a
        // byte at most), previous positions past EPS.
        let (mut words, mut steps, mut prev_past, mut prev_worst) = (0u32, 0u32, 0u32, 0f32);
        let (mut forms, mut first_word, mut apart) = (0u32, None, 0f32);
        for i in 0..if primary { n } else { 0 } {
            let (Some(a), Some(b)) = (block(&cpu.bytes, i), block(got, i)) else { continue };
            let (wa, wb) = (le32(&a, 0), le32(&b, 0));
            if wa != wb {
                words += 1;
                forms += u32::from((wa ^ wb) >> 31 != 0);
                first_word.get_or_insert((i, wa, wb));
                let (na, nb) = (range_normal(wa), range_normal(wb));
                apart = apart.max((na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2]).clamp(-1.0, 1.0).acos().to_degrees());
                let field = |w: u32, k: u32| if k < 2 { w >> (8 * k) & 0xff } else { w >> (16 + 7 * (k - 2)) & 0x7f };
                steps = steps.max((0..4).map(|k| field(wa, k).abs_diff(field(wb, k))).max().unwrap_or(0));
            }
            let (pa, pb) = (range_position(&a[4..], format), range_position(&b[4..], format));
            let diff = if pa[0].is_nan() && pb[0].is_nan() { 0.0 } else { (0..3).map(|k| (pa[k] - pb[k]).abs()).fold(0.0, |m: f32, x| if x.is_nan() { f32::INFINITY } else { m.max(x) }) };
            prev_worst = prev_worst.max(diff);
            prev_past += u32::from(diff > EPS);
        }
        let extra = format!("{words} frame words ({forms} in the other form, normals at most {apart:.2} degrees apart, largest step {steps}, first {}), previous positions past {EPS} {prev_past} (largest {prev_worst:.6})", first_word.map_or("none".into(), |(i, a, b)| format!("v{i} reference {a:#010x} program {b:#010x}")));
        let header = got.get(..8) == cpu.bytes.get(..8);
        line(format!("{rule:?} {before:?} +0x5c {:#x}: header {}, positions past {EPS} {past} of {}, largest {worst:.6}, first {}; extra blocks differing: {extra}; {seconds:.2} s", le32(&inp.b7, 0x5c), if header { "equal" } else { "DIFFERENT" }, want_pos.len(), first.map_or("none".into(), |(i, a, b)| format!("v{i} reference {a:?} program {b:?}"))));
        let e = per.entry(variant).or_default();
        *e = (e.0 + 1, e.1 + u64::from(n), e.2 + past, e.3.max(worst), e.4 + u64::from(words), e.5 + u64::from(prev_past));
    }
    for (variant, (dispatches, vertices, past, worst, words, prev)) in per {
        println!("oracle variant {variant}: {dispatches} dispatches, {vertices} vertices, {past} past {EPS}, largest position difference {worst:.6}; frame words differing {words}, previous positions past {EPS} {prev}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() <= eps)
    }

    /// Each position format through the reference's writer and this decode:
    /// a made-up position comes back within the format's step; a hidden
    /// vertex is NaN.
    #[test]
    fn range_formats_round_trip() {
        // One vertex, one identity bone moved by (0.5, -1.25, 2): bind (0.25, 0.5, 0.125).
        let mut vb = vec![0u8; 0x100];
        vb[0x6c..0x70].copy_from_slice(&2u32.to_le_bytes());
        for (k, s) in [0.25f32, 0.5, 0.125].iter().enumerate() {
            vb[0x70 + 4 * k..0x74 + 4 * k].copy_from_slice(&s.to_le_bytes());
        }
        // Position q = (1, 1, 1) at 0x80; weights 255,0,0,0 at 0x90; indices 0 at 0x94.
        vb[0x80..0x84].copy_from_slice(&(1u32 << 8 | 1).to_le_bytes());
        vb[0x84..0x88].copy_from_slice(&(1u32 << 16).to_le_bytes());
        vb[0x90] = 255;
        let mut palette = vec![0u8; 48];
        for (r, t) in [0.5f32, -1.25, 2.0].iter().enumerate() {
            palette[16 * r + 4 * r..16 * r + 4 * r + 4].copy_from_slice(&1f32.to_le_bytes());
            palette[16 * r + 12..16 * r + 16].copy_from_slice(&t.to_le_bytes());
        }
        let want = [0.75, -0.75, 2.125];
        for (bits, format, eps) in [(0x2000_0000u32, PosFormat::Float3, 0.0), (0x1000_0000, PosFormat::Packed10, 6.0 / 1023.0), (0, PosFormat::Packed24, 6.0 / 65535.0)] {
            let word = 0x4000_0010 | bits;
            let mut b7 = vec![0u8; 96];
            let put = |b: &mut Vec<u8>, at: usize, v: u32| b[at..at + 4].copy_from_slice(&v.to_le_bytes());
            put(&mut b7, 0, 8 | 0x80 << 8);
            put(&mut b7, 4, 4 | 0x98 << 8);
            put(&mut b7, 0xc, 4 | 0x94 << 8);
            put(&mut b7, 0x10, 4 | 0x90 << 8);
            put(&mut b7, 0x34, word);
            put(&mut b7, 0x38, 1);
            let w = ps::cpu(&ps::Inputs { b7: &b7, vb: &vb, remap: &[0], palette: &palette, ..Default::default() });
            assert_eq!(ps::range(word).2, format);
            let (pos, normals) = range_decode(&w.bytes, word);
            assert_eq!((pos.len(), normals.len()), (1, 1));
            assert!(close(pos[0], want, eps), "{format:?}: {:?}", pos[0]);
            assert!((normals[0].iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-3);
        }
        assert!(range_position(&[0xff; 12], PosFormat::Float3)[0].is_nan());
        assert!(range_position(&[0xff; 4], PosFormat::Packed10)[0].is_nan());
        assert!(range_position(&[0xff; 8], PosFormat::Packed24)[0].is_nan());
        // The decode's own constants: the largest codes are +3.
        assert!(close(range_position(&[0xff, 0xff, 0xff, 0x3f], PosFormat::Packed10), [3.0; 3], 1e-3));
        assert!(close(range_position(&[0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff], PosFormat::Packed24), [3.0; 3], 1e-3));
        assert!(close(range_position(&[0; 8], PosFormat::Packed24), [-3.0; 3], 0.0));
    }

    /// The palette base: rows cut from the capture stand at row 0, the
    /// previous frame's after them; without previous rows both bases are 0.
    #[test]
    fn palette_base_moves_to_the_cut_rows() {
        let mut b7 = vec![0u8; 96];
        b7[0x28..0x2c].copy_from_slice(&0x6ab3eu32.to_le_bytes());
        b7[0x2c..0x30].copy_from_slice(&0x1234u32.to_le_bytes());
        let (cur, prev) = (vec![1u8; 96 + 5], vec![2u8; 48]);
        let (b, p) = palette_rebase(&b7, &cur, Some(&prev));
        assert_eq!((le32(&b, 0x28), le32(&b, 0x2c), p.len()), (0, 2, 144));
        assert_eq!((p[95], p[96]), (1, 2));
        let (b, p) = palette_rebase(&b7, &cur, None);
        assert_eq!((le32(&b, 0x28), le32(&b, 0x2c), p.len()), (0, 0, 96));
        assert_eq!(remap_elements(&[1, 0, 2, 0, 0xef, 1, 0, 0], 2), [1, 2, 0x1ef, 0]);
        assert_eq!(remap_elements(&[1, 0, 2, 0, 0xef, 1, 0, 0], 4), [0x2_0001, 0x1ef]);
    }

    /// The vertex-buffer copy: float3 streams appended on a 4-byte boundary,
    /// stream words 0 and 8 pointing at them, the skin words off, and the
    /// crate's own stream reader gives the positions back.
    #[test]
    fn copy_reads_back_as_float3_without_skin() {
        let mut vb = vec![7u8; 0x86];
        vb[0x18..0x28].fill(0x55);
        let pos = [[1.0f32, 2.0, 3.0], [-4.0, 5.5, 6.25]];
        let nrm = [[0.0f32, 0.0, 1.0], [1.0, 0.0, 0.0]];
        let out = with_positions(&vb, &pos, &nrm).unwrap();
        assert_eq!(out.len(), 0x88 + 48);
        let s = crate::decode::Streams::parse(&out).unwrap();
        assert_eq!((s.pos.format, s.pos.stride, s.pos.offset), (3, 12, 0x88));
        assert_eq!((s.normal.format, s.normal.stride, s.normal.offset), (3, 12, 0x88 + 24));
        assert_eq!([s.position(&out, 0), s.position(&out, 1)], [Some(pos[0]), Some(pos[1])]);
        assert!(s.weights.iter().chain(&s.indices).all(|w| w.stride == 0));
        // Without normals the normal word stays.
        let out = with_positions(&vb, &pos, &[]).unwrap();
        assert_eq!(le32(&out, 8), 0x0707_0707);
        assert!(with_positions(&[0; 0x40], &pos, &[]).is_none());
        assert_eq!(variant_of("CS_PedStandardSkinning_11_15f19d85_Wrapped"), Some(11));
        assert_eq!(variant_of("CS_PosOnlyLodSkinning_1"), None);
    }
}
