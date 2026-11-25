//! XNW: descriptors read through a waterfall loop, and joins with an
//! undefined edge.
//!
//! A program that may see a different descriptor in each lane keeps it in
//! VGPRs and walks the distinct values with a waterfall loop:
//!
//! ```text
//! loop: v_readfirstlane_b32 s28, v73        ; ... s31, v76
//!       v_cmp_eq_u64 vcc, s[28:29], v[73:74]
//!       v_cmp_eq_u64 s[6:7], s[30:31], v[75:76]
//!       s_and_b64 s[6:7], vcc, s[6:7]
//!       s_and_saveexec_b64 s[6:7], s[6:7]
//!       image_sample ..., s[28:31]            ; only lanes equal to the first
//!       s_xor_b64 exec, exec, s[6:7]
//!       s_cbranch_execnz loop
//! ```
//!
//! Under the `s_and_saveexec` every active lane's VGPR words equal the read
//! ones, so the descriptor there is each lane's own: the source of the VGPR
//! words, traced per lane. The words are copies of scalar descriptor words
//! (`v_mov_b32 v73, s8`) made on the paths of a divergent branch, so they
//! meet in VGPR phis; a phi edge where a word is an undefined register (a
//! VGPR the stage does not initialise, the "skip the then-part" edge of an
//! if/else) carries no descriptor, and a phi whose defined edges agree is
//! that one source. The host runs one thread per lane, so a per-thread
//! choice ([`Source::Phi`]) is exact and the loop runs once per thread.

use super::{Descriptor, MAX_JOINS, Role, Site, Source, Term, Tracer};
use crate::gfx10::{Fields, Operand};
use crate::ir::{EXEC, Exit, Inst, Reg, Value};
use crate::native::HwStage;

impl Tracer<'_> {
    /// Whether `t` is a register the hardware leaves undefined on entry:
    /// a VGPR past the ones the stage initialises (conservative: none for
    /// stages whose VGPR inputs vary).
    pub(super) fn undefined(&self, t: Term) -> bool {
        let Term::Value(v) = t else { return false };
        let Some(Reg::V(n)) = self.inputs[v.0 as usize] else {
            return false;
        };
        match self.stage {
            // NGG: v0-v8 at most (vertex and primitive ids, the group).
            HwStage::Ms | HwStage::Gs | HwStage::Es => n > 8,
            HwStage::Cs => n > 2,
            _ => false,
        }
    }

    /// For descriptor words `words` that `inst` at `site` reads: when each
    /// is a `v_readfirstlane_b32` of a VGPR in the same block, compared for
    /// equality with it and then masked into `exec` before `inst`, the VGPR
    /// values read.
    pub(super) fn waterfall_lanes(&self, site: (usize, usize), words: &[Value]) -> Option<Vec<Value>> {
        let (b, i) = site;
        let insts = &self.p.blocks[b].insts[..i];
        words
            .iter()
            .map(|&w| {
                let (wb, j) = self.defs[w.0 as usize]?;
                let read = self.inst((wb, j));
                if wb != b || read.name() != "v_readfirstlane_b32" {
                    return None;
                }
                let lane = read.reads.iter().find(|r| matches!(r.0, Reg::V(_)))?.1;
                let cmp = (j + 1..i).find(|&k| {
                    let c = &insts[k];
                    (c.name().starts_with("v_cmp_eq_u") || c.name().starts_with("v_cmpx_eq_u"))
                        && c.reads.iter().any(|r| r.1 == w)
                        && c.reads.iter().any(|r| r.1 == lane)
                })?;
                let masked = |c: &Inst| c.name().starts_with("v_cmpx") || c.name().contains("saveexec");
                (cmp..i).any(|k| masked(&insts[k])).then_some(lane)
            })
            .collect()
    }

    /// The scalar loads behind the samplers that waterfall loops walk
    /// ([`super::Resources::lane_samplers`]).
    pub(super) fn lane_samplers(&self, descriptors: &[Descriptor]) -> Vec<Site> {
        let mut loads = Vec::new();
        for d in descriptors.iter().filter(|d| d.role == Role::Sampler) {
            let inst = self.inst(d.site);
            let Fields::Image { sampler: Some(Operand::Scalar { code, dwords }), .. } = inst.fields else {
                continue;
            };
            let walked = (0..dwords)
                .map(|k| inst.read(Reg::S(code + k)))
                .collect::<Option<Vec<_>>>()
                .is_some_and(|read| self.waterfall_lanes(d.site, &read).is_some());
            if !walked {
                continue;
            }
            for leaf in d.source.leaves() {
                if let Source::Loaded { load, .. } = *leaf
                    && !loads.contains(&load)
                {
                    loads.push(load);
                }
            }
        }
        loads.retain(|&load| self.only_copied(load));
        loads
    }

    /// Whether the dwords the scalar load at `load` writes, and their
    /// scalar copies, are read only by copies (`s_mov`, a plain
    /// `v_mov_b32`) and as the sampler of an image instruction: nothing
    /// computes with them.
    fn only_copied(&self, load: Site) -> bool {
        let words: Vec<Value> = self.inst(load).writes.iter().map(|w| w.1).collect();
        let ours = |v: Value| matches!(self.origin(v), Term::Value(o) if words.contains(&o));
        self.p.blocks.iter().flat_map(|b| &b.insts).all(|i| {
            i.reads.iter().filter(|r| ours(r.1)).all(|&(reg, _)| match &i.fields {
                Fields::Image { sampler: Some(Operand::Scalar { code, dwords }), .. } => {
                    matches!(reg, Reg::S(c) if (*code..*code + *dwords).contains(&c))
                }
                Fields::Sop { .. } => !i.writes.is_empty() && i.writes.iter().all(|w| ours(w.1)),
                Fields::Valu(f) => {
                    i.name() == "v_mov_b32" && f.abs == 0 && f.neg == 0 && f.sdwa.is_none() && f.dpp.is_none()
                }
                _ => false,
            })
        })
    }

    /// `t` through VGPR copies and copy-propagated phis, to a fixed point.
    fn lane_origin(&self, mut t: Term) -> Term {
        for _ in 0..super::MAX_ROUNDS {
            let Term::Value(v) = t else { break };
            let next = self.vorigin(v);
            if next == t {
                break;
            }
            t = next;
        }
        t
    }

    /// The source of per-lane descriptor words: through VGPR copies and
    /// VGPR phis (undefined edges dropped) to scalar words, then
    /// [`Tracer::classify`].
    pub(super) fn lane_classify(&self, words: &[Term], joins: &mut Vec<usize>) -> Result<Source, &'static str> {
        self.lane_join(words, joins, None)
    }

    /// [`Tracer::lane_classify`]; `skip` names a block and the predecessor
    /// edges into it no lane that goes on to the caller's join takes.
    fn lane_join(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        skip: Option<(usize, Vec<usize>)>,
    ) -> Result<Source, &'static str> {
        let words: Vec<Term> = words
            .iter()
            .map(|&w| self.lane_origin(w))
            .collect();
        let vphi = |w: &Term| match *w {
            Term::Value(v) => self.phis[v.0 as usize].filter(|&(b, i)| matches!(self.p.blocks[b].phis[i].reg, Reg::V(_))),
            Term::Const(_) => None,
        };
        let Some((block, _)) = words.iter().find_map(vphi) else {
            return self.classify(&words, joins);
        };
        if words.iter().filter_map(vphi).any(|(b, _)| b != block) {
            return Err("a descriptor joined in several blocks");
        }
        if joins.contains(&block) {
            return Err("a descriptor that changes in a loop");
        }
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(block);
        let phis = &self.p.blocks[block].phis;
        let mut args = Vec::new();
        let dropped = skip.filter(|s| s.0 == block).map(|s| s.1).unwrap_or_default();
        for (j, &q) in self.p.blocks[block].preds.iter().enumerate() {
            let mut own = true;
            let edge: Vec<Term> = words
                .iter()
                .map(|w| match vphi(w) {
                    Some((_, i)) => {
                        let a = phis[i].args[j];
                        own &= a == phis[i].value;
                        self.lane_origin(Term::Value(a))
                    }
                    None => {
                        own = false;
                        *w
                    }
                })
                .collect();
            // An edge that carries no descriptor: one no lane takes, an
            // undefined word, or the phis themselves (a loop that leaves
            // them unchanged).
            if dropped.contains(&j) || own || edge.iter().any(|&t| self.undefined(t)) {
                args.push(None);
                continue;
            }
            match self.lane_join(&edge, joins, Some((q, self.not_on_to(q, block)))) {
                Ok(s) => args.push(Some(s)),
                Err(e) => {
                    joins.pop();
                    return Err(e);
                }
            }
        }
        joins.pop();
        let first = args.iter().flatten().next().cloned().ok_or("a descriptor undefined on every edge")?;
        if args.iter().flatten().all(|a| *a == first) {
            return Ok(first);
        }
        // An edge without a descriptor may take any: the first.
        Ok(Source::Phi {
            block,
            args: args.into_iter().map(|a| a.unwrap_or_else(|| first.clone())).collect(),
        })
    }

    /// The predecessor edges (indices into `q`'s preds) no lane takes into
    /// `q` and then on from `q` to `join`, for the if/else of exec masks:
    ///
    /// ```text
    /// a: s_and_saveexec_b64 s[60:61], cond ; s_cbranch_execz q    (skip the then-part)
    ///    ...then-part...
    /// q: s_andn2_b64 exec, s[60:61], exec  ; s_cbranch_execz join (skip the else-part)
    /// ```
    ///
    /// A lane that skips the then-part from `a` had its bit in s[60:61]
    /// and none in exec, so the `s_andn2` leaves its bit set and it runs
    /// the else-part: it never takes `q`'s skip to `join` directly.
    fn not_on_to(&self, q: usize, join: usize) -> Vec<usize> {
        let execz = |b: usize, to: usize| {
            matches!(self.p.blocks[b].exit, Exit::Branch { taken, .. } if taken == to)
                && self.p.blocks[b].insts.last().is_some_and(|i| i.name() == "s_cbranch_execz")
        };
        if !execz(q, join) {
            return Vec::new();
        }
        // q: exec = saved & !exec, reading `saved` as written in `a`.
        let Some(andn2) = self.p.blocks[q].insts.iter().find(|i| {
            matches!(i.name(), "s_andn2_b64" | "s_andn2_b32")
                && i.write(Reg::S(EXEC)).is_some()
                && matches!(&i.fields, Fields::Sop { src, .. }
                    if matches!(src.get(1), Some(Operand::Scalar { code: EXEC, .. }))
                        && matches!(src.first(), Some(Operand::Scalar { code, .. }) if *code != EXEC))
        }) else {
            return Vec::new();
        };
        let Fields::Sop { src, .. } = &andn2.fields else { return Vec::new() };
        let Some(&Operand::Scalar { code: saved, .. }) = src.first() else { return Vec::new() };
        let Some(value) = andn2.read(Reg::S(saved)) else { return Vec::new() };
        self.p.blocks[q]
            .preds
            .iter()
            .enumerate()
            .filter(|&(_, &a)| {
                execz(a, q)
                    && self.p.blocks[a].insts.iter().any(|i| {
                        i.name().starts_with("s_and_saveexec_b") && i.write(Reg::S(saved)) == Some(value)
                    })
            })
            .map(|(j, _)| j)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::native::{HwStage, RootParam, RootType};
    use crate::resource::{Role, Source, track};

    const fn sop1(op: u8, sdst: u8, src0: u8) -> u32 {
        0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | src0 as u32
    }
    const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
        0x8000_0000 | (op as u32) << 23 | (sdst as u32) << 16 | (src1 as u32) << 8 | src0 as u32
    }
    const fn sopp(op: u8, simm16: i16) -> u32 {
        0xBF80_0000 | (op as u32) << 16 | simm16 as u16 as u32
    }
    const fn vop1(op: u8, vdst: u8, src0: u16) -> u32 {
        0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | src0 as u32
    }
    const fn vopc(op: u8, src0: u16, vsrc1: u8) -> u32 {
        0x7C00_0000 | (op as u32) << 17 | (vsrc1 as u32) << 9 | src0 as u32
    }
    const fn s_load_x4(dst: u8, base: u8, offset: u32) -> [u32; 2] {
        [0xF408_0000 | (dst as u32) << 6 | (base as u32 >> 1), 0xFA00_0000 | offset]
    }
    const EXEC: u8 = 126;
    const VCC: u8 = 106;
    const fn v(n: u8) -> u16 {
        256 + n as u16
    }

    /// An if/else that copies one of two table descriptors into v[20:23]
    /// (the then-part's words undefined on the else path), then a
    /// waterfall loop sampling with them as the sampler.
    fn program(else_offset: u32, defined: bool) -> Vec<u32> {
        // With `defined`, v[20:23] hold a lane value before the if/else.
        let mut w: Vec<u32> = (0..4).filter(|_| defined).map(|k| vop1(1, 20 + k, v(0))).collect();
        w.extend([
            sop1(0x03, 2, 9),   // s_mov_b32 s2, s9
            sop1(0x03, 9, 128), // s_mov_b32 s9, 0
            sop1(0x03, 3, 128), // s_mov_b32 s3, 0
            vopc(0xC2, 128, 0), // v_cmp_eq_u32 vcc, 0, v0
            sop1(0x24, 6, VCC), // s_and_saveexec_b64 s[6:7], vcc
            sopp(8, 7),         // s_cbranch_execz L1
        ]);
        let copies = |w: &mut Vec<u32>, offset| {
            w.extend(s_load_x4(12, 8, offset)); // s_load_dwordx4 s[12:15], s[8:9], offset
            w.push(sopp(12, 0)); // s_waitcnt 0
            w.extend((0..4).map(|k| vop1(1, 20 + k, 12 + k as u16))); // v_mov_b32 v20+k, s12+k
        };
        copies(&mut w, 0x10);
        w.push(sop2(0x15, EXEC, 6, EXEC)); // s_andn2_b64 exec, s[6:7], exec
        w.push(sopp(8, 7)); // s_cbranch_execz L2
        copies(&mut w, else_offset);
        w.extend([
            sop2(0x11, EXEC, EXEC, 6), // s_or_b64 exec, exec, s[6:7]
            0xF40C_0401,
            0xFA00_0000,                // s_load_dwordx8 s[16:23], s[2:3], null
            sopp(12, 0),                // s_waitcnt 0
            sop1(0x04, 24, EXEC),       // s_mov_b64 s[24:25], exec
        ]);
        w.extend((0..4).map(|k| vop1(2, 28 + k, v(20 + k)))); // v_readfirstlane_b32 s28+k, v20+k
        w.extend([
            vopc(0xE2, 28, 20), // v_cmp_eq_u64 vcc, s[28:29], v[20:21]
            0xD4E2_001A,
            0x0002_2C1E,              // v_cmp_eq_u64 s[26:27], s[30:31], v[22:23]
            sop2(0x0F, 26, VCC, 26),  // s_and_b64 s[26:27], vcc, s[26:27]
            sop1(0x24, 26, 26),       // s_and_saveexec_b64 s[26:27], s[26:27]
            0xF080_0F08,
            0x00E4_0004,              // image_sample v[0:3], v[4:5], s[16:23], s[28:31]
            sop2(0x13, EXEC, EXEC, 26), // s_xor_b64 exec, exec, s[26:27]
            sopp(9, -13),             // s_cbranch_execnz L3
            sop1(0x04, EXEC, 24),     // s_mov_b64 exec, s[24:25]
            0xBF81_0000,              // s_endpgm
        ]);
        w
    }

    fn sampler(words: &[u32]) -> (Source, Vec<u32>) {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, true, 256).unwrap();
        let table = |user_data| RootParam { root: RootType::Table, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data, slot: 0 };
        let r = track(&p, &[table(8), table(9)], HwStage::Cs).unwrap();
        let d = r.descriptors.iter().find(|d| d.role == Role::Sampler).unwrap();
        let offset = |s: &Source| match *s {
            Source::Loaded { load, first: 0 } => r.scalar.iter().find(|a| a.0 == load).unwrap().1.offset,
            _ => panic!("{s:?}"),
        };
        (d.source.clone(), d.source.leaves().into_iter().map(offset).collect())
    }

    #[test]
    fn decodes_as_written() {
        let code: Vec<u8> = program(0x20, false).iter().flat_map(|w| w.to_le_bytes()).collect();
        let text: Vec<String> = crate::gfx10::decode(&code).unwrap().iter().map(|i| i.text(true)).collect();
        let has = |s: &str| text.iter().any(|t| t.starts_with(s));
        assert!(has("v_cmp_eq_u64 s[26:27], s[30:31], v[22:23]"), "{text:#?}");
        assert!(has("image_sample v[0:3], v[4:5], s[16:23], s[28:31]"), "{text:#?}");
        assert!(has("s_cbranch_execnz label_006C"), "{text:#?}");
    }

    #[test]
    fn waterfall_sampler_is_each_lanes_own() {
        // Two loads: chosen per thread at the join after the else-part.
        for defined in [false, true] {
            let (s, offsets) = sampler(&program(0x20, defined));
            assert!(matches!(s, Source::Phi { .. }), "{s:?}");
            assert_eq!(offsets, [0x10, 0x20]);
        }
    }

    #[test]
    fn one_descriptor_on_both_paths_is_that_one() {
        // The same table entry loaded on both paths: still one load per
        // path (two sites), chosen per thread.
        let (s, offsets) = sampler(&program(0x10, true));
        assert!(matches!(s, Source::Phi { .. }), "{s:?}");
        assert_eq!(offsets, [0x10, 0x10]);
    }
}
