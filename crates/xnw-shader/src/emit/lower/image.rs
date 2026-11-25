// Part of emit/lower.rs: atomic loops, 64-bit image atomics, image ops, cube directions.
// Origin as in lower.rs's header: VIctory crates/victory-shader/src/emit/lower.rs, d2a4bae1f495, and the
// later VIctory ports named there; GPL-3.0-or-later. Moved out of lower.rs unchanged (XNW BLD2): one stage,
// one codegen unit.
// With DexNom-VIctory (ddb7332), GPL-3.0-or-later: a uniform branch choosing between
// two immediate samplers (Lower::image, image_with_sampler).
use super::*;

impl<'a> Lower<'a> {
    /// `x` (an f16 in the low bits) as the low half of VGPR `reg`: 16-bit
    /// results keep the register's high half (GFX10).
    pub(super) fn low_half(&mut self, inst: &Inst, reg: u8, x: V) -> V {
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
    pub(super) fn atomic_loop(&mut self, h: V, coords: &[V], op: &str, vals: &[V]) -> V {
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
        let limit = self.s.m.i32(super::super::loop_limit_for(self.x.native.stage, self.x.fallback));
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
    pub(super) fn image_atomic64(
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

    /// DexNom-VIctory (ddb7332, GPL-3.0-or-later): a uniform branch that
    /// picks one of two immediate samplers (a phi of two constant S#).
    pub(super) fn image(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let sampler = self.descriptors.get(&((b, i), true)).copied();
        if let Some(Source::Phi { block, args }) = sampler
            && let [first @ Source::Constant(_), second @ Source::Constant(_)] = args.as_slice()
            && let [left, right] = self.x.p.blocks[*block].preds.as_slice()
        {
            // Immediate samplers are separate static bindings. Sample both with
            // the original derivatives, then select the texel on the same edge
            // that supplied the console's sampler.
            if self.block[*left].is_none() {
                return self.image_with_sampler(inst, b, i, Some(second));
            }
            if self.block[*right].is_none() {
                return self.image_with_sampler(inst, b, i, Some(first));
            }
            let edge = |pred: usize| {
                self.edges
                    .get(&(pred, *block))
                    .copied()
                    .or(self.tail[pred])
                    .ok_or("a sampler choice with an unfinished predecessor")
            };
            let (left, right) = (edge(*left)?, edge(*right)?);
            let join = self.block[*block].ok_or("a sampler choice without a host block")?;
            let chosen = self.s.m.phi(join, self.s.t.i1);
            let (yes, no) = (self.s.m.bool(true), self.s.m.bool(false));
            self.s.m.set_phi(chosen, vec![(yes, left), (no, right)]);
            self.image_with_sampler(inst, b, i, Some(first))?;
            let left = inst
                .writes
                .iter()
                .map(|&(_, v)| self.i32_of(v))
                .collect::<Result<Vec<_>, _>>()?;
            self.image_with_sampler(inst, b, i, Some(second))?;
            for (&(_, v), a) in inst.writes.iter().zip(left) {
                let b = self.i32_of(v)?;
                let result = self.s.m.select(self.cur, chosen, a, b);
                self.vals.insert(v, Val::I32(result));
            }
            return Ok(());
        }
        self.image_with_sampler(inst, b, i, sampler)
    }

    fn image_with_sampler(
        &mut self,
        inst: &Inst,
        b: usize,
        i: usize,
        sampler: Option<&Source>,
    ) -> Result<(), Why> {
        if let Some(h) = self.s.host_copies.as_mut() {
            *h |= super::super::host_copies::TEXTURES;
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
            let ss = sampler.ok_or("an untracked sampler")?;
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
    pub(super) fn cube_direction(&mut self, s: V, t: V, face: V) -> [V; 3] {
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
}
