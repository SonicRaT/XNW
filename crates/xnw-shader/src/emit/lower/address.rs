// Part of emit/lower.rs: address offsets, relative offsets, resource handles, heap slots.
// Origin as in lower.rs's header: VIctory crates/victory-shader/src/emit/lower.rs, d2a4bae1f495, and the
// later VIctory ports named there; GPL-3.0-or-later. Moved out of lower.rs unchanged (XNW BLD2): one stage,
// one codegen unit.
use super::*;

impl<'a> Lower<'a> {
    /// The byte offset a buffer descriptor built around a root descriptor adds
    /// (constant and computed parts); none for other descriptors.
    pub(super) fn built_offset(&mut self, src: &Source) -> Result<(u32, Option<V>), Why> {
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
    pub(super) fn address_offset(&mut self, a: Address) -> Result<V, Why> {
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

    pub(super) fn op_value(inst: &Inst, op: &Operand) -> Option<Value> {
        match *op {
            Operand::Scalar { code, .. } if code != NULL => inst.read(Reg::S(code)),
            Operand::Vector { reg, .. } => inst.read(Reg::V(reg)),
            _ => None,
        }
    }

    /// Follows unmodified, unguarded `v_mov_b32` chains back to an SGPR value.
    pub(super) fn uniform_vgpr_src(&self, mut inst: &'a Inst, mut op: &'a Operand) -> Option<Value> {
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
    pub(super) fn spilled_value(&self, inst: &Inst, src: &Operand, lane: &Operand) -> Option<Value> {
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

    pub(super) fn spilled_in_vgpr(
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

    pub(super) fn op_reaches_base(&self, inst: &Inst, op: &Operand, base: Value) -> bool {
        Self::op_value(inst, op).is_some_and(|v| self.reaches_base(v, base))
    }

    pub(super) fn reaches_base(&self, v: Value, base: Value) -> bool {
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

    pub(super) fn rel_offset_op(
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
    pub(super) fn rel_offset(
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
    pub(super) fn used_kind(&self, src: &Source) -> Option<&'a UsedResource> {
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
    pub(super) fn handle(
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
                let (index, own) = self.own_index(*load, *first, sampler, props)?;
                let h = if own { self.s.heap_handle_own(self.cur, index, props) } else { self.s.heap_handle(self.cur, index, sampler, props) };
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
                    let index = self.own_index(*load, *first, sampler, props);
                    self.cur = here;
                    incoming.push((index?.0, at));
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
    pub(super) fn heap_picked(&self, src: &Source) -> bool {
        let Source::Loaded { load, .. } = src else {
            return false;
        };
        self.scalar
            .get(load)
            .and_then(|a| self.x.native.layout.get(a.param))
            .is_some_and(|p| p.root == RootType::ResourceHeap)
    }

    pub(super) fn heap_slot(&mut self, src: &Source, sampler: bool) -> Result<V, Why> {
        // A buffer's slot: confined as a raw buffer's handle is (BND1).
        let props = (11, 0);
        match src {
            Source::Loaded { load, first } => Ok(self.own_index(*load, *first, sampler, props)?.0),
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
                    let index = self.own_index(*load, *first, sampler, props);
                    self.cur = here;
                    incoming.push((index?.0, at));
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
    /// [`super::super::STRIDE_SPACE`]) for the view-heap slot of `src`, read in the
    /// current block: the slot holding a raw view of the buffer's bytes
    /// (the slot itself when the table names none), the stride and the bias
    /// ([`super::super::StrideEntry`]).
    pub(super) fn stride_entry(&mut self, src: &Source) -> Result<[V; 3], Why> {
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
        // HANG1, after VIctory-Belobug-AMD 71652a2 (`slot_dword`,
        // GPL-3.0-or-later): the table is a root view, which has no bounds
        // on the card, so a slot past the heap (a wild index from title
        // data) reads entry 0, not past the table; the handle below is
        // bounded as every heap handle is.
        let limit = self.s.heap_limit_words()[0];
        let inside = self.s.m.cmp(c, pred::IULT, slot, limit);
        let zero = self.s.m.i32(0);
        let read_at = self.s.m.select(c, inside, slot, zero);
        let bytes = self.s.m.i32(super::super::StrideEntry::BYTES);
        let addr = self.s.m.bin(c, Bin::Mul, read_at, bytes);
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
        let low = self.s.m.i32(super::super::StrideEntry::RAW_SLOTS);
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
    pub(super) fn runtime_view(
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

    /// [`Lower::heap_index`] as the draw's own ([`super::super::Options::confine`],
    /// BND1); true when the index needs no bound any more. Off, or for a
    /// sampler (the sampler heap is the draw's own list, whole): the
    /// index as it is. On:
    /// - a view picked from the heap by an index the program computed (a
    ///   `ResourceHeap` parameter): [`super::super::shader::Shader::confined`];
    /// - a table's descriptor: at a fixed slot the program's resource list
    ///   names (the host's block of that table reaches past every one of
    ///   them and holds a view or typed null of ours at each), the index
    ///   as it is; at a fixed slot past them, the null of `props`' kind;
    ///   at a computed offset, refused (no bridged program has one).
    pub(super) fn own_index(&mut self, load: Site, first: u8, sampler: bool, props: (u32, u32)) -> Result<(V, bool), Why> {
        let index = self.heap_index(load, first, sampler)?;
        if sampler || !super::super::CONFINE.with(std::cell::Cell::get) {
            return Ok((index, false));
        }
        let a = *self.scalar.get(&load).ok_or("an untracked descriptor load")?;
        let Some(rp) = self.x.native.layout.get(a.param).copied() else {
            return Ok((index, false));
        };
        match rp.root {
            RootType::ResourceHeap => Ok((self.s.confined(self.cur, index, props), true)),
            RootType::Table => {
                let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
                let fixed = matches!(&li.fields, Fields::Smem { soffset: Operand::Scalar { code: NULL, .. }, .. }) && a.dynamic.is_none() && a.rel.is_none();
                if !fixed {
                    return Err("a table descriptor at a computed offset (confined)");
                }
                let slot = a.offset.wrapping_add(4 * u32::from(first)) / 32;
                let named = self.x.native.used.iter().filter(|u| u.root == RootType::Table && u.user_data == rp.user_data).map(|u| u.table_offset_dwords / 8 + 1).max().unwrap_or(0);
                if slot < named { Ok((index, false)) } else { Ok((self.s.null_index(self.cur, props), true)) }
            }
            _ => Ok((index, false)),
        }
    }

    /// The heap index of the descriptor at dword `first` of what the scalar
    /// load at `load` reads, computed in the current block.
    pub(super) fn heap_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        let addr = self.heap_byte(load, first)?;
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        Ok(self.s.m.bin(self.cur, Bin::LShr, addr, shift))
    }

    /// The byte offset in its heap of dword `first` of what the scalar load
    /// at `load` reads (the console's entry sizes), computed in the current
    /// block.
    pub(super) fn heap_byte(&mut self, load: Site, first: u8) -> Result<V, Why> {
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
}
