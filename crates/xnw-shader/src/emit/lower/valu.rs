// Part of emit/lower.rs: vector ALU: dpp, the 32-bit op table, 64-bit sources.
// Origin as in lower.rs's header: VIctory crates/victory-shader/src/emit/lower.rs, d2a4bae1f495, and the
// later VIctory ports named there; GPL-3.0-or-later. Moved out of lower.rs unchanged (XNW BLD2): one stage,
// one codegen unit.
use super::*;

impl<'a> Lower<'a> {
    pub(super) fn valu(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        if let Some(rest) = name.strip_prefix("v_cmp_").or(name.strip_prefix("v_cmpx_")) {
            let r = if v.dpp.is_some() {
                Err("a compare with DPP")
            } else {
                self.compare(inst, rest)
            };
            if let Err(why) = r {
                for &(_, w) in &inst.writes {
                    self.lane_fail.insert(w, why);
                }
            }
            return r;
        }
        let mut guard = self.guard(b, i)?;
        if let Some(d) = v.dpp {
            let (x, enable) = self.dpp(inst, d)?;
            guard = match (guard, enable) {
                (Some(g), Some(e)) => Some(self.s.m.bin(self.cur, Bin::And, g, e)),
                (g, e) => g.or(e),
            };
            self.src0 = Some(x);
        }
        let r = self.valu_op(inst, guard);
        self.src0 = None;
        r
    }

    /// DPP: src0 as read from another lane, and which lanes write their
    /// result (none: all).
    pub(super) fn dpp(&mut self, inst: &Inst, dpp: Dpp) -> Result<(V, Option<V>), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(&Operand::Vector { reg, .. }) = v.src.first() else {
            return Err("a DPP operand");
        };
        let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("a DPP operand")?)?;
        // Quad permutations and DPP8 stay in groups the pixel layout shares.
        let quad = matches!(dpp, Dpp::Row { ctrl: 0..=0xFF, .. });
        let mesh_quad = self.s.mesh.is_some() && quad;
        if !mesh_quad {
            self.cross_lane(quad)?;
        }
        let c = self.cur;
        let lane = self.lane_index();
        let k = |l: &mut Self, n: u32| l.s.m.i32(n);
        let (sixteen_mask, low) = (k(self, !15), k(self, 15));
        let row = self.s.m.bin(c, Bin::And, lane, sixteen_mask);
        let pos = self.s.m.bin(c, Bin::And, lane, low);
        // Source lane, and whether it lies in the row.
        let (src, inside, masks, zero) = match dpp {
            Dpp::Lanes { sel, .. } => {
                let (seven, not7, three) = (k(self, 7), k(self, !7), k(self, 3));
                let p = self.s.m.bin(c, Bin::And, lane, seven);
                let sh = self.s.m.bin(c, Bin::Mul, p, three);
                let sel = k(self, sel);
                let pick = self.s.m.bin(c, Bin::LShr, sel, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, seven);
                let base = self.s.m.bin(c, Bin::And, lane, not7);
                (self.s.m.bin(c, Bin::Or, base, pick), None, None, false)
            }
            Dpp::Row {
                ctrl,
                row_mask,
                bank_mask,
                bound_ctrl,
                ..
            } => {
                let n = u32::from(ctrl & 15);
                let (src, inside) = match ctrl {
                    0..=0xFF => {
                        let (three, one, not3) = (k(self, 3), k(self, 1), k(self, !3));
                        let p = self.s.m.bin(c, Bin::And, lane, three);
                        let sh = self.s.m.bin(c, Bin::Shl, p, one);
                        let perm = k(self, u32::from(ctrl));
                        let pick = self.s.m.bin(c, Bin::LShr, perm, sh);
                        let pick = self.s.m.bin(c, Bin::And, pick, three);
                        let base = self.s.m.bin(c, Bin::And, lane, not3);
                        (self.s.m.bin(c, Bin::Or, base, pick), None)
                    }
                    0x101..=0x10F | 0x111..=0x11F => {
                        let nn = k(self, n);
                        let left = ctrl < 0x110;
                        let (op, p) = if left {
                            (Bin::Add, pred::IULT)
                        } else {
                            (Bin::Sub, pred::IUGE)
                        };
                        let s = self.s.m.bin(c, op, lane, nn);
                        // Left: pos + n < 16; right: pos >= n.
                        let bound = if left { k(self, 16 - n) } else { nn };
                        (s, Some(self.s.m.cmp(c, p, pos, bound)))
                    }
                    0x121..=0x12F => {
                        let nn = k(self, n);
                        let t = self.s.m.bin(c, Bin::Sub, pos, nn);
                        let t = self.s.m.bin(c, Bin::And, t, low);
                        (self.s.m.bin(c, Bin::Or, row, t), None)
                    }
                    0x140 | 0x141 => {
                        let m = k(self, if ctrl == 0x140 { 15 } else { 7 });
                        let t = self.s.m.bin(c, Bin::Xor, lane, m);
                        (t, None)
                    }
                    0x150..=0x15F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Or, row, nn), None)
                    }
                    0x160..=0x16F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Xor, lane, nn), None)
                    }
                    _ => return Err("a DPP lane pattern"),
                };
                let masks = (row_mask != 0xF || bank_mask != 0xF).then_some((row_mask, bank_mask));
                (src, inside, masks, bound_ctrl)
            }
        };
        let mut r = if mesh_quad || self.helpers_read(quad) {
            self.quad_pick(x, src)
        } else {
            self.read_lane(x, src)
        };
        let mut enable = None;
        if let Some(inside) = inside {
            if zero {
                let z = k(self, 0);
                r = self.s.m.select(c, inside, r, z);
            } else {
                enable = Some(inside);
            }
        }
        if let Some((rows, banks)) = masks {
            // Row (16 lanes) and bank (4 lanes of a row) write masks.
            let (four, two, three) = (k(self, 4), k(self, 2), k(self, 3));
            let ri = self.s.m.bin(c, Bin::LShr, lane, four);
            let ri = self.s.m.bin(c, Bin::And, ri, three);
            let bi = self.s.m.bin(c, Bin::LShr, lane, two);
            let bi = self.s.m.bin(c, Bin::And, bi, three);
            let (rm, bm) = (k(self, u32::from(rows)), k(self, u32::from(banks)));
            let rb = self.s.m.bin(c, Bin::LShr, rm, ri);
            let bb = self.s.m.bin(c, Bin::LShr, bm, bi);
            let both = self.s.m.bin(c, Bin::And, rb, bb);
            let (one, z) = (k(self, 1), k(self, 0));
            let both = self.s.m.bin(c, Bin::And, both, one);
            let on = self.s.m.cmp(c, pred::NE, both, z);
            enable = Some(match enable {
                Some(e) => self.s.m.bin(c, Bin::And, e, on),
                None => on,
            });
        }
        Ok((r, enable))
    }

    pub(super) fn valu_op(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let c = self.cur;
        let f = |l: &mut Self, k| l.fsrc(inst, k);
        let s = |l: &mut Self, k| l.src(inst, k);
        let float = |l: &mut Self, r: V| l.write_f(inst, r, guard);
        let int = |l: &mut Self, r: V| l.write_i(inst, r, guard);
        if let Some(r) = self.valu16(inst, name, guard) {
            return r;
        }
        if let Some(op) = name.strip_prefix("v_pk_") {
            return self.packed16(inst, op, guard);
        }
        match name {
            "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64" => {
                let n = s(self, 0)?;
                let x = self.src64(inst, 1)?;
                let i64t = self.s.m.int(64);
                let k63 = self.s.m.i32(63);
                let n = self.s.m.bin(c, Bin::And, n, k63);
                let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
                let op = match name {
                    "v_lshlrev_b64" => Bin::Shl,
                    "v_lshrrev_b64" => Bin::LShr,
                    _ => Bin::AShr,
                };
                let r = self.s.m.bin(c, op, x, n);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a 64-bit result");
                };
                let (lo, hi) = self.split64(r);
                self.write(inst, Reg::V(reg), lo, guard);
                self.write(inst, Reg::V(reg + 1), hi, guard);
                Ok(())
            }
            "v_cubeid_f32" | "v_cubesc_f32" | "v_cubetc_f32" | "v_cubema_f32" => {
                // Cube-map face coordinates (AMD's ISA): the major axis is z
                // when |z| >= |x| and |z| >= |y|, else y when |y| >= |x|,
                // else x.
                let (x, y, z) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let (ax, ay, az) = (self.unary(6, x), self.unary(6, y), self.unary(6, z));
                let zx = self.s.m.cmp(c, pred::OGE, az, ax);
                let zy = self.s.m.cmp(c, pred::OGE, az, ay);
                let zmaj = self.s.m.bin(c, Bin::And, zx, zy);
                let ymaj = self.s.m.cmp(c, pred::OGE, ay, ax);
                let zero = self.s.m.f32(0);
                let neg = |l: &mut Self, v: V| l.s.m.cmp(c, pred::OLT, v, zero);
                let minus = |l: &mut Self, v: V| {
                    let z = l.s.m.f32(0x8000_0000);
                    l.s.m.bin(c, Bin::Sub, z, v)
                };
                let pick = |l: &mut Self, zv: V, yv: V, xv: V| {
                    let t = l.s.m.select(c, ymaj, yv, xv);
                    l.s.m.select(c, zmaj, zv, t)
                };
                let r = match name {
                    "v_cubeid_f32" => {
                        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
                        let (nz, ny, nx) = (neg(self, z), neg(self, y), neg(self, x));
                        let (f5, f4, f3, f2, f1, f0) = (
                            k(self, 5.0),
                            k(self, 4.0),
                            k(self, 3.0),
                            k(self, 2.0),
                            k(self, 1.0),
                            k(self, 0.0),
                        );
                        let zf = self.s.m.select(c, nz, f5, f4);
                        let yf = self.s.m.select(c, ny, f3, f2);
                        let xf = self.s.m.select(c, nx, f1, f0);
                        pick(self, zf, yf, xf)
                    }
                    "v_cubesc_f32" => {
                        let (nz, nx) = (neg(self, z), neg(self, x));
                        let (mx, mz) = (minus(self, x), minus(self, z));
                        let zf = self.s.m.select(c, nz, mx, x);
                        let xf = self.s.m.select(c, nx, z, mz);
                        pick(self, zf, x, xf)
                    }
                    "v_cubetc_f32" => {
                        let ny = neg(self, y);
                        let (my, mz) = (minus(self, y), minus(self, z));
                        let yf = self.s.m.select(c, ny, mz, z);
                        pick(self, my, yf, my)
                    }
                    _ => {
                        let t = pick(self, z, y, x);
                        let two = self.s.m.f32(2.0f32.to_bits());
                        self.s.m.bin(c, Bin::Mul, t, two)
                    }
                };
                float(self, r)
            }
            "v_movrels_b32" | "v_movreld_b32" | "v_movrelsd_b32" => {
                // M0 indexes the source (movrels), the destination (movreld)
                // or both, from the encoded register up to the program's
                // VGPR count (the registers the instruction reads or writes).
                let m0 = self.i32_of(inst.read(Reg::S(crate::ir::M0)).ok_or("M0")?)?;
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("an M0-relative move");
                };
                let indexed_src = name != "v_movreld_b32";
                let value = if indexed_src {
                    let mut r = self.s.m.i32(0);
                    for k in 0..=255u8.saturating_sub(sreg) {
                        let Some(x) = inst.read(Reg::V(sreg + k)) else {
                            break;
                        };
                        let x = self.i32_of(x)?;
                        let kk = self.s.m.i32(u32::from(k));
                        let is = self.s.m.cmp(c, pred::EQ, m0, kk);
                        r = self.s.m.select(c, is, x, r);
                    }
                    r
                } else {
                    s(self, 0)?
                };
                if name == "v_movrels_b32" {
                    return int(self, value);
                }
                for k in 0..=255u8.saturating_sub(dreg) {
                    let Some(_) = inst.write(Reg::V(dreg + k)) else {
                        break;
                    };
                    let old = match inst.read(Reg::V(dreg + k)) {
                        Some(o) => self.i32_of(o)?,
                        None => self.s.m.i32(0),
                    };
                    let kk = self.s.m.i32(u32::from(k));
                    let is = self.s.m.cmp(c, pred::EQ, m0, kk);
                    let x = self.s.m.select(c, is, value, old);
                    self.write_placed(inst, Reg::V(dreg + k), x, guard);
                }
                Ok(())
            }
            "v_swap_b32" => {
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("v_swap_b32");
                };
                let a = self.i32_of(inst.read(Reg::V(dreg)).ok_or("v_swap_b32")?)?;
                let b = self.i32_of(inst.read(Reg::V(sreg)).ok_or("v_swap_b32")?)?;
                self.write_placed(inst, Reg::V(dreg), b, guard);
                if dreg != sreg {
                    self.write_placed(inst, Reg::V(sreg), a, guard);
                }
                Ok(())
            }
            "v_ffbh_u32" | "v_ffbh_i32" => {
                // Counted from the top bit, -1 when none (DXIL's too).
                let a = s(self, 0)?;
                let op = if name == "v_ffbh_u32" { 33 } else { 34 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                int(self, r)
            }
            "v_mov_b32" => {
                if v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0 {
                    let a = f(self, 0)?;
                    return float(self, a);
                }
                let a = s(self, 0)?;
                int(self, a)
            }
            "v_add_f32" | "v_sub_f32" | "v_subrev_f32" | "v_mul_f32" | "v_mul_legacy_f32"
            | "v_max_f32" | "v_min_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let r = match name {
                    "v_add_f32" => self.s.m.bin(c, Bin::Add, a, bb),
                    "v_sub_f32" => self.s.m.bin(c, Bin::Sub, a, bb),
                    "v_subrev_f32" => self.s.m.bin(c, Bin::Sub, bb, a),
                    "v_mul_f32" => self.s.m.bin(c, Bin::Mul, a, bb),
                    "v_mul_legacy_f32" => self.legacy_mul(a, bb),
                    "v_max_f32" => self.max_min(true, a, bb),
                    _ => self.max_min(false, a, bb),
                };
                float(self, r)
            }
            "v_fma_f32" | "v_fmaak_f32" | "v_fmamk_f32" => {
                // shortcut: DXIL has no fused f32 multiply-add; FMad lets the driver fuse.
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = self.fused(a, bb, cc);
                float(self, r)
            }
            "v_div_scale_f32" => {
                // RDNA2 ISA: src0 = value to scale, src1 = denominator, src2 = numerator.
                let (a, den, num) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let (ai, ni, di) = (self.as_int(a), self.as_int(num), self.as_int(den));
                let (k23, k255, kabs) = (
                    self.s.m.i32(23),
                    self.s.m.i32(0xFF),
                    self.s.m.i32(0x7FFF_FFFF),
                );
                let (zero, k23_exp, k232_exp, k96) = (
                    self.s.m.i32(0),
                    self.s.m.i32(23),
                    self.s.m.i32(232),
                    self.s.m.i32(96),
                );
                let exp_of = |l: &mut Self, x: V| {
                    let s = l.s.m.bin(c, Bin::LShr, x, k23);
                    l.s.m.bin(c, Bin::And, s, k255)
                };
                let (e1, e2) = (exp_of(self, ni), exp_of(self, di));
                let n_abs = self.s.m.bin(c, Bin::And, ni, kabs);
                let d_abs = self.s.m.bin(c, Bin::And, di, kabs);
                let z1 = self.s.m.cmp(c, pred::EQ, n_abs, zero);
                let z2 = self.s.m.cmp(c, pred::EQ, d_abs, zero);
                let any_zero = self.s.m.bin(c, Bin::Or, z1, z2);
                let special = {
                    let s1 = self.s.m.cmp(c, pred::EQ, e1, k255);
                    let s2 = self.s.m.cmp(c, pred::EQ, e2, k255);
                    let a12 = self.s.m.bin(c, Bin::Or, s1, s2);
                    self.s.m.bin(c, Bin::Or, a12, any_zero)
                };
                let is_den = self.s.m.cmp(c, pred::EQ, ai, di);
                let diff = self.s.m.bin(c, Bin::Sub, e2, e1);
                let den_tiny = self.s.m.cmp(c, pred::IULE, e2, k23_exp);
                let den_huge = self.s.m.cmp(c, pred::IUGE, e2, k232_exp);
                let num_tiny_exp = self.s.m.cmp(c, pred::IULE, e1, k23_exp);
                let num_huge = self.s.m.cmp(c, pred::IUGE, e1, k232_exp);
                let num_gap = self.s.m.cmp(c, pred::SGE, diff, k96);
                let not_den_huge = {
                    let t = self.s.m.bool(true);
                    self.s.m.bin(c, Bin::Xor, den_huge, t)
                };
                let num_gap = self.s.m.bin(c, Bin::And, num_gap, not_den_huge);
                let num_tiny = self.s.m.bin(c, Bin::Or, num_tiny_exp, num_gap);
                let up_raw = self.s.m.select(c, is_den, den_tiny, num_tiny);
                let dn_raw = self.s.m.select(c, is_den, den_huge, num_huge);
                let not_special = {
                    let t = self.s.m.bool(true);
                    self.s.m.bin(c, Bin::Xor, special, t)
                };
                let up = self.s.m.bin(c, Bin::And, up_raw, not_special);
                let dn = self.s.m.bin(c, Bin::And, dn_raw, not_special);
                let uncancelled = {
                    let x_tiny = self.s.m.bin(c, Bin::Xor, den_tiny, num_tiny);
                    let x_huge = self.s.m.bin(c, Bin::Xor, den_huge, num_huge);
                    self.s.m.bin(c, Bin::Or, x_tiny, x_huge)
                };
                let flag_raw = self.s.m.select(c, is_den, uncancelled, up_raw);
                let flag = self.s.m.bin(c, Bin::And, flag_raw, not_special);
                let k_up = self.s.m.f32(0x5F80_0000);
                let k_dn = self.s.m.f32(0x1F80_0000);
                let a_up = self.s.m.bin(c, Bin::Mul, a, k_up);
                let a_dn = self.s.m.bin(c, Bin::Mul, a, k_dn);
                let r = self.s.m.select(c, dn, a_dn, a);
                let r = self.s.m.select(c, up, a_up, r);
                let qnan = self.s.m.f32(0x7FC0_0000);
                let r = self.s.m.select(c, any_zero, qnan, r);
                float(self, r)?;
                if let Some(Operand::Scalar { code, .. }) = v.carry
                    && let Some(w) = inst.write(Reg::S(code))
                {
                    self.lane_bits.insert(w, flag);
                }
                for &(_, w) in &inst.writes {
                    self.vals
                        .entry(w)
                        .or_insert(Val::Missing("a lane mask read as data"));
                }
                Ok(())
            }
            "v_div_fmas_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let m = inst
                    .read(Reg::S(crate::ir::VCC))
                    .and_then(|v| self.x.st.masks.get(&v).copied())
                    .ok_or("v_div_fmas_f32 without VCC")?;
                let vcc = self.boolean(m, c)?;
                let tmp = self.fused(a, bb, cc);
                let ci = self.as_int(cc);
                let (k23, k255, k128) = (self.s.m.i32(23), self.s.m.i32(0xFF), self.s.m.i32(128));
                let sh = self.s.m.bin(c, Bin::LShr, ci, k23);
                let exp = self.s.m.bin(c, Bin::And, sh, k255);
                let exp_ge = self.s.m.cmp(c, pred::IUGE, exp, k128);
                let (k_up, k_dn) = (self.s.m.f32(0x5F80_0000), self.s.m.f32(0x1F80_0000));
                let scale = self.s.m.select(c, exp_ge, k_up, k_dn);
                let scaled = self.s.m.bin(c, Bin::Mul, tmp, scale);
                let r = self.s.m.select(c, vcc, scaled, tmp);
                float(self, r)
            }
            "v_div_fixup_f32" => {
                let (q, den, num) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let (qi, di, ni) = (self.as_int(q), self.as_int(den), self.as_int(num));
                let (ksign, kabs, kinf) = (
                    self.s.m.i32(0x8000_0000),
                    self.s.m.i32(0x7FFF_FFFF),
                    self.s.m.i32(0x7F80_0000),
                );
                let (kqnan_bit, kneg_qnan, zero) = (
                    self.s.m.i32(0x0040_0000),
                    self.s.m.i32(0xFFC0_0000),
                    self.s.m.i32(0),
                );
                let sign = {
                    let x = self.s.m.bin(c, Bin::Xor, ni, di);
                    self.s.m.bin(c, Bin::And, x, ksign)
                };
                let q_abs = self.s.m.bin(c, Bin::And, qi, kabs);
                let d_abs = self.s.m.bin(c, Bin::And, di, kabs);
                let n_abs = self.s.m.bin(c, Bin::And, ni, kabs);
                let q_nan = self.s.m.cmp(c, pred::IUGT, q_abs, kinf);
                let d_nan = self.s.m.cmp(c, pred::IUGT, d_abs, kinf);
                let n_nan = self.s.m.cmp(c, pred::IUGT, n_abs, kinf);
                let d_inf = self.s.m.cmp(c, pred::EQ, d_abs, kinf);
                let n_inf = self.s.m.cmp(c, pred::EQ, n_abs, kinf);
                let d_zero = self.s.m.cmp(c, pred::EQ, d_abs, zero);
                let n_zero = self.s.m.cmp(c, pred::EQ, n_abs, zero);
                let both_inf = self.s.m.bin(c, Bin::And, n_inf, d_inf);
                let both_zero = self.s.m.bin(c, Bin::And, n_zero, d_zero);
                let invalid = self.s.m.bin(c, Bin::Or, both_inf, both_zero);
                let make_inf = {
                    let t = self.s.m.bin(c, Bin::Or, n_inf, d_zero);
                    self.s.m.bin(c, Bin::Or, t, q_nan)
                };
                let make_zero = self.s.m.bin(c, Bin::Or, d_inf, n_zero);
                let signed_q = self.s.m.bin(c, Bin::Or, sign, q_abs);
                let signed_inf = self.s.m.bin(c, Bin::Or, sign, kinf);
                let n_qnan = self.s.m.bin(c, Bin::Or, ni, kqnan_bit);
                let d_qnan = self.s.m.bin(c, Bin::Or, di, kqnan_bit);
                let r = self.s.m.select(c, make_zero, sign, signed_q);
                let r = self.s.m.select(c, make_inf, signed_inf, r);
                let r = self.s.m.select(c, invalid, kneg_qnan, r);
                let r = self.s.m.select(c, d_nan, d_qnan, r);
                let r = self.s.m.select(c, n_nan, n_qnan, r);
                let rf = self.as_float(r);
                float(self, rf)
            }
            "v_fmac_f32" | "v_mac_f32" | "v_fmac_legacy_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("an accumulator");
                };
                let acc = inst.read(Reg::V(reg)).ok_or("an accumulator")?;
                let acc = match self.floats.get(&acc) {
                    Some(&x) => x,
                    None => {
                        let x = self.i32_of(acc)?;
                        self.as_float(x)
                    }
                };
                let r = match name {
                    "v_fmac_f32" => self.fused(a, bb, acc),
                    "v_mac_f32" => {
                        let m = self.s.m.bin(c, Bin::Mul, a, bb);
                        self.s.m.bin(c, Bin::Add, m, acc)
                    }
                    _ => self.fma_legacy(a, bb, acc),
                };
                float(self, r)
            }
            "v_fma_legacy_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = self.fma_legacy(a, bb, cc);
                float(self, r)
            }
            "v_mad_f32" | "v_madak_f32" | "v_madmk_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let m = self.s.m.bin(c, Bin::Mul, a, bb);
                let r = self.s.m.bin(c, Bin::Add, m, cc);
                float(self, r)
            }
            "v_max3_f32" | "v_min3_f32" | "v_med3_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = match name {
                    "v_max3_f32" => {
                        let t = self.max_min(true, a, bb);
                        self.max_min(true, t, cc)
                    }
                    "v_min3_f32" => {
                        let t = self.max_min(false, a, bb);
                        self.max_min(false, t, cc)
                    }
                    _ => self.med3(a, bb, cc),
                };
                float(self, r)
            }
            "v_rcp_f32" | "v_rcp_iflag_f32" => {
                // shortcut: the hardware reciprocal is approximate (1 ULP); this divides.
                let a = f(self, 0)?;
                let one = self.s.m.f32(1.0f32.to_bits());
                let r = self.s.m.bin(c, Bin::SDiv, one, a);
                float(self, r)
            }
            "v_rsq_f32" | "v_sqrt_f32" | "v_log_f32" | "v_exp_f32" | "v_fract_f32"
            | "v_trunc_f32" | "v_floor_f32" | "v_ceil_f32" | "v_rndne_f32" | "v_sin_f32"
            | "v_cos_f32" => {
                let a = f(self, 0)?;
                let op = match name {
                    "v_rsq_f32" => 25,
                    "v_sqrt_f32" => 24,
                    "v_log_f32" => 23,
                    "v_exp_f32" => 21,
                    "v_fract_f32" => 22,
                    "v_trunc_f32" => 29,
                    "v_floor_f32" => 27,
                    "v_ceil_f32" => 28,
                    "v_rndne_f32" => 26,
                    "v_sin_f32" => 13,
                    _ => 12,
                };
                let r = match op {
                    12 | 13 => self.trig_turns(a, op == 12),
                    22 => self.fract(a, 1.0, f32::from_bits(0x3F7F_FFFF)),
                    _ => self.unary(op, a),
                };
                float(self, r)
            }
            "v_cvt_i32_f32" | "v_cvt_u32_f32" | "v_cvt_flr_i32_f32" | "v_cvt_rpi_i32_f32" => {
                let mut a = f(self, 0)?;
                if name == "v_cvt_rpi_i32_f32" {
                    let h = self.s.m.f32(0.5f32.to_bits());
                    a = self.s.m.bin(c, Bin::Add, a, h);
                }
                if name != "v_cvt_i32_f32" && name != "v_cvt_u32_f32" {
                    a = self.unary(27, a);
                }
                // D3D's conversions saturate and give 0 for NaN, as the
                // hardware does (checked on the RX 6700 XT and WARP).
                let op = if name == "v_cvt_u32_f32" {
                    Cast::FpToUi
                } else {
                    Cast::FpToSi
                };
                let r = self.s.m.cast(c, op, a, self.s.t.i32);
                int(self, r)
            }
            "v_cvt_f32_i32" | "v_cvt_f32_u32" | "v_cvt_f32_ubyte0" | "v_cvt_f32_ubyte1"
            | "v_cvt_f32_ubyte2" | "v_cvt_f32_ubyte3" => {
                let mut a = s(self, 0)?;
                if let Some(n) = name.strip_prefix("v_cvt_f32_ubyte") {
                    let sh = self.s.m.i32(8 * u32::from(n.as_bytes()[0] - b'0'));
                    let ff = self.s.m.i32(0xff);
                    a = self.s.m.bin(c, Bin::LShr, a, sh);
                    a = self.s.m.bin(c, Bin::And, a, ff);
                }
                let op = if name == "v_cvt_f32_i32" {
                    Cast::SiToFp
                } else {
                    Cast::UiToFp
                };
                let r = self.s.m.cast(c, op, a, self.s.t.f32);
                float(self, r)
            }
            // victory-fork 3bff182 (VIctory-7900XTX 342245e), GPL-3.0-or-later.
            "v_cvt_f32_f64" => {
                let (lo, hi) = self.f64_pair(inst, 0)?;
                let bits = self.f64_to_f32_bits(lo, hi);
                let r = self.as_float(bits);
                float(self, r)
            }
            "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => {
                // Each source is f32, or with op_sel_hi the low (op_sel: high)
                // f16 half; neg_hi is the absolute value here.
                let mut xs = Vec::new();
                for i in 0..3 {
                    let op = v.src.get(i).ok_or("a missing operand")?;
                    let x = self.operand(inst, op)?;
                    let mut x = if v.opsel_hi >> i & 1 != 0 {
                        let h = if v.opsel >> i & 1 != 0 {
                            let k16 = self.s.m.i32(16);
                            self.s.m.bin(c, Bin::LShr, x, k16)
                        } else {
                            x
                        };
                        self.s
                            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h])
                    } else {
                        self.as_float(x)
                    };
                    if v.neg_hi >> i & 1 != 0 {
                        x = self.unary(6, x);
                    }
                    if v.neg >> i & 1 != 0 {
                        let z = self.s.m.f32(0x8000_0000);
                        x = self.s.m.bin(c, Bin::Sub, z, x);
                    }
                    xs.push(x);
                }
                // f16 sources are exact in f32 (an f16 product fits), so
                // only the f32 form needs the exact path; it is fused.
                let mut r = self.fused(xs[0], xs[1], xs[2]);
                if name == "v_fma_mix_f32" {
                    return float(self, r);
                }
                if v.clamp {
                    r = self.unary(7, r);
                }
                // The f16 result goes to one half; the other half stays.
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a mixed-precision result");
                };
                let old = match inst.read(Reg::V(reg)) {
                    Some(o) => self.i32_of(o)?,
                    None => self.s.m.i32(0),
                };
                let h = self
                    .s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[r]);
                let (keep, h) = if name == "v_fma_mixhi_f16" {
                    let k16 = self.s.m.i32(16);
                    (0xFFFF, self.s.m.bin(c, Bin::Shl, h, k16))
                } else {
                    (0xFFFF_0000, h)
                };
                let keep = self.s.m.i32(keep);
                let old = self.s.m.bin(c, Bin::And, old, keep);
                let x = self.s.m.bin(c, Bin::Or, old, h);
                self.write(inst, Reg::V(reg), x, guard);
                Ok(())
            }
            "v_cvt_pkrtz_f16_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let lo = self.f16_toward_zero(a);
                let hi = self.f16_toward_zero(bb);
                let sh = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, hi, sh);
                let r = self.s.m.bin(c, Bin::Or, lo, hi);
                int(self, r)
            }
            "v_cvt_f16_f32" if v.opsel == 0 => {
                // Rounded to nearest even (the MODE default) into the low half.
                let a = f(self, 0)?;
                let r = self
                    .s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[a]);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a 16-bit result outside a VGPR");
                };
                let r = self.low_half(inst, reg, r);
                int(self, r)
            }
            "v_pack_b32_f16" => {
                // Two f16 sources (op_sel: their high halves; abs and neg on
                // the sign bit) into the low and high half.
                let mut halves = Vec::new();
                for k in 0..2 {
                    let op = v.src.get(k).ok_or("a missing operand")?;
                    let mut x = self.operand(inst, op)?;
                    if v.opsel >> k & 1 != 0 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let mask = if v.abs >> k & 1 != 0 { 0x7FFF } else { 0xFFFF };
                    let mask = self.s.m.i32(mask);
                    x = self.s.m.bin(c, Bin::And, x, mask);
                    if v.neg >> k & 1 != 0 {
                        let sign = self.s.m.i32(0x8000);
                        x = self.s.m.bin(c, Bin::Xor, x, sign);
                    }
                    halves.push(x);
                }
                let k16 = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
                let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
                int(self, r)
            }
            "v_cvt_f32_f16" if v.opsel == 0 => {
                let a = s(self, 0)?;
                let r = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[a]);
                float(self, r)
            }
            "v_cndmask_b32" => {
                let cond =
                    self.mask_cond(inst, v.src.get(2), c, "a select on an untracked mask")?;
                if v.abs != 0 || v.neg != 0 {
                    let (a, bb) = (f(self, 0)?, f(self, 1)?);
                    let r = self.s.m.select(c, cond, bb, a);
                    return float(self, r);
                }
                let (a, bb) = (s(self, 0)?, s(self, 1)?);
                let r = self.s.m.select(c, cond, bb, a);
                int(self, r)
            }
            "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" | "v_add_co_ci_u32"
            | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32" => {
                // The carry (borrow) out is a lane mask: this lane's bit.
                let (mut a, mut bb) = (s(self, 0)?, s(self, 1)?);
                if name.contains("subrev") {
                    std::mem::swap(&mut a, &mut bb);
                }
                let cin = if name.ends_with("_ci_u32") {
                    Some(self.mask_cond(
                        inst,
                        v.src.get(2),
                        c,
                        "a carry-in from an untracked mask",
                    )?)
                } else {
                    None
                };
                let (r, carry) = self.add_sub_carry(c, name.starts_with("v_add"), a, bb, cin);
                int(self, r)?;
                if let Some(Operand::Scalar { code, .. }) = v.carry
                    && let Some(w) = inst.write(Reg::S(code))
                {
                    self.lane_bits.insert(w, carry);
                }
                for &(_, w) in &inst.writes {
                    self.vals
                        .entry(w)
                        .or_insert(Val::Missing("a lane mask read as data"));
                }
                Ok(())
            }
            "v_readlane_b32" => {
                // XNW: a constant lane of a VGPR the program spilled an
                // SGPR to (`v_writelane_b32`) reads the SGPR spilled there.
                // (Outside compute only: compute reads lanes for real, and
                // the search through joins is costly on its large programs.)
                if self.s.threads.is_none()
                    && let (Some(src), Some(l)) = (v.src.first(), v.src.get(1))
                    && let Some(sv) = self.spilled_value(inst, src, l)
                {
                    let r = self.i32_of(sv)?;
                    return int(self, r);
                }
                let (a, lane) = (s(self, 0)?, s(self, 1)?);
                // A computed lane in a pixel or vertex program is a host lane
                // (from a mask or `v_mbcnt`); a constant one is the console's.
                if self.mask_data == MaskData::Host && !matches!(lane, V::Const(_)) {
                    self.s.flags |= flags::WAVE_OPS;
                } else {
                    self.cross_lane(false)?;
                }
                let r = self.read_lane(a, lane);
                int(self, r)
            }
            "v_writelane_b32" => {
                // XNW: outside compute, a lane write is only an SGPR
                // spill, which `v_readlane_b32` reads back as the SGPR; the
                // VGPR itself is left without a value.
                if self.s.threads.is_none() {
                    return Err("a lane write outside compute");
                }
                // The lane given takes the scalar; the others keep their value.
                let (a, lane) = (s(self, 0)?, s(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a lane write");
                };
                let old = match inst.read(Reg::V(reg)) {
                    Some(o) => self.i32_of(o)?,
                    None => self.s.m.i32(0),
                };
                self.cross_lane(false)?;
                let me = self.lane_index();
                let is = self.s.m.cmp(c, pred::EQ, me, lane);
                let r = self.s.m.select(c, is, a, old);
                int(self, r)
            }
            "v_readfirstlane_b32" => {
                let a = s(self, 0)?;
                self.s.flags |= flags::WAVE_OPS;
                let r = self.s.op(
                    c,
                    "waveReadLaneFirst.i32",
                    Attr::NoUnwind,
                    self.s.t.i32,
                    118,
                    &[a],
                );
                let r = self.helper_reads_own(a, r);
                int(self, r)
            }
            _ => {
                let r = self.integer(inst, name)?;
                int(self, r)
            }
        }
    }

    /// A 64-bit source (a register pair; inline constants sign-extend).
    pub(super) fn src64(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 || v.sdwa.is_some() || v.dpp.is_some() {
            return Err("64-bit operand modifiers");
        }
        let c = self.cur;
        let i64t = self.s.m.int(64);
        self.s.flags |= flags::INT64;
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let pair = |l: &mut Self, r: Reg, r1: Reg| -> Result<V, Why> {
            let lo = l.i32_of(inst.read(r).ok_or("an unread operand")?)?;
            let hi = l.i32_of(inst.read(r1).ok_or("an unread operand")?)?;
            let lo = l.s.m.cast(c, Cast::ZExt, lo, i64t);
            let hi = l.s.m.cast(c, Cast::ZExt, hi, i64t);
            let k32 = l.s.m.int_const(i64t, 32);
            let hi = l.s.m.bin(c, Bin::Shl, hi, k32);
            Ok(l.s.m.bin(c, Bin::Or, lo, hi))
        };
        match op {
            Operand::Vector { reg, .. } => pair(self, Reg::V(reg), Reg::V(reg + 1)),
            Operand::Scalar { code, .. } if code != NULL => {
                pair(self, Reg::S(code), Reg::S(code + 1))
            }
            Operand::Const(k @ 128..=208) => {
                let x = crate::resource::inline(k).ok_or("a special operand")?;
                Ok(self.s.m.int_const(i64t, i64::from(x as i32)))
            }
            _ => {
                // Literals and the null register: zero-extended.
                let x = self.operand(inst, &op)?;
                Ok(self.s.m.cast(c, Cast::ZExt, x, i64t))
            }
        }
    }
}

// victory-fork 3bff182 (VIctory-7900XTX 342245e: `v_cvt_f32_f64` in 32-bit IEEE-754 integer arithmetic),
// GPL-3.0-or-later, verbatim.
impl<'a> Lower<'a> {
    /// Reads an IEEE-754 binary64 operand as `(lo, hi)` 32-bit words, applying
    /// `abs` and `neg` input modifiers on operand `k`.
    pub(super) fn f64_pair(&mut self, inst: &Inst, k: usize) -> Result<(V, V), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.sdwa.is_some() || v.dpp.is_some() {
            return Err("64-bit operand modifiers");
        }
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let pair = |l: &mut Self, r: Reg, r1: Reg| -> Result<(V, V), Why> {
            let lo = l.i32_of(inst.read(r).ok_or("an unread operand")?)?;
            let hi = l.i32_of(inst.read(r1).ok_or("an unread operand")?)?;
            Ok((lo, hi))
        };
        let (lo, mut hi) = match op {
            Operand::Vector { reg, .. } => pair(self, Reg::V(reg), Reg::V(reg + 1))?,
            Operand::Scalar { code, .. } if code != NULL => {
                pair(self, Reg::S(code), Reg::S(code + 1))?
            }
            Operand::Const(128) => (self.s.m.i32(0), self.s.m.i32(0)),
            Operand::Const(c @ 240..=248) => {
                const INLINE_F64: [f64; 9] = [
                    0.5,
                    -0.5,
                    1.0,
                    -1.0,
                    2.0,
                    -2.0,
                    4.0,
                    -4.0,
                    1.0 / std::f64::consts::TAU,
                ];
                let bits = INLINE_F64[(c - 240) as usize].to_bits();
                (self.s.m.i32(bits as u32), self.s.m.i32((bits >> 32) as u32))
            }
            // GFX10 places a 32-bit literal in the high 32 bits of a 64-bit float operand.
            Operand::Literal(x) => (self.s.m.i32(0), self.s.m.i32(x)),
            _ => return Err("a 64-bit float operand"),
        };
        let c = self.cur;
        if v.abs >> k & 1 != 0 {
            let mask = self.s.m.i32(0x7FFF_FFFF);
            hi = self.s.m.bin(c, Bin::And, hi, mask);
        }
        if v.neg >> k & 1 != 0 {
            let sign = self.s.m.i32(0x8000_0000);
            hi = self.s.m.bin(c, Bin::Xor, hi, sign);
        }
        Ok((lo, hi))
    }

    /// Converts IEEE-754 binary64 `(lo, hi)` to binary32 bits with round-to-nearest-even,
    /// without requiring host `DOUBLES` (`f64`) support.
    pub(super) fn f64_to_f32_bits(&mut self, lo: V, hi: V) -> V {
        let c = self.cur;
        let k0 = self.s.m.i32(0);
        let k1 = self.s.m.i32(1);
        let sign_mask = self.s.m.i32(0x8000_0000);
        let sign = self.s.m.bin(c, Bin::And, hi, sign_mask);

        let k20 = self.s.m.i32(20);
        let k7ff = self.s.m.i32(0x7FF);
        let exp64 = self.s.m.bin(c, Bin::LShr, hi, k20);
        let exp64 = self.s.m.bin(c, Bin::And, exp64, k7ff);

        let frac_mask = self.s.m.i32(0x000F_FFFF);
        let frac_hi = self.s.m.bin(c, Bin::And, hi, frac_mask);
        let frac_any = self.s.m.bin(c, Bin::Or, frac_hi, lo);
        let frac_nz = self.s.m.cmp(c, pred::NE, frac_any, k0);

        let k3 = self.s.m.i32(3);
        let k29 = self.s.m.i32(29);
        let m23_hi = self.s.m.bin(c, Bin::Shl, frac_hi, k3);
        let m23_lo = self.s.m.bin(c, Bin::LShr, lo, k29);
        let m23 = self.s.m.bin(c, Bin::Or, m23_hi, m23_lo);

        let rem_mask = self.s.m.i32(0x1FFF_FFFF);
        let rem = self.s.m.bin(c, Bin::And, lo, rem_mask);
        let half_rem = self.s.m.i32(0x1000_0000);
        let gt_half = self.s.m.cmp(c, pred::IUGT, rem, half_rem);
        let eq_half = self.s.m.cmp(c, pred::EQ, rem, half_rem);
        let m23_odd_bit = self.s.m.bin(c, Bin::And, m23, k1);
        let m23_odd = self.s.m.cmp(c, pred::NE, m23_odd_bit, k0);
        let tie_up = self.s.m.bin(c, Bin::And, eq_half, m23_odd);
        let norm_round = self.s.m.bin(c, Bin::Or, gt_half, tie_up);
        let norm_inc = self.s.m.select(c, norm_round, k1, k0);

        // Normal range: exp64 in [897, 1150] -> exp32 in [1, 254].
        let k896 = self.s.m.i32(896);
        let k23 = self.s.m.i32(23);
        let exp32 = self.s.m.bin(c, Bin::Sub, exp64, k896);
        let exp32_sh = self.s.m.bin(c, Bin::Shl, exp32, k23);
        let norm_body = self.s.m.bin(c, Bin::Add, exp32_sh, m23);
        let norm_body = self.s.m.bin(c, Bin::Add, norm_body, norm_inc);
        let norm_res = self.s.m.bin(c, Bin::Or, sign, norm_body);

        // Subnormal range: exp64 in [874, 896] -> shift in [1, 23].
        let k897 = self.s.m.i32(897);
        let k32 = self.s.m.i32(32);
        let k31 = self.s.m.i32(31);
        let imp = self.s.m.i32(0x0080_0000);
        let sig24 = self.s.m.bin(c, Bin::Or, imp, m23);
        let raw_shift = self.s.m.bin(c, Bin::Sub, k897, exp64);
        // Clamp shift to [1, 23] so shifts never produce poison outside the subnormal branch.
        let lt1 = self.s.m.cmp(c, pred::IULT, raw_shift, k1);
        let shift = self.s.m.select(c, lt1, k1, raw_shift);
        let gt23 = self.s.m.cmp(c, pred::IUGT, shift, k23);
        let shift = self.s.m.select(c, gt23, k23, shift);
        let sub_m = self.s.m.bin(c, Bin::LShr, sig24, shift);
        let inv_shift = self.s.m.bin(c, Bin::Sub, k32, shift);
        let lost = self.s.m.bin(c, Bin::Shl, sig24, inv_shift);
        let round_bit = self.s.m.bin(c, Bin::LShr, lost, k31);
        let round_on = self.s.m.cmp(c, pred::NE, round_bit, k0);
        let low31 = self.s.m.i32(0x7FFF_FFFF);
        let lost_lo = self.s.m.bin(c, Bin::And, lost, low31);
        let sticky = self.s.m.bin(c, Bin::Or, lost_lo, rem);
        let sticky_on = self.s.m.cmp(c, pred::NE, sticky, k0);
        let sub_odd_bit = self.s.m.bin(c, Bin::And, sub_m, k1);
        let sub_odd = self.s.m.cmp(c, pred::NE, sub_odd_bit, k0);
        let sub_tie = self.s.m.bin(c, Bin::Or, sticky_on, sub_odd);
        let sub_round = self.s.m.bin(c, Bin::And, round_on, sub_tie);
        let sub_inc = self.s.m.select(c, sub_round, k1, k0);
        let sub_body = self.s.m.bin(c, Bin::Add, sub_m, sub_inc);
        let sub_res = self.s.m.bin(c, Bin::Or, sign, sub_body);

        // Underflow range: exp64 < 874. Only exp64 == 873 with non-zero fraction rounds up to min subnormal (1).
        let k873 = self.s.m.i32(873);
        let eq873 = self.s.m.cmp(c, pred::EQ, exp64, k873);
        let round_min = self.s.m.bin(c, Bin::And, eq873, frac_nz);
        let min_bit = self.s.m.select(c, round_min, k1, k0);
        let under_res = self.s.m.bin(c, Bin::Or, sign, min_bit);

        // Overflow / Inf / NaN: exp64 >= 1151.
        let is_exp_all = self.s.m.cmp(c, pred::EQ, exp64, k7ff);
        let is_nan = self.s.m.bin(c, Bin::And, is_exp_all, frac_nz);
        let inf_bits = self.s.m.i32(0x7F80_0000);
        let qnan_bits = self.s.m.i32(0x7FC0_0000);
        let nan_body = self.s.m.bin(c, Bin::Or, qnan_bits, m23);
        let over_body = self.s.m.select(c, is_nan, nan_body, inf_bits);
        let over_res = self.s.m.bin(c, Bin::Or, sign, over_body);

        // Select final result by exponent range.
        let k874 = self.s.m.i32(874);
        let k1151 = self.s.m.i32(1151);
        let is_under = self.s.m.cmp(c, pred::IULT, exp64, k874);
        let is_sub = self.s.m.cmp(c, pred::IULT, exp64, k897);
        let is_over = self.s.m.cmp(c, pred::IUGE, exp64, k1151);
        let r = self.s.m.select(c, is_over, over_res, norm_res);
        let r = self.s.m.select(c, is_sub, sub_res, r);
        self.s.m.select(c, is_under, under_res, r)
    }

}
