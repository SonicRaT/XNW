// Part of emit/lower.rs: buffer ops, paged addresses.
// Origin as in lower.rs's header: VIctory crates/victory-shader/src/emit/lower.rs, d2a4bae1f495, and the
// later VIctory ports named there; GPL-3.0-or-later. Moved out of lower.rs unchanged (XNW BLD2): one stage,
// one codegen unit.
// With victory-fork (open pull request, head 5a55384; GPL-3.0-or-later): halves through views of a table.
use super::*;

impl<'a> Lower<'a> {
    pub(super) fn buffer(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        self.buffer_site = Some((b, i));
        let Fields::Buffer {
            data,
            addr,
            soffset,
            offset,
            offen,
            idxen,
            format,
            lds,
            tfe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if *lds || *tfe {
            return Err("a buffer access to LDS or with tfe");
        }
        if format.is_some() && !name.starts_with("tbuffer_load_format_") {
            return Err(name);
        }
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let store = name.starts_with("buffer_store_");
        // DXIL atomic operations: add, and, or, xor, signed and unsigned
        // min and max, exchange; compare-exchange apart.
        let atomic = match name.strip_prefix("buffer_atomic_") {
            None => None,
            Some("add") => Some(0),
            Some("and") => Some(1),
            Some("or") => Some(2),
            Some("xor") => Some(3),
            Some("smin") => Some(4),
            Some("smax") => Some(5),
            Some("umin") => Some(6),
            Some("umax") => Some(7),
            Some("swap") => Some(8),
            Some("cmpswap") => Some(u32::MAX),
            Some(_) => return Err(name),
        };
        // d16: two 16-bit channels per VGPR, low half first.
        let d16 = name.starts_with("buffer_") && name.contains("_format_d16_");
        let plain = name.replace("_format_d16_", "_format_");
        // XNW: bytes and halves a load reads within their dword
        // (bits, signed), as `global_load_*` does (VIctory).
        let narrow = match name {
            "buffer_store_byte" => Some((8u32, false)),
            "buffer_store_short" => Some((16, false)),
            "buffer_load_ubyte" => Some((8u32, false)),
            "buffer_load_sbyte" => Some((8, true)),
            "buffer_load_ushort" => Some((16, false)),
            "buffer_load_sshort" => Some((16, true)),
            _ => None,
        };
        let (typed, count) = match plain
            .strip_prefix("buffer_load_")
            .or(plain.strip_prefix("buffer_store_"))
            .or(plain.strip_prefix("tbuffer_load_"))
        {
            _ if atomic.is_some() => (false, 1),
            _ if narrow.is_some() => (false, 1),
            Some("dword") => (false, 1),
            Some("dwordx2") => (false, 2),
            Some("dwordx3") => (false, 3),
            Some("dwordx4") => (false, 4),
            Some("format_x") => (true, 1),
            Some("format_xy") => (true, 2),
            Some("format_xyz") => (true, 3),
            Some("format_xyzw") => (true, 4),
            _ => return Err(name),
        };
        let Operand::Vector { reg: areg, .. } = *addr else {
            return Err("a buffer address");
        };
        let vaddr = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a buffer address")?;
            l.i32_of(v)
        };
        let index = if *idxen { Some(vaddr(self, 0)?) } else { None };
        let mut off = self.s.m.i32(u32::from(*offset));
        if *offen {
            let o = vaddr(self, u8::from(*idxen))?;
            off = self.s.m.bin(c, Bin::Add, off, o);
        }
        if !matches!(
            soffset,
            Operand::Scalar { code: NULL, .. } | Operand::Const(128)
        ) {
            let so = self.operand(inst, soffset)?;
            off = self.s.m.bin(c, Bin::Add, off, so);
        }
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked buffer")?;
        let used = self.used_kind(src);
        // Sixteen bits a channel through a view of a table: the bits go as
        // they are, as unsigned integers, and the host is told, which shows
        // the shader a 16-bit float view there as its unsigned twin
        // ([`super::super::HostShader::half_views`]).
        // A format load of whole dwords whose every use reads one half of a
        // result as a half float moves 16-bit integers too (the head
        // program's reads of the face network's output), where every
        // access through its view does ([`Self::places_read_as_halves`]).
        // A d16 access too only where every access does: a whole float
        // read through the same place would see the unsigned twin.
        let place = typed.then(|| self.table_place(src)).flatten().filter(|p| {
            self.halves_only.contains(p) && (d16 || self.format_load_read_as_halves(inst))
        });
        let halves = place.is_some();
        if let Some(place) = place {
            self.half_views.insert(place);
        }
        let (base, extra) = self.built_offset(src)?;
        if base != 0 {
            let b = self.s.m.i32(base);
            off = self.s.m.bin(c, Bin::Add, off, b);
        }
        if let Some(x) = extra {
            off = self.s.m.bin(c, Bin::Add, off, x);
        }
        // stage b: a root view by address (not by index: its stride is the
        // code's, not a record's).
        let root = if typed || index.is_some() { None } else { self.root_view_base(src) };
        let paged = !typed && (self.paged(src) || root.is_some());
        if !paged {
            self.note_unpaged(src, typed);
        }
        if paged {
            self.declared_stride = used.map(|u| u.stride);
            // XNW: stages a and d): the page table is the only path
            // for raw loads, stores and atomics through a record.
            let uniform = !*offen && !*idxen;
            // A constant buffer indexed per lane: 16-byte rows (as below).
            let (index, off) = match index {
                Some(i) if used.is_some_and(|u| u.kind == ResourceType::Cbv) => {
                    let sixteen = self.s.m.i32(16);
                    let rows = self.s.m.bin(c, Bin::Mul, i, sixteen);
                    (None, self.s.m.bin(c, Bin::Add, rows, off))
                }
                other => (other, off),
            };
            if let Some(op) = atomic {
                let d = data.as_ref().ok_or("buffer atomic data")?;
                let vals = self.vgprs(inst, d, if op == u32::MAX { 2 } else { 1 })?;
                let Operand::Vector { reg, .. } = *d else {
                    return Err("buffer atomic data");
                };
                let returns = matches!(inst.fields, Fields::Buffer { cache, .. } if cache.glc);
                let old = match inst.read(Reg::V(reg)) {
                    Some(v) if returns => self.i32_of(v)?,
                    _ => self.s.m.i32(0),
                };
                let r = self.page_atomic(site, root, index, off, op, &vals, old, uniform, guard)?;
                if returns {
                    self.write(inst, Reg::V(reg), r, None);
                }
                return Ok(());
            }
            if store {
                let vals = self.vgprs(inst, data.as_ref().ok_or("buffer store data")?, count as u8)?;
                return self.page_store(site, root, index, off, &vals, uniform, guard);
            }
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            for (k, x) in self.page_load(site, root, index, off, count, uniform)?.into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        // XNW: a structured access through a root descriptor (the BVH
        // builder's root UAVs, stride in the program's metadata) is a raw
        // access at index × stride: D3D12 root views take no stride.
        let mut index = index;
        if let (Some(i), Source::Built { address, .. }, Some(u)) = (index, src, used)
            && !typed
            && u.stride >= 4
            && u.stride.is_multiple_of(4)
            && self.x.native.layout.get(address.param).is_some_and(|r| r.root == RootType::Descriptor)
            && u.kind != ResourceType::Cbv
        {
            let stride = self.s.m.i32(u.stride);
            let at = self.s.m.bin(c, Bin::Mul, i, stride);
            off = self.s.m.bin(c, Bin::Add, at, off);
            index = None;
        }
        // XNW: a byte or half: the raw dword holding it, shifted down.
        // shortcut: the field is taken not to cross a dword; an indexed one
        // needs the stride the program's metadata names.
        if let Some((bits, signed)) = narrow {
            if let Some(i) = index {
                let stride = used.map_or(0, |u| u.stride);
                if stride == 0 {
                    return Err("a byte or half of a structured buffer of unknown stride");
                }
                let k = self.s.m.i32(stride);
                let at = self.s.m.bin(c, Bin::Mul, i, k);
                off = self.s.m.bin(c, Bin::Add, at, off);
            }
            let u = if store || used.is_some_and(|u| u.kind == ResourceType::Uav) { 0x1000 } else { 0 };
            let h = self.handle(site, src, false, (11 | u, 0))?;
            let (three, not3) = (self.s.m.i32(3), self.s.m.i32(!3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            if store {
                // Two atomics, `and` then `or`, so lanes writing the other
                // bytes of the dword keep theirs (as LDS bytes, VIctory).
                let v = self.vgprs(inst, data.as_ref().ok_or("buffer store data")?, 1)?[0];
                let c = self.cur;
                let byte = self.s.m.bin(c, Bin::And, off, three);
                let sh = self.s.m.bin(c, Bin::Shl, byte, three);
                let mask = self.s.m.i32((1u32 << bits) - 1);
                let v = self.s.m.bin(c, Bin::And, v, mask);
                let v = self.s.m.bin(c, Bin::Shl, v, sh);
                let hole = self.s.m.bin(c, Bin::Shl, mask, sh);
                let all = self.s.m.i32(u32::MAX);
                let keep = self.s.m.bin(c, Bin::Xor, hole, all);
                let (iu, i32t) = (self.s.m.undef(self.s.t.i32), self.s.t.i32);
                let (and, or) = (self.s.m.i32(1), self.s.m.i32(2));
                let g = self.begin_guarded(guard);
                self.s.op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &[h, and, at, iu, iu, keep]);
                self.s.op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &[h, or, at, iu, iu, v]);
                self.end_guarded(g, &[]);
                return Ok(());
            }
            let w = self.raw_dword(h, at);
            let c = self.cur;
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            let x = self.s.m.bin(c, Bin::LShr, w, sh);
            let left = self.s.m.i32(32 - bits);
            let x = self.s.m.bin(c, Bin::Shl, x, left);
            let x = self.s.m.bin(c, if signed { Bin::AShr } else { Bin::LShr }, x, left);
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            self.write(inst, Reg::V(reg), x, guard);
            return Ok(());
        }
        if used.is_some_and(|u| u.kind == ResourceType::Cbv) && !typed {
            // A constant buffer read per lane (a dynamically indexed array).
            // shortcut: the console's CBV descriptor is taken to have a 16-byte
            // stride, as the vector loads of whole rows suggest.
            let gpu = self.x.gpu_desc;
            let h = self.handle(site, src, false, if gpu { (11, 0) } else { (13, 0x1_0000) })?;
            let byte = match index {
                Some(i) => {
                    let sixteen = self.s.m.i32(16);
                    let rows = self.s.m.bin(c, Bin::Mul, i, sixteen);
                    self.s.m.bin(c, Bin::Add, rows, off)
                }
                None => off,
            };
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            for k in 0..count {
                let four = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, byte, four);
                let x = if gpu { self.raw_dword(h, a) } else { self.cbuffer_dword(h, a) };
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let uav = store || atomic.is_some() || used.is_some_and(|u| u.kind == ResourceType::Uav);
        let u = if uav { 0x1000 } else { 0 };
        let iu = self.s.m.undef(self.s.t.i32);
        if let Some(op) = atomic {
            let d = data.as_ref().ok_or("buffer atomic data")?;
            let vals = self.vgprs(inst, d, if op == u32::MAX { 2 } else { 1 })?;
            let (h, coords) = match index {
                Some(index) => {
                    let stride = used.map_or(0, |u| u.stride);
                    if stride != 0 && (stride < 4 || !stride.is_multiple_of(4)) {
                        return Err("a structured buffer of unknown stride");
                    }
                    if stride >= 4 && stride.is_multiple_of(4) {
                        (
                            self.handle(site, src, false, (12 | u, stride))?,
                            [index, off, iu],
                        )
                    } else {
                        let (h, byte) = self.runtime_view(src, index, off, 11 | u)?;
                        (h, [byte, iu, iu])
                    }
                }
                None => (self.handle(site, src, false, (11 | u, 0))?, [off, iu, iu]),
            };
            let Operand::Vector { reg, .. } = *d else {
                return Err("buffer atomic data");
            };
            let returns = matches!(inst.fields, Fields::Buffer { cache, .. } if cache.glc);
            let old = match inst.read(Reg::V(reg)) {
                Some(v) if returns => self.i32_of(v)?,
                _ => self.s.m.i32(0),
            };
            let at = self.begin_guarded(guard);
            let i32t = self.s.t.i32;
            let r = if op == u32::MAX {
                // cmpswap: data is (new, compare).
                let mut args = vec![h];
                args.extend(coords);
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
                let mut args = vec![h, self.s.m.i32(op)];
                args.extend(coords);
                args.push(vals[0]);
                self.s
                    .op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &args)
            };
            let r = self.end_guarded(at, &[(r, old)])[0];
            if returns {
                self.write(inst, Reg::V(reg), r, None);
            }
            return Ok(());
        }
        if store {
            let regs = if d16 { count.div_ceil(2) } else { count };
            let mut vals =
                self.vgprs(inst, data.as_ref().ok_or("buffer store data")?, regs as u8)?;
            if d16 {
                // Through a view of a table each half is stored as it is
                // (`halves`). Reached another way:
                // shortcut: d16 data is taken as f16 (integer formats would
                // keep 16 bits), widened for the 32-bit float store.
                let mut wide = Vec::new();
                for k in 0..count {
                    let mut x = vals[(k / 2) as usize];
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    if halves {
                        let low = self.s.m.i32(0xffff);
                        wide.push(self.s.m.bin(c, Bin::And, x, low));
                        continue;
                    }
                    let f32t = self.s.t.f32;
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                    wide.push(self.as_int(f));
                }
                vals = wide;
            }
            let mut args = Vec::new();
            let (op, name) = if typed {
                // shortcut: typed stores are taken to write all four channels
                // of a float format.
                if count != 4 || *offen || *offset != 0 {
                    return Err("a typed buffer store of some channels or at an offset");
                }
                // Four unsigned integers for halves, four floats otherwise.
                let component = if halves { 0x405 } else { 0x409 };
                args.push(self.handle(site, src, false, (10 | u, component))?);
                args.extend([index.unwrap_or(self.s.m.i32(0)), iu]);
                for x in vals {
                    args.push(if halves { x } else { self.as_float(x) });
                }
                (
                    69,
                    if halves {
                        "bufferStore.i32"
                    } else {
                        "bufferStore.f32"
                    },
                )
            } else {
                match index {
                    Some(index) => {
                        let stride = used.map_or(0, |u| u.stride);
                        if stride >= 4 && stride.is_multiple_of(4) {
                            args.push(self.handle(site, src, false, (12 | u, stride))?);
                            args.extend([index, off]);
                        } else {
                            let (h, byte) = self.runtime_view(src, index, off, 11 | u)?;
                            args.push(h);
                            args.extend([byte, iu]);
                        }
                    }
                    None => {
                        args.push(self.handle(site, src, false, (11 | u, 0))?);
                        args.extend([off, iu]);
                    }
                }
                args.extend(&vals);
                args.resize(3 + 4, iu);
                (140, "rawBufferStore.i32")
            };
            args.push(self.s.m.i8(if typed { 0xF } else { (1u8 << count) - 1 }));
            if !typed {
                args.push(self.s.m.i32(4));
            }
            let at = self.begin_guarded(guard);
            let void = self.s.t.void;
            self.s.op(self.cur, name, Attr::NoUnwind, void, op, &args);
            self.end_guarded(at, &[]);
            return Ok(());
        }
        if let Some(f) = *format {
            if index.is_some() {
                return Err("a typed-format load by index");
            }
            let h = self.handle(site, src, false, (11 | u, 0))?;
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            for (k, x) in self.format_load(h, off, f, count)?.into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let r = if typed {
            if *offen || matches!(off, V::Inst(_)) || *offset != 0 {
                return Err("a typed buffer load at an offset");
            }
            if uav {
                self.s.flags |= flags::TYPED_UAV_LOADS;
            }
            let index = index.unwrap_or(self.s.m.i32(0));
            if halves {
                let h = self.handle(site, src, false, (10 | u, 0x405))?;
                let i32t = self.s.t.i32;
                let ret = self.s.m.named_struct("dx.types.ResRet.i32", vec![i32t; 5]);
                self.s.op(
                    c,
                    "bufferLoad.i32",
                    Attr::ReadOnly,
                    ret,
                    68,
                    &[h, index, iu],
                )
            } else {
                let h = self.handle(site, src, false, (10 | u, 0x409))?;
                self.s.op(
                    c,
                    "bufferLoad.f32",
                    Attr::ReadOnly,
                    self.s.t.res_ret,
                    68,
                    &[h, index, iu],
                )
            }
        } else {
            let mask = self.s.m.i8((1u8 << count) - 1);
            let align = self.s.m.i32(4);
            match index {
                Some(index) => {
                    let stride = used.map_or(0, |u| u.stride);
                    if stride != 0 && (stride < 4 || !stride.is_multiple_of(4)) {
                        return Err("a structured buffer of unknown stride");
                    }
                    if stride >= 4 && stride.is_multiple_of(4) {
                        let h = self.handle(site, src, false, (12 | u, stride))?;
                        self.s.op(
                            c,
                            "rawBufferLoad.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            139,
                            &[h, index, off, mask, align],
                        )
                    } else {
                        let (h, byte) = self.runtime_view(src, index, off, 11 | u)?;
                        self.s.op(
                            c,
                            "rawBufferLoad.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            139,
                            &[h, byte, iu, mask, align],
                        )
                    }
                }
                None => {
                    let h = self.handle(site, src, false, (11 | u, 0))?;
                    self.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, off, iu, mask, align],
                    )
                }
            }
        };
        let Some(Operand::Vector { reg, .. }) = *data else {
            return Err("a buffer result");
        };
        if d16 {
            // Two f16 channels per VGPR (an odd last high half 0); through
            // a view of a table, the 16 bits of each as they are.
            let k16 = self.s.m.i32(16);
            let i32t = self.s.t.i32;
            for k in 0..count.div_ceil(2) {
                let mut x = None;
                for ch in (2 * k..(2 * k + 2).min(count)).rev() {
                    let f = self.s.m.extract(c, r, ch);
                    let h = if halves {
                        let low = self.s.m.i32(0xffff);
                        self.s.m.bin(c, Bin::And, f, low)
                    } else {
                        self.s
                            .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
                    };
                    x = Some(match x {
                        Some(hi) => {
                            let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                            self.s.m.bin(c, Bin::Or, h, hi)
                        }
                        None => h,
                    });
                }
                self.write(inst, Reg::V(reg + k as u8), x.expect("a channel"), guard);
            }
            return Ok(());
        }
        for k in 0..count {
            let x = self.s.m.extract(c, r, k);
            // (Integers already where the load moved halves.)
            let x = if halves { x } else { self.as_int(x) };
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// XNW: with the page table on, an access through `src` (and
    /// `typed`) that does not go through it: its kind joins the program's
    /// host-copy bits ([`super::super::host_copies`]).
    pub(super) fn note_unpaged(&mut self, src: &Source, typed: bool) {
        if !(self.x.pagetable && self.x.gpu_desc) {
            return;
        }
        use super::super::host_copies as h;
        let bit = if typed {
            h::TYPED_BUFFERS
        } else {
            match src {
                Source::Built { address, .. }
                    if self.x.native.layout.get(address.param).is_some_and(|r| r.root == RootType::Descriptor) =>
                {
                    h::ROOT_VIEWS
                }
                Source::Built { .. } | Source::Constant(_) => h::BUILT_DESCRIPTORS,
                _ => return,
            }
        };
        let b = self.s.host_copies.get_or_insert(0);
        *b |= bit;
    }

    /// XNW: whether accesses through `src` go through the page
    /// table only: the switch is on, descriptors are GPU-side, and `src` is
    /// a record (not a root descriptor or one the code builds).
    pub(super) fn paged(&self, src: &Source) -> bool {
        self.x.pagetable && self.x.gpu_desc && !matches!(src, Source::Built { .. } | Source::Constant(_))
    }

    /// XNW: stage b): for a buffer the code builds around a root
    /// descriptor (root SRV/CBV/UAV by address) with the page table on, the
    /// view's console address, from the two user-data dwords the runtime
    /// fills for it (spec section 6: root type Descriptor, dword 0 = bits
    /// 0-31, dword 1 = bits 32-63).
    pub(super) fn root_view_base(&mut self, src: &Source) -> Option<(V, V)> {
        if !(self.x.pagetable && self.x.gpu_desc) {
            return None;
        }
        let Source::Built { address, .. } = src else { return None };
        let param = address.param;
        let rp = *self.x.native.layout.get(param)?;
        if rp.root != RootType::Descriptor {
            return None;
        }
        let index = u32::from(rp.user_data);
        let lo = self.user_extra(param, index, 0);
        let hi = self.user_extra(param, index + 1, 1);
        Some((lo, hi))
    }

    /// XNW (page table): the console
    /// address of an access of `count` dwords through the record at `site`,
    /// a GFX10 V# tagged [`super::super::PAGE_TAG`]: base (dword 0, dword 1 bits
    /// 0-15) + index × stride (dword 1 bits 16-29) + `off` in 64 bits, and
    /// whether it is to run: the tag is there and the access in range by
    /// num_records (dword 2) as GFX10 checks it (structured: the index; raw:
    /// the bytes). A record without the tag runs nothing (reads 0).
    pub(super) fn page_address(&mut self, site: Site, root: Option<(V, V)>, index: Option<V>, off: V, count: u32) -> Result<(V, V, V), Why> {
        if let Some((base_lo, base_hi)) = root {
            // A root view: base + offset, no record and no range (the runtime
            // maps the view's pages up to its buffer's end).
            let c = self.cur;
            let lo = self.s.m.bin(c, Bin::Add, base_lo, off);
            let carry = self.s.m.cmp(c, pred::IULT, lo, base_lo);
            let carry = self.s.m.cast(c, Cast::ZExt, carry, self.s.t.i32);
            let hi = self.s.m.bin(c, Bin::Add, base_hi, carry);
            let _ = (site, index, count);
            return Ok((lo, hi, self.s.m.bool(true)));
        }
        let [w0, w1, w2, w3] = self.record_words(site)?;
        let c = self.cur;
        let k = |l: &mut Self, n: u32| l.s.m.i32(n);
        let (tag, k20) = (k(self, super::super::PAGE_TAG), k(self, 20));
        let high = self.s.m.bin(c, Bin::LShr, w3, k20);
        let is_page = self.s.m.cmp(c, pred::EQ, high, tag);
        let (m16, k16, m14) = (k(self, 0xFFFF), k(self, 16), k(self, 0x3FFF));
        let base_hi = self.s.m.bin(c, Bin::And, w1, m16);
        // The program's declared stride where it has one (spec 4: a record
        // may carry stride 0 for strides of 256 or more, which would fail
        // the title's own dword-1 check), else the record's.
        let stride = match self.declared_stride.take() {
            Some(s) if s > 0 => k(self, s),
            _ => {
                let s = self.s.m.bin(c, Bin::LShr, w1, k16);
                self.s.m.bin(c, Bin::And, s, m14)
            }
        };
        let zero = k(self, 0);
        let end = k(self, 4 * count);
        let last = self.s.m.bin(c, Bin::Add, off, end);
        let by_bytes = self.s.m.cmp(c, pred::IULE, last, w2);
        let (byte, in_range) = match index {
            Some(i) => {
                let e = self.s.m.bin(c, Bin::Mul, i, stride);
                let byte = self.s.m.bin(c, Bin::Add, e, off);
                let structured = self.s.m.cmp(c, pred::NE, stride, zero);
                let by_index = self.s.m.cmp(c, pred::IULT, i, w2);
                (byte, self.s.m.select(c, structured, by_index, by_bytes))
            }
            None => (off, by_bytes),
        };
        let lo = self.s.m.bin(c, Bin::Add, w0, byte);
        let carry = self.s.m.cmp(c, pred::IULT, lo, w0);
        let carry = self.s.m.cast(c, Cast::ZExt, carry, self.s.t.i32);
        let hi = self.s.m.bin(c, Bin::Add, base_hi, carry);
        let go = self.s.m.bin(c, Bin::And, is_page, in_range);
        Ok((lo, hi, go))
    }

    /// XNW: the page table's entry for console address (lo, hi): the
    /// handle of the page's raw SRV (`uav`: its UAV, index + 1), the byte of
    /// the address in it, and whether the page is mapped (if not, the handle
    /// is the null raw view, host slot 2 or 6). One lookup at the access's
    /// first byte; `uniform`: the same address in every lane.
    pub(super) fn page_entry(&mut self, lo: V, hi: V, uav: bool, uniform: bool) -> (V, V, V) {
        let c = self.cur;
        let t = self.s.page_table_handle();
        let k = |l: &mut Self, n: u32| l.s.m.i32(n);
        let (spans, none, four, zero) = (k(self, 4096), k(self, u32::MAX), k(self, 4), k(self, 0));
        let inside = self.s.m.cmp(c, pred::IULT, hi, spans);
        let span = self.s.m.select(c, inside, hi, zero);
        let at = self.s.m.bin(c, Bin::Mul, span, four);
        let block = self.raw_dword(t, at);
        let block = self.s.m.select(c, inside, block, none);
        let has = self.s.m.cmp(c, pred::NE, block, none);
        let blk = self.s.m.select(c, has, block, zero);
        let (k16, k3, top, size) = (k(self, 16), k(self, 3), k(self, 16384), k(self, 524_288));
        let page = self.s.m.bin(c, Bin::LShr, lo, k16);
        let page = self.s.m.bin(c, Bin::Shl, page, k3);
        let first = self.s.m.bin(c, Bin::Mul, blk, size);
        let e = self.s.m.bin(c, Bin::Add, first, page);
        let e = self.s.m.bin(c, Bin::Add, e, top);
        let mut index = self.raw_dword(t, e);
        let e4 = self.s.m.bin(c, Bin::Add, e, four);
        let offset = self.raw_dword(t, e4);
        let named = self.s.m.cmp(c, pred::NE, index, none);
        let mapped = self.s.m.bin(c, Bin::And, has, named);
        if uav {
            let one = k(self, 1);
            index = self.s.m.bin(c, Bin::Add, index, one);
        }
        let null = k(self, if uav { 6 } else { 2 });
        let index = self.s.m.select(c, mapped, index, null);
        let props = (if uav { 0x100B } else { 11 }, 0);
        let h = if uniform {
            self.s.heap_handle(c, index, false, props)
        } else {
            self.s.heap_handle_nonuniform(c, index, props)
        };
        let m16 = k(self, 0xFFFF);
        let low = self.s.m.bin(c, Bin::And, lo, m16);
        let byte = self.s.m.bin(c, Bin::Add, offset, low);
        (h, byte, mapped)
    }

    /// XNW: `count` dwords through the record at `site` and the
    /// page table, the only path (0 where the record has no tag, the access
    /// is out of range or the page is missing).
    pub(super) fn page_load(&mut self, site: Site, root: Option<(V, V)>, index: Option<V>, off: V, count: u32, uniform: bool) -> Result<Vec<V>, Why> {
        let (lo, hi, go) = self.page_address(site, root, index, off, count)?;
        let zero = self.s.m.i32(0);
        let at = self.begin_guarded(Some(go));
        let (h, byte, mapped) = self.page_entry(lo, hi, false, uniform);
        let c = self.cur;
        let mut vals = Vec::new();
        for n in 0..count {
            let d = self.s.m.i32(4 * n);
            let a = self.s.m.bin(c, Bin::Add, byte, d);
            let x = self.raw_dword(h, a);
            vals.push((self.s.m.select(c, mapped, x, zero), zero));
        }
        Ok(self.end_guarded(at, &vals))
    }

    /// XNW: stage d): a store of `vals` through the record at `site`
    /// and the page table's UAV (index + 1); dropped where the record has no
    /// tag, the access is out of range, the page is missing or `guard` is off.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn page_store(&mut self, site: Site, root: Option<(V, V)>, index: Option<V>, off: V, vals: &[V], uniform: bool, guard: Option<V>) -> Result<(), Why> {
        let (lo, hi, mut go) = self.page_address(site, root, index, off, vals.len() as u32)?;
        if let Some(g) = guard {
            go = self.s.m.bin(self.cur, Bin::And, go, g);
        }
        let at = self.begin_guarded(Some(go));
        let (h, byte, mapped) = self.page_entry(lo, hi, true, uniform);
        let inner = self.begin_guarded(Some(mapped));
        let c = self.cur;
        let iu = self.s.m.undef(self.s.t.i32);
        let mut args = vec![h, byte, iu];
        args.extend(vals);
        args.resize(3 + 4, iu);
        args.push(self.s.m.i8((1u8 << vals.len()) - 1));
        args.push(self.s.m.i32(4));
        let void = self.s.t.void;
        self.s.op(c, "rawBufferStore.i32", Attr::NoUnwind, void, 140, &args);
        self.end_guarded(inner, &[]);
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// XNW: stage d): a buffer atomic through the record at `site`
    /// and the page table's UAV: DXIL atomic op `code` (`u32::MAX`:
    /// compare-exchange, `vals` = (new, compare)) on the raw dword. Returns
    /// the old value, `old` where it does not run.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn page_atomic(&mut self, site: Site, root: Option<(V, V)>, index: Option<V>, off: V, code: u32, vals: &[V], old: V, uniform: bool, guard: Option<V>) -> Result<V, Why> {
        let (lo, hi, mut go) = self.page_address(site, root, index, off, 1)?;
        if let Some(g) = guard {
            go = self.s.m.bin(self.cur, Bin::And, go, g);
        }
        let at = self.begin_guarded(Some(go));
        let (h, byte, mapped) = self.page_entry(lo, hi, true, uniform);
        let inner = self.begin_guarded(Some(mapped));
        let c = self.cur;
        let i32t = self.s.t.i32;
        let iu = self.s.m.undef(i32t);
        let r = if code == u32::MAX {
            self.s.op(c, "atomicCompareExchange.i32", Attr::NoUnwind, i32t, 79, &[h, byte, iu, iu, vals[1], vals[0]])
        } else {
            let op = self.s.m.i32(code);
            self.s.op(c, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &[h, op, byte, iu, iu, vals[0]])
        };
        // An atomic on a missing page returns 0 (GFX10: out of range).
        let zero = self.s.m.i32(0);
        let r = self.end_guarded(inner, &[(r, zero)])[0];
        Ok(self.end_guarded(at, &[(r, old)])[0])
    }
    /// The four dwords of the descriptor record the instruction at `site`
    /// names (GPU-side descriptors).
    pub(super) fn record_words(&mut self, site: Site) -> Result<[V; 4], Why> {
        let inst: &Inst = &self.x.p.blocks[site.0].insts[site.1];
        let op = match &inst.fields {
            Fields::Buffer { rsrc, .. } => rsrc,
            Fields::Smem { base, .. } => base,
            _ => return Err("a descriptor record read by this instruction"),
        };
        let Operand::Scalar { code, .. } = *op else {
            return Err("a descriptor record not in SGPRs");
        };
        let mut w = [V::Const(0); 4];
        for (k, slot) in w.iter_mut().enumerate() {
            let v = inst.read(Reg::S(code + k as u8)).ok_or("a descriptor record")?;
            *slot = self.i32_of(v)?;
        }
        Ok(w)
    }
}
