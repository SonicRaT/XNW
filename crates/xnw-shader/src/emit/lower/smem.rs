// Part of emit/lower.rs: scalar memory: windows, records, descriptor data, constant-buffer dwords, format loads.
// Origin as in lower.rs's header: VIctory crates/victory-shader/src/emit/lower.rs, d2a4bae1f495, and the
// later VIctory ports named there; GPL-3.0-or-later. Moved out of lower.rs unchanged (XNW BLD2): one stage,
// one codegen unit.
use super::*;

impl<'a> Lower<'a> {
    pub(super) fn smem(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Fields::Smem {
            data,
            offset,
            soffset,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if let Some(a) = self.scalar.get(&site).copied().filter(|a| a.param == CODE) {
            return self.code_smem(inst, a);
        }
        if self.is_global(site) {
            return self.global_smem(inst);
        }
        if self.x.gpu_desc
            && name.starts_with("s_load_dword")
            && let Some(a) = self.scalar.get(&site).copied()
            && matches!(
                self.x.native.layout.get(a.param).map(|r| r.root),
                Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap)
            )
        {
            return self.window_smem(inst, a);
        }
        let Some(rest) = name.strip_prefix("s_buffer_load_dword") else {
            return self.descriptor_data(inst, site);
        };
        let count = match rest {
            "" => 1,
            "x2" => 2,
            "x4" => 4,
            "x8" => 8,
            "x16" => 16,
            _ => return Err("a scalar buffer load"),
        };
        let Some(Operand::Scalar { code, .. }) = *data else {
            return Err("a scalar buffer load");
        };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked descriptor")?;
        let kind = self.used_kind(src);
        let c = self.cur;
        let (base, extra) = self.built_offset(src)?;
        let offset = &((*offset as u32).wrapping_add(base) as i32);
        let mut addr = self.s.m.i32(*offset as u32);
        if let Some(x) = extra {
            addr = self.s.m.bin(c, Bin::Add, addr, x);
        }
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(inst, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        if matches!(src, Source::Built { address, .. } if address.param == CODE) {
            // A descriptor built around a table in the code.
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let x = self.code_dword(a)?;
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        let root = self.root_view_base(src);
        if !self.paged(src) && root.is_none() {
            self.note_unpaged(src, false);
        }
        if self.paged(src) || root.is_some() {
            // XNW: stage c): constants and scalar buffer reads raw at
            // base + offset through the page table, the only path (no host
            // constant buffer view, no 256-byte alignment).
            for (k, x) in self.page_load(site, root, None, addr, count, true)?.into_iter().enumerate() {
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        if let Some(u) = kind
            && matches!(u.kind, ResourceType::Srv | ResourceType::Uav)
        {
            let u_bit = if u.kind == ResourceType::Uav { 0x1000 } else { 0 };
            let structured = u.stride >= 4 && u.stride.is_multiple_of(4);
            let h = self.handle(
                site,
                src,
                false,
                if structured {
                    (12 | u_bit, u.stride)
                } else {
                    (11 | u_bit, 0)
                },
            )?;
            let st = structured.then(|| self.s.m.i32(u.stride));
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let (idx, elem_off) = match st {
                    Some(st) => (
                        self.s.m.bin(c, Bin::UDiv, a, st),
                        self.s.m.bin(c, Bin::URem, a, st),
                    ),
                    None => (a, self.s.m.undef(self.s.t.i32)),
                };
                let mask = self.s.m.i8(1);
                let al = self.s.m.i32(4);
                let r = self.s.op(
                    c,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    self.s.t.res_ret,
                    139,
                    &[h, idx, elem_off, mask, al],
                );
                let x = self.s.m.extract(c, r, 0);
                let x = self.as_int(x);
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        if self.x.gpu_desc {
            // XNW: constants at the console's 16-byte alignment read
            // through the raw view the host makes for a constant record
            // (a CBV needs 256 bytes).
            let h = self.handle(site, src, false, (11, 0))?;
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let x = self.raw_dword(h, a);
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        // victory-fork (fxceless/victory-fork 578d459, crates/victory-shader
        // src/emit/lower.rs 5344-5395, GPL-3.0-or-later): a buffer the
        // program picks from the view heap by an index it computes
        // (`ResourceHeap`) has no used-resource entry, and the heap holds a
        // buffer view there, not a constant buffer; read as one, the loads
        // gave nothing: the title's meshlet culling reads its batch records,
        // mesh headers and transforms this way, so every indirect mesh draw
        // ran empty. Changed from the fork: the slot's stride comes from our
        // runtime stride table (`stride_entry`, VIctory/704) and the
        // bytes through the raw view that table names, not a table of its
        // own. A stride of 0 (no buffer view) is still read as a constant
        // buffer; only the load that fits the slot runs.
        if kind.is_none() && self.heap_picked(src) {
            let [raw, stride, bias] = self.stride_entry(src)?;
            let zero = self.s.m.i32(0);
            let is_buffer = self.s.m.cmp(self.cur, pred::NE, stride, zero);
            let at = self.begin_guarded(Some(is_buffer));
            let h = self.s.heap_handle(self.cur, raw, false, (11, 0));
            let mut bytes = Vec::new();
            for k in 0..count {
                let b = self.cur;
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(b, Bin::Add, addr, off);
                let a = self.s.m.bin(b, Bin::Add, a, bias);
                bytes.push((self.raw_dword(h, a), zero));
            }
            let bytes = self.end_guarded(at, &bytes);
            let other = self.s.m.cmp(self.cur, pred::EQ, stride, zero);
            let at = self.begin_guarded(Some(other));
            let h = self.handle(site, src, false, (13, 0x1_0000))?;
            let mut both = Vec::new();
            for (k, &r) in bytes.iter().enumerate() {
                let b = self.cur;
                let off = self.s.m.i32(4 * k as u32);
                let a = self.s.m.bin(b, Bin::Add, addr, off);
                both.push((self.cbuffer_dword(h, a), r));
            }
            let both = self.end_guarded(at, &both);
            for (k, x) in both.into_iter().enumerate() {
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        let h = self.handle(site, src, false, (13, 0x1_0000))?;
        let constant = match addr {
            V::Const(_)
                if matches!(soffset, Operand::Scalar { code: NULL, .. }) && extra.is_none() =>
            {
                Some(*offset as u32)
            }
            _ => None,
        };
        let mut rows: HashMap<u32, V> = HashMap::new();
        for k in 0..count {
            let x = match constant {
                Some(base) => {
                    let byte = base + 4 * k;
                    let row = match rows.get(&(byte >> 4)) {
                        Some(&r) => r,
                        None => {
                            let ri = self.s.m.i32(byte >> 4);
                            let r = self.s.op(
                                c,
                                "cbufferLoadLegacy.i32",
                                Attr::ReadOnly,
                                self.s.t.cbuf_ret,
                                59,
                                &[h, ri],
                            );
                            rows.insert(byte >> 4, r);
                            r
                        }
                    };
                    self.s.m.extract(c, row, (byte >> 2) & 3)
                }
                None => {
                    let off = self.s.m.i32(4 * k);
                    let a = self.s.m.bin(c, Bin::Add, addr, off);
                    self.cbuffer_dword(h, a)
                }
            };
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// A descriptor load (`s_load_dword*` from a table or heap) whose dwords
    /// the code also reads as numbers: the console compiler computes sizes
    /// from descriptor bits (an image's width, height, depth, levels; a
    /// buffer's record count) instead of querying them. Those fields are
    /// rebuilt from the host view's dimensions in the GFX10 layout (image
    /// dword 1 bits 30-31 and dword 2 bits 0-13 width - 1, dword 2 bits
    /// 14-29 height - 1, dword 3 identity swizzle, last level at 16, type at
    /// 28, dword 4 depth or layers - 1; buffer dword 2 the record count).
    // shortcut: the base level reads 0 (the host view starts at its own
    // level 0, so sizes the code shifts by it come out right) and the other
    // dwords (addresses, formats, tiling) have no host value: reading them
    // stops the program.
    /// Two scalar loads of the same dwords: same opcode and immediate
    /// offset, base read from the same values, and the same or an
    /// equivalent register offset.
    pub(super) fn twin_loads(&self, a: Site, b: Site) -> bool {
        let (ia, ib): (&Inst, &Inst) = (
            &self.x.p.blocks[a.0].insts[a.1],
            &self.x.p.blocks[b.0].insts[b.1],
        );
        let (
            Fields::Smem {
                base: Operand::Scalar { code: ba, .. },
                offset: oa,
                soffset: sa,
                ..
            },
            Fields::Smem {
                base: Operand::Scalar { code: bb, .. },
                offset: ob,
                soffset: sb,
                ..
            },
        ) = (&ia.fields, &ib.fields)
        else {
            return false;
        };
        let same_base = (0..2).all(|k| {
            let va = ia.read(Reg::S(ba + k));
            va.is_some() && va == ib.read(Reg::S(bb + k))
        });
        let same_soffset = match (sa, sb) {
            (Operand::Scalar { code: NULL, .. }, Operand::Scalar { code: NULL, .. }) => true,
            (Operand::Scalar { code: ca, .. }, Operand::Scalar { code: cb, .. }) => {
                match (ia.read(Reg::S(*ca)), ib.read(Reg::S(*cb))) {
                    (Some(x), Some(y)) => self.same_scalar(x, y),
                    _ => false,
                }
            }
            _ => false,
        };
        ia.name() == ib.name() && oa == ob && same_base && same_soffset
    }

    /// The same value, or two `v_readfirstlane_b32` of the same VGPR value
    /// (each waterfall loop reads the index anew).
    pub(super) fn same_scalar(&self, x: Value, y: Value) -> bool {
        if x == y {
            return true;
        }
        let first_lane_of = |v: Value| {
            self.def_site
                .get(&v)
                .map(|&(b, i)| &self.x.p.blocks[b].insts[i])
                .filter(|i| i.name() == "v_readfirstlane_b32")
                .and_then(|i| {
                    i.reads
                        .iter()
                        .find(|r| matches!(r.0, Reg::V(_)))
                        .map(|r| r.1)
                })
        };
        matches!((first_lane_of(x), first_lane_of(y)), (Some(a), Some(b)) if a == b)
    }

    pub(super) fn descriptor_data(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        const OTHER: Why = "a descriptor's address or format read as data";
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar memory read as data");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar memory read as data"),
        };
        // XNW: a sampler a waterfall loop walks. Its dwords are only copied
        // per lane, read from the first lane and compared for equality, so
        // each stands as its own byte offset in the sampler heap (16 bytes
        // an entry, as the console's): lanes compare equal when they hold
        // the same entry, and each samples with its own.
        if self.x.res.lane_samplers.contains(&site) {
            for k in 0..count {
                let x = self.heap_byte(site, k as u8)?;
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        // How the program uses the descriptor tells the view's kind. Prefer an
        // image or vector buffer use when a descriptor has both scalar and
        // vector uses, since those carry format and indexing cues.
        // The first use by an image or buffer instruction, else the first.
        let mut use_at = self.load_uses.get(&site).and_then(|&(first, rich)| rich.or(first));
        // VIctory (5ff639e): else the uses of a twin load of the
        // same dwords (a non-uniform index reloads the descriptor in each
        // waterfall loop, one for the size bits, another for the access).
        if use_at.is_none() {
            use_at = self
                .x
                .res
                .descriptors
                .iter()
                .filter(|d| d.role == Role::Resource)
                .find_map(|d| {
                    d.source.leaves().into_iter().find_map(|l| match *l {
                        Source::Loaded { load, first } if self.twin_loads(site, load) => {
                            Some((d.site, first))
                        }
                        _ => None,
                    })
                });
        }
        let (src, uav, props, image) = if let Some((at, first)) = use_at {
            let src = Source::Loaded { load: site, first };
            let used = self.used_kind(&src);
            let ui: &Inst = &self.x.p.blocks[at.0].insts[at.1];
            let un = ui.name();
            let uav = un.contains("store")
                || un.contains("atomic")
                || used.is_some_and(|u| u.kind == ResourceType::Uav);
            let u = if uav { 0x1000 } else { 0 };
            let (props, image) = match &ui.fields {
                Fields::Image { dim, .. } => {
                    let kind = match dim {
                        0 => 1,
                        1 => 2,
                        2 => 4,
                        4 => 6,
                        5 => 7,
                        _ => return Err("an image dimension (cube or MSAA)"),
                    };
                    ((kind | u, 0x409), Some(kind))
                }
                Fields::Buffer { idxen, format, .. } => {
                    let stride = used.map_or(0, |u| u.stride);
                    let props = if un.starts_with("buffer_load_format")
                        || un.starts_with("buffer_store_format")
                    {
                        // (Unsigned integers where the access moves halves;
                        // victory-fork 5a55384.)
                        let halves = self
                            .table_place(&Source::Loaded { load: site, first })
                            .is_some_and(|p| {
                                self.halves_only.contains(&p)
                                    && (un.contains("_format_d16_")
                                        || self.format_load_read_as_halves(ui))
                            });
                        (10 | u, if halves { 0x405 } else { 0x409 })
                    } else if *idxen && format.is_none() && stride >= 4 && stride.is_multiple_of(4) {
                        (12 | u, stride)
                    } else {
                        (11 | u, 0)
                    };
                    if used.is_some_and(|u| u.kind == ResourceType::Cbv) {
                        return Err("a constant buffer's descriptor read as data");
                    }
                    (props, None)
                }
                Fields::Smem { .. } if un.starts_with("s_buffer_") => {
                    if used.is_some_and(|u| u.kind == ResourceType::Cbv) {
                        return Err("a constant buffer's descriptor read as data");
                    }
                    let stride = used.map_or(0, |u| u.stride);
                    let props = if stride >= 4 && stride.is_multiple_of(4) {
                        (12 | u, stride)
                    } else {
                        (11 | u, 0)
                    };
                    (props, None)
                }
                _ => return Err("a scalar memory read as data"),
            };
            (src, uav, props, image)
        } else {
            // VIctory (f800f1c): a descriptor read only for its size
            // (GetDimensions) is typed by the program's resource metadata.
            let src = Source::Loaded {
                load: site,
                first: 0,
            };
            let Some(used) = self.used_kind(&src) else {
                // VIctory: no access names it, so the console words
                // are all there is.
                let words = self.descriptor_words(site, count)?;
                for (k, x) in words.into_iter().enumerate() {
                    self.write(inst, Reg::S(code + k as u8), x, None);
                }
                return Ok(());
            };
            if used.kind == ResourceType::Cbv {
                return Err("a constant buffer's descriptor read as data");
            }
            if !matches!(used.kind, ResourceType::Srv | ResourceType::Uav) {
                return Err("a descriptor read only as data");
            }
            let uav = used.kind == ResourceType::Uav;
            let u = if uav { 0x1000 } else { 0 };
            let (props, image) = if used.dwords == 4 {
                let stride = used.stride;
                let props = if stride >= 4 && stride.is_multiple_of(4) {
                    (12 | u, stride)
                } else {
                    (11 | u, 0)
                };
                (props, None)
            } else {
                ((2 | u, 0x409), Some(2))
            };
            (src, uav, props, image)
        };
        let Source::Loaded { first, .. } = src else {
            unreachable!()
        };
        let h = self.handle(site, &src, false, props)?;
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let mip = if image.is_some() && !uav {
            self.s.m.i32(0)
        } else {
            self.s.m.undef(i32t)
        };
        let d = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
        let field = |l: &mut Self, k: u32| l.s.m.extract(c, d, k);
        let words: Vec<(u32, V)> = match image {
            Some(kind) => {
                let one = self.s.m.i32(1);
                let w = field(self, 0);
                let (height, depth) = match kind {
                    1 => (one, one),
                    2 => (field(self, 1), one),
                    6 => (one, field(self, 1)),
                    _ => (field(self, 1), field(self, 2)),
                };
                let levels = if uav { one } else { field(self, 3) };
                let less = |l: &mut Self, x: V| l.s.m.bin(c, Bin::Sub, x, one);
                let (w, height, depth, levels) = (
                    less(self, w),
                    less(self, height),
                    less(self, depth),
                    less(self, levels),
                );
                let k = |l: &mut Self, v: u32| l.s.m.i32(v);
                let (k30, k2, k14, k16, mask) = (
                    k(self, 30),
                    k(self, 2),
                    k(self, 14),
                    k(self, 16),
                    k(self, 0x3FFF),
                );
                let d1 = self.s.m.bin(c, Bin::Shl, w, k30);
                let hi = self.s.m.bin(c, Bin::LShr, w, k2);
                let hi = self.s.m.bin(c, Bin::And, hi, mask);
                let ht = self.s.m.bin(c, Bin::Shl, height, k14);
                let d2 = self.s.m.bin(c, Bin::Or, hi, ht);
                let ty = [0, 8, 9, 0, 10, 0, 12, 13][kind as usize];
                let base = k(self, 0xFAC | ty << 28);
                let last = self.s.m.bin(c, Bin::Shl, levels, k16);
                let d3 = self.s.m.bin(c, Bin::Or, base, last);
                vec![(1, d1), (2, d2), (3, d3), (4, depth)]
            }
            None => vec![(2, field(self, 0))],
        };
        // VIctory: fields the host view gives are rebuilt from it;
        // the console's other words (addresses, formats) come from the
        // descriptor words. Changed from VIctory (always): only in programs
        // that walk a BVH, so programs that translate today bind nothing new.
        let missing = (0..count).any(|k| !words.iter().any(|w| w.0 == k.wrapping_sub(u32::from(first))));
        let console = if missing && self.walks_bvh() { self.descriptor_words(site, count).ok() } else { None };
        for k in 0..count {
            let rel = k.wrapping_sub(u32::from(first));
            match (words.iter().find(|w| w.0 == rel), &console) {
                (Some(&(_, x)), _) => self.write(inst, Reg::S(code + k as u8), x, None),
                (None, Some(c)) => self.write(inst, Reg::S(code + k as u8), c[k as usize], None),
                (None, None) => {
                    if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                        self.vals.insert(v, Val::Missing(OTHER));
                    }
                }
            }
        }
        Ok(())
    }

    /// `s_load_dword*` from the program's constant table.
    pub(super) fn code_smem(&mut self, inst: &Inst, a: Address) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar load");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar load from the code"),
        };
        let c = self.cur;
        let addr = self.address_offset(a)?;
        for k in 0..count {
            let off = self.s.m.i32(4 * k);
            let at = self.s.m.bin(c, Bin::Add, addr, off);
            let x = self.code_dword(at)?;
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// The dword at a computed byte address of a constant buffer.
    /// The dword at byte `a` of raw buffer `h` (XNW).
    pub(super) fn raw_dword(&mut self, h: V, a: V) -> V {
        let c = self.cur;
        let (u, mask, al) = (self.s.m.undef(self.s.t.i32), self.s.m.i8(1), self.s.m.i32(4));
        let r = self.s.op(c, "rawBufferLoad.f32", Attr::ReadOnly, self.s.t.res_ret, 139, &[h, a, u, mask, al]);
        let x = self.s.m.extract(c, r, 0);
        self.as_int(x)
    }

    pub(super) fn cbuffer_dword(&mut self, h: V, a: V) -> V {
        let c = self.cur;
        let four = self.s.m.i32(4);
        let ri = self.s.m.bin(c, Bin::LShr, a, four);
        let r = self.s.op(
            c,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, ri],
        );
        let two = self.s.m.i32(2);
        let three = self.s.m.i32(3);
        let comp = self.s.m.bin(c, Bin::LShr, a, two);
        let comp = self.s.m.bin(c, Bin::And, comp, three);
        let mut x = self.s.m.extract(c, r, 0);
        for j in 1..4 {
            let e = self.s.m.extract(c, r, j);
            let jc = self.s.m.i32(j);
            let is = self.s.m.cmp(c, pred::EQ, comp, jc);
            x = self.s.m.select(c, is, e, x);
        }
        x
    }

    /// `channels` channels of the element at byte `off` of raw buffer `h`,
    /// converted from the GFX10 buffer format `format` as the hardware's
    /// typed loads do (channels past the format's read 0, 0, 0, 1).
    // shortcut: the descriptor's channel swizzle is taken as the identity
    // (it is for raw views and the descriptors code builds), elements of 4
    // bytes or more as dword-aligned. Norm formats divide, which the AMD
    // driver does 1 ULP off the hardware's correctly rounded conversion for
    // some values; correct it with an integer check if a title compares
    // converted values exactly.
    pub(super) fn format_load(&mut self, h: V, off: V, format: u8, channels: u32) -> Result<Vec<V>, Why> {
        let (bits, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let total: u32 = bits.iter().sum();
        let (u, align) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
        let load = |l: &mut Self, at: V, n: u32| -> Vec<V> {
            let mask = l.s.m.i8((1u8 << n) - 1);
            let r = l.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                l.s.t.res_ret,
                139,
                &[h, at, u, mask, align],
            );
            (0..n)
                .map(|k| {
                    let x = l.s.m.extract(c, r, k);
                    l.as_int(x)
                })
                .collect()
        };
        let words = if total < 32 {
            // A byte or half inside the dword that holds it.
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            let w = load(self, at, 1)[0];
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            vec![self.s.m.bin(c, Bin::LShr, w, sh)]
        } else {
            load(self, off, total / 32)
        };
        let float = |l: &mut Self, f: V| l.as_int(f);
        let mut out = Vec::new();
        let mut at = 0;
        for &w in bits {
            let word = words[(at / 32) as usize];
            let field = if w == 32 {
                word
            } else {
                let sh = self.s.m.i32(at % 32);
                let t = self.s.m.bin(c, Bin::LShr, word, sh);
                let m = self.s.m.i32((1 << w) - 1);
                self.s.m.bin(c, Bin::And, t, m)
            };
            at += w;
            // Sign-extended from the field's top bit.
            let signed = |l: &mut Self| {
                if w == 32 {
                    return field;
                }
                let k = l.s.m.i32(32 - w);
                let t = l.s.m.bin(c, Bin::Shl, field, k);
                l.s.m.bin(c, Bin::AShr, t, k)
            };
            let f32t = self.s.t.f32;
            let x = match numeric {
                Numeric::Unorm => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    let max = self.s.m.f32((((1u64 << w) - 1) as f32).to_bits());
                    let q = self.s.m.bin(c, Bin::SDiv, f, max);
                    float(self, q)
                }
                Numeric::Snorm => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    let max = self.s.m.f32((((1u64 << (w - 1)) - 1) as f32).to_bits());
                    let q = self.s.m.bin(c, Bin::SDiv, f, max);
                    let lo = self.s.m.f32((-1.0f32).to_bits());
                    let q = self.binary(35, q, lo);
                    float(self, q)
                }
                Numeric::Uscaled => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    float(self, f)
                }
                Numeric::Sscaled => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    float(self, f)
                }
                Numeric::Uint => field,
                Numeric::Sint => signed(self),
                Numeric::Float if w == 32 => field,
                Numeric::Float => {
                    let h = match half_shift(w) {
                        0 => field,
                        k => {
                            let sh = self.s.m.i32(k);
                            self.s.m.bin(c, Bin::Shl, field, sh)
                        }
                    };
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    float(self, f)
                }
            };
            out.push(x);
        }
        let one = match numeric {
            Numeric::Uint | Numeric::Sint => 1,
            _ => 1.0f32.to_bits(),
        };
        for k in out.len() as u32..channels {
            out.push(self.s.m.i32(if k == 3 { one } else { 0 }));
        }
        out.truncate(channels as usize);
        Ok(out)
    }
}
