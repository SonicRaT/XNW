// Part of emit/lower.rs: exports, exits, loop counts, flow, phis.
// Origin as in lower.rs's header: VIctory crates/victory-shader/src/emit/lower.rs, d2a4bae1f495, and the
// later VIctory ports named there; GPL-3.0-or-later. Moved out of lower.rs unchanged (XNW BLD2): one stage,
// one codegen unit.
use super::*;

impl<'a> Lower<'a> {
    pub(super) fn export(&mut self, b: usize, i: usize) -> Result<(), Why> {
        let e = *self
            .exports
            .get(&(b, i))
            .ok_or("an export I/O raising did not see")?;
        if e.compressed {
            return Err("a compressed export");
        }
        let guard = self.guard(b, i)?;
        let mesh = self.s.mesh.is_some();
        let c = self.cur;
        if let Some(g) = guard
            && !mesh
        {
            if !(e.done && self.s.pixel) {
                return Err("a conditional export");
            }
            // XNW (KILL1): `Stub::kill` leaves the discard out.
            crate::emit::stub_site("kill", 0);
            if !crate::emit::STUB.with(std::cell::Cell::get).kill {
                let t = self.s.m.bool(true);
                let kill = self.s.m.bin(c, Bin::Xor, g, t);
                self.s
                    .op(c, "discard", Attr::NoUnwind, self.s.t.void, 82, &[kill]);
            }
        }
        if mesh && e.target == Target::Primitive {
            if self.s.fallback {
                return self.fallback_indices(e, guard);
            }
            return self.emit_indices(e, guard);
        }
        if self.s.fallback && matches!(e.target, Target::PrimitiveParam { .. }) {
            return Err("a per-primitive output in the compute fallback");
        }
        let param_row = |usage: Usage, row: u8| match usage {
            // The 2020 compiler leaves a mesh program's first entry zeroed
            // (usage "position"): its row is the one no other entry names.
            Usage::Other(0) if mesh => self.unnamed_row(),
            _ => Some(row),
        };
        // (output element, channel) per exported channel; per-primitive
        // outputs are elements of `prims`.
        let mut stores = Vec::new();
        let primitive = matches!(e.target, Target::PrimitiveParam { .. });
        let outputs = if primitive {
            &self.s.prims
        } else {
            &self.s.outputs
        };
        let by_kind = |kind: u8| outputs.iter().position(|o| o.kind == kind);
        // 16-bit vertex outputs: the halves each dword carries, as (output
        // element, channel, high half) per channel.
        let halves = match e.target {
            Target::Param { usage, row } => {
                param_row(usage, row).and_then(|r| self.half_outputs.get(&r).map(|&h| (r, h)))
            }
            _ => None,
        };
        if let Some((row, (pack2, b))) = halves {
            for ch in 0..4u8 {
                let Some(value) = e.values[usize::from(ch)] else {
                    continue;
                };
                for high in [false, true] {
                    let h = 2 * ch + u8::from(high);
                    let src = if pack2 {
                        packed_half(outputs, row, b, h)
                    } else {
                        (!high).then_some((row, ch))
                    };
                    let found = src.and_then(|(r, c)| {
                        outputs.iter().position(|o| {
                            o.row == Some(u32::from(r)) && c >= o.col && c < o.col + o.cols
                        })
                    });
                    if let Some(id) = found {
                        let col = src.expect("found").1 - outputs[id].col;
                        stores.push((id, col, value, Some(high)));
                    }
                }
            }
        }
        for ch in 0..4u8 {
            let Some(value) = e.values[usize::from(ch)] else {
                continue;
            };
            if halves.is_some() {
                break;
            }
            let found = match e.target {
                Target::Color(n) => outputs
                    .iter()
                    .position(|o| o.kind == 16 && o.row == Some(u32::from(n)))
                    .map(|id| (id, ch)),
                Target::Position(0) => by_kind(3).map(|id| (id, ch)),
                Target::Depth => match ch {
                    0 => by_kind(17)
                        .or(by_kind(18))
                        .or(by_kind(19))
                        .map(|id| (id, 0)),
                    1 => by_kind(20).map(|id| (id, 0)),
                    2 => by_kind(14).map(|id| (id, 0)),
                    _ => None,
                },
                Target::Param { usage, row } | Target::PrimitiveParam { usage, row } => {
                    let row = param_row(usage, row);
                    outputs
                        .iter()
                        .position(|o| {
                            row.is_some_and(|r| o.row == Some(u32::from(r)))
                                && ch >= o.col
                                && ch < o.col + o.cols
                        })
                        .map(|id| (id, ch - outputs[id].col))
                }
                Target::Null | Target::Unread => None,
                Target::Position(n @ 1..=2) => {
                    // shortcut: without the misc vector (point size, render
                    // target or viewport index) pos1 and pos2 carry the clip
                    // then cull distances, in semantic order.
                    if outputs.iter().any(|o| matches!(o.kind, 4 | 5)) {
                        return Err("render-target or viewport index exports");
                    }
                    let mut dists: Vec<usize> = (0..outputs.len())
                        .filter(|&i| matches!(outputs[i].kind, 6 | 7))
                        .collect();
                    dists.sort_by_key(|&i| (outputs[i].kind, outputs[i].index));
                    let flat: Vec<(usize, u8)> = dists
                        .iter()
                        .flat_map(|&i| (0..outputs[i].cols).map(move |c| (i, c)))
                        .collect();
                    // Channels past the distances the signature declares are
                    // ones the clipper does not enable (`PA_CL_VS_OUT_CNTL`).
                    let k = usize::from(n - 1) * 4 + usize::from(ch);
                    flat.get(k).copied()
                }
                Target::Position(_) => return Err("clip, cull or point-size exports"),
                Target::Primitive => return Err("a primitive export"),
            };
            if let Some((id, col)) = found {
                stores.push((id, col, value, None));
            }
        }
        fn elements<'l>(l: &'l mut Lower, primitive: bool) -> &'l mut Vec<SigElement> {
            if primitive {
                &mut l.s.prims
            } else {
                &mut l.s.outputs
            }
        }
        if stores
            .iter()
            .any(|s| s.3.is_none() && elements(self, primitive)[s.0].half)
        {
            return Err("a 16-bit output the stage data does not mark");
        }
        // A mesh lane writes its own vertex or primitive, where it exports.
        let at = if mesh {
            self.begin_guarded(guard)
        } else {
            None
        };
        let (name, opcode) = match (mesh, primitive) {
            (false, _) => ("storeOutput", 5),
            (true, false) => ("storeVertexOutput", 171),
            (true, true) => ("storePrimitiveOutput", 172),
        };
        // XNW (KILL1): `Stub::export` stores 0.5 in a colour target's channels.
        let stubbed = match e.target {
            Target::Color(n) if self.s.pixel => {
                crate::emit::stub_site("export", u32::from(n));
                crate::emit::STUB.with(std::cell::Cell::get).exports >> n.min(7) & 1 != 0
            }
            _ => false,
        };
        for (id, col, value, half) in stores {
            let mut x = if stubbed { self.s.m.i32(0.5f32.to_bits()) } else { self.i32_of(value)? };
            if let Some(high) = half {
                // An f16 half, declared a 32-bit float on the host.
                let c = self.cur;
                if high {
                    let k16 = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, k16);
                }
                let f32t = self.s.t.f32;
                let f = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                x = self.as_int(f);
            }
            if self.s.fallback {
                self.fallback_store(id, col, x)?;
                continue;
            }
            let el = &mut elements(self, primitive)[id];
            el.usage |= 1 << col;
            let (suffix, x) = match el.comp {
                9 if half.is_some() => (".f32", self.as_float(x)),
                9 => {
                    let f = match self.floats.get(&value) {
                        Some(&f) if !stubbed => f,
                        _ => self.as_float(x),
                    };
                    (".f32", f)
                }
                _ => (".i32", x),
            };
            let mut args = vec![
                self.s.m.i32(id as u32),
                self.s.m.i32(0),
                self.s.m.i8(col),
                x,
            ];
            if mesh {
                args.push(self.mesh_thread());
            }
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                &format!("{name}{suffix}"),
                Attr::NoUnwind,
                void,
                opcode,
                &args,
            );
        }
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Mesh: `exp prim` as the lane's primitive. Bit 31 culls it (all
    /// indices 0: no area).
    pub(super) fn emit_indices(&mut self, e: &Export, guard: Option<V>) -> Result<(), Why> {
        let v = e.values[0].ok_or("a primitive export without its indices")?;
        let x = self.i32_of(v)?;
        let at = self.begin_guarded(guard);
        let c = self.cur;
        let z = self.s.m.i32(0);
        let culled = self.s.m.cmp(c, pred::SLT, x, z);
        let mut args = vec![self.mesh_thread()];
        for k in 0..3 {
            // VIctory (c0228d9): a 9-bit index; bits 9, 19, 29 are
            // edge flags, no part of it.
            let (shift, mask) = (self.s.m.i32(10 * k), self.s.m.i32(0x1FF));
            let i = self.s.m.bin(c, Bin::LShr, x, shift);
            let i = self.s.m.bin(c, Bin::And, i, mask);
            args.push(self.s.m.select(c, culled, z, i));
        }
        let void = self.s.t.void;
        self.s
            .op(c, "emitIndices", Attr::NoUnwind, void, 169, &args);
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// The per-vertex arbitrary row no mesh output entry names but one
    /// (the 2020 compiler's zeroed entry).
    pub(super) fn unnamed_row(&self) -> Option<u8> {
        let StageIo::Ms(ms) = &self.x.native.io else {
            return None;
        };
        let named = |r: u32| {
            ms.outputs
                .iter()
                .any(|o| o.param & 0x80 == 0 && o.usage != Usage::Other(0) && u32::from(o.row) == r)
        };
        let free: std::collections::BTreeSet<u32> = self
            .s
            .outputs
            .iter()
            .filter(|o| o.kind == 0)
            .filter_map(|o| o.row)
            .filter(|&r| !named(r))
            .collect();
        match free.into_iter().collect::<Vec<_>>()[..] {
            [r] => u8::try_from(r).ok(),
            _ => None,
        }
    }

    pub(super) fn exit(&mut self, b: usize) -> Result<(), Stop> {
        let offset = self.x.p.blocks[b]
            .insts
            .last()
            .map_or(self.x.p.blocks[b].offset, |i| i.machine.offset);
        let term = match self.x.st.exits[b] {
            LaneExit::Jump(t) if self.counters.contains_key(&t) && self.rpo[t] <= self.rpo[b] => {
                let (next, ok) = self.count_iteration(pred::IULT);
                self.report_loop(t, next, None);
                self.flow(t, self.cur, next);
                let abort = self.abort_block();
                Term::CondBr(ok, self.block[t].expect("reachable"), abort)
            }
            LaneExit::Jump(t) => {
                let budget = self.budget.expect("block started");
                self.flow(t, self.cur, budget);
                Term::Br(self.block[t].expect("reachable"))
            }
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                // A negated condition branches the other way round.
                let (cond, taken, not_taken) = match self.x.st.bools.node(when) {
                    // Wave-wide (FACE1, `Structured::wide`): the wave takes
                    // it when every lane would.
                    _ if self.x.st.wide[b] => {
                        let c = self
                            .boolean(when, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        self.s.flags |= flags::WAVE_OPS;
                        let i1 = self.s.t.i1;
                        let all = self.s.op(self.cur, "waveAllTrue", Attr::NoUnwind, i1, 114, &[c]);
                        (all, taken, not_taken)
                    }
                    Some((atom, Bool::TRUE, Bool::FALSE)) => {
                        let a = self
                            .atom(atom, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (a, not_taken, taken)
                    }
                    _ => {
                        let c = self
                            .boolean(when, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (c, taken, not_taken)
                    }
                };
                // A back edge is taken only below the iteration limit.
                let back =
                    |l: &Self, to: usize| l.counters.contains_key(&to) && l.rpo[to] <= l.rpo[b];
                // (No `xor` of a compare: a loop exit testing one hung the
                // AMD driver.)
                let budget = self.budget.expect("block started");
                let (cond, next) = if back(self, taken) {
                    let (next, ok) = self.count_iteration(pred::IULT);
                    let no = self.s.m.bool(false);
                    let leaving = self.s.m.cmp(self.cur, pred::EQ, cond, no);
                    let c2 = self.s.m.bin(self.cur, Bin::And, cond, ok);
                    self.report_loop(taken, next, Some(leaving));
                    (c2, next)
                } else if back(self, not_taken) {
                    let (next, stop) = self.count_iteration(pred::IUGE);
                    let c2 = self.s.m.bin(self.cur, Bin::Or, cond, stop);
                    self.report_loop(not_taken, next, Some(cond));
                    (c2, next)
                } else {
                    (cond, budget)
                };
                // Critical edges get a block of their own, so loops have
                // dedicated exits and preheaders as DXC emits them.
                let (t, f) = (self.edge(b, taken), self.edge(b, not_taken));
                for (to, via) in [(taken, t), (not_taken, f)] {
                    // From an edge block, or from here.
                    let from = if Some(via) == self.block[to] {
                        self.cur
                    } else {
                        via
                    };
                    // The budget leaves counted (back edge) or as it came.
                    let value = if back(self, to) { next } else { budget };
                    self.flow(to, from, value);
                }
                Term::CondBr(cond, t, f)
            }
            LaneExit::End => {
                if self.s.pixel && !self.exported[b] && !self.has_done_export(b) && {
                    crate::emit::stub_site("kill", 0);
                    !crate::emit::STUB.with(std::cell::Cell::get).kill
                } {
                    // A lane leaving without its final export was killed.
                    let t = self.s.m.bool(true);
                    self.s
                        .op(self.cur, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
                }
                Term::Ret
            }
            LaneExit::Indirect => return Err(unsupported(offset, "an indirect jump")),
        };
        let _ = Exit::End;
        // XNW: a loop-count report (`report_loop`) may have moved on
        // to a block of its own; the block's values leave from there.
        self.tail[b] = Some(self.cur);
        self.s.m.terminate(self.cur, term);
        Ok(())
    }

    /// XNW: `XNW_LOOP_COUNTS=1`, contract in xnw-gpu
    /// `d3d12x/loopcount.rs`): at a back edge to loop `header`, every 64
    /// passes and where a lane leaves (`leaving`), the wave's largest count
    /// so far goes to word `slot * 8 + k` (k: the loop's place in program
    /// order, past the 8th: 7) of the runtime's buffer by an atomic max of
    /// its first active lane. The count is the invocation's loop budget
    /// spent so far (one budget for all its loops).
    pub(super) fn report_loop(&mut self, header: usize, count: V, leaving: Option<V>) {
        let Some(slot) = self.x.loop_counts else { return };
        let p = self.x.p;
        let mut headers: Vec<usize> = (0..p.blocks.len())
            .filter(|&h| self.block[h].is_some() && p.blocks[h].preds.iter().any(|&q| self.block[q].is_some() && self.rpo[h] <= self.rpo[q]))
            .collect();
        headers.sort_by_key(|&h| p.blocks[h].offset);
        let k = headers.iter().position(|&h| h == header).unwrap_or(0).min(LOOP_COUNT_LOOPS as usize - 1) as u32;
        let c = self.cur;
        let (k63, zero) = (self.s.m.i32(63), self.s.m.i32(0));
        let low = self.s.m.bin(c, Bin::And, count, k63);
        let mut report = self.s.m.cmp(c, pred::EQ, low, zero);
        if let Some(l) = leaving {
            report = self.s.m.bin(c, Bin::Or, report, l);
        }
        let index = self.loop_count_buffer();
        let at = self.begin_guarded(Some(report));
        let c = self.cur;
        let i32t = self.s.t.i32;
        let (max, unsigned) = (self.s.m.i8(3), self.s.m.i8(1));
        let most = self.s.op(c, "waveActiveOp.i32", Attr::NoUnwind, i32t, 119, &[count, max, unsigned]);
        let i1 = self.s.t.i1;
        let first = self.s.op(c, "waveIsFirstLane", Attr::NoUnwind, i1, 110, &[]);
        self.s.flags |= flags::WAVE_OPS;
        let inner = self.begin_guarded(Some(first));
        let c = self.cur;
        let h = self.s.heap_handle(c, index, false, (11 | 0x1000, 0));
        let byte = self.s.m.i32((slot * LOOP_COUNT_LOOPS + k) * 4);
        let iu = self.s.m.undef(i32t);
        let umax = self.s.m.i32(7);
        self.s.op(c, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &[h, umax, byte, iu, iu, most]);
        self.end_guarded(inner, &[]);
        self.end_guarded(at, &[]);
    }

    /// XNW: the heap index of the loop-count buffer, from the
    /// user-data dword of root type `Other(LOOP_COUNT_ROOT)` the host fills:
    /// placed after every dword the program's parameters and their window
    /// dwords may take.
    pub(super) fn loop_count_buffer(&mut self) -> V {
        if let Some(v) = self.loop_count_index {
            return v;
        }
        let tables = self
            .x
            .native
            .layout
            .iter()
            .filter(|r| matches!(r.root, RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap))
            .count() as u32;
        let index = self.window_base() + 2 * tables;
        self.user_reads.insert(index, UserData { index, slot: 0, root: RootType::Other(LOOP_COUNT_ROOT), dword: 0 });
        self.s.user_data = self.s.user_data.max(index + 1);
        let h = match self.user_data {
            Some(h) => h,
            None => {
                let h = self.s.user_data_handle();
                self.user_data = Some(h);
                h
            }
        };
        let entry = self.s.entry;
        let row = self.s.m.i32(index / 4);
        let r = self.s.op(entry, "cbufferLoadLegacy.i32", Attr::ReadOnly, self.s.t.cbuf_ret, 59, &[h, row]);
        let v = self.s.m.extract(entry, r, index % 4);
        self.loop_count_index = Some(v);
        v
    }

    /// The budget `value` arrives at IR block `to` from DXIL block `via`.
    pub(super) fn flow(&mut self, to: usize, via: usize, value: V) {
        if let Some((_, incoming)) = self.counters.get_mut(&to) {
            incoming.push((value, via));
        }
    }

    /// A back edge (it keeps its block, no critical-edge block): the
    /// budget goes up by one; the comparison of it with the stage's limit by
    /// `p` (below: go on; at or above: stop).
    pub(super) fn count_iteration(&mut self, p: u8) -> (V, V) {
        let c = self.cur;
        let budget = self.budget.expect("block started");
        let one = self.s.m.i32(1);
        let next = self.s.m.bin(c, Bin::Add, budget, one);
        let limit = self.s.m.i32(super::super::loop_limit_for(self.x.native.stage, self.x.fallback));
        (next, self.s.m.cmp(c, p, next, limit))
    }

    /// The block a lane stopped by [`LOOP_LIMIT`] leaves through.
    pub(super) fn abort_block(&mut self) -> usize {
        if let Some(a) = self.abort {
            return a;
        }
        let a = self.s.m.new_block();
        if self.s.pixel && self.x.probe && !self.s.fallback {
            // XNW: `XNW_LOOP_PROBE`): magenta in every target, so one
            // run shows where the budget ran out.
            let targets: Vec<usize> = (0..self.s.outputs.len()).filter(|&i| self.s.outputs[i].kind == 16).collect();
            for id in targets {
                let e = &mut self.s.outputs[id];
                let (cols, float) = (e.cols, e.comp == 9);
                e.usage |= (1 << cols) - 1;
                for col in 0..cols {
                    let on = col != 1;
                    let (x, suffix) = if float {
                        (self.s.m.f32(if on { 1.0f32.to_bits() } else { 0 }), ".f32")
                    } else {
                        (self.s.m.i32(if on { u32::MAX } else { 0 }), ".i32")
                    };
                    let args = [self.s.m.i32(id as u32), self.s.m.i32(0), self.s.m.i8(col), x];
                    let void = self.s.t.void;
                    self.s.op(a, &format!("storeOutput{suffix}"), Attr::NoUnwind, void, 5, &args);
                }
            }
        } else if self.s.pixel {
            let t = self.s.m.bool(true);
            self.s
                .op(a, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
        }
        self.s.m.terminate(a, Term::Ret);
        self.abort = Some(a);
        a
    }

    /// The DXIL block a branch from IR block `from` to `to` goes to.
    pub(super) fn edge(&mut self, from: usize, to: usize) -> usize {
        let target = self.block[to].expect("reachable");
        let preds = self.x.p.blocks[to]
            .preds
            .iter()
            .filter(|&&q| self.block[q].is_some())
            .count();
        // Back edges stay as they are: the exiting block is the latch.
        if preds < 2 || self.rpo[to] <= self.rpo[from] {
            return target;
        }
        let e = self.s.m.new_block();
        self.s.m.terminate(e, Term::Br(target));
        self.edges.insert((from, to), e);
        e
    }

    /// A dword a scalar load writes, where the load is a descriptor the
    /// program uses as a resource or sampler.
    pub(super) fn descriptor_dword(&self, v: Value) -> bool {
        let Some(load) = self.x.p.blocks.iter().enumerate().find_map(|(b, block)| {
            block
                .insts
                .iter()
                .position(|i| i.writes.iter().any(|w| w.1 == v))
                .map(|i| (b, i))
        }) else {
            return false;
        };
        self.x.p.blocks[load.0].insts[load.1]
            .name()
            .starts_with("s_load_dword")
            && self.x.res.descriptors.iter().any(|d| {
                d.source
                    .leaves()
                    .iter()
                    .any(|l| matches!(l, Source::Loaded { load: at, .. } if *at == load))
            })
    }

    pub(super) fn fill_phis(&mut self) -> Result<(), Stop> {
        let p = self.x.p;
        // A phi that only another read phi consumes is live too. Without this, a value with no DXIL form that
        // reaches a read through two joins is taken for dead at the first join and replaced by zero below, and the
        // lowering accepts what it refuses when the same value is read directly.
        let mut work: Vec<Value> = self.read_phis.iter().copied().collect();
        while let Some(v) = work.pop() {
            let Some(&(b, i)) = self.phi_at.get(&v) else {
                continue;
            };
            for &arg in &p.blocks[b].phis[i].args {
                if self.phi_at.contains_key(&arg) && self.read_phis.insert(arg) {
                    work.push(arg);
                }
            }
        }
        while let Some((phi, b, i, base)) = self.pending_addr_phis.pop() {
            let ir_phi = &p.blocks[b].phis[i];
            let offset = p.blocks[b].offset;
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let saved_cur = self.cur;
                let saved_ir = self.cur_ir;
                self.cur = qb;
                self.cur_ir = q;
                let mut local = HashMap::new();
                let x = self.rel_offset(ir_phi.args[n], base, &mut local);
                self.cur = saved_cur;
                self.cur_ir = saved_ir;
                let x = x.map_err(|w| unsupported(offset, w))?;
                incoming.push((x, qb));
            }
            self.s.m.set_phi(phi, incoming);
        }
        // XNW change: every phi found unreadable in one pass goes back at
        // once (was the first only: a program with 100 such phis was
        // lowered 100 times).
        let mut retry = Vec::new();
        for (v, b, i) in std::mem::take(&mut self.data_phis) {
            let phi = &p.blocks[b].phis[i];
            let bit = phi.reg == Reg::Scc;
            // A register initialized only on one branch may carry a dead
            // descriptor dword left over from earlier register reuse on the
            // other branch (guarded by a separate predicate). Only treat a
            // descriptor-dword `Missing` as fatal when no incoming edge carries
            // a real data value.
            let has_data = !bit
                && p.blocks[b]
                    .preds
                    .iter()
                    .enumerate()
                    .filter(|&(_, &q)| self.tail[q].is_some())
                    .any(|(n, _)| matches!(self.vals.get(&phi.args[n]), Some(Val::I32(_))));
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let arg = phi.args[n];
                let x = match (self.vals.get(&arg).copied(), bit) {
                    (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => x,
                    (Some(Val::Missing(why)), false)
                        if has_data
                            && matches!(
                                why,
                                "a descriptor's address or format read as data"
                                    | "a descriptor read only as data"
                                    // XNW: the same for a constant
                                    // buffer's (ped pixel programs, 957fc3f6
                                    // at 0x2ac8: s[20:27] is data on one
                                    // edge, a CBV descriptor on the other).
                                    | "a constant buffer's descriptor read as data"
                                    // XNW: and for a sampler heap entry's
                                    // (ped pixel programs: s[12:15] hold the
                                    // entry on one edge, constants on the other).
                                    | "a sampler descriptor read as data"
                            )
                            // XNW: an SGPR that held a lane mask and is
                            // then reused for data: the mask edge of the join
                            // is taken as dead (as the descriptor dwords above).
                            || has_data && why == LANE_MASK =>
                    {
                        self.s.m.i32(0)
                    }
                    (Some(Val::Bit(_)), false) if self.read_phis.contains(&phi.value) => {
                        retry.push((phi.value, "an SCC joined with SGPR values"));
                        break;
                    }
                    // VIctory (permadiego 7ab94ab): a lane mask's dword
                    // reaching a join that also takes data: the registers
                    // are reused once the mask is dead.
                    // shortcut: undefined on that edge, assuming the program
                    // reads the joined value only on paths from a data edge.
                    // Changed from upstream: only where another edge brings
                    // data (`has_data`); a join of masks on every edge (a
                    // carry-in from a divergent join) stays refused (test
                    // a_carry_in_from_a_divergent_join_is_refused).
                    (Some(Val::Missing(_)), false)
                        if has_data
                            && (self.x.st.masks.contains_key(&arg)
                                || self.x.st.highs.contains_key(&arg)) =>
                    {
                        self.s.m.undef(self.s.t.i32)
                    }
                    // A descriptor's dwords reaching a join that also takes
                    // data (the registers are reused after the last access).
                    // shortcut: undefined on that edge, assuming the program
                    // reads the joined value only after the data edge.
                    // Changed from VIctory: compute only; on mesh
                    // programs it turned 15 translations into lane-mask
                    // refusals (pack_listing).
                    (Some(Val::Missing(_)), _)
                        if self.x.native.stage == HwStage::Cs && self.descriptor_dword(arg) =>
                    {
                        if bit {
                            self.s.m.undef(self.s.t.i1)
                        } else {
                            self.s.m.undef(self.s.t.i32)
                        }
                    }
                    (Some(Val::Missing(why)), _) if self.read_phis.contains(&phi.value) => {
                        retry.push((phi.value, why));
                        break;
                    }
                    _ if self.read_phis.contains(&phi.value) => {
                        retry.push((phi.value, "a join of mismatched values"));
                        break;
                    }
                    _ if bit => self.s.m.bool(false),
                    _ => self.s.m.i32(0),
                };
                incoming.push((x, qb));
            }
            if !retry.is_empty() {
                continue;
            }
            self.s.m.set_phi(v, incoming);
        }
        if !retry.is_empty() {
            return Err(Stop::Retry(retry));
        }
        for (phi, incoming) in std::mem::take(&mut self.counters).into_values() {
            self.s.m.set_phi(phi, incoming);
        }
        // victory-fork (578d459, GPL-3.0-or-later): each edge's mask as the
        // structure tracked it (`phi_masks`), a constant or a scalar-built
        // one included. Changed from upstream: where the incoming value has
        // its own entry in `masks` (every join that translated before) that
        // entry is taken, as before: with `phi_masks` on every edge the DXIL
        // of 2,030 programs that already translated changed (43 drawn),
        // none shown equal yet; only joins with an untracked edge take
        // `phi_masks`.
        let st = self.x.st;
        while let Some((v, b, i)) = self.pending_masks.pop() {
            let phi = &p.blocks[b].phis[i];
            let offset = p.blocks[b].offset;
            let untracked = || unsupported(offset, "a lane-mask join of an untracked mask");
            let own: Option<Vec<_>> = phi.args.iter().map(|a| st.masks.get(a).copied()).collect();
            let args = match own {
                Some(own) => own,
                None => st.phi_masks.get(&phi.value).ok_or_else(untracked)?.clone(),
            };
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let x = self.boolean(args[n], qb).map_err(|w| unsupported(offset, w))?;
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        Ok(())
    }
}
