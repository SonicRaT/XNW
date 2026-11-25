// XNW. Not ported: VIctory and its forks
// (victory-fork 0b0074e lower.rs `handle`, DexNom-VIctory 8cca4d2, VIctory
// 940d32f) declare non-uniform only the heap indices of descriptors a
// waterfall loop walks per lane, and of memory reached through the page
// table. Their rule misses the two cases below (measured in
//).
//! Uniformity of descriptor heap indices: after the clean-up passes, every
//! `createHandleFromHeap` whose index may differ between the lanes of a
//! wave gets its non-uniform operand set (`NonUniformResourceIndex`).
//!
//! An index differs between lanes when it is
//! - varying: the lanes compute different values: a thread id or input,
//!   a lane's own data, an atomic's result, a join that lanes reach from
//!   different sides of a branch on a varying condition, a value carried
//!   round a loop that lanes leave at different times. A driver reads a
//!   uniform-declared index from one lane and every lane uses that lane's
//!   descriptor (on AMD a wrong-kind or out-of-bounds read).
//! - masked: read (memory, a wave operation, a join) in a block that only
//!   some lanes run. The lanes that run it agree, but the others never
//!   read it, and WARP 1.0.22 takes a uniform-declared index from lane 0
//!   whether lane 0 runs or not (CS_Prop_BatchCull 0451d340,
//!   the slot `s_buffer_load` reads inside `v_cmpx_eq_u32 exec, 0, v26`:
//!   lane 0 slot 0, lane 1 slot 729, an access violation in WARP).
//!
//! Arithmetic carries both from its operands; constants and values
//! computed in every lane from uniform ones (a root constant, a table
//! slot) stay uniform, so most handles keep the cheaper uniform form.
//!
//! "Only some lanes run" a block when it is control dependent on a branch
//! whose condition varies: it lies between such a branch and the branch's
//! immediate post-dominator (a loop with a varying exit: the whole loop).

use super::{Module, Op, Term, V, for_each_operand, successors};

/// How a `dx.op` call's result relates to its operands.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    /// A function of the operands, computed in every lane.
    Pure,
    /// A read of memory or of neighbouring lanes: uniform from uniform
    /// operands, masked in a block only some lanes run.
    Read,
    /// The same in every active lane whatever the operands (a wave-wide
    /// reduction, a read of the first lane); masked like a read.
    Wave,
    /// `waveReadLaneAt`: uniform when the lane is (operand 2).
    LaneAt,
    /// Differs between lanes (thread ids, inputs, atomics, prefix and
    /// quad operations, anything not named here).
    Varying,
}

/// The class of `dx.op` opcode `op` (DXIL.rst's opcode table).
fn class(op: i64) -> Class {
    match op {
        // FAbs .. Dot4: unary, binary and tertiary arithmetic; CreateHandle.
        6..=57 => Class::Pure,
        // GroupId, WaveGetLaneCount, MakeDouble, SplitDouble.
        94 | 101 | 102 | 112 => Class::Pure,
        // AnnotateHandle, CreateHandleFromBinding, CreateHandleFromHeap.
        216..=218 => Class::Pure,
        // CBufferLoad(Legacy), Sample*, TextureLoad, BufferLoad,
        // CheckAccessFullyMapped, GetDimensions, TextureGather(Cmp),
        // CalculateLOD, derivatives, RawBufferLoad.
        58..=66 | 68 | 71..=74 | 81 | 83..=86 | 139 => Class::Read,
        // WaveAnyTrue, WaveAllTrue, WaveActiveAllEqual, WaveActiveBallot,
        // WaveReadLaneFirst, WaveActiveOp, WaveActiveBit, WaveAllBitCount.
        113..=116 | 118..=120 | 135 => Class::Wave,
        117 => Class::LaneAt,
        _ => Class::Varying,
    }
}

/// Immediate dominators of the graph over `0..=n` (node `n` the root) that
/// `next` walks from the root; `None` for nodes it does not reach (Cooper,
/// Harvey and Kennedy's iteration).
fn idoms(n: usize, next: &dyn Fn(usize) -> Vec<usize>, back: &dyn Fn(usize) -> Vec<usize>) -> Vec<Option<usize>> {
    let mut seen = vec![false; n + 1];
    let mut post = Vec::new();
    let mut stack = vec![(n, next(n), 0usize)];
    seen[n] = true;
    while let Some(top) = stack.last_mut() {
        let k = top.2;
        top.2 += 1;
        match top.1.get(k).copied() {
            Some(s) if !seen[s] => {
                seen[s] = true;
                let succ = next(s);
                stack.push((s, succ, 0));
            }
            Some(_) => {}
            None => {
                post.push(top.0);
                stack.pop();
            }
        }
    }
    let mut number = vec![usize::MAX; n + 1];
    for (k, &b) in post.iter().enumerate() {
        number[b] = k;
    }
    let mut idom = vec![usize::MAX; n + 1];
    idom[n] = n;
    let mut changed = true;
    while changed {
        changed = false;
        for &b in post.iter().rev().skip(1) {
            let mut new = usize::MAX;
            for p in back(b) {
                if idom[p] == usize::MAX {
                    continue;
                }
                if new == usize::MAX {
                    new = p;
                    continue;
                }
                let (mut x, mut y) = (p, new);
                while x != y {
                    while number[x] < number[y] {
                        x = idom[x];
                    }
                    while number[y] < number[x] {
                        y = idom[y];
                    }
                }
                new = x;
            }
            if new != usize::MAX && idom[b] != new {
                idom[b] = new;
                changed = true;
            }
        }
    }
    idom.into_iter().map(|d| (d != usize::MAX).then_some(d)).collect()
}

impl Module {
    /// The `dx.op` opcode of instruction `i`, when it is such a call.
    fn dx_op(&self, i: u32) -> Option<i64> {
        let Op::Call(f, args) = &self.insts[i as usize].op else {
            return None;
        };
        if !self.funcs[*f as usize].name.starts_with("dx.op.") {
            return None;
        }
        self.int_value(*args.first()?)
    }

    /// Sets the non-uniform operand of every heap handle whose index is
    /// varying, and with `masked` also of those whose index is masked (see
    /// the module notes). Returns the handles marked here, those of them
    /// whose index is varying, and all heap handles.
    pub(crate) fn mark_nonuniform(&mut self, masked: bool) -> (usize, usize, usize) {
        let n = self.blocks.len();
        if n == 0 {
            return (0, 0, 0);
        }
        let reached = self.reached();
        let succs = |b: usize| -> Vec<usize> { successors(self.blocks[b].term).to_vec() };
        let mut preds = vec![Vec::new(); n];
        for b in (0..n).filter(|&b| reached[b]) {
            for s in succs(b) {
                preds[s].push(b);
            }
        }
        // Post-dominators: the reverse graph from a virtual exit (node n)
        // that every block without a successor leads to.
        let exits: Vec<usize> = (0..n).filter(|&b| reached[b] && succs(b).is_empty()).collect();
        let ipdom = idoms(
            n,
            &|b| if b == n { exits.clone() } else { preds[b].clone() },
            &|b| if succs(b).is_empty() { vec![n] } else { succs(b) },
        );
        let mut block_of = vec![usize::MAX; self.insts.len()];
        for b in (0..n).filter(|&b| reached[b]) {
            for &i in self.blocks[b].phis.iter().chain(&self.blocks[b].body) {
                block_of[i as usize] = b;
            }
        }
        let order: Vec<u32> = (0..n)
            .filter(|&b| reached[b])
            .flat_map(|b| self.blocks[b].phis.iter().chain(&self.blocks[b].body).copied())
            .collect();
        let ops: Vec<Option<i64>> = (0..self.insts.len() as u32).map(|i| self.dx_op(i)).collect();
        let is = |set: &[bool], v: V| matches!(v, V::Inst(i) if set[i as usize]);

        // Varying values, blocks only some lanes run, and joins of lanes
        // that split: each grows the others until nothing changes.
        let mut varying = vec![false; self.insts.len()];
        let mut partial = vec![false; n];
        let mut join = vec![false; n];
        let mut split = vec![false; n];
        loop {
            let mut changed = true;
            while changed {
                changed = false;
                for &i in &order {
                    if varying[i as usize] {
                        continue;
                    }
                    let inst = &self.insts[i as usize];
                    let mut any = false;
                    for_each_operand(&inst.op, |v| any |= is(&varying, v));
                    let now = match &inst.op {
                        Op::Phi(_) => any || join[block_of[i as usize]],
                        Op::AtomicRmw(..) | Op::CmpXchg(..) => true,
                        Op::Call(_, args) => match ops[i as usize].map(class) {
                            Some(Class::Pure | Class::Read) => any,
                            Some(Class::Wave) => false,
                            Some(Class::LaneAt) => args.get(2).is_some_and(|&l| is(&varying, l)),
                            Some(Class::Varying) | None => true,
                        },
                        _ => any,
                    };
                    if now {
                        varying[i as usize] = true;
                        changed = true;
                    }
                }
            }
            let mut grew = false;
            for x in 0..n {
                if !reached[x] || split[x] {
                    continue;
                }
                let Some(Term::CondBr(c, ..)) = self.blocks[x].term else {
                    continue;
                };
                if !is(&varying, c) {
                    continue;
                }
                split[x] = true;
                grew = true;
                // The region: what the two sides reach before they meet.
                let stop = ipdom[x].unwrap_or(n);
                let mut stack = succs(x);
                let mut seen = vec![false; n];
                while let Some(b) = stack.pop() {
                    if b == stop || seen[b] {
                        continue;
                    }
                    seen[b] = true;
                    partial[b] = true;
                    if preds[b].len() > 1 {
                        join[b] = true;
                    }
                    stack.extend(succs(b));
                }
                if stop < n && preds[stop].len() > 1 {
                    join[stop] = true;
                }
            }
            if !grew {
                break;
            }
        }

        // Masked: read in a block only some lanes run, or computed from
        // such a read (or from a varying value).
        let check = masked;
        let mut masked = varying.clone();
        let mut changed = check;
        while changed {
            changed = false;
            for &i in &order {
                if masked[i as usize] {
                    continue;
                }
                let inst = &self.insts[i as usize];
                let read = partial[block_of[i as usize]]
                    && match &inst.op {
                        Op::Phi(_) | Op::Load(_) => true,
                        Op::Call(..) => !matches!(ops[i as usize].map(class), Some(Class::Pure)),
                        _ => false,
                    };
                let mut any = read;
                for_each_operand(&inst.op, |v| any |= is(&masked, v));
                if any {
                    masked[i as usize] = true;
                    changed = true;
                }
            }
        }

        let (mut marked, mut lanes, mut handles) = (0, 0, 0);
        // The constant only when a handle takes it: an unused one would
        // change the module of every program.
        let mut yes = None;
        for &i in &order {
            if ops[i as usize] != Some(218) {
                continue;
            }
            handles += 1;
            let Op::Call(_, args) = &self.insts[i as usize].op else {
                continue;
            };
            let (Some(&index), Some(&flag)) = (args.get(1), args.get(3)) else {
                continue;
            };
            if !is(&masked, index) || self.int_value(flag) == Some(-1) {
                continue;
            }
            let k = *yes.get_or_insert_with(|| self.bool(true));
            if let Op::Call(_, args) = &mut self.insts[i as usize].op {
                args[3] = k;
            }
            marked += 1;
            lanes += usize::from(is(&varying, index));
        }
        (marked, lanes, handles)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Attr, Bin, Module, Op, Term, V, pred};

    /// A module with `main` and the calls the pass tells apart: each takes
    /// its opcode first, as `dx.op` calls do.
    struct Kit {
        m: Module,
        f: V,
        heap: V,
    }

    fn kit() -> Kit {
        let mut m = Module::default();
        let i32t = m.int(32);
        let void = m.void();
        let fty = m.func_ty(i32t, vec![i32t, i32t]);
        let hty = m.func_ty(i32t, vec![i32t, i32t, i32t, i32t]);
        let f = m.function("dx.op.any.i32", fty, Some(Attr::ReadOnly), false);
        let heap = m.function("dx.op.createHandleFromHeap", hty, Some(Attr::ReadNone), false);
        let main_ty = m.func_ty(void, vec![]);
        m.function("main", main_ty, None, true);
        Kit { m, f, heap }
    }

    impl Kit {
        fn op(&mut self, b: usize, code: u32, x: V) -> V {
            let c = self.m.i32(code);
            self.m.call(b, self.f, vec![c, x])
        }
        fn handle(&mut self, b: usize, index: V) -> V {
            let (c, no) = (self.m.i32(218), self.m.bool(false));
            self.m.call(b, self.heap, vec![c, index, no, no])
        }
        fn nonuniform(&self, h: V) -> bool {
            let V::Inst(i) = h else { panic!("a call") };
            let Op::Call(_, args) = &self.m.insts[i as usize].op else { panic!("a call") };
            self.m.int_value(args[3]) == Some(-1)
        }
    }

    #[test]
    fn a_slot_read_where_only_some_lanes_run_is_non_uniform() {
        // the shape (0451d340): lanes with thread id 0 read a slot from
        // a buffer and index the heap with it.
        let mut k = kit();
        let (b0, b1, b2) = (k.m.new_block(), k.m.new_block(), k.m.new_block());
        let (zero, five) = (k.m.i32(0), k.m.i32(5));
        let table = k.op(b0, 139, zero);
        let entry = k.handle(b0, table);
        let tid = k.op(b0, 93, zero);
        let first = k.m.cmp(b0, pred::EQ, tid, zero);
        k.m.terminate(b0, Term::CondBr(first, b1, b2));
        let slot = k.op(b1, 139, five);
        let index = k.m.bin(b1, Bin::LShr, slot, five);
        let read = k.handle(b1, index);
        let again = k.handle(b1, table);
        let shifted = k.m.bin(b1, Bin::Add, table, five);
        let computed = k.handle(b1, shifted);
        k.m.terminate(b1, Term::Br(b2));
        k.m.terminate(b2, Term::Ret);
        assert_eq!(k.m.clone().mark_nonuniform(false), (0, 0, 4), "the hardware: no lane varies");
        assert_eq!(k.m.mark_nonuniform(true), (1, 0, 4));
        assert!(k.nonuniform(read), "read inside the lanes' branch");
        // Read where every lane runs: uniform, also when used or computed on
        // inside the branch.
        assert!(!k.nonuniform(entry) && !k.nonuniform(again) && !k.nonuniform(computed));
    }

    #[test]
    fn lane_values_and_joins_of_split_lanes_are_non_uniform() {
        let mut k = kit();
        let i32t = k.m.int(32);
        let blocks: Vec<usize> = (0..7).map(|_| k.m.new_block()).collect();
        let [b0, b1, b2, b3, b4, b5, b6] = blocks[..] else { unreachable!() };
        let (zero, one, two) = (k.m.i32(0), k.m.i32(1), k.m.i32(2));
        let tid = k.op(b0, 93, zero);
        let own = k.handle(b0, tid);
        let root = k.op(b0, 59, zero);
        // A branch on a constant buffer value: every lane goes one way.
        let set = k.m.cmp(b0, pred::NE, root, zero);
        k.m.terminate(b0, Term::CondBr(set, b1, b2));
        k.m.terminate(b1, Term::Br(b2));
        let same = k.m.phi(b2, i32t);
        k.m.set_phi(same, vec![(one, b0), (two, b1)]);
        let uniform = k.handle(b2, same);
        // A branch on the thread id: the join holds each lane's side.
        let odd = k.m.cmp(b2, pred::NE, tid, zero);
        k.m.terminate(b2, Term::CondBr(odd, b3, b4));
        k.m.terminate(b3, Term::Br(b5));
        k.m.terminate(b4, Term::Br(b5));
        let side = k.m.phi(b5, i32t);
        k.m.set_phi(side, vec![(one, b3), (two, b4)]);
        let joined = k.handle(b5, side);
        k.m.terminate(b5, Term::Br(b6));
        k.m.terminate(b6, Term::Ret);
        assert_eq!(k.m.mark_nonuniform(false), (2, 2, 3));
        assert!(k.nonuniform(own) && k.nonuniform(joined));
        assert!(!k.nonuniform(uniform));
    }

    #[test]
    fn a_counter_of_a_loop_lanes_leave_apart_is_non_uniform() {
        let mut k = kit();
        let i32t = k.m.int(32);
        let (b0, head, exit) = (k.m.new_block(), k.m.new_block(), k.m.new_block());
        let (zero, one) = (k.m.i32(0), k.m.i32(1));
        let tid = k.op(b0, 93, zero);
        k.m.terminate(b0, Term::Br(head));
        let count = k.m.phi(head, i32t);
        let next = k.m.bin(head, Bin::Add, count, one);
        let more = k.m.cmp(head, pred::IULT, next, tid);
        k.m.terminate(head, Term::CondBr(more, head, exit));
        k.m.set_phi(count, vec![(zero, b0), (next, head)]);
        let after = k.handle(exit, next);
        k.m.terminate(exit, Term::Ret);
        assert_eq!(k.m.mark_nonuniform(false), (1, 1, 1));
        assert!(k.nonuniform(after));
    }
}
