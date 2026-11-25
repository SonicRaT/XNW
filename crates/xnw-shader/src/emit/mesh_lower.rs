//! XNW): a mesh program lowered as a compute shader that
//! writes one fixed-size record per group instead of mesh outputs; the
//! record layout is [`super::super::mesh::Layout`], the draw side
//! [`super::super::mesh::passthrough_vs`]. A child of `lower` (it reuses the
//! mesh lowering and only redirects the outputs).
//!
//! What changes against the mesh-shader form (`lower.rs` `mesh_prologue`,
//! `export`, `emit_indices`):
//!
//! * prologue: no `SetMeshOutputCounts`; thread 0 zeroes the record header
//!   and every thread writes its primitive slot as a degenerate triangle
//!   (the console draws nothing for a primitive no lane exports);
//! * `s_sendmsg` GS_ALLOC_REQ: the lane stores the requested vertex and
//!   primitive counts in the header (VIctory found culled groups skip
//!   the request: their header stays 0 and the draw drops every vertex);
//! * `exp pos`/`exp param`: each exported dword is stored raw at its
//!   signature row and channel in the lane's vertex slot;
//! * `exp prim`: the packed word is stored in the record and decoded into
//!   the group's index triple (`group * max_vertices + index`); bit 31, or an
//!   index past the maximum (VIctory's bound), makes it degenerate.

use super::{Bin, Lower, V, Val, Why, pred};
use crate::emit::bitcode::Attr;
use crate::emit::mesh::{HEADER_BYTES, INDEX_REGISTER, Layout, RECORD_REGISTER};
use crate::io::Export;
use crate::ir::Inst;

impl Lower<'_> {
    fn fallback_layout(&self) -> Layout {
        let m = self.s.mesh.expect("mesh");
        Layout::new(&self.s.outputs, m.max_vertices, m.max_primitives)
    }

    /// The group id (x) of the compute dispatch.
    fn fallback_group(&mut self) -> V {
        let Val::I32(g) = self.compute_id("groupId.i32", 94, 0) else {
            unreachable!()
        };
        g
    }

    /// `rawBufferStore.i32` of `values` (1 to 4 dwords) at byte `offset`.
    fn fallback_raw_store(&mut self, uav: u32, offset: V, values: &[V]) {
        let h = self.s.raw_view(true, uav);
        let c = self.cur;
        let iu = self.s.m.undef(self.s.t.i32);
        let mut args = vec![h, offset, iu];
        args.extend(values);
        args.resize(3 + 4, iu);
        args.push(self.s.m.i8((1u8 << values.len()) - 1));
        args.push(self.s.m.i32(4));
        let void = self.s.t.void;
        self.s
            .op(c, "rawBufferStore.i32", Attr::NoUnwind, void, 140, &args);
    }

    pub(super) fn fallback_prologue(&mut self, mesh: crate::emit::shader::Mesh) -> Result<(), Why> {
        let threads = self.s.threads.expect("mesh")[0];
        if mesh.topology != 2 {
            return Err("a mesh shader not drawing triangles");
        }
        if threads > 1024 || mesh.max_vertices > 256 || mesh.max_primitives > 256 {
            return Err("a mesh shader beyond the host's limits");
        }
        if self.s.outputs.iter().any(|o| o.row.is_none()) {
            return Err("a mesh output without a register (compute fallback)");
        }
        let l = self.fallback_layout();
        let flat = self.mesh_thread();
        let g = self.fallback_group();
        let e = self.s.entry;
        let rb = self.s.m.i32(l.record_bytes());
        let rec = self.s.m.bin(e, Bin::Mul, g, rb);
        // Thread 0: the header's counts start at 0 (no request seen).
        let z = self.s.m.i32(0);
        let first = self.s.m.cmp(self.cur, pred::EQ, flat, z);
        let at = self.begin_guarded(Some(first));
        self.fallback_raw_store(RECORD_REGISTER, rec, &[z, z, z, z]);
        self.end_guarded(at, &[]);
        // Every primitive slot: degenerate until exported. A thread owns
        // slots `flat`, `flat + threads`, ... (with fewer threads than
        // primitives the slots past the threads were left as whatever the
        // memory held, indices into any record).
        let p = self.s.m.i32(mesh.max_primitives);
        let mut first_slot = 0;
        while first_slot < mesh.max_primitives {
            let c = self.cur;
            let k = self.s.m.i32(first_slot);
            let slot_in = self.s.m.bin(c, Bin::Add, flat, k);
            let guard = (first_slot + threads > mesh.max_primitives).then(|| self.s.m.cmp(self.cur, pred::IULT, slot_in, p));
            let at = self.begin_guarded(guard);
            let c = self.cur;
            let mv = self.s.m.i32(mesh.max_vertices);
            let base = self.s.m.bin(c, Bin::Mul, g, mv);
            let slot = self.s.m.bin(c, Bin::Mul, g, p);
            let slot = self.s.m.bin(c, Bin::Add, slot, slot_in);
            let twelve = self.s.m.i32(12);
            let off = self.s.m.bin(c, Bin::Mul, slot, twelve);
            self.fallback_raw_store(INDEX_REGISTER, off, &[base, base, base]);
            self.end_guarded(at, &[]);
            first_slot += threads.max(1);
        }
        Ok(())
    }

    /// Byte offset of this lane's vertex slot in its group's record.
    fn fallback_vertex(&mut self, l: &Layout) -> V {
        let flat = self.mesh_thread();
        let g = self.fallback_group();
        let c = self.cur;
        let rb = self.s.m.i32(l.record_bytes());
        let rec = self.s.m.bin(c, Bin::Mul, g, rb);
        let s = self.s.m.i32(l.vertex_stride());
        let v = self.s.m.bin(c, Bin::Mul, flat, s);
        let h = self.s.m.i32(HEADER_BYTES);
        let v = self.s.m.bin(c, Bin::Add, v, h);
        self.s.m.bin(c, Bin::Add, rec, v)
    }

    /// One exported dword of output element `id`, channel `col` (relative
    /// to the element), as raw bits.
    pub(super) fn fallback_store(&mut self, id: usize, col: u8, x: V) -> Result<(), Why> {
        let o = &self.s.outputs[id];
        let row = o.row.ok_or("a mesh output without a register (compute fallback)")?;
        let ch = u32::from(o.col + col);
        let l = self.fallback_layout();
        // A lane past the record's vertices writes nothing (it wrote
        // into the next group's record, or past the segment's end).
        let guard = self.past_guard(l.max_vertices);
        let g = self.begin_guarded(guard);
        let at = self.fallback_vertex(&l);
        let c = self.cur;
        let k = self.s.m.i32(16 * row + 4 * ch);
        let off = self.s.m.bin(c, Bin::Add, at, k);
        self.fallback_raw_store(RECORD_REGISTER, off, &[x]);
        self.end_guarded(g, &[]);
        Ok(())
    }

    /// `exp prim`: the packed word into the record, the triple into the
    /// group's index slots.
    pub(super) fn fallback_indices(&mut self, e: &Export, guard: Option<V>) -> Result<(), Why> {
        let v = e.values[0].ok_or("a primitive export without its indices")?;
        let x = self.i32_of(v)?;
        let l = self.fallback_layout();
        let at = self.begin_guarded(guard);
        // A lane past the record's primitives writes nothing.
        let past = self.past_guard(l.max_prims);
        let inner = self.begin_guarded(past);
        let flat = self.mesh_thread();
        let g = self.fallback_group();
        let c = self.cur;
        let z = self.s.m.i32(0);
        let mut dead = self.s.m.cmp(c, pred::SLT, x, z);
        let mv = self.s.m.i32(l.max_vertices);
        let base = self.s.m.bin(c, Bin::Mul, g, mv);
        let mut idx = Vec::new();
        for k in 0..3 {
            // VIctory (c0228d9): a 9-bit index; bits 9, 19, 29 are
            // edge flags, no part of it.
            let (shift, mask) = (self.s.m.i32(10 * k), self.s.m.i32(0x1FF));
            let i = self.s.m.bin(c, Bin::LShr, x, shift);
            let i = self.s.m.bin(c, Bin::And, i, mask);
            let past = self.s.m.cmp(c, pred::IUGE, i, mv);
            dead = self.s.m.bin(c, Bin::Or, dead, past);
            idx.push(self.s.m.bin(c, Bin::Add, base, i));
        }
        let idx: Vec<V> = idx.into_iter().map(|i| self.s.m.select(c, dead, base, i)).collect();
        let mp = self.s.m.i32(l.max_prims);
        let slot = self.s.m.bin(c, Bin::Mul, g, mp);
        let slot = self.s.m.bin(c, Bin::Add, slot, flat);
        let twelve = self.s.m.i32(12);
        let off = self.s.m.bin(c, Bin::Mul, slot, twelve);
        self.fallback_raw_store(INDEX_REGISTER, off, &idx);
        let rb = self.s.m.i32(l.record_bytes());
        let rec = self.s.m.bin(c, Bin::Mul, g, rb);
        let four = self.s.m.i32(4);
        let p = self.s.m.bin(c, Bin::Mul, flat, four);
        let po = self.s.m.i32(l.prim_offset());
        let p = self.s.m.bin(c, Bin::Add, p, po);
        let p = self.s.m.bin(c, Bin::Add, rec, p);
        self.fallback_raw_store(RECORD_REGISTER, p, &[x]);
        self.end_guarded(inner, &[]);
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// `flat < n` when the group has more threads than `n` (else no guard).
    fn past_guard(&mut self, n: u32) -> Option<V> {
        let threads = self.s.threads.expect("mesh")[0];
        if threads <= n {
            return None;
        }
        let flat = self.mesh_thread();
        let k = self.s.m.i32(n);
        Some(self.s.m.cmp(self.cur, pred::IULT, flat, k))
    }

    /// `s_sendmsg` GS_ALLOC_REQ: M0 = vertices | primitives << 12 into the
    /// header (dwords 0 and 1), and M0 itself (dword 2).
    pub(super) fn fallback_alloc(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let m0 = inst.read(crate::ir::Reg::S(crate::ir::M0)).ok_or("M0")?;
        let m0 = self.i32_of(m0)?;
        let guard = self.guard(b, i)?;
        let l = self.fallback_layout();
        let at = self.begin_guarded(guard);
        let g = self.fallback_group();
        let c = self.cur;
        let (k12, kfff) = (self.s.m.i32(12), self.s.m.i32(0xFFF));
        let verts = self.s.m.bin(c, Bin::And, m0, kfff);
        let prims = self.s.m.bin(c, Bin::LShr, m0, k12);
        let prims = self.s.m.bin(c, Bin::And, prims, kfff);
        let rb = self.s.m.i32(l.record_bytes());
        let rec = self.s.m.bin(c, Bin::Mul, g, rb);
        self.fallback_raw_store(RECORD_REGISTER, rec, &[verts, prims, m0]);
        self.end_guarded(at, &[]);
        Ok(())
    }
}
