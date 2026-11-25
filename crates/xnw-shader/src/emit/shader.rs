// Ported from VIctory crates/victory-shader/src/emit/shader.rs, commit d2a4bae1f49533ef5cb512fd1b1ba1513f2e1989, GPL-3.0-or-later.
// With VIctory (head 42538cd, on upstream 4322b61; GPL-3.0-or-later): unknown-stride structured buffers read through the runtime stride table (shader side).
//! A DXIL vertex, pixel, compute or mesh shader under construction: the module,
//! the `dx.op` declarations it calls, its signatures and bound resources,
//! and the metadata and container that wrap it up (SM 6.6, validator 1.6).

use super::bitcode::{Attr, Md, Module, Ty, V};
use super::container::{self, Binding, Psv, SigElement};

/// Where the shader reads its user data: `b0` in this space.
pub const USER_DATA_SPACE: u32 = 0;
/// Where immediate samplers are bound: `s<n>` in this space.
pub const SAMPLER_SPACE: u32 = 0;
/// Where root CBVs and SRVs are bound: `b<root parameter index>` and
/// `t<root parameter index>` in this space.
pub const ROOT_CBV_SPACE: u32 = 1;
/// Where the host's runtime stride table is bound: `t0` in this space
/// (one [`StrideEntry`] per view-heap slot).
/// Changed from VIctory (2): space 2 holds the mesh fallback's record views
/// here ([`super::mesh::RECORD_SPACE`], XNW).
pub const STRIDE_SPACE: u32 = 3;
/// XNW: the page table
/// (`XNW_PAGETABLE=1`), `t0` in this space: it takes the place of the
/// address map (VIctory) at the same binding.
pub const PAGE_TABLE_SPACE: u32 = 4;
/// Where a mesh fallback program reads the bytes its root SRVs may read
/// (XNW): `b0` in this space, one dword per root SRV in the order of
/// [`Shader::root_srvs`]; a raw load through a root SRV at an offset past
/// it reads offset 0 instead (a root descriptor has no bounds: VIctory,
/// mesh shaders reading megabytes past a root SRV's copy page-fault).
pub const ROOT_LEN_SPACE: u32 = 4;
/// Where the address map is bound: `t0` in this space (a root SRV), when
/// the shader reads title memory at addresses it computes
/// ([`crate::resource::GLOBAL`]; see [`super::HostShader::address_map`]).
/// VIctory (1a2a3a1, GPL-3.0-or-later); changed from VIctory (2):
/// spaces 2 to 4 are taken here (XNW; 4 until the root
/// lengths took 4).
pub const ADDRESS_MAP_SPACE: u32 = 5;
/// Where the descriptor words are bound: `t0` in this space (a root SRV),
/// when the shader reads descriptors as numbers beyond what host views
/// tell (see [`super::HostShader::descriptor_words`]). VIctory;
/// changed from VIctory (3), XNW.
pub const DESCRIPTOR_WORDS_SPACE: u32 = 6;
/// XNW: the heap bounds, `b0` in this space: 8
/// root constants the runtime sets per draw (view slots, the guard slots of
/// each kind, sampler slots, the sampler guard).
pub const HEAP_BOUNDS_SPACE: u32 = 7;

/// What the host writes per view-heap slot in the runtime stride table, for
/// buffers a shader reads without a stride known when it was translated.
///
/// The shader reads such a buffer through a raw handle, and a raw handle
/// reads a raw view only: over a structured or typed view D3D12 leaves the
/// result undefined (an RTX 5070 reads other dwords than a Radeon or WARP
/// do). So the host makes a raw view of the same bytes in another slot for
/// every buffer view that is not raw, and says here where it is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StrideEntry {
    /// Bytes per element of the view in the slot (0: no buffer view).
    pub stride: u32,
    /// The heap slot holding a raw view of the bytes the slot's view reads;
    /// none when the slot's own view is raw (or reads nothing).
    pub raw_slot: Option<u32>,
    /// Bytes from the start of that raw view to the slot's view's first
    /// byte: a multiple of 4 below 16 ([`RawWindow`]).
    pub bias: u32,
}

impl StrideEntry {
    /// Bytes per entry in the table: slot `i`'s entry is at `i * BYTES`.
    pub const BYTES: u32 = 8;
    /// The raw slots an entry can name.
    pub const RAW_SLOTS: u32 = (1 << 28) - 1;

    /// The entry as the shader reads it: the stride, then the raw slot plus
    /// one in the low 28 bits (0: the slot itself) under the bias. A table
    /// of zeros sends every slot to itself.
    pub fn words(self) -> [u32; 2] {
        debug_assert!(self.bias < 16);
        let raw = self.raw_slot.map_or(0, |s| {
            debug_assert!(s < Self::RAW_SLOTS);
            s + 1
        });
        [self.stride, raw | self.bias << 28]
    }
}

/// A raw view covering bytes `[start, start + len)` of a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawWindow {
    /// The view's first element (4 bytes each).
    pub first: u64,
    /// Its element count.
    pub dwords: u32,
    /// Bytes from the view's start to `start` ([`StrideEntry::bias`]).
    pub bias: u32,
}

impl RawWindow {
    /// The window for bytes `[start, start + len)`. The host wants raw
    /// views to start on 16 bytes (`D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT`), the
    /// console on 4, so the view starts up to 12 bytes early and the shader
    /// adds the difference. None for a start off a dword (raw loads drop
    /// the low address bits) or a view too long to count.
    pub fn covering(start: u64, len: u64) -> Option<RawWindow> {
        if !start.is_multiple_of(4) {
            return None;
        }
        let bias = start % 16;
        let dwords = u32::try_from((bias + len).div_ceil(4).max(1)).ok()?;
        Some(RawWindow {
            first: (start - bias) / 4,
            dwords,
            bias: bias as u32,
        })
    }
}

thread_local! {
    /// XNW: the last module this thread finished: its instruction count
    /// (after any clean-up pass), for the shader gate ([`super::last_insts`]).
    pub(crate) static INSTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
thread_local! {
    /// XNW: the last finished module's instruction counts before and
    /// after the clean-up passes, for the world tests.
    pub(crate) static OPT_STATS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
    /// XNW: the pass sets to finish each module with, and what came
    /// out (see `world_tests::opt_sweep`).
    pub(crate) static SWEEP: std::cell::RefCell<Option<Sweep>> = const { std::cell::RefCell::new(None) };
    static SWEEP_FLAGS: std::cell::RefCell<(Vec<String>, usize)> = const { std::cell::RefCell::new((Vec::new(), 0)) };
}

/// XNW: a pass sweep over one thread's translations.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct Sweep {
    pub sets: Vec<u32>,
    pub out: Vec<SweepResult>,
}

/// XNW: one module finished with one pass set.
#[cfg(test)]
pub(crate) struct SweepResult {
    pub passes: u32,
    pub dxil: Vec<u8>,
    pub before: usize,
    pub after: usize,
    /// The differential's flags, and the merges it allowed.
    pub flags: (Vec<String>, usize),
    pub micros: u64,
}

/// `ShaderFlags` bits (metadata) and their `SFI0` feature bits.
pub(crate) mod flags {
    pub const DOUBLES: u64 = 1 << 2;
    pub const DOUBLE_EXTENSIONS: u64 = 1 << 6;
    pub const RAW_BUFFERS: u64 = 1 << 4;
    pub const STENCIL_REF: u64 = 1 << 11;
    pub const TYPED_UAV_LOADS: u64 = 1 << 13;
    pub const WAVE_OPS: u64 = 1 << 19;
    pub const INT64: u64 = 1 << 20;
    /// XNW: 64-bit atomics on a typed resource, and on one reached
    /// through the descriptor heap.
    pub const ATOMIC_INT64_TYPED: u64 = 1 << 27;
    pub const RESOURCE_HEAP: u64 = 1 << 30;
    pub const SAMPLER_HEAP: u64 = 1 << 31;
    pub const ATOMIC_INT64_HEAP: u64 = 1 << 32;
}

#[derive(Clone)]
pub(crate) struct Types {
    pub void: Ty,
    pub i1: Ty,
    pub i32: Ty,
    pub f32: Ty,
    pub handle: Ty,
    pub props: Ty,
    pub bind: Ty,
    pub cbuf_ret: Ty,
    pub res_ret: Ty,
}

/// What a mesh shader declares beyond its thread-group size.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Mesh {
    pub max_vertices: u32,
    pub max_primitives: u32,
    /// 1 lines, 2 triangles.
    pub topology: u8,
}

#[derive(Clone)]
pub(crate) struct Shader {
    pub m: Module,
    pub t: Types,
    pub pixel: bool,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    /// A mesh shader's per-primitive outputs.
    pub prims: Vec<SigElement>,
    /// A mesh shader (`threads` is its thread-group size).
    pub mesh: Option<Mesh>,
    /// Dwords of user data the shader reads (0: no user-data buffer).
    pub user_data: u32,
    /// Immediate samplers, bound as `s0..` in [`SAMPLER_SPACE`].
    pub samplers: u32,
    /// Bit n: immediate sampler n is a comparison sampler.
    pub cmp_samplers: u64,
    /// Root CBVs read, by root parameter index, with their handles.
    pub root_cbvs: Vec<(u8, V)>,
    /// Root SRVs (raw buffers) read, likewise.
    pub root_srvs: Vec<(u8, V)>,
    /// Root UAVs (raw buffers) read or written, likewise (XNW).
    pub root_uavs: Vec<(u8, V)>,
    /// The address map's handle, once made (VIctory).
    pub address_map: Option<V>,
    /// The descriptor words' handle, once made (VIctory).
    pub descriptor_words: Option<V>,
    /// Handle of the runtime stride table (`t0`, [`STRIDE_SPACE`]), if read.
    pub stride_table: Option<V>,
    /// XNW: handle of the page table (`t0`, [`PAGE_TABLE_SPACE`]).
    pub page_table: Option<V>,
    /// XNW: page table on): what the program still reads through host
    /// copies ([`super::host_copies`] bits), written to the container's
    /// `PRIV` part as `XNWH` + the bits.
    pub host_copies: Option<u32>,
    /// XNW: the eight heap-limit words, loaded once in the entry block.
    heap_limits: Option<[V; 8]>,
    /// Mesh fallback: the root SRV lengths' buffer, and per root SRV
    /// handle the highest offset a 16-byte load may start at.
    root_lens: Option<V>,
    root_limits: Vec<(V, V)>,
    pub flags: u64,
    /// A compute shader with this thread-group size.
    pub threads: Option<[u32; 3]>,
    /// The wave size a compute shader requires (`WaveSize`).
    pub wave_size: Option<u32>,
    /// Thread-group shared memory, in dwords.
    pub lds_dwords: u32,
    /// A mesh program emitted as compute (XNW: `super::mesh`): lowered
    /// as a mesh shader, finished as `cs_6_6` without signatures.
    pub fallback: bool,
    /// Raw buffers bound by register in [`super::mesh::RECORD_SPACE`]
    /// (XNW): (UAV, register, handle).
    pub raw_views: Vec<(bool, u32, V)>,
    /// A geometry shader (XNW: `super::rect`): a triangle in, a strip
    /// of four vertices out, stream 0.
    pub geometry: bool,
    lds: Option<V>,
    main: V,
    pub entry: usize,
}

impl Shader {
    pub fn new(pixel: bool) -> Self {
        let mut m = Module::default();
        let void = m.void();
        let i1 = m.int(1);
        let i8 = m.int(8);
        let i32 = m.int(32);
        let f32 = m.float();
        let i8p = m.ptr(i8);
        let handle = m.named_struct("dx.types.Handle", vec![i8p]);
        let props = m.named_struct("dx.types.ResourceProperties", vec![i32, i32]);
        let bind = m.named_struct("dx.types.ResBind", vec![i32, i32, i32, i8]);
        let cbuf_ret = m.named_struct("dx.types.CBufRet.i32", vec![i32, i32, i32, i32]);
        let res_ret = m.named_struct("dx.types.ResRet.f32", vec![f32, f32, f32, f32, i32]);
        let main_ty = m.func_ty(void, vec![]);
        m.ptr(main_ty);
        let main = m.function("main", main_ty, None, true);
        let entry = m.new_block();
        Shader {
            m,
            t: Types {
                void,
                i1,
                i32,
                f32,
                handle,
                props,
                bind,
                cbuf_ret,
                res_ret,
            },
            pixel,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh: None,
            user_data: 0,
            samplers: 0,
            cmp_samplers: 0,
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            address_map: None,
            descriptor_words: None,
            stride_table: None,
            page_table: None,
            host_copies: None,
            heap_limits: None,
            root_lens: None,
            root_limits: Vec::new(),
            flags: 0,
            threads: None,
            wave_size: None,
            lds_dwords: 0,
            fallback: false,
            raw_views: Vec::new(),
            geometry: false,
            lds: None,
            main,
            entry,
        }
    }

    /// The thread-group shared memory: `[lds_dwords x i32]` in address
    /// space 3.
    pub fn lds(&mut self) -> V {
        if let Some(g) = self.lds {
            return g;
        }
        let arr = self.m.array(self.lds_dwords, self.t.i32);
        let g = self.m.global("lds", arr, 3);
        self.lds = Some(g);
        g
    }

    /// Calls `dx.op.<name>` (class and overload) with `opcode` and `args`.
    pub fn op(
        &mut self,
        block: usize,
        name: &str,
        attr: Attr,
        ret: Ty,
        opcode: u32,
        args: &[V],
    ) -> V {
        let mut params = vec![self.t.i32];
        params.extend(args.iter().map(|&a| self.m.type_of(a)));
        let fty = self.m.func_ty(ret, params);
        self.m.ptr(fty);
        let f = self
            .m
            .function(&format!("dx.op.{name}"), fty, Some(attr), false);
        let mut all = vec![self.m.i32(opcode)];
        all.extend(args);
        // A raw load through a bounded root SRV past its limit reads
        // offset 0.
        if opcode == 139
            && let Some(&(_, limit)) = self.root_limits.iter().find(|r| Some(&r.0) == args.first())
        {
            let at = all[2];
            let ok = self.m.cmp(block, super::bitcode::pred::IULE, at, limit);
            let zero = self.m.i32(0);
            all[2] = self.m.select(block, ok, at, zero);
        }
        self.m.call(block, f, all)
    }

    /// The handle of the user-data constant buffer, made in the entry block.
    pub fn user_data_handle(&mut self) -> V {
        let bind = [
            self.m.i32(0),
            self.m.i32(0),
            self.m.i32(USER_DATA_SPACE),
            self.m.i8(2), // CBV
        ];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let block = self.entry;
        let (i, f) = (self.m.i32(0), self.m.bool(false));
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, i, f],
        );
        // The size is known in `finish`, which fixes this placeholder.
        self.annotate(block, h, 13, u32::MAX)
    }

    /// The handle of the root CBV of root parameter `slot`, made in the
    /// entry block.
    pub fn root_cbv_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_cbvs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(2)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 13, 65536);
        self.root_cbvs.push((slot, h));
        h
    }

    /// The handle of the root SRV (a raw buffer) of root parameter `slot`.
    pub fn root_srv_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_srvs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(0)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 11, 0);
        self.flags |= flags::RAW_BUFFERS;
        let k = self.root_srvs.len() as u32;
        self.root_srvs.push((slot, h));
        if self.fallback {
            // The bytes this root SRV may read, from the lengths buffer:
            // a 16-byte load may start at most 16 bytes before the end.
            let lens = self.root_lens_handle();
            let row = self.m.i32(k / 4);
            let r = self.op(block, "cbufferLoadLegacy.i32", Attr::ReadOnly, self.t.cbuf_ret, 59, &[lens, row]);
            let len = self.m.extract(block, r, k % 4);
            let sixteen = self.m.i32(16);
            let room = self.m.cmp(block, super::bitcode::pred::IUGT, len, sixteen);
            let less = self.m.bin(block, super::bitcode::Bin::Sub, len, sixteen);
            let zero = self.m.i32(0);
            let limit = self.m.select(block, room, less, zero);
            self.root_limits.push((h, limit));
        }
        h
    }

    /// The handle of the root SRV lengths' buffer (`b0` in
    /// [`ROOT_LEN_SPACE`]), made once in the entry block; its size is
    /// fixed in `finish`.
    fn root_lens_handle(&mut self) -> V {
        if let Some(h) = self.root_lens {
            return h;
        }
        let z = self.m.i32(0);
        let bind = [z, z, self.m.i32(ROOT_LEN_SPACE), self.m.i8(2)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(block, "createHandleFromBinding", Attr::ReadNone, self.t.handle, 217, &[bind, z, f]);
        let h = self.annotate(block, h, 13, u32::MAX - 1);
        self.root_lens = Some(h);
        h
    }

    /// The handle of the root UAV (a raw buffer, `u<slot>` in
    /// [`ROOT_CBV_SPACE`]) of root parameter `slot` (VIctory,
    /// bffaefd, GPL-3.0-or-later; XNW).
    pub fn root_uav_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_uavs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(1)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(block, "createHandleFromBinding", Attr::ReadNone, self.t.handle, 217, &[bind, r, f]);
        let h = self.annotate(block, h, 0x100B, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.root_uavs.push((slot, h));
        h
    }

    /// The handle of the address map (`t0` in [`ADDRESS_MAP_SPACE`], a raw
    /// buffer), made in the entry block (VIctory).
    pub fn address_map_handle(&mut self) -> V {
        if let Some(h) = self.address_map {
            return h;
        }
        let h = self.space_handle(ADDRESS_MAP_SPACE);
        self.address_map = Some(h);
        h
    }

    /// The handle of the descriptor words (`t0` in
    /// [`DESCRIPTOR_WORDS_SPACE`], a raw buffer), made in the entry block.
    pub fn descriptor_words_handle(&mut self) -> V {
        if let Some(h) = self.descriptor_words {
            return h;
        }
        let h = self.space_handle(DESCRIPTOR_WORDS_SPACE);
        self.descriptor_words = Some(h);
        h
    }

    /// A raw buffer bound at `t0` in `space`, made in the entry block.
    fn space_handle(&mut self, space: u32) -> V {
        let r = self.m.i32(0);
        let bind = [r, r, self.m.i32(space), self.m.i8(0)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(block, "createHandleFromBinding", Attr::ReadNone, self.t.handle, 217, &[bind, r, f]);
        let h = self.annotate(block, h, 11, 0);
        self.flags |= flags::RAW_BUFFERS;
        h
    }

    /// The handle of the raw buffer `u<register>` (UAV) or `t<register>`
    /// in [`super::mesh::RECORD_SPACE`], made in the entry block (XNW).
    pub fn raw_view(&mut self, uav: bool, register: u32) -> V {
        if let Some(&(_, _, h)) = self.raw_views.iter().find(|r| r.0 == uav && r.1 == register) {
            return h;
        }
        let r = self.m.i32(register);
        let class = if uav { 1 } else { 0 };
        let bind = [r, r, self.m.i32(super::mesh::RECORD_SPACE), self.m.i8(class)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        // Raw buffer (11), UAV bit 12.
        let h = self.annotate(block, h, if uav { 0x100B } else { 11 }, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.raw_views.push((uav, register, h));
        h
    }

    /// The handle of the host's runtime stride table (`t0` in
    /// [`STRIDE_SPACE`], a raw buffer of [`StrideEntry`] words indexed by
    /// view-heap slot).
    pub fn stride_table_handle(&mut self) -> V {
        if let Some(h) = self.stride_table {
            return h;
        }
        let z = self.m.i32(0);
        let bind = [z, z, self.m.i32(STRIDE_SPACE), self.m.i8(0)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, z, f],
        );
        let h = self.annotate(block, h, 11, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.stride_table = Some(h);
        h
    }

    /// XNW: the handle of the page table (`t0` in [`PAGE_TABLE_SPACE`]).
    pub fn page_table_handle(&mut self) -> V {
        if let Some(h) = self.page_table {
            return h;
        }
        let z = self.m.i32(0);
        let bind = [z, z, self.m.i32(PAGE_TABLE_SPACE), self.m.i8(0)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(block, "createHandleFromBinding", Attr::ReadNone, self.t.handle, 217, &[bind, z, f]);
        let h = self.annotate(block, h, 11, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.page_table = Some(h);
        h
    }

    /// A descriptor heap handle whose index may differ between lanes
    /// (`NonUniformResourceIndex`), annotated with `props`.
    pub fn heap_handle_nonuniform(&mut self, block: usize, index: V, props: (u32, u32)) -> V {
        self.heap_handle_with(block, index, false, true, props)
    }

    /// The handle of immediate sampler `n` (a comparison sampler when its
    /// bit in `cmp_samplers` is set).
    pub fn sampler_handle(&mut self, block: usize, n: u32) -> V {
        let bind = [
            self.m.i32(n),
            self.m.i32(n),
            self.m.i32(SAMPLER_SPACE),
            self.m.i8(3), // sampler
        ];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let (i, f) = (self.m.i32(n), self.m.bool(false));
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, i, f],
        );
        let cmp = if self.cmp_samplers >> n & 1 != 0 {
            0x8000
        } else {
            0
        };
        self.annotate(block, h, 14 | cmp, 0)
    }

    pub fn annotate(&mut self, block: usize, h: V, props0: u32, props1: u32) -> V {
        let p = [self.m.i32(props0), self.m.i32(props1)];
        let p = self.m.aggregate(self.t.props, &p);
        self.op(
            block,
            "annotateHandle",
            Attr::ReadNone,
            self.t.handle,
            216,
            &[h, p],
        )
    }

    /// XNW: after VIctory 05c681c, GPL-3.0-or-later; words of
    /// spec section 7): `index` if it lies in its heap, else the heap's
    /// permanent null of the kind `props` names: a select on the index
    /// itself, before any access (an index past the heap hung the RX 5700 XT
    /// even on a path whose result is discarded). The eight words, `b0` in
    /// [`HEAP_BOUNDS_SPACE`]: view slots, null 2D texture SRV, null raw
    /// buffer SRV, null raw buffer UAV, null 2D texture UAV, null CBV,
    /// sampler slots, null sampler.
    fn bounded(&mut self, block: usize, index: V, sampler: bool, props: (u32, u32)) -> V {
        let w = match self.heap_limits {
            Some(w) => w,
            None => {
                let z = self.m.i32(0);
                let bind = [z, z, self.m.i32(HEAP_BOUNDS_SPACE), self.m.i8(2)];
                let bind = self.m.aggregate(self.t.bind, &bind);
                let f = self.m.bool(false);
                let entry = self.entry;
                let h = self.op(entry, "createHandleFromBinding", Attr::ReadNone, self.t.handle, 217, &[bind, z, f]);
                let h = self.annotate(entry, h, 13, 32);
                let one = self.m.i32(1);
                let r0 = self.op(entry, "cbufferLoadLegacy.i32", Attr::ReadOnly, self.t.cbuf_ret, 59, &[h, z]);
                let r1 = self.op(entry, "cbufferLoadLegacy.i32", Attr::ReadOnly, self.t.cbuf_ret, 59, &[h, one]);
                let mut w = [z; 8];
                for (k, slot) in w.iter_mut().enumerate() {
                    *slot = self.m.extract(entry, if k < 4 { r0 } else { r1 }, k as u32 % 4);
                }
                self.heap_limits = Some(w);
                w
            }
        };
        let uav = props.0 & 0x1000 != 0;
        let (limit, null) = match (sampler, props.0 & 0xFF, uav) {
            (true, _, _) => (w[6], w[7]),
            (false, 13, _) => (w[0], w[5]),
            (false, 10..=12, false) => (w[0], w[2]),
            (false, 10..=12, true) => (w[0], w[3]),
            (false, _, false) => (w[0], w[1]),
            (false, _, true) => (w[0], w[4]),
        };
        let inside = self.m.cmp(block, super::bitcode::pred::IULT, index, limit);
        self.m.select(block, inside, index, null)
    }

    /// A descriptor heap handle, annotated with `props`; the index is
    /// bounded first ([`Shader::bounded`]).
    pub fn heap_handle(&mut self, block: usize, index: V, sampler: bool, props: (u32, u32)) -> V {
        self.heap_handle_with(block, index, sampler, false, props)
    }

    /// [`Shader::heap_handle`], with a non-uniform index when `nonuniform`
    /// (lanes may name different slots; XNW, for the address map).
    pub fn heap_handle_with(&mut self, block: usize, index: V, sampler: bool, nonuniform: bool, props: (u32, u32)) -> V {
        self.flags |= if sampler {
            flags::SAMPLER_HEAP
        } else {
            flags::RESOURCE_HEAP
        };
        let index = self.bounded(block, index, sampler, props);
        let (s, n) = (self.m.bool(sampler), self.m.bool(nonuniform));
        let h = self.op(
            block,
            "createHandleFromHeap",
            Attr::ReadNone,
            self.t.handle,
            218,
            &[index, s, n],
        );
        self.annotate(block, h, props.0, props.1)
    }

    /// The finished container.
    pub fn finish(self) -> Vec<u8> {
        // XNW: with a sweep asked for, the module is also finished
        // with each pass set of the sweep, on a copy.
        #[cfg(test)]
        if let Some(sets) = SWEEP.with(|s| s.borrow().as_ref().map(|s| s.sets.clone())) {
            for passes in sets {
                let copy = self.clone();
                let started = std::time::Instant::now();
                let dxil = copy.finish_with(passes, true);
                let (before, after) = OPT_STATS.with(|c| c.get());
                let flags = SWEEP_FLAGS.with(|f| std::mem::take(&mut *f.borrow_mut()));
                SWEEP.with(|s| {
                    if let Some(s) = s.borrow_mut().as_mut() {
                        s.out.push(SweepResult { passes, dxil, before, after, flags, micros: started.elapsed().as_micros() as u64 });
                    }
                });
            }
        }
        self.finish_with(super::shader_opt(), false)
    }

    fn finish_with(mut self, passes: u32, check: bool) -> Vec<u8> {
        let _ = check;
        if self.fallback {
            // A compute shader from here on: no signatures, no mesh state.
            self.mesh = None;
            self.inputs.clear();
            self.outputs.clear();
            self.prims.clear();
        }
        let ud_size = (self.user_data * 4).next_multiple_of(16);
        self.fix_user_data_size(ud_size);
        // The root SRV lengths' buffer, one dword per root SRV.
        let lens_size = (self.root_srvs.len() as u32 * 4).next_multiple_of(16).max(16);
        if self.root_lens.is_some() {
            let old = [self.m.i32(13), self.m.i32(u32::MAX - 1)];
            let old = self.m.aggregate(self.t.props, &old);
            let new = [self.m.i32(13), self.m.i32(lens_size)];
            let new = self.m.aggregate(self.t.props, &new);
            self.m.replace_const(old, new);
        }
        // victory-fork shader.rs `finish`: the clean-up, before
        // anything is declared about the code: a feature whose last use it
        // removed must not be declared (the validator compares the flags
        // with what the code uses).
        let passes = passes & super::bitcode::opt::MODULE;
        #[cfg(test)]
        let before = self.m.inst_count();
        #[cfg(test)]
        let off = if check { Some(self.m.clone()) } else { None };
        if passes != 0 {
            self.m.optimise(passes);
            #[cfg(test)]
            if let Some(off) = &off {
                let (flags, merged) = off.differential(&self.m);
                SWEEP_FLAGS.with(|f| *f.borrow_mut() = (flags, merged));
            }
            let used = self.m.usage();
            for (flag, used) in [
                (flags::WAVE_OPS, used.wave_ops),
                (flags::INT64, used.int64),
                (flags::TYPED_UAV_LOADS, used.typed_uav_loads),
                (flags::RESOURCE_HEAP, used.heap_views),
                (flags::SAMPLER_HEAP, used.heap_samplers),
                (flags::ATOMIC_INT64_TYPED | flags::ATOMIC_INT64_HEAP, used.atomic64),
                (flags::DOUBLES | flags::DOUBLE_EXTENSIONS, used.doubles),
            ] {
                if !used {
                    self.flags &= !flag;
                }
            }
        }
        let insts = self.m.inst_count();
        INSTS.with(|c| c.set(insts));
        #[cfg(test)]
        OPT_STATS.with(|c| c.set((before, insts)));
        let mut resources = None;
        let mut bindings = Vec::new();
        let i32c = |s: &mut Self, v: u32| {
            let c = s.m.i32(v);
            s.m.md_value(c)
        };
        if self.user_data > 0
            || self.samplers > 0
            || !self.root_cbvs.is_empty()
            || !self.root_srvs.is_empty()
            || !self.root_uavs.is_empty()
            || !self.raw_views.is_empty()
            || self.stride_table.is_some()
            || self.address_map.is_some()
            || self.descriptor_words.is_some()
            || self.page_table.is_some()
            || self.heap_limits.is_some()
        {
            let empty = self.m.md_str("");
            let mut cbvs = None;
            let mut cbv_list = Vec::new();
            if self.user_data > 0 {
                let arr = self.m.array(self.user_data.next_multiple_of(4), self.t.i32);
                let ty = self.m.named_struct("UserData", vec![arr]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let ops = vec![
                    Some(i32c(&mut self, 0)),
                    Some(self.m.md_value(u)),
                    Some(empty),
                    Some(i32c(&mut self, USER_DATA_SPACE)),
                    Some(i32c(&mut self, 0)),
                    Some(i32c(&mut self, 1)),
                    Some(i32c(&mut self, ud_size)),
                    None,
                ];
                cbv_list.push(Some(self.m.md_node(ops)));
                bindings.push(Binding {
                    kind: 2,
                    space: USER_DATA_SPACE,
                    register: 0,
                    dxil_kind: 13,
                });
            }
            if self.heap_limits.is_some() {
                let arr = self.m.array(8, self.t.i32);
                let ty = self.m.named_struct("HeapLimits", vec![arr]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let ops = vec![
                    Some(i32c(&mut self, cbv_list.len() as u32)),
                    Some(self.m.md_value(u)),
                    Some(empty),
                    Some(i32c(&mut self, HEAP_BOUNDS_SPACE)),
                    Some(i32c(&mut self, 0)),
                    Some(i32c(&mut self, 1)),
                    Some(i32c(&mut self, 32)),
                    None,
                ];
                cbv_list.push(Some(self.m.md_node(ops)));
                bindings.push(Binding {
                    kind: 2,
                    space: HEAP_BOUNDS_SPACE,
                    register: 0,
                    dxil_kind: 13,
                });
            }
            if self.root_lens.is_some() {
                let arr = self.m.array(lens_size / 4, self.t.i32);
                let ty = self.m.named_struct("RootLengths", vec![arr]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let ops = vec![
                    Some(i32c(&mut self, cbv_list.len() as u32)),
                    Some(self.m.md_value(u)),
                    Some(empty),
                    Some(i32c(&mut self, ROOT_LEN_SPACE)),
                    Some(i32c(&mut self, 0)),
                    Some(i32c(&mut self, 1)),
                    Some(i32c(&mut self, lens_size)),
                    None,
                ];
                cbv_list.push(Some(self.m.md_node(ops)));
                bindings.push(Binding {
                    kind: 2,
                    space: ROOT_LEN_SPACE,
                    register: 0,
                    dxil_kind: 13,
                });
            }
            let blob = self.m.array(16384, self.t.i32);
            let ty = self.m.named_struct("RootCbv", vec![blob]);
            let p = self.m.ptr(ty);
            let u = self.m.undef(p);
            for (slot, _) in self.root_cbvs.clone() {
                let ops = vec![
                    Some(i32c(&mut self, cbv_list.len() as u32)),
                    Some(self.m.md_value(u)),
                    Some(empty),
                    Some(i32c(&mut self, ROOT_CBV_SPACE)),
                    Some(i32c(&mut self, u32::from(slot))),
                    Some(i32c(&mut self, 1)),
                    Some(i32c(&mut self, 65536)),
                    None,
                ];
                cbv_list.push(Some(self.m.md_node(ops)));
                bindings.push(Binding {
                    kind: 2,
                    space: ROOT_CBV_SPACE,
                    register: u32::from(slot),
                    dxil_kind: 13,
                });
            }
            if !cbv_list.is_empty() {
                cbvs = Some(self.m.md_node(cbv_list));
            }
            let mut srvs = None;
            let raw_srvs: Vec<u32> = self.raw_views.iter().filter(|r| !r.0).map(|r| r.1).collect();
            // (space, register): root UAVs (XNW), then the record views.
            let raw_uavs: Vec<(u32, u32)> = self
                .root_uavs
                .iter()
                .map(|r| (ROOT_CBV_SPACE, u32::from(r.0)))
                .chain(self.raw_views.iter().filter(|r| r.0).map(|r| (super::mesh::RECORD_SPACE, r.1)))
                .collect();
            if !self.root_srvs.is_empty()
                || !raw_srvs.is_empty()
                || self.stride_table.is_some()
                || self.address_map.is_some()
                || self.descriptor_words.is_some()
                || self.page_table.is_some()
            {
                let mut specs: Vec<(u32, u32)> = self
                    .root_srvs
                    .iter()
                    .map(|&(slot, _)| (ROOT_CBV_SPACE, u32::from(slot)))
                    .collect();
                if self.stride_table.is_some() {
                    specs.push((STRIDE_SPACE, 0));
                }
                // VIctory: the address map, then the descriptor words.
                if self.address_map.is_some() {
                    specs.push((ADDRESS_MAP_SPACE, 0));
                }
                if self.descriptor_words.is_some() {
                    specs.push((DESCRIPTOR_WORDS_SPACE, 0));
                }
                if self.page_table.is_some() {
                    specs.push((PAGE_TABLE_SPACE, 0));
                }
                let ty = self
                    .m
                    .named_struct("struct.ByteAddressBuffer", vec![self.t.i32]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let mut list = Vec::new();
                for (n, (space, reg)) in specs.into_iter().enumerate() {
                    let ops = vec![
                        Some(i32c(&mut self, n as u32)),
                        Some(self.m.md_value(u)),
                        Some(empty),
                        Some(i32c(&mut self, space)),
                        Some(i32c(&mut self, reg)),
                        Some(i32c(&mut self, 1)),
                        Some(i32c(&mut self, 11)),
                        Some(i32c(&mut self, 0)),
                        None,
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 4,
                        space,
                        register: reg,
                        dxil_kind: 11,
                    });
                }
                for &reg in &raw_srvs {
                    let ops = vec![
                        Some(i32c(&mut self, list.len() as u32)),
                        Some(self.m.md_value(u)),
                        Some(empty),
                        Some(i32c(&mut self, super::mesh::RECORD_SPACE)),
                        Some(i32c(&mut self, reg)),
                        Some(i32c(&mut self, 1)),
                        Some(i32c(&mut self, 11)),
                        Some(i32c(&mut self, 0)),
                        None,
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 4,
                        space: super::mesh::RECORD_SPACE,
                        register: reg,
                        dxil_kind: 11,
                    });
                }
                srvs = Some(self.m.md_node(list));
            }
            let mut uavs = None;
            if !raw_uavs.is_empty() {
                let ty = self
                    .m
                    .named_struct("struct.RWByteAddressBuffer", vec![self.t.i32]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let no = self.m.bool(false);
                let no = self.m.md_value(no);
                let mut list = Vec::new();
                for (n, &(space, reg)) in raw_uavs.iter().enumerate() {
                    let ops = vec![
                        Some(i32c(&mut self, n as u32)),
                        Some(self.m.md_value(u)),
                        Some(empty),
                        Some(i32c(&mut self, space)),
                        Some(i32c(&mut self, reg)),
                        Some(i32c(&mut self, 1)),
                        Some(i32c(&mut self, 11)),
                        Some(no),
                        Some(no),
                        Some(no),
                        None,
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 7,
                        space,
                        register: reg,
                        dxil_kind: 11,
                    });
                }
                uavs = Some(self.m.md_node(list));
            }
            let mut samplers = None;
            if self.samplers > 0 {
                let ty = self.m.named_struct("struct.SamplerState", vec![self.t.i32]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let mut list = Vec::new();
                for n in 0..self.samplers {
                    let kind = (self.cmp_samplers >> n & 1) as u32;
                    let ops = vec![
                        Some(i32c(&mut self, n)),
                        Some(self.m.md_value(u)),
                        Some(empty),
                        Some(i32c(&mut self, SAMPLER_SPACE)),
                        Some(i32c(&mut self, n)),
                        Some(i32c(&mut self, 1)),
                        // Sampler kind: 1 comparison.
                        Some(i32c(&mut self, kind)),
                        None,
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 1,
                        space: SAMPLER_SPACE,
                        register: n,
                        dxil_kind: 14,
                    });
                }
                samplers = Some(self.m.md_node(list));
            }
            resources = Some(self.m.md_node(vec![srvs, uavs, cbvs, samplers]));
            // PSV0 lists them as the module does: CBVs, samplers, SRVs.
            bindings.sort_by_key(|b| match b.kind {
                2 => 0,
                1 => 1,
                _ => 2,
            });
        }
        if let Some(r) = resources {
            self.m.named_md("dx.resources", vec![r]);
        }

        let graphics = self.threads.is_none() || self.mesh.is_some();
        if graphics {
            // shortcut: every output is declared to depend on every input; the
            // exact sets need a dependence pass. Drivers read this state.
            let scalars = |els: &[SigElement]| {
                els.iter()
                    .filter_map(|e| e.row.map(|r| r * 4 + u32::from(e.col + e.cols)))
                    .max()
                    .unwrap_or(0)
            };
            let (ni, no) = (scalars(&self.inputs), scalars(&self.outputs));
            let mut view = vec![ni, no];
            // Mesh shaders have no inputs: then the primitive outputs' count.
            if self.mesh.is_some() {
                view.push(scalars(&self.prims));
            }
            for _ in 0..ni {
                for k in 0..no.div_ceil(32) {
                    let bits = (no - 32 * k).min(32);
                    view.push(if bits == 32 {
                        u32::MAX
                    } else {
                        (1 << bits) - 1
                    });
                }
            }
            // Geometry shaders: then streams 1-3, each its output count
            // (0 here) and no table.
            if self.geometry {
                view.extend([0, 0, 0]);
            }
            let arr = self.m.array(view.len() as u32, self.t.i32);
            let elems: Vec<V> = view.iter().map(|&v| self.m.i32(v)).collect();
            let arr = self.m.aggregate(arr, &elems);
            let arr = self.m.md_value(arr);
            let view_node = self.m.md_node(vec![Some(arr)]);
            self.m.named_md("dx.viewIdState", vec![view_node]);
        }

        let sigs = if graphics {
            let ins = self.signature_md(self.inputs.clone());
            let outs = self.signature_md(self.outputs.clone());
            let prims = self.signature_md(self.prims.clone());
            Some(self.m.md_node(vec![ins, outs, prims]))
        } else {
            None
        };
        // A pixel shader with an SV_StencilRef output (as DXC collects it).
        if self.pixel && self.outputs.iter().any(|o| o.kind == 20) {
            self.flags |= flags::STENCIL_REF;
        }
        // Entry properties: shader flags (tag 0), thread-group size (4),
        // wave size (11).
        let mut props = Vec::new();
        if self.flags != 0 {
            props.push(i32c(&mut self, 0));
            let i64t = self.m.int(64);
            let f = self.m.int_const(i64t, self.flags as i64);
            props.push(self.m.md_value(f));
        }
        if let Some(threads) = self.threads {
            let t = threads.map(|n| Some(i32c(&mut self, n)));
            let t = self.m.md_node(t.to_vec());
            match self.mesh {
                // Mesh state (tag 9): threads, maximum vertices and
                // primitives, topology, payload bytes.
                Some(mesh) => {
                    props.push(i32c(&mut self, 9));
                    let state = [
                        mesh.max_vertices,
                        mesh.max_primitives,
                        mesh.topology.into(),
                        0,
                    ]
                    .map(|n| Some(i32c(&mut self, n)));
                    let mut node = vec![Some(t)];
                    node.extend(state);
                    props.push(self.m.md_node(node));
                }
                None => {
                    props.push(i32c(&mut self, 4));
                    props.push(t);
                }
            }
        }
        if self.geometry {
            // Geometry state (tag 1): input primitive (triangle), maximum
            // vertices, stream mask, output topology (strip), instances.
            props.push(i32c(&mut self, 1));
            let state = [3, 4, 1, 5, 1].map(|n| Some(i32c(&mut self, n)));
            props.push(self.m.md_node(state.to_vec()));
        }
        if let Some(n) = self.wave_size {
            props.push(i32c(&mut self, 11));
            let n = Some(i32c(&mut self, n));
            props.push(self.m.md_node(vec![n]));
        }
        let props =
            (!props.is_empty()).then(|| self.m.md_node(props.into_iter().map(Some).collect()));
        let main = self.m.md_value(self.main);
        let name = self.m.md_str("main");
        let entry = self
            .m
            .md_node(vec![Some(main), Some(name), sigs, resources, props]);

        let ident = self.m.md_str("victory");
        let ident = self.m.md_node(vec![Some(ident)]);
        self.m.named_md("llvm.ident", vec![ident]);
        let one = i32c(&mut self, 1);
        let six = i32c(&mut self, 6);
        let version = self.m.md_node(vec![Some(one), Some(six)]);
        self.m.named_md("dx.version", vec![version]);
        self.m.named_md("dx.valver", vec![version]);
        // Program kinds: 0 pixel, 1 vertex, 5 compute, 13 mesh.
        let (name, kind) = match (self.threads, self.pixel) {
            _ if self.mesh.is_some() => ("ms", 13u32),
            _ if self.geometry => ("gs", 2u32),
            (Some(_), _) => ("cs", 5u32),
            (None, true) => ("ps", 0),
            (None, false) => ("vs", 1),
        };
        let name = self.m.md_str(name);
        let sm = self.m.md_node(vec![Some(name), Some(six), Some(six)]);
        self.m.named_md("dx.shaderModel", vec![sm]);
        self.m.named_md("dx.entryPoints", vec![entry]);

        let t = std::time::Instant::now();
        let bitcode = self.m.write();
        super::lap("write", t);
        let mut dxil = Vec::with_capacity(24 + bitcode.len());
        dxil.extend((kind << 16 | 0x66).to_le_bytes());
        dxil.extend((((24 + bitcode.len()) / 4) as u32).to_le_bytes());
        dxil.extend(b"DXIL");
        dxil.extend(0x106u32.to_le_bytes());
        dxil.extend(16u32.to_le_bytes());
        dxil.extend((bitcode.len() as u32).to_le_bytes());
        dxil.extend(&bitcode);

        let mut features = 0u64;
        if self.flags & flags::STENCIL_REF != 0 {
            features |= 0x200;
        }
        if self.flags & flags::DOUBLES != 0 {
            features |= 0x1;
        }
        if self.flags & flags::DOUBLE_EXTENSIONS != 0 {
            features |= 0x20;
        }
        if self.flags & flags::TYPED_UAV_LOADS != 0 {
            features |= 0x800;
        }
        if self.flags & flags::WAVE_OPS != 0 {
            features |= 0x4000;
        }
        if self.flags & flags::INT64 != 0 {
            features |= 0x8000;
        }
        if self.flags & flags::RESOURCE_HEAP != 0 {
            features |= 0x200_0000;
        }
        if self.flags & flags::SAMPLER_HEAP != 0 {
            features |= 0x400_0000;
        }
        if self.flags & flags::ATOMIC_INT64_TYPED != 0 {
            features |= 0x40_0000;
        }
        if self.flags & flags::ATOMIC_INT64_HEAP != 0 {
            features |= 0x1000_0000;
        }
        let sfi0 = features.to_le_bytes();
        let isg1 = container::signature(&self.inputs, true);
        let osg1 = container::signature(&self.outputs, false);
        let flag = if self.pixel {
            self.outputs.iter().any(|o| (17..=19).contains(&o.kind))
        } else {
            self.outputs.iter().any(|o| o.kind == 3)
        };
        let psv0 = container::psv(&Psv {
            stage: kind as u8,
            flag: flag && self.threads.is_none(),
            inputs: &self.inputs,
            outputs: &self.outputs,
            prims: &self.prims,
            mesh: self.mesh.map(|m| (m, self.lds_dwords * 4)),
            geometry: self.geometry,
            bindings: &bindings,
            threads: self.threads.unwrap_or_default(),
            wave_size: self.wave_size,
        });
        let hash = [0u8; 20];
        let psg1 = container::signature(&self.prims, false);
        let mut parts: Vec<(&[u8; 4], &[u8])> =
            vec![(b"SFI0", &sfi0), (b"ISG1", &isg1), (b"OSG1", &osg1)];
        if self.mesh.is_some() {
            parts.push((b"PSG1", &psg1));
        }
        parts.extend([(b"PSV0", &psv0[..]), (b"HASH", &hash), (b"DXIL", &dxil)]);
        let private: Vec<u8> = self.host_copies.map(|b| [*b"XNWH", b.to_le_bytes()].concat()).unwrap_or_default();
        if self.host_copies.is_some() {
            parts.push((b"PRIV", &private));
        }
        container::build(&parts)
    }

    /// The user-data buffer's annotation carries its size, known only now.
    fn fix_user_data_size(&mut self, size: u32) {
        let old = [self.m.i32(13), self.m.i32(u32::MAX)];
        let old = self.m.aggregate(self.t.props, &old);
        let new = [self.m.i32(13), self.m.i32(size)];
        let new = self.m.aggregate(self.t.props, &new);
        self.m.replace_const(old, new);
    }

    fn signature_md(&mut self, elements: Vec<SigElement>) -> Option<Md> {
        if elements.is_empty() {
            return None;
        }
        let mut list = Vec::new();
        for (id, e) in elements.iter().enumerate() {
            let c32 = |s: &mut Self, v: u32| {
                let c = s.m.i32(v);
                Some(s.m.md_value(c))
            };
            let c8 = |s: &mut Self, v: u8| {
                let c = s.m.i8(v);
                Some(s.m.md_value(c))
            };
            let idx = c32(self, e.index);
            let indices = Some(self.m.md_node(vec![idx]));
            let usage = if e.usage != 0 {
                let (tag, mask) = (c32(self, 3), c32(self, u32::from(e.usage)));
                Some(self.m.md_node(vec![tag, mask]))
            } else {
                None
            };
            let ops = vec![
                c32(self, id as u32),
                Some(self.m.md_str(&e.name)),
                c8(self, e.comp),
                c8(self, e.kind),
                indices,
                c8(self, e.interp),
                c32(self, 1),
                c8(self, e.cols),
                c32(self, e.row.unwrap_or(u32::MAX)),
                c8(self, if e.row.is_some() { e.col } else { u8::MAX }),
                usage,
            ];
            list.push(Some(self.m.md_node(ops)));
        }
        Some(self.m.md_node(list))
    }
}
