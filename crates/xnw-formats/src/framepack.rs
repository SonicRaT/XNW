//! Frame pack v2: the files a `pack=2` capture
//! writes under `frame-NNNN\pack\` and their reader. Pure: no title, no
//! hook, no hashing (the writer names blobs by `xnw_core::hash128`, which
//! this module takes as given).
//!
//! Files: `manifest.tsv` (key, value lines), `refs.bin` (40-byte records),
//! `sinks.tsv`, `programs.tsv`, `textures.tsv`; raw bytes live in the
//! capture's shared `blobs\r-<32 hex>`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

pub const FORMAT: &str = "xnw-pack";
pub const VERSION: u32 = 2;
pub const REC_BYTES: usize = 40;
/// `Ref::owner` of a frame-scope object (the preamble).
pub const FRAME_SCOPE: u32 = u32::MAX;
/// `Ref::owner` bit of a hook-side object: the low bits are the v1 draw.
pub const HOOK: u32 = 0x8000_0000;

/// Record kinds (PACK.md section 3). Unknown kinds are kept, not refused.
pub mod kind {
    pub const ROOTS: u16 = 1;
    pub const TABLE: u16 = 2;
    pub const CBV: u16 = 3;
    pub const ROOT_VIEW: u16 = 4;
    pub const HEAP_SLOT: u16 = 5;
    /// v2.4, owner = a sink: the bound sampler heap's entries
    /// (`SamplerEntry` each; addr = the heap's first slot).
    pub const SAMPLERS: u16 = 6;
    /// Bytes behind a buffer view in a table (slot = root * 96 + slot).
    pub const BUFFER: u16 = 7;
    pub const T8: u16 = 16;
    pub const FRAME_CB: u16 = 17;
    pub const HEAPS: u16 = 18;
    pub const STATES: u16 = 19;
    pub const SHADER: u16 = 32;
    pub const MAT_CB: u16 = 33;
    pub const TEX_VARS: u16 = 34;
    pub const PER_BATCH: u16 = 35;
    pub const WORLD: u16 = 36;
    pub const VIEW_B0: u16 = 37;
    /// Skinned draw's palette at the hook (slot 0 current, 1 previous).
    pub const PALETTE: u16 = 38;
    /// The draw's model head (0x40 B).
    pub const MODEL: u16 = 39;
    /// v2.1, frame scope: a descriptor handle a hook draw named (albedo,
    /// layers), resolved at the hook: addr = the handle, blob = one SlotView.
    pub const HANDLE_VIEW: u16 = 40;
    /// v2.1, hook side: the skinned draw's handler head (entity+0x28, 0x400 B).
    pub const HANDLER: u16 = 41;
    /// v2.1, frame scope: one imposter submit 0x1451D0AC0 call (slot = call).
    pub const IMPOSTER: u16 = 42;
    /// v2.1: arena words at a slot-0 record's +0x74 (slot as the record's);
    /// frame scope: the arena base (addr, no bytes).
    pub const ARENA: u16 = 43;
    /// Owner = a dispatch sink: the dispatch itself (`Dispatch`,
    /// 272 B; addr = the title's 144-B item, 0 when not found).
    pub const DISPATCH: u16 = 44;
    /// Dispatch sink: the view at a heap index the program's constant
    /// buffer names (slot k: b7+0x40 + 4k); addr = the handle, blob = one
    /// SlotView whose slot is the heap index.
    pub const HEAP_VIEW: u16 = 45;
    /// Dispatch sink: the bytes of HEAP_VIEW k's range.
    pub const HEAP_BYTES: u16 = 46;
    /// v2.3, frame scope: the instances of IMPOSTER call `slot` (same slot):
    /// `ImpInstances`; addr = the instance list L = *(desc+0).
    pub const IMP_INST: u16 = 47;
    /// v2.3, draw sink of an instanced wheel program: 256 x 48 B of its
    /// palette (root 0 slot 5) at one per-instance base (slot = k-th
    /// distinct base; addr = view addr + 48 * base).
    pub const PAL_WINDOW: u16 = 48;
    /// v2.4, draw sink: slot 0 = `DRAW_WORDS_BYTES` at root 0 slot 86's
    /// view address + root 4 constant dword 1 (addr = that address); slot 1
    /// = those words resolved as view heap indices, SlotViews (slot = heap
    /// index; addr = the heap's first slot).
    pub const DRAW_WORDS: u16 = 49;
    /// v2.4, dispatch sink of a displacing pre-skin program: the view at the
    /// heap index of element `slot` of root 0 slot 8 (144-B elements, the
    /// index at +0x8C); addr = the handle (the index when it did not
    /// resolve), blob = one SlotView whose slot is the heap index.
    pub const DISP_MAP: u16 = 50;
    /// v2.6, sink of a Draw or DrawIndexed (op 4, 3): `DRAW_ARGS_BYTES`,
    /// four u32: instances, first vertex (or index), base vertex (indexed;
    /// as i32), first instance. Strand hair: strands per draw = instances.
    pub const DRAW_ARGS: u16 = 51;
    /// v2.6, dispatch sink of a strand-hair input program (`smh_pose_*`,
    /// `smh_gen_*`): the view at the heap index in word k (`STRAND_WORDS`)
    /// of strand group record r of root 0 slot 21 (`STRAND_RECORD` bytes
    /// each); slot = r * 8 + k; addr = the handle (the index when it did
    /// not resolve), blob = one SlotView whose slot is the heap index.
    /// Once per frame per record buffer content (the first sink that names it).
    pub const STRAND_VIEW: u16 = 52;
    /// v2.6: the bytes of STRAND_VIEW `slot`'s range (buffers <= 16 MB,
    /// constant buffers <= 64 KB; an unordered view: reason 10, address only).
    pub const STRAND_BYTES: u16 = 53;
    /// v2.7, frame scope: a pipeline's targets once per frame (addr = the
    /// pipeline): `TARGETS_BYTES`, u32 count, 8 render-target DXGI formats,
    /// the depth format.
    pub const TARGETS: u16 = 54;
    /// v2.7, frame scope: the CPU scene-light array at the frame's first
    /// lighting dispatch (slot 0: `LIGHT_ARRAY_BYTES` at addr; slot 1: 8 B,
    /// the buffer index and the light count, addr = the count's address).
    pub const LIGHTS: u16 = 55;
    /// v2.8, hook side (owner HOOK | draw), `hookin=1`: the hook's inputs
    /// for a replay of the hook: `HookIn` parts (see `HOOKIN_PARTS`).
    pub const HOOKIN: u16 = 56;
    /// v2.8, frame scope: the hook's globals, 10 B: u16 matrix slot word,
    /// u64 palette base (addr = the slot word's address).
    pub const HOOK_GLOBALS: u16 = 57;
    /// v2.9, strand-hair input sink: per group record r the palette bones
    /// it names (root 0 slot 1, 48 B a bone): slot 2r = from base rec+0xe8,
    /// 2r+1 = from base rec+0xec (the previous frame's); bones = the
    /// largest remap entry (rec+0xe4 .. +256, & 0xfff) + 1; addr = the
    /// palette view's address + 48 * base. CUT at the view's end.
    pub const STRAND_PALETTE: u16 = 58;
    /// v2.9, strand-hair input sink: per group record r the collision
    /// shapes `CS_UpdatePosePosition` loops over in root 0 slot 22: byte
    /// offset rec+0xf8, count rec+0xa8 >> 24, `STRAND_SHAPE` bytes each;
    /// slot r, addr = slot 22's view address + offset. 1 = count 0.
    pub const STRAND_SPAN: u16 = 59;
    /// v2.9, any dispatch sink: 12 B, the x, y, z group counts.
    pub const GROUPS: u16 = 60;

    pub fn name(k: u16) -> &'static str {
        match k {
            ROOTS => "roots",
            TABLE => "table",
            CBV => "cbv",
            ROOT_VIEW => "rootview",
            HEAP_SLOT => "heapslot",
            SAMPLERS => "samplers",
            BUFFER => "buffer",
            T8 => "t8",
            FRAME_CB => "framecb",
            HEAPS => "heaps",
            STATES => "states",
            SHADER => "shader",
            MAT_CB => "matcb",
            TEX_VARS => "texvars",
            PER_BATCH => "perbatch",
            WORLD => "world",
            VIEW_B0 => "b0",
            PALETTE => "palette",
            MODEL => "model",
            HANDLE_VIEW => "handleview",
            HANDLER => "handler",
            IMPOSTER => "imposter",
            ARENA => "arena",
            DISPATCH => "dispatch",
            HEAP_VIEW => "heapview",
            HEAP_BYTES => "heapbytes",
            IMP_INST => "impinst",
            PAL_WINDOW => "palwin",
            DRAW_WORDS => "drawwords",
            DISP_MAP => "dispmap",
            DRAW_ARGS => "drawargs",
            STRAND_VIEW => "strandview",
            STRAND_BYTES => "strandbytes",
            TARGETS => "targets",
            LIGHTS => "lights",
            HOOKIN => "hookin",
            HOOK_GLOBALS => "hookglobals",
            STRAND_PALETTE => "strandpal",
            STRAND_SPAN => "strandspan",
            GROUPS => "groups",
            _ => "other",
        }
    }
}

/// Why a record holds no bytes (PACK.md section 5).
pub mod reason {
    pub const OK: u32 = 0;
    pub const NULL: u32 = 1;
    pub const UNREADABLE: u32 = 2;
    pub const IMPLAUSIBLE: u32 = 3;
    pub const BUDGET: u32 = 4;
    pub const NOT_RESIDENT: u32 = 5;
    pub const NO_HOOK_DRAW: u32 = 6;
    pub const ZERO_FILLED: u32 = 7;
    pub const NOT_CAPTURED: u32 = 8;
    /// The object is longer than its cap: the blob holds its first `len`
    /// bytes (the hash is valid).
    pub const CUT: u32 = 9;
    /// A view the GPU writes (unordered access) at a dispatch sink:
    /// its content does not exist on the CPU; not read.
    pub const GPU_OUTPUT: u32 = 10;
    /// Bytes present (whole or cut).
    pub fn has_bytes(r: u32) -> bool {
        r == OK || r == CUT
    }
}

/// One object a draw (or the frame) references.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ref {
    pub owner: u32,
    pub kind: u16,
    pub slot: u16,
    pub addr: u64,
    pub len: u32,
    pub reason: u32,
    pub hash: u128,
}

impl Ref {
    pub fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.owner.to_le_bytes());
        out.extend_from_slice(&self.kind.to_le_bytes());
        out.extend_from_slice(&self.slot.to_le_bytes());
        out.extend_from_slice(&self.addr.to_le_bytes());
        out.extend_from_slice(&self.len.to_le_bytes());
        out.extend_from_slice(&self.reason.to_le_bytes());
        out.extend_from_slice(&self.hash.to_le_bytes());
    }

    pub fn decode(b: &[u8]) -> Option<Self> {
        let b: &[u8; REC_BYTES] = b.get(..REC_BYTES)?.try_into().ok()?;
        let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        Some(Ref {
            owner: u32_at(0),
            kind: u16::from_le_bytes([b[4], b[5]]),
            slot: u16::from_le_bytes([b[6], b[7]]),
            addr: u64::from_le_bytes(b[8..16].try_into().ok()?),
            len: u32_at(16),
            reason: u32_at(20),
            hash: u128::from_le_bytes(b[24..40].try_into().ok()?),
        })
    }

    /// Frame scope first, then by (owner, kind, slot).
    fn order(&self) -> (bool, u32, u16, u16) {
        (self.owner != FRAME_SCOPE, self.owner, self.kind, self.slot)
    }
}

/// refs.bin: the records sorted (frame scope first).
pub fn encode_refs(refs: &mut [Ref]) -> Vec<u8> {
    refs.sort_by_key(Ref::order);
    let mut out = Vec::with_capacity(refs.len() * REC_BYTES);
    for r in refs.iter() {
        r.encode(&mut out);
    }
    out
}

pub fn decode_refs(b: &[u8]) -> Result<Vec<Ref>, String> {
    if b.len() % REC_BYTES != 0 {
        return Err(format!("refs.bin: {} bytes, not a multiple of {REC_BYTES}", b.len()));
    }
    Ok(b.chunks_exact(REC_BYTES).filter_map(Ref::decode).collect())
}

/// A raw object's file name in the capture's shared `blobs\`.
pub fn blob_name(hash: u128) -> String {
    format!("r-{hash:032x}")
}

/// What a descriptor slot holds (a resolved view; `SlotView::kind`).
pub mod view {
    pub const NONE: u32 = 0;
    pub const CONSTANT: u32 = 1;
    pub const TEXTURE: u32 = 2;
    pub const BUFFER: u32 = 3;
    pub const PLACED: u32 = 4;
    pub const PLACED_BUFFER: u32 = 5;
    pub const SAMPLER: u32 = 6;
    pub const OTHER: u32 = 7;
    /// `SlotView::layout` of a BUFFER view that is an unordered access view.
    pub const UNORDERED: u32 = 1;
}

pub const SLOT_BYTES: usize = 176;

/// One resolved descriptor slot of a table (kinds `TABLE`, `HEAP_SLOT`):
/// a table's blob is its non-empty slots in order, `SLOT_BYTES` each.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SlotView {
    pub slot: u32,
    pub kind: u32,
    pub addr: u64,
    pub bytes: u64,
    pub resource: u64,
    pub width: u64,
    pub height: u32,
    pub depth: u16,
    pub mips: u16,
    pub format: u32,
    pub layout: u32,
    pub words: [u32; 30],
}

impl SlotView {
    pub fn encode(&self, out: &mut Vec<u8>) {
        for v in [self.slot, self.kind] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in [self.addr, self.bytes, self.resource, self.width] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.height.to_le_bytes());
        out.extend_from_slice(&self.depth.to_le_bytes());
        out.extend_from_slice(&self.mips.to_le_bytes());
        out.extend_from_slice(&self.format.to_le_bytes());
        out.extend_from_slice(&self.layout.to_le_bytes());
        for w in self.words {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }

    pub fn decode(b: &[u8]) -> Option<SlotView> {
        let b = b.get(..SLOT_BYTES)?;
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap_or_default());
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap_or_default());
        let u16_at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
        Some(SlotView {
            slot: u32_at(0),
            kind: u32_at(4),
            addr: u64_at(8),
            bytes: u64_at(16),
            resource: u64_at(24),
            width: u64_at(32),
            height: u32_at(40),
            depth: u16_at(44),
            mips: u16_at(46),
            format: u32_at(48),
            layout: u32_at(52),
            words: std::array::from_fn(|i| u32_at(56 + 4 * i)),
        })
    }
}

pub fn encode_slots(slots: &[SlotView]) -> Vec<u8> {
    let mut out = Vec::with_capacity(slots.len() * SLOT_BYTES);
    for s in slots {
        s.encode(&mut out);
    }
    out
}

pub fn decode_slots(b: &[u8]) -> Vec<SlotView> {
    b.chunks_exact(SLOT_BYTES).filter_map(SlotView::decode).collect()
}

/// How a sink draw was issued.
pub mod op {
    pub const MESH: u8 = 1;
    pub const MESH_X: u8 = 2;
    pub const INDEXED: u8 = 3;
    pub const DRAW: u8 = 4;
    /// ExecuteIndirect of mesh dispatches (count = max commands).
    pub const INDIRECT_MESH: u8 = 5;
    /// ExecuteIndirect of other draws or dispatches.
    pub const INDIRECT: u8 = 6;
    /// A direct compute Dispatch (count = x groups; y, z in its
    /// DISPATCH record). Not a draw: never paired with a hook draw.
    pub const DISPATCH: u8 = 7;
    /// v2.9: ExecuteIndirect of compute dispatches (count = max commands),
    /// with the compute roots (before: op INDIRECT with the graphics roots).
    /// Not a draw: never paired with a hook draw.
    pub const INDIRECT_DISPATCH: u8 = 8;
    /// A compute dispatch, direct or indirect (no draw).
    pub fn is_dispatch(o: u8) -> bool {
        o == DISPATCH || o == INDIRECT_DISPATCH
    }
    pub fn name(o: u8) -> &'static str {
        match o {
            MESH => "mesh",
            MESH_X => "mesh_x",
            INDEXED => "indexed",
            DRAW => "draw",
            INDIRECT_MESH => "indirect_mesh",
            INDIRECT => "indirect",
            DISPATCH => "dispatch",
            INDIRECT_DISPATCH => "indirect_dispatch",
            _ => "?",
        }
    }
    pub fn parse(s: &str) -> u8 {
        [MESH, MESH_X, INDEXED, DRAW, INDIRECT_MESH, INDIRECT, DISPATCH, INDIRECT_DISPATCH].into_iter().find(|&o| name(o) == s).unwrap_or(0)
    }
}

/// The DISPATCH record of a dispatch sink (kind 44, `DISPATCH_BYTES`).
/// The ped pre-skin pass (title 0x144E3CDD0) dispatches one 144-B item per
/// ped geometry: its constant buffer b7 is item+32..+127, and the range
/// word (b7+0x34 = item+136) is what the PreSkinD draw's record holds at
/// +0x54.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Dispatch {
    pub groups: [u32; 3],
    /// `B7_FOUND`, `ITEM_FOUND`.
    pub flags: u32,
    /// Where b7 was bound: root * 96 + slot.
    pub b7_slot: u32,
    /// b7+0x34: the output range (offset >> 4, bit 24 the buffer, format bits above).
    pub word: u32,
    /// b7+0x38.
    pub vertices: u32,
    /// 96 B (zeros when not found).
    pub b7: Vec<u8>,
    /// 144 B (zeros when not found).
    pub item: Vec<u8>,
}

pub const DISPATCH_BYTES: usize = 32 + B7_BYTES + ITEM_BYTES;
pub const B7_BYTES: usize = 96;
pub const ITEM_BYTES: usize = 144;
pub const B7_FOUND: u32 = 1;
pub const ITEM_FOUND: u32 = 2;
/// The bits of a range word that name the output range: offset >> 4 (0-23)
/// and the buffer (24-25); the bits above are the vertex format.
pub const RANGE_MASK: u32 = 0x03FF_FFFF;

impl Dispatch {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(DISPATCH_BYTES);
        for w in [self.groups[0], self.groups[1], self.groups[2], self.flags, self.b7_slot, self.word, self.vertices, 0] {
            out.extend_from_slice(&w.to_le_bytes());
        }
        for (part, len) in [(&self.b7, B7_BYTES), (&self.item, ITEM_BYTES)] {
            let at = out.len();
            out.extend_from_slice(&part[..part.len().min(len)]);
            out.resize(at + len, 0);
        }
        out
    }

    pub fn decode(b: &[u8]) -> Option<Dispatch> {
        let b = b.get(..DISPATCH_BYTES)?;
        let w = |i: usize| u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        Some(Dispatch { groups: [w(0), w(1), w(2)], flags: w(3), b7_slot: w(4), word: w(5), vertices: w(6), b7: b[32..32 + B7_BYTES].to_vec(), item: b[32 + B7_BYTES..].to_vec() })
    }
}

/// The IMP_INST record of one imposter submit call (kind 47): the batch's
/// instance list L = *(desc+0) (0xA0 B; records at L+0x90, u16 count at
/// L+0x98, 64 B each, as for GPU-buffer LOD batches), the visible list at
/// IcRec+0x48 (u32 x IcRec+0x50), the indirect arguments behind
/// IcRec+0x58, the batch handler's head (r9 - 0x308; its +0x100 is the
/// imposter type id) and the type record R the title's lookup 0x145248630
/// finds for the call's type (224 B; size floats R+0x40..0x74, atlas
/// R+0x8C). Each part has a reason (`reason::OK` = present); a part that
/// failed is zeros (head, args, handler, type) or empty (visible, records).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImpInstances {
    /// L, L+0x90, IcRec+0x48, IcRec+0x58, the handler, R.
    pub list: u64,
    pub records_at: u64,
    pub visible_at: u64,
    pub args_at: u64,
    pub handler_at: u64,
    pub type_at: u64,
    /// u16 L+0x98.
    pub count: u32,
    /// The batch's instance count (u32 desc+8).
    pub n: u32,
    /// The visible list's length (u32 IcRec+0x50).
    pub visible_n: u32,
    /// Reasons by part: `IMP_HEAD`, `IMP_ARGS`, `IMP_VISIBLE`,
    /// `IMP_RECORDS`, `IMP_HANDLER`, `IMP_TYPE`.
    pub reasons: [u32; 6],
    /// L's 0xA0 bytes.
    pub head: Vec<u8>,
    /// `IMP_ARGS_BYTES` behind IcRec+0x58.
    pub args: Vec<u8>,
    /// `IMP_HANDLER_BYTES` of the handler.
    pub handler: Vec<u8>,
    /// `IMP_TYPE_BYTES` of R.
    pub type_record: Vec<u8>,
    /// u32 x min(visible_n, `IMP_MAX`) behind IcRec+0x48.
    pub visible: Vec<u32>,
    /// 64 B x min(count, `IMP_MAX`).
    pub records: Vec<u8>,
}

pub const IMP_HEAD_BYTES: usize = 0xA0;
/// DrawIndexedIndirect arguments: 5 u32.
pub const IMP_ARGS_BYTES: usize = 20;
pub const IMP_HANDLER_BYTES: usize = 0x400;
pub const IMP_TYPE_BYTES: usize = 0xE0;
pub const IMP_RECORD_BYTES: usize = 64;
/// Instances (records, visible words) kept per call.
pub const IMP_MAX: usize = 4096;
/// Indices of `ImpInstances::reasons`.
pub const IMP_HEAD: usize = 0;
pub const IMP_ARGS: usize = 1;
pub const IMP_VISIBLE: usize = 2;
pub const IMP_RECORDS: usize = 3;
pub const IMP_HANDLER: usize = 4;
pub const IMP_TYPE: usize = 5;
/// Header: 6 u64, then 11 u32 and one spare.
const IMP_WORDS: usize = 48;
const IMP_HEADER: usize = 96;
const IMP_FIXED: usize = IMP_HEADER + IMP_HEAD_BYTES + IMP_ARGS_BYTES + IMP_HANDLER_BYTES + IMP_TYPE_BYTES;

impl ImpInstances {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(IMP_FIXED + 4 * self.visible.len() + self.records.len());
        for q in [self.list, self.records_at, self.visible_at, self.args_at, self.handler_at, self.type_at] {
            out.extend_from_slice(&q.to_le_bytes());
        }
        let records = (self.records.len() / IMP_RECORD_BYTES).min(IMP_MAX);
        let visible = self.visible.len().min(IMP_MAX);
        for w in [self.count, self.n, self.visible_n, visible as u32, records as u32].into_iter().chain(self.reasons).chain([0]) {
            out.extend_from_slice(&w.to_le_bytes());
        }
        for (part, len) in [(&self.head, IMP_HEAD_BYTES), (&self.args, IMP_ARGS_BYTES), (&self.handler, IMP_HANDLER_BYTES), (&self.type_record, IMP_TYPE_BYTES)] {
            let at = out.len();
            out.extend_from_slice(&part[..part.len().min(len)]);
            out.resize(at + len, 0);
        }
        for w in &self.visible[..visible] {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.extend_from_slice(&self.records[..records * IMP_RECORD_BYTES]);
        out
    }

    /// None when `b` is shorter or longer than its header says.
    pub fn decode(b: &[u8]) -> Option<ImpInstances> {
        let fixed = b.get(..IMP_FIXED)?;
        let q = |i: usize| u64::from_le_bytes(fixed[8 * i..8 * i + 8].try_into().unwrap_or([0; 8]));
        let w = |i: usize| u32::from_le_bytes(fixed[IMP_WORDS + 4 * i..IMP_WORDS + 4 * i + 4].try_into().unwrap_or([0; 4]));
        let (visible, records) = (w(3) as usize, w(4) as usize);
        if visible > IMP_MAX || records > IMP_MAX || b.len() != IMP_FIXED + 4 * visible + IMP_RECORD_BYTES * records {
            return None;
        }
        let part = |at: usize, len: usize| fixed[IMP_HEADER + at..IMP_HEADER + at + len].to_vec();
        let rest = &b[IMP_FIXED..];
        Some(ImpInstances {
            list: q(0),
            records_at: q(1),
            visible_at: q(2),
            args_at: q(3),
            handler_at: q(4),
            type_at: q(5),
            count: w(0),
            n: w(1),
            visible_n: w(2),
            reasons: [w(5), w(6), w(7), w(8), w(9), w(10)],
            head: part(0, IMP_HEAD_BYTES),
            args: part(IMP_HEAD_BYTES, IMP_ARGS_BYTES),
            handler: part(IMP_HEAD_BYTES + IMP_ARGS_BYTES, IMP_HANDLER_BYTES),
            type_record: part(IMP_HEAD_BYTES + IMP_ARGS_BYTES + IMP_HANDLER_BYTES, IMP_TYPE_BYTES),
            visible: rest[..4 * visible].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            records: rest[4 * visible..].to_vec(),
        })
    }

    /// Instance `i`'s 64-B record.
    pub fn record(&self, i: usize) -> Option<&[u8]> {
        self.records.get(IMP_RECORD_BYTES * i..IMP_RECORD_BYTES * (i + 1))
    }
}

/// The bytes a DRAW_WORDS slot 0 record holds (16 u32).
pub const DRAW_WORDS_BYTES: usize = 64;
/// Table slots per root in a ref's slot (`root * TABLE_ROW + slot`);
/// v2.7: slots TABLE_ROW..2*TABLE_ROW of a root (the frame table's upper
/// half) are `UPPER | (root * TABLE_ROW + slot - TABLE_ROW)`.
pub const TABLE_ROW: u32 = 96;
pub const UPPER: u16 = 0x8000;

/// The ref slot of table root `root` slot `slot` (kinds CBV, BUFFER).
pub fn table_slot(root: u32, slot: u32) -> u16 {
    if slot < TABLE_ROW { (root * TABLE_ROW + slot) as u16 } else { UPPER | (root * TABLE_ROW + slot - TABLE_ROW) as u16 }
}

/// The (root, slot) of a ref slot made by [`table_slot`].
pub fn table_slot_of(at: u16) -> (u32, u32) {
    let v = u32::from(at & !UPPER);
    (v / TABLE_ROW, v % TABLE_ROW + if at & UPPER != 0 { TABLE_ROW } else { 0 })
}

/// v2.8 HOOKIN blob: u32 part count n, then n x (u64 address, u32 length,
/// u32 1 if read), then the parts' bytes in order (zeros where not read).
/// Parts (k: what, bytes): 0 TLS+0x950 stage 4; 1 TLS+0x1110 4; 2 TLS+0x1294
/// 1; 3 TLS+0x11D8 0xF0 (drawable .. technique); 4 the geometry (rcx) 0x48;
/// 5 *(g+8) vb desc 0x20; 6 *(g+0x10) ib desc 0x20; 7 **(g+0x40) meshlet
/// desc 0x20; 8 ctx+0x40+8*slot 8 (bound); 9 ctx+0x6200+2*slot 2 (lazy);
/// 10 ctx+0x12190 4 (matrix index i); 11 ctx+0x12328 8; 12 ctx+0x12360
/// 0x1A; 13 bound 0x80; 14 matrix cache + 128*i 0x80; 15 viewport 0xC0;
/// 16 viewport+0x708 8; 17 model 0x38; 18 palette + 48 * model[0x33] 48.
/// ctx = TLS+0x12A8, viewport = TLS+0x12B8, model = TLS+0x1208, palette =
/// TLS+0x11E8, slot = (matrix slot word) >> 2.
pub const HOOKIN_PARTS: usize = 19;

/// One HOOKIN part: (address, length, read, bytes).
pub fn decode_hookin(b: &[u8]) -> Option<Vec<(u64, u32, bool, Vec<u8>)>> {
    let n = u32::from_le_bytes(b.get(..4)?.try_into().ok()?) as usize;
    let mut at = 4 + 16 * n;
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let h = b.get(4 + 16 * k..20 + 16 * k)?;
        let addr = u64::from_le_bytes(h[..8].try_into().ok()?);
        let len = u32::from_le_bytes(h[8..12].try_into().ok()?);
        let ok = u32::from_le_bytes(h[12..16].try_into().ok()?) != 0;
        out.push((addr, len, ok, b.get(at..at + len as usize)?.to_vec()));
        at += len as usize;
    }
    Some(out)
}

/// v2.7 TARGETS blob size.
pub const TARGETS_BYTES: usize = 40;
/// v2.7 the scene-light array's bytes per buffer (0x1487BAF40 + 374784 *
/// index [V ]).
pub const LIGHT_ARRAY_BYTES: usize = 374_784;

/// v2.6 `DRAW_ARGS`: instances, first vertex or index, base vertex, first instance.
pub const DRAW_ARGS_BYTES: usize = 16;
/// v2.6 strand group record (root 0 slot 21 of the `smh_*` programs).
pub const STRAND_RECORD: usize = 336;
/// v2.9: one collision shape of root 0 slot 22 (`CS_UpdatePosePosition`
/// steps 0x50 per shape, bc6245e7 0x97c-0x9a0) [V read].
pub const STRAND_SHAPE: usize = 0x50;
/// v2.9: the bytes of one palette bone (3x4 f32).
pub const BONE_BYTES: usize = 48;
/// v2.6: the record words holding view heap indices, in STRAND_VIEW slot
/// order k = 0..: +0xe0 (bone remap), +0x120 (asset buffer) [V the programs read],
/// +0x114, +0x11c, +0x124, +0x128, +0x12c [I].
pub const STRAND_WORDS: [usize; 7] = [0xe0, 0x120, 0x114, 0x11c, 0x124, 0x128, 0x12c];

/// One slot of a sampler heap (kind `SAMPLERS`), `SAMPLER_BYTES` encoded:
/// u32 index, u32 form, u32 x4 raw, u32 x27 desc.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SamplerEntry {
    /// Slot index in the heap.
    pub index: u32,
    /// `SAMPLER_D3D12` (desc holds a D3D12 sampler description, 13 words),
    /// `SAMPLER_CONSOLE` (the console description, 27 words) or
    /// `SAMPLER_RAW` (the raw words are not a record of ours).
    pub form: u32,
    /// The 16 bytes at the heap's first slot + 16 * index.
    pub raw: [u32; 4],
    pub desc: [u32; 27],
}

pub const SAMPLER_BYTES: usize = 132;
pub const SAMPLER_RAW: u32 = 0;
pub const SAMPLER_D3D12: u32 = 13;
pub const SAMPLER_CONSOLE: u32 = 27;
/// Sampler heap slots looked at per heap.
pub const SAMPLER_MAX: u32 = 4096;

impl SamplerEntry {
    pub fn encode(&self, out: &mut Vec<u8>) {
        for w in [self.index, self.form].into_iter().chain(self.raw).chain(self.desc) {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }

    pub fn decode(b: &[u8]) -> Option<SamplerEntry> {
        let b = b.get(..SAMPLER_BYTES)?;
        let w = |i: usize| u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        Some(SamplerEntry { index: w(0), form: w(1), raw: std::array::from_fn(|i| w(2 + i)), desc: std::array::from_fn(|i| w(6 + i)) })
    }
}

pub fn encode_samplers(entries: &[SamplerEntry]) -> Vec<u8> {
    let mut out = Vec::with_capacity(entries.len() * SAMPLER_BYTES);
    for e in entries {
        e.encode(&mut out);
    }
    out
}

/// None when the length is not a whole number of entries.
pub fn decode_samplers(b: &[u8]) -> Option<Vec<SamplerEntry>> {
    (b.len() % SAMPLER_BYTES == 0).then(|| b.chunks_exact(SAMPLER_BYTES).filter_map(SamplerEntry::decode).collect())
}

/// One draw at the list sink, tagged with the hook draw on its thread.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Sink {
    pub index: u32,
    pub thread: u32,
    pub hook_draw: Option<u32>,
    pub op: u8,
    pub count: u32,
    pub pipeline: u64,
}

/// One stage of a pipeline the frame's sinks bound.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Program {
    pub pipeline: u64,
    pub stage: String,
    /// 1 container, 2 cache code (the xnw-gpu `Form`), 0 unknown.
    pub form: u8,
    /// hash128 of the bytes the title passed (PackIndex::by_hash key).
    pub container: u128,
    pub size: u32,
    pub prefix: [u8; 24],
    pub reason: u32,
}

/// A fill fingerprint: permille of zero bytes over spread windows and a
/// 64-bit FNV-1a of those windows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fill {
    pub zero_permille: u16,
    pub hash: u64,
}

pub const FILL_WINDOWS: usize = 16;
pub const FILL_WINDOW: usize = 256;

impl Fill {
    /// The windows a fingerprint reads of a `len`-byte object: (offset, length).
    pub fn windows(len: usize) -> Vec<(usize, usize)> {
        if len == 0 {
            return Vec::new();
        }
        let w = FILL_WINDOW.min(len);
        let span = len - w;
        (0..FILL_WINDOWS).map(|i| (span * i / (FILL_WINDOWS - 1), w)).collect::<std::collections::BTreeSet<_>>().into_iter().collect()
    }

    /// The fingerprint of the bytes read at [`Fill::windows`], concatenated.
    pub fn of(sampled: &[u8]) -> Fill {
        if sampled.is_empty() {
            return Fill { zero_permille: 1000, hash: 0 };
        }
        let zeros = sampled.iter().filter(|&&b| b == 0).count();
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in sampled {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
        Fill { zero_permille: (zeros * 1000 / sampled.len()) as u16, hash: h }
    }

    /// All sampled bytes zero: not resident yet (or truly empty).
    pub fn empty(&self) -> bool {
        self.zero_permille == 1000
    }

    fn text(&self) -> String {
        format!("z{}:{:016x}", self.zero_permille, self.hash)
    }

    fn parse(s: &str) -> Option<Fill> {
        let (z, h) = s.strip_prefix('z')?.split_once(':')?;
        Some(Fill { zero_permille: z.parse().ok()?, hash: u64::from_str_radix(h, 16).ok()? })
    }
}

/// One image descriptor reached by the frame's draws.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Texture {
    /// hash128 of the 32 descriptor bytes (blob).
    pub desc: u128,
    /// Texel address the descriptor names.
    pub at: u64,
    /// Bytes from `at` the writer knew of (0: unknown).
    pub len: u64,
    /// Pack origin `<file>#<texture>`, when the origin registry holds it.
    pub origin: Option<String>,
    /// hash128 of the stored texel bytes (0: none stored).
    pub texels: u128,
    pub texel_bytes: u64,
    pub fill: Fill,
    pub reason: u32,
}

/// Per-frame header and totals.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Manifest {
    pub version: u32,
    pub frame: u64,
    pub commit: String,
    pub hook_draws: u32,
    pub sink_draws: u32,
    /// Hook draws with 0, 1 and more than 1 sink draw.
    pub pairs: [u32; 3],
    /// Sink draws with no hook draw on their thread in the frame.
    pub unpaired_sinks: u32,
    /// Dispatch sinks recorded (not in `sink_draws`), and the direct
    /// dispatches the title made in the frame (recorded or not).
    pub dispatches: [u32; 2],
    /// (kind, objects, new bytes written).
    pub bytes: Vec<(u16, u64, u64)>,
    /// (kind, reason, count).
    pub reasons: Vec<(u16, u32, u64)>,
}

impl Manifest {
    pub fn text(&self) -> String {
        let mut t = format!("format\t{FORMAT}\nversion\t{}\nframe\t{}\ncommit\t{}\n", self.version, self.frame, self.commit);
        let _ = writeln!(t, "hook_draws\t{}\nsink_draws\t{}", self.hook_draws, self.sink_draws);
        let _ = writeln!(t, "pairs\t{}\t{}\t{}\nunpaired_sinks\t{}", self.pairs[0], self.pairs[1], self.pairs[2], self.unpaired_sinks);
        let _ = writeln!(t, "dispatches	{}	{}", self.dispatches[0], self.dispatches[1]);
        for (k, n, b) in &self.bytes {
            let _ = writeln!(t, "bytes\t{k}\t{n}\t{b}");
        }
        for (k, r, n) in &self.reasons {
            let _ = writeln!(t, "reason\t{k}\t{r}\t{n}");
        }
        t
    }

    pub fn parse(text: &str) -> Result<Manifest, String> {
        let mut m = Manifest::default();
        let mut format = false;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split('\t').collect();
            let n = |i: usize| f.get(i).and_then(|s| s.parse::<u64>().ok()).ok_or_else(|| format!("manifest: bad line `{line}`"));
            match f[0] {
                "format" => format = f.get(1) == Some(&FORMAT),
                "version" => m.version = n(1)? as u32,
                "frame" => m.frame = n(1)?,
                "commit" => m.commit = f.get(1).unwrap_or(&"").to_string(),
                "hook_draws" => m.hook_draws = n(1)? as u32,
                "sink_draws" => m.sink_draws = n(1)? as u32,
                "pairs" => m.pairs = [n(1)? as u32, n(2)? as u32, n(3)? as u32],
                "unpaired_sinks" => m.unpaired_sinks = n(1)? as u32,
                "dispatches" => m.dispatches = [n(1)? as u32, n(2)? as u32],
                "bytes" => m.bytes.push((n(1)? as u16, n(2)?, n(3)?)),
                "reason" => m.reasons.push((n(1)? as u16, n(2)? as u32, n(3)?)),
                _ => {}
            }
        }
        if !format {
            return Err(format!("manifest: not `{FORMAT}`"));
        }
        if m.version != VERSION {
            return Err(format!("manifest: version {} (this reader: {VERSION})", m.version));
        }
        Ok(m)
    }
}

fn hex_or_dash(v: u128) -> String {
    if v == 0 { "-".into() } else { format!("{v:032x}") }
}

fn parse_hex128(s: &str) -> Option<u128> {
    if s == "-" { Some(0) } else { u128::from_str_radix(s, 16).ok() }
}

fn parse_hex64(s: &str) -> Option<u64> {
    u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()
}

const SINKS_HEAD: &str = "sink\tthread\thook_draw\top\tcount\tpipeline";
const PROGRAMS_HEAD: &str = "pipeline\tstage\tform\tcontainer\tsize\tprefix24\treason";
const TEXTURES_HEAD: &str = "desc\tat\tlen\torigin\ttexels\ttexel_bytes\tfill\treason";

pub fn sinks_text(rows: &[Sink]) -> String {
    let mut t = format!("{SINKS_HEAD}\n");
    for s in rows {
        let hook = s.hook_draw.map_or("-".into(), |d| d.to_string());
        let _ = writeln!(t, "{}\t{}\t{hook}\t{}\t{}\t{:#x}", s.index, s.thread, op::name(s.op), s.count, s.pipeline);
    }
    t
}

pub fn programs_text(rows: &[Program]) -> String {
    let mut t = format!("{PROGRAMS_HEAD}\n");
    for p in rows {
        let prefix: String = p.prefix.iter().map(|b| format!("{b:02x}")).collect();
        let _ = writeln!(t, "{:#x}\t{}\t{}\t{}\t{}\t{prefix}\t{}", p.pipeline, p.stage, p.form, hex_or_dash(p.container), p.size, p.reason);
    }
    t
}

pub fn textures_text(rows: &[Texture]) -> String {
    let mut t = format!("{TEXTURES_HEAD}\n");
    for x in rows {
        let origin = x.origin.as_deref().unwrap_or("-");
        let _ = writeln!(t, "{}\t{:#x}\t{}\t{origin}\t{}\t{}\t{}\t{}", hex_or_dash(x.desc), x.at, x.len, hex_or_dash(x.texels), x.texel_bytes, x.fill.text(), x.reason);
    }
    t
}

fn rows<'a>(text: &'a str, head: &str, file: &str) -> Result<Vec<Vec<&'a str>>, String> {
    let mut lines = text.lines();
    if lines.next() != Some(head) {
        return Err(format!("{file}: unexpected header"));
    }
    Ok(lines.filter(|l| !l.is_empty()).map(|l| l.split('\t').collect()).collect())
}

pub fn parse_sinks(text: &str) -> Result<Vec<Sink>, String> {
    rows(text, SINKS_HEAD, "sinks.tsv")?
        .into_iter()
        .map(|f| {
            let bad = || format!("sinks.tsv: bad row {f:?}");
            if f.len() < 6 {
                return Err(bad());
            }
            Ok(Sink {
                index: f[0].parse().map_err(|_| bad())?,
                thread: f[1].parse().map_err(|_| bad())?,
                hook_draw: if f[2] == "-" { None } else { Some(f[2].parse().map_err(|_| bad())?) },
                op: op::parse(f[3]),
                count: f[4].parse().map_err(|_| bad())?,
                pipeline: parse_hex64(f[5]).ok_or_else(bad)?,
            })
        })
        .collect()
}

pub fn parse_programs(text: &str) -> Result<Vec<Program>, String> {
    rows(text, PROGRAMS_HEAD, "programs.tsv")?
        .into_iter()
        .map(|f| {
            let bad = || format!("programs.tsv: bad row {f:?}");
            if f.len() < 7 || f[5].len() != 48 {
                return Err(bad());
            }
            let mut prefix = [0u8; 24];
            for (i, b) in prefix.iter_mut().enumerate() {
                *b = u8::from_str_radix(&f[5][2 * i..2 * i + 2], 16).map_err(|_| bad())?;
            }
            Ok(Program {
                pipeline: parse_hex64(f[0]).ok_or_else(bad)?,
                stage: f[1].to_string(),
                form: f[2].parse().map_err(|_| bad())?,
                container: parse_hex128(f[3]).ok_or_else(bad)?,
                size: f[4].parse().map_err(|_| bad())?,
                prefix,
                reason: f[6].parse().map_err(|_| bad())?,
            })
        })
        .collect()
}

pub fn parse_textures(text: &str) -> Result<Vec<Texture>, String> {
    rows(text, TEXTURES_HEAD, "textures.tsv")?
        .into_iter()
        .map(|f| {
            let bad = || format!("textures.tsv: bad row {f:?}");
            if f.len() < 8 {
                return Err(bad());
            }
            Ok(Texture {
                desc: parse_hex128(f[0]).ok_or_else(bad)?,
                at: parse_hex64(f[1]).ok_or_else(bad)?,
                len: f[2].parse().map_err(|_| bad())?,
                origin: (f[3] != "-").then(|| f[3].to_string()),
                texels: parse_hex128(f[4]).ok_or_else(bad)?,
                texel_bytes: f[5].parse().map_err(|_| bad())?,
                fill: Fill::parse(f[6]).ok_or_else(bad)?,
                reason: f[7].parse().map_err(|_| bad())?,
            })
        })
        .collect()
}

/// Everything of one frame's pack, in memory (what the writer fills and
/// the reader returns).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FramePack {
    pub manifest: Manifest,
    pub refs: Vec<Ref>,
    pub sinks: Vec<Sink>,
    pub programs: Vec<Program>,
    pub textures: Vec<Texture>,
}

/// `frame-NNNN\pack\`.
pub fn pack_dir(frame_dir: &Path) -> PathBuf {
    frame_dir.join("pack")
}

/// The capture's shared blob directory for a frame directory.
pub fn blob_dir(frame_dir: &Path) -> PathBuf {
    frame_dir.parent().unwrap_or(frame_dir).join("blobs")
}

impl FramePack {
    /// Writes the pack files (not the blobs); the manifest last.
    pub fn write(&mut self, frame_dir: &Path) -> std::io::Result<()> {
        let dir = pack_dir(frame_dir);
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join("refs.bin"), encode_refs(&mut self.refs))?;
        std::fs::write(dir.join("sinks.tsv"), sinks_text(&self.sinks))?;
        std::fs::write(dir.join("programs.tsv"), programs_text(&self.programs))?;
        std::fs::write(dir.join("textures.tsv"), textures_text(&self.textures))?;
        std::fs::write(dir.join("manifest.tsv"), self.manifest.text())
    }

    /// Reads a frame's pack; `Ok(None)` for a v1 frame (no `pack\manifest.tsv`).
    pub fn open(frame_dir: &Path) -> Result<Option<FramePack>, String> {
        let dir = pack_dir(frame_dir);
        let read = |name: &str| std::fs::read(dir.join(name)).map_err(|e| format!("{}: {e}", dir.join(name).display()));
        let text = |name: &str| read(name).map(|b| String::from_utf8_lossy(&b).into_owned());
        if !dir.join("manifest.tsv").exists() {
            return Ok(None);
        }
        Ok(Some(FramePack {
            manifest: Manifest::parse(&text("manifest.tsv")?)?,
            refs: decode_refs(&read("refs.bin")?)?,
            sinks: parse_sinks(&text("sinks.tsv")?)?,
            programs: parse_programs(&text("programs.tsv")?)?,
            textures: parse_textures(&text("textures.tsv")?)?,
        }))
    }

    /// A raw object's bytes from the capture's shared blobs: a v2.1 bundle,
    /// else its own `r-<hash>` file (v2.0).
    pub fn blob(frame_dir: &Path, hash: u128) -> Option<Vec<u8>> {
        let dir = blob_dir(frame_dir);
        bundled(&dir, hash).or_else(|| std::fs::read(dir.join(blob_name(hash))).ok())
    }

    /// The records of one owner, in order.
    pub fn refs_of(&self, owner: u32) -> impl Iterator<Item = &Ref> {
        let start = self.refs.partition_point(|r| r.order() < (owner != FRAME_SCOPE, owner, 0, 0));
        self.refs[start..].iter().take_while(move |r| r.owner == owner)
    }
}

/// Writes `bytes` as blob `hash` unless it exists; true when written.
/// v2.1 bundle index record: hash u128, offset u64, length u64.
pub const BUNDLE_REC: usize = 32;

/// v2.1: one frame's new blobs in one file `blobs-<name>.bin`, indexed
/// by `b-<name>.idx` (written last, through a temporary name: a bundle
/// without its index is not read). Returns the bytes written.
pub fn write_bundle(blob_dir: &Path, name: &str, blobs: &[(u128, &[u8])]) -> std::io::Result<u64> {
    use std::io::Write as _;
    if blobs.is_empty() {
        return Ok(0);
    }
    std::fs::create_dir_all(blob_dir)?;
    let mut idx = Vec::with_capacity(blobs.len() * BUNDLE_REC);
    let mut f = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(blob_dir.join(format!("b-{name}.bin")))?);
    let mut at = 0u64;
    for (h, b) in blobs {
        f.write_all(b)?;
        idx.extend_from_slice(&h.to_le_bytes());
        idx.extend_from_slice(&at.to_le_bytes());
        idx.extend_from_slice(&(b.len() as u64).to_le_bytes());
        at += b.len() as u64;
    }
    f.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    let tmp = blob_dir.join(format!("b-{name}.idx.tmp"));
    std::fs::write(&tmp, &idx)?;
    std::fs::rename(&tmp, blob_dir.join(format!("b-{name}.idx")))?;
    Ok(at)
}

/// The bundles of one blob directory: hash -> (bundle file, offset, length).
#[derive(Default)]
struct Bundles {
    stamp: Option<std::time::SystemTime>,
    files: Vec<PathBuf>,
    seen: std::collections::HashSet<PathBuf>,
    at: std::collections::HashMap<u128, (u32, u64, u64)>,
}

static BUNDLES: std::sync::Mutex<Option<std::collections::HashMap<PathBuf, Bundles>>> = std::sync::Mutex::new(None);

/// A blob from the directory's bundles (indexes read once; read again when
/// the directory changed and a hash is missing).
fn bundled(dir: &Path, hash: u128) -> Option<Vec<u8>> {
    use std::io::{Read as _, Seek as _};
    let (file, off, len) = {
        let mut g = BUNDLES.lock().unwrap_or_else(|e| e.into_inner());
        let b = g.get_or_insert_with(Default::default).entry(dir.to_path_buf()).or_default();
        if !b.at.contains_key(&hash) {
            let stamp = std::fs::metadata(dir).and_then(|m| m.modified()).ok();
            if stamp.is_none() || stamp != b.stamp {
                b.stamp = stamp;
                for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
                    let p = e.path();
                    let n = e.file_name().to_string_lossy().into_owned();
                    if !(n.starts_with("b-") && n.ends_with(".idx")) || b.seen.contains(&p) {
                        continue;
                    }
                    let Ok(idx) = std::fs::read(&p) else { continue };
                    b.seen.insert(p.clone());
                    let k = b.files.len() as u32;
                    b.files.push(p.with_extension("bin"));
                    for r in idx.chunks_exact(BUNDLE_REC) {
                        let h = u128::from_le_bytes(r[..16].try_into().ok()?);
                        let o = u64::from_le_bytes(r[16..24].try_into().ok()?);
                        let l = u64::from_le_bytes(r[24..32].try_into().ok()?);
                        b.at.entry(h).or_insert((k, o, l));
                    }
                }
            }
        }
        let (k, o, l) = *b.at.get(&hash)?;
        (b.files[k as usize].clone(), o, l)
    };
    let mut f = std::fs::File::open(file).ok()?;
    f.seek(std::io::SeekFrom::Start(off)).ok()?;
    let mut out = vec![0u8; usize::try_from(len).ok()?];
    f.read_exact(&mut out).ok()?;
    Some(out)
}

pub fn write_blob(blob_dir: &Path, hash: u128, bytes: &[u8]) -> std::io::Result<bool> {
    let path = blob_dir.join(blob_name(hash));
    if path.exists() {
        return Ok(false);
    }
    std::fs::create_dir_all(blob_dir)?;
    let tmp = blob_dir.join(format!("{}.tmp", blob_name(hash)));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, &path)?;
    Ok(true)
}

/// The `pack-info` coverage of one frame.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Coverage {
    pub sinks: u32,
    pub paired: u32,
    /// Sinks whose pipeline has a named mesh-or-vertex and pixel program.
    pub both_named: u32,
    pub textures: u32,
    /// Textures with stored texels or a pack origin.
    pub textures_held: u32,
    /// (kind, bytes) of records with bytes, by kind.
    pub bytes: Vec<(u16, u64)>,
}

impl Coverage {
    /// `named(container hash)` names a program (the pack join).
    pub fn of(p: &FramePack, named: &dyn Fn(u128) -> bool) -> Coverage {
        // Dispatch sinks are not draws.
        let mut c = Coverage { sinks: p.sinks.iter().filter(|s| !op::is_dispatch(s.op)).count() as u32, ..Coverage::default() };
        c.paired = p.sinks.iter().filter(|s| s.hook_draw.is_some()).count() as u32;
        let stage_named = |pipe: u64, pixel: bool| p.programs.iter().any(|g| g.pipeline == pipe && (g.stage == "pixel" || g.stage == "PS") == pixel && g.reason == 0 && named(g.container));
        c.both_named = p.sinks.iter().filter(|s| !op::is_dispatch(s.op)).filter(|s| stage_named(s.pipeline, false) && stage_named(s.pipeline, true)).count() as u32;
        c.textures = p.textures.len() as u32;
        c.textures_held = p.textures.iter().filter(|t| t.texels != 0 || t.origin.is_some()).count() as u32;
        // Distinct objects only (pk1: summing every reference declared 99.9 GB
        // of buffer bytes for 1,427 distinct blobs).
        let mut by: std::collections::BTreeMap<u16, u64> = std::collections::BTreeMap::new();
        let mut seen = std::collections::HashSet::new();
        for r in p.refs.iter().filter(|r| reason::has_bytes(r.reason) && seen.insert(r.hash)) {
            *by.entry(r.kind).or_default() += u64::from(r.len);
        }
        *by.entry(u16::MAX).or_default() += p.textures.iter().map(|t| t.texel_bytes).sum::<u64>();
        c.bytes = by.into_iter().collect();
        c
    }

    pub fn line(&self) -> String {
        let pct = |a: u32, b: u32| if b == 0 { 0.0 } else { 100.0 * f64::from(a) / f64::from(b) };
        let bytes: Vec<String> = self.bytes.iter().map(|(k, b)| format!("{} {b}", if *k == u16::MAX { "texels" } else { kind::name(*k) })).collect();
        format!(
            "pack: sink draws {} (paired {}), both programs named {} ({:.1} %), textures with texels or origin {} of {} ({:.1} %), bytes {}",
            self.sinks,
            self.paired,
            self.both_named,
            pct(self.both_named, self.sinks),
            self.textures_held,
            self.textures,
            pct(self.textures_held, self.textures),
            bytes.join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    /// v2.7: slots past one row (the frame table's upper half) keep their own
    /// ref slots, apart from the next root's; lower slots as before.
    #[test]
    fn table_slots_past_one_row() {
        assert_eq!((table_slot(4, 1), table_slot(0, 95), table_slot(0, 96), table_slot(0, 181), table_slot(1, 0)), (4 * 96 + 1, 95, UPPER, UPPER | 85, 96));
        for (root, slot) in [(0, 0), (0, 95), (0, 96), (0, 191), (5, 8), (31, 95), (3, 150)] {
            assert_eq!(table_slot_of(table_slot(root, slot)), (root, slot));
        }
    }

    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("xnw-framepack-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d.join("frame-0007")
    }

    fn sample() -> FramePack {
        let r = |owner, kind, slot, len, reason, hash| Ref { owner, kind, slot, addr: 0x1_0000 + u64::from(slot), len, reason, hash };
        FramePack {
            manifest: Manifest { version: VERSION, frame: 7, commit: "abc1234".into(), hook_draws: 2, sink_draws: 2, pairs: [0, 2, 0], unpaired_sinks: 0, dispatches: [3, 1600], bytes: vec![(kind::ROOTS, 2, 64)], reasons: vec![(kind::CBV, reason::UNREADABLE, 1)] },
            refs: vec![
                r(1, kind::CBV, 0x20, 256, reason::UNREADABLE, 0),
                r(0, kind::ROOTS, 3, 32, 0, 0x11),
                r(FRAME_SCOPE, kind::T8, 0, 2880, 0, 0x22),
                r(HOOK | 5, kind::MAT_CB, 1, 96, 0, 0x33),
                r(0, kind::TABLE, 0, 256, 0, 0x44),
                r(1, kind::ROOTS, 3, 32, 0, 0x11),
            ],
            sinks: vec![
                Sink { index: 0, thread: 9140, hook_draw: Some(5), op: op::MESH, count: 12, pipeline: 0xA000 },
                Sink { index: 1, thread: 9140, hook_draw: None, op: op::INDEXED, count: 36, pipeline: 0xB000 },
            ],
            programs: vec![
                Program { pipeline: 0xA000, stage: "MS".into(), form: 1, container: 0xAAAA, size: 4096, prefix: [7; 24], reason: 0 },
                Program { pipeline: 0xA000, stage: "PS".into(), form: 1, container: 0xBBBB, size: 2048, prefix: [8; 24], reason: 0 },
                Program { pipeline: 0xB000, stage: "VS".into(), form: 1, container: 0xCCCC, size: 1024, prefix: [9; 24], reason: 0 },
                Program { pipeline: 0xB000, stage: "PS".into(), form: 0, container: 0, size: 0, prefix: [0; 24], reason: reason::UNREADABLE },
            ],
            textures: vec![
                Texture { desc: 0x55, at: 0x3_0000_0000, len: 65536, origin: Some("x64a.rpf#12".into()), texels: 0, texel_bytes: 0, fill: Fill::of(&[1, 2, 3]), reason: 0 },
                Texture { desc: 0x66, at: 0x3_1000_0000, len: 4096, origin: None, texels: 0x77, texel_bytes: 4096, fill: Fill::of(&[0; 16]), reason: reason::ZERO_FILLED },
                Texture { desc: 0x88, at: 0, len: 0, origin: None, texels: 0, texel_bytes: 0, fill: Fill::default(), reason: reason::NULL },
            ],
        }
    }

    /// pk2 frame 1002's shape (23,252 new blobs, 402 MB) written one file per
    /// blob (v2.0) and as one bundle (v2.1); prints both times. Run by hand:
    /// `cargo test -p xnw-formats bundle_vs_files -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bundle_vs_files_write_time() {
        let dir = tmp("time");
        let n = 23_252usize;
        let each = 402_000_000 / n;
        let blob: Vec<u8> = (0..each).map(|i| i as u8).collect();
        let blobs: Vec<(u128, &[u8])> = (0..n).map(|i| (i as u128 + 1, &blob[..])).collect();
        let t = std::time::Instant::now();
        for (h, b) in &blobs {
            write_blob(&dir.join("files"), *h, b).unwrap();
        }
        let files = t.elapsed();
        let t = std::time::Instant::now();
        write_bundle(&dir.join("bundle"), "1002", &blobs).unwrap();
        let bundle = t.elapsed();
        eprintln!("{n} blobs, {} MB: one file each {:.1} s, one bundle {:.2} s", n * each / 1_000_000, files.as_secs_f64(), bundle.as_secs_f64());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ref_round_trips_every_field() {
        let r = Ref { owner: HOOK | 3, kind: kind::PER_BATCH, slot: 0xBEEF, addr: 0x1_4ABC_E2A0, len: 128, reason: reason::BUDGET, hash: u128::MAX - 5 };
        let mut b = Vec::new();
        r.encode(&mut b);
        assert_eq!(b.len(), REC_BYTES);
        assert_eq!(Ref::decode(&b), Some(r));
        assert!(decode_refs(&b[..39]).is_err());
    }

    #[test]
    fn refs_sort_frame_scope_first_and_select_by_owner() {
        let mut p = sample();
        let bytes = encode_refs(&mut p.refs);
        let back = decode_refs(&bytes).unwrap();
        assert_eq!(back[0].owner, FRAME_SCOPE);
        assert_eq!(back, p.refs);
        p.refs = back;
        assert_eq!(p.refs_of(1).map(|r| r.kind).collect::<Vec<_>>(), [kind::ROOTS, kind::CBV]);
        assert_eq!(p.refs_of(FRAME_SCOPE).count(), 1);
        assert_eq!(p.refs_of(HOOK | 5).next().map(|r| r.hash), Some(0x33));
        assert_eq!(p.refs_of(2).count(), 0);
    }

    #[test]
    fn manifest_round_trips_and_refuses_other_versions() {
        let m = sample().manifest;
        assert_eq!(Manifest::parse(&m.text()), Ok(m.clone()));
        assert!(Manifest::parse(&m.text().replace("version\t2", "version\t3")).is_err());
        assert!(Manifest::parse("frame\t1\n").is_err());
    }

    #[test]
    fn sinks_programs_textures_round_trip() {
        let p = sample();
        assert_eq!(parse_sinks(&sinks_text(&p.sinks)), Ok(p.sinks.clone()));
        assert_eq!(parse_programs(&programs_text(&p.programs)), Ok(p.programs.clone()));
        assert_eq!(parse_textures(&textures_text(&p.textures)), Ok(p.textures.clone()));
    }

    #[test]
    fn dispatch_record_round_trips() {
        let d = Dispatch { groups: [39, 1, 1], flags: B7_FOUND | ITEM_FOUND, b7_slot: 5 * 96 + 8, word: 0x6100_1234, vertices: 1234, b7: (0..96).collect(), item: (100..244u32).map(|i| i as u8).collect() };
        let b = d.encode();
        assert_eq!(b.len(), DISPATCH_BYTES);
        assert_eq!(Dispatch::decode(&b), Some(d));
        // A dispatch without b7 or item still decodes (zeros).
        let none = Dispatch { groups: [1, 2, 3], ..Dispatch::default() };
        let got = Dispatch::decode(&none.encode()).unwrap();
        assert_eq!((got.groups, got.flags, got.b7.len(), got.item.len()), ([1, 2, 3], 0, B7_BYTES, ITEM_BYTES));
        assert_eq!(Dispatch::decode(&b[..100]), None);
        assert_eq!((op::parse("dispatch"), op::name(op::DISPATCH), kind::name(kind::DISPATCH)), (op::DISPATCH, "dispatch", "dispatch"));
    }

    #[test]
    fn sampler_entries_round_trip() {
        let e = [
            SamplerEntry { index: 3, form: SAMPLER_CONSOLE, raw: [7, 8, 1, 0x4549_5676], desc: std::array::from_fn(|i| i as u32 + 100) },
            SamplerEntry { index: 4095, form: SAMPLER_RAW, raw: [0xdead, 0xbeef, 1, 2], desc: [0; 27] },
        ];
        let b = encode_samplers(&e);
        assert_eq!(b.len(), 2 * SAMPLER_BYTES);
        assert_eq!(&b[..8], &[3, 0, 0, 0, 27, 0, 0, 0]);
        assert_eq!(decode_samplers(&b), Some(e.to_vec()));
        assert_eq!(decode_samplers(&b[..SAMPLER_BYTES + 1]), None);
        assert_eq!(decode_samplers(&[]), Some(Vec::new()));
        assert_eq!(kind::name(kind::SAMPLERS), "samplers");
        assert_eq!(kind::name(kind::DRAW_WORDS), "drawwords");
        assert_eq!(kind::name(kind::DISP_MAP), "dispmap");
    }

    #[test]
    fn imposter_instances_round_trip() {
        let i = ImpInstances {
            list: 0x3_0000_1000,
            records_at: 0x3_0000_2000,
            visible_at: 0x305b_0000,
            args_at: 0x3_0000_3000,
            handler_at: 0x3_0000_4000,
            type_at: 0x3_0000_5000,
            count: 3,
            n: 3,
            visible_n: 3,
            reasons: [reason::OK; 6],
            head: (0..0xA0u32).map(|i| i as u8).collect(),
            args: vec![6, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            handler: (0..0x400u32).map(|i| (i >> 2) as u8).collect(),
            type_record: vec![0x42; 0xE0],
            visible: vec![2, 0, 1],
            records: (0..192u32).map(|i| i as u8).collect(),
        };
        let b = i.encode();
        assert_eq!(b.len(), 96 + 0xA0 + 20 + 0x400 + 0xE0 + 12 + 192);
        assert_eq!(ImpInstances::decode(&b), Some(i.clone()));
        assert_eq!(i.record(2).map(|r| r[0]), Some(128));
        assert_eq!(i.record(3), None);
        // A failed part: zeros or empty, its reason kept.
        let none = ImpInstances { list: 0, reasons: [reason::NULL, reason::UNREADABLE, reason::NULL, reason::NULL, reason::NULL, reason::IMPLAUSIBLE], ..ImpInstances::default() };
        let got = ImpInstances::decode(&none.encode()).unwrap();
        assert_eq!((got.head.len(), got.args.len(), got.handler.len(), got.type_record.len(), got.visible.len(), got.records.len()), (IMP_HEAD_BYTES, IMP_ARGS_BYTES, IMP_HANDLER_BYTES, IMP_TYPE_BYTES, 0, 0));
        assert_eq!(got.reasons, none.reasons);
        // Lengths the header does not account for are refused.
        assert_eq!(ImpInstances::decode(&b[..b.len() - 1]), None);
        assert_eq!(ImpInstances::decode(&b[..100]), None);
        let mut long = b.clone();
        long.push(0);
        assert_eq!(ImpInstances::decode(&long), None);
        assert_eq!(kind::name(kind::IMP_INST), "impinst");
    }

    #[test]
    fn slot_views_round_trip() {
        let a = SlotView { slot: 86, kind: view::CONSTANT, addr: 0x2_0000_1000, bytes: 4096, ..SlotView::default() };
        let b = SlotView { slot: 3, kind: view::PLACED, addr: 0x3_0000_0000, bytes: 1 << 20, width: 512, height: 256, depth: 1, mips: 10, format: 71, layout: 0x105, words: std::array::from_fn(|i| i as u32 * 7), ..SlotView::default() };
        let bytes = encode_slots(&[a, b]);
        assert_eq!(bytes.len(), 2 * SLOT_BYTES);
        assert_eq!(decode_slots(&bytes), vec![a, b]);
    }

    #[test]
    fn fill_tells_empty_from_dark() {
        assert!(Fill::of(&[0; 4096]).empty());
        assert!(!Fill::of(&[0x10; 4096]).empty());
        assert_eq!(Fill::of(&[0, 1]).zero_permille, 500);
        let w = Fill::windows(1 << 20);
        assert_eq!(w.len(), FILL_WINDOWS);
        assert_eq!(w[0], (0, FILL_WINDOW));
        assert_eq!(w[FILL_WINDOWS - 1], ((1 << 20) - FILL_WINDOW, FILL_WINDOW));
        assert_eq!(Fill::windows(100), vec![(0, 100)]);
        assert!(Fill::windows(0).is_empty());
    }

    #[test]
    fn frame_pack_writes_reads_and_blobs() {
        let dir = tmp("rt");
        let mut p = sample();
        p.write(&dir).unwrap();
        let back = FramePack::open(&dir).unwrap().unwrap();
        assert_eq!(back.manifest, p.manifest);
        assert_eq!(back.refs, p.refs);
        assert_eq!(back.sinks, p.sinks);
        assert_eq!(back.programs, p.programs);
        assert_eq!(back.textures, p.textures);
        assert!(write_blob(&blob_dir(&dir), 0x44, b"descriptor words").unwrap());
        assert!(!write_blob(&blob_dir(&dir), 0x44, b"descriptor words").unwrap());
        assert_eq!(FramePack::blob(&dir, 0x44).as_deref(), Some(&b"descriptor words"[..]));
        // v2.1: bundles, two frames into one directory; a later bundle is
        // found after the first was indexed; a bundle without .idx is not read.
        let blobs = blob_dir(&dir);
        assert_eq!(write_bundle(&blobs, "7", &[(0x51, b"first"), (0x52, b"second!")]).unwrap(), 12);
        assert_eq!(FramePack::blob(&dir, 0x52).as_deref(), Some(&b"second!"[..]));
        std::thread::sleep(std::time::Duration::from_millis(20));
        write_bundle(&blobs, "8", &[(0x53, &[9u8; 70_000][..])]).unwrap();
        assert_eq!(FramePack::blob(&dir, 0x53).map(|b| b.len()), Some(70_000));
        assert_eq!(FramePack::blob(&dir, 0x51).as_deref(), Some(&b"first"[..]));
        std::fs::remove_file(blobs.join("b-8.idx")).unwrap();
        assert!(FramePack::blob(&dir, 0x99).is_none());
        assert_eq!(FramePack::open(&dir.with_file_name("frame-0008")), Ok(None));
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    #[test]
    fn coverage_counts_named_pairs_and_held_textures() {
        let p = sample();
        let c = Coverage::of(&p, &|h| h == 0xAAAA || h == 0xBBBB || h == 0xCCCC);
        assert_eq!((c.sinks, c.paired, c.both_named, c.textures, c.textures_held), (2, 1, 1, 3, 2));
        assert!(c.line().starts_with("pack: sink draws 2 (paired 1), both programs named 1 (50.0 %), textures with texels or origin 2 of 3 (66.7 %)"), "{}", c.line());
        assert!(c.line().contains("texels 4096"));
    }
}
