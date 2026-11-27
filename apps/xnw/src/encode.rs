//! The world draws' GPU-side encode, timed and, under null, left out (ENG8).
//!
//! Every world draw ends in a geometry draw, `title+0x55e50` (one geometry)
//! or `title+0x2ec470` (instanced), or in an instance batch (`title+0x6b210`,
//! `title+0x13efb0`). Our draw hook sits on their entries (`worldprobe.rs`
//! `ENTRIES`) and reads there what our pass draws from: the TLS block's
//! shader object, model and palette, the geometry head, the transform slot
//! and the per-draw record. What follows the entry is the title encoding the
//! draw for its GPU. In both geometry draws, in this order (machine code):
//!
//! - prestep `title+0x43c5570`: batched instancing; `al = 1` says it drew
//!   the geometry and the function leaves (`test al, al; jne` to its exit,
//!   which clears TLS+0x1170);
//! - kind `title+0x4e34030`: the program's kind (0 none, 1 vertex, 2 mesh),
//!   with the input stream map;
//! - mode `title+0x2495400`: the indirect-mode draw (true: drawn there);
//! - world `title+0x52db2e0`: only when TLS+0x1148.. is not the default;
//! - meshlet `title+0x43c5b80` (kind 2): the cull job or the direct path,
//!   the state flush (`title+0x32520`, `title+0x32130`: tables, constants,
//!   pipeline) and DispatchMesh / ExecuteIndirect on the list;
//! - indexed `title+0x274750` / `title+0x248ce00` (kind 1): flush and
//!   DrawIndexedInstanced.
//!
//! Each is one `call rel32`, pointed at a hook of ours (`titlemeter.rs`
//! `redirect`), which calls the title's function with the five arguments.
//!
//! `XNW_ENCODE_TIME=1`: each call is timed (cycle counter, elapsed on the
//! calling thread); one `title encode:` line per stage-meter window (60
//! world frames; it needs the stage detour and installs it): the rows of
//! the two geometry draws and of the instance batches' draw calls (their
//! sum, and its split by stage, is the encode below our hook), the rows
//! inside them (the direct and indirect list draws with their flush), and
//! above the hook the material bind `title+0x4da8bc0`.
//!
//! `XNW_PATCH=noencode` (or `XNW_AB_PATCH=noencode`: from the A/B switch
//! on): under null (`XNW_GPU=null...`, from its point on) the prestep hook
//! answers 1 without calling anything, the title's own "the prestep drew
//! it" exit, so none of the above runs for that draw. What still reads a
//! draw at the list are the list sink's consumers (`worldprobe.rs`
//! `sink_want`: PreSkinD draws for the pre-skin join, instanced wheels,
//! bridged pixel programs) and the frame pack's live modes (`pack.rs`
//! `live_draw_want`: UI-class draws, hair strands), each by the draw's
//! pipeline, which is made in the flush. So a draw is left out only when its program set (context
//! +0x12328, with the three stage programs it names) is known as one no
//! consumer wants: a set not seen yet is encoded, and `sink_want`'s answer
//! at the list (`learn`) classes it. No such reader registered: every draw
//! is left out. A capture's frame pack (`pack=2`) reads every draw: nothing
//! is left out. The classes are forgotten every 512 world frames (a freed
//! program set's address can come back as another's).
//!
//! `XNW_PATCH=noencodebatch` does the same for the instance batches, at
//! the title's own early exits: `title+0x6b210` first asks
//! `title+0x55ad130` (a per-thread mode table; not 0: return), answered 1;
//! `title+0x13efb0` asks the program kind, answered 0 (no program: it
//! clears its stream bindings and returns).
//!
//! Not run by a skipped draw: the cull job record, the flush, the list
//! call, the kick test after the draw (`title+0x4e88f60`; a pass still
//! ends with its own flush), the topology byte reset (context +0x1237d,
//! set only after a draw changed it). Run as before: everything above the
//! entry (the draw-list walk, the handlers, BeginDraw, the material bind,
//! whose texture lookups keep the streaming stamps).
//!
//! ENG9: one call below the gate is
//! not encode. Both geometry draws invoke the per-geometry delegate at
//! TLS+0x1148 (`title+0x52db2e0`, when it is not the cleared one); three
//! drawable classes install `title+0x4ef140` there, which sets the stencil
//! reference from the geometry's shader (`title+0x4e55390`: the per-slot
//! word at `title+0x8fcf8e0` and context +0x12368). Our hook reads context
//! +0x12360..+0x1237A at the next draw's entry, so a left-out geometry draw
//! still gets that call from the gate (`keep_delegate`). The mesh program's
//! data slot (`title+0x75e8ec2`, set in the kind step), the four draw
//! record slots the pre-draw callback binds (context +0x122C0 ->
//! `title+0x4e30830` / `title+0x4fb9b00`; handles `title+0x8fcf926`..)
//! and the cull dispatch's two (`title+0x840632a`, `..2c`)
//! must not be the matrix slot the hook reads (`title+0x75eacb8`): the
//! gates leave nothing out while one of them names that slot (`slots_same`
//! at every gate, `slots_differ` once a window).
//!
//! The matrix slot itself is rewritten by the encode: the binding flush's
//! table fill (`title+0x4d9efd0`) turns a lazy slot (CPU block, size at
//! context +0x6200 + 2 * slot) into a view and clears the size. The matrix
//! setter above the hook (`title+0x4e3d960`) binds the context's own cache
//! entry, `title+0x75e8f80 + 128 * context[+0x12190]`, with size 128; the
//! hook reads the block while the size is set and that cache entry once it
//! is clear (`worldprobe.rs` `transform_direct`): the same 128 bytes with
//! or without the flush.
//!
//! `XNW_ENCODE_CHECK=1`: what the hook reads (TLS draw words, context
//! matrix index, program set, state bytes, the world matrix as the hook
//! picks it, palette pointer, the 128-B draw record) is hashed before and
//! after one metered call in four; one `title encode check:` line per
//! window names every row inside which a field changed. Only the `world`
//! rows (the delegate, kept) and the prestep (which calls it per part) may
//! show `state`: anything else is an encode write that changes what our
//! hook reads at the next draw, which the gate would take away.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// The title's executable 1.0.2.1 (PE header timestamp).
const BUILD_1_0_2_1: u32 = 0x6AF7_83D4;
/// The switches' names in `XNW_PATCH` and `XNW_AB_PATCH`: the geometry draws' gate, the instance batches'.
pub(crate) const NAMES: [&str; 2] = ["noencode", "noencodebatch"];

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    /// The first call of a geometry draw (group 0) or an instance batch (group 1) after our hook, whose
    /// answer `Site::left` makes the title's function leave: the gate.
    Gate(usize),
    /// A call of a geometry draw that draws: a row of the encode sum; a list draw inside it is `learn`ed.
    Draw,
    /// A call of a geometry draw that does not draw: a row of the encode sum.
    Step,
    /// A call inside a `Draw` row (its time is in that row already).
    Inner,
    /// The material bind, above our hook: not part of the encode sum.
    Bind,
}

/// A metered call: `title+site` is `call rel32` to `title+target`.
struct Site {
    name: &'static str,
    site: usize,
    target: usize,
    kind: Kind,
    /// A gate's answer that leaves the draw out.
    left: usize,
}

const fn s(name: &'static str, site: usize, target: usize, kind: Kind) -> Site {
    Site { name, site, target, kind, left: 1 }
}

/// Sites and targets read from the exe (unit test).
const SITES: [Site; 30] = [
    // The geometry draw `title+0x55e50`, in call order.
    s("geometry.prestep", 0x55e74, 0x43c5570, Kind::Gate(0)),
    s("geometry.kind", 0x55e84, 0x4e34030, Kind::Step),
    s("geometry.mode", 0x55e97, 0x2495400, Kind::Draw),
    s("geometry.world", 0x55f6c, 0x52db2e0, Kind::Step),
    s("geometry.meshlet", 0x55f7c, 0x43c5b80, Kind::Draw),
    s("geometry.indexed", 0x55fc7, 0x274750, Kind::Draw),
    // The instanced geometry draw `title+0x2ec470`, the same calls.
    s("instanced.prestep", 0x2ec49a, 0x43c5570, Kind::Gate(0)),
    s("instanced.kind", 0x2ec4aa, 0x4e34030, Kind::Step),
    s("instanced.mode", 0x2ec4bd, 0x2495400, Kind::Draw),
    s("instanced.world", 0x2ec592, 0x52db2e0, Kind::Step),
    s("instanced.meshlet", 0x2ec5a2, 0x43c5b80, Kind::Draw),
    s("instanced.indexed", 0x2ec5f5, 0x248ce00, Kind::Draw),
    // The instance batch `title+0x6b210`: its mode test, the kind, then one of four list draws with its flush.
    s("batch.mode", 0x6b240, 0x55ad130, Kind::Gate(1)),
    s("batch.kind", 0x6b250, 0x4e34030, Kind::Step),
    s("batch.mesh", 0x6b3ea, 0x555e640, Kind::Draw),
    s("batch.indexed", 0x6b641, 0x248d500, Kind::Draw),
    s("batch.indexed2", 0x6b6d5, 0x248ce00, Kind::Draw),
    s("batch.meshx", 0x6b7d6, 0x555e800, Kind::Draw),
    // The instance batch `title+0x13efb0`: the kind (0: it leaves), then one of two list draws.
    Site { name: "batch2.kind", site: 0x13effa, target: 0x4e34030, kind: Kind::Gate(1), left: 0 },
    s("batch2.meshx", 0x13f0ec, 0x555e800, Kind::Draw),
    s("batch2.indexed", 0x13f257, 0x248d500, Kind::Draw),
    // Inside the meshlet and mode rows: the list draws with their state flush.
    s("meshlet.direct", 0x43c5c74, 0x555e640, Kind::Inner),
    s("meshlet.cull.direct", 0x325b653, 0x555e640, Kind::Inner),
    s("meshlet.cull.indirect", 0x325c070, 0x4e33e10, Kind::Inner),
    s("meshlet.flush.indirect", 0x325b44c, 0x4e33e10, Kind::Inner),
    s("mode.indirect", 0x2495616, 0x4e33e10, Kind::Inner),
    // Above the hook: the material bind, by BeginDraw (`title+0x4f4d380`, `title+0x4e7e620`), the instance
    // batch's caller (`title+0x4ebd630`) and the record list's (`title+0x51ad540`).
    s("bind.shaded", 0x4f4d60c, 0x4da8bc0, Kind::Bind),
    s("bind.plain", 0x4e7e87e, 0x4da8bc0, Kind::Bind),
    s("bind.batch", 0x4ebd9ce, 0x4da8bc0, Kind::Bind),
    s("bind.records", 0x51ad81e, 0x4da8bc0, Kind::Bind),
];
const N: usize = SITES.len();

/// Classes are forgotten after this many world frames.
const FORGET_EVERY: u64 = 512;

static TARGETS: [AtomicUsize; N] = [const { AtomicUsize::new(0) }; N];
static CYCLES: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static CALLS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
/// The encode sum (`Gate`, `Draw`, `Step` rows) by the stage of the calling thread.
static BY_STAGE: [AtomicU64; crate::stagemeter::STAGES] = [const { AtomicU64::new(0) }; crate::stagemeter::STAGES];
static TIME: AtomicBool = AtomicBool::new(false);
/// The gates leave draws out (by group: geometry draws, instance batches).
static SKIP: [AtomicBool; 2] = [const { AtomicBool::new(false) }; 2];
/// The gates' hooks are in place (the switch can come later: `XNW_AB_PATCH`).
static GATED: AtomicBool = AtomicBool::new(false);
/// Draws at a group's gates since the last line: left out; encoded for a consumer; encoded, set not classed
/// yet; encoded for another reason (not null yet, the frame pack's tap, no program set).
static GATE_N: [[AtomicU64; 4]; 2] = [const { [const { AtomicU64::new(0) }; 4] }; 2];
static FRAMES: AtomicU64 = AtomicU64::new(0);
/// The mapped title (0: nothing installed) and its delegate invoke `title+0x52db2e0`.
static BASE: AtomicUsize = AtomicUsize::new(0);
/// Left-out geometry draws that had a per-geometry delegate, which the gate called.
static DELEGATES: AtomicU64 = AtomicU64::new(0);
/// `XNW_ENCODE_CHECK`: on; per row the calls sampled and, per field of `FIELDS`, those that changed it.
static CHECK: AtomicBool = AtomicBool::new(false);
const FIELDS: [&str; 7] = ["tls", "matrix index", "program set", "state", "world matrix", "palette", "draw record"];
static CHECKED: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static CHANGED: [[AtomicU64; 7]; N] = [const { [const { AtomicU64::new(0) }; 7] }; N];
/// The slot guard's answer was logged; the gates were turned off by it.
static SLOTS_SAID: AtomicBool = AtomicBool::new(false);

/// The delegate invoke both geometry draws call after the mode step (unit test: its call sites).
const DELEGATE_INVOKE: usize = 0x52d_b2e0;
/// TLS block +0x1148: the per-geometry delegate (object, function, -, argument, kind u32).
const TLS_DELEGATE: usize = 0x1148;
/// The resource-slot handles (u16, slot = handle >> 2) the encode stores into the context's slot table
/// (+0x40 + 8 * slot, lazy size at +0x6200 + 2 * slot): the mesh program's data slot the kind step sets
/// (`title+0x555ee70`), the four per-draw records the pre-draw callback binds when they are dirty
/// (`title+0x4e308e0`, `title+0x4fb9bb0` through `title+0x4e31090`: 128, 68, 340 and 288 bytes), and the
/// cull dispatch's two (`title+0x325c500`).
const MESH_SLOT_HANDLE: usize = 0x75e_8ec2;
const RECORD_SLOT_HANDLES: [usize; 4] = [0x8fc_f926, 0x8fc_f928, 0x8fc_f92a, 0x8fc_f92c];
const CULL_SLOT_HANDLES: [usize; 2] = [0x840_632a, 0x840_632c];
/// The matrix slot the hook reads (`worldprobe.rs` `MATRIX_SLOT`; bound by the matrix setter above the hook,
/// `title+0x4e3d960`, to the context's entry of the cache at `MATRIX_CACHE`, 128 B each).
const MATRIX_SLOT_HANDLE: usize = 0x75e_acb8;
const MATRIX_CACHE: usize = 0x75e_8f80;

/// A program set's class, by key: `key & !3 | class` (0 free). Open addressing, 8 probes.
const SETS: usize = 4096;
const WANTED: u64 = 2;
const UNWANTED: u64 = 1;
static CLASS: [AtomicU64; SETS] = [const { AtomicU64::new(0) }; SETS];

thread_local! {
    /// The key of the draw at this thread's last gate (0: none, or the gate keeps everything).
    static KEY: Cell<u64> = const { Cell::new(0) };
    /// The key list draws are classed under now (set inside a `Draw` row of a gated draw).
    static LEARNING: Cell<u64> = const { Cell::new(0) };
    /// `XNW_ENCODE_CHECK`: this thread's metered calls (one in four is checked).
    static TICK: Cell<u32> = const { Cell::new(0) };
}

/// The calling thread's title TLS block (0: none).
#[cfg(not(test))]
fn tls_block() -> usize {
    let index = xnw_core::TITLE_TLS_INDEX.load(Ordering::Relaxed) as usize;
    let array: usize;
    // SAFETY: reads TEB.ThreadLocalStoragePointer of the calling thread.
    unsafe { std::arch::asm!("mov {}, gs:[0x58]", out(reg) array, options(nostack, readonly, preserves_flags)) };
    // SAFETY: the thread's TLS array holds a slot for every module index.
    unsafe { ((array + 8 * index) as *const usize).read() }
}

/// A left-out geometry draw's per-geometry delegate, called as the title's draw does (`lea rcx,
/// [tls+1148h]; xor edx, edx; call title+0x52db2e0` when the delegate is not the cleared one).
#[cfg(not(test))]
fn keep_delegate() {
    let (base, block) = (BASE.load(Ordering::Relaxed), tls_block());
    if base == 0 || block == 0 {
        return;
    }
    // SAFETY: the calling title thread's TLS block; the title's draw reads these four words next.
    let d = unsafe { ((block + TLS_DELEGATE) as *const [u64; 5]).read_unaligned() };
    if d[0] != 0 || d[1] != 0 || d[3] != 0 || d[4] as u32 != 0 {
        // SAFETY: the title's delegate invoke, with the arguments its two call sites pass.
        let f: extern "system" fn(usize, usize) = unsafe { std::mem::transmute(base + DELEGATE_INVOKE) };
        f(block + TLS_DELEGATE, 0);
        DELEGATES.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
fn keep_delegate() {
    DELEGATES.fetch_add(1, Ordering::Relaxed);
}

/// What the draw hook reads at an entry (`worldprobe.rs` `capture_light`, `transform_direct`), one hash
/// per field of `FIELDS`.
#[cfg(not(test))]
fn hook_inputs() -> [u64; 7] {
    let mut out = [0u64; 7];
    let (base, block) = (BASE.load(Ordering::Relaxed), tls_block());
    if base == 0 || block == 0 {
        return out;
    }
    let fnv = |bytes: &[u8]| bytes.iter().fold(0xCBF2_9CE4_8422_2325u64, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01B3));
    // SAFETY: the calling title thread inside a draw: its TLS block, its render context and the title's
    // image, all mapped; plain reads of what the hook reads.
    unsafe {
        let r = |a: usize, n: usize| std::slice::from_raw_parts(a as *const u8, n);
        // TLS: drawable, palette count and pointers, model, geometry index, shader, context, viewport,
        // technique, the record index (+0x1110, +0x1294) and the stage (+0x950).
        for (at, n) in [(0x11D8, 8), (0x11E4, 4), (0x11E8, 16), (0x1208, 8), (0x1210, 4), (0x1220, 8), (0x12A8, 8), (0x12B8, 8), (0x12C0, 1), (0x1294, 1), (0x1110, 4), (0x950, 4)] {
            out[0] = out[0].rotate_left(7) ^ fnv(r(block + at, n));
        }
        let ctx = ((block + 0x12A8) as *const usize).read();
        if ctx != 0 {
            out[1] = u64::from(((ctx + 0x1_2190) as *const u32).read());
            out[2] = ((ctx + 0x1_2328) as *const u64).read();
            out[3] = fnv(r(ctx + 0x1_2360, 0x1A));
            // The world matrix as `transform_direct` picks it: the slot's CPU block while its lazy size is
            // set (the block the flush would copy: readable for that size), else the context's cache entry.
            let slot = usize::from(((base + MATRIX_SLOT_HANDLE) as *const u16).read() >> 2);
            let (bound, lazy) = (((ctx + 0x40 + 8 * slot) as *const usize).read(), usize::from(((ctx + 0x6200 + 2 * slot) as *const u16).read()));
            let index = ((ctx + 0x1_2190) as *const i32).read();
            let mut block = [0u8; 128];
            out[4] = if lazy != 0 && bound != 0 {
                // A guarded copy: nothing says a lazy block never flushed is still mapped.
                let n = lazy.min(128);
                if crate::watch::guarded_read(bound, &mut block[..n]) { fnv(&block[..n]) } else { bound as u64 }
            } else if (0..4096).contains(&index) {
                fnv(r(base + MATRIX_CACHE + 128 * index as usize, 128))
            } else {
                bound as u64
            };
            out[5] = ((ctx + 0xA10) as *const u64).read();
        }
        let record = ((block + 0x1110) as *const u32).read() as usize + usize::from(((block + 0x1294) as *const u8).read());
        // Six records of 128 B (the next table starts at `title+0x8fca520`).
        if record < 6 {
            out[6] = fnv(r(base + 0x8FC_A220 + 128 * record, 128));
        }
    }
    out
}

#[cfg(test)]
thread_local! {
    static TEST_INPUTS: Cell<[u64; 7]> = const { Cell::new([0; 7]) };
}
#[cfg(test)]
fn hook_inputs() -> [u64; 7] {
    TEST_INPUTS.with(Cell::get)
}

/// Whether the mesh program's data slot and the hook's matrix slot differ (true before either handle is
/// registered: both 0, no mesh program can draw then).
/// The slots the encode binds (mesh data, the four records, the cull dispatch's two) and the hook's
/// matrix slot (0: not registered yet).
#[cfg(not(test))]
fn slots() -> ([u16; 7], u16) {
    let base = BASE.load(Ordering::Relaxed);
    if base == 0 {
        return ([0; 7], 0);
    }
    // SAFETY: eight words of the mapped title's data.
    let slot = |rva: usize| unsafe { ((base + rva) as *const u16).read() >> 2 };
    let (r, c) = (RECORD_SLOT_HANDLES, CULL_SLOT_HANDLES);
    ([slot(MESH_SLOT_HANDLE), slot(r[0]), slot(r[1]), slot(r[2]), slot(r[3]), slot(c[0]), slot(c[1])], slot(MATRIX_SLOT_HANDLE))
}

#[cfg(test)]
static TEST_SLOTS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
#[cfg(test)]
fn slots() -> ([u16; 7], u16) {
    ([TEST_SLOTS[0].load(Ordering::Relaxed) as u16, 2, 3, 4, 0, 6, 7], TEST_SLOTS[1].load(Ordering::Relaxed) as u16)
}

/// The slot guard: a slot the encode binds is the hook's matrix slot (the encode would write what the
/// hook reads next). Asked at every gate (eight words), so no draw is left out before a window has looked.
fn slots_same() -> bool {
    let (encode, matrix) = slots();
    matrix != 0 && encode.contains(&matrix)
}

/// The guard's line, at a window's end while a gate is on: the slots once the matrix slot is registered;
/// a shared slot turns the gates off for good.
fn slots_differ() -> bool {
    let (encode, matrix) = slots();
    if matrix == 0 {
        return true;
    }
    if encode.contains(&matrix) {
        for g in &SKIP {
            g.store(false, Ordering::Relaxed);
        }
        if !SLOTS_SAID.swap(true, Ordering::Relaxed) {
            xnw_core::warn!("title encode gate: the hook's matrix slot {matrix} is one the encode binds (mesh data, draw records, cull inputs {encode:?}); noencode turned off");
        }
        return false;
    }
    if !SLOTS_SAID.swap(true, Ordering::Relaxed) {
        xnw_core::info!("title encode gate: the encode binds slots {encode:?} (mesh data, four draw records, two cull inputs), the hook's matrix slot is {matrix} (they differ)");
    }
    true
}

fn slot_of(key: u64) -> usize {
    ((key >> 2).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 52) as usize % SETS
}

/// The class of `key` (0: not seen).
fn class_of(key: u64) -> u64 {
    let start = slot_of(key);
    for k in 0..8 {
        let e = CLASS[(start + k) % SETS].load(Ordering::Relaxed);
        if e == 0 {
            return 0;
        }
        if e & !3 == key & !3 {
            return e & 3;
        }
    }
    0
}

/// Classes `key`; a wanted set stays wanted. A full neighbourhood leaves it unclassed (always encoded).
fn class_set(key: u64, wanted: bool) {
    let start = slot_of(key);
    let new = key & !3 | if wanted { WANTED } else { UNWANTED };
    for k in 0..8 {
        let cell = &CLASS[(start + k) % SETS];
        let e = cell.load(Ordering::Relaxed);
        if e == 0 {
            if cell.compare_exchange(0, new, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                return;
            }
        } else if e & !3 == key & !3 {
            if e & 3 != WANTED && wanted {
                cell.store(new, Ordering::Relaxed);
            }
            return;
        }
    }
}

/// The list sink's answer for the draw being recorded on this thread (`worldprobe.rs` `sink_want`): whether
/// a consumer wants its pipeline. Classes the program set of the gated draw it belongs to, if any.
pub(crate) fn learn(wanted: bool) {
    let key = LEARNING.with(Cell::get);
    if key != 0 {
        class_set(key, wanted);
    }
}

fn mix(words: [u64; 6]) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for w in words {
        h = (h ^ w).wrapping_mul(0x0000_0100_0000_01B3);
        h ^= h >> 29;
    }
    // Never 0 (none), the class bits clear.
    (h | 4) & !3
}

/// The key of the calling thread's program set: its address and the three stage programs it names, with
/// the context's technique variant (+0x12743) and pass record (+0x12748), which BeginDraw set (0: none).
#[cfg(not(test))]
fn program_key() -> u64 {
    // TLS block +0x12A8: the render context; context +0x12328: the program set.
    const TLS_CTX: usize = 0x12A8;
    const CTX_PROGRAMS: usize = 0x1_2328;
    const CTX_VARIANT: usize = 0x1_2743;
    const CTX_PASS: usize = 0x1_2748;
    let block = tls_block();
    if block == 0 {
        return 0;
    }
    // SAFETY: a title thread inside a geometry draw: its TLS block, the context and the program set are the
    // ones the draw itself reads next (prestep `title+0x43c5570` reads set+8, the meshlet draw set+0x20).
    unsafe {
        let ctx = ((block + TLS_CTX) as *const usize).read();
        if ctx == 0 {
            return 0;
        }
        let set = ((ctx + CTX_PROGRAMS) as *const usize).read();
        if set == 0 {
            return 0;
        }
        let w = ((set + 8) as *const [u64; 3]).read_unaligned();
        let (variant, pass) = (((ctx + CTX_VARIANT) as *const u8).read(), ((ctx + CTX_PASS) as *const u64).read_unaligned());
        mix([set as u64, w[0], w[1], w[2], u64::from(variant), pass])
    }
}

#[cfg(test)]
thread_local! {
    static TEST_KEY: Cell<u64> = const { Cell::new(0) };
}
#[cfg(test)]
fn program_key() -> u64 {
    TEST_KEY.with(Cell::get)
}

/// What the list's readers are: (null is on, a capture's frame pack reads every draw, a reader that picks
/// draws by pipeline is registered).
#[cfg(not(test))]
fn readers() -> (bool, bool, bool) {
    let (every, picking) = crate::worldprobe::list_readers();
    (xnw_gpu::d3d12x::null_now(), every, picking)
}

#[cfg(test)]
static TEST_READERS: [AtomicBool; 3] = [const { AtomicBool::new(false) }; 3];
#[cfg(test)]
fn readers() -> (bool, bool, bool) {
    (TEST_READERS[0].load(Ordering::Relaxed), TEST_READERS[1].load(Ordering::Relaxed), TEST_READERS[2].load(Ordering::Relaxed))
}

/// A gate's decision for the draw on this thread: true = leave it out. Counts it and keeps its key.
fn leave_out(group: usize) -> bool {
    KEY.with(|k| k.set(0));
    if !SKIP[group].load(Ordering::Relaxed) {
        return false;
    }
    let (null, pack, sinks) = readers();
    let n = if !null || pack || slots_same() {
        3
    } else if !sinks {
        0
    } else {
        let key = program_key();
        KEY.with(|k| k.set(key));
        match (key, class_of(key)) {
            (0, _) => 3,
            (_, UNWANTED) => 0,
            (_, WANTED) => 1,
            _ => 2,
        }
    };
    GATE_N[group][n].fetch_add(1, Ordering::Relaxed);
    n == 0
}

fn tsc() -> u64 {
    // SAFETY: reads the time-stamp counter.
    #[allow(unused_unsafe)]
    unsafe {
        core::arch::x86_64::_rdtsc()
    }
}

fn run(i: usize, a: usize, b: usize, c: usize, d: usize, e: usize) -> usize {
    let kind = SITES[i].kind;
    if let Kind::Gate(group) = kind
        && leave_out(group)
    {
        if group == 0 {
            keep_delegate();
        }
        return SITES[i].left;
    }
    let before = (CHECK.load(Ordering::Relaxed) && TICK.with(|t| t.replace(t.get().wrapping_add(1))) & 3 == 0).then(hook_inputs);
    // SAFETY: the title's function this call site named, set at install; none takes more than five
    // integer arguments or a float.
    let f: extern "system" fn(usize, usize, usize, usize, usize) -> usize = unsafe { std::mem::transmute(TARGETS[i].load(Ordering::Relaxed)) };
    let draws = matches!(kind, Kind::Gate(_) | Kind::Draw);
    if draws {
        LEARNING.with(|l| l.set(KEY.with(Cell::get)));
    }
    let r = if TIME.load(Ordering::Relaxed) {
        let t0 = tsc();
        let r = f(a, b, c, d, e);
        let dt = tsc().wrapping_sub(t0);
        CYCLES[i].fetch_add(dt, Ordering::Relaxed);
        CALLS[i].fetch_add(1, Ordering::Relaxed);
        if !matches!(kind, Kind::Inner | Kind::Bind) {
            BY_STAGE[crate::stagemeter::STAGE.with(Cell::get)].fetch_add(dt, Ordering::Relaxed);
        }
        r
    } else {
        f(a, b, c, d, e)
    };
    if draws {
        LEARNING.with(|l| l.set(0));
    }
    if let Some(before) = before {
        let after = hook_inputs();
        CHECKED[i].fetch_add(1, Ordering::Relaxed);
        for (k, _) in before.iter().zip(&after).enumerate().filter(|(_, (a, b))| a != b) {
            CHANGED[i][k].fetch_add(1, Ordering::Relaxed);
        }
    }
    r
}

/// One hook per row: each knows its row.
macro_rules! hooks {
    ($($i:literal)*) => {
        [$({
            extern "system" fn h(a: usize, b: usize, c: usize, d: usize, e: usize) -> usize {
                run($i, a, b, c, d, e)
            }
            h as usize
        }),*]
    };
}

fn hook_table() -> [usize; N] {
    hooks!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29)
}

/// Which of `NAMES` a comma list holds.
fn named(list: &str) -> [bool; 2] {
    NAMES.map(|name| list.split(',').any(|n| n.trim().eq_ignore_ascii_case(name)))
}

/// Reads `XNW_ENCODE_TIME`; whether the stage detour is wanted: for the meter's line, and for a gate, whose
/// classes are forgotten by the stage meter's frame count.
pub(crate) fn configure() -> bool {
    let on = match std::env::var("XNW_ENCODE_TIME").unwrap_or_default().trim() {
        "" | "0" => false,
        "1" => true,
        other => {
            xnw_core::warn!("XNW_ENCODE_TIME={other}: not 0 or 1; the encode meter stays off");
            false
        }
    };
    TIME.store(on, Ordering::Relaxed);
    let check = std::env::var("XNW_ENCODE_CHECK").is_ok_and(|v| v.trim() == "1");
    CHECK.store(check, Ordering::Relaxed);
    let gate = |k: &str| named(&std::env::var(k).unwrap_or_default()).contains(&true);
    on || check || gate("XNW_PATCH") || gate("XNW_AB_PATCH")
}

/// Turns on, while the title runs, the gates `names` holds (`ab.rs`, at the B switch); the names turned on
/// (none when the hooks are not in place).
pub(crate) fn skip_live(names: &[String]) -> Vec<String> {
    let mut done = Vec::new();
    if GATED.load(Ordering::Relaxed) {
        for (g, name) in NAMES.iter().enumerate() {
            if names.iter().any(|n| n == name) {
                SKIP[g].store(true, Ordering::Relaxed);
                done.push((*name).to_owned());
            }
        }
    }
    done
}

/// Installs what `XNW_ENCODE_TIME`, `XNW_PATCH=noencode[,noencodebatch]` and `XNW_AB_PATCH` with those
/// names ask for.
///
/// # Safety
/// `base` is the mapped title and none of its code runs yet.
pub unsafe fn install(base: usize) {
    let env = |k: &str| std::env::var(k).unwrap_or_default();
    // The check hooks every row, as the meter does.
    let (time, check, skip, later) = (TIME.load(Ordering::Relaxed), CHECK.load(Ordering::Relaxed), named(&env("XNW_PATCH")), named(&env("XNW_AB_PATCH")));
    let gate = skip.contains(&true) || later.contains(&true);
    if !time && !check && !gate {
        return;
    }
    // SAFETY: forwarded; SizeOfImage is at the optional header's +0x38.
    let (stamp, size) = unsafe {
        let nt = (base as *const u8).add(0x3c).cast::<u32>().read_unaligned() as usize;
        (crate::patches::timestamp(base), (base as *const u8).add(nt + 0x50).cast::<u32>().read_unaligned() as usize)
    };
    if stamp != BUILD_1_0_2_1 {
        xnw_core::warn!("title encode: title timestamp {stamp:#x} is not build 1.0.2.1; nothing installed");
        return;
    }
    let Some(page) = crate::titlemeter::near_page(base, base + size) else {
        xnw_core::warn!("title encode: no page within 2 GiB of the title; nothing installed");
        return;
    };
    crate::stagemeter::cycles_per_second();
    BASE.store(base, Ordering::Relaxed);
    // The gates alone need their four calls and the rows that draw after them (the list's answer is
    // taken inside those); the meter all rows.
    let wanted = |k: Kind| time || check || matches!(k, Kind::Gate(_) | Kind::Draw);
    let (hooks, mut done, mut gates) = (hook_table(), 0, 0);
    for (i, s) in SITES.iter().enumerate().filter(|(_, s)| wanted(s.kind)) {
        TARGETS[i].store(base + s.target, Ordering::Relaxed);
        // SAFETY: inside the mapped .text of this build, no code running; 16 bytes of our page per row.
        match unsafe { crate::titlemeter::redirect((base + s.site) as *mut u8, base + s.target, page.add(16 * i), hooks[i]) } {
            Ok(()) => {
                done += 1;
                gates += usize::from(matches!(s.kind, Kind::Gate(_)));
            }
            Err(why) => xnw_core::warn!("title encode: {} at title+{:#x} skipped: {why}", s.name, s.site),
        }
    }
    let rows = SITES.iter().filter(|s| wanted(s.kind)).count();
    if check {
        xnw_core::info!("title encode check: on, the hook's inputs hashed around one call in four of {done} call sites; one line per stage-meter window");
    }
    if time {
        xnw_core::info!("title encode: {done} of {rows} call sites metered (geometry draws title+0x55e50 and title+0x2ec470, instance batches, material bind); one line per stage-meter window");
    }
    if !gate {
        return;
    }
    // All four gates or none: a draw row classes under the key its function's gate left.
    let all = gates == SITES.iter().filter(|s| matches!(s.kind, Kind::Gate(_))).count();
    GATED.store(all, Ordering::Relaxed);
    for (g, name) in NAMES.iter().enumerate().filter(|&(g, _)| skip[g] || later[g]) {
        if !all {
            xnw_core::warn!("patch {name} skipped: {gates} of the gate calls are the expected ones");
            continue;
        }
        SKIP[g].store(skip[g], Ordering::Relaxed);
        xnw_core::info!(
            "patch {name} {}: under null {} whose program set no list consumer wants leaves at its first call; {done} of {rows} call sites hooked",
            if skip[g] { "applied" } else { "armed for the A/B switch" },
            if g == 0 { "a geometry draw (title+0x55e74, title+0x2ec49a)" } else { "an instance batch (title+0x6b240, title+0x13effa)" }
        );
    }
}

/// The line for the last `frames` frames; the sums start again.
fn take_line(frames: u64) -> String {
    let (hz, f) = (crate::stagemeter::cycles_per_second(), frames.max(1) as f64);
    let ms = |c: u64| c as f64 / hz * 1000.0 / f;
    let (mut sum, mut groups) = (0u64, [const { Vec::new() }; 3]);
    for (i, s) in SITES.iter().enumerate() {
        let (c, n) = (CYCLES[i].swap(0, Ordering::Relaxed), CALLS[i].swap(0, Ordering::Relaxed));
        let g = match s.kind {
            Kind::Inner => 1,
            Kind::Bind => 2,
            _ => {
                sum += c;
                0
            }
        };
        if n != 0 {
            groups[g].push(format!("{} {:.2} ({:.1})", s.name, ms(c), n as f64 / f));
        }
    }
    let mut stages: Vec<(usize, u64)> = BY_STAGE.iter().enumerate().map(|(i, c)| (i, c.swap(0, Ordering::Relaxed))).filter(|r| r.1 != 0).collect();
    stages.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let by: Vec<String> = stages.iter().map(|&(i, c)| if i == crate::stagemeter::NO_STAGE { format!("none {:.2}", ms(c)) } else { format!("{i} {:.2}", ms(c)) }).collect();
    let join = |v: &Vec<String>| if v.is_empty() { "none".to_owned() } else { v.join(", ") };
    format!(
        "title encode: {frames} frames, ms a frame elapsed (calls a frame) below the draw hook: {}; all {:.2}; by stage: {}; inside them: {}; above the hook: {}",
        join(&groups[0]),
        ms(sum),
        join(&by),
        join(&groups[1]),
        join(&groups[2])
    )
}

/// The gates' line for the last `frames` frames; the counts start again.
fn take_gate_line(frames: u64) -> String {
    let f = frames.max(1) as f64;
    let part = |g: usize| {
        let n: Vec<f64> = GATE_N[g].iter().map(|c| c.swap(0, Ordering::Relaxed) as f64 / f).collect();
        format!("left out {:.1}, encoded {:.1} (for a list consumer {:.1}, program set not classed yet {:.1}, other {:.1})", n[0], n[1] + n[2] + n[3], n[1], n[2], n[3])
    };
    let (mut wanted, mut unwanted) = (0, 0);
    for c in &CLASS {
        match c.load(Ordering::Relaxed) & 3 {
            WANTED => wanted += 1,
            UNWANTED => unwanted += 1,
            _ => {}
        }
    }
    let delegates = DELEGATES.swap(0, Ordering::Relaxed) as f64 / f;
    format!(
        "title encode gate: {frames} frames, a frame: geometry draws {} (per-geometry delegate still called {delegates:.1}); instance batches {}; program sets classed: wanted {wanted}, not wanted {unwanted}",
        part(0),
        part(1)
    )
}

/// The check's line for the last `frames` frames; the counts start again.
fn take_check_line(frames: u64) -> String {
    let (mut sampled, mut rows) = (0, Vec::new());
    for (i, s) in SITES.iter().enumerate() {
        let n = CHECKED[i].swap(0, Ordering::Relaxed);
        sampled += n;
        let changed: Vec<String> = FIELDS.iter().enumerate().map(|(k, name)| (name, CHANGED[i][k].swap(0, Ordering::Relaxed))).filter(|r| r.1 != 0).map(|(name, c)| format!("{name} {c}")).collect();
        if !changed.is_empty() {
            rows.push(format!("{} ({} of {n})", s.name, changed.join(", ")));
        }
    }
    format!(
        "title encode check: {frames} frames, {sampled} calls sampled; hook inputs changed inside: {}",
        if rows.is_empty() { "none".to_owned() } else { rows.join("; ") }
    )
}

/// The end of a stage-meter window of `frames` world frames (`stagemeter.rs`): the lines, tagged `tag`
/// (`[A]`, `[B]` or nothing), and the classes' forgetting.
pub(crate) fn window_end(frames: u64, tag: &str) {
    if TIME.load(Ordering::Relaxed) {
        xnw_core::info!("{}", take_line(frames).replacen("title encode:", &format!("title encode{tag}:"), 1));
    }
    if CHECK.load(Ordering::Relaxed) {
        xnw_core::info!("{}", take_check_line(frames).replacen("title encode check:", &format!("title encode check{tag}:"), 1));
    }
    if SKIP.iter().any(|s| s.load(Ordering::Relaxed)) && slots_differ() {
        xnw_core::info!("{}", take_gate_line(frames).replacen("title encode gate:", &format!("title encode gate{tag}:"), 1));
        let n = FRAMES.fetch_add(frames, Ordering::Relaxed);
        if n / FORGET_EVERY != (n + frames) / FORGET_EVERY {
            for c in &CLASS {
                c.store(0, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xnw_core::win::{MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc};

    /// Every site is a `call rel32` to its target in the executable; the two gates are followed by the
    /// title's own test of the prestep's answer, whose taken branch is the function's exit.
    #[test]
    fn the_sites_call_their_targets_in_the_executable() {
        let Some(exe) = crate::patches::Exe::open() else {
            eprintln!("no title executable here (XNW_TITLE_EXE): the encode sites were not checked against it");
            return;
        };
        for s in &SITES {
            assert_eq!((exe.at(s.site, 1)[0], exe.rel32_target(s.site)), (0xE8, s.target), "{} at title+{:#x}", s.name, s.site);
        }
        assert_eq!(hook_table().len(), SITES.len());
        // Call order inside each geometry draw: the gate first, then kind, mode, world, meshlet, indexed.
        for (first, entry, end) in [(0, 0x55e50, 0x560a1), (6, 0x2ec470, 0x2ec6d2)] {
            assert!(SITES[first..first + 6].windows(2).all(|w| w[0].site < w[1].site));
            assert!(SITES[first..first + 6].iter().all(|s| (entry..end).contains(&s.site)));
            assert_eq!((SITES[first].kind, SITES[first].left), (Kind::Gate(0), 1));
            assert!(SITES[first + 1..first + 6].iter().all(|s| !matches!(s.kind, Kind::Gate(_))));
        }
        // The instance batches: the gate is the first metered call of its function, the draw rows follow it.
        for (first, n, entry, end) in [(12, 6, 0x6b210, 0x6b93b), (18, 3, 0x13efb0, 0x13f396)] {
            assert!(SITES[first..first + n].windows(2).all(|w| w[0].site < w[1].site));
            assert!(SITES[first..first + n].iter().all(|s| (entry..end).contains(&s.site)));
            assert_eq!(SITES[first].kind, Kind::Gate(1));
            assert!(SITES[first + 1..first + n].iter().all(|s| matches!(s.kind, Kind::Draw | Kind::Step)));
        }
        assert_eq!(SITES.iter().filter(|s| matches!(s.kind, Kind::Gate(_))).count(), 4);
        // title+0x6b240: the mode test; test al, al; jne title+0x6b8d0, the epilogue (add rsp, 28h; 8 pops;
        // vzeroupper; ret). The function it asks is a table read by the TLS byte +0x1294.
        assert_eq!((SITES[12].site, SITES[12].left), (0x6b240, 1));
        assert_eq!(exe.at(0x6b245, 8), [0x84, 0xC0, 0x0F, 0x85, 0x83, 0x06, 0x00, 0x00]);
        assert_eq!(0x6b24d + 0x683, 0x6b8d0);
        assert_eq!(exe.at(0x6b8d0, 20), [0x48, 0x83, 0xC4, 0x28, 0x5B, 0x5D, 0x5F, 0x5E, 0x41, 0x5C, 0x41, 0x5D, 0x41, 0x5E, 0x41, 0x5F, 0xC5, 0xF8, 0x77, 0xC3]);
        // title+0x13effa: the kind; cmp eax, 1; je ..; cmp eax, 2; jne title+0x13f27b, which clears the
        // stream bindings when context +0x12530 is not 0 (cmp dword [rax+12530h], 0) and returns.
        assert_eq!((SITES[18].site, SITES[18].left), (0x13effa, 0));
        assert_eq!(exe.at(0x13efff, 18), [0x83, 0xF8, 0x01, 0x0F, 0x84, 0x36, 0x01, 0x00, 0x00, 0x83, 0xF8, 0x02, 0x0F, 0x85, 0x6A, 0x02, 0x00, 0x00]);
        assert_eq!(0x13f011 + 0x26a, 0x13f27b);
        assert_eq!(exe.at(0x13f2af, 7), [0x83, 0xB8, 0x30, 0x25, 0x01, 0x00, 0x00]);
        assert_eq!(exe.at(0x13f351, 18), [0x48, 0x83, 0xC4, 0x20, 0x5B, 0x5D, 0x5F, 0x5E, 0x41, 0x5C, 0x41, 0x5E, 0x41, 0x5F, 0xC5, 0xF8, 0x77, 0xC3]);
        // title+0x55e6d: mov [r14+0x1170], rcx; the gate; test al, al; jne title+0x56056, which is
        // lea rax, [r14+0x1170]; mov qword [rax], 0; add rsp, 48h; pop rbx; pop rdi; pop rsi; pop r14; ret.
        assert_eq!(exe.at(0x55e6d, 7), [0x49, 0x89, 0x8E, 0x70, 0x11, 0x00, 0x00]);
        assert_eq!(exe.at(0x55e79, 8), [0x84, 0xC0, 0x0F, 0x85, 0xD5, 0x01, 0x00, 0x00]);
        assert_eq!(0x55e81 + 0x1d5, 0x56056);
        assert_eq!(exe.at(0x56056, 24), [0x49, 0x8D, 0x86, 0x70, 0x11, 0x00, 0x00, 0x48, 0xC7, 0x00, 0, 0, 0, 0, 0x48, 0x83, 0xC4, 0x48, 0x5B, 0x5F, 0x5E, 0x41, 0x5E, 0xC3]);
        // title+0x2ec493: mov [r15+0x1170], rcx; the gate; test al, al; jne title+0x2ec684, the same exit
        // with r15 and six pops.
        assert_eq!(exe.at(0x2ec493, 7), [0x49, 0x89, 0x8F, 0x70, 0x11, 0x00, 0x00]);
        assert_eq!(exe.at(0x2ec49f, 8), [0x84, 0xC0, 0x0F, 0x85, 0xDD, 0x01, 0x00, 0x00]);
        assert_eq!(0x2ec4a7 + 0x1dd, 0x2ec684);
        assert_eq!(exe.at(0x2ec684, 27), [0x49, 0x8D, 0x87, 0x70, 0x11, 0x00, 0x00, 0x48, 0xC7, 0x00, 0, 0, 0, 0, 0x48, 0x83, 0xC4, 0x58, 0x5B, 0x5D, 0x5F, 0x5E, 0x41, 0x5E, 0x41, 0x5F, 0xC3]);
        // The hook's reads: the context at TLS+0x12A8 (mov rdi, [rax+12A8h] in the indexed path) and the
        // program set at context +0x12328 (the meshlet draw title+0x43c5b80 loads it: mov r14, [rsi+12328h]).
        assert_eq!(exe.at(0x55fbd, 7), [0x48, 0x8B, 0xB8, 0xA8, 0x12, 0x00, 0x00]);
        let meshlet = exe.at(0x43c5b80, 361);
        assert!(meshlet.windows(4).any(|w| w == [0x28, 0x23, 0x01, 0x00]), "context +0x12328 in the meshlet draw");
        // The variant byte beside it: cmp byte [rsi+12743h], 0Ch.
        assert!(meshlet.windows(7).any(|w| w == [0x80, 0xBE, 0x43, 0x27, 0x01, 0x00, 0x0C]), "context +0x12743 in the meshlet draw");
        // The material bind takes (shader instance, context, pass record): its body tests [rcx+30h].
        assert_eq!(exe.at(0x4da8bc0, 5), [0x48, 0x83, 0x79, 0x30, 0x00]);
    }

    /// ENG9: the per-geometry delegate both geometry draws call below the gate, as `keep_delegate` calls
    /// it; what it reaches (the stencil reference our hook reads); the mesh program's data slot handle.
    #[test]
    fn the_delegate_and_the_slot_handles_are_the_executables() {
        let Some(exe) = crate::patches::Exe::open() else {
            eprintln!("no title executable here (XNW_TITLE_EXE): the delegate sites were not checked against it");
            return;
        };
        // Both draws: mov rax, [rax+1148h] / +1150h / +1160h and mov eax, [rax+1168h] compared with a
        // cleared delegate; then xor edx, edx; mov rax, [rax]; lea rcx, [rax+1148h]; call title+0x52db2e0.
        for (cmp, call) in [(0x55ede, 0x55f6c), (0x2ec504, 0x2ec592)] {
            for (at, field) in [(0, 0x48u8), (0x18, 0x50), (0x32, 0x60)] {
                assert_eq!(exe.at(cmp + at, 7), [0x48, 0x8B, 0x80, field, 0x11, 0x00, 0x00], "delegate word {field:#x} before title+{call:#x}");
            }
            assert_eq!(exe.at(cmp + 0x4c, 6), [0x8B, 0x80, 0x68, 0x11, 0x00, 0x00]);
            assert_eq!(exe.at(call - 12, 12), [0x31, 0xD2, 0x48, 0x8B, 0x00, 0x48, 0x8D, 0x88, 0x48, 0x11, 0x00, 0x00]);
            assert_eq!((exe.at(call, 1)[0], exe.rel32_target(call)), (0xE8, DELEGATE_INVOKE));
        }
        assert_eq!(TLS_DELEGATE, 0x1148);
        // The cleared delegate (title+0x4e72fd0) is 40 zero bytes: mov qword [rcx+20h], 0; vmovups [rcx], ymm0.
        assert_eq!(exe.at(0x4e72fd3, 16), [0xC5, 0xF8, 0x57, 0xC0, 0x48, 0xC7, 0x41, 0x20, 0, 0, 0, 0, 0xC5, 0xFC, 0x11, 0x01]);
        // The invoke leaves at once on kind 0: mov r8, rdx; mov edx, [rcx+20h]; test edx, edx; jz.
        assert_eq!(exe.at(DELEGATE_INVOKE, 9), [0x49, 0x89, 0xD0, 0x8B, 0x51, 0x20, 0x85, 0xD2, 0x74]);
        // The delegate three drawable classes install (title+0x4ef140) ends in a jump to the stencil
        // reference setter title+0x4e55390, which stores context +0x12368: mov [rax+12368h], esi.
        assert_eq!((exe.at(0x4ef216, 1)[0], exe.rel32_target(0x4ef216)), (0xE9, 0x4e5_5390));
        assert_eq!(exe.at(0x4e5548f, 6), [0x89, 0xB0, 0x68, 0x23, 0x01, 0x00]);
        // The kind step's mesh data slot: title+0x555ee79 movzx edx, word [rip+..] reads the handle.
        assert_eq!(exe.at(0x555ee79, 3), [0x0F, 0xB7, 0x15]);
        let rel = i32::from_le_bytes(exe.at(0x555ee7c, 4).try_into().unwrap());
        assert_eq!((0x555ee80i64 + i64::from(rel)) as usize, MESH_SLOT_HANDLE);
        assert!(!RECORD_SLOT_HANDLES.contains(&MATRIX_SLOT_HANDLE) && !CULL_SLOT_HANDLES.contains(&MATRIX_SLOT_HANDLE) && MESH_SLOT_HANDLE != MATRIX_SLOT_HANDLE);
        // The matrix setter title+0x4e3d960: lea rdi, [cache]; its index is context +0x12190 (movsxd r??,
        // [rcx+12190h]); it binds the slot of the handle at title+0x75eacb8 with the lazy size 128
        // (mov word [.. + 6200h], 80h).
        let setter = exe.at(0x4e3_d960, 0x180);
        let rip = |i: usize, len: usize| (0x4e3_d960 + i as i64 + len as i64 + i64::from(i32::from_le_bytes(setter[i + len - 4..i + len].try_into().unwrap()))) as usize;
        assert!((0..setter.len() - 7).any(|i| setter[i] & 0xFB == 0x48 && setter[i + 1] == 0x8D && setter[i + 2] & 0xC7 == 0x05 && rip(i, 7) == MATRIX_CACHE), "the cache");
        assert!((0..setter.len() - 8).any(|i| setter[i + 1..i + 3] == [0x0F, 0xB7] && setter[i + 3] & 0xC7 == 0x05 && rip(i, 8) == MATRIX_SLOT_HANDLE) || (0..setter.len() - 7).any(|i| setter[i..i + 2] == [0x0F, 0xB7] && setter[i + 2] & 0xC7 == 0x05 && rip(i, 7) == MATRIX_SLOT_HANDLE), "the matrix handle");
        assert!(setter.windows(4).any(|w| w == [0x90, 0x21, 0x01, 0x00]), "context +0x12190");
        assert!(setter.windows(6).any(|w| w == [0x00, 0x62, 0x00, 0x00, 0x80, 0x00]), "the lazy size 128 at context +0x6200");
        // The table fill title+0x4d9efd0 (binding flush) clears a slot's lazy size and stores the view:
        // it addresses both from the list wrapper (context +0x12780): -50560 = +0x6200, -75584 = +0x40.
        assert_eq!((0x1_2780 - 50560, 0x1_2780 - 75584), (0x6200, 0x40));
        let fill = exe.at(0x4d9_efd0, 0x200);
        assert!(fill.windows(4).any(|w| w == (-50560i32).to_le_bytes()) && fill.windows(4).any(|w| w == (-75584i32).to_le_bytes()), "the table fill's slot and lazy stores");
        // The cull dispatch title+0x325c500 reads its two handles: movzx r32, word [rip+..].
        let cull = exe.at(0x325_c500, 0xd98);
        for handle in CULL_SLOT_HANDLES {
            let found = (0..cull.len() - 7).any(|i| {
                cull[i..i + 2] == [0x0F, 0xB7] && cull[i + 2] & 0xC7 == 0x05 && (0x325_c500 + i as i64 + 7 + i64::from(i32::from_le_bytes(cull[i + 3..i + 7].try_into().unwrap()))) as usize == handle
            });
            assert!(found, "title+0x325c500 reads the handle at title+{handle:#x}");
        }
        // The pre-draw callback's record binds: movzx r9d, word [rip+..] of each handle in title+0x4e308e0,
        // then title+0x4e31090, which stores context +0x40 + 8 * slot and +0x6200 + 2 * slot.
        let upload = exe.at(0x4e3_08e0, 0x7a3);
        for handle in RECORD_SLOT_HANDLES {
            let found = (0..upload.len() - 8).any(|i| {
                upload[i..i + 4] == [0x44, 0x0F, 0xB7, 0x0D] && (0x4e3_08e0 + i as i64 + 8 + i64::from(i32::from_le_bytes(upload[i + 4..i + 8].try_into().unwrap()))) as usize == handle
            });
            assert!(found, "title+0x4e308e0 reads the handle at title+{handle:#x}");
        }
        let bind = exe.at(0x4e3_1090, 0x60);
        assert!(bind.windows(4).any(|w| w == [0x00, 0x62, 0x00, 0x00]), "title+0x4e31090 stores the lazy size at context +0x6200");
    }

    static RAN: AtomicU64 = AtomicU64::new(0);
    extern "system" fn prestep(a: usize, b: usize, c: usize, d: usize, e: usize) -> usize {
        RAN.fetch_add(1, Ordering::Relaxed);
        assert_eq!((a, b, c, d, e), (11, 12, 13, 14, 15), "the five arguments reach the title's function");
        0
    }
    /// A draw row's function: the list sink answers inside it.
    static SINK: AtomicU64 = AtomicU64::new(0);
    extern "system" fn draw(_: usize, _: usize, _: usize, _: usize, _: usize) -> usize {
        RAN.fetch_add(0x100, Ordering::Relaxed);
        match SINK.load(Ordering::Relaxed) {
            1 => learn(false),
            2 => learn(true),
            _ => {}
        }
        7
    }

    /// The world row's function in the check: it changes the state bytes the hook reads.
    extern "system" fn changes_state(_: usize, _: usize, _: usize, _: usize, _: usize) -> usize {
        TEST_INPUTS.with(|i| i.set([0, 0, 0, 7, 0, 0, 0]));
        0
    }

    /// The gate and a draw row as at install, driven through their hooks (one test: the statics are shared).
    #[test]
    fn the_gate_leaves_out_only_draws_no_consumer_wants() {
        let (gate, row) = (0, 4);
        assert_eq!((SITES[gate].kind, SITES[row].kind), (Kind::Gate(0), Kind::Draw));
        // A call site made in memory, redirected as at install: it runs the function and is counted.
        // SAFETY: a fresh executable allocation of ours.
        let code = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) }.cast::<u8>();
        assert!(!code.is_null());
        let page = crate::titlemeter::near_page(code as usize, code as usize + 4096).expect("a page near our code");
        // sub rsp, 38h; mov qword [rsp+20h], 15; call code+0x100; add rsp, 38h; ret. At code+0x100: jmp [rip+0] to prestep.
        let mut site = vec![0x48, 0x83, 0xEC, 0x38, 0x48, 0xC7, 0x44, 0x24, 0x20, 15, 0, 0, 0, 0xE8];
        site.extend((0x100i32 - 18).to_le_bytes());
        site.extend([0x48, 0x83, 0xC4, 0x38, 0xC3]);
        let mut thunk = vec![0xFF, 0x25, 0, 0, 0, 0];
        thunk.extend((prestep as usize as u64).to_le_bytes());
        // SAFETY: inside the allocation; the function just written, of this type; our page.
        unsafe {
            std::ptr::copy_nonoverlapping(site.as_ptr(), code, site.len());
            std::ptr::copy_nonoverlapping(thunk.as_ptr(), code.add(0x100), thunk.len());
            let f: extern "system" fn(usize, usize, usize, usize) -> usize = std::mem::transmute(code);
            assert_eq!(f(11, 12, 13, 14), 0);
            TARGETS[gate].store(code as usize + 0x100, Ordering::Relaxed);
            assert_eq!(crate::titlemeter::redirect(code.add(13), code as usize + 0x200, page, hook_table()[gate]), Err("not the expected call"));
            crate::titlemeter::redirect(code.add(13), code as usize + 0x100, page, hook_table()[gate]).unwrap();
            assert_eq!(f(11, 12, 13, 14), 0, "the gate off: the prestep runs through the hook");
        }
        assert_eq!(RAN.swap(0, Ordering::Relaxed), 2);
        TARGETS[gate].store(prestep as usize, Ordering::Relaxed);
        TARGETS[row].store(draw as usize, Ordering::Relaxed);
        let hooks = hook_table();
        // SAFETY: two of the hooks made above, of this type.
        let (g, d): (extern "system" fn(usize, usize, usize, usize, usize) -> usize, extern "system" fn(usize, usize, usize, usize, usize) -> usize) = unsafe { (std::mem::transmute(hooks[gate]), std::mem::transmute(hooks[row])) };
        let one = |key: u64, sink: u64| -> (usize, u64) {
            TEST_KEY.with(|k| k.set(key));
            SINK.store(sink, Ordering::Relaxed);
            let left = g(11, 12, 13, 14, 15);
            if left == 0 {
                assert_eq!(d(1, 2, 3, 4, 5), 7);
            }
            (left, RAN.swap(0, Ordering::Relaxed))
        };
        let (a, b, c) = (mix([0x1000, 1, 2, 3, 8, 5]), mix([0x2000, 1, 2, 3, 8, 5]), mix([0x1000, 1, 2, 3, 9, 5]));
        assert!(a != b && a != c && a & 3 == 0 && a != 0);
        // Off: everything runs, nothing is classed.
        assert_eq!(one(a, 1), (0, 0x101));
        assert_eq!(class_of(a), 0);
        SKIP[0].store(true, Ordering::Relaxed);
        // Not under null: encoded, not classed.
        TEST_READERS[2].store(true, Ordering::Relaxed);
        assert_eq!(one(a, 1), (0, 0x101));
        assert_eq!(class_of(a), 0);
        TEST_READERS[0].store(true, Ordering::Relaxed);
        // Null, a consumer registered: the first draw of a set is encoded and classed at the list.
        assert_eq!(one(a, 1), (0, 0x101));
        assert_eq!(class_of(a), UNWANTED);
        assert_eq!(one(a, 1), (1, 0), "left out: neither the prestep nor the draw ran");
        // A set a consumer wants stays encoded, also after a draw of it that no consumer wanted.
        assert_eq!(one(b, 2), (0, 0x101));
        assert_eq!(one(b, 1), (0, 0x101));
        assert_eq!(class_of(b), WANTED);
        // A draw that never reaches the list stays unclassed and encoded; no program set: encoded.
        assert_eq!(one(c, 0), (0, 0x101));
        assert_eq!((class_of(c), one(c, 0)), (0, (0, 0x101)));
        assert_eq!(one(0, 1), (0, 0x101));
        // A list draw outside a gated draw classes nothing.
        learn(true);
        assert_eq!(class_of(a), UNWANTED);
        // The frame pack's tap: nothing is left out.
        TEST_READERS[1].store(true, Ordering::Relaxed);
        assert_eq!(one(a, 1), (0, 0x101));
        TEST_READERS[1].store(false, Ordering::Relaxed);
        // The instance batches' gates: off until their own switch; then the same rule, with each site's answer.
        let (bg, bg2) = (12, 18);
        TARGETS[bg].store(prestep as usize, Ordering::Relaxed);
        TARGETS[bg2].store(prestep as usize, Ordering::Relaxed);
        // SAFETY: two more of the hooks, of this type.
        let (m, k): (extern "system" fn(usize, usize, usize, usize, usize) -> usize, extern "system" fn(usize, usize, usize, usize, usize) -> usize) = unsafe { (std::mem::transmute(hooks[bg]), std::mem::transmute(hooks[bg2])) };
        TEST_KEY.with(|k| k.set(a));
        assert_eq!((m(11, 12, 13, 14, 15), k(11, 12, 13, 14, 15), RAN.swap(0, Ordering::Relaxed)), (0, 0, 2), "their switch is off: the title's functions run");
        SKIP[1].store(true, Ordering::Relaxed);
        assert_eq!((m(11, 12, 13, 14, 15), RAN.swap(0, Ordering::Relaxed)), (1, 0), "the mode test answers 1");
        TEST_KEY.with(|k| k.set(b));
        assert_eq!((m(11, 12, 13, 14, 15), RAN.swap(0, Ordering::Relaxed)), (0, 1), "a wanted set: asked");
        // The kind answers 0 for a set no consumer wants; the title's own answer (here 0 too) is asked for a wanted one.
        TEST_KEY.with(|k| k.set(a));
        assert_eq!((k(11, 12, 13, 14, 15), RAN.swap(0, Ordering::Relaxed)), (0, 0));
        TEST_KEY.with(|k| k.set(b));
        assert_eq!((k(11, 12, 13, 14, 15), RAN.swap(0, Ordering::Relaxed)), (0, 1));
        SKIP[1].store(false, Ordering::Relaxed);
        let line = take_gate_line(1);
        assert!(line.contains("geometry draws left out 1.0, encoded 8.0 (for a list consumer 1.0, program set not classed yet 4.0, other 3.0) (per-geometry delegate still called 1.0); instance batches left out 2.0, encoded 2.0 (for a list consumer 2.0,") && line.ends_with("wanted 1, not wanted 1"), "{line}");
        // No consumer at all: every draw is left out, classed or not.
        TEST_READERS[2].store(false, Ordering::Relaxed);
        assert_eq!((one(b, 2), one(c, 0), one(0, 0)), ((1, 0), (1, 0), (1, 0)));
        // ENG9: each left-out geometry draw passed its delegate on (these three; the gate line above took
        // the one before); a left-out batch has none.
        assert_eq!(DELEGATES.swap(0, Ordering::Relaxed), 3);
        SKIP[1].store(true, Ordering::Relaxed);
        TEST_KEY.with(|k| k.set(a));
        assert_eq!((m(11, 12, 13, 14, 15), DELEGATES.load(Ordering::Relaxed)), (1, 0));
        SKIP[1].store(false, Ordering::Relaxed);
        for n in &GATE_N[1] {
            n.store(0, Ordering::Relaxed);
        }
        // The slot guard: the mesh data slot on the hook's matrix slot keeps every draw encoded at once,
        // and turns the gates off at the window's end.
        TEST_SLOTS[0].store(5, Ordering::Relaxed);
        TEST_SLOTS[1].store(5, Ordering::Relaxed);
        assert!(slots_same());
        assert_eq!(one(b, 2), (0, 0x101));
        window_end(1, "");
        assert!(!SKIP[0].load(Ordering::Relaxed), "one slot for both: the gate is off");
        TEST_SLOTS[0].store(9, Ordering::Relaxed);
        SLOTS_SAID.store(false, Ordering::Relaxed);
        SKIP[0].store(true, Ordering::Relaxed);
        assert_eq!(one(b, 2), (1, 0));
        window_end(1, "");
        assert!(SKIP[0].load(Ordering::Relaxed) && SLOTS_SAID.load(Ordering::Relaxed));
        FRAMES.store(0, Ordering::Relaxed);
        DELEGATES.store(0, Ordering::Relaxed);
        for n in &GATE_N[0] {
            n.store(0, Ordering::Relaxed);
        }
        // The check: a row inside which a hook input changes is named with the field.
        SKIP[0].store(false, Ordering::Relaxed);
        CHECK.store(true, Ordering::Relaxed);
        TICK.with(|t| t.set(0));
        TARGETS[3].store(changes_state as usize, Ordering::Relaxed);
        // SAFETY: one more of the hooks, of this type.
        let w: extern "system" fn(usize, usize, usize, usize, usize) -> usize = unsafe { std::mem::transmute(hooks[3]) };
        assert_eq!(w(0, 0, 0, 0, 0), 0);
        TICK.with(|t| t.set(0));
        assert_eq!(g(11, 12, 13, 14, 15), 0);
        CHECK.store(false, Ordering::Relaxed);
        TEST_INPUTS.with(|i| i.set([0; 7]));
        RAN.store(0, Ordering::Relaxed);
        let line = take_check_line(60);
        assert_eq!(line, "title encode check: 60 frames, 2 calls sampled; hook inputs changed inside: geometry.world (state 1 of 1)");
        assert!(take_check_line(60).ends_with("0 calls sampled; hook inputs changed inside: none"));
        SKIP[0].store(true, Ordering::Relaxed);
        // The meter: rows by group, the encode sum by stage, sums starting again; the classes are forgotten
        // when the frame count passes a multiple of FORGET_EVERY.
        SKIP[0].store(false, Ordering::Relaxed);
        TIME.store(true, Ordering::Relaxed);
        crate::stagemeter::STAGE.with(|s| s.set(67));
        assert_eq!((one(a, 0), d(1, 2, 3, 4, 5)), ((0, 0x101), 7));
        crate::stagemeter::STAGE.with(|s| s.set(crate::stagemeter::NO_STAGE));
        TIME.store(false, Ordering::Relaxed);
        let line = take_line(1);
        assert!(line.contains("below the draw hook: geometry.prestep ") && line.contains(" (1.0), geometry.meshlet ") && line.contains(" (2.0); all "), "{line}");
        assert!(line.contains("; by stage: 67 ") && line.ends_with("; inside them: none; above the hook: none"), "{line}");
        assert!(take_line(1).contains("below the draw hook: none; all 0.00; by stage: none"), "the sums start again");
        SKIP[0].store(true, Ordering::Relaxed);
        window_end(FORGET_EVERY - 1, "");
        assert_eq!(class_of(a), UNWANTED);
        window_end(1, "");
        assert_eq!((class_of(a), class_of(b)), (0, 0));
        SKIP[0].store(false, Ordering::Relaxed);
        assert_eq!((named("watchdog, NoEncode"), named("noencodex,noencodebatch"), named("")), ([true, false], [false, true], [false, false]));
        // The switch that comes later needs the hooks in place.
        assert!(skip_live(&["noencode".to_owned()]).is_empty());
        GATED.store(true, Ordering::Relaxed);
        assert_eq!(skip_live(&["nolighting".to_owned(), "noencodebatch".to_owned()]), ["noencodebatch"]);
        assert_eq!((SKIP[0].load(Ordering::Relaxed), SKIP[1].swap(false, Ordering::Relaxed)), (false, true));
        GATED.store(false, Ordering::Relaxed);
    }
}
