//! INT1 (`vehmat=on`, default on with `XNW_WORLD_RENDER`): a vehicle draw's
//! material colour table, read at its list sink from the title's own GPU
//! records (xnw_world::vehmat has the layout and the evidence):
//! Root 1: slot 8 (slot 7 for the deform PS), the vehicle's remap (256 u32 slots, base at +0x400),
//! and T8 (root 0) slot 42, the 76-byte material elements the CPU writer
//! the function fills. Asked only for the sink that follows a hook draw of
//! a vehicle-body effect (WHL1: or a wheel or tyre) on the same thread (`set_pending`), so other draws
//! cost nothing. Tables are kept per (frame, remap view, record view): the
//! draws of one vehicle share them.
//!
//! PAL1: slots whose record names a palette row take it from the draw's
//! palette, draw word 8 (`s14 + 32` of the T8+0xAC0 words [V a8000c5c
//! 0x098, 0x1cc, 0x26a8]; pk8a: heap 35700, a 128 x 256 x 8 sRGB array):
//! its texels from title memory (decoded as a pack's stored texels), or in
//! a burst from the frame pack's provider (the ttpf origin, else stored
//! texels; `set_texels`). Palettes are kept by view (address, bytes,
//! format, size) for `PAL_FRAMES` frames.
//!
//! POL2 (`vehlivery=on`, default off): the livery of PS_VehicleLiveryOn*
//! (xnw_world::vehmat::Livery has the rules and the evidence): the draw's
//! T8+0xA00 record (index from T9+0x40 and T8+0xB20, 0x62c bytes each: its
//! no-livery mask at +0x36c, the record base at +0x3a8), T9+0x60 (root 1 slot 3, +12 of the first 256
//! elements: the layer bytes by u), the T8+0x5c0 records (root 0 slot 46)
//! they name and their mask textures (heap index -> descriptor slot, at
//! most two a draw, given to the pass as [`Block::livery`]).

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use xnw_formats::framepack::view;
use xnw_gpu::d3d12x::tap::{self, TapDraw, TapRoot};
use xnw_world::vehmat as vm;

/// T8+0x540 (root 0 slot 42) and T9+0x100 (root 1 slot 8) [V PS a8000c5c 0x40, 0x7fc].
/// The deform variant PS_VehicleLiveryOffOnlyDeferred_7095f810 (9aa8da02) reads the
/// same remap at T9+0xe0 (root 1 slot 7: its table has one view fewer before it)
/// [V disasm 0x808 `s_buffer_load_dword T9+0xe0 +0x400`; pk8a f1030 row 238: the
/// 1040-B buffer 0x9888e6320 at slot 7, row 214 (a8000c5c) has it at slot 8] (INT2).
/// WHL1: the wheel and tyre programs read it at their own slot, the record at
/// the same T8+0x540: e668ad56 (`_wheel_inst`, NoDeform) T9+0x160 (slot 11),
/// d68d7c01 (`_tyre_inst`) T9+0x180 (slot 12), a2c80c6f (`_wheel_inst`, deform)
/// T9+0x100 [V disasm `s_buffer_load_dword T9+X#0 +0x400`: e668ad56 0x920,
/// d68d7c01 0xb34, a2c80c6f 0x920]. The first remap-sized buffer in this order
/// is each program's own slot on all 152 stage-67 livery sinks of pk8a
/// f1027-1032 [V refs.bin; earlier slots hold 16-512 B there].
const RECORD_ROOT: usize = 0;
const RECORD_SLOT: u32 = 0x540 / 32;
const REMAP_ROOT: usize = 1;
const REMAP_SLOTS: [u32; 4] = [0x100 / 32, 0xe0 / 32, 0x160 / 32, 0x180 / 32];

/// A draw's block: 256 entries, then its palette rows (xnw_world::vehmat::table),
/// then (POL2) its livery section.
pub struct Block {
    pub words: Vec<u32>,
    /// The livery section's dword index in `words` (0: none).
    pub livery_at: u32,
    /// Its mask textures' descriptor slots (0: none).
    pub livery: [u64; 2],
    /// NPC1: a far ped's block (`slodtint.rs`; 0: a vehicle's), as `Material::slod`.
    pub slod: u8,
}
pub type Table = Arc<Block>;

/// T8+0xA00 (root 0 slot 80), T8+0x5c0 (root 0 slot 46), T9+0x60 (root 1
/// slot 3), T9+0x40 (root 1 slot 2), T8+0xB20 (root 0 slot 89) [V PS
/// 05483074 0x29c, 0x2b4, 0xbc, 0x40, 0x48; MS 42c87283 0xd48].
const LIVERY_DRAW_SLOT: u32 = 0xA00 / 32;
const LIVERY_J_SLOT: u32 = 0x40 / 32;
const LIVERY_K_SLOT: u32 = 0xB20 / 32;
const LIVERY_RECORD_SLOT: u32 = 0x5C0 / 32;
const LIVERY_LAYER_SLOT: u32 = 0x60 / 32;

thread_local! {
    /// The last hook draw on this thread was a vehicle-body draw.
    static PENDING: Cell<bool> = const { Cell::new(false) };
    /// POL2: pipeline object -> its pixel program is a PS_VehicleLiveryOn* one.
    static LIVERY_PIPES: std::cell::RefCell<HashMap<usize, bool>> = std::cell::RefCell::new(HashMap::new());
}

/// POL2: whether the draw's pixel program is one of the 55
/// PS_VehicleLiveryOn* programs (xnw_world::bridge::LIVERY_PS, by the
/// container hash's first 32 bits): only those read the livery records;
/// other vehicle programs' T8+0xA00 words at +0x3a8 are other data [V pk8nort
/// f1027: 0x3c041e93 there on a LiveryOff row].
fn livery_program(pipeline: usize) -> bool {
    LIVERY_PIPES.with(|m| {
        let mut m = m.borrow_mut();
        if let Some(&k) = m.get(&pipeline) {
            return k;
        }
        if m.len() > 4096 {
            m.clear();
        }
        let k = tap::stages(pipeline).iter().any(|s| s.name == "pixel" && s.container && xnw_world::bridge::LIVERY_PS.contains(&((s.hash >> 96) as u32)));
        m.insert(pipeline, k);
        k
    })
}

/// At the hook: whether the next sink on this thread is asked.
pub(super) fn set_pending(on: bool) {
    PENDING.with(|p| p.set(on));
}

pub(super) fn pending() -> bool {
    PENDING.with(Cell::get)
}

/// `XNW_WORLD_RENDER` set and not `vehmat=off` (read once).
pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| vm::parse_switch(&v)))
}

/// `vehlivery=on` (read once).
pub(super) fn livery_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| on() && std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| vm::parse_livery(&v)))
}

/// POL2: tables asked a livery, then by outcome: no record index, no T8+0xA00
/// view, record base 0, no T9+0x60 view, no T8+0x5c0 view, no layer word
/// names a record, sections made; records in them, masks given, masks the
/// two fields could not take, records unreadable; POL3: sections with a UV
/// transform (the mesh program's s44 >= 0); POL4: transforms refused as
/// collapsed (vm::uvx_ok: taken as identity).
pub(super) static LV: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];
const LV_NAMES: [&str; 14] = ["asked", "no draw index", "no draw view", "base 0", "no layer view", "no record view", "no layer", "sections", "records", "masks", "masks lost", "records unreadable", "uv transforms", "uv transforms refused"];

/// POL4: one section's inputs as read, for the frame line's `livery
/// detail` (the first section made since the last line): CB b4 +0xb4..+0xbc
/// (s44 s45 s46), the transform, its mask views.
static DETAIL: Mutex<Option<String>> = Mutex::new(None);

/// A mask's view as the title described it: kind, the resource's console
/// format, the view's format and component mapping (words 14, 16), size,
/// mips, the view's MostDetailedMip and MipLevels (words 18, 19); POL5: the
/// layout, the texel address and its first [`MASK_BYTES_READ`] bytes in title
/// memory (vm::mask_bytes_text: blocks not zero, the first, its BC7 texels
/// and the weight they give the records of `section` whose mask is `field`).
fn mask_view_text(h: u64, section: &[u32], records: u32, field: u32) -> String {
    // The slot's record first (fault-safe), as `palette_view`.
    if super::dcopy(h, tap::SLOT as usize).is_none() {
        return "unreadable".into();
    }
    let v = tap::view(h);
    let n = v.bytes.min(MASK_BYTES_READ) as usize;
    let mem = if v.addr == 0 || n == 0 { "no texel address".to_string() } else { super::dcopy(v.addr, n).map_or_else(|| "texels unreadable".to_string(), |b| vm::mask_bytes_text(&b, matches!(v.format, 97..=99), section, records, field)) };
    format!("kind {} format {} view format {} mapping 0x{:x} {}x{} mips {} view mips {}+{} layout {} at 0x{:x} {mem}", v.kind, v.format, v.words[14], v.words[16], v.width, v.height, v.mips, v.words[18], v.words[19], v.layout, v.addr)
}

/// POL5: a mask's texel bytes read for the `livery detail` text at most
/// (once per frame line; a 4x4 BC7 is one 16-byte block in a 64 KiB tile at most).
const MASK_BYTES_READ: u64 = 64 << 10;

/// `vehmat=mul` (read once).
fn mul_on() -> bool {
    static MUL: OnceLock<bool> = OnceLock::new();
    *MUL.get_or_init(|| std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| vm::parse_mul(&v)))
}

/// Sinks since the last line by outcome; PAL1: tables with palette rows,
/// tables whose palette rows got no texels (and why: no palette view, no texels).
pub(super) static WHY: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];
pub(super) const NAMES: [&str; 9] = ["asked", "no remap view", "no record view", "unreadable", "tables made", "given", "with palette rows", "palette rows without texels", "no palette view"];

pub(super) fn take() -> [u64; 9] {
    std::array::from_fn(|i| WHY[i].swap(0, Ordering::Relaxed))
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

type Key = (u64, u64, u64, u32);
static CACHE: Mutex<Option<(u64, HashMap<Key, Option<Table>>)>> = Mutex::new(None);

pub(super) fn table_view(d: &TapDraw, root: usize, slot: u32) -> Option<tap::TapView> {
    match d.roots.get(root) {
        Some(TapRoot::Table { handle, slots, .. }) if slot < *slots => Some(tap::view(handle + tap::SLOT * u64::from(slot))).filter(|v| v.kind != view::NONE),
        _ => None,
    }
}

/// The draw words' palette entry (word 8) [V a8000c5c 0x098 s26 = word at
/// s14 + 32, 0x1cc s86 = s26 << 5, 0x26a8 the palette's descriptor at s86].
const PALETTE_WORD: usize = 8;
/// Frames a palette read from title memory is kept (live; a burst's frames
/// are other captures of the same texture).
const PAL_FRAMES: u64 = 600;
/// A palette's bytes in console memory at most (pk8a's: 1 MiB).
const PAL_BYTES_MOST: u64 = 16 << 20;

/// The burst's texel source for this frame (the frame pack's provider).
static TEXELS: Mutex<Option<xnw_world::textures::Provider>> = Mutex::new(None);

/// A burst gives the frame pack's texel provider (None: none).
pub(super) fn set_texels(p: Option<xnw_world::textures::Provider>) {
    *lock(&TEXELS) = p;
}

type PalKey = (u64, u64, u32, u64, u32, u16);
static PALETTES: Mutex<Option<HashMap<PalKey, (u64, Option<Arc<vm::Palette>>)>>> = Mutex::new(None);

/// The draw's palette descriptor (heap slot handle) and view: draw word 8
/// of the T8+0xAC0 words (root 0 slot 86 at root 4's constant dword 1, as
/// the bridge reads them) in the draw's view heap.
fn palette_view(d: &TapDraw) -> Option<(u64, tap::TapView)> {
    let (_, h) = tap::heap_slot(d.view_heap, draw_word(d, PALETTE_WORD)?)?;
    // The slot's record first (fault-safe): `tap::view` reads it in place.
    super::dcopy(h, tap::SLOT as usize)?;
    Some((h, tap::view(h))).filter(|(_, v)| matches!(v.kind, view::TEXTURE | view::PLACED) && v.bytes != 0 && v.bytes <= PAL_BYTES_MOST)
}

/// Draw word `k` of the T8+0xAC0 words (root 0 slot 86 at root 4's
/// constant dword 1, as the bridge reads them).
fn draw_word(d: &TapDraw, k: usize) -> Option<u32> {
    use xnw_world::bridged::{WORDS, WORDS_DWORD, WORDS_ROOT, WORDS_SLOT};
    let v = table_view(d, 0, WORDS_SLOT).filter(|v| matches!(v.kind, view::BUFFER | view::PLACED_BUFFER | view::CONSTANT))?;
    let off = match d.roots.get(WORDS_ROOT) {
        Some(TapRoot::Constants(c)) => u64::from(*c.get(WORDS_DWORD)?),
        _ => return None,
    };
    if off + 4 * WORDS as u64 > v.bytes || k >= WORDS {
        return None;
    }
    let w = super::dcopy(v.addr + off + 4 * k as u64, 4)?;
    Some(u32::from_le_bytes(w.try_into().ok()?))
}

fn lv(i: usize) {
    LV[i].fetch_add(1, Ordering::Relaxed);
}

/// POL2: the livery inputs of a vehicle draw (None: none, counted in [`LV`]):
/// the T8+0xA00 record's mask and base, T9+0x60's layer words, the
/// T8+0x5c0 view; POL3: the mesh program's UV transform (None: identity);
/// POL4: CB b4 +0xb4..+0xbc and the transform as read (before vm::uvx_ok).
type LiveryInputs = (u32, [u32; 8], Vec<u32>, tap::TapView, Option<[f32; 6]>, (Option<[u32; 3]>, Option<[f32; 6]>));
fn livery_inputs(d: &TapDraw) -> Option<LiveryInputs> {
    let any = |v: &tap::TapView| matches!(v.kind, view::BUFFER | view::PLACED_BUFFER | view::CONSTANT);
    lv(0);
    let rd = |v: &tap::TapView, off: u64, n: usize| (off + n as u64 <= v.bytes).then_some(()).and_then(|_| super::dcopy(v.addr + off, n));
    let words = |b: Vec<u8>| b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<u32>>();
    let dword = |root: usize, slot: u32, off: u64| table_view(d, root, slot).filter(any).and_then(|v| rd(&v, off, 4)).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let j = dword(REMAP_ROOT, LIVERY_J_SLOT, vm::LIVERY_J_OFF);
    let Some(k) = j.and_then(|j| dword(RECORD_ROOT, LIVERY_K_SLOT, u64::from(j) * vm::LIVERY_K_STRIDE + vm::LIVERY_K_OFF)) else {
        lv(1);
        return None;
    };
    // POL3: CB b4 +0xb4 (s44 s45 s46), the T8+0x600 row r = T8+0xB20[48 j + 4] >> 16.
    let cb = table_view(d, vm::LIVERY_UVX_CB_ROOT, vm::LIVERY_UVX_CB_SLOT).filter(any).and_then(|v| rd(&v, vm::LIVERY_UVX_CB_OFF, 12)).map(words).map(|w| [w[0], w[1], w[2]]);
    let read = (|| {
        let cb = cb?;
        let e = vm::uvx_index(cb[0])?;
        let r = dword(RECORD_ROOT, LIVERY_K_SLOT, u64::from(j?) * vm::LIVERY_K_STRIDE + vm::LIVERY_UVX_ROW_OFF)? >> 16;
        let t = table_view(d, RECORD_ROOT, vm::LIVERY_UVX_TABLE_SLOT).filter(any)?;
        let at = u64::from(r) * vm::LIVERY_UVX_STRIDE + e * vm::LIVERY_UVX_ENTRY;
        let row = |off: u64| rd(&t, off, 12).map(words).map(|w| [f32::from_bits(w[0]), f32::from_bits(w[1]), f32::from_bits(w[2])]);
        Some(vm::uvx(row(at)?, row(at + vm::LIVERY_UVX_V)?, cb[1], cb[2]))
    })();
    // POL4: a collapsed transform is a misread (vm::uvx_ok): identity, counted.
    let uvx = read.filter(|x| {
        let ok = vm::uvx_ok(x);
        if !ok {
            lv(13);
        }
        ok
    });
    let at = u64::from(k >> 16) * vm::LIVERY_DRAW_STRIDE;
    let Some(dv) = table_view(d, RECORD_ROOT, LIVERY_DRAW_SLOT).filter(any) else {
        lv(2);
        return None;
    };
    let base = rd(&dv, at + vm::LIVERY_BASE_OFF, 4).map(words).map_or(0, |w| w[0]);
    if base == 0 {
        lv(3);
        return None;
    }
    let mut off = [0u32; 8];
    if let Some(w) = rd(&dv, at + vm::LIVERY_MASK_OFF, 32).map(words) {
        off.copy_from_slice(&w);
    }
    // The layer elements' stride is the view's (a structured buffer's width).
    let Some(lw) = table_view(d, REMAP_ROOT, LIVERY_LAYER_SLOT).filter(|v| any(v) && v.width >= vm::LIVERY_LAYERS_OFF + 4) else {
        lv(4);
        return None;
    };
    let n = (lw.bytes / lw.width).min(256) as usize;
    let layers = rd(&lw, 0, n * lw.width as usize).map_or_else(Vec::new, |b| (0..n).map(|u| {
        let o = u * lw.width as usize + vm::LIVERY_LAYERS_OFF as usize;
        u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
    }).collect());
    let Some(rv) = table_view(d, RECORD_ROOT, LIVERY_RECORD_SLOT).filter(|v| any(v) && v.width >= 60) else {
        lv(5);
        return None;
    };
    Some((base, off, layers, rv, uvx, (cb, read)))
}

/// The palette's texels: a burst's provider by handle, else title memory.
fn palette_texels(h: u64, v: &tap::TapView) -> Option<vm::Palette> {
    if super::burst::serving() {
        let p = lock(&TEXELS).as_mut().and_then(|f| f(h))?;
        return vm::Palette::from_provided(&p);
    }
    let bytes = super::dcopy(v.addr, v.bytes as usize)?;
    let sv = xnw_formats::framepack::SlotView { slot: 0, kind: v.kind, addr: v.addr, bytes: v.bytes, resource: v.resource, width: v.width, height: v.height, depth: v.depth, mips: v.mips, format: v.format, layout: v.layout, words: v.words };
    vm::Palette::from_provided(&xnw_frame::pack::texels_of(&sv, &bytes)?)
}

/// The draw's palette (None: no palette view, or no texels), kept by view.
fn palette_of(d: &TapDraw, frame: u64) -> Option<Arc<vm::Palette>> {
    let _site = super::fault_site_or_kept(super::SITE_BRIDGE);
    let Some((h, v)) = palette_view(d) else {
        WHY[8].fetch_add(1, Ordering::Relaxed);
        return None;
    };
    let key = (v.addr, v.bytes, v.format, v.width, v.height, v.depth);
    if let Some((at, p)) = lock(&PALETTES).as_ref().and_then(|m| m.get(&key)).cloned()
        && frame.wrapping_sub(at) < PAL_FRAMES
    {
        return p;
    }
    let p = palette_texels(h, &v).map(Arc::new);
    lock(&PALETTES).get_or_insert_with(HashMap::new).insert(key, (frame, p.clone()));
    p
}

/// The table for this sink (None: not a vehicle material draw, or unreadable).
fn table_of(d: &TapDraw, frame: u64) -> Option<Table> {
    WHY[0].fetch_add(1, Ordering::Relaxed);
    let buffer = |v: &tap::TapView| matches!(v.kind, view::BUFFER | view::PLACED_BUFFER);
    let Some(rm) = REMAP_SLOTS.iter().find_map(|&s| table_view(d, REMAP_ROOT, s).filter(|v| buffer(v) && v.bytes >= vm::REMAP_BYTES as u64)) else {
        WHY[1].fetch_add(1, Ordering::Relaxed);
        return None;
    };
    // The record view's stride is the element size (pk8a: width 76, 25,083,648 B).
    let Some(rec) = table_view(d, RECORD_ROOT, RECORD_SLOT).filter(|v| v.kind == view::BUFFER && v.width == vm::ELEMENT && v.bytes >= vm::ELEMENT) else {
        WHY[2].fetch_add(1, Ordering::Relaxed);
        return None;
    };
    let livery = if livery_on() && livery_program(d.pipeline) { livery_inputs(d) } else { None };
    let key = (rm.addr, rec.addr, rec.bytes, livery.as_ref().map_or(0, |l| l.0 ^ l.1.iter().copied().chain(l.4.iter().flatten().map(|f| f.to_bits())).fold(0u32, |a, w| a.rotate_left(5) ^ w)));
    {
        let mut g = lock(&CACHE);
        let c = g.get_or_insert_with(|| (frame, HashMap::new()));
        if c.0 != frame {
            *c = (frame, HashMap::new());
        }
        if let Some(t) = c.1.get(&key) {
            return t.clone();
        }
    }
    let palette = palette_of(d, frame);
    let _site = super::fault_site_or_kept(super::SITE_BRIDGE);
    let made = super::dcopy(rm.addr, vm::REMAP_BYTES).and_then(|b| vm::remap(&b)).map(|(remap, base)| {
        let read = |off: u64| {
            (off + 8 <= rec.bytes).then_some(())?;
            super::dcopy(rec.addr + off, 8).and_then(|b| b.try_into().ok())
        };
        let mut masks = [0u64; 2];
        let (t, n, at) = match &livery {
            Some((lbase, off, layers, rv, uvx, (cb, uvx_read))) => {
                let mut record = |i: u32| {
                    let o = u64::from(i) * rv.width;
                    (o + vm::LIVERY_RECORD as u64 <= rv.bytes).then_some(())?;
                    super::dcopy(rv.addr + o, vm::LIVERY_RECORD).and_then(|b| b.try_into().ok())
                };
                let mut mask = |t: u32| {
                    let (_, h) = tap::heap_slot(d.view_heap, t)?;
                    let k = masks.iter().position(|m| *m == h).or_else(|| masks.iter().position(|m| *m == 0))?;
                    masks[k] = h;
                    Some(k as u32)
                };
                let mut l = vm::Livery { base: *lbase, off: *off, layers, record: &mut record, mask: &mut mask, uvx: *uvx, n: Default::default() };
                let r = vm::table_with(&remap, base, palette.as_deref(), mul_on(), xnw_world::carshade::on(), read, Some(&mut l));
                let c = l.n;
                for (i, v) in [(8, c.records), (10, c.masks_lost), (11, c.unreadable)] {
                    LV[i].fetch_add(u64::from(v), Ordering::Relaxed);
                }
                lv(if r.2.is_some() { 7 } else { 6 });
                if r.2.is_some() && uvx.is_some() {
                    lv(12);
                }
                if r.2.is_some() {
                    let mut g = lock(&DETAIL);
                    if g.is_none() {
                        let cbs = cb.map_or_else(|| "none".to_string(), |c| format!("{} {} {}", f32::from_bits(c[0]), f32::from_bits(c[1]), f32::from_bits(c[2])));
                        let xs = uvx_read.map_or_else(|| "identity".to_string(), |x| format!("u' = {} u + {} v + {}, v' = {} u + {} v + {}{}", x[0], x[1], x[2], x[3], x[4], x[5], if uvx.is_none() { " (refused)" } else { "" }));
                        let sec = r.2.and_then(|at| r.0.get(at as usize..)).unwrap_or(&[]);
                        let ms: Vec<String> = masks.iter().enumerate().filter(|(_, h)| **h != 0).map(|(k, h)| format!("mask {k}: {}", mask_view_text(*h, sec, c.records, k as u32))).collect();
                        *g = Some(format!("pipeline 0x{:x}; CB b4 +0xb4 {cbs}; {xs}; {}; {}", d.pipeline, if ms.is_empty() { "no mask".to_string() } else { ms.join("; ") }, vm::section_text(sec, c.records)));
                    }
                }
                r
            }
            None => vm::table_with(&remap, base, palette.as_deref(), mul_on(), xnw_world::carshade::on(), read, None),
        };
        LV[9].fetch_add(masks.iter().filter(|m| **m != 0).count() as u64, Ordering::Relaxed);
        if n.palette != 0 {
            WHY[6].fetch_add(1, Ordering::Relaxed);
        }
        if n.no_palette != 0 {
            WHY[7].fetch_add(1, Ordering::Relaxed);
        }
        Arc::new(Block { words: t, livery_at: at.unwrap_or(0), livery: if at.is_some() { masks } else { [0; 2] }, slod: 0 })
    });
    WHY[if made.is_some() { 4 } else { 3 }].fetch_add(1, Ordering::Relaxed);
    if let Some(c) = lock(&CACHE).as_mut().filter(|c| c.0 == frame) {
        c.1.insert(key, made.clone());
    }
    made
}

/// The draw tap (title thread, at the list sink, after a vehicle-body hook draw).
pub(super) fn on_draw_sink(d: &TapDraw) {
    set_pending(false);
    let Some(t) = table_of(d, super::frame_key()) else { return };
    if super::attach_vehmat(t) {
        WHY[5].fetch_add(1, Ordering::Relaxed);
    }
}

/// The frame line: rows with a table of the vehicle-body rows, distinct
/// tables, and the sinks by outcome since the last line.
pub(super) fn line(frame: u64, rows: u32, vehicle: usize, tables: usize) -> String {
    let why: Vec<String> = NAMES.iter().zip(take()).map(|(n, v)| format!("{n} {v}")).collect();
    let mut l = format!("world render: frame {frame}: vehmat: {rows} of {vehicle} vehicle body, wheel and tyre rows coloured from {tables} material tables; sinks: {}", why.join(", "));
    if livery_on() {
        let v: Vec<String> = LV_NAMES.iter().zip(LV.iter()).map(|(n, v)| format!("{n} {}", v.swap(0, Ordering::Relaxed))).collect();
        l += &format!("; livery: {}", v.join(", "));
        if let Some(s) = lock(&DETAIL).take() {
            l += &format!("; livery detail: {s}");
        }
    }
    l
}

/// Puts a draw's block into the frame's instance floats (264 dwords and its
/// palette rows, whole instances); the same block (by pointer) once a list.
/// Returns its dword index.
pub(super) fn place(list: &mut xnw_world::FrameList, placed: &mut HashMap<usize, u32>, t: &Table) -> Option<u32> {
    let key = Arc::as_ptr(t) as usize;
    if let Some(&at) = placed.get(&key) {
        return Some(at);
    }
    let at = u32::try_from(list.instances.len()).ok()?;
    list.instances.extend(t.words.iter().map(|w| f32::from_bits(*w)));
    list.instances.resize(at as usize + t.words.len().max(vm::LUT_DWORDS).div_ceil(12) * 12, 0.0);
    placed.insert(key, at);
    Some(at)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tables join the instance floats whole (264 dwords, a multiple of 12:
    /// later instance indices stay aligned), once per table.
    #[test]
    fn tables_join_the_instances_once() {
        let mut list = xnw_world::FrameList { instances: vec![0.0; 24], ..Default::default() };
        let mut placed = HashMap::new();
        let mut a = vec![0u32; vm::LUT_DWORDS];
        a[42] = vm::HAS_COLOUR | 0x08_0808;
        // A block with two palette rows (PAL1): 256 + 2 x 128 dwords, padded to 12.
        let blk = |words: Vec<u32>| Arc::new(Block { words, livery_at: 0, livery: [0; 2], slod: 0 });
        let (a, b): (Table, Table) = (blk(a), blk(vec![0; (256 + 2 * vm::PAL_TEXELS).div_ceil(12) * 12]));
        assert_eq!(place(&mut list, &mut placed, &a), Some(24));
        assert_eq!(place(&mut list, &mut placed, &a), Some(24));
        assert_eq!(place(&mut list, &mut placed, &b), Some(24 + 264));
        assert_eq!(list.instances.len(), 24 + 264 + 516);
        assert_eq!(list.instances.len() % 12, 0);
        assert_eq!(list.instances[24 + 42].to_bits(), vm::HAS_COLOUR | 0x08_0808);
        assert_eq!((RECORD_SLOT, REMAP_SLOTS), (42, [8, 7, 11, 12]));
        assert_eq!((LIVERY_DRAW_SLOT, LIVERY_RECORD_SLOT, LIVERY_LAYER_SLOT), (80, 46, 3));
    }
}
