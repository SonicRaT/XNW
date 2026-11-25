//! The ped pre-skin in the live pass (`XNW_WORLD_RENDER=...,preskin=on`,
//! default off). At each direct dispatch of a covered pre-skin program the
//! inputs are read from title memory where the frame pack finds them: the
//! 96-B constant block (table root 1 slot 8), the vertex and remap views
//! its +0x40 / +0x44 name in the bound view heap, the palette rows from its
//! +0x28 base (table root 0 slot 1), and the hide-mask record picked through
//! root 0 slots 9, 0 and 6. Vertex and remap bytes are kept by address,
//! length and stream header (static per model); the palette rows and the
//! block are read every frame. The frame's items go with its list; a draw
//! whose own record's +0x54 range word (T9 slot 0, read at its list sink)
//! names an item's output takes its positions from our compute pass. Variant 11 (the face) is read with the header of
//! its position table (root 1 slots 9 and 0) and runs, while that table is
//! unfilled, from its bind shape. `preskinbits` (hex, default 0) widens the
//! cover: bit 0 variants 8/9 without their maps, bit 1 every face from its
//! bind shape. Every read is fault-safe: a dispatch that cannot be read is
//! counted by reason, never a fault; dispatches of the standard variants
//! our pass does not run are counted by variant.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use xnw_formats::framepack::{RANGE_MASK, view};
use xnw_gpu::d3d12x::tap::{self, TapDraw, TapRoot, TapView};
use xnw_world::preskin::{self as ps, Inputs};
use xnw_world::preskin_gpu::{Item, MASK_RECORD, PreskinFrame, Reads, WHY_FACE, address_key, covered, rule_for, screen_bits, screen_switch, takes_delta};
use xnw_world::{FrameList, NO_PALETTE};

/// `preskin=on` (read once).
pub fn on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| screen_switch(&v)));
    *ON
}

/// Registers the dispatch tap (once).
pub(super) fn install() {
    static DONE: std::sync::Once = std::sync::Once::new();
    DONE.call_once(|| {
        tap::set_dispatch_tap(Some((want, on_dispatch)));
        xnw_core::info!("world probe: pre-skin on: dispatches of the covered ped_skinning_cs variants are read for our compute pass");
    });
}

/// `preskinbits` (read once).
pub fn bits() -> u32 {
    static BITS: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| std::env::var("XNW_WORLD_RENDER").map_or(0, |v| screen_bits(&v)));
    *BITS
}

/// The variant of a pipeline's compute program (None: not one of the
/// standard pre-skin variants), asked once per pipeline.
static RULES: Mutex<Option<HashMap<usize, Option<u32>>>> = Mutex::new(None);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The variant of a program hash: its place among the standard programs.
pub fn variant_of(hash: u128) -> Option<u32> {
    super::pack::PED_SKINNING[..13].iter().position(|h| *h == hash).map(|v| v as u32)
}

fn rule_of(pipeline: usize) -> Option<u32> {
    *lock(&RULES).get_or_insert_with(HashMap::new).entry(pipeline).or_insert_with(|| tap::stages(pipeline).iter().filter(|s| s.name == "compute").find_map(|s| variant_of(s.hash)))
}

fn want(pipeline: usize) -> bool {
    rule_of(pipeline).is_some()
}

/// The PreSkinD mesh programs (163 in shaders.rpf: every name holding
/// `PreSkinD` in `<out>/shader-names.tsv`, the container hash >> 96;
/// sorted, unique in the pack). Their draws read the pre-skin output
/// through their own record's +0x54.
pub const DRAW_PROGRAMS: [u32; 163] = [
    0x0397_681c, 0x0464_0cf5, 0x068d_873d, 0x0c24_9ede, 0x0df9_246d, 0x0e9c_8cd0, 0x0eed_e673, 0x0f03_d887,
    0x0f2b_55c2, 0x12c2_d189, 0x12d3_0758, 0x1359_217b, 0x15c2_507d, 0x18aa_62be, 0x196b_43dc, 0x1a31_663e,
    0x1c95_e2d7, 0x1ccb_b6bb, 0x1d8b_b275, 0x1d9b_55b0, 0x1e3d_d729, 0x1e94_461e, 0x208d_cdf8, 0x2683_d442,
    0x26fe_3d4a, 0x2ae9_3660, 0x2d38_8641, 0x2dae_f9eb, 0x2e93_b7b8, 0x2faf_83ac, 0x3055_766a, 0x33d0_9b90,
    0x34b4_6f0e, 0x3518_9b54, 0x36f9_07c1, 0x37f9_b27b, 0x39c5_775e, 0x3c08_c84c, 0x3c41_ffc6, 0x3cd2_f671,
    0x3de7_2418, 0x4094_3a69, 0x40bd_5f22, 0x4179_3408, 0x421f_c6f1, 0x4712_0965, 0x47ec_6b6e, 0x48c3_63e8,
    0x4ad2_e86a, 0x4e5f_a07b, 0x4eed_2d4d, 0x50f9_36fb, 0x5136_6fb3, 0x51e8_c757, 0x56f5_59c4, 0x5807_eeb6,
    0x5bad_4edb, 0x5f8a_d1d5, 0x6011_e9a5, 0x617a_226a, 0x6429_bba6, 0x643c_cef7, 0x6839_cec0, 0x6a3c_8030,
    0x6d43_fc73, 0x6d74_0592, 0x6fec_37a8, 0x71ee_86a5, 0x74dc_7bd0, 0x75c2_077f, 0x76a9_3382, 0x7816_1419,
    0x798a_38fb, 0x7b12_952c, 0x7e40_4476, 0x81ef_5608, 0x83d1_a12b, 0x83e3_2400, 0x8509_f034, 0x85a1_37eb,
    0x8767_dd95, 0x8948_5ff4, 0x8bc1_485c, 0x8c76_5143, 0x8cc1_f7e8, 0x8db6_457b, 0x9043_bb59, 0x9090_854b,
    0x93b3_848f, 0x93fc_7d02, 0x94f3_0a6d, 0x95c7_d110, 0x9969_6c9a, 0x999c_98ec, 0x9a17_e53a, 0x9a64_16f7,
    0x9bc8_6409, 0x9d4a_39f5, 0x9f29_891b, 0xa134_f4dd, 0xa298_38a5, 0xa33f_8f31, 0xa413_0a4a, 0xa6b1_c50a,
    0xa70f_54bb, 0xaa07_fcd9, 0xaa5d_e3a0, 0xab69_356f, 0xabe8_4930, 0xb0a3_9ce2, 0xb718_9cfe, 0xb8b2_13e5,
    0xba5f_4676, 0xbb07_df5f, 0xbd05_76e5, 0xbeb0_b803, 0xc3a2_453b, 0xc4fb_a1f9, 0xc504_c5ad, 0xc583_26fb,
    0xc64f_ecc3, 0xc6f0_0e34, 0xc8ba_7959, 0xc978_96f1, 0xcb67_cf90, 0xcc7f_d522, 0xcc9b_68d7, 0xcd8b_0522,
    0xce32_2575, 0xcf56_bcd8, 0xcf75_21c5, 0xd089_ab02, 0xd1a7_0cb6, 0xd24c_fd16, 0xd3b5_ea85, 0xd3fb_f12b,
    0xd66f_39a8, 0xd854_0223, 0xdd33_18ff, 0xdee0_9f8b, 0xe020_21fc, 0xe44a_e8ff, 0xe554_3f22, 0xe5d2_7632,
    0xe81b_6e82, 0xe8b1_2020, 0xe8be_db27, 0xe9ff_b2dd, 0xeb19_cac4, 0xeb55_3134, 0xef9f_b6b1, 0xf0e3_4211,
    0xf314_72d2, 0xf397_42bc, 0xf4c9_ad8a, 0xf4dd_cdd5, 0xf555_82ab, 0xf64f_36da, 0xf779_2a16, 0xf95f_c768,
    0xfccd_e9b5, 0xfdda_cb36, 0xfff1_285f,
];

thread_local! {
    /// Pipeline object -> it draws with a PreSkinD program.
    static DRAW_PIPES: std::cell::RefCell<HashMap<usize, bool>> = std::cell::RefCell::new(HashMap::new());
}

/// The draw tap's filter: a PreSkinD program (asked once per pipeline).
pub(super) fn draw_want(pipeline: usize) -> bool {
    DRAW_PIPES.with(|m| {
        let mut m = m.borrow_mut();
        if let Some(&w) = m.get(&pipeline) {
            return w;
        }
        if m.len() > 4096 {
            m.clear();
        }
        let w = tap::stages(pipeline).iter().any(|s| (s.name == "mesh" || s.name == "vertex") && DRAW_PROGRAMS.binary_search(&((s.hash >> 96) as u32)).is_ok());
        m.insert(pipeline, w);
        w
    })
}

/// Sinks of PreSkinD draws since the last join line: seen, given to their
/// hook draw, no hook draw to give it to, no 128-B record at a T9 slot 0.
static SINK_N: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The range word of a draw's own record: +0x54 of the 128-B record at
/// slot 0 of one of its tables (T9; the frame pack's `range_word`).
fn own_word(src: &dyn Source, roots: usize) -> Option<u32> {
    let v = (0..roots).find_map(|r| src.table(r, 0).filter(|v| v.bytes == 128 && v.addr != 0))?;
    let b = src.read(v.addr + 0x54, 4)?;
    Some(le32(&b, 0))
}

/// The draw tap (title thread, at the list sink, after the draw's hook):
/// a PreSkinD draw's own range word, given to this thread's last hooked
/// draw (the join key; the hook's TLS record can be another draw's).
pub(super) fn on_draw_sink(d: &TapDraw) {
    SINK_N[0].fetch_add(1, Ordering::Relaxed);
    let _site = super::fault_site(super::SITE_PRESKIN);
    let k = match own_word(&Title(d), d.roots.len()) {
        Some(w) if super::attach_sink_range(w) => 1,
        Some(_) => 2,
        None => 3,
    };
    SINK_N[k].fetch_add(1, Ordering::Relaxed);
}

/// The join line (every 60 frames): the PreSkinD sinks since the last line,
/// and this frame's rows (`words`: per list row, its own record's range word
/// and the hook's): with their own word, of them the hook's word names
/// another range (the old join's key), rows joined to an item.
pub fn join_line(frame: u64, words: &[(Option<u32>, Option<u32>)], joined: usize) -> String {
    let n: Vec<u64> = SINK_N.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect();
    let own = words.iter().filter(|w| w.0.is_some()).count();
    let other = words.iter().filter(|w| w.0.is_some_and(|o| w.1.is_none_or(|h| h & RANGE_MASK != o & RANGE_MASK))).count();
    format!("world render: frame {frame}: pre-skin join: PreSkinD sinks (60 frames) {}, given to their draw {}, no hook draw {}, no record {}; this frame rows with their own record {own}, of them the hook's record names another range {other}; rows joined {joined}", n[0], n[1], n[2], n[3])
}

/// Frames with items not yet joined to their list (oldest first).
struct Open {
    frame: u64,
    items: Vec<Item>,
    unreadable: u32,
    ns: u64,
    reads: Reads,
}

const OPEN_MAX: usize = 8;
static OPEN: Mutex<Vec<Open>> = Mutex::new(Vec::new());

/// Static streams by (address, length, header hash): the bytes and the
/// frame they were last used in.
type Kept<T> = HashMap<(u64, u64, u64), (Arc<Vec<T>>, u64)>;
/// A kept stream and its key on the device.
type Stream<T> = (Arc<Vec<T>>, u64);
static VB: Mutex<Option<Kept<u8>>> = Mutex::new(None);
static REMAP: Mutex<Option<Kept<u32>>> = Mutex::new(None);
/// Frames a kept stream stays unused before it goes.
const KEEP_FRAMES: u64 = 600;
static SWEPT: AtomicU64 = AtomicU64::new(0);

fn read(addr: u64, len: u64) -> Option<Vec<u8>> {
    if addr == 0 || len == 0 || !(0x1_0000..0x8000_0000_0000).contains(&addr) || addr.checked_add(len).is_none_or(|e| e > 0x8000_0000_0000) {
        return None;
    }
    let _site = super::fault_site_or_kept(super::SITE_PRESKIN);
    super::dcopy(addr, len as usize)
}

fn le32(b: &[u8], at: usize) -> u32 {
    b.get(at..at + 4).map_or(0, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

/// The tap: a covered pre-skin dispatch's item, or one more not run (by
/// reason, or by variant when our pass does not cover it).
fn on_dispatch(d: &TapDraw) {
    let t0 = std::time::Instant::now();
    let frame = super::frame_key();
    let mut new = New::default();
    let item = match rule_of(d.pipeline) {
        Some(v) if covered(v, bits()) => Some(item_of(&Title(d), frame, v, bits(), &mut new)),
        _ => None,
    };
    let ns = t0.elapsed().as_nanos() as u64;
    let mut open = lock(&OPEN);
    let k = match open.iter().position(|o| o.frame == frame) {
        Some(k) => k,
        None => {
            if open.len() >= OPEN_MAX {
                open.remove(0);
            }
            open.push(Open { frame, items: Vec::new(), unreadable: 0, ns: 0, reads: Reads::default() });
            open.len() - 1
        }
    };
    let o = &mut open[k];
    o.ns += ns;
    o.reads.max_ms = o.reads.max_ms.max(ns as f64 / 1e6);
    (o.reads.new, o.reads.new_kb, o.reads.new_ms) = (o.reads.new + new.streams, o.reads.new_kb + new.bytes / 1024, o.reads.new_ms + new.ns as f64 / 1e6);
    match item {
        Some(Ok(i)) => o.items.push(i),
        Some(Err(why)) => {
            o.unreadable += 1;
            o.reads.count(why);
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| xnw_core::warn!("world probe: pre-skin: a dispatch not read ({why}); counted per frame by reason"));
        }
        None => {
            if let Some(n) = rule_of(d.pipeline).and_then(|v| o.reads.uncovered.get_mut(v as usize)) {
                *n += 1;
            }
        }
    }
}

/// Streams read whole by one dispatch (not yet kept): count, bytes, time.
#[derive(Default)]
pub struct New {
    streams: u32,
    bytes: u64,
    ns: u64,
}

/// What the reader needs of a dispatch's binding and of title memory
/// (a trait so the test can stand in for both).
pub trait Source {
    /// The view at table root `root`, slot `slot` (None: no view there).
    fn table(&self, root: usize, slot: u32) -> Option<TapView>;
    /// The view at heap index `index` of the bound view heap.
    fn heap(&self, index: u32) -> Option<TapView>;
    fn read(&self, addr: u64, len: u64) -> Option<Vec<u8>>;
}

struct Title<'a>(&'a TapDraw<'a>);

impl Source for Title<'_> {
    fn table(&self, root: usize, slot: u32) -> Option<TapView> {
        match self.0.roots.get(root) {
            Some(TapRoot::Table { handle, slots, .. }) if slot < *slots => Some(tap::view(handle + tap::SLOT * u64::from(slot))).filter(|v| v.kind != view::NONE),
            _ => None,
        }
    }

    fn heap(&self, index: u32) -> Option<TapView> {
        tap::heap_slot(self.0.view_heap, index).map(|(_, h)| tap::view(h)).filter(|v| v.kind != view::NONE)
    }

    fn read(&self, addr: u64, len: u64) -> Option<Vec<u8>> {
        read(addr, len)
    }
}

/// A buffer view's bytes from `at` (byte offset in the view), at most
/// `len`, cut at the view's end.
fn view_read(src: &dyn Source, v: &TapView, at: u64, len: u64) -> Option<Vec<u8>> {
    if !matches!(v.kind, view::BUFFER | view::PLACED_BUFFER | view::CONSTANT) || at >= v.bytes {
        return None;
    }
    src.read(v.addr + at, len.min(v.bytes - at))
}

/// A static stream kept by address, length and the hash of its first
/// 128 bytes; read whole the first time (outside the lock: other title
/// threads' dispatches do not wait on it; counted in `new`). With it, the
/// key our pass keeps it on the device by (the same three).
fn kept<T: Clone>(cache: &Mutex<Option<Kept<T>>>, src: &dyn Source, v: &TapView, frame: u64, max: u64, new: &mut New, make: impl Fn(Vec<u8>) -> Vec<T>) -> Result<Stream<T>, &'static str> {
    if v.bytes == 0 || v.bytes > max {
        return Err("stream size");
    }
    let head = view_read(src, v, 0, 128).ok_or("stream unreadable")?;
    let key = (v.addr, v.bytes, xnw_core::hash128(&[&head]) as u64);
    let device = address_key(v.addr, v.bytes, &head);
    if let Some((a, last)) = lock(cache).get_or_insert_with(HashMap::new).get_mut(&key) {
        *last = frame;
        return Ok((a.clone(), device));
    }
    let t = std::time::Instant::now();
    let bytes = view_read(src, v, 0, v.bytes).ok_or("stream unreadable")?;
    (new.streams, new.bytes) = (new.streams + 1, new.bytes + bytes.len() as u64);
    let a = Arc::new(make(bytes));
    new.ns += t.elapsed().as_nanos() as u64;
    // A second thread may have read it meanwhile: the first one kept stays.
    let mut g = lock(cache);
    let (a, last) = g.get_or_insert_with(HashMap::new).entry(key).or_insert((a, frame));
    *last = frame;
    Ok((a.clone(), device))
}

/// Ok when the face's position table is unfilled: b6 (root 1 slot 9)
/// names the table (b6+0xc set) and its header lies at b6+0 in t74 (root 1
/// slot 0); only the header is read. Err: why the face is not run from its
/// bind shape (the line counts it under that reason).
fn face_unfilled(src: &dyn Source) -> Result<(), &'static str> {
    let _site = super::fault_site(super::SITE_PRESKIN_FACE);
    let Some(mut b6) = src.table(1, 9).and_then(|v| view_read(src, &v, 0, 16)).filter(|b| b.len() == 16) else { return Err("face b6 unreadable") };
    if le32(&b6, 0xc) == 0 {
        return Err("face delta step");
    }
    let base = u64::from(le32(&b6, 0));
    let Some(head) = src.table(1, 0).and_then(|v| view_read(src, &v, base, 28)).filter(|h| h.len() == 28) else { return Err("face table unreadable") };
    // The header read alone: the table at offset 0 of it.
    b6[..4].fill(0);
    if ps::table_unfilled(&Inputs { b6: &b6, t74: &head, ..Default::default() }) { Ok(()) } else { Err("face table filled") }
}

/// Vertex views past this are refused (the capture's buffer cap).
const VB_MAX: u64 = 16 << 20;
const REMAP_MAX: u64 = 1 << 20;
/// Variant 2's delta tables past this are refused.
const DELTA_MAX: u64 = 1 << 20;

/// One dispatch's item: the same inputs `xnw_frame::preskin::load` takes
/// from a capture, cut to what the program reads (palette rows up to the
/// largest remapped bone; the one hide-mask record), under the variant's
/// rule ([`rule_for`], `bits`: `preskinbits`).
pub fn item_of(src: &dyn Source, frame: u64, variant: u32, bits: u32, new: &mut New) -> Result<Item, &'static str> {
    let b7v = src.table(1, 8).filter(|v| v.kind == view::CONSTANT).ok_or("no constant block at root 1 slot 8")?;
    let mut b7 = view_read(src, &b7v, 0, 96).filter(|b| b.len() == 96).ok_or("constant block unreadable")?;
    let face = if variant == 11 { face_unfilled(src) } else { Ok(()) };
    let (rule, neutral, unmapped) = rule_for(variant, &b7, bits, || face.is_ok()).map_err(|e| face.err().filter(|_| e == WHY_FACE).unwrap_or(e))?;
    let c = |off: usize| le32(&b7, off);
    let vb_view = src.heap(c(0x40)).ok_or("no vertex view")?;
    let (vb, vb_key) = kept(&VB, src, &vb_view, frame, VB_MAX, new, |b| b)?;
    let remap_view = src.heap(c(0x44)).ok_or("no remap view")?;
    let (remap, remap_key) = kept(&REMAP, src, &remap_view, frame, REMAP_MAX, new, |b| b.chunks_exact(2).map(|w| u32::from(u16::from_le_bytes([w[0], w[1]]))).collect())?;
    let rows = remap.iter().map(|r| r & 0xfff).max().map_or(1, |m| m + 1);
    let pal = src.table(0, 1).ok_or("no palette view")?;
    let palette = view_read(src, &pal, 48 * u64::from(c(0x28)), 48 * u64::from(rows)).ok_or("palette unreadable")?;
    let palette = palette[..palette.len() / 48 * 48].to_vec();
    // The hide-mask record, as the reference picks it.
    let mask_site = super::fault_site(super::SITE_PRESKIN_MASK);
    let dword = |root: usize, slot: u32, at: u64| src.table(root, slot).and_then(|v| view_read(src, &v, at, 4)).map_or(0, |b| le32(&b, 0));
    // The pick and its record are read only when the block asks for a mask
    // (the same result: no mask otherwise); unread, they cannot fault.
    let asks = c(0x8) != 0 && c(0x20) & c(0x24) != u32::MAX;
    let rec = if asks { dword(0, 0, u64::from(dword(0, 9, u64::from(c(0x3c)) * 48 + 28) >> 16) * 416 + 0xa0) } else { u32::MAX };
    let mask = (asks && (rec as i32) >= 0).then(|| {
        let mut m = src.table(0, 6).and_then(|v| view_read(src, &v, u64::from(rec) * MASK_RECORD as u64, MASK_RECORD as u64)).unwrap_or_default();
        m.resize(MASK_RECORD, 0);
        m
    });
    drop(mask_site);
    // The rows start at the block's base.
    b7[0x28..0x2c].copy_from_slice(&0u32.to_le_bytes());
    // Variant 2: t76 (root 1 slot 5) from its header, whole (its entries are indexed per vertex).
    let delta = if takes_delta(variant, bits) {
        let _site = super::fault_site(super::SITE_PRESKIN_DELTA);
        let v = src.table(1, 5).ok_or("no delta table")?;
        if v.bytes > DELTA_MAX {
            return Err("delta table size");
        }
        view_read(src, &v, 0, v.bytes).ok_or("delta table unreadable")?
    } else {
        Vec::new()
    };
    Ok(Item { b7, vb, remap, palette, mask, rule, neutral, unmapped, vb_key, remap_key, delta })
}

/// The pre-skin of `list` (frame `frame`): its items, and the rows whose
/// range word (`range(row)`: the row's own record +0x54, u32::MAX none)
/// names an item's output. Rows of a draw without a palette and without
/// the ped flag are left out (a vehicle's rec+0x54 is a bone offset).
pub fn frame_for(frame: u64, list: &FrameList, range: impl Fn(usize) -> u32) -> PreskinFrame {
    let mut open = lock(&OPEN);
    let Some(k) = open.iter().position(|o| o.frame == frame) else { return PreskinFrame::default() };
    let o = open.remove(k);
    open.retain(|x| x.frame > frame);
    drop(open);
    sweep(frame);
    let by_word: HashMap<u32, u32> = o.items.iter().enumerate().map(|(i, it)| (it.word() & RANGE_MASK, i as u32)).collect();
    let draws = list
        .draws
        .iter()
        .enumerate()
        .filter(|(_, r)| r.ped || r.palette != NO_PALETTE)
        .filter_map(|(row, _)| {
            let w = range(row);
            let item = *by_word.get(&(w & RANGE_MASK)).filter(|_| w != u32::MAX)?;
            Some((row as u32, item, ps::range(w).3))
        })
        .collect();
    PreskinFrame { items: o.items, draws, unreadable: o.unreadable, read_ms: o.ns as f64 / 1e6, reads: o.reads }
}

/// Drops kept streams unused for KEEP_FRAMES (every KEEP_FRAMES frames).
fn sweep(frame: u64) {
    if frame < SWEPT.load(Ordering::Relaxed) + KEEP_FRAMES {
        return;
    }
    SWEPT.store(frame, Ordering::Relaxed);
    if let Some(m) = lock(&VB).as_mut() {
        m.retain(|_, (_, last)| *last + KEEP_FRAMES > frame);
    }
    if let Some(m) = lock(&REMAP).as_mut() {
        m.retain(|_, (_, last)| *last + KEEP_FRAMES > frame);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xnw_world::preskin::Bones;

    /// Title memory and a binding made up in this process: the views point
    /// at our own vectors.
    struct Made {
        tables: Vec<(usize, u32, Vec<u8>, u32)>,
        heap: Vec<(u32, Vec<u8>)>,
    }

    fn view_of(b: &[u8], kind: u32) -> TapView {
        TapView { kind, addr: b.as_ptr() as u64, bytes: b.len() as u64, ..TapView::default() }
    }

    impl Source for Made {
        fn table(&self, root: usize, slot: u32) -> Option<TapView> {
            self.tables.iter().find(|t| t.0 == root && t.1 == slot).map(|t| view_of(&t.2, t.3))
        }
        fn heap(&self, index: u32) -> Option<TapView> {
            self.heap.iter().find(|h| h.0 == index).map(|h| view_of(&h.1, view::BUFFER))
        }
        fn read(&self, addr: u64, len: u64) -> Option<Vec<u8>> {
            // SAFETY: the test's own vectors (every view above points into one).
            Some(unsafe { std::slice::from_raw_parts(addr as *const u8, len as usize) }.to_vec())
        }
    }

    fn put(b: &mut [u8], at: usize, v: u32) {
        b[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }

    /// The live read gives the item the capture route gives
    /// (`Item::from_inputs` over whole views), with the palette cut to the
    /// remapped rows and the mask record picked; the vertex view is kept
    /// and shared by a second read; a missing block is a reason.
    #[test]
    fn live_item_equals_capture_item() {
        let mut b7 = vec![0u8; 96];
        put(&mut b7, 0x0, 8 | 0x80 << 8);
        put(&mut b7, 0x4, 4 | 0x90 << 8);
        put(&mut b7, 0x8, 4 | 0x90 << 8);
        put(&mut b7, 0xc, 4 | 0x98 << 8);
        put(&mut b7, 0x10, 4 | 0xa0 << 8);
        put(&mut b7, 0x20, 3);
        put(&mut b7, 0x24, u32::MAX);
        put(&mut b7, 0x28, 2);
        put(&mut b7, 0x30, 1);
        put(&mut b7, 0x34, 0x6000_0010);
        put(&mut b7, 0x38, 2);
        put(&mut b7, 0x3c, 1);
        put(&mut b7, 0x40, 70);
        put(&mut b7, 0x44, 71);
        let mut vb = vec![0u8; 0xb0];
        vb[0x98] = 3;
        let remap: Vec<u8> = [0u16, 0, 0, 0, 7, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let palette: Vec<u8> = (0..48 * 20).map(|i| i as u8).collect();
        let mut t7 = vec![0u8; 96];
        put(&mut t7, 48 + 28, 2 << 16);
        let mut t71 = vec![0u8; 3 * 416];
        put(&mut t71, 2 * 416 + 0xa0, 1);
        let mut t78 = vec![0u8; 2 * MASK_RECORD];
        t78[MASK_RECORD + 3] = 0x5a;
        let src = Made { tables: vec![(1, 8, b7.clone(), view::CONSTANT), (0, 1, palette.clone(), view::BUFFER), (0, 9, t7.clone(), view::BUFFER), (0, 0, t71.clone(), view::BUFFER), (0, 6, t78.clone(), view::BUFFER)], heap: vec![(70, vb.clone()), (71, remap.clone())] };
        let live = item_of(&src, 5, 3, 0, &mut New::default()).unwrap();
        let mut rebased = b7.clone();
        put(&mut rebased, 0x28, 0);
        let remap32: Vec<u32> = vec![0, 0, 0, 0, 7, 0];
        let want = Item::from_inputs(&Inputs { b7: &rebased, vb: &vb, remap: &remap32, palette: &palette[2 * 48..], t7: &t7, t71: &t71, t78: &t78, ..Default::default() }, Bones::Eight);
        assert_eq!((&live.b7, &live.vb, &live.remap, &live.mask, live.rule, live.neutral), (&want.b7, &want.vb, &want.remap, &want.mask, want.rule, false));
        assert_eq!(live.mask.as_ref().map(|m| m[3]), Some(0x5a));
        // Rows up to the largest remapped bone (7): 8 rows from the base.
        assert_eq!(live.palette, palette[2 * 48..10 * 48]);
        let again = item_of(&src, 6, 3, 0, &mut New::default()).unwrap();
        assert!(Arc::ptr_eq(&live.vb, &again.vb));
        assert_eq!((again.vb_key, again.remap_key), (live.vb_key, live.remap_key));
        assert_ne!(live.vb_key, live.remap_key);
        let none = Made { tables: vec![], heap: vec![] };
        assert_eq!(item_of(&none, 5, 0, 0, &mut New::default()).err(), Some("no constant block at root 1 slot 8"));
    }

    /// The face (variant 11): its bones by b7+0x5c, run from the bind shape
    /// while the table's header (at b6+0 in t74) has a zero scale; a filled
    /// table, or no table named (the delta step), is not covered and named
    /// as such; with `preskinbits` bit 1 it runs from its bind shape too.
    #[test]
    fn face_runs_neutral_while_its_table_is_unfilled() {
        let mut b7 = vec![0u8; 96];
        put(&mut b7, 0x34, 0x6000_0010);
        put(&mut b7, 0x38, 2);
        put(&mut b7, 0x40, 70);
        put(&mut b7, 0x44, 71);
        put(&mut b7, 0x5c, 2);
        let vb = vec![0u8; 0x80];
        let remap: Vec<u8> = [0u16, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
        let palette = vec![0u8; 48 * 2];
        let mut b6 = vec![0u8; 16];
        put(&mut b6, 0, 0x40);
        put(&mut b6, 0xc, 1);
        let mut t74 = vec![0u8; 0x40 + 28];
        let src = |b6: &[u8], t74: &[u8]| Made { tables: vec![(1, 8, b7.clone(), view::CONSTANT), (0, 1, palette.clone(), view::BUFFER), (1, 9, b6.to_vec(), view::CONSTANT), (1, 0, t74.to_vec(), view::BUFFER)], heap: vec![(70, vb.clone()), (71, remap.clone())] };
        let it = item_of(&src(&b6, &t74), 7, 11, 0, &mut New::default()).unwrap();
        assert_eq!((it.rule, it.neutral), (Bones::FourAll, true));
        put(&mut t74, 0x40 + 20, 1f32.to_bits());
        assert_eq!(item_of(&src(&b6, &t74), 7, 11, 0, &mut New::default()).err(), Some("face table filled"));
        // `preskinbits` bit 1: that face from its bind shape too.
        let any = item_of(&src(&b6, &t74), 7, 11, xnw_world::preskin_gpu::BIT_FACE_ANY, &mut New::default()).unwrap();
        assert!(any.neutral);
        let mut no_table = b6.clone();
        put(&mut no_table, 0xc, 0);
        assert_eq!(item_of(&src(&no_table, &vec![0u8; 0x40 + 28]), 7, 11, 0, &mut New::default()).err(), Some("face delta step"));
        assert_eq!(item_of(&src(&b6, &[0u8; 8]), 7, 11, 0, &mut New::default()).err(), Some("face table unreadable"));
        assert!(!covered(8, 0) && covered(11, 0) && covered(0, 0));
        // Bit 0: variant 8 under 4's rule, counted as unmapped; the new streams counted.
        let mut new = New::default();
        let v8 = item_of(&src(&b6, &t74), 9, 8, xnw_world::preskin_gpu::BIT_UNMAPPED, &mut new).unwrap();
        assert_eq!((v8.rule, v8.neutral, v8.unmapped), (Bones::Eight, false, true));
    }

    /// The join: a row with a palette or the ped flag whose range word
    /// names an item; the normals by the row's own primary bit.
    #[test]
    fn rows_join_items_by_range_word() {
        use xnw_world::{Albedo, DrawRecord, GeomKey, Material, NO_REMAP};
        let it = |word: u32| {
            let mut b7 = vec![0u8; 96];
            put(&mut b7, 0x34, word);
            Item { b7, vb: Arc::new(Vec::new()), remap: Arc::new(Vec::new()), palette: Vec::new(), mask: None, rule: Bones::Four, neutral: false, unmapped: false, vb_key: 1, remap_key: 1, delta: Vec::new() }
        };
        // Below the first sweep (frame 600): the other test's kept streams stay.
        let frame = 500;
        lock(&OPEN).push(Open { frame, items: vec![it(0x6000_0010), it(0x2100_0020)], unreadable: 1, ns: 2_000_000, reads: Reads::default() });
        let rec = |ped: bool, palette: u32| DrawRecord { geom: GeomKey::default(), world: [0.0; 16], viewport: 1, effect: 0, albedo: Albedo::Flat([1.0; 3]), palette, stage: 67, inst_first: 0, inst_count: 0, bone_first: 0, bone_count: 0, material: Material::default(), state: None, remap: NO_REMAP, ped };
        let list = FrameList { draws: vec![rec(true, NO_PALETTE), rec(false, 3), rec(false, NO_PALETTE), rec(true, NO_PALETTE)], ..Default::default() };
        let words = [0x2000_0010u32, 0x4100_0020, 0x6000_0010, u32::MAX];
        let pre = frame_for(frame, &list, |row| words[row]);
        assert_eq!(pre.draws, [(0, 0, false), (1, 1, true)]);
        assert_eq!((pre.items.len(), pre.unreadable, pre.read_ms), (2, 1, 2.0));
        assert!(frame_for(frame, &list, |row| words[row]).items.is_empty());
    }

    /// The join key is the draw's own record: +0x54 of the 128-B view at
    /// slot 0 of a table (not root 0's 64-B one); none without such a view.
    /// The line counts rows whose hook word names another range.
    #[test]
    fn own_record_is_the_join_key() {
        let mut rec = vec![0u8; 128];
        put(&mut rec, 0x54, 0x6000_0123);
        let small = vec![7u8; 64];
        let made = Made { tables: vec![(0, 0, small.clone(), view::BUFFER), (2, 0, rec, view::BUFFER)], heap: Vec::new() };
        assert_eq!(own_word(&made, 4), Some(0x6000_0123));
        assert_eq!(own_word(&made, 2), None);
        assert_eq!(own_word(&Made { tables: vec![(0, 0, small, view::BUFFER)], heap: Vec::new() }, 4), None);
        assert!(DRAW_PROGRAMS.windows(2).all(|w| w[0] < w[1]));
        for a in &SINK_N {
            a.store(0, Ordering::Relaxed);
        }
        SINK_N[0].store(3, Ordering::Relaxed);
        SINK_N[1].store(2, Ordering::Relaxed);
        SINK_N[3].store(1, Ordering::Relaxed);
        // Own 0x6000_0123: the hook's 0x2000_0123 is the same range (bits 30-31 are format), 0x6000_0456 another; no hook word: another.
        let words = [(Some(0x6000_0123), Some(0x2000_0123)), (Some(0x6000_0123), Some(0x6000_0456)), (Some(0x6000_0123), None), (None, Some(1))];
        assert_eq!(
            join_line(60, &words, 3),
            "world render: frame 60: pre-skin join: PreSkinD sinks (60 frames) 3, given to their draw 2, no hook draw 0, no record 1; this frame rows with their own record 3, of them the hook's record names another range 2; rows joined 3"
        );
        assert!(SINK_N.iter().all(|a| a.load(Ordering::Relaxed) == 0));
    }
}
