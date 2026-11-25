//! Upload ahead of a camera cut, keyed on residency.
//!
//! The title streams the next shot's drawables in seconds before the cut
//! (the cinematic prestream lists, lead 4 s by default). Each resource
//! that becomes resident passes 0x1454BD9C0(streamer, u32 strIndex, flag),
//! "mark loaded" [V; callers 0x14258CA80 (in-deadline place, after its
//! module's post-place vt+0x50), 0x1400E9520, 0x1409D9750, 0x142590E60].
//!
//! On the title's (streaming) thread the entry hook only stores the index
//! and a time in a lock-free ring ([`RING`]): no lock, no allocation, no
//! title call. Once a frame, on the writer thread ([`drain`]):
//! 1. strIndex -> module: the 16-byte table at qword_14ABCD200 (count u16
//!    dword_14ABCD208), entry {module*, u32 end}, binary search for the
//!    first `end > index` (0x144DA24E0) [V]; module +8 = first
//!    index, local = (index - first) & 0x3FFFFF (0x144DA2470) [V].
//! 2. Only the DrawableStore for now (vtable off_1461D0050, object
//!    qword_14A92E7D0; others once they are named): GetPtr(local) is
//!    vt+0x68 = 0x14002C890 -> 0x14003B530: `bit (local & 7) of byte
//!    [*(m+0x60) + local/8]` set -> `*(*(m+0x38) + *(u32)(m+0x40) * local)`
//!    [V]. Replicated with guarded reads, never called.
//! 3. Drawable +0x50 + 8*lod -> {Model** data, u16 n} for lod 0..3; Model
//!    +0x08 Geometry**, u16 +0x36 count [V]. Each geometry's
//!    key is made exactly as the draw hook makes it (+0x30 hash, +0x08 VB
//!    grc, +0x40 meshlet buffer else +0x10 IB), and offered to the pass when
//!    `known().wants` it: the pass uploads it after the frame's drawn
//!    geometries, inside its budgets (geometry.rs: `uploaded ahead`).
//!
//! Every read is a guarded copy; counts and sizes are bounded.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use xnw_world::{FrameList, GeomKey, HostGeometry};

const IMAGE_BASE: usize = 0x1_4000_0000;
/// "Mark loaded" (streamer, strIndex, flag) and its entry bytes.
const MARK_LOADED: usize = 0x1_454B_D9C0;
const MARK_LOADED_ENTRY: [u8; 15] = [0x41, 0x57, 0x41, 0x56, 0x56, 0x57, 0x55, 0x53, 0x48, 0x81, 0xEC, 0x48, 0x06, 0x00, 0x00];
const MODULE_TABLE: usize = 0x1_4ABC_D200;
const MODULE_COUNT: usize = 0x1_4ABC_D208;
const DRAWABLE_STORE_VT: usize = 0x1_461D_0050;
/// The three stores share that vtable [V]; told apart by their object.
const DRAWABLE_STORE: usize = 0x1_4A92_E7D0;
const DWD_STORE: usize = 0x1_4A92_E700;
const FRAGMENT_STORE: usize = 0x1_4A92_E8A0;
/// Hook stub id (watch.rs stubs pass it back; unused here).
const HOOK_ID: usize = 9;

/// Measured: 16,950 lost at world entry with 16k slots (40k+ loads in a frame).
const RING_N: usize = 1 << 17;
/// Each slot: time in ms since start (high 32) | strIndex (low 32), 0 = empty.
static RING: [AtomicU64; RING_N] = [const { AtomicU64::new(0) }; RING_N];
static HEAD: AtomicU64 = AtomicU64::new(0);
static T0: OnceLock<std::time::Instant> = OnceLock::new();

/// Title addresses (rebased).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Addrs {
    pub(crate) table: u64,
    pub(crate) count: u64,
    pub(crate) drawable_vt: u64,
    /// DrawableStore, DwdStore, FragmentStore objects.
    pub(crate) stores: [u64; 3],
}

static ADDRS: OnceLock<Addrs> = OnceLock::new();

/// Which parts a `XNW_WORLD_RENDER` value selects: (install the hooks
/// and log/dump placements and scene events, offer geometry bytes). Every
/// recording mode (`1`, `capture`, `screen..`) installs; only `screen..`
/// offers (capture and `1` never feed a pass: no copies for nobody).
pub(crate) fn modes(render: Option<&str>) -> (bool, bool) {
    let r = render.map(str::trim).unwrap_or("");
    let off = r.split(',').any(|p| p.trim() == "ahead=off");
    // The street was lost with residency offers on: off by default
    // in screen mode until the attach predictor (`ahead=on` to offer).
    let on = r.split(',').any(|p| p.trim() == "ahead=on");
    let recording = r == "1" || r.starts_with("capture") || r.starts_with("screen");
    (recording && !off, recording && !off && on && r.starts_with("screen"))
}

static OFFER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn set_offer(on: bool) {
    OFFER.store(on, Ordering::Relaxed);
}

/// `XNW_WORLD_RENDER=...,ahead=off` leaves the title alone.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| !std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| v.split(',').any(|p| p.trim() == "ahead=off")));
    *ON
}

/// The entry hook (title thread): rdx = strIndex.
extern "system" fn on_mark_loaded(frame: usize, _id: usize) {
    // SAFETY: the stub's frame: rdx at frame-32.
    let idx = unsafe { ((frame - 32) as *const u64).read() } as u32;
    if idx & 0x0FFF_FFFF == 0x0FFF_FFFF {
        return;
    }
    push(idx);
}

fn push(idx: u32) {
    let ms = T0.get().map_or(0, |t| t.elapsed().as_millis() as u64).max(1);
    let i = HEAD.fetch_add(1, Ordering::Relaxed);
    RING[(i as usize) & (RING_N - 1)].store(ms << 32 | u64::from(idx), Ordering::Release);
}

// ---- scene event ticks (which event id is the camera cut) ----

/// The scene event tick 0x1450FE750(ev, inst, ...) [V] and its entry.
const SCENE_EVENT: usize = 0x1_450F_E750;
const SCENE_EVENT_ENTRY: [u8; 19] = [0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x56, 0x57, 0x55, 0x53, 0x48, 0x81, 0xEC, 0x48, 0x01, 0x00, 0x00];
const EV_N: usize = 1024;
static EVENTS: [AtomicU64; EV_N] = [const { AtomicU64::new(0) }; EV_N];
static EV_HEAD: AtomicU64 = AtomicU64::new(0);
static EV_INST: AtomicU64 = AtomicU64::new(0);

/// The entry hook (main thread): rcx = event, rdx = cinematic instance [I].
extern "system" fn on_scene_event(frame: usize, _id: usize) {
    // SAFETY: the stub's frame: rcx at frame-24, rdx at frame-32.
    let (ev, inst) = unsafe { (((frame - 24) as *const u64).read(), ((frame - 32) as *const u64).read()) };
    let i = EV_HEAD.fetch_add(1, Ordering::Relaxed);
    EVENTS[(i as usize) & (EV_N - 1)].store(ev, Ordering::Relaxed);
    EV_INST.store(inst, Ordering::Relaxed);
}

/// One ticked event: type id (vt[6] = `mov eax, imm; ret`, read, not called),
/// def+0x28 start s, def+0x30 lead s.
pub(crate) fn event_of(ev: u64) -> Option<(u32, f32, f32)> {
    let vt = q(ev)?;
    let f = q(vt + 0x30)?;
    let b = guarded(f, 6)?;
    let ty = if b[0] == 0xB8 && b[5] == 0xC3 { le(&b, 1, 4) as u32 } else { u32::MAX };
    let def = guarded(q(ev + 8)?, 0x34)?;
    Some((ty, f32::from_bits(le(&def, 0x28, 4) as u32), f32::from_bits(le(&def, 0x30, 4) as u32)))
}

struct EvDrain {
    tail: u64,
    last_log: Option<std::time::Instant>,
    rows: Vec<String>,
}
static EV_DRAIN: Mutex<Option<EvDrain>> = Mutex::new(None);

/// Once a frame: the events ticked since the last frame and the scene clock
/// (inst+0x460 f32 s): a line per second with the table, a line whenever an
/// event starts within this frame (the cut ids show up next to the pass's
/// cut lines), rows for the seq dump (cine.tsv).
pub(crate) fn drain_events(frame: u64) {
    let Ok(mut g) = EV_DRAIN.lock() else { return };
    let d = g.get_or_insert_with(|| EvDrain { tail: 0, last_log: None, rows: Vec::new() });
    let head = EV_HEAD.load(Ordering::Relaxed);
    let from = d.tail.max(head.saturating_sub(EV_N as u64));
    d.tail = head;
    if from == head {
        return;
    }
    let mut evs: Vec<u64> = (from..head).map(|i| EVENTS[(i as usize) & (EV_N - 1)].load(Ordering::Relaxed)).filter(|&e| e != 0).collect();
    evs.sort_unstable();
    evs.dedup();
    let inst = EV_INST.load(Ordering::Relaxed);
    let t = guarded(inst + 0x460, 4).map(|b| f32::from_bits(le(&b, 0, 4) as u32)).filter(|t| t.is_finite());
    let mut table: Vec<(u32, f32, f32, u64)> = evs.iter().take(512).filter_map(|&e| event_of(e).map(|(ty, st, ld)| (ty, st, ld, e))).collect();
    table.sort_by(|a, b| a.1.total_cmp(&b.1));
    if let Some(t) = t {
        let dt = 1.5 / 30.0;
        for (ty, st, _, ev) in table.iter().filter(|e| (e.1 - t).abs() <= dt) {
            xnw_core::info!("world cine: frame {frame}: event type {ty:#x} ({ev:#x}) starts at {st:.3} s, scene clock {t:.3} s");
        }
    }
    let due = d.last_log.is_none_or(|l| l.elapsed().as_secs_f64() >= 1.0);
    if due {
        d.last_log = Some(std::time::Instant::now());
        let shown: Vec<String> = table.iter().take(32).map(|(ty, st, ld, _)| format!("{ty:#x}@{st:.2}/{ld:.1}")).collect();
        xnw_core::info!("world cine: frame {frame}: scene clock {}, {} events ticked (type@start/lead): {}", t.map_or("?".into(), |t| format!("{t:.3} s")), table.len(), shown.join(" "));
    }
    if d.rows.len() < 8192 {
        for (ty, st, ld, ev) in &table {
            d.rows.push(format!("{frame}\t{}\t{ty:#x}\t{st}\t{ld}\t{ev:#x}", t.map_or(f32::NAN, |t| t)));
        }
    }
}

/// The event rows since the last call (seq dumps: cine.tsv).
pub(crate) fn take_event_rows() -> Vec<String> {
    EV_DRAIN.lock().ok().and_then(|mut g| g.as_mut().map(|d| std::mem::take(&mut d.rows))).unwrap_or_default()
}

pub(crate) const CINE_HEADER: &str = "frame\tscene_clock_s\ttype\tstart_s\tlead_s\tevent\n";

// ---- the inverse: a store slot freed (eviction) ----

/// Store vt+0xE8 0x142226DA0(store, local): clears the occupancy bit
/// (`_InterlockedAnd8` on *(store+0x60)), decrements the count, puts the
/// slot on the free list [V]: the unload of one store object.
const SLOT_REMOVE: usize = 0x1_4222_6DA0;
const SLOT_REMOVE_ENTRY: [u8; 12] = [0x56, 0x57, 0x53, 0x48, 0x83, 0xEC, 0x20, 0x89, 0xD3, 0x48, 0x89, 0xCE];
/// Marks not occupied yet are tried again at this many later drains.
const RETRIES: u8 = 30;
const UNLOAD_N: usize = 1 << 16;
static UNLOADS: [AtomicU64; UNLOAD_N] = [const { AtomicU64::new(0) }; UNLOAD_N];
static EV_UNLOAD_HEAD: AtomicU64 = AtomicU64::new(0);

/// The entry hook (title thread): rcx = store, rdx = local index. For our
/// three stores, the object in that slot (still there at entry) is queued.
extern "system" fn on_slot_remove(frame: usize, _id: usize) {
    let Some(a) = ADDRS.get() else { return };
    // SAFETY: the stub's frame.
    let (store, local) = unsafe { (((frame - 24) as *const u64).read(), ((frame - 32) as *const u64).read() as u32 & 0x3F_FFFF) };
    if local == 0x3F_FFFF || !a.stores.contains(&store) {
        return;
    }
    let Some(object) = guarded(store, 0x68).and_then(|h| {
        let (pool, stride) = (le(&h, 0x38, 8), le(&h, 0x40, 4));
        (stride != 0 && stride <= 4096).then(|| q(pool.wrapping_add(stride * u64::from(local)))).flatten()
    }) else {
        return;
    };
    unload(object);
}

fn unload(object: u64) {
    if object == 0 {
        return;
    }
    let i = EV_UNLOAD_HEAD.fetch_add(1, Ordering::Relaxed);
    UNLOADS[(i as usize) & (UNLOAD_N - 1)].store(object, Ordering::Release);
}

// ---- entity drawable attach (the sharper "about to be drawn" signal) ----

/// 0x144F40940(entity): called by CreateDrawable 0x144F407B0 only when the
/// entity's draw handler was made (0x144F407B0 switches on entity+0x30, the
/// type byte; the caller 0x144F3FDE0 stores the handler at entity+0x28 after
/// it returns; entity+0x20 is the archetype: its +0x84 flags and +0x58 index
/// are read there) [V]. Log only: the entity, its type, archetype,
/// the archetype's u32 +0x58 (I: its model's store index) and, through the
/// draw handler, the drawable, its geometries, position and camera distance.
const ATTACH: usize = 0x1_44F4_0940;
const ATTACH_ENTRY: [u8; 15] = [0x56, 0x48, 0x83, 0xEC, 0x20, 0x48, 0x89, 0xCE, 0xE8, 0x03, 0x01, 0x00, 0x00, 0x84, 0xC0];
const FRAME_ID: usize = 0x1_4ABC_E2A0;
const ATTACH_N: usize = 1 << 15;
/// Each: entity, title frame id at the hook.
static ATTACHES: [[AtomicU64; 2]; ATTACH_N] = [const { [const { AtomicU64::new(0) }; 2] }; ATTACH_N];
static ATTACH_HEAD: AtomicU64 = AtomicU64::new(0);
static FRAME_AT: AtomicU64 = AtomicU64::new(0);

extern "system" fn on_attach(frame: usize, _id: usize) {
    // SAFETY: the stub's frame: rcx at frame-24.
    let entity = unsafe { ((frame - 24) as *const u64).read() };
    let f = FRAME_AT.load(Ordering::Relaxed);
    // SAFETY: a title global (always mapped) when set.
    let fid = if f != 0 { u64::from(unsafe { (f as *const u32).read_volatile() }) } else { 0 };
    attach(entity, fid);
}

fn attach(entity: u64, fid: u64) {
    let i = ATTACH_HEAD.fetch_add(1, Ordering::Relaxed) as usize & (ATTACH_N - 1);
    ATTACHES[i][1].store(fid, Ordering::Relaxed);
    ATTACHES[i][0].store(entity, Ordering::Release);
}

struct AttachDrain {
    tail: u64,
    rows: Vec<String>,
    n: u64,
    near: u64,
    by_type: HashMap<u8, u64>,
}
static ATTACH_DRAIN: Mutex<Option<AttachDrain>> = Mutex::new(None);

/// One attach row. The draw handler (entity+0x28, set by 0x144F3FDE0 just
/// after the hooked call) holds the drawable at +0x08 and the world matrix
/// rows at +0x40/+0x50/+0x60 with the position at +0x70 [V: 0x144FD0F40,
/// vt+0xB0 0x1454A45B0]: the entity's drawable, position and distance to
/// `cam` (the frame's main camera) need no float matching.
pub(crate) fn attach_row(frame: u64, fid: u64, entity: u64, cam: Option<[f32; 3]>) -> Option<(u8, String)> {
    let e = guarded(entity, 0x90)?;
    let (ty, arch, handler) = (e[0x30], le(&e, 0x20, 8), le(&e, 0x28, 8));
    let idx = if arch != 0 { guarded(arch + 0x58, 4).map_or(u64::MAX, |b| le(&b, 0, 4)) } else { u64::MAX };
    let h = if handler != 0 { guarded(handler, 0x80) } else { None };
    let drawable = h.as_ref().map_or(0, |h| le(h, 8, 8));
    let pos: [f32; 3] = h.as_ref().map_or([f32::NAN; 3], |h| std::array::from_fn(|k| f32::from_bits(le(h, 0x70 + 4 * k, 4) as u32)));
    let dist = cam.map_or(f32::NAN, |c| ((pos[0] - c[0]).powi(2) + (pos[1] - c[1]).powi(2) + (pos[2] - c[2]).powi(2)).sqrt());
    let geoms = if drawable != 0 { geometries(drawable, 512).len() } else { 0 };
    Some((ty, format!("{frame}\t{fid}\t{entity:#x}\t{ty}\t{arch:#x}\t{idx:#x}\t{handler:#x}\t{drawable:#x}\t{geoms}\t{:.1} {:.1} {:.1}\t{dist:.1}", pos[0], pos[1], pos[2])))
}

pub(crate) const ATTACH_HEADER: &str = "frame\ttitle_frame\tentity\ttype\tarchetype\tarch_0x58\thandler\tdrawable\tgeometries\tposition\tdistance\n";

/// Once a frame (writer thread): the attaches since the last frame as rows
/// (seq dumps: attach.tsv) and a count line every 60 frames.
pub(crate) fn drain_attaches(frame: u64, cam: Option<[f32; 3]>) {
    let Ok(mut g) = ATTACH_DRAIN.lock() else { return };
    let d = g.get_or_insert_with(|| AttachDrain { tail: 0, rows: Vec::new(), n: 0, near: 0, by_type: HashMap::new() });
    let head = ATTACH_HEAD.load(Ordering::Acquire);
    let from = d.tail.max(head.saturating_sub(ATTACH_N as u64));
    for i in from..head {
        let slot = &ATTACHES[i as usize & (ATTACH_N - 1)];
        let (entity, fid) = (slot[0].load(Ordering::Acquire), slot[1].load(Ordering::Relaxed));
        if let Some((ty, row)) = attach_row(frame, fid, entity, cam) {
            d.n += 1;
            *d.by_type.entry(ty).or_default() += 1;
            if row.rsplit('\t').next().and_then(|v| v.parse::<f32>().ok()).is_some_and(|v| v <= 200.0) {
                d.near += 1;
            }
            if d.rows.len() < 65_536 {
                d.rows.push(row);
            }
        }
    }
    d.tail = head;
    if frame % 60 == 0 && d.n > 0 {
        let mut t: Vec<(u8, u64)> = d.by_type.drain().collect();
        t.sort();
        xnw_core::info!("world ahead: frame {frame}: entity drawables attached {} ({} within 200 m of the camera; by entity type {t:?})", d.n, d.near);
        (d.n, d.near) = (0, 0);
    }
}

pub(crate) fn take_attach_rows() -> Vec<String> {
    ATTACH_DRAIN.lock().ok().and_then(|mut g| g.as_mut().map(|d| std::mem::take(&mut d.rows))).unwrap_or_default()
}

/// Hooks "mark loaded" when the entry bytes are the expected ones.
///
/// # Safety
/// `base` is the mapped title and none of its code runs yet.
pub(crate) unsafe fn install(base: usize) {
    if !enabled() {
        return;
    }
    let real = |va: usize| (base + (va - IMAGE_BASE)) as u64;
    let _ = ADDRS.set(Addrs { table: real(MODULE_TABLE), count: real(MODULE_COUNT), drawable_vt: real(DRAWABLE_STORE_VT), stores: [real(DRAWABLE_STORE), real(DWD_STORE), real(FRAGMENT_STORE)] });
    let _ = T0.set(std::time::Instant::now());
    let at = real(MARK_LOADED) as usize;
    let mut got = [0u8; 15];
    if !crate::watch::guarded_read(at, &mut got) || got != MARK_LOADED_ENTRY {
        xnw_core::warn!("world ahead: mark-loaded 0x1454BD9C0 entry {got:02x?} not the expected one; no upload ahead");
        return;
    }
    // SAFETY: nothing runs the title yet; a plain extern fn.
    match unsafe { crate::watch::hook_entry_with(at as *mut u8, HOOK_ID, on_mark_loaded as usize) } {
        Ok(_) => xnw_core::info!("world ahead: mark-loaded hooked at title+{:#x} (DrawableStore and DwdStore geometries offered ahead)", MARK_LOADED - IMAGE_BASE),
        Err(e) => xnw_core::warn!("world ahead: mark-loaded not hooked: {e}"),
    }
    // Entity drawable attach (log only).
    FRAME_AT.store(real(FRAME_ID), Ordering::Relaxed);
    let at = real(ATTACH) as usize;
    let mut got = [0u8; 15];
    if crate::watch::guarded_read(at, &mut got) && got == ATTACH_ENTRY {
        // SAFETY: as above.
        match unsafe { crate::watch::hook_entry_with(at as *mut u8, HOOK_ID + 3, on_attach as usize) } {
            Ok(_) => xnw_core::info!("world ahead: entity drawable attach hooked at title+{:#x} (attach.tsv)", ATTACH - IMAGE_BASE),
            Err(e) => xnw_core::warn!("world ahead: entity drawable attach not hooked: {e}"),
        }
    } else {
        xnw_core::warn!("world ahead: attach 0x144F40940 entry {got:02x?} not the expected one; no attach log");
    }
    // The unload path (store slot freed).
    let at = real(SLOT_REMOVE) as usize;
    let mut got = [0u8; 12];
    if crate::watch::guarded_read(at, &mut got) && got == SLOT_REMOVE_ENTRY {
        // SAFETY: as above.
        match unsafe { crate::watch::hook_entry_with(at as *mut u8, HOOK_ID + 2, on_slot_remove as usize) } {
            Ok(_) => xnw_core::info!("world ahead: store slot remove hooked at title+{:#x} (unloads drop our geometry)", SLOT_REMOVE - IMAGE_BASE),
            Err(e) => xnw_core::warn!("world ahead: store slot remove not hooked: {e}"),
        }
    } else {
        xnw_core::warn!("world ahead: store slot remove 0x142226DA0 entry {got:02x?} not the expected one; no unload path");
    }
    // The scene event ticks.
    let at = real(SCENE_EVENT) as usize;
    let mut got = [0u8; 19];
    if !crate::watch::guarded_read(at, &mut got) || got != SCENE_EVENT_ENTRY {
        xnw_core::warn!("world cine: scene event tick 0x1450FE750 entry {got:02x?} not the expected one; no event log");
        return;
    }
    // SAFETY: as above.
    match unsafe { crate::watch::hook_entry_with(at as *mut u8, HOOK_ID + 1, on_scene_event as usize) } {
        Ok(_) => xnw_core::info!("world cine: scene event tick hooked at title+{:#x}", SCENE_EVENT - IMAGE_BASE),
        Err(e) => xnw_core::warn!("world cine: scene event tick not hooked: {e}"),
    }
}

fn guarded(addr: u64, len: usize) -> Option<Vec<u8>> {
    if !(0x1_0000..0x8000_0000_0000).contains(&addr) || len > (64 << 20) {
        return None;
    }
    let mut b = vec![0u8; len];
    crate::watch::guarded_read(addr as usize, &mut b).then_some(b)
}

fn le(b: &[u8], at: usize, n: usize) -> u64 {
    b.get(at..at + n).map_or(0, |s| s.iter().rev().fold(0u64, |v, x| v << 8 | u64::from(*x)))
}

fn q(addr: u64) -> Option<u64> {
    guarded(addr, 8).map(|b| le(&b, 0, 8))
}

/// The module table's 16-byte entries {module*, u32 end} (read once a drain).
pub(crate) fn read_table(a: &Addrs) -> Option<Vec<u8>> {
    let table = q(a.table)?;
    let n = guarded(a.count, 2).map(|b| le(&b, 0, 2))? as usize;
    if n == 0 || n > 4096 {
        return None;
    }
    guarded(table, 16 * n)
}

/// strIndex -> (module object, its vtable, local index).
#[cfg(test)]
pub(crate) fn module_of(a: &Addrs, idx: u32) -> Option<(u64, u64, u32)> {
    module_in(&read_table(a)?, idx)
}

fn module_in(t: &[u8], idx: u32) -> Option<(u64, u64, u32)> {
    let idx = idx & 0x0FFF_FFFF;
    let n = t.len() / 16;
    // The first entry whose end is past the index (ends ascend: 0x144DA24E0's binary search).
    let i = (0..n).find(|&i| le(t, 16 * i + 8, 4) as u32 > idx)?;
    let m = le(&t, 16 * i, 8);
    let h = guarded(m, 0x10)?;
    Some((m, le(&h, 0, 8), idx.wrapping_sub(le(&h, 8, 4) as u32) & 0x3F_FFFF))
}

/// GetPtr of a DrawableStore-like pool: the object at `local` when its
/// occupancy bit is set.
pub(crate) fn get_ptr(m: &[u8], local: u32) -> Option<u64> {
    let (pool, stride, bits) = (le(m, 0x38, 8), le(m, 0x40, 4), le(m, 0x60, 8));
    let byte = guarded(bits + u64::from(local >> 3), 1)?[0];
    if byte >> (local & 7) & 1 == 0 || stride == 0 || stride > 4096 {
        return None;
    }
    q(pool.wrapping_add(stride * u64::from(local))).filter(|&p| p != 0)
}

/// The geometry pointers of a drawable (every LOD), at most `cap`.
pub(crate) fn geometries(drawable: u64, cap: usize) -> Vec<(u64, u8)> {
    let mut out = Vec::new();
    for lod in 0..4u64 {
        let Some(arr) = q(drawable + 0x50 + 8 * lod).filter(|&p| p != 0) else { continue };
        let Some(h) = guarded(arr, 10) else { continue };
        let (models, n) = (le(&h, 0, 8), le(&h, 8, 2) as usize);
        let Some(ms) = guarded(models, 8 * n.min(256)) else { continue };
        for k in 0..n.min(256) {
            let model = le(&ms, 8 * k, 8);
            let Some(mh) = guarded(model, 0x38) else { continue };
            let (geoms, ng) = (le(&mh, 8, 8), le(&mh, 0x36, 2) as usize);
            let Some(gs) = guarded(geoms, 8 * ng.min(256)) else { continue };
            for j in 0..ng.min(256) {
                let g = le(&gs, 8 * j, 8);
                if g != 0 && !out.iter().any(|e: &(u64, u8)| e.0 == g) {
                    out.push((g, lod as u8));
                    if out.len() >= cap {
                        return out;
                    }
                }
            }
        }
    }
    out
}

/// A drawable dictionary's drawables (pgDictionary: values atArray at +0x30
/// {ptr, u16 count} [I: the engine's lineage; each value checked by the reads]).
pub(crate) fn dictionary_geometries(dict: u64, cap: usize) -> Vec<(u64, u8)> {
    let mut out = Vec::new();
    let Some(h) = guarded(dict + 0x30, 10) else { return out };
    let (vals, n) = (le(&h, 0, 8), le(&h, 8, 2) as usize);
    let Some(v) = guarded(vals, 8 * n.min(256)) else { return out };
    for k in 0..n.min(256) {
        for g in geometries(le(&v, 8 * k, 8), cap - out.len()) {
            if !out.iter().any(|e| e.0 == g.0) {
                out.push(g);
            }
        }
        if out.len() >= cap {
            break;
        }
    }
    out
}

/// A store object's drawable and geometry pointers by store (0 drawable,
/// 1 drawable dictionary, 2 fragment), as 0x140441020 reads them [V].
pub(crate) fn object_geometries(kind: usize, object: u64, cap: usize) -> (u64, Vec<(u64, u8)>) {
    match kind {
        1 => (object, dictionary_geometries(object, cap)),
        2 => {
            let dr = q(object + 0x18).unwrap_or(0);
            (dr, if dr != 0 { geometries(dr, cap) } else { Vec::new() })
        }
        _ => (object, geometries(object, cap)),
    }
}

fn grc(obj: &[u8]) -> (u64, u64) {
    (le(obj, 0x18, 8), le(obj, 8, 4) * (le(obj, 0xC, 2) & 0xFFF))
}

/// The draw hook's key for a geometry (worldprobe.rs hook_draw), from guarded reads.
pub(crate) fn key_of_geometry(geom: u64) -> Option<GeomKey> {
    let g = guarded(geom, 0x48)?;
    let mut k = GeomKey { hash: le(&g, 0x30, 8), ..Default::default() };
    if let Some(vb) = guarded(le(&g, 8, 8), 0x20) {
        (k.vb_at, k.vb_len) = grc(&vb);
    }
    if let Some(buf) = q(le(&g, 0x40, 8)).and_then(|m| guarded(m, 0x20)) {
        (k.mesh_at, k.mesh_len) = grc(&buf);
    } else if let Some(ib) = guarded(le(&g, 0x10, 8), 0x20) {
        k.mesh_at = le(&ib, 0x18, 8);
        k.ib = le(&ib, 8, 4) << 8 | (le(&ib, 0xC, 2) & 0xFFF);
    }
    (k.vb_at != 0 && k.mesh_at != 0).then_some(k)
}

/// The bytes the pass uploads for `key` (as render_frame copies them).
pub(crate) fn host_geometry(key: &GeomKey) -> Option<HostGeometry> {
    let vb = guarded(key.vb_at, key.vb_len.min(64 << 20) as usize)?;
    // Not before the title has filled the header (None: offered again later).
    if !xnw_frame::vb_ready(&vb) {
        return None;
    }
    let indices = if key.mesh_len != 0 {
        xnw_frame::expand_meshlets(&guarded(key.mesh_at, key.mesh_len.min(16 << 20) as usize)?).ok()?
    } else {
        let (n, elem) = ((key.ib >> 8).min(3 << 20) as usize, (key.ib & 0xFF) as usize);
        let ib = guarded(key.mesh_at, n * elem)?;
        match elem {
            2 => ib.chunks_exact(2).map(|c| u32::from(u16::from_le_bytes([c[0], c[1]]))).collect(),
            4 => ib.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect(),
            _ => return None,
        }
    };
    Some(HostGeometry { vertices: Vec::new(), indices, raw_vb: Some(vb) })
}

/// Counters since the last log line, and totals.
#[derive(Default, Debug, Clone)]
pub(crate) struct Stats {
    pub(crate) placed: u64,
    pub(crate) drawables: u64,
    pub(crate) not_drawable: u64,
    pub(crate) no_module: u64,
    pub(crate) not_occupied: u64,
    pub(crate) late: u64,
    pub(crate) unloads: u64,
    pub(crate) unloads_ours: u64,
    pub(crate) keys_dropped: u64,
    pub(crate) fragments: u64,
    /// Capture mode: geometry keys read (no uploader) and their bytes.
    pub(crate) keyed: u64,
    pub(crate) keyed_bytes: u64,
    pub(crate) geometries: u64,
    pub(crate) offered: u64,
    pub(crate) offered_bytes: u64,
    pub(crate) known: u64,
    pub(crate) unreadable: u64,
    pub(crate) overrun: u64,
    /// Placed indices per module vtable (title rva) that are not the DrawableStore.
    pub(crate) modules: HashMap<u64, u64>,
}

struct Drain {
    tail: u64,
    /// Placed indices waiting for their turn (bounded), with their time.
    /// (ms, strIndex, tries): a mark whose store slot was not occupied yet
    /// is tried again at later drains, up to [`RETRIES`].
    queue: VecDeque<(u32, u32, u8)>,
    retry: Vec<(u32, u32, u8)>,
    /// Geometry keys per resolved store object (for the unload hook).
    owned: HashMap<u64, Vec<GeomKey>>,
    evict_tail: u64,
    stats: Stats,
    /// place.tsv rows since the last dump.
    rows: Vec<String>,
    /// ahead-keys.tsv rows since the last dump (capture mode).
    keys: Vec<String>,
}

static DRAIN: Mutex<Option<Drain>> = Mutex::new(None);

/// Per frame at most this many geometries and bytes are offered.
const FRAME_GEOMS: usize = 2048;
const FRAME_BYTES: u64 = 64 << 20;

/// Once a frame on the writer thread: placed drawables -> geometries
/// offered on `list` (`want` is the pass's `known().wants`).
pub(crate) fn drain(frame: u64, list: &mut FrameList, want: &mut dyn FnMut(&GeomKey) -> bool) {
    let Some(a) = ADDRS.get() else { return };
    drain_with(a, frame, list, want, OFFER.load(Ordering::Relaxed));
}

pub(crate) fn drain_with(a: &Addrs, frame: u64, list: &mut FrameList, want: &mut dyn FnMut(&GeomKey) -> bool, offer: bool) {
    let Ok(mut g) = DRAIN.lock() else { return };
    let d = g.get_or_insert_with(|| Drain { tail: 0, queue: VecDeque::new(), retry: Vec::new(), owned: HashMap::new(), evict_tail: 0, stats: Stats::default(), rows: Vec::new(), keys: Vec::new() });
    let head = HEAD.load(Ordering::Acquire);
    if head - d.tail > RING_N as u64 {
        d.stats.overrun += head - d.tail - RING_N as u64;
        d.tail = head - RING_N as u64;
    }
    while d.tail < head {
        let v = RING[(d.tail as usize) & (RING_N - 1)].load(Ordering::Acquire);
        d.tail += 1;
        if v != 0 && d.queue.len() < 65_536 {
            d.queue.push_back(((v >> 32) as u32, v as u32, 0));
        }
    }
    // Earlier marks whose slot was not occupied yet: first in line.
    for r in std::mem::take(&mut d.retry).into_iter().rev() {
        d.queue.push_front(r);
    }
    // Unloads (0x142226DA0) of objects we resolved: their keys leave our cache.
    let eh = EV_UNLOAD_HEAD.load(Ordering::Acquire);
    let from = d.evict_tail.max(eh.saturating_sub(UNLOAD_N as u64));
    let mut dropped = Vec::new();
    for i in from..eh {
        let o = UNLOADS[(i as usize) & (UNLOAD_N - 1)].load(Ordering::Acquire);
        d.stats.unloads += 1;
        if let Some(keys) = d.owned.remove(&o) {
            d.stats.unloads_ours += 1;
            dropped.extend(keys);
        }
    }
    d.evict_tail = eh;
    d.stats.keys_dropped += dropped.len() as u64;
    if offer && !dropped.is_empty() {
        xnw_world::geometry::request_evict(dropped);
    }
    let base = a.drawable_vt.wrapping_sub((DRAWABLE_STORE_VT - IMAGE_BASE) as u64);
    let (mut n, mut bytes) = (0usize, 0u64);
    let table = if d.queue.is_empty() { None } else { read_table(a) };
    while n < FRAME_GEOMS && bytes < FRAME_BYTES {
        let Some((ms, idx, tries)) = d.queue.pop_front() else { break };
        d.stats.placed += u64::from(tries == 0);
        let Some((m, vt, local)) = table.as_deref().and_then(|t| module_in(t, idx)) else {
            d.stats.no_module += 1;
            continue;
        };
        // The stores' run-time vtable is not 0x1461D0050 (the ctor's
        // last write; read it statically), so the 41,944 loads at world entry
        // all fell to "other": the store is known by its object [V].
        let kind = a.stores.iter().position(|&o| o == m);
        if kind.is_some() && vt != a.drawable_vt {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| xnw_core::info!("world ahead: store {:#x} has vtable {:#x} at run time (static read: {:#x})", m.wrapping_sub(base) as usize + IMAGE_BASE, vt.wrapping_sub(base) as usize + IMAGE_BASE, DRAWABLE_STORE_VT));
        }
        if kind.is_none() {
            d.stats.not_drawable += 1;
            *d.stats.modules.entry(m.wrapping_sub(base)).or_default() += 1;
            continue;
        }
        // Not occupied by drain time (unloaded again, or marked before its
        // object was set: 3,350 of 57,697 in one run).
        let Some(object) = guarded(m, 0x68).and_then(|h| get_ptr(&h, local)) else {
            if tries < RETRIES {
                d.retry.push((ms, idx, tries + 1));
            } else {
                d.stats.not_occupied += 1;
            }
            continue;
        };
        d.stats.late += u64::from(tries > 0);
        // 0x140441020 (strIndex -> drawable, all three stores) [V]: a
        // DrawableStore object is the drawable; a Dwd's first drawable is
        // **(obj+0x30) (values array, count u16 +0x38 [I]); a fragment's
        // drawable is *(obj+0x18) (also 0x1425DB670) [V].
        d.stats.fragments += u64::from(kind == Some(2));
        let (drawable, gs) = object_geometries(kind.unwrap_or(0), object, 512);
        d.stats.drawables += 1;
        let mut db = 0u64;
        let mut mine = Vec::new();
        for &(geom, lod) in &gs {
            d.stats.geometries += 1;
            let Some(key) = key_of_geometry(geom) else {
                d.stats.unreadable += 1;
                continue;
            };
            mine.push(key);
            if !offer {
                // Capture mode has no uploader: the key and its size only
                // (ahead-keys.tsv: which later draws these placements cover).
                let b = key.vb_len + if key.mesh_len != 0 { key.mesh_len } else { (key.ib >> 8) * (key.ib & 0xFF) };
                d.stats.keyed += 1;
                d.stats.keyed_bytes += b;
                db += b;
                if d.keys.len() < 65_536 {
                    // The first 4 KiB of the vertex buffer as placed (FNV-1a), to compare
                    // with the draw-time blob (were the bytes final at "mark loaded"?).
                    let vbh = guarded(key.vb_at, key.vb_len.min(4096) as usize).map_or(0, |v| v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ u64::from(x)).wrapping_mul(0x100_0000_01b3)));
                    d.keys.push(format!("{ms}\t{frame}\t{:#x}\t{:#x}\t{}\t{}\t{lod}\t{vbh:#x}", key.mesh_at, key.hash, b, kind.unwrap_or(0)));
                }
                continue;
            }
            if !want(&key) {
                d.stats.known += 1;
                continue;
            }
            match host_geometry(&key) {
                Some(hg) => {
                    let b = (hg.raw_vb.as_ref().map_or(0, Vec::len) + 4 * hg.indices.len()) as u64;
                    (bytes, db, n) = (bytes + b, db + b, n + 1);
                    d.stats.offered += 1;
                    d.stats.offered_bytes += b;
                    list.ahead_geoms.push((key, hg));
                }
                None => {
                    d.stats.unreadable += 1;
                    xnw_world::geometry::known().not_sent(&key);
                }
            }
        }
        if d.owned.len() > 262_144 {
            d.owned.clear();
        }
        d.owned.insert(object, mine);
        if d.rows.len() < 4096 {
            d.rows.push(format!("{ms}\t{frame}\t{idx:#x}\t{local}\t{drawable:#x}\t{}\t{db}", gs.len()));
        }
    }
    if frame % 60 == 0 {
        let s = std::mem::take(&mut d.stats);
        let mut mods: Vec<(u64, u64)> = s.modules.iter().map(|(k, v)| (*k, *v)).collect();
        mods.sort_by_key(|m| std::cmp::Reverse(m.1));
        let mods: Vec<String> = mods.iter().take(6).map(|(k, v)| format!("{:#x}: {v}", (*k as usize).wrapping_add(IMAGE_BASE))).collect();
        xnw_core::info!(
            "world ahead: frame {frame}: placed {}, drawables {} (of them fragments {}; {} geometries: offered {} ({:.1} MiB), already known {}, unreadable {}; keys only (no uploader) {} ({:.1} MiB)), other modules {} [{}], no module {}, not occupied after {RETRIES} drains {}, resolved on a retry {}, queued {}, ring overrun {}; unloads {} (ours {}, keys dropped {})",
            s.placed, s.drawables, s.fragments, s.geometries, s.offered, s.offered_bytes as f64 / f64::from(1 << 20), s.known, s.unreadable, s.keyed, s.keyed_bytes as f64 / f64::from(1 << 20), s.not_drawable, mods.join(", "), s.no_module, s.not_occupied, s.late, d.queue.len() + d.retry.len(), s.overrun, s.unloads, s.unloads_ours, s.keys_dropped
        );
    }
}

/// The place rows since the last call (seq dumps: place.tsv).
pub(crate) fn take_rows() -> Vec<String> {
    DRAIN.lock().ok().and_then(|mut g| g.as_mut().map(|d| std::mem::take(&mut d.rows))).unwrap_or_default()
}

/// The ahead-keys rows since the last call (seq dumps in capture mode).
pub(crate) fn take_key_rows() -> Vec<String> {
    DRAIN.lock().ok().and_then(|mut g| g.as_mut().map(|d| std::mem::take(&mut d.keys))).unwrap_or_default()
}

pub(crate) const KEYS_HEADER: &str = "ms\tframe\tmesh_at\thash\tbytes\tstore\tlod\tvb4k_fnv\n";

pub(crate) const PLACE_HEADER: &str = "ms\tframe\tstr_index\tlocal\tdrawable\tgeometries\tbytes_offered\n";

#[cfg(test)]
mod tests {
    use super::*;

    fn leak<T>(v: Vec<T>) -> &'static mut [T] {
        Box::leak(v.into_boxed_slice())
    }

    /// A synthetic module table, DrawableStore pool, drawable, model and
    /// two geometries (one meshlet, one classic): a "mark loaded" of its
    /// index offers both geometries under the draw hook's keys, once.
    /// Capture mode once installed nothing. The profiles' exact strings.
    #[test]
    #[ignore = "needs the capture run profiles (not published)"]
    fn capture_profiles_install_the_hooks_without_offering() {
        for (f, want) in [("null-capture-warp-103.env", (true, false)), ("null-render-capture-warp.env", (true, false))] {
            let text = std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/runs").join(f)).unwrap();
            let v = text.lines().find_map(|l| l.trim().strip_prefix("XNW_WORLD_RENDER=")).unwrap();
            assert_eq!(modes(Some(v)), want, "{f}: {v}");
        }
        assert_eq!(modes(Some("capture")), (true, false));
        assert_eq!(modes(Some("1")), (true, false));
        assert_eq!(modes(Some("screen,tex=off")), (true, false), "offers off by default");
        assert_eq!(modes(Some("screen,ahead=on")), (true, true));
        assert_eq!(modes(Some("screen,ahead=off")), (false, false));
        assert_eq!(modes(None), (false, false));
    }

    /// An attach of a synthetic entity becomes one row with its type,
    /// archetype, archetype +0x58, handler and the title frame.
    #[test]
    fn entity_attach_rows() {
        let arch = leak(vec![0u8; 0x60]);
        arch[0x58..0x5C].copy_from_slice(&0x1234u32.to_le_bytes());
        let handler = leak(vec![0u8; 0x80]);
        handler[8..16].copy_from_slice(&0x7777u64.to_le_bytes());
        for (k, v) in [3.0f32, 4.0, 0.0].iter().enumerate() {
            handler[0x70 + 4 * k..0x74 + 4 * k].copy_from_slice(&v.to_le_bytes());
        }
        let ent = leak(vec![0u8; 0x90]);
        ent[0x20..0x28].copy_from_slice(&(arch.as_ptr() as u64).to_le_bytes());
        ent[0x28..0x30].copy_from_slice(&(handler.as_ptr() as u64).to_le_bytes());
        ent[0x30] = 4;
        ent[0x50..0x54].copy_from_slice(&12.5f32.to_le_bytes());
        let _ = take_attach_rows();
        attach(ent.as_ptr() as u64, 777);
        drain_attaches(3, Some([0.0, 0.0, 0.0]));
        let rows = take_attach_rows();
        assert_eq!(rows.len(), 1, "{rows:?}");
        let f: Vec<&str> = rows[0].split('\t').collect();
        assert_eq!((f[1], f[3], f[5], f[7], f[9], f[10]), ("777", "4", "0x1234", "0x7777", "3.0 4.0 0.0", "5.0"), "{f:?}");
    }

    #[test]
    fn event_type_read_from_its_vtable() {
        // vt[6] = mov eax, 0x2A; ret
        let f = leak(vec![0xB8u8, 0x2A, 0, 0, 0, 0xC3]);
        let vt = leak(vec![0u64; 8]);
        vt[6] = f.as_ptr() as u64;
        let def = leak(vec![0u8; 0x34]);
        def[0x28..0x2C].copy_from_slice(&6.85f32.to_le_bytes());
        def[0x30..0x34].copy_from_slice(&(-1.0f32).to_le_bytes());
        let ev = leak(vec![vt.as_ptr() as u64, def.as_ptr() as u64]);
        assert_eq!(event_of(ev.as_ptr() as u64), Some((0x2A, 6.85, -1.0)));
        f[5] = 0x90;
        assert_eq!(event_of(ev.as_ptr() as u64).map(|e| e.0), Some(u32::MAX), "not the imm-return shape");
    }

    #[test]
    fn placed_drawable_offers_its_geometries() {
        // Geometry bytes: a VB of 3 RGB32F vertices with the raw header word.
        let vb_data = leak(vec![0u8; 0x80 + 36]);
        // A filled header (float3 positions), else the buffer is "not ready".
        vb_data[0..4].copy_from_slice(&0x3063u32.to_le_bytes());
        let vb_obj = leak(vec![0u8; 0x20]);
        vb_obj[8..12].copy_from_slice(&(((0x80 + 36) / 4) as u32).to_le_bytes());
        vb_obj[0xC..0xE].copy_from_slice(&4u16.to_le_bytes());
        vb_obj[0x18..0x20].copy_from_slice(&(vb_data.as_ptr() as u64).to_le_bytes());
        let ib_data = leak(vec![0u16, 1, 2]);
        let ib_obj = leak(vec![0u8; 0x20]);
        ib_obj[8..12].copy_from_slice(&3u32.to_le_bytes());
        ib_obj[0xC..0xE].copy_from_slice(&2u16.to_le_bytes());
        ib_obj[0x18..0x20].copy_from_slice(&(ib_data.as_ptr() as u64).to_le_bytes());
        let geom = |hash: u64| {
            let g = leak(vec![0u8; 0x48]);
            g[8..16].copy_from_slice(&(vb_obj.as_ptr() as u64).to_le_bytes());
            g[0x10..0x18].copy_from_slice(&(ib_obj.as_ptr() as u64).to_le_bytes());
            g[0x30..0x38].copy_from_slice(&hash.to_le_bytes());
            g.as_ptr() as u64
        };
        let geoms = leak(vec![geom(0xAA00), geom(0xBB00)]);
        let model = leak(vec![0u8; 0x38]);
        model[8..16].copy_from_slice(&(geoms.as_ptr() as u64).to_le_bytes());
        model[0x36..0x38].copy_from_slice(&2u16.to_le_bytes());
        let models = leak(vec![model.as_ptr() as u64]);
        let lod0 = leak(vec![0u8; 16]);
        lod0[..8].copy_from_slice(&(models.as_ptr() as u64).to_le_bytes());
        lod0[8..10].copy_from_slice(&1u16.to_le_bytes());
        let drawable = leak(vec![0u8; 0x70]);
        drawable[0x50..0x58].copy_from_slice(&(lod0.as_ptr() as u64).to_le_bytes());
        // The pool: stride 16, local 5 occupied.
        let pool = leak(vec![0u64; 16]);
        pool[10] = drawable.as_ptr() as u64;
        let bits = leak(vec![0b0010_0000u8, 0]);
        let vt = 0x7777_0000u64;
        let store = leak(vec![0u8; 0x68]);
        store[..8].copy_from_slice(&vt.to_le_bytes());
        store[8..12].copy_from_slice(&1000u32.to_le_bytes());
        store[0x38..0x40].copy_from_slice(&(pool.as_ptr() as u64).to_le_bytes());
        store[0x40..0x44].copy_from_slice(&16u32.to_le_bytes());
        store[0x60..0x68].copy_from_slice(&(bits.as_ptr() as u64).to_le_bytes());
        let other = leak(vec![0u8; 0x68]);
        other[..8].copy_from_slice(&0x8888_0000u64.to_le_bytes());
        // Module table: [0, 1000) other, [1000, 2000) the store.
        let table = leak(vec![0u8; 32]);
        table[..8].copy_from_slice(&(other.as_ptr() as u64).to_le_bytes());
        table[8..12].copy_from_slice(&1000u32.to_le_bytes());
        table[16..24].copy_from_slice(&(store.as_ptr() as u64).to_le_bytes());
        table[24..28].copy_from_slice(&2000u32.to_le_bytes());
        let table_ptr = leak(vec![table.as_ptr() as u64]);
        let count = leak(vec![2u16]);
        let a = Addrs { table: table_ptr.as_ptr() as u64, count: count.as_ptr() as u64, drawable_vt: vt, stores: [store.as_ptr() as u64, 1, 2] };
        assert_eq!(module_of(&a, 1005), Some((store.as_ptr() as u64, vt, 5)));
        assert_eq!(module_of(&a, 7).map(|m| m.1), Some(0x8888_0000));
        assert_eq!(module_of(&a, 2500), None, "past the last module");
        assert_eq!(get_ptr(store, 4), None, "not occupied");
        let with_lod: Vec<(u64, u8)> = geoms.iter().map(|g| (*g, 0)).collect();
        assert_eq!(geometries(drawable.as_ptr() as u64, 512), with_lod);
        // A fragment: its drawable at +0x18; a dictionary: values at +0x30, count +0x38.
        let frag = leak(vec![0u64; 4]);
        frag[3] = drawable.as_ptr() as u64;
        assert_eq!(object_geometries(2, frag.as_ptr() as u64, 512), (drawable.as_ptr() as u64, with_lod.clone()));
        let vals = leak(vec![drawable.as_ptr() as u64]);
        let dict = leak(vec![0u8; 0x40]);
        dict[0x30..0x38].copy_from_slice(&(vals.as_ptr() as u64).to_le_bytes());
        dict[0x38..0x3A].copy_from_slice(&1u16.to_le_bytes());
        assert_eq!(object_geometries(1, dict.as_ptr() as u64, 512).1, with_lod);
        let k = key_of_geometry(geoms[0]).unwrap();
        assert_eq!((k.hash, k.vb_at, k.vb_len, k.mesh_at, k.mesh_len, k.ib), (0xAA00, vb_data.as_ptr() as u64, (0x80 + 36) as u64, ib_data.as_ptr() as u64, 0, 3 << 8 | 2));
        let hg = host_geometry(&k);
        let vb_ok = guarded(k.vb_at, k.vb_len as usize).is_some();
        let ib_ok = guarded(k.mesh_at, 6).map(|b| b.len());
        assert!(hg.as_ref().is_some_and(|g| g.indices == [0, 1, 2]), "host geometry of {k:?}: vb read {vb_ok}, ib read {ib_ok:?}, got {:?}", hg.as_ref().map(|g| (g.indices.clone(), g.raw_vb.as_ref().map(Vec::len))));
        assert!(key_of_geometry(geoms[1]).is_some());
        // Through the ring and the drain: both geometries offered, then known.
        push(7); // another module
        push(1005);
        push(1004); // not occupied
        let mut list = FrameList::default();
        let mut seen = std::collections::HashSet::new();
        drain_with(&a, 1, &mut list, &mut |k| seen.insert(*k), true);
        let st = DRAIN.lock().unwrap().as_ref().map(|d| (d.stats.clone(), d.queue.len(), d.tail));
        assert_eq!(list.ahead_geoms.len(), 2, "{st:?} head {}", HEAD.load(Ordering::Relaxed));
        assert_eq!(list.ahead_geoms[1].0.hash, 0xBB00);
        assert_eq!(list.ahead_geoms[0].1.indices, vec![0, 1, 2]);
        let rows = take_rows();
        assert_eq!(rows.len(), 1, "{rows:?}");
        assert!(rows[0].contains(&format!("{:#x}", drawable.as_ptr() as u64)));
        push(1005);
        let mut list = FrameList::default();
        drain_with(&a, 2, &mut list, &mut |k| seen.insert(*k), true);
        assert!(list.ahead_geoms.is_empty(), "already sent: not offered again");
        let g = DRAIN.lock().unwrap();
        let s = &g.as_ref().unwrap().stats;
        assert_eq!((s.drawables, s.offered, s.known, s.not_drawable, s.not_occupied), (2, 2, 2, 1, 0));
        assert_eq!(g.as_ref().unwrap().retry.len(), 1, "1004 waits for its slot (retried later)");
        drop(g);
        // Its slot fills: the retry resolves it (offered: its keys are new to `seen`? no: the same geometries, already known).
        bits[0] |= 0b0001_0000;
        pool[8] = drawable.as_ptr() as u64;
        let mut list = FrameList::default();
        drain_with(&a, 3, &mut list, &mut |k| seen.insert(*k), true);
        assert_eq!(DRAIN.lock().unwrap().as_ref().unwrap().stats.late, 1);
        // The unload of the object we resolved: its keys are dropped from the pass.
        let _ = xnw_world::geometry::take_evict();
        unload(drawable.as_ptr() as u64);
        let mut list = FrameList::default();
        drain_with(&a, 4, &mut list, &mut |k| seen.insert(*k), true);
        let s = DRAIN.lock().unwrap().as_ref().unwrap().stats.clone();
        assert_eq!((s.unloads_ours, s.keys_dropped), (1, 2));
        let ev = xnw_world::geometry::take_evict();
        assert!(ev.iter().any(|k| k.hash == 0xAA00) && ev.iter().any(|k| k.hash == 0xBB00), "{ev:?}");
    }
}
