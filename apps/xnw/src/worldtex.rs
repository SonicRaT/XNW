//! At the geometry-draw hook, a draw's albedo descriptor
//! slot and its alpha-clip flag, for our world pass's textures.
//!
//! Recipe:
//! `sh` = TLS+0x1220, `fx = *sh`, template `t = *fx`; its variables are
//! 16-byte records at `*(t+0x50)`, `u8 t+4` of them, {hashA, hashB, u16
//! index +8, u16 member +10, u8 kind +12}. The albedo variable is the first
//! kind-0 match, by hashA then hashB, in the order of 0x145429040: 6994ef3f,
//! DiffuseTex 0xC8E8C282, 7d4f12e2, InteriorDiffuseMap 0x17D3B5C2 [V order,
//! I meaning]. Its value `texVars[index]` (`texVars = sh[2]`) is a 32-byte
//! view, or with bit 63 a streaming handle: record `qword_149C3E108 + 16 *
//! u16[h+6]`, resident when `min(rec[12], rec[14]) != 0`, view = `*(*rec +
//! 0x28)` (0x144EE12A0) [V]. The view's vtable is 0x1461ECDA0 [V]; its
//! descriptor slot `view+0x18` is what HD swaps and residency rewrite, so
//! it is read here, per draw [V]. Alpha clip: the effect name
//! (`*(fx+0xD08)`) names a cut-out class (cutout, trees, grass, fence) [I
//! class table; the `_AlphaClip` program is the exact test].
//!
//! Cost: per (shader, effect) the variable index and flag are found once
//! with guarded reads and kept in a fixed 4,096-entry table (no allocation
//! at the hook). Per draw the reads are the ones the title's own bind makes
//! for this draw (sh, texVars, the handle record, the view): direct, shape
//! checked, no syscall (as `dread`).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

const IMAGE_BASE: usize = 0x1_4000_0000;
const VIEW_VTBL: usize = 0x1_461E_CDA0;
const TEX_HANDLES: usize = 0x1_49C3_E108;
const ALBEDO_NAMES: [u32; 4] = [0x6994_EF3F, 0xC8E8_C282, 0x7D4F_12E2, 0x17D3_B5C2];
const CLIP_WORDS: [&str; 4] = ["cutout", "trees", "grass", "fence"];
const FX_NAME: u64 = 0xD08;
const NO_VAR: u16 = u16::MAX;

/// Rebased title addresses the hook compares and reads.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TexGlobals {
    /// The view vtable (0x1461ECDA0).
    pub(crate) view_vtbl: u64,
    /// The qword holding the handle record table (0x149C3E108).
    pub(crate) handles: u64,
}

static GLOBALS: OnceLock<TexGlobals> = OnceLock::new();

pub(crate) fn set_globals(title: usize) {
    let real = |va: usize| (title + (va - IMAGE_BASE)) as u64;
    let _ = GLOBALS.set(TexGlobals { view_vtbl: real(VIEW_VTBL), handles: real(TEX_HANDLES) });
}

pub(crate) fn globals() -> TexGlobals {
    GLOBALS.get().copied().unwrap_or_default()
}

/// `XNW_WORLD_RENDER=...,tex=off` skips the hook reads too.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| !std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| v.split(',').any(|p| p.trim() == "tex=off")));
    *ON
}

/// Per-draw outcomes: found, no albedo variable, not resident (0 or handle
/// not resident), not a view (vtable), table misses (guarded lookups).
pub(crate) static COUNTS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
const C_FOUND: usize = 0;
const C_NO_VAR: usize = 1;
const C_NOT_RESIDENT: usize = 2;
const C_NOT_VIEW: usize = 3;
const C_MISS: usize = 4;

thread_local! {
    /// This thread's outcome counts not yet added to [`COUNTS`], and how many.
    static PENDING: std::cell::Cell<([u32; 4], u32)> = const { std::cell::Cell::new(([0; 4], 0)) };
}

/// One draw's outcome, added to [`COUNTS`] 256 draws at a time per thread
/// (the shared counters, bumped per draw by every recording thread, were
/// part of the hook's cost); the 60-frame line lags by at most 255 a thread.
fn count(k: usize) {
    PENDING.with(|p| {
        let (mut a, mut n) = p.get();
        a[k] += 1;
        n += 1;
        if n >= 256 {
            for (i, v) in a.iter().enumerate().filter(|(_, v)| **v != 0) {
                COUNTS[i].fetch_add(u64::from(*v), Ordering::Relaxed);
            }
            (a, n) = ([0; 4], 0);
        }
        p.set((a, n));
    });
}

/// Counts since the last call, for the frame line: (found, no var, not
/// resident, not a view, lookups).
pub(crate) fn take_counts() -> [u64; 5] {
    std::array::from_fn(|i| COUNTS[i].swap(0, Ordering::Relaxed))
}

#[derive(Clone, Copy)]
struct Entry {
    shader: u64,
    fx: u64,
    var: u16,
    clip: bool,
    /// For layered effects (`*lyr*`), the texVars index of each
    /// layer's diffuse (NO_VAR: none).
    layers: [u16; 4],
}

const TABLE: usize = 4096;
const EMPTY: Entry = Entry { shader: 0, fx: 0, var: NO_VAR, clip: false, layers: [NO_VAR; 4] };
static KNOWN: Mutex<[Entry; TABLE]> = Mutex::new([EMPTY; TABLE]);

/// Per recording thread, direct-mapped, in front of [`KNOWN`] (made at the
/// thread's first draw: other threads carry only the empty slot).
const L1_N: usize = 1024;
thread_local! {
    static L1: std::cell::RefCell<Option<Box<[Entry; L1_N]>>> = const { std::cell::RefCell::new(None) };
}

/// Layer diffuse names, joaat [V hash of the name, I that these
/// are the names the layered effects use]: lyr0..3diffusetex (lineage: 7d4f12e2
/// = lyr1diffusetex), then DiffuseTex, DiffuseTex2..4. When no name matches,
/// the effect's kind-0 variables in template order stand in [I].
const LAYER_NAMES: [[u32; 2]; 4] = [[0x3463_94EA, 0xC8E8_C282], [0x7D4F_12E2, 0x0DF4_7048], [0x7D07_2B80, 0xE466_1D2C], [0x694D_88E2, 0x3918_C690]];

/// Layer lookups by name and by template order (counted once per effect).
pub(crate) static LAYER_LOOKUPS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

thread_local! {
    /// The last `albedo()` call's layer descriptor slots (0: none).
    static LAYERS: std::cell::Cell<[u64; 4]> = const { std::cell::Cell::new([0; 4]) };
}

/// The layer descriptor slots of the draw [`albedo`] was last asked for
/// on this thread (all 0 unless its effect is layered).
pub(crate) fn last_layers() -> [u64; 4] {
    LAYERS.with(std::cell::Cell::get)
}

/// A read of memory this draw's own bind reads: only the address's shape is
/// checked.
///
/// # Safety
/// `addr` is readable for `T`'s size.
unsafe fn dread<T: Copy>(addr: u64) -> Option<T> {
    // SAFETY: the caller's.
    (addr >= 0x1_0000 && addr < 0x8000_0000_0000).then(|| unsafe { (addr as *const T).read_unaligned() })
}

fn guarded(addr: u64, len: usize) -> Option<Vec<u8>> {
    if !(0x1_0000..0x8000_0000_0000).contains(&addr) {
        return None;
    }
    let mut b = vec![0u8; len];
    crate::watch::guarded_read(addr as usize, &mut b).then_some(b)
}

fn le(b: &[u8], at: usize, n: usize) -> u64 {
    b.get(at..at + n).map_or(0, |s| s.iter().rev().fold(0u64, |v, x| v << 8 | u64::from(*x)))
}

/// Once per (shader, effect): the albedo variable's texVars index and the
/// clip flag (guarded reads).
fn lookup(fx: u64) -> (u16, bool, [u16; 4]) {
    let name = guarded(fx.wrapping_add(FX_NAME), 8).map(|p| le(&p, 0, 8)).and_then(|p| guarded(p, 48)).map(|s| {
        let name: Vec<u8> = s.iter().copied().take_while(|b| *b != 0).map(|b| b.to_ascii_lowercase()).collect();
        String::from_utf8_lossy(&name).into_owned()
    }).unwrap_or_default();
    let clip = CLIP_WORDS.iter().any(|w| name.contains(w));
    let layered = name.contains("lyr");
    let none = (NO_VAR, clip, [NO_VAR; 4]);
    let Some(t) = guarded(fx, 8).map(|b| le(&b, 0, 8)) else { return none };
    let Some(head) = guarded(t, 0x58) else { return none };
    let n = usize::from(head[4]);
    let Some(vars) = guarded(le(&head, 0x50, 8), 16 * n) else { return none };
    let find = |name: u32| {
        let hit = |h: usize| (0..n).find(|i| le(&vars, 16 * i + h, 4) == u64::from(name) && vars[16 * i + 12] == 0);
        hit(0).or_else(|| hit(4)).map(|r| le(&vars, 16 * r + 8, 2) as u16)
    };
    let mut layers = [NO_VAR; 4];
    if layered {
        for (i, names) in LAYER_NAMES.iter().enumerate() {
            layers[i] = names.iter().find_map(|h| find(*h)).unwrap_or(NO_VAR);
        }
        if layers.iter().all(|l| *l == NO_VAR) {
            // No layer name: the kind-0 variables in template order [I].
            for (k, i) in (0..n).filter(|i| vars[16 * i + 12] == 0).take(4).enumerate() {
                layers[k] = le(&vars, 16 * i + 8, 2) as u16;
            }
            LAYER_LOOKUPS[1].fetch_add(1, Ordering::Relaxed);
        } else {
            LAYER_LOOKUPS[0].fetch_add(1, Ordering::Relaxed);
        }
    }
    let var = ALBEDO_NAMES.iter().find_map(|h| find(*h)).unwrap_or(NO_VAR);
    (var, clip, layers)
}

thread_local! {
    /// Census: the last `albedo()` outcome on this thread: 0 found,
    /// 1 no albedo variable, 2 not resident, 3 not a view, 4 no effect.
    static WHY: std::cell::Cell<u8> = const { std::cell::Cell::new(4) };
}

pub(crate) fn last_why() -> u8 {
    WHY.with(std::cell::Cell::get)
}

pub(crate) const WHY_NAMES: [&str; 5] = ["found", "no albedo variable", "not resident", "not a view", "no effect"];

/// The draw's albedo descriptor slot (0: none, not resident, not a view)
/// and its alpha-clip flag. `shader` is the TLS+0x1220 value of the draw.
/// Both halves at once: [`head`] (the draw's own shader object and texVars,
/// which the title reuses after the draw) then [`tail`] (the views they name).
pub(crate) fn albedo(g: &TexGlobals, shader: u64) -> (u64, bool) {
    let h = head(shader);
    tail(g, &h, false)
}

/// What [`albedo`] reads of the draw's own objects: the effect, the
/// variable's and the layers' texVars values. `XNW_HOOK_THIN=on` reads it at
/// the hook and leaves [`tail`] to our threads (worldprobe/thin.rs).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Head {
    /// The effect `*sh` (0: the shader was unreadable, no albedo).
    pub(crate) fx: u64,
    /// The albedo variable's texVars value (0: unreadable), when it has one.
    var: Option<u64>,
    clip: bool,
    /// Layered: each layer's texVars value (None: no such layer, unreadable).
    layered: bool,
    layers: [Option<u64>; 4],
}

/// The first half of [`albedo`]: the effect's variable (table, else guarded
/// lookups) and the draw's texVars values; the outcome "no albedo variable"
/// is counted here.
pub(crate) fn head(shader: u64) -> Head {
    WHY.with(|w| w.set(4));
    // SAFETY: the shader object of the draw being made.
    let Some(fx) = (unsafe { dread::<u64>(shader) }) else { return Head::default() };
    // Two ways per set (entries `s` and `s + 1`); a new entry goes to an
    // empty way, else first, the older one moving to the second.
    let h = (shader.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ fx).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let s = ((h >> 52) as usize % TABLE) & !1;
    // This thread's own copy first: the shared table's lock, taken per
    // draw by every recording thread, cost 0.46 us a draw on five threads
    // against 0.05 on one (albedo_cost_five_threads).
    let l1 = (h >> 40) as usize & (L1_N - 1);
    let mine = L1.with(|c| c.borrow().as_ref().map(|t| t[l1]).filter(|e| e.shader == shader && e.fx == fx));
    let known = mine.or_else(|| {
        let e = KNOWN.lock().ok().and_then(|t| [t[s], t[s + 1]].into_iter().find(|e| e.shader == shader && e.fx == fx));
        if let Some(e) = e {
            L1.with(|c| c.borrow_mut().get_or_insert_with(|| Box::new([EMPTY; L1_N]))[l1] = e);
        }
        e
    });
    let (var, clip, layers) = match known {
        Some(e) => (e.var, e.clip, e.layers),
        None => {
            COUNTS[C_MISS].fetch_add(1, Ordering::Relaxed);
            let (var, clip, layers) = lookup(fx);
            let e = Entry { shader, fx, var, clip, layers };
            L1.with(|c| c.borrow_mut().get_or_insert_with(|| Box::new([EMPTY; L1_N]))[l1] = e);
            if let Ok(mut t) = KNOWN.lock() {
                if t[s].shader == 0 {
                    t[s] = e;
                } else if t[s + 1].shader == 0 {
                    t[s + 1] = e;
                } else {
                    t[s + 1] = t[s];
                    t[s] = e;
                }
            }
            (var, clip, layers)
        }
    };
    // sh[2], the texVars array this draw binds from, and its entries.
    // SAFETY: the draw's own texVars (direct reads, as for the albedo below).
    let value = |var: u16| unsafe { dread::<u64>(shader.wrapping_add(0x10)).and_then(|tv| dread::<u64>(tv.wrapping_add(8 * u64::from(var)))) };
    let layered = !(layers[0] == NO_VAR && layers[1] == NO_VAR);
    let layers = if layered { layers.map(|l| if l == NO_VAR { None } else { value(l) }) } else { [None; 4] };
    if var == NO_VAR {
        count(C_NO_VAR);
        WHY.with(|w| w.set(1));
        return Head { fx, var: None, clip, layered, layers };
    }
    Head { fx, var: Some(value(var).unwrap_or(0)), clip, layered, layers }
}

/// The second half of [`albedo`]: the views the texVars values name (a
/// handle's residency record, the view's vtable and descriptor slot), the
/// layers' slots ([`last_layers`]) and the outcome. `fixed`: the reads are
/// fault-fixed (our threads, after the draw); else direct, as at the hook.
pub(crate) fn tail(g: &TexGlobals, h: &Head, fixed: bool) -> (u64, bool) {
    if h.fx == 0 {
        // As before the split: the layers of the last draw are left as they were.
        WHY.with(|w| w.set(4));
        return (0, false);
    }
    LAYERS.with(|c| c.set(if h.layered { h.layers.map(|v| v.and_then(|v| slot_of(g, v, fixed)).unwrap_or(0)) } else { [0; 4] }));
    let Some(v) = h.var else {
        WHY.with(|w| w.set(1));
        return (0, h.clip);
    };
    let view = resolve(g, v, fixed);
    if view == 0 {
        count(C_NOT_RESIDENT);
        WHY.with(|w| w.set(2));
        return (0, h.clip);
    }
    // SAFETY: the view this draw binds.
    let (vt, slot) = unsafe { (rd::<u64>(view, fixed), rd::<u64>(view.wrapping_add(0x18), fixed)) };
    if vt != Some(g.view_vtbl) || g.view_vtbl == 0 {
        count(C_NOT_VIEW);
        WHY.with(|w| w.set(3));
        return (0, h.clip);
    }
    count(C_FOUND);
    WHY.with(|w| w.set(0));
    (slot.unwrap_or(0), h.clip)
}

/// A texVars value's view: itself, or with bit 63 a streaming handle's
/// (record `table + 16 * u16[h+6]`, resident when `min(rec[12], rec[14])
/// != 0`, view `*(*rec + 0x28)`, as 0x144EE12A0 reads them); 0: none.
fn resolve(g: &TexGlobals, v: u64, fixed: bool) -> u64 {
    if v >> 63 == 0 {
        return v;
    }
    // SAFETY: the handle, its record and object, as 0x144EE12A0 reads them for this draw.
    unsafe {
        let h = v & !(1 << 63);
        let rec = rd::<u16>(h.wrapping_add(6), fixed).zip(rd::<u64>(g.handles, fixed)).map(|(i, table)| table.wrapping_add(16 * u64::from(i)));
        match rec.and_then(|r| rd::<[u8; 16]>(r, fixed)) {
            Some(r) if r[14].min(r[12]) != 0 => rd::<u64>(le(&r, 0, 8).wrapping_add(0x28), fixed).unwrap_or(0),
            _ => 0,
        }
    }
}

/// A direct read ([`dread`]) or, `fixed`, the fault-fixed copy of worldprobe.
///
/// # Safety
/// Without `fixed`: as [`dread`].
unsafe fn rd<T: Copy>(addr: u64, fixed: bool) -> Option<T> {
    // SAFETY: the caller's; the fixed read survives a fault.
    if fixed { unsafe { crate::worldprobe::fixed_read::<T>(addr) } } else { unsafe { dread::<T>(addr) } }
}

/// The descriptor slot of a layer's texVars value (as [`albedo`] resolves
/// its one variable): None when not resident or not a view.
fn slot_of(g: &TexGlobals, v: u64, fixed: bool) -> Option<u64> {
    let view = resolve(g, v, fixed);
    // SAFETY: as in `albedo`: the view this draw binds.
    unsafe {
        if view == 0 || g.view_vtbl == 0 || rd::<u64>(view, fixed)? != g.view_vtbl {
            return None;
        }
        rd::<u64>(view.wrapping_add(0x18), fixed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic title objects: shader -> effect -> template -> variables,
    /// texVars, a view (or a handle -> record -> object -> view).
    struct World {
        shader: Vec<u64>,
        _fx: Vec<u8>,
        _template: Vec<u8>,
        _vars: Vec<u8>,
        tex_vars: Vec<u64>,
        view: Vec<u64>,
        _name: Vec<u8>,
        handle: Vec<u16>,
        record: Vec<u8>,
        object: Vec<u64>,
        table: u64,
    }

    const VTBL: u64 = 0x1_461E_CDA0;

    fn world(name: &str, var_hash: u32, kind: u8) -> World {
        let mut w = World { shader: vec![0; 3], _fx: vec![0; 0xD10], _template: vec![0; 0x58], _vars: vec![0; 32], tex_vars: vec![0; 4], view: vec![VTBL, 0, 0, 0x5_1000_0020], _name: name.bytes().chain([0]).collect(), handle: vec![0; 4], record: vec![0; 16], object: vec![0; 6], table: 0 };
        // Two variables: a constant, then the texture at texVars[2].
        w._vars[0..4].copy_from_slice(&0x1111_1111u32.to_le_bytes());
        w._vars[12] = 3;
        w._vars[16..20].copy_from_slice(&0x2222_2222u32.to_le_bytes());
        w._vars[20..24].copy_from_slice(&var_hash.to_le_bytes());
        w._vars[24..26].copy_from_slice(&2u16.to_le_bytes());
        w._vars[28] = kind;
        w._template[4] = 2;
        w._template[0x50..0x58].copy_from_slice(&(w._vars.as_ptr() as u64).to_le_bytes());
        w._fx[0..8].copy_from_slice(&(w._template.as_ptr() as u64).to_le_bytes());
        w._fx[0xD08..0xD10].copy_from_slice(&(w._name.as_ptr() as u64).to_le_bytes());
        w.shader[0] = w._fx.as_ptr() as u64;
        w.shader[2] = w.tex_vars.as_ptr() as u64;
        w.tex_vars[2] = w.view.as_ptr() as u64;
        // The handle path: record 3 of the table, object +0x28 = the view.
        w.handle[3] = 3;
        w.object[5] = w.view.as_ptr() as u64;
        w.record[0..8].copy_from_slice(&(w.object.as_ptr() as u64).to_le_bytes());
        (w.record[12], w.record[14]) = (2, 1);
        w.table = w.record.as_ptr() as u64 - 48;
        w
    }

    fn g(w: &World) -> TexGlobals {
        TexGlobals { view_vtbl: VTBL, handles: &w.table as *const u64 as u64 }
    }

    #[test]
    fn albedo_slot_from_a_view_and_a_handle() {
        // Leaked: a freed world's addresses could come back with another test's.
        let w = Box::leak(Box::new(world("normal_spec", 0xC8E8_C282, 0)));
        let sh = w.shader.as_ptr() as u64;
        assert_eq!(albedo(&g(&w), sh), (0x5_1000_0020, false), "DiffuseTex by hashB, a plain view");
        // A resident handle.
        w.tex_vars[2] = 1 << 63 | w.handle.as_ptr() as u64;
        assert_eq!(albedo(&g(&w), sh).0, 0x5_1000_0020);
        // Not resident: resident level 0.
        w.record[14] = 0;
        assert_eq!(albedo(&g(&w), sh).0, 0);
        // Not a view (another vtable).
        w.tex_vars[2] = w.view.as_ptr() as u64;
        w.view[0] = 0x1234;
        assert_eq!(albedo(&g(&w), sh).0, 0);
    }

    #[test]
    fn no_albedo_variable_and_the_clip_flag() {
        let w = Box::leak(Box::new(world("ped_standard_Cutout", 0xD7C3_8241, 0)));
        assert_eq!(albedo(&g(&w), w.shader.as_ptr() as u64), (0, true), "BumpTexture only: no albedo; cut-out by name");
        assert_eq!(last_why(), 1);
        let w = Box::leak(Box::new(world("trees_lod", 0xC8E8_C282, 3)));
        assert_eq!(albedo(&g(&w), w.shader.as_ptr() as u64), (0, true), "DiffuseTex as a constant (kind 3) is not a texture");
        let w = Box::leak(Box::new(world("default", 0x17D3_B5C2, 0)));
        assert_eq!(albedo(&g(&w), w.shader.as_ptr() as u64), (0x5_1000_0020, false), "InteriorDiffuseMap");
        assert_eq!(albedo(&g(&w), 0), (0, false));
    }

    /// A layered effect (`*lyr*`) records its layer diffuse slots: by
    /// name (lyr0diffusetex as layer 0), else kind-0 variables in order.
    #[test]
    fn layered_effects_record_layer_slots() {
        let w = Box::leak(Box::new(world("standard_uber_4lyr_spx_dm_hbb", 0x3463_94EA, 0)));
        assert_eq!(albedo(&g(w), w.shader.as_ptr() as u64).0, 0, "no albedo variable");
        assert_eq!(last_layers(), [0x5_1000_0020, 0, 0, 0], "layer 0 by name");
        let w = Box::leak(Box::new(world("standard_2lyr_ground", 0x1234_5678, 0)));
        albedo(&g(w), w.shader.as_ptr() as u64);
        assert_eq!(last_layers()[0], 0x5_1000_0020, "no name: the first kind-0 variable");
        let w = Box::leak(Box::new(world("normal_spec", 0xC8E8_C282, 0)));
        albedo(&g(w), w.shader.as_ptr() as u64);
        assert_eq!(last_layers(), [0; 4], "not layered");
    }

    /// The per-draw cost: 5,000 draws over 500 shaders, table warm.
    #[test]
    fn albedo_cost_5000_draws() {
        let worlds: &Vec<World> = Box::leak(Box::new((0..500).map(|_| world("normal_spec", 0xC8E8_C282, 0)).collect()));
        let gl = g(&worlds[0]);
        for w in worlds {
            albedo(&gl, w.shader.as_ptr() as u64);
        }
        // The best of five frames (the host may run other tests alongside).
        let misses = COUNTS[C_MISS].load(Ordering::Relaxed);
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            let mut found = 0;
            for i in 0..5000 {
                found += usize::from(albedo(&gl, worlds[i % 500].shader.as_ptr() as u64).0 != 0);
            }
            best = best.min(t0.elapsed().as_secs_f64() * 1e3);
            assert_eq!(found, 5000);
        }
        let misses = COUNTS[C_MISS].load(Ordering::Relaxed).saturating_sub(misses);
        eprintln!("albedo slot: 5,000 draws in {best:.3} ms ({:.3} us/draw), {misses} table misses in 25,000 (other tests count too)", best * 1000.0 / 5000.0);
        assert!(best < 2.0, "{best} ms");
        assert!(misses < 2_500, "{misses} misses: the table thrashes");
    }

    /// The hook's case: five title threads at once over the same 500
    /// shaders (1,200 draws each, after a warm pass), time summed over the threads. Best of five.
    #[test]
    fn albedo_cost_five_threads() {
        let worlds: &Vec<World> = Box::leak(Box::new((0..500).map(|_| world("normal_spec", 0xC8E8_C282, 0)).collect()));
        let gl = g(&worlds[0]);
        for w in worlds {
            albedo(&gl, w.shader.as_ptr() as u64);
        }
        let mut best = f64::MAX;
        for _ in 0..5 {
            let ns: u64 = std::thread::scope(|sc| {
                let hs: Vec<_> = (0..5usize)
                    .map(|k| {
                        sc.spawn(move || {
                            // A long-lived title thread: its own copies warm.
                            for w in worlds {
                                albedo(&gl, w.shader.as_ptr() as u64);
                            }
                            let t0 = std::time::Instant::now();
                            let mut found = 0;
                            for i in 0..1200 {
                                found += usize::from(albedo(&gl, worlds[(k * 97 + i) % 500].shader.as_ptr() as u64).0 != 0);
                            }
                            assert_eq!(found, 1200);
                            t0.elapsed().as_nanos() as u64
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).sum()
            });
            best = best.min(ns as f64 / 1e6);
        }
        eprintln!("albedo slot, 5 threads x 1,200 draws: {best:.3} ms summed ({:.3} us/draw)", best * 1000.0 / 6000.0);
    }
}
