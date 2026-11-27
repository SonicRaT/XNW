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
//! Fallback (`XNW_ALBEDO_FALLBACK`, default on; `=off` as before): an effect
//! with none of the four names (ped_standard_muv*, prop_batch_slod*,
//! grass_imposter_slod, uber_4lyr: ~170 rows a frame, a test run) takes a known
//! colour name ([`COLOUR_NAMES`]), else its first kind-0 variable in
//! template order that is not a known non-colour name ([`NOT_COLOUR`]), as
//! the layered path does for layers [I]; outcome "fallback". Such effects'
//! kind-0 variables are logged once each (64 effects), joaat-labelled.
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
/// Colour texture names, joaat of the an earlier title shader names
/// [V hash, I meaning]: AlbedoMap, DiffuseTexture, BaseTexture, ColorTexture,
/// DiffuseSampler, imposterTexture, lyr0DiffuseTex, splatDiffuseTex, DiffuseHfTex;
/// VIS2: DiffuseTexture_layer0 (the uber_4lyr ground's first colour layer).
const COLOUR_NAMES: [u32; 10] = [0x6B9B_34BC, 0x2D14_F172, 0xA6EE_7CEB, 0x37AE_E870, 0xF1FE_2B71, 0x4E41_4593, 0x3463_94EA, 0xEAED_8EB0, 0x2118_3F1A, 0xCE7B_968E];
/// Known non-colour texture names the fallback skips (same source): bump,
/// normal, specular, detail, palette, mask, imposter normal/shadow/material;
/// VIS2: the uber_4lyr ground's MaterialMapATexture_layer0..3, lookupTexture,
/// BumpTexture_layer0..3 and DetailTexArrayN (an earlier title terrain_uber_4lyr declares
/// these with DiffuseTexture_layer0..3, tintTexture and tintBumpTexture; all
/// 15 named of the title's 17 kind-0 variables are in that set).
const NOT_COLOUR: [u32; 34] = [
    0xD7C3_8241, 0x1E77_BF4C, 0x46B7_C64F, 0x1A01_6C7E, 0xDFB4_2A92, 0xCDD7_9A22, 0x3EB8_A8A1, 0x84C6_9350, 0x58B6_E707, 0x9E37_22E4, 0xC6F4_6E79, 0xC1C0_1CC4,
    0x6087_99C6, 0x399A_985F, 0xCA42_99E4, 0x2C8D_7182, 0x6967_FE62, 0xBFDC_C9B6, 0x86BD_9FB0, 0x6239_614E, 0x0E37_38BB, 0x8B40_545A, 0x288E_DB30, 0x4300_C09F,
    0x3022_DE38, 0x4211_0214, 0x0CD6_179F, 0x3EA3_7B39, 0xC075_F9B6, 0x27DB_D2BE, 0x7E1F_7F44, 0x4C75_9BF1, 0xDE84_C011, 0x0CD2_7763,
];
/// Labels for the fallback log: the names above and the four of the title.
const LABELS: [(u32, &str); 51] = [
    (0x6994_EF3F, "albedo0(6994ef3f)"), (0xC8E8_C282, "DiffuseTex"), (0x7D4F_12E2, "lyr1DiffuseTex"), (0x17D3_B5C2, "InteriorDiffuseMap"),
    (0x6B9B_34BC, "AlbedoMap"), (0x2D14_F172, "DiffuseTexture"), (0xA6EE_7CEB, "BaseTexture"), (0x37AE_E870, "ColorTexture"), (0xF1FE_2B71, "DiffuseSampler"),
    (0x4E41_4593, "imposterTexture"), (0x3463_94EA, "lyr0DiffuseTex"), (0xEAED_8EB0, "splatDiffuseTex"), (0x2118_3F1A, "DiffuseHfTex"),
    (0xD7C3_8241, "BumpTexture"), (0x1E77_BF4C, "BumpTex"), (0x46B7_C64F, "BumpSampler"), (0x1A01_6C7E, "tintBumpTexture"), (0xDFB4_2A92, "NormalMap"),
    (0xCDD7_9A22, "NormalTexture"), (0x3EB8_A8A1, "NormalMapTex"), (0x84C6_9350, "lyr0NormalTex"), (0x58B6_E707, "lyr1NormalTex"), (0x9E37_22E4, "SpecularTexture"),
    (0xC6F4_6E79, "SpecularTex"), (0xC1C0_1CC4, "specTexture"), (0x6087_99C6, "SpecSampler"), (0x399A_985F, "DetailTexture"), (0xCA42_99E4, "DetailSampler"),
    (0x2C8D_7182, "TintPaletteTex"), (0x6967_FE62, "PaletteTexture"), (0xBFDC_C9B6, "albedoPaletteTex"), (0x86BD_9FB0, "skinToneBlendPaletteTex"),
    (0x6239_614E, "WrinkleMaskTexture"), (0x0E37_38BB, "MaskTex"), (0x8B40_545A, "imposterNormalTexture"), (0x288E_DB30, "imposterShadowTexture"),
    (0x4300_C09F, "imposterMaterialTexture"),
    (0xCE7B_968E, "DiffuseTexture_layer0"), (0xD883_AA9E, "DiffuseTexture_layer1"), (0x32B9_DF09, "DiffuseTexture_layer2"), (0x3D10_F3B7, "DiffuseTexture_layer3"),
    (0x3022_DE38, "MaterialMapATexture_layer0"), (0x4211_0214, "MaterialMapATexture_layer1"), (0x0CD6_179F, "MaterialMapATexture_layer2"), (0x3EA3_7B39, "MaterialMapATexture_layer3"),
    (0xC075_F9B6, "lookupTexture"), (0x27DB_D2BE, "BumpTexture_layer0"), (0x7E1F_7F44, "BumpTexture_layer1"), (0x4C75_9BF1, "BumpTexture_layer2"), (0xDE84_C011, "BumpTexture_layer3"),
    (0x0CD2_7763, "DetailTexArrayN"),
];
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
/// Live (SW1 `tex`, capped; asked per draw at the hook).
pub(crate) fn enabled() -> bool {
    xnw_core::live::on(xnw_core::live::TEX, tex_start)
}

/// SW1 live `tex` start: not `tex=off`.
pub(crate) fn tex_start() -> u32 {
    u32::from(!std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| v.split(',').any(|p| p.trim() == "tex=off")))
}

/// Per-draw outcomes: found, no albedo variable, not resident (0 or handle
/// not resident), not a view (vtable), table misses (guarded lookups),
/// found by the fallback.
pub(crate) static COUNTS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
const C_FOUND: usize = 0;
const C_NO_VAR: usize = 1;
const C_NOT_RESIDENT: usize = 2;
const C_NOT_VIEW: usize = 3;
const C_MISS: usize = 4;
const C_FALLBACK: usize = 5;

/// `XNW_ALBEDO_FALLBACK=off`: effects with none of the four names stay
/// without albedo (before ALB1). Default on.
fn fallback_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XNW_ALBEDO_FALLBACK").is_ok_and(|v| v.trim() == "off"))
}

thread_local! {
    /// This thread's outcome counts not yet added to [`COUNTS`], and how many.
    static PENDING: std::cell::Cell<([u32; 6], u32)> = const { std::cell::Cell::new(([0; 6], 0)) };
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
            (a, n) = ([0; 6], 0);
        }
        p.set((a, n));
    });
}

/// Counts since the last call, for the frame line: (found, no var, not
/// resident, not a view, lookups, fallback).
pub(crate) fn take_counts() -> [u64; 6] {
    std::array::from_fn(|i| COUNTS[i].swap(0, Ordering::Relaxed))
}

#[derive(Clone, Copy)]
struct Entry {
    shader: u64,
    fx: u64,
    var: u16,
    /// `var` came from the fallback (none of the four names).
    fallback: bool,
    clip: bool,
    /// For layered effects (`*lyr*`), the texVars index of each
    /// layer's diffuse (NO_VAR: none).
    layers: [u16; 4],
    /// Ped tint: TintPaletteTex's texVars index, the cbData index of
    /// TintPaletteSelector's buffer and the selector's byte offset in it
    /// (NO_VAR: none); the effect's class.
    tint: (u16, u16, u16),
    class: u8,
}

const TABLE: usize = 4096;
const EMPTY: Entry = Entry { shader: 0, fx: 0, var: NO_VAR, fallback: false, clip: false, layers: [NO_VAR; 4], tint: (NO_VAR, NO_VAR, NO_VAR), class: CLASS_OTHER };

/// Effect classes for the material features.
pub(crate) const CLASS_OTHER: u8 = 0;
pub(crate) const CLASS_PED: u8 = 1;
pub(crate) const CLASS_PAINT: u8 = 2;
/// WHL1: the instanced wheel and tyre materials. Their pixel programs are the
/// body's livery family (e668ad56, a2c80c6f, d68d7c01) and bind the parent
/// vehicle's own remap and T8+0x540 records [V pk8a f1027-1032 refs: remap
/// 0x9888e6320, base 23909, as the body rows]; no paint (their layers
/// drop where the table is drawn: precord).
pub(crate) const CLASS_WHEEL: u8 = 3;
/// NPC1: the far peds' `ped_slod_tint` (no texture: a per-ped palette by
/// vertex, read at its sink; `worldprobe/slodtint.rs`).
pub(crate) const CLASS_SLOD: u8 = 4;

/// The effect name's class: `ped*` (tint), the vehicle body materials
/// (`standard_vehicle`, `standard_vehicle_2lyr`, `standard_vehicle_2lyr_lod`:
/// paint) [I from a test run's census names].
pub(crate) fn class_of(name: &str) -> u8 {
    if name == "ped_slod_tint" {
        CLASS_SLOD
    } else if name.starts_with("ped") {
        CLASS_PED
    } else if matches!(name, "standard_vehicle" | "standard_vehicle_2lyr" | "standard_vehicle_2lyr_lod") {
        CLASS_PAINT
    } else if matches!(name, "standard_vehicle_2lyr_wheel_inst" | "standard_vehicle_2lyr_tyre_inst" | "standard_vehicle_2lyr_tyre_inst_lod") {
        CLASS_WHEEL
    } else {
        CLASS_OTHER
    }
}

/// INT1/WHL1: an effect class whose sink is asked for the vehicle material table.
pub(crate) fn vehmat_class(class: u8) -> bool {
    class == CLASS_PAINT || class == CLASS_WHEEL
}

/// TintPaletteTex (kind 0) and TintPaletteSelector (kind 3,
/// a constant buffer variable) joaat.
const TINT_PALETTE: u32 = 0x2C8D_7182;
const TINT_SELECTOR: u32 = 0xFDD7_96CF;

/// `XNW_WORLD_RENDER=...,paint=off` (read once): no vehicle handler
/// hooks, vehicle 2lyr materials layered as before.
pub(crate) fn paint_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| v.split(',').any(|p| p.trim() == "paint=off")))
}

/// W16P: ped draws by tint outcome, since the last [`take_tint`]: ped-effect
/// draws, no tint variables (not a _Tint effect), selector block unreadable,
/// block not a tint (neither at the selector nor 4 bytes before it),
/// palette not resolved, tinted.
pub(crate) static TINT_WHY: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
pub(crate) const TINT_NAMES: [&str; 6] = ["ped draws", "no tint vars", "block unreadable", "not a tint", "palette unresolved", "tinted"];

pub(crate) fn take_tint() -> [u64; 6] {
    std::array::from_fn(|i| TINT_WHY[i].swap(0, Ordering::Relaxed))
}

/// One-shot logs: ped effects' tint variables (first 16) and rejected tint
/// blocks (first 8).
static TINT_LOGS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// once per (shader, effect) beside [`lookup`]: the class and, for a
/// ped effect, the tint variables (guarded reads; the template walk as
/// `lookup` makes it). W16P: the selector is a constant at
/// `cbData[r.index] + *(u16*)(fx->varDecl + 24*(n + r.member) + 12)`,
/// n = `*(u16*)varDecl`, varDecl = `fx+0xC88` (sec. 1, 0x145429040 [V]);
/// read `cbData[r.index] + 0`. An unreadable declaration keeps offset 0.
fn extra_lookup(fx: u64) -> ((u16, u16, u16), u8) {
    let name = guarded(fx.wrapping_add(FX_NAME), 8).map(|p| le(&p, 0, 8)).and_then(|p| guarded(p, 48)).map(|s| {
        let name: Vec<u8> = s.iter().copied().take_while(|b| *b != 0).map(|b| b.to_ascii_lowercase()).collect();
        String::from_utf8_lossy(&name).into_owned()
    }).unwrap_or_default();
    let class = class_of(&name);
    let none = ((NO_VAR, NO_VAR, NO_VAR), class);
    if class != CLASS_PED {
        return none;
    }
    let Some(t) = guarded(fx, 8).map(|b| le(&b, 0, 8)) else { return none };
    let Some(head) = guarded(t, 0x58) else { return none };
    let n = usize::from(head[4]);
    let Some(vars) = guarded(le(&head, 0x50, 8), 16 * n) else { return none };
    let find = |name: u32, kind: u8| {
        let hit = |h: usize| (0..n).find(|i| le(&vars, 16 * i + h, 4) == u64::from(name) && vars[16 * i + 12] == kind);
        hit(0).or_else(|| hit(4))
    };
    let palette = find(TINT_PALETTE, 0).map_or(NO_VAR, |r| le(&vars, 16 * r + 8, 2) as u16);
    let sel = find(TINT_SELECTOR, 3);
    let (index, member) = sel.map_or((NO_VAR, NO_VAR), |r| (le(&vars, 16 * r + 8, 2) as u16, le(&vars, 16 * r + 10, 2) as u16));
    let off = sel
        .and_then(|_| guarded(fx.wrapping_add(0xC88), 8))
        .map(|p| le(&p, 0, 8))
        .and_then(|decl| guarded(decl, 2).map(|c| (decl, le(&c, 0, 2))))
        .and_then(|(decl, cn)| guarded(decl.wrapping_add(24 * (cn + u64::from(member)) + 12), 2))
        .map_or(0, |b| le(&b, 0, 2) as u16);
    if TINT_LOGS[0].fetch_add(1, Ordering::Relaxed) < 16 {
        // Every record naming the selector, any kind: (kind, index, member).
        let any: Vec<String> = (0..n).filter(|i| le(&vars, 16 * i, 4) == u64::from(TINT_SELECTOR) || le(&vars, 16 * i + 4, 4) == u64::from(TINT_SELECTOR)).map(|i| format!("kind {} index {} member {}", vars[16 * i + 12], le(&vars, 16 * i + 8, 2), le(&vars, 16 * i + 10, 2))).collect();
        xnw_core::info!("tint vars: effect {name}: {n} vars; TintPaletteTex texVars {}; TintPaletteSelector [{}] -> cbData[{}] + {off}", if palette == NO_VAR { "none".to_string() } else { palette.to_string() }, any.join("; "), if index == NO_VAR { "none".to_string() } else { index.to_string() });
    }
    ((palette, index, if index == NO_VAR { NO_VAR } else { off }), class)
}

/// W16P: a ped draw's tint at the hook (counted by outcome in [`TINT_WHY`]):
/// the palette's texVars value, rows and 1/rows.
fn tint_read(shader: u64, fx: u64, tint: (u16, u16, u16), value: &dyn Fn(u16) -> Option<u64>) -> Option<(u64, [u8; 3], f32)> {
    TINT_WHY[0].fetch_add(1, Ordering::Relaxed);
    if tint.0 == NO_VAR || tint.1 == NO_VAR {
        TINT_WHY[1].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let off = u64::from(tint.2);
    // SAFETY: the draw's own cbData array and the block it binds (direct reads, as above).
    let Some(base) = (unsafe { dread::<u64>(shader.wrapping_add(8)).and_then(|cbs| dread::<u64>(cbs.wrapping_add(8 * u64::from(tint.1)))) }) else {
        TINT_WHY[2].fetch_add(1, Ordering::Relaxed);
        return None;
    };
    // The selector as the rows word (1/rows the 4 bytes before it), then as 1/rows itself;
    // never past selector + 8 bytes.
    // SAFETY: inside the block the selector lives in (as above).
    let before = if off >= 4 { unsafe { dread::<[u8; 8]>(base.wrapping_add(off - 4)) } } else { None };
    // SAFETY: as above.
    let here = unsafe { dread::<[u8; 8]>(base.wrapping_add(off)) };
    if before.is_none() && here.is_none() {
        TINT_WHY[2].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let Some((rows, inv)) = before.and_then(tint_block).or_else(|| here.and_then(tint_block)) else {
        TINT_WHY[3].fetch_add(1, Ordering::Relaxed);
        if TINT_LOGS[1].fetch_add(1, Ordering::Relaxed) < 8 {
            xnw_core::info!("tint block rejected: fx {fx:#x}: cbData[{}] {base:#x} + {off}: at -4 {before:02x?}, at 0 {here:02x?}", tint.1);
        }
        return None;
    };
    let Some(v) = value(tint.0) else {
        TINT_WHY[4].fetch_add(1, Ordering::Relaxed);
        return None;
    };
    Some((v, rows, inv))
}

/// What the last [`tail`] found for the material features.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Extra {
    pub(crate) class: u8,
    /// Ped tint: the palette's descriptor slot (0: none), rows r0..r2
    /// (0xFF: none) and 1/rows, from the b4 block ([V layout]; read as
    /// cbData[selector's buffer] +0 [I which buffer], validated).
    pub(crate) tint_slot: u64,
    pub(crate) tint_rows: [u8; 3],
    pub(crate) tint_inv: f32,
}

pub(crate) const NO_EXTRA: Extra = Extra { class: CLASS_OTHER, tint_slot: 0, tint_rows: [0xFF; 3], tint_inv: 0.0 };

impl Default for Extra {
    fn default() -> Self {
        NO_EXTRA
    }
}

thread_local! {
    static EXTRA: std::cell::Cell<Extra> = const { std::cell::Cell::new(NO_EXTRA) };
}

/// The material features of the draw [`tail`] (or [`albedo`]) last
/// resolved on this thread.
pub(crate) fn last_extra() -> Extra {
    EXTRA.with(std::cell::Cell::get)
}

/// A b4 block's first 8 bytes as the tint: 1/rows (rows 1..=256,
/// a whole number) and rows r0..r2 (0xFF none, else below rows); None when
/// the bytes do not read as one.
pub(crate) fn tint_block(b: [u8; 8]) -> Option<([u8; 3], f32)> {
    let inv = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    if !(inv.is_finite() && inv > 0.0 && inv <= 1.0) {
        return None;
    }
    let rows = (1.0 / inv).round();
    if !(1.0..=256.0).contains(&rows) || ((1.0 / inv) - rows).abs() > 0.01 {
        return None;
    }
    let r = [b[4], b[5], b[6]];
    r.iter().all(|x| *x == 0xFF || f32::from(*x) < rows).then_some((r, inv))
}
static KNOWN: Mutex<[Entry; TABLE]> = Mutex::new([EMPTY; TABLE]);

/// Per recording thread, direct-mapped, in front of [`KNOWN`] (made at the
/// thread's first draw: other threads carry only the empty slot).
const L1_N: usize = 1024;
thread_local! {
    static L1: std::cell::RefCell<Option<Box<[Entry; L1_N]>>> = const { std::cell::RefCell::new(None) };
}

/// Layer diffuse names, joaat [V hash of the name, I that these
/// are the names the layered effects use]: lyr0..3diffusetex (lineage: 7d4f12e2
/// = lyr1diffusetex), then DiffuseTex, DiffuseTex2..4, then (VIS2) the uber_4lyr
/// ground's DiffuseTexture_layer0..3. When no name matches, the effect's kind-0
/// variables in template order stand in, known non-colour names skipped [I].
/// VIS2: before the uber names, the uber ground (kerbs, verges, planters) took
/// @0..@3 = MaterialMapATexture_layer1, _layer3, lookupTexture, 07bc8533 as its
/// colour layers (a test run `albedo names` line): the red/teal patches.
const LAYER_NAMES: [[u32; 3]; 4] = [
    [0x3463_94EA, 0xC8E8_C282, 0xCE7B_968E],
    [0x7D4F_12E2, 0x0DF4_7048, 0xD883_AA9E],
    [0x7D07_2B80, 0xE466_1D2C, 0x32B9_DF09],
    [0x694D_88E2, 0x3918_C690, 0x3D10_F3B7],
];

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

/// The albedo variable of a template's variable table (`n` 16-byte records):
/// its texVars index (NO_VAR: none) and whether the fallback chose it. By
/// name: the first of [`ALBEDO_NAMES`] a kind-0 record carries as hashA,
/// then as hashB. Else, `fallback`: a [`COLOUR_NAMES`] hit, else the first
/// kind-0 record whose hashes are not [`NOT_COLOUR`].
fn pick(vars: &[u8], n: usize, fallback: bool) -> (u16, bool) {
    let n = n.min(vars.len() / 16);
    let kind0 = move |i: usize| vars[16 * i + 12] == 0;
    let hashes = |i: usize| [le(vars, 16 * i, 4) as u32, le(vars, 16 * i + 4, 4) as u32];
    let index = |i: usize| le(vars, 16 * i + 8, 2) as u16;
    let find = |name: u32| {
        let hit = |h: usize| (0..n).find(|i| kind0(*i) && hashes(*i)[h] == name);
        hit(0).or_else(|| hit(1)).map(index)
    };
    if let Some(v) = ALBEDO_NAMES.iter().find_map(|h| find(*h)) {
        return (v, false);
    }
    if !fallback {
        return (NO_VAR, false);
    }
    let v = COLOUR_NAMES.iter().find_map(|h| find(*h)).or_else(|| (0..n).find(|i| kind0(*i) && !hashes(*i).iter().any(|h| NOT_COLOUR.contains(h))).map(index));
    v.map_or((NO_VAR, false), |v| (v, true))
}

/// One log line per effect with none of the four names (first 64): its
/// kind-0 variables `hashA/hashB@index` (label), and the fallback's choice.
fn log_unnamed(name: &str, vars: &[u8], n: usize, chosen: u16) {
    static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let Ok(mut seen) = SEEN.lock() else { return };
    if seen.len() >= 64 || seen.iter().any(|s| s == name) {
        return;
    }
    seen.push(name.to_string());
    let label = |h: u32| LABELS.iter().find(|l| l.0 == h).map(|l| format!(" ({})", l.1)).unwrap_or_default();
    let list: Vec<String> = (0..n.min(vars.len() / 16))
        .filter(|i| vars[16 * i + 12] == 0)
        .map(|i| {
            let (a, b) = (le(vars, 16 * i, 4) as u32, le(vars, 16 * i + 4, 4) as u32);
            format!("{a:08x}/{b:08x}@{}{}{}", le(vars, 16 * i + 8, 2), label(a), label(b))
        })
        .collect();
    let pick = if chosen == NO_VAR { "none".to_string() } else { format!("@{chosen}") };
    xnw_core::info!("albedo names: effect {name}: {n} vars, kind-0 vars {}; fallback {pick}", list.join(" "));
}

/// A layered effect's layer texVars indices and whether a name found them:
/// [`LAYER_NAMES`] (kind 0, hashA then hashB), else the first four kind-0
/// variables in template order that are not [`NOT_COLOUR`] names.
fn layer_vars(vars: &[u8], n: usize) -> ([u16; 4], bool) {
    let n = n.min(vars.len() / 16);
    let kind0 = |i: usize| vars[16 * i + 12] == 0;
    let hashes = |i: usize| [le(vars, 16 * i, 4) as u32, le(vars, 16 * i + 4, 4) as u32];
    let index = |i: usize| le(vars, 16 * i + 8, 2) as u16;
    let find = |name: u32| {
        let hit = |h: usize| (0..n).find(|i| kind0(*i) && hashes(*i)[h] == name);
        hit(0).or_else(|| hit(1)).map(index)
    };
    let mut layers = [NO_VAR; 4];
    for (i, names) in LAYER_NAMES.iter().enumerate() {
        layers[i] = names.iter().find_map(|h| find(*h)).unwrap_or(NO_VAR);
    }
    if layers.iter().any(|l| *l != NO_VAR) {
        return (layers, true);
    }
    for (k, i) in (0..n).filter(|i| kind0(*i) && !hashes(*i).iter().any(|h| NOT_COLOUR.contains(h))).take(4).enumerate() {
        layers[k] = index(i);
    }
    (layers, false)
}

/// Once per (shader, effect): the albedo variable's texVars index, whether
/// the fallback chose it, the clip flag and the layers (guarded reads).
fn lookup(fx: u64) -> (u16, bool, bool, [u16; 4]) {
    let name = guarded(fx.wrapping_add(FX_NAME), 8).map(|p| le(&p, 0, 8)).and_then(|p| guarded(p, 48)).map(|s| {
        let name: Vec<u8> = s.iter().copied().take_while(|b| *b != 0).map(|b| b.to_ascii_lowercase()).collect();
        String::from_utf8_lossy(&name).into_owned()
    }).unwrap_or_default();
    let clip = CLIP_WORDS.iter().any(|w| name.contains(w));
    let layered = name.contains("lyr");
    let none = (NO_VAR, false, clip, [NO_VAR; 4]);
    let Some(t) = guarded(fx, 8).map(|b| le(&b, 0, 8)) else { return none };
    let Some(head) = guarded(t, 0x58) else { return none };
    let n = usize::from(head[4]);
    let Some(vars) = guarded(le(&head, 0x50, 8), 16 * n) else { return none };
    let layers = if layered {
        let (layers, by_name) = layer_vars(&vars, n);
        LAYER_LOOKUPS[usize::from(!by_name)].fetch_add(1, Ordering::Relaxed);
        layers
    } else {
        [NO_VAR; 4]
    };
    let (var, fallback) = pick(&vars, n, fallback_on());
    if var == NO_VAR || fallback {
        log_unnamed(&name, &vars, n, var);
    }
    (var, fallback, clip, layers)
}

thread_local! {
    /// Census: the last `albedo()` outcome on this thread: 0 found,
    /// 1 no albedo variable, 2 not resident, 3 not a view, 4 no effect,
    /// 5 found by the fallback.
    static WHY: std::cell::Cell<u8> = const { std::cell::Cell::new(4) };
}

pub(crate) fn last_why() -> u8 {
    WHY.with(std::cell::Cell::get)
}

pub(crate) const WHY_NAMES: [&str; 6] = ["found", "no albedo variable", "not resident", "not a view", "no effect", "fallback"];

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
    /// `var` is the fallback's (outcome "fallback" when it resolves).
    fallback: bool,
    clip: bool,
    /// Layered: each layer's texVars value (None: no such layer, unreadable).
    layered: bool,
    layers: [Option<u64>; 4],
    /// The effect's class; ped tint: the palette's texVars value, rows
    /// and 1/rows (read at the hook: the b4 block is the draw's).
    pub(crate) class: u8,
    tint: Option<(u64, [u8; 3], f32)>,
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
    let (var, fallback, clip, layers, tint, class) = match known {
        Some(e) => (e.var, e.fallback, e.clip, e.layers, e.tint, e.class),
        None => {
            COUNTS[C_MISS].fetch_add(1, Ordering::Relaxed);
            let (var, fallback, clip, mut layers) = lookup(fx);
            let (tint, class) = extra_lookup(fx);
            // Vehicles' 2lyr materials are paint and dirt, not ground layers (paint on).
            if class == CLASS_PAINT && paint_enabled() {
                layers = [NO_VAR; 4];
            }
            let e = Entry { shader, fx, var, fallback, clip, layers, tint, class };
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
            (var, fallback, clip, layers, tint, class)
        }
    };
    // sh[2], the texVars array this draw binds from, and its entries.
    // SAFETY: the draw's own texVars (direct reads, as for the albedo below).
    let value = |var: u16| unsafe { dread::<u64>(shader.wrapping_add(0x10)).and_then(|tv| dread::<u64>(tv.wrapping_add(8 * u64::from(var)))) };
    let layered = !(layers[0] == NO_VAR && layers[1] == NO_VAR);
    let layers = if layered { layers.map(|l| if l == NO_VAR { None } else { value(l) }) } else { [None; 4] };
    // The tint block: sh[1] (the draw's cbData array), the selector's buffer at the
    // selector's offset (W16P): 1/rows then the rows there, or the selector is the rows word
    // and 1/rows the 4 bytes before it.
    let tint = if class == CLASS_PED { tint_read(shader, fx, tint, &value) } else { None };
    if var == NO_VAR {
        count(C_NO_VAR);
        WHY.with(|w| w.set(1));
        return Head { fx, var: None, fallback: false, clip, layered, layers, class, tint };
    }
    Head { fx, var: Some(value(var).unwrap_or(0)), fallback, clip, layered, layers, class, tint }
}

/// The second half of [`albedo`]: the views the texVars values name (a
/// handle's residency record, the view's vtable and descriptor slot), the
/// layers' slots ([`last_layers`]) and the outcome. `fixed`: the reads are
/// fault-fixed (our threads, after the draw); else direct, as at the hook.
pub(crate) fn tail(g: &TexGlobals, h: &Head, fixed: bool) -> (u64, bool) {
    if h.fx == 0 {
        // As before the split: the layers of the last draw are left as they were.
        WHY.with(|w| w.set(4));
        EXTRA.with(|c| c.set(NO_EXTRA));
        return (0, false);
    }
    LAYERS.with(|c| c.set(if h.layered { h.layers.map(|v| v.and_then(|v| slot_of(g, v, fixed)).unwrap_or(0)) } else { [0; 4] }));
    // The class and the tint palette's slot.
    let mut extra = Extra { class: h.class, ..NO_EXTRA };
    if let Some((v, rows, inv)) = h.tint {
        if let Some(slot) = slot_of(g, v, fixed).filter(|s| *s != 0) {
            (extra.tint_slot, extra.tint_rows, extra.tint_inv) = (slot, rows, inv);
            TINT_WHY[5].fetch_add(1, Ordering::Relaxed);
        } else {
            TINT_WHY[4].fetch_add(1, Ordering::Relaxed);
        }
    }
    EXTRA.with(|c| c.set(extra));
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
    let (c, why) = if h.fallback { (C_FALLBACK, 5) } else { (C_FOUND, 0) };
    count(c);
    WHY.with(|w| w.set(why));
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
        assert_eq!(albedo(&g(w), w.shader.as_ptr() as u64).0, 0x5_1000_0020, "no albedo name: lyr0DiffuseTex by the fallback");
        assert_eq!(last_why(), 5, "outcome fallback");
        assert_eq!(last_layers(), [0x5_1000_0020, 0, 0, 0], "layer 0 by name");
        let w = Box::leak(Box::new(world("standard_2lyr_ground", 0x1234_5678, 0)));
        albedo(&g(w), w.shader.as_ptr() as u64);
        assert_eq!(last_layers()[0], 0x5_1000_0020, "no name: the first kind-0 variable");
        let w = Box::leak(Box::new(world("normal_spec", 0xC8E8_C282, 0)));
        albedo(&g(w), w.shader.as_ptr() as u64);
        assert_eq!(last_layers(), [0; 4], "not layered");
    }

    /// A synthetic variable table: {hashA, hashB, index, member, kind}.
    fn table(rows: &[(u32, u32, u16, u8)]) -> Vec<u8> {
        rows.iter()
            .flat_map(|(a, b, i, k)| {
                let mut r = [0u8; 16];
                r[0..4].copy_from_slice(&a.to_le_bytes());
                r[4..8].copy_from_slice(&b.to_le_bytes());
                r[8..10].copy_from_slice(&i.to_le_bytes());
                r[12] = *k;
                r
            })
            .collect()
    }

    /// A name hit is kept with or without the fallback; no name hit: the
    /// fallback takes a colour name, else the first kind-0 variable that is
    /// not a known non-colour name; off: none, as before.
    #[test]
    fn albedo_variable_fallback() {
        // Constant, BumpTexture (kind 0), an unknown kind-0 var, DiffuseTex by hashB.
        let named = table(&[(0x1111_1111, 0, 0, 3), (0xD7C3_8241, 0, 1, 0), (0xAAAA_0001, 0, 2, 0), (0x2222_2222, 0xC8E8_C282, 3, 0)]);
        assert_eq!(pick(&named, 4, true), (3, false), "DiffuseTex by name, fallback unused");
        assert_eq!(pick(&named, 4, false), (3, false));
        // The muv case: no name; the bump map is skipped, the first other kind-0 taken.
        let muv = table(&[(0x1111_1111, 0, 0, 3), (0xD7C3_8241, 0, 1, 0), (0xAAAA_0001, 0, 2, 0), (0xAAAA_0002, 0, 5, 0)]);
        assert_eq!(pick(&muv, 4, true), (2, true), "first kind-0 not bump/normal/spec");
        assert_eq!(pick(&muv, 4, false), (NO_VAR, false), "XNW_ALBEDO_FALLBACK=off: as before");
        // A colour name later in the table beats the first kind-0.
        let imp = table(&[(0xAAAA_0001, 0, 0, 0), (0x8B40_545A, 0, 1, 0), (0x4E41_4593, 0, 7, 0)]);
        assert_eq!(pick(&imp, 3, true), (7, true), "imposterTexture");
        // Only non-colour or non-texture variables: none.
        let none = table(&[(0x1111_1111, 0, 0, 3), (0xD7C3_8241, 0, 1, 0), (0x9E37_22E4, 0, 2, 0)]);
        assert_eq!(pick(&none, 3, true), (NO_VAR, false));
        // A count past the table is clamped.
        assert_eq!(pick(&muv, 200, true), (2, true));
    }

    /// VIS2: the uber_4lyr ground (a test run `albedo names` line, kind-0 vars
    /// in template order): its colour layers are DiffuseTexture_layer0..3,
    /// not the first four kind-0 variables (material maps, lookupTexture).
    #[test]
    fn uber_ground_layers_by_name() {
        let uber = table(&[
            (0x4211_0214, 0x4211_0214, 0, 0), (0x3EA3_7B39, 0x3EA3_7B39, 1, 0), (0xC075_F9B6, 0xC075_F9B6, 2, 0), (0x07BC_8533, 0x07BC_8533, 3, 0),
            (0x3022_DE38, 0x3022_DE38, 4, 0), (0x1A01_6C7E, 0x1A01_6C7E, 8, 0), (0x32B9_DF09, 0x32B9_DF09, 9, 0), (0xD883_AA9E, 0xD883_AA9E, 12, 0),
            (0x3D10_F3B7, 0x3D10_F3B7, 13, 0), (0xCE7B_968E, 0xCE7B_968E, 14, 0), (0x1111_1111, 0, 20, 3),
        ]);
        assert_eq!(layer_vars(&uber, 11), ([14, 12, 9, 13], true), "DiffuseTexture_layer0..3");
        assert_eq!(pick(&uber, 11, true), (14, true), "albedo fallback: DiffuseTexture_layer0, not MaterialMapATexture_layer1 @0");
        // No layer name: kind-0 in template order, known non-colour names skipped.
        let plain = table(&[(0xC075_F9B6, 0, 0, 0), (0xAAAA_0001, 0, 1, 0), (0x27DB_D2BE, 0, 2, 0), (0xAAAA_0002, 0, 3, 0)]);
        assert_eq!(layer_vars(&plain, 4), ([1, 3, NO_VAR, NO_VAR], false));
        // The vehicle 2lyr names keep their places (DiffuseTex, DiffuseTex2).
        let veh = table(&[(0x0DF4_7048, 0, 5, 0), (0x2222_2222, 0xC8E8_C282, 2, 0)]);
        assert_eq!(layer_vars(&veh, 2), ([2, 5, NO_VAR, NO_VAR], true));
    }

    /// The b4 tint block's validation, the effect classes, and a
    /// ped effect's tint through head and tail: palette view slot, rows and
    /// 1/rows from cbData[selector's buffer].
    #[test]
    fn ped_tint_from_the_b4_block() {
        let mut b = [0u8; 8];
        b[..4].copy_from_slice(&0.25f32.to_le_bytes());
        b[4..8].copy_from_slice(&[1, 3, 0xFF, 0]);
        assert_eq!(tint_block(b), Some(([1, 3, 0xFF], 0.25)), "4 rows; r2 none");
        let mut c = b;
        c[5] = 4;
        assert_eq!(tint_block(c), None, "a row past the palette");
        c[..4].copy_from_slice(&0.3f32.to_le_bytes());
        assert_eq!(tint_block(c), None, "not 1/rows");
        assert_eq!(tint_block([0; 8]), None, "zero");
        assert_eq!((class_of("ped_standard_muv"), class_of("standard_vehicle_2lyr"), class_of("standard_vehicle_glass"), class_of("standard")), (CLASS_PED, CLASS_PAINT, CLASS_OTHER, CLASS_OTHER));
        // WHL1: wheels and tyres ask for the table, no paint; other 2lyr effects do not.
        assert_eq!((class_of("standard_vehicle_2lyr_wheel_inst"), class_of("standard_vehicle_2lyr_tyre_inst_lod"), class_of("standard_vehicle_2lyr_decal")), (CLASS_WHEEL, CLASS_WHEEL, CLASS_OTHER));
        assert!(vehmat_class(CLASS_PAINT) && vehmat_class(CLASS_WHEEL) && !vehmat_class(CLASS_PED) && !vehmat_class(CLASS_OTHER));
        assert_eq!((class_of("ped_slod_tint"), class_of("ped_slod_vertex_color"), class_of("ped_standard_lod")), (CLASS_SLOD, CLASS_PED, CLASS_PED));
        // A ped effect: DiffuseTex at texVars[2], TintPaletteTex (kind 0) at texVars[3],
        // TintPaletteSelector (kind 3) naming cbData[1].
        let w = Box::leak(Box::new(world("ped_standard_tint", 0xC8E8_C282, 0)));
        let vars: &mut Vec<u8> = Box::leak(Box::new(vec![0u8; 64]));
        vars[..32].copy_from_slice(&w._vars);
        vars[32..36].copy_from_slice(&TINT_PALETTE.to_le_bytes());
        vars[40..42].copy_from_slice(&3u16.to_le_bytes());
        vars[52..56].copy_from_slice(&TINT_SELECTOR.to_le_bytes());
        vars[56..58].copy_from_slice(&1u16.to_le_bytes());
        vars[60] = 3;
        w._template[4] = 4;
        w._template[0x50..0x58].copy_from_slice(&(vars.as_ptr() as u64).to_le_bytes());
        w.tex_vars[3] = w.view.as_ptr() as u64;
        let block: &[u8; 8] = Box::leak(Box::new(b));
        let cbs: &[u64; 2] = Box::leak(Box::new([0, block.as_ptr() as u64]));
        w.shader[1] = cbs.as_ptr() as u64;
        let sh = w.shader.as_ptr() as u64;
        assert_eq!(albedo(&g(w), sh), (0x5_1000_0020, false));
        assert_eq!(last_extra(), Extra { class: CLASS_PED, tint_slot: 0x5_1000_0020, tint_rows: [1, 3, 0xFF], tint_inv: 0.25 });
        // The thin split: head at the hook, tail later.
        let h = head(sh);
        assert_eq!(h.class, CLASS_PED);
        tail(&g(w), &h, false);
        assert_eq!(last_extra().tint_rows, [1, 3, 0xFF]);
        // A block that does not read as a tint: no tint, class kept.
        let bad: &[u8; 8] = Box::leak(Box::new([0xFF; 8]));
        let cbs2: &[u64; 2] = Box::leak(Box::new([0, bad.as_ptr() as u64]));
        w.shader[1] = cbs2.as_ptr() as u64;
        albedo(&g(w), sh);
        assert_eq!(last_extra(), Extra { class: CLASS_PED, ..NO_EXTRA });
        // W16P: the selector at +4 of its buffer (declaration n = 0, member 1 at offset 4,
        // formula): read as 1/rows itself, or as the rows word after 1/rows.
        let w2 = Box::leak(Box::new(world("ped_standard_tint", 0xC8E8_C282, 0)));
        let vars2: &mut Vec<u8> = Box::leak(Box::new(vars.clone()));
        vars2[58..60].copy_from_slice(&1u16.to_le_bytes());
        w2._template[4] = 4;
        w2._template[0x50..0x58].copy_from_slice(&(vars2.as_ptr() as u64).to_le_bytes());
        w2.tex_vars[3] = w2.view.as_ptr() as u64;
        let decl: &mut Vec<u8> = Box::leak(Box::new(vec![0u8; 64]));
        decl[36..38].copy_from_slice(&4u16.to_le_bytes());
        w2._fx[0xC88..0xC90].copy_from_slice(&(decl.as_ptr() as u64).to_le_bytes());
        let mut shifted = [0xAAu8; 12];
        shifted[4..].copy_from_slice(&b);
        let shifted: &[u8; 12] = Box::leak(Box::new(shifted));
        let cbs3: &[u64; 2] = Box::leak(Box::new([0, shifted.as_ptr() as u64]));
        w2.shader[1] = cbs3.as_ptr() as u64;
        let sh2 = w2.shader.as_ptr() as u64;
        albedo(&g(w2), sh2);
        assert_eq!(last_extra().tint_rows, [1, 3, 0xFF], "1/rows at the selector's offset");
        let mut packed = [0u8; 12];
        packed[..8].copy_from_slice(&b);
        let packed: &[u8; 12] = Box::leak(Box::new(packed));
        let cbs4: &[u64; 2] = Box::leak(Box::new([0, packed.as_ptr() as u64]));
        w2.shader[1] = cbs4.as_ptr() as u64;
        albedo(&g(w2), sh2);
        assert_eq!((last_extra().tint_rows, last_extra().tint_inv), ([1, 3, 0xFF], 0.25), "the selector is the rows word, 1/rows before it");
        // Not a ped: no tint lookup at all.
        let v = Box::leak(Box::new(world("standard_vehicle", 0xC8E8_C282, 0)));
        albedo(&g(v), v.shader.as_ptr() as u64);
        assert_eq!(last_extra(), Extra { class: CLASS_PAINT, ..NO_EXTRA });
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
