//! NPC1 (`slodtint=on`, default on with `vehmat`): the far peds'
//! `ped_slod_tint` rows take the title's per-vertex palette colour instead
//! of the pastel stand-in (`flat_colour`). xnw_world::vehmat::slod_lut has
//! the rule and the evidence: root 4 dword 0 (s21) names the ped's 512-byte
//! block in the frame table's T8+0x60 bytes (root 0 slot 3), an index table
//! at +0x100 and its RGB8 entries; the vertex program picks the entry by
//! mesh word 0x04's stream, the pixel program decodes it.
//!
//! As INT1's vehicle tables (vehmat.rs): asked only for the sink that
//! follows a `ped_slod_tint` hook draw on the same thread (`set_pending`;
//! burst: its hook draw's effect), the block given to that draw as its
//! vehmat block ([`super::vehmat::Block::slod`] set), kept per (frame,
//! block address, mode): a ped's sections share one.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use xnw_formats::framepack::view;
use xnw_gpu::d3d12x::tap::{TapDraw, TapRoot};
use xnw_world::vehmat as vm;

/// T8+0x60 (root 0 slot 3) and root 4 dword 0 [V MS 132ea7e6 0x91c-0x940:
/// `s_load_dwordx4 T8+0x60`, `s_lshl_b32 s0, s21, 9`; pk8a f1027 sinks
/// 112/119/124: root 4 = (0x10035, 0), slot 3 a 307,200-B buffer].
const PALETTE_ROOT: usize = 0;
const PALETTE_SLOT: u32 = 0x60 / 32;
const S21_ROOT: usize = 4;

thread_local! {
    /// The last hook draw on this thread was a `ped_slod_tint` draw.
    static PENDING: Cell<bool> = const { Cell::new(false) };
}

pub(super) fn set_pending(on: bool) {
    PENDING.with(|p| p.set(on));
}

pub(super) fn pending() -> bool {
    PENDING.with(Cell::get)
}

/// `XNW_WORLD_RENDER` set, `vehmat` on (its block path) and not `slodtint=off` (read once).
pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| vm::parse_switch(&v) && vm::parse_slod(&v)))
}

/// Sinks since the last line by outcome.
static WHY: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
const NAMES: [&str; 5] = ["asked", "no root constant", "no palette view", "unreadable", "given"];

fn why(i: usize) {
    WHY[i].fetch_add(1, Ordering::Relaxed);
}

type Key = (u64, u64, u8);
static CACHE: Mutex<Option<(u64, HashMap<Key, Option<super::vehmat::Table>>)>> = Mutex::new(None);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The draw's block (None: counted in [`WHY`]).
fn block_of(d: &TapDraw, frame: u64) -> Option<super::vehmat::Table> {
    why(0);
    let Some(s21) = (match d.roots.get(S21_ROOT) {
        Some(TapRoot::Constants(c)) => c.first().copied(),
        _ => None,
    }) else {
        why(1);
        return None;
    };
    let at = vm::slod_block_at(s21);
    let Some(v) = super::vehmat::table_view(d, PALETTE_ROOT, PALETTE_SLOT).filter(|v| matches!(v.kind, view::BUFFER | view::PLACED_BUFFER) && at + vm::SLOD_BLOCK as u64 <= v.bytes) else {
        why(2);
        return None;
    };
    let mode = vm::slod_mode(s21);
    let key = (v.addr, at, mode);
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
    let _site = super::fault_site_or_kept(super::SITE_BRIDGE);
    let made = super::dcopy(v.addr + at, vm::SLOD_BLOCK).and_then(|b| vm::slod_lut(&b)).map(|words| Arc::new(super::vehmat::Block { words, livery_at: 0, livery: [0; 2], slod: mode }));
    if made.is_none() {
        why(3);
    }
    if let Some(c) = lock(&CACHE).as_mut().filter(|c| c.0 == frame) {
        c.1.insert(key, made.clone());
    }
    made
}

/// The draw tap (title thread, at the list sink, after a `ped_slod_tint` hook draw).
pub(super) fn on_draw_sink(d: &TapDraw) {
    set_pending(false);
    let Some(t) = block_of(d, super::frame_key()) else { return };
    if super::attach_vehmat(t) {
        why(4);
    }
}

/// The frame line: slod rows with a block of the frame's slod rows, and
/// the sinks by outcome since the last line.
pub(super) fn line(frame: u64, rows: usize, slod: usize) -> String {
    let w: Vec<String> = NAMES.iter().zip(WHY.iter()).map(|(n, v)| format!("{n} {}", v.swap(0, Ordering::Relaxed))).collect();
    format!("world render: frame {frame}: slodtint: {rows} of {slod} ped_slod_tint rows coloured from the title's palette blocks (T8+0x60 by root 4 dword 0); sinks: {}", w.join(", "))
}
