//! Vehicle paint: the vehicle handler Draw entries (vt[6] of the
//! vehicle handler families; rcx = the handler = `*(entity+0x28)`) are
//! hooked; on entry the handler's paint colours A and B (its type-21
//! component row, +0x124 and +0x134) are kept for this thread's following
//! geometry draws, which take them when their effect is a vehicle body
//! material ([`crate::worldtex::CLASS_PAINT`]). Ported from commit
//! 37b37eb6.
//!
//! The fix: the first port hooked 0x1412BB560 (frag family 0x14646D210) and
//! 0x1435B0DA0 / 0x1435B2A00 (skinned family 0x14646E4E0 and its twin): none
//! is a vehicle handler, and a test run counted 724-4235 calls a window, 0 with
//! a type-21 row. The vehicle handlers are the families whose prepare
//! (vt[2]) reaches `veh_chassis_skeleton_setup` 0x141648F60 (VEH1 [V]):
//! vtables 0x14646F010 (vt[6] 0x1435B7040), 0x146470D20 (0x1435A85F0),
//! 0x14646F650 (0x1435A9A50) [V vtable words read from the exe].

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// The vehicle handler Draw entries and the bytes a hook moves [V the exe
/// 1.0.2.1, timestamp 0x6AF783D4]:
/// 0x1435B7040 push rsi; push rdi; sub rsp,28h; mov rsi,rdx; mov rdi,rcx; call 0x1455648B0 (17);
/// 0x1435A85F0 push r14/rsi/rdi/rbp/rbx; sub rsp,20h; mov eax,[rip+386052Ch] (16; watch.rs moves the rip load);
/// 0x1435A9A50 eight pushes; sub rsp,0D8h (19).
/// No title patch (patches.rs) and no other hook of ours is at these addresses [V grep].
pub(crate) const VEHICLE_DRAWS: [(usize, &[u8]); 3] = [
    (0x1_435B_7040, &[0x56, 0x57, 0x48, 0x83, 0xEC, 0x28, 0x48, 0x89, 0xD6, 0x48, 0x89, 0xCF, 0xE8, 0x5F, 0xD8, 0xFA, 0x01]),
    (0x1_435A_85F0, &[0x41, 0x56, 0x56, 0x57, 0x55, 0x53, 0x48, 0x83, 0xEC, 0x20, 0x8B, 0x05, 0x2C, 0x05, 0x86, 0x03]),
    (0x1_435A_9A50, &[0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x56, 0x57, 0x55, 0x53, 0x48, 0x81, 0xEC, 0xD8, 0x00, 0x00, 0x00]),
];

/// The hook ids (watch.rs ids are only labels for the stubs).
const HOOK_ID: usize = 40;

thread_local! {
    /// The paint of the vehicle handler whose Draw ran last on this thread.
    static PAINT: std::cell::Cell<Option<[f32; 6]>> = const { std::cell::Cell::new(None) };
}

/// Why a handler's paint read failed (the frame line's reasons).
pub(crate) const REASONS: [&str; 5] = ["container unreadable", "no type-21 flag", "no type-21 row", "colour unreadable", "colour out of range"];

/// Calls, calls with paint, then one count per [`REASONS`] entry, since the
/// last [`take`].
static CALLS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
/// Rejections logged so far (the first 8 are).
static LOGGED: AtomicU32 = AtomicU32::new(0);
/// The title's base, for static reading addresses in the log.
static BASE: OnceLock<usize> = OnceLock::new();

/// Handler Draws seen, those with a paint row, and the rejections by
/// reason, since the last call (the frame line).
pub(crate) fn take() -> [u64; 7] {
    std::array::from_fn(|i| CALLS[i].swap(0, Ordering::Relaxed))
}

/// The paint for a draw on this thread now (the last handler Draw's).
pub(crate) fn now() -> Option<[f32; 6]> {
    PAINT.with(std::cell::Cell::get)
}

extern "system" fn on_vehicle_draw(frame: usize, id: usize) {
    // SAFETY: the stub's frame: rcx (the handler) at frame - 24.
    let handler = unsafe { ((frame - 24) as *const u64).read() };
    let p = paint_read(handler);
    CALLS[0].fetch_add(1, Ordering::Relaxed);
    match p {
        Ok(_) => {
            CALLS[1].fetch_add(1, Ordering::Relaxed);
        }
        Err((why, raw)) => {
            CALLS[2 + usize::from(why)].fetch_add(1, Ordering::Relaxed);
            let k = LOGGED.fetch_add(1, Ordering::Relaxed);
            if k < 8 {
                let vt = guarded(handler, 8).map_or(0, |b| le(&b, 0, 8));
                let base = BASE.get().copied().unwrap_or(0) as u64;
                let vt_ida = if base != 0 && vt >= base { vt - base + 0x1_4000_0000 } else { vt };
                let entry = VEHICLE_DRAWS.get(id.wrapping_sub(HOOK_ID)).map_or(0, |e| e.0);
                xnw_core::info!("world probe: paint read rejected ({}/8) at {entry:#x}: handler {handler:#x} vt {vt_ida:#x}: {}; raw {:#x} {:#x} {:#x} {:#x}", k + 1, REASONS[usize::from(why)], raw[0], raw[1], raw[2], raw[3]);
            }
        }
    }
    PAINT.with(|c| c.set(p.ok()));
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

/// A vehicle handler's paint colours A and B: the type-21 row of its
/// component container (handler+0x18: {u32 cap, u32 n, rows**, u64 flags;
/// bit 31+21 = has a type-21 row}; row type u32 at +0x10), colour A float3
/// at +0x124, B at +0x134 (0x1450705D0 / 0x144F56F90 find the row, 0x140967400
/// writes A = RGBA8 x 1/255 [V]), with guarded reads. A failed read gives
/// the reason ([`REASONS`] index) and raw values: container (cap|n, rows,
/// flags), the first row types, the colour words.
fn paint_read(handler: u64) -> Result<[f32; 6], (u8, [u64; 4])> {
    let c = guarded(handler.wrapping_add(0x18), 0x18).ok_or((0, [handler, 0, 0, 0]))?;
    let (cn, rows, flags) = (le(&c, 0, 8), le(&c, 8, 8), le(&c, 0x10, 8));
    if (flags >> 31) & (1 << 21) == 0 {
        return Err((1, [cn, rows, flags, 0]));
    }
    let n = (le(&c, 4, 4) as usize).min(64);
    let list = guarded(rows, 8 * n).ok_or((2, [cn, rows, flags, 0]))?;
    let types = |k: usize| (0..k.min(n)).map(|i| guarded(le(&list, 8 * i, 8).wrapping_add(0x10), 4).map_or(0xFFFF, |t| le(&t, 0, 4) & 0xFFFF)).fold(0u64, |a, t| a << 16 | t);
    let comp = (0..n).map(|i| le(&list, 8 * i, 8)).find(|r| guarded(r.wrapping_add(0x10), 4).is_some_and(|t| le(&t, 0, 4) == 21)).ok_or((2, [cn, rows, flags, types(4)]))?;
    let b = guarded(comp + 0x124, 0x1C).ok_or((3, [comp, 0, 0, 0]))?;
    let f = |o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let v = [f(0), f(4), f(8), f(0x10), f(0x14), f(0x18)];
    if v.iter().all(|x| x.is_finite() && (0.0..=1.01).contains(x)) {
        Ok(v)
    } else {
        Err((4, [comp, le(&b, 0, 8), le(&b, 8, 4), le(&b, 0x10, 8)]))
    }
}

/// Hooks the handler Draw entries (`paint=off`: none). `real` rebases an
/// static reading address. Entries whose first bytes differ are skipped with a warning.
pub(crate) fn install(base: usize, real: &dyn Fn(usize) -> usize) {
    let _ = BASE.set(base);
    if !crate::worldtex::paint_enabled() {
        xnw_core::info!("world probe: vehicle paint off (paint=off): no handler Draw hooks");
        return;
    }
    let mut hooked = 0;
    for (k, (va, want)) in VEHICLE_DRAWS.iter().enumerate() {
        let at = real(*va);
        let mut got = vec![0u8; want.len()];
        if !(crate::watch::guarded_read(at, &mut got) && got == *want) {
            xnw_core::warn!("world probe: vehicle draw {va:#x} entry {got:02x?} not the expected one; no paint from it");
            continue;
        }
        // SAFETY: nothing runs the title yet; the entry bytes were checked.
        match unsafe { crate::watch::hook_entry_with(at as *mut u8, HOOK_ID + k, on_vehicle_draw as usize) } {
            Ok(_) => hooked += 1,
            Err(e) => xnw_core::warn!("world probe: vehicle draw {va:#x} not hooked: {e}"),
        }
    }
    xnw_core::info!("world probe: vehicle paint: {hooked} of {} vehicle handler Draw entries hooked (0x1435B7040, 0x1435A85F0, 0x1435A9A50)", VEHICLE_DRAWS.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A handler whose container holds a type-7 and a type-21 row: colours
    /// A and B read; no type-21 flag, no type-21 row, or a colour out of
    /// 0..1: none, with the reason and raw values.
    #[test]
    fn paint_read_from_the_type21_row() {
        let comp: &mut Vec<u8> = Box::leak(Box::new(vec![0u8; 0x200]));
        comp[0x10..0x14].copy_from_slice(&21u32.to_le_bytes());
        for (i, v) in [0.8f32, 0.1, 0.1, 0.2, 0.3, 0.4].iter().enumerate() {
            let o = if i < 3 { 0x124 + 4 * i } else { 0x134 + 4 * (i - 3) };
            comp[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }
        let other: &mut Vec<u8> = Box::leak(Box::new(vec![0u8; 0x20]));
        other[0x10] = 7;
        let rows: &[u64; 2] = Box::leak(Box::new([other.as_ptr() as u64, comp.as_ptr() as u64]));
        let h: &mut Vec<u8> = Box::leak(Box::new(vec![0u8; 0x40]));
        h[0x18 + 4..0x18 + 8].copy_from_slice(&2u32.to_le_bytes());
        h[0x18 + 8..0x18 + 16].copy_from_slice(&(rows.as_ptr() as u64).to_le_bytes());
        h[0x18 + 16..0x18 + 24].copy_from_slice(&((1u64 << 21) << 31).to_le_bytes());
        assert_eq!(paint_read(h.as_ptr() as u64).ok(), Some([0.8, 0.1, 0.1, 0.2, 0.3, 0.4]));
        comp[0x128..0x12C].copy_from_slice(&f32::NAN.to_le_bytes());
        assert_eq!(paint_read(h.as_ptr() as u64).map_err(|e| e.0), Err(4), "a colour not a number");
        comp[0x128..0x12C].copy_from_slice(&0.1f32.to_le_bytes());
        comp[0x10] = 22;
        assert_eq!(paint_read(h.as_ptr() as u64).map_err(|e| (e.0, e.1[3])), Err((2, 7 << 16 | 22)), "no type-21 row; the row types");
        comp[0x10] = 21;
        h[0x18 + 16..0x18 + 24].copy_from_slice(&0u64.to_le_bytes());
        assert_eq!(paint_read(h.as_ptr() as u64).map_err(|e| e.0), Err(1), "no type-21 row flag");
        assert_eq!(paint_read(0).map_err(|e| e.0), Err(0), "no handler");
    }

    /// Each entry's bytes relocate to a trampoline (whole instructions; the
    /// call and the rip load made absolute).
    #[test]
    fn vehicle_draw_entries_relocate() {
        for (va, b) in VEHICLE_DRAWS {
            assert_eq!(crate::watch::relocated_len(b), Some(b.len()), "{va:#x}");
        }
    }
}
