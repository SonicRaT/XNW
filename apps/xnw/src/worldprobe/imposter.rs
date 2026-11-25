//! Imposters in the live pass (`XNW_WORLD_RENDER=...,imposters=on`, default
//! off). At each call of the title's imposter submit 0x1451D0AC0 the call's
//! inputs are read from title memory by the frame pack's own reader
//! (`pack::imposter_read`: IcRec, desc, T in the frame's arena, the list L
//! and its records, the type record R through the type table) and added to
//! the frame's batches by the replay's own `xnw_frame::imposter::add_call`,
//! so the live pass and the replay of a capture make the same batches. The
//! frame's batches go with its list. Every read is bounds-checked and
//! fault-safe: a call that cannot be read is counted as skipped, never a
//! fault.

use std::sync::Mutex;

use xnw_world::imposter::{self as imp, Imposters};

/// `imposters=on` (read once).
pub fn on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| imp::parse_screen(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default()).on);
    *ON
}

/// Frames whose calls are not yet joined to their list (oldest first).
struct Open {
    frame: u64,
    /// The frame's arena base, read at its first call.
    arena: u64,
    im: Imposters,
}

/// Frames held at most (a frame whose list never comes is dropped).
const OPEN_MAX: usize = 8;
static OPEN: Mutex<Vec<Open>> = Mutex::new(Vec::new());

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// At the hook: one submit call of `frame` (`args` as the pack's kind 42).
pub(super) fn on_call(frame: u64, args: [u64; 6], icrec: u64) {
    let _site = super::fault_site(super::SITE_IMPOSTERS);
    let mut open = lock(&OPEN);
    let k = match open.iter().position(|o| o.frame == frame) {
        Some(k) => k,
        None => {
            if open.len() >= OPEN_MAX {
                open.remove(0);
            }
            open.push(Open { frame, arena: super::pack::arena_base(), im: Imposters::default() });
            open.len() - 1
        }
    };
    let o = &mut open[k];
    add(&mut o.im, args, icrec, o.arena);
}

/// One call read and added (the cap is counted without reading).
fn add(im: &mut Imposters, args: [u64; 6], icrec: u64, arena: u64) {
    if im.calls >= imp::MAX_CALLS {
        xnw_frame::imposter::add_call(im, &[], None);
        return;
    }
    // Only the parts the card pass uses (not the handler, args, visible list).
    let (blob, inst) = super::pack::imposter_read(args, icrec, arena, false);
    xnw_frame::imposter::add_call(im, &blob, Some(&inst));
}

/// The imposter batches of frame `frame` (None: no call met), taken; older
/// frames still held are dropped.
pub fn take(frame: u64) -> Option<std::sync::Arc<Imposters>> {
    let mut open = lock(&OPEN);
    let k = open.iter().position(|o| o.frame == frame);
    let got = k.map(|k| open.remove(k));
    open.retain(|o| o.frame > frame);
    got.map(|o| std::sync::Arc::new(o.im))
}

#[cfg(test)]
mod tests {
    use super::*;
    use xnw_formats::framepack as fp;

    /// Made-up title memory for one call: IcRec -> desc -> L (two records),
    /// T at arena + 0x40, the type table with type 0x181 at index 0 (R+0x58
    /// = 90). The live reader builds the same batch as the capture reader
    /// (the pack's kind 42 + 47 through `load_imposters`' path).
    #[test]
    fn live_batch_equals_capture_batch() {
        use xnw_world::imposter::synthetic_record;
        // The type table global is shared with the pack's tests.
        let _one = super::super::pack::tests::fresh();
        let recs = [synthetic_record([10.0, 20.0, 30.0], 1.0), synthetic_record([0.0, 300.0, 5.0], 2.0)];
        let rec_bytes: Vec<u8> = recs.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        let mut l = vec![0u8; 0xA0];
        l[0x90..0x98].copy_from_slice(&(rec_bytes.as_ptr() as u64).to_le_bytes());
        l[0x98..0x9A].copy_from_slice(&2u16.to_le_bytes());
        let mut desc = vec![0u8; 0x100];
        desc[..8].copy_from_slice(&(l.as_ptr() as u64).to_le_bytes());
        desc[8..12].copy_from_slice(&2u32.to_le_bytes());
        for (k, v) in [-5.0f32, -5.0, 0.0, 5.0, 5.0, 14.0].iter().enumerate() {
            desc[0x10 + 4 * k + if k >= 3 { 4 } else { 0 }..][..4].copy_from_slice(&v.to_le_bytes());
        }
        let mut arena = vec![0u8; 0x100];
        for (k, v) in [-1.0f32, -1.0, -1.0, -1.0, 400.0, 420.0].iter().enumerate() {
            arena[0x40 + 0x48 + 4 * k..][..4].copy_from_slice(&v.to_le_bytes());
        }
        let t_off = [0u8, 0, 0, 0, 0, 0, 0, 0, 0x40, 0, 0, 0];
        let mut ic = vec![0u8; 0x100];
        ic[..8].copy_from_slice(&(desc.as_ptr() as u64).to_le_bytes());
        ic[0x60..0x68].copy_from_slice(&(t_off.as_ptr() as u64).to_le_bytes());
        let mut types = vec![0u8; 224];
        types[0x58..0x5C].copy_from_slice(&90f32.to_le_bytes());
        let mut groups = vec![0x80u8; 144];
        groups[0] = 0x01;
        groups[16..24].copy_from_slice(&[0x81, 0x01, 0, 0, 0, 0, 0, 0]);
        let mut globals = (types.as_ptr() as u64).to_le_bytes().to_vec();
        globals.extend(1u64.to_le_bytes());
        globals.extend((groups.as_ptr() as u64).to_le_bytes());
        super::super::pack::set_imposter_types(globals.as_ptr() as u64);
        let args = [0x181, 76, 3, 0, 0, 0];
        // Live: the hook's path.
        let mut live = Imposters::default();
        add(&mut live, args, ic.as_ptr() as u64, arena.as_ptr() as u64);
        add(&mut live, [7, 76, 3, 0, 0, 0], 0, arena.as_ptr() as u64);
        // Capture: the same call as the pack writes it, read back as the replay does.
        let (blob, inst) = super::super::pack::imposter_read(args, ic.as_ptr() as u64, arena.as_ptr() as u64, true);
        let inst = fp::ImpInstances::decode(&inst.encode()).unwrap();
        let (blob0, inst0) = super::super::pack::imposter_read([7, 76, 3, 0, 0, 0], 0, arena.as_ptr() as u64, true);
        let mut capture = Imposters::default();
        xnw_frame::imposter::add_call(&mut capture, &blob, Some(&inst));
        xnw_frame::imposter::add_call(&mut capture, &blob0, Some(&fp::ImpInstances::decode(&inst0.encode()).unwrap()));
        super::super::pack::set_imposter_types(0);
        assert_eq!(live, capture);
        assert_eq!((live.calls, live.skipped, live.batches.len()), (2, [1, 0, 0], 1));
        let b = &live.batches[0];
        assert_eq!((b.kind, b.n, b.records.clone(), b.start, b.fade, b.box_max), (0x181, 2, recs.to_vec(), 90.0, [400.0, 420.0], [5.0, 5.0, 14.0]));
    }

    /// The live read leaves out the parts the card pass does not use: r9 at
    /// the start of a fresh allocation puts the handler (r9 - 0x308) in an
    /// unmapped page (a capture: 793 faults, last at 0x2eb8efcf8 = 0x2eb8f0000 - 0x308).
    #[test]
    fn live_read_skips_the_handler() {
        // SAFETY: reserve only (faults on read); freed below.
        let bad = unsafe { xnw_core::win::VirtualAlloc(std::ptr::null(), 0x1_0000, 0x2000, 0x01) } as u64;
        assert_ne!(bad, 0);
        let ic = vec![0u8; 0x100];
        let args = [0x181, 76, 3, bad + 0x308, 0, 0];
        let (_, live) = super::super::pack::imposter_read(args, ic.as_ptr() as u64, 0, false);
        let (_, full) = super::super::pack::imposter_read(args, ic.as_ptr() as u64, 0, true);
        for p in [fp::IMP_ARGS, fp::IMP_VISIBLE, fp::IMP_HANDLER] {
            assert_eq!(live.reasons[p], fp::reason::NOT_CAPTURED, "part {p}");
        }
        assert_eq!((full.reasons[fp::IMP_HANDLER], full.handler_at), (fp::reason::UNREADABLE, bad), "the capture still reads it");
        // SAFETY: the reservation above.
        unsafe { xnw_core::win::VirtualFree(bad as *mut _, 0, 0x8000) };
    }

    /// Frames are taken once; older frames go with a newer take.
    #[test]
    fn frames_are_taken_once() {
        for f in [7_000_001u64, 7_000_002, 7_000_003] {
            on_call(f, [1, 76, 3, 0, 0, 0], 0);
        }
        let got = take(7_000_002).unwrap();
        assert_eq!((got.calls, got.skipped[imp::SKIP_UNREADABLE], got.batches.len()), (1, 1, 0));
        assert!(take(7_000_002).is_none() && take(7_000_001).is_none(), "taken, and older dropped");
        assert!(take(7_000_003).is_some());
    }
}
