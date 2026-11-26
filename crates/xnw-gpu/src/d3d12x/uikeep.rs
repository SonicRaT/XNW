//! `XNW_GPU=null...,ui=keep` (default off): in null mode the title's
//! own UI still reaches the host GPU, over our renderer's picture.
//!
//! The rule (static V): our list methods run on the title's
//! recording thread, so its static TLS block (`TEB.TLS[TITLE_TLS_INDEX]`)
//! says which pass records. `slot = u8 [tls+0x1294]`, `phase = u32
//! [title 0x149C3B750 + 4*slot]` (pass begin writes it, pass end writes -1:
//! 0x1450041B0), `stage = u32 [tls+0x950]` (0x14325D2A0). A command is
//! **kept** when `phase == -1` (outside any pass: frame-begin callbacks,
//! loading and front-end render functions, which never begin a pass) or
//! `stage` is 30 (UI, HUD, Scaleform), 16 (plane-1 alpha compensation) or
//! 8 (Scaleform to texture). TLS+0x950 is not cleared at pass end
//! (0x145004160 clears only +0x972), so the phase is tested first.
//!
//! Per command, not per list. In dropped stretches state calls and
//! barriers are still recorded (post binds plane 1 in stage 15), and
//! draws, dispatches, indirect executions, GPU copies and clears are
//! dropped. A list holding kept work is executed at submission; the others
//! do only their CPU part (`run_skipped`). A kept draw whose host pipeline is
//! not made already is dropped, never waited for, and nothing is compiled
//! for it (only programs that ran before the null point draw). Inside a
//! stage-30 pass a draw is also dropped when its pixel or compute program
//! is named and not of the UI class (stage 30 can run world post on a
//! second view, 0x1417BD7F0): Scaleform `F[A-Z]*`, `PS_Bink*`,
//! `PS_Minimap*`, `*_IM_*`, `PS_BlitUI*`, `CS_UIOverlay*`.
//!
//! Present: our picture (`XNW_WORLD_RENDER=screen`) as plane 0 and the
//! title's UI plane over it (premultiplied, present.rs). A frame with no
//! dropped command (front end, loading) shows the title's planes as they
//! are. Not fed (left empty for now): the menu backdrop blur (sub-passes
//! 184/185) and depth-occluded markers read the title's scene target and
//! depth, which null mode never draws; feeding them means writing our
//! picture and depth into the title's plane-0 and depth host textures
//! before the UI list runs. Single-plane mode composes the UI into plane 0
//! (0x140125BA0): logged once; then ours is not shown over it.
//!
//! One line a second, `null ui: kept ...`: kept and dropped commands by
//! stage, drops for a pipeline not made or a non-UI program in stage 30,
//! lists executed and skipped, pipelines made since the null point.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

/// `,ui=keep` after `null[@...]` in `XNW_GPU`.
pub(crate) fn parse(v: Option<&str>) -> bool {
    let Some(v) = v else { return false };
    let mut parts = v.trim().split(',');
    parts.next().is_some_and(|p| p.trim().starts_with("null")) && parts.any(|p| p.trim() == "ui=keep")
}

/// Whether the switch is on.
pub(crate) fn on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        let on = parse(std::env::var("XNW_GPU").ok().as_deref());
        if on {
            xnw_core::warn!("null ui: the title's UI passes reach the host GPU in null mode (XNW_GPU=...,ui=keep)");
        }
        on
    });
    *ON
}

/// The keep rule over (phase, stage).
pub(crate) fn keep(phase: u32, stage: u32) -> bool {
    phase == u32::MAX || matches!(stage, 30 | 16 | 8)
}

/// (phase, stage) from a title TLS block and the phase table.
///
/// # Safety
/// `block` is a title TLS block (0x1298 bytes readable) and `phases` the
/// title's phase table (256 entries).
pub(crate) unsafe fn read(block: *const u8, phases: *const u32) -> (u32, u32) {
    // SAFETY: the caller's guarantee.
    unsafe {
        let slot = block.add(0x1294).read();
        let phase = phases.add(usize::from(slot)).read_unaligned();
        let stage = block.add(0x950).cast::<u32>().read_unaligned();
        (phase, stage)
    }
}

/// This thread's (phase, stage), if it has a title TLS block.
pub(crate) fn here() -> Option<(u32, u32)> {
    let base = super::object::title_base();
    let index = xnw_core::TITLE_TLS_INDEX.load(Relaxed) as usize;
    if base == 0 {
        return None;
    }
    let array: usize;
    // SAFETY: reads the TEB's ThreadLocalStoragePointer.
    unsafe { std::arch::asm!("mov {}, gs:[0x58]", out(reg) array, options(nostack, readonly, preserves_flags)) };
    if array == 0 {
        return None;
    }
    // SAFETY: the title is a module with static TLS: every thread's array
    // has its index (the loader extends them all).
    let block = unsafe { ((array + 8 * index) as *const usize).read() };
    if block == 0 {
        return None;
    }
    let phases = (base + (0x1_49c3_b750 - 0x1_4000_0000)) as *const u32;
    // SAFETY: a title TLS block and the title's table.
    Some(unsafe { read(block as *const u8, phases) })
}

thread_local! {
    /// Set while kept lists are executed (their submission-time uploads
    /// and copies are made).
    static EXECUTING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` as the execution of kept lists.
pub(crate) fn executing<R>(f: impl FnOnce() -> R) -> R {
    EXECUTING.with(|e| e.set(true));
    let r = f();
    EXECUTING.with(|e| e.set(false));
    r
}

/// Whether the command being recorded on this thread is kept (null mode
/// and the switch on; false otherwise).
pub(crate) fn keep_now() -> bool {
    on() && (EXECUTING.with(|e| e.get()) || here().is_some_and(|(p, s)| keep(p, s)))
}

/// Whether a null-mode refusal (uploads, host copies, pipelines) stands:
/// `null_now() && !keep_now()`.
pub(crate) fn refuse() -> bool {
    super::frameskip::null_now() && !keep_now()
}

/// What a recorded command is, for the decision and the counts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Command {
    /// Pipeline, root arguments, targets, viewports, heaps, buffers: never
    /// asked about (always recorded), named for [`records`].
    #[allow(dead_code)]
    State,
    #[allow(dead_code)]
    Barrier,
    Draw,
    Dispatch,
    Indirect,
    Copy,
    Clear,
}

/// Whether `command` is recorded in a stretch that `keep` says.
pub(crate) fn records(command: Command, keep: bool) -> bool {
    keep || matches!(command, Command::State | Command::Barrier)
}

const STAGES: usize = 128;
/// Index for commands outside any pass.
const OUTSIDE: usize = STAGES - 1;
static KEPT: [AtomicU64; STAGES] = [const { AtomicU64::new(0) }; STAGES];
static DROPPED: [AtomicU64; STAGES] = [const { AtomicU64::new(0) }; STAGES];
/// Dropped commands since the last present (a frame without any shows the
/// title's planes).
static DROPPED_FRAME: AtomicU64 = AtomicU64::new(0);
/// Kept draws without a made pipeline; kept stage-30 draws of a non-UI
/// program; lists executed, lists skipped; pipelines made since the null
/// point.
pub(crate) static NOT_MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static NOT_UI: AtomicU64 = AtomicU64::new(0);
pub(crate) static EXECUTED: AtomicU64 = AtomicU64::new(0);
pub(crate) static SKIPPED: AtomicU64 = AtomicU64::new(0);

/// A draw-like command (draw, dispatch, indirect, GPU copy, clear) being
/// recorded: whether it is kept (counted by stage). Always true with the
/// switch off or before the null point.
pub(crate) fn command(c: Command) -> bool {
    if !on() || !super::frameskip::null_now() {
        return true;
    }
    let (phase, stage) = here().unwrap_or((0, OUTSIDE as u32 - 1));
    let k = keep(phase, stage);
    let at = if phase == u32::MAX { OUTSIDE } else { (stage as usize).min(STAGES - 2) };
    if k {
        KEPT[at].fetch_add(1, Relaxed);
    } else {
        DROPPED[at].fetch_add(1, Relaxed);
        DROPPED_FRAME.fetch_add(1, Relaxed);
    }
    records(c, k)
}

/// This thread's stage when inside a pass (for the stage-30 name check).
pub(crate) fn stage_in_pass() -> Option<u32> {
    here().and_then(|(p, s)| (p != u32::MAX).then_some(s))
}

/// Whether a program's pack name is of the UI class.
pub(crate) fn ui_name(name: &str) -> bool {
    let n = name.rsplit(':').next().unwrap_or(name);
    let mut c = n.chars();
    (c.next() == Some('F') && c.next().is_some_and(|x| x.is_ascii_uppercase()))
        || n.starts_with("PS_Bink")
        || n.starts_with("PS_Minimap")
        || n.contains("_IM_")
        || n.starts_with("PS_BlitUI")
        || n.starts_with("CS_UIOverlay")
}

/// Whether the frame now presented had no dropped command; resets.
pub(crate) fn frame_all_kept() -> bool {
    DROPPED_FRAME.swap(0, Relaxed) == 0
}

/// Once a second (called at present): the `null ui:` line.
pub(crate) fn tick() {
    if !on() || !super::frameskip::null_now() {
        return;
    }
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = xnw_core::uptime().as_millis() as u64;
    let last = LAST.load(Relaxed);
    if now < last + 1000 || LAST.compare_exchange(last, now, Relaxed, Relaxed).is_err() {
        return;
    }
    let by = |t: &[AtomicU64; STAGES]| -> String {
        let v: Vec<String> = t.iter().enumerate().filter_map(|(i, c)| {
            let n = c.swap(0, Relaxed);
            (n != 0).then(|| if i == OUTSIDE { format!("outside {n}") } else { format!("{i}: {n}") })
        }).collect();
        if v.is_empty() { "none".into() } else { v.join(", ") }
    };
    xnw_core::info!(
        "null ui: kept by stage {}; dropped by stage {}; kept but no pipeline made {}, stage-30 non-UI program {}; lists executed {}, skipped {}; host pipelines made since the null point {}",
        by(&KEPT),
        by(&DROPPED),
        NOT_MADE.swap(0, Relaxed),
        NOT_UI.swap(0, Relaxed),
        EXECUTED.swap(0, Relaxed),
        SKIPPED.swap(0, Relaxed),
        super::pipeline::made_in_null()
    );
}

/// Single-plane presents, logged once.
pub(crate) fn single_plane() {
    static SAID: AtomicBool = AtomicBool::new(false);
    if !SAID.swap(true, Relaxed) {
        xnw_core::warn!("null ui: the title presents one plane: its UI is composed into plane 0 (0x140125BA0), so ours is not shown over it; the title's plane is shown");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake title TLS block and phase table: slot, phase in that slot,
    /// stage.
    fn state(slot: u8, phase: u32, stage: u32) -> (u32, u32) {
        let mut block = vec![0u8; 0x1300];
        let mut phases = vec![0x1234u32; 256];
        block[0x1294] = slot;
        block[0x950..0x954].copy_from_slice(&stage.to_le_bytes());
        phases[usize::from(slot)] = phase;
        // SAFETY: both are large enough.
        unsafe { read(block.as_ptr(), phases.as_ptr()) }
    }

    #[test]
    fn the_keep_rule_over_title_tls_states() {
        // Outside any pass: kept whatever stage TLS+0x950 still holds.
        for stage in [0, 15, 30, 65, 67, 99] {
            let (p, s) = state(3, u32::MAX, stage);
            assert!(keep(p, s), "outside a pass, stale stage {stage}");
        }
        // Inside a pass: only UI (30), plane-1 compute (16), Scaleform to
        // texture (8).
        for (stage, want) in [(30, true), (16, true), (8, true), (15, false), (65, false), (67, false), (74, false), (1, false), (21, false), (12, false)] {
            let (p, s) = state(7, 12 + 3, stage);
            assert_eq!(keep(p, s), want, "stage {stage} inside a pass");
        }
        // The phase is read from the slot the block names.
        let (p, _) = state(200, 0xffff_ffff, 67);
        assert_eq!(p, u32::MAX);
        let (p, _) = state(0, 5, 67);
        assert_eq!(p, 5);
    }

    #[test]
    fn dropped_stretches_still_record_state_and_barriers() {
        use Command::*;
        for c in [State, Barrier] {
            assert!(records(c, false) && records(c, true), "{c:?}");
        }
        for c in [Draw, Dispatch, Indirect, Copy, Clear] {
            assert!(!records(c, false) && records(c, true), "{c:?}");
        }
    }

    #[test]
    fn the_switch_and_the_ui_names() {
        assert!(parse(Some("null@frame+1250,ui=keep")));
        assert!(parse(Some("null,ui=keep")));
        assert!(parse(Some(" null@world , ui=keep ")));
        assert!(!parse(Some("null@frame+1250")));
        assert!(!parse(Some("ui=keep")));
        assert!(!parse(None));
        for n in ["FTexTGCxformMul_1a2b3c4d_Wrapped", "scaleform:FVertexCxform_00000000_Wrapped", "PS_Bink_a55b8f10_Wrapped", "PS_MinimapWaterDraw_1c76317a_Wrapped", "PS_Textured_IM_c38bc63c_Wrapped", "PS_BlitUISDRToBackbuffer_8e9c2e8a_Wrapped", "im:CS_UIOverlayAlphaCompensation_a4006c24_Wrapped"] {
            assert!(ui_name(n), "{n}");
        }
        for n in ["PS_PostFX_1_Wrapped", "Fxaa_0_Wrapped", "VS_Bink_a55b8f10_Wrapped", "PS_DeferredLighting_0_Wrapped", "avgLuminanceCompute:LuminanceFromHDRBinkCS_8ab9a11c_Wrapped"] {
            assert!(!ui_name(n), "{n}");
        }
    }
}
