//! `XNW_GPU=null...,ui=keep` (default off): in null mode the title's
//! own UI still reaches the host GPU, over our renderer's picture.
//!
//! The rule (static V): our list methods run on the title's
//! recording thread, so its static TLS block (`TEB.TLS[TITLE_TLS_INDEX]`)
//! says which pass records. `slot = u8 [tls+0x1294]`, `phase = u32
//! [title 0x149C3B750 + 4*slot]` (pass begin writes it, pass end writes -1:
//! 0x1450041B0), `stage = u32 [tls+0x950]` (0x14325D2A0). A command is
//! **kept** when `stage` is 30 (UI, HUD, Scaleform), 16 (plane-1 alpha
//! compensation) or 8 (Scaleform to texture) inside a pass; TLS+0x950 is not
//! cleared at pass end (0x145004160 clears only +0x972), so the phase is
//! tested first. (a test run hung the card at world entry, 50.6 s, with
//! every kept command "outside" a pass): outside a pass (`phase == -1`) a
//! command is kept only while the title's loading/front-end renderer slot
//! (0x14A210D60) is set and no world stage began this frame, and only as a
//! plain draw. Never kept anywhere: indirect executions, mesh draws, and
//! dispatches outside stage 16. No host pipeline is begun after the null
//! point by any path (make.rs refuses and counts), and kept draws use only
//! pipelines made before it.
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

/// PASS1: `XNW_WORLD_RENDER=...,uiwait=on|off` (default off): kept UI lists
/// are waited for on the submitting thread (on: as before), or sent without
/// a wait and finished at their fence (`host::Host::execute_nowait`).
pub(crate) fn ui_wait() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| switch(std::env::var("XNW_WORLD_RENDER").ok().as_deref(), "uiwait"));
    *ON
}

/// PASS1: `XNW_WORLD_RENDER=...,showwait=on|off` (default off): the show's
/// compose is waited for in PresentX (on: as before), or sent without a wait.
pub(crate) fn show_wait() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| switch(std::env::var("XNW_WORLD_RENDER").ok().as_deref(), "showwait"));
    *ON
}

/// Whether `name=on` (or `=1`) is among the comma-separated switches (the last one wins).
pub(crate) fn switch(v: Option<&str>, name: &str) -> bool {
    v.unwrap_or_default().split(',').filter_map(|p| p.trim().strip_prefix(name)?.strip_prefix('=')).last().is_some_and(|v| matches!(v.trim(), "on" | "1"))
}

/// The phase read for a pass recorded inline.
pub(crate) const INLINE: u32 = 0xffff_fffe;

/// (phase, stage) from a title TLS block and the phase table.
///
/// Pass begin 0x140369A80 writes the phase table from the phase's
/// +12 field, which is -1 for a pass recorded inline (then the table reads
/// -1 inside the pass too), and stores the phase's vt+8 byte at TLS+0x972,
/// which pass end 0x145004160 clears [V the code]. WARP g27w2 never saw
/// stages 1, 15, 16, 30, 36 inside a pass: the render thread's inline ones
/// So a -1 phase with TLS+0x972 set reads as [`INLINE`]: inside a pass.
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
        let inside = block.add(0x972).read() != 0;
        (if phase == u32::MAX && inside { INLINE } else { phase }, stage)
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

/// The title's loading/front-end render function slot (
/// `qword_14A210D60`, non-zero while a loading or front-end renderer
/// replaces the world frame; watch `loading_render_fn`, a test run: set from
/// 2.55 s, cleared at 49.70 s, world entry).
fn loading() -> bool {
    let base = super::object::title_base();
    // SAFETY: a u64 in the title's mapped data.
    base != 0 && unsafe { ((base + (0x1_4a21_0d60 - 0x1_4000_0000)) as *const u64).read_volatile() } != 0
}

/// A world stage (inside a pass, not 30/16/8) was recorded since the
/// last null present.
static WORLD_BEGUN: AtomicBool = AtomicBool::new(false);

/// A card run: the front-end and loading renderers record inside an
/// inline pass whose TLS stage is 0 ("dropped by reason world stage 672;
/// inline passes by stage 0: 672; loading renderer on", nothing kept). Stage
/// 0 is no world stage, so while the loading renderer slot is set such a
/// command reads as outside a pass (the earlier front-end rule then applies).
fn front(phase: u32, stage: u32) -> u32 {
    // KEEP1: not once a world stage recorded in this frame (the frames
    // around world entry, the loading slot still set).
    if phase == INLINE && stage == 0 && loading() && !WORLD_BEGUN.load(Relaxed) { u32::MAX } else { phase }
}

/// Whether a stretch (phase, stage) is kept, given the loading flag and
/// whether a world stage began this frame: inside a pass only
/// stages 30/16/8; outside one only while the title shows a loading or
/// front-end renderer and no world stage began.
pub(crate) fn keep_stretch(phase: u32, stage: u32, loading: bool, world_begun: bool) -> bool {
    if phase != u32::MAX {
        return matches!(stage, 30 | 16 | 8 | 6);
    }
    loading && !world_begun
}

/// Whether the stretch recorded now on this thread is kept (uploads and
/// host copies for it are made).
pub(crate) fn keep_now() -> bool {
    on() && (EXECUTING.with(|e| e.get()) || DRAWING.with(|d| d.get()) || here().is_some_and(|(p, s)| keep_stretch(front(p, s), s, loading(), WORLD_BEGUN.load(Relaxed))))
}

/// Whether a null-mode refusal (uploads, host copies) stands.
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
    /// A plain (direct or indexed) draw.
    Draw,
    /// A mesh draw (runs as compute plus a draw).
    Mesh,
    Dispatch,
    Indirect,
    Copy,
    Clear,
}
const KINDS: [&str; 8] = ["state", "barrier", "draw", "mesh", "dispatch", "indirect", "copy", "clear"];

/// Whether `command` is recorded in a stretch that `keep` says.
pub(crate) fn records(command: Command, keep: bool) -> bool {
    keep || matches!(command, Command::State | Command::Barrier)
}

/// Why a command is dropped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Drop {
    /// Inside a pass of a world stage.
    WorldStage,
    /// Outside a pass while the title shows no loading/front-end renderer.
    OutsideNotLoading,
    /// Outside a pass after a world stage began this frame.
    OutsideWorldBegun,
    /// Outside a pass and not a plain draw.
    OutsideNotDraw,
    /// An indirect execution or mesh draw (never kept).
    IndirectOrMesh,
    /// A dispatch outside stage 16.
    Dispatch,
}
const DROPS: [&str; 6] = ["world stage", "outside, not loading", "outside, world begun", "outside, not a plain draw", "indirect or mesh", "dispatch outside stage 16"];

/// How a kept command is kept: freely, or only if its program is named
/// and of the UI class.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kept {
    Free,
    UiNameOnly,
}

/// the earlier rule over one command. (WARP run g27w2, 140 s of world: no
/// command ever read as inside a stage-30/16/8 pass, while 37,000 a second
/// read as outside one): outside a pass with TLS+0x950 holding 30/16/8 (an
/// inline UI pass, or the stale stage after it), a plain draw is kept if
/// its program is named and of the UI class.
pub(crate) fn decide_kept(phase: u32, stage: u32, loading: bool, world_begun: bool, c: Command) -> Result<Kept, Drop> {
    match decide(phase, stage, loading, world_begun, c) {
        Ok(()) => Ok(if phase == u32::MAX { Kept::Free } else if matches!(stage, 30 | 6) { Kept::UiNameOnly } else { Kept::Free }),
        // KEEP1: and at a stale stage 6, the Bink movie: in the
        // world its draws are recorded outside a pass too, and were dropped.
        Err(Drop::OutsideNotLoading | Drop::OutsideWorldBegun) if c == Command::Draw && matches!(stage, 30 | 16 | 8 | 6) => Ok(Kept::UiNameOnly),
        Err(e) => Err(e),
    }
}

/// the earlier rule over one command (without the earlier outside UI draws).
pub(crate) fn decide(phase: u32, stage: u32, loading: bool, world_begun: bool, c: Command) -> Result<(), Drop> {
    use Command::*;
    let outside = phase == u32::MAX;
    // KEEP1: stage 6 draws the Bink movie into its texture: plain draws,
    // by name only ([`decide_kept`]).
    if !outside && stage == 6 && c == Draw {
        return Ok(());
    }
    if !outside && !matches!(stage, 30 | 16 | 8) {
        return Err(Drop::WorldStage);
    }
    if matches!(c, Indirect | Mesh) {
        return Err(Drop::IndirectOrMesh);
    }
    if outside {
        if !loading {
            return Err(Drop::OutsideNotLoading);
        }
        if world_begun {
            return Err(Drop::OutsideWorldBegun);
        }
        // KEEP1: and clears (the front end clears its planes; a test run kept
        // every loading command on the card for 16 s).
        if c != Draw && c != Clear {
            return Err(Drop::OutsideNotDraw);
        }
        return Ok(());
    }
    if c == Dispatch && stage != 16 {
        return Err(Drop::Dispatch);
    }
    Ok(())
}

const STAGES: usize = 128;
/// Index for commands outside any pass.
const OUTSIDE: usize = STAGES - 1;
static KEPT: [AtomicU64; STAGES] = [const { AtomicU64::new(0) }; STAGES];
static DROPPED: [AtomicU64; STAGES] = [const { AtomicU64::new(0) }; STAGES];
/// Commands read as inside an inline pass, by stage.
static INLINE_SEEN: [AtomicU64; STAGES] = [const { AtomicU64::new(0) }; STAGES];
static KEPT_KIND: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
static DROPPED_WHY: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
/// Dropped commands since the last present (a frame without any shows the
/// title's planes).
static DROPPED_FRAME: AtomicU64 = AtomicU64::new(0);
/// Kept draws without a pipeline made before the null point; kept stage-30
/// draws of a non-UI program; lists executed, lists skipped.
pub(crate) static NOT_MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static NOT_UI: AtomicU64 = AtomicU64::new(0);
pub(crate) static EXECUTED: AtomicU64 = AtomicU64::new(0);
pub(crate) static SKIPPED: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// The decision for the command being recorded on this thread.
    static LAST: std::cell::Cell<Option<Kept>> = const { std::cell::Cell::new(None) };
}

/// How the command being recorded on this thread was kept, if it was (for
/// its pipeline lookup).
pub(crate) fn command_kept() -> Option<Kept> {
    LAST.with(|l| l.get())
}

/// The first world frames (presented after the null point with no
/// loading renderer), whose kept commands are listed in the log.
static WORLD_FRAMES: AtomicU64 = AtomicU64::new(0);
static DUMPED: AtomicU64 = AtomicU64::new(0);

/// Whether kept commands are listed now (world frames 1-3, 3,000 lines at most).
pub(crate) fn dumping() -> bool {
    (1..=3).contains(&WORLD_FRAMES.load(Relaxed)) && DUMPED.fetch_add(1, Relaxed) < 3000
}

/// A draw-like command being recorded: whether it is kept (counted by
/// stage, kind and reason). Always true with the switch off or before the
/// null point.
pub(crate) fn command(c: Command) -> bool {
    if !on() || !super::frameskip::null_now() {
        LAST.with(|l| l.set(Some(Kept::Free)));
        return true;
    }
    let (phase, stage) = here().unwrap_or((0, OUTSIDE as u32 - 1));
    if phase == INLINE {
        INLINE_SEEN[(stage as usize).min(STAGES - 2)].fetch_add(1, Relaxed);
    }
    let phase = front(phase, stage);
    if phase != u32::MAX && !matches!(stage, 30 | 16 | 8 | 6) {
        WORLD_BEGUN.store(true, Relaxed);
    }
    let d = decide_kept(phase, stage, loading(), WORLD_BEGUN.load(Relaxed), c);
    WHY.with(|w| w.set(d.err()));
    let at = if phase == u32::MAX { OUTSIDE } else { (stage as usize).min(STAGES - 2) };
    match d {
        Ok(_) => {
            KEPT[at].fetch_add(1, Relaxed);
            KEPT_KIND[c as usize].fetch_add(1, Relaxed);
            if dumping() && c != Command::Draw {
                xnw_core::info!("null ui: world frame {} kept {:?} at stage {}", WORLD_FRAMES.load(Relaxed), c, if phase == u32::MAX { "outside".to_string() } else { stage.to_string() });
            }
        }
        Err(why) => {
            DROPPED[at].fetch_add(1, Relaxed);
            DROPPED_WHY[why as usize].fetch_add(1, Relaxed);
            DROPPED_FRAME.fetch_add(1, Relaxed);
        }
    }
    // Only a draw or dispatch looks its pipeline up; any other command
    // leaves no decision behind (see [`done`]).
    LAST.with(|l| l.set(d.ok().filter(|_| matches!(c, Command::Draw | Command::Dispatch))));
    records(c, d.is_ok())
}

/// This thread's stage when inside a pass (for the stage-30 name check
/// and the world-frame list).
pub(crate) fn stage_in_pass() -> Option<u32> {
    here().and_then(|(p, s)| (p != u32::MAX).then_some(s))
}

// ---- KEEP1 (follow-up): where kept UI work goes ----
//
// Runs 172/173 (card): the Scaleform text draws are kept and made, yet the
// credits text and the "TWO YEARS LATER" card are missing in the world. So
// the log says, once per (program, target): what each kept draw draws into
// (a presented plane, a texture kept UI draws wrote, another texture), which
// stage-30 draws are refused by name, and every draw, copy and clear, kept or
// dropped, that touches a presented plane or a UI-written texture: the
// consumer of the Scaleform target is among them.

/// The title resources of the planes last presented in a null frame:
/// scene, UI.
/// Each plane is a chain of buffers the title alternates (keep1d: UI
/// 0x..8420 / 0x..84f0, scene 0x..8830 / 0x..8900): the last two of each
/// ([scene, scene before, UI, UI before]).
static PLANES: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// Targets (title resources) kept UI draws drew into, newest last (a ring).
static WRITTEN: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
static WRITTEN_AT: AtomicU64 = AtomicU64::new(0);
/// Null presents by kind since the last line: composed (ours under the
/// UI plane), shown as the title drew them, single plane.
pub(crate) static PRESENTS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Clears since the last line: kept into the UI plane, the scene plane, a
/// UI-written texture; dropped into the same three.
static CLEARS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

thread_local! {
    /// The first colour target (title resource) and its size of the draw
    /// being recorded on this thread.
    static TARGET: std::cell::Cell<(usize, u32, u32)> = const { std::cell::Cell::new((0, 0, 0)) };
}

/// PresentX in a null frame: the planes' title resources.
pub(crate) fn note_planes(scene: usize, ui: usize) {
    for (at, r) in [(0, scene as u64), (2, ui as u64)] {
        let last = PLANES[at].load(Relaxed);
        if r != 0 && r != last {
            if r != PLANES[at + 1].load(Relaxed) {
                EPOCH.fetch_add(1, Relaxed);
            }
            PLANES[at + 1].store(last, Relaxed);
            PLANES[at].store(r, Relaxed);
        }
    }
}

/// What a title resource is to the kept UI: 1 the UI plane, 0 the scene
/// plane, 2 a texture kept UI draws wrote, 3 another texture, 4 none.
pub(crate) fn class(resource: usize) -> usize {
    let r = resource as u64;
    if r == 0 {
        4
    } else if r == PLANES[2].load(Relaxed) || r == PLANES[3].load(Relaxed) {
        1
    } else if r == PLANES[0].load(Relaxed) || r == PLANES[1].load(Relaxed) {
        0
    } else if WRITTEN.iter().any(|w| w.load(Relaxed) == r) {
        2
    } else if WANTED.iter().any(|w| w.load(Relaxed) == r) {
        5
    } else {
        3
    }
}
const CLASSES: [&str; 6] = ["the scene plane", "the UI plane", "a UI-written texture", "a texture", "no target", "a wanted source target"];

/// Render targets a refused UI-plane consumer sampled (the
/// 2560x1440 sRGB target `ps_srgb_write` puts on the plane at world entry
/// and at 77 s, which nothing kept wrote): every command that writes one is
/// named, kept or dropped, with the reason ([`draw_seen`], [`clear_seen`],
/// [`copy_seen`]), in loading frames too.
static WANTED: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
static WANTED_AT: AtomicU64 = AtomicU64::new(0);
/// Commands into wanted targets since the last line: draws kept, draws
/// dropped, clears, copies.
static WANTED_SEEN: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Counts every change of the planes, the UI-written and the wanted lists.
static EPOCH: AtomicU64 = AtomicU64::new(1);

thread_local! {
    /// The class of the last draw target on this thread and the [`EPOCH`] it was read at.
    static CLASS: std::cell::Cell<(usize, u64, usize)> = const { std::cell::Cell::new((0, 0, 4)) };
}

fn want(resource: usize) {
    if class(resource) == 3 {
        let at = WANTED_AT.fetch_add(1, Relaxed) as usize % WANTED.len();
        WANTED[at].store(resource as u64, Relaxed);
        EPOCH.fetch_add(1, Relaxed);
    }
}

thread_local! {
    /// Why the command being recorded on this thread is dropped.
    static WHY: std::cell::Cell<Option<Drop>> = const { std::cell::Cell::new(None) };
}

fn why_text() -> &'static str {
    WHY.with(|w| w.get()).map_or("kept by the stage rule", |d| DROPS[d as usize])
}

/// The pass state of this thread, as text.
fn where_text() -> String {
    match here() {
        None => "no title thread".into(),
        Some((INLINE, s)) => format!("inline pass stage {s}"),
        Some((u32::MAX, s)) => format!("outside a pass (TLS stage {s})"),
        Some((_, s)) => format!("pass stage {s}"),
    }
}

/// Whether a line about (`a`, `b`, `kind`) was not written yet (600 lines).
fn first_seen(a: usize, b: usize, kind: u8) -> bool {
    static SEEN: std::sync::Mutex<Option<std::collections::HashSet<(usize, usize, u8)>>> = std::sync::Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let set = g.get_or_insert_with(Default::default);
    set.len() < 600 && set.insert((a, b, kind))
}

/// Whether the instrument runs: the switch on, null mode, the world (no
/// loading renderer).
fn watching() -> bool {
    on() && super::frameskip::null_now() && !loading()
}

/// A draw is recorded (`kept`: the decision of [`command`]): its first
/// colour target is remembered for [`drew`] and [`refused`]; a dropped draw
/// into a plane or a UI-written texture is named once (`names` of
/// title pipeline `key`).
pub(crate) fn draw_seen(kept: bool, target: usize, size: (u32, u32), key: usize, names: impl FnOnce() -> String) {
    if !on() {
        return;
    }
    TARGET.with(|t| t.set((target, size.0, size.1)));
    CONSUMER.with(|c| c.set(false));
    if !super::frameskip::null_now() {
        return;
    }
    // `XNW_UI_WATCH=0`: dropped draws are not looked at (to measure what
    // the instrument costs; kept UI work is the same either way).
    static WATCH: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_UI_WATCH").map_or(true, |v| v.trim() != "0"));
    if !kept && !*WATCH {
        return;
    }
    // Every draw of the world comes here (60-90 k a second): the class of
    // the last target on this thread is kept while the lists do not change.
    let epoch = EPOCH.load(Relaxed);
    let (was, at, cached) = CLASS.with(|c| c.get());
    let c = if was == target && at == epoch {
        cached
    } else {
        let c = class(target);
        CLASS.with(|k| k.set((target, epoch, c)));
        c
    };
    if c == 5 {
        WANTED_SEEN[usize::from(!kept)].fetch_add(1, Relaxed);
    }
    if kept || !(c == 5 || c <= 2 && watching()) {
        return;
    }
    if first_seen(key, target, 0) {
        xnw_core::info!("null ui: dropped draw ({}) into {} {target:#x} {}x{}, {}: {}", why_text(), CLASSES[c], size.0, size.1, where_text(), names());
    }
}

/// The kept draw being recorded has its pipeline and is drawn: its target
/// counts as UI-written (unless a plane); named once per target.
pub(crate) fn drew(key: usize, kind: u8, names: impl FnOnce() -> String) {
    let (target, w, h) = TARGET.with(|t| t.get());
    movie_drew(kind, target);
    let c = class(target);
    // In the world only: while loading every draw is kept. A wanted
    // source target a UI-class program draws into becomes UI-written.
    if matches!(c, 3 | 5) && !loading() {
        let at = WRITTEN_AT.fetch_add(1, Relaxed) as usize % WRITTEN.len();
        WRITTEN[at].store(target as u64, Relaxed);
        EPOCH.fetch_add(1, Relaxed);
    }
    if watching() && first_seen(key, target, 1) {
        xnw_core::info!("null ui: kept draw into {} {target:#x} {w}x{h}, {}: {}", CLASSES[class(target)], where_text(), names());
    }
}

thread_local! {
    /// The draw being recorded was admitted as a consumer ([`consumer_name`]):
    /// its sources are checked once its pipeline is known ([`consumer_ok`]).
    static CONSUMER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// A kept draw is being prepared on this thread ([`Drawing`]).
    static DRAWING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// FLK3: the draw being recorded is the kept Bink draw ([`movie_drew`]):
    /// its sampled planes are noted once its pipeline is known ([`take_bink`]).
    static BINK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// While a kept draw is prepared and recorded: what it reads is brought up
/// to date as for kept work ([`keep_now`]). Runs 172/173: in the world the
/// kept Scaleform text (FText: the glyph cache the title writes on the CPU)
/// and the kept Bink draw (the planes its decoder writes) drew nothing: the
/// stretch they are recorded in (outside a pass, no loading renderer) is not
/// a kept one, so their textures' uploads were refused (host.rs upload gate).
pub(crate) struct Drawing;

impl Drawing {
    pub(crate) fn begin() -> Drawing {
        DRAWING.with(|d| d.set(true));
        Drawing
    }
}

impl std::ops::Drop for Drawing {
    fn drop(&mut self) {
        DRAWING.with(|d| d.set(false));
        done();
    }
}

/// The kept draw or dispatch is recorded: its decision and target do not
/// outlive it. A test run: pipeline look-ups outside a draw (a pipeline being
/// set in a world pass) ran with the last kept draw's decision and target,
/// so world programs (ps_sky_paraboloid_reflection in pass stage 18,
/// PS_DistantLights, the trail map clear) were logged as refused "into the
/// UI plane": they never drew there.
pub(crate) fn done() {
    LAST.with(|l| l.set(None));
    TARGET.with(|t| t.set((0, 0, 0)));
    CONSUMER.with(|c| c.set(false));
    BINK.with(|b| b.set(false));
}

/// The world's blits that put kept UI work on the UI plane, by name: made
/// in null mode and drawn only into the UI plane and only when every
/// texture they sample is one kept UI draws wrote ([`consumer_ok`]): never
/// a blit of a scene buffer the title did not draw. keep1d (box, WARP):
/// `ps_srgb_write` and `PS_BlitDiffuse` are drawn into the UI plane every
/// world frame at a stale stage 30; `PS_BlitColour` is the front end's.
pub(crate) fn consumer_name(name: &str) -> bool {
    let n = name.rsplit(':').next().unwrap_or(name);
    n.starts_with("ps_srgb_write_") || n.starts_with("PS_BlitDiffuse_") || n.starts_with("PS_BlitColour_")
}

/// Whether the draw being recorded, of a consumer program, is into the UI
/// plane; then its sources are checked at the draw ([`take_consumer`]).
pub(crate) fn consumer_admit() -> bool {
    let ok = !loading() && class(TARGET.with(|t| t.get()).0) == 1;
    CONSUMER.with(|c| c.set(ok));
    ok
}

/// Whether the draw being recorded was admitted as a consumer; resets.
pub(crate) fn take_consumer() -> bool {
    CONSUMER.with(|c| c.replace(false))
}

/// A consumer draw's sampled textures (title resources): drawn only when
/// each is a texture kept UI draws wrote. Named once per source set.
///
/// A test run: `ps_srgb_write` into the UI plane was refused three times, each
/// with another plain texture as its source: pictures that reach the host
/// by upload from title memory (a decoded image or movie frame), which
/// null mode does not lose. So a source is also good when the title writes
/// it itself ([`uploaded`]): never a render, depth or compute target (those
/// hold only what dropped work would have drawn), never the destination of
/// a host copy.
pub(crate) fn consumer_ok(key: usize, sources: &[usize], names: impl FnOnce() -> String) -> bool {
    let ok = sources.iter().all(|&s| class(s) == 2 || uploaded(s));
    if !ok {
        // What writes the refused sources is named from now on.
        for &s in sources.iter().filter(|&&s| class(s) == 3 && !uploaded(s)) {
            want(s);
        }
    }
    if first_seen(key, sources.first().copied().unwrap_or(0), 7 + u8::from(ok)) {
        let text: Vec<String> = sources.iter().map(|&s| source_text(s)).collect();
        xnw_core::info!("null ui: consumer draw into the UI plane {}: {}; samples [{}]; {}", if ok { "kept" } else { "refused (a source neither kept UI draws nor the title's own uploads wrote)" }, names(), text.join(", "), where_text());
    }
    if ok {
        movie_blit(sources);
    }
    ok
}

/// Whether the title writes `resource` itself (its texels are uploaded from
/// title memory): it has a host copy without render-target, depth or
/// unordered-access flags, no host copy or resolve wrote it, and it is no
/// presented plane.
fn uploaded(resource: usize) -> bool {
    let r = resource as super::object::This;
    class(resource) == 3 && super::resource::host_copy(r).is_some() && !super::texture::gpu_output(r) && !super::texture::is_host_written(r)
}

/// A source texture for the log: class, address, size, format, flags and
/// how it is written.
fn source_text(resource: usize) -> String {
    let r = resource as super::object::This;
    let Some(copy) = super::resource::host_copy(r) else {
        return format!("{} {resource:#x} (no host copy)", CLASSES[class(resource)]);
    };
    // SAFETY: a live host texture.
    let d = unsafe { copy.texture.GetDesc() };
    let how = if class(resource) == 2 {
        "written by kept UI draws"
    } else if super::texture::is_host_written(r) {
        "written by a host copy"
    } else if super::texture::gpu_output(r) {
        "a render, depth or compute target (nothing kept wrote it)"
    } else {
        "uploaded from title memory"
    };
    format!("{} {resource:#x} {}x{} format {} mips {} flags {:#x}: {how}", CLASSES[class(resource)], d.Width, d.Height, d.Format.0, d.MipLevels, d.Flags.0)
}

/// Programs the pack does not name that are of the UI class, by the hash
/// of their last program: the pixel program e97a4bf1 (964 B; with vertex
/// 9ed51e41, one root SRV, no texture) is drawn into the Scaleform texture
/// at stage 8 and into the UI plane (keep1d 95.670 s, 118.932 s).
pub(crate) fn ui_hash(hash: u128) -> bool {
    hash == 0xe97a4bf11dc82ec6de45359cc20d9f0f
}

/// A target clear dropped by the stage rule is kept all the same in a world
/// frame when it clears the UI plane or a texture kept UI draws wrote (the
/// title clears the plane once a frame at a stale stage 15, the Scaleform
/// texture at 7: keep1d `clears dropped: UI plane 3-4, UI-written texture 85`
/// a second): without it kept UI draws pile up frame over frame.
///
/// A card run (a card flickers): the plane's clear is kept at once (its
/// writers draw every frame); the clear of a UI-written texture is HELD
/// (returns 2) and done only when a kept draw into that texture follows in
/// the same frame ([`held_clear`]): such a texture is redrawn at its own
/// rate (a Scaleform movie's, a Bink movie's) while the title frames run
/// faster, and a clear kept in a frame without its writer leaves it empty
/// for whoever shows it that frame.
pub(crate) fn clear_admit(target: usize) -> u8 {
    if !on() || !super::frameskip::null_now() || loading() {
        return 0;
    }
    match class(target) {
        1 => 1,
        2 => 2,
        _ => 0,
    }
}

/// Clears held for a kept draw of the same frame: target, view handle,
/// colour, frame.
static HELD: std::sync::Mutex<Vec<(usize, u64, [f32; 4], u64)>> = std::sync::Mutex::new(Vec::new());
static HELD_N: AtomicU64 = AtomicU64::new(0);
/// Clears held and clears done at a kept draw, since the last line.
static HELD_COUNTS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Holds the clear of `target` (the latest of a frame wins).
pub(crate) fn hold_clear(target: usize, handle: u64, color: [f32; 4]) {
    let frame = super::queue::presented();
    let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
    held.retain(|h| h.0 != target && h.3 == frame);
    if held.len() < 16 {
        held.push((target, handle, color, frame));
    }
    HELD_N.store(held.len() as u64, Relaxed);
    HELD_COUNTS[0].fetch_add(1, Relaxed);
}

/// The clear held this frame for the target of the kept draw being
/// recorded, taken: the caller does it before the draw.
pub(crate) fn held_clear() -> Option<(usize, u64, [f32; 4])> {
    if HELD_N.load(Relaxed) == 0 {
        return None;
    }
    let target = TARGET.with(|t| t.get()).0;
    let frame = super::queue::presented();
    let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
    let at = held.iter().position(|h| h.0 == target)?;
    let (_, handle, color, of) = held.swap_remove(at);
    HELD_N.store(held.len() as u64, Relaxed);
    if of != frame {
        return None;
    }
    HELD_COUNTS[1].fetch_add(1, Relaxed);
    Some((target, handle, color))
}

/// The kept draw being recorded is refused by its program's name.
pub(crate) fn refused(key: usize, names: impl FnOnce() -> String) {
    let (target, w, h) = TARGET.with(|t| t.get());
    if watching() && first_seen(key, target, 2) {
        xnw_core::info!("null ui: refused draw (program not of the UI class) into {} {target:#x} {w}x{h}, {}: {}", CLASSES[class(target)], where_text(), names());
    }
}

/// A GPU copy is recorded: named once when either side is a plane or a
/// UI-written texture.
pub(crate) fn copy_seen(kept: bool, dst: usize, src: usize) {
    if !watching() {
        return;
    }
    let (d, s) = (class(dst), class(src));
    if d == 5 {
        WANTED_SEEN[3].fetch_add(1, Relaxed);
    }
    if (d <= 2 || s <= 2 || d == 5 || s == 5) && first_seen(dst, src, 3 + u8::from(kept)) {
        xnw_core::info!("null ui: {} copy from {} {src:#x} to {} {dst:#x}, {}", if kept { "kept" } else { "dropped" }, CLASSES[s], CLASSES[d], where_text());
    }
}

/// A target clear is recorded: counted when it clears a plane or a
/// UI-written texture.
pub(crate) fn clear_seen(kept: bool, target: usize) {
    if !on() || !super::frameskip::null_now() {
        return;
    }
    let c = class(target);
    if kept && target as u64 == MOVIE_TARGET.load(Relaxed) {
        MOVIE_FRAME.fetch_or(8, Relaxed);
    }
    if c == 5 {
        WANTED_SEEN[2].fetch_add(1, Relaxed);
        if first_seen(target, c, 5 + u8::from(kept)) {
            xnw_core::info!("null ui: {} clear of {} {target:#x}, {}", if kept { "kept" } else { "dropped" }, CLASSES[c], where_text());
        }
    }
    if c <= 2 {
        CLEARS[c + if kept { 0 } else { 3 }].fetch_add(1, Relaxed);
        if watching() && first_seen(target, c, 5 + u8::from(kept)) {
            xnw_core::info!("null ui: {} clear of {} {target:#x}, {}", if kept { "kept" } else { "dropped" }, CLASSES[c], where_text());
        }
    }
}

/// KEEP1: whether a not-UI program is still named in the log (64 lines).
pub(crate) fn note_not_ui() -> bool {
    static SAID: AtomicU64 = AtomicU64::new(0);
    SAID.fetch_add(1, Relaxed) < 64
}

/// The world frame number for the kept-command list.
pub(crate) fn world_frame() -> u64 {
    WORLD_FRAMES.load(Relaxed)
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

/// KEEP1: the front end's own pixel programs that are not of the UI class
/// and are made in null mode for draws kept freely (the loading renderer
/// runs): `PS_BlitColour` (intro video and title cards missing
/// without it; 28 instructions) and `ps_srgb_write` (box capture keep1c,
/// 10.6 s, with the Bink movie; 70 instructions); no wave ops, no buffer
/// reads. Not in [`ui_name`]: that answer is cached per pipeline
/// (`Desc::ui_class`) and admits world draws at a stale stage 30, where
/// a blit of a scene buffer the title never drew would cover our picture.
pub(crate) fn front_name(name: &str) -> bool {
    let n = name.rsplit(':').next().unwrap_or(name);
    n.starts_with("PS_BlitColour_") || n.starts_with("ps_srgb_write_")
}

// ---- The movie chain, frame by frame (the "VI" card flickers) ----
//
// A Bink card in the world is: the unnamed fill pair and the Bink draw into
// a 2560x1440 target, then `ps_srgb_write` from that target onto the UI
// plane. Counted per presented frame in which the blit ran: whether the Bink
// draw, the fill and a clear of the target ran in that frame too.

/// The target the Bink draw last drew into.
static MOVIE_TARGET: AtomicU64 = AtomicU64::new(0);
/// This frame: 1 Bink drew, 2 the fill drew into the movie target, 4 the
/// blit sampled it, 8 it was cleared.
static MOVIE_FRAME: AtomicU64 = AtomicU64::new(0);
/// Frames with the blit since the last line, by the other three bits
/// (index: 1 Bink, 2 fill, 4 clear), and Bink draws, fills, blits in all.
static MOVIE_FRAMES: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
static MOVIE_COUNTS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

fn movie_drew(kind: u8, target: usize) {
    if kind == 1 && class(target) != 1 {
        BINK.with(|b| b.set(true));
        MOVIE_TARGET.store(target as u64, Relaxed);
        MOVIE_FRAME.fetch_or(1, Relaxed);
        MOVIE_COUNTS[0].fetch_add(1, Relaxed);
    } else if kind == 2 && target as u64 == MOVIE_TARGET.load(Relaxed) && target != 0 {
        MOVIE_FRAME.fetch_or(2, Relaxed);
        MOVIE_COUNTS[1].fetch_add(1, Relaxed);
    }
}

/// FLK3: the plane (first texture of its tables) the last Bink draw of
/// this frame sampled: the decoder's two plane sets alternate per movie
/// frame, so a flicker at the movie's rate names the set on screen.
static MOVIE_PLANE: AtomicU64 = AtomicU64::new(0);

/// FLK3: whether the draw being recorded is the kept Bink draw; resets.
pub(crate) fn take_bink() -> bool {
    BINK.with(|b| b.replace(false))
}

/// FLK3: the kept Bink draw being recorded samples `sources` (title
/// resources, its tables' order).
pub(crate) fn bink_planes(sources: &[usize]) {
    MOVIE_PLANE.store(sources.first().copied().unwrap_or(0) as u64, Relaxed);
}

/// A kept consumer blit sampled `sources`.
fn movie_blit(sources: &[usize]) {
    let m = MOVIE_TARGET.load(Relaxed);
    if m != 0 && sources.iter().any(|&s| s as u64 == m) {
        MOVIE_FRAME.fetch_or(4, Relaxed);
        MOVIE_COUNTS[2].fetch_add(1, Relaxed);
    }
}

/// A frame is presented: its movie bits are counted.
fn movie_present() {
    let f = MOVIE_FRAME.swap(0, Relaxed);
    // FLK3: the flicker counter names each present's movie chain (bits as
    // MOVIE_FRAME's, the Bink draw's plane set).
    xnw_core::live::note_movie(f as u8, MOVIE_PLANE.swap(0, Relaxed) as u16);
    if f & 4 != 0 {
        MOVIE_FRAMES[((f & 3) | (f & 8) >> 1) as usize].fetch_add(1, Relaxed);
    }
}

/// Whether the frame now presented had no dropped command; resets the
/// frame's state (dropped count, world begun) and counts world frames.
pub(crate) fn frame_all_kept() -> bool {
    WORLD_BEGUN.store(false, Relaxed);
    movie_present();
    if !loading() || WORLD_FRAMES.load(Relaxed) > 0 {
        WORLD_FRAMES.fetch_add(1, Relaxed);
    }
    // KEEP1: in the world always ours under the UI plane: a world frame
    // in which the render thread recorded nothing (keep1d: 2-3 presents a
    // second) would show the title's scene plane, which null mode never draws.
    let none_dropped = DROPPED_FRAME.swap(0, Relaxed) == 0;
    none_dropped && (loading() || WORLD_FRAMES.load(Relaxed) == 0)
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
    let named = |t: &[AtomicU64], names: &[&str]| -> String {
        let v: Vec<String> = t.iter().zip(names).filter_map(|(c, n)| {
            let k = c.swap(0, Relaxed);
            (k != 0).then(|| format!("{n} {k}"))
        }).collect();
        if v.is_empty() { "none".into() } else { v.join(", ") }
    };
    let (started, finished, ui) = super::pipeline::made_in_null();
    let n = |t: &[AtomicU64], i: usize| t[i].swap(0, Relaxed);
    let (held, done) = (n(&HELD_COUNTS, 0), n(&HELD_COUNTS, 1));
    if held != 0 {
        xnw_core::info!("null ui: clears of UI-written textures: held {held}, done with a kept draw of the same frame {done}");
    }
    if MOVIE_COUNTS.iter().any(|c| c.load(Relaxed) != 0) {
        let f: Vec<u64> = (0..8).map(|i| n(&MOVIE_FRAMES, i)).collect();
        let e = super::texture::movie_exits();
        xnw_core::info!(
            "null ui: movie: Bink draws {}, fills {}, blits {}; frames with the blit: with Bink and fill {}, Bink only {}, fill only {}, neither {}; of all those with a clear of the target {}; target {:#x}; plane refreshes (MOV1, {}): uploads {}, exits gate {} watch {} hash {} origin {}",
            n(&MOVIE_COUNTS, 0), n(&MOVIE_COUNTS, 1), n(&MOVIE_COUNTS, 2), f[3] + f[7], f[1] + f[5], f[2] + f[6], f[0] + f[4], f[4] + f[5] + f[6] + f[7], MOVIE_TARGET.load(Relaxed),
            if super::texture::movie_fresh() { "forced" } else { "XNW_MOVIE_FRESH=0" }, e[4], e[0], e[1], e[2], e[3]
        );
    }
    if WANTED_AT.load(Relaxed) != 0 {
        xnw_core::info!("null ui: wanted source targets: draws kept {}, draws dropped {}, clears {}, copies into {}", n(&WANTED_SEEN, 0), n(&WANTED_SEEN, 1), n(&WANTED_SEEN, 2), n(&WANTED_SEEN, 3));
    }
    xnw_core::info!(
        "null ui: presents: ours under the UI plane {}, the title's planes as drawn {}, one plane {}; planes scene {:#x} UI {:#x}; clears kept: scene plane {}, UI plane {}, UI-written texture {}; clears dropped: scene plane {}, UI plane {}, UI-written texture {}",
        n(&PRESENTS, 0), n(&PRESENTS, 1), n(&PRESENTS, 2), PLANES[0].load(Relaxed), PLANES[2].load(Relaxed),
        n(&CLEARS, 0), n(&CLEARS, 1), n(&CLEARS, 2), n(&CLEARS, 3), n(&CLEARS, 4), n(&CLEARS, 5)
    );
    xnw_core::info!(
        "null ui: kept by stage {}; kept by kind {}; dropped by stage {}; dropped by reason {}; inline passes by stage {}; loading renderer {}; kept but no pipeline made before the null point {}, stage-30 non-UI program {}; lists executed {}, skipped {}; host pipelines made since the null point {started} (in flight at it and not used: {finished}; for kept UI programs: {ui})",
        by(&KEPT),
        named(&KEPT_KIND, &KINDS),
        by(&DROPPED),
        named(&DROPPED_WHY, &DROPS),
        by(&INLINE_SEEN),
        if loading() { "on" } else { "off" },
        NOT_MADE.swap(0, Relaxed),
        NOT_UI.swap(0, Relaxed),
        EXECUTED.swap(0, Relaxed),
        SKIPPED.swap(0, Relaxed),
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
        state_inline(slot, phase, stage, false)
    }

    fn state_inline(slot: u8, phase: u32, stage: u32, inside: bool) -> (u32, u32) {
        let mut block = vec![0u8; 0x1300];
        block[0x972] = u8::from(inside);
        let mut phases = vec![0x1234u32; 256];
        block[0x1294] = slot;
        block[0x950..0x954].copy_from_slice(&stage.to_le_bytes());
        phases[usize::from(slot)] = phase;
        // SAFETY: both are large enough.
        unsafe { read(block.as_ptr(), phases.as_ptr()) }
    }

    /// PASS1: `uiwait=` and `showwait=` among XNW_WORLD_RENDER's switches (default off).
    #[test]
    fn wait_switches() {
        assert!(!switch(None, "uiwait") && !switch(Some("screen,gbuf=on"), "uiwait"));
        assert!(switch(Some("screen, uiwait=on ,x"), "uiwait") && !switch(Some("screen,uiwait=on"), "showwait"));
        assert!(!switch(Some("uiwait=on,uiwait=off"), "uiwait") && switch(Some("showwait=1"), "showwait"));
    }

    #[test]
    fn the_keep_rule_over_title_tls_states() {
        // Outside any pass (whatever stage TLS+0x950 still holds): kept only
        // while a loading/front-end renderer runs and no world stage began.
        for stage in [0, 15, 30, 65, 67, 99] {
            let (p, s) = state(3, u32::MAX, stage);
            assert!(keep_stretch(p, s, true, false), "outside a pass, stale stage {stage}");
            assert!(!keep_stretch(p, s, false, false), "outside, no loading renderer (a card run's world entry)");
            assert!(!keep_stretch(p, s, true, true), "outside, a world stage began this frame");
        }
        // Inside a pass: only UI (30), plane-1 compute (16), Scaleform to
        // texture (8), loading or not.
        for (stage, want) in [(30, true), (16, true), (8, true), (15, false), (65, false), (67, false), (74, false), (1, false), (21, false), (12, false)] {
            let (p, s) = state(7, 12 + 3, stage);
            for (loading, begun) in [(false, false), (true, false), (false, true)] {
                assert_eq!(keep_stretch(p, s, loading, begun), want, "stage {stage} inside a pass");
            }
        }
        // The phase is read from the slot the block names.
        let (p, _) = state(200, 0xffff_ffff, 67);
        assert_eq!(p, u32::MAX);
        let (p, _) = state(0, 5, 67);
        assert_eq!(p, 5);
        // An inline pass (phase -1, TLS+0x972 set) is inside a pass.
        let (p, s) = state_inline(4, u32::MAX, 30, true);
        assert_eq!(p, INLINE);
        assert!(keep_stretch(p, s, false, true));
        assert_eq!(decide_kept(p, s, false, true, Command::Draw), Ok(Kept::UiNameOnly));
        let (p, s) = state_inline(4, u32::MAX, 67, true);
        assert_eq!(decide(p, s, true, false, Command::Draw), Err(Drop::WorldStage));
    }

    #[test]
    fn g27_commands_by_kind() {
        use Command::*;
        const OUT: u32 = u32::MAX;
        // Outside a pass, loading, no world stage: plain draws only.
        assert_eq!(decide(OUT, 0, true, false, Draw), Ok(()));
        for c in [Dispatch, Copy] {
            assert_eq!(decide(OUT, 0, true, false, c), Err(Drop::OutsideNotDraw), "{c:?}");
        }
        assert_eq!(decide(OUT, 0, true, false, Clear), Ok(()));
        // KEEP1: stage 6 (Bink): plain draws only, by name.
        assert_eq!(decide_kept(15, 6, false, true, Draw), Ok(Kept::UiNameOnly));
        for c in [Dispatch, Indirect, Copy, Clear, Mesh] {
            assert_eq!(decide(15, 6, false, true, c), Err(Drop::WorldStage), "{c:?}");
        }
        for c in [Indirect, Mesh] {
            assert_eq!(decide(OUT, 0, true, false, c), Err(Drop::IndirectOrMesh), "{c:?}");
        }
        assert_eq!(decide(OUT, 0, false, false, Draw), Err(Drop::OutsideNotLoading));
        assert_eq!(decide(OUT, 0, true, true, Draw), Err(Drop::OutsideWorldBegun));
        // UI stages: draws, clears, copies; dispatch only in 16; never indirect or mesh.
        for stage in [30, 16, 8] {
            for c in [Draw, Copy, Clear] {
                assert_eq!(decide(15, stage, false, true, c), Ok(()), "{c:?} in {stage}");
            }
            assert_eq!(decide(15, stage, false, true, Indirect), Err(Drop::IndirectOrMesh));
            assert_eq!(decide(15, stage, false, true, Mesh), Err(Drop::IndirectOrMesh));
        }
        assert_eq!(decide(15, 16, false, true, Dispatch), Ok(()));
        assert_eq!(decide(15, 30, false, true, Dispatch), Err(Drop::Dispatch));
        // World stages: nothing.
        for c in [Draw, Dispatch, Indirect, Copy, Clear, Mesh] {
            assert_eq!(decide(15, 67, true, false, c), Err(Drop::WorldStage), "{c:?}");
        }
    }

    #[test]
    fn dropped_stretches_still_record_state_and_barriers() {
        use Command::*;
        for c in [State, Barrier] {
            assert!(records(c, false) && records(c, true), "{c:?}");
        }
        for c in [Draw, Mesh, Dispatch, Indirect, Copy, Clear] {
            assert!(!records(c, false) && records(c, true), "{c:?}");
        }
    }

    #[test]
    fn g27b_outside_ui_stage_draws_need_a_ui_name() {
        use Command::*;
        const OUT: u32 = u32::MAX;
        // Outside a pass, no loading renderer, TLS stage 30/16/8: plain draws by UI name only.
        for stage in [30, 16, 8] {
            assert_eq!(decide_kept(OUT, stage, false, true, Draw), Ok(Kept::UiNameOnly));
            for c in [Dispatch, Indirect, Mesh, Copy, Clear] {
                assert!(decide_kept(OUT, stage, false, true, c).is_err(), "{c:?}");
            }
        }
        // A world stage left in TLS: nothing.
        assert_eq!(decide_kept(OUT, 67, false, false, Draw), Err(Drop::OutsideNotLoading));
        // Loading renderer, no world stage: free; inside stage 30: by name (unnamed kept).
        assert_eq!(decide_kept(OUT, 67, true, false, Draw), Ok(Kept::Free));
        assert_eq!(decide_kept(15, 30, false, true, Draw), Ok(Kept::UiNameOnly));
        assert_eq!(decide_kept(15, 16, false, true, Dispatch), Ok(Kept::Free));
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
        assert!(front_name("PS_BlitColour_6e5520d1_Wrapped") && front_name("ps_srgb_write_dd820410_Wrapped"));
        assert!(!front_name("PS_BlitColourAndDepth_9a8e8f91_Wrapped") && !front_name("PS_PostFX_1_Wrapped") && !ui_name("PS_BlitColour_6e5520d1_Wrapped"));
        for n in ["PS_PostFX_1_Wrapped", "Fxaa_0_Wrapped", "VS_Bink_a55b8f10_Wrapped", "PS_DeferredLighting_0_Wrapped", "avgLuminanceCompute:LuminanceFromHDRBinkCS_8ab9a11c_Wrapped"] {
            assert!(!ui_name(n), "{n}");
        }
    }
}
