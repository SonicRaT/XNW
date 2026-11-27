//! Test-speed levers on the title's own game code (ENG1/ENG2): each calls a
//! function of the title from a detour on the thread that owns the state,
//! off by default, for build 1.0.2.1 only (the entry bytes of every function
//! called are checked first). Evidence:.
//!
//! `XNW_CLOCK=hh:mm`: the game clock (a 32-byte date-time at
//! `title+0xab5cdc0`: hour +0x10, minute +0x14, second +0x18) is held at
//! that time of day. The detour is on the clock tick `title+0x15120b0`
//! (it adds a minute every 5000 ms of game time): after each tick, when
//! the hour or minute is another, the title's raw setter
//! `title+0x1ab9b20(h, m, s)` is called (it writes the time through
//! `title+0x4ebb720` and restarts the minute; no date step). The timecycle
//! blend `title+0x5dcc40` reads the clock every frame, so sun and sky
//! follow. A scene that sets the time itself is set back at the next tick.
//!
//! `XNW_CINE=skip`: every in-engine cinematic (the instances of the manager
//! `qword_14A9ECAC8`) is skipped through the title's own skip as soon
//! as its gate allows. The detour is on the instance state machine's tick
//! `title+0x50f89c0(instance, flag)`; after it, for an instance that plays
//! (state +0x43C = 6 or 8) and is not being skipped (+0x497 bits 4-5 and
//! +0x496 bit 6 clear), the steps of the title's by-handle skip
//! `title+0x31602a0` are taken: its context test (`title+0x524c060`,
//! `title+0x524c030`), the group lock (`title+0x524c040` /
//! `title+0x52ec1a0`, both idle inside the cinematic update), the time stamp
//! at +0xE48, then `cine_skip(instance)` `title+0x3146250`, which acts only
//! when `cine_can_skip` `title+0x31462c0` holds (+0x496 bit 5, set by a scene
//! event `title+0x4c8ec30`; the definition's +0x158 bit 2; the skip point).
//! `XNW_CINE=force` sets +0x496 bit 5 itself first: a scene that is
//! skippable by definition is skipped before its event opens the window.
//! `XNW_CINE=all` also sets the definition's +0x158 bit 2 (FAST1: the
//! opening cinematic's instances have the window open and the definition
//! not skippable, `skip gate closed (window flag true, definition
//! skippable false)` at 53.5 s of box run fast1a), so the title's own skip
//! takes scenes it would not offer to skip. It breaks the title: box run
//! fast1d asked 115 skips in 30 s, then the title's int3 at
//! `title+0x22106e0` under the instance tick (161 s). Kept for study only.
//!
//! `XNW_INTRO=seekend[:<n>[:<seconds>]]` (ENG10): the opening's in-engine
//! scenes that the title does not offer to skip are taken to their end
//! through the title's own seek. Same detour as `XNW_CINE`. After the tick,
//! for an instance that plays (state 6), whose definition is not skippable
//! (+0x158 bit 2 clear), with no seek or skip under way (+0x495 bit 2,
//! +0x497 bits 4-5, +0x496 bit 6 clear) and whose section ends (+0x480, set
//! from the section's +0x50 by `title+0x24eb40`) at least `<seconds>`
//! (default 30) in, the steps of the title's seek method of a scene handle
//! `title+0x3162b60` (vtable 0x1463CDD30 slot 1) are taken: the group lock,
//! the +0x49 & 0x18 refusal, the stamp at +0xE48, then `cine_seek(instance,
//! t, 0, 0)` `title+0x518880` -> `title+0x5188b0` with t = end - 0.5 s.
//! The seek only sets +0x495 bit 2 and the target +0x484; the state machine
//! then goes 6 -> 8 (`title+0x50f92b8`: the seek state, which streams for
//! the target) -> 6, plays the last half second and ends as it does when
//! played through: `title+0x50f92f3` (+0x7FD, the timeline's end) ->
//! state 10, or the next section (`title+0x24ba00`). Half a second short of
//! the end on purpose: a seek to the end itself (t >= end - 1e-6) makes the
//! title mark the scene skipped (+0x496 bit 6, `title+0x51893d`), which its
//! by-handle getter `title+0x31603c0` reports. `<n>` seeks at most
//! (default 5), then the lever is off: the opening scene is one instance
//! of five sections (54.93, 78.57, 78.57, 78.57 and 69.90 s) that starts
//! over at the first when the fifth ends (box run eng10a: sought through
//! in 112.8-124.1 s, the anim-scene camera off at 124.09 s; played through
//! it is 360 s of scene time). What follows the opening did not come
//! sooner for it (eng10a, to 180 s): the script polls no scene method
//! after it starts the scenes, so it waits on something else (ENG10).
//! Lines: `intro: seek k of n asked ...`, `intro: instance .. state a -> b`.
//!
//! `XNW_SAVE_AT=<n>|<n>,force` (SAVE2): the title writes its autosave
//! once, its own way. After the clock tick, from world frame n (the stage
//! meter's count), `save_request_autosave` 0x1444817F0() is called, the body
//! of the title's script native 0x144461180; its gates are logged first
//! (`save: gates ...`), tried every 60 ticks, 30 tries. It queues the op
//! `g_saveop_autosave` (qword_1478B2680), which the title's save queue pump
//! runs over the next frames (steps 0-12; step 10 runs the write machine
//! 0x14289A590, state dword_1478CCD78, whose state 14 writes). Each change
//! of the op's queue state, step and result, the write state and the opcode
//! is one `save: op ...` line; at the end one line names the containers
//! under the saves folder with a newer version (`written slot SARC0015~..
//! vN (k bytes)`) or why not. `,force` sets the profile's autosave flag
//! (+0x410 of the object at vslot 0x48 of `qword_14AAD3CE0+8`) to 1 when it
//! is 0.
//!
//! `XNW_LOD_SCALE=<f>|<f>,force|read` (ENG6): the global LOD scale, the
//! settings field `lodScale` (graphics +0xD4, slider 0.1..1.0, file
//! `settings.xml`). The title's setter is one store in `settings_apply`
//! (`title+0x224360a`: `vmovss xmm0, [rsi+0xD4]; vmovss [title+0xa15e200],
//! xmm0`); its one reader is the getter `title+0x45f460`, called once a
//! frame by the scan setup (`title+0x80db3b`), by the LOD scale block
//! builder (`title+0x3e573e6`) and at `title+0x26507ed`. The getter returns
//! max(scale, 1.0) while `g_stream_slice_flags` bit 1 is set or the mode
//! test `title+0x45f4a0` holds (either of the two objects qword_1498C14B0/B8
//! with flag +0x25E clear and state +0x258 other than 3), else the scale:
//! a `lodScale` below 1.0 only counts outside those modes. The two objects
//! are camera directors of class `camAnimSceneDirector` (ENG7: made by name
//! and class hash in `title+0x5715680`; the first is `ANIM_SCENE_DIRECTOR`),
//! so the mode is "an anim-scene camera renders": in-engine cinematics.
//! The LOD block builder also divides its final scale by the stored value
//! (`title+0x3e579ed`), which is why the hook stores what it returns. The three call
//! sites are pointed at a hook that stores the value where the title keeps
//! it and then asks the title's getter, so the title's own raise to 1.0
//! stays. `,force` returns the value itself. `read` stores nothing and
//! logs what the getter returns (is the file's `lodScale` honoured?). One
//! `lod scale:` line at the first call and at each change of the result
//! (16 at most).

use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering};

/// `title+rva` of the clock tick and its first whole instructions (push rsi;
/// push rdi; push rbp; push rbx; sub rsp, 58h; vmovaps [rsp+40h], xmm7).
pub(crate) const CLOCK_TICK_RVA: usize = 0x15120b0;
pub(crate) const CLOCK_TICK_BYTES: [u8; 14] = [0x56, 0x57, 0x55, 0x53, 0x48, 0x83, 0xEC, 0x58, 0xC5, 0xF8, 0x29, 0x7C, 0x24, 0x40];
/// The raw time setter (h, m, s) and its first bytes (push rsi; push rdi;
/// push rbx; sub rsp, 50h).
const CLOCK_SET_RVA: usize = 0x1ab9b20;
const CLOCK_SET_BYTES: [u8; 7] = [0x56, 0x57, 0x53, 0x48, 0x83, 0xEC, 0x50];
/// The clock's hour and minute.
const CLOCK_HOUR_RVA: usize = 0xab5cdd0;
const CLOCK_MINUTE_RVA: usize = 0xab5cdd4;

/// The trampoline of the detoured clock tick (set by `detours.rs`).
pub(crate) static CLOCK_ORIGINAL: AtomicUsize = AtomicUsize::new(0);
static BASE: AtomicUsize = AtomicUsize::new(0);
/// hour << 8 | minute, or `NO_TIME`.
static CLOCK_TARGET: AtomicU32 = AtomicU32::new(NO_TIME);
static CLOCK_SETS: AtomicU64 = AtomicU64::new(0);
const NO_TIME: u32 = u32::MAX;

/// `hh:mm` as (hour, minute).
fn parse_clock(v: &str) -> Option<(u32, u32)> {
    let (h, m) = v.trim().split_once(':')?;
    let ok = |s: &str| !s.is_empty() && s.len() <= 2 && s.bytes().all(|b| b.is_ascii_digit());
    if !ok(h) || !ok(m) {
        return None;
    }
    let (h, m) = (h.parse::<u32>().ok()?, m.parse::<u32>().ok()?);
    (h < 24 && m < 60).then_some((h, m))
}

/// What to set after a tick that left the clock at `now`: the target when
/// the hour or minute is another.
fn clock_step(now: (u32, u32), target: (u32, u32)) -> Option<(u32, u32)> {
    (now != target).then_some(target)
}

/// `mov r64, gs:[58h]` (`65 REX 8B ModRM 25 58 00 00 00`, the title's TLS
/// block load) and its length.
const TLS_LOAD_LEN: usize = 9;
fn is_tls_load(b: &[u8]) -> bool {
    b.len() >= TLS_LOAD_LEN && b[0] == 0x65 && (b[1] == 0x48 || b[1] == 0x4C) && b[2] == 0x8B && b[3] & 0xC7 == 0x04 && b[4..TLS_LOAD_LEN] == [0x25, 0x58, 0, 0, 0]
}

/// Whether mapped code `now` is the exe's `expect`, where the loader may
/// have turned each TLS load of `expect` into `call thunk; nop4`
/// (` rel32 0F 1F 40 00`, `xnw-loader` tls.rs `redirect`, done before
/// the levers are configured whenever Windows gave the title a TLS index
/// other than 0: index 10).
fn code_is(now: &[u8], expect: &[u8]) -> bool {
    if now.len() != expect.len() {
        return false;
    }
    let mut i = 0;
    while i < expect.len() {
        if is_tls_load(&expect[i..]) && now[i] == 0xE8 && now[i + 5..i + TLS_LOAD_LEN] == [0x0F, 0x1F, 0x40, 0x00] {
            i += TLS_LOAD_LEN;
        } else if now[i] == expect[i] {
            i += 1;
        } else {
            return false;
        }
    }
    true
}

/// Whether the code at `base + rva` starts with `expect` (TLS loads as the
/// loader left them).
///
/// # Safety
/// `base` is the mapped title; `rva` is inside its .text.
unsafe fn starts_with(base: usize, rva: usize, expect: &[u8]) -> bool {
    // SAFETY: the caller's mapped code.
    code_is(unsafe { std::slice::from_raw_parts((base + rva) as *const u8, expect.len()) }, expect)
}

/// Reads `XNW_CLOCK`; whether the clock detour is wanted.
///
/// # Safety
/// `base` is the mapped title of build 1.0.2.1.
pub(crate) unsafe fn clock_configure(base: usize) -> bool {
    let v = std::env::var("XNW_CLOCK").unwrap_or_default();
    if v.trim().is_empty() {
        return false;
    }
    let Some((h, m)) = parse_clock(&v) else {
        xnw_core::warn!("XNW_CLOCK={}: not hh:mm (00:00..23:59); the clock is left alone", v.trim());
        return false;
    };
    // SAFETY: forwarded.
    if !unsafe { starts_with(base, CLOCK_SET_RVA, &CLOCK_SET_BYTES) } {
        xnw_core::warn!("clock: the setter at title+{CLOCK_SET_RVA:#x} is not the expected code; XNW_CLOCK ignored");
        return false;
    }
    BASE.store(base, Ordering::Relaxed);
    CLOCK_TARGET.store(h << 8 | m, Ordering::Relaxed);
    xnw_core::info!("clock: held at {h:02}:{m:02} after each tick of the title's clock (title+{CLOCK_TICK_RVA:#x}; setter title+{CLOCK_SET_RVA:#x})");
    true
}

/// The detour of the clock tick `title+0x15120b0` (no arguments).
pub(crate) extern "system" fn clock_hook() -> usize {
    // SAFETY: the trampoline made at install: the tick's own entry.
    let original: extern "system" fn() -> usize = unsafe { std::mem::transmute(CLOCK_ORIGINAL.load(Ordering::Relaxed)) };
    let result = original();
    if SAVE_AT.load(Ordering::Relaxed) != 0 {
        save_tick();
    }
    let (base, t) = (BASE.load(Ordering::Relaxed), CLOCK_TARGET.load(Ordering::Relaxed));
    if base == 0 || t == NO_TIME {
        return result;
    }
    // SAFETY: the title's clock, two dwords of its .data, on the thread that ticks it.
    let now = unsafe { (((base + CLOCK_HOUR_RVA) as *const u32).read(), ((base + CLOCK_MINUTE_RVA) as *const u32).read()) };
    if let Some((h, m)) = clock_step(now, (t >> 8, t & 0xFF)) {
        // SAFETY: the title's setter (hour, minute, second), its entry checked at configure.
        let set: extern "system" fn(u32, u32, u32) -> usize = unsafe { std::mem::transmute(base + CLOCK_SET_RVA) };
        set(h, m, 0);
        let n = CLOCK_SETS.fetch_add(1, Ordering::Relaxed) + 1;
        // The first, then powers of two: a held clock is set once per game minute.
        if n.is_power_of_two() {
            xnw_core::info!("clock: set {n} (was {:02}:{:02}, now {h:02}:{m:02})", now.0, now.1);
        }
    }
    result
}

// ---- XNW_CINE ---------------------------------------------------------------

/// `title+rva` of the cinematic instance tick and its first whole
/// instructions (push r15; push r14; push rsi; push rdi; push rbx; sub rsp,
/// 20h; mov rsi, rcx).
pub(crate) const CINE_TICK_RVA: usize = 0x50f89c0;
pub(crate) const CINE_TICK_BYTES: [u8; 14] = [0x41, 0x57, 0x41, 0x56, 0x56, 0x57, 0x53, 0x48, 0x83, 0xEC, 0x20, 0x48, 0x89, 0xCE];
const CINE_SKIP_RVA: usize = 0x3146250;
const CINE_CAN_SKIP_RVA: usize = 0x31462c0;
const CINE_IN_CONTEXT_RVA: usize = 0x524c060;
const CINE_CONTEXT_TEST_RVA: usize = 0x524c030;
const CINE_LOCK_RVA: usize = 0x524c040;
const CINE_UNLOCK_RVA: usize = 0x52ec1a0;
const CINE_STAMP_RVA: usize = 0x5228e90;
/// The group locks (40 bytes each, index i8 at instance +0xE24).
const CINE_LOCKS_RVA: usize = 0xab73240;
/// The functions called and the code each starts with (read from the exe).
const CINE_CALLS: [(usize, &[u8]); 7] = [
    (CINE_SKIP_RVA, &[0x56, 0x57, 0x53, 0x48, 0x83, 0xEC, 0x20, 0x48, 0x89, 0xCE]),
    (CINE_CAN_SKIP_RVA, &[0x56, 0x48, 0x83, 0xEC, 0x40, 0xF6, 0x81, 0x96, 0x04, 0x00, 0x00, 0x20]),
    (CINE_IN_CONTEXT_RVA, &[0x48, 0x83, 0xEC, 0x28, 0x8B, 0x05]),
    (CINE_CONTEXT_TEST_RVA, &[0xC1, 0xE9, 0x0E, 0x83, 0x3D]),
    (CINE_LOCK_RVA, &[0x65, 0x48, 0x8B, 0x04, 0x25, 0x58, 0x00, 0x00, 0x00, 0x48, 0x8B, 0x00, 0x80, 0xB8, 0x30, 0x14, 0x00, 0x00, 0x00]),
    (CINE_UNLOCK_RVA, &[0x65, 0x48, 0x8B, 0x04, 0x25, 0x58, 0x00, 0x00, 0x00, 0x48, 0x8B, 0x00, 0x80, 0xB8, 0x30, 0x14, 0x00, 0x00, 0x00]),
    (CINE_STAMP_RVA, &[0xF6, 0xC1, 0x40, 0x48, 0x8D, 0x05]),
];

/// The trampoline of the detoured instance tick (set by `detours.rs`).
pub(crate) static CINE_ORIGINAL: AtomicUsize = AtomicUsize::new(0);
/// 0 off, 1 skip, 2 force, 3 all.
static CINE_MODE: AtomicU8 = AtomicU8::new(0);
static CINE_SKIPS: AtomicU64 = AtomicU64::new(0);
/// The instances whose closed gate was logged last (a ring).
static CINE_CLOSED: [AtomicUsize; 8] = [const { AtomicUsize::new(0) }; 8];
static CINE_CLOSED_NEXT: AtomicUsize = AtomicUsize::new(0);

/// Whether `inst` is new to the ring (it is in it afterwards).
fn first_closed(inst: usize) -> bool {
    if CINE_CLOSED.iter().any(|c| c.load(Ordering::Relaxed) == inst) {
        return false;
    }
    CINE_CLOSED[CINE_CLOSED_NEXT.fetch_add(1, Ordering::Relaxed) % CINE_CLOSED.len()].store(inst, Ordering::Relaxed);
    true
}

fn parse_cine(v: &str) -> Option<u8> {
    match v.trim() {
        "skip" => Some(1),
        "force" => Some(2),
        "all" => Some(3),
        _ => None,
    }
}

/// Whether an instance in `state` with flag bytes +0x496 and +0x497 is one
/// to try the skip on: playing, and no skip under way or done.
fn cine_wants_skip(state: u32, f496: u8, f497: u8) -> bool {
    (state == 6 || state == 8) && f497 & 0x30 == 0 && f496 & 0x40 == 0
}

/// Reads `XNW_CINE`; whether the cinematic detour is wanted.
///
/// # Safety
/// `base` is the mapped title of build 1.0.2.1.
pub(crate) unsafe fn cine_configure(base: usize) -> bool {
    let v = std::env::var("XNW_CINE").unwrap_or_default();
    if v.trim().is_empty() {
        return false;
    }
    let Some(mode) = parse_cine(&v) else {
        xnw_core::warn!("XNW_CINE={}: not skip, force or all; cinematics are left alone", v.trim());
        return false;
    };
    for (rva, bytes) in CINE_CALLS {
        // SAFETY: forwarded.
        if !unsafe { starts_with(base, rva, bytes) } {
            xnw_core::warn!("cine: the code at title+{rva:#x} is not the expected one; XNW_CINE ignored");
            return false;
        }
    }
    if mode == 3 {
        xnw_core::warn!("XNW_CINE=all: forcing non-skippable scenes ended box run fast1d at the title's int3 title+0x22106e0; for study only");
    }
    BASE.store(base, Ordering::Relaxed);
    CINE_MODE.store(mode, Ordering::Relaxed);
    xnw_core::info!("cine: {} through the title's own skip (title+{CINE_SKIP_RVA:#x}) after each instance tick (title+{CINE_TICK_RVA:#x})", if mode == 3 { "every playing cinematic skipped, its skip window and its definition's skip flag forced" } else if mode == 2 { "every playing cinematic skipped, its skip window forced open" } else { "every playing cinematic skipped once its gate allows" });
    true
}

/// The detour of the instance tick `title+0x50f89c0(instance, flag)`.
pub(crate) extern "system" fn cine_hook(inst: usize, flag: usize) -> usize {
    // SAFETY: the trampoline made at install: the tick's own entry.
    let original: extern "system" fn(usize, usize) -> usize = unsafe { std::mem::transmute(CINE_ORIGINAL.load(Ordering::Relaxed)) };
    let result = original(inst, flag);
    let (base, mode) = (BASE.load(Ordering::Relaxed), CINE_MODE.load(Ordering::Relaxed));
    if base == 0 || inst == 0 {
        return result;
    }
    if INTRO_COUNT.load(Ordering::Relaxed) != 0 {
        // SAFETY: the instance the title just ticked, on its thread; `base` is the title configured.
        unsafe { intro_step(base, inst) };
    }
    if mode == 0 {
        return result;
    }
    // SAFETY: the instance the title just ticked (0xEB0 bytes), on its thread.
    let (state, f496, f497) = unsafe { (((inst + 0x43C) as *const u32).read(), ((inst + 0x496) as *const u8).read(), ((inst + 0x497) as *const u8).read()) };
    if !cine_wants_skip(state, f496, f497) {
        return result;
    }
    // SAFETY: the title's functions, their entries checked at configure, with the
    // arguments its own by-handle skip (title+0x31602a0) gives them.
    unsafe {
        if mode >= 2 {
            ((inst + 0x496) as *mut u8).write(f496 | 0x20);
        }
        if mode == 3 {
            let def = ((inst + 8) as *const usize).read();
            if def != 0 {
                let d = (def + 0x158) as *mut u8;
                d.write(d.read() | 4);
            }
        }
        let can: extern "system" fn(usize) -> u8 = std::mem::transmute(base + CINE_CAN_SKIP_RVA);
        let time = ((inst + 0x460) as *const f32).read();
        if can(inst) == 0 {
            if first_closed(inst) {
                let def = ((inst + 8) as *const usize).read();
                let skippable = def != 0 && ((def + 0x158) as *const u8).read() & 4 != 0;
                xnw_core::info!("cine: instance {inst:#x} state {state} at {time:.2} s: skip gate closed (window flag {}, definition skippable {skippable})", ((inst + 0x496) as *const u8).read() & 0x20 != 0);
            }
            return result;
        }
        let in_context: extern "system" fn() -> u8 = std::mem::transmute(base + CINE_IN_CONTEXT_RVA);
        let context_test: extern "system" fn(u32) -> u8 = std::mem::transmute(base + CINE_CONTEXT_TEST_RVA);
        if in_context() != 0 && context_test(((inst + 0x48) as *const u32).read()) != 0 {
            return result;
        }
        let lock_of = base + CINE_LOCKS_RVA + 40 * ((inst + 0xE24) as *const i8).read() as isize as usize;
        let (lock, unlock): (extern "system" fn(usize) -> usize, extern "system" fn(usize) -> usize) = (std::mem::transmute(base + CINE_LOCK_RVA), std::mem::transmute(base + CINE_UNLOCK_RVA));
        lock(lock_of);
        let mut asked = false;
        if ((inst + 0x49) as *const u8).read() & 0x18 == 0 {
            let stamp: extern "system" fn(u8) -> u32 = std::mem::transmute(base + CINE_STAMP_RVA);
            ((inst + 0xE48) as *mut u32).write(stamp(f497));
            let skip: extern "system" fn(usize) -> usize = std::mem::transmute(base + CINE_SKIP_RVA);
            skip(inst);
            asked = true;
        }
        unlock(lock_of);
        if asked {
            let n = CINE_SKIPS.fetch_add(1, Ordering::Relaxed) + 1;
            xnw_core::info!("cine: skip {n} asked on instance {inst:#x} (state {state}, scene time {time:.2} s)");
        }
    }
    result
}

// ---- XNW_INTRO --------------------------------------------------------------

/// `cine_seek(instance, t, a3, force)`: the thunk every caller uses (`jmp
/// title+0x5188b0`) and the body.
const CINE_SEEK_THUNK_RVA: usize = 0x518880;
const CINE_SEEK_RVA: usize = 0x5188b0;
/// The seek's entry (push rsi; push rdi; push rbx; sub rsp, 30h; vmovss
/// [rsp+2Ch], xmm1; mov edi, [rcx+43Ch]).
const CINE_SEEK_BYTES: [u8; 19] = [0x56, 0x57, 0x53, 0x48, 0x83, 0xEC, 0x30, 0xC5, 0xFA, 0x11, 0x4C, 0x24, 0x2C, 0x8B, 0xB9, 0x3C, 0x04, 0x00, 0x00];
/// What the seek does to a playing instance (`title+0x51890e`): `or byte
/// [rsi+495h], 4; vmovss xmm0, [rsp+2Ch]; vmovss [rsi+484h], xmm0; test bl,
/// bl; jnz out; vmovss xmm1, [rsi+480h]; vaddss xmm1, xmm1, [-1e-6]; vucomiss
/// xmm0, xmm1; jb out; or byte [rsi+496h], 40h`.
const CINE_SEEK_SET_RVA: usize = 0x51890e;
const CINE_SEEK_SET_BYTES: [u8; 54] = [
    0x80, 0x8E, 0x95, 0x04, 0x00, 0x00, 0x04, 0xC5, 0xFA, 0x10, 0x44, 0x24, 0x2C, 0xC5, 0xFA, 0x11, 0x86, 0x84, 0x04, 0x00, 0x00, 0x84, 0xDB, 0x75, 0x40, 0xC5, 0xFA, 0x10, 0x8E, 0x80, 0x04, 0x00, 0x00, 0xC5, 0xF2, 0x58, 0x0D,
    0x65, 0x16, 0xC4, 0x05, 0xC5, 0xF8, 0x2E, 0xC1, 0x72, 0x2A, 0x80, 0x8E, 0x96, 0x04, 0x00, 0x00, 0x40,
];
/// The title's seek method of a scene handle (vtable 0x1463CDD30 slot 1),
/// whose steps the lever takes; read by the unit test only.
#[cfg(test)]
const CINE_SEEK_BY_HANDLE_RVA: usize = 0x3162b60;
/// The two anim-scene camera directors (qword_1498C14B0, ..B8): state u32
/// at +0x258, flag u8 at +0x25E (`title+0x45f4a0`: one renders when its
/// flag is clear and its state is not 3).
const DIRECTORS_RVA: usize = 0x98c14b0;
/// The functions called and the code each starts with.
const INTRO_CALLS: [(usize, &[u8]); 3] = [(CINE_SEEK_THUNK_RVA, &[0xE9, 0x2B, 0x00, 0x00, 0x00]), (CINE_SEEK_RVA, &CINE_SEEK_BYTES), (CINE_SEEK_SET_RVA, &CINE_SEEK_SET_BYTES)];
/// Seconds short of the section's end the seek lands: past `end - 1e-6`
/// the title marks the scene skipped.
const INTRO_MARGIN: f32 = 0.5;

/// Seeks left (0 = off).
static INTRO_COUNT: AtomicU32 = AtomicU32::new(0);
static INTRO_OF: AtomicU32 = AtomicU32::new(0);
/// The shortest section end taken, as f32 bits.
static INTRO_MIN: AtomicU32 = AtomicU32::new(0);
/// The instance sought last, and the state and section (a name hash at
/// +0x9A8) logged for it.
static INTRO_INST: AtomicUsize = AtomicUsize::new(0);
static INTRO_STATE: AtomicU32 = AtomicU32::new(0);
static INTRO_SECTION: AtomicU32 = AtomicU32::new(0);
static INTRO_LINES: AtomicU32 = AtomicU32::new(0);

/// `seekend[:<n>[:<seconds>]]` as (seeks, shortest section end).
fn parse_intro(v: &str) -> Option<(u32, f32)> {
    let mut parts = v.trim().split(':');
    if parts.next()? != "seekend" {
        return None;
    }
    let n = match parts.next() {
        Some(n) => n.parse::<u32>().ok().filter(|&n| (1..=64).contains(&n))?,
        None => 5,
    };
    let min = match parts.next() {
        Some(s) => s.parse::<f32>().ok().filter(|s| (2.0..=3600.0).contains(s))?,
        None => 30.0,
    };
    parts.next().is_none().then_some((n, min))
}

/// Where to seek an instance to, if it is one to take: it plays (state 6),
/// its definition is not skippable (`def158` bit 2), the handle methods do
/// not refuse it (`f49` & 0x18), no seek (`f495` bit 2) or skip (`f497` bits
/// 4-5, `f496` bit 6) is under way, its section ends at `min` or later, and
/// it is more than a second before the landing point.
#[allow(clippy::too_many_arguments)]
fn intro_target(state: u32, def158: u8, f49: u8, f495: u8, f496: u8, f497: u8, time: f32, end: f32, min: f32) -> Option<f32> {
    let to = end - INTRO_MARGIN;
    (state == 6 && def158 & 4 == 0 && f49 & 0x18 == 0 && f495 & 4 == 0 && f496 & 0x40 == 0 && f497 & 0x30 == 0 && end.is_finite() && time.is_finite() && end >= min && time < to - 1.0).then_some(to)
}

/// Reads `XNW_INTRO`; whether the cinematic detour is wanted for it.
///
/// # Safety
/// `base` is the mapped title of build 1.0.2.1.
pub(crate) unsafe fn intro_configure(base: usize) -> bool {
    let v = std::env::var("XNW_INTRO").unwrap_or_default();
    if v.trim().is_empty() {
        return false;
    }
    let Some((n, min)) = parse_intro(&v) else {
        xnw_core::warn!("XNW_INTRO={}: not seekend[:<seeks 1..64>[:<seconds 2..3600>]]; the opening is left alone", v.trim());
        return false;
    };
    let lock_calls = CINE_CALLS.iter().filter(|c| [CINE_LOCK_RVA, CINE_UNLOCK_RVA, CINE_STAMP_RVA].contains(&c.0));
    for &(rva, bytes) in INTRO_CALLS.iter().chain(lock_calls) {
        // SAFETY: forwarded.
        if !unsafe { starts_with(base, rva, bytes) } {
            xnw_core::warn!("intro: the code at title+{rva:#x} is not the expected one; XNW_INTRO ignored");
            return false;
        }
    }
    BASE.store(base, Ordering::Relaxed);
    INTRO_MIN.store(min.to_bits(), Ordering::Relaxed);
    INTRO_OF.store(n, Ordering::Relaxed);
    INTRO_COUNT.store(n, Ordering::Relaxed);
    xnw_core::info!("intro: the first {n} playing scene sections that are not skippable and end {min} s in or later are sought to {INTRO_MARGIN} s before their end through the title's own seek (title+{CINE_SEEK_THUNK_RVA:#x}) after the instance tick (title+{CINE_TICK_RVA:#x})");
    true
}

/// The anim-scene camera directors as text: `state/flag` each, `-` for none.
///
/// # Safety
/// `base` is the mapped title, on the game thread.
unsafe fn directors(base: usize) -> String {
    let one = |i: usize| {
        // SAFETY: a pointer of the title's .data; the director object is at least 0x260 bytes (title+0x45f4a0 reads both fields).
        unsafe {
            let d = ((base + DIRECTORS_RVA + 8 * i) as *const usize).read();
            if d == 0 { "-".to_owned() } else { format!("{}/{}", ((d + 0x258) as *const u32).read(), ((d + 0x25E) as *const u8).read()) }
        }
    };
    format!("{} {}", one(0), one(1))
}

/// One instance after its tick: the seek when it is one to take, and a line
/// per state or section change of the instance sought last.
///
/// # Safety
/// `inst` is the instance the title just ticked (0xEB0 bytes), on its
/// thread; `base` is the title `intro_configure` checked.
unsafe fn intro_step(base: usize, inst: usize) {
    // SAFETY: the caller's instance and its definition (read only up to +0x15C).
    unsafe {
        let u8_at = |o: usize| ((inst + o) as *const u8).read();
        let f32_at = |o: usize| ((inst + o) as *const f32).read();
        let (state, section) = (((inst + 0x43C) as *const u32).read(), ((inst + 0x9A8) as *const u32).read());
        if inst == INTRO_INST.load(Ordering::Relaxed) {
            let (was, was_section) = (INTRO_STATE.swap(state, Ordering::Relaxed), INTRO_SECTION.swap(section, Ordering::Relaxed));
            if (was != state || was_section != section) && INTRO_LINES.fetch_add(1, Ordering::Relaxed) < 64 {
                xnw_core::info!("intro: instance {inst:#x} state {was} -> {state}, section {was_section:#x} -> {section:#x}, scene time {:.2} s of {:.2} s, flags +0x495 {:#04x} +0x496 {:#04x}, anim-scene cameras (state/flag) {}", f32_at(0x460), f32_at(0x480), u8_at(0x495), u8_at(0x496), directors(base));
            }
        }
        let left = INTRO_COUNT.load(Ordering::Relaxed);
        if left == 0 {
            return;
        }
        let def = ((inst + 8) as *const usize).read();
        if def == 0 {
            return;
        }
        let def158 = ((def + 0x158) as *const u32).read();
        let (time, end) = (f32_at(0x460), f32_at(0x480));
        let f497 = u8_at(0x497);
        let Some(to) = intro_target(state, def158 as u8, u8_at(0x49), u8_at(0x495), u8_at(0x496), f497, time, end, f32::from_bits(INTRO_MIN.load(Ordering::Relaxed))) else {
            return;
        };
        // The steps of the title's seek method of a scene handle (title+0x3162b60).
        let lock_of = base + CINE_LOCKS_RVA + 40 * ((inst + 0xE24) as *const i8).read() as isize as usize;
        let (lock, unlock): (extern "system" fn(usize) -> usize, extern "system" fn(usize) -> usize) = (std::mem::transmute(base + CINE_LOCK_RVA), std::mem::transmute(base + CINE_UNLOCK_RVA));
        lock(lock_of);
        let stamp: extern "system" fn(u8) -> u32 = std::mem::transmute(base + CINE_STAMP_RVA);
        ((inst + 0xE48) as *mut u32).write(stamp(f497));
        let seek: extern "system" fn(usize, f32, u8, u8) -> usize = std::mem::transmute(base + CINE_SEEK_THUNK_RVA);
        seek(inst, to, 0, 0);
        unlock(lock_of);
        INTRO_COUNT.store(left - 1, Ordering::Relaxed);
        INTRO_INST.store(inst, Ordering::Relaxed);
        INTRO_STATE.store(state, Ordering::Relaxed);
        INTRO_SECTION.store(section, Ordering::Relaxed);
        let (of, words) = (INTRO_OF.load(Ordering::Relaxed), [0x8usize, 0xC, 0x10, 0x14, 0x18].map(|o| ((def + o) as *const u32).read()));
        xnw_core::info!(
            "intro: seek {} of {of} asked on instance {inst:#x}: section {section:#x}, scene time {time:.2} s -> {:.2} s (end {end:.2} s; pending +0x495 {:#04x}, target +0x484 {:.2}); definition {def:#x} flags +0x158 {def158:#x} words +8.. {words:x?}; play flags +0x48 {:#x} +0x80 {:#x}; anim-scene cameras (state/flag) {}",
            of - left + 1,
            to,
            u8_at(0x495),
            f32_at(0x484),
            ((inst + 0x48) as *const u32).read(),
            ((inst + 0x80) as *const u32).read(),
            directors(base)
        );
        if left == 1 {
            xnw_core::info!("intro: {of} seeks asked; the lever is off from here");
        }
    }
}

// ---- XNW_SAVE_AT ------------------------------------------------------------

/// `save_request_autosave` 0x1444817F0() (no arguments, al = queued): the
/// body of the script native 0x144461180. Its whole code: the blocked test
/// (call 0x144481830 -> jmp 0x1428BBB80 `mov al, [byte_1478B27F1]`), the
/// profile's autosave flag (`[[qword_14AAD3CE0+8]]` vslot 0x48 -> +0x410 == 1),
/// `byte_149DF77C0 == 0`, then `jmp save_queue_autosave_op` 0x1428BB500.
const SAVE_REQUEST_RVA: usize = 0x44817f0;
const SAVE_REQUEST_BYTES: [u8; 0x40] = [
    0x48, 0x83, 0xEC, 0x28, 0xE8, 0x37, 0x00, 0x00, 0x00, 0x84, 0xC0, 0x75, 0x23, 0x48, 0x8B, 0x05, 0xDC, 0x24, 0x65, 0x06, 0x48, 0x8B, 0x48, 0x08, 0x48, 0x8B, 0x01, 0xFF, 0x50, 0x48, 0x80, 0xB8,
    0x10, 0x04, 0x00, 0x00, 0x01, 0x75, 0x09, 0x80, 0x3D, 0xA2, 0x5F, 0x97, 0x05, 0x00, 0x74, 0x07, 0x31, 0xC0, 0x48, 0x83, 0xC4, 0x28, 0xC3, 0x48, 0x83, 0xC4, 0x28, 0xE9, 0xD0, 0x9C, 0x43, 0xFE,
];
/// The blocked test's thunk and body.
#[cfg(test)]
const SAVE_BLOCKED_THUNK_RVA: usize = 0x4481830;
const SAVE_BLOCKED_RVA: usize = 0x28bbb80;
const SAVE_BLOCKED_BYTES: [u8; 7] = [0x8A, 0x05, 0x6B, 0x6C, 0xFF, 0x04, 0xC3];
/// `save_autosave_op_idle` 0x1428BB4E0, the queue step's gate: `byte_1478B2778 == 0` and the op's +8 != 1.
const SAVE_OP_IDLE_RVA: usize = 0x28bb4e0;
const SAVE_OP_IDLE_BYTES: [u8; 28] = [0x80, 0x3D, 0x91, 0x72, 0xFF, 0x04, 0x00, 0x74, 0x03, 0x31, 0xC0, 0xC3, 0x48, 0x8B, 0x0D, 0x8D, 0x71, 0xFF, 0x04, 0xB0, 0x01, 0x83, 0x79, 0x08, 0x01, 0x74, 0xEE, 0xC3];
/// `save_queue_autosave_op` 0x1428BB500 (the request's tail jump).
#[cfg(test)]
const SAVE_QUEUE_RVA: usize = 0x28bb500;
/// `save_arm_write_slot` 0x1428B82D0, reached from the op's step 9: the same
/// profile flag (slot < 0), then `dword_1478CCD78 | dword_149DF77C4 == 0`.
#[cfg(test)]
const SAVE_ARM_RVA: usize = 0x28b82d0;
#[cfg(test)]
const SAVE_ARM_BYTES: [u8; 62] = [
    0x56, 0x57, 0x48, 0x83, 0xEC, 0x28, 0x85, 0xC9, 0x79, 0x26, 0x48, 0x8B, 0x05, 0xFF, 0xB9, 0x21, 0x08, 0x89, 0xCE, 0x89, 0xD7, 0x48, 0x8B, 0x40, 0x08, 0x4C, 0x8B, 0x00, 0x48, 0x89, 0xC1, 0x41,
    0xFF, 0x50, 0x48, 0x80, 0xB8, 0x10, 0x04, 0x00, 0x00, 0x01, 0x89, 0xF1, 0x89, 0xFA, 0x75, 0x19, 0x8B, 0x05, 0x72, 0x4A, 0x01, 0x05, 0x0B, 0x05, 0xB8, 0xF4, 0x53, 0x07, 0x75, 0x0B,
];
/// The gates' and the watch's data (`title+rva`).
const SAVE_BLOCKED_FLAG_RVA: usize = 0x78b27f1;
const SAVE_PROFILE_RVA: usize = 0xaad3ce0;
const SAVE_PROFILE_FLAG: usize = 0x410;
const SAVE_REFUSE_RVA: usize = 0x9df77c0;
const SAVE_OP_BUSY_RVA: usize = 0x78b2778;
/// `g_saveop_autosave`: the op (+8 queue state: 1 queued, 0 done, 2 failed;
/// +0x10 step 0..12; +0x18 result 1 ok, 2 failed).
const SAVE_OP_RVA: usize = 0x78b2680;
const SAVE_OPCODE_RVA: usize = 0x9df77c4;
const SAVE_FSM_RVA: usize = 0x78ccd78;
const SAVE_MENU_RVA: usize = 0x78cb7f4;
/// The write machine's state that calls the WRITE step 0x14289B800.
const SAVE_FSM_WRITE: u32 = 14;

/// The world frame to ask at (0: off), `,force`, the phase (0 waiting, 1
/// asked, 2 done), clock ticks since the first try, refusals, watch lines.
static SAVE_AT: AtomicU64 = AtomicU64::new(0);
static SAVE_FORCE: AtomicU8 = AtomicU8::new(0);
static SAVE_PHASE: AtomicU8 = AtomicU8::new(0);
static SAVE_TICKS: AtomicU64 = AtomicU64::new(0);
static SAVE_TRIES: AtomicU32 = AtomicU32::new(0);
static SAVE_LAST: AtomicU64 = AtomicU64::new(u64::MAX);
static SAVE_LINES: AtomicU32 = AtomicU32::new(0);
static SAVE_WROTE: AtomicU8 = AtomicU8::new(0);
/// The world frame of the request.
static SAVE_ASKED_AT: AtomicU64 = AtomicU64::new(0);
/// Tries before giving up (one each `SAVE_RETRY` ticks) and ticks to wait for the write.
const SAVE_RETRY: u64 = 60;
const SAVE_MAX_TRIES: u32 = 30;
const SAVE_WAIT_TICKS: u64 = 30_000;
/// The containers' versions before the request (name, version).
static SAVE_BEFORE: std::sync::Mutex<Vec<(String, u64)>> = std::sync::Mutex::new(Vec::new());

/// `<n>` or `<n>,force` as (frame, force).
fn parse_save_at(v: &str) -> Option<(u64, bool)> {
    let (n, force) = match v.trim().split_once(',') {
        Some((n, f)) if f.trim().eq_ignore_ascii_case("force") => (n, true),
        Some(_) => return None,
        None => (v, false),
    };
    n.trim().parse::<u64>().ok().filter(|&n| n > 0).map(|n| (n, force))
}

/// The gates of the request and the write, as the title reads them.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SaveGates {
    blocked: u8,
    /// None: no profile object yet.
    flag: Option<u8>,
    refuse: u8,
    busy: u8,
    /// The op's +8, +0x10, +0x18 (None: no op yet).
    op: Option<(u32, u32, u32)>,
    opcode: u32,
    fsm: u32,
    menu: u32,
}

impl SaveGates {
    /// Whether `save_request_autosave` would queue the op.
    fn open(&self) -> bool {
        self.blocked == 0 && self.flag == Some(1) && self.refuse == 0 && self.busy == 0 && self.op.is_some_and(|o| o.0 != 1)
    }
    fn line(&self) -> String {
        let flag = self.flag.map_or("no profile object".to_owned(), |f| f.to_string());
        let op = self.op.map_or("no op".to_owned(), |o| format!("queue {} step {} result {}", o.0, o.1, o.2));
        format!(
            "blocked byte_1478B27F1={}, profile autosave flag +0x410={flag}, refuse byte_149DF77C0={}, op busy byte_1478B2778={}, op {op}, opcode dword_149DF77C4={}, write fsm dword_1478CCD78={}, menu dword_1478CB7F4={}",
            self.blocked, self.refuse, self.busy, self.opcode, self.fsm, self.menu
        )
    }
}

/// The profile object's autosave flag byte (`[[qword_14AAD3CE0+8]]` vslot 0x48, +0x410).
///
/// # Safety
/// `base` is the mapped title; on its game thread.
unsafe fn save_flag_at(base: usize) -> Option<*mut u8> {
    // SAFETY: the title's pointer and the object's vtable getter, as 0x1444817F0 calls it.
    unsafe {
        let holder = ((base + SAVE_PROFILE_RVA) as *const usize).read();
        if holder == 0 {
            return None;
        }
        let obj = ((holder + 8) as *const usize).read();
        if obj == 0 {
            return None;
        }
        let slot = ((obj as *const usize).read() + 0x48) as *const usize;
        let get: extern "system" fn(usize) -> usize = std::mem::transmute(slot.read());
        let p = get(obj);
        (p != 0).then_some((p + SAVE_PROFILE_FLAG) as *mut u8)
    }
}

/// # Safety
/// As `save_flag_at`.
unsafe fn save_gates(base: usize) -> SaveGates {
    // SAFETY: the title's .data globals and its op (0x30 bytes at least).
    unsafe {
        let rd8 = |rva: usize| ((base + rva) as *const u8).read();
        let rd32 = |rva: usize| ((base + rva) as *const u32).read();
        let op = ((base + SAVE_OP_RVA) as *const usize).read();
        SaveGates {
            blocked: rd8(SAVE_BLOCKED_FLAG_RVA),
            flag: save_flag_at(base).map(|p| p.read()),
            refuse: rd8(SAVE_REFUSE_RVA),
            busy: rd8(SAVE_OP_BUSY_RVA),
            op: (op != 0).then(|| (((op + 8) as *const u32).read(), ((op + 0x10) as *const u32).read(), ((op + 0x18) as *const u32).read())),
            opcode: rd32(SAVE_OPCODE_RVA),
            fsm: rd32(SAVE_FSM_RVA),
            menu: rd32(SAVE_MENU_RVA),
        }
    }
}

/// The game save containers under `root` (`<scid>\<user>\<name>~<hash>\v<N>\`)
/// with their newest committed version and its blob bytes (manifest left out).
fn save_versions(root: &std::path::Path) -> Vec<(String, u64, u64)> {
    let mut out = Vec::new();
    let dirs = |p: &std::path::Path| -> Vec<std::path::PathBuf> { std::fs::read_dir(p).map(|r| r.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect()).unwrap_or_default() };
    for scid in dirs(root) {
        for user in dirs(&scid) {
            for c in dirs(&user) {
                let newest = dirs(&c).iter().filter_map(|v| v.file_name()?.to_str()?.strip_prefix('v')?.parse::<u64>().ok().map(|n| (n, v.clone()))).max_by_key(|x| x.0);
                if let Some((n, v)) = newest {
                    let bytes = std::fs::read_dir(&v).map(|r| r.flatten().filter(|e| e.file_name() != "manifest").filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum()).unwrap_or(0);
                    out.push((c.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(), n, bytes));
                }
            }
        }
    }
    out.sort();
    out
}

/// The containers of `now` that are new or have a newer version than in `before`, SARC slots first.
fn save_written(before: &[(String, u64)], now: &[(String, u64, u64)]) -> Vec<(String, u64, u64)> {
    let mut w: Vec<(String, u64, u64)> = now.iter().filter(|c| before.iter().find(|b| b.0 == c.0).is_none_or(|b| b.1 < c.1)).cloned().collect();
    w.sort_by_key(|c| !c.0.starts_with("SARC"));
    w
}

/// Reads `XNW_SAVE_AT`; whether the clock tick detour (and the stage
/// meter's world frame count) is wanted for it.
///
/// # Safety
/// `base` is the mapped title of build 1.0.2.1.
pub(crate) unsafe fn save_configure(base: usize) -> bool {
    let v = std::env::var("XNW_SAVE_AT").unwrap_or_default();
    if v.trim().is_empty() {
        return false;
    }
    let Some((at, force)) = parse_save_at(&v) else {
        xnw_core::warn!("XNW_SAVE_AT={}: not <world frame> or <world frame>,force; no save asked", v.trim());
        return false;
    };
    let checks: [(usize, &[u8]); 3] = [(SAVE_REQUEST_RVA, &SAVE_REQUEST_BYTES), (SAVE_BLOCKED_RVA, &SAVE_BLOCKED_BYTES), (SAVE_OP_IDLE_RVA, &SAVE_OP_IDLE_BYTES)];
    for (rva, bytes) in checks {
        // SAFETY: forwarded.
        if !unsafe { starts_with(base, rva, bytes) } {
            xnw_core::warn!("save: the code at title+{rva:#x} is not the expected one; XNW_SAVE_AT ignored");
            return false;
        }
    }
    BASE.store(base, Ordering::Relaxed);
    SAVE_FORCE.store(force as u8, Ordering::Relaxed);
    SAVE_AT.store(at, Ordering::Relaxed);
    xnw_core::info!("save: the title's autosave (title+{SAVE_REQUEST_RVA:#x}) asked once at world frame {at}, on the clock tick's thread (title+{CLOCK_TICK_RVA:#x}){}; saves under {}", if force { "; a profile autosave flag of 0 is set to 1 first" } else { "" }, xnw_hle::saves_folder().display());
    true
}

/// After each clock tick: ask once the world frame is reached, then watch the op to its end.
fn save_tick() {
    let (base, at) = (BASE.load(Ordering::Relaxed), SAVE_AT.load(Ordering::Relaxed));
    let phase = SAVE_PHASE.load(Ordering::Relaxed);
    let frame = crate::stagemeter::world_frames();
    if base == 0 || phase == 2 || frame < at {
        return;
    }
    let tick = SAVE_TICKS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: the title's globals and profile object, on the thread that ticks its clock.
    let g = unsafe { save_gates(base) };
    if phase == 0 {
        if tick % SAVE_RETRY != 0 {
            return;
        }
        let tries = SAVE_TRIES.fetch_add(1, Ordering::Relaxed) + 1;
        if tries == 1 || tries == SAVE_MAX_TRIES {
            xnw_core::info!("save: gates at world frame {frame} (try {tries}): {}", g.line());
        }
        let mut g = g;
        if g.flag == Some(0) && SAVE_FORCE.load(Ordering::Relaxed) != 0 {
            // SAFETY: the flag byte just read, on the same thread.
            if let Some(p) = unsafe { save_flag_at(base) } {
                // SAFETY: the profile object's flag byte, as the title's own setter leaves it (0 or 1).
                unsafe { p.write(1) };
                g.flag = Some(1);
                xnw_core::info!("save: profile autosave flag +0x410 was 0; set to 1 (XNW_SAVE_AT=..,force)");
            }
        }
        if !g.open() {
            if tries >= SAVE_MAX_TRIES {
                SAVE_PHASE.store(2, Ordering::Relaxed);
                xnw_core::warn!("save: autosave not requested by world frame {frame}: a gate stayed closed after {tries} tries ({})", g.line());
            }
            return;
        }
        *SAVE_BEFORE.lock().unwrap_or_else(|e| e.into_inner()) = save_versions(&xnw_hle::saves_folder()).into_iter().map(|c| (c.0, c.1)).collect();
        // SAFETY: the title's request (no arguments), its code checked at configure.
        let request: extern "system" fn() -> u8 = unsafe { std::mem::transmute(base + SAVE_REQUEST_RVA) };
        let queued = request() != 0;
        // SAFETY: as above.
        let after = unsafe { save_gates(base) };
        xnw_core::info!("save: autosave requested at world frame {frame}: returned {} ({})", queued as u8, after.line());
        if queued {
            SAVE_ASKED_AT.store(frame, Ordering::Relaxed);
            SAVE_TICKS.store(1, Ordering::Relaxed);
            SAVE_PHASE.store(1, Ordering::Relaxed);
        } else if tries >= SAVE_MAX_TRIES {
            SAVE_PHASE.store(2, Ordering::Relaxed);
        }
        return;
    }
    // Asked: one line per change of queue state, step, result, write state, opcode.
    let Some(op) = g.op else { return };
    let key = u64::from(op.0 & 0xFF) | u64::from(op.1 & 0xFF) << 8 | u64::from(op.2 & 0xFF) << 16 | u64::from(g.fsm & 0xFF) << 24 | u64::from(g.opcode & 0xFF) << 32;
    if SAVE_LAST.swap(key, Ordering::Relaxed) != key && SAVE_LINES.fetch_add(1, Ordering::Relaxed) < 64 {
        xnw_core::info!("save: op queue {} step {} result {}, write fsm {}, opcode {} at world frame {frame} (tick {tick})", op.0, op.1, op.2, g.fsm, g.opcode);
    }
    if g.fsm == SAVE_FSM_WRITE {
        SAVE_WROTE.store(1, Ordering::Relaxed);
    }
    let asked = SAVE_ASKED_AT.load(Ordering::Relaxed);
    let wrote = if SAVE_WROTE.load(Ordering::Relaxed) != 0 { "write state 14 reached" } else { "write state 14 not seen" };
    if op.0 != 1 {
        SAVE_PHASE.store(2, Ordering::Relaxed);
        let before = SAVE_BEFORE.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let written = save_written(&before, &save_versions(&xnw_hle::saves_folder()));
        let slots: Vec<String> = written.iter().map(|c| format!("{} v{} ({} bytes)", c.0, c.1, c.2)).collect();
        if op.0 == 0 && !written.is_empty() {
            xnw_core::info!("save: autosave requested at frame {asked}; written slot {} at world frame {frame} ({wrote}; op result {})", slots.join(", "), op.2);
        } else {
            xnw_core::warn!("save: autosave requested at frame {asked}; not written: op queue {} result {} at world frame {frame} ({wrote}; new containers [{}])", op.0, op.2, slots.join(", "));
        }
    } else if tick >= SAVE_WAIT_TICKS {
        SAVE_PHASE.store(2, Ordering::Relaxed);
        xnw_core::warn!("save: autosave requested at frame {asked}; not written after {tick} ticks ({wrote}; {})", g.line());
    }
}

// ---- XNW_LOD_SCALE ----------------------------------------------------------

/// The global LOD scale (f32), stored by `settings_apply` from graphics +0xD4.
const LOD_SCALE_RVA: usize = 0xa15e200;
/// The title's getter of it and the three sites that call it.
const LOD_GETTER_RVA: usize = 0x45f460;
const LOD_SITES: [usize; 3] = [0x80db3b, 0x26507ed, 0x3e573e6];
/// The getter's first bytes: sub rsp, 28h; test byte [rip+..], 2.
const LOD_GETTER_BYTES: [u8; 6] = [0x48, 0x83, 0xEC, 0x28, 0xF6, 0x05];

/// 0 off, 1 read, 2 set, 3 set and force.
static LOD_MODE: AtomicU8 = AtomicU8::new(0);
/// The wanted scale's bits.
static LOD_VALUE: AtomicU32 = AtomicU32::new(0);
/// Where the scale lives and the title's getter (addresses; set at install).
static LOD_SLOT: AtomicUsize = AtomicUsize::new(0);
static LOD_GETTER: AtomicUsize = AtomicUsize::new(0);
/// The last result logged (bits; `NO_TIME` = none yet) and the lines so far.
static LOD_LAST: AtomicU32 = AtomicU32::new(NO_TIME);
static LOD_LINES: AtomicU32 = AtomicU32::new(0);

/// `read`, `<f>` or `<f>,force` as (mode, value); the value within 0.05..=4.0.
fn parse_lod(v: &str) -> Option<(u8, f32)> {
    let v = v.trim();
    if v.eq_ignore_ascii_case("read") {
        return Some((1, 0.0));
    }
    let (num, mode) = match v.split_once(',') {
        Some((n, f)) if f.trim().eq_ignore_ascii_case("force") => (n, 3),
        Some(_) => return None,
        None => (v, 2),
    };
    let f = num.trim().parse::<f32>().ok()?;
    (0.05..=4.0).contains(&f).then_some((mode, f))
}

/// The hook at the getter's call sites: the scale the title uses this frame.
pub(crate) extern "system" fn lod_hook() -> f32 {
    let (mode, want) = (LOD_MODE.load(Ordering::Relaxed), f32::from_bits(LOD_VALUE.load(Ordering::Relaxed)));
    let slot = LOD_SLOT.load(Ordering::Relaxed) as *mut f32;
    // SAFETY: the title's f32 (set at install), written as `settings_apply` writes it.
    let was = unsafe {
        let was = slot.read();
        if mode >= 2 {
            slot.write(want);
        }
        was
    };
    // SAFETY: the title's getter (no arguments, f32 in xmm0), checked at install.
    let title: extern "system" fn() -> f32 = unsafe { std::mem::transmute(LOD_GETTER.load(Ordering::Relaxed)) };
    let got = title();
    let out = if mode == 3 { want } else { got };
    if LOD_LAST.swap(out.to_bits(), Ordering::Relaxed) != out.to_bits() && LOD_LINES.fetch_add(1, Ordering::Relaxed) < 16 {
        let how = match mode {
            1 => "read only".to_owned(),
            2 => format!("{want} stored"),
            _ => format!("{want} stored and returned"),
        };
        xnw_core::info!("lod scale: the title uses {out} (its getter returns {got}; the stored value was {was}; {how})");
    }
    out
}

/// Reads `XNW_LOD_SCALE` and points the getter's call sites at the hook.
///
/// # Safety
/// `base` is the mapped title and none of its code runs yet.
pub(crate) unsafe fn lod_install(base: usize) {
    let v = std::env::var("XNW_LOD_SCALE").unwrap_or_default();
    if v.trim().is_empty() {
        return;
    }
    let Some((mode, value)) = parse_lod(&v) else {
        xnw_core::warn!("XNW_LOD_SCALE={}: not read, <f> or <f>,force (0.05..4.0); the LOD scale is left alone", v.trim());
        return;
    };
    // SAFETY: forwarded; SizeOfImage is at the optional header's +0x38.
    let (stamp, size) = unsafe {
        let nt = (base as *const u8).add(0x3c).cast::<u32>().read_unaligned() as usize;
        (crate::patches::timestamp(base), (base as *const u8).add(nt + 0x50).cast::<u32>().read_unaligned() as usize)
    };
    // SAFETY: inside the mapped .text when the build is the one named.
    if stamp != 0x6AF7_83D4 || !unsafe { starts_with(base, LOD_GETTER_RVA, &LOD_GETTER_BYTES) } {
        xnw_core::warn!("lod scale: the getter at title+{LOD_GETTER_RVA:#x} is not the expected code; XNW_LOD_SCALE ignored");
        return;
    }
    let Some(page) = crate::titlemeter::near_page(base, base + size) else {
        xnw_core::warn!("lod scale: no page within 2 GiB of the title; XNW_LOD_SCALE ignored");
        return;
    };
    LOD_SLOT.store(base + LOD_SCALE_RVA, Ordering::Relaxed);
    LOD_GETTER.store(base + LOD_GETTER_RVA, Ordering::Relaxed);
    LOD_VALUE.store(value.to_bits(), Ordering::Relaxed);
    LOD_MODE.store(mode, Ordering::Relaxed);
    let mut done = 0;
    for (i, &site) in LOD_SITES.iter().enumerate() {
        // SAFETY: a call site of this build, no code running; 16 bytes of our page each.
        match unsafe { crate::titlemeter::redirect((base + site) as *mut u8, base + LOD_GETTER_RVA, page.add(16 * i), lod_hook as usize) } {
            Ok(()) => done += 1,
            Err(why) => xnw_core::warn!("lod scale: the call at title+{site:#x} skipped: {why}"),
        }
    }
    let what = match mode {
        1 => "read only".to_owned(),
        2 => format!("{value} stored at title+{LOD_SCALE_RVA:#x} before each read"),
        _ => format!("{value} stored and returned (the title's raise to 1.0 in its modes is bypassed)"),
    };
    xnw_core::info!("lod scale: {done} of {} call sites of the getter title+{LOD_GETTER_RVA:#x} hooked; {what}", LOD_SITES.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_values_parse() {
        assert_eq!(parse_clock("12:00"), Some((12, 0)));
        assert_eq!(parse_clock(" 7:05 "), Some((7, 5)));
        assert_eq!(parse_clock("23:59"), Some((23, 59)));
        for bad in ["24:00", "12:60", "12", "12:", ":30", "noon", "1:2:3", "-1:00", "+1:00", "012:00"] {
            assert_eq!(parse_clock(bad), None, "{bad}");
        }
    }

    #[test]
    fn cine_values_parse_and_only_a_playing_unskipped_instance_is_tried() {
        assert_eq!((parse_cine("skip"), parse_cine(" force "), parse_cine("all"), parse_cine("1"), parse_cine("seek")), (Some(1), Some(2), Some(3), None, None));
        assert!(cine_wants_skip(6, 0, 0) && cine_wants_skip(8, 0x20, 0x40));
        for state in [1, 2, 3, 4, 5, 7, 9, 10] {
            assert!(!cine_wants_skip(state, 0, 0), "state {state}");
        }
        assert!(!cine_wants_skip(6, 0, 0x10), "a skip under way (+0x497 bits 4-5)");
        assert!(!cine_wants_skip(6, 0x40, 0), "skipped (+0x496 bit 6)");
        // Every call's bytes are whole and at least a few instructions long.
        assert!(CINE_CALLS.iter().all(|c| c.1.len() >= 5));
        // A closed gate is logged once per instance, two instances in turn included.
        assert_eq!([first_closed(0x10), first_closed(0x20), first_closed(0x10), first_closed(0x20)], [true, true, false, false]);
    }

    /// The cinematic lock's expected bytes match the code as the loader leaves it: the TLS load replaced by
    /// a call (any nine bytes), the rest as in the file; another tail, or another head without a TLS load, does not.
    #[test]
    fn a_function_whose_tls_load_the_loader_redirected_is_still_the_expected_one() {
        let (_, expect) = CINE_CALLS.iter().find(|c| c.0 == CINE_LOCK_RVA).unwrap();
        let mut code = expect.to_vec();
        // SAFETY: local buffers of the expected length.
        unsafe {
            assert!(starts_with(code.as_ptr() as usize, 0, expect), "the file's bytes");
            code[..9].copy_from_slice(&[0xE8, 0x11, 0x22, 0x33, 0x44, 0x0F, 0x1F, 0x40, 0x00]);
            assert!(starts_with(code.as_ptr() as usize, 0, expect), "a redirected TLS load");
            code[12] ^= 1;
            assert!(!starts_with(code.as_ptr() as usize, 0, expect), "another instruction after it");
            let plain = CLOCK_SET_BYTES;
            let mut other = plain;
            other[0] ^= 1;
            assert!(starts_with(plain.as_ptr() as usize, 0, &CLOCK_SET_BYTES) && !starts_with(other.as_ptr() as usize, 0, &CLOCK_SET_BYTES));
        }
    }

    /// Every entry the levers check is the exe's own code, also after the
    /// loader's TLS rewrite (a run logged `the code at
    /// title+0x524c040 is not the expected one`: its `mov rax, gs:[58h]`
    /// had become `call thunk; nop4` at 0.415 s, the check ran at 0.453 s).
    #[test]
    fn the_checked_entries_are_the_exes_also_after_the_tls_rewrite() {
        // The rewrite as xnw-loader tls.rs `patch` makes it, on a copy.
        let rewrite = |b: &[u8]| -> Vec<u8> {
            let mut v = b.to_vec();
            let mut i = 0;
            while i < v.len() {
                if is_tls_load(&v[i..]) {
                    v[i..i + TLS_LOAD_LEN].copy_from_slice(&[0xE8, 0x11, 0x22, 0x33, 0x44, 0x0F, 0x1F, 0x40, 0x00]);
                    i += TLS_LOAD_LEN;
                } else {
                    i += 1;
                }
            }
            v
        };
        // The lock's own bytes, in memory: plain, rewritten, one byte off.
        let lock = CINE_CALLS.iter().find(|c| c.0 == CINE_LOCK_RVA).unwrap().1;
        let moved = rewrite(lock);
        assert_ne!(moved.as_slice(), lock, "the lock opens with a TLS load");
        assert!(code_is(lock, lock) && code_is(&moved, lock));
        let mut off = moved.clone();
        off[TLS_LOAD_LEN] ^= 1;
        assert!(!code_is(&off, lock), "a byte after the load still counts");
        let mut not_call = moved.clone();
        not_call[0] = 0xE9;
        assert!(!code_is(&not_call, lock), "only the loader's call form");
        let Some(exe) = crate::patches::Exe::open() else {
            eprintln!("no title executable here (XNW_TITLE_EXE): the lever entries were not checked against it");
            return;
        };
        let mut all: Vec<(usize, &[u8])> = CINE_CALLS.to_vec();
        all.extend([(CINE_TICK_RVA, &CINE_TICK_BYTES[..]), (CLOCK_TICK_RVA, &CLOCK_TICK_BYTES[..]), (CLOCK_SET_RVA, &CLOCK_SET_BYTES[..])]);
        for (rva, bytes) in all {
            let file = exe.at(rva, bytes.len());
            assert_eq!(file, bytes, "title+{rva:#x}");
            assert!(code_is(&rewrite(file), bytes), "title+{rva:#x} after the TLS rewrite");
        }
        // The two group-lock calls are the ones that carry a TLS load.
        let with_tls: Vec<usize> = CINE_CALLS.iter().filter(|c| c.1 != rewrite(c.1).as_slice()).map(|c| c.0).collect();
        assert_eq!(with_tls, [CINE_LOCK_RVA, CINE_UNLOCK_RVA]);
    }

    #[test]
    fn intro_values_parse_and_only_a_long_unskippable_playing_section_is_sought() {
        assert_eq!((parse_intro("seekend"), parse_intro(" seekend:6 "), parse_intro("seekend:4:20")), (Some((5, 30.0)), Some((6, 30.0)), Some((4, 20.0))));
        for bad in ["", "seek", "seekend:", "seekend:0", "seekend:65", "seekend:2:1", "seekend:2:30:1", "skip"] {
            assert_eq!(parse_intro(bad), None, "{bad:?}");
        }
        // The opening's first section as the log has it: playing, 0.07 s of 55 s.
        assert_eq!(intro_target(6, 0, 0, 0, 0x20, 0, 0.07, 55.0, 30.0), Some(54.5));
        // The landing point is short of where the title marks a scene skipped (end - 1e-6).
        assert!(54.5 < 55.0 - 1e-6);
        for state in [1, 2, 3, 4, 5, 7, 8, 9, 10] {
            assert_eq!(intro_target(state, 0, 0, 0, 0, 0, 0.0, 55.0, 30.0), None, "state {state}");
        }
        assert_eq!(intro_target(6, 4, 0, 0, 0, 0, 0.0, 55.0, 30.0), None, "skippable by definition: XNW_CINE's");
        assert_eq!(intro_target(6, 0, 0x08, 0, 0, 0, 0.0, 55.0, 30.0), None, "refused by the handle methods (+0x49 & 0x18)");
        assert_eq!(intro_target(6, 0, 0, 4, 0, 0, 0.0, 55.0, 30.0), None, "a seek is pending");
        assert_eq!(intro_target(6, 0, 0, 0, 0x40, 0, 0.0, 55.0, 30.0), None, "skipped");
        assert_eq!(intro_target(6, 0, 0, 0, 0, 0x10, 0.0, 55.0, 30.0), None, "a skip under way");
        assert_eq!(intro_target(6, 0, 0, 0, 0, 0, 0.0, 17.4, 30.0), None, "a short section (the ambient scenes)");
        assert_eq!(intro_target(6, 0, 0, 0, 0, 0, 53.6, 55.0, 30.0), None, "within a second of the landing point: no second seek after the first lands");
        assert_eq!(intro_target(6, 0, 0, 0, 0, 0, f32::NAN, 55.0, 30.0), None);
    }

    /// The seek, the title's handle method whose steps the lever takes, and
    /// the two ways a playing instance leaves state 6 are the exe's code.
    #[test]
    fn the_seek_its_handle_method_and_the_end_of_play_are_the_executables() {
        let Some(exe) = crate::patches::Exe::open() else {
            eprintln!("no title executable here (XNW_TITLE_EXE): the intro lever was not checked against it");
            return;
        };
        for (rva, bytes) in INTRO_CALLS {
            assert_eq!(exe.at(rva, bytes.len()), bytes, "title+{rva:#x}");
        }
        assert_eq!(exe.rel32_target(CINE_SEEK_THUNK_RVA), CINE_SEEK_RVA);
        // The seek marks a scene skipped from end - 1e-6 on: the vaddss operand at title+0x51892f.
        let disp = i32::from_le_bytes(exe.at(0x51892f + 4, 4).try_into().unwrap());
        let eps = (0x51892f + 8usize).wrapping_add(disp as isize as usize);
        assert_eq!((eps, f32::from_le_bytes(exe.at(eps, 4).try_into().unwrap())), (0x6159f9c, -1e-6));
        // The handle's seek method title+0x3162b60: context call, group lock, refusal test, stamp, seek(inst, t, 0, 0), unlock.
        let m = CINE_SEEK_BY_HANDLE_RVA;
        assert_eq!(exe.rel32_target(m + 0x30), CINE_IN_CONTEXT_RVA);
        assert_eq!(exe.at(m + 0x35, 8), [0x48, 0x0F, 0xBE, 0x86, 0x24, 0x0E, 0x00, 0x00], "movsx rax, byte [rsi+0E24h]: the lock index");
        assert_eq!(exe.rel32_target(m + 0x4C), CINE_LOCK_RVA);
        assert_eq!(exe.at(m + 0x51, 6), [0xF6, 0x46, 0x49, 0x18, 0x75, 0x51], "test byte [rsi+49h], 18h; jnz unlock");
        assert_eq!(exe.at(m + 0x57, 6), [0x8A, 0x8E, 0x97, 0x04, 0x00, 0x00], "mov cl, [rsi+497h]");
        assert_eq!(exe.rel32_target(m + 0x5D), CINE_STAMP_RVA);
        assert_eq!(exe.at(m + 0x65, 16), [0xC5, 0xF8, 0x28, 0xCE, 0x45, 0x31, 0xC0, 0x45, 0x31, 0xC9, 0x89, 0x86, 0x48, 0x0E, 0x00, 0x00], "xmm1 = t; r8d = r9d = 0; [rsi+0E48h] = stamp");
        assert_eq!(exe.rel32_target(m + 0x75), CINE_SEEK_THUNK_RVA);
        assert_eq!(exe.rel32_target(m + 0x96), CINE_UNLOCK_RVA);
        // Its vtable: slot 1 of 0x1463CDD30.
        assert_eq!(exe.at(0x63cdd38, 8), (0x1_4000_0000u64 + m as u64).to_le_bytes());
        // The section end the lever reads: title+0x24ec3b `vmovss xmm0, [rdi+50h]` ... `vmovss [rsi+480h], xmm0`.
        assert_eq!(exe.at(0x24ec3b, 5), [0xC5, 0xFA, 0x10, 0x47, 0x50]);
        assert_eq!(exe.at(0x24ec48, 8), [0xC5, 0xFA, 0x11, 0x86, 0x80, 0x04, 0x00, 0x00]);
        // State 6 after its update (title+0x50f9250): a pending seek goes to state 8 ...
        assert_eq!(exe.at(0x50f92b8, 13), [0xF6, 0x86, 0x95, 0x04, 0x00, 0x00, 0x04, 0x0F, 0x85, 0x93, 0x00, 0x00, 0x00], "test byte [rsi+495h], 4; jnz title+0x50f9358");
        assert_eq!(exe.at(0x50f9358, 8), [0x48, 0x89, 0xF1, 0xBA, 0x08, 0x00, 0x00, 0x00], "mov rcx, rsi; mov edx, 8");
        assert_eq!(exe.rel32_target(0x50f9367), 0x23cbb0, "jmp set_state");
        // ... and the end of play, sought or not: +0x7FD (or +0x495 bit 0) -> state 10 through the same setter.
        assert_eq!(exe.at(0x50f92f3, 18), [0x80, 0xBE, 0xFD, 0x07, 0x00, 0x00, 0x00, 0x75, 0x27, 0xF6, 0x86, 0x95, 0x04, 0x00, 0x00, 0x01, 0x75, 0x1E]);
        assert_eq!(exe.at(0x50f9323, 8), [0x48, 0x89, 0xF1, 0xBA, 0x0A, 0x00, 0x00, 0x00], "mov rcx, rsi; mov edx, 0Ah");
        assert_eq!(exe.rel32_target(0x50f932b), 0x23cbb0);
        // What reports a skipped scene to the handle's users: +0x496 bit 6 (title+0x316040a).
        assert_eq!(exe.at(0x316040a, 11), [0x40, 0x8A, 0xB7, 0x96, 0x04, 0x00, 0x00, 0x40, 0x80, 0xE6, 0x40]);
        // What a waiting script asks: the scene method `IsFinished` (registrar title+0x1d7b380: `mov edx, joaat`,
        // then `lea rcx, binding`), binding title+0x4bfe790 -> title+0x4bfe750 -> title+0xe21ce0(handle, 1):
        // state 10, or 9. The lever writes no state: the title's own end of play sets 10.
        let name = "IsFinished".bytes().fold(0u32, |h, b| {
            let h = h.wrapping_add(u32::from(b));
            let h = h.wrapping_add(h << 10);
            h ^ (h >> 6)
        });
        let name = name.wrapping_add(name << 3);
        let name = name ^ (name >> 11);
        let name = name.wrapping_add(name << 15);
        assert_eq!(exe.at(0x1d7cd35, 5), [&[0xBA][..], &name.to_le_bytes()].concat(), "the method name's case-sensitive joaat");
        let lea = i32::from_le_bytes(exe.at(0x1d7cd78 + 3, 4).try_into().unwrap());
        assert_eq!((exe.at(0x1d7cd78, 3), (0x1d7cd78 + 7usize).wrapping_add(lea as isize as usize)), (&[0x48, 0x8D, 0x0D][..], 0x4bfe790));
        assert_eq!(exe.at(0x4bfe777, 2), [0xB2, 0x01], "mov dl, 1");
        assert_eq!(exe.rel32_target(0x4bfe779), 0xe21ce0);
        assert_eq!(exe.at(0xe21d2c, 24), [0x8B, 0x87, 0x3C, 0x04, 0x00, 0x00, 0x83, 0xF8, 0x0A, 0x0F, 0x94, 0xC1, 0x83, 0xF8, 0x09, 0x0F, 0x94, 0xC0, 0x40, 0x20, 0xC6, 0x40, 0x08, 0xCE]);
        // The directors' fields the lines print: title+0x45f4ab `mov cl, [rax+25Eh]; mov eax, [rax+258h]`.
        assert_eq!(exe.at(0x45f4ab, 12), [0x8A, 0x88, 0x5E, 0x02, 0x00, 0x00, 0x8B, 0x80, 0x58, 0x02, 0x00, 0x00]);
        assert_eq!(exe.at(0x4e74840, 8), [0x48, 0x8B, 0x05, 0x69, 0xCC, 0xA4, 0x04, 0xC3], "mov rax, [qword_1498C14B0]; ret");
    }

    #[test]
    fn save_values_parse_and_the_gates_open_only_as_the_titles() {
        assert_eq!((parse_save_at("3600"), parse_save_at(" 1200 , Force "), parse_save_at("0"), parse_save_at("12,always"), parse_save_at("x")), (Some((3600, false)), Some((1200, true)), None, None, None));
        let open = SaveGates { blocked: 0, flag: Some(1), refuse: 0, busy: 0, op: Some((0, 12, 1)), opcode: 0, fsm: 0, menu: 0 };
        assert!(open.open());
        for g in [
            SaveGates { blocked: 1, ..open },
            SaveGates { flag: Some(0), ..open },
            SaveGates { flag: None, ..open },
            SaveGates { refuse: 1, ..open },
            SaveGates { busy: 1, ..open },
            SaveGates { op: Some((1, 3, 0)), ..open },
            SaveGates { op: None, ..open },
        ] {
            assert!(!g.open(), "{g:?}");
        }
        assert!(open.line().contains("profile autosave flag +0x410=1") && open.line().contains("op queue 0 step 12 result 1"));
    }

    /// The saves folder's layout as xgamesave.rs commits it: the newest
    /// version per container and its blob bytes; a new or newer one counts.
    #[test]
    fn a_save_written_is_a_container_with_a_newer_version() {
        let root = std::env::temp_dir().join(format!("xnw-save2-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let user = root.join("0009000000000001").join("user");
        let put = |c: &str, v: u64, blob: usize| {
            let d = user.join(c).join(format!("v{v}"));
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("0"), vec![0u8; blob]).unwrap();
            std::fs::write(d.join("manifest"), b"m").unwrap();
        };
        put("Player~1", 1, 2048);
        put("SARC0015~f3", 1, 100);
        let before: Vec<(String, u64)> = save_versions(&root).into_iter().map(|c| (c.0, c.1)).collect();
        assert_eq!(before, [("Player~1".to_owned(), 1), ("SARC0015~f3".to_owned(), 1)]);
        assert!(save_written(&before, &save_versions(&root)).is_empty());
        put("SARC0015~f3", 2, 255_088);
        std::fs::create_dir_all(user.join("SARC0015~f3").join("v3.new")).unwrap();
        put("Player~1", 2, 10);
        let w = save_written(&before, &save_versions(&root));
        assert_eq!(w, [("SARC0015~f3".to_owned(), 2, 255_088), ("Player~1".to_owned(), 2, 10)], "SARC first; a staged v3.new is no version");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The request, its gates and the write's globals against the
    /// executable: every address the lever reads is one the title's own
    /// code reads at those instructions.
    #[test]
    fn the_autosave_request_and_its_gates_are_the_executables() {
        let Some(exe) = crate::patches::Exe::open() else {
            eprintln!("no title executable here (XNW_TITLE_EXE): the autosave request was not checked against it");
            return;
        };
        // rip-relative operand: disp32 at `at + disp`, instruction length `len`.
        let rip = |at: usize, disp: usize, len: usize| (at + len).wrapping_add(i32::from_le_bytes(exe.at(at + disp, 4).try_into().unwrap()) as isize as usize);
        for (rva, bytes) in [(SAVE_REQUEST_RVA, &SAVE_REQUEST_BYTES[..]), (SAVE_BLOCKED_RVA, &SAVE_BLOCKED_BYTES[..]), (SAVE_OP_IDLE_RVA, &SAVE_OP_IDLE_BYTES[..]), (SAVE_ARM_RVA, &SAVE_ARM_BYTES[..])] {
            assert_eq!(exe.at(rva, bytes.len()), bytes, "title+{rva:#x}");
            assert!(code_is(exe.at(rva, bytes.len()), bytes), "title+{rva:#x} as mapped (no TLS load)");
        }
        let r = SAVE_REQUEST_RVA;
        // call blocked thunk -> jmp body: mov al, [byte_1478B27F1]; ret.
        assert_eq!((exe.rel32_target(r + 4), exe.at(SAVE_BLOCKED_THUNK_RVA, 1)[0], exe.rel32_target(SAVE_BLOCKED_THUNK_RVA)), (SAVE_BLOCKED_THUNK_RVA, 0xE9, SAVE_BLOCKED_RVA));
        assert_eq!(rip(SAVE_BLOCKED_RVA, 2, 6), SAVE_BLOCKED_FLAG_RVA);
        // mov rax, [qword_14AAD3CE0]; mov rcx, [rax+8]; mov rax, [rcx]; call [rax+48h]; cmp byte [rax+410h], 1.
        assert_eq!(rip(r + 0xD, 3, 7), SAVE_PROFILE_RVA);
        assert_eq!(exe.at(r + 0x14, 17), [0x48, 0x8B, 0x48, 0x08, 0x48, 0x8B, 0x01, 0xFF, 0x50, 0x48, 0x80, 0xB8, 0x10, 0x04, 0x00, 0x00, 0x01]);
        assert_eq!(SAVE_PROFILE_FLAG, 0x410);
        // cmp byte [byte_149DF77C0], 0 (imm8 after the disp); the tail jmp to the queue step.
        assert_eq!(rip(r + 0x27, 2, 7), SAVE_REFUSE_RVA);
        assert_eq!((exe.at(r + 0x3B, 1)[0], exe.rel32_target(r + 0x3B)), (0xE9, SAVE_QUEUE_RVA));
        // The queue step calls the idle gate: cmp [byte_1478B2778], 0; mov rcx, [g_saveop_autosave]; cmp dword [rcx+8], 1.
        assert_eq!((exe.at(SAVE_QUEUE_RVA + 4, 1)[0], exe.rel32_target(SAVE_QUEUE_RVA + 4)), (0xE8, SAVE_OP_IDLE_RVA));
        assert_eq!((rip(SAVE_OP_IDLE_RVA, 2, 7), rip(SAVE_OP_IDLE_RVA + 0xC, 3, 7)), (SAVE_OP_BUSY_RVA, SAVE_OP_RVA));
        assert_eq!(rip(SAVE_QUEUE_RVA + 0x14, 3, 7), SAVE_OP_RVA, "the op pushed");
        // The arm step: the same profile flag; mov eax, [write fsm]; or eax, [opcode].
        assert_eq!((rip(SAVE_ARM_RVA + 0xA, 3, 7), rip(SAVE_ARM_RVA + 0x30, 2, 6), rip(SAVE_ARM_RVA + 0x36, 2, 6)), (SAVE_PROFILE_RVA, SAVE_FSM_RVA, SAVE_OPCODE_RVA));
        // The menu UI state, read by save_menu_pick_slot 0x1428BC010: cmp [dword_1478CB7F4], ...
        assert_eq!(exe.at(0x28bc014, 2), [0x83, 0x3D]);
        assert_eq!(rip(0x28bc014, 2, 7), SAVE_MENU_RVA);
    }

    #[test]
    fn the_clock_is_set_only_when_it_left_the_target() {
        assert_eq!(clock_step((12, 0), (12, 0)), None);
        assert_eq!(clock_step((12, 1), (12, 0)), Some((12, 0)), "the tick added a minute");
        assert_eq!(clock_step((3, 0), (12, 0)), Some((12, 0)), "a scene set its own time");
    }

    #[test]
    fn lod_values_parse() {
        assert_eq!(parse_lod("0.25"), Some((2, 0.25)));
        assert_eq!(parse_lod(" 1.5 , Force "), Some((3, 1.5)));
        assert_eq!(parse_lod("read"), Some((1, 0.0)));
        for bad in ["", "0", "0.01", "5", "0.5,always", "nan", "inf", "low"] {
            assert_eq!(parse_lod(bad), None, "{bad}");
        }
    }

    static FAKE_SCALE: AtomicU32 = AtomicU32::new(0);
    static FAKE_RAISED: AtomicU8 = AtomicU8::new(0);

    /// The title's getter in small: the scale, raised to 1.0 while the mode holds.
    extern "system" fn fake_getter() -> f32 {
        let v = f32::from_bits(FAKE_SCALE.load(Ordering::Relaxed));
        if FAKE_RAISED.load(Ordering::Relaxed) != 0 { v.max(1.0) } else { v }
    }

    /// The hook stores the value where the title keeps it, returns what the
    /// title's getter makes of it, and only `force` goes past the getter.
    #[test]
    fn the_lod_hook_stores_then_asks_the_titles_getter() {
        LOD_SLOT.store(FAKE_SCALE.as_ptr() as usize, Ordering::Relaxed);
        LOD_GETTER.store(fake_getter as usize, Ordering::Relaxed);
        let set = |mode: u8, value: f32, file: f32, raised: u8| {
            LOD_MODE.store(mode, Ordering::Relaxed);
            LOD_VALUE.store(value.to_bits(), Ordering::Relaxed);
            FAKE_SCALE.store(file.to_bits(), Ordering::Relaxed);
            FAKE_RAISED.store(raised, Ordering::Relaxed);
        };
        let stored = || f32::from_bits(FAKE_SCALE.load(Ordering::Relaxed));
        set(1, 0.0, 0.25, 0);
        assert_eq!((lod_hook(), stored()), (0.25, 0.25), "read: nothing stored");
        set(1, 0.0, 0.25, 1);
        assert_eq!((lod_hook(), stored()), (1.0, 0.25), "read: the title's raise shows");
        set(2, 0.5, 1.0, 0);
        assert_eq!((lod_hook(), stored()), (0.5, 0.5), "set: stored, then the getter's answer");
        set(2, 0.5, 1.0, 1);
        assert_eq!((lod_hook(), stored()), (1.0, 0.5), "set: the title's raise stays");
        set(3, 0.5, 1.0, 1);
        assert_eq!((lod_hook(), stored()), (0.5, 0.5), "force: the value itself");
        LOD_MODE.store(0, Ordering::Relaxed);
    }

    /// The addresses against the executable: the three sites call the
    /// getter, the getter loads the scale from the address `settings_apply`
    /// stores graphics +0xD4 to, and raises it to 1.0 on its mode branch.
    #[test]
    fn the_lod_scale_sites_are_the_executables() {
        let Some(exe) = crate::patches::Exe::open() else {
            eprintln!("no title executable here (XNW_TITLE_EXE): the LOD scale sites were not checked against it");
            return;
        };
        let rip = |at: usize, len: usize| (at + len).wrapping_add(i32::from_le_bytes(exe.at(at + len - 4, 4).try_into().unwrap()) as isize as usize);
        for site in LOD_SITES {
            assert_eq!((exe.at(site, 1)[0], exe.rel32_target(site)), (0xE8, LOD_GETTER_RVA), "title+{site:#x}");
        }
        assert_eq!(exe.at(LOD_GETTER_RVA, LOD_GETTER_BYTES.len()), LOD_GETTER_BYTES);
        // The getter: test [g_stream_slice_flags], 2; jne raise; call mode test; test al, al; jne raise.
        assert_eq!(((0x45f46busize).wrapping_add(i32::from_le_bytes(exe.at(0x45f466, 4).try_into().unwrap()) as isize as usize), exe.at(0x45f46a, 1)[0]), (0xab44324, 2));
        assert_eq!((exe.at(0x45f46d, 1)[0], exe.rel32_target(0x45f46d)), (0xE8, 0x45f4a0));
        // Plain path: vmovss xmm0, [scale]; raise path: vmovss xmm0, [scale]; vmaxss xmm0, xmm0, [1.0].
        assert_eq!((&exe.at(0x45f476, 4)[..], rip(0x45f476, 8)), (&[0xC5, 0xFA, 0x10, 0x05][..], LOD_SCALE_RVA));
        assert_eq!((&exe.at(0x45f483, 4)[..], rip(0x45f483, 8)), (&[0xC5, 0xFA, 0x10, 0x05][..], LOD_SCALE_RVA));
        assert_eq!(&exe.at(0x45f48b, 4)[..], &[0xC5, 0xFA, 0x5F, 0x05][..]);
        assert_eq!(exe.at(rip(0x45f48b, 8), 4), 1.0f32.to_le_bytes());
        // The title's setter: settings_apply loads graphics +0xD4 and stores it to the scale.
        assert_eq!(exe.at(0x224360a, 8), [0xC5, 0xFA, 0x10, 0x86, 0xD4, 0x00, 0x00, 0x00]);
        assert_eq!((&exe.at(0x2243612, 4)[..], rip(0x2243612, 8)), (&[0xC5, 0xFA, 0x11, 0x05][..], LOD_SCALE_RVA));
        // The builder divides its final scale by the stored one: vdivss xmm5, xmm10, [scale]; vmovss [scale+0xC], xmm5.
        assert_eq!((rip(0x3e579ed, 8), rip(0x3e579f5, 8)), (LOD_SCALE_RVA, LOD_SCALE_RVA + 0xC));
        // The mode test asks two objects (accessors of qword_1498C14B0/B8) for flag +0x25E and state +0x258.
        assert_eq!((exe.rel32_target(0x45f4a6), rip(0x4e74840, 7), exe.rel32_target(0x45f4d7), rip(0x4e74850, 7)), (0x4e74840, 0x98c14b0, 0x4e74850, 0x98c14b8));
        assert_eq!(exe.at(0x45f4ab, 12), [0x8A, 0x88, 0x5E, 0x02, 0x00, 0x00, 0x8B, 0x80, 0x58, 0x02, 0x00, 0x00]);
        // The first is made as joaat("ANIM_SCENE_DIRECTOR") of class joaat("camAnimSceneDirector") and stored there.
        assert_eq!(exe.at(0x57156e2, 10), [0xB9, 0x86, 0xD4, 0xD6, 0xEA, 0xBA, 0xD2, 0x61, 0xD3, 0xD4]);
        assert_eq!((exe.rel32_target(0x57156ec), rip(0x57156f1, 7)), (0xd9c4b0, 0x98c14b0));
        assert_eq!((joaat("ANIM_SCENE_DIRECTOR"), joaat("camAnimSceneDirector")), (0xEAD6_D486, 0xD4D3_61D2));
    }

    /// The title's name hash (lower-cased one-at-a-time).
    fn joaat(s: &str) -> u32 {
        let mut h = 0u32;
        for b in s.bytes() {
            h = h.wrapping_add(u32::from(b.to_ascii_lowercase()));
            h = h.wrapping_add(h << 10);
            h ^= h >> 6;
        }
        h = h.wrapping_add(h << 3);
        h ^= h >> 11;
        h.wrapping_add(h << 15)
    }
}
