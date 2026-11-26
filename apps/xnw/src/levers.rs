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
    if base == 0 || mode == 0 || inst == 0 {
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
