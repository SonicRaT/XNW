//! The title's render stages, timed and skipped (ENG1). Every stage the
//! title queues in a frame (`drawlist_append_entry` 0x144FF03C0's 48-byte
//! ring entries) is run through one function, `title+0x43c3890`
//! (`mov edx, [rcx+18h]; mov rax, [rcx+8]; mov rcx, [rcx+10h]; jmp rax`:
//! `cb(phase, cbArg)` of the entry in rcx), by the render thread for inline
//! entries and by the five Subrender workers for the rest. The entry
//! holds the stage id at +0x1C. A detour there (`detours.rs`) gives:
//!
//! - `XNW_STAGE_TIME=1`: the CPU time of the calling thread and the wall
//!   time around each callback, summed per stage; one `title stages:` line
//!   per 60 frames (a frame ends with stage 64, queued last). The CPU time
//!   holds our own hooks' work inside the stage too.
//! - `XNW_STAGE_TIME=2` (ENG6): the same, and a second line `title stage
//!   callbacks:` for the stages that run through more than one callback in
//!   the window (36: the sky LUTs and the probe capture; 67; 19), one row
//!   per callback `stage title+rva cpu (calls a frame)`.
//! - `XNW_STAGE_SKIP=<ids>` (decimal, comma list): those stages' callbacks
//!   are not called. An empty callback is a case the title has itself
//!   (stage 20 registers a `nullsub`); what each stage's work feeds on the
//!   CPU is.
//!
//! Both off by default; with neither (nor `XNW_AB` or
//! `XNW_STOP_WORLD_FRAMES`, `ab.rs`, which count its frames, nor
//! `XNW_ENCODE_TIME` or `XNW_PATCH=noencode`, `encode.rs`, whose lines
//! close with the window) the detour is not installed.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use xnw_core::win::{GetCurrentThread, QueryThreadCycleTime};

/// `title+rva` of the entry runner and its whole 14 bytes (read from the exe).
pub(crate) const ENTRY_RVA: usize = 0x43c3890;
pub(crate) const ENTRY_BYTES: [u8; 14] = [0x8B, 0x51, 0x18, 0x48, 0x8B, 0x41, 0x08, 0x48, 0x8B, 0x49, 0x10, 0x48, 0xFF, 0xE0];

/// Stage ids are below this (the largest queued is 90); larger ones share the last slot.
pub(crate) const STAGES: usize = 128;
/// No stage callback runs on the thread.
pub(crate) const NO_STAGE: usize = STAGES - 1;

thread_local! {
    /// The stage whose callback runs on this thread now (`encode.rs` splits its rows by it).
    pub(crate) static STAGE: std::cell::Cell<usize> = const { std::cell::Cell::new(NO_STAGE) };
}
/// The frame's last stage (frame end 0x142246A40, queued by 0x140511A40).
const FRAME_END: usize = 64;
/// Frames per line.
const EVERY: u64 = 60;

/// The trampoline of the detoured function (set by `detours.rs`).
pub(crate) static ORIGINAL: AtomicUsize = AtomicUsize::new(0);
static TIME: AtomicBool = AtomicBool::new(false);
/// `XNW_STAGE_TIME=2`: rows per callback too.
static SPLIT: AtomicBool = AtomicBool::new(false);
/// The title's base (for the callbacks' `title+rva`).
static BASE: AtomicUsize = AtomicUsize::new(0);
/// Per callback: its address (0 = free), its stage, cycles and calls. Open addressing from the address's hash.
const CALLBACKS: usize = 128;
static CB_KEY: [AtomicUsize; CALLBACKS] = [const { AtomicUsize::new(0) }; CALLBACKS];
static CB_STAGE: [AtomicUsize; CALLBACKS] = [const { AtomicUsize::new(0) }; CALLBACKS];
static CB_CYCLES: [AtomicU64; CALLBACKS] = [const { AtomicU64::new(0) }; CALLBACKS];
static CB_CALLS: [AtomicU64; CALLBACKS] = [const { AtomicU64::new(0) }; CALLBACKS];
static SKIP: [AtomicBool; STAGES] = [const { AtomicBool::new(false) }; STAGES];
static CYCLES: [AtomicU64; STAGES] = [const { AtomicU64::new(0) }; STAGES];
static WALL_NS: [AtomicU64; STAGES] = [const { AtomicU64::new(0) }; STAGES];
static CALLS: [AtomicU64; STAGES] = [const { AtomicU64::new(0) }; STAGES];
static SKIPPED: [AtomicU64; STAGES] = [const { AtomicU64::new(0) }; STAGES];
static FRAMES: AtomicU64 = AtomicU64::new(0);
/// A callback has come through the detour.
static FIRST: AtomicBool = AtomicBool::new(false);
/// Where the cycle counter and the clock stood when the meter was set up.
static START: OnceLock<(Instant, u64)> = OnceLock::new();

fn tsc() -> u64 {
    // SAFETY: reads the time-stamp counter.
    #[allow(unused_unsafe)]
    unsafe {
        core::arch::x86_64::_rdtsc()
    }
}

/// The cycle counter's rate since the meter's clock started (started here if not yet).
pub(crate) fn cycles_per_second() -> f64 {
    let (t, c) = *START.get_or_init(|| (Instant::now(), tsc()));
    (tsc().wrapping_sub(c) as f64 / t.elapsed().as_secs_f64().max(1e-9)).max(1.0)
}

/// Nanoseconds per cycle, times 2^20 (0: not measured yet); renewed with every line.
static NS_PER_CYCLE_Q20: AtomicU64 = AtomicU64::new(0);

/// `cycles` of thread time in nanoseconds, at the last measured rate.
fn cycles_ns(cycles: u64) -> u64 {
    let mut q = NS_PER_CYCLE_Q20.load(Ordering::Relaxed);
    if q == 0 {
        q = (1e9 / cycles_per_second() * (1u64 << 20) as f64) as u64;
        NS_PER_CYCLE_Q20.store(q, Ordering::Relaxed);
    }
    ((u128::from(cycles) * u128::from(q)) >> 20) as u64
}

pub(crate) fn thread_cycles() -> u64 {
    let mut c = 0u64;
    // SAFETY: the calling thread's pseudo handle and a u64 of ours.
    unsafe { QueryThreadCycleTime(GetCurrentThread(), &mut c) };
    c
}

/// The stage ids of a `XNW_STAGE_SKIP` value, and the items that are none.
fn parse_skip(v: &str) -> (Vec<usize>, Vec<String>) {
    let (mut ids, mut bad) = (Vec::new(), Vec::new());
    for item in v.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match item.parse::<usize>() {
            Ok(n) if n < STAGES => ids.push(n),
            _ => bad.push(item.to_owned()),
        }
    }
    (ids, bad)
}

/// The slot of callback `cb` (claimed if new); None when the table is full.
fn callback_slot(cb: usize, stage: usize) -> Option<usize> {
    let start = (cb >> 4).wrapping_mul(0x9E37_79B9) % CALLBACKS;
    for k in 0..CALLBACKS {
        let i = (start + k) % CALLBACKS;
        match CB_KEY[i].compare_exchange(0, cb, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => {
                CB_STAGE[i].store(stage, Ordering::Relaxed);
                return Some(i);
            }
            Err(now) if now == cb => return Some(i),
            Err(_) => {}
        }
    }
    None
}

/// Reads `XNW_STAGE_TIME` and `XNW_STAGE_SKIP`; whether the detour is wanted.
pub(crate) fn configure(base: usize) -> bool {
    BASE.store(base, Ordering::Relaxed);
    let (time, split) = match std::env::var("XNW_STAGE_TIME").unwrap_or_default().trim() {
        "" | "0" => (false, false),
        "1" => (true, false),
        "2" => (true, true),
        other => {
            xnw_core::warn!("XNW_STAGE_TIME={other}: not 0, 1 or 2; the stage meter stays off");
            (false, false)
        }
    };
    SPLIT.store(split, Ordering::Relaxed);
    let (ids, bad) = parse_skip(&std::env::var("XNW_STAGE_SKIP").unwrap_or_default());
    for b in &bad {
        xnw_core::warn!("XNW_STAGE_SKIP={b}: not a stage id (0..{STAGES}); ignored");
    }
    if !time && ids.is_empty() {
        return false;
    }
    set(time, &ids);
    xnw_core::info!("title stages: meter {}, skipped stages {ids:?} (callbacks run through title+{ENTRY_RVA:#x})", if time { "on" } else { "off" });
    true
}

fn set(time: bool, skip: &[usize]) {
    START.get_or_init(|| (Instant::now(), tsc()));
    TIME.store(time, Ordering::Relaxed);
    for (i, s) in SKIP.iter().enumerate() {
        s.store(skip.contains(&i), Ordering::Relaxed);
    }
}

/// The detour of `title+0x43c3890`: rcx = the ring entry.
pub(crate) extern "system" fn hook(entry: usize) -> usize {
    // SAFETY: the ring entry the title passes (48 bytes, stage at +0x1C).
    let stage = (unsafe { ((entry + 0x1C) as *const u32).read() } as usize).min(STAGES - 1);
    // SAFETY: the trampoline made at install: the function's own bytes.
    let original: extern "system" fn(usize) -> usize = unsafe { std::mem::transmute(ORIGINAL.load(Ordering::Relaxed)) };
    STAGE.with(|s| s.set(stage));
    let result = if SKIP[stage].load(Ordering::Relaxed) {
        SKIPPED[stage].fetch_add(1, Ordering::Relaxed);
        0
    } else if TIME.load(Ordering::Relaxed) {
        let (c0, t0) = (thread_cycles(), Instant::now());
        let r = original(entry);
        let cycles = thread_cycles().wrapping_sub(c0);
        CYCLES[stage].fetch_add(cycles, Ordering::Relaxed);
        if SPLIT.load(Ordering::Relaxed) {
            // SAFETY: the entry's callback pointer at +8.
            if let Some(i) = callback_slot(unsafe { ((entry + 8) as *const usize).read() }, stage) {
                CB_CYCLES[i].fetch_add(cycles, Ordering::Relaxed);
                CB_CALLS[i].fetch_add(1, Ordering::Relaxed);
            }
        }
        let wall = t0.elapsed().as_nanos() as u64;
        WALL_NS[stage].fetch_add(wall, Ordering::Relaxed);
        xnw_core::framewall::add(xnw_core::framewall::STAGE_WALL, wall);
        xnw_core::framewall::add(xnw_core::framewall::STAGE_CPU, cycles_ns(cycles));
        CALLS[stage].fetch_add(1, Ordering::Relaxed);
        r
    } else {
        original(entry)
    };
    STAGE.with(|s| s.set(NO_STAGE));
    if !FIRST.swap(true, Ordering::Relaxed) {
        xnw_core::info!("title stages: first callback, stage {stage}, at {:.1} s", START.get().map_or(0.0, |s| s.0.elapsed().as_secs_f64()));
    }
    if stage == FRAME_END {
        frame_end(FRAMES.fetch_add(1, Ordering::Relaxed) + 1);
    }
    result
}

/// World frames so far (stage 64 ends one).
pub(crate) fn world_frames() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// World frames in the meter's open window.
static WINDOW: AtomicU64 = AtomicU64::new(0);

/// The end of world frame `n` (1 = the first): the meter's line every
/// `EVERY` frames, and what `XNW_AB` / `XNW_STOP_WORLD_FRAMES` ask for
/// there (`ab.rs`: the window closes early at the switch and at the stop).
fn frame_end(n: u64) {
    let w = WINDOW.fetch_add(1, Ordering::Relaxed) + 1;
    let act = crate::ab::at(n);
    if w < EVERY && act == crate::ab::At::Nothing {
        return;
    }
    WINDOW.store(0, Ordering::Relaxed);
    let line = take_line(w);
    if TIME.load(Ordering::Relaxed) || !line.ends_with("none") {
        if crate::ab::armed() {
            xnw_core::info!("{}", line.replacen("title stages:", &format!("title stages [{}]:", crate::ab::phase()), 1));
        } else {
            xnw_core::info!("{line}");
        }
        if SPLIT.load(Ordering::Relaxed) {
            xnw_core::info!("{}", take_callback_line(w));
        }
    }
    crate::encode::window_end(w, &if crate::ab::armed() { format!(" [{}]", crate::ab::phase()) } else { String::new() });
    match act {
        crate::ab::At::Switch => crate::ab::switch(n),
        crate::ab::At::Stop => crate::ab::stop(n),
        crate::ab::At::Nothing => {}
    }
}

/// Stages skipped from now on as well (`XNW_AB_SKIP`).
pub(crate) fn skip_more(ids: &[usize]) {
    for &i in ids.iter().filter(|&&i| i < STAGES) {
        SKIP[i].store(true, Ordering::Relaxed);
    }
}

/// The line for the last `frames` frames; the sums start again.
fn take_line(frames: u64) -> String {
    let hz = cycles_per_second();
    let f = frames.max(1) as f64;
    let take = |a: &[AtomicU64; STAGES]| -> Vec<u64> { a.iter().map(|x| x.swap(0, Ordering::Relaxed)).collect() };
    let (cycles, wall, calls, skipped) = (take(&CYCLES), take(&WALL_NS), take(&CALLS), take(&SKIPPED));
    let mut rows: Vec<usize> = (0..STAGES).filter(|&i| calls[i] != 0).collect();
    rows.sort_by(|a, b| cycles[*b].cmp(&cycles[*a]).then(a.cmp(b)));
    let ms = |c: u64| c as f64 / hz * 1000.0 / f;
    let by: Vec<String> = rows.iter().map(|&i| format!("{i} {:.2} ({:.1})", ms(cycles[i]), calls[i] as f64 / f)).collect();
    let skips: Vec<String> = (0..STAGES).filter(|&i| skipped[i] != 0).map(|i| format!("{i} x{:.1}", skipped[i] as f64 / f)).collect();
    // WALL1: the six longest by wall (a stage's wall above its CPU is a wait inside it); the rate for the frame wall line.
    NS_PER_CYCLE_Q20.store((1e9 / hz * (1u64 << 20) as f64) as u64, Ordering::Relaxed);
    format!(
        "title stages: {frames} frames, cpu ms a frame by stage (calls a frame): {}; all {:.2} cpu, {:.2} wall; wall ms a frame, longest: {}; skipped a frame: {}",
        if by.is_empty() { "none".to_owned() } else { by.join(", ") },
        ms(cycles.iter().sum()),
        wall.iter().sum::<u64>() as f64 / 1e6 / f,
        wall_top(&wall, &cycles, hz, f),
        if skips.is_empty() { "none".to_owned() } else { skips.join(", ") }
    )
}

/// The six stages longest by wall: `stage wall (cpu)` ms a frame; "none" without any.
fn wall_top(wall: &[u64], cycles: &[u64], hz: f64, f: f64) -> String {
    let mut rows: Vec<usize> = (0..wall.len()).filter(|&i| wall[i] != 0).collect();
    rows.sort_by(|a, b| wall[*b].cmp(&wall[*a]).then(a.cmp(b)));
    let top: Vec<String> = rows.iter().take(6).map(|&i| format!("{i} {:.2} ({:.2})", wall[i] as f64 / 1e6 / f, cycles[i] as f64 / hz * 1000.0 / f)).collect();
    if top.is_empty() { "none".to_owned() } else { top.join(", ") }
}

/// The per-callback line for the last `frames` frames: the stages that ran
/// through more than one callback, a row per callback; the sums start again.
fn take_callback_line(frames: u64) -> String {
    let (hz, f, base) = (cycles_per_second(), frames.max(1) as f64, BASE.load(Ordering::Relaxed));
    // (stage, callback, cycles, calls) of every callback that ran.
    let mut rows: Vec<(usize, usize, u64, u64)> = (0..CALLBACKS)
        .map(|i| (CB_STAGE[i].load(Ordering::Relaxed), CB_KEY[i].load(Ordering::Relaxed), CB_CYCLES[i].swap(0, Ordering::Relaxed), CB_CALLS[i].swap(0, Ordering::Relaxed)))
        .filter(|r| r.1 != 0 && r.3 != 0)
        .collect();
    let shared = |stage: usize, rows: &[(usize, usize, u64, u64)]| rows.iter().filter(|r| r.0 == stage).count() > 1;
    let all = rows.clone();
    rows.retain(|r| shared(r.0, &all));
    rows.sort_by(|a, b| a.0.cmp(&b.0).then(b.2.cmp(&a.2)).then(a.1.cmp(&b.1)));
    let by: Vec<String> = rows.iter().map(|&(stage, cb, c, n)| format!("{stage} title+{:#x} {:.2} ({:.1})", cb.wrapping_sub(base), c as f64 / hz * 1000.0 / f, n as f64 / f)).collect();
    format!("title stage callbacks: {frames} frames, stages with more than one callback, cpu ms a frame (calls a frame): {}", if by.is_empty() { "none".to_owned() } else { by.join(", ") })
}

#[cfg(test)]
mod tests {
    use super::*;
    use xnw_core::win::{MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc};

    static SEEN: AtomicU64 = AtomicU64::new(0);

    /// A stage callback: counts and keeps (phase, cbArg).
    extern "system" fn callback(phase: usize, arg: u32) -> usize {
        SEEN.fetch_add(1 + ((phase as u64) << 16) + ((arg as u64) << 32), Ordering::Relaxed);
        5
    }

    /// Two callbacks of one stage get a row each, a stage with one callback none.
    #[test]
    fn a_stage_with_two_callbacks_is_split_by_callback() {
        BASE.store(0x1000, Ordering::Relaxed);
        for (cb, stage, cycles, calls) in [(0x4413b0 + 0x1000, 36, 3_000u64, 2u64), (0xd3ac60 + 0x1000, 36, 1_000, 2), (0x11544b0 + 0x1000, 13, 500, 2)] {
            let i = callback_slot(cb, stage).unwrap();
            assert_eq!(callback_slot(cb, stage), Some(i), "the same slot again");
            CB_CYCLES[i].fetch_add(cycles, Ordering::Relaxed);
            CB_CALLS[i].fetch_add(calls, Ordering::Relaxed);
        }
        let line = take_callback_line(2);
        let rows = line.split_once("): ").unwrap().1;
        let names: Vec<&str> = rows.split(", ").map(|r| r.rsplitn(3, ' ').last().unwrap()).collect();
        assert_eq!(names, ["36 title+0x4413b0", "36 title+0xd3ac60"], "{line}");
        assert!(rows.contains("(1.0)") && !rows.contains("0x11544b0"), "{line}");
        assert!(take_callback_line(2).ends_with(": none"), "the sums start again");
    }

    /// The six longest stages by wall, each with its CPU beside it.
    #[test]
    fn the_longest_stages_by_wall_are_named() {
        let (mut wall, mut cycles) = (vec![0u64; STAGES], vec![0u64; STAGES]);
        for (stage, w_ms, c_ms) in [(69u64, 60, 36), (67, 20, 16), (73, 24, 12), (76, 10, 10), (1, 12, 10), (70, 8, 6), (74, 6, 6), (11, 2, 2)] {
            // 2 frames; 1000 cycles a millisecond.
            (wall[stage as usize], cycles[stage as usize]) = (w_ms * 1_000_000, c_ms * 1000);
        }
        assert_eq!(wall_top(&wall, &cycles, 1e6, 2.0), "69 30.00 (18.00), 73 12.00 (6.00), 67 10.00 (8.00), 1 6.00 (5.00), 76 5.00 (5.00), 70 4.00 (3.00)");
        assert_eq!(wall_top(&[0; 4], &[0; 4], 1e6, 2.0), "none");
    }

    #[test]
    fn skip_lists_parse() {
        assert_eq!(parse_skip("21, 65,"), (vec![21, 65], vec![]));
        assert_eq!(parse_skip(""), (vec![], vec![]));
        assert_eq!(parse_skip("21,x,128,0x15"), (vec![21], vec!["x".to_owned(), "128".to_owned(), "0x15".to_owned()]));
    }

    /// The title's 14 bytes, made in memory and detoured as at install: a
    /// timed stage runs its callback with the entry's arguments, a skipped
    /// one does not, and stage 64 closes the frame for the line.
    #[test]
    fn a_stage_is_timed_or_skipped_and_the_line_names_both() {
        // SAFETY: a fresh executable allocation of ours.
        let code = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) }.cast::<u8>();
        assert!(!code.is_null());
        // SAFETY: inside the allocation; int3 after the function, as in the title.
        unsafe {
            std::ptr::copy_nonoverlapping(ENTRY_BYTES.as_ptr(), code, ENTRY_BYTES.len());
            code.add(ENTRY_BYTES.len()).write_bytes(0xCC, 2);
        }
        // A ring entry: {view, cb, phase, cbArg, stage, seq, 0}.
        let entry = |stage: u32| -> [u64; 6] { [0, callback as usize as u64, 7, 9 | ((stage as u64) << 32), 0, 0] };
        // SAFETY: the function just written: the title's own.
        let run: extern "system" fn(usize) -> usize = unsafe { std::mem::transmute(code) };
        let e21 = entry(21);
        assert_eq!(run(e21.as_ptr() as usize), 5, "undetoured: cb(phase, cbArg)");
        assert_eq!(SEEN.swap(0, Ordering::Relaxed), 1 + (7 << 16) + (9 << 32));

        // SAFETY: our code, nothing runs it now.
        let tramp = unsafe { crate::detours::detour(code, &ENTRY_BYTES, hook as usize) }.unwrap();
        ORIGINAL.store(tramp, Ordering::Relaxed);
        set(true, &[65]);
        let (e65, e64) = (entry(65), entry(64));
        for _ in 0..3 {
            assert_eq!(run(e21.as_ptr() as usize), 5, "timed: the callback's result");
        }
        assert_eq!(run(e65.as_ptr() as usize), 0, "skipped: not called");
        assert_eq!(run(e64.as_ptr() as usize), 5);
        assert_eq!(SEEN.swap(0, Ordering::Relaxed), 4 * (1 + (7 << 16) + (9 << 32)), "three of stage 21 and one of 64; none of 65");
        assert_eq!(FRAMES.load(Ordering::Relaxed), 1, "stage 64 ends a frame");
        let line = take_line(1);
        assert!(line.starts_with("title stages: 1 frames, cpu ms a frame by stage (calls a frame): "), "{line}");
        assert!(line.contains("21 ") && line.contains(" (3.0)") && line.contains("64 ") && line.contains(" (1.0)"), "{line}");
        assert!(line.ends_with("skipped a frame: 65 x1.0"), "{line}");
        assert!(take_line(1).contains("by stage (calls a frame): none; all 0.00 cpu"), "the sums start again");
        set(false, &[]);
        assert_eq!(run(e65.as_ptr() as usize), 5, "off: every callback runs");
        SEEN.store(0, Ordering::Relaxed);
    }
}
