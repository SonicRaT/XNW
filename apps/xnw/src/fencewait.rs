//! The title's waits for the GPU, metered and answered by an event (ENG5).
//!
//! Every wait of the title for submitted work is one function,
//! `title+0x4fb0ad0(fence, token, predicate, spin)`: it reads the
//! fence's value through the pointer of fence slot 12 and loops until the
//! predicate holds (default `title+0x10a330`: `((value - token) >> 55) & 1
//! == 0`). With `spin` set (every caller found: the frame pacing on the
//! present token of frame N-2 `title+0x24747aa`, the flush-and-wait of a
//! synchronous readback `title+0x979738`, flush-all `title+0x5eb2a00`) the
//! loop is `monitorx; check; mwaitx; check; Sleep(0)` (`title+0x4fb0bd0` to
//! `0x4fb0c2f`). The console parks the core in `mwaitx`; on a CPU without
//! it our loader makes that a `pause` (xnw-loader `rewrite.rs`), so the
//! wait is a `Sleep(0)` spin: it costs its whole length in CPU on the
//! waiting thread, and a stage that waits shows the wait as CPU time.
//!
//! The value is written by the queue's `Signal`, which the title's submit
//! thread (`title+0x2481e90`) calls after each batch it has executed, so
//! under the null GPU a wait is the submit thread's backlog: lists, the
//! frame-origin wait (our pacing) and the present.
//!
//! `XNW_FENCE_WAIT=time`: a detour on the wait times each one (wall, CPU)
//! by the site that asked, one `fence waits:` line per 60 frames (a frame
//! is one pacing wait). `XNW_FENCE_WAIT=block`: the same, and a wait with
//! the default predicate that is not met after a short spin blocks in the
//! fence's own `SetEventOnCompletion(value, null)` (slot 9: returns once
//! the value is reached), woken by the signal; the title's function then
//! finds its predicate true. Off by default.

use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use xnw_core::win::RtlCaptureStackBackTrace;

/// `title+rva` of the wait and its first whole instructions (push r15; push
/// r14; push r13; push r12; push rsi; push rdi; push rbp; push rbx; sub
/// rsp, 58h), read from the exe.
pub(crate) const WAIT_RVA: usize = 0x4fb0ad0;
pub(crate) const WAIT_BYTES: [u8; 16] = [0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x56, 0x57, 0x55, 0x53, 0x48, 0x83, 0xEC, 0x58];
/// The default predicate and its code (sub rcx, rdx; bt rcx, 37h; setae al; ret).
const PREDICATE_RVA: usize = 0x10a330;
const PREDICATE_BYTES: [u8; 12] = [0x48, 0x29, 0xD1, 0x48, 0x0F, 0xBA, 0xE1, 0x37, 0x0F, 0x93, 0xC0, 0xC3];
/// Where the frame pacing's wait returns to (`call` at `title+0x24747aa`).
const PACING_RETURN: usize = 0x24747af;
/// Code ranges a wait passes through before the site that asked: the wait
/// functions, the flush-and-wait `title+0x9796d0` and the readback stamp
/// `title+0x53f1950` (both end in a jump, so they leave no frame anyway).
const INNER: [(usize, usize); 3] = [(0x4fb0910, 0x4fb0e90), (0x9796d0, 0x97974f), (0x53f1950, 0x53f1a3c)];
/// Sites by name (return addresses; flush-all is a range of eleven calls).
const NAMED: [(usize, usize, &str); 3] = [(PACING_RETURN, PACING_RETURN + 1, "pacing"), (0x5eb2a00, 0x5eb2be0, "flush all"), (0x23cf363, 0x23cf364, "render sync")];

/// Frames per line.
const EVERY: u64 = 60;
/// How long a blocking wait spins first: a wake costs about this much.
const SPIN: Duration = Duration::from_micros(50);
const SITES: usize = 24;

const OFF: u8 = 0;
const TIME: u8 = 1;
const BLOCK: u8 = 2;

/// The trampoline of the detoured wait (set by `detours.rs`).
pub(crate) static ORIGINAL: AtomicUsize = AtomicUsize::new(0);
static MODE: AtomicU8 = AtomicU8::new(OFF);
static BASE: AtomicUsize = AtomicUsize::new(0);
/// The default predicate's address in this process.
static PREDICATE: AtomicUsize = AtomicUsize::new(0);
/// Per site (`title+rva` of the return address + 1; 0 = free; the last slot takes the overflow).
static SITE: [AtomicUsize; SITES] = [const { AtomicUsize::new(0) }; SITES];
static WAITS: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];
static UNMET: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];
static WALL_NS: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];
static CYCLES: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];
static LONGEST_NS: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];
static BLOCKED: AtomicU64 = AtomicU64::new(0);
static FRAMES: AtomicU64 = AtomicU64::new(0);

/// The title's default predicate: the value has reached the token (56-bit wrap).
fn met(value: u64, token: u64) -> bool {
    value.wrapping_sub(token) & (1 << 55) == 0
}

fn parse_mode(v: &str) -> Option<u8> {
    match v.trim() {
        "" | "0" | "off" => Some(OFF),
        "time" => Some(TIME),
        "block" => Some(BLOCK),
        _ => None,
    }
}

/// Reads `XNW_FENCE_WAIT`; whether the detour is wanted.
///
/// # Safety
/// `base` is the mapped title of build 1.0.2.1.
pub(crate) unsafe fn configure(base: usize) -> bool {
    let v = std::env::var("XNW_FENCE_WAIT").unwrap_or_default();
    let Some(mode) = parse_mode(&v) else {
        xnw_core::warn!("XNW_FENCE_WAIT={}: not time or block; the fence waits are left alone", v.trim());
        return false;
    };
    if mode == OFF {
        return false;
    }
    // SAFETY: the caller's mapped code.
    if unsafe { std::slice::from_raw_parts((base + PREDICATE_RVA) as *const u8, PREDICATE_BYTES.len()) } != PREDICATE_BYTES {
        xnw_core::warn!("fence waits: the predicate at title+{PREDICATE_RVA:#x} is not the expected code; XNW_FENCE_WAIT ignored");
        return false;
    }
    BASE.store(base, Ordering::Relaxed);
    PREDICATE.store(base + PREDICATE_RVA, Ordering::Relaxed);
    MODE.store(mode, Ordering::Relaxed);
    crate::stagemeter::cycles_per_second();
    xnw_core::info!(
        "fence waits: {} (title+{WAIT_RVA:#x}); one line per {EVERY} frames",
        if mode == BLOCK { "timed by site, and a wait not met at once blocks until the fence's signal instead of spinning" } else { "timed by site" }
    );
    true
}

/// The slot of the site a wait came from: the first return address above
/// the wait's own functions, as `title+rva` (0 outside the title).
fn site_slot(rva: usize) -> usize {
    let key = rva + 1;
    for (i, s) in SITE.iter().enumerate().take(SITES - 1) {
        match s.compare_exchange(0, key, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return i,
            Err(k) if k == key => return i,
            Err(_) => {}
        }
    }
    SITES - 1
}

/// The asking site among the return addresses above the hook, as an rva.
fn asking(frames: &[usize], base: usize, size: usize) -> usize {
    frames.iter().filter_map(|&f| f.checked_sub(base).filter(|&r| r < size)).find(|r| !INNER.iter().any(|&(lo, hi)| (lo..hi).contains(r))).unwrap_or(0)
}

/// The detour of `title+0x4fb0ad0`: the fence, the token (56 bits), the
/// predicate record `{context, function}` and the spin flag.
pub(crate) extern "system" fn hook(fence: usize, token: u64, predicate: usize, spin: usize) -> usize {
    // SAFETY: the trampoline made at install: the function's own entry.
    let original: extern "system" fn(usize, u64, usize, usize) -> usize = unsafe { std::mem::transmute(ORIGINAL.load(Ordering::Relaxed)) };
    let mode = MODE.load(Ordering::Relaxed);
    if mode == OFF {
        return original(fence, token, predicate, spin);
    }
    let mut frames = [std::ptr::null_mut(); 4];
    // SAFETY: room for four frames above this one.
    let n = unsafe { RtlCaptureStackBackTrace(1, frames.len() as u32, frames.as_mut_ptr(), std::ptr::null_mut()) } as usize;
    let rva = asking(&frames.map(|f| f as usize)[..n], BASE.load(Ordering::Relaxed), 0x1000_0000);
    let slot = site_slot(rva);
    // SAFETY: the title's own first step: fence slot 12 gives where the value lives; the record is two pointers.
    let (value, default) = unsafe {
        let table = (fence as *const *const usize).read();
        let address: extern "system" fn(usize) -> *const AtomicU64 = std::mem::transmute(table.add(12).read());
        let (context, function) = ((predicate as *const usize).read(), ((predicate + 8) as *const usize).read());
        (&*address(fence), context == 0 && function == PREDICATE.load(Ordering::Relaxed))
    };
    let (c0, t0) = (crate::stagemeter::thread_cycles(), Instant::now());
    let done = |v: u64| v == u64::MAX || met(v, token);
    let unmet = default && !done(value.load(Ordering::Acquire));
    if unmet && mode == BLOCK {
        while !done(value.load(Ordering::Acquire)) && t0.elapsed() < SPIN {
            for _ in 0..32 {
                std::hint::spin_loop();
            }
        }
        if !done(value.load(Ordering::Acquire)) {
            BLOCKED.fetch_add(1, Ordering::Relaxed);
            // SAFETY: fence slot 9, SetEventOnCompletion(value, event): a null event returns once the value is reached.
            unsafe {
                let table = (fence as *const *const usize).read();
                let wait: extern "system" fn(usize, u64, usize) -> i32 = std::mem::transmute(table.add(9).read());
                wait(fence, token, 0);
            }
        }
    }
    let result = original(fence, token, predicate, spin);
    let ns = t0.elapsed().as_nanos() as u64;
    WAITS[slot].fetch_add(1, Ordering::Relaxed);
    UNMET[slot].fetch_add(u64::from(unmet), Ordering::Relaxed);
    WALL_NS[slot].fetch_add(ns, Ordering::Relaxed);
    LONGEST_NS[slot].fetch_max(ns, Ordering::Relaxed);
    CYCLES[slot].fetch_add(crate::stagemeter::thread_cycles().wrapping_sub(c0), Ordering::Relaxed);
    if rva == PACING_RETURN && FRAMES.fetch_add(1, Ordering::Relaxed) % EVERY == EVERY - 1 {
        xnw_core::info!("{}", take_line(EVERY));
    }
    result
}

fn site_name(rva: usize) -> String {
    match NAMED.iter().find(|&&(lo, hi, _)| (lo..hi).contains(&rva)) {
        Some(&(_, _, name)) => format!("{name} title+{rva:#x}"),
        None if rva == 0 => "elsewhere".to_owned(),
        None => format!("title+{rva:#x}"),
    }
}

/// The line for the last `frames` frames; the sums start again (the sites stay).
fn take_line(frames: u64) -> String {
    let (hz, f) = (crate::stagemeter::cycles_per_second(), frames.max(1) as f64);
    let mut rows: Vec<(u64, String)> = Vec::new();
    let (mut wall, mut cpu) = (0u64, 0u64);
    for i in 0..SITES {
        let (n, unmet, w, c, longest) = (WAITS[i].swap(0, Ordering::Relaxed), UNMET[i].swap(0, Ordering::Relaxed), WALL_NS[i].swap(0, Ordering::Relaxed), CYCLES[i].swap(0, Ordering::Relaxed), LONGEST_NS[i].swap(0, Ordering::Relaxed));
        if n == 0 {
            continue;
        }
        let name = if i == SITES - 1 { "other sites".to_owned() } else { site_name(SITE[i].load(Ordering::Relaxed).saturating_sub(1)) };
        wall += w;
        cpu += c;
        rows.push((w, format!("{name} {:.2}/{:.2} ({:.1}, {:.1} not met, longest {:.2})", w as f64 / 1e6 / f, c as f64 / hz * 1000.0 / f, n as f64 / f, unmet as f64 / f, longest as f64 / 1e6)));
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    format!(
        "fence waits: {frames} frames, ms a frame wall/cpu by site (waits a frame, not met at entry, longest ms): {}; all {:.2} wall, {:.2} cpu; blocked {:.1} a frame",
        if rows.is_empty() { "none".to_owned() } else { rows.into_iter().map(|r| r.1).collect::<Vec<_>>().join(", ") },
        wall as f64 / 1e6 / f,
        cpu as f64 / hz * 1000.0 / f,
        BLOCKED.swap(0, Ordering::Relaxed) as f64 / f
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use xnw_core::win::{MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc};

    /// What the title's fence looks like to the wait: a table whose slot 12
    /// gives the value's address and whose slot 9 returns once it is reached.
    #[repr(C)]
    struct Fence {
        table: *const [usize; 13],
        value: AtomicU64,
        blocked: AtomicU64,
    }

    extern "system" fn address(this: usize) -> *const AtomicU64 {
        // SAFETY: the test's fence.
        unsafe { &(*(this as *const Fence)).value }
    }

    extern "system" fn until(this: usize, value: u64, event: usize) -> i32 {
        assert_eq!(event, 0, "a null event: block");
        // SAFETY: the test's fence.
        let fence = unsafe { &*(this as *const Fence) };
        fence.blocked.fetch_add(1, Ordering::Relaxed);
        while fence.value.load(Ordering::Acquire) < value {
            std::thread::sleep(Duration::from_millis(1));
        }
        0
    }

    fn table() -> [usize; 13] {
        let mut t = [0usize; 13];
        t[9] = until as usize;
        t[12] = address as usize;
        t
    }
    /// What the stand-in for the title's wait saw: calls << 32 | calls that found the default predicate met.
    static SEEN: AtomicU64 = AtomicU64::new(0);

    extern "system" fn title_wait(fence: usize, token: u64, _predicate: usize, _spin: usize) -> usize {
        let v = address(fence);
        // SAFETY: the test's fence value.
        let now = unsafe { &*v }.load(Ordering::Acquire);
        SEEN.fetch_add((1 << 32) + u64::from(met(now, token)), Ordering::Relaxed);
        7
    }

    #[test]
    fn the_predicate_is_the_titles() {
        assert!(met(5, 5) && met(6, 5) && !met(4, 5));
        assert!(!met(u64::MAX, 5), "a removed device (-1) is not met: the title's function handles it");
        assert!(met(2, (1 << 56) - 3) && !met((1 << 56) - 3, 2), "56-bit wrap, as bt rcx, 37h reads it");
        assert_eq!(parse_mode(" block "), Some(BLOCK));
        assert_eq!(parse_mode("time"), Some(TIME));
        assert_eq!(parse_mode(""), Some(OFF));
        assert_eq!(parse_mode("1"), None);
    }

    #[test]
    fn the_asking_site_is_the_first_frame_outside_the_wait() {
        let base = 0x7ff6_0000_0000usize;
        // The pacing: the wait's own return (in 0x4fb0a90), then the caller of 0x4fb0a50.
        assert_eq!(asking(&[base + 0x4fb0ac4, base + PACING_RETURN, base + 0xc9e603], base, 0x1000_0000), PACING_RETURN);
        // A readback: the stamp and the flush-and-wait jump, so its caller is next.
        assert_eq!(asking(&[base + 0x4fb0ac4, base + 0x328f391], base, 0x1000_0000), 0x328f391);
        assert_eq!(asking(&[0x1234, base + 0x4fb0ac4], base, 0x1000_0000), 0, "nothing of the title outside the wait");
        assert_eq!(site_name(PACING_RETURN), "pacing title+0x24747af");
        assert_eq!(site_name(0x5eb2a2e), "flush all title+0x5eb2a2e");
        assert_eq!(site_name(0x328f391), "title+0x328f391");
    }

    /// The byte tables and the call sites named here are the executable's.
    #[test]
    fn the_wait_its_predicate_and_its_callers_are_the_executables() {
        let Some(exe) = crate::patches::Exe::open() else {
            eprintln!("no title executable here (XNW_TITLE_EXE): the fence wait's bytes were not checked against it");
            return;
        };
        assert_eq!(exe.at(WAIT_RVA, WAIT_BYTES.len()), WAIT_BYTES);
        assert_eq!(exe.at(PREDICATE_RVA, PREDICATE_BYTES.len()), PREDICATE_BYTES);
        // The one call of the wait (in 0x4fb0a90, reached by a jump from 0x4fb0a50), and its return address.
        assert_eq!((exe.at(0x4fb0abf, 1)[0], exe.rel32_target(0x4fb0abf)), (0xE8, WAIT_RVA));
        assert_eq!((exe.at(0x4fb0a79, 1)[0], exe.rel32_target(0x4fb0a79)), (0xE9, 0x4fb0a90));
        // The pacing: mov dl, 1; call 0x4fb0a50, returning to PACING_RETURN.
        assert_eq!(exe.at(PACING_RETURN - 7, 3), [0xB2, 0x01, 0xE8]);
        assert_eq!(exe.rel32_target(PACING_RETURN - 5), 0x4fb0a50);
        // The flush-and-wait ends in mov dl, 1; ...; jmp 0x4fb0a50, and the readback stamp in a jump to it.
        assert_eq!(exe.at(0x979738, 2), [0xB2, 0x01]);
        assert_eq!((exe.at(0x979741, 1)[0], exe.rel32_target(0x979741)), (0xE9, 0x4fb0a50));
        assert_eq!((exe.at(0x53f1a37, 1)[0], exe.rel32_target(0x53f1a37)), (0xE9, 0x9796d0));
        // The spin loop: mwaitx at 0x4fb0c00, then call Sleep(0)'s wrapper, then back to the monitorx.
        assert_eq!(exe.at(0x4fb0c00, 3), [0x0F, 0x01, 0xFB]);
        assert_eq!(exe.rel32_target(0x4fb0c22), 0x4fb0e90);
        assert_eq!(exe.at(0x4fb0c2f, 2), [0xEB, 0x9F]);
    }

    /// The hook over a stand-in fence and a stand-in for the title's
    /// function: timed only, the title's function is called with the wait
    /// unmet; blocking, it is called once the value is there.
    #[test]
    fn a_wait_is_timed_and_in_block_mode_met_before_the_titles_function_runs() {
        let table = table();
        let fence = Fence { table: &table, value: AtomicU64::new(3), blocked: AtomicU64::new(0) };
        let this = std::ptr::from_ref(&fence) as usize;
        let marker = 0x1234_5678usize;
        PREDICATE.store(marker, Ordering::Relaxed);
        ORIGINAL.store(title_wait as usize, Ordering::Relaxed);
        let default: [usize; 4] = [0, marker, 0, 0];
        let other: [usize; 4] = [0, marker + 1, 0, 0];

        MODE.store(TIME, Ordering::Relaxed);
        assert_eq!(hook(this, 5, default.as_ptr() as usize, 1), 7);
        assert_eq!(SEEN.swap(0, Ordering::Relaxed), 1 << 32, "timed only: the title's function does the waiting");
        assert_eq!(fence.blocked.load(Ordering::Relaxed), 0);

        MODE.store(BLOCK, Ordering::Relaxed);
        let value = &fence.value;
        std::thread::scope(|s| {
            s.spawn(move || {
                std::thread::sleep(Duration::from_millis(20));
                value.store(5, Ordering::Release);
            });
            let t = Instant::now();
            assert_eq!(hook(this, 5, default.as_ptr() as usize, 1), 7);
            assert!(t.elapsed() >= Duration::from_millis(15), "it waited for the signal");
        });
        assert_eq!(SEEN.swap(0, Ordering::Relaxed), (1 << 32) + 1, "the title's function found the value there");
        assert_eq!(fence.blocked.load(Ordering::Relaxed), 1, "through the fence's own blocking wait");
        assert_eq!(hook(this, 5, default.as_ptr() as usize, 1), 7);
        assert_eq!(fence.blocked.load(Ordering::Relaxed), 1, "met at entry: no block");
        assert_eq!(hook(this, 9, other.as_ptr() as usize, 1), 7);
        assert_eq!(fence.blocked.load(Ordering::Relaxed), 1, "another predicate is the title's business");
        fence.value.store(u64::MAX, Ordering::Release);
        assert_eq!(hook(this, 9, default.as_ptr() as usize, 1), 7);
        assert_eq!(fence.blocked.load(Ordering::Relaxed), 1, "a removed device goes straight to the title's function");
        SEEN.store(0, Ordering::Relaxed);

        let line = take_line(1);
        assert!(line.starts_with("fence waits: 1 frames, ms a frame wall/cpu by site (waits a frame, not met at entry, longest ms): elsewhere "), "{line}");
        assert!(line.contains("(5.0, 2.0 not met, longest ") && line.ends_with("blocked 1.0 a frame"), "{line}");
        assert!(take_line(1).contains("by site (waits a frame, not met at entry, longest ms): none; all 0.00 wall"), "the sums start again");
        MODE.store(OFF, Ordering::Relaxed);
    }

    /// The title's 16 entry bytes, made in memory with a body and detoured
    /// as at install: the hook runs and the original still returns its result.
    #[test]
    fn the_entry_bytes_move_to_a_trampoline() {
        // prologue; mov rax, rdx; add rsp, 58h; pop rbx; pop rbp; pop rdi; pop rsi; pop r12; pop r13; pop r14; pop r15; ret
        let body: [u8; 20] = [0x48, 0x89, 0xD0, 0x48, 0x83, 0xC4, 0x58, 0x5B, 0x5D, 0x5F, 0x5E, 0x41, 0x5C, 0x41, 0x5D, 0x41, 0x5E, 0x41, 0x5F, 0xC3];
        // SAFETY: a fresh executable allocation of ours.
        let code = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) }.cast::<u8>();
        assert!(!code.is_null());
        // SAFETY: inside the allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(WAIT_BYTES.as_ptr(), code, WAIT_BYTES.len());
            std::ptr::copy_nonoverlapping(body.as_ptr(), code.add(WAIT_BYTES.len()), body.len());
        }
        static COUNT: AtomicU64 = AtomicU64::new(0);
        static TRAMP: AtomicUsize = AtomicUsize::new(0);
        extern "system" fn counted(a: usize, b: u64, c: usize, d: usize) -> usize {
            COUNT.fetch_add(1, Ordering::Relaxed);
            // SAFETY: the trampoline of the function below, of this type.
            let original: extern "system" fn(usize, u64, usize, usize) -> usize = unsafe { std::mem::transmute(TRAMP.load(Ordering::Relaxed)) };
            original(a, b, c, d)
        }
        // SAFETY: the function just written.
        let f: extern "system" fn(usize, u64, usize, usize) -> usize = unsafe { std::mem::transmute(code) };
        assert_eq!(f(1, 42, 3, 4), 42);
        // SAFETY: our code, nothing runs it now.
        TRAMP.store(unsafe { crate::detours::detour(code, &WAIT_BYTES, counted as usize) }.unwrap(), Ordering::Relaxed);
        assert_eq!(f(1, 43, 3, 4), 43, "through the hook and the trampoline");
        assert_eq!(COUNT.load(Ordering::Relaxed), 1);
    }
}
