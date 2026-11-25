//! The title's blocking waits, seen. Kernel waits and sleeps the
//! title imports (`WaitForSingleObject(Ex)`, `WaitForMultipleObjects(Ex)`,
//! `Sleep(Ex)`, `SleepConditionVariableSRW/CS`, `WaitOnAddress`) are bound
//! to these wrappers, which call the host's at once. With
//! `XNW_WAIT_STATS=1` each is timed by the title address that called it,
//! and every 10 s a line names the five that blocked longest (calls, total,
//! longest, the timeout asked): what a null-GPU frame of the title waits on
//! when nothing of our GPU path is in it (a test run's 100 ms frames).
//!
//! (The timer is already at 1 ms: `audio::use_console_timers`, a test run's
//! "timers: resolution 0.997 ms".)
//!
//! Cost unset: one call level and one cached test per wait.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

type H = *mut c_void;

fn on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_WAIT_STATS").is_ok_and(|v| v.trim() == "1"));
    *ON
}

/// The functions timed, by index.
const NAMES: [&str; 9] = ["WaitForSingleObject", "WaitForSingleObjectEx", "WaitForMultipleObjects", "WaitForMultipleObjectsEx", "Sleep", "SleepEx", "SleepConditionVariableSRW", "SleepConditionVariableCS", "WaitOnAddress"];

/// (function, caller, timeout asked) -> calls, microseconds, longest,
/// timeout (by timeout too, so a 1 ms poll is not averaged with the
/// same caller's 2 s waits).
type Stats = HashMap<(usize, usize, u32), (u64, u64, u64, u32)>;
static STATS: Mutex<Option<Stats>> = Mutex::new(None);
static LAST_LINE: AtomicU64 = AtomicU64::new(0);

/// The title address that called our wrapper.
#[inline(never)]
fn caller() -> usize {
    let mut frames = [std::ptr::null_mut(); 1];
    // SAFETY: room for one frame; frame 0 is this function, 1 the wrapper,
    // 2 its caller.
    unsafe { xnw_core::win::RtlCaptureStackBackTrace(2, 1, frames.as_mut_ptr(), std::ptr::null_mut()) };
    frames[0] as usize
}

/// The switches, read at start-up (`read_shim_switches`).
pub(crate) fn read_switches() {
    let _ = (on(), detail(), cs_cap());
}

fn cs_cap() -> Option<u32> {
    static CAP: std::sync::LazyLock<Option<u32>> = std::sync::LazyLock::new(|| std::env::var("XNW_CS_SPIN").ok().and_then(|v| v.trim().parse().ok()));
    *CAP
}

/// `XNW_WAIT_DETAIL=<rva,rva>[,from=<s>,to=<s>]`: title callers whose
/// waits are logged one by one (within the time range, at most 200
/// lines per caller per 10 s).
fn detail_rvas() -> &'static [usize] {
    &detail().0
}

fn detail() -> &'static (Vec<usize>, u64, u64) {
    static D: std::sync::LazyLock<(Vec<usize>, u64, u64)> = std::sync::LazyLock::new(|| {
        let v = std::env::var("XNW_WAIT_DETAIL").unwrap_or_default();
        parse_detail(&v)
    });
    &D
}

/// `<rva,rva>[,from=<s>,to=<s>]`.
pub(crate) fn parse_detail(v: &str) -> (Vec<usize>, u64, u64) {
    let (mut rvas, mut from, mut to) = (Vec::new(), 0, u64::MAX);
    for s in v.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if let Some(x) = s.strip_prefix("from=") {
            from = x.parse().unwrap_or(0);
        } else if let Some(x) = s.strip_prefix("to=") {
            to = x.parse().unwrap_or(u64::MAX);
        } else if let Ok(r) = usize::from_str_radix(s.trim_start_matches("0x"), 16) {
            rvas.push(r);
        }
    }
    (rvas, from, to)
}

/// Times `f` as function `which` called from `at` with `timeout` asked;
/// `handles` (for wait details) are the objects waited on.
#[inline(always)]
fn timed<T: Copy + std::fmt::Debug>(which: usize, timeout: u32, handles: &[H], f: impl FnOnce() -> T) -> T {
    let detail = !detail_rvas().is_empty();
    if !on() && !detail {
        return f();
    }
    // The caller walk may set last error: the title's own value goes to the host call.
    let at = {
        let _keep = crate::threads::KeepError::now();
        caller()
    };
    let started = std::time::Instant::now();
    let r = f();
    let _keep = crate::threads::KeepError::now();
    let us = started.elapsed().as_micros() as u64;
    if on() {
        note(which, at, timeout, us);
    }
    if detail {
        let base = crate::title_base();
        let rva = at.wrapping_sub(base);
        if let Some(k) = detail_rvas().iter().position(|&d| d == rva) {
            detail_line(k, which, rva, handles, timeout, r, us);
        }
    }
    r
}

/// One wait of a chosen caller: handles and their kinds, result, thread,
/// duration.
fn detail_line<T: std::fmt::Debug>(k: usize, which: usize, rva: usize, handles: &[H], timeout: u32, r: T, us: u64) {
    // Per caller: the 10 s window (number) and lines in it.
    static COUNTS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
    let now = xnw_core::uptime().as_secs();
    let (_, from, to) = detail();
    if k >= 8 || now < *from || now > *to {
        return;
    }
    let win = now / 10;
    let c = COUNTS[k].load(Ordering::Relaxed);
    if c >> 16 != win {
        COUNTS[k].store(win << 16, Ordering::Relaxed);
    }
    if COUNTS[k].fetch_add(1, Ordering::Relaxed) & 0xffff >= 200 {
        return;
    }
    let kinds: Vec<String> = handles.iter().take(4).map(|&h| format!("{h:p} {}", object_kind(h))).collect();
    // SAFETY: plain queries.
    let tid = unsafe { xnw_core::win::GetCurrentThreadId() };
    xnw_core::info!(
        "waits: {} from title+{rva:#x} on thread {tid}: [{}]{}, timeout {}, returned {r:?} after {:.3} ms (our HLE makes no title events: creator unknown)",
        NAMES[which],
        kinds.join(", "),
        if handles.len() > 4 { format!(" and {} more", handles.len() - 4) } else { String::new() },
        if timeout == u32::MAX { "INFINITE".to_string() } else { format!("{timeout} ms") },
        us as f64 / 1000.0
    );
}

/// A handle's object kind (Event, Semaphore, Thread, ...).
fn object_kind(h: H) -> String {
    let mut buf = [0u64; 64];
    // SAFETY: a query into our buffer; its start is a UNICODE_STRING whose
    // characters follow inside it.
    unsafe {
        if xnw_core::win::NtQueryObject(h, 2, buf.as_mut_ptr().cast(), 512, std::ptr::null_mut()) < 0 {
            return "?".into();
        }
        let len = (buf[0] & 0xffff) as usize / 2;
        let p = buf[1] as *const u16;
        if p.is_null() || len > 200 {
            return "?".into();
        }
        String::from_utf16_lossy(std::slice::from_raw_parts(p, len))
    }
}

fn note(which: usize, at: usize, timeout: u32, us: u64) {
    {
        let mut g = STATS.lock().unwrap_or_else(|e| e.into_inner());
        let e = g.get_or_insert_with(HashMap::new).entry((which, at, timeout)).or_insert((0, 0, 0, 0));
        e.0 += 1;
        e.1 += us;
        e.2 = e.2.max(us);
        e.3 = timeout;
    }
    let now = xnw_core::uptime().as_secs();
    let last = LAST_LINE.load(Ordering::Relaxed);
    if now >= last + 10 && LAST_LINE.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
        let taken = STATS.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default();
        xnw_core::info!("waits: last 10 s, title threads blocked longest: {}", top_text(&taken, 5));
    }
}

/// The `n` waits that blocked longest, as text.
fn top_text(stats: &Stats, n: usize) -> String {
    let mut v: Vec<(&(usize, usize, u32), &(u64, u64, u64, u32))> = stats.iter().collect();
    v.sort_by(|a, b| b.1.1.cmp(&a.1.1));
    let base = crate::title_base();
    v.iter()
        .take(n)
        .map(|((which, at, _), (calls, us, max, timeout))| {
            let at = if base != 0 && *at > base { format!("title+{:#x}", at - base) } else { format!("{at:#x}") };
            let timeout = if *timeout == u32::MAX { "INFINITE".to_string() } else { format!("{timeout} ms") };
            format!("{} from {at}: {calls} calls, {:.3} s, longest {:.1} ms (timeout {timeout})", NAMES[*which], *us as f64 / 1e6, *max as f64 / 1000.0)
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// The handles of a multiple-object wait.
fn handles<'a>(n: u32, hs: *const H) -> &'a [H] {
    if hs.is_null() || n == 0 || n > 64 {
        return &[];
    }
    // SAFETY: the title passes `n` handles.
    unsafe { std::slice::from_raw_parts(hs, n as usize) }
}

/// The title's critical-section spin counts. It imports
/// `InitializeCriticalSectionAndSpinCount` (and `InitializeCriticalSection`,
/// no spin); each distinct (caller, spin) is logged once. `XNW_CS_SPIN=<n>`
/// caps the spin count (a console with 8 free cores spins cheaply; a
/// squeezed host does not).
pub extern "system" fn InitializeCriticalSectionAndSpinCount(cs: H, spin: u32) -> i32 {
    static SEEN: std::sync::Mutex<Vec<(usize, u32)>> = std::sync::Mutex::new(Vec::new());
    let cap = cs_cap();
    let applied = cap.map_or(spin, |c| spin.min(c));
    let at = caller().wrapping_sub(crate::title_base());
    {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if seen.len() < 128 && !seen.contains(&(at, spin)) {
            seen.push((at, spin));
            xnw_core::info!("waits: InitializeCriticalSectionAndSpinCount from title+{at:#x}: spin {spin:#x}, applied {applied:#x}{}", if cap.is_some() { " (XNW_CS_SPIN)" } else { "" });
        }
    }
    host!("kernel32.dll", "InitializeCriticalSectionAndSpinCount": extern "system" fn(H, u32) -> i32)(cs, applied)
}

pub extern "system" fn WaitForSingleObject(h: H, ms: u32) -> u32 {
    timed(0, ms, &[h], || host!("kernel32.dll", "WaitForSingleObject": extern "system" fn(H, u32) -> u32)(h, ms))
}
pub extern "system" fn WaitForSingleObjectEx(h: H, ms: u32, alertable: i32) -> u32 {
    timed(1, ms, &[h], || host!("kernel32.dll", "WaitForSingleObjectEx": extern "system" fn(H, u32, i32) -> u32)(h, ms, alertable))
}
pub extern "system" fn WaitForMultipleObjects(n: u32, hs: *const H, all: i32, ms: u32) -> u32 {
    timed(2, ms, handles(n, hs), || host!("kernel32.dll", "WaitForMultipleObjects": extern "system" fn(u32, *const H, i32, u32) -> u32)(n, hs, all, ms))
}
pub extern "system" fn WaitForMultipleObjectsEx(n: u32, hs: *const H, all: i32, ms: u32, alertable: i32) -> u32 {
    timed(3, ms, handles(n, hs), || host!("kernel32.dll", "WaitForMultipleObjectsEx": extern "system" fn(u32, *const H, i32, u32, i32) -> u32)(n, hs, all, ms, alertable))
}
pub extern "system" fn Sleep(ms: u32) {
    timed(4, ms, &[], || host!("kernel32.dll", "Sleep": extern "system" fn(u32))(ms))
}
pub extern "system" fn SleepEx(ms: u32, alertable: i32) -> u32 {
    timed(5, ms, &[], || host!("kernel32.dll", "SleepEx": extern "system" fn(u32, i32) -> u32)(ms, alertable))
}
pub extern "system" fn SleepConditionVariableSRW(cv: H, lock: H, ms: u32, flags: u32) -> i32 {
    timed(6, ms, &[], || host!("kernel32.dll", "SleepConditionVariableSRW": extern "system" fn(H, H, u32, u32) -> i32)(cv, lock, ms, flags))
}
pub extern "system" fn SleepConditionVariableCS(cv: H, cs: H, ms: u32) -> i32 {
    timed(7, ms, &[], || host!("kernel32.dll", "SleepConditionVariableCS": extern "system" fn(H, H, u32) -> i32)(cv, cs, ms))
}
pub extern "system" fn WaitOnAddress(address: *const c_void, compare: *const c_void, size: usize, ms: u32) -> i32 {
    timed(8, ms, &[], || host!("kernelbase.dll", "WaitOnAddress": extern "system" fn(*const c_void, *const c_void, usize, u32) -> i32)(address, compare, size, ms))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The longest blocked waits come first, by title address, with their
    /// counts, totals, longest and timeout; calls go through to the host.
    #[test]
    fn the_longest_waits_are_named() {
        let mut s = Stats::new();
        s.insert((4, 0x10, 1), (100, 1_500_000, 16_000, 1));
        s.insert((0, 0x20, 100), (3, 300_000, 100_000, 100));
        s.insert((8, 0x30, u32::MAX), (9, 9, 1, u32::MAX));
        let text = top_text(&s, 2);
        assert!(text.starts_with("Sleep from 0x10: 100 calls, 1.500 s, longest 16.0 ms (timeout 1 ms); WaitForSingleObject from 0x20"), "{text}");
        assert!(!text.contains("WaitOnAddress"));
        assert_eq!(parse_detail("43c421e, 0x22511f ,from=60,to=212"), (vec![0x43c421e, 0x22511f], 60, 212));
        assert_eq!(parse_detail(""), (vec![], 0, u64::MAX));
        // A real call through: a zero sleep returns.
        Sleep(0);
        assert_eq!(SleepEx(0, 0), 0);
    }
}
