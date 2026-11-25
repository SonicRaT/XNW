//! The title's own stack walks and memory queries. It imports
//! `RtlLookupFunctionEntry` and `RtlVirtualUnwind`
//! (api-ms-win-core-rtlsupport) and `VirtualQueryEx`
//! (api-ms-win-core-memory) directly; a test run found 127 ms a frame of title
//! CPU inside NtQueryVirtualMemory with no title frame near. These wrappers
//! call the host's at once; with `XNW_WAIT_STATS=1` each call is counted
//! and timed by its title caller, and function-entry lookups by where the
//! asked PC is (the title image, which the OS loader registered: it is a
//! DLL to it; our code; elsewhere: our VirtualAlloc'd thunks and stubs have
//! no function table, and a lookup for them takes the slow path), one line
//! per 10 s.
//!
//! `XNW_UNWIND=fast`: a lookup for a PC inside the title image is answered
//! from the image's own `.pdata` by binary search (the host's answer for a
//! registered image, without its locks and searches).
//!
//! Cost unset: one call level and one cached test per call.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::threads::KeepError;

type H = *mut c_void;

fn on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_WAIT_STATS").is_ok_and(|v| v.trim() == "1"));
    *ON
}

fn fast() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_UNWIND").is_ok_and(|v| v.trim() == "fast"));
    *ON
}

/// The switches, read at start-up (`read_shim_switches`).
pub(crate) fn read_switches() {
    let _ = fast();
    if on() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(start_report);
    }
}

const NAMES: [&str; 3] = ["RtlLookupFunctionEntry", "RtlVirtualUnwind", "VirtualQueryEx"];

/// (function, caller rva) -> calls, microseconds.
type Stats = HashMap<(usize, usize), (u64, u64)>;
static STATS: Mutex<Option<Stats>> = Mutex::new(None);
/// Lookups by the PC's place: title image, ours, elsewhere (none found
/// counted apart).
static WHERE: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

#[inline(never)]
fn caller() -> usize {
    let mut f = [std::ptr::null_mut(); 1];
    // SAFETY: room for one frame: this, the wrapper, its caller.
    unsafe { xnw_core::win::RtlCaptureStackBackTrace(2, 1, f.as_mut_ptr(), std::ptr::null_mut()) };
    f[0] as usize
}

fn note(which: usize, at: usize, us: u64) {
    {
        let mut g = STATS.lock().unwrap_or_else(|e| e.into_inner());
        let e = g.get_or_insert_with(HashMap::new).entry((which, at.wrapping_sub(crate::title_base()))).or_insert((0, 0));
        e.0 += 1;
        e.1 += us;
    }
}

/// The line every 10 s from a thread of its own, zeros included (a test run
/// printed none when the line came only from a call).
fn start_report() {
    let _ = std::thread::Builder::new().name("xnw walks".into()).spawn(|| {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(10));
            xnw_core::info!("{}", report_text());
        }
    });
}

fn report_text() -> String {
    let taken = STATS.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default();
    let w: Vec<u64> = WHERE.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
    let bound: Vec<&str> = NAMES.iter().zip(&BOUND).filter(|(_, b)| b.load(Ordering::Relaxed)).map(|(n, _)| *n).collect();
    let bound = if bound.is_empty() { "none".to_owned() } else { bound.join(", ") };
    format!("walks: last 10 s: {}; function-entry lookups by PC: title image {}, ours {}, elsewhere {}, none found {}; bound: {bound}", top_text(&taken, 6), w[0], w[1], w[2], w[3])
}

/// Which of [`NAMES`] an import was bound to (the binder asks
/// `host_override`).
static BOUND: [std::sync::atomic::AtomicBool; 3] = [const { std::sync::atomic::AtomicBool::new(false) }; 3];

/// Notes an import of `symbol` from `dll` bound to our wrapper; logs the
/// first of each.
pub(crate) fn note_bound(dll: &str, symbol: &str) {
    if let Some(i) = NAMES.iter().position(|n| *n == symbol)
        && !BOUND[i].swap(true, Ordering::Relaxed)
    {
        xnw_core::info!("walks: {dll}!{symbol} bound to our wrapper");
    }
}

fn top_text(stats: &HashMap<(usize, usize), (u64, u64)>, n: usize) -> String {
    let mut v: Vec<_> = stats.iter().collect();
    v.sort_by(|a, b| b.1.1.cmp(&a.1.1));
    let parts: Vec<String> = v.iter().take(n).map(|((w, at), (c, us))| format!("{} from title+{at:#x}: {c} calls, {:.1} ms", NAMES[*w], *us as f64 / 1000.0)).collect();
    if parts.is_empty() { "none".into() } else { parts.join("; ") }
}

/// The title image's `.pdata`: (begin, end, unwind) rvas, sorted.
fn title_pdata() -> Option<&'static [[u32; 3]]> {
    static P: std::sync::OnceLock<Option<&'static [[u32; 3]]>> = std::sync::OnceLock::new();
    *P.get_or_init(|| {
        let base = crate::title_base();
        (base != 0).then(|| pdata_of(base)).flatten()
    })
}

/// An image's exception directory as RUNTIME_FUNCTION rows.
pub(crate) fn pdata_of(base: usize) -> Option<&'static [[u32; 3]]> {
    // SAFETY: a mapped image's headers and exception directory.
    unsafe {
        let rd = |at: usize| (at as *const u32).read_unaligned();
        let nt = base + rd(base + 0x3c) as usize;
        let (rva, size) = (rd(nt + 0xa0) as usize, rd(nt + 0xa4) as usize);
        (rva != 0 && size >= 12).then(|| std::slice::from_raw_parts((base + rva) as *const [u32; 3], size / 12))
    }
}

/// The RUNTIME_FUNCTION of `rva` in sorted rows (an indirect one followed).
pub(crate) fn lookup_in(base: usize, rows: &'static [[u32; 3]], rva: u32) -> Option<*const [u32; 3]> {
    let i = rows.partition_point(|r| r[0] <= rva).checked_sub(1)?;
    let r = &rows[i];
    if rva >= r[1] {
        return None;
    }
    // RUNTIME_FUNCTION_INDIRECT: the unwind field names another entry.
    if r[2] & 1 != 0 {
        return Some((base + (r[2] & !1) as usize) as *const [u32; 3]);
    }
    Some(std::ptr::from_ref(r))
}

pub extern "system" fn RtlLookupFunctionEntry(pc: u64, image_base: *mut u64, history: H) -> *const c_void {
    let base = crate::title_base();
    let in_title = base != 0 && (pc as usize) >= base && (pc as usize) < base + 0x0b00_0000;
    if fast()
        && in_title
        && let Some(rows) = title_pdata()
    {
        let found = lookup_in(base, rows, (pc as usize - base) as u32);
        if let Some(f) = found {
            if !image_base.is_null() {
                // SAFETY: the caller passes room for the image base.
                unsafe { *image_base = base as u64 };
            }
            return f.cast();
        }
    }
    let real = host!("ntdll.dll", "RtlLookupFunctionEntry": extern "system" fn(u64, *mut u64, H) -> *const c_void);
    if !on() {
        return real(pc, image_base, history);
    }
    let at = caller();
    let started = std::time::Instant::now();
    let r = real(pc, image_base, history);
    let _keep = KeepError::now();
    // SAFETY: our own module.
    let ours = unsafe { xnw_core::win::GetModuleHandleW(std::ptr::null()) as usize };
    let place = if r.is_null() { 3 } else if in_title { 0 } else if (pc as usize) >= ours && (pc as usize) < ours + 0x0400_0000 { 1 } else { 2 };
    WHERE[place].fetch_add(1, Ordering::Relaxed);
    note(0, at, started.elapsed().as_micros() as u64);
    r
}

#[allow(clippy::too_many_arguments)]
pub extern "system" fn RtlVirtualUnwind(kind: u32, image_base: u64, pc: u64, entry: *const c_void, context: H, handler_data: *mut H, frame: *mut u64, pointers: H) -> *const c_void {
    let real = host!("ntdll.dll", "RtlVirtualUnwind": extern "system" fn(u32, u64, u64, *const c_void, H, *mut H, *mut u64, H) -> *const c_void);
    if !on() {
        return real(kind, image_base, pc, entry, context, handler_data, frame, pointers);
    }
    let at = caller();
    let started = std::time::Instant::now();
    let r = real(kind, image_base, pc, entry, context, handler_data, frame, pointers);
    let _keep = KeepError::now();
    note(1, at, started.elapsed().as_micros() as u64);
    r
}

pub extern "system" fn VirtualQueryEx(process: H, address: *const c_void, info: *mut c_void, len: usize) -> usize {
    let real = host!("kernel32.dll", "VirtualQueryEx": extern "system" fn(H, *const c_void, *mut c_void, usize) -> usize);
    if !on() {
        return real(process, address, info, len);
    }
    let at = caller();
    let started = std::time::Instant::now();
    let r = real(process, address, info, len);
    let _keep = KeepError::now();
    note(2, at, started.elapsed().as_micros() as u64);
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fast lookup answers as the host does for code of our own
    /// (registered) test image; the wrappers return what the host returns
    /// and leave its last error.
    #[test]
    fn lookups_match_the_host() {
        // SAFETY: our own module.
        let base = unsafe { xnw_core::win::GetModuleHandleW(std::ptr::null()) as usize };
        let rows = pdata_of(base).expect("our .pdata");
        let host = host!("ntdll.dll", "RtlLookupFunctionEntry": extern "system" fn(u64, *mut u64, H) -> *const c_void);
        let step = (rows.len() / 500).max(1);
        let mut checked = 0;
        for r in rows.iter().step_by(step) {
            for rva in [r[0], (r[0] + r[1]) / 2, r[1] - 1] {
                let mut b = 0u64;
                let want = host((base + rva as usize) as u64, &mut b, std::ptr::null_mut());
                let got = lookup_in(base, rows, rva).map_or(std::ptr::null(), |p| p.cast::<c_void>());
                assert_eq!(got, want, "rva {rva:#x}");
                assert!(want.is_null() || b as usize == base);
                checked += 1;
            }
        }
        assert!(checked > 300);
        // The wrappers: same answers, last error kept.
        use xnw_core::win::{GetLastError, SetLastError};
        let pc = lookups_match_the_host as usize as u64;
        let (mut b1, mut b2) = (0u64, 0u64);
        let (r1, r2) = (host(pc, &mut b1, std::ptr::null_mut()), RtlLookupFunctionEntry(pc, &mut b2, std::ptr::null_mut()));
        assert_eq!((r1, b1), (r2, b2));
        let mut info = [0u8; 48];
        // SAFETY: plain calls on this thread.
        let (v, e) = unsafe { SetLastError(77); (VirtualQueryEx(xnw_core::win::GetCurrentProcess(), pc as *const c_void, info.as_mut_ptr().cast(), 48), GetLastError()) };
        assert_eq!((v, e), (48, 77));
    }
}
