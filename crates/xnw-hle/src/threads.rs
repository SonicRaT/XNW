//! The title's threads, their affinity and the processor count it
//! sees. The title imports `CreateThread`
//! (api-ms-win-core-processthreads), `Get/SetThreadGroupAffinity`
//! (api-ms-win-core-processtopology) and `GetLogicalProcessorInformation`
//! (api-ms-win-core-sysinfo); nothing else about processors (V: its import
//! table).
//!
//! - `CreateThread`: every title thread starts at one trampoline
//!   (0x142215EA0) whose parameter's first word is the real entry.
//!   The thread is named `title <name or entry rva>` (`SetThreadDescription`)
//!   so the frame-threads line and the CPU sampler tell them apart.
//! - `SetThreadGroupAffinity`: the title asks console processor numbers.
//!   Subrender workers ask 0x1F00, audio and others 0x3F00 (bits 8-13: a
//!   14-thread console), the job pools ~(-1 << n) with n from
//!   `GetLogicalProcessorInformation`. On a 12-processor host a mask naming
//!   bit 12 or 13 is refused by Windows (the thread keeps all processors).
//!   `XNW_AFFINITY=title` (default: as before, the call goes through),
//!   `host` (never applied: the threads run anywhere), `map` (another runtime's rule:
//!   cut to the processors there are, else each console processor to the
//!   host one at its index modulo their count; for 0x1F00 that pins five
//!   Subrender threads onto processors 8-11, two physical cores). Every
//!   distinct (thread, mask) is logged once with what was applied and the
//!   result.
//! - `SetThreadPriority`: as asked, or with `XNW_TITLE_PRIORITIES=lift`
//!   idle and lowest raised to below normal (VIctory-7900XTX 0d065d3).
//! - `GetLogicalProcessorInformation`: `XNW_CPUS=<n>` reports only the
//!   first n logical processors (default: the host's, as before); `console`
//!   is not offered, since 14 is more than this host has and masks naming
//!   the missing ones are refused.

use std::ffi::c_void;
use std::sync::Mutex;

type H = *mut c_void;

/// `GROUP_AFFINITY`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GroupAffinity {
    mask: usize,
    group: u16,
    reserved: [u16; 3],
}

/// The trampoline all title threads start at.
const TRAMPOLINE: usize = 0x2215ea0;

/// Names of title thread entries (rva) known from a list of its threads.
const ENTRIES: [(usize, &str); 8] = [
    (0x43c4140, "Subrender"),
    (0x23cefb0, "render"),
    (0x2481e90, "submit"),
    (0x41f18b0, "socket"),
    (0x25785c0, "nav"),
    (0x5c60800, "audio a"),
    (0x43adc30, "audio b"),
    (0x2218c60, "job worker"),
];

fn entry_name(rva: usize) -> String {
    ENTRIES.iter().find(|(r, _)| *r == rva).map_or_else(|| format!("title fn {:#x}", 0x1_4000_0000 + rva), |(_, n)| format!("title {n}"))
}

/// The thread's last error, kept across work of ours after a host call.
pub(crate) struct KeepError(u32);
impl KeepError {
    pub(crate) fn now() -> KeepError {
        // SAFETY: plain calls on the current thread.
        KeepError(unsafe { xnw_core::win::GetLastError() })
    }
}
impl Drop for KeepError {
    fn drop(&mut self) {
        // SAFETY: as above.
        unsafe { xnw_core::win::SetLastError(self.0) };
    }
}

/// The switches, read at start-up (`read_shim_switches`).
pub(crate) fn read_switches() {
    let _ = (affinity_mode(), cpus(), lift_priorities());
}

/// `XNW_TITLE_PRIORITIES=lift`: the title's idle and lowest threads run
/// below normal; off (default): as the title sets them.
fn lift_priorities() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        let on = std::env::var("XNW_TITLE_PRIORITIES").is_ok_and(|v| v.trim() == "lift");
        if on {
            xnw_core::info!("threads: the title's idle and lowest threads run below normal (XNW_TITLE_PRIORITIES=lift)");
        }
        on
    });
    *ON
}

/// The priority a title thread gets for `asked`: with
/// `XNW_TITLE_PRIORITIES=lift`, idle (-15) up to lowest (-2) become below
/// normal (-1); everything else as asked. The title puts several threads at
/// idle (the record: 16 of 71 calls negative), the streaming pager's among
/// them; on the console they have what the rest leaves of their cores, here
/// they compete with our own workers (DirectStorage's 12, pipelines) and other
/// programs, and loading waits on them (VIctory-7900XTX 0d065d3, run
/// stream-3: loads seen done 1.1 s late on average, 4.8 s at worst).
pub(crate) fn title_priority(asked: i32, lift: bool) -> i32 {
    const IDLE: i32 = -15;
    const LOWEST: i32 = -2;
    const BELOW_NORMAL: i32 = -1;
    match asked {
        IDLE..=LOWEST if lift => BELOW_NORMAL,
        other => other,
    }
}

/// After VIctory-7900XTX `crates/victory-hle/src/modules/kernel.rs`
/// `SetThreadPriority` at 0d065d3 (GPL-3.0-or-later); off by default here.
pub extern "system" fn SetThreadPriority(thread: H, priority: i32) -> i32 {
    let priority = title_priority(priority, lift_priorities());
    host!("kernel32.dll", "SetThreadPriority": extern "system" fn(H, i32) -> i32)(thread, priority)
}

pub extern "system" fn CreateThread(attributes: H, stack: usize, start: usize, param: H, flags: u32, id: *mut u32) -> H {
    let h = host!("kernel32.dll", "CreateThread": extern "system" fn(H, usize, usize, H, u32, *mut u32) -> H)(attributes, stack, start, param, flags, id);
    let _keep = KeepError::now();
    if !h.is_null() {
        let base = crate::title_base();
        let mut entry = start;
        if base != 0 && start == base + TRAMPOLINE && !param.is_null() {
            let mut w = [0u8; 8];
            // SAFETY: a read of the title's parameter block through the kernel.
            if unsafe { xnw_core::win::ReadProcessMemory(xnw_core::win::GetCurrentProcess(), param.cast_const(), w.as_mut_ptr().cast(), 8, std::ptr::null_mut()) } != 0 {
                entry = usize::from_le_bytes(w);
            }
        }
        let name = if base != 0 && entry > base && entry < base + 0x0b00_0000 { entry_name(entry - base) } else { format!("title start {entry:#x}") };
        let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
        // SAFETY: a handle just made, a NUL-terminated name.
        unsafe { xnw_core::win::SetThreadDescription(h, wide.as_ptr()) };
    }
    h
}

/// `XNW_AFFINITY`: 0 title, 1 host, 2 map.
fn affinity_mode() -> u8 {
    static MODE: std::sync::LazyLock<u8> = std::sync::LazyLock::new(|| match std::env::var("XNW_AFFINITY").as_deref().map(str::trim) {
        Ok("host") => 1,
        Ok("map") => 2,
        _ => 0,
    });
    *MODE
}

/// Another runtime's fit (read for the idea: `crates/xcl/src/hle/kernelx.rs`
/// `fit_affinity_mask`, GPL-2.0-only, no code taken): a mask inside the
/// allowed one passes; else cut to it; else each bit to the allowed bit at
/// its index modulo their count.
pub(crate) fn fit(mask: usize, allowed: usize) -> usize {
    if mask == 0 || allowed == 0 || mask & !allowed == 0 {
        return mask;
    }
    if mask & allowed != 0 {
        return mask & allowed;
    }
    let host: Vec<u32> = (0..usize::BITS).filter(|b| allowed >> b & 1 != 0).collect();
    (0..usize::BITS).filter(|b| mask >> b & 1 != 0).fold(0, |m, b| m | 1 << host[b as usize % host.len()])
}

fn process_mask() -> usize {
    let (mut p, mut s) = (0usize, 0usize);
    // SAFETY: two out-parameters.
    unsafe { xnw_core::win::GetProcessAffinityMask(xnw_core::win::GetCurrentProcess(), &mut p, &mut s) };
    p
}

pub extern "system" fn SetThreadGroupAffinity(thread: H, asked: *const GroupAffinity, previous: *mut GroupAffinity) -> i32 {
    let real = host!("kernel32.dll", "SetThreadGroupAffinity": extern "system" fn(H, *const GroupAffinity, *mut GroupAffinity) -> i32);
    if asked.is_null() {
        return real(thread, asked, previous);
    }
    // SAFETY: the title passes a whole GROUP_AFFINITY.
    let mut a = unsafe { *asked };
    let mask = a.mask;
    let (applied, r) = match affinity_mode() {
        1 => {
            if !previous.is_null() {
                // SAFETY: the title passes room for the previous affinity.
                let get = host!("kernel32.dll", "GetThreadGroupAffinity": extern "system" fn(H, *mut GroupAffinity) -> i32);
                get(thread, previous);
            }
            (None, 1)
        }
        2 => {
            a.mask = fit(mask, process_mask());
            (Some(a.mask), real(thread, &a, previous))
        }
        _ => (Some(mask), real(thread, asked, previous)),
    };
    let _keep = KeepError::now();
    log_once(thread, mask, applied, r);
    r
}

/// Each distinct (thread name, mask) once.
fn log_once(thread: H, mask: usize, applied: Option<usize>, r: i32) {
    static SEEN: Mutex<Vec<(String, usize)>> = Mutex::new(Vec::new());
    let name = name_of(thread);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.iter().any(|(n, m)| *n == name && *m == mask) || seen.len() > 256 {
        return;
    }
    seen.push((name.clone(), mask));
    // SAFETY: a plain query.
    let error = if r == 0 { unsafe { xnw_core::win::GetLastError() } } else { 0 };
    xnw_core::info!(
        "affinity: {name} asked {mask:#x} (process {:#x}): {} -> {} (XNW_AFFINITY={})",
        process_mask(),
        applied.map_or("not applied".to_string(), |m| format!("applied {m:#x}")),
        if r != 0 { "ok".to_string() } else { format!("refused, error {error}") },
        ["title", "host", "map"][affinity_mode() as usize]
    );
}

fn name_of(thread: H) -> String {
    let mut p: *mut u16 = std::ptr::null_mut();
    // SAFETY: a query; the string is freed here.
    unsafe {
        if xnw_core::win::GetThreadDescription(thread, &mut p) < 0 || p.is_null() {
            return "?".into();
        }
        let mut n = 0;
        while *p.add(n) != 0 && n < 128 {
            n += 1;
        }
        let s = String::from_utf16_lossy(std::slice::from_raw_parts(p, n));
        xnw_core::win::LocalFree(p.cast());
        if s.is_empty() { "unnamed".into() } else { s }
    }
}

/// `XNW_CPUS=<n>`: logical processors reported (0: the host's).
fn cpus() -> usize {
    static N: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| std::env::var("XNW_CPUS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0));
    *N
}

/// `SYSTEM_LOGICAL_PROCESSOR_INFORMATION`: mask, relationship, union.
const ENTRY: usize = 32;

/// Keeps only processors `0..n` in `n_bytes` of entries: masks cut,
/// entries left empty dropped; the bytes kept.
pub(crate) fn cut(buf: &mut [u8], n: usize) -> usize {
    let keep = if n >= usize::BITS as usize { usize::MAX } else { (1usize << n) - 1 };
    let mut out = 0;
    for i in 0..buf.len() / ENTRY {
        let at = i * ENTRY;
        let mask = usize::from_le_bytes(buf[at..at + 8].try_into().unwrap_or([0; 8])) & keep;
        if mask == 0 {
            continue;
        }
        buf.copy_within(at..at + ENTRY, out);
        buf[out..out + 8].copy_from_slice(&mask.to_le_bytes());
        out += ENTRY;
    }
    out
}

pub extern "system" fn GetLogicalProcessorInformation(info: *mut u8, len: *mut u32) -> i32 {
    let real = host!("kernel32.dll", "GetLogicalProcessorInformation": extern "system" fn(*mut u8, *mut u32) -> i32);
    let r = real(info, len);
    let _keep = KeepError::now();
    let n = cpus();
    if r != 0 && n > 0 && !info.is_null() && !len.is_null() {
        // SAFETY: the call filled `*len` bytes at `info`.
        unsafe {
            let bytes = std::slice::from_raw_parts_mut(info, *len as usize);
            *len = cut(bytes, n) as u32;
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| xnw_core::info!("cpus: GetLogicalProcessorInformation reports the first {n} logical processors (XNW_CPUS)"));
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With the switch, idle and lowest become below normal; the others,
    /// and everything without it, as asked.
    #[test]
    fn idle_title_threads_are_lifted_only_when_asked() {
        let given = |lift| [-15, -2, -1, 0, 1, 2, 15].map(|p| title_priority(p, lift));
        assert_eq!(given(true), [-1, -1, -1, 0, 1, 2, 15]);
        assert_eq!(given(false), [-15, -2, -1, 0, 1, 2, 15]);
    }

    /// Each shim, called as the title calls it, returns what
    /// the host returns and leaves the thread's last error as the host left
    /// it.
    #[test]
    fn shims_keep_the_hosts_result_and_last_error() {
        use xnw_core::win::{GetLastError, SetLastError};
        // GetLogicalProcessorInformation with no buffer: FALSE, 122.
        let host_glpi = host!("kernel32.dll", "GetLogicalProcessorInformation": extern "system" fn(*mut u8, *mut u32) -> i32);
        let (mut a, mut b) = (0u32, 0u32);
        // SAFETY: plain calls on this thread.
        let (ra, ea) = unsafe { SetLastError(0); (host_glpi(std::ptr::null_mut(), &mut a), GetLastError()) };
        let (rb, eb) = unsafe { SetLastError(0); (GetLogicalProcessorInformation(std::ptr::null_mut(), &mut b), GetLastError()) };
        assert_eq!((ra, ea, a), (0, 122, a));
        assert_eq!((rb, eb, b), (ra, ea, a), "the shim as the host");
        // A timed-out wait: WAIT_TIMEOUT, last error untouched.
        // SAFETY: an event of our own, never signalled.
        let event = unsafe { xnw_core::win::CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
        let (rw, ew) = unsafe { SetLastError(1234); (crate::waits::WaitForSingleObject(event, 0), GetLastError()) };
        assert_eq!((rw, ew), (0x102, 1234));
        // SetThreadGroupAffinity with a processor this host has not.
        let bad = GroupAffinity { mask: 1 << 40, group: 0, reserved: [0; 3] };
        let host_set = host!("kernel32.dll", "SetThreadGroupAffinity": extern "system" fn(H, *const GroupAffinity, *mut GroupAffinity) -> i32);
        // SAFETY: the current thread's pseudo-handle, a whole GROUP_AFFINITY.
        let (rh, eh) = unsafe { SetLastError(0); (host_set(xnw_core::win::GetCurrentThread(), &bad, std::ptr::null_mut()), GetLastError()) };
        let (rs, es) = unsafe { SetLastError(0); (SetThreadGroupAffinity(xnw_core::win::GetCurrentThread(), &bad, std::ptr::null_mut()), GetLastError()) };
        assert_eq!(rh, 0, "the host refuses a missing processor");
        assert_eq!((rs, es), (rh, eh), "the shim as the host (error {eh})");
    }

    #[test]
    fn console_masks_fit_the_host() {
        // 12 host processors.
        assert_eq!(fit(0x1f00, 0xfff), 0xf00, "Subrender: cut to 8-11");
        assert_eq!(fit(0x3f00, 0xfff), 0xf00);
        assert_eq!(fit(0x3000, 0xfff), 0x3, "none there: modulo (12 -> 0, 13 -> 1)");
        assert_eq!(fit(0x0ff, 0xfff), 0x0ff, "inside: as asked");
    }

    #[test]
    fn processors_past_n_are_cut() {
        let mut buf = vec![0u8; 3 * ENTRY];
        for (i, mask) in [0x3usize, 0xc, 0xfff].iter().enumerate() {
            buf[i * ENTRY..i * ENTRY + 8].copy_from_slice(&mask.to_le_bytes());
            buf[i * ENTRY + 8] = i as u8;
        }
        let kept = cut(&mut buf, 2);
        assert_eq!(kept, 2 * ENTRY, "the core of processors 2-3 goes");
        assert_eq!(usize::from_le_bytes(buf[ENTRY..ENTRY + 8].try_into().unwrap()), 0x3, "the package mask cut to 0-1");
        assert_eq!(buf[ENTRY + 8], 2, "the third entry moved up");
    }

    #[test]
    fn thread_entries_are_named() {
        assert_eq!(entry_name(0x43c4140), "title Subrender");
        assert_eq!(entry_name(0x1234), "title fn 0x140001234");
    }

    /// What a 1 ms wait costs through our wrappers on this host,
    /// before and after timeBeginPeriod(1) (the runtime calls it at start:
    /// `audio::use_console_timers`). Printed; the only assertion is that
    /// with the 1 ms period a 1 ms wait takes well under a 15.6 ms tick.
    #[test]
    fn one_ms_waits_cost() {
        let measure = |f: &dyn Fn()| {
            let at = std::time::Instant::now();
            for _ in 0..20 {
                f();
            }
            at.elapsed().as_secs_f64() * 1000.0 / 20.0
        };
        // SAFETY: an unnamed manual-reset event of our own, never signalled.
        let event = unsafe { xnw_core::win::CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
        let sleep = || crate::waits::Sleep(1);
        let wait = || {
            crate::waits::WaitForSingleObject(event, 1);
        };
        let (s0, w0) = (measure(&sleep), measure(&wait));
        // SAFETY: a plain call.
        unsafe { xnw_core::win::timeBeginPeriod(1) };
        let (s1, w1) = (measure(&sleep), measure(&wait));
        println!("waits: 1 ms waits: Sleep {s0:.2} ms, WaitForSingleObject {w0:.2} ms before timeBeginPeriod(1); {s1:.2} / {w1:.2} ms after");
        // Timing-sensitive: asserted only on a quiet host (XNW_TEST_TIMING=1).
        if std::env::var("XNW_TEST_TIMING").is_ok_and(|v| v == "1") {
            assert!(s1 < 5.0 && w1 < 5.0, "with the 1 ms period: Sleep {s1:.2} ms, wait {w1:.2} ms");
        }
    }
}
