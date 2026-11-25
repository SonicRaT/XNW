//! `XNW_OBJECT_GUARD=1` (or `keep`): catch the title writing past the
//! objects and blocks we hand it. The title's x64 code runs in our process
//! and reads and writes our objects by fixed offsets (its inlined code);
//! an object smaller than what it writes would have the overrun land in
//! whatever the heap put next, the driver's allocations included (the
//! driver faults on descriptor-shaped words in its own memory, in two test runs).
//!
//! Each guarded object or block gets its own VirtualAlloc region:
//! `[guard page][data pages][guard page]`, the object's end flush against
//! the trailing guard (its size is a multiple of its alignment, so flush
//! is aligned), the leading guard before its first page. A touch of a
//! guard page is reported by our first vectored handler (kind, our size,
//! how far past the end or before the start, read or write, the faulting
//! code address), then the crash logger logs the registers and stack and
//! stops the run (`xnw_core::FATAL_FAULT`). `keep` leaves a released
//! object's pages no-access for the run: a later touch is "after release".
//! The pad after a list's console layout stays zero (the title reads it,
//! A canary there was dereferenced) and is checked for bytes the
//! title left at each submission and at release (`list::guard_check`).
//!
//! Cost unset: one cached test per object made and released.

use std::collections::BTreeMap;
use std::sync::Mutex;

use xnw_core::win;

const PAGE: usize = 4096;

/// 0 off, 1 on, 2 keep.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::LazyLock<u8> = std::sync::LazyLock::new(|| {
        let m = match std::env::var("XNW_OBJECT_GUARD").as_deref().map(str::trim) {
            Ok("1") => 1,
            Ok("keep") => 2,
            _ => 0,
        };
        start_log(m);
        if m != 0 {
            install();
        }
        m
    });
    *MODE
}

/// A guarded region: its base and length, where the object starts, its
/// size and kind, and whether it was released (`keep`).
#[derive(Clone, Copy)]
struct Region {
    len: usize,
    start: usize,
    size: usize,
    kind: &'static str,
    released: bool,
    /// Its slack was reported written.
    reported: bool,
}

static REGIONS: Mutex<BTreeMap<usize, Region>> = Mutex::new(BTreeMap::new());

/// The alignment every guarded block gets at least: the process heap's on
/// x64. The title relies on it: `vmovaps xmm0,[rdx+0x10]` on our pipeline
/// object at title+0x32786 (and the same pair at title+0x248c30d); an
/// 8-aligned pipeline flush against the guard faulted there.
pub(crate) const MIN_ALIGN: usize = 16;

/// `size` bytes aligned to `align` (at least [`MIN_ALIGN`]) ending against
/// a no-access page: flush when `size` is a multiple of the alignment,
/// else the slack between (under 16 bytes for most) is a canary checked at
/// release and, for lists, at each submission ([`slack_check`]); one
/// no-access page before them. `None` if the guard is off, the alignment
/// is past 64 KiB or the memory cannot be had.
pub(crate) fn alloc(kind: &'static str, size: usize, align: usize) -> Option<*mut u8> {
    if mode() == 0 {
        return None;
    }
    alloc_any(kind, size, align)
}

fn alloc_any(kind: &'static str, size: usize, align: usize) -> Option<*mut u8> {
    if align > 1 << 16 {
        return None;
    }
    let align = align.max(MIN_ALIGN);
    let body = size.max(1);
    let pages = (body + align).div_ceil(PAGE);
    let len = (pages + 2) * PAGE;
    // SAFETY: a fresh region of ours.
    let base = unsafe { win::VirtualAlloc(std::ptr::null(), len, win::MEM_COMMIT | win::MEM_RESERVE, win::PAGE_READWRITE) } as usize;
    if base == 0 {
        return None;
    }
    let mut old = 0;
    // SAFETY: the region's first and last pages, ours.
    unsafe {
        win::VirtualProtect(base as *const _, PAGE, win::PAGE_NOACCESS, &mut old);
        win::VirtualProtect((base + len - PAGE) as *const _, PAGE, win::PAGE_NOACCESS, &mut old);
    }
    let start = (base + len - PAGE - body) / align * align;
    // Slack after the block (its size not a multiple of the alignment):
    // the canary, checked at release.
    // SAFETY: inside the committed pages.
    unsafe { std::ptr::write_bytes((start + size) as *mut u8, 0xA5, base + len - PAGE - start - size) };
    REGIONS.lock().unwrap_or_else(|e| e.into_inner()).insert(base, Region { len, start, size, kind, released: false, reported: false });
    Some(start as *mut u8)
}

/// Frees the guarded block at `start` (or, under `keep`, makes it no
/// access for the run); false if it is not one of ours.
pub(crate) fn free(start: *mut u8) -> bool {
    if mode() == 0 {
        return false;
    }
    let mut regions = REGIONS.lock().unwrap_or_else(|e| e.into_inner());
    let Some((&base, r)) = regions.range(..=start as usize).next_back() else { return false };
    if r.start != start as usize || r.released {
        return false;
    }
    let r = *r;
    slack_report(base, &r);
    if mode() == 2 {
        let mut old = 0;
        // SAFETY: our region, kept unreachable.
        unsafe { win::VirtualProtect(base as *const _, r.len, win::PAGE_NOACCESS, &mut old) };
        regions.insert(base, Region { released: true, ..r });
    } else {
        regions.remove(&base);
        // SAFETY: our region, its object already dropped.
        unsafe { win::VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }
    true
}
const MEM_RELEASE: u32 = 0x8000;

/// The first slack byte (after the block, before the guard) that is not
/// the canary: its distance past the block's end.
fn slack_written(base: usize, r: &Region) -> Option<usize> {
    // SAFETY: inside the committed pages of a live region.
    let slack = unsafe { std::slice::from_raw_parts((r.start + r.size) as *const u8, base + r.len - PAGE - r.start - r.size) };
    slack.iter().position(|&b| b != 0xA5)
}

fn slack_report(base: usize, r: &Region) {
    if let Some(i) = slack_written(base, r) {
        xnw_core::warn!("OBJECT GUARD: {} at {:#x} ({} bytes): slack written {} bytes past its end", r.kind, r.start, r.size, i + 1);
    }
}

/// The slack check of the guarded block at `start` (lists, at each
/// submission); reported once per block.
pub(crate) fn slack_check(start: *const u8) {
    if mode() == 0 {
        return;
    }
    let mut regions = REGIONS.lock().unwrap_or_else(|e| e.into_inner());
    let Some((&base, r)) = regions.range_mut(..=start as usize).next_back() else { return };
    if r.start != start as usize || r.released || r.reported {
        return;
    }
    if slack_written(base, r).is_some() {
        r.reported = true;
        slack_report(base, r);
    }
}

/// What `address` is to the guard: past an object's end, before its
/// start, or in a released one; `None` if none of ours.
pub(crate) fn describe(address: usize) -> Option<String> {
    let regions = REGIONS.try_lock().ok()?;
    let (&base, r) = regions.range(..=address).next_back()?;
    if address >= base + r.len {
        return None;
    }
    let end = r.start + r.size;
    Some(if r.released {
        format!("{} at {:#x} ({} bytes) touched after its release, at offset {:#x}", r.kind, r.start, r.size, address as isize - r.start as isize)
    } else if address >= end {
        format!("{} bytes past the end of {} at {:#x} ({} bytes; offset {:#x})", address - end + 1, r.kind, r.start, r.size, address - r.start)
    } else if address < r.start {
        format!("{} bytes before the start of {} at {:#x} ({} bytes)", r.start - address, r.kind, r.start, r.size)
    } else {
        format!("inside {} at {:#x} ({} bytes), offset {:#x}", r.kind, r.start, r.size, address - r.start)
    })
}

/// The first vectored handler: a guard page touch is reported and made
/// fatal; anything else passes on.
fn install() {
    unsafe extern "system" fn on_fault(info: *mut win::ExceptionPointers) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).record };
        if r.code != win::EXCEPTION_ACCESS_VIOLATION || r.param_count < 2 {
            return win::EXCEPTION_CONTINUE_SEARCH;
        }
        if let Some(what) = describe(r.params[1]) {
            let how = if r.params[0] == 1 { "write" } else if r.params[0] == 8 { "execute" } else { "read" };
            xnw_core::warn!("OBJECT GUARD: {how} at {:#x}: {what}; by code at {:#x} (the exception line below names it)", r.params[1], r.address);
            xnw_core::FATAL_FAULT.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        win::EXCEPTION_CONTINUE_SEARCH
    }
    // SAFETY: the handler reads the record and our region map only.
    unsafe { win::AddVectoredExceptionHandler(1, on_fault) };
}

/// Once at start: the process's mitigations and whether the page heap is
/// on for xnw.exe (GFlags `FLG_HEAP_PAGE_ALLOCS`).
fn start_log(m: u8) {
    let mut words = Vec::new();
    for (policy, name) in [(0, "DEP"), (1, "ASLR"), (2, "dynamic code"), (3, "strict handles"), (7, "control flow guard"), (8, "signature")] {
        let mut v = 0u32;
        // SAFETY: a policy query into a 4-byte local.
        let ok = unsafe { win::GetProcessMitigationPolicy(win::GetCurrentProcess(), policy, std::ptr::from_mut(&mut v).cast(), 4) } != 0;
        words.push(if ok { format!("{name} {v:#x}") } else { format!("{name} ?") });
    }
    let key: Vec<u16> = "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Image File Execution Options\\xnw.exe\0".encode_utf16().collect();
    let value: Vec<u16> = "GlobalFlag\0".encode_utf16().collect();
    let (mut flags, mut size) = (0u32, 4u32);
    // SAFETY: a DWORD read into a local of its size.
    let read = unsafe { win::RegGetValueW(win::HKEY_LOCAL_MACHINE, key.as_ptr(), value.as_ptr(), win::RRF_RT_REG_DWORD, std::ptr::null_mut(), std::ptr::from_mut(&mut flags).cast(), &mut size) } == 0;
    const FLG_HEAP_PAGE_ALLOCS: u32 = 0x0200_0000;
    xnw_core::info!(
        "object guard: XNW_OBJECT_GUARD {}; mitigations: {}; page heap for xnw.exe: {}",
        ["off", "on", "keep"][m as usize],
        words.join(", "),
        if !read { "no GlobalFlag".to_string() } else if flags & FLG_HEAP_PAGE_ALLOCS != 0 { format!("ON (GlobalFlag {flags:#x})") } else { format!("off (GlobalFlag {flags:#x})") }
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `describe` only try-locks REGIONS: tests that hold or take it run one at a time.
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        static ONE: Mutex<()> = Mutex::new(());
        ONE.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Every guarded block is at least 16-aligned (the title loads our
    /// pipeline with `vmovaps` at +0x10), whatever its size and asked
    /// alignment; the slack before the guard is the canary, and a slack
    /// write is found.
    #[test]
    fn guarded_blocks_are_16_aligned() {
        let _serial = serial();
        for (size, align) in [(0x38, 8), (1, 1), (0x560 + 1024 + 200, 8), (40, 0x100), (4096, 16), (4095, 4), (24, 64)] {
            let at = alloc_any("test", size, align).expect("a guarded block");
            assert!((at as usize).is_multiple_of(align.max(MIN_ALIGN)), "{size} bytes aligned {align}: {at:p}");
            let end = at as usize + size;
            assert!(end.next_multiple_of(PAGE) - end < MIN_ALIGN.max(align), "{size}/{align}: ends {} bytes before the guard", end.next_multiple_of(PAGE) - end);
            let (base, r) = {
                let g = REGIONS.lock().unwrap();
                let (&b, r) = g.range(..=at as usize).next_back().unwrap();
                (b, *r)
            };
            assert_eq!(slack_written(base, &r), None, "the slack is the canary");
            if end % PAGE != 0 {
                // SAFETY: the first slack byte, committed.
                unsafe { (end as *mut u8).write(0) };
                assert_eq!(slack_written(base, &r), Some(0));
            }
            REGIONS.lock().unwrap().remove(&base);
            // SAFETY: our region.
            unsafe { win::VirtualFree(base as *mut _, 0, MEM_RELEASE) };
        }
    }

    /// An object's end is flush against a guard page; its last byte is
    /// writable, the byte after it is not (the fault is described with the
    /// object and the offset).
    #[test]
    fn a_one_byte_overrun_is_on_the_guard() {
        let _serial = serial();
        // The guard is off in tests: regions are made by hand.
        let body = 200;
        let len = 3 * PAGE;
        // SAFETY: a region of ours, as `alloc` makes it.
        let base = unsafe { win::VirtualAlloc(std::ptr::null(), len, win::MEM_COMMIT | win::MEM_RESERVE, win::PAGE_READWRITE) } as usize;
        assert_ne!(base, 0);
        let mut old = 0;
        // SAFETY: our region's last page.
        unsafe { win::VirtualProtect((base + len - PAGE) as *const _, PAGE, win::PAGE_NOACCESS, &mut old) };
        let start = base + len - PAGE - body;
        REGIONS.lock().unwrap().insert(base, Region { len, start, size: body, kind: "list", released: false, reported: false });
        // SAFETY: the object's last byte, inside the committed page.
        unsafe { ((start + body - 1) as *mut u8).write(1) };
        let query = |a: usize| {
            // SAFETY: zero is a valid value; a query about our memory.
            let mut info: win::MemoryBasicInformation = unsafe { std::mem::zeroed() };
            unsafe { win::VirtualQueryEx(win::GetCurrentProcess(), a as *const _, &mut info, size_of::<win::MemoryBasicInformation>()) };
            info.protect
        };
        assert_eq!(query(start + body), win::PAGE_NOACCESS, "the byte after the object is on the guard");
        let text = describe(start + body).unwrap();
        assert!(text.starts_with("1 bytes past the end of list at") && text.contains("(200 bytes; offset 0xc8)"), "{text}");
        REGIONS.lock().unwrap().remove(&base);
        // SAFETY: our region.
        unsafe { win::VirtualFree(base as *mut _, 0, MEM_RELEASE) };

    }
}
