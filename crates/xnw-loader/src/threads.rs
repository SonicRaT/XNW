//! Thread and process-exit notices for title images.
//!
//! Adapted from VIctory, `crates/victory-loader/src/threads.rs` at d2a4bae
//! (GPL-3.0-or-later), on our own Win32 declarations and image record.
//!
//! The prepared copies have no entry point and no TLS callbacks, so the
//! Windows loader tells them nothing when a thread starts or ends. Our own
//! TLS callback, which Windows calls on every thread of the process whoever
//! created it (title, host CRT or our runtime), passes each notice on to
//! every attached image: its TLS callbacks, then a DLL's `DllMain`. At
//! process exit every attached image gets `DLL_PROCESS_DETACH`, in reverse
//! order.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use xnw_core::win::*;

/// What a notice needs of a mapped title image.
#[derive(Debug)]
pub(crate) struct Record {
    /// For the log.
    pub name: String,
    /// Its module handle (base address).
    pub base: usize,
    pub is_dll: bool,
    /// It has static TLS (then `DisableThreadLibraryCalls` fails).
    pub has_tls: bool,
    /// Absolute address of `DllMain`, 0 for none.
    pub entry: usize,
    /// Absolute addresses of its TLS callbacks.
    pub callbacks: Vec<usize>,
}

type Callback = unsafe extern "system" fn(*mut c_void, u32, *mut c_void);
type DllMain = unsafe extern "system" fn(*mut c_void, u32, *mut c_void) -> i32;

impl Record {
    /// Its TLS callbacks, then (when `dll_main`) a DLL's `DllMain`.
    ///
    /// # Safety
    /// Runs title code; the image is bound.
    unsafe fn notify(&self, reason: u32, dll_main: bool, reserved: *mut c_void) {
        let module = self.base as *mut c_void;
        for &f in &self.callbacks {
            // SAFETY: an address from the image's TLS directory.
            unsafe { std::mem::transmute::<usize, Callback>(f)(module, reason, reserved) };
        }
        if dll_main && self.is_dll && self.entry != 0 {
            // SAFETY: the entry point from the image's header.
            unsafe { std::mem::transmute::<usize, DllMain>(self.entry)(module, reason, reserved) };
        }
    }
}

struct Mapped {
    image: Arc<Record>,
    /// Its `DLL_PROCESS_ATTACH` is done; only then does it get thread notices.
    attached: bool,
    /// `DisableThreadLibraryCalls`: no more `DllMain` thread notices.
    quiet: bool,
}

/// Every title image we mapped, in attach order once attached.
static IMAGES: Mutex<Vec<Mapped>> = Mutex::new(Vec::new());

fn images() -> std::sync::MutexGuard<'static, Vec<Mapped>> {
    IMAGES.lock().unwrap_or_else(|e| e.into_inner())
}

pub(crate) fn mapped(image: Record) {
    // SAFETY: reads our own static, so the linker keeps it (and its section).
    unsafe { std::ptr::from_ref(&ON_THREAD).read_volatile() };
    let mut images = images();
    images.retain(|m| m.image.base != image.base);
    images.push(Mapped { image: Arc::new(image), attached: false, quiet: false });
}

/// Its `DLL_PROCESS_ATTACH` is done. The list keeps attach order, which is
/// the Windows loader's initialisation order.
pub(crate) fn attached(base: usize) {
    {
        let mut images = images();
        if let Some(i) = images.iter().position(|m| m.image.base == base) {
            let mut m = images.remove(i);
            m.attached = true;
            images.push(m);
        }
    }
    static AT_EXIT: std::sync::Once = std::sync::Once::new();
    // Registered before the title's own exit functions, so it runs after
    // them, as `ExitProcess` follows `exit`'s functions on Windows.
    // SAFETY: registers a function with the host CRT's exit table.
    AT_EXIT.call_once(|| unsafe {
        atexit(detach_all);
    });
}

/// Off the list: a DLL whose initialisation failed, unmapped again.
pub(crate) fn unmapped(base: usize) {
    images().retain(|m| m.image.base != base);
}

unsafe extern "C" {
    fn atexit(f: extern "C" fn()) -> i32;
}

/// `DisableThreadLibraryCalls` for a title image; `None` if `base` is not
/// one. Fails, as on Windows, for an image with static TLS.
pub fn disable_thread_calls(base: usize) -> Option<bool> {
    let mut images = images();
    let m = images.iter_mut().find(|m| m.image.base == base)?;
    if m.image.has_tls {
        return Some(false);
    }
    m.quiet = true;
    Some(true)
}

/// Set once the process is ending: no more thread notices.
static EXITING: AtomicBool = AtomicBool::new(false);

/// How long title images may take to detach before the process ends anyway
/// (a thread ended while holding a lock their detach code wants).
const DETACH_DEADLINE: Duration = Duration::from_secs(5);

/// Takes every attached image off the list: DLLs in reverse attach order,
/// then the exe. `None` if another thread holds the list.
fn take_attached(wait: bool) -> Option<Vec<Arc<Record>>> {
    let mut images = if wait { images() } else { IMAGES.try_lock().ok()? };
    let attached: Vec<_> = images
        .iter_mut()
        .filter(|m| m.attached)
        .map(|m| {
            m.attached = false;
            m.image.clone()
        })
        .collect();
    let (mut dlls, exes): (Vec<_>, Vec<_>) = attached.into_iter().rev().partition(|i| i.is_dll);
    dlls.extend(exes);
    Some(dlls)
}

/// `DLL_PROCESS_DETACH` to each of `images`, in order. A non-null third
/// argument tells the entry points the process is ending.
fn detach(images: &[Arc<Record>]) {
    for image in images {
        // SAFETY: attached images are bound.
        unsafe { image.notify(DLL_PROCESS_DETACH, true, std::ptr::without_provenance_mut(1)) };
    }
}

/// `ExitProcess` (which Rust's `std::process::exit` and a title calling the
/// host's `ExitProcess` both reach) has ended every other thread and calls
/// our TLS callback: the images get their detach notices here, under the
/// loader lock, as DLLs would. Takes no lock that an ended thread may hold.
fn on_process_exit() {
    EXITING.store(true, Ordering::SeqCst);
    if let Some(images) = take_attached(false) {
        detach(&images);
    }
}

/// Process exit through the CRT's `exit` (returning from `main`), in
/// `ExitProcess`'s order: the exit functions registered after ours have
/// run; every other thread ends (title threads and ours); every attached
/// image gets `DLL_PROCESS_DETACH`, DLLs in reverse initialisation order,
/// then the exe's TLS callbacks; then the CRT calls `ExitProcess`, and
/// [`on_process_exit`] finds nothing left to do.
extern "C" fn detach_all() {
    let Some(images) = take_attached(true) else { return };
    if images.is_empty() {
        return;
    }
    let names: Vec<&str> = images.iter().map(|i| i.name.as_str()).collect();
    xnw_core::info!("process exit: ending other threads, detaching {}", names.join(", "));
    crate::crash::report("process exit");
    EXITING.store(true, Ordering::SeqCst);
    // As on Windows, a lock a thread held when it ended stays taken; detach
    // code that waits for one would hang the exit. The watchdog starts
    // before the other threads end: a thread that ends can hold the heap
    // lock or the loader lock, and a new thread needs both to start. After
    // the deadline it only ends the process; it does not log (that needs
    // the heap and the log's lock).
    // Its thread id, set when it runs its own code; 0 until then.
    static WATCHDOG: AtomicU32 = AtomicU32::new(0);
    drop(std::thread::spawn(|| {
        // SAFETY: a plain query of the calling thread.
        WATCHDOG.store(unsafe { GetCurrentThreadId() }, Ordering::SeqCst);
        std::thread::sleep(DETACH_DEADLINE);
        end_process();
    }));
    // Its start (loader notices, Rust's thread set-up) is done only when it
    // runs its own code. Those notices need the loader lock, which this
    // thread or one waiting for it can hold; then the watchdog never
    // starts, so this wait has the same deadline.
    let waiting = Instant::now();
    let watchdog = loop {
        match WATCHDOG.load(Ordering::SeqCst) {
            0 if waiting.elapsed() >= DETACH_DEADLINE => end_process(),
            0 => std::thread::yield_now(),
            id => break id,
        }
    };
    end_other_threads(&[watchdog]);
    detach(&images);
}

/// Ends this process when the title images do not detach in time. Uses no
/// heap and no lock: a thread that ended can hold them.
fn end_process() {
    // SAFETY: ends this process, which is exiting anyway.
    unsafe { TerminateProcess(GetCurrentProcess(), 1) };
}

/// Ends every thread of the process but this one and those in `keep` (and
/// the ones started while doing so), as `ExitProcess` does first; except the
/// system's thread-pool workers, which `TerminateThread` refuses (the
/// process fails fast) and which run no title code.
fn end_other_threads(keep: &[u32]) {
    // SAFETY: plain queries of the calling process and thread.
    let (process, me) = unsafe { (GetCurrentProcessId(), GetCurrentThreadId()) };
    let mut ended = std::collections::HashSet::new();
    loop {
        // SAFETY: a snapshot of every thread, walked with a sized entry.
        let others: Vec<u32> = unsafe {
            let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
            if snapshot == INVALID_HANDLE_VALUE {
                break;
            }
            let mut entry = ThreadEntry32 { size: size_of::<ThreadEntry32>() as u32, usage: 0, thread_id: 0, owner_process_id: 0, base_priority: 0, delta_priority: 0, flags: 0 };
            let mut ids = Vec::new();
            let mut more = Thread32First(snapshot, &mut entry) != 0;
            while more {
                if entry.owner_process_id == process && entry.thread_id != me && !keep.contains(&entry.thread_id) {
                    ids.push(entry.thread_id);
                }
                more = Thread32Next(snapshot, &mut entry) != 0;
            }
            CloseHandle(snapshot);
            ids
        };
        let new: Vec<u32> = others.into_iter().filter(|id| ended.insert(*id)).collect();
        if new.is_empty() {
            break;
        }
        for id in new {
            // SAFETY: a thread of this process, which is exiting.
            unsafe {
                let thread = OpenThread(THREAD_TERMINATE | THREAD_QUERY_INFORMATION, 0, id);
                if !thread.is_null() {
                    if !starts_in(thread, Some(c"ntdll.dll")) {
                        TerminateThread(thread, 0);
                    }
                    CloseHandle(thread);
                }
            }
        }
    }
}

/// Windows calls every pointer between the CRT's markers `.CRT$XLA` and
/// `.CRT$XLZ`; `XLAV` sorts ahead of the usual `XLB`, so ours runs before
/// any other callback linked into the exe (Rust's std's among them).
#[used]
#[unsafe(link_section = ".CRT$XLAV")]
static ON_THREAD: Callback = on_thread;

unsafe extern "system" fn on_thread(_: *mut c_void, reason: u32, _: *mut c_void) {
    if reason == DLL_PROCESS_DETACH {
        on_process_exit();
        return;
    }
    if reason != DLL_THREAD_ATTACH && reason != DLL_THREAD_DETACH {
        return;
    }
    if EXITING.load(Ordering::SeqCst) {
        return; // our exit watchdog, or threads ending with the process
    }
    // A snapshot: title code may map and attach another DLL from here.
    let notify: Vec<_> = images().iter().filter(|m| m.attached).map(|m| (m.image.clone(), !m.quiet)).collect();
    if notify.is_empty() {
        return;
    }
    // Rust's std sets its threads up after this callback and aborts the
    // process if their per-thread state was touched first (the current
    // thread's id, which our logging asks for when title code calls us).
    // That state is put back as the thread started.
    let saved = (reason == DLL_THREAD_ATTACH && starts_in(current_thread(), None)).then(OwnTls::save);
    // Starts in load order, ends in reverse, as the Windows loader does.
    let notify_one = |(image, dll_main): &(Arc<Record>, bool)| {
        // SAFETY: attached images have their imports bound.
        unsafe { image.notify(reason, *dll_main, std::ptr::null_mut()) };
    };
    if reason == DLL_THREAD_ATTACH {
        notify.iter().for_each(notify_one);
    } else {
        notify.iter().rev().for_each(notify_one);
    }
    if let Some(saved) = saved {
        saved.restore();
    }
}

fn current_thread() -> HANDLE {
    // SAFETY: the calling thread's pseudo handle.
    unsafe { GetCurrentThread() }
}

/// Our own image's static TLS block on this thread (Rust's std keeps its
/// per-thread state there), as copied before title code ran.
struct OwnTls {
    block: *mut u8,
    saved: Vec<u8>,
}

/// The TLS directory the linker emits for our image (the CRT's symbol).
#[repr(C)]
struct TlsDirectory {
    start: usize,
    end: usize,
    index: *const u32,
    callbacks: usize,
    zero_fill: u32,
    characteristics: u32,
}

unsafe extern "C" {
    static _tls_used: TlsDirectory;
}

impl OwnTls {
    fn save() -> OwnTls {
        let array: *const *mut u8;
        // SAFETY: reads TEB.ThreadLocalStoragePointer; the directory is our
        // image's, and Windows set up its block on this thread before
        // calling TLS callbacks.
        let (block, len) = unsafe {
            std::arch::asm!("mov {}, gs:[0x58]", out(reg) array, options(nostack, readonly));
            let dir = &raw const _tls_used;
            let block = *array.add((*dir).index.read() as usize);
            (block, (*dir).end - (*dir).start + (*dir).zero_fill as usize)
        };
        // SAFETY: the block is `len` bytes, live for the thread's life.
        let saved = unsafe { std::slice::from_raw_parts(block, len) }.to_vec();
        OwnTls { block, saved }
    }

    fn restore(self) {
        // SAFETY: the same block on the same thread, no reference to it held.
        unsafe { std::ptr::copy_nonoverlapping(self.saved.as_ptr(), self.block, self.saved.len()) };
    }
}

/// Whether `thread`'s start routine lies in the module named `module`
/// (`None`: our exe). Our exe's are Rust's std threads: the title's main
/// thread and our runtime's workers, which run title callbacks.
fn starts_in(thread: HANDLE, module: Option<&std::ffi::CStr>) -> bool {
    let mut start = 0usize;
    // SAFETY: writes one pointer-sized start address.
    let status = unsafe { NtQueryInformationThread(thread, THREAD_QUERY_SET_WIN32_START_ADDRESS, (&raw mut start).cast(), size_of::<usize>() as u32, std::ptr::null_mut()) };
    let mut found = std::ptr::null_mut();
    let name = module.map_or(std::ptr::null(), |m| m.as_ptr().cast());
    status == 0
        // FROM_ADDRESS | UNCHANGED_REFCOUNT
        // SAFETY: looks up the module containing an address, taking no reference.
        && unsafe { GetModuleHandleExW(4 | 2, start as *const u16, &mut found) } != 0
        // SAFETY: a module handle by name (or our exe's); no reference taken.
        && found == unsafe { GetModuleHandleA(name) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    /// Notices seen: (fake base, thread id, reason, from DllMain).
    static SEEN: Mutex<Vec<(usize, u32, u32, bool)>> = Mutex::new(Vec::new());

    fn seen() -> std::sync::MutexGuard<'static, Vec<(usize, u32, u32, bool)>> {
        SEEN.lock().unwrap_or_else(|e| e.into_inner())
    }

    unsafe extern "system" fn tls_callback(module: *mut c_void, reason: u32, _: *mut c_void) {
        // SAFETY: a plain query.
        let tid = unsafe { GetCurrentThreadId() };
        seen().push((module as usize, tid, reason, false));
    }

    unsafe extern "system" fn dll_main(module: *mut c_void, reason: u32, _: *mut c_void) -> i32 {
        // SAFETY: a plain query.
        let tid = unsafe { GetCurrentThreadId() };
        seen().push((module as usize, tid, reason, true));
        1
    }

    /// A fake image at `base` with one counting TLS callback and, for a
    /// DLL, a counting `DllMain`.
    fn fake(name: &str, base: usize, is_dll: bool) -> Record {
        Record { name: name.into(), base, is_dll, has_tls: false, entry: if is_dll { dll_main as usize } else { 0 }, callbacks: vec![tls_callback as usize] }
    }

    fn count(base: usize, tid: u32, reason: u32, from_main: bool) -> usize {
        seen().iter().filter(|s| **s == (base, tid, reason, from_main)).count()
    }

    #[test]
    fn every_thread_reaches_every_attached_image() {
        const DLL: usize = 0x7100_0000;
        const QUIET: usize = 0x7200_0000;
        const EXE: usize = 0x7300_0000;
        const NOT_ATTACHED: usize = 0x7400_0000;
        mapped(fake("a.dll", DLL, true));
        mapped(fake("quiet.dll", QUIET, true));
        mapped(fake("title", EXE, false));
        mapped(fake("late.dll", NOT_ATTACHED, true));
        attached(DLL);
        attached(QUIET);
        attached(EXE);
        assert_eq!(disable_thread_calls(QUIET), Some(true));
        assert_eq!(disable_thread_calls(1), None);

        static RAN: AtomicU64 = AtomicU64::new(0);
        let threads: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(|| {
                    RAN.fetch_add(1, Ordering::SeqCst);
                    // SAFETY: a plain query.
                    unsafe { GetCurrentThreadId() }
                })
            })
            .collect();
        let ids: Vec<u32> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(RAN.load(Ordering::SeqCst), 8);
        // A joined thread's detach notices are done: they come before the
        // thread object is signalled.
        for id in ids {
            for reason in [DLL_THREAD_ATTACH, DLL_THREAD_DETACH] {
                assert_eq!(count(DLL, id, reason, false), 1, "TLS callback of a.dll, thread {id} reason {reason}");
                assert_eq!(count(DLL, id, reason, true), 1, "DllMain of a.dll");
                assert_eq!(count(QUIET, id, reason, false), 1, "TLS callback of quiet.dll");
                assert_eq!(count(QUIET, id, reason, true), 0, "quiet.dll's DllMain");
                assert_eq!(count(EXE, id, reason, false), 1, "TLS callback of the exe");
                assert_eq!(count(EXE, id, reason, true), 0, "an exe has no DllMain");
                assert_eq!(seen().iter().filter(|s| s.0 == NOT_ATTACHED && s.1 == id).count(), 0);
            }
            // Attach in attach order, detach in reverse.
            let order: Vec<_> = seen().iter().filter(|s| s.1 == id && !s.3).map(|s| (s.0, s.2)).collect();
            assert_eq!(order, [(DLL, 2), (QUIET, 2), (EXE, 2), (EXE, 3), (QUIET, 3), (DLL, 3)]);
        }
        for base in [DLL, QUIET, EXE, NOT_ATTACHED] {
            unmapped(base);
        }
    }

    /// In a child process of the next test: attaches two DLLs and an exe,
    /// starts a thread that never ends, and exits.
    #[test]
    fn exit_child() {
        if std::env::var_os("XNW_EXIT_CHILD").is_none() {
            return;
        }
        unsafe extern "system" fn say(module: *mut c_void, reason: u32, reserved: *mut c_void) {
            use std::io::Write;
            let line = format!("NOTICE {:#x} {reason} {}\n", module as usize, reserved as usize);
            let _ = std::io::stdout().write_all(line.as_bytes());
        }
        unsafe extern "system" fn say_main(module: *mut c_void, reason: u32, reserved: *mut c_void) -> i32 {
            // SAFETY: forwarded.
            unsafe { say((module as usize + 1) as _, reason, reserved) };
            1
        }
        for (base, is_dll) in [(0x1000, true), (0x2000, true), (0x3000, false)] {
            mapped(Record { name: String::new(), base, is_dll, has_tls: false, entry: say_main as usize, callbacks: vec![say as usize] });
            attached(base);
        }
        static TICKS: AtomicU64 = AtomicU64::new(0);
        std::thread::spawn(|| {
            loop {
                TICKS.fetch_add(1, Ordering::SeqCst);
                std::thread::yield_now();
            }
        });
        while TICKS.load(Ordering::SeqCst) == 0 {
            std::thread::yield_now();
        }
        unsafe extern "C" {
            fn exit(code: i32) -> !;
        }
        match std::env::var("XNW_EXIT_CHILD").as_deref() {
            // SAFETY: the CRT's exit: our atexit function, then ExitProcess.
            Ok("crt") => unsafe { exit(0) },
            _ => std::process::exit(0), // ExitProcess
        }
    }

    #[test]
    fn exit_detaches_in_reverse_order() {
        for how in ["crt", "ExitProcess"] {
            exits_by(how);
        }
    }

    fn exits_by(how: &str) {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "threads::tests::exit_child", "--nocapture", "--test-threads=1"])
            .env("XNW_EXIT_CHILD", how)
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        let notices: Vec<&str> = text.lines().filter(|l| l.starts_with("NOTICE") && l.split(' ').nth(2) == Some("0")).collect();
        // DLLs in reverse (TLS callbacks, then DllMain at base + 1), then the
        // exe's TLS callbacks; reason 0, reserved non-null.
        assert_eq!(notices, ["NOTICE 0x2000 0 1", "NOTICE 0x2001 0 1", "NOTICE 0x1000 0 1", "NOTICE 0x1001 0 1", "NOTICE 0x3000 0 1"], "{how}: {text}");
        assert!(out.status.success(), "{how}: {text}");
    }
}
