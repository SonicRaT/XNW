//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early.
//!
//! Ported from VIctory `crates/victory-gpu/src/pipeline/library.rs` 1-524
//! at xnw/base 5d0516e (VIctory d2a4bae; no merged PR touches it),
//! GPL-3.0-or-later. Changed: the log macros of xnw-core, the content
//! hash of xnw-core, the host device from `d3d12x::gpu`, thread names,
//! the test scratch folder with no extra crate.
//! With VIctory-Belobug-AMD (fork of VIctory at 8e79c80; read at 4cbe9d7),
//! `crates/victory-gpu/src/pipeline/library.rs` 108-352 (0d2f6a6, 1b0c6b5,
//! 95e935d, 2655edb), GPL-3.0-or-later: the guard over the driver's
//! compiler (a crash parks the thread, an endless compile is suspended,
//! both are left out from then on). Changed: Win32 through xnw-core, a
//! lost worker is not replaced, the crashed list holds this run's too.
//!
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//! 2. The title pipelines a run drew or dispatched with are listed by a key
//!    known when the title creates them (their shader containers and
//!    graphics state), with the time of their first use; the next run's
//!    prebuilder (`prebuild.rs`) builds those once the title has
//!    created its pipelines, earliest use first. The list is
//!    `pipeline-list.txt` (one file for every translator version, the
//!    key names console code and state only, so it only grows): a `#`
//!    header line, then `<key>\t<first use in ms, relative to the story
//!    load start>[\t<recipe>]` per line, merged from run to run (a key used
//!    again takes the newer time). The recipe (`warm.rs`) names the shaders'
//!    pack containers and holds the graphics state, so `xnw prebuild` and
//!    the launch prebuild rebuild the pipeline without the title. Made the
//!    first time a draw asks for the pipeline (made or not, so a pipeline
//!    that fails is retried after a translator change). The first run with
//!    no such file imports every older `early-pipelines*.txt` (keys
//!    without recipes: the in-run prebuild still matches them, and adds
//!    their recipes when it makes them).
//!
//! All library calls are serialised: the interface does not promise that
//! stores, loads and serialising may overlap.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, mpsc};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// Host pipelines created and loaded from the library since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
/// Title pipelines first used in this run ("recorded").
pub(crate) static RECORDED: AtomicU64 = AtomicU64::new(0);
/// Milliseconds since the start at which the story load began (0: not yet).
pub(crate) static STORY_MS: AtomicU64 = AtomicU64::new(0);

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Vec<u8>,
    /// Pipelines stored since the last save.
    added: u64,
}

// SAFETY: D3D12 objects are free-threaded; every call is made under the
// mutex that holds the library.
unsafe impl Send for Library {}

const LIBRARY_FILE: &str = "host-pipelines.bin";
/// The pipeline list's file name (not versioned).
const LIST_FILE: &str = "pipeline-list.txt";
/// Lists of earlier builds (one per translator version), imported once.
const OLD_PREFIX: &str = "early-pipelines";
/// Title pipelines listed for an early start, at most (sized for full
/// open-world pipeline counts while keeping memory bounded).
const EARLY_MAX: usize = 65_536;
/// Set whenever a new pipeline is stored into the library since the last save.
static LIBRARY_DIRTY: AtomicBool = AtomicBool::new(false);
/// Prevents overlapping periodic saves from queuing redundant work.
static SAVE_IN_FLIGHT: AtomicBool = AtomicBool::new(false);

fn library() -> Option<&'static Mutex<Library>> {
    static LIBRARY: OnceLock<Option<Mutex<Library>>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            let device = super::super::host::host_device()?.device.cast::<ID3D12Device1>().ok()?;
            let blob = DIR
                .get()
                .and_then(|d| std::fs::read(d.join(LIBRARY_FILE)).ok())
                .unwrap_or_default();
            let library = open(&device, blob);
            if library.is_none() {
                xnw_core::warn!("d3d12x: no host pipeline library: pipelines are not kept");
            }
            library.map(Mutex::new)
        })
        .as_ref()
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
fn open(device: &ID3D12Device1, blob: Vec<u8>) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // SAFETY: the bytes outlive the library (kept beside it).
        unsafe {
            device
                .CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?
                .cast()
        }
    };
    if !blob.is_empty() {
        match make(&blob) {
            Ok(library) => {
                xnw_core::info!("d3d12x: host pipeline library loaded ({} bytes)", blob.len());
                return Some(Library {
                    library,
                    _blob: blob,
                    added: 0,
                });
            }
            Err(e) => {
                xnw_core::info!("d3d12x: host pipeline library rejected ({e}); starting an empty one")
            }
        }
    } else {
        xnw_core::info!("d3d12x: no host pipeline library yet; starting an empty one");
    }
    let library = make(&[]).ok()?;
    Some(Library {
        library,
        _blob: Vec::new(),
        added: 0,
    })
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let parts: Vec<&[u8]> = [kind.as_bytes(), root].into_iter().chain(code.iter().copied()).chain([state.as_bytes()]).collect();
    // hash128 length-prefixes each part, so parts do not run into each other.
    format!("{:032x}", xnw_core::hash128(&parts))
}

/// Pipelines a driver's compiler crashed on (names, one per line).
const CRASHED_FILE: &str = "crashed.txt";

/// Pipelines the driver crashed on, or never ended, in this run.
static CRASHED_NOW: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// The pipeline named `name` is left out from now on, and from the next
/// run on.
fn note_crashed(name: &str) {
    use std::io::Write;
    CRASHED_NOW.lock().unwrap_or_else(|e| e.into_inner()).push(name.to_owned());
    let Some(dir) = DIR.get() else { return };
    let file = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(CRASHED_FILE));
    if let Ok(mut f) = file {
        let _ = writeln!(f, "{name}");
    }
}

/// Whether the driver crashed making the pipeline named `name` before.
fn crashed(name: &str) -> bool {
    static NAMES: OnceLock<std::collections::HashSet<String>> = OnceLock::new();
    NAMES
        .get_or_init(|| {
            DIR.get()
                .and_then(|d| std::fs::read_to_string(d.join(CRASHED_FILE)).ok())
                .map(|t| t.lines().map(str::to_owned).collect())
                .unwrap_or_default()
        })
        .contains(name)
        || CRASHED_NOW.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|n| n == name)
}

thread_local! {
    /// The pipeline the driver makes on this thread.
    static MAKING: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

/// The driver makes a pipeline on this thread. If its compiler crashes there
/// (AMD's does on a few of the world's compute shaders), the pipeline is
/// noted and left out from then on, and the thread parked: the others, and
/// the title, go on.
struct Making;

impl Making {
    fn new(name: &str) -> Making {
        static HANDLER: std::sync::Once = std::sync::Once::new();
        // SAFETY: a handler that lives as long as the process.
        HANDLER.call_once(|| unsafe {
            xnw_core::win::AddVectoredExceptionHandler(1, compiler_crashed);
        });
        MAKING.with(|m| name.clone_into(&mut m.borrow_mut()));
        endless(name, true);
        Making
    }
}

/// The process's private bytes.
fn committed() -> u64 {
    use xnw_core::win::{GetCurrentProcess, K32GetProcessMemoryInfo, ProcessMemoryCounters};
    // SAFETY: a structure of the size passed, all integers.
    unsafe {
        let mut info: ProcessMemoryCounters = std::mem::zeroed();
        info.cb = size_of_val(&info) as u32;
        if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut info, info.cb) == 0 {
            return 0;
        }
        info.pagefile_usage as u64
    }
}

/// Pipeline workers lost to compiles that crashed or never ended. They are
/// not replaced: the others take the queue.
static LOST: AtomicU64 = AtomicU64::new(0);

/// Suspends thread `id`, at work in the driver's compiler (never in a heap
/// function, whose lock it would keep); false if it was not caught there.
fn suspend(id: u32) -> bool {
    use xnw_core::win::{CloseHandle, Context, GetModuleHandleExW, GetThreadContext, OpenThread, ResumeThread, SuspendThread};
    const THREAD_SUSPEND_RESUME: u32 = 0x2;
    const THREAD_GET_CONTEXT: u32 = 0x8;
    const CONTEXT_CONTROL_AMD64: u32 = 0x10_0001;
    const CONTEXT_FLAGS: usize = 0x30;
    const FROM_ADDRESS: u32 = 0x4;
    const UNCHANGED_REFCOUNT: u32 = 0x2;
    // SAFETY: plain calls on a thread of ours; the context is whole and
    // aligned; nothing that may need the suspended thread's locks is called
    // while it is suspended (the modules are asked about after it runs on).
    unsafe {
        let thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, 0, id);
        if thread.is_null() {
            return false;
        }
        let module = |at: usize, flags: u32| {
            let mut m = std::ptr::null_mut();
            let _ = GetModuleHandleExW(flags | UNCHANGED_REFCOUNT, at as *const u16, &mut m);
            m as usize
        };
        let named = |s: &str| {
            let wide: Vec<u16> = s.encode_utf16().chain([0]).collect();
            module(wide.as_ptr() as usize, 0)
        };
        let (ntdll, ucrt) = (named("ntdll.dll"), named("ucrtbase.dll"));
        let mut caught = false;
        let mut last = Vec::new();
        for _ in 0..500 {
            if SuspendThread(thread) == u32::MAX {
                break;
            }
            let mut context = Context::zeroed();
            context.0[CONTEXT_FLAGS..CONTEXT_FLAGS + 4].copy_from_slice(&CONTEXT_CONTROL_AMD64.to_le_bytes());
            let known = GetThreadContext(thread, &mut context) != 0;
            let at = context.rip() >> 12;
            if known && last.contains(&at) {
                caught = true;
                break;
            }
            ResumeThread(thread);
            // A page of the driver's own code: caught there next time.
            let m = module(at << 12, FROM_ADDRESS);
            if known && m != 0 && m != ntdll && m != ucrt {
                last.push(at);
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let _ = CloseHandle(thread);
        caught
    }
}

/// Notes pipelines the driver's compiler has worked on for long: they are
/// left out from then on. AMD's never ends some programs, and takes memory
/// all the while (16 GB in two minutes), so their threads are stopped.
fn endless(name: &str, begins: bool) {
    use std::time::{Duration, Instant};
    // Some long programs take a while; what takes memory all along goes
    // first.
    const ENDLESS: Duration = Duration::from_secs(20);
    const EVEN_SO: Duration = Duration::from_secs(150);
    const GROWTH: u64 = 3 << 30;
    static AT_WORK: Mutex<Vec<(String, Instant, u32)>> = Mutex::new(Vec::new());
    static WATCH: std::sync::Once = std::sync::Once::new();
    {
        let mut at_work = AT_WORK.lock().unwrap_or_else(|e| e.into_inner());
        if begins {
            // SAFETY: a plain call.
            let id = unsafe { xnw_core::win::GetCurrentThreadId() };
            at_work.push((name.to_owned(), Instant::now(), id));
        } else if let Some(i) = at_work.iter().position(|w| w.0 == name) {
            at_work.swap_remove(i);
        }
    }
    WATCH.call_once(|| {
        let _ = std::thread::Builder::new().name("xnw-compiles".into()).spawn(|| {
            let mut noted: Vec<String> = Vec::new();
            // The process's commit when a compile was first seen late.
            let mut late: Option<u64> = None;
            let older = |than: Duration, noted: &[String]| -> Vec<(String, u32)> {
                AT_WORK
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .iter()
                    .filter(|w| w.1.elapsed() > than && !noted.contains(&w.0))
                    .map(|w| (w.0.clone(), w.2))
                    .collect()
            };
            loop {
                std::thread::sleep(Duration::from_secs(2));
                let long = older(ENDLESS, &noted);
                if long.is_empty() {
                    late = None;
                    continue;
                }
                let now = committed();
                let grown = now.saturating_sub(*late.get_or_insert(now)) > GROWTH;
                let long = if grown {
                    late = None;
                    long
                } else {
                    older(EVEN_SO, &noted)
                };
                for (name, id) in long {
                    let stopped = suspend(id);
                    if stopped {
                        LOST.fetch_add(1, Ordering::Relaxed);
                    }
                    xnw_core::warn!(
                        "d3d12x: the driver's compiler does not end pipeline {name} (thread stopped: {stopped}, workers lost: {}): it is left out from now on",
                        LOST.load(Ordering::Relaxed)
                    );
                    note_crashed(&name);
                    noted.push(name);
                }
            }
        });
    });
}

impl Drop for Making {
    fn drop(&mut self) {
        let name = MAKING.with(|m| std::mem::take(&mut *m.borrow_mut()));
        endless(&name, false);
    }
}

unsafe extern "system" fn compiler_crashed(info: *mut xnw_core::win::ExceptionPointers) -> i32 {
    const ACCESS_VIOLATION: u32 = 0xc000_0005;
    // SAFETY: the system passes a live record.
    let record = unsafe { &*(*info).record };
    let name = MAKING.try_with(|m| m.try_borrow().map(|m| m.clone()).unwrap_or_default());
    let name = name.unwrap_or_default();
    // Only reads or writes of the first pages: our own guards' faults are
    // elsewhere.
    if record.code != ACCESS_VIOLATION || name.is_empty() || record.params[1] >= 0x10000 {
        return xnw_core::win::EXCEPTION_CONTINUE_SEARCH;
    }
    note_crashed(&name);
    endless(&name, false);
    LOST.fetch_add(1, Ordering::Relaxed);
    xnw_core::warn!(
        "d3d12x: the driver's compiler crashed making pipeline {name} (at {:#x}, workers lost: {}): it is left out",
        record.address,
        LOST.load(Ordering::Relaxed)
    );
    loop {
        std::thread::park();
    }
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored.
pub(crate) fn pipeline(
    name: &str,
    load: impl FnOnce(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    if crashed(name) {
        // E_FAIL.
        return Err(windows::core::Error::from_hresult(windows::core::HRESULT(0x8000_4005_u32 as i32)));
    }
    let create = || {
        let _making = Making::new(name);
        create()
    };
    let wide = HSTRING::from(name);
    let Some(library) = library() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    if let Ok(pso) = load(
        &library.lock().unwrap_or_else(|e| e.into_inner()).library,
        PCWSTR(wide.as_ptr()),
    ) {
        LOADED.fetch_add(1, Ordering::Relaxed);
        return Ok(pso);
    }
    // Compiled outside the lock: compiles take up to seconds.
    let pso = match create() {
        Ok(pso) => pso,
        Err(e) => {
            // DXGI_ERROR_DEVICE_REMOVED / _HUNG / _RESET: what was in
            // flight becomes a suspect.
            if matches!(e.code().0 as u32, 0x887a_0005 | 0x887a_0006 | 0x887a_0007) {
                super::compile::device_lost(&format!("host device removed ({:#010x})", e.code().0 as u32));
            }
            return Err(e);
        }
    };
    MADE.fetch_add(1, Ordering::Relaxed);
    let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: a live pipeline; a name already stored (a description that
    // differs) is refused, which leaves the library as it was.
    if unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
        l.added += 1;
        ADDED_SINCE_SAVE.fetch_add(1, Ordering::Relaxed);
        LIBRARY_DIRTY.store(true, Ordering::Relaxed);
    }
    Ok(pso)
}

/// Pipelines stored since the last save that reached the disk (a save
/// is due at [`SAVE_AFTER`] of them).
static ADDED_SINCE_SAVE: AtomicU64 = AtomicU64::new(0);

/// Pipelines stored that are not on disk yet (0 after a save that
/// wrote them all).
pub(crate) fn unsaved() -> u64 {
    ADDED_SINCE_SAVE.load(Ordering::Relaxed)
}
/// A save is due after this many new pipelines, or [`SAVE_EVERY`].
const SAVE_AFTER: u64 = 50;
const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(5);

struct SaveAck {
    done: Mutex<bool>,
    cv: Condvar,
}

impl SaveAck {
    fn new() -> Self {
        Self {
            done: Mutex::new(false),
            cv: Condvar::new(),
        }
    }

    fn signal(&self) {
        let mut done = self.done.lock().unwrap_or_else(|e| e.into_inner());
        *done = true;
        self.cv.notify_all();
    }

    /// Waits for the save, at most `limit` (a failing run must not hang
    /// on a saver blocked by a thread that died holding the library).
    fn wait(&self, limit: std::time::Duration) -> bool {
        let until = std::time::Instant::now() + limit;
        let mut done = self.done.lock().unwrap_or_else(|e| e.into_inner());
        while !*done {
            let left = until.saturating_duration_since(std::time::Instant::now());
            if left.is_zero() {
                return false;
            }
            done = self.cv.wait_timeout(done, left).unwrap_or_else(|e| e.into_inner()).0;
        }
        true
    }
}

struct SaveRequest {
    dir: PathBuf,
    ack: Option<Arc<SaveAck>>,
}

/// Dedicated background saver thread so periodic saves never block PSO
/// compilation workers on serialization or disk I/O, while shutdown (`now =
/// true`) can still wait for a clean flush.
fn saver() -> Option<&'static mpsc::Sender<SaveRequest>> {
    static SAVER: OnceLock<Option<mpsc::Sender<SaveRequest>>> = OnceLock::new();
    SAVER
        .get_or_init(|| {
            let (tx, rx) = mpsc::channel::<SaveRequest>();
            let spawned = std::thread::Builder::new()
                .name("xnw-pipeline-save".into())
                .spawn(move || {
                    // SAFETY: plain priority call on our own background thread.
                    unsafe {
                        use windows::Win32::System::Threading::*;
                        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
                    }
                    let mut buf = Vec::new();
                    while let Ok(req) = rx.recv() {
                        perform_save(&req.dir, &mut buf);
                        SAVE_IN_FLIGHT.store(false, Ordering::Release);
                        if let Some(ack) = req.ack {
                            ack.signal();
                        }
                    }
                });
            match spawned {
                Ok(_) => Some(tx),
                Err(e) => {
                    xnw_core::warn!("d3d12x: no pipeline saver thread: {e}");
                    None
                }
            }
        })
        .as_ref()
}

/// Serializes the library into `buf` (growing `buf` outside `library.lock()` so
/// multi-megabyte page faults never stall PSO workers), returning the serialized
/// byte length and count of newly stored pipelines.
fn serialize_library(library: &Mutex<Library>, buf: &mut Vec<u8>) -> Option<(usize, u64)> {
    let needed = {
        let l = library.lock().unwrap_or_else(|e| e.into_inner());
        if l.added == 0 {
            LIBRARY_DIRTY.store(false, Ordering::Relaxed);
            return None;
        }
        // SAFETY: plain size query on a live pipeline library under its mutex.
        unsafe { l.library.GetSerializedSize() }
    };
    if needed == 0 {
        return None;
    }
    if buf.len() < needed {
        buf.resize(needed.saturating_add(needed / 16).max(needed), 0);
    }
    loop {
        let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
        if l.added == 0 {
            LIBRARY_DIRTY.store(false, Ordering::Relaxed);
            return None;
        }
        // SAFETY: plain size query on a live pipeline library under its mutex.
        let exact = unsafe { l.library.GetSerializedSize() };
        if exact > buf.len() {
            drop(l);
            buf.resize(exact.saturating_add(exact / 16).max(exact), 0);
            continue;
        }
        // SAFETY: `buf[..exact]` has the exact size requested by `GetSerializedSize`.
        match unsafe { l.library.Serialize(&mut buf[..exact]) } {
            Ok(()) => {
                let added = std::mem::take(&mut l.added);
                LIBRARY_DIRTY.store(false, Ordering::Relaxed);
                return Some((exact, added));
            }
            Err(e) => {
                xnw_core::warn!("d3d12x: host pipeline library not serialised: {e}");
                return None;
            }
        }
    }
}

/// Performs the actual snapshot + unlocked disk I/O for `host-pipelines.bin`
/// and `early-pipelines.txt`.
fn perform_save(dir: &Path, buf: &mut Vec<u8>) {
    let Some(library) = library() else { return };
    let serialized = serialize_library(library, buf);
    let used_snapshot = USED_CHANGED.swap(false, Ordering::AcqRel).then(early_text);
    if serialized.is_none() && used_snapshot.is_none() {
        return;
    }

    if let Err(e) = std::fs::create_dir_all(dir) {
        if let Some((_, added)) = serialized {
            library.lock().unwrap_or_else(|e| e.into_inner()).added += added;
            LIBRARY_DIRTY.store(true, Ordering::Relaxed);
        }
        if used_snapshot.is_some() {
            USED_CHANGED.store(true, Ordering::Relaxed);
        }
        xnw_core::warn!("d3d12x: host pipeline library directory not created: {e}");
        return;
    }

    let mut lib_bytes = 0usize;
    let mut ok = true;
    if let Some((len, added)) = serialized {
        if let Err(e) = write_atomically(&dir.join(LIBRARY_FILE), &buf[..len]) {
            library.lock().unwrap_or_else(|e| e.into_inner()).added += added;
            LIBRARY_DIRTY.store(true, Ordering::Relaxed);
            xnw_core::warn!("d3d12x: host pipeline library not saved: {e}");
            ok = false;
        } else {
            lib_bytes = len;
            let _ = ADDED_SINCE_SAVE.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| Some(n.saturating_sub(added)));
        }
    }
    let mut early_count = 0usize;
    if let Some(used) = used_snapshot {
        early_count = used.lines().count().saturating_sub(1);
        if let Err(e) = write_atomically(&dir.join(LIST_FILE), used.as_bytes()) {
            USED_CHANGED.store(true, Ordering::Relaxed);
            xnw_core::warn!("d3d12x: early pipeline list not saved: {e}");
            ok = false;
        }
    }
    if ok {
        xnw_core::info!(
            "d3d12x: host pipeline library saved: {lib_bytes} bytes, {early_count} early, {} made, {} loaded, {} recorded",
            MADE.load(Ordering::Relaxed),
            LOADED.load(Ordering::Relaxed),
            RECORDED.load(Ordering::Relaxed)
        );
    }
}

/// Writes the library and the early list if pipelines were added since the
/// last save and [`SAVE_EVERY`] has passed or [`SAVE_AFTER`] pipelines were
/// added (or `now`). Periodic saves run asynchronously on the dedicated
/// saver thread; `now = true` waits for disk completion (at most 5 s). The
/// file is written to a temporary name and renamed: a run that dies while
/// saving leaves the last whole file.
pub(crate) fn save(now: bool) {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let Some(dir) = DIR.get() else { return };
    if !now && !LIBRARY_DIRTY.load(Ordering::Relaxed) && !USED_CHANGED.load(Ordering::Relaxed) {
        return;
    }
    if !now && SAVE_IN_FLIGHT.load(Ordering::Acquire) {
        return;
    }
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) && ADDED_SINCE_SAVE.load(Ordering::Relaxed) < SAVE_AFTER {
            return;
        }
        if !now && SAVE_IN_FLIGHT.swap(true, Ordering::AcqRel) {
            return;
        }
        *last = Some(std::time::Instant::now());
    }
    if let Some(tx) = saver() {
        let ack = now.then(|| Arc::new(SaveAck::new()));
        let req = SaveRequest {
            dir: dir.clone(),
            ack: ack.clone(),
        };
        if tx.send(req).is_ok() {
            if let Some(ack) = ack
                && !ack.wait(std::time::Duration::from_secs(5))
            {
                xnw_core::warn!("d3d12x: host pipeline library save not finished within 5 s");
            }
            return;
        }
    }
    // Fallback if no background saver thread could be created.
    let mut buf = Vec::new();
    perform_save(dir, &mut buf);
    SAVE_IN_FLIGHT.store(false, Ordering::Release);
}

/// A failure ends the run (device removed, an exception): what was
/// built goes to disk now, waited for at most 5 s.
pub(crate) fn flush(why: &str) {
    let pending = ADDED_SINCE_SAVE.load(Ordering::Relaxed);
    if DIR.get().is_some() && (pending > 0 || USED_CHANGED.load(Ordering::Relaxed)) {
        xnw_core::info!("d3d12x: host pipeline library flushed ({why}): {pending} pipelines not saved yet");
        save(true);
    }
}

/// Writes `bytes` to a unique temporary file beside `path`, then renames it
/// over `path`, removing the temporary file if either write or rename fails.
fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("cache");
    let tmp = path.with_file_name(format!(".{file_name}.tmp-{}-{n}", std::process::id()));
    if let Err(err) = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path)) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err);
    }
    Ok(())
}

/// One listed pipeline: first use (ms) and its recipe, if known.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Row {
    pub ms: i64,
    pub recipe: Option<String>,
}

/// The early list: the keys an earlier run listed with their first-use
/// time (relative to its story load start), and this run's keys with their
/// first-use time since the start (made relative at saving).
#[derive(Default)]
struct Early {
    listed: HashMap<String, Row>,
    new: HashMap<String, u64>,
    /// Recipes learnt this run (for listed and new keys).
    recipes: HashMap<String, String>,
}

static USED: Mutex<Option<Early>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used() -> std::sync::MutexGuard<'static, Option<Early>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let (listed, imported) = DIR.get().map(|d| load_list(d)).unwrap_or_default();
        if imported > 0 {
            xnw_core::info!("d3d12x: pipeline list imported from {imported} older lists ({} pipelines, no recipes)", listed.len());
            USED_CHANGED.store(true, Ordering::Relaxed);
        }
        let recipes = listed.values().filter(|r| r.recipe.is_some()).count();
        xnw_core::info!("d3d12x: {} title pipelines listed for prebuild, {recipes} with recipes (translator {})", listed.len(), xnw_shader::emit::VERSION);
        *used = Some(Early { listed, ..Early::default() });
    }
    used
}

/// The list in `dir`; without one, the older lists merged (oldest first, so
/// a newer time wins), with how many were imported.
fn load_list(dir: &Path) -> (HashMap<String, Row>, usize) {
    if let Ok(text) = std::fs::read_to_string(dir.join(LIST_FILE)) {
        return (parse(&text), 0);
    }
    let mut old: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with(OLD_PREFIX))
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect();
    old.sort();
    let mut all = HashMap::new();
    for (_, p) in &old {
        all.extend(parse(&std::fs::read_to_string(p).unwrap_or_default()));
    }
    all.shrink_to_fit();
    (all, old.len())
}

/// The rows of a list (header and damaged lines skipped).
fn parse(text: &str) -> HashMap<String, Row> {
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let mut f = l.split('\t');
            let (key, ms) = (f.next()?, f.next()?);
            let recipe = f.next().map(str::trim).filter(|r| !r.is_empty()).map(str::to_owned);
            (key.len() == 32).then_some((key.to_owned(), Row { ms: ms.trim().parse().ok()?, recipe }))
        })
        .take(EARLY_MAX)
        .collect()
}

/// The list as saved: this run's keys merged over the listed ones, by time,
/// with the recipes learnt this run over the listed ones.
fn early_text() -> String {
    let story = STORY_MS.load(Ordering::Relaxed) as i64;
    let used = used();
    let early = used.as_ref().expect("loaded by used()");
    list_text(&merged(early, story), story)
}

fn merged(early: &Early, story: i64) -> HashMap<String, Row> {
    let mut all = early.listed.clone();
    for (key, ms) in &early.new {
        if all.len() < EARLY_MAX || all.contains_key(key) {
            let recipe = all.get(key).and_then(|r| r.recipe.clone());
            all.insert(key.clone(), Row { ms: *ms as i64 - story, recipe });
        }
    }
    for (key, recipe) in &early.recipes {
        if let Some(row) = all.get_mut(key) {
            row.recipe = Some(recipe.clone());
        }
    }
    all
}

fn list_text(all: &HashMap<String, Row>, story: i64) -> String {
    let mut rows: Vec<(&String, &Row)> = all.iter().collect();
    rows.sort_by_key(|(k, r)| (r.ms, (*k).clone()));
    let mut text = format!("# xnw pipeline list v2 translator {} story_ms {story} count {}\n", xnw_shader::emit::VERSION, rows.len());
    for (key, r) in rows {
        match &r.recipe {
            Some(recipe) => text.push_str(&format!("{key}\t{}\t{recipe}\n", r.ms)),
            None => text.push_str(&format!("{key}\t{}\n", r.ms)),
        }
    }
    text
}

/// How many pipelines the list holds (loads it).
pub(crate) fn listed() -> usize {
    used().as_ref().map_or(0, |e| e.listed.len())
}

/// The first-use time (ms, relative to the story load start) the list gives
/// the title pipeline with `key`, if it lists it.
pub(crate) fn wanted(key: &str) -> Option<i64> {
    used().as_ref()?.listed.get(key).map(|r| r.ms)
}

/// The listed pipelines (key, first use, recipe), as loaded.
pub(crate) fn rows() -> Vec<(String, Row)> {
    used().as_ref().map(|e| e.listed.iter().map(|(k, r)| (k.clone(), r.clone())).collect()).unwrap_or_default()
}

/// Whether the list or this run has a recipe for `key`.
pub(crate) fn has_recipe(key: &str) -> bool {
    used().as_ref().is_some_and(|e| e.recipes.contains_key(key) || e.listed.get(key).is_some_and(|r| r.recipe.is_some()))
}

/// The recipe of a pipeline made this run, kept if the list has or
/// will have its key and it has none yet.
pub(crate) fn note_recipe(key: &str, recipe: String) {
    let mut used = used();
    let early = used.get_or_insert_with(Early::default);
    let known = early.listed.get(key).is_some_and(|r| r.recipe.as_deref() == Some(recipe.as_str()));
    if !known && early.recipes.insert(key.to_owned(), recipe).is_none() {
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

/// The story load starts now (the times of the list are relative to it).
pub(crate) fn story_started() {
    let ms = xnw_core::uptime().as_millis() as u64;
    let _ = STORY_MS.compare_exchange(0, ms.max(1), Ordering::Relaxed, Ordering::Relaxed);
}

/// Inserts `key` into `new` if the list is below [`EARLY_MAX`], returning
/// `true` if newly added.
fn record_used(new: &mut HashMap<String, u64>, listed: usize, key: &str, ms: u64) -> bool {
    if listed + new.len() < EARLY_MAX && !new.contains_key(key) {
        new.insert(key.to_owned(), ms);
        true
    } else {
        false
    }
}

/// The title pipeline with `key` was drawn or dispatched with (first time
/// in this run).
pub(crate) fn note_used(key: &str) {
    let ms = xnw_core::uptime().as_millis() as u64;
    let mut used = used();
    let early = used.get_or_insert_with(Early::default);
    if record_used(&mut early.new, early.listed.len(), key, ms) {
        RECORDED.fetch_add(1, Ordering::Relaxed);
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    /// A compile that faults on the first pages parks its thread and leaves
    /// its pipeline out; asking for it again fails at once, without a
    /// compile.
    #[test]
    fn a_compile_that_crashes_is_parked_and_left_out() {
        let name = "test: a compile that crashes";
        // The guard `pipeline` puts around a compile (no device here).
        let _parked = std::thread::spawn(move || {
            let _making = Making::new(name);
            // SAFETY: none: the fault is the test (the guard parks this
            // thread before anything reads the value).
            let x = unsafe { std::ptr::read_volatile(0x18 as *const u32) };
            panic!("read {x} from page 0")
        });
        let started = std::time::Instant::now();
        while !CRASHED_NOW.lock().unwrap().iter().any(|n| n == name) {
            assert!(started.elapsed().as_secs() < 10, "the crash was not noted");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(LOST.load(Ordering::Relaxed) >= 1);
        let again = pipeline(name, |_, _| unreachable!("no load"), || unreachable!("no second compile"));
        assert_eq!(again.err().map(|e| e.code().0 as u32), Some(0x8000_4005));
    }

    /// A thread at work in code outside the system's modules is caught and
    /// stays suspended.
    #[test]
    fn a_busy_thread_is_suspended() {
        let (tx, rx) = std::sync::mpsc::channel();
        let _busy = std::thread::spawn(move || {
            // SAFETY: a plain call.
            tx.send(unsafe { xnw_core::win::GetCurrentThreadId() }).unwrap();
            let mut n = 0u64;
            loop {
                n = std::hint::black_box(n.wrapping_add(1));
            }
        });
        let id = rx.recv().unwrap();
        assert!(suspend(id), "the busy thread was not caught");
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        if std::env::var_os("XNW_AGILITY").is_none() {
            return; // runs on the hardware device only
        }
        let Some(device) = super::super::super::host::host_device().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
        else {
            return; // no host GPU here
        };
        let library =
            open(&device, b"not a pipeline library".repeat(16)).expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(library._blob.is_empty());
    }

    #[test]
    fn early_list_records_deduplicates_and_parses() {
        const { assert!(EARLY_MAX >= 65_536) };
        let mut new = HashMap::new();
        assert!(record_used(&mut new, 0, "a", 5));
        assert!(!record_used(&mut new, 0, "a", 9));
        assert_eq!(new["a"], 5);
        for i in 0..EARLY_MAX {
            record_used(&mut new, 0, &format!("k{i}"), 1);
        }
        assert_eq!(new.len(), EARLY_MAX, "keys beyond EARLY_MAX are ignored");
        let key = "0123456789abcdef0123456789abcdef";
        let list = parse(&format!("# header\n{key}\t-250\nshort\t3\n{key}x\t1\n{key}\n"));
        assert_eq!(list.len(), 1);
        assert_eq!(list[key], Row { ms: -250, recipe: None });
    }

    /// The list survives translator versions: older versioned lists are
    /// imported once (newer time wins), recipes learnt in a run are saved
    /// and read back, and the saved list only grows.
    #[test]
    fn list_imports_old_lists_and_keeps_recipes() {
        let dir = std::env::temp_dir().join(format!("xnw-test-pipeline-list-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (a, b, c) = ("a".repeat(32), "b".repeat(32), "c".repeat(32));
        std::fs::write(dir.join("early-pipelines-v11.txt"), format!("# v1\n{a}\t100\n{b}\t200\n")).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(30));
        std::fs::write(dir.join("early-pipelines-v13.txt"), format!("# v1\n{b}\t50\n")).unwrap();
        let (listed, imported) = load_list(&dir);
        assert_eq!(imported, 2);
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[&b].ms, 50, "the newer list's time wins");

        // A run: c is new, a gets its recipe.
        let early = Early { listed, new: HashMap::from([(c.clone(), 1500)]), recipes: HashMap::from([(a.clone(), "vertex:0:1:1;-".to_owned()), ("d".repeat(32), "x".into())]) };
        let all = merged(&early, 1000);
        assert_eq!(all.len(), 3, "a recipe of an unlisted key adds no row");
        assert_eq!(all[&c].ms, 500);
        std::fs::write(dir.join(LIST_FILE), list_text(&all, 1000)).unwrap();

        // The next load reads the list, not the old ones, recipes included.
        let (again, imported) = load_list(&dir);
        assert_eq!(imported, 0);
        assert_eq!(again, all);
        assert_eq!(again[&a].recipe.as_deref(), Some("vertex:0:1:1;-"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
