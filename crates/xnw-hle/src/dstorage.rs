//! The console's DirectStorage (`XCRDAPI.dll`): a factory that opens files
//! and creates request queues and status arrays the title polls.
//!
//! Adapted from VIctory, `crates/victory-hle/src/modules/dstorage.rs`
//! (GPL-3.0-or-later): the interface ids, slot numbers and request layout
//! at d2a4bae; the scheduler, the request execution and the BCPack path at
//! main 58fed37 (the file as of 2849477). Submitted requests of every queue
//! run on one pool of workers: in parallel, started in queue order, queues
//! served by weighted round robin over their priorities (about 10x per
//! level, counted in bytes), with at most the staging buffer's worth of
//! compressed source in flight, and a full queue blocks the caller. Each
//! request reads its source (a file or memory), inflates zlib data and/or
//! unpacks BCPack data (`xnw_formats::bcpack`, VIctory's decoder), and
//! writes the destination (an address, or physical pages). Completion is
//! what stays in order: a status entry is written once every request
//! enqueued before it on its queue has finished, with the first failure (in
//! queue order) since the previous status entry, or success.
//!
//! Ours, kept from before the port:
//! - Nothing runs before `Submit` unless the queue is full (another runtime's
//!   measurement: the title fills a request's page array after enqueueing
//!   it; VIctory submits on its own at half a queue).
//! - A queued read holds a reference on its file (after VIctory).
//! - `Close` returns once everything submitted has run and its status
//!   entries are written (VIctory); VIctory main drops what has not
//!   started.
//! - Read failures set the queue's error event (slot 9) and fill its error
//!   record (slot 10), after VIctory (8632869).
//!
//! zlib requests inflate with `miniz_oxide` (a stream that ends short of the
//! destination passes). BCPack modes 1 and 3-7 decode; another mode, or an
//! option we do not know, fails `E_NOTIMPL`, a stream that does not decode
//! `HRESULT_FROM_WIN32(ERROR_INVALID_DATA)`. Each BCPack failure is counted
//! by mode in the log (the first of each mode, then every power of two).
//! `XNW_TRACE=dstorage` logs every request with its file, offset and sizes,
//! and a summary every 5 s.
//!
//! shortcut: VIctory main also writes page destinations in place through
//! one xmem view (`page_window`, 4e94c2b) and opens the GPU's guard on
//! raw-read destinations; both live outside this file. VIctory's overlapped
//! reads are not taken (it changes xmem too).

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ffi::{CStr, c_char, c_void};
use std::os::windows::fs::FileExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};

use xnw_formats::bcpack::{self, Tables};

use crate::images::caller;
use crate::xgameruntime::xasync::E_PENDING;
use crate::xgameruntime::{E_INVALIDARG, E_NOINTERFACE, E_NOTIMPL, Guid, HRESULT, Object, S_OK, Table, methods, put};

/// The factory interface the title asks for.
const FACTORY: Guid = Guid::new(0x6295_9d11, 0xd45f, 0x4884, [0xa0, 0xa3, 0xf7, 0x86, 0x2e, 0xa2, 0x43, 0xae]);
const NONE: Guid = Guid::new(0, 0, 0, [0; 8]);
const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;
/// `HRESULT_FROM_WIN32(ERROR_HANDLE_EOF)`: the source ends early.
const END_OF_FILE: HRESULT = 0x8007_0026_u32 as i32;
/// `HRESULT_FROM_WIN32(ERROR_INVALID_DATA)`: a bad compressed stream.
const INVALID_DATA: HRESULT = 0x8007_000d_u32 as i32;

type Out = *mut *mut c_void;

static FACTORY_OBJECT: Object = Object::new("dstorage factory", FACTORY, FACTORY, &FACTORY_VTABLE);

pub extern "system" fn DStorageGetFactory(iid: *const Guid, out: Out) -> HRESULT {
    // SAFETY: the title passes a GUID (null tolerated).
    match unsafe { iid.as_ref() } {
        Some(&iid) if iid == FACTORY => {
            xnw_core::info!("DStorageGetFactory from {}", caller());
            put(out, std::ptr::from_ref(&FACTORY_OBJECT).cast_mut().cast())
        }
        iid => {
            xnw_core::warn!("DStorageGetFactory: unknown interface {}; from {}", iid_text(iid.map_or(std::ptr::null(), |g| g)), caller());
            E_NOINTERFACE
        }
    }
}

/// One of our objects with state: the runtime's object header (method
/// table first), a reference count, then `T`.
#[repr(C)]
struct Instance<T> {
    base: Object,
    refs: AtomicU32,
    state: T,
}

impl<T> Instance<T> {
    fn create(area: &'static str, vtable: &'static Table, state: T) -> *mut c_void {
        Box::into_raw(Box::new(Instance { base: Object::new(area, FACTORY, NONE, vtable), refs: AtomicU32::new(1), state })).cast()
    }
}

extern "system" fn add_ref_instance<T>(this: *const Instance<T>) -> u32 {
    // SAFETY: an instance we created and the title still holds.
    unsafe { &*this }.refs.fetch_add(1, Ordering::Relaxed) + 1
}

extern "system" fn release<T>(this: *mut Instance<T>) -> u32 {
    // SAFETY: as above.
    let left = unsafe { &*this }.refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if left == 0 {
        // SAFETY: the last reference; created by `Instance::create`.
        drop(unsafe { Box::from_raw(this) });
    }
    left
}

fn iid_text(iid: *const Guid) -> String {
    // SAFETY: the title passes a GUID (null tolerated).
    unsafe { iid.as_ref() }.map_or_else(|| "null".into(), Guid::to_string)
}

/// `HRESULT_FROM_WIN32` of a host error.
fn from_win32(error: &std::io::Error) -> HRESULT {
    error.raw_os_error().map_or(E_INVALIDARG, |e| (0x8007_0000 | (e as u32 & 0xffff)) as i32)
}

fn text(name: *const c_char) -> String {
    if name.is_null() {
        return String::new();
    }
    // SAFETY: a NUL-terminated name from the title.
    unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned()
}

// The factory.

type Factory = *const Object;

/// Opens `path` for reading; the file object owns the handle.
extern "system" fn open_file(_: Factory, path: *const u16, iid: *const Guid, out: Out) -> HRESULT {
    if path.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title passes a NUL-terminated path.
    let name = unsafe {
        let len = (0..).take_while(|&i| *path.add(i) != 0).count();
        String::from_utf16_lossy(std::slice::from_raw_parts(path, len))
    };
    let Some(host) = crate::files::host_path(&name) else {
        xnw_core::warn!("dstorage: {name} is not a path we serve; from {}", caller());
        return E_INVALIDARG;
    };
    let started = xnw_core::perf::start();
    let opened = std::fs::File::open(&host);
    xnw_core::perf::since(xnw_core::perf::Area::DsOpen, started, 0);
    match opened {
        Ok(file) => {
            xnw_core::info!("dstorage: opened {name} ({}), iid {}", host.display(), iid_text(iid));
            if name.contains("story000") {
                xnw_gpu::d3d12x::pipeline_phase("story load");
            }
            let reads = file_reads(&host);
            crate::texbypass::start();
            crate::texbypass::opened(&name);
            let origin = texture_index(&host);
            put(out, Instance::create("dstorage file", &FILE_VTABLE, File { file, name, reads, origin }))
        }
        Err(error) => {
            xnw_core::warn!("dstorage: opening {name} ({}) failed: {error}; from {}", host.display(), caller());
            from_win32(&error)
        }
    }
}

/// `(capacity, name, iid, out)`.
extern "system" fn create_status_array(_: Factory, capacity: u32, name: *const c_char, iid: *const Guid, out: Out) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    xnw_core::info!("dstorage: status array \"{}\" of {capacity}, iid {}", text(name), iid_text(iid));
    let entries = (0..capacity).map(|_| AtomicI32::new(S_OK)).collect();
    put(out, Instance::create("dstorage status array", &STATUS_VTABLE, StatusArray { entries }))
}

/// The queue's description: as in 2020 (source type, capacity, priority,
/// name), plus a field the title sets to 1.
#[repr(C)]
struct QueueDesc {
    source_type: u64,
    capacity: u16,
    priority: i8,
    name: *const c_char,
    unknown: u64,
}

extern "system" fn create_queue(_: Factory, desc: *const QueueDesc, iid: *const Guid, out: Out) -> HRESULT {
    // SAFETY: the title passes its description (null tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if out.is_null() {
        return E_INVALIDARG;
    }
    xnw_core::info!(
        "dstorage: queue \"{}\" (source {}, capacity {}, priority {}, field {}), iid {}",
        text(desc.name),
        desc.source_type,
        desc.capacity,
        desc.priority,
        desc.unknown,
        iid_text(iid)
    );
    let Some(scheduler) = scheduler() else {
        return E_OUTOFMEMORY;
    };
    // SAFETY: a private, unsignalled, manual-reset event.
    let event = unsafe { xnw_core::win::CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
    let mut lanes = scheduler.lock();
    let id = lanes.next_id;
    lanes.next_id += 1;
    lanes.lanes.insert(id, Lane::new(desc.priority, desc.capacity, event));
    drop(lanes);
    put(out, Instance::create("dstorage queue", &QUEUE_VTABLE, Queue { id, event }))
}

/// The title asks for 32 MiB and stops if this fails.
extern "system" fn set_staging_buffer_size(_: Factory, size: u32) -> HRESULT {
    xnw_core::info!("dstorage: staging buffer size {size:#x}");
    if let Some(scheduler) = scheduler() {
        scheduler.lock().staging = u64::from(size);
    }
    S_OK
}

static FACTORY_VTABLE: Table = methods(&[
    // 0x18: 2020 queue creation; 0x30 CPU affinity; 0x38 debug flags.
    (4, open_file as *const ()),               // 0x20
    (5, create_status_array as *const ()),     // 0x28
    (8, set_staging_buffer_size as *const ()), // 0x40
    (10, create_queue as *const ()),           // 0x50
]);

// Files.

struct File {
    file: std::fs::File,
    /// The title's path, for the log.
    name: String,
    /// What has been read from the host file, for `XNW_TRACE=perf`.
    reads: &'static FileReads,
    /// A texture page file's id in the origin registry
    /// (`xnw_formats::origin`): its completed page reads are recorded there.
    origin: Option<u32>,
}

/// A `.ttpf`'s index, parsed once per host file (`xnw_formats::origin`);
/// `None` for other files or one that does not parse (warned once).
fn texture_index(host: &std::path::Path) -> Option<u32> {
    if !xnw_formats::origin::is_ttpf(host) {
        return None;
    }
    let started = std::time::Instant::now();
    match xnw_formats::origin::registry().open(host) {
        Ok(id) => {
            xnw_core::info!("dstorage: texture index {id} of {} in {:.1} ms", host.display(), started.elapsed().as_secs_f64() * 1e3);
            Some(id)
        }
        Err(why) => {
            xnw_core::warn!("dstorage: {} has no texture index: {why}", host.display());
            None
        }
    }
}

/// Reads, bytes and nanoseconds spent reading, of one host file (all its
/// openings), and whether it lies on the full package (slow storage).
struct FileReads {
    path: PathBuf,
    package: bool,
    counts: [AtomicU64; 3],
    /// Bytes at the last summary.
    logged: AtomicU64,
}

/// Every host file DirectStorage opened, with its reads.
static FILE_READS: Mutex<Vec<&'static FileReads>> = Mutex::new(Vec::new());

fn file_reads(path: &std::path::Path) -> &'static FileReads {
    let mut all = FILE_READS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(f) = all.iter().find(|f| f.path == path) {
        return f;
    }
    let f: &'static FileReads = Box::leak(Box::new(FileReads {
        path: path.to_owned(),
        package: crate::files::on_package(path),
        counts: Default::default(),
        logged: AtomicU64::new(0),
    }));
    all.push(f);
    f
}

/// Counts a file read of `bytes` that took `time`.
fn count_read(request: &Request, bytes: usize, time: std::time::Duration) {
    if !xnw_core::perf::on() {
        return;
    }
    // SAFETY: a file object we created; the queued read holds it.
    let reads = unsafe { &*request.source.cast::<Instance<File>>() }.state.reads;
    let area = if reads.package { xnw_core::perf::Area::DsReadPackage } else { xnw_core::perf::Area::DsRead };
    xnw_core::perf::add(area, time, bytes as u64);
    for (c, v) in reads.counts.iter().zip([1, bytes as u64, time.as_nanos() as u64]) {
        c.fetch_add(v, Ordering::Relaxed);
    }
}

/// `XNW_TRACE=perf`: a line per file read from since the last one.
fn log_file_reads() {
    for f in FILE_READS.lock().unwrap_or_else(|e| e.into_inner()).iter() {
        let [n, bytes, nanos] = [0, 1, 2].map(|i| f.counts[i].load(Ordering::Relaxed));
        if f.logged.swap(bytes, Ordering::Relaxed) == bytes {
            continue;
        }
        xnw_core::log_line(
            "perf",
            format_args!(
                "dstorage: {}{}: {n} reads, {:.1} MB, {:.2} s reading ({:.1} MB/s per busy thread)",
                f.path.display(),
                if f.package { " (full package)" } else { "" },
                bytes as f64 / 1e6,
                nanos as f64 / 1e9,
                bytes as f64 / 1e6 / (nanos as f64 / 1e9).max(1e-9)
            ),
        );
    }
}

/// `XNW_TRACE=perf`: the origin registry's counters.
fn log_origins() {
    let s = xnw_formats::origin::registry().stats();
    if s.recorded + s.lookups == 0 {
        return;
    }
    xnw_core::log_line(
        "perf",
        format_args!(
            "origin: {} page reads recorded, {} lookups ({} hits), {} frame runs, {} address runs; decoded pages {:.1} MB, {} hits {} misses",
            s.recorded,
            s.lookups,
            s.hits,
            s.phys_runs,
            s.va_runs,
            s.cache_bytes as f64 / 1e6,
            s.cache_hits,
            s.cache_misses
        ),
    );
    let b = xnw_formats::origin::registry().bypass_counts();
    if b[2] > 0 {
        let parts: Vec<String> = xnw_formats::origin::BYPASS_COUNTS.iter().zip(b).map(|(n, c)| format!("{n} {c}")).collect();
        xnw_core::log_line("perf", format_args!("texture pages: {}", parts.join(", ")));
    }
    let reasons: Vec<String> = xnw_formats::origin::REASONS.iter().zip(s.reasons).filter(|(n, c)| !n.is_empty() && *c > 0).map(|(n, c)| format!("{n} {c}")).collect();
    xnw_core::log_line("perf", format_args!("origin lookups by outcome: {}", reasons.join(", ")));
}

static FILE_VTABLE: Table = methods(&[(1, add_ref_instance::<File> as *const ()), (2, release::<File> as *const ())]);

// Status arrays.

struct StatusArray {
    entries: Box<[AtomicI32]>,
}

type Array = *mut Instance<StatusArray>;

/// The result of the requests before entry `index`: pending, the first
/// failure, or success.
extern "system" fn status_result(this: Array, index: u32) -> HRESULT {
    // SAFETY: an array we created and the title still holds.
    let entries = &unsafe { &*this }.state.entries;
    entries.get(index as usize).map_or(E_INVALIDARG, |e| e.load(Ordering::Acquire))
}

/// The title polls this (entry 0, a `Sleep(1)` loop).
extern "system" fn is_complete(this: Array, index: u32) -> bool {
    status_result(this, index) != E_PENDING
}

static STATUS_VTABLE: Table = methods(&[
    (1, add_ref_instance::<StatusArray> as *const ()),
    (2, release::<StatusArray> as *const ()),
    (3, is_complete as *const ()),   // 0x18
    (4, status_result as *const ()), // 0x20
]);

// Queues.

/// A queue object: its lane in the scheduler, and its error event (manual
/// reset), set on each failure.
struct Queue {
    id: u64,
    event: xnw_core::win::HANDLE,
}

impl Drop for Queue {
    fn drop(&mut self) {
        close_lane(self.id);
        // SAFETY: our own event; the lane that set it is gone.
        unsafe { xnw_core::win::CloseHandle(self.event) };
    }
}

type QueueRef = *const Instance<Queue>;

/// The console's error record: a failure count, then the first failure.
/// The layout is Microsoft's (`dstorage_win32.h`), as VIctory
/// gives it, 0x290 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
struct ErrorRecord {
    failure_count: u32,
    _pad: u32,
    first: ReadFailure,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ReadFailure {
    result: HRESULT,
    /// 0: a request; -1: none.
    command_type: i32,
    options: u64,
    destination: usize,
    destination_size: u32,
    filename: [u16; 260],
    _pad: u32,
    file_offset: u64,
    source: usize,
    source_size: u32,
    _pad2: u32,
    tag: u64,
    name: [u8; 64],
}

const _: () = {
    assert!(size_of::<ErrorRecord>() == 0x290);
    assert!(std::mem::offset_of!(ReadFailure, filename) == 0x1c);
    assert!(std::mem::offset_of!(ReadFailure, file_offset) == 0x228);
    assert!(std::mem::offset_of!(ReadFailure, name) == 0x248);
};

impl Default for ErrorRecord {
    fn default() -> Self {
        // SAFETY: plain integers; then the "none" command type.
        let mut record: Self = unsafe { std::mem::zeroed() };
        record.first.command_type = -1;
        record
    }
}

impl ReadFailure {
    fn new(request: &Request, result: HRESULT) -> Self {
        // SAFETY: plain integers.
        let mut failure: Self = unsafe { std::mem::zeroed() };
        failure.result = result;
        failure.options = request.options;
        failure.destination = request.destination as usize;
        failure.destination_size = request.destination_size;
        failure.source_size = request.source_size;
        failure.tag = request.tag;
        if request.options & MEMORY_SOURCE != 0 {
            failure.source = request.source as usize;
        } else {
            failure.file_offset = request.offset;
            // SAFETY: a file object we created, or null.
            if let Some(file) = unsafe { request.source.cast::<Instance<File>>().as_ref() } {
                for (to, from) in failure.filename[..259].iter_mut().zip(file.state.name.encode_utf16()) {
                    *to = from;
                }
            }
        }
        for (to, from) in failure.name[..63].iter_mut().zip(text(request.name).bytes()) {
            *to = from;
        }
        failure
    }
}

/// Every queue's requests, run by a pool of workers.
struct Scheduler {
    lanes: Mutex<Lanes>,
    /// Requests to start, or staging space freed.
    work: Condvar,
    /// A request finished (for full queues and `close`).
    finished: Condvar,
}

/// Priority weights per class (low, normal, high and real-time), as the
/// console's: about 10x per level, counted against request sizes.
const COST: [u64; 3] = [100, 10, 1];
/// The console's default staging buffer.
const STAGING: u64 = 32 << 20;
/// Reads from the full package (the last `G:` folder, a hard disk here)
/// run at most this many at once. Ours: in the story load all 12 workers
/// sat in reads of `levels.rpf` on the disk (3-4 MB/s each, 288 reads
/// waiting) while requests on fast storage waited behind them (L1 run
/// 0001); the disk's own queue gains little past a few.
const PACKAGE_READERS: usize = 4;

/// Whether `request` reads a file on the full package.
fn on_package(request: &Request) -> bool {
    // SAFETY: a file object we created; the queued read holds it.
    file_of(request).is_some_and(|f| unsafe { &*f }.state.reads.package)
}

struct Lanes {
    lanes: HashMap<u64, Lane>,
    next_id: u64,
    /// Submission order over all queues.
    next_seq: u64,
    /// Bytes of staging held by running requests, and the limit.
    staged: u64,
    staging: u64,
    /// A start waited for staging space (or a package reader) since the
    /// last finish.
    starved: bool,
    /// Reads from the full package running.
    package_reads: usize,
    /// Weighted round robin: each class's virtual time (bytes × cost), and
    /// the time of the last start; within a class, the lane clock of the
    /// last start.
    clock: [u64; 3],
    now: u64,
    lane_now: [u64; 3],
    /// Since the last summary: requests, bytes written, time spent running
    /// them.
    done: (u64, u64, std::time::Duration),
    refused: u64,
    /// BCPack bytes written and time spent on them.
    bcpack: (u64, std::time::Duration),
    since: Option<std::time::Instant>,
}

/// A queue's requests in the scheduler.
struct Lane {
    class: usize,
    capacity: usize,
    /// Bytes started, for round robin among the lanes of a class.
    clock: u64,
    /// Enqueued, not yet submitted.
    unsubmitted: Vec<Command>,
    /// Submitted, by sequence number: reads not started, status entries
    /// not written, reads running, and failures not yet reported.
    reads: VecDeque<(u64, Read)>,
    statuses: VecDeque<(u64, Status)>,
    running: BTreeSet<u64>,
    failures: BTreeMap<u64, HRESULT>,
    closed: bool,
    /// The queue's error event (owned by the queue object) and record.
    event: xnw_core::win::HANDLE,
    record: ErrorRecord,
}

// SAFETY: requests point into title memory and our own objects, which stay
// valid until the title has seen them complete; status arrays and files
// count their references atomically.
unsafe impl Send for Lane {}

/// A read a worker runs.
struct Job {
    lane: u64,
    seq: u64,
    read: Read,
    staging: u64,
}

// SAFETY: as for `Lane`.
unsafe impl Send for Job {}

/// The scheduler, with its workers started on first use; `None` if not
/// even one worker could start.
fn scheduler() -> Option<&'static Scheduler> {
    static SCHEDULER: OnceLock<Option<Scheduler>> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let scheduler = SCHEDULER
        .get_or_init(|| {
            Some(Scheduler {
                lanes: Mutex::new(Lanes {
                    lanes: HashMap::new(),
                    next_id: 0,
                    next_seq: 0,
                    staged: 0,
                    staging: STAGING,
                    starved: false,
                    package_reads: 0,
                    clock: [0; 3],
                    now: 0,
                    lane_now: [0; 3],
                    done: Default::default(),
                    refused: 0,
                    bcpack: Default::default(),
                    since: None,
                }),
                work: Condvar::new(),
                finished: Condvar::new(),
            })
        })
        .as_ref()?;
    STARTED.call_once(|| {
        // `XNW_DS_WORKERS=<n>` (default: one per logical CPU) and
        // `XNW_DS_PRIORITY=below` (workers below normal priority, so the
        // title's 20-odd job threads come first on a 12-thread CPU).
        let workers = std::env::var("XNW_DS_WORKERS").ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|&n| n > 0).unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
        let below = std::env::var("XNW_DS_PRIORITY").is_ok_and(|v| v.trim() == "below");
        let started = (0..workers)
            .filter(|i| {
                std::thread::Builder::new()
                    .name(format!("dstorage {i}"))
                    .spawn(move || {
                        if below {
                            // SAFETY: this thread's own priority.
                            unsafe { xnw_core::win::SetThreadPriority(xnw_core::win::GetCurrentThread(), -1) };
                        }
                        work(scheduler)
                    })
                    .is_ok()
            })
            .count();
        xnw_core::info!("dstorage: {started} workers{}", if below { " (below normal priority, XNW_DS_PRIORITY)" } else { "" });
    });
    Some(scheduler)
}

impl Scheduler {
    fn lock(&self) -> MutexGuard<'_, Lanes> {
        self.lanes.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// A worker: runs the next read of the most deserving queue.
fn work(scheduler: &'static Scheduler) {
    let mut lanes = scheduler.lock();
    loop {
        let Some(job) = lanes.start() else {
            lanes = scheduler.work.wait(lanes).unwrap_or_else(|e| e.into_inner());
            continue;
        };
        drop(lanes);
        let started = std::time::Instant::now();
        let result = execute(&job.read.0);
        lanes = scheduler.lock();
        lanes.finish(&job, result);
        lanes.count(&job, started);
        if std::mem::take(&mut lanes.starved) {
            scheduler.work.notify_all();
        }
        scheduler.finished.notify_all();
    }
}

impl Lanes {
    /// Counts a finished request; with `XNW_TRACE=dstorage`, or when
    /// BCPack data was decoded, a line every 5 s sums them up.
    fn count(&mut self, job: &Job, started: std::time::Instant) {
        let request = &job.read.0;
        let (requests, bytes, busy) = &mut self.done;
        *requests += 1;
        *bytes += u64::from(request.destination_size);
        *busy += started.elapsed();
        if xnw_core::perf::on() {
            xnw_core::perf::add(xnw_core::perf::Area::DsRequest, started.elapsed(), u64::from(request.destination_size));
        }
        if request.options & BCPACK != 0 {
            self.bcpack.0 += u64::from(request.destination_size);
            self.bcpack.1 += started.elapsed();
        }
        let since = *self.since.get_or_insert(started);
        let elapsed = since.elapsed();
        if elapsed.as_secs() >= 5 {
            let waiting: usize = self.lanes.values().map(|l| l.reads.len()).sum();
            let reading = READING.swap(0, Ordering::Relaxed) as f64 / 1e9;
            if tracing() || self.bcpack.0 > 0 {
                let modes: Vec<String> = (1..8)
                    .filter_map(|m| {
                        let (ok, failed) = (BCPACK_OK[m].load(Ordering::Relaxed), BCPACK_FAILED[m].load(Ordering::Relaxed));
                        (ok + failed > 0).then(|| format!("BC{m} {ok} ok {failed} failed"))
                    })
                    .collect();
                xnw_core::info!(
                    "dstorage: last {:.1} s: {} requests, {:.1} MB/s, workers busy {:.2} reading {:.2}, staging waits {}, BCPack {} MB on {:.2} workers, {waiting} waiting; BCPack since start: {}",
                    elapsed.as_secs_f64(),
                    *requests,
                    *bytes as f64 / elapsed.as_secs_f64() / 1e6,
                    busy.as_secs_f64() / elapsed.as_secs_f64(),
                    reading / elapsed.as_secs_f64(),
                    std::mem::take(&mut self.refused),
                    self.bcpack.0 / 1_000_000,
                    self.bcpack.1.as_secs_f64() / elapsed.as_secs_f64(),
                    if modes.is_empty() { "none".into() } else { modes.join(", ") }
                );
            }
            if xnw_core::perf::on() {
                log_file_reads();
                log_origins();
            }
            self.done = Default::default();
            self.bcpack = Default::default();
            self.since = None;
        }
    }

    /// Starts the next read of the class with the least virtual time, from
    /// its lane that has started the fewest bytes, if the staging buffer
    /// has room for it (or nothing else holds any).
    fn start(&mut self) -> Option<Job> {
        // A lane whose next read is from the full package waits while
        // `PACKAGE_READERS` such reads run, leaving workers to the rest.
        let capped = self.package_reads >= PACKAGE_READERS;
        if capped && self.lanes.values().any(|l| l.reads.front().is_some_and(|r| on_package(&r.1.0))) {
            self.starved = true;
        }
        let busy = || self.lanes.iter().filter(move |(_, l)| l.reads.front().is_some_and(|r| !(capped && on_package(&r.1.0))));
        let (_, class) = busy().map(|(_, l)| (self.clock[l.class], l.class)).min()?;
        let (_, _, id) = busy().filter(|(_, l)| l.class == class).filter_map(|(&id, l)| Some((l.clock, l.reads.front()?.0, id))).min()?;
        let lane = self.lanes.get_mut(&id)?;
        let request = lane.reads.front()?.1.0;
        let staging = staging_bytes(&request);
        if self.staged > 0 && self.staged + staging > self.staging {
            self.starved = true;
            self.refused += 1;
            return None;
        }
        let (seq, read) = lane.reads.pop_front()?;
        lane.running.insert(seq);
        self.staged += staging;
        let size = u64::from(request.destination_size.max(request.source_size)).max(1);
        self.now = self.clock[class];
        self.clock[class] += size * COST[class];
        self.lane_now[class] = lane.clock;
        lane.clock += size;
        self.package_reads += usize::from(on_package(&request));
        Some(Job { lane: id, seq, read, staging })
    }

    fn finish(&mut self, job: &Job, result: Result<(), HRESULT>) {
        self.staged -= job.staging;
        self.package_reads -= usize::from(on_package(&job.read.0));
        if let Some(lane) = self.lanes.get_mut(&job.lane) {
            lane.running.remove(&job.seq);
            if let Err(hr) = result {
                lane.failures.insert(job.seq, hr);
                lane.report_failure(&job.read.0, hr);
            }
            lane.flush();
        }
    }

    /// Moves a queue's enqueued commands to the workers.
    fn submit(&mut self, id: u64) {
        let Some(class) = self.lanes.get(&id).map(|l| l.class) else {
            return;
        };
        // A class that had nothing waiting starts from now, not from the
        // credit it left behind.
        if !self.lanes.values().any(|l| l.class == class && !l.reads.is_empty()) {
            self.clock[class] = self.clock[class].max(self.now);
        }
        let Some(lane) = self.lanes.get_mut(&id) else {
            return;
        };
        if lane.reads.is_empty() {
            lane.clock = lane.clock.max(self.lane_now[class]);
        }
        for command in std::mem::take(&mut lane.unsubmitted) {
            let seq = self.next_seq;
            self.next_seq += 1;
            match command {
                Command::Read(read) => lane.reads.push_back((seq, read)),
                Command::Status(status) => lane.statuses.push_back((seq, status)),
            }
        }
        lane.flush();
        if xnw_core::perf::on() {
            xnw_core::perf::peak(xnw_core::perf::Area::DsDepth, self.lanes.values().map(|l| l.reads.len() as u64).sum());
        }
    }
}

impl Lane {
    fn new(priority: i8, capacity: u16, event: xnw_core::win::HANDLE) -> Lane {
        Lane {
            class: priority.clamp(-1, 1).wrapping_add(1) as usize,
            capacity: usize::from(capacity),
            clock: 0,
            unsubmitted: Vec::new(),
            reads: VecDeque::new(),
            statuses: VecDeque::new(),
            running: BTreeSet::new(),
            failures: BTreeMap::new(),
            closed: false,
            event,
            record: ErrorRecord::default(),
        }
    }

    /// Requests and status entries held, submitted or not.
    fn entries(&self) -> usize {
        self.unsubmitted.len() + self.reads.len() + self.statuses.len() + self.running.len()
    }

    /// Submitted work not yet done: reads waiting or running, status
    /// entries not written.
    fn busy(&self) -> bool {
        !(self.reads.is_empty() && self.running.is_empty() && self.statuses.is_empty())
    }

    /// Writes the status entries whose earlier reads have all finished.
    fn flush(&mut self) {
        while let Some(&(seq, _)) = self.statuses.front() {
            let waiting = self.reads.front().map(|r| r.0);
            if waiting.into_iter().chain(self.running.first().copied()).any(|s| s < seq) {
                return;
            }
            let Some((_, status)) = self.statuses.pop_front() else {
                return;
            };
            let later = self.failures.split_off(&seq);
            let first = std::mem::replace(&mut self.failures, later).into_values().next();
            status.entry().store(first.unwrap_or(S_OK), Ordering::Release);
        }
    }

    /// Counts a failed read in the queue's error record and sets its event.
    fn report_failure(&mut self, request: &Request, hr: HRESULT) {
        if self.record.failure_count == 0 {
            self.record.first = ReadFailure::new(request, hr);
        }
        self.record.failure_count = self.record.failure_count.saturating_add(1);
        if !self.event.is_null() {
            // SAFETY: the queue's event; the queue outlives its lane.
            unsafe { xnw_core::win::SetEvent(self.event) };
        }
    }
}

/// Staging bytes a request holds while it runs: what it buffers between
/// the file and the destination. Raw reads go straight to their
/// destination, memory sources need none (as on the console).
fn staging_bytes(request: &Request) -> u64 {
    let buffered = request.options & (ZLIB | BCPACK | PAGE_DESTINATION) != 0;
    if buffered && request.options & MEMORY_SOURCE == 0 { u64::from(request.source_size) } else { 0 }
}

/// Ends a queue: what was not submitted is dropped, and this waits until
/// everything submitted has completed or failed, its status entries
/// written (VIctory).
fn close_lane(id: u64) {
    let Some(scheduler) = scheduler() else {
        return;
    };
    let mut lanes = scheduler.lock();
    let Some(lane) = lanes.lanes.get_mut(&id) else {
        return;
    };
    lane.closed = true;
    lane.unsubmitted.clear(); // releases arrays and files
    let started = xnw_core::perf::start();
    while lanes.lanes.get(&id).is_some_and(Lane::busy) {
        lanes = scheduler.finished.wait(lanes).unwrap_or_else(|e| e.into_inner());
    }
    xnw_core::perf::since(xnw_core::perf::Area::DsClose, started, 0);
    lanes.lanes.remove(&id);
    scheduler.finished.notify_all();
}

/// A read request as the title builds it, 0x48 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
struct Request {
    options: u64,
    /// An address, or with `PAGE_DESTINATION` an array of 64 KiB page
    /// numbers.
    destination: *mut c_void,
    /// With `PAGE_DESTINATION`, the low 16 bits are where in the first page
    /// the destination starts.
    page_offset: u64,
    destination_size: u32,
    /// Our file object, or with `MEMORY_SOURCE` an address.
    source: *const c_void,
    offset: u64,
    source_size: u32,
    /// Compressed size between two decompression stages (zlib, then BCPack).
    intermediate_size: u32,
    tag: u64,
    name: *const c_char,
}

const _: () = assert!(size_of::<Request>() == 0x48);

// Request options.
/// Inflate zlib (RFC 1950) data.
const ZLIB: u64 = 1;
/// Bits 1-4: decode BCPack data of this BC format number (0: none).
const BCPACK: u64 = 0xf << 1;
const PAGE_DESTINATION: u64 = 1 << 21;
const MEMORY_SOURCE: u64 = 1 << 22;
// Bits 5-20 select a further transform, bit 23 a source in physical pages;
// the title's packages need neither, so requests with them fail.

const PAGE: usize = 64 * 1024;

enum Command {
    Read(Read),
    Status(Status),
}

/// A queued read, holding a reference on its file object until it has run:
/// the title may release the file once it has enqueued (after VIctory).
struct Read(Request);

impl Read {
    fn new(request: Request) -> Read {
        if let Some(file) = file_of(&request) {
            add_ref_instance(file);
        }
        Read(request)
    }
}

impl Drop for Read {
    fn drop(&mut self) {
        if let Some(file) = file_of(&self.0) {
            release(file.cast_mut());
        }
    }
}

/// The request's file object, unless it reads memory.
fn file_of(request: &Request) -> Option<*const Instance<File>> {
    (request.options & MEMORY_SOURCE == 0 && !request.source.is_null()).then_some(request.source.cast())
}

/// A status entry to write, holding a reference on its array.
struct Status {
    array: Array,
    index: u32,
}

impl Drop for Status {
    fn drop(&mut self) {
        release(self.array);
    }
}

impl Status {
    fn entry(&self) -> &AtomicI32 {
        // SAFETY: we hold a reference; the index was checked when enqueued.
        &unsafe { &*self.array }.state.entries[self.index as usize]
    }
}

extern "system" fn enqueue_request(this: QueueRef, request: *const Request) {
    // SAFETY: the title passes its request (null tolerated).
    let Some(&request) = (unsafe { request.as_ref() }) else {
        return;
    };
    queue(this, Command::Read(Read::new(request)));
}

extern "system" fn enqueue_status(this: QueueRef, array: Array, index: u32) {
    // SAFETY: the title passes an array we created (null tolerated).
    let Some(entries) = (unsafe { array.as_ref() }).map(|a| &a.state.entries) else {
        return;
    };
    let Some(entry) = entries.get(index as usize) else {
        xnw_core::warn!("dstorage: status entry {index} out of range");
        return;
    };
    entry.store(E_PENDING, Ordering::Release);
    add_ref_instance(array);
    queue(this, Command::Status(Status { array, index }));
}

/// Adds a command. Nothing runs before `Submit` unless the queue is full:
/// the title fills a request's page array after enqueueing it and before
/// submitting (another runtime's measurement; a half-full auto-submit read unfinished
/// requests there). A full queue submits what it holds and blocks the
/// caller until a slot frees, as the console does; one slot stays free, as
/// the console's count of empty slots says.
fn queue(this: QueueRef, command: Command) {
    // SAFETY: a queue we created and the title still holds.
    let id = unsafe { &*this }.state.id;
    let Some(scheduler) = scheduler() else {
        return;
    };
    let mut lanes = scheduler.lock();
    let mut blocked = None;
    loop {
        let Some(lane) = lanes.lanes.get(&id).filter(|l| !l.closed) else {
            return; // closed: dropped (a status releases its array)
        };
        if lane.entries() + 1 < lane.capacity.max(2) {
            xnw_core::perf::since(xnw_core::perf::Area::DsBlocked, blocked, 0);
            break;
        }
        blocked = blocked.or_else(xnw_core::perf::start);
        if !lane.unsubmitted.is_empty() {
            lanes.submit(id);
            scheduler.work.notify_all();
        }
        lanes = scheduler.finished.wait(lanes).unwrap_or_else(|e| e.into_inner());
    }
    let Some(lane) = lanes.lanes.get_mut(&id) else {
        return;
    };
    lane.unsubmitted.push(command);
}

extern "system" fn submit(this: QueueRef) {
    // SAFETY: as above.
    let id = unsafe { &*this }.state.id;
    let Some(scheduler) = scheduler() else {
        return;
    };
    scheduler.lock().submit(id);
    scheduler.work.notify_all();
}

/// Ends the queue whatever its reference count: what was not submitted
/// never completes, and this returns once everything submitted has
/// completed or failed, its status entries written. The title calls this,
/// not `Release`.
extern "system" fn close(this: QueueRef) {
    // SAFETY: as above.
    close_lane(unsafe { &*this }.state.id);
}

/// Slot 9: a duplicate of the queue's error event; the title owns it.
extern "system" fn error_event(this: QueueRef) -> xnw_core::win::HANDLE {
    use xnw_core::win::{DUPLICATE_SAME_ACCESS, DuplicateHandle, GetCurrentProcess};
    // SAFETY: a queue we created and the title still holds.
    let queue = &unsafe { &*this }.state;
    let mut out = std::ptr::null_mut();
    // SAFETY: our own event, duplicated into our own process.
    unsafe { DuplicateHandle(GetCurrentProcess(), queue.event, GetCurrentProcess(), &mut out, 0, 0, DUPLICATE_SAME_ACCESS) };
    out
}

/// Slot 10: hands over the error record and starts a new one; the event
/// is reset under the scheduler's lock, so no failure is lost.
extern "system" fn retrieve_error_record(this: QueueRef, out: *mut ErrorRecord) {
    // SAFETY: as above.
    let queue = &unsafe { &*this }.state;
    let record = match scheduler() {
        Some(scheduler) => {
            let mut lanes = scheduler.lock();
            // SAFETY: our own event.
            unsafe { xnw_core::win::ResetEvent(queue.event) };
            lanes.lanes.get_mut(&queue.id).map(|l| std::mem::take(&mut l.record)).unwrap_or_default()
        }
        None => ErrorRecord::default(),
    };
    if !out.is_null() {
        // SAFETY: the title passes room for a record.
        unsafe { out.write_unaligned(record) };
    }
}

/// Names an unsupported kind of request once; every one fails.
fn unsupported(what: String, request: &Request) -> HRESULT {
    static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&what) {
        xnw_core::warn!("UNIMPLEMENTED dstorage {what}: {}; such requests fail", describe(request));
        seen.push(what);
    }
    E_NOTIMPL
}

/// BCPack requests per mode (the option's BC format number), decoded and
/// failed, since the start.
static BCPACK_OK: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
static BCPACK_FAILED: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];

/// Counts a BCPack request by mode; failures are logged, the first of
/// each mode and then every power of two.
fn count_bcpack(bc: u8, result: &Result<(), HRESULT>) {
    let Err(hr) = result else {
        BCPACK_OK[usize::from(bc)].fetch_add(1, Ordering::Relaxed);
        return;
    };
    let n = BCPACK_FAILED[usize::from(bc)].fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        xnw_core::warn!("dstorage: {n} BCPack mode {bc} request(s) failed so far (last {:#010x})", *hr as u32);
    }
}

/// Whether `XNW_TRACE` names `dstorage`: every request is then logged.
fn tracing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XNW_TRACE").is_ok_and(|t| t.to_ascii_lowercase().split(',').any(|f| f.trim() == "*" || f.trim() == "dstorage")))
}

/// A request for the log: options, source (file and offset, or memory),
/// sizes and the request's own name.
fn describe(request: &Request) -> String {
    let source = if request.options & MEMORY_SOURCE != 0 {
        format!("memory {:#x}", request.source as usize)
    } else if request.source.is_null() {
        "no source".into()
    } else {
        // SAFETY: a file object we created; the queued read holds it.
        let file = &unsafe { &*request.source.cast::<Instance<File>>() }.state;
        format!("{} @ {:#x}", file.name, request.offset)
    };
    format!(
        "options {:#x}, {source}, {} -> {} bytes (intermediate {}), tag {:#x}, \"{}\"",
        request.options,
        request.source_size,
        request.destination_size,
        request.intermediate_size,
        request.tag,
        text(request.name)
    )
}

/// Runs one request and logs it when traced or when it fails.
fn execute(request: &Request) -> Result<(), HRESULT> {
    let result = perform(request);
    match result {
        Ok(()) if tracing() => xnw_core::trace!("dstorage: {} ok", describe(request)),
        Err(hr) if hr != E_NOTIMPL => xnw_core::warn!("dstorage: {} failed {:#010x}", describe(request), hr as u32),
        _ => {}
    }
    let bc = ((request.options & BCPACK) >> 1) as u8;
    if bc != 0 {
        count_bcpack(bc, &result);
    }
    result
}

/// Reads, inflates and/or unpacks, and writes one request.
fn perform(request: &Request) -> Result<(), HRESULT> {
    let options = request.options;
    if options & !(ZLIB | BCPACK | PAGE_DESTINATION | MEMORY_SOURCE) != 0 {
        return Err(unsupported("request option".into(), request));
    }
    let bc = ((options & BCPACK) >> 1) as u8;
    if bc != 0 && !matches!(bc, 1 | 3..=7) {
        return Err(unsupported(format!("BCPack mode {bc}"), request));
    }
    if request.source.is_null() || request.destination.is_null() {
        return Err(E_INVALIDARG);
    }
    let size = request.destination_size as usize;
    // A texture page the GPU takes from the pack is not read.
    if options & (PAGE_DESTINATION | MEMORY_SOURCE) == PAGE_DESTINATION
        // SAFETY: a file object we created; the queued read holds it.
        && let Some(file) = file_of(request).and_then(|f| unsafe { &*f }.state.origin)
    {
        let offset = (request.page_offset & 0xffff) as usize;
        // SAFETY: the title passes a page number per 64 KiB of destination.
        let pages = unsafe { std::slice::from_raw_parts(request.destination.cast::<usize>(), (offset + size).div_ceil(PAGE)) };
        if crate::texbypass::bypass(file, request.offset, pages, offset, size) {
            return Ok(());
        }
    }
    if options & (ZLIB | BCPACK) == 0 {
        if request.source_size > request.destination_size {
            return Err(E_INVALIDARG);
        }
        let size = request.source_size as usize;
        return into_destination(request, size, |out| read_into(request, out).map(|()| size));
    }
    let tables = if bc != 0 { Some(bcpack_tables()?) } else { None };
    let source = read_source(request)?;
    // The console's order: zlib first, then BCPack.
    let source = if bc != 0 && options & ZLIB != 0 {
        let mut packed = vec![0; request.intermediate_size as usize];
        let n = inflate(source.bytes(), &mut packed)?;
        packed.truncate(n);
        Source::Owned(packed)
    } else {
        source
    };
    into_destination(request, size, |out| match tables {
        Some(tables) => xnw_core::perf::time(xnw_core::perf::Area::DsBcpack, out.len() as u64, || bcpack::decode(bc, source.bytes(), out, tables)).map(|()| out.len()).map_err(|e| {
            xnw_core::warn!("dstorage: {e}: {}", describe(request));
            INVALID_DATA
        }),
        None => inflate(source.bytes(), out),
    })
}

/// Inflates a zlib (RFC 1950) stream into `out`, returning the bytes
/// written. A stream that ends short of `out` passes (VIctory's rule: the
/// title's chunks fill it exactly, and nothing says that is required).
fn inflate(source: &[u8], out: &mut [u8]) -> Result<usize, HRESULT> {
    let len = out.len() as u64;
    xnw_core::perf::time(xnw_core::perf::Area::DsInflate, len, || miniz_oxide::inflate::decompress_slice_iter_to_slice(out, std::iter::once(source), true, false)).map_err(|status| {
        xnw_core::warn!("dstorage: zlib stream does not inflate: {status:?}");
        INVALID_DATA
    })
}

/// The BCPack format's constant tables, parsed once for the process. If
/// they do not parse every BCPack request fails: one warning.
pub(crate) fn bcpack_tables() -> Result<&'static Tables, HRESULT> {
    static TABLES: OnceLock<Result<Tables, String>> = OnceLock::new();
    static WARNED: AtomicBool = AtomicBool::new(false);
    match TABLES.get_or_init(load_bcpack_tables) {
        Ok(tables) => Ok(tables),
        Err(why) => {
            if !WARNED.swap(true, Ordering::Relaxed) {
                xnw_core::warn!("dstorage: BCPack requests fail: {why}");
            }
            Err(E_NOTIMPL)
        }
    }
}

/// The tables VIctory ships with.
#[cfg(not(test))]
fn load_bcpack_tables() -> Result<Tables, String> {
    Tables::builtin().map_err(|e| format!("the built-in BCPack tables: {e}"))
}

/// Tests decode streams made against tables of their own (VIctory's:
/// uniform models; BC4 index pairs: low 3 bits, next 3 bits).
#[cfg(test)]
fn load_bcpack_tables() -> Result<Tables, String> {
    let mut t = Vec::new();
    for m in 0..128 {
        let g = m / 16;
        for b in 0..9 {
            let f: u16 = if b < g + 2 { 1 << (15 - g) } else { 0 };
            t.extend_from_slice(&f.to_le_bytes());
        }
    }
    t.extend([0; 256 + 64 + 256]);
    t.extend((0..64u8).map(|v| (v & 7) | (v >> 3) << 4));
    t.resize(bcpack::TABLES_SIZE, 0);
    Tables::parse(&t).map_err(|e| e.to_string())
}

/// A compressed request's source: borrowed from memory, or read from the
/// file.
enum Source {
    Borrowed(*const u8, usize),
    Owned(Vec<u8>),
}

impl Source {
    fn bytes(&self) -> &[u8] {
        match self {
            // SAFETY: the title's source holds `len` bytes until completion.
            Source::Borrowed(at, len) => unsafe { std::slice::from_raw_parts(*at, *len) },
            Source::Owned(bytes) => bytes,
        }
    }
}

fn read_source(request: &Request) -> Result<Source, HRESULT> {
    let size = request.source_size as usize;
    if request.options & MEMORY_SOURCE != 0 {
        return Ok(Source::Borrowed(request.source.cast(), size));
    }
    let mut bytes = vec![0; size];
    read_into(request, &mut bytes)?;
    Ok(Source::Owned(bytes))
}

/// Nanoseconds workers spent reading files, for the summary.
static READING: AtomicU64 = AtomicU64::new(0);

/// Fills `out` (the source size) from the request's file or memory.
fn read_into(request: &Request, out: &mut [u8]) -> Result<(), HRESULT> {
    if request.options & MEMORY_SOURCE != 0 {
        // SAFETY: the title's source holds `out.len()` bytes until completion.
        out.copy_from_slice(unsafe { std::slice::from_raw_parts(request.source.cast(), out.len()) });
        return Ok(());
    }
    // SAFETY: a file object we created; the queued read holds it.
    let file = &unsafe { &*request.source.cast::<Instance<File>>() }.state.file;
    let started = std::time::Instant::now();
    let mut done = 0;
    let result = loop {
        if done == out.len() {
            break Ok(());
        }
        match file.seek_read(&mut out[done..], request.offset + done as u64) {
            Ok(0) => break Err(END_OF_FILE),
            Ok(n) => done += n,
            Err(error) => break Err(from_win32(&error)),
        }
    };
    READING.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    count_read(request, done, started.elapsed());
    result
}

/// Runs `fill` on the destination's first `len` bytes; `fill` returns how
/// many it produced. An address is written in place; physical pages go
/// through a buffer.
fn into_destination(request: &Request, len: usize, fill: impl FnOnce(&mut [u8]) -> Result<usize, HRESULT>) -> Result<(), HRESULT> {
    if request.options & PAGE_DESTINATION == 0 {
        // Raw reads land through the kernel, which does not fault into the
        // GPU side's guard on title buffers; open them first (VIctory-base
        // dstorage.rs 1046).
        xnw_gpu::d3d12x::kernel_will_write(request.destination as usize, len);
        xnw_formats::origin::registry().forget_va(request.destination as usize, len);
        // SAFETY: the title's destination holds `destination_size` bytes, at
        // least `len`, and nothing else writes it until completion.
        let out = unsafe { std::slice::from_raw_parts_mut(request.destination.cast::<u8>(), len) };
        return fill(out).map(drop);
    }
    let offset = (request.page_offset & 0xffff) as usize;
    // SAFETY: the title passes a page number per 64 KiB of destination,
    // which holds at least `len` bytes from `offset`.
    let pages = unsafe { std::slice::from_raw_parts(request.destination.cast::<usize>(), (offset + len).div_ceil(PAGE)) };
    let origins = xnw_formats::origin::registry();
    // What the pages held goes before they change; a texture page read is
    // recorded once written.
    origins.forget_pages(pages, offset, len);
    let mut data = vec![0; len];
    let n = fill(&mut data)?;
    if !crate::xmem::write_physical(pages, offset, &data[..n]) {
        return Err(E_INVALIDARG);
    }
    // SAFETY: a file object we created; the queued read holds it.
    let file = file_of(request).and_then(|f| unsafe { &*f }.state.origin);
    let Some(file) = file else { return Ok(()) };
    // The decoded page is kept before the read is recorded: recording
    // tells the GPU layer, which may prepare the texture from it at once.
    let index = origins.file(file);
    if let Some(r) = index.as_ref().and_then(|f| f.resolve(request.offset))
        && index.as_ref().and_then(|f| f.ttpf.pages.get(r.page as usize).map(|p| p.size)) == Some(n as u64)
    {
        data.truncate(n);
        origins.put_decoded(file, r.page, data);
    }
    origins.record_pages(file, request.offset, pages, offset, n);
    Ok(())
}

/// Slots by the 2020 order, which the argument shapes at the title's call
/// sites match (0x18, 0x20, 0x30 and 0x40 are called).
static QUEUE_VTABLE: Table = methods(&[
    (1, add_ref_instance::<Queue> as *const ()),
    (2, release::<Queue> as *const ()),
    (3, enqueue_request as *const ()),        // 0x18
    (4, enqueue_status as *const ()),         // 0x20
    (6, submit as *const ()),                 // 0x30
    (8, close as *const ()),                  // 0x40
    (9, error_event as *const ()),            // 0x48
    (10, retrieve_error_record as *const ()), // 0x50
]);

#[cfg(test)]
mod tests {
    use super::*;

    fn call<F: Copy>(object: *mut c_void, slot: usize) -> F {
        // SAFETY: every object here starts with the runtime header.
        crate::xgameruntime::method(unsafe { &*object.cast::<Object>() }, slot)
    }

    fn factory() -> *mut c_void {
        let mut factory = std::ptr::null_mut();
        assert_eq!(DStorageGetFactory(&FACTORY, &mut factory), S_OK);
        factory
    }

    /// A folder served as the game drive, while the guard lives.
    struct Drive {
        dir: std::path::PathBuf,
        _guard: std::sync::MutexGuard<'static, ()>,
    }

    impl Drive {
        fn new(name: &str) -> Drive {
            let guard = crate::files::TEST_DRIVES.lock().unwrap_or_else(|e| e.into_inner());
            let dir = std::env::temp_dir().join(format!("xnw-dstorage-{name}-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            crate::files::set_game_drive(std::slice::from_ref(&dir));
            Drive { dir, _guard: guard }
        }

        /// Writes `bytes` as `name` and opens it as the title does.
        fn file(&self, name: &str, bytes: &[u8]) -> *mut c_void {
            std::fs::write(self.dir.join(name), bytes).unwrap();
            let wide: Vec<u16> = format!("G:/{name}").encode_utf16().chain([0]).collect();
            let open: extern "system" fn(*mut c_void, *const u16, *const Guid, Out) -> HRESULT = call(factory(), 0x20);
            let mut file = std::ptr::null_mut();
            assert_eq!(open(factory(), wide.as_ptr(), std::ptr::null(), &mut file), S_OK);
            file
        }
    }

    impl Drop for Drive {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn status_array(capacity: u32) -> *mut c_void {
        let create: extern "system" fn(*mut c_void, u32, *const c_char, *const Guid, Out) -> HRESULT = call(factory(), 0x28);
        let mut array = std::ptr::null_mut();
        assert_eq!(create(factory(), capacity, c"test".as_ptr(), std::ptr::null(), &mut array), S_OK);
        array
    }

    fn new_queue(capacity: u16) -> *mut c_void {
        let create: extern "system" fn(*mut c_void, *const QueueDesc, *const Guid, Out) -> HRESULT = call(factory(), 0x50);
        let desc = QueueDesc { source_type: 0, capacity, priority: 0, name: c"test".as_ptr(), unknown: 1 };
        let mut queue = std::ptr::null_mut();
        assert_eq!(create(factory(), &desc, std::ptr::null(), &mut queue), S_OK);
        queue
    }

    fn request(options: u64, destination: &mut [u8], source: *const c_void, offset: u64, source_size: usize) -> Request {
        Request {
            options,
            destination: destination.as_mut_ptr().cast(),
            page_offset: 0,
            destination_size: destination.len() as u32,
            source,
            offset,
            source_size: source_size as u32,
            intermediate_size: 0,
            tag: 0,
            name: std::ptr::null(),
        }
    }

    fn enqueue(queue: *mut c_void, request: &Request) {
        let f: extern "system" fn(*mut c_void, *const Request) = call(queue, 0x18);
        f(queue, request);
    }

    fn enqueue_status(queue: *mut c_void, array: *mut c_void, index: u32) {
        let f: extern "system" fn(*mut c_void, *mut c_void, u32) = call(queue, 0x20);
        f(queue, array, index);
    }

    fn void(object: *mut c_void, slot: usize) {
        let f: extern "system" fn(*mut c_void) = call(object, slot);
        f(object);
    }

    fn result(array: *mut c_void, index: u32) -> HRESULT {
        let f: extern "system" fn(*mut c_void, u32) -> HRESULT = call(array, 0x20);
        f(array, index)
    }

    /// Polls as the title does, then returns the entry's result.
    fn wait(array: *mut c_void, index: u32) -> HRESULT {
        let done: extern "system" fn(*mut c_void, u32) -> bool = call(array, 0x18);
        let start = std::time::Instant::now();
        while !done(array, index) {
            assert!(start.elapsed().as_secs() < 10, "never completed");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        result(array, index)
    }

    #[test]
    fn reads_complete_in_order_with_status() {
        let drive = Drive::new("reads");
        let data: Vec<u8> = (0..=255u8).cycle().take(100_000).collect();
        let file = drive.file("data.bin", &data);
        let plain: Vec<u8> = (0..60_000u32).map(|i| (i * 7 / 13) as u8).collect();
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);
        let packed_file = drive.file("packed.bin", &packed);
        let (queue, array) = (new_queue(0x80), status_array(2));

        let mut out = vec![0u8; 50_000];
        let mut inflated = vec![0u8; 65_536];
        enqueue(queue, &request(0, &mut out, file, 1000, 50_000));
        enqueue_status(queue, array, 0);
        enqueue(queue, &request(ZLIB, &mut inflated, packed_file, 0, packed.len()));
        enqueue_status(queue, array, 1);
        // Nothing runs before Submit.
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(result(array, 0), E_PENDING);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out[..], data[1000..51_000]);
        assert_eq!(wait(array, 1), S_OK);
        assert_eq!(inflated[..plain.len()], plain[..]);

        // A stream that does not inflate fails, and is reported.
        let event: extern "system" fn(*mut c_void) -> xnw_core::win::HANDLE = call(queue, 0x48);
        let retrieve: extern "system" fn(*mut c_void, *mut ErrorRecord) = call(queue, 0x50);
        let handle = event(queue);
        assert!(!handle.is_null());
        let mut garbage = vec![0u8; 16];
        enqueue(queue, &request(ZLIB, &mut garbage, file, 0, 16)); // plain bytes, not zlib
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        void(queue, 0x40); // close: waits for what was submitted
        assert_eq!(result(array, 0), INVALID_DATA);
        let mut record = ErrorRecord::default();
        retrieve(queue, &mut record);
        // After Close the lane is gone; the record was taken with it.
        assert_eq!(record.failure_count, 0);
        // SAFETY: the duplicate is ours to close.
        unsafe { xnw_core::win::CloseHandle(handle) };
    }

    #[test]
    fn failures_fill_the_error_record() {
        let drive = Drive::new("record");
        let file = drive.file("data.bin", &[0u8; 64]);
        let (queue, array) = (new_queue(0x80), status_array(1));
        let mut garbage = vec![0u8; 16];
        enqueue(queue, &request(ZLIB, &mut garbage, file, 0, 16));
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), INVALID_DATA);
        let retrieve: extern "system" fn(*mut c_void, *mut ErrorRecord) = call(queue, 0x50);
        let mut record = ErrorRecord::default();
        retrieve(queue, &mut record);
        assert_eq!(record.failure_count, 1);
        assert_eq!(record.first.result, INVALID_DATA);
        assert_eq!(record.first.command_type, 0);
        assert_eq!(String::from_utf16_lossy(&record.first.filename[..11]), "G:/data.bin");
        void(queue, 0x40);
    }

    #[test]
    fn statuses_complete_in_order_with_the_first_failure() {
        let drive = Drive::new("order");
        let contents: Vec<u8> = (0..4096).map(|i| (i * 7 + i / 251) as u8).collect();
        let file = drive.file("data.bin", &contents);
        let (queue, array) = (new_queue(0x80), status_array(4));
        // Requests in flight together run in any order, so none of them
        // shares a destination (failed ones leave theirs undefined).
        let mut buffers = [[0u8; 16]; 6];
        let [failed, failed_too, out, other, bcpack, bc2] = buffers.each_mut();

        // Segment 0: a read past the end, then a bad stream; the first wins.
        enqueue(queue, &request(0, failed, file, 4090, 16));
        enqueue(queue, &request(ZLIB, failed_too, file, 0, 16));
        enqueue_status(queue, array, 0);
        // Segment 1: two reads.
        enqueue(queue, &request(0, other, file, 0, 16));
        enqueue(queue, &request(0, out, file, 100, 16));
        enqueue_status(queue, array, 1);
        void(queue, 0x30);
        // Segment 2: a BCPack request (BC7) whose stream is empty fails.
        enqueue(queue, &request(7 << 1, bcpack, file, 0, 0));
        enqueue_status(queue, array, 2);
        // Segment 3: BCPack mode 2 (BC2) has no decoder.
        enqueue(queue, &request(2 << 1, bc2, file, 0, 16));
        enqueue_status(queue, array, 3);
        void(queue, 0x30);

        assert_eq!(wait(array, 3), E_NOTIMPL);
        assert_eq!(result(array, 2), INVALID_DATA);
        assert_eq!(result(array, 0), END_OF_FILE);
        assert_eq!(result(array, 1), S_OK);
        assert_eq!(buffers[2], contents[100..116]);
        assert_eq!(buffers[3], contents[..16]);
        assert!(BCPACK_FAILED[7].load(Ordering::Relaxed) >= 1);
        assert!(BCPACK_FAILED[2].load(Ordering::Relaxed) >= 1);
        void(queue, 0x40);
    }

    #[test]
    fn a_full_queue_submits_itself() {
        let drive = Drive::new("full");
        let contents: Vec<u8> = (0..64u8).collect();
        let file = drive.file("data.bin", &contents);
        let (queue, array) = (new_queue(4), status_array(1));
        let mut out = [[0u8; 16]; 3];
        let [a, b, c] = out.each_mut();
        enqueue(queue, &request(0, a, file, 0, 16));
        enqueue(queue, &request(0, b, file, 16, 16));
        enqueue_status(queue, array, 0);
        // The fourth entry would fill the queue: what it holds is submitted.
        enqueue(queue, &request(0, c, file, 32, 16));
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out[1], contents[16..32]);
        void(queue, 0x40);
    }

    /// Two BC4 blocks from the format's test encoder: endpoints 10, -3 with
    /// indices p -> p & 7, then the same block plus 1 everywhere (VIctory's).
    const BC4_STREAM: [u8; 88] = [
        0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 4, 4, 4, 4, 1, 1, 1, 1, 0x10, 0x10, 0x10, 0x10, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0x14, 5, 0x20, 0x68, 0xb0, 0xf8, 3, 3, 2, 2,
        0x24, 0x24, 0x24, 0x24, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 0x10, 0x10, 0x10, 0x10, 0, 0, 0, 0, 0x20, 0x68, 0xb0, 0xf8, 0x24, 0x24, 0x24, 0x24, 0, 0,
        0, 0, 0, 0, 0, 0,
    ];

    /// What `BC4_STREAM` unpacks to.
    fn bc4_blocks() -> [u8; 16] {
        let code = [0u64, 2, 3, 4, 5, 6, 7, 1];
        let block = |e0: u64, e1: u64, add: usize| {
            let bits = (0..16).fold(0, |a, p| a | code[(p + add) & 7] << (16 + 3 * p));
            (e0 | e1 << 8 | bits).to_le_bytes()
        };
        let mut out = [0; 16];
        out[..8].copy_from_slice(&block(138, 125, 0));
        out[8..].copy_from_slice(&block(139, 126, 1));
        out
    }

    #[test]
    fn bcpack_requests_unpack_into_the_destination() {
        let drive = Drive::new("bcpack");
        let mut contents: Vec<u8> = (0..100u8).collect();
        contents.extend_from_slice(&BC4_STREAM);
        let file = drive.file("data.bin", &contents);
        let (queue, array) = (new_queue(0x80), status_array(1));
        let mut out = [0u8; 16];
        enqueue(queue, &request(4 << 1, &mut out, file, 100, 88));
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, bc4_blocks());
        void(queue, 0x40);
    }

    #[test]
    fn zlib_then_bcpack() {
        let drive = Drive::new("zlib-bcpack");
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(&BC4_STREAM, 6);
        let file = drive.file("data.bin", &packed);
        let (queue, array) = (new_queue(0x80), status_array(1));
        let mut out = [0u8; 16];
        let mut r = request(ZLIB | 4 << 1, &mut out, file, 0, packed.len());
        r.intermediate_size = BC4_STREAM.len() as u32;
        enqueue(queue, &r);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, bc4_blocks());
        void(queue, 0x40);
    }

    /// Lanes without workers: what `start` picks, and when statuses land.
    fn lanes(staging: u64) -> Lanes {
        Lanes {
            lanes: HashMap::new(),
            next_id: 0,
            next_seq: 0,
            staged: 0,
            staging,
            starved: false,
            package_reads: 0,
            clock: [0; 3],
            now: 0,
            lane_now: [0; 3],
            done: Default::default(),
            refused: 0,
            bcpack: Default::default(),
            since: None,
        }
    }

    fn read(lanes: &mut Lanes, id: u64, options: u64, size: u32) {
        let mut r = request(options, &mut [], std::ptr::null(), 0, 0);
        (r.destination_size, r.source_size) = (size, size);
        lanes.lanes.get_mut(&id).unwrap().unsubmitted.push(Command::Read(Read::new(r)));
    }

    fn lane(priority: i8, capacity: u16) -> Lane {
        Lane::new(priority, capacity, std::ptr::null_mut())
    }

    #[test]
    fn priorities_share_by_bytes_about_tenfold() {
        let mut l = lanes(STAGING);
        l.lanes.insert(0, lane(0, 4096));
        l.lanes.insert(1, lane(1, 4096));
        l.lanes.insert(2, lane(2, 4096)); // real-time: as high
        for _ in 0..400 {
            for id in 0..3 {
                read(&mut l, id, 0, 4096);
            }
        }
        (0..3).for_each(|id| l.submit(id));
        let mut started = [0u32; 3];
        for _ in 0..330 {
            let job = l.start().unwrap();
            started[job.lane as usize] += 1;
            l.finish(&job, Ok(()));
        }
        // Normal gets about a tenth of what high and real-time get together.
        assert_eq!(started[1] + started[2], 300, "{started:?}");
        assert_eq!(started[0], 30, "{started:?}");
        assert!(started[1].abs_diff(started[2]) <= 1, "{started:?}");
    }

    #[test]
    fn staging_bounds_compressed_bytes_in_flight() {
        let mut l = lanes(100);
        l.lanes.insert(0, lane(0, 64));
        read(&mut l, 0, ZLIB, 60);
        read(&mut l, 0, ZLIB, 60);
        read(&mut l, 0, 0, 1000); // raw: straight to its destination
        l.submit(0);
        let first = l.start().unwrap();
        assert!(l.start().is_none(), "60 + 60 > 100");
        l.finish(&first, Ok(()));
        let second = l.start().unwrap();
        let raw = l.start().unwrap();
        assert_eq!((second.staging, raw.staging), (60, 0));
        // Alone, a request larger than the buffer still runs.
        read(&mut l, 0, ZLIB, 500);
        l.submit(0);
        assert!(l.start().is_none());
        l.finish(&second, Ok(()));
        assert!(l.start().is_some());
    }

    /// At most `PACKAGE_READERS` reads of the full package run; other
    /// queues' reads still start meanwhile.
    #[test]
    fn package_reads_leave_workers_to_the_rest() {
        let reads: &'static FileReads =
            Box::leak(Box::new(FileReads { path: PathBuf::from("slow"), package: true, counts: Default::default(), logged: AtomicU64::new(0) }));
        let file = std::fs::File::open(env!("CARGO_MANIFEST_PATH")).unwrap();
        let slow = Instance::create("dstorage file", &FILE_VTABLE, File { file, name: "slow".into(), reads, origin: None });
        let mut l = lanes(STAGING);
        l.lanes.insert(0, lane(0, 64));
        l.lanes.insert(1, lane(0, 64));
        for _ in 0..PACKAGE_READERS + 1 {
            let mut r = request(0, &mut [], slow, 0, 0);
            (r.destination_size, r.source_size) = (10, 10);
            l.lanes.get_mut(&0).unwrap().unsubmitted.push(Command::Read(Read::new(r)));
        }
        read(&mut l, 1, 0, 10);
        l.submit(0);
        l.submit(1);
        let jobs: Vec<Job> = std::iter::from_fn(|| l.start()).collect();
        assert_eq!(jobs.len(), PACKAGE_READERS + 1, "the cap, and the fast read");
        assert_eq!(jobs.iter().filter(|j| on_package(&j.read.0)).count(), PACKAGE_READERS);
        assert!(std::mem::take(&mut l.starved));
        l.finish(&jobs[0], Ok(()));
        assert!(l.start().is_some_and(|j| on_package(&j.read.0)));
    }

    #[test]
    fn a_status_waits_for_every_earlier_read() {
        let mut l = lanes(STAGING);
        l.lanes.insert(0, lane(0, 64));
        let array = status_array(2).cast::<Instance<StatusArray>>();
        let status = |index| {
            add_ref_instance(array);
            Command::Status(Status { array, index })
        };
        read(&mut l, 0, 0, 16);
        read(&mut l, 0, 0, 16);
        l.lanes.get_mut(&0).unwrap().unsubmitted.push(status(0));
        read(&mut l, 0, 0, 16);
        l.lanes.get_mut(&0).unwrap().unsubmitted.push(status(1));
        // SAFETY: an array we created.
        for entry in unsafe { &(*array).state.entries }.iter() {
            entry.store(E_PENDING, Ordering::Release);
        }
        l.submit(0);
        let jobs: Vec<Job> = (0..3).map(|_| l.start().unwrap()).collect();
        // The last read and the first finish; the second then fails.
        l.finish(&jobs[2], Ok(()));
        l.finish(&jobs[0], Ok(()));
        let results = || (result(array.cast(), 0), result(array.cast(), 1));
        assert_eq!(results(), (E_PENDING, E_PENDING));
        l.finish(&jobs[1], Err(INVALID_DATA));
        assert_eq!(results(), (INVALID_DATA, S_OK));
    }
}
