//! Naming addresses, walking stacks and reporting faults.

use std::sync::RwLock;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

use xnw_core::win::*;

/// Title images we mapped: (name, base, size).
static IMAGES: RwLock<Vec<(String, usize, usize)>> = RwLock::new(Vec::new());

pub(crate) fn register(name: &str, base: usize, size: usize) {
    IMAGES.write().unwrap_or_else(|e| e.into_inner()).push((name.to_owned(), base, size));
    // The first image mapped is the title's (the handler tests
    // addresses against it without a lock).
    if TITLE.0.compare_exchange(0, base, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
        TITLE.1.store(size, Ordering::Relaxed);
    }
}

/// The title image: base, size (0 before it is mapped).
static TITLE: (AtomicUsize, AtomicUsize) = (AtomicUsize::new(0), AtomicUsize::new(0));

/// Whether `at` is in the title image.
pub(crate) fn in_title(at: usize) -> bool {
    let base = TITLE.0.load(Ordering::Relaxed);
    base != 0 && at >= base && at < base + TITLE.1.load(Ordering::Relaxed)
}

pub(crate) fn unregister(base: usize) {
    IMAGES.write().unwrap_or_else(|e| e.into_inner()).retain(|&(_, b, _)| b != base);
}

/// `module+0xRVA` for any code address; for title code also the start of the
/// function it is in (from the unwind tables).
pub fn locate(address: usize) -> String {
    if let Ok(images) = IMAGES.try_read()
        && let Some((name, base, _)) = images.iter().find(|(_, b, s)| address >= *b && address < b + s)
    {
        let mut image_base = 0u64;
        // SAFETY: a table lookup for a code address in a loaded module.
        let f = unsafe { RtlLookupFunctionEntry(address as u64, &mut image_base, std::ptr::null_mut()) };
        // SAFETY: a non-null result points into the module's .pdata.
        return match unsafe { f.as_ref() } {
            Some(f) => format!("{name}+{:#x} (fn {:#x})", address - base, f.begin),
            None => format!("{name}+{:#x}", address - base),
        };
    }
    let mut module: HMODULE = std::ptr::null_mut();
    // GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | UNCHANGED_REFCOUNT
    // SAFETY: asks which module holds an address.
    if unsafe { GetModuleHandleExW(4 | 2, address as *const u16, &mut module) } != 0 {
        let mut path = [0u16; 260];
        // SAFETY: the buffer and its length.
        let n = unsafe { GetModuleFileNameW(module, path.as_mut_ptr(), 260) } as usize;
        let path = String::from_utf16_lossy(&path[..n]);
        let name = path.rsplit('\\').next().unwrap_or("?");
        return format!("{name}+{:#x}", address - module as usize);
    }
    format!("{address:#x}")
}

/// Return addresses from `ctx` outward (the faulting `rip` first).
fn walk(ctx: &mut Context, max: usize) -> Vec<usize> {
    let mut out = Vec::new();
    while out.len() < max {
        let rip = ctx.rip();
        if rip == 0 {
            break;
        }
        out.push(rip);
        let mut image_base = 0u64;
        // SAFETY: lookup and unwind over a context we own; frames without
        // unwind data are leaf frames (return address at `rsp`).
        unsafe {
            let f = RtlLookupFunctionEntry(rip as u64, &mut image_base, std::ptr::null_mut());
            if f.is_null() {
                let rsp = ctx.rsp();
                if rsp == 0 || rsp % 8 != 0 {
                    break;
                }
                let ret = (rsp as *const u64).read();
                ctx.0[Context::RIP..Context::RIP + 8].copy_from_slice(&ret.to_le_bytes());
                ctx.0[Context::RSP..Context::RSP + 8].copy_from_slice(&(rsp as u64 + 8).to_le_bytes());
            } else {
                let (mut data, mut frame) = (std::ptr::null_mut(), 0u64);
                RtlVirtualUnwind(0, image_base, rip as u64, f, ctx, &mut data, &mut frame, std::ptr::null_mut());
            }
        }
    }
    out
}

/// `own_too`: our frames too (a fault in a driver under our host calls
/// named none of ours).
fn log_frames(frames: &[usize], own_too: bool) {
    let own = std::env::current_exe().ok().and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned())).unwrap_or_default();
    for (i, &rip) in frames.iter().enumerate() {
        let at = locate(rip);
        // Our own frames (handlers, trampolines) are noise in a title stack.
        if !own_too && !own.is_empty() && at.starts_with(&own) {
            continue;
        }
        xnw_core::warn!("  #{i:<2} {at}");
    }
}

/// Logs the calling thread's stack.
pub fn log_stack_here() {
    let mut ctx = Context::zeroed();
    // SAFETY: fills our own buffer.
    unsafe { RtlCaptureContext(&mut ctx) };
    log_frames(&walk(&mut ctx, 48), false);
}

static REPORTED: AtomicU32 = AtomicU32::new(0);

/// A line the runner wants in the log at the first reported exception and
/// at process exit (`xnw run`: the memory line), with the reason.
static ON_REPORT: std::sync::OnceLock<fn(&str)> = std::sync::OnceLock::new();

/// Sets the [`ON_REPORT`] function, once.
pub fn on_report(f: fn(&str)) {
    let _ = ON_REPORT.set(f);
}

/// A function the runner wants called at the first access violation
/// outside every module (a JIT's), with the integer registers by name
/// (`xnw run`: what each register pointing into a descriptor heap holds).
static ON_FAULT: std::sync::OnceLock<fn(&[(&str, u64)])> = std::sync::OnceLock::new();

/// Sets the [`ON_FAULT`] function, once.
pub fn on_fault(f: fn(&[(&str, u64)])) {
    let _ = ON_FAULT.set(f);
}

/// Calls the [`ON_REPORT`] function, if set.
pub(crate) fn report(why: &str) {
    if let Some(f) = ON_REPORT.get() {
        f(why);
    }
}

fn code_name(code: u32) -> Option<&'static str> {
    Some(match code {
        0xC000_0005 => "access violation",
        0xC000_001D => "illegal instruction",
        0xC000_0096 => "privileged instruction",
        0xC000_00FD => "stack overflow",
        0xC000_0094 => "integer divide by zero",
        0xC000_0409 => "fail fast",
        0x8000_0003 => "breakpoint",
        _ => return None,
    })
}

/// Instructions done by [`crate::amd`] so far (tests read it).
pub(crate) static EMULATED: AtomicU64 = AtomicU64::new(0);

/// Sites already logged; a site is logged once, the first 256 of them.
static SITES: [AtomicUsize; 256] = [const { AtomicUsize::new(0) }; 256];

/// Whether `rip` is a site not seen before. Lock-free: the faulting thread
/// may hold any lock.
fn first_time(rip: usize) -> bool {
    for slot in &SITES {
        match slot.compare_exchange(0, rip, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return true,
            Err(seen) if seen == rip => return false,
            Err(_) => {}
        }
    }
    false
}

/// Exceptions that reach this handler, by (code, address, title
/// caller), with the TSC cycles spent in the handler. Lock-free, no
/// allocation: the faulting thread may hold any lock. A slot is claimed by
/// its key's hash; the key words are written after the claim (a report in
/// the same instant may show the slot without them, then it is skipped).
struct Slot {
    tag: AtomicU64,
    code: AtomicU32,
    at: AtomicUsize,
    from: AtomicUsize,
    count: AtomicU64,
    ticks: AtomicU64,
}
const SLOTS: usize = 1024;
static HIST: [Slot; SLOTS] = [const { Slot { tag: AtomicU64::new(0), code: AtomicU32::new(0), at: AtomicUsize::new(0), from: AtomicUsize::new(0), count: AtomicU64::new(0), ticks: AtomicU64::new(0) } }; SLOTS];
/// Exceptions that found no free slot: count, ticks.
static HIST_FULL: (AtomicU64, AtomicU64) = (AtomicU64::new(0), AtomicU64::new(0));

fn tsc() -> u64 {
    // SAFETY: RDTSC is always there on x86-64.
    unsafe { core::arch::x86_64::_rdtsc() }
}

fn note_exception(code: u32, at: usize, from: usize, ticks: u64) {
    let mut h = (u64::from(code) ^ (at as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (from as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f)) | 1;
    h ^= h >> 29;
    let h = h | 1;
    for i in 0..16 {
        let s = &HIST[(h as usize).wrapping_add(i) % SLOTS];
        let seen = match s.tag.compare_exchange(0, h, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => {
                s.code.store(code, Ordering::Relaxed);
                s.at.store(at, Ordering::Relaxed);
                s.from.store(from, Ordering::Release);
                h
            }
            Err(t) => t,
        };
        if seen == h {
            s.count.fetch_add(1, Ordering::Relaxed);
            s.ticks.fetch_add(ticks, Ordering::Relaxed);
            return;
        }
    }
    HIST_FULL.0.fetch_add(1, Ordering::Relaxed);
    HIST_FULL.1.fetch_add(ticks, Ordering::Relaxed);
}

/// One histogram row: code, address, title caller (0: the address is in
/// the title, or none found), count, ticks.
pub type ExceptionRow = (u32, usize, usize, u64, u64);

/// The histogram's rows with a count, taken (counts reset) or read.
pub fn exception_rows(take: bool) -> Vec<ExceptionRow> {
    let mut out = Vec::new();
    for s in &HIST {
        if s.tag.load(Ordering::Acquire) == 0 {
            continue;
        }
        let (count, ticks) = if take { (s.count.swap(0, Ordering::Relaxed), s.ticks.swap(0, Ordering::Relaxed)) } else { (s.count.load(Ordering::Relaxed), s.ticks.load(Ordering::Relaxed)) };
        let code = s.code.load(Ordering::Relaxed);
        if count != 0 && code != 0 {
            out.push((code, s.at.load(Ordering::Relaxed), s.from.load(Ordering::Acquire), count, ticks));
        }
    }
    out
}

/// For an exception raised outside the title (RaiseException, a C++
/// throw, our code): the first title return address in at most 8 frames.
fn title_caller(ctx: &Context) -> usize {
    let mut copy = Context(ctx.0);
    for _ in 0..8 {
        let rip = copy.rip();
        if rip == 0 {
            return 0;
        }
        if in_title(rip) {
            return rip;
        }
        let mut image_base = 0u64;
        // SAFETY: lookup and unwind over a copy of the context; a frame
        // without unwind data ends the search (no guessing at `rsp`).
        unsafe {
            let f = RtlLookupFunctionEntry(rip as u64, &mut image_base, std::ptr::null_mut());
            if f.is_null() {
                return 0;
            }
            let (mut data, mut frame) = (std::ptr::null_mut(), 0u64);
            RtlVirtualUnwind(0, image_base, rip as u64, f, &mut copy, &mut data, &mut frame, std::ptr::null_mut());
        }
    }
    0
}

/// Title sites rewritten at load ([`crate::rewrite`]), for the
/// exceptions line.
static REWRITTEN: AtomicU64 = AtomicU64::new(0);

pub(crate) fn set_rewritten(n: u64) {
    REWRITTEN.store(n, Ordering::Relaxed);
}

unsafe extern "system" fn on_exception(info: *mut ExceptionPointers) -> i32 {
    let start = tsc();
    // SAFETY: Windows passes valid pointers for the duration of the call.
    let (code, at) = unsafe { ((*(*info).record).code, (*(*info).record).address) };
    // SAFETY: forwarded.
    let r = unsafe { handle(info) };
    // SAFETY: as above; the context is read only.
    let from = if in_title(at) { 0 } else { title_caller(unsafe { &*(*info).context }) };
    note_exception(code, at, from, tsc().wrapping_sub(start));
    r
}

unsafe fn handle(info: *mut ExceptionPointers) -> i32 {
    // SAFETY: Windows passes valid pointers for the duration of the call.
    if unsafe { (*(*info).record).code } == 0xC000_001D {
        // SAFETY: Windows passes valid pointers; the context is ours to change.
        let ctx = unsafe { &mut *(*info).context };
        let rip = ctx.rip();
        // SAFETY: the thread faulted at `rip`, in mapped code.
        if let Some(name) = unsafe { crate::amd::emulate(ctx) } {
            EMULATED.fetch_add(1, Ordering::Relaxed);
            if first_time(rip) {
                xnw_core::info!("emulating {name} at {}", locate(rip));
            }
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    // SAFETY: Windows passes valid pointers for the duration of the call.
    let (rec, ctx) = unsafe { (&*(*info).record, &*(*info).context) };
    let Some(name) = code_name(rec.code) else { return EXCEPTION_CONTINUE_SEARCH };
    // First-chance: the title may handle it. Report the first few in full.
    let reported = REPORTED.fetch_add(1, Ordering::Relaxed);
    if reported >= 4 {
        // A guard page hit ends the run even past the reported few.
        if xnw_core::FATAL_FAULT.load(Ordering::Relaxed) {
            xnw_core::warn!("EXCEPTION {:#010x} ({name}) at {:#x}: stopping the run: an object guard page was hit (XNW_OBJECT_GUARD)", rec.code, rec.address);
            std::process::abort();
        }
        return EXCEPTION_CONTINUE_SEARCH;
    }
    if reported == 0 {
        report(name);
    }
    // SAFETY: plain query.
    let tid = unsafe { GetCurrentThreadId() };
    xnw_core::warn!("EXCEPTION {:#010x} ({name}) at {} on thread {tid}", rec.code, locate(rec.address));
    if rec.code == 0xC000_0005 && rec.param_count >= 2 {
        let how = ["reading", "writing", "?", "?", "?", "?", "?", "?", "executing"];
        xnw_core::warn!("  {} address {:#x}: {}", how.get(rec.params[0]).unwrap_or(&"?"), rec.params[1], memory_at(rec.params[1]));
        // Once: the code around a fault outside every module (a JIT's, such
        // as WARP's shaders), from 0x180 before to 0x40 after, inside its
        // page (what a WARP shader read at 0x388 through).
        static CODE_SHOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let loc = locate(rec.address);
        if !loc.contains('+') && !CODE_SHOWN.swap(true, Ordering::Relaxed) {
            let page = rec.address & !0xfff;
            let (from, to) = (rec.address.saturating_sub(0x180).max(page), (rec.address + 0x40).min(page + 0x1000));
            // SAFETY: the faulting instruction's own page, which the CPU fetched from.
            let bytes = unsafe { std::slice::from_raw_parts(from as *const u8, to - from) };
            let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
            xnw_core::warn!("  code from {from:#x} (fault at +{:#x}): {hex}", rec.address - from);
            if let Some(f) = ON_FAULT.get() {
                let mut regs: Vec<(&str, u64)> = Context::INT_NAMES.iter().enumerate().map(|(i, n)| (*n, ctx.get(Context::RAX + i * 8))).collect();
                // The low 128 bits of xmm0-7 by dword lane (CONTEXT.Xmm0 at
                // 0x1A0): a JIT keeps per-lane descriptor indices there.
                const LANES: [&str; 32] = [
                    "xmm0.0", "xmm0.1", "xmm0.2", "xmm0.3", "xmm1.0", "xmm1.1", "xmm1.2", "xmm1.3", "xmm2.0", "xmm2.1", "xmm2.2", "xmm2.3", "xmm3.0", "xmm3.1", "xmm3.2", "xmm3.3",
                    "xmm4.0", "xmm4.1", "xmm4.2", "xmm4.3", "xmm5.0", "xmm5.1", "xmm5.2", "xmm5.3", "xmm6.0", "xmm6.1", "xmm6.2", "xmm6.3", "xmm7.0", "xmm7.1", "xmm7.2", "xmm7.3",
                ];
                for (k, name) in LANES.iter().enumerate() {
                    regs.push((name, ctx.get(0x1A0 + k * 4) & 0xFFFF_FFFF));
                }
                f(&regs);
            }
        }
    }
    if rec.code == 0xC000_001D || rec.code == 0xC000_0096 {
        // SAFETY: the CPU refused the instruction at the address, so it fetched it.
        xnw_core::warn!("  code {}", unsafe { instruction_bytes(rec.address) });
    }
    for row in Context::INT_NAMES.chunks(4).enumerate() {
        let line: Vec<String> = row.1.iter().enumerate().map(|(i, n)| format!("{n:<3} {:016x}", ctx.get(Context::RAX + (row.0 * 4 + i) * 8))).collect();
        xnw_core::warn!("  {}", line.join("  "));
    }
    let mut copy = Context(ctx.0);
    log_frames(&walk(&mut copy, 48), true);
    // An object guard page was hit: no title handler may carry on.
    if xnw_core::FATAL_FAULT.load(Ordering::Relaxed) {
        xnw_core::warn!("stopping the run: an object guard page was hit (XNW_OBJECT_GUARD)");
        std::process::abort();
    }
    EXCEPTION_CONTINUE_SEARCH
}

/// The bytes at `at` as hex, as many as the longest x86 instruction (15),
/// but none past the end of `at`'s page: the next page may not be mapped.
/// For an instruction the CPU refused, they name it (`0f 01 fd` is
/// `rdpru`), which the address alone does not. (VIctory babe2db.)
///
/// # Safety
/// `at`'s page is readable.
unsafe fn instruction_bytes(at: usize) -> String {
    let n = (0x1000 - (at & 0xfff)).min(15);
    // SAFETY: `n` bytes from `at` stay inside its page, which the caller
    // says is readable.
    let bytes = unsafe { std::slice::from_raw_parts(at as *const u8, n) };
    let hex: Vec<_> = bytes.iter().map(|b| format!("{b:02x}")).collect();
    hex.join(" ")
}

/// What is at `address`, for an access violation there: nothing, memory
/// reserved but never committed, a guard page, or the protection the access
/// broke (with the module, for an image's pages). Each points at a different
/// cause: a wild pointer, an allocation whose commit failed, a stack
/// overrun, a write to code or constants. (VIctory 3390e47, `memory_at`.)
fn memory_at(address: usize) -> String {
    const MEM_IMAGE: u32 = 0x100_0000;
    const PAGE_GUARD: u32 = 0x100;
    // SAFETY: a zeroed MEMORY_BASIC_INFORMATION is valid.
    let mut info: MemoryBasicInformation = unsafe { std::mem::zeroed() };
    // SAFETY: reads the address space's bookkeeping, not `address` itself.
    let n = unsafe { VirtualQueryEx(GetCurrentProcess(), address as *const _, &mut info, size_of_val(&info)) };
    if n == 0 {
        // Above the process's address space: Windows reports a
        // non-canonical address as all ones.
        return "outside the address space".into();
    }
    let protect = match info.state {
        MEM_FREE => return "nothing mapped there".into(),
        MEM_RESERVE => return "reserved, never committed".into(),
        _ if info.protect & PAGE_GUARD != 0 => return "a guard page".into(),
        _ => match info.protect & 0xff {
            PAGE_NOACCESS => "no access".into(),
            PAGE_READONLY => "read-only".into(),
            PAGE_READWRITE => "read-write".into(),
            0x08 => "copy-on-write".into(),
            0x10 => "execute-only".into(),
            PAGE_EXECUTE_READ => "read-execute".into(),
            PAGE_EXECUTE_READWRITE => "read-write-execute".into(),
            0x80 => "copy-on-write, executable".into(),
            _ => format!("protection {:#x}", info.protect),
        },
    };
    if info.kind == MEM_IMAGE { format!("{protect}, in {}", locate(address)) } else { protect }
}

// ---- stall dump -------------------------------------------------------
//
// A run can stop presenting with nothing in the log: no exception, threads
// alive. When the frame count given to [`watch_stalls`] has not moved for
// 10 s, every thread's stack is logged once (until frames move again), with
// the start address of each thread and notes of who holds our locks.
// Threads are suspended one at a time and only while their registers and
// return addresses are taken into memory reserved beforehand: a suspended
// thread may hold the heap's or the log's lock.

/// Notes for a stall dump (lock owners): set by the runner.
static STALL_NOTES: std::sync::OnceLock<fn() -> String> = std::sync::OnceLock::new();

/// Starts the stall watch over `frames` (the presented-frame count), once.
/// `notes` adds lines to a dump (who holds which lock).
pub fn watch_stalls(frames: fn() -> u64, notes: fn() -> String) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = STALL_NOTES.set(notes);
        let _ = std::thread::Builder::new().name("xnw stall watch".into()).spawn(move || {
            let (mut last, mut since, mut dumped) = (frames(), std::time::Instant::now(), false);
            loop {
                std::thread::sleep(std::time::Duration::from_millis(500));
                let now = frames();
                if now != last {
                    (last, since, dumped) = (now, std::time::Instant::now(), false);
                    continue;
                }
                // Before the first frame (loading) only after a minute.
                let limit = if now == 0 { 60 } else { 10 };
                if !dumped && since.elapsed().as_secs() >= limit {
                    dumped = true;
                    dump_threads(&format!("no frame presented for {limit} s (frame {now})"));
                }
            }
        });
    });
}

/// Logs every thread's stack but the caller's.
pub fn dump_threads(why: &str) {
    const FRAMES: usize = 32;
    // SAFETY: plain queries.
    let (pid, me) = unsafe { (GetCurrentProcessId(), GetCurrentThreadId()) };
    let mut ids = Vec::new();
    // SAFETY: a thread snapshot of the system, walked with a sized entry.
    unsafe {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snap != INVALID_HANDLE_VALUE {
            let mut e: ThreadEntry32 = std::mem::zeroed();
            e.size = size_of::<ThreadEntry32>() as u32;
            let mut ok = Thread32First(snap, &mut e) != 0;
            while ok {
                if e.owner_process_id == pid && e.thread_id != me {
                    ids.push(e.thread_id);
                }
                ok = Thread32Next(snap, &mut e) != 0;
            }
            CloseHandle(snap);
        }
    }
    // Room for every stack before any thread is stopped.
    let mut stacks: Vec<(u32, usize, Vec<usize>)> = ids.iter().map(|&t| (t, 0, Vec::with_capacity(FRAMES))).collect();
    for (tid, start, frames) in &mut stacks {
        // SAFETY: rights to suspend, read the context and query the thread;
        // the thread is resumed before anything that may take a lock.
        unsafe {
            let h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, 0, *tid);
            if h.is_null() {
                continue;
            }
            let mut at = 0usize;
            NtQueryInformationThread(h, THREAD_QUERY_SET_WIN32_START_ADDRESS, (&raw mut at).cast(), size_of::<usize>() as u32, std::ptr::null_mut());
            *start = at;
            if SuspendThread(h) != u32::MAX {
                let mut ctx = Context::zeroed();
                ctx.0[Context::FLAGS..Context::FLAGS + 4].copy_from_slice(&CONTEXT_CONTROL_INTEGER.to_le_bytes());
                if GetThreadContext(h, &mut ctx) != 0 {
                    walk_into(&mut ctx, frames, FRAMES);
                }
                ResumeThread(h);
            }
            CloseHandle(h);
        }
    }
    xnw_core::warn!("STALL: {why}: {} threads", stacks.len());
    if let Some(notes) = STALL_NOTES.get() {
        for line in notes().lines() {
            xnw_core::warn!("  {line}");
        }
    }
    for (tid, start, frames) in &stacks {
        xnw_core::warn!("thread {tid} (start {})", locate(*start));
        for (i, &rip) in frames.iter().enumerate() {
            match symbol(rip) {
                Some(name) => xnw_core::warn!("  #{i:<2} {} {name}", locate(rip)),
                None => xnw_core::warn!("  #{i:<2} {}", locate(rip)),
            }
        }
    }
}

/// The function at `address` from a PDB (ours, beside xnw.exe; system
/// modules' exports), through dbghelp, set up at the first use: for stall
/// dumps only (dbghelp is not for a faulting thread).
fn symbol(address: usize) -> Option<String> {
    static READY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // SAFETY: dbghelp set up once for our own process.
    let ready = *READY.get_or_init(|| unsafe {
        const UNDNAME: u32 = 0x2;
        const DEFERRED_LOADS: u32 = 0x4;
        SymSetOptions(UNDNAME | DEFERRED_LOADS);
        SymInitializeW(GetCurrentProcess(), std::ptr::null(), 1) != 0
    });
    if !ready {
        return None;
    }
    // SAFETY: zeroed is a valid SYMBOL_INFOW; dbghelp fills it.
    let mut info: SymbolInfoW = unsafe { std::mem::zeroed() };
    info.size_of_struct = SYMBOL_INFOW_SIZE;
    info.max_name_len = 512;
    let mut displacement = 0u64;
    // SAFETY: the struct and its room, for our own process.
    if unsafe { SymFromAddrW(GetCurrentProcess(), address as u64, &mut displacement, &mut info) } == 0 {
        return None;
    }
    let n = (info.name_len as usize).min(512);
    Some(format!("{}+{displacement:#x}", String::from_utf16_lossy(&info.name[..n])))
}

/// [`walk`] into `out` (reserved: no allocation while a thread is stopped).
fn walk_into(ctx: &mut Context, out: &mut Vec<usize>, max: usize) {
    while out.len() < max.min(out.capacity()) {
        let rip = ctx.rip();
        if rip == 0 {
            break;
        }
        out.push(rip);
        let mut image_base = 0u64;
        // SAFETY: as in `walk`; the stack belongs to the stopped thread.
        unsafe {
            let f = RtlLookupFunctionEntry(rip as u64, &mut image_base, std::ptr::null_mut());
            if f.is_null() {
                let rsp = ctx.rsp();
                if rsp == 0 || rsp % 8 != 0 {
                    break;
                }
                let ret = (rsp as *const u64).read();
                ctx.0[Context::RIP..Context::RIP + 8].copy_from_slice(&ret.to_le_bytes());
                ctx.0[Context::RSP..Context::RSP + 8].copy_from_slice(&(rsp as u64 + 8).to_le_bytes());
            } else {
                let (mut data, mut frame) = (std::ptr::null_mut(), 0u64);
                RtlVirtualUnwind(0, image_base, rip as u64, f, ctx, &mut data, &mut frame, std::ptr::null_mut());
            }
        }
    }
}

/// Installs the fault reporter (first in the handler chain), once. It also
/// does the Zen 2 instructions this CPU lacks ([`crate::amd`]).
pub fn install() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: registers a handler that reads the exception it is given and
    // changes only the context of an instruction it emulates.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, on_exception);
    });
}

/// The exception histogram's top 10 by handler time every 10 s, on
/// one line, `none` when empty (a thread of its own, started once).
pub fn report_exceptions() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = std::thread::Builder::new().name("xnw exceptions".into()).spawn(|| {
            loop {
                let (t0, c0) = (std::time::Instant::now(), tsc());
                std::thread::sleep(std::time::Duration::from_secs(10));
                let per_ms = tsc().wrapping_sub(c0) as f64 / (t0.elapsed().as_secs_f64() * 1000.0).max(1e-3);
                xnw_core::info!("{}", exceptions_text(exception_rows(true), per_ms));
            }
        });
    });
}

fn exceptions_text(mut rows: Vec<ExceptionRow>, ticks_per_ms: f64) -> String {
    let (full, full_ticks) = (HIST_FULL.0.swap(0, Ordering::Relaxed), HIST_FULL.1.swap(0, Ordering::Relaxed));
    let total: u64 = rows.iter().map(|r| r.3).sum::<u64>() + full;
    let ms = |t: u64| t as f64 / ticks_per_ms;
    let all_ms = ms(rows.iter().map(|r| r.4).sum::<u64>() + full_ticks);
    rows.sort_by(|a, b| b.4.cmp(&a.4));
    let parts: Vec<String> = rows
        .iter()
        .take(10)
        .map(|&(code, at, from, count, ticks)| {
            let name = code_name(code).map(|n| format!(" ({n})")).unwrap_or_default();
            let from = if from == 0 { String::new() } else { format!(" from {}", locate(from)) };
            format!("{code:#010x}{name} at {}{from}: {count}, {:.1} ms", locate(at), ms(ticks))
        })
        .collect();
    let top = if parts.is_empty() { "none".to_owned() } else { parts.join("; ") };
    format!("exceptions: last 10 s: {total} in this handler, {all_ms:.1} ms in it, {} Zen 2 sites rewritten at load (XNW_CPU_REWRITE), {full} unkeyed; {top}", REWRITTEN.load(Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_at_names_what_an_access_hit() {
        const MEM_COMMIT: u32 = 0x1000;
        const MEM_RELEASE: u32 = 0x8000;
        // SAFETY: reserves four pages of ours and commits three of them;
        // nothing reads or writes them.
        let base = unsafe {
            let base = VirtualAlloc(std::ptr::null(), 0x4000, MEM_RESERVE, PAGE_NOACCESS);
            assert!(!base.is_null());
            let page = |i: usize, protect| {
                let at = base.cast::<u8>().add(i * 0x1000).cast();
                assert!(!VirtualAlloc(at, 0x1000, MEM_COMMIT, protect).is_null());
            };
            page(1, PAGE_READONLY);
            page(2, PAGE_READWRITE | 0x100);
            page(3, PAGE_NOACCESS);
            base as usize
        };
        assert_eq!(memory_at(base + 8), "reserved, never committed");
        assert_eq!(memory_at(base + 0x1008), "read-only");
        assert_eq!(memory_at(base + 0x2008), "a guard page");
        assert_eq!(memory_at(base + 0x3008), "no access");
        // SAFETY: releases the reservation made above.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
        // Not asserted "nothing mapped there": another test thread may map the freed
        // range at once (CI render e879853: an executable arena landed there, "read-execute");
        // the unmapped case is the address-0 line below.
        assert_eq!(memory_at(0), "nothing mapped there");
        assert_eq!(memory_at(usize::MAX), "outside the address space");
        // This test's own code: an image page, named by its module.
        let code = memory_at(memory_at_names_what_an_access_hit as *const () as usize);
        assert!(code.starts_with("read-execute, in xnw_loader-"), "{code}");
    }

    /// A stall dump stops and walks every other thread and lets it go.
    #[test]
    fn stall_dump_walks_other_threads() {
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let other = std::thread::spawn(move || {
            let _ = rx.recv();
        });
        dump_threads("test");
        tx.send(()).unwrap();
        other.join().unwrap();
    }

    #[test]
    fn instruction_bytes_stop_at_the_end_of_their_page() {
        const MEM_COMMIT: u32 = 0x1000;
        const MEM_RELEASE: u32 = 0x8000;
        // SAFETY: two fresh pages of ours, the second never committed, so a
        // read past the end of the first faults.
        unsafe {
            let page = VirtualAlloc(std::ptr::null(), 0x2000, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!VirtualAlloc(page.cast(), 0x1000, MEM_COMMIT, PAGE_READWRITE).is_null());
            let code = [0x0f_u8, 0x01, 0xfd, 0x48, 0x89, 0xc1];
            std::ptr::copy_nonoverlapping(code.as_ptr(), page, 6);
            assert_eq!(instruction_bytes(page as usize), "0f 01 fd 48 89 c1 00 00 00 00 00 00 00 00 00");
            let end = page.add(0x1000 - 3);
            std::ptr::copy_nonoverlapping(code.as_ptr(), end, 3);
            assert_eq!(instruction_bytes(end as usize), "0f 01 fd");
            VirtualFree(page.cast(), 0, MEM_RELEASE);
        }
    }
}
