//! `XNW_IMPORT_TRACE=<file>[,until=<s>]`: every title import call the
//! binder bound to us or to the host (not title DLLs), recorded compactly
//! until `<s>` seconds of the log clock (default 30): name, caller,
//! arguments, result, the thread's last error before and after, and for
//! the pure calls of `xnw_core::imprec::PURE` the bytes they read and
//! write. Format and limits: `xnw_core::imprec`. The replay test is
//! `xnw-hle/tests/import_replay.rs`.
//!
//! The thunk calls the target from a frame of its own (with unwind data,
//! so exceptions and unwinds pass through it) after copying 16 stack
//! arguments (as far as the thread's stack goes): calls with more than 20
//! arguments, and the few that must see their caller's frame (setjmp,
//! unwinding, exception handlers), are left unrecorded. The thread's last
//! error is restored after our work. Unset: nothing is installed.

use std::collections::HashSet;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use xnw_core::imprec::{Arg, MAX_BYTES, PER_SITE, PURE, Record};

/// What a recording thunk knows. `target` first and `pure` second: the
/// trampoline reads them.
#[repr(C)]
pub struct RecSite {
    pub target: usize,
    /// Index in `PURE` plus 1, or 0.
    pub pure: usize,
    pub dll: String,
    pub symbol: String,
    pub provider: char,
    pub calls: AtomicU64,
    pub distinct: AtomicU64,
}

const _: () = assert!(std::mem::offset_of!(RecSite, target) == 0 && std::mem::offset_of!(RecSite, pure) == 8);

/// The trampoline's save area (`rsp + 0xA0`); the copied stack arguments
/// are the 16 words below it.
#[repr(C)]
struct Frame {
    args: [u64; 4],
    le_before: u64,
    rax: u64,
    le_after: u64,
}

std::arch::global_asm!(
    r#"
    .globl xnw_record_entry
    .def xnw_record_entry; .scl 2; .type 32; .endef
    .seh_proc xnw_record_entry
xnw_record_entry:
    push rbx
    .seh_pushreg rbx
    sub rsp, 0x170
    .seh_stackalloc 0x170
    .seh_endprologue
    mov rbx, r10
    mov [rsp + 0xA0], rcx
    mov [rsp + 0xA8], rdx
    mov [rsp + 0xB0], r8
    mov [rsp + 0xB8], r9
    movups [rsp + 0xE0], xmm0
    movups [rsp + 0xF0], xmm1
    movups [rsp + 0x100], xmm2
    movups [rsp + 0x110], xmm3
    mov eax, dword ptr gs:[0x68]
    mov [rsp + 0xC0], rax
    mov r10, qword ptr gs:[0x08]
    lea rax, [rsp + 0x1A0]
    sub r10, rax
    shr r10, 3
    cmp r10, 16
    jbe 1f
    mov r10d, 16
1:
    xor r11d, r11d
2:
    cmp r11, r10
    jae 3f
    mov rax, [rsp + r11*8 + 0x1A0]
    mov [rsp + r11*8 + 0x20], rax
    inc r11
    jmp 2b
3:
    cmp qword ptr [rbx + 8], 0
    je 4f
    mov rcx, rbx
    lea rdx, [rsp + 0xA0]
    call xnw_record_pre
    mov eax, [rsp + 0xC0]
    mov dword ptr gs:[0x68], eax
4:
    mov rcx, [rsp + 0xA0]
    mov rdx, [rsp + 0xA8]
    mov r8, [rsp + 0xB0]
    mov r9, [rsp + 0xB8]
    movups xmm0, [rsp + 0xE0]
    movups xmm1, [rsp + 0xF0]
    movups xmm2, [rsp + 0x100]
    movups xmm3, [rsp + 0x110]
    call qword ptr [rbx]
    mov [rsp + 0xC8], rax
    mov [rsp + 0xD8], rdx
    movups [rsp + 0x120], xmm0
    mov eax, dword ptr gs:[0x68]
    mov [rsp + 0xD0], rax
    mov rcx, rbx
    lea rdx, [rsp + 0xA0]
    mov r8, [rsp + 0x178]
    call xnw_record_post
    mov eax, [rsp + 0xD0]
    mov dword ptr gs:[0x68], eax
    mov rax, [rsp + 0xC8]
    mov rdx, [rsp + 0xD8]
    movups xmm0, [rsp + 0x120]
    add rsp, 0x170
    pop rbx
    ret
    .seh_endproc
"#
);

unsafe extern "C" {
    fn xnw_record_entry();
}

pub(crate) fn entry() -> usize {
    xnw_record_entry as usize
}

struct State {
    out: Mutex<std::io::BufWriter<std::fs::File>>,
    until: f64,
    seen: Vec<Mutex<HashSet<u64>>>,
    done: AtomicBool,
    sites: Mutex<Vec<&'static RecSite>>,
}

fn state() -> Option<&'static State> {
    static S: OnceLock<Option<State>> = OnceLock::new();
    S.get_or_init(|| {
        let v = std::env::var("XNW_IMPORT_TRACE").ok()?;
        let mut parts = v.split(',');
        let path = parts.next()?.trim().to_owned();
        let until = parts.find_map(|p| p.trim().strip_prefix("until=").and_then(|s| s.parse().ok())).unwrap_or(30.0);
        if let Some(d) = std::path::Path::new(&path).parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let file = match std::fs::File::create(&path) {
            Ok(f) => f,
            Err(e) => {
                xnw_core::warn!("import trace: {path}: {e}; nothing recorded");
                return None;
            }
        };
        let mut out = std::io::BufWriter::new(file);
        let _ = writeln!(out, "# xnw import trace v1 (XNW_IMPORT_TRACE={v})");
        xnw_core::info!("import trace: every import call bound to us or the host, to {until} s, into {path}");
        Some(State { out: Mutex::new(out), until, seen: (0..16).map(|_| Mutex::new(HashSet::new())).collect(), done: AtomicBool::new(false), sites: Mutex::new(Vec::new()) })
    })
    .as_ref()
}

/// Calls that must see their caller's own frame or never return normally.
fn unsafe_to_wrap(symbol: &str) -> bool {
    let s = symbol.to_ascii_lowercase();
    s.contains("setjmp") || s.contains("longjmp") || s.contains("unwind") || s.contains("handler") || s.starts_with("__cxx") || s.starts_with("_cxx")
        || s.starts_with("__c_specific") || s.contains("restorecontext") || s == "ntcontinue" || s == "rtlcapturecontext" || s.contains("exitthread")
        || s.contains("exitprocess") || s == "raiseexception" || s.contains("terminate") || s == "abort" || s == "_exit" || s == "exit"
}

/// Whether `dll!symbol` gets a recording thunk (XNW_IMPORT_TRACE set).
pub(crate) fn wanted(symbol: &str) -> bool {
    state().is_some() && !unsafe_to_wrap(symbol)
}

pub(crate) fn site(dll: &str, symbol: &str, target: usize, hle: bool) -> &'static RecSite {
    let pure = xnw_core::imprec::pure(symbol).map_or(0, |(i, _)| i + 1);
    let s: &'static RecSite = Box::leak(Box::new(RecSite {
        target,
        pure,
        dll: dll.to_owned(),
        symbol: symbol.to_owned(),
        provider: if hle { 'H' } else { 'W' },
        calls: AtomicU64::new(0),
        distinct: AtomicU64::new(0),
    }));
    if let Some(st) = state() {
        st.sites.lock().unwrap_or_else(|e| e.into_inner()).push(s);
    }
    s
}

thread_local! {
    /// A pure call's arguments and input bytes, from before the call.
    static PRE: std::cell::RefCell<Option<(usize, Vec<u64>, Vec<(bool, usize, Vec<u8>)>)>> = const { std::cell::RefCell::new(None) };
}

/// The `n` arguments of a call: 4 registers, then the copied stack words.
fn args_of(frame: &Frame, n: usize) -> Vec<u64> {
    let stack = std::ptr::from_ref(frame).cast::<u64>().wrapping_sub(16);
    (0..n)
        .map(|i| {
            if i < 4 {
                frame.args[i]
            } else {
                // SAFETY: the trampoline's 16 copied stack words.
                unsafe { stack.add(i - 4).read() }
            }
        })
        .collect()
}

/// # Safety
/// `p` points at `n` readable bytes (the title's argument to a pure call).
unsafe fn bytes(p: u64, n: usize) -> Vec<u8> {
    if p == 0 {
        return Vec::new();
    }
    // SAFETY: the caller's guarantee.
    unsafe { std::slice::from_raw_parts(p as *const u8, n.min(MAX_BYTES)) }.to_vec()
}

/// # Safety
/// `p` is null or a terminated string of `unit`-byte characters.
unsafe fn string(p: u64, len: Option<u64>, unit: usize) -> Vec<u8> {
    if p == 0 {
        return Vec::new();
    }
    let n = match len.filter(|&l| l as i32 >= 0) {
        Some(l) => l as usize,
        None => {
            let mut n = 0;
            // SAFETY: the caller's guarantee: terminated.
            while n < MAX_BYTES / unit && unsafe { std::slice::from_raw_parts((p as usize + n * unit) as *const u8, unit) }.iter().any(|&b| b != 0) {
                n += 1;
            }
            n + 1
        }
    };
    // SAFETY: as above.
    unsafe { bytes(p, n * unit) }
}

#[unsafe(no_mangle)]
extern "system" fn xnw_record_pre(site: &RecSite, frame: &Frame) {
    let Some(st) = state() else { return };
    if st.done.load(Ordering::Relaxed) {
        return;
    }
    let spec = &PURE[site.pure - 1];
    let args = args_of(frame, spec.args.len());
    let mut inputs = Vec::new();
    for (i, a) in spec.args.iter().enumerate() {
        let len = |k: i8| (k >= 0).then(|| args[k as usize]);
        // SAFETY: the title passes valid buffers to these calls.
        let b = unsafe {
            match *a {
                Arg::InA(k) => string(args[i], len(k), 1),
                Arg::InW(k) => string(args[i], len(k), 2),
                Arg::InOut4 => bytes(args[i], 4),
                _ => continue,
            }
        };
        if args[i] != 0 {
            inputs.push((true, i, b));
        }
    }
    PRE.with(|p| *p.borrow_mut() = Some((std::ptr::from_ref(frame) as usize, args, inputs)));
}

fn shown(v: u64) -> Option<u64> {
    if v < 0x1_0000 || v >= 0xFFFF_FFFF_0000_0000 { Some(v) } else { None }
}

#[unsafe(no_mangle)]
extern "system" fn xnw_record_post(site: &RecSite, frame: &Frame, ret: usize) {
    let Some(st) = state() else { return };
    site.calls.fetch_add(1, Ordering::Relaxed);
    if st.done.load(Ordering::Relaxed) {
        return;
    }
    if xnw_core::uptime().as_secs_f64() > st.until {
        finish(st);
        return;
    }
    if site.distinct.load(Ordering::Relaxed) >= PER_SITE {
        return;
    }
    let (args, mut b) = if site.pure != 0 {
        let pre = PRE.with(|p| p.borrow_mut().take()).filter(|(f, _, _)| *f == std::ptr::from_ref(frame) as usize);
        let Some((_, args, mut b)) = pre else { return };
        let spec = &PURE[site.pure - 1];
        for (i, a) in spec.args.iter().enumerate() {
            let p = args[i];
            // SAFETY: the call wrote these buffers.
            let out = unsafe {
                match *a {
                    Arg::Out(n) => bytes(p, n),
                    Arg::OutR { mult, .. } => bytes(p, frame.rax as u32 as usize * mult as usize),
                    Arg::OutLenPtr { len } => {
                        let n = if args[len as usize] != 0 { (args[len as usize] as *const u32).read_unaligned() as usize } else { 0 };
                        bytes(p, n)
                    }
                    Arg::InOut4 => bytes(p, 4),
                    _ => continue,
                }
            };
            if p != 0 {
                b.push((false, i, out));
            }
        }
        let shown: Vec<Option<u64>> = args.iter().zip(spec.args).map(|(&v, a)| if *a == Arg::S || v == 0 { Some(v) } else { None }).collect();
        (shown, b)
    } else {
        (frame.args.iter().map(|&v| shown(v)).collect(), Vec::new())
    };
    b.sort_by_key(|(i, n, _)| (!*i, *n));
    let rec = Record {
        dll: site.dll.clone(),
        symbol: site.symbol.clone(),
        provider: site.provider,
        caller: String::new(),
        args,
        ret: frame.rax,
        le_before: frame.le_before as u32,
        le_after: frame.le_after as u32,
        bytes: b,
    };
    let key = {
        use std::hash::{Hash, Hasher};
        let mut h = std::hash::DefaultHasher::new();
        (std::ptr::from_ref(site) as usize, ret, &rec.args, rec.ret, rec.le_before, rec.le_after, &rec.bytes).hash(&mut h);
        h.finish()
    };
    let shard = &st.seen[(std::ptr::from_ref(site) as usize >> 4) % st.seen.len()];
    if !shard.lock().unwrap_or_else(|e| e.into_inner()).insert(key) {
        return;
    }
    if site.distinct.fetch_add(1, Ordering::Relaxed) >= PER_SITE {
        return;
    }
    let caller = crate::crash::locate(ret);
    let rec = Record { caller: caller.split_whitespace().next().unwrap_or("?").to_owned(), ..rec };
    let mut out = st.out.lock().unwrap_or_else(|e| e.into_inner());
    let _ = writeln!(out, "{}", rec.line());
    let _ = out.flush();
}

/// The end of the recording: every import's call count.
fn finish(st: &State) {
    if st.done.swap(true, Ordering::Relaxed) {
        return;
    }
    let mut out = st.out.lock().unwrap_or_else(|e| e.into_inner());
    let sites = st.sites.lock().unwrap_or_else(|e| e.into_inner());
    let mut calls = 0;
    for s in sites.iter() {
        let n = s.calls.load(Ordering::Relaxed);
        calls += n;
        if n > 0 {
            let _ = writeln!(out, "C\t{}\t{}\t{n}\t{}", s.dll, s.symbol, s.distinct.load(Ordering::Relaxed).min(PER_SITE));
        }
    }
    let _ = writeln!(out, "# end at {:.1} s: {calls} calls through {} imports", st.until, sites.len());
    let _ = out.flush();
    xnw_core::info!("import trace: ended at {:.1} s, {calls} calls through {} recorded imports", st.until, sites.len());
}

#[cfg(test)]
mod tests {
    use crate::stubs::Thunks;

    extern "system" fn six(a: u64, b: u64, c: u64, d: u64, e: u64, f: u64) -> u64 {
        // SAFETY: plain call on this thread.
        unsafe { xnw_core::win::SetLastError(77) };
        a + b * 10 + c * 100 + d * 1000 + e * 10_000 + f * 100_000
    }

    fn last_error() -> i32 {
        std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
    }

    /// Six arguments (two on the stack) arrive, the result and the callee's
    /// last error come back, and a pure host call records its bytes.
    #[test]
    fn a_recorded_call_keeps_arguments_result_and_last_error() {
        let path = std::env::temp_dir().join(format!("xnw-record-{}.tsv", std::process::id()));
        // SAFETY: the only test in this binary that reads the variable.
        unsafe { std::env::set_var("XNW_IMPORT_TRACE", format!("{},until=600", path.display())) };
        let qpf = crate::host_export("kernel32.dll", &xnw_pe::Symbol::Name("QueryPerformanceFrequency".into())).unwrap();
        let mut t = Thunks::new(2).unwrap();
        let thunk = t.recorded("test.dll", "six", six as usize, true);
        let pure = t.recorded("kernel32.dll", "QueryPerformanceFrequency", qpf, false);
        t.seal().unwrap();
        // SAFETY: the thunks forward to functions of these signatures.
        let f: extern "system" fn(u64, u64, u64, u64, u64, u64) -> u64 = unsafe { std::mem::transmute(thunk) };
        let q: extern "system" fn(*mut u64) -> i32 = unsafe { std::mem::transmute(pure) };
        // SAFETY: plain call.
        unsafe { xnw_core::win::SetLastError(5) };
        let r = f(1, 2, 3, 4, 5, 6);
        let le = last_error();
        assert_eq!((r, le), (654_321, 77));
        let mut freq = 0u64;
        assert_eq!(q(&mut freq), 1);
        assert!(freq > 0);
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let recs: Vec<_> = text.lines().filter_map(xnw_core::imprec::Record::parse).collect();
        let six = recs.iter().find(|r| r.symbol == "six").expect(&text);
        assert_eq!((six.provider, six.ret, six.le_before, six.le_after), ('H', 654_321, 5, 77));
        assert_eq!(six.args, [Some(1), Some(2), Some(3), Some(4)]);
        let q = recs.iter().find(|r| r.symbol == "QueryPerformanceFrequency").expect(&text);
        assert_eq!(q.output(0), Some(&freq.to_le_bytes()[..]));
        assert_eq!(q.args, [None]);
    }
}
