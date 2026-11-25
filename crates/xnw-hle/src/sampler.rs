//! `XNW_CPU_SAMPLE=<hz>[,from=<s>,to=<s>]`: which title code burns
//! the CPU. A thread of ours stops each title thread `hz` times a second
//! (default 200), takes its RIP and the word at its RSP, and lets it go
//! (nothing is allocated or locked while a thread is stopped). Samples are
//! bucketed by function start from a table of function starts (`XNW_FUNCS`);
//! the word at RSP counts as a caller only where the bytes before it are a
//! `call`.
//! Outside the image a sample goes to its module: xnw (ours), ntdll,
//! kernelbase/kernel32, other. Every 10 s: all title threads' top 8
//! functions with share and their commonest caller, then the six busiest
//! threads' top 3. Title threads are those without a name (ours are named).
//!
//! Cost: about 30 us per stopped thread; 200 Hz over ~30 threads is under
//! a fifth of one core of twelve.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use xnw_core::win::*;

/// The functions table: starts (rva), ends (rva), names; sorted by start.
struct Funcs {
    starts: Vec<u32>,
    ends: Vec<u32>,
    names: Vec<String>,
}

fn load_funcs() -> Option<Funcs> {
    let path = std::env::var("XNW_FUNCS").unwrap_or_else(|_| r"functions.tsv".into());
    let text = std::fs::read_to_string(&path).ok()?;
    let mut rows: Vec<(u32, u32, String)> = text
        .lines()
        .skip(1)
        .filter_map(|l| {
            let mut f = l.split('\t');
            let start = u64::from_str_radix(f.next()?.trim_start_matches("0x"), 16).ok()?;
            let end = u64::from_str_radix(f.next()?.trim_start_matches("0x"), 16).ok()?;
            let _size = f.next();
            let name = f.next().unwrap_or("").to_string();
            Some(((start - 0x1_4000_0000) as u32, (end - 0x1_4000_0000) as u32, name))
        })
        .collect();
    rows.sort_unstable_by_key(|r| r.0);
    xnw_core::info!("cpu sample: {} functions from {path}", rows.len());
    Some(Funcs { starts: rows.iter().map(|r| r.0).collect(), ends: rows.iter().map(|r| r.1).collect(), names: rows.into_iter().map(|r| r.2).collect() })
}

impl Funcs {
    /// The function holding `rva`: its start, or `None`.
    fn of(&self, rva: u32) -> Option<usize> {
        let i = self.starts.partition_point(|&s| s <= rva).checked_sub(1)?;
        (rva < self.ends[i]).then_some(i)
    }
}

/// `XNW_CPU_SAMPLE=<hz>[,from=<s>,to=<s>]`.
pub(crate) fn parse(v: &str) -> Option<(u32, u64, u64)> {
    let mut parts = v.split(',');
    let hz = parts.next()?.trim();
    let hz = if hz.is_empty() { 200 } else { hz.parse().ok()? };
    let (mut from, mut to) = (0, u64::MAX);
    for p in parts {
        match p.trim().split_once('=') {
            Some(("from", s)) => from = s.parse().ok()?,
            Some(("to", s)) => to = s.parse().ok()?,
            _ => return None,
        }
    }
    (hz > 0).then_some((hz, from, to))
}

/// Starts the sampler once, if asked for.
pub(crate) fn ensure() {
    static STARTED: AtomicBool = AtomicBool::new(false);
    if STARTED.load(Ordering::Relaxed) || STARTED.swap(true, Ordering::Relaxed) {
        return;
    }
    let Some((hz, from, to)) = std::env::var("XNW_CPU_SAMPLE").ok().as_deref().and_then(parse) else { return };
    let _ = std::thread::Builder::new().name("xnw cpu sampler".into()).spawn(move || run(hz, from, to));
}

/// Where a sample landed: a title function (or image page) and, when the
/// thread was outside the title (ntdll, kernelbase, ours), the leaf there.
#[derive(Hash, PartialEq, Eq, Clone, Copy)]
enum Place {
    Func(usize, Leaf),
    Image(u32, Leaf),
    Module(Leaf),
}

/// The leaf of a sample outside the title: module and nearest export.
#[derive(Hash, PartialEq, Eq, Clone, Copy)]
enum Leaf {
    Title,
    /// Module index ([`MODULES`]) and export index (`usize::MAX`: none).
    In(usize, usize),
}

/// Modules a leaf is named in: name, base, end, exports (rva, name) sorted.
struct Module {
    name: &'static str,
    base: usize,
    end: usize,
    exports: Vec<(u32, String)>,
}

/// A loaded module's range and export names (its export directory).
fn module_info(name: &'static str) -> Option<Module> {
    let w: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
    // SAFETY: a loaded system module's headers and export directory, read
    // within its image.
    unsafe {
        let base = GetModuleHandleW(w.as_ptr()) as usize;
        if base == 0 {
            return None;
        }
        let rd32 = |at: usize| (at as *const u32).read_unaligned();
        let nt = base + rd32(base + 0x3c) as usize;
        let size = rd32(nt + 0x50) as usize;
        let dir = rd32(nt + 0x88) as usize;
        let mut exports = Vec::new();
        if dir != 0 {
            let d = base + dir;
            let (names, funcs, names_at, ords) = (rd32(d + 0x18) as usize, rd32(d + 0x1c) as usize, rd32(d + 0x20) as usize, rd32(d + 0x24) as usize);
            for i in 0..names {
                let name_rva = rd32(base + names_at + 4 * i) as usize;
                let ord = ((base + ords + 2 * i) as *const u16).read_unaligned() as usize;
                let f = rd32(base + funcs + 4 * ord);
                let s = std::ffi::CStr::from_ptr((base + name_rva) as *const std::ffi::c_char).to_string_lossy().into_owned();
                exports.push((f, s));
            }
            exports.sort_unstable_by_key(|e| e.0);
        }
        Some(Module { name, base, end: base + size, exports })
    }
}

fn run(hz: u32, from: u64, to: u64) {
    let funcs = load_funcs();
    let period = std::time::Duration::from_micros(1_000_000 / u64::from(hz));
    // SAFETY: plain queries.
    let (pid, me) = unsafe { (GetCurrentProcessId(), GetCurrentThreadId()) };
    let modules: Vec<Module> = ["ntdll.dll", "kernelbase.dll", "kernel32.dll"].into_iter().filter_map(module_info).collect();
    // SAFETY: our own module.
    let ours = unsafe { GetModuleHandleW(std::ptr::null()) as usize };
    let mut threads: Vec<(u32, HANDLE, u64)> = Vec::new();
    let mut names: HashMap<u32, String> = HashMap::new();
    let mut cpu_start: HashMap<u32, u64> = HashMap::new();
    let mut refreshed = std::time::Instant::now() - std::time::Duration::from_secs(10);
    let mut window = std::time::Instant::now();
    let mut all: HashMap<Place, u64> = HashMap::new();
    let mut by_thread: HashMap<u32, HashMap<Place, u64>> = HashMap::new();
    let mut chains: HashMap<Chain, u64> = HashMap::new();
    let mut total = 0u64;
    let mut stack = [0usize; STACK];
    xnw_core::info!("cpu sample: {hz} Hz from {from} s to {}, running threads only (cycle-time deltas)", if to == u64::MAX { "the end".into() } else { format!("{to} s") });
    let cpu_ms = |h: HANDLE| -> u64 {
        let (mut c, mut e, mut k, mut u) = (0, 0, 0, 0);
        // SAFETY: an open thread, out-parameters.
        unsafe { GetThreadTimes(h, &mut c, &mut e, &mut k, &mut u) };
        (k + u) / 10_000
    };
    loop {
        std::thread::sleep(period);
        let now = xnw_core::uptime().as_secs();
        if now < from {
            continue;
        }
        if now > to {
            return;
        }
        let base = crate::title_base();
        if refreshed.elapsed() >= std::time::Duration::from_secs(1) {
            refreshed = std::time::Instant::now();
            let mut kept: HashMap<u32, u64> = threads.iter().map(|&(t, _, c)| (t, c)).collect();
            for (_, h, _) in threads.drain(..) {
                // SAFETY: a handle we opened.
                unsafe { CloseHandle(h) };
            }
            // SAFETY: a thread snapshot walked with a sized entry; title
            // threads opened to stop and read them.
            unsafe {
                let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
                if snap != INVALID_HANDLE_VALUE {
                    let mut e: ThreadEntry32 = std::mem::zeroed();
                    e.size = size_of::<ThreadEntry32>() as u32;
                    let mut ok = Thread32First(snap, &mut e) != 0;
                    while ok {
                        if e.owner_process_id == pid && e.thread_id != me {
                            let h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, 0, e.thread_id);
                            if !h.is_null() {
                                if named(h) {
                                    CloseHandle(h);
                                } else {
                                    names.insert(e.thread_id, describe(h));
                                    cpu_start.entry(e.thread_id).or_insert_with(|| cpu_ms(h));
                                    threads.push((e.thread_id, h, kept.remove(&e.thread_id).unwrap_or(0)));
                                }
                            }
                        }
                        ok = Thread32Next(snap, &mut e) != 0;
                    }
                    CloseHandle(snap);
                }
            }
        }
        let mut ctx = Context::zeroed();
        let image = Image { base, len: 0x0b00_0000, ours };
        for (tid, h, last_cycles) in threads.iter_mut() {
            if let Some((place, ran, chain)) = sample_thread(*h, last_cycles, &image, funcs.as_ref(), &modules, &mut stack, &mut ctx) {
                if let Some(c) = chain {
                    *chains.entry(c).or_default() += ran;
                }
                *all.entry(place).or_default() += ran;
                *by_thread.entry(*tid).or_default().entry(place).or_default() += ran;
                total += ran;
            }
        }
        if window.elapsed() >= std::time::Duration::from_secs(10) {
            window = std::time::Instant::now();
            let cpu: HashMap<u32, u64> = threads.iter().map(|&(t, h, _)| (t, cpu_ms(h).saturating_sub(cpu_start.get(&t).copied().unwrap_or(0)))).collect();
            report(funcs.as_ref(), &modules, total, &all, &by_thread, &names, &cpu);
            report_chains(funcs.as_ref(), &modules, &image, total, &chains);
            chains.clear();
            for &(t, h, _) in threads.iter() {
                cpu_start.insert(t, cpu_ms(h));
            }
            (total, all, by_thread) = (0, HashMap::new(), HashMap::new());
        }
    }
}

/// Stack words read per sample (4 KiB: an exception dispatch puts a
/// CONTEXT and an EXCEPTION_RECORD, ~1.4 KiB, between a leaf and its
/// caller; a test run found no title frame within 64 words).
const STACK: usize = 512;

/// For a sample in ntdll with no title frame, the first six return
/// addresses on its stack (into any known module: title, ours, ntdll,
/// kernelbase/kernel32), call-preceded.
type Chain = [usize; 6];

/// The title image's range, and our own module's base.
struct Image {
    base: usize,
    len: usize,
    ours: usize,
}

/// One tick of one thread: `None` if it did not run since the last tick
/// (cycle delta under 20,000) or could not be read; else where it is and
/// the cycles it ran. The thread is stopped only for its context and 64
/// stack words.
fn sample_thread(h: HANDLE, last_cycles: &mut u64, image: &Image, funcs: Option<&Funcs>, modules: &[Module], stack: &mut [usize; STACK], ctx: &mut Context) -> Option<(Place, u64, Option<Chain>)> {
    let mut cycles = 0u64;
    // SAFETY: an open thread, an out-parameter.
    unsafe { QueryThreadCycleTime(h, &mut cycles) };
    let ran = cycles.saturating_sub(*last_cycles);
    let first = *last_cycles == 0;
    *last_cycles = cycles;
    if first || ran < 20_000 {
        return None;
    }
    let (mut rip, mut words) = (0usize, 0usize);
    // SAFETY: stopped for a register and stack read only; resumed before
    // anything that allocates or locks.
    unsafe {
        if SuspendThread(h) == u32::MAX {
            return None;
        }
        ctx.0[Context::FLAGS..Context::FLAGS + 4].copy_from_slice(&CONTEXT_CONTROL_INTEGER.to_le_bytes());
        if GetThreadContext(h, ctx) != 0 {
            rip = ctx.rip();
            // A shallow stack ends before 4 KiB past RSP and the whole read
            // fails: halve until it fits.
            let mut len = size_of_val(stack);
            while len >= 64 {
                let mut n = 0;
                if ReadProcessMemory(GetCurrentProcess(), ctx.rsp() as *const _, stack.as_mut_ptr().cast(), len, &mut n) != 0 {
                    words = n / 8;
                    break;
                }
                len /= 2;
            }
        }
        ResumeThread(h);
    }
    if rip == 0 {
        return None;
    }
    let base = image.base;
    let in_title = |a: usize| base != 0 && a >= base && a < base + image.len;
    let title_place = |a: usize, leaf: Leaf| -> Place {
        let rva = (a - base) as u32;
        funcs.and_then(|f| f.of(rva)).map_or(Place::Image(rva & !0xfff, leaf), |i| Place::Func(i, leaf))
    };
    let place = if in_title(rip) {
        title_place(rip, Leaf::Title)
    } else {
        let leaf = modules
            .iter()
            .position(|m| rip >= m.base && rip < m.end)
            .map(|k| {
                let rva = (rip - modules[k].base) as u32;
                let i = modules[k].exports.partition_point(|e| e.0 <= rva);
                Leaf::In(k, i.checked_sub(1).unwrap_or(usize::MAX))
            })
            .unwrap_or(Leaf::In(if rip >= image.ours && rip < image.ours + 0x0400_0000 { usize::MAX - 1 } else { usize::MAX }, usize::MAX));
        // The first return address into the title on the stack (a call
        // precedes it).
        let caller = stack[..words].iter().copied().find(|&w| in_title(w) && w > base + 6 && is_after_call(w));
        caller.map_or(Place::Module(leaf), |c| title_place(c, leaf))
    };
    // Where a title-less ntdll sample came from.
    let chain = (matches!(place, Place::Module(Leaf::In(0, _)))).then(|| {
        let known = |w: usize| in_title(w) || (w >= image.ours && w < image.ours + 0x0400_0000) || modules.iter().any(|m| w >= m.base && w < m.end);
        let mut c: Chain = [0; 6];
        let mut k = 0;
        for &w in &stack[..words] {
            if k == c.len() {
                break;
            }
            if w > 0x10000 && known(w) && is_after_call(w) {
                c[k] = w;
                k += 1;
            }
        }
        c
    });
    Some((place, ran, chain))
}

/// Whether the bytes before `ret` (in our own image) are a call.
fn is_after_call(ret: usize) -> bool {
    let mut b = [0u8; 7];
    let mut n = 0;
    // SAFETY: a read of our own memory through the kernel.
    if unsafe { ReadProcessMemory(GetCurrentProcess(), (ret - 7) as *const _, b.as_mut_ptr().cast(), 7, &mut n) } == 0 {
        return false;
    }
    // call rel32 (e8 xx xx xx xx), call [rip+x] (ff 15 x4), call reg/mem (ff /2 short forms).
    b[2] == 0xe8 || (b[1] == 0xff && b[2] == 0x15) || (b[5] == 0xff && b[6] & 0x38 == 0x10) || (b[4] == 0xff && b[5] & 0x38 == 0x10) || (b[3] == 0xff && b[4] & 0x38 == 0x10)
}

/// Whether a thread is ours: it has a description not starting "title "
/// (the title's threads are named "title <entry>").
fn named(h: HANDLE) -> bool {
    let mut p: *mut u16 = std::ptr::null_mut();
    // SAFETY: a query; the string is freed here.
    unsafe {
        if GetThreadDescription(h, &mut p) < 0 || p.is_null() {
            return false;
        }
        let mut n = 0;
        while *p.add(n) != 0 && n < 6 {
            n += 1;
        }
        let ours = n > 0 && String::from_utf16_lossy(std::slice::from_raw_parts(p, n)) != "title ";
        LocalFree(p.cast());
        ours
    }
}

/// A thread's description, or "unnamed".
fn describe(h: HANDLE) -> String {
    let mut p: *mut u16 = std::ptr::null_mut();
    // SAFETY: a query; the string is freed here.
    unsafe {
        if GetThreadDescription(h, &mut p) < 0 || p.is_null() {
            return "unnamed".into();
        }
        let mut n = 0;
        while *p.add(n) != 0 && n < 128 {
            n += 1;
        }
        let s = String::from_utf16_lossy(std::slice::from_raw_parts(p, n));
        LocalFree(p.cast());
        if s.is_empty() { "unnamed".into() } else { s }
    }
}

fn leaf_text(modules: &[Module], leaf: Leaf) -> String {
    match leaf {
        Leaf::Title => String::new(),
        Leaf::In(k, i) if k < modules.len() => {
            let m = &modules[k];
            match m.exports.get(i) {
                Some((_, name)) => format!(" via {}!{name}", m.name.trim_end_matches(".dll")),
                None => format!(" via {}", m.name.trim_end_matches(".dll")),
            }
        }
        Leaf::In(k, _) if k == usize::MAX - 1 => " via xnw".into(),
        Leaf::In(..) => " via another module".into(),
    }
}

fn label(funcs: Option<&Funcs>, modules: &[Module], p: Place) -> String {
    match p {
        Place::Func(i, leaf) => funcs.map_or_else(String::new, |f| format!("{:#x} {}{}", f.starts[i], f.names[i], leaf_text(modules, leaf))),
        Place::Image(page, leaf) => format!("image page {page:#x}{}", leaf_text(modules, leaf)),
        Place::Module(leaf) => format!("(no title frame){}", leaf_text(modules, leaf)),
    }
}

/// An address as module+offset: title function, our symbol (dbghelp, the
/// exe's pdb), the nearest ntdll/kernelbase export.
fn address_text(funcs: Option<&Funcs>, modules: &[Module], image: &Image, a: usize) -> String {
    if image.base != 0 && a >= image.base && a < image.base + image.len {
        let rva = (a - image.base) as u32;
        return funcs.and_then(|f| f.of(rva).map(|i| format!("title {}+{:#x}", f.names[i], rva - f.starts[i]))).unwrap_or_else(|| format!("title+{rva:#x}"));
    }
    if let Some(m) = modules.iter().find(|m| a >= m.base && a < m.end) {
        let rva = (a - m.base) as u32;
        let i = m.exports.partition_point(|e| e.0 <= rva);
        return match i.checked_sub(1).and_then(|i| m.exports.get(i)) {
            Some((s, n)) => format!("{}!{n}+{:#x}", m.name.trim_end_matches(".dll"), rva - s),
            None => format!("{}+{rva:#x}", m.name.trim_end_matches(".dll")),
        };
    }
    if a >= image.ours && a < image.ours + 0x0400_0000 {
        return format!("xnw!{}", our_symbol(a).unwrap_or_else(|| format!("+{:#x}", a - image.ours)));
    }
    format!("{a:#x}")
}

/// Our function at `a` through dbghelp (set up once, on this thread).
fn our_symbol(a: usize) -> Option<String> {
    static READY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // SAFETY: dbghelp for our own process, from the sampler thread only.
    let ready = *READY.get_or_init(|| unsafe {
        SymSetOptions(0x2 | 0x4);
        SymInitializeW(GetCurrentProcess(), std::ptr::null(), 1) != 0
    });
    if !ready {
        return None;
    }
    // SAFETY: zeroed is a valid SYMBOL_INFOW; dbghelp fills it.
    let mut info: SymbolInfoW = unsafe { std::mem::zeroed() };
    info.size_of_struct = SYMBOL_INFOW_SIZE;
    info.max_name_len = 512;
    let mut d = 0u64;
    // SAFETY: as above.
    if unsafe { SymFromAddrW(GetCurrentProcess(), a as u64, &mut d, &mut info) } == 0 {
        return None;
    }
    let n = (info.name_len as usize).min(512);
    Some(format!("{}+{d:#x}", String::from_utf16_lossy(&info.name[..n])))
}

/// The ten heaviest chains of title-less ntdll samples.
fn report_chains(funcs: Option<&Funcs>, modules: &[Module], image: &Image, total: u64, chains: &HashMap<Chain, u64>) {
    if chains.is_empty() || total == 0 {
        return;
    }
    let mut v: Vec<(&Chain, &u64)> = chains.iter().collect();
    v.sort_unstable_by(|a, b| b.1.cmp(a.1));
    for (c, w) in v.into_iter().take(10) {
        let frames: Vec<String> = c.iter().filter(|&&a| a != 0).map(|&a| address_text(funcs, modules, image, a)).collect();
        xnw_core::info!("cpu sample: ntdll without a title frame, {:.1} %: {}", *w as f64 * 100.0 / total as f64, frames.join(" < "));
    }
}

fn report(funcs: Option<&Funcs>, modules: &[Module], total: u64, all: &HashMap<Place, u64>, by_thread: &HashMap<u32, HashMap<Place, u64>>, names: &HashMap<u32, String>, cpu: &HashMap<u32, u64>) {
    if total == 0 {
        return;
    }
    let top = |m: &HashMap<Place, u64>, n: usize| -> Vec<(Place, u64)> {
        let mut v: Vec<(Place, u64)> = m.iter().map(|(&p, &c)| (p, c)).collect();
        v.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        v.truncate(n);
        v
    };
    let cpu_total: u64 = cpu.values().sum();
    let lines: Vec<String> = top(all, 8).into_iter().map(|(p, c)| format!("{} {:.1} %", label(funcs, modules, p), c as f64 * 100.0 / total as f64)).collect();
    xnw_core::info!("cpu sample: last 10 s, title threads ran {cpu_total} ms of CPU; by running time: {}", lines.join("; "));
    let mut threads: Vec<(u32, u64)> = cpu.iter().map(|(&t, &ms)| (t, ms)).collect();
    threads.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    for (tid, ms) in threads.into_iter().take(8).filter(|&(_, ms)| ms > 0) {
        let parts: Vec<String> = by_thread.get(&tid).map(|m| top(m, 3).into_iter().map(|(p, c)| format!("{} {:.0} %", label(funcs, modules, p), c as f64 * 100.0 / m.values().sum::<u64>().max(1) as f64)).collect()).unwrap_or_default();
        xnw_core::info!("cpu sample: thread {tid} ({}): {ms} ms CPU: {}", names.get(&tid).map_or("?", String::as_str), parts.join("; "));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    /// A function's code: through an incremental-link `jmp rel32` thunk if
    /// its pointer is one (the remote build's are).
    fn code_of(f: usize) -> usize {
        // SAFETY: reads the first bytes of a function of ours.
        unsafe {
            if *(f as *const u8) == 0xe9 {
                let rel = (f as *const u8).add(1).cast::<i32>().read_unaligned();
                return (f as isize + 5 + rel as isize) as usize;
            }
        }
        f
    }

    #[inline(never)]
    fn query_here(stop: &AtomicBool) {
        let mut info: MemoryBasicInformation = unsafe { std::mem::zeroed() };
        while !stop.load(Ordering::Relaxed) {
            // SAFETY: a query about our own address space into a local.
            unsafe { VirtualQueryEx(GetCurrentProcess(), query_here as *const _, &mut info, size_of::<MemoryBasicInformation>()) };
        }
    }

    /// A thread looping on VirtualQuery outside the title shows as
    /// ntdll with its caller chain, the calling function in it.
    #[test]
    fn a_title_less_ntdll_sample_carries_its_callers() {
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel();
        let s = stop.clone();
        let j = std::thread::spawn(move || {
            // SAFETY: a plain query.
            tx.send(unsafe { GetCurrentThreadId() }).unwrap();
            query_here(&s);
        });
        let id = rx.recv().unwrap();
        // SAFETY: our own module.
        let ours = unsafe { GetModuleHandleW(std::ptr::null()) as usize };
        let image = Image { base: 0, len: 0, ours };
        let modules: Vec<Module> = ["ntdll.dll", "kernelbase.dll", "kernel32.dll"].into_iter().filter_map(module_info).collect();
        let (mut stack, mut ctx) = ([0usize; STACK], Context::zeroed());
        // SAFETY: rights to stop and read a thread of ours.
        let h = unsafe { OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, 0, id) };
        let mut last = 0u64;
        let f = code_of(query_here as usize);
        let (mut chained, mut with_caller) = (0, 0);
        for _ in 0..300 {
            std::thread::sleep(std::time::Duration::from_millis(1));
            if let Some((_, _, Some(c))) = sample_thread(h, &mut last, &image, None, &modules, &mut stack, &mut ctx) {
                chained += 1;
                with_caller += usize::from(c.iter().any(|&a| a > f && a < f + 0x200));
            }
        }
        stop.store(true, Ordering::Relaxed);
        j.join().unwrap();
        // SAFETY: our handle.
        unsafe { CloseHandle(h) };
        assert!(chained > 10, "ntdll samples with chains: {chained}");
        assert!(with_caller * 10 >= chained * 8, "{with_caller} of {chained} chains name the calling function");
        let text = address_text(None, &modules, &image, f + 1);
        assert!(text.starts_with("xnw!"), "{text}");
    }

    #[inline(never)]
    fn spin_here(stop: &AtomicBool) {
        while !stop.load(Ordering::Relaxed) {
            std::hint::spin_loop();
        }
    }

    #[inline(never)]
    fn sleep0_here(stop: &AtomicBool) {
        while !stop.load(Ordering::Relaxed) {
            // SAFETY: a plain call.
            unsafe { Sleep(0) };
        }
    }

    /// A thread spinning in a known function is attributed to it, a
    /// sleeping one gets no weight, and one inside a system call made from
    /// a known function is attributed to that function "via" the module.
    #[test]
    fn running_threads_are_attributed_to_their_functions() {
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel();
        let spawn = |which: u8| {
            let (stop, tx) = (stop.clone(), tx.clone());
            std::thread::spawn(move || {
                // SAFETY: a plain query.
                tx.send((which, unsafe { GetCurrentThreadId() })).unwrap();
                match which {
                    0 => spin_here(&stop),
                    1 => sleep0_here(&stop),
                    // Asleep for longer than the sampling lasts.
                    _ => std::thread::sleep(std::time::Duration::from_millis(1500)),
                }
            })
        };
        let joins: Vec<_> = (0..3).map(spawn).collect();
        let mut ids = [0u32; 3];
        for _ in 0..3 {
            let (w, id) = rx.recv().unwrap();
            ids[w as usize] = id;
        }
        // The fake title image: both functions, each given a 0x100 range.
        let (a, b) = (code_of(spin_here as usize), code_of(sleep0_here as usize));
        let base = a.min(b) - 0x1000;
        let funcs = Funcs { starts: vec![(a - base) as u32, (b - base) as u32], ends: vec![(a - base) as u32 + 0x100, (b - base) as u32 + 0x100], names: vec!["spin".into(), "sleep0".into()] };
        let mut order: Vec<usize> = vec![0, 1];
        order.sort_by_key(|&i| funcs.starts[i]);
        let funcs = Funcs { starts: order.iter().map(|&i| funcs.starts[i]).collect(), ends: order.iter().map(|&i| funcs.ends[i]).collect(), names: order.iter().map(|&i| funcs.names[i].clone()).collect() };
        let index = |name: &str| funcs.names.iter().position(|n| n == name).unwrap();
        let image = Image { base, len: a.max(b) - base + 0x1000, ours: 0 };
        let modules: Vec<Module> = ["ntdll.dll", "kernelbase.dll", "kernel32.dll"].into_iter().filter_map(module_info).collect();
        let (mut stack, mut ctx) = ([0usize; STACK], Context::zeroed());
        let mut got: [Vec<Place>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let handles: Vec<HANDLE> = ids.iter().map(|&id| unsafe { OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, 0, id) }).collect();
        let mut last = [0u64; 3];
        for _ in 0..300 {
            std::thread::sleep(std::time::Duration::from_millis(1));
            for k in 0..3 {
                if let Some((p, _, _)) = sample_thread(handles[k], &mut last[k], &image, Some(&funcs), &modules, &mut stack, &mut ctx) {
                    got[k].push(p);
                }
            }
        }
        stop.store(true, Ordering::Relaxed);
        for j in joins {
            j.join().unwrap();
        }
        for h in handles {
            // SAFETY: handles we opened.
            unsafe { CloseHandle(h) };
        }
        let spin = index("spin");
        assert!(got[0].len() > 50, "the spinner ran: {} samples", got[0].len());
        let in_spin = got[0].iter().filter(|p| matches!(p, Place::Func(i, Leaf::Title) if *i == spin)).count();
        assert!(in_spin * 10 >= got[0].len() * 9, "{in_spin} of {} spinner samples in its function", got[0].len());
        assert!(got[2].len() <= 1, "the sleeper is not counted: {} samples", got[2].len());
        let sleep0 = index("sleep0");
        let via = got[1].iter().filter(|p| matches!(p, Place::Func(i, Leaf::In(..)) if *i == sleep0)).count();
        let elsewhere = got[1].iter().filter(|p| !matches!(p, Place::Func(i, _) if *i == sleep0)).count();
        assert!(via > 0, "a Sleep(0) loop is seen via a system module: {} samples", got[1].len());
        assert!(elsewhere * 10 <= got[1].len(), "{elsewhere} of {} not in its function", got[1].len());
    }

    #[test]
    fn the_switch_parses() {
        assert_eq!(parse("200"), Some((200, 0, u64::MAX)));
        assert_eq!(parse(",from=60,to=212"), Some((200, 60, 212)));
        assert_eq!(parse("100,from=5"), Some((100, 5, u64::MAX)));
        assert_eq!(parse("0"), None);
        assert_eq!(parse("50,when=3"), None);
    }

    #[test]
    fn a_function_is_found_by_its_rva() {
        let f = Funcs { starts: vec![0x1000, 0x2000], ends: vec![0x10ec, 0x2100], names: vec!["a".into(), "b".into()] };
        assert_eq!(f.of(0x1000), Some(0));
        assert_eq!(f.of(0x10eb), Some(0));
        assert_eq!(f.of(0x10ec), None, "between functions");
        assert_eq!(f.of(0x20ff), Some(1));
        assert_eq!(f.of(0x0fff), None);
    }
}
