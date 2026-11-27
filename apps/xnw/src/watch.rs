//! A generic runtime probe for checklists of title addresses. Two env switches,
//! both off (no thread, no hook) when unset; grammar below.
//!
//! `XNW_WATCH=<file>|<inline list>`: title globals sampled every 250 ms, a
//! line per change (or per interval). `XNW_CALLS=<file>|<inline list>`: an
//! entry detour per function that counts calls and logs registers and the
//! return address. Numbers are hexadecimal (`0x` optional) except
//! `args=`, `every=` of a call and the `every=1s|500ms` interval.
//!
//! With `:ret` a sampled call also logs rax at its return: the stub swaps the
//! return address for a per-hook exit stub (original kept on a per-thread
//! shadow stack keyed by the stack slot, so recursion and unwinding that
//! skips frames are handled), which saves rax, rdx and xmm0, logs, and
//! returns to the caller.
//!
//! The call detour is a generated stub per function: it saves flags, the
//! volatile registers and xmm0-5, calls `call_hook(frame, id)`, restores all
//! and jumps to a trampoline holding the entry's moved instructions. The
//! entry's whole instructions (at least 14 bytes) are measured by a small
//! length decoder that refuses rip-relative operands and relative branches;
//! such a function is logged as not hooked. A branch from elsewhere into the
//! middle of the first 14 bytes is not detected (I).

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use xnw_core::win::{FlushInstructionCache, GetCurrentProcess, MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, ReadProcessMemory, VirtualAlloc, VirtualProtect};

/// The image base the listed addresses are relative to.
const IMAGE_BASE: usize = 0x1_4000_0000;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Ty {
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Ptr,
}

impl Ty {
    fn parse(s: &str) -> Option<Ty> {
        Some(match s {
            "u8" => Ty::U8,
            "u16" => Ty::U16,
            "u32" => Ty::U32,
            "u64" => Ty::U64,
            "f32" => Ty::F32,
            "f64" => Ty::F64,
            "ptr" => Ty::Ptr,
            _ => return None,
        })
    }
    fn size(self) -> usize {
        match self {
            Ty::U8 => 1,
            Ty::U16 => 2,
            Ty::U32 | Ty::F32 => 4,
            Ty::U64 | Ty::F64 | Ty::Ptr => 8,
        }
    }
    fn show(self, raw: u64) -> String {
        match self {
            Ty::F32 => format!("{}", f32::from_bits(raw as u32)),
            Ty::F64 => format!("{}", f64::from_bits(raw)),
            Ty::Ptr | Ty::U64 => format!("{raw:#x}"),
            _ => format!("{raw}"),
        }
    }
    fn delta(self, old: u64, new: u64) -> String {
        match self {
            Ty::F32 => format!("{:+}", f32::from_bits(new as u32) - f32::from_bits(old as u32)),
            Ty::F64 => format!("{:+}", f64::from_bits(new) - f64::from_bits(old)),
            _ => format!("{:+}", new.wrapping_sub(old) as i64),
        }
    }
}

#[derive(Debug, PartialEq)]
struct WatchItem {
    /// Image address of the first step.
    addr: usize,
    /// Offsets of the pointer chain: each step reads a pointer and adds one.
    chain: Vec<usize>,
    ty: Ty,
    name: String,
    /// Log at this interval instead of on change.
    interval: Option<Duration>,
}

#[derive(Debug, PartialEq)]
struct CallItem {
    addr: usize,
    name: String,
    args: usize,
    every: u64,
    ret: bool,
}

fn num(s: &str) -> Result<usize, String> {
    let t = s.trim();
    let t = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")).unwrap_or(t);
    usize::from_str_radix(t, 16).map_err(|_| format!("bad number `{s}`"))
}

fn interval(s: &str) -> Result<Duration, String> {
    let bad = || format!("bad interval `{s}` (3s or 500ms)");
    if let Some(ms) = s.strip_suffix("ms") {
        return ms.parse().map(Duration::from_millis).map_err(|_| bad());
    }
    let sec = s.strip_suffix('s').ok_or_else(bad)?;
    sec.parse::<f64>().ok().filter(|v| *v > 0.0).map(Duration::from_secs_f64).ok_or_else(bad)
}

/// `addr[->+off...]:type[:name][:every=1s]`
fn parse_watch(item: &str) -> Result<WatchItem, String> {
    let mut parts = item.split(':').map(str::trim);
    let path = parts.next().unwrap_or("");
    let ty = parts.next().and_then(Ty::parse).ok_or_else(|| format!("`{item}`: missing or unknown type (u8 u16 u32 u64 f32 f64 ptr)"))?;
    let mut steps = path.split("->").map(str::trim);
    let addr = num(steps.next().unwrap_or(""))?;
    let mut chain = Vec::new();
    for s in steps {
        chain.push(num(s.strip_prefix('+').ok_or_else(|| format!("`{item}`: chain step `{s}` must be +offset"))?)?);
    }
    let mut name = String::new();
    let mut every = None;
    for p in parts {
        if let Some(v) = p.strip_prefix("every=") {
            every = Some(interval(v)?);
        } else if name.is_empty() {
            name = p.to_string();
        } else {
            return Err(format!("`{item}`: unexpected field `{p}`"));
        }
    }
    if name.is_empty() {
        name = path.to_string();
    }
    Ok(WatchItem { addr, chain, ty, name, interval: every })
}

/// `addr[:name][:args=N][:every=K][:ret]`
fn parse_call(item: &str) -> Result<CallItem, String> {
    let mut parts = item.split(':').map(str::trim);
    let addr = num(parts.next().unwrap_or(""))?;
    let (mut name, mut args, mut every, mut ret) = (String::new(), 2usize, 1000u64, false);
    for p in parts {
        if let Some(v) = p.strip_prefix("args=") {
            args = v.parse().ok().filter(|n| *n <= 4).ok_or_else(|| format!("`{item}`: args= is 0..4"))?;
        } else if let Some(v) = p.strip_prefix("every=") {
            every = v.parse().ok().filter(|n| *n > 0).ok_or_else(|| format!("`{item}`: bad every="))?;
        } else if p == "ret" {
            ret = true;
        } else if name.is_empty() {
            name = p.to_string();
        } else {
            return Err(format!("`{item}`: unexpected field `{p}`"));
        }
    }
    if name.is_empty() {
        name = format!("{addr:#x}");
    }
    Ok(CallItem { addr, name, args, every, ret })
}

/// The items of a switch value: a file (one item per line, `#` comments) if
/// it names one, else an inline comma list. Bad items are logged and skipped.
fn items<T>(value: &str, parse: fn(&str) -> Result<T, String>) -> Vec<T> {
    let text = std::fs::read_to_string(value).unwrap_or_else(|_| value.to_string());
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("");
        for item in line.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            match parse(item) {
                Ok(i) => out.push(i),
                Err(e) => xnw_core::warn!("watch: skipped item: {e}"),
            }
        }
    }
    out
}

/// Reads `buf.len()` bytes of our own memory; false when unreadable.
/// In place first with `XNW_PLAINCOPY=on` (`xnw_core::plaincopy`: a page
/// that is not there goes on to the kernel's copy, so the result is the
/// kernel's); the kernel's copies are counted as `ReadProcessMemory probe`.
pub(crate) fn guarded_read(addr: usize, buf: &mut [u8]) -> bool {
    if xnw_core::plaincopy::copy(buf.as_mut_ptr(), addr as *const u8, buf.len()) {
        return true;
    }
    let mut done = 0;
    // SAFETY: the kernel reports an unmapped page instead of faulting.
    let ok = xnw_core::syscalls::timed(xnw_core::syscalls::Call::ReadProbe, || unsafe {
        ReadProcessMemory(GetCurrentProcess(), addr as *const _, buf.as_mut_ptr().cast(), buf.len(), &mut done)
    });
    ok != 0 && done == buf.len()
}

/// The value at the end of a chain from the real address `addr`; `None`
/// when any step is unreadable.
fn read_chain(addr: usize, chain: &[usize], ty: Ty) -> Option<u64> {
    let mut at = addr;
    for off in chain {
        let mut p = [0u8; 8];
        if !guarded_read(at, &mut p) {
            return None;
        }
        at = (u64::from_le_bytes(p) as usize).checked_add(*off)?;
    }
    let mut b = [0u8; 8];
    if !guarded_read(at, &mut b[..ty.size()]) {
        return None;
    }
    Some(u64::from_le_bytes(b))
}

fn real(base: usize, va: usize) -> usize {
    base.wrapping_add(va.wrapping_sub(IMAGE_BASE))
}

struct Sampled {
    item: WatchItem,
    last: Option<u64>,
    seen: bool,
    printed_at: Instant,
    printed: Option<u64>,
    window: Instant,
    logged: u32,
    suppressed: u32,
}

fn show(ty: Ty, v: Option<u64>) -> String {
    v.map_or("?".to_string(), |v| ty.show(v))
}

fn watch_thread(base: usize, list: Vec<WatchItem>) {
    let now = Instant::now();
    let mut s: Vec<Sampled> = list.into_iter().map(|item| Sampled { item, last: None, seen: false, printed_at: now, printed: None, window: now, logged: 0, suppressed: 0 }).collect();
    loop {
        let now = Instant::now();
        let frame = xnw_gpu::d3d12x::frames_presented();
        let t = xnw_core::uptime().as_secs_f64();
        for w in &mut s {
            let v = read_chain(real(base, w.item.addr), &w.item.chain, w.item.ty);
            // The state-keyed key script reads these.
            xnw_hle::note_watch(&w.item.name, v);
            let ty = w.item.ty;
            if !w.seen {
                w.seen = true;
                w.last = v;
                w.printed = v;
                w.printed_at = now;
                xnw_core::info!("watch: {} = {} at {t:.3} s frame {frame}", w.item.name, show(ty, v));
                continue;
            }
            if let Some(every) = w.item.interval {
                if now.duration_since(w.printed_at) >= every {
                    let delta = match (w.printed, v) {
                        (Some(o), Some(n)) => ty.delta(o, n),
                        _ => "?".into(),
                    };
                    xnw_core::info!("watch: {} {} (delta {delta}) at {t:.3} s frame {frame}", w.item.name, show(ty, v));
                    w.printed = v;
                    w.printed_at = now;
                }
                continue;
            }
            if now.duration_since(w.window) >= Duration::from_secs(1) {
                if w.suppressed > 0 {
                    xnw_core::info!("watch: {} {} changes suppressed, now {} at {t:.3} s frame {frame}", w.item.name, w.suppressed, show(ty, w.last));
                }
                (w.window, w.logged, w.suppressed) = (now, 0, 0);
            }
            if v != w.last {
                if w.logged < 4 {
                    w.logged += 1;
                    xnw_core::info!("watch: {} {} -> {} at {t:.3} s frame {frame}", w.item.name, show(ty, w.last), show(ty, v));
                } else {
                    w.suppressed += 1;
                }
                w.last = v;
            }
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

// ---- calls ----

struct CallHook {
    name: String,
    args: usize,
    every: u64,
    ret: bool,
    /// The exit stub, set once the entry is hooked.
    ret_stub: AtomicUsize,
    count: AtomicU64,
}

struct Record {
    id: usize,
    n: u64,
    regs: [u64; 4],
    ret: usize,
}

static HOOKS: OnceLock<Vec<CallHook>> = OnceLock::new();
static RECORDS: Mutex<Vec<Record>> = Mutex::new(Vec::new());
struct Shadow {
    slot: usize,
    ret: usize,
    id: usize,
    n: u64,
}

thread_local! {
    static SHADOW: RefCell<Vec<Shadow>> = const { RefCell::new(Vec::new()) };
}

/// Called from an exit stub: logs rax, writes the original return address
/// to the stub's return slot.
extern "system" fn ret_hook(rax: u64, _id: usize, slot: usize) {
    let found = SHADOW.with(|s| {
        let mut s = s.borrow_mut();
        // Entries above the slot are frames that never returned here.
        while let Some(e) = s.pop() {
            if e.slot == slot {
                return Some(e);
            }
            if e.slot > slot {
                s.push(e);
                break;
            }
        }
        None
    });
    let Some(e) = found else {
        xnw_core::warn!("calls: exit with no shadow entry (slot {slot:#x}); aborting");
        std::process::abort();
    };
    if let Ok(mut r) = RETS.lock() {
        r.push((e.id, e.n, rax));
    }
    // SAFETY: the exit stub's slot, which the `ret` after us pops.
    unsafe { (slot as *mut usize).write(e.ret) };
}

static RETS: Mutex<Vec<(usize, u64, u64)>> = Mutex::new(Vec::new());
static TITLE: AtomicU64 = AtomicU64::new(0);

/// Called from a stub with the frame pointer it set up: `[frame+8]` the
/// return address, rcx, rdx, r8, r9 at `frame-24`, `-32`, `-40`, `-48`.
extern "system" fn call_hook(frame: usize, id: usize) {
    let Some(h) = HOOKS.get().and_then(|h| h.get(id)) else { return };
    let n = h.count.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 8 || n % h.every == 0 {
        // SAFETY: the stub's own frame, readable.
        let (regs, ret) = unsafe { ([(frame - 24) as *const u64, (frame - 32) as *const u64, (frame - 40) as *const u64, (frame - 48) as *const u64].map(|p| p.read()), ((frame + 8) as *const usize).read()) };
        if let Ok(mut r) = RECORDS.lock() {
            r.push(Record { id, n, regs, ret });
        }
        let stub = h.ret_stub.load(Ordering::Relaxed);
        if h.ret && stub != 0 {
            let slot = frame + 8;
            SHADOW.with(|s| s.borrow_mut().push(Shadow { slot, ret, id, n }));
            // SAFETY: the return address slot of the hooked call.
            unsafe { (slot as *mut usize).write(stub) };
        }
    }
}

fn where_(ret: usize) -> String {
    let base = TITLE.load(Ordering::Relaxed) as usize;
    if ret >= base && ret - base < 0x1000_0000 { format!("title+{:#x}", ret - base) } else { format!("{ret:#x}") }
}

fn calls_thread() {
    let hooks = HOOKS.get().map(Vec::as_slice).unwrap_or(&[]);
    let mut last = vec![0u64; hooks.len()];
    loop {
        std::thread::sleep(Duration::from_secs(1));
        let recs = RECORDS.lock().map(|mut r| std::mem::take(&mut *r)).unwrap_or_default();
        for r in recs {
            let h = &hooks[r.id];
            let mut regs = String::new();
            for (i, n) in ["rcx", "rdx", "r8", "r9"].iter().enumerate().take(h.args) {
                regs += &format!(" {n}={:#x}", r.regs[i]);
            }
            xnw_core::info!("calls: {} #{}{regs} ret={}", h.name, r.n, where_(r.ret));
        }
        let rets = RETS.lock().map(|mut r| std::mem::take(&mut *r)).unwrap_or_default();
        for (id, n, rax) in rets {
            xnw_core::info!("calls: {} #{n} returned rax={rax:#x}", hooks[id].name);
        }
        for (i, h) in hooks.iter().enumerate() {
            let c = h.count.load(Ordering::Relaxed);
            if c != last[i] {
                xnw_core::info!("calls: {} {} ({c} total)", h.name, c - last[i]);
                last[i] = c;
            }
        }
    }
}

/// Length of one instruction at the start of `b`, or `None` when unknown,
/// position dependent (rip-relative, relative branch) or a return.
fn insn_len(b: &[u8]) -> Option<usize> {
    let mut i = 0;
    let mut opsize16 = false;
    let mut rexw = false;
    while let Some(&p) = b.get(i) {
        match p {
            0x66 => opsize16 = true,
            // `fs`/`gs` overrides (`mov rax, gs:[58h]` opens most prologues).
            0xF2 | 0xF3 | 0x64 | 0x65 => {}
            0x40..=0x4F => {
                rexw = p & 8 != 0;
                i += 1;
                break;
            }
            _ => break,
        }
        i += 1;
    }
    let op = *b.get(i)?;
    i += 1;
    let modrm = |i: usize| -> Option<usize> {
        let m = *b.get(i)?;
        let (md, rm) = (m >> 6, m & 7);
        let mut n = 1;
        if md != 3 && rm == 4 {
            let sib = *b.get(i + 1)?;
            n += 1;
            if md == 0 && sib & 7 == 5 {
                n += 4;
            }
        }
        match md {
            0 if rm == 5 => return None,
            1 => n += 1,
            2 => n += 4,
            _ => {}
        }
        Some(n)
    };
    let imm_z = if opsize16 { 2 } else { 4 };
    Some(i + match op {
        // `ret` and `int3` padding (a 12-byte function plus its padding:
        // the trampoline runs the ret; the int3 bytes are never reached).
        0x50..=0x5F | 0x90 | 0xC9 | 0xC3 | 0xCC => 0,
        0x6A | 0xA8 | 0x04 | 0x0C | 0x24 | 0x2C | 0x3C => 1,
        0x68 | 0xA9 | 0x05 | 0x0D | 0x25 | 0x2D | 0x3D => imm_z,
        0xB0..=0xB7 => 1,
        0xB8..=0xBF => if rexw { 8 } else { imm_z },
        0x00..=0x03 | 0x08..=0x0B | 0x10..=0x13 | 0x18..=0x1B | 0x20..=0x23 | 0x28..=0x2B | 0x30..=0x33 | 0x38..=0x3B | 0x63 | 0x84..=0x8B | 0x8D | 0xFF => {
            // `FF /2 /4` are indirect calls and jumps: position independent.
            modrm(i)?
        }
        0x80 | 0x83 | 0xC6 | 0x6B => modrm(i)? + 1,
        0x81 | 0xC7 | 0x69 => modrm(i)? + imm_z,
        0xF6 | 0xF7 => {
            let reg = (*b.get(i)? >> 3) & 7;
            modrm(i)? + if reg < 2 { if op == 0xF6 { 1 } else { imm_z } } else { 0 }
        }
        0x0F => {
            let op2 = *b.get(i)?;
            i += 1;
            match op2 {
                0x10 | 0x11 | 0x1E | 0x1F | 0x28 | 0x29 | 0x2A | 0x2C | 0x2E | 0x2F | 0x40..=0x4F | 0x51 | 0x54 | 0x57 | 0x58 | 0x59 | 0x5C | 0x5E | 0x6E | 0x6F | 0x7E | 0x7F | 0xAF | 0xB6 | 0xB7 | 0xBE | 0xBF | 0xD6 | 0xEF => return Some(i + modrm(i)?),
                0xA2 => return Some(i),
                _ => return None,
            }
        }
        _ => return None,
    })
}

/// Bytes of whole instructions at the start of `code` covering at least 14.
#[cfg(test)]
fn entry_len(code: &[u8]) -> Option<usize> {
    relocate(code, 0).map(|r| r.0)
}

/// The whole instructions at the start of `code` (which sits at `from`)
/// covering at least 14 bytes, and their copy for a trampoline: a
/// `call rel32` becomes `mov r11, target; call r11` (the loader turns
/// `mov r, gs:[58h]` into a call to a TLS thunk, so most prologues hold one;
/// r11 is volatile at a function's entry). An earlier hook's own entry,
/// `jmp [rip+0]` and its 8-byte target, moves whole (it is absolute and
/// ends the path), so a second hook on one address runs before the first.
/// A `jcc rel8` becomes the inverse jcc over `jmp [rip+0]` to its target,
/// when that target lies outside the moved bytes (handler Draws such as
/// 0x1435B0DA0 open with `test byte [rcx+..], 2; jnz +7` at entry + 6).
/// Other relative forms: `None`.
fn relocate(code: &[u8], from: usize) -> Option<(usize, Vec<u8>)> {
    let (mut n, mut out) = (0, Vec::new());
    let mut targets = Vec::new();
    while n < 14 {
        let c = code.get(n..)?;
        // `mov r32, [rip+d32]` (no REX; eax, ecx, edx, ebx, esi, edi) becomes
        // `mov r64, target; mov r32, [r64]` (the register is written either way;
        // W16P: the vehicle handler Draw 0x1435A85F0 reads a TLS index so at entry + 10).
        if c.first() == Some(&0x8B) && c.get(1).is_some_and(|m| m & 0xC7 == 0x05 && !matches!((m >> 3) & 7, 4 | 5)) {
            let reg = (c[1] >> 3) & 7;
            let disp = i32::from_le_bytes(c.get(2..6)?.try_into().ok()?);
            let target = (from + n + 6).wrapping_add(disp as isize as usize);
            out.extend([0x48, 0xB8 + reg]);
            out.extend((target as u64).to_le_bytes());
            out.extend([0x8B, reg << 3 | reg]);
            n += 6;
            continue;
        }
        if let Some(&op @ 0x70..=0x7F) = c.first() {
            let rel = *c.get(1)? as i8;
            let target = (from + n + 2).wrapping_add(rel as isize as usize);
            out.extend([op ^ 1, 14]);
            out.extend(jump(target));
            targets.push(target);
            n += 2;
            continue;
        }
        if c.starts_with(&[0xFF, 0x25, 0, 0, 0, 0]) {
            out.extend(c.get(..14)?);
            n += 14;
            continue;
        }
        if c.first() == Some(&0xE8) {
            let rel = i32::from_le_bytes(c.get(1..5)?.try_into().ok()?);
            let target = (from + n + 5).wrapping_add(rel as isize as usize);
            out.extend([0x49, 0xBB]);
            out.extend((target as u64).to_le_bytes());
            out.extend([0x41, 0xFF, 0xD3]);
            n += 5;
            continue;
        }
        let k = insn_len(c)?;
        out.extend(&c[..k]);
        n += k;
    }
    if targets.iter().any(|t| (from..from + n).contains(t)) {
        return None;
    }
    Some((n, out))
}

/// Bytes a hook at `code` moves to its trampoline (whole instructions
/// covering at least 14), or None when they cannot be moved.
#[cfg(test)]
pub(crate) fn relocated_len(code: &[u8]) -> Option<usize> {
    relocate(code, 0).map(|r| r.0)
}

/// The exit stub of hook `id`: entered by the hooked function's `ret`.
fn make_exit_stub(id: usize) -> Vec<u8> {
    let mut s: Vec<u8> = vec![0x6A, 0x00, 0x48, 0x83, 0xEC, 0x58]; // push 0 (return slot); sub rsp,58h
    s.extend([0x48, 0x89, 0x44, 0x24, 0x20, 0x48, 0x89, 0x54, 0x24, 0x28, 0x0F, 0x11, 0x44, 0x24, 0x30]); // save rax rdx xmm0
    s.extend([0x48, 0x89, 0xC1, 0x4C, 0x8D, 0x44, 0x24, 0x58]); // mov rcx,rax; lea r8,[rsp+58h]
    s.extend([0xBA]);
    s.extend((id as u32).to_le_bytes());
    s.extend([0x48, 0xB8]);
    s.extend((ret_hook as usize as u64).to_le_bytes());
    s.extend([0xFF, 0xD0]);
    s.extend([0x48, 0x8B, 0x44, 0x24, 0x20, 0x48, 0x8B, 0x54, 0x24, 0x28, 0x0F, 0x10, 0x44, 0x24, 0x30]); // restore
    s.extend([0x48, 0x83, 0xC4, 0x58, 0xC3]); // add rsp,58h; ret
    s
}

fn jump(to: usize) -> [u8; 14] {
    let mut j = [0xFF, 0x25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    j[6..].copy_from_slice(&(to as u64).to_le_bytes());
    j
}

/// The stub for hook `id` (see the module header): calls `hook(frame, id)`,
/// then jumps to `tramp`. With `early` a nonzero rax from the hook
/// returns it to the hooked function's caller instead: volatile registers
/// and flags restored, rsp as at the entry, the function never runs.
/// `site`: entered inside a function body (rsp a multiple of
/// 16, not 8 as at an entry): 8 more bytes keep the hook's call aligned.
fn make_stub_at(id: usize, tramp: usize, hook: usize, early: bool, site: bool) -> Vec<u8> {
    let room: u8 = if site { 0x88 } else { 0x80 };
    let mut s: Vec<u8> = vec![0x55, 0x48, 0x89, 0xE5, 0x9C, 0xFC]; // push rbp; mov rbp,rsp; pushfq; cld
    s.extend([0x50, 0x51, 0x52, 0x41, 0x50, 0x41, 0x51, 0x41, 0x52, 0x41, 0x53]); // rax rcx rdx r8 r9 r10 r11
    s.extend([0x48, 0x81, 0xEC, room, 0, 0, 0]); // sub rsp,128 (136 at a site)
    for x in 0..6u8 {
        s.extend([0x0F, 0x11, 0x44 | x << 3, 0x24, 32 + 16 * x]); // movups [rsp+..],xmmX
    }
    s.extend([0x48, 0x89, 0xE9, 0xBA]); // mov rcx,rbp; mov edx,id
    s.extend((id as u32).to_le_bytes());
    s.extend([0x48, 0xB8]);
    s.extend((hook as u64).to_le_bytes());
    s.extend([0xFF, 0xD0]); // call rax
    for x in 0..6u8 {
        s.extend([0x0F, 0x10, 0x44 | x << 3, 0x24, 32 + 16 * x]);
    }
    let pops = [0x41, 0x5B, 0x41, 0x5A, 0x41, 0x59, 0x41, 0x58, 0x5A, 0x59]; // r11 r10 r9 r8 rdx rcx
    if !early {
        s.extend([0x48, 0x81, 0xC4, room, 0, 0, 0]);
        s.extend(pops);
        s.extend([0x58, 0x9D, 0x5D]); // rax popfq rbp
        s.extend(jump(tramp));
        return s;
    }
    s.extend([0x48, 0x8D, 0xA4, 0x24, room, 0, 0, 0]); // lea rsp,[rsp+128] (flags kept)
    s.extend([0x48, 0x85, 0xC0, 0x75, 27]); // test rax,rax; jnz early
    s.extend(pops);
    s.extend([0x58, 0x9D, 0x5D]);
    s.extend(jump(tramp)); // 13 + 14 = 27 bytes
    s.extend(pops); // early:
    s.extend([0x48, 0x83, 0xC4, 0x08, 0x9D, 0x5D, 0xC3]); // add rsp,8 (saved rax); popfq; pop rbp; ret
    s
}

/// Hooks the function at `at` as hook `id`.
///
/// # Safety
/// `at` is code of ours or the title; nothing may be executing its first
/// bytes now.
unsafe fn hook_entry(at: *mut u8, id: usize) -> Result<usize, String> {
    // SAFETY: forwarded.
    unsafe { hook_entry_with(at, id, call_hook as usize) }
}

/// Hooks the function at `at`: on entry `hook(frame, id)` runs
/// (`extern "system" fn(usize, usize)`; `frame` as for [`call_hook`]), then
/// the function. Returns the exit stub's address (meaningful only for
/// [`call_hook`]).
///
/// # Safety
/// As [`hook_entry`]; `hook` preserves what the Microsoft x64 convention
/// asks a callee to preserve.
pub(crate) unsafe fn hook_entry_with(at: *mut u8, id: usize, hook: usize) -> Result<usize, String> {
    // SAFETY: forwarded.
    unsafe { hook_entry_impl(at, id, hook, false, &[]) }
}

/// As [`hook_entry_with`] at any instruction `at` (a call site), with
/// `prelude` run first: position-independent code that may change only r11
/// (volatile at a call site; the hook reads it at `frame - 64`), e.g. to
/// hand over a value held in a nonvolatile register.
///
/// # Safety
/// As [`hook_entry_with`]; no branch may target `at + 1 ..` the relocated
/// bytes, r11 must be dead at `at`, and `prelude` must be at most 48 bytes.
pub(crate) unsafe fn hook_site_with(at: *mut u8, id: usize, hook: usize, prelude: &[u8]) -> Result<usize, String> {
    if prelude.len() > 48 {
        return Err("prelude too long".into());
    }
    // SAFETY: forwarded.
    unsafe { hook_entry_impl(at, id, hook, false, prelude) }
}

/// As [`hook_entry_with`], but `hook` (`extern "system" fn(usize, usize) ->
/// u64`) may replace the call: a nonzero result returns to the caller with
/// that rax and the function does not run; zero runs it. `frame` as for
/// [`call_hook`]; the fifth argument is at `frame + 0x30`.
///
/// # Safety
/// As [`hook_entry_with`]; returning early is right only when the skipped
/// function's effects are not needed.
pub(crate) unsafe fn hook_entry_early(at: *mut u8, id: usize, hook: usize) -> Result<usize, String> {
    // SAFETY: forwarded.
    unsafe { hook_entry_impl(at, id, hook, true, &[]) }
}

unsafe fn hook_entry_impl(at: *mut u8, id: usize, hook: usize, early: bool, prelude: &[u8]) -> Result<usize, String> {
    let mut code = [0u8; 32];
    if !guarded_read(at as usize, &mut code) {
        return Err("entry unreadable".into());
    }
    let (n, moved) = relocate(&code, at as usize).ok_or("entry instructions not relocatable (unknown, rip-relative, branch or too short)")?;
    // SAFETY: a fresh executable allocation of ours.
    let mem = unsafe { VirtualAlloc(std::ptr::null(), 512, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) }.cast::<u8>();
    if mem.is_null() {
        return Err("no stub memory".into());
    }
    let tramp = mem as usize + 256;
    let mut stub = prelude.to_vec();
    stub.extend(make_stub_at(id, tramp, hook, early, !prelude.is_empty()));
    if stub.len() > 256 {
        return Err("stub too long".into());
    }
    let back = jump(at as usize + n);
    // 32 bytes of entry grow by at most 8 per call: inside 256..384.
    debug_assert!(moved.len() + 14 <= 128);
    // SAFETY: inside the 512 bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(stub.as_ptr(), mem, stub.len());
        std::ptr::copy_nonoverlapping(moved.as_ptr(), mem.add(256), moved.len());
        std::ptr::copy_nonoverlapping(back.as_ptr(), mem.add(256 + moved.len()), 14);
        let exit = make_exit_stub(id);
        std::ptr::copy_nonoverlapping(exit.as_ptr(), mem.add(384), exit.len());
        FlushInstructionCache(GetCurrentProcess(), mem.cast(), 512);
    }
    let mut entry = code[..n].to_vec();
    entry[..14].copy_from_slice(&jump(mem as usize));
    entry[14..].fill(0xCC);
    let mut old = 0;
    // SAFETY: the entry bytes only; protection restored.
    if unsafe { VirtualProtect(at.cast(), n, PAGE_EXECUTE_READWRITE, &mut old) } == 0 {
        return Err("VirtualProtect failed".into());
    }
    // SAFETY: writable now.
    unsafe {
        std::ptr::copy_nonoverlapping(entry.as_ptr(), at, n);
        VirtualProtect(at.cast(), n, old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), at.cast(), n);
    }
    Ok(mem as usize + 384)
}

/// Starts what `XNW_WATCH` and `XNW_CALLS` ask for; nothing when unset.
///
/// # Safety
/// `base` is the mapped title and none of its code runs yet.
pub unsafe fn start(base: usize) {
    TITLE.store(base as u64, Ordering::Relaxed);
    if let Some(v) = std::env::var("XNW_CALLS").ok().filter(|v| !v.is_empty()) {
        let list = items(&v, parse_call);
        let mut hooks = Vec::new();
        let mut want = Vec::new();
        for c in list {
            // SAFETY: forwarded; ids are the order in `hooks`.
            match unsafe { hook_entry(real(base, c.addr) as *mut u8, hooks.len()) } {
                Ok(exit) => {
                    xnw_core::info!("calls: {} hooked at title+{:#x}", c.name, c.addr - IMAGE_BASE);
                    hooks.push(CallHook { name: c.name, args: c.args, every: c.every, ret: c.ret, ret_stub: AtomicUsize::new(exit), count: AtomicU64::new(0) });
                }
                Err(e) => xnw_core::warn!("calls: {} at {:#x} not hooked: {e}", c.name, c.addr),
            }
            want.push(());
        }
        let _ = HOOKS.set(hooks);
        if !want.is_empty() {
            std::thread::spawn(calls_thread);
        }
    }
    if let Some(v) = std::env::var("XNW_WATCH").ok().filter(|v| !v.is_empty()) {
        let list = items(&v, parse_watch);
        if !list.is_empty() {
            xnw_core::info!("watch: {} items", list.len());
            std::thread::spawn(move || watch_thread(base, list));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static SITE_R11: AtomicU64 = AtomicU64::new(0);
    static SITE_FRAME: AtomicU64 = AtomicU64::new(0);

    extern "system" fn site_hook(frame: usize, _id: usize) {
        // SAFETY: the stub's frame.
        SITE_R11.store(unsafe { ((frame - 64) as *const u64).read() }, Ordering::Relaxed);
        SITE_FRAME.store(frame as u64, Ordering::Relaxed);
    }

    /// A hook inside a function body with a prelude: the hook
    /// sees r11 as the prelude left it (`[r15+28h] + r14`), its call is
    /// 16-byte aligned, and the body goes on unchanged.
    #[test]
    fn site_hook_with_prelude() {
        let list = Box::leak(Box::new([0u64; 8]));
        list[5] = 0x1000;
        let mut code: Vec<u8> = vec![0x41, 0x56, 0x41, 0x57, 0x48, 0x83, 0xEC, 0x28]; // push r14; push r15; sub rsp,28h
        code.extend([0x49, 0xBF]);
        code.extend((list.as_ptr() as u64).to_le_bytes()); // mov r15, list
        code.extend([0x49, 0xBE]);
        code.extend(0x40u64.to_le_bytes()); // mov r14, 0x40
        code.extend([0x49, 0xBB]);
        code.extend(0x5555u64.to_le_bytes()); // mov r11, junk
        let site = code.len();
        code.extend([0x90; 16]);
        code.extend([0x4C, 0x89, 0xF0]); // mov rax, r14
        code.extend([0x48, 0x83, 0xC4, 0x28, 0x41, 0x5F, 0x41, 0x5E, 0xC3]); // add rsp,28h; pop r15; pop r14; ret
        // SAFETY: a fresh executable allocation of ours.
        let mem = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) }.cast::<u8>();
        // SAFETY: inside it.
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), mem, code.len()) };
        // SAFETY: the function above.
        let f: extern "system" fn() -> u64 = unsafe { std::mem::transmute(mem) };
        assert_eq!(f(), 0x40);
        // SAFETY: nothing runs it now; r11 is dead at the site.
        unsafe { hook_site_with(mem.wrapping_add(site), 0, site_hook as usize, &[0x4D, 0x8B, 0x5F, 0x28, 0x4D, 0x01, 0xF3]) }.unwrap();
        assert_eq!(f(), 0x40);
        assert_eq!(SITE_R11.load(Ordering::Relaxed), 0x1040);
        assert_eq!(SITE_FRAME.load(Ordering::Relaxed) % 16, 8, "the hook's call is aligned");
    }

    #[test]
    fn watch_grammar() {
        let w = parse_watch("0x1498F828C:u32:session").unwrap();
        assert_eq!((w.addr, w.ty, w.name.as_str(), w.chain.len()), (0x1498F828C, Ty::U32, "session", 0));
        let w = parse_watch("1471273A0->+10->+0x12:u16:every=2s").unwrap();
        assert_eq!((w.chain.clone(), w.name.as_str(), w.interval), (vec![0x10, 0x12], "1471273A0->+10->+0x12", Some(Duration::from_secs(2))));
        assert_eq!(parse_watch("14ABBEF40:u32:ms:every=500ms").unwrap().interval, Some(Duration::from_millis(500)));
        assert!(parse_watch("1000:u128").is_err());
        assert!(parse_watch("1000->8:u8").is_err());
        assert!(parse_watch("zz:u8").is_err());
    }

    #[test]
    fn call_grammar() {
        let c = parse_call("141AC7760:setter:args=1:every=50").unwrap();
        assert_eq!((c.addr, c.name.as_str(), c.args, c.every), (0x141AC7760, "setter", 1, 50));
        let c = parse_call("0x145862370").unwrap();
        assert_eq!((c.name.as_str(), c.args, c.every), ("0x145862370", 2, 1000));
        assert!(parse_call("1:a:args=9").is_err());
        assert!(parse_call("1:a:b").is_err());
    }

    #[test]
    fn file_items_skip_comments() {
        let dir = std::env::temp_dir().join("xnw_w8_items.txt");
        std::fs::write(&dir, "# c\n1000:u8:a # tail\n\n2000:u32, 3000:ptr:p\nbad\n").unwrap();
        let v = items(dir.to_str().unwrap(), parse_watch);
        assert_eq!(v.len(), 3);
        assert_eq!(items("1000:u8,2000:f32", parse_watch).len(), 2);
        let _ = std::fs::remove_file(dir);
    }

    #[test]
    fn chain_reads_local_memory() {
        let leaf: [u32; 4] = [0, 0, 77, 0];
        let mid: [u64; 2] = [0, leaf.as_ptr() as u64 + 4]; // +8 -> mid[1]; +4 more
        let top: u64 = mid.as_ptr() as u64;
        let at = &top as *const u64 as usize;
        // top -> mid(+8) holds a pointer; add 4 -> leaf[2]
        assert_eq!(read_chain(at, &[8, 4], Ty::U32), Some(77));
        assert_eq!(read_chain(at, &[], Ty::U64), Some(top));
        assert_eq!(read_chain(at, &[0x1000_0000_0000], Ty::U8), None);
        assert_eq!(read_chain(8, &[], Ty::U8), None);
        assert_eq!(Ty::F32.show(1.5f32.to_bits() as u64), "1.5");
        assert_eq!(Ty::U32.delta(5, 9), "+4");
    }

    #[test]
    fn decoder() {
        // push r15; push r14; push rsi; push rdi; push rbp; push rbx; sub rsp,28h; mov rsi,rdx
        let p = [0x41, 0x57, 0x41, 0x56, 0x56, 0x57, 0x55, 0x53, 0x48, 0x83, 0xEC, 0x28, 0x48, 0x89, 0xD6, 0xC3];
        assert_eq!(entry_len(&p), Some(15));
        let mut rip = p;
        rip[13] = 0x8B;
        rip[14] = 0x05; // mov rax,[rip+..]
        assert_eq!(entry_len(&rip), None);
        assert_eq!(entry_len(&[0xE9, 0, 0, 0, 0, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90]), None);
        // An entry an earlier hook patched (an XNW_CALLS detour on 0x1451D0AC0
        // or 0x14013EFB0 before the world probe): its jump moves whole.
        let mut hooked = jump(0x7FF0_1234_5678).to_vec();
        hooked.extend([0xCC; 6]);
        assert_eq!(relocate(&hooked, 0x1000), Some((14, hooked[..14].to_vec())));
        assert_eq!(entry_len(&[0xFF, 0x25, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), None);
        // The 12-byte render flip 0x1423CF3E0 and its int3 padding.
        assert_eq!(entry_len(&[0xB8, 0x01, 0x00, 0x00, 0x00, 0x2B, 0x41, 0x5C, 0x89, 0x41, 0x5C, 0xC3, 0xCC, 0xCC, 0xCC, 0xCC]), Some(14));
        // Title 0x140055E50 (geometry draw): pushes, sub rsp,48h, then
        // `mov rax, gs:[58h]` as the title has it and as the loader
        // rewrites it (`call thunk; nop4`).
        let gs = [0x41, 0x56, 0x56, 0x57, 0x53, 0x48, 0x83, 0xEC, 0x48, 0x65, 0x48, 0x8B, 0x04, 0x25, 0x58, 0, 0, 0, 0x31, 0xD2];
        assert_eq!(relocate(&gs, 0x1000).map(|r| r.0), Some(18));
        let mut call = gs;
        call[9..18].copy_from_slice(&[0xE8, 0x10, 0, 0, 0, 0x0F, 0x1F, 0x40, 0x00]);
        let (n, moved) = relocate(&call, 0x1000).unwrap();
        assert_eq!(n, 14);
        assert_eq!(&moved[..9], &call[..9]);
        // mov r11, 0x1000 + 9 + 5 + 0x10; call r11
        assert_eq!(&moved[9..11], &[0x49, 0xBB]);
        assert_eq!(u64::from_le_bytes(moved[11..19].try_into().unwrap()), 0x101E);
        assert_eq!(&moved[19..], &[0x41, 0xFF, 0xD3]);
        // Title 0x1435B0DA0 (a vehicle handler Draw): push rsi; push rdi; sub rsp,28h;
        // test byte [rcx+7A4h],2; jnz +7: the jnz becomes jz +14 over jmp [rip+0] to entry + 22.
        let veh = [0x56, 0x57, 0x48, 0x83, 0xEC, 0x28, 0xF6, 0x81, 0xA4, 0x07, 0x00, 0x00, 0x02, 0x75, 0x07, 0x48];
        let (n, moved) = relocate(&veh, 0x1000).unwrap();
        assert_eq!(n, 15);
        assert_eq!(&moved[..13], &veh[..13]);
        assert_eq!(&moved[13..15], &[0x74, 14]);
        assert_eq!(&moved[15..], &jump(0x1000 + 22));
        // A jcc back into the moved bytes cannot be relocated.
        let mut back = veh;
        back[14] = 0xF5; // jnz -11: entry + 4
        assert_eq!(relocate(&back, 0x1000), None);
        // W16P: title 0x1435A85F0 (a vehicle handler Draw): pushes, sub rsp,20h, mov eax,[rip+386052Ch].
        let veh = [0x41, 0x56, 0x56, 0x57, 0x55, 0x53, 0x48, 0x83, 0xEC, 0x20, 0x8B, 0x05, 0x2C, 0x05, 0x86, 0x03];
        let (n, moved) = relocate(&veh, 0x1000).unwrap();
        assert_eq!(n, 16);
        assert_eq!(&moved[..10], &veh[..10]);
        assert_eq!(&moved[10..12], &[0x48, 0xB8]);
        assert_eq!(u64::from_le_bytes(moved[12..20].try_into().unwrap()), 0x1000 + 16 + 0x0386_052C);
        assert_eq!(&moved[20..], &[0x8B, 0x00]);
        // mov ebp,[rip+..] (rm 5 as a base needs a displacement): not moved.
        assert_eq!(relocate(&[0x56, 0x57, 0x48, 0x83, 0xEC, 0x28, 0x8B, 0x2D, 0, 0, 0, 0, 0x90, 0x90, 0x90, 0x90], 0x1000), None);
    }

    /// A function with 17 bytes of prologue, hooked: the result is intact,
    /// callee-saved registers survive and the count is one.
    #[test]
    fn hooked_function_still_works() {
        let code = [
            0x40, 0x53, 0x56, 0x57, 0x55, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57, // pushes
            0x48, 0x83, 0xEC, 0x20, // sub rsp,20h
            0x48, 0x89, 0xC8, 0x48, 0x01, 0xD0, // mov rax,rcx; add rax,rdx
            0x48, 0x83, 0xC4, 0x20, // add rsp,20h
            0x41, 0x5F, 0x41, 0x5E, 0x41, 0x5D, 0x41, 0x5C, 0x5D, 0x5F, 0x5E, 0x5B, 0xC3,
        ];
        // SAFETY: a fresh executable allocation of ours.
        let mem = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) }.cast::<u8>();
        // SAFETY: inside the allocation.
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), mem, code.len()) };
        // SAFETY: the function written above.
        let f: extern "system" fn(usize, usize) -> usize = unsafe { std::mem::transmute(mem) };
        assert_eq!(f(2, 3), 5);
        let _ = HOOKS.set(vec![CallHook { name: "t".into(), args: 2, every: 1000, ret: true, ret_stub: AtomicUsize::new(0), count: AtomicU64::new(0) }]);
        // SAFETY: nothing runs the function now.
        let exit = unsafe { hook_entry(mem, 0) }.unwrap();
        HOOKS.get().unwrap()[0].ret_stub.store(exit, Ordering::Relaxed);
        assert_eq!(f(20, 3), 23);
        assert_eq!(f(1, 1), 2);
        assert_eq!(HOOKS.get().unwrap()[0].count.load(Ordering::Relaxed), 2);
        let r = RECORDS.lock().unwrap();
        assert_eq!((r[0].regs[0], r[0].regs[1], r[0].n), (20, 3, 1));
        let rets = RETS.lock().unwrap();
        assert_eq!((rets.len(), rets[0].2, rets[1].2), (2, 23, 2));
        drop((r, rets));
        // A second hook on the same entry runs first, then the first hook,
        // then the function.
        static SECOND: AtomicU64 = AtomicU64::new(0);
        extern "system" fn second(frame: usize, _id: usize) {
            // SAFETY: the stub's frame: rcx saved at frame - 24.
            SECOND.store(unsafe { ((frame - 24) as *const u64).read() }, Ordering::Relaxed);
        }
        // SAFETY: nothing runs the function now.
        unsafe { hook_entry_with(mem, 7, second as usize) }.unwrap();
        assert_eq!(f(40, 2), 42);
        assert_eq!((SECOND.load(Ordering::Relaxed), HOOKS.get().unwrap()[0].count.load(Ordering::Relaxed)), (40, 3));
    }

    /// A function whose moved bytes end in a `jnz rel8` (as the vehicle
    /// handler Draws do), hooked: both branches still return the right value
    /// and the hook sees rcx.
    #[test]
    fn hooked_function_with_a_jcc_still_works() {
        let code = [
            0x56, 0x57, 0x48, 0x83, 0xEC, 0x28, // push rsi; push rdi; sub rsp,28h
            0x48, 0x89, 0xC8, // mov rax,rcx
            0xF6, 0xC1, 0x01, // test cl,1
            0x75, 0x0A, // jnz +10 (odd: entry + 24)
            0x48, 0x01, 0xD0, 0x48, 0x83, 0xC4, 0x28, 0x5F, 0x5E, 0xC3, // add rax,rdx; add rsp,28h; pop rdi; pop rsi; ret
            0x48, 0x29, 0xD0, 0x48, 0x83, 0xC4, 0x28, 0x5F, 0x5E, 0xC3, // sub rax,rdx; ...
        ];
        // SAFETY: a fresh executable allocation of ours.
        let mem = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) }.cast::<u8>();
        // SAFETY: inside the allocation.
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), mem, code.len()) };
        // SAFETY: the function written above.
        let f: extern "system" fn(usize, usize) -> usize = unsafe { std::mem::transmute(mem) };
        assert_eq!((f(10, 3), f(11, 3)), (13, 8));
        static SEEN: AtomicU64 = AtomicU64::new(0);
        extern "system" fn seen(frame: usize, _id: usize) {
            // SAFETY: the stub's frame: rcx saved at frame - 24.
            SEEN.store(unsafe { ((frame - 24) as *const u64).read() }, Ordering::Relaxed);
        }
        // SAFETY: nothing runs the function now.
        unsafe { hook_entry_with(mem, 40, seen as usize) }.unwrap();
        assert_eq!(f(20, 3), 23);
        assert_eq!(SEEN.load(Ordering::Relaxed), 20);
        assert_eq!(f(21, 3), 18);
        assert_eq!(SEEN.load(Ordering::Relaxed), 21);
    }

    /// W16P: a function whose moved bytes hold `mov eax, [rip+d32]`, hooked:
    /// it still returns the word it reads.
    #[test]
    fn hooked_function_with_a_rip_load_still_works() {
        let mut code = vec![
            0x56, 0x57, 0x48, 0x83, 0xEC, 0x28, // push rsi; push rdi; sub rsp,28h
            0x8B, 0x05, 20, 0, 0, 0, // mov eax,[rip+20] (the word at 32)
            0x48, 0x83, 0xC4, 0x28, 0x5F, 0x5E, 0xC3, // add rsp,28h; pop rdi; pop rsi; ret
        ];
        code.resize(32, 0xCC);
        code.extend(0x1234_5678u32.to_le_bytes());
        // SAFETY: a fresh executable allocation of ours.
        let mem = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) }.cast::<u8>();
        // SAFETY: inside the allocation.
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), mem, code.len()) };
        // SAFETY: the function written above.
        let f: extern "system" fn(usize, usize) -> usize = unsafe { std::mem::transmute(mem) };
        assert_eq!(f(0, 0), 0x1234_5678);
        static SEEN: AtomicU64 = AtomicU64::new(0);
        extern "system" fn seen(frame: usize, _id: usize) {
            // SAFETY: the stub's frame: rcx saved at frame - 24.
            SEEN.store(unsafe { ((frame - 24) as *const u64).read() }, Ordering::Relaxed);
        }
        // SAFETY: nothing runs the function now.
        unsafe { hook_entry_with(mem, 41, seen as usize) }.unwrap();
        assert_eq!(f(77, 0), 0x1234_5678);
        assert_eq!(SEEN.load(Ordering::Relaxed), 77);
    }
}

#[cfg(test)]
mod ready_files {
    use super::*;

    #[test]
    #[ignore = "needs the watch lists (not published)"]
    fn every_ready_file_parses_completely() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/watch");
        for (f, calls) in [("load.txt", false), ("cine.txt", false), ("scene.txt", false), ("load-calls.txt", true), ("scene-calls.txt", true), ("cine-play.txt", false), ("cine-play-calls.txt", true), ("gates.txt", false), ("stream.txt", false), ("stream-calls.txt", true), ("dirty.txt", false), ("dirty-calls.txt", true)] {
            let text = std::fs::read_to_string(dir.join(f)).unwrap();
            let lines = text.lines().filter(|l| !l.split('#').next().unwrap().trim().is_empty()).count();
            let got = if calls { items(dir.join(f).to_str().unwrap(), parse_call).len() } else { items(dir.join(f).to_str().unwrap(), parse_watch).len() };
            assert_eq!(got, lines, "{f}");
        }
    }

    /// `guarded_read` across a page that is not there: false and nothing
    /// claimed, with `XNW_PLAINCOPY=on` (fault, then the kernel's copy) as
    /// without it; a read inside the page is whole either way.
    #[test]
    fn guarded_read_across_a_hole() {
        use xnw_core::win::{MEM_RELEASE, VirtualFree};
        const PAGE: usize = 4096;
        // SAFETY: memory of the test's own: two pages reserved, the first committed.
        let base = unsafe {
            let base = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, xnw_core::win::PAGE_NOACCESS);
            assert!(!base.is_null());
            assert!(!VirtualAlloc(base, PAGE, MEM_COMMIT, xnw_core::win::PAGE_READWRITE).is_null());
            base as usize
        };
        // SAFETY: the committed page.
        unsafe { std::slice::from_raw_parts_mut(base as *mut u8, PAGE) }.fill(0xAB);
        let mut b = vec![0u8; PAGE];
        assert!(!guarded_read(base + PAGE / 2, &mut b));
        assert!(!guarded_read(base + PAGE / 2, &mut b), "again, remembered as faulted");
        assert!(guarded_read(base, &mut b) && b.iter().all(|&x| x == 0xAB));
        assert!(guarded_read(base + 8, &mut b[..8]) && b[..8].iter().all(|&x| x == 0xAB), "under the in-place floor");
        // SAFETY: the test's reservation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }
}
