//! A data write watch on one title address by a debug register
//! (`XNW_DWATCH=<image address>[:len 1|2|4|8]`, off when unset). Every write
//! by any thread of the process logs the writer (`title+rva` of the
//! instruction after the write: data breakpoints trap after it), the
//! thread, the new value and the title's frame id (dword_14ABCE2A0).
//!
//! Why a debug register and not the page watch: the target 0x14765A1C3
//! (the TAA history-reset countdown) shares its page with the postfx
//! state the render thread rewrites every frame, so a guard-page watch would
//! fault thousands of times a frame. DR0 traps only on the watched bytes.
//!
//! Arming: a thread of ours enumerates the process's threads once a second
//! and sets DR0/DR7 on each new one (suspend, set context, resume; never its
//! own). The vectored handler takes `EXCEPTION_SINGLE_STEP` with DR6 bit 0,
//! records into a lock-free ring, clears DR6 and continues; anything else
//! goes on to the next handler. A logging thread prints the ring (the first
//! 64 writes, then every 100th, and a count per second).

use std::collections::HashSet;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use xnw_core::win;

const IMAGE_BASE: usize = 0x1_4000_0000;
const FRAME_ID: usize = 0x1_4ABC_E2A0;
const SINGLE_STEP: u32 = 0x8000_0004;
const CONTEXT_DEBUG_REGISTERS: u64 = 0x0010_0010;
const THREAD_ACCESS: u32 = 0x0002 | 0x0008 | 0x0010; // suspend/resume, get, set context

#[derive(Clone, Copy, Debug)]
struct Target {
    addr: usize,
    len: usize,
    base: usize,
}

static TARGET: OnceLock<Target> = OnceLock::new();
const RING_N: usize = 4096;
/// Each record: rip, thread, value, frame (4 x u64).
static RING: [[AtomicU64; 4]; RING_N] = [const { [const { AtomicU64::new(0) }; 4] }; RING_N];
static HEAD: AtomicU64 = AtomicU64::new(0);

/// `XNW_DWATCH` parsed: (image address, length).
pub(crate) fn parse(v: &str) -> Option<(usize, usize)> {
    let mut p = v.trim().split(':');
    let a = usize::from_str_radix(p.next()?.trim().trim_start_matches("0x"), 16).ok()?;
    let len = p.next().map_or(Some(1), |l| l.trim().parse().ok())?;
    matches!(len, 1 | 2 | 4 | 8).then_some((a, len))
}

/// DR7 for slot 0: local enable, break on write, the length.
pub(crate) fn dr7(len: usize) -> u64 {
    let l = match len {
        1 => 0,
        2 => 1,
        8 => 2,
        _ => 3,
    };
    1 | 1 << 16 | (l as u64) << 18
}

unsafe extern "system" fn on_step(info: *mut win::ExceptionPointers) -> i32 {
    // SAFETY: the system's exception record and context for this thread.
    let (rec, ctx) = unsafe { (&*(*info).record, &mut *(*info).context) };
    if rec.code != SINGLE_STEP || ctx.get(win::Context::DR6) & 1 == 0 {
        return 0; // EXCEPTION_CONTINUE_SEARCH
    }
    let Some(t) = TARGET.get() else { return 0 };
    // SAFETY: the watched bytes and the frame id are title globals (mapped).
    let (value, frame) = unsafe {
        let mut v = [0u8; 8];
        std::ptr::copy_nonoverlapping(t.addr as *const u8, v.as_mut_ptr(), t.len);
        (u64::from_le_bytes(v), if t.base != 0 { ((t.base + (FRAME_ID - IMAGE_BASE)) as *const u32).read_volatile() } else { 0 })
    };
    // SAFETY: plain call.
    let tid = unsafe { win::GetCurrentThreadId() };
    let i = HEAD.fetch_add(1, Ordering::Relaxed) as usize & (RING_N - 1);
    RING[i][0].store(ctx.rip() as u64, Ordering::Relaxed);
    RING[i][1].store(u64::from(tid), Ordering::Relaxed);
    RING[i][2].store(value, Ordering::Relaxed);
    RING[i][3].store(u64::from(frame), Ordering::Release);
    ctx.set(win::Context::DR6, 0);
    -1 // EXCEPTION_CONTINUE_EXECUTION
}

/// Sets DR0/DR7 on thread `id` (not the calling thread).
fn arm(id: u32, t: &Target) -> bool {
    // SAFETY: a thread handle of our process; suspended around the context write.
    unsafe {
        let h = win::OpenThread(THREAD_ACCESS, 0, id);
        if h.is_null() {
            return false;
        }
        let mut ok = false;
        if win::SuspendThread(h) != u32::MAX {
            let mut c = win::Context::zeroed();
            c.set(win::Context::FLAGS, CONTEXT_DEBUG_REGISTERS);
            if win::GetThreadContext(h, &mut c) != 0 {
                c.set(win::Context::DR0, t.addr as u64);
                c.set(win::Context::DR7, (c.get(win::Context::DR7) & !0xF_0003) | dr7(t.len));
                c.set(win::Context::FLAGS, CONTEXT_DEBUG_REGISTERS);
                ok = win::SetThreadContext(h, &c) != 0;
            }
            win::ResumeThread(h);
        }
        win::CloseHandle(h);
        ok
    }
}

/// The process's thread ids.
fn threads() -> Vec<u32> {
    let mut out = Vec::new();
    // SAFETY: a toolhelp snapshot walked with a sized entry, then closed.
    unsafe {
        let snap = win::CreateToolhelp32Snapshot(win::TH32CS_SNAPTHREAD, 0);
        if snap == win::INVALID_HANDLE_VALUE {
            return out;
        }
        let me = win::GetCurrentProcessId();
        let mut e: win::ThreadEntry32 = std::mem::zeroed();
        e.size = std::mem::size_of::<win::ThreadEntry32>() as u32;
        let mut more = win::Thread32First(snap, &mut e) != 0;
        while more {
            if e.owner_process_id == me {
                out.push(e.thread_id);
            }
            more = win::Thread32Next(snap, &mut e) != 0;
        }
        win::CloseHandle(snap);
    }
    out
}

fn install_target(t: Target) {
    let _ = TARGET.set(t);
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: a plain extern handler, first in the chain.
    ONCE.call_once(|| unsafe {
        win::AddVectoredExceptionHandler(1, on_step);
    });
}

/// The records since `from` (ring index) as (rip, thread, value, frame).
fn records(from: &mut u64) -> Vec<[u64; 4]> {
    let head = HEAD.load(Ordering::Acquire);
    let start = (*from).max(head.saturating_sub(RING_N as u64));
    let out = (start..head).map(|i| std::array::from_fn(|k| RING[i as usize & (RING_N - 1)][k].load(Ordering::Acquire))).collect();
    *from = head;
    out
}

/// Starts the watch when `XNW_DWATCH` is set.
///
/// # Safety
/// `base` is the mapped title.
pub unsafe fn start(base: usize) {
    let Some((va, len)) = std::env::var("XNW_DWATCH").ok().and_then(|v| parse(&v)) else { return };
    let t = Target { addr: base + (va - IMAGE_BASE), len, base };
    install_target(t);
    xnw_core::info!("dwatch: DR0 write watch on {va:#x} ({len} B) for every thread of the process");
    std::thread::spawn(move || {
        // SAFETY: plain call.
        let me = unsafe { win::GetCurrentThreadId() };
        let mut armed: HashSet<u32> = HashSet::new();
        let (mut from, mut n) = (0u64, 0u64);
        loop {
            for id in threads() {
                if id != me && !armed.contains(&id) && arm(id, &t) {
                    armed.insert(id);
                }
            }
            for [rip, tid, value, frame] in records(&mut from) {
                n += 1;
                if n <= 64 || n % 100 == 0 {
                    let at = if (rip as usize) >= base && (rip as usize) - base < 0x1000_0000 { format!("title+{:#x}", rip as usize - base) } else { format!("{rip:#x}") };
                    xnw_core::info!("dwatch: write #{n} to {va:#x}: value {value:#x} by {at} (instruction after the write), thread {tid}, title frame {frame}");
                }
            }
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switch_and_dr7() {
        assert_eq!(parse("0x14765A1C3"), Some((0x1_4765_A1C3, 1)));
        assert_eq!(parse("14765A1C0:4"), Some((0x1_4765_A1C0, 4)));
        assert_eq!(parse("14765A1C0:3"), None);
        assert_eq!(dr7(1), 0x1_0001);
        assert_eq!(dr7(4), 0xD_0001);
        assert_eq!(dr7(8), 0x9_0001);
    }

    /// The handler path without the title: a thread of ours is armed on a
    /// local byte, writes it twice; both writes are recorded with the value
    /// and the thread, and the thread runs on.
    #[test]
    fn a_write_by_an_armed_thread_is_recorded() {
        static CELL: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
        let t = Target { addr: CELL.as_ptr() as usize, len: 1, base: 0 };
        install_target(t);
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let (idtx, idrx) = std::sync::mpsc::channel::<u32>();
        let worker = std::thread::spawn(move || {
            // SAFETY: plain call.
            idtx.send(unsafe { win::GetCurrentThreadId() }).unwrap();
            rx.recv().unwrap();
            CELL.store(7, Ordering::SeqCst);
            CELL.store(9, Ordering::SeqCst);
            CELL.load(Ordering::SeqCst)
        });
        let id = idrx.recv().unwrap();
        let mut from = HEAD.load(Ordering::Acquire);
        assert!(arm(id, &t), "armed");
        tx.send(()).unwrap();
        assert_eq!(worker.join().unwrap(), 9, "the thread ran on");
        let r = records(&mut from);
        assert_eq!(r.len(), 2, "{r:x?}");
        assert_eq!((r[0][1], r[0][2], r[1][2]), (u64::from(id), 7, 9));
    }
}
