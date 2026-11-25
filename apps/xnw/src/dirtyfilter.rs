//! `XNW_DIRTYFILTER=a|ab|count` (off when unset): stop the title's
//! constant-buffer bind from dirtying a descriptor table when the bind
//! changes nothing.
//!
//! - `a`: detour on the CB bind sink 0x144DA8E70 `(sub = ctx+0x12780, code,
//!   base, offset, size)`: returns `0x1000 << (code & 3)` without running
//!   when the bind is from the persistent pool and the slot's record already
//!   holds it (all fields V). The original ORs that bit into
//!   ctx+0x20 and rewrites record, CBV and cb[s] with the same values.
//! - `ab`: also 0x1400378C0 `(sub, code, data, size)`, the ring copy: when
//!   the slot's ring block holds the same bytes, one page reference
//!   (0x144D9F420) and the old block back instead of a new block (the
//!   caller's 0x144DA8E20 drops one reference and stores the same block).
//! - `count`: both detours, never skipping: counts what would be skipped.
//!
//! A line `dirty filter: ...` per second with binds seen, skipped (or would
//! skip) and the reasons not to. The checks read the context through the
//! pointer the original uses (the hook runs before it touches the same
//! fields); a pointer or slot index out of range means "run the original".

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::Duration;

const IMAGE_BASE: usize = 0x1_4000_0000;
const CB_SET: usize = 0x1_44DA_8E70;
const CB_RING: usize = 0x1_4003_78C0;
const POOL_BASE: usize = 0x1_475D_F140;
const RING_BASE32: usize = 0x1_475D_F1B8;
const PAGE_REF: usize = 0x1_44D9_F420;
const CB_SET_ENTRY: [u8; 14] = [0x41, 0x56, 0x56, 0x57, 0x53, 0x48, 0x83, 0xEC, 0x38, 0x0F, 0xB7, 0xFA, 0x89, 0xD6];
const CB_RING_ENTRY: [u8; 14] = [0x41, 0x57, 0x41, 0x56, 0x56, 0x57, 0x53, 0x48, 0x83, 0xEC, 0x30, 0x4C, 0x89, 0xCE];

/// Context offsets from `sub` (= ctx + 0x12780): records ctx+0x7440 (64 B
/// each), cb[] ctx+0x5540, cbLazy ctx+0x70C0 (u16), pageRef ctx+0xDB40.
const SUB: u64 = 0x1_2780;
const REC: u64 = 0x7440;
const CB: u64 = 0x5540;
const LAZY: u64 = 0x70C0;
const PAGE: u64 = 0xDB40;
/// Records fit below pageRef: (0xDB40 - 0x7440) / 64 slots.
const MAX_SLOT: u64 = (PAGE - REC) / 64;
const MAX_RING_BYTES: u64 = 0x1_0000;

const OFF: u8 = 0;
const COUNT: u8 = 1;
const A: u8 = 2;
const AB: u8 = 3;
static MODE: AtomicU8 = AtomicU8::new(OFF);

/// Why a bind is not skipped.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Why {
    NotPool,
    Lazy,
    Ring,
    Differs,
    Bytes,
    Bad,
}
const WHY: [&str; 6] = ["not pool", "lazy", "ring block", "record differs", "bytes differ", "bad pointer"];

struct Counts {
    seen: AtomicU64,
    skip: AtomicU64,
    why: [AtomicU64; 6],
}
static CA: Counts = Counts { seen: AtomicU64::new(0), skip: AtomicU64::new(0), why: [const { AtomicU64::new(0) }; 6] };
static CBR: Counts = Counts { seen: AtomicU64::new(0), skip: AtomicU64::new(0), why: [const { AtomicU64::new(0) }; 6] };
static TITLE: AtomicU64 = AtomicU64::new(0);

/// # Safety
/// `at` is readable for 8 bytes.
unsafe fn rd<T: Copy>(at: u64) -> T {
    // SAFETY: the caller's.
    unsafe { (at as *const T).read_unaligned() }
}

/// The slot's record, cb[] and lazy: `Err(Bad)` when `sub` or the slot
/// cannot be the context's.
fn slot(sub: u64, code: u16) -> Result<(u64, u64), Why> {
    let s = u64::from(code >> 2);
    if sub < 0x1_0000 + SUB || sub >= 0x8000_0000_0000 || s >= MAX_SLOT {
        return Err(Why::Bad);
    }
    let ctx = sub - SUB;
    Ok((ctx, s))
}

/// The slot holds its own record, not lazy data.
///
/// # Safety
/// `ctx` is a render context.
unsafe fn own_record(ctx: u64, s: u64) -> Result<u64, Why> {
    let rec = ctx + REC + 64 * s;
    // SAFETY: fields of the context.
    let (cb, lazy) = unsafe { (rd::<u64>(ctx + CB + 8 * s), rd::<u16>(ctx + LAZY + 2 * s)) };
    if cb != rec || lazy != 0 {
        return Err(Why::Lazy);
    }
    Ok(rec)
}

/// The record already holds (offset, size), CBV kind 0x10, +0x20 clear.
///
/// # Safety
/// `rec` is a CB record of a context.
unsafe fn record_equal(rec: u64, off: u32, size: u64) -> bool {
    // SAFETY: the record's fields.
    unsafe { rd::<u32>(rec + 0x14) == off && rd::<u32>(rec + 8) == (size >> 4) as u32 && rd::<u16>(rec + 0xC) & 0xFFF == 0x10 && rd::<u64>(rec + 0x20) == 0 }
}

/// Detour A's condition.
///
/// # Safety
/// `sub` is what the title passes 0x144DA8E70 (ctx + 0x12780).
pub(crate) unsafe fn check_a(sub: u64, code: u16, base: u64, off: u32, size: u64, pool: u64) -> Result<(), Why> {
    let (ctx, s) = slot(sub, code)?;
    if pool == 0 || base != pool || size & 15 != 0 {
        return Err(Why::NotPool);
    }
    // SAFETY: the context the original writes.
    let rec = unsafe { own_record(ctx, s)? };
    // SAFETY: as above.
    if unsafe { rd::<u64>(ctx + PAGE + 8 * s) } != 0 {
        return Err(Why::Ring);
    }
    // SAFETY: as above.
    if unsafe { !record_equal(rec, off, size) } {
        return Err(Why::Differs);
    }
    Ok(())
}

/// Detour B's condition: the ring block the slot holds, when it holds
/// these bytes and the record points at it.
///
/// # Safety
/// `sub` as for [`check_a`]; `data` readable for `size` bytes.
pub(crate) unsafe fn check_b(sub: u64, code: u16, data: u64, size: u64, ring_base32: u32) -> Result<u64, Why> {
    let (ctx, s) = slot(sub, code)?;
    // SAFETY: the context.
    let p = unsafe { rd::<u64>(ctx + PAGE + 8 * s) };
    if p == 0 {
        return Err(Why::Ring);
    }
    if p < 0x1_0000 || data < 0x1_0000 || size == 0 || size & 15 != 0 || size > MAX_RING_BYTES {
        return Err(Why::Bad);
    }
    // SAFETY: as above.
    let rec = unsafe { own_record(ctx, s)? };
    // SAFETY: as above.
    if unsafe { !record_equal(rec, (p as u32).wrapping_sub(ring_base32), size) } {
        return Err(Why::Differs);
    }
    // SAFETY: p is the slot's live (referenced) block of at least the
    // recorded size, equal to `size`; data is the caller's source.
    let same = unsafe { std::slice::from_raw_parts(p as *const u8, size as usize) == std::slice::from_raw_parts(data as *const u8, size as usize) };
    if !same {
        return Err(Why::Bytes);
    }
    Ok(p)
}

fn real(va: usize) -> u64 {
    (TITLE.load(Ordering::Relaxed) as usize).wrapping_add(va - IMAGE_BASE) as u64
}

/// Arguments from the stub frame: rcx, rdx, r8, r9, and the fifth.
///
/// # Safety
/// `frame` is the stub's.
unsafe fn args(frame: usize) -> [u64; 5] {
    // SAFETY: the stub saved them there; the fifth is the caller's stack.
    unsafe { [frame - 24, frame - 32, frame - 40, frame - 48, frame + 0x30].map(|at| (at as *const u64).read()) }
}

extern "system" fn on_cb_set(frame: usize, _id: usize) -> u64 {
    // SAFETY: the stub's frame.
    let [sub, code, base, off, size] = unsafe { args(frame) };
    let code = code as u16;
    CA.seen.fetch_add(1, Ordering::Relaxed);
    // SAFETY: a title global; `sub` is the original's argument.
    let r = unsafe { check_a(sub, code, base, off as u32, size, rd::<u64>(real(POOL_BASE))) };
    match r {
        Ok(()) => {
            CA.skip.fetch_add(1, Ordering::Relaxed);
            if MODE.load(Ordering::Relaxed) == COUNT { 0 } else { 0x1000u64 << (code & 3) }
        }
        Err(w) => {
            CA.why[w as usize].fetch_add(1, Ordering::Relaxed);
            0
        }
    }
}

extern "system" fn on_cb_ring(frame: usize, _id: usize) -> u64 {
    // SAFETY: the stub's frame.
    let [sub, code, data, size, _] = unsafe { args(frame) };
    CBR.seen.fetch_add(1, Ordering::Relaxed);
    // SAFETY: a title global; arguments the original receives.
    let r = unsafe { check_b(sub, code as u16, data, size, rd::<u32>(real(RING_BASE32))) };
    match r {
        Ok(p) => {
            CBR.skip.fetch_add(1, Ordering::Relaxed);
            if MODE.load(Ordering::Relaxed) != AB {
                return 0;
            }
            // SAFETY: the title's page reference +1 (0x144D9F420(block)),
            // on the context's own thread as the title calls it.
            let page_ref: extern "system" fn(u64) -> u64 = unsafe { std::mem::transmute(real(PAGE_REF) as usize) };
            page_ref(p);
            p
        }
        Err(w) => {
            CBR.why[w as usize].fetch_add(1, Ordering::Relaxed);
            0
        }
    }
}

fn line(name: &str, c: &Counts, last: &mut [u64; 8]) -> Option<String> {
    let now = [c.seen.load(Ordering::Relaxed), c.skip.load(Ordering::Relaxed)].into_iter().chain(c.why.iter().map(|w| w.load(Ordering::Relaxed))).collect::<Vec<_>>();
    let d: Vec<u64> = now.iter().zip(last.iter()).map(|(n, l)| n - l).collect();
    last.copy_from_slice(&now);
    if d[0] == 0 {
        return None;
    }
    let why: Vec<String> = WHY.iter().zip(&d[2..]).filter(|(_, n)| **n > 0).map(|(w, n)| format!("{w} {n}")).collect();
    Some(format!("{name} {} seen, {} {} ({:.1} %), not: {}", d[0], d[1], if MODE.load(Ordering::Relaxed) == COUNT { "would skip" } else { "skipped" }, 100.0 * d[1] as f64 / d[0] as f64, if why.is_empty() { "-".into() } else { why.join(", ") }))
}

fn parse(v: &str) -> u8 {
    match v.trim() {
        "a" => A,
        "ab" => AB,
        "count" => COUNT,
        _ => OFF,
    }
}

/// Installs what `XNW_DIRTYFILTER` asks for; nothing when unset.
///
/// # Safety
/// `base` is the mapped title and none of its code runs yet.
pub unsafe fn start(base: usize) {
    let Some(v) = std::env::var("XNW_DIRTYFILTER").ok() else { return };
    let mode = parse(&v);
    if mode == OFF {
        xnw_core::warn!("dirty filter: XNW_DIRTYFILTER={v}: expected a, ab or count; off");
        return;
    }
    TITLE.store(base as u64, Ordering::Relaxed);
    MODE.store(mode, Ordering::Relaxed);
    let mut any = false;
    let targets: &[(usize, [u8; 14], usize, &str)] = &[(CB_SET, CB_SET_ENTRY, on_cb_set as usize, "CB bind 0x144DA8E70"), (CB_RING, CB_RING_ENTRY, on_cb_ring as usize, "ring CB 0x1400378C0")];
    for (k, (at, want, hook, name)) in targets.iter().enumerate() {
        if k == 1 && mode == A {
            break;
        }
        let at = real(*at) as usize;
        let mut got = [0u8; 14];
        if !crate::watch::guarded_read(at, &mut got) || got != *want {
            xnw_core::warn!("dirty filter: {name}: entry {got:02x?} is not the expected one; not hooked");
            continue;
        }
        // SAFETY: nothing runs the title yet; the hook keeps the ABI.
        match unsafe { crate::watch::hook_entry_early(at as *mut u8, k, *hook) } {
            Ok(_) => any = true,
            Err(e) => xnw_core::warn!("dirty filter: {name}: not hooked: {e}"),
        }
    }
    if !any {
        return;
    }
    xnw_core::info!("dirty filter: mode {}", v.trim());
    std::thread::spawn(|| {
        let (mut la, mut lb) = ([0u64; 8], [0u64; 8]);
        loop {
            std::thread::sleep(Duration::from_secs(1));
            let a = line("CB binds", &CA, &mut la);
            let b = line("ring CB copies", &CBR, &mut lb);
            if a.is_some() || b.is_some() {
                xnw_core::info!("dirty filter: {}{}{}", a.as_deref().unwrap_or(""), if a.is_some() && b.is_some() { "; " } else { "" }, b.as_deref().unwrap_or(""));
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use xnw_core::win::{MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc};

    /// A zeroed context; returns (memory, sub).
    fn context() -> (Vec<u8>, u64) {
        let m = vec![0u8; 0x1_4000];
        let sub = m.as_ptr() as u64 + SUB;
        (m, sub)
    }

    fn put<T: Copy>(m: &mut [u8], at: u64, v: T) {
        // SAFETY: inside the test buffer.
        unsafe { (m.as_mut_ptr().add(at as usize) as *mut T).write_unaligned(v) };
    }

    /// Slot 3 bound from the pool: record (offset 0x200, 0x40 bytes), cb[3] = rec.
    fn bound(m: &mut [u8]) {
        let base = m.as_ptr() as u64;
        let rec = REC + 64 * 3;
        put(m, CB + 8 * 3, base + rec);
        put(m, rec + 0x14, 0x200u32);
        put(m, rec + 8, 4u32);
        put(m, rec + 0xC, 0x7010u16);
    }

    #[test]
    fn condition_a() {
        let (mut m, sub) = context();
        let code = 3 << 2 | 1;
        let pool = 0x2_0000_0000;
        // SAFETY (all calls): the synthetic context.
        unsafe {
            assert_eq!(check_a(sub, code, pool, 0x200, 0x40, pool), Err(Why::Lazy), "first bind: cb[s] is not the record");
            bound(&mut m);
            assert_eq!(check_a(sub, code, pool, 0x200, 0x40, pool), Ok(()));
            assert_eq!(check_a(sub, code, pool + 0x10, 0x200, 0x40, pool), Err(Why::NotPool));
            assert_eq!(check_a(sub, code, pool, 0x200, 0x44, pool), Err(Why::NotPool), "size not a multiple of 16");
            assert_eq!(check_a(sub, code, pool, 0x210, 0x40, pool), Err(Why::Differs));
            assert_eq!(check_a(sub, code, pool, 0x200, 0x50, pool), Err(Why::Differs));
            put(&mut m, LAZY + 2 * 3, 480u16);
            assert_eq!(check_a(sub, code, pool, 0x200, 0x40, pool), Err(Why::Lazy));
            put(&mut m, LAZY + 2 * 3, 0u16);
            put(&mut m, PAGE + 8 * 3, 0x1234_0000u64);
            assert_eq!(check_a(sub, code, pool, 0x200, 0x40, pool), Err(Why::Ring));
            put(&mut m, PAGE + 8 * 3, 0u64);
            put(&mut m, REC + 64 * 3 + 0x20, 1u64);
            assert_eq!(check_a(sub, code, pool, 0x200, 0x40, pool), Err(Why::Differs));
            assert_eq!(check_a(0x100, code, pool, 0x200, 0x40, pool), Err(Why::Bad));
            assert_eq!(check_a(sub, u16::MAX, pool, 0x200, 0x40, pool), Err(Why::Bad));
            assert_eq!(check_a(sub, code, 0, 0x200, 0x40, 0), Err(Why::NotPool));
        }
    }

    #[test]
    fn condition_b() {
        let (mut m, sub) = context();
        let code = 3 << 2;
        let block = vec![7u8; 0x40];
        let p = block.as_ptr() as u64;
        let ring32 = (p as u32).wrapping_sub(0x200);
        let same = vec![7u8; 0x40];
        let mut other = same.clone();
        other[0x3F] = 8;
        // SAFETY (all calls): the synthetic context and buffers.
        unsafe {
            bound(&mut m);
            assert_eq!(check_b(sub, code, same.as_ptr() as u64, 0x40, ring32), Err(Why::Ring), "no block held");
            put(&mut m, PAGE + 8 * 3, p);
            assert_eq!(check_b(sub, code, same.as_ptr() as u64, 0x40, ring32), Ok(p));
            assert_eq!(check_b(sub, code, other.as_ptr() as u64, 0x40, ring32), Err(Why::Bytes));
            assert_eq!(check_b(sub, code, same.as_ptr() as u64, 0x30, ring32), Err(Why::Differs));
            assert_eq!(check_b(sub, code, same.as_ptr() as u64, 0x40, ring32 + 16), Err(Why::Differs));
            assert_eq!(check_b(sub, code, same.as_ptr() as u64, 0x48, ring32), Err(Why::Bad));
            put(&mut m, CB + 8 * 3, 0u64);
            assert_eq!(check_b(sub, code, same.as_ptr() as u64, 0x40, ring32), Err(Why::Lazy));
        }
    }

    static BODY_RUNS: AtomicU64 = AtomicU64::new(0);

    /// rcx == 1: replace with `0x1000 << (rdx & 3)`; rcx == 2: replace with
    /// the fifth argument; else run the body.
    extern "system" fn test_hook(frame: usize, _id: usize) -> u64 {
        // SAFETY: the stub's frame.
        let [rcx, rdx, _, _, fifth] = unsafe { args(frame) };
        match rcx {
            1 => 0x1000 << (rdx & 3),
            2 => fifth,
            _ => 0,
        }
    }

    /// A function with 0x144DA8E70's prologue; its body counts a run and
    /// returns 0x777 + r9.
    #[test]
    fn early_return_stub() {
        let counter = BODY_RUNS.as_ptr() as u64;
        let mut code = CB_SET_ENTRY.to_vec();
        code.extend([0x48, 0xB8]);
        code.extend(counter.to_le_bytes()); // mov rax, counter
        code.extend([0x48, 0xFF, 0x00]); // inc qword [rax]
        code.extend([0x4C, 0x89, 0xC8, 0x48, 0x05, 0x77, 0x07, 0, 0]); // mov rax,r9; add rax,0x777
        code.extend([0x48, 0x83, 0xC4, 0x38, 0x5B, 0x5F, 0x5E, 0x41, 0x5E, 0xC3]);
        // SAFETY: a fresh executable allocation of ours.
        let mem = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) }.cast::<u8>();
        // SAFETY: inside it.
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), mem, code.len()) };
        // SAFETY: the function above.
        let f: extern "system" fn(u64, u64, u64, u64, u64) -> u64 = unsafe { std::mem::transmute(mem) };
        assert_eq!(f(0, 0, 0, 5, 0), 0x77C);
        // SAFETY: nothing runs it now.
        unsafe { crate::watch::hook_entry_early(mem, 0, test_hook as usize) }.unwrap();
        let before = BODY_RUNS.load(Ordering::Relaxed);
        // Values kept in locals across the calls check callee-saved registers.
        let keep: Vec<u64> = (0..8).map(|i| i * 0x1111).collect();
        let mut sum = 0u64;
        for i in 0..100u64 {
            assert_eq!(f(1, i, 0, 0, 0), 0x1000 << (i & 3));
            assert_eq!(f(2, 0, 0, 0, 0xABC0 + i), 0xABC0 + i);
            assert_eq!(f(3, 0, 0, i, 0), 0x777 + i);
            sum += keep[(i % 8) as usize];
        }
        assert_eq!(BODY_RUNS.load(Ordering::Relaxed) - before, 100, "the body runs only when the hook returns 0");
        assert_eq!(sum, (0..100u64).map(|i| (i % 8) * 0x1111).sum::<u64>());
    }

    #[test]
    fn modes() {
        assert_eq!((parse("a"), parse(" ab "), parse("count"), parse("1")), (A, AB, COUNT, OFF));
    }
}
