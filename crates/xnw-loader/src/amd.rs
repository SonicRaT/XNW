//! Zen 2 instructions this CPU lacks, done in the fault handler instead.
//!
//! Adapted from VIctory, `crates/victory-loader/src/crash.rs` at d2a4bae
//! (GPL-3.0-or-later): `emulate_amd_only`, `clwb_len`, `clwb_loop`,
//! `cmp_flags`, `sse4a`. Rewritten on our byte-offset `Context`, with the
//! write-back loop generalised to `clzero` loops.
//!
//! The console CPU is Zen 2, so the title uses `clzero`, `clwb`,
//! `monitorx`/`mwaitx` and SSE4a's `insertq`, which an Intel CPU before Ice
//! Lake rejects with an illegal-instruction fault (32 sites).
//! [`emulate`] does the instruction's job
//! on the faulting thread's `CONTEXT` and moves `rip` past it.

use xnw_core::win::*;

/// Bytes at `rip`, as many as are readable up to 32. The faulting
/// instruction's own bytes always are (the CPU fetched them); the page after
/// it may not be mapped.
pub(crate) unsafe fn fetch(rip: usize) -> ([u8; 32], usize) {
    let mut bytes = [0u8; 32];
    let room = 0x1000 - (rip & 0xfff);
    let mut n = room.min(bytes.len());
    // SAFETY: `rip..rip + n` is in the faulting instruction's page.
    unsafe { std::ptr::copy_nonoverlapping(rip as *const u8, bytes.as_mut_ptr(), n) };
    if n < bytes.len() {
        let next = rip + room;
        if readable(next) {
            let more = bytes.len() - n;
            // SAFETY: the next page is committed and readable.
            unsafe { std::ptr::copy_nonoverlapping(next as *const u8, bytes.as_mut_ptr().add(n), more) };
            n += more;
        }
    }
    (bytes, n)
}

/// Title pages found readable, which stay so (a test run spent 2 % of
/// title CPU querying the page after an `insertq` 15 bytes before a page
/// end, on every emulation).
static READABLE: [std::sync::atomic::AtomicUsize; 64] = [const { std::sync::atomic::AtomicUsize::new(0) }; 64];

/// Whether the page at `page` is committed, not no-access, not a guard page.
fn readable(page: usize) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let slot = &READABLE[(page >> 12) % READABLE.len()];
    if slot.load(Relaxed) == page {
        return true;
    }
    let mut m = MemoryBasicInformation { base: 0, allocation_base: 0, allocation_protect: 0, partition: 0, region_size: 0, state: 0, protect: 0, kind: 0 };
    // SAFETY: a query of our own address space into a sized buffer.
    let ok = unsafe { VirtualQueryEx(GetCurrentProcess(), page as _, &mut m, size_of::<MemoryBasicInformation>()) } != 0;
    let yes = ok && m.state == MEM_COMMIT && m.protect & 0xff != PAGE_NOACCESS && m.protect & 0x100 == 0;
    if yes && crate::crash::in_title(page) {
        slot.store(page, Relaxed);
    }
    yes
}

/// Performs the instruction at `ctx.rip` if it is one of the Zen 2 ones,
/// advancing `rip`; returns its mnemonic, or `None` (nothing changed).
///
/// # Safety
/// `ctx` is the context of a thread that faulted at `rip`, whose code is
/// mapped; `clzero` writes the memory the title asked for.
pub(crate) unsafe fn emulate(ctx: &mut Context) -> Option<&'static str> {
    let rip = ctx.rip();
    // SAFETY: the caller's guarantee.
    let (bytes, n) = unsafe { fetch(rip) };
    // SAFETY: forwarded.
    unsafe { emulate_bytes(&bytes[..n], ctx) }
}

/// [`emulate`] on `code`, the bytes at `ctx.rip`.
unsafe fn emulate_bytes(code: &[u8], ctx: &mut Context) -> Option<&'static str> {
    let (name, len) = match code.get(..3)? {
        [0x0f, 0x01, 0xfc] => {
            if let Some(l) = line_loop(code) {
                // SAFETY: the title's own loop would write these lines.
                unsafe { l.run(ctx, true) };
                ("clzero", l.len)
            } else {
                // SAFETY: the line the title names.
                unsafe { ((ctx.gpr(0) & !63) as *mut u8).write_bytes(0, 64) };
                ("clzero", 3)
            }
        }
        [0x0f, 0x01, 0xfa] => ("monitorx", 3),
        [0x0f, 0x01, 0xfb] => {
            // A timed wait for a store to the monitored line: a yield,
            // so the spin loop around it goes on.
            std::thread::yield_now();
            ("mwaitx", 3)
        }
        _ => {
            if let Some(l) = line_loop(code) {
                // SAFETY: a write-back hint writes nothing.
                unsafe { l.run(ctx, false) };
                ("clwb", l.len)
            } else if let Some(len) = clwb_len(code) {
                // The PC's memory is coherent: nothing to write back.
                ("clwb", len)
            } else {
                sse4a(code, ctx)?
            }
        }
    };
    ctx.set(Context::RIP, rip_after(ctx, len));
    Some(name)
}

fn rip_after(ctx: &Context, len: usize) -> u64 {
    ctx.get(Context::RIP).wrapping_add(len as u64)
}

/// `clwb m8` (`66 [rex] 0f ae /6`, a memory operand), which Intel CPUs before
/// Ice Lake lack: its length, or `None` for any other instruction.
pub(crate) fn clwb_len(code: &[u8]) -> Option<usize> {
    let at = if code.get(1)? & 0xf0 == 0x40 { 2 } else { 1 };
    if code[0] != 0x66 || code.get(at..at + 2)? != [0x0f, 0xae] {
        return None;
    }
    let modrm = *code.get(at + 2)?;
    let (mode, rm) = (modrm >> 6, modrm & 7);
    if mode == 3 || (modrm >> 3) & 7 != 6 {
        return None;
    }
    let mut len = at + 3;
    let mut base = rm;
    if rm == 4 {
        base = code.get(len)? & 7;
        len += 1;
    }
    len += match mode {
        0 if rm == 5 || (rm == 4 && base == 5) => 4,
        1 => 1,
        2 => 4,
        _ => 0,
    };
    Some(len)
}

/// A cache-line loop as the title's compilers build it, one line a pass
/// (`base` and `limit` any registers; `clzero` always works on `rax`):
///
/// ```text
/// top: clzero          | clwb [base]
///      add  base, 0x40
///      cmp  base, limit
///      jb   top        | jbe top
/// ```
///
/// An exception costs about a microsecond and the title clears and writes
/// back whole buffers this way, so the loop is done in one exception.
#[derive(Debug)]
struct LineLoop {
    base: usize,
    limit: usize,
    /// `jbe` (loop while `base <= limit`) rather than `jb`.
    or_equal: bool,
    /// Bytes from the first instruction to the one after the jump.
    len: usize,
}

fn line_loop(code: &[u8]) -> Option<LineLoop> {
    let (base, add) = if code.get(..3)? == [0x0f, 0x01, 0xfc] {
        (0, 3)
    } else {
        // clwb [base]: 66 [40|41] 0f ae, ModRM mod 00, reg 6, no SIB or displacement.
        let (rex, at) = match *code.get(1)? {
            r @ (0x40 | 0x41) => (r, 2),
            _ => (0, 1),
        };
        if code[0] != 0x66 || code.get(at..at + 2)? != [0x0f, 0xae] {
            return None;
        }
        let modrm = *code.get(at + 2)?;
        let rm = modrm & 7;
        if modrm >> 6 != 0 || (modrm >> 3) & 7 != 6 || rm == 4 || rm == 5 {
            return None;
        }
        (usize::from(rm | (rex & 1) << 3), at + 3)
    };
    // add base, 0x40: REX.W(+B) 83 /0 ib.
    let b = base as u8;
    if *code.get(add..add + 4)? != [0x48 | b >> 3, 0x83, 0xc0 | (b & 7), 0x40] {
        return None;
    }
    // cmp base, limit: REX.W 39 /r with base in r/m.
    let cmp = add + 4;
    let &[rex @ 0x48..=0x4f, 0x39, modrm] = code.get(cmp..cmp + 3)? else { return None };
    if modrm >> 6 != 3 || usize::from(modrm & 7 | (rex & 1) << 3) != base {
        return None;
    }
    let limit = usize::from((modrm >> 3) & 7 | (rex & 4) << 1);
    let jump = cmp + 3;
    let len = jump + 2;
    let or_equal = match *code.get(jump)? {
        0x72 => false,
        0x76 => true,
        _ => return None,
    };
    if i64::from(*code.get(jump + 1)? as i8) != -(len as i64) || limit == base {
        return None;
    }
    Some(LineLoop { base, limit, or_equal, len })
}

impl LineLoop {
    /// Runs every pass (zeroing each line for `clzero`) and leaves `base`
    /// and the flags as the last `cmp` does.
    unsafe fn run(&self, ctx: &mut Context, zero: bool) {
        let (start, limit) = (ctx.gpr(self.base), ctx.gpr(self.limit));
        let goes_on = |b: u64| if self.or_equal { b <= limit } else { b < limit };
        let passes = if !goes_on(start.wrapping_add(64)) {
            1
        } else if self.or_equal {
            (limit - start) / 64 + 1
        } else {
            (limit - start).div_ceil(64)
        };
        if zero {
            for i in 0..passes {
                // SAFETY: the lines the title's loop zeroes.
                unsafe { ((start.wrapping_add(i * 64) & !63) as *mut u8).write_bytes(0, 64) };
            }
        }
        let end = start.wrapping_add(passes * 64);
        ctx.set_gpr(self.base, end);
        ctx.set_eflags(ctx.eflags() & !CMP_FLAGS | cmp_flags(end, limit));
    }
}

/// The status flags `cmp a, b` leaves (CF, PF, AF, ZF, SF, OF) as `EFLAGS` bits.
fn cmp_flags(a: u64, b: u64) -> u32 {
    let r = a.wrapping_sub(b);
    u32::from(a < b)
        | u32::from((r as u8).count_ones().is_multiple_of(2)) << 2
        | (((a ^ b ^ r) >> 4) as u32 & 1) << 4
        | u32::from(r == 0) << 6
        | ((r >> 63) as u32) << 7
        | ((((a ^ b) & (a ^ r)) >> 63) as u32) << 11
}
const CMP_FLAGS: u32 = 0x8d5;

/// SSE4a's `insertq`/`extrq` (register forms, the only ones there are) on
/// the saved XMM registers; returns the mnemonic and length.
///
/// `insertq dst, src, len, idx` puts the low `len` bits of `src` at bit
/// `idx` of `dst`'s low quadword (the register form takes `len`/`idx` from
/// `src[64..80]`); `dst`'s high quadword is undefined and left as it is.
/// `extrq` moves `len` bits from bit `idx` down to bit 0.
fn sse4a(code: &[u8], ctx: &mut Context) -> Option<(&'static str, usize)> {
    let prefix = *code.first()?;
    let (rex, at) = if code.get(1)? & 0xf0 == 0x40 { (code[1], 2) } else { (0, 1) };
    if !matches!(prefix, 0x66 | 0xf2) || *code.get(at)? != 0x0f || !matches!(code.get(at + 1)?, 0x78 | 0x79) {
        return None;
    }
    let modrm = *code.get(at + 2)?;
    if modrm >> 6 != 3 {
        return None;
    }
    let dst = usize::from((modrm >> 3) & 7 | (rex & 4) << 1);
    let src = usize::from(modrm & 7 | (rex & 1) << 3);
    let immediate = code[at + 1] == 0x78;
    let (fields, len) = if immediate {
        ((*code.get(at + 3)?, *code.get(at + 4)?), at + 5)
    } else {
        // extrq reads its fields from src's low half, insertq from its high.
        let f = ctx.xmm(src, usize::from(prefix == 0xf2));
        ((f as u8, (f >> 8) as u8), at + 3)
    };
    let (n, at_bit) = (u32::from(fields.0 & 63), u32::from(fields.1 & 63));
    let mask = if n == 0 { u64::MAX } else { (1u64 << n) - 1 };
    if prefix == 0x66 {
        // extrq with immediates is `66 0f 78 /0 ib ib`, working on r/m.
        if immediate && (modrm >> 3) & 7 != 0 {
            return None;
        }
        let target = if immediate { src } else { dst };
        ctx.set_xmm(target, 0, (ctx.xmm(target, 0) >> at_bit) & mask);
        Some(("extrq", len))
    } else {
        let value = (ctx.xmm(src, 0) & mask) << at_bit;
        ctx.set_xmm(dst, 0, ctx.xmm(dst, 0) & !(mask << at_bit) | value);
        Some(("insertq", len))
    }
}

#[cfg(test)]
mod tests {
    //! Each instruction form the title uses (made-up encodings of the same
    //! instructions, operand kinds and loop shapes) runs from
    //! an executable buffer on this CPU with the fault handler installed;
    //! registers, flags and memory are compared with a plain software model.
    //! On a CPU that has the instructions they run natively and the model
    //! checks the hardware instead.

    use super::*;
    use std::arch::x86_64::__cpuid;
    use std::sync::atomic::Ordering;

    /// What the buffer loads before the instruction and stores after it.
    #[repr(C, align(16))]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct State {
        xmm: [[u64; 2]; 8],
        /// rax, rcx, rdx, rbx, r8, r9, r10, r11.
        gpr: [u64; 8],
        saved: [[u64; 2]; 2],
        flags: u64,
    }
    const GPR: usize = 0x80;
    const SAVED: usize = 0xC0;
    const FLAGS: usize = 0xE0;
    const GPRS: [u8; 8] = [0, 1, 2, 3, 8, 9, 10, 11];

    impl State {
        fn new() -> Self {
            let mut s = State { xmm: [[0; 2]; 8], gpr: [0; 8], saved: [[0; 2]; 2], flags: 0 };
            for (i, x) in s.xmm.iter_mut().enumerate() {
                let k = i as u64 + 1;
                *x = [0x0123_4567_89ab_cdef_u64.rotate_left(8 * k as u32) ^ k, 0xfedc_ba98_7654_3210_u64.rotate_right(4 * k as u32)];
            }
            s
        }
    }

    fn disp32(code: &mut Vec<u8>, at: usize) {
        code.extend((at as u32).to_le_bytes());
    }

    /// `fn(*mut State)`: saves xmm6/xmm7, loads the state, runs `body`,
    /// stores the state and the flags, restores xmm6/xmm7.
    fn harness(body: &[u8]) -> Vec<u8> {
        let mut c = vec![0x53, 0x57, 0x48, 0x89, 0xcf]; // push rbx; push rdi; mov rdi, rcx
        for i in 0..2u8 {
            c.extend([0xf3, 0x0f, 0x7f, 0x87 | (6 + i) << 3]); // movdqu [rdi+saved], xmm6/7
            disp32(&mut c, SAVED + 16 * usize::from(i));
        }
        for i in 0..8u8 {
            c.extend([0xf3, 0x0f, 0x6f, 0x87 | i << 3]); // movdqu xmmi, [rdi+16i]
            disp32(&mut c, 16 * usize::from(i));
        }
        let gpr = |c: &mut Vec<u8>, op: u8, j: usize, r: u8| {
            c.extend([0x48 | (r >> 3) << 2, op, 0x87 | (r & 7) << 3]);
            disp32(c, GPR + 8 * j);
        };
        for (j, &r) in GPRS.iter().enumerate() {
            gpr(&mut c, 0x8b, j, r);
        }
        c.extend(body);
        c.extend([0x9c, 0x41, 0x5c]); // pushfq; pop r12 (r12 restored below)
        for (j, &r) in GPRS.iter().enumerate() {
            gpr(&mut c, 0x89, j, r);
        }
        gpr(&mut c, 0x89, (FLAGS - GPR) / 8, 12);
        for i in 0..8u8 {
            c.extend([0xf3, 0x0f, 0x7f, 0x87 | i << 3]);
            disp32(&mut c, 16 * usize::from(i));
        }
        for i in 0..2u8 {
            c.extend([0xf3, 0x0f, 0x6f, 0x87 | (6 + i) << 3]);
            disp32(&mut c, SAVED + 16 * usize::from(i));
        }
        c.extend([0x5f, 0x5b, 0xc3]); // pop rdi; pop rbx; ret
        // r12 is callee-saved: wrap the whole thing in push/pop r12.
        let mut out = vec![0x41, 0x54];
        out.extend(&c[..c.len() - 1]);
        out.extend([0x41, 0x5c, 0xc3]);
        out
    }

    /// Runs `body` on `state` from an executable page.
    fn run(body: &[u8], state: &mut State) {
        crate::crash::install();
        let code = harness(body);
        // SAFETY: a fresh page of ours, filled then made executable; the
        // code follows the Win64 ABI and touches only `state` and the
        // memory the case's registers point at.
        unsafe {
            let page = VirtualAlloc(std::ptr::null(), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE).cast::<u8>();
            assert!(!page.is_null());
            std::ptr::copy_nonoverlapping(code.as_ptr(), page, code.len());
            let mut old = 0;
            assert!(VirtualProtect(page.cast(), 4096, PAGE_EXECUTE_READ, &mut old) != 0);
            FlushInstructionCache(GetCurrentProcess(), page.cast(), 4096);
            let f: unsafe extern "system" fn(*mut State) = std::mem::transmute(page);
            f(state);
            VirtualFree(page.cast(), 0, MEM_RELEASE);
        }
    }

    /// Whether this CPU has each instruction (CPUID).
    fn native(name: &str) -> bool {
        // SAFETY: CPUID is always there on x86-64.
        unsafe {
            let ext = __cpuid(0x8000_0000).eax;
            match name {
                "insertq" => __cpuid(0x8000_0001).ecx & 1 << 6 != 0,
                "monitorx" | "mwaitx" => __cpuid(0x8000_0001).ecx & 1 << 29 != 0,
                "clzero" => ext >= 0x8000_0008 && __cpuid(0x8000_0008).ebx & 1 != 0,
                "clwb" => __cpuid(7).ebx & 1 << 24 != 0,
                _ => unreachable!(),
            }
        }
    }

    /// Runs `body`, checks the result against `model`, and that the handler
    /// took `faults` exceptions when the CPU lacks the instruction.
    fn check(name: &str, body: &[u8], mut state: State, faults: u64, model: impl FnOnce(&mut State)) {
        let mut want = state;
        model(&mut want);
        let before = crate::crash::EMULATED.load(Ordering::SeqCst);
        run(body, &mut state);
        let took = crate::crash::EMULATED.load(Ordering::SeqCst) - before;
        // Flags: only CF and ZF are modelled, and only for loops (the model
        // marks them with bit 63).
        state.flags = if want.flags >> 63 != 0 { 1 << 63 | state.flags & 0x41 } else { want.flags };
        state.saved = want.saved;
        assert_eq!(state, want, "{name} {body:02x?}");
        if !native(name) {
            // Other tests fault on other threads at the same time.
            assert!(took >= faults, "{name}: {took} emulated, want {faults}");
        }
    }

    /// Plain model of `insertq`: bit by bit.
    fn model_insertq(dst: u64, src: u64, len: u8, idx: u8) -> u64 {
        let len = if len & 63 == 0 { 64 } else { u32::from(len & 63) };
        let mut out = dst;
        for i in 0..len {
            let to = u32::from(idx & 63) + i;
            if to < 64 {
                out = out & !(1 << to) | ((src >> i) & 1) << to;
            }
        }
        out
    }

    #[test]
    fn insertq_with_immediates() {
        // (modrm, len, idx): distinct and equal registers, fields at 0, in
        // the middle and reaching bit 63.
        let forms: [(u8, u8, u8); 8] = [
            (0xca, 0x0c, 0x04), // insertq xmm1, xmm2, 12, 4
            (0xd3, 0x10, 0x20), // xmm2, xmm3, 16, 32
            (0xe6, 0x08, 0x00), // xmm4, xmm6, 8, 0
            (0xfc, 0x18, 0x08), // xmm7, xmm4, 24, 8
            (0xdb, 0x04, 0x3c), // xmm3, xmm3, 4, 60
            (0xf7, 0x20, 0x10), // xmm6, xmm7, 32, 16
            (0xed, 0x20, 0x20), // xmm5, xmm5, 32, 32
            (0xc0, 0x10, 0x28), // xmm0, xmm0, 16, 40
        ];
        for (modrm, len, idx) in forms {
            let body = [0xf2, 0x0f, 0x78, modrm, len, idx];
            let (dst, src) = (usize::from((modrm >> 3) & 7), usize::from(modrm & 7));
            check("insertq", &body, State::new(), 1, |s| s.xmm[dst][0] = model_insertq(s.xmm[dst][0], s.xmm[src][0], len, idx));
        }
    }

    #[test]
    fn insertq_with_fields_in_a_register() {
        // insertq xmm2, xmm5: length and index in xmm5[64..80]. Not in the
        // title; covered for completeness.
        for (len, idx) in [(16u8, 16u8), (8, 24), (0, 0), (5, 59)] {
            let mut s = State::new();
            s.xmm[5][1] = 0xdead_0000 | u64::from(idx) << 8 | u64::from(len);
            check("insertq", &[0xf2, 0x0f, 0x79, 0xd5], s, 1, |s| s.xmm[2][0] = model_insertq(s.xmm[2][0], s.xmm[5][0], len, idx));
        }
    }

    /// 256 bytes, four cache lines, filled with 0xaa.
    #[repr(C, align(64))]
    struct Lines([u8; 256]);

    #[test]
    fn clzero_alone() {
        // The instruction followed by nops: the bare form.
        let mut mem = Lines([0xaa; 256]);
        let at = mem.0.as_mut_ptr() as u64;
        let mut s = State::new();
        s.gpr[0] = at + 64 + 5;
        check("clzero", &[0x0f, 0x01, 0xfc, 0x90, 0x90], s, 1, |_| {});
        assert!(mem.0[..64].iter().all(|&b| b == 0xaa));
        assert!(mem.0[64..128].iter().all(|&b| b == 0));
        assert!(mem.0[128..].iter().all(|&b| b == 0xaa));
    }

    /// Model of a whole line loop: `do { op(base); base += 64 } while
    /// base < limit (or <=)`, with the flags of the last `cmp`.
    fn model_loop(s: &mut State, base: usize, limit: usize, or_equal: bool, mut op: impl FnMut(u64)) {
        loop {
            op(s.gpr[base]);
            s.gpr[base] += 64;
            let (a, b) = (s.gpr[base], s.gpr[limit]);
            s.flags = 1 << 63 | u64::from(a < b) | u64::from(a == b) << 6;
            if !(a < b || or_equal && a == b) {
                break;
            }
        }
    }

    #[test]
    fn clzero_line_loops() {
        // clzero; add rax, 40h; cmp rax, <limit>; jb top (limit: its State index).
        let loops: [(&[u8], usize); 3] = [
            (&[0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x4c, 0x39, 0xd8, 0x72, 0xf4], 7), // r11
            (&[0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x48, 0x39, 0xd8, 0x72, 0xf4], 3), // rbx
            (&[0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x48, 0x39, 0xc8, 0x72, 0xf4], 1), // rcx
        ];
        for (body, limit) in loops {
            // Limits inside a line, on a line boundary, and below the start.
            for (start, end) in [(0u64, 200u64), (0, 192), (64, 10), (3, 130)] {
                let mut mem = Lines([0xaa; 256]);
                let at = mem.0.as_mut_ptr() as u64;
                let mut s = State::new();
                (s.gpr[0], s.gpr[limit]) = (at + start, at + end);
                let mut zeroed = vec![];
                check("clzero", body, s, 1, |s| model_loop(s, 0, limit, false, |a| zeroed.push((a - at) as usize & !63)));
                for line in 0..4 {
                    let want = if zeroed.contains(&(line * 64)) { 0 } else { 0xaa };
                    assert!(mem.0[line * 64..][..64].iter().all(|&b| b == want), "{body:02x?} {start} {end} line {line}");
                }
            }
        }
    }

    #[test]
    fn clzero_in_a_loop_it_cannot_skip() {
        // top: mov rax, r11; add r11, 40h; clzero; cmp r11, r10; jb top: the
        // pointer is copied before the add, so no line loop. One fault a line.
        let body = [0x4c, 0x89, 0xd8, 0x49, 0x83, 0xc3, 0x40, 0x0f, 0x01, 0xfc, 0x4d, 0x39, 0xd3, 0x72, 0xf1];
        let mut mem = Lines([0xaa; 256]);
        let at = mem.0.as_mut_ptr() as u64;
        let mut s = State::new();
        (s.gpr[7], s.gpr[6]) = (at + 64, at + 192);
        check("clzero", &body, s, 2, |s| {
            s.gpr[0] = at + 128;
            s.gpr[7] = at + 192;
        });
        assert!(mem.0[..64].iter().all(|&b| b == 0xaa));
        assert!(mem.0[64..192].iter().all(|&b| b == 0));
        assert!(mem.0[192..].iter().all(|&b| b == 0xaa));
    }

    #[test]
    fn clwb_line_loops_and_bare() {
        let mut mem = Lines([0xaa; 256]);
        let at = mem.0.as_mut_ptr() as u64;
        // clwb [rdx]; add rdx, 40h; cmp rdx, rbx; jbe top
        let rdx_loop = [0x66, 0x0f, 0xae, 0x32, 0x48, 0x83, 0xc2, 0x40, 0x48, 0x39, 0xda, 0x76, 0xf3];
        // clwb [rbx]; add rbx, 40h; cmp rbx, rdx; jbe top
        let rbx_loop = [0x66, 0x0f, 0xae, 0x33, 0x48, 0x83, 0xc3, 0x40, 0x48, 0x39, 0xd3, 0x76, 0xf3];
        for (start, end) in [(0u64, 200u64), (0, 192), (64, 10)] {
            let mut s = State::new();
            (s.gpr[2], s.gpr[3]) = (at + start, at + end);
            check("clwb", &rdx_loop, s, 1, |s| model_loop(s, 2, 3, true, |_| {}));
            let mut s = State::new();
            (s.gpr[3], s.gpr[2]) = (at + start, at + end);
            check("clwb", &rbx_loop, s, 1, |s| model_loop(s, 3, 2, true, |_| {}));
        }
        // The bare forms, followed by something else.
        let mut s = State::new();
        (s.gpr[2], s.gpr[3]) = (at, at + 64);
        check("clwb", &[0x66, 0x0f, 0xae, 0x32, 0x90], s, 1, |_| {});
        check("clwb", &[0x66, 0x0f, 0xae, 0x73, 0x08, 0x90], s, 1, |_| {});
        assert!(mem.0.iter().all(|&b| b == 0xaa));
    }

    #[test]
    fn monitorx_and_mwaitx_do_nothing() {
        let mut mem = Lines([0xaa; 256]);
        let mut s = State::new();
        // monitorx rax, rcx(0), rdx(0); then mwaitx with ecx 2 (timer on),
        // ebx the timeout, eax 0.
        (s.gpr[0], s.gpr[1], s.gpr[2], s.gpr[3]) = (mem.0.as_mut_ptr() as u64, 0, 0, 1000);
        check("monitorx", &[0x0f, 0x01, 0xfa], s, 1, |_| {});
        (s.gpr[0], s.gpr[1]) = (0, 2);
        check("mwaitx", &[0x0f, 0x01, 0xfb], s, 1, |_| {});
    }

    fn ctx_with(xmm: &[(usize, u64, u64)]) -> Context {
        let mut c = Context::zeroed();
        for &(n, lo, hi) in xmm {
            c.set_xmm(n, 0, lo);
            c.set_xmm(n, 1, hi);
        }
        c
    }

    /// Emulates `code` on `ctx` without running it (any CPU).
    fn emulated(code: &[u8], ctx: &mut Context) -> Option<(&'static str, u64)> {
        let mut bytes = [0xccu8; 32];
        bytes[..code.len()].copy_from_slice(code);
        ctx.set(Context::RIP, 0x1000);
        // SAFETY: no clzero here, so no memory is written.
        let name = unsafe { emulate_bytes(&bytes, ctx) }?;
        Some((name, ctx.get(Context::RIP) - 0x1000))
    }

    #[test]
    fn sse4a_reaches_xmm8_to_xmm15_and_extrq_works() {
        let mut c = ctx_with(&[(8, 0x1111_2222_3333_4444, 7), (9, 0xabcd, 0)]);
        // insertq xmm8, xmm9, 16, 16 (REX.R and REX.B)
        assert_eq!(emulated(&[0xf2, 0x45, 0x0f, 0x78, 0xc1, 0x10, 0x10], &mut c), Some(("insertq", 7)));
        assert_eq!((c.xmm(8, 0), c.xmm(8, 1)), (0x1111_2222_abcd_4444, 7));
        // extrq xmm8, 8, 20 (REX.B)
        assert_eq!(emulated(&[0x66, 0x41, 0x0f, 0x78, 0xc0, 0x08, 0x14], &mut c), Some(("extrq", 7)));
        assert_eq!(c.xmm(8, 0), 0xbc);
        // extrq xmm0, xmm1: fields in xmm1's low half.
        let mut c = ctx_with(&[(0, 0x1111_2222_abcd_4444, 0), (1, 0x14_08, 0)]);
        assert_eq!(emulated(&[0x66, 0x0f, 0x79, 0xc1], &mut c), Some(("extrq", 4)));
        assert_eq!(c.xmm(0, 0), 0xbc);
    }

    #[test]
    fn other_instructions_are_left_alone() {
        let mut c = Context::zeroed();
        assert_eq!(emulated(&[0x0f, 0x0b], &mut c), None); // ud2
        assert_eq!(emulated(&[0x66, 0x0f, 0x79, 0x00], &mut c), None); // extrq, memory operand
        assert_eq!(emulated(&[0x66, 0x0f, 0x78, 0xc8, 0x08, 0x14], &mut c), None); // extrq needs /0
        assert_eq!(emulated(&[0x66, 0x0f, 0xae, 0x38], &mut c), None); // clflushopt
        assert_eq!(emulated(&[0x0f, 0xae, 0x30], &mut c), None); // xsaveopt
        assert_eq!(emulated(&[0x66, 0x0f, 0xae, 0xf0], &mut c), None); // register form
        assert_eq!(c.rip(), 0x1000, "rip unchanged");
    }

    #[test]
    fn clwb_lengths() {
        let len = |b: &[u8]| clwb_len(b);
        assert_eq!(len(&[0x66, 0x41, 0x0f, 0xae, 0x32]), Some(5)); // [r10]
        assert_eq!(len(&[0x66, 0x0f, 0xae, 0x70, 0x40]), Some(5)); // [rax+40h]
        assert_eq!(len(&[0x66, 0x0f, 0xae, 0xb0, 0, 1, 0, 0]), Some(8)); // [rax+disp32]
        assert_eq!(len(&[0x66, 0x0f, 0xae, 0x34, 0x24]), Some(5)); // [rsp]
        assert_eq!(len(&[0x66, 0x0f, 0xae, 0x35, 0, 0, 0, 0]), Some(8)); // [rip+disp32]
    }

    #[test]
    fn loops_that_are_not_line_loops() {
        // Another stride, a jump elsewhere, a cut-off loop, a clzero loop on a register other than rax.
        assert!(line_loop(&[0x66, 0x0f, 0xae, 0x32, 0x48, 0x83, 0xc2, 0x20, 0x48, 0x39, 0xda, 0x76, 0xf3]).is_none());
        assert!(line_loop(&[0x66, 0x0f, 0xae, 0x32, 0x48, 0x83, 0xc2, 0x40, 0x48, 0x39, 0xda, 0x76, 0xf0]).is_none());
        assert!(line_loop(&[0x66, 0x0f, 0xae, 0x32, 0x48, 0x83, 0xc2, 0x40]).is_none());
        assert!(line_loop(&[0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc2, 0x40, 0x48, 0x39, 0xda, 0x72, 0xf4]).is_none());
        // clwb [r10] loop with REX.
        let l = line_loop(&[0x66, 0x41, 0x0f, 0xae, 0x32, 0x49, 0x83, 0xc2, 0x40, 0x49, 0x39, 0xca, 0x76, 0xf2]).unwrap();
        assert_eq!((l.base, l.limit, l.or_equal, l.len), (10, 1, true, 14));
    }

    /// The handler's histogram finds an exception raised N times at
    /// one address.
    #[test]
    fn the_histogram_counts_a_site() {
        if native("mwaitx") {
            return;
        }
        crate::crash::install();
        const N: u32 = 200;
        // mov ecx, N; L: mwaitx; dec ecx; jnz L; ret
        let mut code = vec![0xb9];
        code.extend(N.to_le_bytes());
        code.extend([0x0f, 0x01, 0xfb, 0xff, 0xc9, 0x75, 0xf9, 0xc3]);
        let count = |at: usize| crate::crash::exception_rows(false).iter().filter(|r| r.0 == 0xC000_001D && r.1 == at).map(|r| r.3).sum::<u64>();
        // SAFETY: a fresh page of ours, filled then made executable; the
        // code touches only rcx (volatile).
        unsafe {
            let page = VirtualAlloc(std::ptr::null(), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE).cast::<u8>();
            assert!(!page.is_null());
            std::ptr::copy_nonoverlapping(code.as_ptr(), page, code.len());
            let mut old = 0;
            assert!(VirtualProtect(page.cast(), 4096, PAGE_EXECUTE_READ, &mut old) != 0);
            FlushInstructionCache(GetCurrentProcess(), page.cast(), 4096);
            let f: unsafe extern "system" fn() = std::mem::transmute(page);
            f();
            assert_eq!(count(page as usize + 5), u64::from(N), "one histogram row, N exceptions");
            VirtualFree(page.cast(), 0, MEM_RELEASE);
        }
    }

    #[test]
    fn zen2_nop_forms() {
        let f = crate::rewrite::nop_form;
        assert_eq!(f(&[0x66, 0x41, 0x0f, 0xae, 0x32]), Some((2, [0x0f, 0x1f, 0x02]))); // clwb [r10] -> nop [r10]
        assert_eq!(f(&[0x66, 0x0f, 0xae, 0x70, 0x40]), Some((1, [0x0f, 0x1f, 0x40]))); // clwb [rax+40h]
        assert_eq!(f(&[0x0f, 0x01, 0xfa]), Some((0, [0x0f, 0x1f, 0xc0])));
        assert_eq!(f(&[0x0f, 0x01, 0xfc]), None, "clzero writes memory");
        assert_eq!(f(&[0xf2, 0x0f, 0x78, 0xd1, 0x0c, 0x04]), None, "insertq");
    }

    // The load-time rewrites ([`crate::rewrite`]), run from the harness.

    /// Runs `body` with the site at `body[off..]` rewritten by
    /// [`crate::rewrite::write`] (stub arena in the same page), on `state`.
    fn run_rewritten(body: &[u8], off: usize, site: &crate::rewrite::Site, state: &mut State) -> crate::rewrite::Plan {
        crate::crash::install();
        let code = harness(body);
        let (a, b) = (harness(&[0x90]), harness(&[0x90, 0x90]));
        let at = a.iter().zip(&b).position(|(x, y)| x != y).unwrap() - 1;
        // SAFETY: a fresh page of ours: code, rewritten, made executable,
        // run; the code follows the Win64 ABI.
        unsafe {
            let page = VirtualAlloc(std::ptr::null(), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE).cast::<u8>();
            assert!(!page.is_null() && code.len() < 0x800);
            std::ptr::copy_nonoverlapping(code.as_ptr(), page, code.len());
            let plan = crate::rewrite::write(site, page as usize + at + off, page as usize + 0x800).expect("the rewrite applies");
            let mut old = 0;
            assert!(VirtualProtect(page.cast(), 4096, PAGE_EXECUTE_READ, &mut old) != 0);
            FlushInstructionCache(GetCurrentProcess(), page.cast(), 4096);
            let f: unsafe extern "system" fn(*mut State) = std::mem::transmute(page);
            f(state);
            VirtualFree(page.cast(), 0, MEM_RELEASE);
            plan
        }
    }

    fn site(kind: crate::rewrite::Kind, len: u8, code: &[u8]) -> crate::rewrite::Site {
        crate::rewrite::Site { rva: 0, kind, len, window: code.len() as u8, targets: 0 }
    }

    /// xorshift64*.
    fn rng(seed: &mut u64) -> u64 {
        *seed ^= *seed >> 12;
        *seed ^= *seed << 25;
        *seed ^= *seed >> 27;
        seed.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    #[test]
    fn rewritten_insertq_matches_the_model_over_random_operands() {
        use crate::rewrite::{Kind, Plan, sse};
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for case in 0..300 {
            let r = rng(&mut seed);
            // Registers 2..15 (xmm0/xmm1 carry the operands in and out),
            // REX.R and REX.B forms both; dst == src now and then.
            let d = 2 + (r % 14) as u8;
            let s = if case % 7 == 0 { d } else { 2 + ((r >> 8) % 14) as u8 };
            let (n, idx) = (((r >> 16) % 64) as u8, ((r >> 24) % 64) as u8);
            let (n, idx) = if n as u32 + idx as u32 > 64 { (n, 64 - n) } else { (n, idx) };
            // cmp al, al: known flags, which the stub must keep.
            let mut body = vec![0x38, 0xc0, 0x48, 0x8d, 0x64, 0x24, 0xe0]; // lea rsp, [rsp-32]
            for (i, x) in [d, s].into_iter().enumerate() {
                sse(&mut body, 0xf3, false, 0x7f, x, None, (0x44, &[0x24, 16 * i as u8])); // movdqu [rsp+16i], xmm(x)
            }
            sse(&mut body, 0x66, false, 0x6f, d, Some(0), (0, &[])); // movdqa xmm(d), xmm0
            sse(&mut body, 0x66, false, 0x6f, s, Some(1), (0, &[])); // movdqa xmm(s), xmm1
            let off = body.len();
            let rex = 0x40 | (d >> 3) << 2 | (s >> 3);
            let mut ins = vec![0xf2];
            if rex != 0x40 {
                ins.push(rex);
            }
            ins.extend([0x0f, 0x78, 0xc0 | (d & 7) << 3 | (s & 7), n, idx]);
            body.extend(&ins);
            sse(&mut body, 0x66, false, 0x6f, 0, Some(d), (0, &[])); // movdqa xmm0, xmm(d)
            for (i, x) in [s, d].into_iter().enumerate() {
                sse(&mut body, 0xf3, false, 0x6f, x, None, (0x44, &[0x24, 16 * (1 - i) as u8])); // movdqu xmm(x), [rsp+16(1-i)]
            }
            body.extend([0x48, 0x8d, 0x64, 0x24, 0x20]); // lea rsp, [rsp+32]
            let mut state = State::new();
            state.xmm[0] = [rng(&mut seed), rng(&mut seed)];
            state.xmm[1] = [rng(&mut seed), rng(&mut seed)];
            let mut want = state;
            let dst = if d == s { state.xmm[1] } else { state.xmm[0] };
            want.xmm[0] = [model_insertq(dst[0], state.xmm[1][0], n, idx), dst[1]];
            let mut reference = body.clone();
            reference[off..off + ins.len()].fill(0x90);
            let mut flags = state;
            run(&reference, &mut flags);
            want.flags = flags.flags;
            let st = site(Kind::Insertq, ins.len() as u8, &ins);
            assert_eq!(run_rewritten(&body, off, &st, &mut state), Plan::Stub);
            state.saved = want.saved;
            assert_eq!(state, want, "insertq xmm{d}, xmm{s}, {n}, {idx}: {ins:02x?}");
        }
    }

    #[test]
    fn rewritten_clzero_zeroes_the_line_and_runs_what_it_moved() {
        use crate::rewrite::{Kind, Plan};
        #[repr(C, align(64))]
        struct Lines([u8; 256]);
        let mut seed = 0x1234_5678u64;
        // clzero; add rdx, 40h and
        // clzero; cmp r11, r10: the moved instruction's
        // flags must survive the stub.
        let forms: [(&[u8], &[u8]); 2] = [(&[0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc2, 0x40], &[0x0f, 0x1f, 0x00, 0x48, 0x83, 0xc2, 0x40]), (&[0x0f, 0x01, 0xfc, 0x4d, 0x39, 0xd3], &[0x0f, 0x1f, 0x00, 0x4d, 0x39, 0xd3])];
        for (body, reference) in forms {
            for _ in 0..20 {
                let mut lines = Lines([0xaa; 256]);
                let line = 1 + (rng(&mut seed) % 2) as usize;
                let mut state = State::new();
                state.gpr[0] = lines.0.as_mut_ptr() as u64 + 64 * line as u64 + rng(&mut seed) % 64;
                state.gpr[6] = rng(&mut seed); // r10
                state.gpr[7] = rng(&mut seed); // r11
                let mut want = state;
                run(reference, &mut want); // the moved instruction natively, a nop for clzero
                let st = site(Kind::Clzero, 3, body);
                assert_eq!(run_rewritten(body, 0, &st, &mut state), Plan::Stub);
                state.saved = want.saved;
                assert_eq!(state, want, "{body:02x?}");
                for (i, &b) in lines.0.iter().enumerate() {
                    assert_eq!(b, if i / 64 == line { 0 } else { 0xaa }, "byte {i}, line {line}");
                }
            }
        }
    }

    #[test]
    fn rewritten_hints_run_in_place_without_a_fault() {
        use crate::rewrite::{Kind, Plan};
        // clwb [r10] / clwb [rbx] / monitorx / mwaitx.
        let mut buf = [0u8; 64];
        for (kind, code) in [(Kind::Clwb, &[0x66, 0x41, 0x0f, 0xae, 0x32][..]), (Kind::Clwb, &[0x66, 0x0f, 0xae, 0x33]), (Kind::Monitorx, &[0x0f, 0x01, 0xfa]), (Kind::Mwaitx, &[0x0f, 0x01, 0xfb])] {
            let mut state = State::new();
            state.gpr[0] = buf.as_mut_ptr() as u64;
            (state.gpr[3], state.gpr[6]) = (buf.as_mut_ptr() as u64, buf.as_mut_ptr() as u64); // rbx, r10
            let mut want = state;
            // cmp al, al: known flags, which the rewrite must keep.
            let mut reference = vec![0x38, 0xc0];
            reference.extend(vec![0x90; code.len()]);
            run(&reference, &mut want);
            let mut body = vec![0x38, 0xc0];
            body.extend(code);
            let plan = run_rewritten(&body, 2, &site(kind, code.len() as u8, code), &mut state);
            assert!(matches!(plan, Plan::InPlace(_)), "{kind:?}");
            if let (Kind::Mwaitx, Plan::InPlace(b)) = (kind, &plan) {
                assert_eq!(b[..], [0xf3, 0x90, 0x90], "pause, not a bare nop");
            }
            state.saved = want.saved;
            assert_eq!(state, want, "{kind:?}");
        }
    }

    #[test]
    fn the_rewriter_refuses_other_code_and_a_branch_into_the_window() {
        use crate::rewrite::{Kind, Site, plan};
        let code = [0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc2, 0x40];
        let s = site(Kind::Clzero, 3, &code);
        assert_eq!(plan(&s, &[0x0f, 0x01, 0xfb, 0x48, 0x83, 0xc2, 0x40]), Err("not the instruction the table names"));
        assert_eq!(plan(&s, &[0x0f, 0x01, 0xfc]), Err("the window runs past the code"));
        let branch = Site { targets: 1 << 3, ..s };
        assert_eq!(plan(&branch, &code), Err("a branch target inside the window"));
        // A target at the window's end (the next instruction) is fine.
        let after = Site { targets: 1 << 7, ..site(Kind::Clzero, 3, &code) };
        assert!(plan(&after, &code).is_ok());
        let short = site(Kind::Clzero, 3, &[0x0f, 0x01, 0xfc]);
        assert_eq!(plan(&short, &code), Err("fewer than 5 bytes"));
        // clflushopt [rbx] (/7), not clwb (/6).
        let clwb = site(Kind::Clwb, 4, &[0x66, 0x0f, 0xae, 0x33]);
        assert_eq!(plan(&clwb, &[0x66, 0x0f, 0xae, 0x3b]), Err("not the instruction the table names"));
    }
}
