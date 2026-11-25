//! Generated code in front of imports.
//!
//! Every import the loader binds can be given a 24-byte thunk that loads its
//! [`Site`] into `r10` and jumps to one of two shared trampolines:
//!
//! * *unimplemented*: nobody provides the import. The handler names it, with
//!   the caller and a stack, then ends the run (`Trap`) or returns 0 (`Zero`).
//! * *trace*: the import is provided; the call is logged with its integer
//!   arguments and then continues to the real target with every argument
//!   register intact.

use std::arch::global_asm;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use xnw_core::win::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Unimplemented {
    /// Report the call and end the process.
    Trap = 0,
    /// Report the first call, return 0, keep going.
    Zero = 1,
}

static POLICY: AtomicU8 = AtomicU8::new(0);

pub fn set_policy(p: Unimplemented) {
    POLICY.store(p as u8, Ordering::Relaxed);
}

/// What a thunk knows about its import. Leaked: thunks live as long as the
/// process. `target` must stay the first field (the trace trampoline jumps
/// through it).
#[repr(C)]
pub struct Site {
    pub target: usize,
    pub dll: String,
    pub symbol: String,
    pub calls: AtomicU64,
}

global_asm!(
    r#"
    .globl xnw_unimplemented_entry
    .def xnw_unimplemented_entry; .scl 2; .type 32; .endef
    .seh_proc xnw_unimplemented_entry
xnw_unimplemented_entry:
    sub rsp, 0x48
    .seh_stackalloc 0x48
    .seh_endprologue
    mov [rsp + 0x20], rcx
    mov [rsp + 0x28], rdx
    mov [rsp + 0x30], r8
    mov [rsp + 0x38], r9
    mov rcx, r10
    lea rdx, [rsp + 0x20]
    mov r8, [rsp + 0x48]
    call xnw_unimplemented
    add rsp, 0x48
    ret
    .seh_endproc

    .globl xnw_trace_entry
    .def xnw_trace_entry; .scl 2; .type 32; .endef
    .seh_proc xnw_trace_entry
xnw_trace_entry:
    sub rsp, 0x88
    .seh_stackalloc 0x88
    .seh_endprologue
    mov [rsp + 0x20], rcx
    mov [rsp + 0x28], rdx
    mov [rsp + 0x30], r8
    mov [rsp + 0x38], r9
    movups [rsp + 0x40], xmm0
    movups [rsp + 0x50], xmm1
    movups [rsp + 0x60], xmm2
    movups [rsp + 0x70], xmm3
    mov [rsp + 0x80], r10
    mov rcx, r10
    lea rdx, [rsp + 0x20]
    mov r8, [rsp + 0x88]
    call xnw_trace
    mov r10, [rsp + 0x80]
    mov rcx, [rsp + 0x20]
    mov rdx, [rsp + 0x28]
    mov r8, [rsp + 0x30]
    mov r9, [rsp + 0x38]
    movups xmm0, [rsp + 0x40]
    movups xmm1, [rsp + 0x50]
    movups xmm2, [rsp + 0x60]
    movups xmm3, [rsp + 0x70]
    add rsp, 0x88
    jmp qword ptr [r10]
    .seh_endproc
"#
);

unsafe extern "C" {
    fn xnw_unimplemented_entry();
    fn xnw_trace_entry();
}

#[unsafe(no_mangle)]
extern "system" fn xnw_unimplemented(site: &Site, args: &[u64; 4], ret: usize) -> u64 {
    let n = site.calls.fetch_add(1, Ordering::Relaxed);
    let trap = POLICY.load(Ordering::Relaxed) == Unimplemented::Trap as u8;
    if n == 0 || trap {
        xnw_core::warn!(
            "UNIMPLEMENTED {}!{} called from {}  args {:#x} {:#x} {:#x} {:#x}",
            site.dll,
            site.symbol,
            crate::crash::locate(ret),
            args[0],
            args[1],
            args[2],
            args[3]
        );
    }
    if trap {
        crate::crash::log_stack_here();
        xnw_core::warn!("run ended at the first unimplemented import (policy: trap)");
        // SAFETY: ends our own process; nothing is left to unwind.
        unsafe { TerminateProcess(GetCurrentProcess(), 70) };
    }
    0
}

#[unsafe(no_mangle)]
extern "system" fn xnw_trace(site: &Site, args: &[u64; 4], ret: usize) {
    site.calls.fetch_add(1, Ordering::Relaxed);
    xnw_core::trace!(
        "{}!{}({:#x}, {:#x}, {:#x}, {:#x}) from {}",
        site.dll,
        site.symbol,
        args[0],
        args[1],
        args[2],
        args[3],
        crate::crash::locate(ret)
    );
}

const THUNK: usize = 24;

/// A block of executable memory holding thunks. Leaked on purpose.
pub struct Thunks {
    base: *mut u8,
    used: usize,
    capacity: usize,
    sites: Vec<&'static Site>,
}

impl Thunks {
    pub fn new(count: usize) -> std::io::Result<Self> {
        let capacity = (count.max(1) * THUNK).next_multiple_of(4096);
        // SAFETY: fresh memory of our own.
        let base = unsafe { VirtualAlloc(std::ptr::null(), capacity, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) };
        if base.is_null() {
            return Err(last_error());
        }
        Ok(Thunks { base: base.cast(), used: 0, capacity, sites: Vec::new() })
    }

    fn emit(&mut self, site: &'static Site, entry: usize) -> usize {
        let at = self.emit_raw(std::ptr::from_ref(site) as usize, entry);
        self.sites.push(site);
        at
    }

    /// A thunk that records the call and its result (`record.rs`).
    pub fn recorded(&mut self, dll: &str, symbol: &str, target: usize, hle: bool) -> usize {
        let site = crate::record::site(dll, symbol, target, hle);
        self.emit_raw(std::ptr::from_ref(site) as usize, crate::record::entry())
    }

    fn emit_raw(&mut self, site: usize, entry: usize) -> usize {
        assert!(self.used + THUNK <= self.capacity, "thunk block is full");
        let mut code = [0xCCu8; THUNK];
        code[0..2].copy_from_slice(&[0x49, 0xBA]); // mov r10, imm64
        code[2..10].copy_from_slice(&(site as u64).to_le_bytes());
        code[10..12].copy_from_slice(&[0x48, 0xB8]); // mov rax, imm64
        code[12..20].copy_from_slice(&(entry as u64).to_le_bytes());
        code[20..22].copy_from_slice(&[0xFF, 0xE0]); // jmp rax
        // SAFETY: inside our writable block, checked above.
        let at = unsafe { self.base.add(self.used) };
        // SAFETY: as above.
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), at, THUNK) };
        self.used += THUNK;
        at as usize
    }

    fn site(dll: &str, symbol: &str, target: usize) -> &'static Site {
        Box::leak(Box::new(Site { target, dll: dll.to_owned(), symbol: symbol.to_owned(), calls: AtomicU64::new(0) }))
    }

    /// A thunk for an import nobody provides.
    pub fn unimplemented(&mut self, dll: &str, symbol: &str) -> usize {
        self.emit(Self::site(dll, symbol, 0), xnw_unimplemented_entry as usize)
    }

    /// A thunk that logs the call and continues at `target`.
    pub fn traced(&mut self, dll: &str, symbol: &str, target: usize) -> usize {
        self.emit(Self::site(dll, symbol, target), xnw_trace_entry as usize)
    }

    /// Makes the block executable and returns its sites (for the run report).
    pub fn seal(self) -> std::io::Result<Vec<&'static Site>> {
        let mut old = 0;
        // SAFETY: our own block.
        if unsafe { VirtualProtect(self.base as _, self.capacity, PAGE_EXECUTE_READ, &mut old) } == 0 {
            return Err(last_error());
        }
        Ok(self.sites)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "system" fn add(a: u64, b: u64, c: u64, d: u64) -> u64 {
        a + b * 10 + c * 100 + d * 1000
    }

    #[test]
    fn traced_call_keeps_arguments_and_result() {
        let mut t = Thunks::new(2).unwrap();
        let thunk = t.traced("test.dll", "add", add as usize);
        let sites = t.seal().unwrap();
        // SAFETY: the thunk forwards to `add` with the same signature.
        let f: extern "system" fn(u64, u64, u64, u64) -> u64 = unsafe { std::mem::transmute(thunk) };
        assert_eq!(f(1, 2, 3, 4), 4321);
        assert_eq!(sites[0].calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn zero_policy_returns_zero_and_counts() {
        set_policy(Unimplemented::Zero);
        let mut t = Thunks::new(1).unwrap();
        let thunk = t.unimplemented("console.dll", "Missing");
        let sites = t.seal().unwrap();
        // SAFETY: the thunk returns a u64 whatever it is given.
        let f: extern "system" fn(u64) -> u64 = unsafe { std::mem::transmute(thunk) };
        assert_eq!(f(7), 0);
        assert_eq!(f(7), 0);
        assert_eq!(sites[0].calls.load(Ordering::Relaxed), 2);
    }
}
