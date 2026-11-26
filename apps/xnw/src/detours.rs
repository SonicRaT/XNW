//! Function detours on the title: a 14-byte absolute jump at a
//! function's entry to a handler of ours, which calls the original through
//! a trampoline (the entry's whole instructions moved there, then a jump
//! back). Like the patches (`patches.rs`), each entry is for one build (PE
//! timestamp) and checks the bytes it replaces; applied after mapping,
//! before any title code runs.
//!
//! `lock`: the title's buffer lock for the CPU, `title+0x524dab0` (fn
//! 0x14524dab0, read from the machine code):
//! `(resource, lock)` -> success; on success the lock record holds
//! the CPU pointer at +0 and the bytes at +0x18 (fn 0x145121690 fills
//! them). Every CPU read of a GPU-written buffer goes through it (21
//! callers). With `XNW_ROLES=lock` the GPU's results for the locked range
//! go back to title memory before the caller reads it.
//!
//! `stages`: the runner of the title's render stage callbacks,
//! `title+0x43c3890` (`stagemeter.rs`: `XNW_STAGE_TIME`, `XNW_STAGE_SKIP`).
//!
//! `clock`: the game clock's tick `title+0x15120b0` (`levers.rs`: `XNW_CLOCK`).
//! `cine`: the cinematic instance tick `title+0x50f89c0` (`levers.rs`: `XNW_CINE`).
//! `fencewait`: the title's wait for a fence token `title+0x4fb0ad0`
//! (`fencewait.rs`: `XNW_FENCE_WAIT`).

use std::sync::atomic::{AtomicUsize, Ordering};

use xnw_core::win::{FlushInstructionCache, GetCurrentProcess, MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, RtlCaptureStackBackTrace, VirtualAlloc, VirtualProtect};

/// The title's executable 1.0.2.1 (PE header timestamp).
const BUILD_1_0_2_1: u32 = 0x6AF7_83D4;

struct Detour {
    name: &'static str,
    rva: usize,
    /// The entry's first whole instructions, at least 14 bytes, with no
    /// position-dependent operand: they run again from the trampoline.
    expect: &'static [u8],
    hook: usize,
    original: &'static AtomicUsize,
}

static LOCK_ORIGINAL: AtomicUsize = AtomicUsize::new(0);

/// The lock detour: the original, then the GPU's results for what it
/// locked.
extern "system" fn lock_hook(resource: usize, lock: usize) -> bool {
    // SAFETY: the trampoline made at install, with the original's type.
    let original: extern "system" fn(usize, usize) -> bool = unsafe { std::mem::transmute(LOCK_ORIGINAL.load(Ordering::Relaxed)) };
    let ok = original(resource, lock);
    if ok && lock != 0 {
        // SAFETY: the lock record the original filled (pointer, bytes).
        let (address, len) = unsafe { ((lock as *const usize).read(), ((lock + 0x18) as *const u32).read() as usize) };
        let mut site: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: one frame above this one: the title's call site.
        unsafe { RtlCaptureStackBackTrace(1, 1, &mut site, std::ptr::null_mut()) };
        if address != 0 {
            xnw_gpu::d3d12x::buffer_locked(address, len, site as usize);
        }
    }
    ok
}

/// The detours for this build.
fn detours() -> [Detour; 5] {
    [
        Detour {
            name: "lock",
            rva: 0x524dab0,
            // push r15; push r14; push rsi; push rdi; push rbp; push rbx;
            // sub rsp, 28h; mov rsi, rdx
            expect: &[0x41, 0x57, 0x41, 0x56, 0x56, 0x57, 0x55, 0x53, 0x48, 0x83, 0xEC, 0x28, 0x48, 0x89, 0xD6],
            hook: lock_hook as usize,
            original: &LOCK_ORIGINAL,
        },
        Detour {
            name: "stages",
            rva: crate::stagemeter::ENTRY_RVA,
            // mov edx, [rcx+18h]; mov rax, [rcx+8]; mov rcx, [rcx+10h]; jmp rax
            // (the whole function; its trampoline never comes back).
            expect: &crate::stagemeter::ENTRY_BYTES,
            hook: crate::stagemeter::hook as usize,
            original: &crate::stagemeter::ORIGINAL,
        },
        Detour {
            name: "clock",
            rva: crate::levers::CLOCK_TICK_RVA,
            // push rsi; push rdi; push rbp; push rbx; sub rsp, 58h;
            // vmovaps [rsp+40h], xmm7
            expect: &crate::levers::CLOCK_TICK_BYTES,
            hook: crate::levers::clock_hook as usize,
            original: &crate::levers::CLOCK_ORIGINAL,
        },
        Detour {
            name: "cine",
            rva: crate::levers::CINE_TICK_RVA,
            // push r15; push r14; push rsi; push rdi; push rbx; sub rsp, 20h;
            // mov rsi, rcx
            expect: &crate::levers::CINE_TICK_BYTES,
            hook: crate::levers::cine_hook as usize,
            original: &crate::levers::CINE_ORIGINAL,
        },
        Detour {
            name: "fencewait",
            rva: crate::fencewait::WAIT_RVA,
            // push r15; push r14; push r13; push r12; push rsi; push rdi;
            // push rbp; push rbx; sub rsp, 58h
            expect: &crate::fencewait::WAIT_BYTES,
            hook: crate::fencewait::hook as usize,
            original: &crate::fencewait::ORIGINAL,
        },
    ]
}

/// `jmp qword [rip+0]` followed by the target.
fn jump(to: usize) -> [u8; 14] {
    let mut j = [0xFF, 0x25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    j[6..].copy_from_slice(&(to as u64).to_le_bytes());
    j
}

/// Installs the detours wanted (`XNW_ROLES=lock`: `lock`; `XNW_STAGE_TIME`,
/// `XNW_STAGE_SKIP`, `XNW_AB` or `XNW_STOP_WORLD_FRAMES`: `stages`; `XNW_CLOCK`: `clock`; `XNW_CINE`: `cine`;
/// `XNW_FENCE_WAIT`: `fencewait`) on the title at `base`.
///
/// # Safety
/// `base` is the mapped title and none of its code runs yet.
pub unsafe fn install(base: usize) {
    // SAFETY: forwarded.
    let stamp = unsafe { crate::patches::timestamp(base) };
    let wanted = [
        ("lock", xnw_gpu::d3d12x::roles_lock()),
        // Both read (no short cut): the meter's switches, then the A/B and stop ones that count its frames.
        ("stages", crate::stagemeter::configure(base) | (stamp == BUILD_1_0_2_1 && crate::ab::configure(base))),
        // SAFETY: the mapped title of this build (its code is read only then).
        ("clock", stamp == BUILD_1_0_2_1 && unsafe { crate::levers::clock_configure(base) }),
        // SAFETY: as above.
        ("cine", stamp == BUILD_1_0_2_1 && unsafe { crate::levers::cine_configure(base) }),
        // SAFETY: as above.
        ("fencewait", stamp == BUILD_1_0_2_1 && unsafe { crate::fencewait::configure(base) }),
    ];
    if wanted.iter().all(|w| !w.1) {
        return;
    }
    if stamp != BUILD_1_0_2_1 {
        xnw_core::warn!("detours: title timestamp {stamp:#x} is not build 1.0.2.1; none installed");
        return;
    }
    for d in detours().into_iter().filter(|d| wanted.iter().any(|w| w.0 == d.name && w.1)) {
        // SAFETY: inside the mapped .text of this build, no code running.
        match unsafe { detour((base + d.rva) as *mut u8, d.expect, d.hook) } {
            Ok(tramp) => {
                d.original.store(tramp, Ordering::Relaxed);
                xnw_core::info!("detour {} installed at title+{:#x}", d.name, d.rva);
            }
            Err(why) => xnw_core::warn!("detour {} at title+{:#x} skipped: {why}", d.name, d.rva),
        }
    }
}

/// Puts a jump to `hook` at `at`, whose first bytes must be `expect`
/// (whole instructions, at least 14 bytes, position independent), and
/// returns the trampoline that runs them and goes on after them.
///
/// # Safety
/// `at` is code with at least `expect.len()` bytes that nothing runs now.
pub(crate) unsafe fn detour(at: *mut u8, expect: &[u8], hook: usize) -> Result<usize, &'static str> {
    // SAFETY: the caller's code bytes.
    let now = unsafe { std::slice::from_raw_parts(at, expect.len()) };
    if now != expect || expect.len() < 14 {
        return Err("the bytes are not the expected ones");
    }
    // The trampoline: the moved instructions, then back after them.
    // SAFETY: a fresh executable allocation of ours.
    let tramp = unsafe { VirtualAlloc(std::ptr::null(), 64, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) }.cast::<u8>();
    if tramp.is_null() {
        return Err("no trampoline memory");
    }
    let back = jump(at as usize + expect.len());
    // SAFETY: 64 bytes of ours hold the moved bytes and a jump.
    unsafe {
        std::ptr::copy_nonoverlapping(expect.as_ptr(), tramp, expect.len());
        std::ptr::copy_nonoverlapping(back.as_ptr(), tramp.add(expect.len()), back.len());
        FlushInstructionCache(GetCurrentProcess(), tramp.cast(), 64);
    }
    let mut entry = expect.to_vec();
    entry[..14].copy_from_slice(&jump(hook));
    // The rest of the moved instructions' bytes: int3, never run.
    entry[14..].fill(0xCC);
    let mut old = 0;
    // SAFETY: the checked bytes only; protection restored below.
    if unsafe { VirtualProtect(at.cast(), entry.len(), PAGE_EXECUTE_READWRITE, &mut old) } == 0 {
        return Err("VirtualProtect failed");
    }
    // SAFETY: writable now; nothing runs this code now.
    unsafe {
        std::ptr::copy_nonoverlapping(entry.as_ptr(), at, entry.len());
        VirtualProtect(at.cast(), entry.len(), old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), at.cast(), entry.len());
    }
    Ok(tramp as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    static TRAMP: AtomicUsize = AtomicUsize::new(0);

    extern "system" fn plus_one(a: usize, b: usize) -> usize {
        // SAFETY: the trampoline of the test function, of this type.
        let original: extern "system" fn(usize, usize) -> usize = unsafe { std::mem::transmute(TRAMP.load(Ordering::Relaxed)) };
        original(a, b) + 1
    }

    /// A function with the lock's prologue, made in memory: `a + b`. Its
    /// detour runs the hook, which runs the original through the
    /// trampoline.
    #[test]
    fn a_detour_runs_the_hook_then_the_original() {
        let lock_prologue = detours()[0].expect;
        // prologue; lea rax, [rcx + rdx]... as: mov rax, rcx; add rax, rsi
        // (rsi = rdx from the prologue); epilogue.
        let body: [u8; 15] = [
            0x48, 0x89, 0xC8, // mov rax, rcx
            0x48, 0x01, 0xF0, // add rax, rsi
            0x48, 0x83, 0xC4, 0x28, // add rsp, 28h
            0x5B, 0x5D, 0x5F, 0x5E, // pop rbx; pop rbp; pop rdi; pop rsi
            0x90,
        ];
        let tail: [u8; 5] = [0x41, 0x5E, 0x41, 0x5F, 0xC3]; // pop r14; pop r15; ret
        // SAFETY: a fresh executable allocation of ours.
        let code = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) }.cast::<u8>();
        assert!(!code.is_null());
        // SAFETY: inside the allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(lock_prologue.as_ptr(), code, lock_prologue.len());
            std::ptr::copy_nonoverlapping(body.as_ptr(), code.add(lock_prologue.len()), body.len());
            std::ptr::copy_nonoverlapping(tail.as_ptr(), code.add(lock_prologue.len() + body.len()), tail.len());
        }
        // SAFETY: the function just written, of this type.
        let f: extern "system" fn(usize, usize) -> usize = unsafe { std::mem::transmute(code) };
        assert_eq!(f(2, 3), 5);
        // SAFETY: our code, nothing runs it now.
        let tramp = unsafe { detour(code, lock_prologue, plus_one as usize) }.unwrap();
        TRAMP.store(tramp, Ordering::Relaxed);
        assert_eq!(f(2, 3), 6, "the hook ran and called the original");
        // SAFETY: as above: the bytes are no longer the expected ones.
        assert!(unsafe { detour(code, lock_prologue, plus_one as usize) }.is_err());
    }
}
