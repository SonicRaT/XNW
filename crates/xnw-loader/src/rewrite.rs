//! The console's Zen 2 instructions that the host CPU lacks are
//! rewritten in the title image at load, before any title code runs, so
//! they no longer cost an exception each ([`crate::amd`] emulates them in
//! the fault handler, which stays as the fallback).
//!
//! The sites come from an offline scan of every code head
//! ([`rewrite_table`], a generated table): 19 `insertq`, 10
//! `clzero`, 8 `clwb`, 2 `monitorx`, 2 `mwaitx`; no `extrq`, `movntss`/`sd`,
//! SHA, `rdpid` or `rdpru`. The table holds no code bytes: it applies only
//! to the executable whose SHA-256 it names (any other file: no rewrite,
//! one log line naming both hashes). Each site is gated on this CPU's CPUID
//! bit for its instruction (a Zen host gets no rewrite) and on a decode of
//! the bytes there as the row's instruction, and of any moved ones as
//! register-only instructions.
//!
//! - In place: `clwb m8` becomes `nop m8` (same bytes but the opcode and
//!   ModRM reg; the PC's memory is coherent, a write-back hint does
//!   nothing). `monitorx` becomes `nop eax` (`0f 1f c0`). `mwaitx` becomes
//!   `pause; nop` (`f3 90 90`). Its only user, the title's fence wait,
//!   re-reads the fence after it and then calls
//!   `Sleep(1)` or `Sleep(0)`, and reads no register or flag that mwaitx sets.
//! - Stubs: `insertq` and `clzero` jump to a stub in an arena within 2 GiB of
//!   the image. The stub does the instruction with SSE2 on the site's own
//!   registers. It uses one scratch xmm (and `rcx` for `clzero`), kept on
//!   the stack below `rsp`, and changes no flags (no ALU instruction). A
//!   `clzero` (3 bytes) takes the following whole position-independent
//!   instructions into its stub (`add rax, 40h` or a `cmp`); the row's
//!   window covers them, and no branch lands inside.
//!
//! `XNW_CPU_REWRITE=auto` (default; each site where the CPU lacks its
//! feature) | `off` | `list` (print the plan and exit). It supersedes the earlier
//! `XNW_PATCH_ZEN2` switch, which is removed.

use xnw_core::win::*;

#[path = "rewrite_table.rs"]
mod rewrite_table;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    Clwb,
    Monitorx,
    Mwaitx,
    Insertq,
    #[allow(dead_code)]
    Extrq,
    Clzero,
    /// No safe rewrite: stays on the exception path.
    #[allow(dead_code)]
    Left,
}

#[derive(Debug)]
pub(crate) struct Site {
    pub rva: u32,
    pub kind: Kind,
    /// The instruction's length.
    pub len: u8,
    /// Bytes the rewrite covers: the instruction, then any whole
    /// instructions moved into its stub.
    pub window: u8,
    /// Bit n: a branch or reference lands n bytes into the window.
    pub targets: u16,
}

/// Whether this CPU runs `kind` itself (CPUID).
pub(crate) fn native(kind: Kind) -> bool {
    use std::arch::x86_64::__cpuid;
    // SAFETY: CPUID is always there on x86-64.
    unsafe {
        let ext = __cpuid(0x8000_0000).eax;
        let e1 = if ext >= 0x8000_0001 { __cpuid(0x8000_0001).ecx } else { 0 };
        match kind {
            Kind::Insertq | Kind::Extrq => e1 & 1 << 6 != 0,
            Kind::Monitorx | Kind::Mwaitx => e1 & 1 << 29 != 0,
            Kind::Clzero => ext >= 0x8000_0008 && __cpuid(0x8000_0008).ebx & 1 != 0,
            Kind::Clwb => __cpuid(7).ebx & 1 << 24 != 0,
            Kind::Left => true,
        }
    }
}

/// What to do at one site.
#[derive(Debug, PartialEq)]
pub(crate) enum Plan {
    /// New bytes over the instruction.
    InPlace(Vec<u8>),
    /// A stub; the window becomes `jmp stub` and NOPs.
    Stub,
}

/// The plan for `site` given the bytes now at it (`code` starts at the
/// site and holds at least the window), or why it is refused.
pub(crate) fn plan(site: &Site, code: &[u8]) -> Result<Plan, &'static str> {
    if matches!(site.kind, Kind::Extrq | Kind::Left) {
        return Err("no rewrite for it");
    }
    let (len, window) = (usize::from(site.len), usize::from(site.window));
    if len == 0 || len > window || window > 16 {
        return Err("a malformed row");
    }
    let code = code.get(..window).ok_or("the window runs past the code")?;
    if !decodes_as(site.kind, &code[..len]) {
        return Err("not the instruction the table names");
    }
    if !moved_ok(&code[len..]) {
        return Err("a moved instruction is not register-only");
    }
    if u32::from(site.targets) & ((1u32 << window) - 1) & !1 != 0 {
        return Err("a branch target inside the window");
    }
    match site.kind {
        Kind::Clwb | Kind::Monitorx | Kind::Mwaitx if window == len => {
            let (off, new) = nop_form(code).ok_or("not the instruction the table names")?;
            let mut out = code.to_vec();
            out[off..off + 3].copy_from_slice(&new);
            Ok(Plan::InPlace(out))
        }
        Kind::Clwb | Kind::Monitorx | Kind::Mwaitx => Err("an in-place rewrite moves nothing"),
        Kind::Insertq | Kind::Clzero if window >= 5 => Ok(Plan::Stub),
        Kind::Insertq | Kind::Clzero => Err("fewer than 5 bytes"),
        Kind::Extrq | Kind::Left => Err("no rewrite for it"),
    }
}

/// Whether `code` is exactly one instruction of `kind`.
pub(crate) fn decodes_as(kind: Kind, code: &[u8]) -> bool {
    match kind {
        Kind::Clwb => crate::amd::clwb_len(code) == Some(code.len()),
        Kind::Monitorx => code == [0x0f, 0x01, 0xfa],
        Kind::Mwaitx => code == [0x0f, 0x01, 0xfb],
        Kind::Clzero => code == [0x0f, 0x01, 0xfc],
        Kind::Insertq => insertq_fields(code).is_some(),
        Kind::Extrq | Kind::Left => false,
    }
}

/// Whether `code` is whole register-only instructions (`[rex] op modrm`
/// with mod 11, then an imm8 for `83`: ALU, `test`, `mov`), which touch
/// no memory and no `rip` and so run the same from a stub.
fn moved_ok(mut code: &[u8]) -> bool {
    while !code.is_empty() {
        let at = usize::from(code[0] & 0xf0 == 0x40);
        let Some(&[op, modrm]) = code.get(at..at + 2) else { return false };
        let imm = match op {
            0x83 => 1,
            0x01 | 0x03 | 0x09 | 0x0b | 0x21 | 0x23 | 0x29 | 0x2b | 0x31 | 0x33 | 0x39 | 0x3b | 0x85 | 0x89 | 0x8b => 0,
            _ => return false,
        };
        let n = at + 2 + imm;
        if modrm >> 6 != 3 || code.len() < n {
            return false;
        }
        code = &code[n..];
    }
    true
}

/// The in-place replacement for a hint at the start of `code`: (offset in
/// it, 3 bytes), or `None`.
pub(crate) fn nop_form(code: &[u8]) -> Option<(usize, [u8; 3])> {
    match code.get(..3)? {
        [0x0f, 0x01, 0xfa] => return Some((0, [0x0f, 0x1f, 0xc0])),
        [0x0f, 0x01, 0xfb] => return Some((0, [0xf3, 0x90, 0x90])),
        _ => {}
    }
    crate::amd::clwb_len(code)?;
    let at = if code[1] & 0xf0 == 0x40 { 2 } else { 1 };
    Some((at, [0x0f, 0x1f, code[at + 2] & !0x38]))
}

/// `insertq xmm(dst), xmm(src), len, idx` (`f2 [rex] 0f 78 /r ib ib`,
/// register operands): (dst, src, len, idx).
fn insertq_fields(code: &[u8]) -> Option<(u8, u8, u8, u8)> {
    let (rex, at) = if code.get(1)? & 0xf0 == 0x40 { (code[1], 2) } else { (0, 1) };
    if code[0] != 0xf2 || code.get(at..at + 2)? != [0x0f, 0x78] {
        return None;
    }
    let modrm = *code.get(at + 2)?;
    if modrm >> 6 != 3 || code.len() != at + 5 {
        return None;
    }
    Some(((modrm >> 3) & 7 | (rex & 4) << 1, modrm & 7 | (rex & 1) << 3, code[at + 3] & 63, code[at + 4] & 63))
}

/// A legacy-SSE instruction `pfx [rex] 0f op modrm ...` with `reg` and an
/// r/m register (mod 11) or a memory tail (`mem`: mod/rm bits and what
/// follows, rm < 8).
pub(crate) fn sse(c: &mut Vec<u8>, pfx: u8, w: bool, op: u8, reg: u8, rm: Option<u8>, mem: (u8, &[u8])) {
    c.push(pfx);
    let rm_hi = rm.map_or(0, |r| r >> 3);
    let rex = 0x40 | u8::from(w) << 3 | (reg >> 3) << 2 | rm_hi;
    if rex != 0x40 {
        c.push(rex);
    }
    c.extend([0x0f, op]);
    match rm {
        Some(r) => c.push(0xc0 | (reg & 7) << 3 | (r & 7)),
        None => {
            c.push(mem.0 | (reg & 7) << 3);
            c.extend(mem.1);
        }
    }
}

/// `[rip + disp32]` to `target` from an instruction whose disp32 ends at
/// `c.len() + 4 + after` (bytes after the displacement).
fn rip_to(c: &[u8], at: usize, target: usize, after: usize) -> [u8; 4] {
    let end = at + c.len() + 4 + after;
    ((target as i64 - end as i64) as i32).to_le_bytes()
}

/// The stub for `site` (its window's bytes in `code`, checked by [`plan`])
/// at `at`, with its 16-byte constants at `consts` (written by the caller
/// from the returned array), ending in a jump to `back`.
pub(crate) fn stub(site: &Site, code: &[u8], at: usize, consts: usize, back: usize) -> (Vec<u8>, [[u64; 2]; 2]) {
    let len = usize::from(site.len);
    let mut c = Vec::new();
    let mut k = [[0u64; 2]; 2];
    // movdqu [rsp+d8], xmm(t) / movdqu xmm(t), [rsp+d8]
    let save = |c: &mut Vec<u8>, t: u8, d: u8| sse(c, 0xf3, false, 0x7f, t, None, (0x44, &[0x24, d]));
    let load = |c: &mut Vec<u8>, t: u8, d: u8| sse(c, 0xf3, false, 0x6f, t, None, (0x44, &[0x24, d]));
    // pand xmm(r), [rip + to]
    let pand_rip = |c: &mut Vec<u8>, r: u8, to: usize| {
        let mut tail = vec![];
        let mut probe = c.clone();
        sse(&mut probe, 0x66, false, 0xdb, r, None, (0x05, &[]));
        tail.extend(rip_to(&probe, at, to, 0));
        sse(c, 0x66, false, 0xdb, r, None, (0x05, &tail));
    };
    match site.kind {
        Kind::Insertq => {
            let (d, s, n, idx) = insertq_fields(&code[..len]).expect("checked by plan");
            let mask = if n == 0 { u64::MAX } else { (1u64 << n) - 1 };
            k[0] = [mask, 0];
            k[1] = [!(mask << idx), u64::MAX];
            let t = (0..16u8).find(|&r| r != d && r != s).unwrap();
            c.extend([0x48, 0x8d, 0x64, 0x24, 0xf0]); // lea rsp, [rsp-16]
            save(&mut c, t, 0);
            sse(&mut c, 0xf3, false, 0x7e, t, Some(s), (0, &[])); // movq xmm(t), xmm(s)
            pand_rip(&mut c, t, consts);
            if idx != 0 {
                sse(&mut c, 0x66, false, 0x73, 6, Some(t), (0, &[])); // psllq xmm(t), idx
                c.push(idx);
            }
            pand_rip(&mut c, d, consts + 16);
            sse(&mut c, 0x66, false, 0xeb, d, Some(t), (0, &[])); // por xmm(d), xmm(t)
            load(&mut c, t, 0);
            c.extend([0x48, 0x8d, 0x64, 0x24, 0x10]); // lea rsp, [rsp+16]
        }
        Kind::Clzero => {
            k[0] = [!63, 0];
            c.extend([0x48, 0x8d, 0x64, 0x24, 0xe8]); // lea rsp, [rsp-24]
            c.extend([0x48, 0x89, 0x0c, 0x24]); // mov [rsp], rcx
            save(&mut c, 0, 8);
            c.extend([0x66, 0x48, 0x0f, 0x6e, 0xc0]); // movq xmm0, rax
            pand_rip(&mut c, 0, consts);
            c.extend([0x66, 0x48, 0x0f, 0x7e, 0xc1]); // movq rcx, xmm0
            c.extend([0x66, 0x0f, 0xef, 0xc0]); // pxor xmm0, xmm0
            c.extend([0x66, 0x0f, 0x7f, 0x01]); // movdqa [rcx], xmm0
            for d in [16u8, 32, 48] {
                c.extend([0x66, 0x0f, 0x7f, 0x41, d]); // movdqa [rcx+d], xmm0
            }
            load(&mut c, 0, 8);
            c.extend([0x48, 0x8b, 0x0c, 0x24]); // mov rcx, [rsp]
            c.extend([0x48, 0x8d, 0x64, 0x24, 0x18]); // lea rsp, [rsp+24]
        }
        _ => unreachable!("in-place kinds have no stub"),
    }
    // The moved instructions (position-independent), then back.
    c.extend(&code[len..usize::from(site.window)]);
    c.push(0xe9);
    let end = at + c.len() + 4;
    c.extend(((back as i64 - end as i64) as i32).to_le_bytes());
    (c, k)
}

/// Writes `site`'s rewrite at `addr` (its address now), with a stub (if
/// any) at `arena`, which must be within 2 GiB, writable, and have room
/// for [`STUB_ROOM`] bytes. Returns the plan.
///
/// # Safety
/// `addr..addr + window` is code no thread runs yet; it and `arena` are
/// writable now.
pub(crate) unsafe fn write(site: &Site, addr: usize, arena: usize) -> Result<Plan, &'static str> {
    // SAFETY: the caller's guarantee: the window is mapped.
    let code = unsafe { std::slice::from_raw_parts(addr as *const u8, usize::from(site.window)) };
    let p = plan(site, code)?;
    match &p {
        Plan::InPlace(new) => {
            // SAFETY: as above.
            unsafe { std::ptr::copy_nonoverlapping(new.as_ptr(), addr as *mut u8, new.len()) };
        }
        Plan::Stub => {
            let consts = arena;
            let at = arena + 32;
            let (code, k) = stub(site, code, at, consts, addr + code.len());
            assert!(32 + code.len() <= STUB_ROOM);
            let rel = (at as i64) - (addr as i64 + 5);
            if i32::try_from(rel).is_err() {
                return Err("arena out of reach");
            }
            // SAFETY: the arena is ours and writable; the window as above.
            unsafe {
                std::ptr::copy_nonoverlapping(k.as_ptr().cast::<u8>(), consts as *mut u8, 32);
                std::ptr::copy_nonoverlapping(code.as_ptr(), at as *mut u8, code.len());
                let w = addr as *mut u8;
                w.write(0xe9);
                std::ptr::copy_nonoverlapping((rel as i32).to_le_bytes().as_ptr(), w.add(1), 4);
                for i in 5..usize::from(site.window) {
                    w.add(i).write(0x90);
                }
            }
        }
    }
    Ok(p)
}

/// Bytes per stub in the arena (32 of constants, then code), 16-aligned.
pub(crate) const STUB_ROOM: usize = 160;

/// Free address space for `size` bytes within 2 GiB of `[base, end)`.
fn arena_near(base: usize, end: usize, size: usize) -> usize {
    const GRAIN: usize = 0x1_0000;
    let lo = base.saturating_sub(0x7000_0000) & !(GRAIN - 1);
    let hi = end + 0x7000_0000;
    let mut at = (end + GRAIN - 1) & !(GRAIN - 1);
    for pass in 0..2 {
        if pass == 1 {
            at = lo;
        }
        while at + size < hi {
            // SAFETY: a zeroed MEMORY_BASIC_INFORMATION is valid; a query of our address space.
            let mut m: MemoryBasicInformation = unsafe { std::mem::zeroed() };
            // SAFETY: as above.
            if unsafe { VirtualQueryEx(GetCurrentProcess(), at as *const _, &mut m, size_of_val(&m)) } == 0 {
                break;
            }
            if m.state == MEM_FREE && m.base + m.region_size >= at + size {
                // SAFETY: reserves and commits free space of ours.
                let p = unsafe { VirtualAlloc(at as *const _, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
                if p != 0 {
                    return p;
                }
            }
            at = ((m.base + m.region_size.max(GRAIN)) + GRAIN - 1) & !(GRAIN - 1);
            if pass == 1 && at >= base {
                break;
            }
        }
    }
    0
}

fn mode() -> &'static str {
    static M: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| std::env::var("XNW_CPU_REWRITE").unwrap_or_default().trim().to_ascii_lowercase());
    match M.as_str() {
        "off" => "off",
        "list" => "list",
        _ => "auto",
    }
}

/// Whether `XNW_CPU_REWRITE=list` asks for the plan only.
pub fn list_only() -> bool {
    mode() == "list"
}

/// The SHA-256 of the title's executable file, set by [`identify`].
static TITLE_SHA256: std::sync::OnceLock<[u8; 32]> = std::sync::OnceLock::new();

/// SHA-256 of `data` (CNG), or `None` if the API fails.
pub(crate) fn sha256(data: &[u8]) -> Option<[u8; 32]> {
    let mut h: HANDLE = std::ptr::null_mut();
    let mut out = [0u8; 32];
    // SAFETY: the SHA-256 pseudo-handle; CNG allocates the hash object; the
    // input and output buffers are ours for the calls.
    unsafe {
        if BCryptCreateHash(BCRYPT_SHA256_ALG_HANDLE, &mut h, std::ptr::null_mut(), 0, std::ptr::null(), 0, 0) != 0 {
            return None;
        }
        let mut ok = true;
        for part in data.chunks(1 << 30) {
            ok &= BCryptHashData(h, part.as_ptr(), part.len() as u32, 0) == 0;
        }
        ok &= BCryptFinishHash(h, out.as_mut_ptr(), 32, 0) == 0;
        BCryptDestroyHash(h);
        ok.then_some(out)
    }
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The hash a note holds for the file with `stamp`: the note is the stamp,
/// a newline, then 64 hex digits.
pub(crate) fn remembered(note: &str, stamp: &str) -> Option<[u8; 32]> {
    let (s, h) = note.trim_end().split_once('\n')?;
    if s != stamp || h.len() != 64 || !h.is_ascii() {
        return None;
    }
    let mut out = [0u8; 32];
    for (o, pair) in out.iter_mut().zip(h.as_bytes().chunks(2)) {
        *o = u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()?;
    }
    Some(out)
}

/// Notes the title executable's file contents (`file`) for [`apply`]: its
/// SHA-256 picks the table. Not hashed with `XNW_CPU_REWRITE=off`. The hash
/// is kept in `note` beside the file's length and modification time
/// (`stamp`), and a later load of a file with the same stamp reads it from
/// there; each site is still decoded before it is patched.
pub fn identify(file: &[u8], stamp: Option<&str>, note: &std::path::Path) {
    if mode() == "off" {
        return;
    }
    let kept = stamp.and_then(|s| remembered(&std::fs::read_to_string(note).ok()?, s));
    if let Some(h) = kept {
        let _ = TITLE_SHA256.set(h);
        xnw_core::info!("title file SHA-256 {} ({} bytes, kept from an earlier load: {})", hex(&h), file.len(), note.display());
        return;
    }
    let t = std::time::Instant::now();
    if let Some(h) = sha256(file) {
        let _ = TITLE_SHA256.set(h);
        xnw_core::info!("title file SHA-256 {} ({} bytes, hashed in {} ms)", hex(&h), file.len(), t.elapsed().as_millis());
        if let Some(s) = stamp {
            let _ = std::fs::write(note, format!("{s}\n{}\n", hex(&h)));
        }
    } else {
        xnw_core::warn!("title file SHA-256: the hash API failed");
    }
}

/// The sites of `table` (made for the file with SHA-256 `want`) for the
/// file with SHA-256 `got`: all of them on a match, else why none.
pub(crate) fn sites_for<'a>(got: Option<&[u8; 32]>, want: &[u8; 32], table: &'a [Site]) -> Result<&'a [Site], String> {
    match got {
        Some(g) if g == want => Ok(table),
        Some(g) => Err(format!("the title file's SHA-256 is {}, the table is for {}", hex(g), hex(want))),
        None => Err(format!("the title file's SHA-256 is unknown, the table is for {}", hex(want))),
    }
}

/// Rewrites the title at `base` (`size` bytes), or with `list` prints the
/// plan. Logs one line per refused site and a summary.
///
/// # Safety
/// The title is mapped at `base` and none of its code has run.
pub unsafe fn apply(base: usize, size: usize) {
    let mode = mode();
    if mode == "off" {
        xnw_core::info!("cpu rewrite: off (XNW_CPU_REWRITE=off): Zen 2 instructions stay on the exception path");
        return;
    }
    let sites = match sites_for(TITLE_SHA256.get(), &rewrite_table::SHA256, rewrite_table::SITES) {
        Ok(s) => s,
        Err(why) => {
            xnw_core::warn!("cpu rewrite: none: {why}; Zen 2 instructions stay on the exception path");
            return;
        }
    };
    let wanted: Vec<&Site> = sites.iter().filter(|s| !native(s.kind)).collect();
    let native_n = sites.len() - wanted.len();
    let stubs = wanted.iter().filter(|s| matches!(s.kind, Kind::Insertq | Kind::Clzero)).count();
    let arena = if mode == "list" || stubs == 0 { 0 } else { arena_near(base, base + size, stubs * STUB_ROOM) };
    let (mut in_place, mut stubbed, mut refused) = (0, 0, 0);
    let mut next = arena;
    for s in &wanted {
        let addr = base + s.rva as usize;
        if mode == "list" {
            // SAFETY: the window is in the mapped image.
            let code = unsafe { std::slice::from_raw_parts(addr as *const u8, usize::from(s.window)) };
            let p = plan(s, code);
            xnw_core::info!("cpu rewrite plan: title+{:#x} {:?} ({} bytes): {}", s.rva, s.kind, s.window, match &p { Ok(Plan::InPlace(b)) => format!("in place {b:02x?}"), Ok(Plan::Stub) => "stub".into(), Err(e) => format!("left: {e}") });
            continue;
        }
        let is_stub = matches!(s.kind, Kind::Insertq | Kind::Clzero);
        if is_stub && arena == 0 {
            refused += 1;
            xnw_core::warn!("cpu rewrite: title+{:#x} {:?} left on the exception path: no arena within 2 GiB", s.rva, s.kind);
            continue;
        }
        let mut old = 0u32;
        // SAFETY: the title's code, not yet run; restored below.
        let r = unsafe {
            VirtualProtect(addr as *const _, usize::from(s.window), PAGE_EXECUTE_READWRITE, &mut old);
            let r = write(s, addr, next);
            let mut x = 0u32;
            VirtualProtect(addr as *const _, usize::from(s.window), old, &mut x);
            r
        };
        match r {
            Ok(Plan::InPlace(_)) => in_place += 1,
            Ok(Plan::Stub) => {
                stubbed += 1;
                next += STUB_ROOM;
            }
            Err(e) => {
                refused += 1;
                xnw_core::warn!("cpu rewrite: title+{:#x} {:?} left on the exception path: {e}", s.rva, s.kind);
            }
        }
    }
    if arena != 0 {
        let mut old = 0u32;
        // SAFETY: the arena is ours: stubs written, now executable.
        unsafe {
            VirtualProtect(arena as *const _, stubs * STUB_ROOM, PAGE_EXECUTE_READ, &mut old);
            FlushInstructionCache(GetCurrentProcess(), arena as *const _, stubs * STUB_ROOM);
        }
    }
    // SAFETY: plain flush of the image's code.
    unsafe { FlushInstructionCache(GetCurrentProcess(), base as *const _, size) };
    if mode == "list" {
        xnw_core::info!("cpu rewrite plan: {} sites, {} the CPU runs itself, {} to rewrite", sites.len(), native_n, wanted.len());
    } else {
        crate::crash::set_rewritten((in_place + stubbed) as u64);
        xnw_core::info!("cpu rewrite: {} sites: {in_place} in place, {stubbed} stubs (arena {arena:#x}), {refused} left on the exception path, {native_n} the CPU runs itself (XNW_CPU_REWRITE={mode})", sites.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(kind: Kind, len: usize, window: usize) -> Site {
        Site { rva: 0, kind, len: len as u8, window: window as u8, targets: 0 }
    }

    /// Made-up instances of each kind: (kind, instruction, moved after it).
    const FORMS: [(Kind, &[u8], &[u8]); 6] = [
        (Kind::Clwb, &[0x66, 0x0f, 0xae, 0x72, 0x10], &[]),            // clwb [rdx+10h]
        (Kind::Clwb, &[0x66, 0x43, 0x0f, 0xae, 0x34, 0x18], &[]),      // clwb [r8+r11]
        (Kind::Monitorx, &[0x0f, 0x01, 0xfa], &[]),
        (Kind::Mwaitx, &[0x0f, 0x01, 0xfb], &[]),
        (Kind::Insertq, &[0xf2, 0x0f, 0x78, 0xd3, 0x05, 0x07], &[]),   // insertq xmm2, xmm3, 5, 7
        (Kind::Clzero, &[0x0f, 0x01, 0xfc], &[0x48, 0x83, 0xe9, 0x40]), // clzero; sub rcx, 40h
    ];

    #[test]
    fn each_kind_decodes_and_refuses_the_others() {
        for (kind, ins, moved) in FORMS {
            let code = [ins, moved].concat();
            let s = row(kind, ins.len(), code.len());
            assert!(decodes_as(kind, ins), "{kind:?}");
            let p = plan(&s, &code).unwrap_or_else(|e| panic!("{kind:?}: {e}"));
            assert_eq!(matches!(p, Plan::Stub), matches!(kind, Kind::Insertq | Kind::Clzero), "{kind:?}");
            for (other, ins2, moved2) in FORMS {
                if other == kind {
                    continue;
                }
                let code2 = [ins2, moved2].concat();
                assert!(!decodes_as(kind, ins2), "{kind:?} accepted {other:?}");
                let s2 = row(kind, ins2.len(), code2.len());
                assert!(plan(&s2, &code2).is_err(), "{kind:?} planned over {other:?}");
            }
        }
        // The two kinds with no rewrite plan nothing.
        assert!(!decodes_as(Kind::Extrq, &[0x66, 0x0f, 0x78, 0xc0, 0x08, 0x00]));
        assert_eq!(plan(&row(Kind::Left, 3, 3), &[0x0f, 0x01, 0xfc]), Err("no rewrite for it"));
    }

    #[test]
    fn moved_instructions_must_be_register_only() {
        let clzero = [0x0f, 0x01, 0xfc];
        for (moved, ok) in [
            (&[0x48, 0x83, 0xc3, 0x20][..], true), // add rbx, 20h
            (&[0x4d, 0x39, 0xd3], true),           // cmp r11, r10
            (&[0x48, 0x8b, 0x01], false),          // mov rax, [rcx]
            (&[0xe8, 0, 0, 0, 0], false),          // call rel32
            (&[0x48, 0x83, 0xc0], false),          // cut off
        ] {
            let code = [&clzero[..], moved].concat();
            assert_eq!(plan(&row(Kind::Clzero, 3, code.len()), &code).is_ok(), ok, "{moved:02x?}");
        }
        // An in-place kind moves nothing.
        let code = [0x0f, 0x01, 0xfa, 0x48, 0x89, 0xc1];
        assert_eq!(plan(&row(Kind::Monitorx, 3, 6), &code), Err("an in-place rewrite moves nothing"));
    }

    #[test]
    fn sha256_matches_the_standard_vectors() {
        assert_eq!(hex(&sha256(b"abc").unwrap()), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(hex(&sha256(b"").unwrap()), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }

    #[test]
    fn a_kept_hash_is_read_only_for_the_same_stamp() {
        let h = sha256(b"abc").unwrap();
        let note = format!("120 77\n{}\n", hex(&h));
        assert_eq!(remembered(&note, "120 77"), Some(h));
        assert_eq!(remembered(&note, "120 78"), None);
        assert_eq!(remembered("120 77\nba78\n", "120 77"), None);
        assert_eq!(remembered(&format!("120 77\n{}\n", "zz".repeat(32)), "120 77"), None);
        assert_eq!(remembered("", ""), None);
    }

    #[test]
    fn another_file_hash_applies_nothing() {
        let table = [row(Kind::Monitorx, 3, 3)];
        let want = [7u8; 32];
        assert_eq!(sites_for(Some(&want), &want, &table).map(<[Site]>::len), Ok(1));
        let other = [8u8; 32];
        let why = sites_for(Some(&other), &want, &table).unwrap_err();
        assert!(why.contains(&hex(&other)) && why.contains(&hex(&want)), "{why}");
        assert!(sites_for(None, &want, &table).is_err());
        // The shipped table is not for a file of no bytes.
        assert!(sites_for(sha256(b"").as_ref(), &rewrite_table::SHA256, rewrite_table::SITES).is_err());
    }

    /// `XNW_TITLE_EXE=<path to the executable>`: its hash is the table's and
    /// every site decodes as its row says (read-only).
    #[test]
    #[ignore = "reads the title's executable named by XNW_TITLE_EXE"]
    fn the_table_fits_the_title_executable() {
        let path = std::env::var("XNW_TITLE_EXE").expect("XNW_TITLE_EXE");
        let file = std::fs::read(&path).unwrap();
        let t = std::time::Instant::now();
        let h = sha256(&file).unwrap();
        let ms = t.elapsed().as_millis();
        let sites = sites_for(Some(&h), &rewrite_table::SHA256, rewrite_table::SITES).unwrap();
        let img = xnw_pe::Image::parse(&file).unwrap();
        let (mut in_place, mut stubs) = (0, 0);
        let mut kinds = std::collections::BTreeMap::new();
        for s in sites {
            let off = img.offset(s.rva).unwrap();
            match plan(s, &file[off..]) {
                Ok(Plan::InPlace(_)) => in_place += 1,
                Ok(Plan::Stub) => stubs += 1,
                Err(e) => panic!("rva {:#x} {:?}: {e}", s.rva, s.kind),
            }
            *kinds.entry(format!("{:?}", s.kind)).or_insert(0) += 1;
        }
        println!("sha256 {} ({} bytes, {ms} ms); {} sites: {in_place} in place, {stubs} stubs; {kinds:?}", hex(&h), file.len(), sites.len());
        assert_eq!(in_place + stubs, sites.len());
    }
}
