//! Static TLS of an executable mapped as a DLL.
//!
//! Adapted from VIctory, `crates/victory-loader/src/tls.rs` at d2a4bae
//! (GPL-3.0-or-later).
//!
//! An exe always has TLS index 0, so its code reads its thread block as
//! `mov r64, gs:[0x58]` then `[r64]`, never consulting `_tls_index`. Mapped
//! as a DLL it gets another index. Every such load becomes a call to a thunk
//! that yields `&array[index]`, and `_tls_index` (which CRT code does add)
//! is set to 0.

use std::ffi::c_void;

use xnw_pe::Image;

use xnw_core::win::*;

/// `mov r64, gs:[0x58]` is `65 REX 8B ModRM 25 58 00 00 00`.
const LOAD_LEN: usize = 9;
const THUNK_SIZE: usize = 32;

/// Returns the number of loads patched (0 when the OS gave index 0).
///
/// # Safety
/// `base` is `img` mapped by the Windows loader, and none of its code runs.
pub unsafe fn redirect(base: usize, img: &Image) -> std::io::Result<usize> {
    let Some(tls) = img.tls().map_err(std::io::Error::other)? else { return Ok(0) };
    // SAFETY: the directory is in the image; the OS relocated the pointer
    // to the index and wrote the index.
    let index_at = unsafe { ((base + tls.directory_rva as usize + 16) as *const usize).read() } as *mut u32;
    // SAFETY: as above.
    let index = unsafe { index_at.read() };
    if index == 0 {
        return Ok(0);
    }
    let thunks = thunks(near(base, img.size_of_image as usize)?, index)?;
    let mut patched = 0;
    for s in img.sections.iter().filter(|s| s.flags & 0x2000_0000 != 0) {
        let (start, len) = (base + s.rva as usize, s.virtual_size as usize);
        let mut old = 0;
        // SAFETY: the section is part of the mapped image and nothing runs in it.
        unsafe {
            if VirtualProtect(start as _, len, PAGE_EXECUTE_READWRITE, &mut old) == 0 {
                return Err(last_error());
            }
            patched += patch(std::slice::from_raw_parts_mut(start as *mut u8, len), start, thunks);
            VirtualProtect(start as _, len, old, &mut old);
            FlushInstructionCache(GetCurrentProcess(), start as _, len);
        }
    }
    // SAFETY: `_tls_index` is in the image's writable data.
    unsafe { index_at.write(0) };
    xnw_core::TITLE_TLS_INDEX.store(index, std::sync::atomic::Ordering::Relaxed);
    xnw_core::info!("static TLS index {index}: {patched} loads redirected");
    Ok(patched)
}

/// Rewrites each `mov r64, gs:[0x58]` as `call thunks[r]; nop4`.
// shortcut (VIctory): a byte match, not a decode. In this title all 53,246
// hits are instruction starts and none is missed.
fn patch(code: &mut [u8], at: usize, thunks: usize) -> usize {
    let (mut count, mut i) = (0, 0);
    while i + LOAD_LEN <= code.len() {
        let c = &code[i..i + LOAD_LEN];
        let rex = c[1];
        let reg = usize::from((c[3] >> 3) & 7) | if rex & 4 != 0 { 8 } else { 0 };
        if c[0] == 0x65 && (rex == 0x48 || rex == 0x4C) && c[2] == 0x8B && c[3] & 0xC7 == 0x04 && c[4..] == [0x25, 0x58, 0, 0, 0] && reg != 4 {
            let rel = (thunks + reg * THUNK_SIZE) as i64 - (at + i + 5) as i64;
            code[i] = 0xE8;
            code[i + 1..i + 5].copy_from_slice(&(rel as i32).to_le_bytes());
            code[i + 5..i + 9].copy_from_slice(&[0x0F, 0x1F, 0x40, 0x00]);
            count += 1;
            i += LOAD_LEN;
        } else {
            i += 1;
        }
    }
    count
}

/// One thunk per register: `mov r, gs:[0x58]; lea r, [r + index*8]; ret`.
fn thunks(block: *mut u8, index: u32) -> std::io::Result<usize> {
    let disp = (index * 8).to_le_bytes();
    for reg in 0..16u8 {
        let (r, low) = (reg >> 3, reg & 7);
        let mut code = vec![0x65, 0x48 | r << 2, 0x8B, low << 3 | 4, 0x25, 0x58, 0, 0, 0];
        code.extend([0x48 | r << 2 | r, 0x8D, 0x80 | low << 3 | low]);
        if low == 4 {
            code.push(0x24);
        }
        code.extend(disp);
        code.push(0xC3);
        // SAFETY: `block` is a writable page holding 16 thunks of 32 bytes.
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), block.add(usize::from(reg) * THUNK_SIZE), code.len()) };
    }
    let mut old = 0;
    // SAFETY: our own page.
    if unsafe { VirtualProtect(block as _, 4096, PAGE_EXECUTE_READ, &mut old) } == 0 {
        return Err(last_error());
    }
    Ok(block as usize)
}

/// One writable page within `rel32` reach of `[base, base + size)`; leaked.
pub(crate) fn near(base: usize, size: usize) -> std::io::Result<*mut u8> {
    const GRANULE: usize = 0x1_0000;
    let above = (base + size).next_multiple_of(GRANULE);
    for k in 0..(1usize << 14) {
        for at in [above + k * GRANULE, base.wrapping_sub((k + 1) * GRANULE)] {
            // SAFETY: asks for fresh memory at a hint; fails harmlessly if taken.
            let p = unsafe { VirtualAlloc(at as *const c_void, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) };
            if !p.is_null() {
                return Ok(p.cast());
            }
        }
    }
    Err(std::io::Error::other("no free page within 1 GiB of the image"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patches_only_exact_loads() {
        // mov rax, gs:[58h] ; mov r9, gs:[58h] ; mov rax, gs:[60h]
        let mut code = vec![0x65, 0x48, 0x8B, 0x04, 0x25, 0x58, 0, 0, 0, 0x65, 0x4C, 0x8B, 0x0C, 0x25, 0x58, 0, 0, 0, 0x65, 0x48, 0x8B, 0x04, 0x25, 0x60, 0, 0, 0];
        assert_eq!(patch(&mut code, 0x1000, 0x8000), 2);
        assert_eq!(code[0], 0xE8);
        assert_eq!(i32::from_le_bytes(code[1..5].try_into().unwrap()), 0x8000 - 0x1005);
        assert_eq!(i32::from_le_bytes(code[10..14].try_into().unwrap()), 0x8000 + 9 * 32 - (0x1009 + 5));
        assert_eq!(code[18], 0x65, "another offset is left alone");
    }
}
