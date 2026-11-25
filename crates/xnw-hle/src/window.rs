//! The title's window: our procedure in front of the title's.
//!
//! Ported from VIctory-base `victory-hle/src/modules/lifecycle.rs`
//! (`CreateWindowExW`/`CreateWindowExA`, `hook`, `window_proc`; 5d0516e) and
//! VIctory main ff8694a (SC_KEYMENU), GPL-3.0-or-later. Trimmed: the window
//! is fitted by the presenter (`xnw-gpu` `d3d12x::present`), and closing is
//! left to the title's procedure for now.
//!
//! Alt or F10 sends WM_SYSCOMMAND SC_KEYMENU, and the default handling runs
//! a modal menu loop on the thread that owns the window, which is the
//! title's: it stops drawing and reading input until a key ends the loop,
//! and its watchdog ends it about half a minute later (VIctory: the
//! deliberate write to 0 at +0x23cf683). Our procedure drops SC_KEYMENU.

use std::ffi::c_void;

use xnw_core::win::{
    CallWindowProcW, CreateWindowExA as HostCreateWindowExA, CreateWindowExW as HostCreateWindowExW, GWLP_WNDPROC, GetPropW,
    GetWindowLongPtrW, HANDLE, HMODULE, RemovePropW, SC_KEYMENU, SetPropW, SetWindowLongPtrW, WM_NCDESTROY, WM_SYSCOMMAND, WndProc,
};

/// The window property holding the procedure a title window had before ours.
const TITLE_PROC: &[u16] = &[
    b'x' as u16, b'n' as u16, b'w' as u16, b'.' as u16, b'w' as u16, b'n' as u16, b'd' as u16, b'p' as u16, b'r' as u16, b'o' as u16,
    b'c' as u16, 0,
];

pub extern "system" fn CreateWindowExW(
    ex_style: u32,
    class: *const u16,
    name: *const u16,
    style: u32,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    parent: HANDLE,
    menu: HANDLE,
    instance: HMODULE,
    param: *const c_void,
) -> HANDLE {
    let style = if headless() { style & !WS_VISIBLE } else { style };
    // SAFETY: the title's own arguments, forwarded.
    let window = unsafe { HostCreateWindowExW(ex_style, class, name, style, x, y, width, height, parent, menu, instance, param) };
    hook(window);
    window
}

const WS_VISIBLE: u32 = 0x1000_0000;

/// `XNW_HEADLESS=1`: the title's windows are never shown (smoke runs;
/// null GPU from frame 0 makes no swap chain either). Read once, last
/// error kept (the first call may be a title's).
fn headless() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let keep = std::io::Error::last_os_error().raw_os_error().unwrap_or(0) as u32;
        let on = std::env::var("XNW_HEADLESS").is_ok_and(|v| v.trim() == "1");
        // SAFETY: plain call on this thread.
        unsafe { xnw_core::win::SetLastError(keep) };
        on
    })
}

/// `ShowWindow`: under `XNW_HEADLESS=1` a title window stays hidden (it
/// answers "was hidden", 0); else the host's.
pub extern "system" fn ShowWindow(window: HANDLE, command: i32) -> i32 {
    // SAFETY: a property read on the title's window handle.
    if headless() && !unsafe { GetPropW(window, TITLE_PROC.as_ptr()) }.is_null() {
        return 0;
    }
    // SAFETY: the title's own arguments, forwarded.
    unsafe { xnw_core::win::ShowWindow(window, command) }
}

/// Read before the title runs (run.rs), so no title call does the lookup.
pub fn read_switches() {
    let _ = headless();
}

/// See [`CreateWindowExW`].
pub extern "system" fn CreateWindowExA(
    ex_style: u32,
    class: *const u8,
    name: *const u8,
    style: u32,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    parent: HANDLE,
    menu: HANDLE,
    instance: HMODULE,
    param: *const c_void,
) -> HANDLE {
    let style = if headless() { style & !WS_VISIBLE } else { style };
    // SAFETY: the title's own arguments, forwarded.
    let window = unsafe { HostCreateWindowExA(ex_style, class, name, style, x, y, width, height, parent, menu, instance, param) };
    hook(window);
    window
}

/// Puts our procedure in front of the title's.
fn hook(window: HANDLE) {
    if window.is_null() {
        return;
    }
    // SAFETY: a window this thread just created. The title's procedure is
    // stored before ours can run.
    unsafe {
        let title = GetWindowLongPtrW(window, GWLP_WNDPROC);
        SetPropW(window, TITLE_PROC.as_ptr(), title as HANDLE);
        SetWindowLongPtrW(window, GWLP_WNDPROC, procedure as WndProc as usize as isize);
    }
    xnw_core::info!("window: title window {window:?} subclassed (Alt/F10 menu mode dropped)");
}

unsafe extern "system" fn procedure(window: HANDLE, message: u32, wparam: usize, lparam: isize) -> isize {
    // A console has no window menu: Alt or F10 would freeze the title's
    // thread in a modal menu loop. The mouse still opens it.
    if message == WM_SYSCOMMAND && wparam & 0xFFF0 == SC_KEYMENU {
        return 0;
    }
    // SAFETY: the procedure `hook` stored, called the way the system would.
    unsafe {
        let title = GetPropW(window, TITLE_PROC.as_ptr());
        if message == WM_NCDESTROY {
            RemovePropW(window, TITLE_PROC.as_ptr());
        }
        let title = std::mem::transmute::<HANDLE, Option<WndProc>>(title);
        CallWindowProcW(title, window, message, wparam, lparam)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use xnw_core::win::{DestroyWindow, SendMessageW};

    use super::*;

    thread_local! {
        /// The system commands `recording` received, low four bits cleared.
        static COMMANDS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    }

    /// Stands in for a title's procedure: notes system commands and acts
    /// on none of them.
    unsafe extern "system" fn recording(window: HANDLE, message: u32, wparam: usize, lparam: isize) -> isize {
        if message == WM_SYSCOMMAND {
            COMMANDS.with(|c| c.borrow_mut().push(wparam & 0xFFF0));
            return 0;
        }
        // SAFETY: the system's default procedure.
        unsafe { xnw_core::win::DefWindowProcW(window, message, wparam, lparam) }
    }

    /// VIctory ff8694a `the_keyboard_menu_never_reaches_the_title`.
    #[test]
    fn the_keyboard_menu_never_reaches_the_title() {
        let class: Vec<u16> = "STATIC\0".encode_utf16().collect();
        let null = std::ptr::null_mut();
        let window = CreateWindowExW(0, class.as_ptr(), class.as_ptr(), 0, 0, 0, 10, 10, null, null, null, std::ptr::null());
        assert!(!window.is_null());
        const SC_MINIMIZE: usize = 0xF020;
        // SAFETY: our window, owned by this thread, so each message is
        // handled before SendMessageW returns; the original procedure is
        // put back before the window is destroyed.
        unsafe {
            let title = GetPropW(window, TITLE_PROC.as_ptr());
            SetPropW(window, TITLE_PROC.as_ptr(), recording as WndProc as usize as HANDLE);
            SendMessageW(window, WM_SYSCOMMAND, SC_KEYMENU, 0); // Alt or F10
            SendMessageW(window, WM_SYSCOMMAND, SC_KEYMENU, isize::from(b' ')); // Alt+Space
            SendMessageW(window, WM_SYSCOMMAND, SC_KEYMENU | 1, 0); // the low bits are the system's
            SendMessageW(window, WM_SYSCOMMAND, SC_MINIMIZE, 0);
            SetPropW(window, TITLE_PROC.as_ptr(), title);
            DestroyWindow(window);
        }
        let seen = COMMANDS.with(|c| c.borrow().clone());
        assert_eq!(seen, [SC_MINIMIZE]);
    }
}
