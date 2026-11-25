//! Library loading as the title sees it: `LoadLibrary*`, `GetModuleHandle*`,
//! `GetProcAddress` and `FreeLibrary`, bound in place of the host's.
//!
//! Adapted from VIctory, `crates/victory-hle/src/modules/kernel.rs` at
//! d2a4bae (GPL-3.0-or-later). By name as by import, every DLL follows
//! [`route`]: a title image we mapped is found as that image; a console DLL
//! we implement (`xgameruntime.dll`) gets a pseudo handle, and
//! `GetProcAddress` on it finds our functions; a host DLL loads on the host;
//! a title DLL not mapped yet is mapped, bound and attached by the loader
//! the app installs ([`set_title_loader`]); anything else is refused, and
//! logged. The network DLLs load on the host; `GetProcAddress`
//! on them hands out our vetted functions (`net`).

use std::ffi::{CStr, c_void};
use std::sync::Mutex;

use xnw_core::win::{self, HANDLE, HMODULE};

use crate::route::{HLE, Route, route};

/// One pseudo module per console DLL we serve: its address is the handle.
/// Aligned like nothing a module handle's low flag bits would mark.
#[repr(C, align(16))]
struct Pseudo([u8; 16]);

static PSEUDO: [Pseudo; HLE.len()] = [const { Pseudo([0; 16]) }; HLE.len()];

fn pseudo_handle(dll: &str) -> Option<HMODULE> {
    let i = HLE.iter().position(|d| d.eq_ignore_ascii_case(dll))?;
    Some(std::ptr::from_ref(&PSEUDO[i]).cast_mut().cast())
}

/// The console DLL behind a pseudo handle.
fn pseudo_dll(module: HMODULE) -> Option<&'static str> {
    let at = module as usize;
    let first = PSEUDO.as_ptr() as usize;
    let i = at.checked_sub(first)? / size_of::<Pseudo>();
    (at >= first && i < HLE.len() && at.is_multiple_of(size_of::<Pseudo>())).then(|| HLE[i])
}

fn narrow(text: *const u8) -> String {
    // SAFETY: the title passes a NUL-terminated string.
    unsafe { CStr::from_ptr(text.cast()) }.to_string_lossy().into_owned()
}

fn wide(text: *const u16) -> String {
    // SAFETY: the title passes a NUL-terminated string.
    unsafe {
        let len = (0..).take_while(|&i| *text.add(i) != 0).count();
        String::from_utf16_lossy(std::slice::from_raw_parts(text, len))
    }
}

/// Who answers a DLL name.
enum Found {
    /// A title image or a pseudo module.
    Module(HMODULE),
    /// The host, with the title's own arguments.
    Host,
    /// Nobody: the call fails with `ERROR_MOD_NOT_FOUND`.
    Refused(&'static str),
    /// A title DLL whose initialisation failed: `ERROR_DLL_INIT_FAILED`.
    Failed(String),
}

/// What the app's loader answers for a title DLL asked for by name.
pub enum TitleLoad {
    /// Mapped, bound and attached (now or before): its module handle.
    Loaded(usize),
    /// No such file beside the title.
    NoFile,
    /// Mapped, but its (or a dependency's) `DllMain` failed; unmapped again.
    Failed(String),
}

/// Maps a title DLL the title asks for at run time (installed by whoever
/// runs the title, which owns the loader and the import resolver).
static TITLE_LOADER: std::sync::RwLock<Option<fn(&str) -> TitleLoad>> = std::sync::RwLock::new(None);

/// Installs the loader for title DLLs asked for by name.
pub fn set_title_loader(f: fn(&str) -> TitleLoad) {
    *TITLE_LOADER.write().unwrap_or_else(|e| e.into_inner()) = Some(f);
}

/// A title DLL by name: the loader's answer, or refused without one.
fn load_title(file: &str) -> Found {
    let f = *TITLE_LOADER.read().unwrap_or_else(|e| e.into_inner());
    let Some(f) = f else { return Found::Refused("title DLL that is not mapped") };
    match f(file) {
        TitleLoad::Loaded(base) => {
            xnw_core::info!("LoadLibrary(\"{file}\"): title DLL at {base:#x}, loaded at run time; from {}", crate::images::caller());
            Found::Module(base as HMODULE)
        }
        TitleLoad::NoFile => Found::Refused("title DLL with no file beside the title"),
        TitleLoad::Failed(why) => Found::Failed(why),
    }
}

/// The file a DLL name means, as the Windows loader reads it: no folder,
/// and `.dll` if it has no extension.
fn file_of(path: &str) -> String {
    let file = path.rsplit(['\\', '/']).next().unwrap_or(path);
    if file.contains('.') { file.to_owned() } else { format!("{file}.dll") }
}

/// Who answers `path`; `load` maps a title DLL not mapped yet
/// (`LoadLibrary`), else it is not found (`GetModuleHandle`).
fn find(path: &str, load: bool) -> Found {
    let file = file_of(path);
    if let Some(base) = crate::images::image_named(&file) {
        return Found::Module(base as HMODULE);
    }
    match route(&file) {
        Route::Hle => pseudo_handle(&file).map_or(Found::Refused("console DLL without a handle"), Found::Module),
        Route::Host | Route::HostFiltered => Found::Host,
        Route::Title if load => load_title(&file),
        Route::Title => Found::Refused("title DLL that is not mapped"),
        Route::Unknown => Found::Refused("not in the routing table"),
    }
}

/// Logs a refused name once per (name, reason).
fn refuse(what: &str, name: &str, why: &str) -> HMODULE {
    refuse_with(what, name, why, win::ERROR_MOD_NOT_FOUND)
}

fn refuse_with(what: &str, name: &str, why: &str, code: u32) -> HMODULE {
    static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let key = name.to_ascii_lowercase();
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&key) {
        xnw_core::warn!("{what}(\"{name}\") refused: {why}; from {}", crate::images::caller());
        seen.push(key);
    }
    // SAFETY: sets this thread's error code.
    unsafe { win::SetLastError(code) };
    std::ptr::null_mut()
}

fn loaded(name: &str, module: HMODULE) -> HMODULE {
    if pseudo_dll(module).is_some() {
        static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if !seen.iter().any(|s| s.eq_ignore_ascii_case(name)) {
            xnw_core::info!("LoadLibrary(\"{name}\"): our console module; from {}", crate::images::caller());
            seen.push(name.to_owned());
        }
    }
    module
}

pub extern "system" fn LoadLibraryExA(name: *const u8, file: HANDLE, flags: u32) -> HMODULE {
    if name.is_null() {
        // SAFETY: the host rejects a null name itself.
        return unsafe { win::LoadLibraryExA(name, file, flags) };
    }
    let text = narrow(name);
    match find(&text, true) {
        Found::Module(m) => loaded(&text, m),
        // SAFETY: the title's own arguments, to the host's function.
        Found::Host => unsafe { win::LoadLibraryExA(name, file, flags) },
        Found::Refused(why) => refuse("LoadLibrary", &text, why),
        Found::Failed(why) => refuse_with("LoadLibrary", &text, &why, win::ERROR_DLL_INIT_FAILED),
    }
}

pub extern "system" fn LoadLibraryExW(name: *const u16, file: HANDLE, flags: u32) -> HMODULE {
    if name.is_null() {
        // SAFETY: the host rejects a null name itself.
        return unsafe { win::LoadLibraryExW(name, file, flags) };
    }
    let text = wide(name);
    match find(&text, true) {
        Found::Module(m) => loaded(&text, m),
        // SAFETY: the title's own arguments, to the host's function.
        Found::Host => unsafe { win::LoadLibraryExW(name, file, flags) },
        Found::Refused(why) => refuse("LoadLibrary", &text, why),
        Found::Failed(why) => refuse_with("LoadLibrary", &text, &why, win::ERROR_DLL_INIT_FAILED),
    }
}

pub extern "system" fn LoadLibraryA(name: *const u8) -> HMODULE {
    LoadLibraryExA(name, std::ptr::null_mut(), 0)
}

pub extern "system" fn LoadLibraryW(name: *const u16) -> HMODULE {
    LoadLibraryExW(name, std::ptr::null_mut(), 0)
}

/// `NULL` is the process's executable: for the title, its own image.
pub extern "system" fn GetModuleHandleA(name: *const u8) -> HMODULE {
    if name.is_null() {
        return crate::title_base() as HMODULE;
    }
    let text = narrow(name);
    match find(&text, false) {
        Found::Module(m) => m,
        // SAFETY: the title's own argument, to the host's function.
        Found::Host => unsafe { win::GetModuleHandleA(name) },
        Found::Refused(_) | Found::Failed(_) => {
            // SAFETY: sets this thread's error code.
            unsafe { win::SetLastError(win::ERROR_MOD_NOT_FOUND) };
            std::ptr::null_mut()
        }
    }
}

/// See [`GetModuleHandleA`].
pub extern "system" fn GetModuleHandleW(name: *const u16) -> HMODULE {
    if name.is_null() {
        return crate::title_base() as HMODULE;
    }
    match find(&wide(name), false) {
        Found::Module(m) => m,
        // SAFETY: the title's own argument, to the host's function.
        Found::Host => unsafe { win::GetModuleHandleW(name) },
        Found::Refused(_) | Found::Failed(_) => {
            // SAFETY: sets this thread's error code.
            unsafe { win::SetLastError(win::ERROR_MOD_NOT_FOUND) };
            std::ptr::null_mut()
        }
    }
}

/// The file name of a host module.
fn host_file(module: HMODULE) -> String {
    let mut path = [0u16; 260];
    // SAFETY: our buffer and its size.
    let len = unsafe { win::GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) } as usize;
    let path = String::from_utf16_lossy(&path[..len.min(path.len())]);
    path.rsplit('\\').next().unwrap_or_default().to_owned()
}

fn not_found(dll: &str, symbol: &str) -> *const c_void {
    static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let key = format!("{dll}!{symbol}");
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&key) {
        xnw_core::warn!("GetProcAddress({key}): not implemented; from {}", crate::images::caller());
        seen.push(key);
    }
    // SAFETY: sets this thread's error code.
    unsafe { win::SetLastError(win::ERROR_PROC_NOT_FOUND) };
    std::ptr::null()
}

/// Our functions behind a pseudo handle; the title's own exports in its
/// images; for a host module, our override where import binding would
/// bind one, else the host's function.
pub extern "system" fn GetProcAddress(module: HMODULE, name: *const u8) -> *const c_void {
    let module = if module.is_null() { crate::title_base() as HMODULE } else { module };
    // Values below 0x10000 are ordinals.
    let symbol = if (name as usize) < 0x1_0000 { format!("#{}", name as usize) } else { narrow(name) };
    if let Some(dll) = pseudo_dll(module) {
        return match crate::lookup(dll, &symbol) {
            Some(f) => f as *const c_void,
            None => not_found(dll, &symbol),
        };
    }
    if !crate::images::is_image(module as usize) {
        let file = host_file(module);
        if let Some(f) = crate::host_override(&file, &symbol) {
            return f as *const c_void;
        }
    }
    // SAFETY: the title's own arguments, to the host's function.
    unsafe { win::GetProcAddress(module, name) }
}

/// Pseudo modules and title images stay loaded: loading them took no
/// reference (a title DLL loaded at run time too: a later `LoadLibrary`
/// gets the same module, already initialised).
pub extern "system" fn FreeLibrary(module: HMODULE) -> i32 {
    if pseudo_dll(module).is_some() || crate::images::is_image(module as usize) {
        return 1;
    }
    // SAFETY: the title's own handle, to the host's function.
    unsafe { win::FreeLibrary(module) }
}

/// The loader's `DisableThreadLibraryCalls` for title images: `None` if
/// the module is not one (handed in by whoever runs the title).
static THREAD_CALLS: std::sync::RwLock<Option<fn(usize) -> Option<bool>>> = std::sync::RwLock::new(None);

/// Installs the loader's thread-notice switch for title images.
pub fn set_thread_calls(f: fn(usize) -> Option<bool>) {
    *THREAD_CALLS.write().unwrap_or_else(|e| e.into_inner()) = Some(f);
}

/// A title image stops getting thread notices from our loader (which sends
/// them, not Windows'); a host module asks the host.
pub extern "system" fn DisableThreadLibraryCalls(module: HMODULE) -> i32 {
    let f = *THREAD_CALLS.read().unwrap_or_else(|e| e.into_inner());
    match f.and_then(|f| f(module as usize)) {
        Some(done) => done.into(),
        None => host!("kernel32.dll", "DisableThreadLibraryCalls": extern "system" fn(HMODULE) -> i32)(module),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_loads_as_a_pseudo_module() {
        let m = LoadLibraryExA(c"xgameruntime.dll".as_ptr().cast(), std::ptr::null_mut(), 0x800);
        assert!(!m.is_null());
        assert_eq!(pseudo_dll(m), Some("xgameruntime.dll"));
        assert_eq!(GetModuleHandleA(c"XGameRuntime".as_ptr().cast()), m);
        for name in [c"QueryApiImpl", c"InitializeApiImpl", c"InitializeApiImplEx", c"InitializeApiImplEx2", c"XErrorReport", c"UninitializeApiImpl"] {
            assert!(!GetProcAddress(m, name.as_ptr().cast()).is_null(), "{name:?}");
        }
        assert!(GetProcAddress(m, c"NoSuchExport".as_ptr().cast()).is_null());
        assert_eq!(FreeLibrary(m), 1);
    }

    #[test]
    fn network_dlls_load_with_our_vetted_functions() {
        let w = LoadLibraryA(c"ws2_32.dll".as_ptr().cast());
        assert!(!w.is_null());
        assert_eq!(GetProcAddress(w, c"connect".as_ptr().cast()) as usize, crate::net::connect as usize);
        assert_eq!(GetProcAddress(w, 4 as *const u8) as usize, crate::net::connect as usize);
        assert!(!GetProcAddress(w, c"WSAStartup".as_ptr().cast()).is_null());
        let k = LoadLibraryA(c"kernel32.dll".as_ptr().cast());
        assert!(!k.is_null());
        // Looked up by name, our loader functions still win over the host's.
        assert_eq!(GetProcAddress(k, c"LoadLibraryExA".as_ptr().cast()) as usize, LoadLibraryExA as usize);
        assert!(!GetProcAddress(k, c"GetTickCount".as_ptr().cast()).is_null());
    }
}

#[cfg(test)]
mod title_load_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CALLS: AtomicUsize = AtomicUsize::new(0);

    /// A real module (so `GetProcAddress` on it works) standing in for a
    /// title DLL, entered as the app's loader enters one.
    fn fake_loader(file: &str) -> TitleLoad {
        CALLS.fetch_add(1, Ordering::SeqCst);
        match file {
            "amd_fidelityfx_upscaler_xbox.dll" => {
                // SAFETY: a system DLL, loaded for its handle.
                let m = unsafe { win::LoadLibraryA(c"version.dll".as_ptr().cast()) } as usize;
                crate::images::add_image(file, m, 0x1000);
                TitleLoad::Loaded(m)
            }
            "bink2scarlett.dll" => TitleLoad::Failed("DllMain returned FALSE".into()),
            _ => TitleLoad::NoFile,
        }
    }

    #[test]
    fn title_dll_by_name_maps_once() {
        set_title_loader(fake_loader);
        let name = c"amd_fidelityfx_upscaler_xbox.dll";
        assert!(GetModuleHandleA(name.as_ptr().cast()).is_null(), "GetModuleHandle does not load");
        assert_eq!(CALLS.load(Ordering::SeqCst), 0);
        let m = LoadLibraryExA(name.as_ptr().cast(), std::ptr::null_mut(), 0);
        assert!(!m.is_null());
        assert_eq!(CALLS.load(Ordering::SeqCst), 1);
        // A second request, by any spelling, gets the same module, not mapped again.
        assert_eq!(LoadLibraryA(c"E:/game/AMD_FidelityFX_Upscaler_Xbox".as_ptr().cast()), m);
        assert_eq!(CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(GetModuleHandleA(name.as_ptr().cast()), m);
        assert!(!GetProcAddress(m, c"GetFileVersionInfoSizeW".as_ptr().cast()).is_null());
        assert_eq!(FreeLibrary(m), 1);
        assert_eq!(LoadLibraryA(name.as_ptr().cast()), m);

        assert!(LoadLibraryA(c"bink2scarlett.dll".as_ptr().cast()).is_null());
        assert_eq!(win::last_error().raw_os_error(), Some(win::ERROR_DLL_INIT_FAILED as i32));
        assert!(LoadLibraryA(c"xboxbvhbuilder.dll".as_ptr().cast()).is_null());
        assert_eq!(win::last_error().raw_os_error(), Some(win::ERROR_MOD_NOT_FOUND as i32));
    }
}
