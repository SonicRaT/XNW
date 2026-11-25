//! The title's drives. On the console the package is drive `G:`, read only,
//! and title storage sits on drives of its own; titles build paths on that.
//!
//! Every title import that takes a path is bound here (`kernel32` file
//! functions, the CRT's `fopen` family, `GetModuleFileName`):
//!
//! * `G:` (and relative paths: the console's working folder is `G:\`) is
//!   the host folder holding the package. It is never written: opening for
//!   write, creating, deleting, renaming or changing attributes there fails
//!   with `ERROR_ACCESS_DENIED`, as on a read-only drive.
//! * Any other drive letter `X:` is a folder of the overlay,
//!   `<data>\X\`, created on first use.
//! * UNC and device paths (`\\server\..`, `\\.\..`) are refused with
//!   `ERROR_BAD_NETPATH`: they could leave the machine or reach host devices.
//!
//! So title code never touches a host path outside the package (read) and
//! the overlay (read and write). `GetModuleFileName` names the title's
//! images as the console does, `G:\<file>`.
//!
//! Adapted in idea from VIctory `crates/victory-hle/src/modules/files.rs`
//! (d2a4bae, GPL-3.0-or-later), which mounts virtual drives; the mapping
//! rules and code here are ours.

use std::ffi::{CStr, c_char, c_void};
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, RwLock};

use xnw_core::win;

static GAME_DRIVE: RwLock<Vec<PathBuf>> = RwLock::new(Vec::new());
static OVERLAY: RwLock<Option<PathBuf>> = RwLock::new(None);

/// The host folders behind the title's `G:` drive, searched in order: a
/// file is served from the first folder that has it (fast local copies in
/// front of the full package). All are read only.
// shortcut: a directory listing of `G:` sees the first folder that has the
// directory, not the union; the title opens its packages by name.
pub fn set_game_drive(roots: &[PathBuf]) {
    let roots: Vec<PathBuf> = roots.iter().map(|r| std::path::absolute(r).unwrap_or_else(|_| r.clone())).collect();
    for (i, r) in roots.iter().enumerate() {
        xnw_core::info!("game drive G: folder {}: {} (read only)", i + 1, r.display());
    }
    *GAME_DRIVE.write().unwrap_or_else(|e| e.into_inner()) = roots;
}

/// `rest` on the game drive: in the first folder that has it, else where
/// it would be in the last (the full package).
fn on_game_drive(rest: &str) -> PathBuf {
    let roots = GAME_DRIVE.read().unwrap_or_else(|e| e.into_inner());
    let mut last = under(Path::new("."), rest);
    for root in roots.iter() {
        last = under(root, rest);
        if last.exists() {
            break;
        }
    }
    last
}

/// The host folder whose subfolders are the title's other drives.
pub fn set_overlay(root: &Path) {
    let root = std::path::absolute(root).unwrap_or_else(|_| root.to_owned());
    xnw_core::info!("title drives other than G: live in {}", root.display());
    *OVERLAY.write().unwrap_or_else(|e| e.into_inner()) = Some(root);
}

const ERROR_ACCESS_DENIED: u32 = 5;
const ERROR_INSUFFICIENT_BUFFER: u32 = 122;
const ERROR_BAD_NETPATH: u32 = 53;
const INVALID_HANDLE: win::HANDLE = -1isize as win::HANDLE;
const INVALID_ATTRIBUTES: u32 = u32::MAX;

/// Where a title path lands.
#[derive(Debug, PartialEq)]
pub(crate) enum Place {
    /// In the package: read only.
    Game(PathBuf),
    /// In the overlay: read and write.
    Overlay(PathBuf),
    /// Nowhere we serve (UNC, devices): refused.
    Refused,
}

impl Place {
    fn path(&self) -> Option<&Path> {
        match self {
            Place::Game(p) | Place::Overlay(p) => Some(p),
            Place::Refused => None,
        }
    }
}

/// `rest` (after the drive) under `root`, with `.` and `..` resolved here
/// (`..` stops at the root).
fn under(root: &Path, rest: &str) -> PathBuf {
    let mut parts: Vec<&str> = Vec::new();
    for part in rest.split(['\\', '/']) {
        match part {
            "" | "." => {}
            ".." => drop(parts.pop()),
            _ => parts.push(part),
        }
    }
    let mut out = root.to_owned();
    out.extend(parts);
    out
}

/// Where the title's `path` is on the host.
pub(crate) fn place(path: &str) -> Place {
    let started = xnw_core::perf::start();
    let place = place_of(path);
    // A package archive in no `G:` folder (an incomplete copy) is named
    // once; the title fails much later and elsewhere without it.
    if let Place::Game(p) = &place
        && p.extension().is_some_and(|e| e.eq_ignore_ascii_case("rpf"))
        && !p.exists()
    {
        static MISSING: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
        let mut missing = MISSING.lock().unwrap_or_else(|e| e.into_inner());
        if !missing.contains(p) {
            let keep = std::io::Error::last_os_error().raw_os_error().unwrap_or(0) as u32;
            xnw_core::warn!("files: {path} is in no game drive folder ({} not found)", p.display());
            missing.push(p.clone());
            fail(keep);
        }
    }
    if started.is_some() {
        xnw_core::perf::since(xnw_core::perf::Area::FileOpen, started, 0);
        if let Place::Game(p) = &place
            && on_package(p)
            && p.is_file()
        {
            static SEEN: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            if !seen.contains(p) {
                let size = std::fs::metadata(p).map_or(0, |m| m.len());
                xnw_core::log_line("perf", format_args!("files: {} served from the full package ({} MB)", p.display(), size / 1_000_000));
                seen.push(p.clone());
            }
        }
    }
    place
}

/// Whether host path `p` lies in the last `G:` folder (the full package)
/// while faster folders stand in front of it.
pub(crate) fn on_package(p: &Path) -> bool {
    let roots = GAME_DRIVE.read().unwrap_or_else(|e| e.into_inner());
    roots.len() > 1 && roots.last().is_some_and(|r| p.starts_with(r))
}

fn place_of(path: &str) -> Place {
    let bare = path.strip_prefix(r"\\?\").unwrap_or(path);
    if bare.starts_with(r"\\") || bare.starts_with("//") || path.starts_with(r"\\.\") {
        return Place::Refused;
    }
    let b = bare.as_bytes();
    if b.len() >= 2 && b[1] == b':' && b[0].is_ascii_alphabetic() {
        let letter = b[0].to_ascii_uppercase();
        if letter == b'G' {
            return Place::Game(on_game_drive(&bare[2..]));
        }
        let overlay = OVERLAY.read().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_else(|| PathBuf::from(r"out\data"));
        let root = overlay.join(char::from(letter).to_string());
        let _ = std::fs::create_dir_all(&root);
        return Place::Overlay(under(&root, &bare[2..]));
    }
    // Relative, or rooted without a drive: on the working drive, `G:`.
    Place::Game(on_game_drive(bare))
}

/// The host path the title's path lands on, for readers (DirectStorage).
pub(crate) fn host_path(path: &str) -> Option<PathBuf> {
    place(path).path().map(Path::to_owned)
}

fn wide(text: *const u16) -> String {
    // SAFETY: the title passes a NUL-terminated string.
    unsafe {
        let len = (0..).take_while(|&i| *text.add(i) != 0).count();
        String::from_utf16_lossy(std::slice::from_raw_parts(text, len))
    }
}

fn narrow(text: *const c_char) -> String {
    // SAFETY: the title passes a NUL-terminated string.
    unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned()
}

fn to_wide(path: &Path) -> Vec<u16> {
    path.as_os_str().encode_wide().chain([0]).collect()
}

fn fail(code: u32) {
    // SAFETY: sets this thread's error code.
    unsafe { win::SetLastError(code) };
}

/// Logs a refused path once per (what, path).
fn refused(what: &'static str, path: &str, code: u32) {
    static SEEN: Mutex<Vec<(&str, String)>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.iter().any(|(w, p)| *w == what && p == path) {
        xnw_core::warn!("{what}(\"{path}\") refused ({code}); from {}", crate::images::caller());
        seen.push((what, path.to_owned()));
    }
    fail(code);
}

/// The host path for a call that only reads, or `None` (error set).
fn for_reading(what: &'static str, path: &str) -> Option<Vec<u16>> {
    match place(path) {
        Place::Game(p) | Place::Overlay(p) => Some(to_wide(&p)),
        Place::Refused => {
            refused(what, path, ERROR_BAD_NETPATH);
            None
        }
    }
}

/// The host path for a call that changes the file system, or `None`
/// (error set): only the overlay may change.
fn for_writing(what: &'static str, path: &str) -> Option<Vec<u16>> {
    match place(path) {
        Place::Overlay(p) => Some(to_wide(&p)),
        Place::Game(_) => {
            refused(what, path, ERROR_ACCESS_DENIED);
            None
        }
        Place::Refused => {
            refused(what, path, ERROR_BAD_NETPATH);
            None
        }
    }
}

// Access rights and dispositions that change a file.
const WRITE_RIGHTS: u32 = 0x4000_0000 | 0x1000_0000 | 0x2 | 0x4 | 0x10 | 0x100 | 0x1_0000 | 0x4_0000 | 0x8_0000 | 0x0200_0000;
const OPEN_EXISTING: u32 = 3;
const OPEN_ALWAYS: u32 = 4;
const DELETE_ON_CLOSE: u32 = 0x0400_0000;

type CreateFileW = extern "system" fn(*const u16, u32, u32, *const c_void, u32, u32, win::HANDLE) -> win::HANDLE;

pub extern "system" fn CreateFileW(
    path: *const u16,
    access: u32,
    share: u32,
    security: *const c_void,
    disposition: u32,
    flags: u32,
    template: win::HANDLE,
) -> win::HANDLE {
    let host = host!("kernel32.dll", "CreateFileW": CreateFileW);
    if path.is_null() {
        return host(path, access, share, security, disposition, flags, template);
    }
    let name = wide(path);
    let writes = access & WRITE_RIGHTS != 0 || flags & DELETE_ON_CLOSE != 0 || !matches!(disposition, OPEN_EXISTING | OPEN_ALWAYS);
    let (target, disposition) = match place(&name) {
        // On the package, "open always" can only open.
        Place::Game(p) if !writes => (to_wide(&p), OPEN_EXISTING),
        Place::Overlay(p) => (to_wide(&p), disposition),
        Place::Game(_) => {
            refused("CreateFileW", &name, ERROR_ACCESS_DENIED);
            return INVALID_HANDLE;
        }
        Place::Refused => {
            refused("CreateFileW", &name, ERROR_BAD_NETPATH);
            return INVALID_HANDLE;
        }
    };
    host(target.as_ptr(), access, share, security, disposition, flags, template)
}

pub extern "system" fn CreateDirectoryW(path: *const u16, security: *const c_void) -> i32 {
    let Some(p) = for_writing("CreateDirectoryW", &wide(path)) else { return 0 };
    host!("kernel32.dll", "CreateDirectoryW": extern "system" fn(*const u16, *const c_void) -> i32)(p.as_ptr(), security)
}

pub extern "system" fn DeleteFileW(path: *const u16) -> i32 {
    let Some(p) = for_writing("DeleteFileW", &wide(path)) else { return 0 };
    host!("kernel32.dll", "DeleteFileW": extern "system" fn(*const u16) -> i32)(p.as_ptr())
}

pub extern "system" fn RemoveDirectoryW(path: *const u16) -> i32 {
    let Some(p) = for_writing("RemoveDirectoryW", &wide(path)) else { return 0 };
    host!("kernel32.dll", "RemoveDirectoryW": extern "system" fn(*const u16) -> i32)(p.as_ptr())
}

pub extern "system" fn SetFileAttributesW(path: *const u16, attributes: u32) -> i32 {
    let Some(p) = for_writing("SetFileAttributesW", &wide(path)) else { return 0 };
    host!("kernel32.dll", "SetFileAttributesW": extern "system" fn(*const u16, u32) -> i32)(p.as_ptr(), attributes)
}

/// Both ends must be in the overlay (a move off the package deletes there).
pub extern "system" fn MoveFileExW(from: *const u16, to: *const u16, flags: u32) -> i32 {
    let Some(f) = for_writing("MoveFileExW", &wide(from)) else { return 0 };
    let t = if to.is_null() {
        None
    } else {
        match for_writing("MoveFileExW", &wide(to)) {
            Some(t) => Some(t),
            None => return 0,
        }
    };
    let host = host!("kernel32.dll", "MoveFileExW": extern "system" fn(*const u16, *const u16, u32) -> i32);
    host(f.as_ptr(), t.as_ref().map_or(std::ptr::null(), |t| t.as_ptr()), flags)
}

pub extern "system" fn FindFirstFileW(path: *const u16, data: *mut c_void) -> win::HANDLE {
    let Some(p) = for_reading("FindFirstFileW", &wide(path)) else { return INVALID_HANDLE };
    host!("kernel32.dll", "FindFirstFileW": extern "system" fn(*const u16, *mut c_void) -> win::HANDLE)(p.as_ptr(), data)
}

pub extern "system" fn GetFileAttributesW(path: *const u16) -> u32 {
    let Some(p) = for_reading("GetFileAttributesW", &wide(path)) else { return INVALID_ATTRIBUTES };
    host!("kernel32.dll", "GetFileAttributesW": extern "system" fn(*const u16) -> u32)(p.as_ptr())
}

pub extern "system" fn GetFileAttributesA(path: *const c_char) -> u32 {
    let Some(p) = for_reading("GetFileAttributesA", &narrow(path)) else { return INVALID_ATTRIBUTES };
    host!("kernel32.dll", "GetFileAttributesW": extern "system" fn(*const u16) -> u32)(p.as_ptr())
}

pub extern "system" fn GetFileAttributesExW(path: *const u16, level: u32, info: *mut c_void) -> i32 {
    let Some(p) = for_reading("GetFileAttributesExW", &wide(path)) else { return 0 };
    host!("kernel32.dll", "GetFileAttributesExW": extern "system" fn(*const u16, u32, *mut c_void) -> i32)(p.as_ptr(), level, info)
}

/// Copies `name` into the title's buffer of `size` characters as
/// `GetModuleFileName` does: truncated with a NUL and
/// `ERROR_INSUFFICIENT_BUFFER` when it does not fit.
fn put_name<T: Copy + From<u8>>(name: &[T], buffer: *mut T, size: u32) -> u32 {
    let size = size as usize;
    if buffer.is_null() || size == 0 {
        fail(ERROR_INSUFFICIENT_BUFFER);
        return 0;
    }
    let n = name.len().min(size - 1);
    // SAFETY: the title passes room for `size` characters.
    unsafe {
        buffer.copy_from_nonoverlapping(name.as_ptr(), n);
        buffer.add(n).write(T::from(0));
    }
    if n < name.len() {
        fail(ERROR_INSUFFICIENT_BUFFER);
        return size as u32;
    }
    n as u32
}

/// A title image's console path: `G:\<file>`.
fn image_path(module: win::HMODULE) -> Option<String> {
    let base = if module.is_null() { crate::title_base() } else { module as usize };
    crate::images::image_file(base).map(|f| format!(r"G:\{f}"))
}

pub extern "system" fn GetModuleFileNameW(module: win::HMODULE, buffer: *mut u16, size: u32) -> u32 {
    match image_path(module) {
        Some(path) => put_name(&path.encode_utf16().collect::<Vec<u16>>(), buffer, size),
        // SAFETY: the title's own arguments, to the host's function.
        None => unsafe { win::GetModuleFileNameW(module, buffer, size) },
    }
}

pub extern "system" fn GetModuleFileNameA(module: win::HMODULE, buffer: *mut u8, size: u32) -> u32 {
    match image_path(module) {
        Some(path) => put_name(path.as_bytes(), buffer, size),
        None => host!("kernel32.dll", "GetModuleFileNameA": extern "system" fn(win::HMODULE, *mut u8, u32) -> u32)(module, buffer, size),
    }
}

// The CRT's own path functions (they would reach the host's `CreateFileW`
// without passing us).

/// A CRT open mode changes the file system unless it is plain reading.
fn mode_writes(mode: &str) -> bool {
    mode.contains(['w', 'a', '+'])
}

type Wfopen = extern "C" fn(*const u16, *const u16) -> *mut c_void;

pub extern "C" fn _wfopen(path: *const u16, mode: *const u16) -> *mut c_void {
    let host = host!("api-ms-win-crt-stdio-l1-1-0.dll", "_wfopen": Wfopen);
    if path.is_null() || mode.is_null() {
        return host(path, mode);
    }
    let name = wide(path);
    let p = if mode_writes(&wide(mode)) { for_writing("_wfopen", &name) } else { for_reading("_wfopen", &name) };
    match p {
        Some(p) => host(p.as_ptr(), mode),
        None => std::ptr::null_mut(),
    }
}

pub extern "C" fn fopen(path: *const c_char, mode: *const c_char) -> *mut c_void {
    let host = host!("api-ms-win-crt-stdio-l1-1-0.dll", "_wfopen": Wfopen);
    if path.is_null() || mode.is_null() {
        return host!("api-ms-win-crt-stdio-l1-1-0.dll", "fopen": extern "C" fn(*const c_char, *const c_char) -> *mut c_void)(path, mode);
    }
    let name = narrow(path);
    let mode = narrow(mode);
    let p = if mode_writes(&mode) { for_writing("fopen", &name) } else { for_reading("fopen", &name) };
    let mode: Vec<u16> = mode.encode_utf16().chain([0]).collect();
    match p {
        Some(p) => host(p.as_ptr(), mode.as_ptr()),
        None => std::ptr::null_mut(),
    }
}

pub extern "C" fn _wstat64(path: *const u16, stat: *mut c_void) -> i32 {
    let host = host!("api-ms-win-crt-filesystem-l1-1-0.dll", "_wstat64": extern "C" fn(*const u16, *mut c_void) -> i32);
    if path.is_null() {
        return host(path, stat);
    }
    match for_reading("_wstat64", &wide(path)) {
        Some(p) => host(p.as_ptr(), stat),
        None => -1,
    }
}

pub extern "C" fn remove(path: *const c_char) -> i32 {
    if path.is_null() {
        return -1;
    }
    match for_writing("remove", &narrow(path)) {
        Some(p) => host!("api-ms-win-crt-filesystem-l1-1-0.dll", "_wremove": extern "C" fn(*const u16) -> i32)(p.as_ptr()),
        None => -1,
    }
}

/// Tests that set the drives hold this.

// Kernel reads into title memory. The GPU side leaves blocks of title GPU
// buffers read-only to see the title's next write, and the kernel's writes
// fail there instead of faulting into its handler; the title's
// synchronous reads (title+0x313fc0) read straight into 4 KiB-aligned
// destinations and retry a failed `ReadFile` forever. So the destination
// is opened first. (VIctory-base `files.rs` 513-535, xnw/base 5d0516e,
// GPL-3.0-or-later.)

pub extern "system" fn ReadFile(file: *mut c_void, buffer: *mut u8, len: u32, read: *mut u32, overlapped: *mut c_void) -> i32 {
    // Import replay: our work before the host call must not leave its
    // last error (203 from a switch lookup) for a read that succeeds.
    let keep = std::io::Error::last_os_error().raw_os_error().unwrap_or(0) as u32;
    xnw_gpu::d3d12x::kernel_will_write(buffer as usize, len as usize);
    xnw_formats::origin::registry().forget_va(buffer as usize, len as usize);
    // SAFETY: plain call on this thread.
    unsafe { xnw_core::win::SetLastError(keep) };
    host!("kernel32.dll", "ReadFile": extern "system" fn(*mut c_void, *mut u8, u32, *mut u32, *mut c_void) -> i32)(file, buffer, len, read, overlapped)
}

/// The host CRT reads large requests straight into the caller's buffer.
pub extern "C" fn fread(buffer: *mut u8, size: usize, count: usize, stream: *mut c_void) -> usize {
    xnw_gpu::d3d12x::kernel_will_write(buffer as usize, size.saturating_mul(count));
    xnw_formats::origin::registry().forget_va(buffer as usize, size.saturating_mul(count));
    host!("api-ms-win-crt-stdio-l1-1-0.dll", "fread": extern "C" fn(*mut u8, usize, usize, *mut c_void) -> usize)(buffer, size, count, stream)
}

#[cfg(test)]
pub(crate) static TEST_DRIVES: Mutex<()> = Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_paths_land_in_the_package_or_the_overlay() {
        let _drives = TEST_DRIVES.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("xnw-files-{}", std::process::id()));
        let package = dir.join("pkg");
        std::fs::create_dir_all(&package).unwrap();
        std::fs::write(package.join("a.rpf"), b"package").unwrap();
        set_game_drive(std::slice::from_ref(&package));
        set_overlay(&dir.join("data"));

        assert_eq!(place(r"G:\data\..\a.rpf"), Place::Game(package.join("a.rpf")));
        assert_eq!(place("g:/x/./y"), Place::Game(package.join("x").join("y")));
        assert_eq!(place(r"\\?\G:\a"), Place::Game(package.join("a")));
        assert_eq!(place("common.rpf"), Place::Game(package.join("common.rpf")));
        assert_eq!(place(r"T:\tmp\..\..\z"), Place::Overlay(dir.join("data").join("T").join("z")));
        assert_eq!(place(r"\\server\share\x"), Place::Refused);
        assert_eq!(place(r"\\.\PhysicalDrive0"), Place::Refused);

        let w = |s: &str| s.encode_utf16().chain([0]).collect::<Vec<u16>>();
        let null = std::ptr::null_mut();
        // The package reads, and refuses every change.
        let h = CreateFileW(w(r"G:\a.rpf").as_ptr(), 0x8000_0000, 1, std::ptr::null(), OPEN_EXISTING, 0, null);
        assert_ne!(h, INVALID_HANDLE);
        // SAFETY: our handle.
        unsafe { win::CloseHandle(h) };
        assert_eq!(CreateFileW(w(r"G:\a.rpf").as_ptr(), 0x4000_0000, 0, std::ptr::null(), OPEN_EXISTING, 0, null), INVALID_HANDLE);
        assert_eq!(win::last_error().raw_os_error(), Some(ERROR_ACCESS_DENIED as i32));
        assert_eq!(CreateFileW(w(r"G:\new.bin").as_ptr(), 0x8000_0000, 0, std::ptr::null(), 2, 0, null), INVALID_HANDLE);
        assert_eq!(DeleteFileW(w(r"G:\a.rpf").as_ptr()), 0);
        assert_eq!(CreateDirectoryW(w(r"G:\d").as_ptr(), std::ptr::null()), 0);
        assert!(fopen(c"G:\\a.rpf".as_ptr(), c"wb".as_ptr()).is_null());
        assert_eq!(remove(c"G:\\a.rpf".as_ptr()), -1);
        assert_eq!(std::fs::read(package.join("a.rpf")).unwrap(), b"package");
        assert!(!package.join("new.bin").exists() && !package.join("d").exists());

        // The overlay takes writes.
        let h = CreateFileW(w(r"T:\made.bin").as_ptr(), 0x4000_0000, 0, std::ptr::null(), 2, 0, null);
        assert_ne!(h, INVALID_HANDLE);
        // SAFETY: our handle.
        unsafe { win::CloseHandle(h) };
        assert!(dir.join("data").join("T").join("made.bin").is_file());
        assert_eq!(GetFileAttributesW(w(r"\\server\share").as_ptr()), INVALID_ATTRIBUTES);
        assert_eq!(win::last_error().raw_os_error(), Some(ERROR_BAD_NETPATH as i32));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn module_names_fit_or_truncate() {
        let mut small = [0u16; 4];
        let name: Vec<u16> = r"G:\game.exe".encode_utf16().collect();
        assert_eq!(put_name(&name, small.as_mut_ptr(), 4), 4);
        assert_eq!(small, [b'G' as u16, b':' as u16, b'\\' as u16, 0]);
        let mut big = [0u16; 64];
        assert_eq!(put_name(&name, big.as_mut_ptr(), 64), name.len() as u32);
    }
}
