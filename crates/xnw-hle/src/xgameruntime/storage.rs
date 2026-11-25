//! Persistent local storage: a folder of the title's own with the quota its
//! manifest declares, seen by the title as a drive root.
//!
//! Ported from VIctory-base `crates/victory-hle/src/modules/storage.rs`
//! (5d0516e, GPL-3.0-or-later). Changed: the drive is a fixed letter of our
//! overlay (`files.rs`: every drive but `G:` is a folder of `out\data`), so
//! the folder is `out\data\P\`; the base mounts the highest free host letter
//! over its data directory. The title asks for this area at 0.37 s
//! (`title+0x22367b3`); without it the path stays the bare device name
//! and `CreateDirectoryW("settings:")` / `CreateFileW("settings:\settings.xml")`
//! are refused (`title+0x5a17350`, `title+0x1c09796`, at 121 s into a run), so
//! first-run choices are never saved.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::xasync::{self, AsyncBlock, E_ABORT};
use super::{Guid, HRESULT, Object, S_OK, Table, methods, put, put_string};

pub const PROVIDER: Guid = Guid::new(0xf4fa_f4d4, 0x2d04, 0x4fce, [0xb3, 0xe0, 0x47, 0x4a, 0x71, 0x3a, 0x3e, 0x84]);
pub const IID: Guid = Guid::new(0xd294_11df, 0x0794, 0x4553, [0x8b, 0x27, 0x95, 0xfc, 0x02, 0xd0, 0xf7, 0x5d]);
/// As the title (2026) asks for it; it calls 0x20 (path) and 0x28
/// (space) with the same arguments.
pub const IID_2026: Guid = Guid::new(0x41a4_e10c, 0x5a7e, 0x41d9, [0x8c, 0x37, 0x37, 0xbd, 0xe6, 0x2a, 0x07, 0xd6]);

pub static OBJECT: Object = Object::new("persistent local storage", PROVIDER, IID, &VTABLE);
pub static OBJECT_2026: Object = Object::new("persistent local storage", PROVIDER, IID_2026, &VTABLE);

type This = *const Object;

/// The drive the title sees its storage as.
const DRIVE: &str = "P:\\";

/// The quota in bytes (manifest `PersistentLocalStorage/SizeMB`).
static QUOTA: AtomicU64 = AtomicU64::new(0);

/// Sets the quota from the manifest's size in MiB.
pub fn set_quota_mb(mb: u64) {
    QUOTA.store(mb << 20, Ordering::Relaxed);
}

/// The host folder behind the drive.
pub(crate) fn folder() -> PathBuf {
    crate::files::host_path(DRIVE).unwrap_or_else(|| PathBuf::from(r"out\data\P"))
}

/// What the host disk under `dir` has free (unlimited if unknown); used by
/// game saves, as in the base.
pub(crate) fn disk_free(dir: &Path) -> u64 {
    use std::os::windows::ffi::OsStrExt;
    let wide: Vec<u16> = dir.as_os_str().encode_wide().chain([0]).collect();
    let mut free = 0;
    let null = std::ptr::null_mut();
    // SAFETY: a NUL-terminated path and a valid output.
    if unsafe { xnw_core::win::GetDiskFreeSpaceExW(wide.as_ptr(), &mut free, null, null) } != 0 {
        return free;
    }
    u64::MAX
}

extern "system" fn path_size(_: This, out: *mut usize) -> HRESULT {
    put(out, DRIVE.len() + 1)
}

/// A buffer too small for the path fails with `E_FAIL`, the documented
/// Xbox result (a PC gets `ERROR_INSUFFICIENT_BUFFER`) (VIctory 376b3cf).
extern "system" fn path(_: This, size: usize, buffer: *mut u8, used: *mut usize) -> HRESULT {
    const E_FAIL: HRESULT = 0x8000_4005_u32 as i32;
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| xnw_core::info!("persistent local storage: {DRIVE} = {}", folder().display()));
    if size <= DRIVE.len() {
        return E_FAIL;
    }
    put_string(DRIVE, size, buffer, used)
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpaceInfo {
    /// Writable now.
    pub available_free: u64,
    /// Left in the quota.
    pub total_free: u64,
    pub used: u64,
    /// The quota.
    pub total: u64,
}

fn used_bytes(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|e| match e.metadata() {
            Ok(m) if m.is_dir() => used_bytes(&e.path()),
            Ok(m) => m.len(),
            Err(_) => 0,
        })
        .sum()
}

fn space() -> SpaceInfo {
    let total = QUOTA.load(Ordering::Relaxed);
    let used = used_bytes(&folder());
    let free = total.saturating_sub(used);
    // shortcut: the host disk's free space is not checked (the base asks
    // GetDiskFreeSpaceExW); the quota is 2 GiB.
    SpaceInfo { available_free: free, total_free: free, used, total }
}

extern "system" fn space_info(_: This, out: *mut SpaceInfo) -> HRESULT {
    put(out, space())
}

/// Tells our prompts apart from other calls on the same block.
static PROMPT: u8 = 0;

/// There is nobody to ask: granted when the quota has room, else declined.
extern "system" fn prompt_for_space(_: This, bytes: u64, block: *mut AsyncBlock) -> HRESULT {
    let hr = if bytes <= space().total_free { S_OK } else { E_ABORT };
    xasync::finished(block, (&raw const PROMPT).cast(), hr)
}

extern "system" fn prompt_result(_: This, block: *mut AsyncBlock) -> HRESULT {
    let none = std::ptr::null_mut();
    xasync::XAsyncGetResult(block, (&raw const PROMPT).cast(), 0, none, none.cast())
}

static VTABLE: Table = methods(&[
    (3, path_size as *const ()),        // 0x18
    (4, path as *const ()),             // 0x20
    (5, space_info as *const ()),       // 0x28
    (6, prompt_for_space as *const ()), // 0x30
    (7, prompt_result as *const ()),    // 0x38
]);

#[cfg(test)]
mod tests {
    use super::super::method;
    use super::*;

    #[test]
    fn serves_a_drive_root_and_the_quota() {
        type Size = extern "system" fn(This, *mut usize) -> HRESULT;
        type Get = extern "system" fn(This, usize, *mut u8, *mut usize) -> HRESULT;
        type Space = extern "system" fn(This, *mut SpaceInfo) -> HRESULT;
        let this: This = &OBJECT_2026;
        set_quota_mb(2048);
        let mut size = 0;
        assert_eq!(method::<Size>(&OBJECT_2026, 0x18)(this, &mut size), S_OK);
        assert_eq!(size, 4, "P:\\ and its NUL");
        let (mut buffer, mut used) = ([0u8; 4], 0);
        assert_eq!(method::<Get>(&OBJECT_2026, 0x20)(this, 4, buffer.as_mut_ptr(), &mut used), S_OK);
        assert_eq!(method::<Get>(&OBJECT_2026, 0x20)(this, 3, buffer.as_mut_ptr(), &mut used), 0x8000_4005_u32 as i32);
        assert_eq!(&buffer, b"P:\\\0");
        let mut info = SpaceInfo { available_free: 0, total_free: 0, used: 0, total: 0 };
        assert_eq!(method::<Space>(&OBJECT_2026, 0x28)(this, &mut info), S_OK);
        assert_eq!(info.total, 2048 << 20);
        assert_eq!(info.total_free, info.total - info.used);
    }
}
