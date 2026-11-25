//! The registry the title sees: one key, the console's `ConsoleMode`.
//!
//! The title does not ask the runtime what it runs on: code linked into it
//! (`title+0x55deb88`, the GDK's inline device-type read) opens
//! `HKLM\Software\Microsoft\Windows NT\CurrentVersion\OEM` with
//! `RegOpenKeyW`, reads the DWORD `ConsoleMode` with `RegQueryValueExW`
//! and maps it (0x60 -> 6 Series S, 0x64 -> 7 Series X; no key -> 1, PC).
//! Those are its only registry imports. We answer them for the console
//! chosen by `XNW_CONSOLE` ([`xnw_gpu::d3d12x::console`]); every other key
//! and value fails with `ERROR_FILE_NOT_FOUND`, so nothing of the host's
//! registry reaches the title.
//!
//! After VIctory `crates/victory-hle/src/modules/xsystem.rs` (`device_type`)
//! at d2a4bae (GPL-3.0-or-later). Changed: no host pass-through.

use std::ffi::c_void;

use xnw_gpu::d3d12x::console;

type Key = *mut c_void;

const SUCCESS: u32 = 0;
const FILE_NOT_FOUND: u32 = 2;
const INVALID_HANDLE: u32 = 6;
const INVALID_PARAMETER: u32 = 87;
const MORE_DATA: u32 = 234;
const REG_DWORD: u32 = 4;

/// `HKEY_CLASSES_ROOT` (0x80000000) to `HKEY_CURRENT_USER_LOCAL_SETTINGS`
/// (0x80000007), as titles pass them (sign-extended).
fn predefined(key: Key) -> Option<u32> {
    let k = key as usize as u64;
    let low = k as u32;
    ((k >> 32 == 0xffff_ffff || k >> 32 == 0) && (0x8000_0000..=0x8000_0007).contains(&low)).then_some(low)
}
const LOCAL_MACHINE: u32 = 0x8000_0002;

const OEM_PATH: &str = r"software\microsoft\windows nt\currentversion\oem";
const CONSOLE_MODE: &str = "consolemode";

/// Stands for the OEM key; never a host handle value.
static OEM: u8 = 0;
fn oem() -> Key {
    (&raw const OEM).cast_mut().cast()
}

/// A NUL-terminated UTF-16 string, lower case; `None` for null.
fn text(p: *const u16) -> Option<String> {
    if p.is_null() {
        return None;
    }
    // SAFETY: the title passes a NUL-terminated string.
    let s = unsafe {
        let len = (0..).take_while(|&i| *p.add(i) != 0).count();
        String::from_utf16_lossy(std::slice::from_raw_parts(p, len))
    };
    Some(s.to_ascii_lowercase())
}

fn open(root: Key, sub: *const u16, out: *mut Key) -> u32 {
    if out.is_null() {
        return INVALID_PARAMETER;
    }
    let path = text(sub).unwrap_or_default();
    let path = path.trim_matches('\\');
    let result = match predefined(root) {
        // An empty path opens the key itself (a predefined key stays itself).
        _ if path.is_empty() && (root == oem() || predefined(root).is_some()) => Ok(root),
        Some(LOCAL_MACHINE) if path == OEM_PATH => Ok(oem()),
        Some(_) => Err(FILE_NOT_FOUND),
        None if root == oem() => Err(FILE_NOT_FOUND), // the OEM key has no subkeys
        None => Err(INVALID_HANDLE),
    };
    let (key, code) = match result {
        Ok(k) => (k, SUCCESS),
        Err(e) => (std::ptr::null_mut(), e),
    };
    // SAFETY: checked non-null; the title passes room for a key.
    unsafe { out.write(key) };
    xnw_core::info!("registry: open {root:p}\\{path} -> {}", if code == SUCCESS { "ours".to_string() } else { format!("error {code}") });
    code
}

/// `RegOpenKeyW(root, sub, out)`.
pub extern "system" fn RegOpenKeyW(root: Key, sub: *const u16, out: *mut Key) -> u32 {
    open(root, sub, out)
}

/// `RegOpenKeyExW(root, sub, options, access, out)`: not imported by the
/// title; here so a `GetProcAddress` cannot reach the host's either.
pub extern "system" fn RegOpenKeyExW(root: Key, sub: *const u16, _options: u32, _access: u32, out: *mut Key) -> u32 {
    open(root, sub, out)
}

/// `RegQueryValueExW(key, name, reserved, type, data, size)`, with the real
/// API's size rules: no data -> the size out; data without a size ->
/// `ERROR_INVALID_PARAMETER`; too small -> `ERROR_MORE_DATA` and the size
/// out, data untouched.
pub extern "system" fn RegQueryValueExW(key: Key, name: *const u16, reserved: *mut u32, kind: *mut u32, data: *mut u8, size: *mut u32) -> u32 {
    if key != oem() {
        return if predefined(key).is_some() { FILE_NOT_FOUND } else { INVALID_HANDLE };
    }
    if !reserved.is_null() || (!data.is_null() && size.is_null()) {
        return INVALID_PARAMETER;
    }
    if text(name).as_deref() != Some(CONSOLE_MODE) {
        return FILE_NOT_FOUND;
    }
    let c = console::current();
    // SAFETY: the title passes valid or null pointers; `data` holds `*size` bytes.
    unsafe {
        if !kind.is_null() {
            kind.write(REG_DWORD);
        }
        if size.is_null() {
            return SUCCESS;
        }
        let room = size.read();
        size.write(4);
        if data.is_null() {
            return SUCCESS;
        }
        if room < 4 {
            return MORE_DATA;
        }
        data.cast::<u32>().write_unaligned(c.console_mode());
    }
    xnw_core::info!("registry: ConsoleMode = {:#x} ({}, device type {})", c.console_mode(), c.name(), c.device_type());
    SUCCESS
}

/// `RegCloseKey(key)`.
pub extern "system" fn RegCloseKey(key: Key) -> u32 {
    if key == oem() || predefined(key).is_some() { SUCCESS } else { INVALID_HANDLE }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(s: &str) -> Vec<u16> {
        s.encode_utf16().chain([0]).collect()
    }
    const HKLM: Key = 0xffff_ffff_8000_0002_u64 as usize as Key;
    const HKCU: Key = 0xffff_ffff_8000_0001_u64 as usize as Key;

    fn open_oem() -> Key {
        let mut key = std::ptr::null_mut();
        let path = w(r"Software\Microsoft\Windows NT\CurrentVersion\OEM");
        assert_eq!(RegOpenKeyW(HKLM, path.as_ptr(), &mut key), SUCCESS);
        assert_eq!(key, oem());
        key
    }

    #[test]
    fn the_console_key_answers_the_chosen_model() {
        let key = open_oem();
        let name = w("ConsoleMode");
        let (mut kind, mut value, mut size) = (0, 0u32, 4);
        let r = RegQueryValueExW(key, name.as_ptr(), std::ptr::null_mut(), &mut kind, (&raw mut value).cast(), &mut size);
        assert_eq!((r, kind, size), (SUCCESS, REG_DWORD, 4));
        assert_eq!(value, console::current().console_mode());
        // The default model: what the title maps to type 6.
        assert_eq!(console::Console::default().console_mode(), 0x60);
        assert_eq!(RegCloseKey(key), SUCCESS);
        // Case does not matter, as in the registry.
        let mut k2 = std::ptr::null_mut();
        let lower = w(r"software\microsoft\windows nt\currentversion\oem");
        assert_eq!(RegOpenKeyW(HKLM, lower.as_ptr(), &mut k2), SUCCESS);
    }

    #[test]
    fn unknown_keys_and_values_fail() {
        let mut key = 1usize as Key;
        let other = w(r"Software\Microsoft\Windows NT\CurrentVersion");
        assert_eq!(RegOpenKeyW(HKLM, other.as_ptr(), &mut key), FILE_NOT_FOUND);
        assert!(key.is_null());
        let oem_path = w(r"Software\Microsoft\Windows NT\CurrentVersion\OEM");
        assert_eq!(RegOpenKeyW(HKCU, oem_path.as_ptr(), &mut key), FILE_NOT_FOUND);
        let sub = w("Sub");
        assert_eq!(RegOpenKeyW(oem(), sub.as_ptr(), &mut key), FILE_NOT_FOUND);
        assert_eq!(RegOpenKeyW(0x1234usize as Key, sub.as_ptr(), &mut key), INVALID_HANDLE);
        assert_eq!(RegOpenKeyW(HKLM, oem_path.as_ptr(), std::ptr::null_mut()), INVALID_PARAMETER);

        let (mut value, mut size) = (0u32, 4u32);
        let name = w("ProductName");
        let q = |k: Key, n: &[u16], v: *mut u32, s: *mut u32| RegQueryValueExW(k, n.as_ptr(), std::ptr::null_mut(), std::ptr::null_mut(), v.cast(), s);
        assert_eq!(q(open_oem(), &name, &mut value, &mut size), FILE_NOT_FOUND);
        assert_eq!(q(HKLM, &w("ConsoleMode"), &mut value, &mut size), FILE_NOT_FOUND);
        assert_eq!(q(0x1234usize as Key, &w("ConsoleMode"), &mut value, &mut size), INVALID_HANDLE);
        assert_eq!(RegCloseKey(0x1234usize as Key), INVALID_HANDLE);
    }

    #[test]
    fn buffer_sizes_behave_like_the_real_api() {
        let key = open_oem();
        let name = w("ConsoleMode");
        let mut kind = 0;
        // Too small: ERROR_MORE_DATA, the size out, data untouched.
        let (mut value, mut size) = (0xdead_beefu32, 2u32);
        let r = RegQueryValueExW(key, name.as_ptr(), std::ptr::null_mut(), &mut kind, (&raw mut value).cast(), &mut size);
        assert_eq!((r, size, value, kind), (MORE_DATA, 4, 0xdead_beef, REG_DWORD));
        // No data: the size only.
        let mut size = 0;
        let r = RegQueryValueExW(key, name.as_ptr(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), &mut size);
        assert_eq!((r, size), (SUCCESS, 4));
        // Data without a size.
        let r = RegQueryValueExW(key, name.as_ptr(), std::ptr::null_mut(), std::ptr::null_mut(), (&raw mut value).cast(), std::ptr::null_mut());
        assert_eq!(r, INVALID_PARAMETER);
        // Larger than needed: the value and its real size.
        let mut big = [0u8; 16];
        let mut size = 16;
        let r = RegQueryValueExW(key, name.as_ptr(), std::ptr::null_mut(), std::ptr::null_mut(), big.as_mut_ptr(), &mut size);
        assert_eq!((r, size), (SUCCESS, 4));
        assert_eq!(u32::from_le_bytes(big[..4].try_into().unwrap()), console::current().console_mode());
    }
}
