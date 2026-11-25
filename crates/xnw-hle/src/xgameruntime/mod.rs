//! `xgameruntime.dll`: the console game runtime.
//!
//! Adapted from VIctory, `crates/victory-hle/src/modules/xgameruntime.rs` at
//! d2a4bae (GPL-3.0-or-later). The title does not import the runtime: a
//! stub linked into it (`title+0x55e2a70`) loads the DLL by name (answered
//! with a pseudo handle, see `libloader`), takes the six entry points with
//! `GetProcAddress` and initialises. Each runtime API call then asks
//! `QueryApiImpl` for a COM-style object of its area (provider and
//! interface GUIDs), calls one method and releases it. An area's interface
//! id changes when its method table does, so each object answers only the
//! ids whose slots we know. Unknown pairs fail (the title's thunk returns
//! that error), logged once with their caller; a slot we lack names itself
//! (area, provider, interface, slot, caller) and, under the trap policy,
//! ends the run as an unimplemented import does.

use std::ffi::{CStr, c_char, c_void};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use xnw_core::win::{GetCurrentProcess, TerminateProcess};

use crate::images::{caller, log_stack, trap};

pub mod storage;
pub mod xasync;
pub mod xgame;
pub mod xgamesave;
pub mod xnetworking;
pub mod xsystem;
pub mod xuser;

pub type HRESULT = i32;
pub const S_OK: HRESULT = 0;
pub const E_NOTIMPL: HRESULT = 0x8000_4001_u32 as i32;
pub const E_NOINTERFACE: HRESULT = 0x8000_4002_u32 as i32;
pub const E_INVALIDARG: HRESULT = 0x8007_0057_u32 as i32;
/// `HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED)`.
pub const NOT_SUPPORTED: HRESULT = 0x8007_0032_u32 as i32;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Guid {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}

impl Guid {
    pub const fn new(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> Guid {
        Guid { data1, data2, data3, data4 }
    }
}

impl std::fmt::Display for Guid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let d = self.data4;
        write!(
            f,
            "{:08x}-{:04x}-{:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            self.data1, self.data2, self.data3, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]
        )
    }
}

/// Method slots per object: `QueryInterface`, `AddRef`, `Release`, then the
/// area's methods from offset 0x18.
pub const SLOTS: usize = 128;

/// A method table: every slot a naming stub until set.
pub type Methods = [*const (); SLOTS];

/// A runtime object as title code calls it: its first word points at the
/// method table. Ours are static, so reference counting is a no-op.
#[repr(C)]
pub struct Object {
    vtable: &'static Table,
    /// Named in the log when a title calls a method we lack.
    pub area: &'static str,
    pub provider: Guid,
    pub iid: Guid,
}

/// [`Methods`] that can live in a static.
#[repr(transparent)]
pub struct Table(pub Methods);

// SAFETY: function addresses only.
unsafe impl Sync for Table {}
// SAFETY: a static table pointer, a name and two GUIDs.
unsafe impl Sync for Object {}

impl Object {
    pub const fn new(area: &'static str, provider: Guid, iid: Guid, vtable: &'static Table) -> Object {
        Object { vtable, area, provider, iid }
    }
}

/// `QueryInterface`: our objects answer only for themselves.
extern "system" fn query_interface(_: *const Object, _: *const Guid, out: *mut *mut c_void) -> HRESULT {
    if !out.is_null() {
        // SAFETY: the title passes room for a pointer.
        unsafe { out.write(std::ptr::null_mut()) };
    }
    E_NOINTERFACE
}

/// `AddRef` and `Release`.
extern "system" fn add_ref(_: *const Object) -> u32 {
    1
}

/// A slot nobody has needed yet: names itself and the caller; ends the run
/// under the trap policy, else returns `E_NOTIMPL` (wrong for a slot that
/// returns `bool` or a value).
extern "system" fn missing<const SLOT: usize>(this: *const Object) -> HRESULT {
    // SAFETY: title code calls methods on objects we handed out.
    let o = unsafe { &*this };
    xnw_core::warn!(
        "UNIMPLEMENTED runtime method {} slot {SLOT} (+{:#x}), provider {} iid {}, called from {}",
        o.area,
        SLOT * 8,
        o.provider,
        o.iid,
        caller()
    );
    if trap() {
        log_stack();
        xnw_core::warn!("run ended at the first unimplemented runtime method (policy: trap)");
        // SAFETY: ends our own process; nothing is left to unwind.
        unsafe { TerminateProcess(GetCurrentProcess(), 71) };
    }
    E_NOTIMPL
}

macro_rules! missing_row {
    ($r:literal) => {
        [
            missing::<{ $r * 16 }> as *const (),
            missing::<{ $r * 16 + 1 }> as *const (),
            missing::<{ $r * 16 + 2 }> as *const (),
            missing::<{ $r * 16 + 3 }> as *const (),
            missing::<{ $r * 16 + 4 }> as *const (),
            missing::<{ $r * 16 + 5 }> as *const (),
            missing::<{ $r * 16 + 6 }> as *const (),
            missing::<{ $r * 16 + 7 }> as *const (),
            missing::<{ $r * 16 + 8 }> as *const (),
            missing::<{ $r * 16 + 9 }> as *const (),
            missing::<{ $r * 16 + 10 }> as *const (),
            missing::<{ $r * 16 + 11 }> as *const (),
            missing::<{ $r * 16 + 12 }> as *const (),
            missing::<{ $r * 16 + 13 }> as *const (),
            missing::<{ $r * 16 + 14 }> as *const (),
            missing::<{ $r * 16 + 15 }> as *const (),
        ]
    };
}

const MISSING_ROWS: [[*const (); 16]; SLOTS / 16] =
    [missing_row!(0), missing_row!(1), missing_row!(2), missing_row!(3), missing_row!(4), missing_row!(5), missing_row!(6), missing_row!(7)];

/// A method table with `IUnknown` and the given `(slot, function)` pairs;
/// every other slot names itself when called.
pub const fn methods(set: &[(usize, *const ())]) -> Table {
    let mut t = [std::ptr::null::<()>(); SLOTS];
    let mut i = 0;
    while i < SLOTS {
        t[i] = MISSING_ROWS[i / 16][i % 16];
        i += 1;
    }
    t[0] = query_interface as *const ();
    t[1] = add_ref as *const ();
    t[2] = add_ref as *const ();
    let mut j = 0;
    while j < set.len() {
        t[set[j].0] = set[j].1;
        j += 1;
    }
    Table(t)
}

/// The objects we hand out, found by `(provider, iid)`.
const OBJECTS: &[&Object] = &[
    &xasync::OBJECT,
    &xuser::OBJECT,
    &xuser::OBJECT_2026,
    &xuser::GAMERTAG_OBJECT,
    &xuser::DEVICE_OBJECT,
    &xsystem::OBJECT,
    &xsystem::OBJECT_2026,
    &xsystem::ANALYTICS_OBJECT,
    &xgame::TITLE_OBJECT,
    &xgame::FEATURES_OBJECT,
    &xgame::INVITES_OBJECT,
    &xgame::EVENTS_OBJECT,
    &xnetworking::OBJECT,
    &xnetworking::OBJECT_2026,
    &storage::OBJECT,
    &storage::OBJECT_2026,
    &xgamesave::OBJECT,
    &xgamesave::OBJECT_2026,
];

/// Writes `value` through `out` if it is non-null.
pub fn put<T>(out: *mut T, value: T) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null; the title passes room for a `T`.
    unsafe { out.write(value) };
    S_OK
}

/// Copies `text` and a NUL into the title's `size`-byte buffer and reports
/// the bytes used, as the runtime's string getters do.
pub fn put_string(text: &str, size: usize, buffer: *mut u8, used: *mut usize) -> HRESULT {
    let len = text.len() + 1;
    if buffer.is_null() || size < len {
        return xasync::E_NOT_SUFFICIENT_BUFFER;
    }
    // SAFETY: the buffer holds `size` bytes, checked above.
    unsafe {
        buffer.copy_from_nonoverlapping(text.as_ptr(), text.len());
        buffer.add(text.len()).write(0);
    }
    if !used.is_null() {
        // SAFETY: checked non-null.
        unsafe { used.write(len) };
    }
    S_OK
}

/// The title's NUL-terminated string, if any.
pub fn text(s: *const c_char) -> Option<String> {
    // SAFETY: the title passes a NUL-terminated string or null.
    (!s.is_null()).then(|| unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned())
}

/// A new registration token for callbacks we never call (events nobody
/// raises offline).
pub fn token() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// The arguments identify the title's build of the stub; not checked.
pub extern "system" fn InitializeApiImpl(version: u64, build: u64) -> HRESULT {
    xnw_core::info!("InitializeApiImpl(version {version:#x}, build {build:#x}) from {}", caller());
    S_OK
}

pub extern "system" fn InitializeApiImplEx(version: u64, build: u64, flags: u64) -> HRESULT {
    xnw_core::info!("InitializeApiImplEx(version {version:#x}, build {build:#x}, flags {flags:#x}) from {}", caller());
    S_OK
}

/// The stub prefers this one when present; `options` is null unless the
/// title initialises with options.
pub extern "system" fn InitializeApiImplEx2(version: u64, build: u64, flags: u64, options: *const c_void) -> HRESULT {
    xnw_core::info!(
        "InitializeApiImplEx2(version {version:#x}, build {build:#x}, flags {flags:#x}, options {options:p}) from {}",
        caller()
    );
    S_OK
}

pub extern "system" fn UninitializeApiImpl() {
    xnw_core::info!("UninitializeApiImpl from {}", caller());
}

/// The object for an implemented (provider, interface id); other pairs
/// fail, each logged once with its first caller.
pub extern "system" fn QueryApiImpl(provider: *const Guid, iid: *const Guid, out: *mut *mut c_void) -> HRESULT {
    static QUERIED: Mutex<Vec<(Guid, Guid)>> = Mutex::new(Vec::new());
    // SAFETY: the title passes pointers to GUIDs (null is tolerated).
    let (Some(&provider), Some(&iid)) = (unsafe { provider.as_ref() }, unsafe { iid.as_ref() }) else {
        return E_INVALIDARG;
    };
    let found = OBJECTS.iter().find(|o| (o.provider, o.iid) == (provider, iid));
    let mut queried = QUERIED.lock().unwrap_or_else(|e| e.into_inner());
    let first = !queried.contains(&(provider, iid));
    if first {
        queried.push((provider, iid));
    }
    drop(queried);
    match found {
        Some(&object) if !out.is_null() => {
            if first {
                xnw_core::info!("QueryApiImpl({provider}, {iid}): {}; from {}", object.area, caller());
            }
            // SAFETY: checked non-null; the title passes room for a pointer.
            unsafe { out.write(std::ptr::from_ref(object).cast_mut().cast()) };
            S_OK
        }
        _ => {
            if first {
                xnw_core::warn!("QueryApiImpl({provider}, {iid}): area not implemented; from {}", caller());
            }
            NOT_SUPPORTED
        }
    }
}

pub extern "system" fn XErrorReport(hr: HRESULT, message: *const c_char) {
    let message = if message.is_null() {
        "".into()
    } else {
        // SAFETY: a non-null message is a NUL-terminated string.
        unsafe { CStr::from_ptr(message) }.to_string_lossy()
    };
    xnw_core::warn!("XErrorReport({:#010x}, \"{message}\") from {}", hr as u32, caller());
}

/// Method `slot` (byte offset) of `object`, as title code calls it.
#[cfg(test)]
pub(crate) fn method<F: Copy>(object: &Object, slot: usize) -> F {
    // SAFETY: the caller names the slot's signature.
    unsafe { std::mem::transmute_copy(&object.vtable.0[slot / 8]) }
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_TABLE: Table = methods(&[(3, add_ref as *const ())]);

    #[test]
    fn tables_keep_iunknown_and_set_slots() {
        assert_eq!(TEST_TABLE.0[1], add_ref as *const ());
        assert_eq!(TEST_TABLE.0[3], add_ref as *const ());
        assert_eq!(TEST_TABLE.0[4], missing::<4> as *const ());
        assert_eq!(TEST_TABLE.0[SLOTS - 1], missing::<{ SLOTS - 1 }> as *const ());
    }

    #[test]
    fn unknown_areas_fail_without_touching_out() {
        let g = Guid::new(1, 2, 3, [4; 8]);
        let untouched = 0x1234 as *mut c_void;
        let mut out = untouched;
        assert!(QueryApiImpl(&g, &g, &mut out) < 0);
        assert_eq!(out, untouched);
    }
}
