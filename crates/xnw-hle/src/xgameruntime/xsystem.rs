//! XSystem: the console's identity and sandbox, and how analytics describe
//! the device: one offline retail console of the model `XNW_CONSOLE`
//! chose (`xnw_gpu::d3d12x::console`). This runtime has no device-type
//! call: the title reads its type from the registry (`registry.rs`).
//!
//! Ported from VIctory, `crates/victory-hle/src/modules/xsystem.rs` at
//! d2a4bae (GPL-3.0-or-later).

use super::{Guid, HRESULT, Object, Table, methods, put_string};

pub const PROVIDER: Guid = Guid::new(0xe349_bd1a, 0xfc20, 0x4e40, [0xb9, 0x9c, 0x41, 0x78, 0xcc, 0x6b, 0x40, 0x9f]);

/// The interface as older titles ask for it; the table ended before the
/// device id.
pub const IID: Guid = Guid::new(0x6fd7_1f09, 0x7513, 0x49f0, [0x89, 0xbc, 0xbf, 0xaf, 0x5d, 0xf6, 0xf8, 0x52]);

/// The interface as this title asks for it (`title+0x55deaf4`); same three
/// slots, same arguments (VIctory).
pub const IID_2026: Guid = Guid::new(0x1861_cf2e, 0xe18b, 0x4834, [0xa9, 0xf5, 0xb4, 0xa4, 0xe6, 0xef, 0xb4, 0xcf]);

/// Analytics information; its interface id is the provider's.
pub const ANALYTICS: Guid = Guid::new(0xb884_675d, 0xb738, 0x4a9c, [0x81, 0x5d, 0x9a, 0x9a, 0x1e, 0x0c, 0x6c, 0x9b]);

/// 38 characters (39 bytes with the NUL).
const CONSOLE_ID: &str = "00000000000000000000000000000000000001";
/// The retail sandbox.
const SANDBOX: &str = "RETAIL";
/// Base64 of 32 bytes: 44 characters (45 bytes with the NUL).
const DEVICE_ID: &str = "WE5XIG9mZmxpbmUgZGV2aWNlIGlkZW50aXR5IDAwMDE=";

pub static OBJECT: Object = Object::new("system", PROVIDER, IID, &VTABLE);
pub static OBJECT_2026: Object = Object::new("system", PROVIDER, IID_2026, &VTABLE);
pub static ANALYTICS_OBJECT: Object = Object::new("analytics", ANALYTICS, ANALYTICS, &ANALYTICS_VTABLE);

type This = *const Object;

extern "system" fn console_id(_: This, size: usize, buffer: *mut u8, used: *mut usize) -> HRESULT {
    put_string(CONSOLE_ID, size, buffer, used)
}
extern "system" fn sandbox_id(_: This, size: usize, buffer: *mut u8, used: *mut usize) -> HRESULT {
    put_string(SANDBOX, size, buffer, used)
}
/// The device id this title sees (titles see different ones).
extern "system" fn device_id(_: This, size: usize, buffer: *mut u8, used: *mut usize) -> HRESULT {
    put_string(DEVICE_ID, size, buffer, used)
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Version {
    pub major: u16,
    pub minor: u16,
    pub build: u16,
    pub revision: u16,
}

#[repr(C)]
pub struct AnalyticsInfo {
    pub os: Version,
    pub hosting_os: Version,
    /// NUL-terminated.
    pub family: [u8; 64],
    pub form: [u8; 64],
}

const OS: Version = Version { major: 10, minor: 0, build: 22000, revision: 0 };

fn text64(s: &str) -> [u8; 64] {
    let mut out = [0; 64];
    out[..s.len()].copy_from_slice(s.as_bytes());
    out
}

/// A struct this large comes back through the caller's buffer, whose
/// address is also the return value.
extern "system" fn analytics_info(_: This, out: *mut AnalyticsInfo) -> *mut AnalyticsInfo {
    let info = AnalyticsInfo { os: OS, hosting_os: OS, family: text64("Xbox"), form: text64(xnw_gpu::d3d12x::console::current().name()) };
    // SAFETY: the caller passes room for the struct.
    unsafe { out.write(info) };
    out
}

static VTABLE: Table = methods(&[
    (3, console_id as *const ()), // by the older table; this title does not call it
    (4, sandbox_id as *const ()),
    (5, device_id as *const ()),
]);

static ANALYTICS_VTABLE: Table = methods(&[(3, analytics_info as *const ())]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xgameruntime::{S_OK, method};

    #[test]
    fn ids_fit_their_documented_sizes() {
        type Get = extern "system" fn(This, usize, *mut u8, *mut usize) -> HRESULT;
        let mut buf = [0u8; 64];
        let mut used = 0;
        assert_eq!(method::<Get>(&OBJECT_2026, 0x20)(&OBJECT_2026, 64, buf.as_mut_ptr(), &mut used), S_OK);
        assert_eq!(&buf[..used], b"RETAIL\0");
        assert_eq!(method::<Get>(&OBJECT_2026, 0x28)(&OBJECT_2026, 64, buf.as_mut_ptr(), &mut used), S_OK);
        assert_eq!(used, 45);
        assert_eq!(CONSOLE_ID.len(), 38);
    }
}
