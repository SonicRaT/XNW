//! XUser: one offline local user, signed in, never a guest, with every
//! privilege. Served as the user provider's objects from `QueryApiImpl`;
//! adding a user completes through `xasync` like any other async call. No
//! input device is associated with the user (there are none, `gameinput`).
//!
//! Ported from VIctory, `crates/victory-hle/src/modules/xuser.rs` at d2a4bae
//! (GPL-3.0-or-later). This title asks with the provider's id as the
//! interface id (`title+0x55dcd58`).

use std::ffi::{c_char, c_void};

use super::xasync::{self, AsyncBlock, ProviderData};
use super::{Guid, HRESULT, Object, S_OK, Table, methods, put, put_string, text, token};

/// The user provider.
pub const PROVIDER: Guid = Guid {
    data1: 0x01ac_d177,
    data2: 0x91f9,
    data3: 0x4763,
    data4: [0xa3, 0x8e, 0xcc, 0xbb, 0x55, 0xce, 0x32, 0xe0],
};

/// Its main interface.
pub const IID: Guid = Guid {
    data1: 0xeb9b_f948,
    data2: 0x18dc,
    data3: 0x4d82,
    data4: [0xbb, 0xcc, 0x40, 0xe0, 0xa8, 0x09, 0xc4, 0xc0],
};

/// The interface titles ask for to read a gamertag.
pub const GAMERTAG_IID: Guid = Guid {
    data1: 0xcef4_fac0,
    data2: 0x7676,
    data3: 0x4a94,
    data4: [0xa1, 0x19, 0x4c, 0x43, 0xf9, 0xeb, 0x5b, 0x74],
};

/// The user's devices and audio endpoints.
pub const DEVICE_PROVIDER: Guid = Guid::new(
    0x7d82_4997,
    0x10dc,
    0x45ab,
    [0x86, 0xb7, 0x27, 0x37, 0x76, 0x7c, 0x0b, 0xf1],
);

/// The device interface as Minecraft (2022) asks for it.
pub const DEVICE_IID: Guid = Guid::new(
    0x0cc6_a956,
    0xe7e1,
    0x4fdf,
    [0x93, 0x41, 0x9d, 0x5d, 0xa6, 0x49, 0xeb, 0xc8],
);

/// The documented "no such user" error.
pub const USER_NOT_FOUND: HRESULT = 0x8924_5104_u32 as i32;

pub const XUID: u64 = 0x0009_0000_0000_0001;
pub const LOCAL_ID: u64 = 1;
const GAMERTAG: &str = "Player";

/// The only user; a handle is its address, duplicating and closing are free.
pub struct User;
static USER: User = User;
type Handle = *const User;

fn user() -> Handle {
    &USER
}

fn found(found: bool, out: *mut Handle) -> HRESULT {
    if found {
        put(out, user())
    } else {
        USER_NOT_FOUND
    }
}

pub static OBJECT: Object = Object::new("user", PROVIDER, IID, &VTABLE);
/// The same object under the provider's id, as this title asks for it
/// (VIctory: same slots, same arguments).
pub static OBJECT_2026: Object = Object::new("user", PROVIDER, PROVIDER, &VTABLE);
pub static GAMERTAG_OBJECT: Object = Object::new("user gamertag", PROVIDER, GAMERTAG_IID, &GAMERTAG_VTABLE);
pub static DEVICE_OBJECT: Object = Object::new("user devices", DEVICE_PROVIDER, DEVICE_IID, &DEVICE_VTABLE);

type This = *const Object;
extern "system" fn duplicate(_: This, _: Handle, out: *mut Handle) -> HRESULT {
    put(out, user())
}
extern "system" fn close(_: This, _: Handle) {}
extern "system" fn compare(_: This, a: Handle, b: Handle) -> i32 {
    i32::from(a != b)
}

/// The async provider for adding a user: done at once, the result is a
/// handle.
extern "system" fn add_provider(op: u32, data: *const ProviderData) -> HRESULT {
    const BEGIN: u32 = 0;
    const GET_RESULT: u32 = 2;
    // SAFETY: the async runtime passes valid provider data.
    let data = unsafe { &*data };
    match op {
        BEGIN => xasync::XAsyncComplete(data.block, S_OK, size_of::<Handle>()),
        GET_RESULT => return put(data.buffer.cast::<Handle>(), user()),
        _ => {}
    }
    S_OK
}

/// Tells our add calls apart from other async calls on the same block.
static ADD: u8 = 0;

/// Any options add the one user, silently.
extern "system" fn add_async(_: This, options: u32, block: *mut AsyncBlock) -> HRESULT {
    xnw_core::info!("XUserAddAsync(options {options:#x}): the offline local user");
    xasync::XAsyncBegin(
        block,
        std::ptr::null_mut(),
        (&raw const ADD).cast(),
        std::ptr::null(),
        Some(add_provider),
    )
}
extern "system" fn add_result(_: This, block: *mut AsyncBlock, out: *mut Handle) -> HRESULT {
    xasync::XAsyncGetResult(
        block,
        (&raw const ADD).cast(),
        size_of::<Handle>(),
        out.cast(),
        std::ptr::null_mut(),
    )
}
extern "system" fn get_local_id(_: This, _: Handle, out: *mut u64) -> HRESULT {
    put(out, LOCAL_ID)
}
extern "system" fn find_by_local_id(_: This, id: u64, out: *mut Handle) -> HRESULT {
    found(id == LOCAL_ID, out)
}
extern "system" fn get_id(_: This, _: Handle, out: *mut u64) -> HRESULT {
    put(out, XUID)
}
extern "system" fn find_by_id(_: This, id: u64, out: *mut Handle) -> HRESULT {
    found(id == XUID, out)
}
extern "system" fn get_is_guest(_: This, _: Handle, out: *mut bool) -> HRESULT {
    put(out, false)
}
/// 0: signed in.
extern "system" fn get_state(_: This, _: Handle, out: *mut u32) -> HRESULT {
    put(out, 0)
}
/// The user's gamertag as the component asked for (`XUserGamertagComponent`):
/// it has no suffix, so the modern suffix is empty and every other component
/// (classic, modern, unique modern) is the name itself (VIctory d46b078).
extern "system" fn get_gamertag(
    _: This,
    _: Handle,
    component: u32,
    size: usize,
    buffer: *mut u8,
    used: *mut usize,
) -> HRESULT {
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        xnw_core::info!("XUserGetGamertag(component {component}, {size} bytes): {GAMERTAG:?}");
    }
    const MODERN_SUFFIX: u32 = 2;
    let text = match component {
        MODERN_SUFFIX => "",
        _ => GAMERTAG,
    };
    put_string(text, size, buffer, used)
}
/// Granted; deny reason 0 (none).
extern "system" fn check_privilege(
    _: This,
    _: Handle,
    _options: u32,
    _privilege: u32,
    has: *mut bool,
    reason: *mut u32,
) -> HRESULT {
    if !reason.is_null() {
        // SAFETY: checked non-null.
        unsafe { reason.write(0) };
    }
    put(has, true)
}

/// 3: adult.
extern "system" fn get_age_group(_: This, _: Handle, out: *mut u32) -> HRESULT {
    put(out, 3)
}

static RESOLVE: u8 = 0;
static TOKEN: u8 = 0;

/// Every privilege is granted, so there is nothing to resolve.
extern "system" fn resolve_privilege_async(
    _: This,
    _: Handle,
    _options: u32,
    _privilege: u32,
    block: *mut AsyncBlock,
) -> HRESULT {
    xasync::finished(block, (&raw const RESOLVE).cast(), S_OK)
}
extern "system" fn resolve_privilege_result(_: This, block: *mut AsyncBlock) -> HRESULT {
    let none = std::ptr::null_mut();
    xasync::XAsyncGetResult(block, (&raw const RESOLVE).cast(), 0, none, none.cast())
}

/// The documented "failed to get token": tokens come from the service.
pub const FAILED_TO_GET_TOKEN: HRESULT = 0x8924_5111_u32 as i32;

#[allow(clippy::too_many_arguments)]
extern "system" fn token_async(
    _: This,
    _: Handle,
    _options: u32,
    _method: *const c_char,
    url: *const c_char,
    _header_count: usize,
    _headers: *const c_void,
    _body_size: usize,
    _body: *const c_void,
    block: *mut AsyncBlock,
) -> HRESULT {
    xnw_core::info!("XUserGetTokenAndSignatureAsync({}): offline", text(url).unwrap_or_default());
    xasync::finished(block, (&raw const TOKEN).cast(), FAILED_TO_GET_TOKEN)
}
extern "system" fn token_size(_: This, block: *mut AsyncBlock, out: *mut usize) -> HRESULT {
    xasync::XAsyncGetResultSize(block, out)
}
extern "system" fn token_result(
    _: This,
    block: *mut AsyncBlock,
    size: usize,
    buffer: *mut c_void,
    _data: *mut *const c_void,
    used: *mut usize,
) -> HRESULT {
    xasync::XAsyncGetResult(block, (&raw const TOKEN).cast(), size, buffer, used)
}

/// The user never changes and no device is ever associated, so these
/// callbacks are never called.
extern "system" fn register(
    _: This,
    _queue: *const c_void,
    _context: *mut c_void,
    _callback: *const c_void,
    out: *mut u64,
) -> HRESULT {
    put(out, token())
}
extern "system" fn unregister(_: This, _token: u64, _wait: bool) -> bool {
    true
}

/// No input device belongs to a user.
extern "system" fn find_for_device(_: This, _device: *const c_void, _: *mut Handle) -> HRESULT {
    USER_NOT_FOUND
}

/// Slot 6, maximum users: one offline local user. After VIctory
/// (b4d31d6, GPL-3.0-or-later): on Windows 10 the title calls it right
/// after `XUserAddAsync` and reads the output whatever the result, the
/// null read at title+0xb5a5f8 in their report (I: argument order theirs,
/// not checked against our build's thunk).
extern "system" fn get_max_users(_: This, _: Handle, out: *mut u32) -> HRESULT {
    put(out, 1)
}

// Unset slots (no title seen calls them) name themselves: gamer pictures, the UTF-16 token calls, issue resolution.
static VTABLE: Table = methods(&[
    (3, duplicate as *const ()),
    (4, close as *const ()),
    (5, compare as *const ()),
    (6, get_max_users as *const ()),
    (7, add_async as *const ()),
    (8, add_result as *const ()),
    (9, get_local_id as *const ()),
    (10, find_by_local_id as *const ()),
    (11, get_id as *const ()),
    (12, find_by_id as *const ()),
    (13, get_is_guest as *const ()),
    (14, get_state as *const ()),
    (15, get_gamertag as *const ()), // older gamertag slot
    (19, get_age_group as *const ()),
    (20, check_privilege as *const ()),
    (21, resolve_privilege_async as *const ()), // with UI
    (22, resolve_privilege_result as *const ()),
    (23, token_async as *const ()), // token and signature
    (24, token_size as *const ()),
    (25, token_result as *const ()),
    (33, register as *const ()), // change event
    (34, unregister as *const ()),
]);

static GAMERTAG_VTABLE: Table = methods(&[(3, get_gamertag as *const ())]);

static DEVICE_VTABLE: Table = methods(&[
    (3, find_for_device as *const ()),
    (4, register as *const ()), // device association changed
    (5, unregister as *const ()),
]);

#[cfg(test)]
mod tests {
    use super::*;
    use super::xasync::E_NOT_SUFFICIENT_BUFFER;
    use super::super::method as slot;

    #[test]
    fn a_gamertag_without_a_suffix_has_an_empty_modern_suffix() {
        type Gamertag = extern "system" fn(This, Handle, u32, usize, *mut u8, *mut usize) -> HRESULT;
        let this: This = &OBJECT;
        let user = std::ptr::null();
        // Classic, modern, modern suffix, unique modern; through both slots.
        let expected: [&[u8]; 4] = [b"Player\0", b"Player\0", b"\0", b"Player\0"];
        for (object, at) in [(&GAMERTAG_OBJECT, 0x18), (&OBJECT, 0x78)] {
            let get = slot::<Gamertag>(object, at);
            for (component, expected) in (0..).zip(expected) {
                let (mut name, mut used) = ([0xff_u8; 16], 0);
                let hr = get(this, user, component, 16, name.as_mut_ptr(), &mut used);
                assert_eq!(hr, S_OK);
                assert_eq!(&name[..used], expected, "component {component}");
            }
        }
    }

    #[test]
    fn startup_adds_the_user_and_reads_its_id() {
        type Add = extern "system" fn(This, u32, *mut AsyncBlock) -> HRESULT;
        type AddResult = extern "system" fn(This, *mut AsyncBlock, *mut Handle) -> HRESULT;
        type GetId = extern "system" fn(This, Handle, *mut u64) -> HRESULT;
        type Gamertag =
            extern "system" fn(This, Handle, u32, usize, *mut u8, *mut usize) -> HRESULT;
        let this: This = &OBJECT;
        let mut block = AsyncBlock {
            queue: std::ptr::null(),
            context: std::ptr::null_mut(),
            callback: None,
            internal: [0; 4],
        };
        assert_eq!(slot::<Add>(&OBJECT, 0x38)(this, 1, &mut block), S_OK);
        assert_eq!(xasync::XAsyncGetStatus(&mut block, true), S_OK);
        let mut handle = std::ptr::null();
        assert_eq!(
            slot::<AddResult>(&OBJECT, 0x40)(this, &mut block, &mut handle),
            S_OK
        );
        assert!(!handle.is_null());
        let mut xuid = 0;
        assert_eq!(slot::<GetId>(&OBJECT, 0x58)(this, handle, &mut xuid), S_OK);
        assert_eq!(xuid, XUID);

        let mut name = [0u8; 16];
        let mut used = 0;
        let get = slot::<Gamertag>(&GAMERTAG_OBJECT, 0x18);
        assert_eq!(get(this, handle, 0, 16, name.as_mut_ptr(), &mut used), S_OK);
        assert_eq!(&name[..used], b"Player\0");
        assert_eq!(
            get(this, handle, 0, 2, name.as_mut_ptr(), &mut used),
            E_NOT_SUFFICIENT_BUFFER
        );

        type Age = extern "system" fn(This, Handle, *mut u32) -> HRESULT;
        let mut age = 0;
        assert_eq!(slot::<Age>(&OBJECT, 0x98)(this, handle, &mut age), S_OK);
        assert_eq!(age, 3, "adult");
        type Token = extern "system" fn(
            This,
            Handle,
            u32,
            *const c_char,
            *const c_char,
            usize,
            *const c_void,
            usize,
            *const c_void,
            *mut AsyncBlock,
        ) -> HRESULT;
        let (null, url) = (std::ptr::null(), c"https://example.invalid/");
        let token = slot::<Token>(&OBJECT, 0xb8);
        assert_eq!(
            token(
                this,
                handle,
                0,
                c"GET".as_ptr(),
                url.as_ptr(),
                0,
                null,
                0,
                null,
                &mut block
            ),
            S_OK
        );
        assert_eq!(
            xasync::XAsyncGetStatus(&mut block, true),
            FAILED_TO_GET_TOKEN
        );
    }
}
