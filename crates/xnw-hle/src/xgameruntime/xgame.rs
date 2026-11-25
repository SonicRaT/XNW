//! The title's own id (from its manifest), which runtime areas exist, and
//! game invites and events. Offline no invite arrives and events go
//! nowhere.
//!
//! Ported from VIctory, `crates/victory-hle/src/modules/xgame.rs` at d2a4bae
//! (GPL-3.0-or-later). `XLaunchNewGame` (title slot 0x20) is not here yet:
//! it names itself.

use std::ffi::{CStr, c_char, c_void};

use super::{Guid, HRESULT, Object, S_OK, Table, methods, put, token};

/// Title id and launching another title.
pub const TITLE_PROVIDER: Guid = Guid::new(0x973a_344e, 0x24bf, 0x4d0f, [0x84, 0x57, 0x56, 0xc5, 0x34, 0x89, 0x2b, 0x29]);
/// As this title asks for it (`title+0x55dd264`).
pub const TITLE_IID: Guid = Guid::new(0x2549_f142, 0x6419, 0x4a06, [0x97, 0xb5, 0x93, 0x1a, 0xab, 0x7c, 0x2f, 0x34]);
/// Feature checks; the interface id is the provider's, as for the next two.
pub const FEATURES: Guid = Guid::new(0x8836_fe87, 0xedb9, 0x4fe3, [0x8d, 0xad, 0x05, 0xf0, 0xd2, 0xcd, 0x5b, 0x40]);
pub const INVITES: Guid = Guid::new(0x0651_aae2, 0x4012, 0x4077, [0xbf, 0x84, 0x8b, 0x90, 0x97, 0x09, 0x0e, 0x2c]);
pub const EVENTS: Guid = Guid::new(0xbbfb_dcc7, 0xbfe7, 0x409b, [0xa5, 0xca, 0xed, 0xf0, 0x54, 0x96, 0x0b, 0x4d]);

pub static TITLE_OBJECT: Object = Object::new("title", TITLE_PROVIDER, TITLE_IID, &TITLE_VTABLE);
pub static FEATURES_OBJECT: Object = Object::new("features", FEATURES, FEATURES, &FEATURES_VTABLE);
pub static INVITES_OBJECT: Object = Object::new("invites", INVITES, INVITES, &INVITES_VTABLE);
pub static EVENTS_OBJECT: Object = Object::new("events", EVENTS, EVENTS, &EVENTS_VTABLE);

type This = *const Object;

extern "system" fn title_id(_: This, out: *mut u32) -> HRESULT {
    put(out, crate::title_id())
}

/// Runtime areas by number, in the public feature list's order, that we
/// serve: accessibility (0), app capture (1), async calls and providers
/// (2, 3), display (4), game (5), invites (6), game saves (7), game UI (8),
/// launcher (9), networking (10), packages (11), persistent local storage
/// (12), speech synthesis (13), store (14), system (15), task queues (16),
/// users (18), error reports (19), game events (20). Not threads (17) or
/// game streaming (21).
const AVAILABLE: [u32; 20] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20];

extern "system" fn is_feature_available(_: This, feature: u32) -> bool {
    AVAILABLE.contains(&feature)
}

/// Nothing raises these events offline, so the callback never runs.
extern "system" fn register(_: This, _queue: *const c_void, _context: *mut c_void, _callback: *const c_void, out: *mut u64) -> HRESULT {
    put(out, token())
}
extern "system" fn unregister(_: This, _token: u64, _wait: bool) -> bool {
    true
}

/// Accepted and dropped.
extern "system" fn write_event(
    _: This,
    _user: *const c_void,
    _service_config: *const c_char,
    _session: *const c_char,
    name: *const c_char,
    _dimensions: *const c_char,
    _measurements: *const c_char,
) -> HRESULT {
    if !name.is_null() {
        // SAFETY: a non-null name is a NUL-terminated string.
        let name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
        xnw_core::trace!("game event {name} dropped");
    }
    S_OK
}

static TITLE_VTABLE: Table = methods(&[(3, title_id as *const ())]);
static FEATURES_VTABLE: Table = methods(&[(3, is_feature_available as *const ())]);
static INVITES_VTABLE: Table = methods(&[(3, register as *const ()), (4, unregister as *const ())]);
static EVENTS_VTABLE: Table = methods(&[(3, write_event as *const ())]);
