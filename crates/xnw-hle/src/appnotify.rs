//! `api-ms-win-core-psm-appnotify-l1-1-{0,1}`: suspend/resume and
//! constrained-mode notifications.
//!
//! Adapted from VIctory, `crates/victory-hle/src/modules/lifecycle.rs` at
//! d2a4bae (GPL-3.0-or-later). The routines are kept; nothing raises them
//! yet (a title that is never suspended or constrained never hears).

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// `routine(state, context)`: suspended/quiesced or constrained, as a byte.
type Routine = extern "system" fn(u8, *mut c_void);

/// Which notification a registration is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    State,
    Constrained,
}

/// Registration id, kind, routine and context.
static ROUTINES: Mutex<Vec<(usize, Kind, usize, usize)>> = Mutex::new(Vec::new());

const ERROR_INVALID_PARAMETER: u32 = 87;

fn register(kind: Kind, routine: Option<Routine>, context: *mut c_void, registration: *mut usize) -> u32 {
    let Some(routine) = routine else { return ERROR_INVALID_PARAMETER };
    if registration.is_null() {
        return ERROR_INVALID_PARAMETER;
    }
    static NEXT: AtomicUsize = AtomicUsize::new(1);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    ROUTINES.lock().unwrap_or_else(|e| e.into_inner()).push((id, kind, routine as usize, context as usize));
    xnw_core::info!("{kind:?} change notification {id} registered from {}", crate::images::caller());
    // SAFETY: checked non-null; the title passes room for its registration.
    unsafe { registration.write(id) };
    0
}

fn unregister(registration: usize) {
    ROUTINES.lock().unwrap_or_else(|e| e.into_inner()).retain(|r| r.0 != registration);
}

pub extern "system" fn RegisterAppStateChangeNotification(routine: Option<Routine>, context: *mut c_void, registration: *mut usize) -> u32 {
    register(Kind::State, routine, context, registration)
}

pub extern "system" fn UnregisterAppStateChangeNotification(registration: usize) {
    unregister(registration)
}

pub extern "system" fn RegisterAppConstrainedChangeNotification(
    routine: Option<Routine>,
    context: *mut c_void,
    registration: *mut usize,
) -> u32 {
    register(Kind::Constrained, routine, context, registration)
}

pub extern "system" fn UnregisterAppConstrainedChangeNotification(registration: usize) {
    unregister(registration)
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "system" fn noop(_: u8, _: *mut c_void) {}

    #[test]
    fn registrations_come_and_go() {
        let (mut a, mut b) = (0, 0);
        assert_eq!(RegisterAppStateChangeNotification(Some(noop), std::ptr::null_mut(), &mut a), 0);
        assert_eq!(RegisterAppConstrainedChangeNotification(Some(noop), std::ptr::null_mut(), &mut b), 0);
        assert_eq!(RegisterAppStateChangeNotification(None, std::ptr::null_mut(), &mut a), ERROR_INVALID_PARAMETER);
        UnregisterAppStateChangeNotification(a);
        let left: Vec<_> = ROUTINES.lock().unwrap().iter().map(|r| r.0).filter(|&id| id == a || id == b).collect();
        assert_eq!(left, [b]);
        UnregisterAppConstrainedChangeNotification(b);
    }
}
