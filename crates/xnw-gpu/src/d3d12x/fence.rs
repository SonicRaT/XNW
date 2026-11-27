//! Fences: a 64-bit value in memory and the events waiting for it. Queue
//! submissions are waited for, so queue signals move the value on the CPU
//! at once; titles also read it at the fence's GPU address.
//!
//! Ported from VIctory `crates/victory-gpu/src/fence.rs` at d2a4bae
//! (GPL-3.0-or-later). Changed: a null event blocks until the value is
//! reached (VIctory fails it).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use super::object::{self, Done, E_INVALIDARG, FENCE, HRESULT, Iid, Object, S_OK, Slot, Table, This, guid};
use super::resource::PAGEABLE_IIDS;

pub(crate) const FENCE_IIDS: [Iid; 4] = [
    guid(0x0a753dcf, 0xc4d8, 0x4b91, [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76]),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Fence {
    object: Object,
    value: AtomicU64,
    flags: u32,
    /// Events to set once the value reaches theirs (handles as integers).
    waiters: Mutex<Vec<(u64, usize)>>,
    reached: Condvar,
}

fn fence<'a>(this: This) -> &'a Fence {
    // SAFETY: called from fence methods only.
    unsafe { &*this.cast::<Fence>() }
}

/// Device method 0x120.
pub(crate) extern "system" fn create(_: This, initial: u64, flags: u32, iid: *const Iid, out: *mut This) -> HRESULT {
    let fence = Box::new(Fence {
        object: Object::new(&FUNCTIONS),
        value: AtomicU64::new(initial),
        flags,
        waiters: Mutex::new(Vec::new()),
        reached: Condvar::new(),
    });
    object::hand_out(FENCE, fence, iid, &FENCE_IIDS, out)
}

fn set_event(event: usize) {
    // SAFETY: the title passed an event handle.
    let _ = unsafe { windows::Win32::System::Threading::SetEvent(windows::Win32::Foundation::HANDLE(event as *mut c_void)) };
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(FENCE, iid, object::is(iid, &FENCE_IIDS).then_some(this), out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Fence>(this)
}
extern "system" fn completed(this: This) -> u64 {
    fence(this).value.load(Ordering::Acquire)
}

/// A null event blocks until the value is reached (by a CPU signal, or a
/// queue's on another thread).
extern "system" fn set_event_on_completion(this: This, value: u64, event: *mut c_void) -> HRESULT {
    let fence = fence(this);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    if fence.value.load(Ordering::Acquire) >= value {
        if !event.is_null() {
            set_event(event as usize);
        }
        return S_OK;
    }
    if !event.is_null() {
        waiters.push((value, event as usize));
        return S_OK;
    }
    let mut warned = false;
    while fence.value.load(Ordering::Acquire) < value {
        let (w, timeout) = fence.reached.wait_timeout(waiters, Duration::from_secs(5)).unwrap_or_else(|e| e.into_inner());
        waiters = w;
        if timeout.timed_out() && !warned {
            warned = true;
            xnw_core::warn!("d3d12x: fence {this:?} blocked 5 s waiting for {value} (at {})", fence.value.load(Ordering::Relaxed));
        }
    }
    S_OK
}

/// The fence's signal, also what queue signals end in.
pub(crate) extern "system" fn signal(this: This, value: u64) -> HRESULT {
    if this.is_null() {
        return E_INVALIDARG;
    }
    let fence = fence(this);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    fence.value.store(value, Ordering::Release);
    waiters.retain(|&(wanted, event)| {
        let reached = wanted <= value;
        if reached {
            set_event(event);
        }
        !reached
    });
    fence.reached.notify_all();
    S_OK
}

/// PASS1: a queue signal deferred behind a submission sent without a wait
/// (`host::Host::signal_after` took a reference): made, and the reference let go.
pub(crate) fn signal_held(this: This, value: u64) {
    signal(this, value);
    release(this);
}

extern "system" fn creation_flags(this: This) -> u32 {
    fence(this).flags
}
/// Where the value lives (GPU addresses are CPU addresses).
extern "system" fn gpu_address(this: This) -> u64 {
    (&raw const fence(this).value) as u64
}

/// IUnknown 3, object 4, device child 1, fence 3, fence1 1, console 1.
pub(crate) const SLOTS: &[Slot] = &[
    (0, query as *const (), Done, "fence.rs:139"),
    (2, release as *const (), Done, "fence.rs:140"),
    (7, object::parent_device as *const (), Done, "fence.rs:141"),
    (8, completed as *const (), Done, "fence.rs:142"),
    (9, set_event_on_completion as *const (), Done, "fence.rs:143"),
    (10, signal as *const (), Done, "fence.rs:144"),
    (11, creation_flags as *const (), Done, "fence.rs:145"),
    (12, gpu_address as *const (), Done, "fence.rs:146"),
];
pub(crate) const LEN: usize = 24;
static FUNCTIONS: Table<LEN> = object::table::<FENCE, LEN>(true, SLOTS);
