//! `gameinput.dll`: input devices from the host. Gamepads come from XInput
//! (host slots 0-3); while none is connected, the keyboard drives one
//! stand-in gamepad (`FALLBACK_BUTTONS`). The keyboard is a device of its own
//! too. A poll thread turns host state into readings (a new one per change)
//! and runs device callbacks on connect and disconnect.
//!
//! Ported from VIctory, `crates/victory-hle/src/modules/gameinput.rs` at
//! d2a4bae (GPL-3.0-or-later), built against this exact title. The title
//! imports one function, by ordinal: `#1`, `GameInputCreate`.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, LazyLock, Mutex, MutexGuard, PoisonError};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use crate::xgameruntime::{E_INVALIDARG, E_NOINTERFACE, E_NOTIMPL, HRESULT, S_OK};

/// The documented errors.
pub const DEVICE_DISCONNECTED: HRESULT = 0x838A_0001_u32 as i32;
pub const DEVICE_NOT_FOUND: HRESULT = 0x838A_0002_u32 as i32;
pub const READING_NOT_FOUND: HRESULT = 0x838A_0003_u32 as i32;
pub const REFERENCE_TOO_OLD: HRESULT = 0x838A_0004_u32 as i32;

/// Input kinds (bits) our devices have.
const KEYBOARD_KIND: u32 = 0x10;
const GAMEPAD_KIND: u32 = 0x4_0000;
/// Device status of a connected device: connected, input enabled.
const STATUS_CONNECTED: u32 = 0x3;
const ASYNC_ENUMERATION: u32 = 1;

/// Gamepad buttons; the title maps each bit to one of its pad buttons.
const MENU: u32 = 0x1;
const VIEW: u32 = 0x2;
const A: u32 = 0x4;
const B: u32 = 0x8;
const X: u32 = 0x10;
const Y: u32 = 0x20;
const UP: u32 = 0x40;
const DOWN: u32 = 0x80;
const LEFT: u32 = 0x100;
const RIGHT: u32 = 0x200;
const LEFT_SHOULDER: u32 = 0x400;
const RIGHT_SHOULDER: u32 = 0x800;
const LEFT_STICK: u32 = 0x1000;
const RIGHT_STICK: u32 = 0x2000;

/// Devices: host pad slots, the keyboard's stand-in pad, the keyboard.
const PADS: usize = 4;
const FALLBACK: usize = PADS;
const KEYBOARD: usize = PADS + 1;
const DEVICES: usize = PADS + 2;

/// Readings kept per device for next/previous lookups: 256 polls of 8 ms,
/// about 2 s. At 32 (VIctory-base) a frame slower than 256 ms found the
/// reading it walks from gone, and presses in between were lost (VIctory
/// main 04512df, in VIctory-7900XTX).
const HISTORY: usize = 256;
const POLL: Duration = Duration::from_millis(8);

/// The keyboard's stand-in gamepad (virtual key, button); README lists it.
/// Sticks: W/A/S/D left, I/J/K/L right; triggers Z and C.
const FALLBACK_BUTTONS: [(u8, u32); 14] = [
    (0x1B, MENU), // Esc
    (0x09, VIEW), // Tab
    (0x0D, A),    // Enter
    (0x08, B),    // Backspace
    (b'X', X),
    (b'Y', Y),
    (0x26, UP), // arrows
    (0x28, DOWN),
    (0x25, LEFT),
    (0x27, RIGHT),
    (b'Q', LEFT_SHOULDER),
    (b'E', RIGHT_SHOULDER),
    (b'F', LEFT_STICK),
    (b'H', RIGHT_STICK),
];

/// A gamepad reading: triggers 0 to 1, sticks -1 to 1 (up positive).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PadState {
    buttons: u32,
    left_trigger: f32,
    right_trigger: f32,
    left_x: f32,
    left_y: f32,
    right_x: f32,
    right_y: f32,
}

/// One key held down: scan code (0xE0xx when extended), character, key.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct KeyState {
    scan_code: u32,
    code_point: u32,
    virtual_key: u8,
    dead_key: bool,
}

const _: () = assert!(size_of::<KeyState>() == 12 && size_of::<PadState>() == 28);

/// Host input, or a fake in tests.
trait Source: Send {
    /// The pad in host slot `slot`, `None` when the slot is empty.
    fn pad(&mut self, slot: usize) -> Option<PadState>;
    /// Keys held down; none while the title's window lacks focus.
    fn keys(&mut self) -> Vec<KeyState>;
    /// Motor speeds: low frequency (left), high frequency (right).
    fn rumble(&mut self, slot: usize, low: u16, high: u16);
}

fn fallback(keys: &[KeyState]) -> PadState {
    let on = |vk: u8| {
        if keys.iter().any(|k| k.virtual_key == vk) {
            1.0
        } else {
            0.0
        }
    };
    PadState {
        buttons: FALLBACK_BUTTONS
            .iter()
            .filter(|(vk, _)| on(*vk) != 0.0)
            .fold(0, |all, (_, bit)| all | bit),
        left_trigger: on(b'Z'),
        right_trigger: on(b'C'),
        left_x: on(b'D') - on(b'A'),
        left_y: on(b'W') - on(b'S'),
        right_x: on(b'L') - on(b'J'),
        right_y: on(b'I') - on(b'K'),
    }
}

#[repr(C)]
struct Table<const N: usize>([*const (); N]);
// SAFETY: function addresses only.
unsafe impl<const N: usize> Sync for Table<N> {}

/// The input object; one per hub.
#[repr(C)]
pub struct Object {
    vtable: &'static Table<22>,
    hub: *const Hub,
}

#[repr(C)]
pub struct Device {
    vtable: &'static Table<21>,
    hub: *const Hub,
    index: usize,
    info: DeviceInfo,
}

/// What a device says about itself. The title reads `id` (+0x20); the rest
/// follows the public layout and is zero where we have nothing to say.
#[repr(C)]
#[derive(Default)]
#[allow(dead_code)] // read by title code
struct DeviceInfo {
    size: u32,
    vendor: u16,
    product: u16,
    revision: u16,
    interface: u8,
    collection: u8,
    usage: [u16; 2],
    versions: [u16; 8],
    id: [u8; 32],
    root_id: [u8; 32],
    family: i32,
    capabilities: u32,
    supported_input: u32,
    rumble_motors: u32,
    /// Counts and optional detail pointers.
    rest: [u64; 26],
}

const _: () = assert!(std::mem::offset_of!(DeviceInfo, id) == 0x20);

fn kind(index: usize) -> u32 {
    if index == KEYBOARD {
        KEYBOARD_KIND
    } else {
        GAMEPAD_KIND
    }
}

impl Device {
    fn new(index: usize) -> Device {
        let mut id = [0; 32];
        id[..7].copy_from_slice(b"victory");
        id[31] = index as u8 + 1;
        let info = DeviceInfo {
            size: size_of::<DeviceInfo>() as u32,
            id,
            root_id: id,
            supported_input: kind(index),
            // Low and high frequency.
            rumble_motors: if index < PADS { 0x3 } else { 0 },
            ..Default::default()
        };
        Device {
            vtable: &DEVICE,
            hub: std::ptr::null(),
            index,
            info,
        }
    }

    fn hub(&self) -> &'static Hub {
        // SAFETY: set by `Hub::new`; hubs are never freed.
        unsafe { &*self.hub }
    }
}

#[derive(PartialEq)]
enum Value {
    Pad(PadState),
    Keys(Vec<KeyState>),
}

#[repr(C)]
pub struct Reading {
    vtable: &'static Table<25>,
    refs: AtomicU32,
    device: *const Device,
    sequence: u64,
    timestamp: u64,
    value: Value,
}

impl Reading {
    fn kind(&self) -> u32 {
        match self.value {
            Value::Pad(_) => GAMEPAD_KIND,
            Value::Keys(_) => KEYBOARD_KIND,
        }
    }
}

/// A counted reference to a reading.
struct Ref(NonNull<Reading>);

impl Ref {
    fn new(reading: Reading) -> Ref {
        Ref(NonNull::from(Box::leak(Box::new(reading))))
    }

    /// Another reference, for the title.
    fn share(&self) -> *const Reading {
        reading_add_ref(self.0.as_ptr());
        self.0.as_ptr()
    }
}

impl Deref for Ref {
    type Target = Reading;
    fn deref(&self) -> &Reading {
        // SAFETY: we hold a reference, so it is alive.
        unsafe { self.0.as_ref() }
    }
}

impl Drop for Ref {
    fn drop(&mut self) {
        reading_release(self.0.as_ptr());
    }
}

type DeviceCallback = extern "system" fn(u64, *mut c_void, *const Device, u64, u32, u32);

#[derive(Clone, Copy)]
struct Registration {
    token: u64,
    /// One device's address, or 0 for any.
    device: usize,
    kind: u32,
    status: u32,
    context: usize,
    callback: DeviceCallback,
}

impl Registration {
    fn matches(&self, devices: &[Device; DEVICES], index: usize) -> bool {
        kind(index) & self.kind != 0
            && (self.device == 0 || self.device == &devices[index] as *const Device as usize)
    }

    fn event(&self, device: usize, timestamp: u64, current: u32, previous: u32) -> Event {
        Event {
            registration: *self,
            device,
            timestamp,
            current,
            previous,
        }
    }
}

struct Event {
    registration: Registration,
    device: usize,
    timestamp: u64,
    current: u32,
    previous: u32,
}

#[derive(Default)]
struct Slot {
    connected: bool,
    sequence: u64,
    history: VecDeque<Ref>,
}

struct State {
    slots: [Slot; DEVICES],
    callbacks: Vec<Registration>,
    /// Asynchronous enumerations, delivered by the next poll.
    pending: Vec<Event>,
    /// Device callbacks running now: token and thread.
    running: Vec<(u64, ThreadId)>,
}

pub struct Hub {
    object: Object,
    devices: [Device; DEVICES],
    /// Apart from the state: XInput can be slow, and the title reads
    /// readings every frame.
    source: Mutex<Box<dyn Source>>,
    state: Mutex<State>,
    /// A device callback returned.
    returned: Condvar,
}

fn lock<T: ?Sized>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

// SAFETY: the back pointers are set before the hub is shared and never
// change; what changes sits behind the mutex or is atomic.
unsafe impl Send for Hub {}
// SAFETY: as above.
unsafe impl Sync for Hub {}

static START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Microseconds, the clock of every timestamp we hand out.
fn now() -> u64 {
    START.elapsed().as_micros() as u64
}

impl Hub {
    fn new(source: Box<dyn Source>) -> &'static Hub {
        let hub = Box::into_raw(Box::new(Hub {
            object: Object {
                vtable: &INPUT,
                hub: std::ptr::null(),
            },
            devices: std::array::from_fn(Device::new),
            source: Mutex::new(source),
            state: Mutex::new(State {
                slots: Default::default(),
                callbacks: Vec::new(),
                pending: Vec::new(),
                running: Vec::new(),
            }),
            returned: Condvar::new(),
        }));
        // SAFETY: just allocated and never freed; nothing else sees it yet.
        unsafe {
            (*hub).object.hub = hub;
            for device in &mut (*hub).devices {
                device.hub = hub;
            }
            &*hub
        }
    }

    /// A hub over the host's input, polled from now on.
    fn start(source: Box<dyn Source>) -> &'static Hub {
        let hub = Hub::new(source);
        hub.poll();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(POLL);
                hub.poll();
            }
        });
        hub
    }

    fn state(&self) -> MutexGuard<'_, State> {
        lock(&self.state)
    }

    /// Reads the host once; device callbacks run on the calling thread.
    fn poll(&self) {
        let (pads, keys) = {
            let mut source = lock(&self.source);
            let pads: [_; PADS] = std::array::from_fn(|slot| source.pad(slot));
            (pads, source.keys())
        };
        let events = self.state().poll(&self.devices, pads, keys);
        self.deliver(events);
    }

    /// Runs callbacks, never under our lock: they call back into us. One
    /// unregistered since the event was taken does not run, and
    /// `UnregisterCallback` waits for one that runs.
    fn deliver(&self, events: Vec<Event>) {
        let me = std::thread::current().id();
        for e in events {
            let r = e.registration;
            {
                let mut state = self.state();
                if !state.callbacks.iter().any(|c| c.token == r.token) {
                    continue;
                }
                state.running.push((r.token, me));
            }
            (r.callback)(
                r.token,
                r.context as *mut c_void,
                &self.devices[e.device],
                e.timestamp,
                e.current,
                e.previous,
            );
            let mut state = self.state();
            if let Some(i) = state.running.iter().position(|&x| x == (r.token, me)) {
                state.running.swap_remove(i);
            }
            self.returned.notify_all();
        }
    }

    fn index(&self, device: *const Device) -> Option<usize> {
        self.devices.iter().position(|d| std::ptr::eq(d, device))
    }
}

impl State {
    fn poll(
        &mut self,
        devices: &[Device; DEVICES],
        pads: [Option<PadState>; PADS],
        keys: Vec<KeyState>,
    ) -> Vec<Event> {
        let now = now();
        let mut events = std::mem::take(&mut self.pending);
        let any_pad = pads.iter().any(Option::is_some);
        for (slot, pad) in pads.into_iter().enumerate() {
            self.update(devices, slot, pad.map(Value::Pad), now, &mut events);
        }
        let fallback = (!any_pad).then(|| Value::Pad(fallback(&keys)));
        self.update(devices, FALLBACK, fallback, now, &mut events);
        self.update(devices, KEYBOARD, Some(Value::Keys(keys)), now, &mut events);
        events
    }

    /// Records a device's state: a reading when it changed, callbacks when
    /// the device came or went.
    fn update(
        &mut self,
        devices: &[Device; DEVICES],
        index: usize,
        value: Option<Value>,
        now: u64,
        events: &mut Vec<Event>,
    ) {
        let slot = &mut self.slots[index];
        let was = slot.connected;
        slot.connected = value.is_some();
        if let Some(value) = value
            && (!was || slot.history.back().is_none_or(|r| r.value != value))
        {
            slot.sequence += 1;
            if slot.history.len() == HISTORY {
                slot.history.pop_front();
            }
            slot.history.push_back(Ref::new(Reading {
                vtable: &READING,
                refs: AtomicU32::new(1),
                device: &devices[index],
                sequence: slot.sequence,
                timestamp: now,
                value,
            }));
        }
        if was != slot.connected {
            xnw_core::info!("input device {index} {}", if was { "disconnected" } else { "connected" });
            let (current, previous) = if was {
                (0, STATUS_CONNECTED)
            } else {
                (STATUS_CONNECTED, 0)
            };
            events.extend(
                self.callbacks
                    .iter()
                    .filter(|r| r.matches(devices, index) && r.status & STATUS_CONNECTED != 0)
                    .map(|r| r.event(index, now, current, previous)),
            );
        }
    }
}

fn host() -> Box<dyn Source> {
    Box::new(host::Host::default())
}

static HUB: LazyLock<&'static Hub> = LazyLock::new(|| Hub::start(host()));

type This = *const Object;
type Out = *mut *const c_void;

/// The DLL's only export (ordinal 1): the input object, static and shared.
pub extern "system" fn GameInputCreate(out: *mut This) -> HRESULT {
    xnw_core::info!("GameInputCreate from {}", crate::images::caller());
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null; the title passes room for a pointer.
    unsafe { out.write(&HUB.object) };
    S_OK
}

fn hub(this: This) -> &'static Hub {
    // SAFETY: `this` is a hub's object; hubs are never freed.
    unsafe { &*(*this).hub }
}

/// Clears a COM out pointer and returns `hr`.
fn fail(out: Out, hr: HRESULT) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a pointer.
        unsafe { out.write(std::ptr::null()) };
    }
    hr
}

/// Gives the title a reference to `reading`, or fails as with none.
fn hand(out: Out, reading: Option<&Ref>) -> HRESULT {
    match reading {
        _ if out.is_null() => E_INVALIDARG,
        Some(reading) => {
            // SAFETY: checked non-null; the title passes room for a pointer.
            unsafe { out.write(reading.share().cast()) };
            S_OK
        }
        None => fail(out, READING_NOT_FOUND),
    }
}

extern "system" fn no_interface(_: *const c_void, _iid: *const c_void, out: Out) -> HRESULT {
    xnw_core::warn!("input QueryInterface not implemented; from {}", crate::images::caller());
    fail(out, E_NOINTERFACE)
}
/// The input object and devices live as long as the process.
extern "system" fn add_ref(_: *const c_void) -> u32 {
    1
}

extern "system" fn timestamp(_: This) -> u64 {
    now()
}

extern "system" fn current_reading(
    this: This,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    let hub = hub(this);
    let state = hub.state();
    let newest = |i: usize| {
        state.slots[i]
            .history
            .back()
            .filter(|r| r.kind() & kind != 0)
    };
    let found = if device.is_null() {
        (0..DEVICES)
            .filter(|&i| state.slots[i].connected)
            .filter_map(newest)
            .max_by_key(|r| r.timestamp)
    } else {
        match hub.index(device) {
            None => return fail(out, E_INVALIDARG),
            Some(i) if !state.slots[i].connected => return fail(out, DEVICE_DISCONNECTED),
            Some(i) => newest(i),
        }
    };
    hand(out, found)
}

/// The reading after (`forward`) or before `reference` on its device.
fn relative(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
    forward: bool,
) -> HRESULT {
    if reference.is_null() {
        return fail(out, E_INVALIDARG);
    }
    let hub = hub(this);
    // SAFETY: a reading we handed out and the title still holds.
    let reference = unsafe { &*reference };
    // shortcut: with no device given only the reference's device is walked,
    // not every device's readings in time order; the title walks
    // only keyboard readings (one device) and the current pad's previous one.
    let Some(index) = hub.index(reference.device) else {
        return fail(out, E_INVALIDARG);
    };
    if !device.is_null() && !std::ptr::eq(device, reference.device) {
        return fail(out, E_INVALIDARG);
    }
    let state = hub.state();
    let history = &state.slots[index].history;
    if history
        .front()
        .is_none_or(|oldest| reference.sequence < oldest.sequence)
    {
        return fail(out, REFERENCE_TOO_OLD);
    }
    let wanted = |r: &&Ref| r.kind() & kind != 0;
    let found = if forward {
        history
            .iter()
            .filter(|r| r.sequence > reference.sequence)
            .find(wanted)
    } else {
        history
            .iter()
            .rev()
            .filter(|r| r.sequence < reference.sequence)
            .find(wanted)
    };
    hand(out, found)
}
extern "system" fn next_reading(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    relative(this, reference, kind, device, out, true)
}
extern "system" fn previous_reading(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    relative(this, reference, kind, device, out, false)
}
/// The device's reading at `time`, among those still kept.
extern "system" fn temporal_reading(
    this: This,
    time: u64,
    device: *const Device,
    out: Out,
) -> HRESULT {
    let hub = hub(this);
    let Some(index) = hub.index(device) else {
        return fail(out, E_INVALIDARG);
    };
    let state = hub.state();
    let history = &state.slots[index].history;
    hand(out, history.iter().rev().find(|r| r.timestamp <= time))
}

static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

fn token(out: *mut u64) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null.
        unsafe { out.write(NEXT_TOKEN.fetch_add(1, Ordering::Relaxed)) };
    }
    S_OK
}

// shortcut: reading callbacks never run; the title registers none
// (it polls).
extern "system" fn register_reading(
    _: This,
    _device: *const c_void,
    _kind: u32,
    _threshold: f32,
    _context: *mut c_void,
    _callback: *const c_void,
    out: *mut u64,
) -> HRESULT {
    xnw_core::warn!("input reading callbacks are never called; from {}", crate::images::caller());
    token(out)
}

/// Device callbacks run on our poll thread when a device comes or goes.
/// Enumeration reports the devices already there: blocking on the caller's
/// thread before we return (the title registers holding a lock its callback
/// takes again), asynchronous on the next poll.
extern "system" fn register_device(
    this: This,
    device: *const Device,
    kind: u32,
    status: u32,
    enumeration: u32,
    context: *mut c_void,
    callback: Option<DeviceCallback>,
    out: *mut u64,
) -> HRESULT {
    let Some(callback) = callback else {
        return E_INVALIDARG;
    };
    let hub = hub(this);
    let registration = Registration {
        token: NEXT_TOKEN.fetch_add(1, Ordering::Relaxed),
        device: device as usize,
        kind,
        status,
        context: context as usize,
        callback,
    };
    let mut found = Vec::new();
    {
        let mut state = hub.state();
        if enumeration != 0 && status & STATUS_CONNECTED != 0 {
            let now = now();
            found = (0..DEVICES)
                .filter(|&i| state.slots[i].connected && registration.matches(&hub.devices, i))
                .map(|i| registration.event(i, now, STATUS_CONNECTED, 0))
                .collect();
        }
        xnw_core::info!(
            "input device callback (kind {kind:#x}, enumeration {enumeration}, {} devices) from {}",
            found.len(),
            crate::images::caller()
        );
        state.callbacks.push(registration);
        if enumeration == ASYNC_ENUMERATION {
            state.pending.append(&mut found);
        }
    }
    if !out.is_null() {
        // SAFETY: checked non-null.
        unsafe { out.write(registration.token) };
    }
    hub.deliver(found);
    S_OK
}
/// Guide button and keyboard layout callbacks: neither ever changes here.
extern "system" fn register_other(
    _: This,
    _device: *const c_void,
    _context: *mut c_void,
    _callback: *const c_void,
    out: *mut u64,
) -> HRESULT {
    token(out)
}

fn forget(this: This, token: u64) {
    let mut state = hub(this).state();
    state.callbacks.retain(|r| r.token != token);
    state.pending.retain(|e| e.registration.token != token);
}
extern "system" fn stop_callback(this: This, token: u64) {
    forget(this, token);
}
/// Stops the callback, then waits up to `timeout` µs for a call of it that
/// is running; false on time-out, or from inside that callback (it cannot
/// wait for itself), as documented. Either way it is never called again.
extern "system" fn unregister_callback(this: This, token: u64, timeout: u64) -> bool {
    forget(this, token);
    let hub = hub(this);
    let me = std::thread::current().id();
    let deadline = Instant::now().checked_add(Duration::from_micros(timeout));
    let mut state = hub.state();
    loop {
        let running = state.running.iter().filter(|r| r.0 == token);
        if running.clone().any(|r| r.1 == me) {
            return false;
        }
        if running.count() == 0 {
            return true;
        }
        state = match deadline {
            None => hub.returned.wait(state).unwrap_or_else(PoisonError::into_inner),
            Some(deadline) => {
                let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                    return false;
                };
                let waited = hub.returned.wait_timeout(state, left);
                waited.unwrap_or_else(PoisonError::into_inner).0
            }
        };
    }
}

/// Dispatcher and aggregate device: no title of ours asks for them.
extern "system" fn create_dispatcher(_: This, out: Out) -> HRESULT {
    xnw_core::warn!("input dispatcher not implemented; from {}", crate::images::caller());
    fail(out, E_NOTIMPL)
}
extern "system" fn create_aggregate(_: This, _kind: u32, out: Out) -> HRESULT {
    xnw_core::warn!("aggregate input device not implemented; from {}", crate::images::caller());
    fail(out, E_NOTIMPL)
}
extern "system" fn find_by_id(this: This, id: *const [u8; 32], out: Out) -> HRESULT {
    if id.is_null() || out.is_null() {
        return fail(out, E_INVALIDARG);
    }
    // SAFETY: checked non-null; the title passes a device id.
    let id = unsafe { &*id };
    match hub(this).devices.iter().find(|d| d.info.id == *id) {
        Some(device) => {
            // SAFETY: checked non-null.
            unsafe { out.write((device as *const Device).cast()) };
            S_OK
        }
        None => fail(out, DEVICE_NOT_FOUND),
    }
}
/// Lookup by object, platform handle or platform string: our devices have
/// none of these.
extern "system" fn find_device(_: This, _value: *const c_void, out: Out) -> HRESULT {
    fail(out, DEVICE_NOT_FOUND)
}
extern "system" fn enable_oem_device(
    _: This,
    _vendor: u16,
    _product: u16,
    _interface: u8,
    _collection: u8,
) -> HRESULT {
    S_OK
}
extern "system" fn set_focus_policy(_: This, _policy: u32) {}

static INPUT: Table<22> = Table([
    no_interface as *const (),        // 0x00
    add_ref as *const (),             // 0x08 AddRef
    add_ref as *const (),             // 0x10 Release
    timestamp as *const (),           // 0x18
    current_reading as *const (),     // 0x20
    next_reading as *const (),        // 0x28
    previous_reading as *const (),    // 0x30
    temporal_reading as *const (),    // 0x38
    register_reading as *const (),    // 0x40
    register_device as *const (),     // 0x48
    register_other as *const (),      // 0x50 guide button
    register_other as *const (),      // 0x58 keyboard layout
    stop_callback as *const (),       // 0x60
    unregister_callback as *const (), // 0x68
    create_dispatcher as *const (),   // 0x70
    create_aggregate as *const (),    // 0x78
    find_by_id as *const (),          // 0x80
    find_device as *const (),         // 0x88 by object
    find_device as *const (),         // 0x90 by platform handle
    find_device as *const (),         // 0x98 by platform string
    enable_oem_device as *const (),   // 0xa0
    set_focus_policy as *const (),    // 0xa8
]);

extern "system" fn device_info(this: *const Device) -> *const DeviceInfo {
    // SAFETY: one of our devices.
    unsafe { &(*this).info }
}
extern "system" fn device_status(this: *const Device) -> u32 {
    // SAFETY: one of our devices.
    let device = unsafe { &*this };
    if device.hub().state().slots[device.index].connected {
        STATUS_CONNECTED
    } else {
        0
    }
}
// shortcut: trigger motors are dropped (XInput has none). The host API that
// drives them without a redistributable is Windows.Gaming.Input (WinRT
// `Gamepad.Vibration`), which needs WinRT bindings this crate lacks outside
// diagnostics builds; the title sends trigger levels in slot 0x50.
extern "system" fn set_rumble(this: *const Device, params: *const [f32; 4]) {
    // SAFETY: one of our devices.
    let device = unsafe { &*this };
    if device.index >= PADS {
        return;
    }
    let [low, high, ..] = if params.is_null() {
        [0.0; 4]
    } else {
        // SAFETY: checked non-null; four motor levels.
        unsafe { *params }
    };
    let speed = |level: f32| (level.clamp(0.0, 1.0) * f32::from(u16::MAX)) as u16;
    lock(&device.hub().source).rumble(device.index, speed(low), speed(high));
}
extern "system" fn unsupported<const SLOT: usize>(_: *const c_void) -> HRESULT {
    xnw_core::warn!("input device method {SLOT:#x} not implemented; from {}", crate::images::caller());
    E_NOTIMPL
}

static DEVICE: Table<21> = Table([
    no_interface as *const (),        // 0x00
    add_ref as *const (),             // 0x08 AddRef
    add_ref as *const (),             // 0x10 Release
    device_info as *const (),         // 0x18
    device_status as *const (),       // 0x20
    unsupported::<0x28> as *const (), // battery
    unsupported::<0x30> as *const (), // force feedback
    unsupported::<0x38> as *const (), //
    unsupported::<0x40> as *const (), //
    unsupported::<0x48> as *const (), // haptics
    set_rumble as *const (),          // 0x50
    unsupported::<0x58> as *const (), // input synchronization
    unsupported::<0x60> as *const (), //
    unsupported::<0x68> as *const (), // power off
    unsupported::<0x70> as *const (), // raw device access
    unsupported::<0x78> as *const (), //
    unsupported::<0x80> as *const (), //
    unsupported::<0x88> as *const (), //
    unsupported::<0x90> as *const (), //
    unsupported::<0x98> as *const (), //
    unsupported::<0xa0> as *const (), //
]);

fn reading<'a>(this: *const Reading) -> &'a Reading {
    // SAFETY: a reading we handed out and the title still holds.
    unsafe { &*this }
}
extern "system" fn reading_add_ref(this: *const Reading) -> u32 {
    reading(this).refs.fetch_add(1, Ordering::Relaxed) + 1
}
extern "system" fn reading_release(this: *const Reading) -> u32 {
    let left = reading(this).refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if left == 0 {
        // SAFETY: the last reference is gone; `Ref::new` boxed it.
        drop(unsafe { Box::from_raw(this.cast_mut()) });
    }
    left
}
extern "system" fn input_kind(this: *const Reading) -> u32 {
    reading(this).kind()
}
extern "system" fn sequence_number(this: *const Reading, kind: u32) -> u64 {
    let reading = reading(this);
    if reading.kind() & kind != 0 {
        reading.sequence
    } else {
        0
    }
}
extern "system" fn reading_timestamp(this: *const Reading) -> u64 {
    reading(this).timestamp
}
extern "system" fn reading_device(this: *const Reading, out: *mut *const Device) {
    if !out.is_null() {
        // SAFETY: checked non-null; devices need no reference.
        unsafe { out.write(reading(this).device) };
    }
}
extern "system" fn key_count(this: *const Reading) -> u32 {
    match &reading(this).value {
        Value::Keys(keys) => keys.len() as u32,
        Value::Pad(_) => 0,
    }
}
extern "system" fn key_state(this: *const Reading, count: u32, out: *mut KeyState) -> u32 {
    let Value::Keys(keys) = &reading(this).value else {
        return 0;
    };
    if out.is_null() {
        return 0;
    }
    let n = keys.len().min(count as usize);
    // SAFETY: the title passes room for `count` keys.
    unsafe { std::ptr::copy_nonoverlapping(keys.as_ptr(), out, n) };
    n as u32
}
extern "system" fn gamepad_state(this: *const Reading, out: *mut PadState) -> bool {
    match &reading(this).value {
        Value::Pad(state) if !out.is_null() => {
            // SAFETY: checked non-null.
            unsafe { out.write(*state) };
            true
        }
        _ => false,
    }
}
/// Kinds our devices do not have: no count, no state.
extern "system" fn nothing(_: *const c_void) -> usize {
    0
}

static READING: Table<25> = Table([
    no_interface as *const (),      // 0x00
    reading_add_ref as *const (),   // 0x08
    reading_release as *const (),   // 0x10
    input_kind as *const (),        // 0x18
    sequence_number as *const (),   // 0x20
    reading_timestamp as *const (), // 0x28
    reading_device as *const (),    // 0x30
    nothing as *const (),           // 0x38 raw report
    nothing as *const (),           // 0x40 controller axes, buttons, switches
    nothing as *const (),           // 0x48
    nothing as *const (),           // 0x50
    nothing as *const (),           // 0x58
    nothing as *const (),           // 0x60
    nothing as *const (),           // 0x68
    key_count as *const (),         // 0x70
    key_state as *const (),         // 0x78
    nothing as *const (),           // 0x80 mouse
    nothing as *const (),           // 0x88 touch
    nothing as *const (),           // 0x90
    nothing as *const (),           // 0x98 motion
    nothing as *const (),           // 0xa0 arcade stick
    nothing as *const (),           // 0xa8 flight stick
    gamepad_state as *const (),     // 0xb0
    nothing as *const (),           // 0xb8 racing wheel
    nothing as *const (),           // 0xc0 UI navigation
]);

/// Host input: XInput pads, the keyboard while our process has focus.
mod host {
    use std::time::{Duration, Instant};

    use xnw_core::win::{
        GetAsyncKeyState, GetCurrentProcessId, GetForegroundWindow, GetWindowThreadProcessId, MAPVK_VK_TO_VSC_EX, MapVirtualKeyW,
        XINPUT_GAMEPAD, XINPUT_STATE, XINPUT_VIBRATION, XInputGetState, XInputSetState,
    };

    use super::*;

    /// Asking XInput about an empty slot is slow; look again this often.
    const EMPTY_SLOT_RECHECK: Duration = Duration::from_secs(1);

    /// XInput button bits and ours.
    const BUTTONS: [(u16, u32); 14] = [
        (0x0010, MENU), // start
        (0x0020, VIEW), // back
        (0x1000, A),
        (0x2000, B),
        (0x4000, X),
        (0x8000, Y),
        (0x0001, UP),
        (0x0002, DOWN),
        (0x0004, LEFT),
        (0x0008, RIGHT),
        (0x0100, LEFT_SHOULDER),
        (0x0200, RIGHT_SHOULDER),
        (0x0040, LEFT_STICK),
        (0x0080, RIGHT_STICK),
    ];

    /// Raw values, no dead zone: titles apply their own.
    pub(super) fn convert(pad: &XINPUT_GAMEPAD) -> PadState {
        let stick = |v: i16| (f32::from(v) / 32767.0).max(-1.0);
        PadState {
            buttons: BUTTONS.iter().filter(|(host, _)| pad.buttons & host != 0).fold(0, |all, (_, bit)| all | bit),
            left_trigger: f32::from(pad.left_trigger) / 255.0,
            right_trigger: f32::from(pad.right_trigger) / 255.0,
            left_x: stick(pad.left_x),
            left_y: stick(pad.left_y),
            right_x: stick(pad.right_x),
            right_y: stick(pad.right_y),
        }
    }

    #[derive(Default)]
    pub(super) struct Host {
        empty_until: [Option<Instant>; PADS],
    }

    impl Source for Host {
        fn pad(&mut self, slot: usize) -> Option<PadState> {
            if self.empty_until[slot].is_some_and(|until| Instant::now() < until) {
                return None;
            }
            let mut state = XINPUT_STATE::default();
            // SAFETY: room for the state.
            if unsafe { XInputGetState(slot as u32, &mut state) } != 0 {
                self.empty_until[slot] = Some(Instant::now() + EMPTY_SLOT_RECHECK);
                return None;
            }
            self.empty_until[slot] = None;
            Some(convert(&state.gamepad))
        }

        fn keys(&mut self) -> Vec<KeyState> {
            let mut down = scripted();
            if focused() {
                // Left/right modifier keys only, as devices report them; no
                // mouse buttons (below 0x08).
                let held: Vec<u8> = (0x08..=0xFE_u8)
                    .filter(|vk| !(0x10..=0x12).contains(vk) && !down.contains(vk))
                    // SW1: F1-F12 are the live switches' keys (window.rs), not the title's.
                    .filter(|&vk| !(crate::window::live_keys() && crate::window::LIVE_VK.contains(&usize::from(vk))))
                    // SAFETY: plain queries.
                    .filter(|&vk| unsafe { GetAsyncKeyState(i32::from(vk)) } < 0)
                    .collect();
                down.extend(held);
            }
            down.into_iter()
                .map(|vk| KeyState {
                    // SAFETY: plain query.
                    scan_code: unsafe { MapVirtualKeyW(u32::from(vk), MAPVK_VK_TO_VSC_EX) },
                    virtual_key: vk,
                    // shortcut: no characters (code point 0); the title reads
                    // scan codes and virtual keys only (VIctory).
                    ..Default::default()
                })
                .collect()
        }

        fn rumble(&mut self, slot: usize, low: u16, high: u16) {
            let speeds = XINPUT_VIBRATION { left: low, right: high };
            // SAFETY: a valid vibration; an empty slot just fails.
            unsafe { XInputSetState(slot as u32, &speeds) };
        }
    }

    /// `XNW_KEYS` scripts key presses with or without focus (grammar and
    /// schedule: `Script` below). Keys: a letter, a digit, or `Enter`, `Esc`,
    /// `Tab`, `Back` (or `Backspace`), `Up`, `Down`, `Left`, `Right`; the
    /// stand-in pad maps Enter/Back/X/Y to A/B/X/Y and the arrows to the
    /// D-pad.
    ///
    /// Ported from VIctory-base `victory-hle/src/modules/gameinput.rs`
    /// 1187-1225 (`VICTORY_KEYS`, 5d0516e, GPL-3.0-or-later). Differences:
    /// always built (the base gates it behind its `diagnostics` feature),
    /// timed from the log clock (the base: from the first input poll), and
    /// keyed to presented frames as well as seconds.
    fn scripted() -> Vec<u8> {
        static SCRIPT: LazyLock<Mutex<Script>> = LazyLock::new(|| {
            let script = Script::parse(&std::env::var("XNW_KEYS").unwrap_or_default());
            if !script.entries.is_empty() || !script.rules.is_empty() {
                xnw_core::info!("gameinput: {} scripted key presses and {} state-keyed rules (XNW_KEYS)", script.entries.len(), script.rules.len());
            }
            for bad in &script.refused {
                xnw_core::warn!("gameinput: XNW_KEYS entry refused: {bad:?}");
            }
            Mutex::new(script)
        });
        let mut script = SCRIPT.lock().unwrap_or_else(PoisonError::into_inner);
        if script.entries.is_empty() && script.rules.is_empty() {
            return Vec::new();
        }
        let (now, frame) = (xnw_core::uptime(), xnw_gpu::d3d12x::presented());
        let (down, fresh) = script.tick(now, frame, &super::watched);
        for vk in fresh {
            xnw_core::info!("keys: {} at {:.1} s frame {frame}", key_name(vk), now.as_secs_f64());
        }
        down
    }

    /// The foreground window is one of ours.
    fn focused() -> bool {
        let mut process = 0;
        // SAFETY: plain queries; a null window gives process 0.
        unsafe {
            GetWindowThreadProcessId(GetForegroundWindow(), &mut process);
            process == GetCurrentProcessId()
        }
    }

    #[test]
    fn host_pads_convert_to_console_pads() {
        let pad = XINPUT_GAMEPAD { buttons: 0x1000 | 0x0010 | 0x0004, left_trigger: 255, left_x: -32768, left_y: 32767, ..Default::default() };
        let state = convert(&pad);
        assert_eq!(state.buttons, A | MENU | LEFT);
        assert_eq!((state.left_trigger, state.left_x, state.left_y, state.right_x), (1.0, -1.0, 1.0, 0.0));
    }
}

/// The title values `XNW_WATCH` samples (every 250 ms), by name, for
/// the state-keyed `XNW_KEYS` forms; set by the app's watch thread.
static WATCHED: Mutex<Vec<(String, Option<u64>)>> = Mutex::new(Vec::new());

/// Notes the latest sample of watch `name` (`None`: unreadable).
pub fn note_watch(name: &str, value: Option<u64>) {
    let mut w = WATCHED.lock().unwrap_or_else(PoisonError::into_inner);
    match w.iter_mut().find(|(n, _)| n == name) {
        Some(e) => e.1 = value,
        None => w.push((name.to_owned(), value)),
    }
}

/// The latest sample of watch `name`.
fn watched(name: &str) -> Option<u64> {
    WATCHED.lock().unwrap_or_else(PoisonError::into_inner).iter().find(|(n, _)| n == name).and_then(|e| e.1)
}

/// A comparison of a watched value (`<name><op><value>[|<value>...]`).
#[derive(Clone, Debug, PartialEq)]
struct Cond {
    name: String,
    op: Op,
    /// One value, or several for `==` (any) and `!=` (none).
    values: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    Eq,
    Ne,
    Ge,
    Le,
    Gt,
    Lt,
}

impl Cond {
    fn parse(text: &str) -> Option<Cond> {
        let text = text.trim();
        let (at, op, len) = [("==", Op::Eq), ("!=", Op::Ne), (">=", Op::Ge), ("<=", Op::Le), (">", Op::Gt), ("<", Op::Lt)].into_iter().find_map(|(t, op)| text.find(t).map(|i| (i, op, t.len())))?;
        let name = text[..at].trim();
        if name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            return None;
        }
        let values: Option<Vec<u64>> = text[at + len..].split('|').map(|v| {
            let v = v.trim();
            match v.strip_prefix("0x") {
                Some(h) => u64::from_str_radix(h, 16).ok(),
                None => v.parse().ok(),
            }
        }).collect();
        let values = values?;
        if values.is_empty() || (values.len() > 1 && !matches!(op, Op::Eq | Op::Ne)) {
            return None;
        }
        Some(Cond { name: name.to_owned(), op, values })
    }

    /// Whether it holds now (an unread value never holds).
    fn holds(&self, look: &dyn Fn(&str) -> Option<u64>) -> bool {
        let Some(v) = look(&self.name) else { return false };
        match self.op {
            Op::Eq => self.values.contains(&v),
            Op::Ne => !self.values.contains(&v),
            Op::Ge => v >= self.values[0],
            Op::Le => v <= self.values[0],
            Op::Gt => v > self.values[0],
            Op::Lt => v < self.values[0],
        }
    }
}

/// A state-keyed press.
#[derive(Clone, Debug, PartialEq)]
enum RuleKind {
    /// `when:<cond>[+<ms>]:<key>`: once, `<ms>` after the condition first held.
    Once { cond: Cond, delay: Duration },
    /// `every:<ms>:<key>[ while:<cond>][ until:<cond>]`: every `<ms>` while
    /// `while` holds (always without it), the first `<ms>` after it began or
    /// after its watched value last changed (a new screen settles first);
    /// never again once `until` has held after `while` first held.
    Every { period: Duration, during: Option<Cond>, until: Option<Cond> },
    /// `hold:<key>[ while:<cond>][ until:<cond>]`: the key stays down for as
    /// long as `while` holds (always without it), and is let go for good
    /// once `until` has held after `while` first held (FAST1: holding Enter
    /// skips an in-world cinematic; a press of [`KEY_HOLD`] does not).
    Hold { during: Option<Cond>, until: Option<Cond> },
}

#[derive(Clone, Debug, PartialEq)]
struct Rule {
    kind: RuleKind,
    /// The keys pressed in turn (one for `when`).
    keys: Vec<u8>,
    /// The next of `keys`.
    turn: usize,
    /// The key of the last press (held [`KEY_HOLD`]).
    key: u8,
    /// Once: when the condition first held; Every: the next press.
    armed: Option<Duration>,
    /// Every: the `while` value the period started from.
    from: Option<u64>,
    pressed: Option<Duration>,
    done: bool,
    /// Every: its `while` has held once.
    started: bool,
}

impl Rule {
    fn new(kind: RuleKind, keys: Vec<u8>) -> Rule {
        Rule { kind, key: keys[0], keys, turn: 0, armed: None, from: None, pressed: None, done: false, started: false }
    }

    /// Takes the next key in turn.
    fn next_key(&mut self) {
        self.key = self.keys[self.turn % self.keys.len()];
        self.turn += 1;
    }

    /// One poll: whether the key goes down now.
    fn tick(&mut self, now: Duration, look: &dyn Fn(&str) -> Option<u64>) -> bool {
        if self.done {
            return false;
        }
        match &self.kind {
            RuleKind::Hold { during, until } => {
                if self.started && until.as_ref().is_some_and(|u| u.holds(look)) {
                    self.done = true;
                    return false;
                }
                let on = during.as_ref().is_none_or(|w| w.holds(look));
                self.started |= on;
                on
            }
            RuleKind::Once { cond, delay } => {
                if self.armed.is_none() && cond.holds(look) {
                    self.armed = Some(now);
                }
                if self.armed.is_some_and(|t| now >= t + *delay) {
                    self.done = true;
                    return true;
                }
                false
            }
            RuleKind::Every { period, during, until } => {
                // `until` counts once the rule has started (its `while` held
                // once): session_state 0 before the front end is not the end.
                if self.started && until.as_ref().is_some_and(|u| u.holds(look)) {
                    self.done = true;
                    return false;
                }
                self.started |= during.as_ref().is_none_or(|w| w.holds(look));
                if let Some(w) = during {
                    if !w.holds(look) {
                        self.armed = None;
                        return false;
                    }
                    let v = look(&w.name);
                    if self.armed.is_none() || v != self.from {
                        self.from = v;
                        self.armed = Some(now + *period);
                    }
                } else if self.armed.is_none() {
                    self.armed = Some(now + *period);
                }
                if self.armed.is_some_and(|t| now >= t) {
                    self.armed = Some(now + *period);
                    return true;
                }
                false
            }
        }
    }

    fn parse(entry: &str) -> Option<Rule> {
        let entry = entry.trim();
        if let Some(rest) = entry.strip_prefix("when:") {
            let (cond, key) = rest.rsplit_once(':')?;
            let key = key_code(key)?;
            let (cond, delay) = match cond.rsplit_once('+') {
                Some((c, ms)) => (c, Duration::from_millis(ms.trim().parse().ok()?)),
                None => (cond, Duration::ZERO),
            };
            return Some(Rule::new(RuleKind::Once { cond: Cond::parse(cond)?, delay }, vec![key]));
        }
        let (period, rest) = if let Some(rest) = entry.strip_prefix("hold:") {
            (None, rest)
        } else {
            let rest = entry.strip_prefix("every:")?;
            let (ms, rest) = rest.split_once(':')?;
            (Some(Duration::from_millis(ms.trim().parse().ok().filter(|&n: &u64| n > 0)?)), rest)
        };
        let mut parts = rest.split([' ', ';']).filter(|p| !p.is_empty());
        let keys: Option<Vec<u8>> = parts.next()?.split('|').map(key_code).collect();
        let keys = keys?;
        let (mut during, mut until) = (None, None);
        for p in parts {
            if let Some(c) = p.strip_prefix("while:") {
                during = Some(Cond::parse(c)?);
            } else if let Some(c) = p.strip_prefix("until:") {
                until = Some(Cond::parse(c)?);
            } else {
                return None;
            }
        }
        if period.is_none() && keys.len() != 1 {
            return None;
        }
        Some(Rule::new(
            match period {
                Some(period) => RuleKind::Every { period, during, until },
                None => RuleKind::Hold { during, until },
            },
            keys,
        ))
    }
}

/// When a scripted key goes down.
#[derive(Clone, Copy, Debug, PartialEq)]
enum When {
    Time(Duration),
    Frame(u64),
}

/// The scripted keys (`XNW_KEYS`), comma separated entries `<when>:<key>`:
/// - `<seconds>:<key>`: at that time on the log clock (held 0.2 s);
/// - `f<frame>:<key>`: at that presented frame (held 2 frames);
/// - `+<n>:<key>`: n frames after the previous frame entry (`f`, `+` or the
///   end of a repeat);
/// - `f<from>-<to>/<every>:<key>[|<key>...]`: every `<every>` frames from
///   `<from>` up to (not including) `<to>`, the keys in turn, e.g.
///   `f1305-4000/5:Up|Enter` presses Up, Enter, Up, ...
///
/// keyed to title state (values `XNW_WATCH` samples, by name; numbers
/// decimal or `0x` hex; `==`/`!=` take `a|b|c` sets):
/// - `when:<name><op><value>[+<ms>]:<key>`: once, `<ms>` after the condition
///   first holds;
/// - `every:<ms>:<key>[|<key>...][ while:<cond>][ until:<cond>]` (clauses
///   separated by a space or `;`): every `<ms>` while `while` holds, the keys
///   in turn, restarting the period when its watched value changes, and
///   never again once `until` held (after `while` first held);
/// - `hold:<key>[ while:<cond>][ until:<cond>]`: the key held down for as
///   long as `while` holds, never again once `until` held.
///
/// Entries that do not parse are dropped and listed in `refused` (logged).
/// A key goes down at the first poll at or after its trigger (frames may
/// skip), once.
struct Script {
    entries: Vec<(When, u8)>,
    /// Per entry: the (time, frame) of its press once it has gone down.
    fired: Vec<Option<(Duration, u64)>>,
    /// The state-keyed rules (held [`KEY_HOLD`]).
    rules: Vec<Rule>,
    /// Entries that did not parse.
    refused: Vec<String>,
}

const KEY_HOLD: Duration = Duration::from_millis(200);
const KEY_HOLD_FRAMES: u64 = 2;
/// A repeat longer than this is a typo.
const REPEAT_MAX: u64 = 100_000;

fn key_code(name: &str) -> Option<u8> {
    Some(match name.trim() {
        "Enter" => 0x0D,
        "Esc" => 0x1B,
        "Tab" => 0x09,
        "Back" | "Backspace" => 0x08,
        "Up" => 0x26,
        "Down" => 0x28,
        "Left" => 0x25,
        "Right" => 0x27,
        k if k.len() == 1 && k.as_bytes()[0].is_ascii_alphanumeric() => k.as_bytes()[0].to_ascii_uppercase(),
        _ => return None,
    })
}

fn key_name(vk: u8) -> String {
    match vk {
        0x0D => "Enter".into(),
        0x1B => "Esc".into(),
        0x09 => "Tab".into(),
        0x08 => "Back".into(),
        0x26 => "Up".into(),
        0x28 => "Down".into(),
        0x25 => "Left".into(),
        0x27 => "Right".into(),
        _ => char::from(vk).to_string(),
    }
}

impl Script {
    fn parse(list: &str) -> Script {
        let mut entries = Vec::new();
        let (mut rules, mut refused) = (Vec::new(), Vec::new());
        let mut cursor = 0_u64; // the frame of the previous frame entry
        for entry in list.split(',') {
            if entry.trim().is_empty() {
                continue;
            }
            let before = entries.len();
            let t = entry.trim();
            if t.starts_with("when:") || t.starts_with("every:") || t.starts_with("hold:") {
                match Rule::parse(t) {
                    Some(r) => rules.push(r),
                    None => refused.push(t.to_owned()),
                }
                continue;
            }
            let Some((at, keys)) = entry.split_once(':') else {
                refused.push(t.to_owned());
                continue;
            };
            let at = at.trim();
            if let Some(rest) = at.strip_prefix('f') {
                if let Some((range, every)) = rest.split_once('/') {
                    let (Some((from, to)), Ok(every)) = (range.split_once('-'), every.trim().parse::<u64>()) else {
                        refused.push(t.to_owned());
                        continue;
                    };
                    let (Ok(from), Ok(to)) = (from.trim().parse::<u64>(), to.trim().parse::<u64>()) else {
                        refused.push(t.to_owned());
                        continue;
                    };
                    let codes: Vec<u8> = keys.split('|').filter_map(key_code).collect();
                    if every == 0 || codes.is_empty() || to.saturating_sub(from) / every > REPEAT_MAX {
                        refused.push(t.to_owned());
                        continue;
                    }
                    for (i, frame) in (from..to).step_by(every as usize).enumerate() {
                        entries.push((When::Frame(frame), codes[i % codes.len()]));
                    }
                    cursor = to;
                } else if let (Ok(frame), Some(vk)) = (rest.parse::<u64>(), key_code(keys)) {
                    entries.push((When::Frame(frame), vk));
                    cursor = frame;
                }
            } else if let Some(n) = at.strip_prefix('+') {
                if let (Ok(n), Some(vk)) = (n.parse::<u64>(), key_code(keys)) {
                    cursor += n;
                    entries.push((When::Frame(cursor), vk));
                }
            } else if let (Some(t), Some(vk)) = (at.parse().ok().and_then(|s| Duration::try_from_secs_f64(s).ok()), key_code(keys)) {
                entries.push((When::Time(t), vk));
            }
            if entries.len() == before {
                refused.push(t.to_owned());
            }
        }
        let fired = vec![None; entries.len()];
        Script { entries, fired, rules, refused }
    }

    /// One poll at `now` and presented `frame`: the keys down, and those
    /// that went down at this poll.
    fn tick(&mut self, now: Duration, frame: u64, look: &dyn Fn(&str) -> Option<u64>) -> (Vec<u8>, Vec<u8>) {
        let (mut down, mut fresh) = (Vec::new(), Vec::new());
        for r in &mut self.rules {
            if matches!(r.kind, RuleKind::Hold { .. }) {
                // Down at every poll its condition holds; fresh only as it goes down.
                if r.tick(now, look) {
                    if r.pressed.is_none() {
                        fresh.push(r.key);
                    }
                    r.pressed = Some(now);
                    down.push(r.key);
                } else {
                    r.pressed = None;
                }
                continue;
            }
            if r.tick(now, look) {
                r.next_key();
                r.pressed = Some(now);
                down.push(r.key);
                fresh.push(r.key);
            } else if r.pressed.is_some_and(|t| now < t + KEY_HOLD) {
                down.push(r.key);
            }
        }
        for (i, &(when, vk)) in self.entries.iter().enumerate() {
            let reached = match when {
                When::Time(t) => now >= t,
                When::Frame(f) => frame >= f,
            };
            match self.fired[i] {
                None if reached => {
                    self.fired[i] = Some((now, frame));
                    down.push(vk);
                    fresh.push(vk);
                }
                Some((t, f)) if match when {
                    When::Time(_) => now < t + KEY_HOLD,
                    When::Frame(_) => frame < f + KEY_HOLD_FRAMES,
                } => down.push(vk),
                _ => {}
            }
        }
        (down, fresh)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn key_grammar_parses_seconds_frames_relative_and_repeats() {
        let s = Script::parse("12:Enter, f288:Down,+90:Up,f1000-1015/5:Up|Enter,+10:a,bad,f9:Nope,f5-9/0:Up,x:Enter");
        let f = |n| When::Frame(n);
        assert_eq!(
            s.entries,
            vec![
                (When::Time(Duration::from_secs(12)), 0x0D),
                (f(288), 0x28),
                (f(378), 0x26),
                (f(1000), 0x26),
                (f(1005), 0x0D),
                (f(1010), 0x26),
                (f(1025), b'A'),
            ]
        );
    }

    #[test]
    fn state_keyed_grammar_and_refusals() {
        let s = Script::parse("when:session_state==1+1500:Enter, every:1500:Enter while:session_state==1|3|13 until:session_state==0, every:200:Down;until:save_op>=0x4, f288:Enter");
        assert_eq!(s.entries, vec![(When::Frame(288), 0x0D)]);
        assert_eq!(s.rules.len(), 3);
        assert!(s.refused.is_empty(), "{:?}", s.refused);
        let c = |name: &str, op, values: &[u64]| Cond { name: name.into(), op, values: values.to_vec() };
        assert_eq!(s.rules[0].kind, RuleKind::Once { cond: c("session_state", Op::Eq, &[1]), delay: Duration::from_millis(1500) });
        assert_eq!(s.rules[1].kind, RuleKind::Every { period: Duration::from_millis(1500), during: Some(c("session_state", Op::Eq, &[1, 3, 13])), until: Some(c("session_state", Op::Eq, &[0])) });
        assert_eq!(s.rules[2].kind, RuleKind::Every { period: Duration::from_millis(200), during: None, until: Some(c("save_op", Op::Ge, &[4])) });
        // Malformed state forms are refused, not half-taken.
        for bad in [
            "when:session_state=1:Enter",      // no operator
            "when:==1:Enter",                  // no name
            "when:session_state==x:Enter",     // not a number
            "when:session_state==1:Nope",      // no such key
            "when:session_state==1+abc:Enter", // bad delay
            "when:session_state>=1|2:Enter",   // a set needs == or !=
            "every:0:Enter",                   // zero period
            "every:1500:Enter while:",         // empty condition
            "every:1500:Enter sometimes:x==1", // unknown clause
            "every:abc:Enter",
            "when:session state==1:Enter",     // space in a name
        ] {
            let s = Script::parse(bad);
            assert!(s.rules.is_empty() && s.entries.is_empty(), "{bad}");
            assert_eq!(s.refused, vec![bad.to_string()], "{bad}");
        }
        // The old forms report what they drop too.
        let s = Script::parse("bad,f9:Nope,12:Enter");
        assert_eq!(s.refused, vec!["bad".to_string(), "f9:Nope".to_string()]);
    }

    #[test]
    fn hold_keeps_its_key_down_while_its_condition_holds() {
        let mut s = Script::parse("hold:Enter while:cine0_state==6|8 until:session_state==9");
        assert!(s.refused.is_empty(), "{:?}", s.refused);
        assert_eq!(s.rules[0].kind, RuleKind::Hold { during: Some(Cond::parse("cine0_state==6|8").unwrap()), until: Some(Cond::parse("session_state==9").unwrap()) });
        let state = std::cell::Cell::new((5u64, 0u64));
        let look = |n: &str| match n {
            "cine0_state" => Some(state.get().0),
            "session_state" => Some(state.get().1),
            _ => None,
        };
        let ms = Duration::from_millis;
        let mut at = |t: u64, cine: u64, sess: u64| {
            state.set((cine, sess));
            s.tick(ms(t), 0, &look)
        };
        assert_eq!(at(0, 5, 0), (vec![], vec![]), "not playing: up");
        assert_eq!(at(100, 6, 0), (vec![0x0D], vec![0x0D]), "playing: down, fresh");
        for t in [400, 900, 5000] {
            assert_eq!(at(t, 8, 0), (vec![0x0D], vec![]), "still playing at {t} ms: held, not pressed again");
        }
        assert_eq!(at(5100, 10, 0), (vec![], vec![]), "ended: let go at once");
        assert_eq!(at(5200, 6, 0), (vec![0x0D], vec![0x0D]), "the next cinematic: down again");
        assert_eq!(at(5300, 6, 9), (vec![], vec![]), "until: let go");
        assert_eq!(at(5400, 6, 0), (vec![], vec![]), "and never again");
        // One key only; the clauses as every's.
        for bad in ["hold:Up|Enter while:x==1", "hold:Nope", "hold:Enter sometimes:x==1", "hold:Enter while:"] {
            assert!(!Script::parse(bad).refused.is_empty(), "{bad}");
        }
        // Without while: down from the first poll until `until`.
        let mut h = Script::parse("hold:a until:session_state==0");
        assert_eq!(h.tick(ms(0), 0, &|_| Some(1)), (vec![b'A'], vec![b'A']));
        assert_eq!(h.tick(ms(50), 0, &|_| Some(0)), (vec![], vec![]));
    }

    #[test]
    fn every_presses_its_keys_in_turn() {
        let mut s = Script::parse("every:100:Up|Up|Enter;while:save_op==4;until:session_state==0");
        let mut keys = Vec::new();
        for ms in (0..1000u64).step_by(10) {
            let v = |n: &str| match n { "save_op" => Some(4), "session_state" => Some(if ms < 650 { 13 } else { 0 }), _ => None };
            keys.extend(s.tick(Duration::from_millis(ms), 0, &v).1);
        }
        assert_eq!(keys, vec![0x26, 0x26, 0x0D, 0x26, 0x26, 0x0D]);
        assert!(!Script::parse("every:100:Up|Nope").refused.is_empty());
    }

    #[test]
    fn state_keyed_rules_press_on_state_not_frames() {
        // A front end: state 1 at 4.0 s, 3 at 6.0 s, 13 at 7.7 s, 0 at 22.8 s.
        let state = |t: f64| if t < 4.0 { 0 } else if t < 6.0 { 1 } else if t < 7.7 { 3 } else if t < 22.8 { 13 } else { 0 };
        let mut s = Script::parse("every:1500:Enter while:session_state==1|3|13 until:session_state==0");
        let mut presses = Vec::new();
        for ms in (0..30_000u64).step_by(50) {
            let t = ms as f64 / 1000.0;
            let v = state(t);
            let (_, fresh) = s.tick(Duration::from_millis(ms), 0, &|n| (n == "session_state").then_some(v));
            if !fresh.is_empty() {
                presses.push(ms);
            }
        }
        // 1.5 s after each new state, then every 1.5 s; none after state 0.
        assert_eq!(&presses[..3], &[5500, 7500, 9200], "{presses:?}");
        assert!(presses.iter().all(|&ms| ms < 22_800));
        assert_eq!(presses.iter().filter(|&&ms| ms >= 9200).count(), 10);
        // `when` fires once after its delay; an unread value never holds.
        let mut s = Script::parse("when:session_state==3+1000:Enter");
        assert!(s.tick(Duration::from_millis(100), 0, &|_| None).1.is_empty());
        assert!(s.tick(Duration::from_millis(200), 0, &|_| Some(3)).1.is_empty());
        assert_eq!(s.tick(Duration::from_millis(1200), 0, &|_| Some(0)).1, vec![0x0D]);
        assert!(s.tick(Duration::from_millis(1300), 0, &|_| Some(3)).1.is_empty(), "once");
        assert!(s.tick(Duration::from_millis(1500), 0, &|_| Some(3)).0.is_empty(), "held 0.2 s only");
    }

    #[test]
    fn scripted_keys_fire_once_each_and_are_held() {
        let mut s = Script::parse("1.0:Enter,f3:Down,+4:Up,f10-20/5:Left");
        let mut fresh_all = Vec::new();
        let mut down_frames = Vec::new();
        for frame in 0..30_u64 {
            let (down, fresh) = s.tick(Duration::from_millis(frame * 33), frame, &|_| None);
            fresh_all.extend(fresh);
            if !down.is_empty() {
                down_frames.push(frame);
            }
        }
        // 1.0 s is frame 31 at 33 ms: not reached; the rest in order, once.
        assert_eq!(fresh_all, vec![0x28, 0x26, 0x25, 0x25]);
        assert_eq!(down_frames, vec![3, 4, 7, 8, 10, 11, 15, 16]);
        // a skipped frame still fires at the first poll after
        let mut s = Script::parse("f5:Enter");
        assert_eq!(s.tick(Duration::ZERO, 9, &|_| None).1, vec![0x0D]);
        assert!(s.tick(Duration::ZERO, 12, &|_| None).0.is_empty());
        // seconds keys hold 0.2 s
        let mut s = Script::parse("2:Enter");
        assert_eq!(s.tick(Duration::from_millis(2100), 0, &|_| None).0, vec![0x0D]);
        assert!(s.tick(Duration::from_millis(2301), 0, &|_| None).0.is_empty());
    }

    #[derive(Default)]
    struct FakeHost {
        pads: [Option<PadState>; PADS],
        keys: Vec<KeyState>,
        rumble: Vec<(usize, u16, u16)>,
    }

    struct Fake(Arc<Mutex<FakeHost>>);

    impl Source for Fake {
        fn pad(&mut self, slot: usize) -> Option<PadState> {
            self.0.lock().unwrap().pads[slot]
        }
        fn keys(&mut self) -> Vec<KeyState> {
            self.0.lock().unwrap().keys.clone()
        }
        fn rumble(&mut self, slot: usize, low: u16, high: u16) {
            self.0.lock().unwrap().rumble.push((slot, low, high));
        }
    }

    fn fake() -> (&'static Hub, Arc<Mutex<FakeHost>>) {
        let host = Arc::new(Mutex::new(FakeHost::default()));
        let hub = Hub::new(Box::new(Fake(Arc::clone(&host))));
        hub.poll();
        (hub, host)
    }

    fn key(virtual_key: u8) -> KeyState {
        KeyState {
            scan_code: u32::from(virtual_key) + 0x100,
            virtual_key,
            ..Default::default()
        }
    }

    fn slot<F: Copy, T>(object: *const T, offset: usize) -> F {
        // SAFETY: our objects start with their table, which has the slot.
        let f = unsafe { *(*(object as *const *const *const ())).add(offset / 8) };
        // SAFETY: the caller names the slot's signature.
        unsafe { std::mem::transmute_copy(&f) }
    }

    /// Device callbacks seen: device index, current and previous status.
    type Log = Mutex<Vec<(usize, u32, u32)>>;

    extern "system" fn record(
        _token: u64,
        context: *mut c_void,
        device: *const Device,
        _timestamp: u64,
        current: u32,
        previous: u32,
    ) {
        // SAFETY: the test passes its log and one of the hub's devices.
        let (log, index) = unsafe { (&*(context as *const Log), (*device).index) };
        log.lock().unwrap().push((index, current, previous));
    }

    fn register(hub: &Hub, kind: u32, enumeration: u32, log: &Log) -> u64 {
        type Register = extern "system" fn(
            This,
            *const Device,
            u32,
            u32,
            u32,
            *mut c_void,
            Option<DeviceCallback>,
            *mut u64,
        ) -> HRESULT;
        let mut token = 0;
        let context = log as *const Log as *mut c_void;
        let register = slot::<Register, _>(&hub.object, 0x48);
        let hr = register(
            &hub.object,
            std::ptr::null(),
            kind,
            1,
            enumeration,
            context,
            Some(record),
            &mut token,
        );
        assert_eq!(hr, S_OK);
        assert_ne!(token, 0);
        token
    }

    fn result(hr: HRESULT, out: *const c_void) -> Result<*const Reading, HRESULT> {
        if hr == S_OK {
            Ok(out.cast())
        } else {
            assert!(out.is_null());
            Err(hr)
        }
    }

    fn current(hub: &Hub, kind: u32, device: usize) -> Result<*const Reading, HRESULT> {
        type Current = extern "system" fn(This, u32, *const Device, Out) -> HRESULT;
        let mut out = std::ptr::null();
        let hr = slot::<Current, _>(&hub.object, 0x20)(
            &hub.object,
            kind,
            &hub.devices[device],
            &mut out,
        );
        result(hr, out)
    }

    fn walk(hub: &Hub, offset: usize, from: *const Reading) -> Result<*const Reading, HRESULT> {
        type Walk = extern "system" fn(This, *const Reading, u32, *const Device, Out) -> HRESULT;
        let mut out = std::ptr::null();
        let device = reading(from).device;
        let hr =
            slot::<Walk, _>(&hub.object, offset)(&hub.object, from, GAMEPAD_KIND, device, &mut out);
        result(hr, out)
    }

    fn pad(reading: *const Reading) -> PadState {
        let mut state = PadState::default();
        let get =
            slot::<extern "system" fn(*const Reading, *mut PadState) -> bool, _>(reading, 0xb0);
        assert!(get(reading, &mut state));
        state
    }

    fn stamp(reading: *const Reading) -> (u64, u64) {
        let time = slot::<extern "system" fn(*const Reading) -> u64, _>(reading, 0x28);
        let sequence = slot::<extern "system" fn(*const Reading, u32) -> u64, _>(reading, 0x20);
        (time(reading), sequence(reading, GAMEPAD_KIND))
    }

    fn release(reading: *const Reading) {
        slot::<extern "system" fn(*const Reading) -> u32, _>(reading, 0x10)(reading);
    }

    #[test]
    fn ordinal_one_is_create() {
        assert_eq!(crate::lookup("GameInput.dll", "#1"), Some(GameInputCreate as usize));
        assert_eq!(crate::lookup("GameInput.dll", "#2"), None);
    }

    #[test]
    fn pads_connect_report_and_disconnect() {
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, GAMEPAD_KIND, 2, &log);
        // Blocking enumeration ran before registration returned: with no
        // host pad the keyboard's stand-in is the only one.
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

        host.lock().unwrap().pads[1] = Some(PadState {
            buttons: A,
            left_x: 0.5,
            ..Default::default()
        });
        hub.poll();
        assert_eq!(
            log.lock().unwrap()[1..],
            [(1, STATUS_CONNECTED, 0), (FALLBACK, 0, STATUS_CONNECTED)]
        );

        let first = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(pad(first).buttons, A);
        assert_eq!(pad(first).left_x, 0.5);
        // No change, no new reading.
        hub.poll();
        let again = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(again, first);
        release(again);

        std::thread::sleep(Duration::from_millis(2));
        host.lock().unwrap().pads[1].as_mut().unwrap().buttons = B;
        hub.poll();
        let second = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(pad(second).buttons, B);
        let (t1, s1) = stamp(first);
        let (t2, s2) = stamp(second);
        assert!(t2 > t1 && s2 == s1 + 1);
        assert!(t2 <= now());

        // The title's edge check: previous of current, and back again.
        assert_eq!(walk(hub, 0x30, second), Ok(first));
        assert_eq!(walk(hub, 0x28, first), Ok(second));
        assert_eq!(walk(hub, 0x28, second), Err(READING_NOT_FOUND));
        assert_eq!(walk(hub, 0x30, first), Err(READING_NOT_FOUND));
        for r in [first, first, second, second] {
            release(r);
        }

        host.lock().unwrap().pads[1] = None;
        hub.poll();
        assert_eq!(
            log.lock().unwrap()[3..],
            [(1, 0, STATUS_CONNECTED), (FALLBACK, STATUS_CONNECTED, 0)]
        );
        assert_eq!(current(hub, GAMEPAD_KIND, 1), Err(DEVICE_DISCONNECTED));
        // The stand-in is back, at rest.
        let rest = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(pad(rest), PadState::default());
        release(rest);
    }

    #[test]
    fn keys_feed_the_keyboard_and_the_stand_in_pad() {
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, KEYBOARD_KIND, 2, &log);
        assert_eq!(*log.lock().unwrap(), [(KEYBOARD, STATUS_CONNECTED, 0)]);

        let keys = vec![key(b'W'), key(0x0D), key(b'L'), key(b'C')];
        host.lock().unwrap().keys = keys.clone();
        hub.poll();
        let r = current(hub, KEYBOARD_KIND, KEYBOARD).unwrap();
        let count = slot::<extern "system" fn(*const Reading) -> u32, _>(r, 0x70);
        let state =
            slot::<extern "system" fn(*const Reading, u32, *mut KeyState) -> u32, _>(r, 0x78);
        assert_eq!(count(r), 4);
        let mut got = [KeyState::default(); 8];
        assert_eq!(state(r, 8, got.as_mut_ptr()), 4);
        assert_eq!(got[..4], keys[..]);
        release(r);

        let r = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(
            pad(r),
            PadState {
                buttons: A,
                left_y: 1.0,
                right_x: 1.0,
                right_trigger: 1.0,
                ..Default::default()
            }
        );
        release(r);
        // A keyboard reading is no gamepad reading.
        assert_eq!(current(hub, GAMEPAD_KIND, KEYBOARD), Err(READING_NOT_FOUND));
    }

    #[test]
    fn async_enumeration_waits_for_the_poll_and_unregistering_stops_callbacks() {
        let (hub, host) = fake();
        let log = Log::default();
        let token = register(hub, GAMEPAD_KIND, ASYNC_ENUMERATION, &log);
        assert!(log.lock().unwrap().is_empty());
        hub.poll();
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

        let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(&hub.object, 0x68);
        assert!(unregister(&hub.object, token, 5000));
        host.lock().unwrap().pads[0] = Some(PadState::default());
        hub.poll();
        assert_eq!(log.lock().unwrap().len(), 1);
    }

    #[test]
    fn unregistering_waits_for_a_running_callback() {
        use std::sync::atomic::AtomicBool;
        static INSIDE: AtomicBool = AtomicBool::new(false);
        static LEAVE: AtomicBool = AtomicBool::new(false);
        static FROM_INSIDE: AtomicU32 = AtomicU32::new(0);
        extern "system" fn slow(
            token: u64,
            context: *mut c_void,
            _: *const Device,
            _: u64,
            _: u32,
            _: u32,
        ) {
            // From inside its own callback, unregistering fails at once.
            let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(context, 0x68);
            if !unregister(context.cast(), token, 0) {
                FROM_INSIDE.fetch_add(1, Ordering::SeqCst);
            }
            INSIDE.store(true, Ordering::SeqCst);
            while !LEAVE.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        let (hub, host) = fake();
        type Register = extern "system" fn(
            This,
            *const Device,
            u32,
            u32,
            u32,
            *mut c_void,
            Option<DeviceCallback>,
            *mut u64,
        ) -> HRESULT;
        let register = slot::<Register, _>(&hub.object, 0x48);
        let mut token = 0;
        let context = (&raw const hub.object).cast_mut().cast();
        assert_eq!(
            register(&hub.object, std::ptr::null(), GAMEPAD_KIND, 1, 0, context, Some(slow), &mut token),
            S_OK
        );
        host.lock().unwrap().pads[1] = Some(PadState::default());
        let poller = std::thread::spawn(|| hub.poll());
        while !INSIDE.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(&hub.object, 0x68);
        assert!(!unregister(&hub.object, token, 5000), "times out while it runs");
        let leave = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(50));
            LEAVE.store(true, Ordering::SeqCst);
        });
        let start = Instant::now();
        assert!(unregister(&hub.object, token, u64::MAX), "waits for it");
        assert!(start.elapsed() >= Duration::from_millis(40));
        assert_eq!(FROM_INSIDE.load(Ordering::SeqCst), 1);
        poller.join().unwrap();
        leave.join().unwrap();
    }

    #[test]
    fn devices_are_found_by_id_and_rumble_reaches_the_host() {
        let (hub, host) = fake();
        host.lock().unwrap().pads[2] = Some(PadState::default());
        hub.poll();
        let device = &hub.devices[2] as *const Device;
        let info = slot::<extern "system" fn(*const Device) -> *const u8, _>(device, 0x18);
        // SAFETY: the info has the id at +0x20.
        let id = unsafe { *(info(device).add(0x20) as *const [u8; 32]) };
        assert_ne!(id, [0; 32]);

        type Find = extern "system" fn(This, *const [u8; 32], Out) -> HRESULT;
        let find = slot::<Find, _>(&hub.object, 0x80);
        let mut out = std::ptr::null();
        assert_eq!(find(&hub.object, &id, &mut out), S_OK);
        assert_eq!(out, device.cast());
        assert_eq!(find(&hub.object, &[1; 32], &mut out), DEVICE_NOT_FOUND);
        assert!(out.is_null());

        type Rumble = extern "system" fn(*const Device, *const [f32; 4]);
        let rumble = slot::<Rumble, _>(device, 0x50);
        rumble(device, &[0.5, 1.0, 1.0, 1.0]);
        rumble(&hub.devices[FALLBACK], &[1.0; 4]);
        rumble(device, std::ptr::null());
        assert_eq!(host.lock().unwrap().rumble, [(2, 32767, 65535), (2, 0, 0)]);
    }
}
