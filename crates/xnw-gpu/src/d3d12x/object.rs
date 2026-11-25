//! What every console graphics object starts with: a function table
//! pointer, bookkeeping bits and a reference count. Tables begin with
//! IUnknown's three slots. Every slot we have not written logs
//! `<object> slot <n> (offset 0x..)` with its caller once and fails.
//!
//! Ported from VIctory `crates/victory-gpu/src/object.rs` at d2a4bae
//! (GPL-3.0-or-later): header layout, private data, the naming stubs.
//! Changed: logging through `xnw-core`, a slot inventory per table.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

pub type HRESULT = i32;
pub const S_OK: HRESULT = 0;
pub const S_FALSE: HRESULT = 1;
pub const E_NOTIMPL: HRESULT = 0x8000_4001_u32 as i32;
pub const E_NOINTERFACE: HRESULT = 0x8000_4002_u32 as i32;
pub const E_POINTER: HRESULT = 0x8000_4003_u32 as i32;
#[allow(dead_code)] // no caller since lists record without failing
pub const E_FAIL: HRESULT = 0x8000_4005_u32 as i32;
pub const E_INVALIDARG: HRESULT = 0x8007_0057_u32 as i32;
pub const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;
/// What D3D and DXGI return for a missing item (adapter, output, private data).
pub const NOT_FOUND: HRESULT = 0x887A_0002_u32 as i32;
pub const MORE_DATA: HRESULT = 0x887A_0003_u32 as i32;

pub type Iid = [u8; 16];

/// An interface id from its text form's fields, in memory order.
pub const fn guid(a: u32, b: u16, c: u16, d: [u8; 8]) -> Iid {
    let (a, b, c) = (a.to_le_bytes(), b.to_le_bytes(), c.to_le_bytes());
    [a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]]
}

pub fn guid_text(g: &Iid) -> String {
    let a = u32::from_le_bytes([g[0], g[1], g[2], g[3]]);
    let b = u16::from_le_bytes([g[4], g[5]]);
    let c = u16::from_le_bytes([g[6], g[7]]);
    let d: String = g[8..].iter().map(|x| format!("{x:02x}")).collect();
    format!("{a:08x}-{b:04x}-{c:04x}-{}-{}", &d[..4], &d[4..])
}

/// The header: table at +0, bits at +8, reference count at +0xc.
#[repr(C)]
pub struct Object {
    functions: *const *const (),
    /// Device index (bits 0-2) and bookkeeping bits; all zero.
    bits: u32,
    refs: AtomicU32,
}
// SAFETY: the table is static and immutable; the count is atomic.
unsafe impl Sync for Object {}
// SAFETY: as above.
unsafe impl Send for Object {}

impl Object {
    pub(crate) const fn new<const N: usize>(table: &'static Table<N>) -> Self {
        Object { functions: table.0.as_ptr(), bits: 0, refs: AtomicU32::new(0) }
    }
}

pub type This = *const Object;

#[repr(transparent)]
pub(crate) struct Table<const N: usize>(pub [*const (); N]);
// SAFETY: function addresses only.
unsafe impl<const N: usize> Sync for Table<N> {}

pub(crate) extern "system" fn add_ref(this: This) -> u32 {
    // SAFETY: `this` is one of our objects.
    unsafe { &*this }.refs.fetch_add(1, Ordering::Relaxed) + 1
}

/// A reference, only while the object still has one (at 0 it is being
/// freed). The caller knows its memory is not freed yet.
pub(crate) fn try_add_ref(this: This) -> bool {
    // SAFETY: as the caller states.
    let refs = &unsafe { &*this }.refs;
    refs.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| (n > 0).then_some(n + 1)).is_ok()
}

/// Drops a reference unless it is the last; `false` if it is (the
/// caller lets it go later, where freeing the object takes no lock it holds).
pub(crate) fn release_unless_last(this: This) -> bool {
    // SAFETY: `this` is one of our objects, held by the caller.
    let refs = &unsafe { &*this }.refs;
    refs.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| (n > 1).then(|| n - 1)).is_ok()
}

/// Release of static objects: counts down, never frees.
pub(crate) extern "system" fn release(this: This) -> u32 {
    // SAFETY: `this` is one of our objects.
    let refs = &unsafe { &*this }.refs;
    let _ = refs.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1));
    refs.load(Ordering::Relaxed)
}

/// Answers an interface query (or any call handing out an object): a new
/// reference to `found`, or null and `E_NOINTERFACE`.
pub(crate) fn answer(kind: usize, iid: *const Iid, found: Option<This>, out: *mut This) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    if let Some(object) = found {
        add_ref(object);
        // SAFETY: checked non-null; the title passes room for a pointer.
        unsafe { out.write(object) };
        return S_OK;
    }
    // SAFETY: the title passes an interface id (null is tolerated).
    let iid = unsafe { iid.as_ref() }.map(guid_text).unwrap_or_default();
    xnw_core::warn!("d3d12x: {} asked for unknown interface {iid}", KINDS[kind]);
    // SAFETY: checked non-null.
    unsafe { out.write(std::ptr::null()) };
    E_NOINTERFACE
}

/// Whether `iid` is one of `known`.
pub(crate) fn is(iid: *const Iid, known: &[Iid]) -> bool {
    // SAFETY: the title passes an interface id (null is tolerated).
    unsafe { iid.as_ref() }.is_some_and(|i| known.contains(i))
}

/// Hands a new object to the title as one of `known`, or drops it. `T`
/// starts with its `Object`; its table's release slot is
/// [`release_boxed::<T>`].
pub(crate) fn hand_out<T>(kind: usize, object: Box<T>, iid: *const Iid, known: &[Iid], out: *mut This) -> HRESULT {
    if !is(iid, known) {
        return answer(kind, iid, None, out);
    }
    if out.is_null() {
        // Asked only whether creation would succeed.
        return S_FALSE;
    }
    answer(kind, iid, Some(place(KINDS[kind], object).cast()), out)
}

/// Where a new object lives: its own guarded region under
/// `XNW_OBJECT_GUARD`, else the heap.
pub(crate) fn place<T>(kind: &'static str, object: Box<T>) -> *mut T {
    match super::guard::alloc(kind, size_of::<T>(), align_of::<T>()) {
        Some(at) => {
            let at = at.cast::<T>();
            // SAFETY: fresh memory of the size and alignment of `T`.
            unsafe { at.write(*object) };
            at
        }
        None => Box::into_raw(object),
    }
}

/// Frees an object made by [`place`].
///
/// # Safety
/// `this` is a `T` made by `place` with no reference left.
pub(crate) unsafe fn free<T>(this: *mut T) {
    if super::guard::mode() != 0 {
        // SAFETY: as the caller states.
        unsafe { std::ptr::drop_in_place(this) };
        if super::guard::free(this.cast()) {
            return;
        }
        // Made before the guard was asked (heap memory): its drop ran.
        // SAFETY: the box's memory without dropping its value again.
        unsafe { std::alloc::dealloc(this.cast(), std::alloc::Layout::new::<T>()) };
        return;
    }
    // SAFETY: as the caller states: a `Box<T>`.
    drop(unsafe { Box::from_raw(this) });
}

/// Release for objects made by [`hand_out`]: the last reference frees it.
pub(crate) fn release_boxed<T>(this: This) -> u32 {
    // SAFETY: `this` is the header of a boxed `T`.
    let refs = &unsafe { &*this }.refs;
    match refs.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1)) {
        Ok(1) => {
            private_data().retain(|&(object, _), _| object != this as usize);
            // SAFETY: that was the last reference to an object `place` made.
            unsafe { free(this.cast_mut().cast::<T>()) };
            0
        }
        Ok(n) => n - 1,
        Err(_) => 0,
    }
}

/// Drops a reference; the count left (`None`: it had none).
pub(crate) fn count_down(this: This) -> Option<u32> {
    // SAFETY: `this` is one of our objects.
    let refs = &unsafe { &*this }.refs;
    refs.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1)).ok().map(|n| n - 1)
}

/// Forgets an object's private data (its release, done by its owner).
pub(crate) fn drop_private(this: This) {
    private_data().retain(|&(object, _), _| object != this as usize);
}

/// Private data of every object, by object address and id.
static PRIVATE: Mutex<BTreeMap<(usize, Iid), Vec<u8>>> = Mutex::new(BTreeMap::new());

fn private_data() -> std::sync::MutexGuard<'static, BTreeMap<(usize, Iid), Vec<u8>>> {
    PRIVATE.lock().unwrap_or_else(|e| e.into_inner())
}

extern "system" fn get_private_data(this: This, id: *const Iid, size: *mut u32, data: *mut u8) -> HRESULT {
    // SAFETY: the title passes an id and its buffer size (null is tolerated).
    let (Some(id), Some(size)) = (unsafe { (id.as_ref(), size.as_mut()) }) else {
        return E_INVALIDARG;
    };
    let map = private_data();
    let Some(value) = map.get(&(this as usize, *id)) else {
        *size = 0;
        return NOT_FOUND;
    };
    let len = value.len() as u32;
    let room = std::mem::replace(size, len);
    if data.is_null() {
        return S_OK;
    }
    if room < len {
        return MORE_DATA;
    }
    // SAFETY: the buffer holds `room` >= `len` bytes.
    unsafe { data.copy_from_nonoverlapping(value.as_ptr(), value.len()) };
    S_OK
}

/// Null data removes the entry.
extern "system" fn set_private_data(this: This, id: *const Iid, size: u32, data: *const u8) -> HRESULT {
    // SAFETY: the title passes an id (null is tolerated).
    let Some(&id) = (unsafe { id.as_ref() }) else {
        return E_INVALIDARG;
    };
    let mut map = private_data();
    if data.is_null() {
        map.remove(&(this as usize, id));
    } else {
        // SAFETY: the title passes `size` bytes.
        let value = unsafe { std::slice::from_raw_parts(data, size as usize) };
        map.insert((this as usize, id), value.to_vec());
    }
    S_OK
}

/// The object methods' name setter: names only reach the trace.
extern "system" fn set_name(_: This, _name: *const u16) -> HRESULT {
    S_OK
}

/// The device-child method: the device that made the object.
pub(crate) extern "system" fn parent_device(_: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let device = super::device::device();
    answer(DEVICE, iid, super::device::is_device(iid).then_some(device), out)
}

/// Accepts and ignores (debug-only settings, markers).
pub(crate) extern "system" fn ignore() -> HRESULT {
    S_OK
}

pub(crate) const DEVICE: usize = 0;
pub(crate) const DXGI_DEVICE: usize = 1;
pub(crate) const ADAPTER: usize = 2;
pub(crate) const FACTORY: usize = 3;
pub(crate) const OUTPUT: usize = 4;
pub(crate) const HEAP: usize = 5;
pub(crate) const RESOURCE: usize = 6;
pub(crate) const FENCE: usize = 7;
pub(crate) const QUEUE: usize = 8;
pub(crate) const ALLOCATOR: usize = 9;
pub(crate) const LIST: usize = 10;
pub(crate) const DESCRIPTOR_HEAP: usize = 11;
pub(crate) const ROOT_SIGNATURE: usize = 12;
pub(crate) const QUERY_HEAP: usize = 13;
pub(crate) const COMMAND_SIGNATURE: usize = 14;
pub(crate) const PIPELINE: usize = 15;
pub(crate) const KINDS: [&str; 16] = [
    "device",
    "DXGI device",
    "adapter",
    "factory",
    "output",
    "heap",
    "resource",
    "fence",
    "queue",
    "allocator",
    "list",
    "descriptor heap",
    "root signature",
    "query heap",
    "command signature",
    "pipeline",
];

const MAX_SLOTS: usize = 160;

/// Which unwritten slots have been reported; each is logged once.
static REPORTED: [[AtomicBool; MAX_SLOTS]; KINDS.len()] =
    [const { [const { AtomicBool::new(false) }; MAX_SLOTS] }; KINDS.len()];

/// Base of the title image, for naming callers (`title+0x..`).
static TITLE_BASE: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn title_base() -> usize {
    TITLE_BASE.load(Ordering::Relaxed)
}

pub fn set_title_base(base: usize) {
    TITLE_BASE.store(base, Ordering::Relaxed);
}

/// Logged once per slot, with the caller. Wrong for a slot that returns a
/// value instead of an `HRESULT` (the caller gets `E_NOTIMPL` as one).
extern "system" fn missing<const KIND: usize, const SLOT: usize>() -> HRESULT {
    if !REPORTED[KIND][SLOT].swap(true, Ordering::Relaxed) {
        xnw_core::warn!("d3d12x: {} slot {SLOT} (offset {:#x}) not implemented, caller {}", KINDS[KIND], SLOT * 8, caller());
    }
    E_NOTIMPL
}

/// Our caller's caller's return address: `title+0x..`, `module+0x..`, or raw.
#[inline(never)]
pub(crate) fn caller() -> String {
    use xnw_core::win;
    let mut frame = std::ptr::null_mut();
    // SAFETY: room for one frame; skips this function and its caller.
    if unsafe { win::RtlCaptureStackBackTrace(2, 1, &mut frame, std::ptr::null_mut()) } != 1 {
        return "?".into();
    }
    let at = frame as usize;
    let mut module = std::ptr::null_mut();
    const FROM_ADDRESS: u32 = 4;
    const UNCHANGED_REFCOUNT: u32 = 2;
    // SAFETY: looks up the module holding the address, no reference taken.
    let found = unsafe { win::GetModuleHandleExW(FROM_ADDRESS | UNCHANGED_REFCOUNT, frame as *const u16, &mut module) };
    if found != 0 {
        let mut name = [0u16; 260];
        // SAFETY: room for 260 units.
        let len = unsafe { win::GetModuleFileNameW(module, name.as_mut_ptr(), 260) } as usize;
        let path = String::from_utf16_lossy(&name[..len]);
        let file = path.rsplit('\\').next().unwrap_or("?").to_string();
        return format!("{file}+{:#x}", at - module as usize);
    }
    let base = TITLE_BASE.load(Ordering::Relaxed);
    if base != 0 && at > base {
        return format!("title+{:#x}", at - base);
    }
    format!("{at:#x}")
}

macro_rules! missing {
    ($kind:ident; $($slot:literal)*) => { [$(missing::<$kind, $slot> as *const ()),*] };
}

/// What a written slot does, for the inventory.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Role {
    /// Does what the console method does (over host D3D12 or in memory).
    Done,
    /// Accepted, counted and dropped (draws, pipelines, views, state).
    Dropped,
}
pub(crate) use Role::{Done, Dropped};

/// One written slot: index, function, role, where VIctory has it.
pub(crate) type Slot = (usize, *const (), Role, &'static str);

/// A table of `N` slots: the written `slots`, the object methods (private
/// data, name) when `d3d12_object`, IUnknown's reference counting, and
/// naming stubs everywhere else. Tables are longer than the interfaces we
/// know, so a method a newer title adds logs instead of jumping into
/// whatever follows the table.
pub(crate) const fn table<const KIND: usize, const N: usize>(d3d12_object: bool, slots: &[Slot]) -> Table<N> {
    let all = missing!(KIND;
        0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
        32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60
        61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89
        90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113
        114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135
        136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157
        158 159
    );
    assert!(N <= all.len());
    let mut f = [std::ptr::null(); N];
    let mut i = 0;
    while i < N {
        f[i] = all[i];
        i += 1;
    }
    f[1] = add_ref as *const ();
    f[2] = release as *const ();
    if d3d12_object {
        f[3] = get_private_data as *const (); // 0x18
        f[4] = set_private_data as *const (); // 0x20
        f[6] = set_name as *const (); // 0x30
    }
    let mut i = 0;
    while i < slots.len() {
        f[slots[i].0] = slots[i].1;
        i += 1;
    }
    Table(f)
}

/// Slot counts of one table: (written and working, dropped, naming stubs).
pub(crate) fn census(n: usize, d3d12_object: bool, slots: &[Slot]) -> (usize, usize, usize) {
    let mut done: Vec<usize> = vec![1, 2];
    if d3d12_object {
        done.extend([3, 4, 6]);
    }
    let mut dropped = Vec::new();
    for s in slots {
        match s.2 {
            Done => done.push(s.0),
            Dropped => dropped.push(s.0),
        }
    }
    done.retain(|i| !dropped.contains(i));
    done.sort_unstable();
    done.dedup();
    dropped.sort_unstable();
    dropped.dedup();
    (done.len(), dropped.len(), n - done.len() - dropped.len())
}

/// Calls slot `offset` of `object`'s table as `F` (tests play the title).
///
/// # Safety
/// `object` is one of ours and `F` is the slot's signature.
pub unsafe fn slot<F: Copy>(object: This, offset: usize) -> F {
    // SAFETY: as the caller promises.
    let f = unsafe { *(*object).functions.add(offset / 8) };
    // SAFETY: as the caller promises.
    unsafe { std::mem::transmute_copy(&f) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guids_round_trip_through_text() {
        let g = guid(0x189819f1, 0x1db6, 0x4b57, [0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7]);
        assert_eq!(g[..4], [0xf1, 0x19, 0x98, 0x18]);
        assert_eq!(guid_text(&g), "189819f1-1db6-4b57-be54-1821339b85f7");
    }
}
