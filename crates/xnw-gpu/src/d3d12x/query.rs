//! Query heaps. Queries are answered when resolved (`list.rs`); the heap
//! only remembers its description.
//!
//! Ported from VIctory-base (branch `xnw/base` @ 5d0516e = VIctory d2a4bae
//! + 26 PRs; none touches this file), `crates/victory-gpu/src/query.rs`
//! 1-72, GPL-3.0-or-later. Changed: the method table in our slot form
//! (`object::table`, with the base lines as roles); the debug line per heap
//! left out (no debug level in `xnw_core`).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use super::object::{self, Done, E_INVALIDARG, HRESULT, Iid, Object, QUERY_HEAP, Slot, Table, This, guid};
use super::resource::PAGEABLE_IIDS;

const HEAP_IIDS: [Iid; 4] = [
    guid(0x0d9658ae, 0xed45, 0x469e, [0xa6, 0x1d, 0x97, 0x0e, 0xc5, 0x83, 0xca, 0xb4]),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

/// Kept as the title gave it (nothing reads it yet).
#[allow(dead_code)]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Desc {
    kind: u32,
    count: u32,
    nodes: u32,
}

#[repr(C)]
struct Heap {
    object: Object,
    #[allow(dead_code)] // kept as the base does
    desc: Desc,
}

/// Device method.
pub(crate) extern "system" fn create(_: This, desc: *const Desc, iid: *const Iid, out: *mut This) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let heap = Box::new(Heap { object: Object::new(&FUNCTIONS), desc });
    object::hand_out(QUERY_HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(QUERY_HEAP, iid, object::is(iid, &HEAP_IIDS).then_some(this), out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}

/// IUnknown 3, object 4, device child 1.
pub(crate) const SLOTS: &[Slot] = &[
    (0, query as *const (), Done, "query.rs:68"),
    (2, release as *const (), Done, "query.rs:69"),
    (7, object::parent_device as *const (), Done, "query.rs:70"),
];
pub(crate) const LEN: usize = 24;
static FUNCTIONS: Table<LEN> = object::table::<QUERY_HEAP, LEN>(true, SLOTS);
