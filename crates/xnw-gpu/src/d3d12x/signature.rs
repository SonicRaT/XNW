//! Command signatures: the record layout of indirect execution, read back
//! by `ExecuteIndirect` (`list.rs`) through [`layout`].
//!
//! Ported from VIctory-base (branch `xnw/base` @ 5d0516e = VIctory d2a4bae
//! + 26 PRs; none touches this file), `crates/victory-gpu/src/signature.rs`
//! 1-94, GPL-3.0-or-later. Changed: the method table in our slot form
//! (`object::table`, with the base lines as roles); the debug line per
//! signature left out (no debug level in `xnw_core`).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use super::object::{self, COMMAND_SIGNATURE, Done, E_INVALIDARG, HRESULT, Iid, Object, Slot, Table, This, guid};
use super::resource::PAGEABLE_IIDS;

const SIGNATURE_IIDS: [Iid; 4] = [
    guid(0xc36a797c, 0xec80, 0x4f0a, [0x89, 0x85, 0xa7, 0xb2, 0x47, 0x50, 0x82, 0xd1]),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
pub(crate) struct Desc {
    stride: u32,
    count: u32,
    /// Each a type and up to three 32-bit fields.
    arguments: *const [u32; 4],
    #[allow(dead_code)] // single node
    nodes: u32,
}

#[repr(C)]
struct Signature {
    object: Object,
    stride: u32,
    arguments: Vec<[u32; 4]>,
}

/// Device method.
pub(crate) extern "system" fn create(_: This, desc: *const Desc, _root: This, iid: *const Iid, out: *mut This) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if desc.arguments.is_null() && desc.count > 0 {
        return E_INVALIDARG;
    }
    let arguments = if desc.count == 0 {
        Vec::new()
    } else {
        // SAFETY: the title passes `count` argument descriptions.
        unsafe { std::slice::from_raw_parts(desc.arguments, desc.count as usize) }.to_vec()
    };
    let signature = Box::new(Signature { object: Object::new(&FUNCTIONS), stride: desc.stride, arguments });
    object::hand_out(COMMAND_SIGNATURE, signature, iid, &SIGNATURE_IIDS, out)
}

/// A signature's record stride and arguments (type, then up to three
/// fields each).
pub(crate) fn layout<'a>(this: This) -> Option<(u32, &'a [[u32; 4]])> {
    if this.is_null() {
        return None;
    }
    // SAFETY: the title passes one of our signatures.
    let s = unsafe { &*this.cast::<Signature>() };
    Some((s.stride, &s.arguments))
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &SIGNATURE_IIDS).then_some(this);
    object::answer(COMMAND_SIGNATURE, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Signature>(this)
}

/// IUnknown 3, object 4, device child 1.
pub(crate) const SLOTS: &[Slot] = &[
    (0, query as *const (), Done, "signature.rs:90"),
    (2, release as *const (), Done, "signature.rs:91"),
    (7, object::parent_device as *const (), Done, "signature.rs:92"),
];
pub(crate) const LEN: usize = 24;
static FUNCTIONS: Table<LEN> = object::table::<COMMAND_SIGNATURE, LEN>(true, SLOTS);

#[cfg(test)]
mod tests {
    use super::*;

    /// A signature of one indexed-draw argument hands its layout back
    /// (ours: the base has no test for this file).
    #[test]
    fn layout_reads_back() {
        let arguments = [[1u32, 0, 0, 0]];
        let desc = Desc { stride: 20, count: 1, arguments: arguments.as_ptr(), nodes: 0 };
        let mut out = std::ptr::null();
        assert_eq!(create(std::ptr::null(), &desc, std::ptr::null(), &SIGNATURE_IIDS[0], &mut out), object::S_OK);
        assert_eq!(layout(out), Some((20, &arguments[..])));
        release(out);
    }
}
