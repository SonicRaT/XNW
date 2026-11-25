//! Root signatures: the `d3d12_xs` serialisation exports and the device's
//! root signature objects.
//!
//! Serialising goes to the host's `D3D12SerializeRootSignature` (or its
//! versioned form): the description is laid out as on PC, and the host's
//! blob works as the console's (IUnknown, then data and size at 0x18 and
//! 0x20). The console's own flags (bit 24 and up) are dropped first.
//!
//! Ported from VIctory `crates/victory-gpu/src/root.rs` at d2a4bae
//! (GPL-3.0-or-later). Changed: the host functions come from the
//! `windows` crate's bindings instead of a looked-up export.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;

use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::{
    D3D_ROOT_SIGNATURE_VERSION, D3D12_ROOT_SIGNATURE_DESC, D3D12_VERSIONED_ROOT_SIGNATURE_DESC, D3D12CreateRootSignatureDeserializer,
    D3D12SerializeRootSignature, D3D12SerializeVersionedRootSignature,
};
use windows::core::GUID;

use super::object::{self, Done, E_INVALIDARG, HRESULT, Iid, Object, ROOT_SIGNATURE, S_OK, Slot, Table, This, guid};
use super::resource::PAGEABLE_IIDS;

const ROOT_SIGNATURE_IIDS: [Iid; 3] = [
    guid(0xc54a6b66, 0x72df, 0x4ee8, [0x8b, 0xe5, 0xa9, 0x46, 0xa1, 0x42, 0x92, 0x14]),
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

const CONSOLE_FLAGS: u32 = 0xff00_0000;

/// PC's root signature description; every version's starts with these fields.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Desc {
    pub parameter_count: u32,
    pub parameters: *const c_void,
    pub sampler_count: u32,
    pub samplers: *const c_void,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VersionedDesc {
    pub version: u32,
    pub desc: Desc,
}

const _: () = assert!(size_of::<Desc>() == size_of::<D3D12_ROOT_SIGNATURE_DESC>());

fn without_console_flags(desc: Desc) -> Desc {
    Desc { flags: desc.flags & !CONSOLE_FLAGS, ..desc }
}

fn hr(r: windows::core::Result<()>) -> HRESULT {
    r.map_or_else(|e| e.code().0, |()| S_OK)
}

/// Logs a serialisation: every failure, and the first success per
/// (version, flags) pair, so a host that refuses some flags (Windows 10's
/// in-box runtime and shader-model-6.6 heap indexing, 0xc00) shows itself.
fn log_serialized(what: &str, version: u32, flags: u32, result: HRESULT) {
    static SEEN: std::sync::Mutex<Vec<(u32, u32)>> = std::sync::Mutex::new(Vec::new());
    if result < 0 {
        xnw_core::warn!("d3d12x: {what} version {version} flags {flags:#x} failed {:#010x}", result as u32);
        return;
    }
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&(version, flags)) {
        seen.push((version, flags));
        xnw_core::info!("d3d12x: {what} version {version} flags {flags:#x}: ok (first of these)");
    }
}

/// `d3d12_xs!D3D12SerializeRootSignature`.
pub extern "system" fn serialize(desc: *const Desc, version: u32, blob: *mut *mut c_void, error: *mut *mut c_void) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let desc = without_console_flags(desc);
    let error = (!error.is_null()).then_some(error.cast::<Option<ID3DBlob>>());
    // SAFETY: the same layout as the host's; the out pointers are the
    // title's (an interface pointer slot is an `Option<ID3DBlob>`).
    let result = hr(unsafe { D3D12SerializeRootSignature((&raw const desc).cast(), D3D_ROOT_SIGNATURE_VERSION(version as i32), blob.cast(), error) });
    log_serialized("serialize root signature", version, desc.flags, result);
    result
}

/// `d3d12_xs!D3D12SerializeVersionedRootSignature` (the title and its BVH builder).
pub extern "system" fn serialize_versioned(desc: *const VersionedDesc, blob: *mut *mut c_void, error: *mut *mut c_void) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    // The host reads the whole union: copy the title's into one.
    // SAFETY: plain data.
    let mut full: D3D12_VERSIONED_ROOT_SIGNATURE_DESC = unsafe { std::mem::zeroed() };
    let ours = VersionedDesc { desc: without_console_flags(desc.desc), ..desc };
    // SAFETY: the host's starts with the same fields and is larger.
    unsafe { (&raw mut full).cast::<VersionedDesc>().write(ours) };
    let error = (!error.is_null()).then_some(error.cast::<Option<ID3DBlob>>());
    // SAFETY: as in `serialize`.
    let result = hr(unsafe { D3D12SerializeVersionedRootSignature(&full, blob.cast(), error) });
    log_serialized("serialize versioned root signature", ours.version, ours.desc.flags, result);
    result
}

/// `d3d12_xs!D3D12CreateRootSignatureDeserializer` (the upscaler's): our
/// blobs are the host's serialisation, so the host's deserializer reads
/// them, and its object is laid out as the console's.
pub extern "system" fn create_deserializer(blob: *const u8, size: usize, iid: *const Iid, out: *mut *mut c_void) -> HRESULT {
    if blob.is_null() || out.is_null() || iid.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title passes `size` bytes, an id and an out slot.
    hr(unsafe { D3D12CreateRootSignatureDeserializer(blob.cast(), size, iid.cast::<GUID>(), out) })
}

#[repr(C)]
struct RootSignature {
    object: Object,
    /// The serialised form (the host's: a DXBC container with `RTS0`).
    blob: Vec<u8>,
}

/// The serialised root signature of a title root signature object.
pub(crate) fn blob<'a>(this: This) -> &'a [u8] {
    if this.is_null() {
        return &[];
    }
    // SAFETY: the title passes one of our root signatures.
    &unsafe { &*this.cast::<RootSignature>() }.blob
}

/// A root signature's shape for logs: `RTS0` version, parameters (T table
/// with its range count, C constants with their count, CBV/SRV/UAV root
/// descriptors), static samplers, flags.
pub(crate) fn shape(this: This) -> String {
    let b = blob(this);
    let u = |at: usize| b.get(at..at + 4).map_or(0, |x| u32::from_le_bytes([x[0], x[1], x[2], x[3]])) as usize;
    if b.len() < 32 || &b[..4] != b"DXBC" {
        return format!("(none: {} bytes)", b.len());
    }
    let Some(part) = (0..u(28)).map(|i| u(32 + i * 4)).find(|&o| b.get(o..o + 4) == Some(b"RTS0")) else {
        return "(no RTS0)".into();
    };
    let r = part + 8;
    let (version, count, params, samplers, flags) = (u(r), u(r + 4), u(r + 8), u(r + 12), u(r + 20));
    let mut out = format!("v{version} {count} params [");
    for i in 0..count.min(64) {
        let p = r + params + i * 12;
        let at = r + u(p + 8);
        let item = match u(p) {
            0 => format!("T{}", u(at)),
            1 => format!("C{}", u(at + 8)),
            2 => "CBV".into(),
            3 => "SRV".into(),
            4 => "UAV".into(),
            k => format!("?{k}"),
        };
        out += if i == 0 { "" } else { " " };
        out += &item;
    }
    out + &format!("] {samplers} static samplers flags {flags:#x}")
}

/// Device method 0x80.
pub(crate) extern "system" fn create(_: This, _nodes: u32, blob: *const u8, size: usize, iid: *const Iid, out: *mut This) -> HRESULT {
    if blob.is_null() || size == 0 {
        return E_INVALIDARG;
    }
    // SAFETY: the title passes `size` bytes.
    let blob = unsafe { std::slice::from_raw_parts(blob, size) }.to_vec();
    let root = Box::new(RootSignature { object: Object::new(&FUNCTIONS), blob });
    object::hand_out(ROOT_SIGNATURE, root, iid, &ROOT_SIGNATURE_IIDS, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(ROOT_SIGNATURE, iid, object::is(iid, &ROOT_SIGNATURE_IIDS).then_some(this), out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<RootSignature>(this)
}

/// IUnknown 3, object 4, device child 1.
pub(crate) const SLOTS: &[Slot] = &[
    (0, query as *const (), Done, "root.rs:205"),
    (2, release as *const (), Done, "root.rs:206"),
    (7, object::parent_device as *const (), Done, "root.rs:207"),
];
pub(crate) const LEN: usize = 24;
static FUNCTIONS: Table<LEN> = object::table::<ROOT_SIGNATURE, LEN>(true, SLOTS);
