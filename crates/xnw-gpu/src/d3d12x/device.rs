//! The device titles get from `D3D12XboxCreateDevice`, laid out as their
//! inlined code expects: the object header, then device fields. Its table
//! starts like the PC device interface (IUnknown, the object methods, then
//! the device methods in PC order); console additions follow. Creation
//! makes the host device; feature queries answer for a Series S.
//!
//! Ported from VIctory `crates/victory-gpu/src/device.rs` at d2a4bae
//! (GPL-3.0-or-later): layout, interface ids, feature queries, the table.
//! Changed: creation needs a host device; PC's residency, removal and
//! power-state methods answer (slots 34, 35, 37, 40).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use super::object::{self, DEVICE, Done, Dropped, E_INVALIDARG, E_NOINTERFACE, E_NOTIMPL, HRESULT, Iid, Object, S_FALSE, S_OK, Slot, Table, This, guid, guid_text};
use super::{console, descriptor, dxgi, fence, frame, list, pipeline, pipeline_x, queue, resource, root};

/// What D3D returns when none of the requested feature levels is
/// supported, and device creation without a host GPU.
pub(crate) const UNSUPPORTED: HRESULT = 0x887A_0004_u32 as i32;

/// The device interface id (the same as on PC). The title also imports it as data.
pub static IID: Iid = guid(0x189819f1, 0x1db6, 0x4b57, [0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7]);

/// `ID3D12Device1` to `ID3D12Device8`, and `ID3D12Device11`, which the
/// title creates its device as. Each level's table is the one before plus
/// its methods.
pub const LEVEL_IIDS: [Iid; 9] = [
    guid(0x77acce80, 0x638e, 0x4e65, [0x88, 0x95, 0xc1, 0xf2, 0x33, 0x86, 0x86, 0x3e]),
    guid(0x30baa41e, 0xb15b, 0x475c, [0xa0, 0xbb, 0x1a, 0xf5, 0xc5, 0xb6, 0x43, 0x28]),
    guid(0x81dadc15, 0x2bad, 0x4392, [0x93, 0xc5, 0x10, 0x13, 0x45, 0xc4, 0xaa, 0x98]),
    guid(0xe865df17, 0xa9ee, 0x46f9, [0xa4, 0x63, 0x30, 0x98, 0x31, 0x5a, 0xa2, 0xe5]),
    guid(0x8b4f173b, 0x2fea, 0x4b80, [0x8f, 0x58, 0x43, 0x07, 0x19, 0x1a, 0xb9, 0x5d]),
    guid(0xc70b221b, 0x40e4, 0x4a17, [0x89, 0xaf, 0x02, 0x5a, 0x07, 0x27, 0xa6, 0xdc]),
    guid(0x5c014b53, 0x68a1, 0x4b9b, [0x8b, 0xd1, 0xdd, 0x60, 0x46, 0xb9, 0x35, 0x8b]),
    guid(0x728f1fa2, 0xea11, 0x4a9f, [0x9b, 0x52, 0x9b, 0x42, 0xa4, 0x84, 0xd5, 0xc9]),
    guid(0x5405c344, 0xd457, 0x444e, [0xb4, 0xdd, 0x23, 0x66, 0xe4, 0x5a, 0xee, 0x39]),
];

pub(crate) fn is_device(iid: *const Iid) -> bool {
    object::is(iid, &[IID]) || object::is(iid, &LEVEL_IIDS)
}

/// The interface version these layouts were checked against (2.4, VIctory).
pub const CHECKED_VERSION: u32 = 0x0002_0004;

#[repr(C)]
pub struct Device {
    object: Object,
    /// GPU address of a null constant buffer; none yet.
    null_constant_buffer: u64,
    /// Byte 0 (device +0x18) non-zero: calls the title would otherwise
    /// inline come to the table. The rest is reserved.
    out_of_line: [u32; 16],
}
const _: () = assert!(size_of::<Device>() == 0x58);

/// One device per process, as on the console.
static DEVICE_OBJECT: Device = Device {
    object: Object::new(&FUNCTIONS),
    null_constant_buffer: 0,
    out_of_line: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
};

pub fn device() -> This {
    (&raw const DEVICE_OBJECT).cast()
}

/// Set when the title creates the device as `ID3D12Device11`: its lists
/// and queues then have the newer layouts (VIctory's rule: the creation
/// interface is the only sign it found).
static NEWER: AtomicBool = AtomicBool::new(false);

pub(crate) fn newer() -> bool {
    NEWER.load(Ordering::Relaxed)
}

/// `D3D12XboxCreateDevice(adapter, parameters, iid, out)`. The parameters
/// start with the interface version; the rest (ring and scratch sizes,
/// features to leave out) does not matter to us yet.
pub extern "system" fn create(_adapter: *const c_void, parameters: *const u32, iid: *const Iid, out: *mut This) -> HRESULT {
    if parameters.is_null() || iid.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null; the title passes its parameters and an id.
    let (version, iid) = unsafe { (parameters.read(), &*iid) };
    xnw_core::info!("D3D12XboxCreateDevice: version {version:#x}, as {}", guid_text(iid));
    if version != CHECKED_VERSION {
        xnw_core::warn!("d3d12x: device layouts were checked against version {CHECKED_VERSION:#x} only");
    }
    if !is_device(iid) {
        xnw_core::warn!("d3d12x: device created as unknown interface {}", guid_text(iid));
        return E_NOINTERFACE;
    }
    if super::host::host_device().is_none() {
        return UNSUPPORTED;
    }
    if out.is_null() {
        return S_FALSE;
    }
    if *iid == LEVEL_IIDS[8] {
        NEWER.store(true, Ordering::Relaxed);
    }
    object::answer(DEVICE, iid, Some(device()), out)
}

/// The device, or its DXGI side.
extern "system" fn query_interface(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = if is_device(iid) {
        Some(this)
    } else if object::is(iid, &dxgi::DEVICE_IIDS) {
        Some(dxgi::device())
    } else {
        None
    };
    object::answer(DEVICE, iid, found, out)
}

extern "system" fn node_count(_: This) -> u32 {
    1
}

/// Residency, removal and power state: always resident, never removed.
extern "system" fn always_ok(_: This) -> HRESULT {
    S_OK
}

/// Feature query 2: the highest of the requested feature levels we support.
#[repr(C)]
struct FeatureLevels {
    count: u32,
    requested: *const u32,
    highest: u32,
}

/// Feature queries not answered, each logged once.
static UNANSWERED: Mutex<Vec<u32>> = Mutex::new(Vec::new());

/// `CheckFeatureSupport`: the Series S's answers (an RDNA 2 GPU), never
/// the host's.
extern "system" fn check_feature_support(_: This, feature: u32, data: *mut c_void, size: u32) -> HRESULT {
    const OPTIONS: u32 = 0;
    const ARCHITECTURE: u32 = 1;
    const FEATURE_LEVELS: u32 = 2;
    const FORMAT_SUPPORT: u32 = 3;
    const MULTISAMPLE: u32 = 4;
    let expected = match feature {
        OPTIONS => size_of_val(&console::OPTIONS),
        ARCHITECTURE => 16,
        FEATURE_LEVELS => size_of::<FeatureLevels>(),
        FORMAT_SUPPORT => 12,
        MULTISAMPLE => 16,
        _ => {
            let mut seen = UNANSWERED.lock().unwrap_or_else(|e| e.into_inner());
            if !seen.contains(&feature) {
                seen.push(feature);
                xnw_core::warn!("d3d12x: device feature query {feature} ({size} bytes) not implemented");
            }
            return E_NOTIMPL;
        }
    };
    if data.is_null() || size as usize != expected {
        return E_INVALIDARG;
    }
    match feature {
        OPTIONS => {
            // SAFETY: checked size and non-null.
            unsafe { data.cast::<[u32; 16]>().write_unaligned(console::OPTIONS) };
            S_OK
        }
        ARCHITECTURE => {
            // SAFETY: checked size and non-null.
            let a = unsafe { &mut *data.cast::<[u32; 4]>() };
            if a[0] != 0 {
                return E_INVALIDARG; // one node only
            }
            // Not tile-based; unified, cache-coherent memory.
            a[1..].copy_from_slice(&[0, 1, 1]);
            S_OK
        }
        FORMAT_SUPPORT => {
            // SAFETY: checked size and non-null.
            let q = unsafe { &mut *data.cast::<[u32; 3]>() };
            // shortcut (VIctory's): every use of every format.
            q[1] = 0x7fff_ffff;
            q[2] = 0xffff;
            S_OK
        }
        MULTISAMPLE => {
            // SAFETY: checked size and non-null.
            let q = unsafe { &mut *data.cast::<[u32; 4]>() };
            // One quality level for the counts the hardware has.
            q[3] = u32::from(matches!(q[1], 1 | 2 | 4 | 8));
            S_OK
        }
        _ => feature_levels(data),
    }
}

fn feature_levels(data: *mut c_void) -> HRESULT {
    // SAFETY: the caller checked size and non-null.
    let query = unsafe { &mut *data.cast::<FeatureLevels>() };
    if query.requested.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title passes `count` levels.
    let requested = unsafe { std::slice::from_raw_parts(query.requested, query.count as usize) };
    match requested.iter().copied().filter(|&l| l <= console::FEATURE_LEVEL).max() {
        Some(level) => {
            query.highest = level;
            S_OK
        }
        None => UNSUPPORTED,
    }
}

/// IUnknown 3, object 4, device methods 76 (0x18-0x290), then the methods
/// `ID3D12Device1` to `ID3D12Device8` add (0x298-0x3a0, the console's own
/// among them), then those of `ID3D12Device9` to `ID3D12Device11`.
pub(crate) const SLOTS: &[Slot] = &[
    (0, query_interface as *const (), Done, "device.rs:355"),
    (7, node_count as *const (), Done, "device.rs:356"),
    (8, queue::create as *const (), Done, "device.rs:357"),
    (9, list::create_allocator as *const (), Done, "device.rs:358"),
    (10, pipeline::create_graphics as *const (), Dropped, "device.rs:360 graphics pipeline"),
    (11, pipeline::create_compute as *const (), Dropped, "device.rs:361 compute pipeline"),
    (12, list::create_list as *const (), Done, "device.rs:359"),
    (13, check_feature_support as *const (), Done, "device.rs:362"),
    (14, descriptor::create_heap as *const (), Done, "device.rs:363"),
    (15, descriptor::increment_size as *const (), Done, "device.rs:364"),
    (16, root::create as *const (), Done, "device.rs:365"),
    (17, descriptor::create_cbv as *const (), Done, "device.rs:366"),
    (18, descriptor::create_srv as *const (), Done, "device.rs:367"),
    (19, descriptor::create_uav as *const (), Done, "device.rs:368"),
    (20, descriptor::create_rtv as *const (), Done, "device.rs:369"),
    (21, descriptor::create_dsv as *const (), Done, "device.rs:370"),
    (22, descriptor::create_sampler as *const (), Done, "device.rs:371"),
    (23, descriptor::copy as *const (), Done, "device.rs:372"),
    (24, descriptor::copy_simple as *const (), Done, "device.rs:373"),
    (25, resource::allocation_info as *const (), Done, "device.rs:374"),
    (26, resource::custom_heap_properties as *const (), Done, "device.rs:375"),
    (27, resource::create_committed as *const (), Done, "device.rs:376"),
    (28, resource::create_heap as *const (), Done, "device.rs:377"),
    (29, resource::create_placed as *const (), Done, "device.rs:378"),
    (34, always_ok as *const (), Done, "ours: PC MakeResident"),
    (35, always_ok as *const (), Done, "ours: PC Evict"),
    (36, fence::create as *const (), Done, "device.rs:379"),
    (37, always_ok as *const (), Done, "ours: PC GetDeviceRemovedReason"),
    (38, resource::copyable_footprints as *const (), Done, "device.rs:380"),
    (39, super::query::create as *const (), Done, "device.rs:381"),
    (40, always_ok as *const (), Done, "ours: PC SetStablePowerState"),
    (41, super::signature::create as *const (), Done, "device.rs:382"),
    (48, queue::create_x as *const (), Done, "device.rs:383"),
    // The title asks both for ID3D12PipelineState (765a30f3-..., at
    // title+0x2477557 and title+0x24800bf) with `(desc, iid, out)`; VIctory
    // leaves both unwritten. Ours, from the call sites.
    // `(base pipeline, count, overrides, iid, out)` at title+0x247a5c2,
    // whose out is used whatever the result. 55/57/67: pipeline_x.rs.
    (55, pipeline_x::create_derived as *const (), Dropped, "ours: derived pipeline (title+0x247a5c8)"),
    (57, pipeline_x::create_cached_graphics as *const (), Dropped, "ours: pipeline state from a description (title+0x247757e)"),
    (61, descriptor::create_sampler_x as *const (), Done, "device.rs:384 console sampler"),
    (62, pipeline::create_compute_x as *const (), Dropped, "device.rs:385 compute pipeline X"),
    (67, pipeline_x::create_cached_compute as *const (), Dropped, "ours: pipeline state from a cached description (title+0x24800db)"),
    (69, object::ignore as *const (), Done, "device.rs:386 debug error filter"),
    (70, descriptor::create_placed_srv as *const (), Done, "device.rs:387 placed SRV"),
    (71, descriptor::create_placed_uav as *const (), Done, "device.rs:388 placed UAV"),
    (73, list::create_list_x as *const (), Done, "device.rs:389"),
    (74, resource::register_page_pool as *const (), Done, "device.rs:390"),
    (75, resource::unregister_page_pool as *const (), Done, "device.rs:391"),
    (76, resource::create_committed as *const (), Done, "device.rs:394 committed X"),
    (78, frame::set_interval as *const (), Done, "device.rs:395"),
    (79, frame::schedule as *const (), Done, "device.rs:396"),
    (80, frame::wait as *const (), Done, "device.rs:397"),
    (81, frame::statistics as *const (), Done, "device.rs:398"),
    (86, pipeline::create_stream as *const (), Dropped, "device.rs:399 pipeline stream"),
    (108, resource::allocation_info2 as *const (), Done, "device.rs:400"),
    (127, resource::create_committed3 as *const (), Done, "device.rs:401"),
    (133, resource::create_at_address as *const (), Done, "device.rs:402"),
];
pub(crate) const LEN: usize = 160;
static FUNCTIONS: Table<LEN> = object::table::<DEVICE, LEN>(true, SLOTS);
