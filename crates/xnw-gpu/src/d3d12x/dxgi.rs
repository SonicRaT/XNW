//! The DXGI side of the device: one adapter with one output, reached from
//! the device by interface query. Tables follow the PC interfaces' order
//! (IUnknown, the object methods, then each interface's methods). The
//! adapter describes the console (`console::current()`), not the host GPU.
//!
//! Ported from VIctory `crates/victory-gpu/src/dxgi.rs` at d2a4bae
//! (GPL-3.0-or-later). Changed: the model comes from `console.rs`.
//! `display_size` from victory-fork (fxceless/victory-fork)
//! `crates/victory-gpu/src/dxgi.rs` at 4e17478 (GPL-3.0-or-later). Changed:
//! `XNW_DISPLAY=1080p|1440p|4k`, default 1080p (theirs `WxH`, default 4K).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use super::object::{self, ADAPTER, DXGI_DEVICE, Done, E_INVALIDARG, FACTORY, HRESULT, Iid, MORE_DATA, NOT_FOUND, OUTPUT, Object, S_OK, Slot, Table, This, guid};

/// The DXGI device interfaces, base first (as on PC).
pub(crate) const DEVICE_IIDS: [Iid; 3] = [
    guid(0x54ec77fa, 0x1377, 0x44e6, [0x8c, 0x32, 0x88, 0xfd, 0x5f, 0x44, 0xc8, 0x4c]),
    guid(0x77db970f, 0x6276, 0x48ba, [0xba, 0x28, 0x07, 0x01, 0x43, 0xb4, 0x39, 0x2c]),
    guid(0x05008617, 0xfbfd, 0x4051, [0xa7, 0x90, 0x14, 0x48, 0x84, 0xb4, 0xf6, 0xa9]),
];
pub(crate) const ADAPTER_IIDS: [Iid; 3] = [
    guid(0x2411e7e1, 0x12ac, 0x4ccf, [0xbd, 0x14, 0x97, 0x98, 0xe8, 0x53, 0x4d, 0xc0]),
    guid(0x29038f61, 0x3839, 0x4626, [0x91, 0xfd, 0x08, 0x68, 0x79, 0x01, 0x1a, 0x05]),
    guid(0x0aa1ae0a, 0xfa0e, 0x4b84, [0x86, 0x44, 0xe0, 0x5f, 0xf8, 0xe5, 0xac, 0xb5]),
];
pub(crate) const FACTORY_IIDS: [Iid; 3] = [
    guid(0x7b7166ec, 0x21c7, 0x44ae, [0xb2, 0x1a, 0xc9, 0xae, 0x32, 0x1a, 0xe3, 0x69]),
    guid(0x770aae78, 0xf26f, 0x4dba, [0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87]),
    guid(0x50c83a1c, 0xe072, 0x4c48, [0x87, 0xb0, 0x36, 0x30, 0xfa, 0x36, 0xa6, 0xd0]),
];
const OUTPUT_IIDS: [Iid; 1] = [guid(0xae02eedb, 0xc735, 0x4690, [0x8d, 0x52, 0x5a, 0x8d, 0xc2, 0x02, 0x13, 0xaa])];

static DEVICE: Object = Object::new(&DEVICE_FUNCTIONS);
static ADAPTER_OBJECT: Object = Object::new(&ADAPTER_FUNCTIONS);
static FACTORY_OBJECT: Object = Object::new(&FACTORY_FUNCTIONS);
static OUTPUT_OBJECT: Object = Object::new(&OUTPUT_FUNCTIONS);

pub(crate) fn device() -> This {
    &DEVICE
}

fn query(kind: usize, iid: *const Iid, known: &[Iid], found: This, out: *mut This) -> HRESULT {
    object::answer(kind, iid, object::is(iid, known).then_some(found), out)
}

/// The DXGI device also answers for the D3D12 device.
extern "system" fn device_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    if super::device::is_device(iid) {
        return object::answer(DXGI_DEVICE, iid, Some(super::device::device()), out);
    }
    query(DXGI_DEVICE, iid, &DEVICE_IIDS, this, out)
}
extern "system" fn device_parent(_: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(DXGI_DEVICE, iid, &ADAPTER_IIDS, &ADAPTER_OBJECT, out)
}
extern "system" fn device_adapter(_: This, out: *mut This) -> HRESULT {
    object::answer(DXGI_DEVICE, &ADAPTER_IIDS[0], Some(&ADAPTER_OBJECT), out)
}

extern "system" fn adapter_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(ADAPTER, iid, &ADAPTER_IIDS, this, out)
}
extern "system" fn adapter_parent(_: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(ADAPTER, iid, &FACTORY_IIDS, &FACTORY_OBJECT, out)
}
extern "system" fn adapter_outputs(_: This, index: u32, out: *mut This) -> HRESULT {
    if index != 0 {
        return NOT_FOUND;
    }
    object::answer(ADAPTER, &OUTPUT_IIDS[0], Some(&OUTPUT_OBJECT), out)
}

#[repr(C)]
pub struct AdapterDescription {
    pub name: [u16; 128],
    pub vendor: u32,
    pub device: u32,
    pub subsystem: u32,
    pub revision: u32,
    pub dedicated_video_memory: usize,
    pub dedicated_system_memory: usize,
    pub shared_system_memory: usize,
    pub luid: [u32; 2],
}
const _: () = assert!(size_of::<AdapterDescription>() == 0x130);

extern "system" fn adapter_description(_: This, out: *mut AdapterDescription) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let mut name = [0; 128];
    for (slot, unit) in name.iter_mut().zip(super::console::current().name().encode_utf16()) {
        *slot = unit;
    }
    let (gpu_optimal, standard) = super::console::current().title_memory();
    let description = AdapterDescription {
        name,
        vendor: super::console::VENDOR,
        device: 0,
        subsystem: 0,
        revision: 0,
        dedicated_video_memory: gpu_optimal,
        dedicated_system_memory: 0,
        shared_system_memory: standard,
        luid: [1, 0],
    };
    // SAFETY: checked non-null; the title passes room for the description.
    unsafe { out.write(description) };
    S_OK
}

extern "system" fn factory_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(FACTORY, iid, &FACTORY_IIDS, this, out)
}
extern "system" fn factory_adapters(_: This, index: u32, out: *mut This) -> HRESULT {
    if index != 0 {
        return NOT_FOUND;
    }
    object::answer(FACTORY, &ADAPTER_IIDS[0], Some(&ADAPTER_OBJECT), out)
}

extern "system" fn output_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(OUTPUT, iid, &OUTPUT_IIDS, this, out)
}
extern "system" fn output_parent(_: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(OUTPUT, iid, &ADAPTER_IIDS, &ADAPTER_OBJECT, out)
}

/// A display mode: size, refresh (numerator, denominator), format,
/// scanline order, scaling, flags.
#[repr(C)]
struct Mode {
    width: u32,
    height: u32,
    refresh: [u32; 2],
    format: u32,
    scanlines: u32,
    scaling: u32,
    flags: u32,
}
const _: () = assert!(size_of::<Mode>() == 32);

/// The display size reported to the title: `XNW_DISPLAY=1080p` (default),
/// `1440p` or `4k`, as a console reports the television it is plugged
/// into. Ported from victory-fork (fxceless/victory-fork,
/// crates/victory-gpu/src/dxgi.rs `display_size`, 4e17478, GPL-3.0-or-later;
/// there `VICTORY_DISPLAY_MODE=WxH`, default 3840x2160): every fast report
/// upstream runs 1080p; at 4K the title's output-size passes (upscale, TAA
/// output, post, UI planes) and our swap chain ran at 3840x2160 on 8 GB.
pub(crate) fn display_size() -> (u32, u32) {
    static SIZE: std::sync::OnceLock<(u32, u32)> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        let asked = std::env::var("XNW_DISPLAY").unwrap_or_default().trim().to_ascii_lowercase();
        let size = match asked.as_str() {
            "4k" | "2160p" | "3840x2160" => (3840, 2160),
            "1440p" | "2560x1440" => (2560, 1440),
            "" | "1080p" | "1920x1080" => (1920, 1080),
            other => {
                xnw_core::warn!("d3d12x: XNW_DISPLAY={other} not known (1080p, 1440p, 4k): 1080p");
                (1920, 1080)
            }
        };
        xnw_core::info!("d3d12x: display mode reported to the title {}x{} at 60 Hz (XNW_DISPLAY)", size.0, size.1);
        size
    })
}

/// `GetDisplayModeList`: one mode at 60 Hz, [`display_size`].
/// shortcut: not the host display's (VIctory: the title looks for a 120 Hz
/// mode to allow 120 Hz frame intervals).
extern "system" fn modes(_: This, _format: u32, _flags: u32, count: *mut u32, out: *mut Mode) -> HRESULT {
    // SAFETY: the title passes its count (null is tolerated).
    let Some(count) = (unsafe { count.as_mut() }) else {
        return E_INVALIDARG;
    };
    if out.is_null() {
        *count = 1;
        return S_OK;
    }
    if *count < 1 {
        return MORE_DATA;
    }
    *count = 1;
    let (width, height) = display_size();
    let mode = Mode { width, height, refresh: [60, 1], format: 0, scanlines: 1, scaling: 0, flags: 0 };
    // SAFETY: checked non-null; room for `count` >= 1 modes.
    unsafe { out.write(mode) };
    S_OK
}

/// IUnknown 3, object 4, device 5 + 2 + 3.
pub(crate) const DEVICE_SLOTS: &[Slot] = &[
    (0, device_query as *const (), Done, "dxgi.rs:183"),
    (6, device_parent as *const (), Done, "dxgi.rs:184"),
    (7, device_adapter as *const (), Done, "dxgi.rs:185"),
];
/// IUnknown 3, object 4, adapter 3 + 1 + 1.
pub(crate) const ADAPTER_SLOTS: &[Slot] = &[
    (0, adapter_query as *const (), Done, "dxgi.rs:192"),
    (6, adapter_parent as *const (), Done, "dxgi.rs:193"),
    (7, adapter_outputs as *const (), Done, "dxgi.rs:194"),
    (8, adapter_description as *const (), Done, "dxgi.rs:195"),
];
/// IUnknown 3, object 4, factory 5 + 2 + 11.
pub(crate) const FACTORY_SLOTS: &[Slot] = &[
    (0, factory_query as *const (), Done, "dxgi.rs:202"),
    (7, factory_adapters as *const (), Done, "dxgi.rs:203"),
    (12, factory_adapters as *const (), Done, "dxgi.rs:204"),
];
/// IUnknown 3, object 4, output 15 (12 as on PC, then 3 console ones).
pub(crate) const OUTPUT_SLOTS: &[Slot] = &[
    (0, output_query as *const (), Done, "dxgi.rs:262"),
    (6, output_parent as *const (), Done, "dxgi.rs:263"),
    (21, modes as *const (), Done, "dxgi.rs:264"),
];
pub(crate) const DEVICE_LEN: usize = 32;
pub(crate) const ADAPTER_LEN: usize = 24;
pub(crate) const FACTORY_LEN: usize = 32;
pub(crate) const OUTPUT_LEN: usize = 32;
static DEVICE_FUNCTIONS: Table<DEVICE_LEN> = object::table::<DXGI_DEVICE, DEVICE_LEN>(false, DEVICE_SLOTS);
static ADAPTER_FUNCTIONS: Table<ADAPTER_LEN> = object::table::<ADAPTER, ADAPTER_LEN>(false, ADAPTER_SLOTS);
static FACTORY_FUNCTIONS: Table<FACTORY_LEN> = object::table::<FACTORY, FACTORY_LEN>(false, FACTORY_SLOTS);
static OUTPUT_FUNCTIONS: Table<OUTPUT_LEN> = object::table::<OUTPUT, OUTPUT_LEN>(false, OUTPUT_SLOTS);
