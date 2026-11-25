//! `d3d12_xs.dll`: virtual memory setup, device creation and root
//! signature (de)serialisation. The device and everything it creates live
//! in `xnw-gpu` (`d3d12x`).
//!
//! The exports and the allocator hand-over follow VIctory
//! `crates/victory-hle/src/modules/d3d12x.rs` at d2a4bae (GPL-3.0-or-later).

use std::ffi::c_void;

use xnw_gpu::d3d12x;

/// What `D3D12ConfigureVirtualMemoryX` takes and fills in (layout from
/// VIctory, `crates/victory-gpu/src/device.rs` at d2a4bae).
#[repr(C)]
pub struct VirtualMemory {
    flags: u32,
    /// GPU page table size in 4 MiB pages.
    page_table_pages: u32,
    /// Memory kept across title processes for the frame buffer.
    preserved_bytes: u32,
    /// Out: the range graphics addresses fall in.
    lowest: u64,
    highest: u64,
}
const _: () = assert!(size_of::<VirtualMemory>() == 0x20);

const S_OK: i32 = 0;
const E_INVALIDARG: i32 = 0x8007_0057u32 as i32;

/// Called before any graphics allocation. The inputs size console structures
/// we do not have; the range out is all of user space, because here GPU
/// addresses are CPU addresses.
pub extern "system" fn D3D12ConfigureVirtualMemoryX(config: *mut VirtualMemory) -> i32 {
    // SAFETY: the title passes its configuration; null is tolerated.
    let Some(config) = (unsafe { config.as_mut() }) else { return E_INVALIDARG };
    xnw_core::info!(
        "D3D12ConfigureVirtualMemoryX: flags {:#x}, page table {} x 4 MiB, preserved {:#x} bytes",
        config.flags,
        config.page_table_pages,
        config.preserved_bytes
    );
    // User space on x64 Windows.
    config.lowest = 0x1_0000;
    config.highest = 0x7FFF_FFFE_FFFF;
    S_OK
}

/// `D3D12XboxCreateDevice(adapter, parameters, iid, out)`. First tells the
/// device where the title is (for naming callers) and hands it the title's
/// general allocator (`XMemAlloc`, which honours the title's hooks) for
/// resource memory, as on the console.
pub extern "system" fn D3D12XboxCreateDevice(adapter: *const c_void, parameters: *const u32, iid: *const d3d12x::Iid, out: *mut d3d12x::This) -> i32 {
    d3d12x::set_title_base(crate::title_base());
    d3d12x::use_allocator(crate::xmem::XMemAlloc, crate::xmem::XMemFree);
    d3d12x::use_texture_sizer(crate::xg::texture_size);
    d3d12x::use_page_mapper(crate::xmem::map_alias);
    xnw_gpu::shared::use_memory(crate::xmem::backing, crate::xmem::physical_window);
    // Access traps for the texture pages, and the fill of a page the
    // bypass left out, run on the faulting thread before its access goes on.
    static TRAPS: std::sync::Once = std::sync::Once::new();
    TRAPS.call_once(|| {
        d3d12x::use_read_traps(d3d12x::ReadTraps { trap: d3d12x::trap_reads, stats: d3d12x::read_trap_stats, untrap: d3d12x::untrap });
        d3d12x::on_trap(0x1000..0x2000, fill_texture_page);
    });
    d3d12x::create_device(adapter, parameters, iid, out)
}

/// A trapped texture block is touched: bytes the bypass left out of it are
/// read from disk now (the on-demand fill).
fn fill_texture_page(_tag: u32, address: usize, len: usize) {
    xnw_formats::origin::registry().ensure_present(address as u64, len as u64);
}

/// `D3D12SerializeRootSignature(desc, version, blob, error)`.
pub extern "system" fn D3D12SerializeRootSignature(desc: *const d3d12x::layout::RootDesc, version: u32, blob: *mut *mut c_void, error: *mut *mut c_void) -> i32 {
    d3d12x::serialize(desc, version, blob, error)
}

/// `D3D12SerializeVersionedRootSignature(desc, blob, error)` (the title and its BVH builder).
pub extern "system" fn D3D12SerializeVersionedRootSignature(desc: *const d3d12x::layout::VersionedRootDesc, blob: *mut *mut c_void, error: *mut *mut c_void) -> i32 {
    d3d12x::serialize_versioned(desc, blob, error)
}

/// `D3D12CreateRootSignatureDeserializer(blob, size, iid, out)` (the upscaler's).
pub extern "system" fn D3D12CreateRootSignatureDeserializer(blob: *const u8, size: usize, iid: *const d3d12x::Iid, out: *mut *mut c_void) -> i32 {
    d3d12x::create_deserializer(blob, size, iid, out)
}
