//! D3D12.X on host D3D12: the device the title creates, and everything it
//! needs up to its first `PresentX`.
//!
//! The title's inlined code reads objects by offset and calls their tables
//! by slot, so every object here is laid out as the console's and every
//! table has every slot: a slot we have not written logs `<object> slot <n>
//! (offset 0x..)` with its caller once and fails. Lists keep their command
//! limits at zero, so every command arrives as a call and records onto a
//! host list (`host.rs`, `list.rs`); clears, copies, resolves, barriers and
//! fences run on the host; draws and dispatches bind and record in
//! `draw.rs` (dropped and counted there when what they need has no host
//! object yet), pipelines in `pipeline.rs`, shader views in `descriptor.rs`.
//! GPU addresses are CPU addresses (console memory is unified).
//!
//! Layouts and slots come from VIctory (GPL-3.0-or-later, commit d2a4bae,
//! `crates/victory-gpu/src/`), which targets this build; each file names
//! what it took. [`inventory`] lists every slot's role.

pub mod console;
mod address_map;
mod capture;
mod crumbs;
mod guard;
mod loopcount;
pub mod counters;
pub mod descriptor;
mod device;
mod draw;
mod dxgi;
mod fence;
mod frameskip;
mod frame;
pub mod host;
mod indirect;
mod late;
mod list;
pub mod object;
pub mod pack;
mod page_table;
mod pipeline;
pub mod pipeline_x;
mod present;
mod primitive;
mod query;
mod queue;
mod refuse;
pub mod resource;
mod root;
pub mod sampler;
mod shader_cache;
mod shadow;
mod signature;
pub mod tap;
pub mod texture;
mod uikeep;
mod watch;
mod roles;

pub use device::{CHECKED_VERSION, IID, LEVEL_IIDS, create as create_device, device};
pub use object::{HRESULT, Iid, Object, S_OK, This, set_title_base, slot};
pub use host::{lock_owners, on_failure};
pub use queue::frames_presented;
pub use frameskip::render_all as frameskip_render_all;
pub use pipeline::note_phase as pipeline_phase;
pub use pipeline::warm::sweep as pipeline_sweep;
pub use pipeline::warm::{Outcome as PrebuildOutcome, Settings as PrebuildSettings, at_launch as prebuild_at_launch, summary as prebuild_summary, warm as prebuild};

/// The pipelines an earlier run found too slow to compile in-run (key, programs).
pub fn slow_pipelines() -> Vec<(String, String)> {
    pipeline::slow_pipelines()
}
pub(crate) use host::host_device;

/// Our world pass's picture for the scene plane (`XNW_WORLD_RENDER=screen`): xnw-world's `install_screen` sets it
/// at start, PresentX (queue.rs) reads it. The world pass is its own crate above this one (BLD2).
static SCREEN_SOURCE: std::sync::OnceLock<fn() -> Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource>> = std::sync::OnceLock::new();

/// Sets the source of our picture for the presenter (first call wins).
pub fn set_screen_source(f: fn() -> Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource>) {
    let _ = SCREEN_SOURCE.set(f);
}

pub(crate) fn screen_source() -> Option<fn() -> Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource>> {
    SCREEN_SOURCE.get().copied()
}

/// PASS1: called once a present is done with the picture the screen source
/// gave (its compose is sent; `host::wait_shown` waits for its GPU work).
static SCREEN_SHOWN: std::sync::OnceLock<fn()> = std::sync::OnceLock::new();

/// Sets what a present calls when it is done with our picture (first call wins).
pub fn set_screen_shown(f: fn()) {
    let _ = SCREEN_SHOWN.set(f);
}

/// Held from the screen source's picture to the end of its present.
pub(crate) struct Shown;

impl Drop for Shown {
    fn drop(&mut self) {
        if let Some(f) = SCREEN_SHOWN.get() {
            f();
        }
    }
}
pub(crate) use shadow::gpu_written_copy;
pub(crate) use descriptor::host::bindless_viewed_changed;
pub(crate) use page_table::{remapped as page_table_remapped, touch as page_table_touch, window_gone as page_table_window_gone};

/// The title image's base (0 before it is mapped).
pub(crate) fn title_image_base() -> usize {
    object::title_base()
}

/// Reads `len` bytes of title memory at `at` into `out`, zeros
/// where unreadable.
pub(crate) fn read_title(at: usize, out: *mut u8, len: usize) {
    host::copy_mapped(out, at as *const u8, len, false);
}

/// Whether `address` is inside a buffer object the title made.
pub(crate) fn has_host_copy_object(address: u64) -> bool {
    shadow::in_buffer(address, 1)
}

/// Whether a title buffer's host copy covers `[address, address + len)`.
pub(crate) fn has_host_copy(address: u64, len: u64) -> bool {
    shadow::find(address, len).is_some()
}
pub(crate) use descriptor::host::shared_window_gone;

/// The title locked `[address, address + len)` of a buffer for the CPU
/// (`XNW_ROLES=lock`, the loader's detour of its lock function), called
/// from `site`: the GPU's results there go back to title memory first.
/// Returns the bytes written.
pub fn buffer_locked(address: usize, len: usize, site: usize) -> usize {
    shadow::buffer_locked(address, len, site)
}

/// Whether `XNW_ROLES=lock` (the loader then detours the title's lock).
/// The title memory (address, bytes) the buffer view recorded at
/// descriptor slot `handle` reads: the record is read with a guarded copy and
/// its generation checked; None for anything but a live buffer view.
pub fn buffer_view_range(handle: u64) -> Option<(u64, u64)> {
    if !(0x1_0000..0x8000_0000_0000).contains(&handle) {
        return None;
    }
    let mut rec = [0u8; 32];
    host::copy_mapped(rec.as_mut_ptr(), handle as *const u8, rec.len(), false);
    // SAFETY: our own 32-byte copy of the record.
    let (i, generation) = unsafe { descriptor::record_id(rec.as_ptr() as u64) }?;
    let (view, current, _) = descriptor::entry(i)?;
    (current == generation).then_some(())?;
    descriptor::host::buffer_range(&view)
}

pub fn roles_lock() -> bool {
    roles::mode() == roles::Mode::Lock
}
pub use resource::{AllocRoutine, FreeRoutine, PageMapper, TextureSizer, use_allocator, use_page_mapper, use_texture_sizer};
pub use watch::{on_trap, read_trap_stats, reading as kernel_will_read, set_title_image, title_memory_remapped, trap_reads, untrap, use_pair, writing as kernel_will_write};
pub use texture::{ReadTraps, preload_textures, use_read_traps};
pub use root::{create_deserializer, serialize, serialize_versioned};

/// Public layouts the tests (playing the title) build arguments with.
pub mod layout {
    pub use super::descriptor::HeapDesc as DescriptorHeapDesc;
    pub use super::dxgi::AdapterDescription;
    pub use super::list::ListDesc;
    pub use super::queue::{QueueDesc, QueueDescX};
    pub use super::resource::{BUFFER, HeapProperties, ResourceDesc};
    pub use super::root::{Desc as RootDesc, VersionedDesc as VersionedRootDesc};
    pub const DXGI_DEVICE_IIDS: [super::Iid; 3] = super::dxgi::DEVICE_IIDS;
    pub const QUEUE_IID: super::Iid = super::queue::QUEUE_IIDS[0];
    pub const ALLOCATOR_IID: super::Iid = super::list::ALLOCATOR_IIDS[0];
    pub const LIST6_IID: super::Iid = super::list::LIST_IIDS[7];
    pub const FENCE_IID: super::Iid = super::fence::FENCE_IIDS[0];
    pub const RESOURCE_IID: super::Iid = super::resource::RESOURCE_IIDS[0];
}

/// Whether null mode (`XNW_GPU=null...`) has reached its point: from then on
/// no title draw reaches the host GPU (`frameskip.rs`).
pub fn null_now() -> bool {
    frameskip::null_now()
}

/// Frames presented so far.
pub fn presented() -> u64 {
    queue::presented()
}

/// The presented frame the world started at, once detected (only when
/// something wants it: `XNW_GPU=null@world`, world captures).
pub fn world_start() -> Option<u64> {
    capture::world_start()
}

/// Writes the memory line now (process commit, video memory, host and
/// placed textures), naming `why` (process exit, an exception).
pub fn log_memory(why: &str) {
    resource::log_memory(why);
    // The crash report's hook: the pipeline library goes to disk too.
    pipeline::flush_library(why);
}

/// One table's slots: object, layout, length, whether it has the object
/// methods, its written slots.
type Entry = (&'static str, usize, bool, &'static [object::Slot]);

fn tables() -> [Entry; 19] {
    [
        ("device", device::LEN, true, device::SLOTS),
        ("queue (older layout)", queue::LEN, true, &queue::SLOTS),
        ("queue (newer, the title's)", queue::LEN, true, &queue::NEWER_SLOTS),
        ("allocator", list::ALLOCATOR_LEN, true, list::ALLOCATOR_SLOTS),
        ("list (older layout)", list::LEN, true, &list::SLOTS),
        ("list (newer, the title's)", list::LEN, true, &list::NEWER_SLOTS),
        ("fence", fence::LEN, true, fence::SLOTS),
        ("descriptor heap", descriptor::LEN, true, descriptor::SLOTS),
        ("heap", resource::HEAP_LEN, true, resource::HEAP_SLOTS),
        ("resource", resource::RESOURCE_LEN, true, resource::RESOURCE_SLOTS),
        ("root signature", root::LEN, true, root::SLOTS),
        ("pipeline", pipeline::LEN, true, pipeline::SLOTS),
        ("query heap", query::LEN, true, query::SLOTS),
        ("command signature", signature::LEN, true, signature::SLOTS),
        ("DXGI device", dxgi::DEVICE_LEN, false, dxgi::DEVICE_SLOTS),
        ("DXGI adapter", dxgi::ADAPTER_LEN, false, dxgi::ADAPTER_SLOTS),
        ("DXGI factory", dxgi::FACTORY_LEN, false, dxgi::FACTORY_SLOTS),
        ("DXGI output", dxgi::OUTPUT_LEN, false, dxgi::OUTPUT_SLOTS),
        ("(object methods: 1 AddRef, 2 Release, 3-4 private data, 6 SetName)", 0, false, &[]),
    ]
}

/// Every table's slot roles as Markdown: counts per object (implemented,
/// recorded-and-dropped, naming stub), then each written slot with the
/// VIctory source line it came from (`victory-gpu/src/<file>:<line>`).
pub fn inventory() -> String {
    let mut out = String::from("| object | slots | implemented | recorded-and-dropped | naming stub |\n|---|---|---|---|---|\n");
    let all = tables();
    for &(name, n, objects, slots) in &all {
        if n == 0 {
            continue;
        }
        let (done, dropped, stub) = object::census(n, objects, slots);
        out += &format!("| {name} | {n} | {done} | {dropped} | {stub} |\n");
    }
    out += "\n| object | slot | offset | role | VIctory source |\n|---|---|---|---|---|\n";
    for &(name, n, _, slots) in &all {
        if n == 0 {
            out += &format!("\n{name}\n");
            continue;
        }
        let mut sorted: Vec<_> = slots.iter().filter(|s| !s.3.is_empty()).collect();
        sorted.sort_by_key(|s| s.0);
        sorted.dedup_by_key(|s| s.0);
        for s in sorted {
            let role = if s.2 == object::Done { "implemented" } else { "dropped" };
            out += &format!("| {name} | {} | {:#x} | {role} | {} |\n", s.0, s.0 * 8, s.3);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_table_counts_add_up() {
        let text = super::inventory();
        assert!(text.contains("| device | 160 |"));
    }
}
