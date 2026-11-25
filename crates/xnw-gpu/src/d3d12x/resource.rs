//! Heaps, committed and placed resources, and resource sizes.
//!
//! `room` from victory-fork (fxceless/victory-fork)
//! `crates/victory-gpu/src/resource.rs` at 4e17478 (GPL-3.0-or-later).
//! Changed: gates sampled host copies by their bytes (theirs: host copies of
//! filled textures past a 2 GiB floor); their idle release not ported.
//!
//! Heaps and resources are host memory and a resource's GPU address is its
//! CPU address, as on the console; textures also get a host copy, and
//! buffers a host copy when a draw reads them (`shadow`).
//!
//! Ported from VIctory-base (branch `xnw/base` @
//! 5d0516e = VIctory d2a4bae + PRs #135, #413, #581)
//! `crates/victory-gpu/src/resource.rs` 1-1966, GPL-3.0-or-later, replacing
//! our d2a4bae subset. Changed: `victory_formats` -> `xnw_formats`;
//! `tracing` -> `xnw_core` (debug lines dropped); `cfg(windows)` and
//! diagnostics dropped; the host layer through [`super::host`] (our `host.rs`).
//! Kept from ours, each for a measured reason: struct-returning slots 25/26/108 pick their
//! result by address ([`return_slot`]: either argument order),
//! textures sized by the title's layout library when the runtime names it
//! ([`use_texture_sizer`]: the Series S back-buffer pool), the
//! target-shape log, `displayed` (PresentX), our slot tables and public
//! layouts. Added: the memory line every 256 frames ([`frame_presented`]).
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ffi::c_void;
use std::ptr::NonNull;

use xnw_formats::texture;

use super::host;
use super::object::{self, Done, E_INVALIDARG, E_OUTOFMEMORY, HEAP, HRESULT, Iid, Object, RESOURCE, S_OK, Slot, Table, This, guid};

/// The interfaces every device child also answers for (pageable, device
/// child, object).
pub(crate) const PAGEABLE_IIDS: [Iid; 3] = [
    guid(0x63ee58fb, 0x1268, 0x4835, [0x86, 0xda, 0xf0, 0x08, 0xce, 0x62, 0xf0, 0xd6]),
    guid(0x905db94b, 0xa00c, 0x4140, [0x9d, 0xf5, 0x2b, 0x64, 0xca, 0x9e, 0xa3, 0x57]),
    guid(0xc4fec28f, 0x7966, 0x4e95, [0x9f, 0x94, 0xf4, 0x31, 0xcb, 0x56, 0xc3, 0xb8]),
];
const HEAP_IIDS: [Iid; 4] = [
    guid(0x6b3b2502, 0x6e51, 0x45b3, [0x90, 0xee, 0x98, 0x84, 0x26, 0x5e, 0x8d, 0xf3]),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];
pub(crate) const RESOURCE_IIDS: [Iid; 4] = [
    guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

const KIB_64: u64 = 64 << 10;
const MIB_4: u64 = 4 << 20;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HeapProperties {
    pub kind: u32,
    pub cpu_pages: u32,
    pub pool: u32,
    pub creation_nodes: u32,
    pub visible_nodes: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HeapDesc {
    pub size: u64,
    pub properties: HeapProperties,
    pub alignment: u64,
    pub flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ResourceDesc {
    pub dimension: u32,
    pub alignment: u64,
    pub width: u64,
    pub height: u32,
    pub depth_or_array_size: u16,
    pub mip_levels: u16,
    pub format: u32,
    pub sample_count: u32,
    pub sample_quality: u32,
    pub layout: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AllocationInfo {
    size: u64,
    alignment: u64,
}

const _: () = assert!(size_of::<HeapProperties>() == 20);
const _: () = assert!(size_of::<HeapDesc>() == 48);
const _: () = assert!(size_of::<ResourceDesc>() == 56);

pub const BUFFER: u32 = 1;
const TEXTURE_3D: u32 = 4;

/// Bytes per element and the element's width in texels (4 for block
/// compression), by DXGI format number. `None` for 1-bit and video
/// formats. The console adds 116/117 (10-bit float color), 118-120 (16-bit
/// depth with 8-bit stencil, two planes: 3 bytes per texel), 189 (signed
/// 10-bit color), 190 (two 4-bit channels) and 191 (24/8 color).
pub(crate) fn element(format: u32) -> Option<(u64, u64)> {
    Some(match format {
        1..=4 => (16, 1),
        5..=8 => (12, 1),
        9..=22 => (8, 1),
        23..=47 | 67 | 87..=93 | 116 | 117 | 189 | 191 => (4, 1),
        118..=120 => (3, 1),
        48..=59 | 68 | 69 | 85 | 86 | 115 => (2, 1),
        60..=65 | 190 => (1, 1),
        70..=72 | 79..=81 => (8, 4),
        73..=78 | 82..=84 | 94..=99 => (16, 4),
        _ => return None,
    })
}

/// A texture's layout in console memory, as the console's layout library
/// gives it (`victory-hle`'s `xg` serves the same): the description's
/// layout field is `0x100 | swizzle mode`, 1 linear, 0 the library's pick.
pub(crate) fn texture_layout(desc: &ResourceDesc) -> Option<texture::Texture> {
    use texture::{Dimension, Swizzle, TextureDesc};
    const TEXTURE_1D: u32 = 2;
    const TEXTURE_2D: u32 = 3;
    const RENDER_TARGET: u32 = 1;
    const DEPTH: u32 = 2;
    let dimension = match desc.dimension {
        TEXTURE_1D => Dimension::Tex1d,
        TEXTURE_2D => Dimension::Tex2d,
        TEXTURE_3D => Dimension::Tex3d,
        _ => return None,
    };
    let swizzle = match desc.layout {
        0 if dimension == Dimension::Tex1d => Swizzle::LINEAR,
        0 if desc.flags & DEPTH != 0 => Swizzle::SW_64KB_Z_X,
        0 if desc.flags & RENDER_TARGET != 0 => Swizzle::SW_64KB_R_X,
        0 => Swizzle::SW_4KB_S,
        1 => Swizzle::LINEAR,
        l if l & !0xff == 0x100 => Swizzle(l & 0xff),
        _ => return None,
    };
    texture::Texture::new(&TextureDesc {
        dimension,
        width: u32::try_from(desc.width).ok()?,
        height: desc.height,
        depth_or_array: u32::from(desc.depth_or_array_size),
        mips: u32::from(desc.mip_levels),
        format: desc.format,
        samples: desc.sample_count.max(1),
        swizzle,
    })
    .ok()
}

/// Sizes a texture as the console's layout library does: `(bytes,
/// alignment)`, `None` if it cannot.
pub type TextureSizer = fn(&ResourceDesc) -> Option<(u64, u64)>;
static SIZER: std::sync::OnceLock<TextureSizer> = std::sync::OnceLock::new();

/// Textures take the size the console's layout library gives them (the
/// title's `XGCreateTextureComputer` answers, `xnw-hle` `xg.rs`), so pools
/// the title sizes from console sizes fit what we allocate: its
/// back-buffer pool (`title+0x247ad10`) traps when more than 2 MiB is left
/// over. Set before the device is created.
pub fn use_texture_sizer(sizer: TextureSizer) {
    let _ = SIZER.set(sizer);
}

/// Bytes and alignment a resource takes: buffers their width (64 KiB
/// aligned), textures their console layout (formats it does not know:
/// every mip of every array slice and sample, packed, 64 KiB aligned),
/// rounded up to the alignment.
/// For the frame pack's tap: a resource's bytes in console memory (its footprint; a
/// buffer's width), 0 when unknown.
pub(crate) fn footprint_bytes(desc: &ResourceDesc) -> u64 {
    if desc.dimension == BUFFER { desc.width } else { footprint(desc).map_or(0, |i| i.size) }
}

fn footprint(desc: &ResourceDesc) -> Option<AllocationInfo> {
    // Ours: the title's own layout library first (the Series S
    // back-buffer pool), then the same library here.
    let laid_out = SIZER.get().and_then(|f| f(desc)).or_else(|| texture_layout(desc).map(|t| (t.size, t.alignment)));
    if desc.dimension != BUFFER
        && let Some((size, layout_alignment)) = laid_out
    {
        // As the console: the layout's own alignment unless more is asked
        // (4 KiB for 4 KiB modes, where PC says 64 KiB). The title
        // keeps these sizes in its texture tables, and its streaming
        // asserts when they disagree with the streamed layout.
        let alignment = desc.alignment.max(layout_alignment);
        if !alignment.is_power_of_two() {
            return None;
        }
        return Some(AllocationInfo {
            size: size.checked_next_multiple_of(alignment)?,
            alignment,
        });
    }
    let bytes = if desc.dimension == BUFFER {
        desc.width
    } else {
        let (element, block) = element(desc.format)?;
        let (depth, slices) = if desc.dimension == TEXTURE_3D {
            (u64::from(desc.depth_or_array_size), 1)
        } else {
            (1, u64::from(desc.depth_or_array_size))
        };
        let height = u64::from(desc.height);
        let mips = mip_count(desc);
        let mut chain = 0u64;
        for mip in 0..mips.min(64) {
            let w = (desc.width >> mip).max(1).div_ceil(block);
            let h = (height >> mip).max(1).div_ceil(block);
            let d = (depth >> mip).max(1);
            chain = chain.checked_add(w.checked_mul(h)?.checked_mul(d)?.checked_mul(element)?)?;
        }
        chain
            .checked_mul(slices)?
            .checked_mul(u64::from(desc.sample_count.max(1)))?
    };
    if bytes == 0 {
        return None;
    }
    let minimum = if desc.sample_count > 1 { MIB_4 } else { KIB_64 };
    let alignment = desc.alignment.max(minimum);
    if !alignment.is_power_of_two() {
        return None;
    }
    Some(AllocationInfo {
        size: bytes.checked_next_multiple_of(KIB_64)?,
        alignment,
    })
}

/// A format as the host has it. The console's 16-bit depth with stencil
/// (and its typeless views) has no PC format: the 24-bit ones stand in.
pub fn pc_format(format: u32) -> u32 {
    match format {
        118 => 45,
        119 => 46,
        120 => 47,
        other => other,
    }
}

/// The typeless format of a format's family (itself if it has none).
pub fn family(format: u32) -> u32 {
    const FAMILIES: [(u32, u32); 22] = [
        (1, 4),
        (5, 8),
        (9, 14),
        (15, 18),
        (19, 22),
        (23, 25),
        (27, 32),
        (33, 38),
        (39, 43),
        (44, 47),
        (48, 52),
        (53, 59),
        (60, 65),
        (70, 72),
        (73, 75),
        (76, 78),
        (79, 81),
        (82, 84),
        (90, 91),
        (92, 93),
        (94, 96),
        (97, 99),
    ];
    match format {
        87 => 90,
        88 => 92,
        f => FAMILIES
            .iter()
            .find(|&&(first, last)| (first..=last).contains(&f))
            .map_or(f, |&(first, _)| first),
    }
}

/// sRGB formats and their linear twins.
pub(crate) const SRGB: [(u32, u32); 3] = [(29, 28), (91, 87), (93, 88)];

// Typeless views, from VIctory (GPL-3.0-or-later)
// (Xvania7, branch fix/issue-850-typeless-view, commits 5e3c9e2..2be93af,
// crates/victory-gpu/src/resource.rs) with the depth and BC6H rows of
// pushkarreddyy. The host refuses a view in a typeless format:
// it becomes a null view, its writes dropped and its reads zero.

/// Whether a format is the typeless root of a family of several.
pub fn is_typeless(format: u32) -> bool {
    format.checked_add(1).is_some_and(|next| family(format) == format && family(next) == format)
}

/// The first typed format of a typeless format's family (others unchanged).
pub fn first_concrete(format: u32) -> u32 {
    match format {
        90 => 87, // B8G8R8A8_TYPELESS -> B8G8R8A8_UNORM
        92 => 88, // B8G8R8X8_TYPELESS -> B8G8R8X8_UNORM
        39 => 41, // R32_TYPELESS -> R32_FLOAT (not D32_FLOAT)
        94 => 95, // BC6H_TYPELESS -> BC6H_UF16
        f if is_typeless(f) => f + 1,
        f => f,
    }
}

/// The readable (shader view) format of a typeless depth family: the depth
/// aspect, never the depth target format (VIctory).
pub(crate) fn depth_readable(format: u32) -> Option<u32> {
    match format {
        19 => Some(21), // R32G8X24_TYPELESS -> R32_FLOAT_X8X24_TYPELESS
        44 => Some(46), // R24G8_TYPELESS -> R24_UNORM_X8_TYPELESS
        39 => Some(41), // R32_TYPELESS -> R32_FLOAT
        53 => Some(56), // R16_TYPELESS -> R16_UNORM
        _ => None,
    }
}

/// The depth target format of a typeless depth family (the reverse of
/// [`DEPTH_FORMATS`]).
pub(crate) fn depth_target(format: u32) -> Option<u32> {
    DEPTH_FORMATS.iter().find(|&&(_, typeless)| typeless == format).map(|&(typed, _)| typed)
}

/// The format a view of a title texture must name when its host copy is
/// typeless (see [`DEPTH_FORMATS`], [`SRGB`]): the title's own; `None` if
/// the host copy is typed.
pub(crate) fn view_format(this: This) -> Option<u32> {
    let own = pc_format(resource(this).desc.format);
    let host = host_resource(this)?;
    // SAFETY: a live resource.
    let format = unsafe { host.GetDesc() }.Format.0 as u32;
    (format != own).then_some(own)
}

/// Depth formats with their typeless family: depth textures that shaders
/// read are made typeless on the host, which only allows views of
/// typeless depth resources.
pub(crate) const DEPTH_FORMATS: [(u32, u32); 4] = [(20, 19), (40, 39), (45, 44), (55, 53)];

/// Mip levels, the full chain for 0.
pub(crate) fn mip_count(desc: &ResourceDesc) -> u32 {
    let depth = if desc.dimension == TEXTURE_3D {
        u64::from(desc.depth_or_array_size)
    } else {
        1
    };
    match desc.mip_levels {
        0 => {
            64 - desc
                .width
                .max(u64::from(desc.height))
                .max(depth)
                .max(1)
                .leading_zeros()
        }
        n => u32::from(n),
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlacedFootprint {
    pub(crate) offset: u64,
    format: u32,
    width: u32,
    height: u32,
    depth: u32,
    row_pitch: u32,
}

impl PlacedFootprint {
    /// The bytes it spans in a buffer: every row pitched but the last.
    pub(crate) fn bytes(&self) -> Option<u64> {
        let (element, block) = element(self.format)?;
        let lines = u64::from(self.height).div_ceil(block) * u64::from(self.depth);
        let last = u64::from(self.width).div_ceil(block) * element;
        Some(u64::from(self.row_pitch) * lines.saturating_sub(1) + last)
    }
}
const _: () = assert!(size_of::<PlacedFootprint>() == 32);

const NO_FOOTPRINT: PlacedFootprint = PlacedFootprint {
    offset: u64::MAX,
    format: 0,
    width: 0,
    height: 0,
    depth: 0,
    row_pitch: 0,
};

/// One subresource's size in an upload buffer: footprint (offset 0), rows
/// and bytes per row. Planes of depth-stencil formats are not told apart.
fn subresource(desc: &ResourceDesc, index: u32) -> Option<(PlacedFootprint, u32, u64)> {
    const PITCH: u64 = 256;
    if desc.dimension == BUFFER {
        let footprint = PlacedFootprint {
            offset: 0,
            format: 0,
            width: u32::try_from(desc.width).ok()?,
            height: 1,
            depth: 1,
            row_pitch: u32::try_from(desc.width.next_multiple_of(PITCH)).ok()?,
        };
        return (index == 0).then_some((footprint, 1, desc.width));
    }
    let mips = mip_count(desc);
    let slices = if desc.dimension == TEXTURE_3D {
        1
    } else {
        u32::from(desc.depth_or_array_size).max(1)
    };
    let (mip, slice) = (index % mips, index / mips);
    if slice >= slices {
        return None;
    }
    let (element, block) = element(desc.format)?;
    let width = (desc.width >> mip).max(1).next_multiple_of(block);
    let height = (u64::from(desc.height) >> mip)
        .max(1)
        .next_multiple_of(block);
    let depth = if desc.dimension == TEXTURE_3D {
        (u32::from(desc.depth_or_array_size) >> mip).max(1)
    } else {
        1
    };
    let row_size = width / block * element;
    let footprint = PlacedFootprint {
        offset: 0,
        format: desc.format,
        width: u32::try_from(width).ok()?,
        height: u32::try_from(height).ok()?,
        depth,
        row_pitch: u32::try_from(row_size.next_multiple_of(PITCH)).ok()?,
    };
    Some((footprint, u32::try_from(height / block).ok()?, row_size))
}

/// Device method: where each subresource goes in an upload buffer, rows
/// 256-byte and subresources 512-byte aligned, as on PC. Unknown sizes give
/// the documented all-ones values.
pub(crate) extern "system" fn copyable_footprints(
    _: This,
    desc: *const ResourceDesc,
    first: u32,
    count: u32,
    base: u64,
    layouts: *mut PlacedFootprint,
    rows: *mut u32,
    row_sizes: *mut u64,
    total: *mut u64,
) {
    const PLACEMENT: u64 = 512;
    // SAFETY: the title passes a description (null is tolerated).
    let desc = unsafe { desc.as_ref() };
    let mut end = base;
    let mut known = desc.is_some();
    for i in 0..count {
        let one = desc.and_then(|d| subresource(d, first + i));
        let (layout, row_count, row_size) = match one {
            Some((mut footprint, row_count, row_size)) => {
                footprint.offset = end.next_multiple_of(PLACEMENT);
                let pitch = u64::from(footprint.row_pitch);
                let lines = u64::from(row_count) * u64::from(footprint.depth);
                end = footprint.offset + pitch * lines.saturating_sub(1) + row_size;
                (footprint, row_count, row_size)
            }
            None => {
                known = false;
                (NO_FOOTPRINT, u32::MAX, u64::MAX)
            }
        };
        let i = i as usize;
        // SAFETY: each output is null or holds `count` entries.
        unsafe {
            if !layouts.is_null() {
                layouts.add(i).write(layout);
            }
            if !rows.is_null() {
                rows.add(i).write(row_count);
            }
            if !row_sizes.is_null() {
                row_sizes.add(i).write(row_size);
            }
        }
    }
    if !known {
        xnw_core::warn!("d3d12x: no footprint for resource {desc:?}");
    }
    if !total.is_null() {
        // SAFETY: checked non-null.
        unsafe { total.write(if known { end - base } else { u64::MAX }) };
    }
}

/// The result pointer of a device method returning a struct through a
/// hidden argument (slots 25, 26, 108). MSVC member functions pass `this`
/// first and the result second; VIctory has the result first, TuxBox
/// (README 42) too; another runtime (idea only) saw both, and a wrong pick
/// overwrote the device on the first A press. The device is ours and
/// static, so the pick compares addresses, never what they hold (a stale
/// device or vtable pointer in the title's return buffer cannot fool it):
/// whichever of the two is not the device is the result. Neither being
/// the device (not seen): the one on this thread's stack, else the first.
/// Null when the pick is the device itself, so nothing writes over it.
fn return_slot<T>(a: *mut c_void, b: *mut c_void) -> *mut T {
    let device = super::device::device() as usize;
    let (a, b) = (a as usize, b as usize);
    let out = if a == device {
        b
    } else if b == device {
        a
    } else {
        let (mut low, mut high) = (0usize, 0usize);
        // SAFETY: two plain outputs.
        unsafe { xnw_core::win::GetCurrentThreadStackLimits(&mut low, &mut high) };
        let on_stack = |p: usize| (low..high).contains(&p);
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            xnw_core::warn!("d3d12x: struct-returning device method without the device ({a:#x}, {b:#x}); picked the stack one");
        }
        if on_stack(b) && !on_stack(a) { b } else { a }
    };
    if out == device { std::ptr::null_mut() } else { out as *mut T }
}

/// Device method 0xc8: the footprint of resources placed one after
/// another, a 16-byte value returned through a hidden first argument.
pub(crate) extern "system" fn allocation_info(a: *mut c_void, b: *mut c_void, _visible: u32, count: u32, descs: *const ResourceDesc) -> *mut AllocationInfo {
    let out = return_slot::<AllocationInfo>(a, b);
    let descs = if descs.is_null() {
        &[][..]
    } else {
        // SAFETY: the title passes `count` descriptions.
        unsafe { std::slice::from_raw_parts(descs, count as usize) }
    };
    let total = place(descs.iter(), |_, _| ());
    if !out.is_null() {
        // SAFETY: the caller passes room for the value.
        unsafe { out.write(total) };
    }
    out
}

/// The newer description: the plain one, then a sampler feedback mip region.
#[repr(C)]
pub(crate) struct ResourceDesc1 {
    desc: ResourceDesc,
    mip_region: [u32; 3],
}
const _: () = assert!(size_of::<ResourceDesc1>() == 72);

#[repr(C)]
pub(crate) struct AllocationInfo1 {
    offset: u64,
    alignment: u64,
    size: u64,
}

/// Device method 0x360 (`ID3D12Device8`): [`allocation_info`] for newer
/// descriptions, also telling where each resource goes.
pub(crate) extern "system" fn allocation_info2(
    a: *mut c_void,
    b: *mut c_void,
    _visible: u32,
    count: u32,
    descs: *const ResourceDesc1,
    each: *mut AllocationInfo1,
) -> *mut AllocationInfo {
    let out = return_slot::<AllocationInfo>(a, b);
    let descs = if descs.is_null() {
        &[][..]
    } else {
        // SAFETY: the title passes `count` descriptions.
        unsafe { std::slice::from_raw_parts(descs, count as usize) }
    };
    let total = place(descs.iter().map(|d| &d.desc), |i, info| {
        if !each.is_null() {
            // SAFETY: the title passes room for `count` entries.
            unsafe { each.add(i).write(info) };
        }
    });
    if !out.is_null() {
        // SAFETY: the caller passes room for the value.
        unsafe { out.write(total) };
    }
    out
}

/// Resources placed one after another: the whole footprint, telling `each`
/// where every resource goes. Unknown sizes give the documented `u64::MAX`.
fn place<'a>(
    descs: impl Iterator<Item = &'a ResourceDesc>,
    mut each: impl FnMut(usize, AllocationInfo1),
) -> AllocationInfo {
    let mut total = AllocationInfo {
        size: 0,
        alignment: KIB_64,
    };
    for (i, desc) in descs.enumerate() {
        let Some(one) = footprint(desc) else {
            xnw_core::warn!("d3d12x: no size for resource {desc:?}");
            each(
                i,
                AllocationInfo1 {
                    offset: u64::MAX,
                    alignment: 0,
                    size: u64::MAX,
                },
            );
            return AllocationInfo {
                size: u64::MAX,
                alignment: 0,
            };
        };
        let offset = total.size.next_multiple_of(one.alignment);
        each(
            i,
            AllocationInfo1 {
                offset,
                alignment: one.alignment,
                size: one.size,
            },
        );
        total.size = offset + one.size;
        total.alignment = total.alignment.max(one.alignment);
    }
    total
}

/// Zeroed host memory standing in for GPU memory.
pub struct Memory {
    ptr: NonNull<u8>,
    layout: Layout,
    /// Set when the memory came from the title's allocator, with these
    /// allocation attributes (it goes back there).
    attributes: Option<u64>,
}

pub type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
pub type FreeRoutine = extern "system" fn(*mut c_void, u64);
static ALLOCATOR: std::sync::OnceLock<(AllocRoutine, FreeRoutine)> = std::sync::OnceLock::new();

/// Committed resources get their memory from the title's general
/// allocator, as on the console, where titles hook it to place graphics
/// memory themselves (the title places its back buffers in a
/// block of its own and checks they fill it). The runtime names it before
/// the device exists.
pub fn use_allocator(alloc: AllocRoutine, free: FreeRoutine) {
    let _ = ALLOCATOR.set((alloc, free));
}

/// Maps the physical pages behind one range of title memory at another,
/// `(to, from, len)` (`from` 0 unmaps); false on failure.
pub type PageMapper = fn(u64, u64, u64) -> bool;
static PAGE_MAPPER: std::sync::OnceLock<PageMapper> = std::sync::OnceLock::new();

/// The runtime's page mapper, for the queues' page-mapping copies (memory
/// is the runtime's, as on the console).
pub fn use_page_mapper(mapper: PageMapper) {
    let _ = PAGE_MAPPER.set(mapper);
}

pub(crate) fn page_mapper() -> Option<PageMapper> {
    PAGE_MAPPER.get().copied()
}

/// Allocation attributes for resource memory: allocator id, page size
/// (bits 14-15: 2 MiB pages from 2 MiB up, else 64 KiB), alignment as log2
/// (bits 24-28) and memory type (bits 29-32: write-combined for GPU-only
/// heaps, cacheable for the CPU-visible ones).
/// shortcut: the console runtime's own choice of page size and memory type
/// is not known; this is the plausible one.
fn resource_attributes(info: AllocationInfo, heap_kind: u32) -> u64 {
    const GRAPHICS_ALLOCATOR: u64 = 128 << 16;
    const DEFAULT_HEAP: u32 = 1;
    let pages: u64 = if info.size >= 2 << 20 { 2 } else { 1 };
    let memory: u64 = if heap_kind == DEFAULT_HEAP { 1 } else { 3 };
    let alignment = u64::from(info.alignment.max(1).trailing_zeros());
    GRAPHICS_ALLOCATOR | pages << 14 | alignment << 24 | memory << 29
}

impl Memory {
    /// A committed resource's memory: from the title's allocator if the
    /// runtime named one, else our own; zeroed either way.
    fn for_resource(info: AllocationInfo, heap_kind: u32) -> Result<Memory, HRESULT> {
        let Some(&(alloc, _)) = ALLOCATOR.get() else {
            return Memory::new(info);
        };
        let layout = Layout::from_size_align(info.size as usize, info.alignment as usize)
            .map_err(|_| E_INVALIDARG)?;
        let attributes = resource_attributes(info, heap_kind);
        let ptr =
            NonNull::new(alloc(layout.size(), attributes).cast::<u8>()).ok_or(E_OUTOFMEMORY)?;
        // SAFETY: the allocator handed out `size` bytes.
        unsafe { ptr.as_ptr().write_bytes(0, layout.size()) };
        Ok(Memory {
            ptr,
            layout,
            attributes: Some(attributes),
        })
    }

    /// `size` zeroed bytes aligned to `alignment` (a power of two).
    pub fn zeroed(size: u64, alignment: u64) -> Result<Memory, HRESULT> {
        Memory::new(AllocationInfo { size, alignment })
    }

    pub fn address(&self) -> u64 {
        self.ptr.as_ptr() as u64
    }

    fn new(info: AllocationInfo) -> Result<Memory, HRESULT> {
        let layout = usize::try_from(info.size)
            .ok()
            .zip(usize::try_from(info.alignment).ok())
            .filter(|&(size, _)| size > 0)
            .and_then(|(size, align)| Layout::from_size_align(size, align).ok())
            .ok_or(E_INVALIDARG)?;
        // Our own memory the title reads and writes (resource and
        // heap memory, descriptor heap storage), against a guard page.
        if let Some(at) = super::guard::alloc("memory (resource, heap or descriptor storage)", layout.size(), layout.align()) {
            // SAFETY: fresh committed pages are zero; the block is ours.
            return Ok(Memory { ptr: NonNull::new(at).ok_or(E_OUTOFMEMORY)?, layout, attributes: None });
        }
        // SAFETY: the size is non-zero.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) }).ok_or(E_OUTOFMEMORY)?;
        Ok(Memory {
            ptr,
            layout,
            attributes: None,
        })
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        if self.attributes.is_none() && super::guard::free(self.ptr.as_ptr()) {
            return;
        }
        match (self.attributes, ALLOCATOR.get()) {
            (Some(attributes), Some(&(_, free))) => free(self.ptr.as_ptr().cast(), attributes),
            // SAFETY: allocated in `new` with this layout.
            _ => unsafe { dealloc(self.ptr.as_ptr(), self.layout) },
        }
    }
}

#[repr(C)]
struct Heap {
    object: Object,
    desc: HeapDesc,
    memory: Memory,
}

pub(crate) extern "system" fn create_heap(
    _: This,
    desc: *const HeapDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if desc.alignment != 0 && !desc.alignment.is_power_of_two() {
        return E_INVALIDARG;
    }
    let info = AllocationInfo {
        size: desc.size,
        alignment: desc.alignment.max(KIB_64),
    };
    let memory = match Memory::new(info) {
        Ok(memory) => memory,
        Err(hr) => return hr,
    };
    let heap = Box::new(Heap {
        object: Object::new(&HEAP_FUNCTIONS),
        desc,
        memory,
    });
    object::hand_out(HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn heap_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(HEAP, iid, object::is(iid, &HEAP_IIDS).then_some(this), out)
}
extern "system" fn heap_release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}
extern "system" fn heap_desc(this: This, out: *mut HeapDesc) -> *mut HeapDesc {
    if !out.is_null() {
        // SAFETY: `this` is a heap; the title passes room for a description.
        unsafe { out.write((*this.cast::<Heap>()).desc) };
    }
    out
}

/// A resource's memory: its own, or a range of a heap it holds a reference
/// to.
enum Backing {
    Committed(Memory),
    Placed {
        heap: This,
        offset: u64,
    },
    /// Memory the title mapped itself, at this address.
    Address(*mut u8),
}

impl Drop for Backing {
    fn drop(&mut self) {
        if let Backing::Placed { heap, .. } = *self {
            heap_release(heap);
        }
    }
}

#[repr(C)]
struct Resource {
    object: Object,
    desc: ResourceDesc,
    heap: HeapProperties,
    heap_flags: u32,
    backing: Backing,
    /// Textures also live on the host GPU (decision 0007); buffers do not
    /// yet. Made on first use (a view, a target, a copy, a barrier), not at
    /// creation: a texture the title makes and the GPU never uses costs no
    /// host memory (VIctory, 23ced70: 3,630 host textures, 8.9 GB,
    /// halved in their world; ours 12,921, 7.3 GiB at 217 s).
    host: std::sync::OnceLock<Option<std::sync::Arc<HostTexture>>>,
    /// The whole-chain host copy that replaced a reduced one when the
    /// title first wrote the texture on the GPU (see [`full_shape`]).
    full: std::sync::OnceLock<std::sync::Arc<HostTexture>>,
    /// The creation state and clear value the host copy is made with.
    made_with: (u32, Option<[u32; 5]>),
    /// Released by the title (memory quarantined), see `let_go`.
    released: std::sync::atomic::AtomicBool,
}

impl Resource {
    /// The host copy, made now if this is its first use (VIctory).
    fn host(&self) -> Option<&std::sync::Arc<HostTexture>> {
        if let Some(full) = self.full.get() {
            return Some(full);
        }
        self.host
            .get_or_init(|| {
                if self.desc.dimension == BUFFER {
                    return None;
                }
                // Memory that holds one streamed texture's pages shares
                // the host copy of the same origin (any address, any time).
                let origin = super::texture::origin_key(self.data(), &self.desc);
                if let Some(copy) = origin.as_ref().and_then(super::texture::origin_cached) {
                    PLACED_COUNTS[PC_ORIGIN_SHARED].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Some(copy);
                }
                PLACED_COUNTS[if origin.is_some() { PC_ORIGIN_MADE } else { PC_ORIGIN_NONE }].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let (state, clear) = self.made_with;
                let clear = clear.as_ref().map_or(std::ptr::null(), |c| std::ptr::from_ref(c).cast::<c_void>());
                let dropped = origin.as_ref().map_or(0, |k| super::texture::mips_to_drop(&self.desc, mip_count(&self.desc)).max(super::texture::pack_drop(k)));
                let copy = host_copy_made(&self.desc, state, clear, origin, dropped)?;
                if let Some(key) = &origin {
                    super::texture::origin_insert(*key, &copy);
                }
                Some(copy)
            })
            .as_ref()
    }
}

/// Before the title writes `this` on the GPU (a copy or resolve into
/// it): a host copy made without top mips (a `.ttpf` copy: budget, pack or
/// `XNW_MIP_DROP`) is replaced by one with the whole chain, so the title's
/// subresources all exist. True if it was replaced now (the caller then
/// uploads what the old one held from title memory and the pack).
pub(crate) fn full_shape(this: This) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    if this.is_null() {
        return false;
    }
    let r = resource(this);
    let Some(old) = r.host() else { return false };
    if old.dropped == 0 || r.full.get().is_some() {
        return false;
    }
    let (state, clear) = r.made_with;
    let clear = clear.as_ref().map_or(std::ptr::null(), |c| std::ptr::from_ref(c).cast::<c_void>());
    let Some(copy) = host_copy_made(&r.desc, state, clear, None, 0) else {
        return false;
    };
    if r.full.set(copy).is_err() {
        return false;
    }
    super::texture::forget(this as usize);
    PLACED_COUNTS[PC_REMADE_FULL].fetch_add(1, Relaxed);
    true
}

/// The host copy's subresource for the title's subresource `index` of
/// `this` (mip + slice x mips + plane x mips x slices, as D3D12): the mip
/// shifted by the mips the host copy leaves out; `None` for a mip it does
/// not have, or no host copy.
pub(crate) fn host_subresource(this: This, index: u32) -> Option<u32> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    let host = r.host()?;
    map_subresource(&r.desc, host.dropped, index)
}

fn map_subresource(desc: &ResourceDesc, dropped: u32, index: u32) -> Option<u32> {
    let mips = mip_count(desc);
    let slices = if desc.dimension == TEXTURE_3D { 1 } else { u32::from(desc.depth_or_array_size).max(1) };
    let (mip, rest) = (index % mips, index / mips);
    let (slice, plane) = (rest % slices, rest / slices);
    if mip < dropped || plane > 1 {
        return None;
    }
    let kept = mips - dropped;
    Some(mip - dropped + slice * kept + plane * kept * slices)
}

/// A copy the title recorded into or from a mip its host copy lacks:
/// skipped (never recorded invalid), counted.
pub(crate) fn skipped_copy() {
    PLACED_COUNTS[PC_COPIES_SKIPPED].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// A new host copy of a texture described by `desc` (for `origin`,
/// without `dropped` top mips), registered for video memory eviction when
/// the title writes its contents.
pub(crate) fn host_copy_made(
    desc: &ResourceDesc,
    state: u32,
    clear: *const c_void,
    origin: Option<super::texture::OriginKey>,
    dropped: u32,
) -> Option<std::sync::Arc<HostTexture>> {
    use std::sync::atomic::Ordering::Relaxed;
    // A sampled-only texture lives within the texture budget and in the
    // pool: room is made by releasing kept copies nobody else holds; failing
    // that, it gets only the coarsest mips that fit; a committed resource is
    // the last resort (counted), never the answer to a full pool.
    let sampled = desc.flags & 0xf == 0 && desc.sample_count <= 1 && clear.is_null() && desc.dimension != BUFFER;
    // A target takes back a free host texture of its kind.
    let target = if origin.is_none() && dropped == 0 { target_key(desc, clear) } else { None };
    if let Some(key) = &target
        && let Some((texture, state)) = take_free_target(key)
    {
        HOST_TEXTURES.count(&texture, true);
        return Some(std::sync::Arc::new(HostTexture {
            texture,
            state: state.into(),
            used: super::queue::presented().into(),
            evicted: false.into(),
            origin: std::sync::OnceLock::new(),
            uploaded: 0.into(),
            dropped: 0,
            prepared: std::sync::Mutex::new(None),
            pooled: None,
            pending: false.into(),
            target,
        }));
    }
    let mut dropped = dropped;
    // `XNW_TEX_POOL_MB=0`: committed textures, still within the budget.
    let pooled_only = pool_budget() > 0;
    let made = if sampled {
        // Only a copy of `.ttpf` pages may lose mips to the budget; a
        // texture of the title's own keeps its whole chain (the title
        // copies into its mips: a test run, BC3 targets of R32G32B32A32 copies).
        let max = if origin.is_some() { super::texture::max_drop(desc, mip_count(desc)).max(dropped) } else { dropped };
        loop {
            let need = host_bytes(desc, dropped);
            // And the card itself has room (`room`): our texture budget
            // is a share of DXGI's taken once at start, so targets, buffers and
            // everything else of ours could still take local video memory past
            // DXGI's budget (8,081-8,370 of 7,322 MiB in the runs without mip
            // drop that ended in the driver's access violation).
            let fits = super::texture::make_room(need) && room(need);
            if (fits || dropped >= max) && let Some(made) = host_texture(desc, state, clear, dropped, pooled_only) {
                break Some(made);
            }
            if dropped >= max {
                break None;
            }
            dropped += 1;
            PLACED_COUNTS[PC_BUDGET_DROPS].fetch_add(1, Relaxed);
        }
    } else {
        None
    };
    let (texture, pooled) = match made {
        Some(made) => made,
        None => {
            if sampled {
                PLACED_COUNTS[PC_COMMITTED_FALLBACK].fetch_add(1, Relaxed);
            }
            host_texture(desc, state, clear, dropped, false)?
        }
    };
    let state = host::legal_state(&texture, host::texture_state(state));
    let copy = std::sync::Arc::new(HostTexture {
        texture,
        state: state.into(),
        used: super::queue::presented().into(),
        evicted: false.into(),
        origin: {
            let l = std::sync::OnceLock::new();
            if let Some(k) = origin {
                let _ = l.set(k);
            }
            l
        },
        uploaded: 0.into(),
        dropped,
        prepared: std::sync::Mutex::new(None),
        pooled,
        pending: false.into(),
        // Targets are never in the sampled texture pool (`host_texture`).
        target,
    });
    // Pooled textures are not demoted one by one (a placed resource has no
    // residency priority of its own); the pool's budget bounds them.
    if copy.pooled.is_none() && super::texture::title_written(&copy.texture) {
        let key = windows::core::Interface::as_raw(&copy.texture) as usize;
        let mut all = EVICTABLE.lock().unwrap_or_else(|e| e.into_inner());
        all.get_or_insert_with(Default::default).insert(key, std::sync::Arc::downgrade(&copy));
    }
    Some(copy)
}

/// A title texture's host copy and the state it is in. The title's barriers
/// name console states (looser than the host's), so we transition from this
/// one. Lists that use the texture hold it until they are reset: the title
/// may release a resource whose commands it recorded but not yet submitted
/// (console memory outlives the object; a host texture must outlive the
/// lists that name it).
pub struct HostTexture {
    pub texture: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    pub state: std::sync::atomic::AtomicU32,
    /// The presented frame a submitted list last used it in, and whether it
    /// is evicted for video memory ([`evict_over_budget`]).
    used: std::sync::atomic::AtomicU64,
    evicted: std::sync::atomic::AtomicBool,
    /// The streamed texture origin it was made for (shared by every
    /// resource over the same pages), the origin stamp its contents were
    /// last uploaded from (0: none), mips dropped from the top
    /// (`XNW_MIP_DROP`), and an upload prepared off the frame.
    pub origin: std::sync::OnceLock<super::texture::OriginKey>,
    pub(crate) uploaded: std::sync::atomic::AtomicU64,
    pub dropped: u32,
    pub(crate) prepared: std::sync::Mutex<Option<super::texture::Prepared>>,
    /// Its range of a pool heap (placed), given back after it.
    pooled: Option<PoolRange>,
    /// An upload of it was put off, or holds only the mips arrived so
    /// far: whoever binds it must ask again next frame (bindless slots
    /// otherwise look again only every 60 frames, 8 a frame).
    pub pending: std::sync::atomic::AtomicBool,
    /// A target's free-pool key: let go, it is kept for the next
    /// target of its kind ([ree_target]).
    target: Option<TargetKey>,
}
impl Drop for HostTexture {
    fn drop(&mut self) {
        HOST_TEXTURES.count(&self.texture, false);
        match self.target.take() {
            Some(key) => free_target(key, self.texture.clone(), self.state.load(std::sync::atomic::Ordering::Relaxed)),
            None => host::retire(self.texture.clone()),
        }
    }
}

/// Moves the host texture to `wanted` (a host texture state); the host
/// resource and the state it leaves, if that is a change.
pub(crate) fn host_transition(this: This, wanted: u32) -> Option<(*mut c_void, u32)> {
    use windows::core::Interface;
    let host = resource(this).host()?;
    let before = host
        .state
        .swap(wanted, std::sync::atomic::Ordering::Relaxed);
    (before != wanted).then(|| (host.texture.as_raw(), before))
}

/// The host GPU's copy of a title resource, if it has one.
pub(crate) fn host_resource<'a>(
    this: This,
) -> Option<&'a windows::Win32::Graphics::Direct3D12::ID3D12Resource> {
    if this.is_null() {
        return None;
    }
    resource(this).host().map(|h| &h.texture)
}

/// A title texture's host copy, shared, for a list to hold.
pub fn host_copy(this: This) -> Option<std::sync::Arc<HostTexture>> {
    if this.is_null() {
        return None;
    }
    resource(this).host().cloned()
}

/// A host texture like the title's: same description, in GPU memory,
/// without the console's layout and flags.
fn host_texture(
    desc: &ResourceDesc,
    state: u32,
    clear: *const c_void,
    dropped: u32,
    pool_only: bool,
) -> Option<(windows::Win32::Graphics::Direct3D12::ID3D12Resource, Option<PoolRange>)> {
    use windows::Win32::Graphics::Direct3D12::*;
    const PC_FLAGS: u32 = 0xf; // render target, depth, UAV, no shader access
    if desc.dimension == BUFFER {
        return None;
    }
    // hook: a video memory budget check before making it
    // (against `IDXGIAdapter3::QueryVideoMemoryInfo`, see
    // `memory_line`). hook: render, depth and unordered-access
    // targets go on the shared page pool instead of a texture of their own.
    let host = host::host()?;
    let mut d = *desc;
    d.layout = 0;
    d.alignment = 0;
    d.flags &= PC_FLAGS;
    // `XNW_MIP_DROP` (VIctory, 9a85db7): without the top mips.
    if dropped > 0 {
        d.width = (d.width >> dropped).max(1);
        d.height = (d.height >> dropped).max(1);
        d.mip_levels = (mip_count(desc) - dropped) as u16;
    }
    // PC refuses depth targets that are also unordered-access (the console
    // allows both; the title's shadow maps ask for it).
    // shortcut: the host copy loses UAV use; give it a second view-only
    // copy if draws write it as a UAV.
    const DEPTH: u32 = 2;
    const UNORDERED: u32 = 4;
    if d.flags & DEPTH != 0 {
        d.flags &= !UNORDERED;
    }
    d.format = pc_format(d.format);
    const DENY_SHADER_RESOURCE: u32 = 8;
    if d.flags & DENY_SHADER_RESOURCE == 0
        && let Some(&(_, typeless)) = DEPTH_FORMATS.iter().find(|&&(f, _)| f == d.format)
    {
        d.format = typeless;
    }
    // The host has no sRGB unordered access: such textures are typeless,
    // their unordered access views plain UNORM.
    if d.flags & UNORDERED != 0 && SRGB.iter().any(|&(srgb, _)| srgb == d.format) {
        d.format = family(d.format);
    }
    // NVIDIA GPUs reject packed 16-bit color formats (B5G5R5A1_UNORM = 86,
    // B4G4R4A4_UNORM = 115) with RENDER_TARGET or UNORDERED_ACCESS, which
    // turns every view into a null descriptor. Store them as R16_TYPELESS
    // (same 16 bpp) so the host texture exists.
    const RENDER_TARGET: u32 = 1;
    const R16_TYPELESS: u32 = 53;
    const R16_UNORM: u32 = 56;
    let packed16_fallback =
        matches!(d.format, 86 | 115) && d.flags & (RENDER_TARGET | UNORDERED) != 0;
    if packed16_fallback {
        d.format = R16_TYPELESS;
    }
    // The clear value names the format too: format, then color or depth
    // and stencil (5 words).
    let clear_value = (!clear.is_null()).then(|| {
        // SAFETY: the title passes a whole clear value.
        let mut value = unsafe { clear.cast::<[u32; 5]>().read_unaligned() };
        value[0] = if packed16_fallback {
            R16_UNORM
        } else {
            pc_format(value[0])
        };
        value
    });
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let started = std::time::Instant::now();
    // Sampled-only textures go in a pooled heap (placing
    // costs ~0.03 ms, committing ~0.4-1 ms).
    if d.flags == 0 && d.sample_count <= 1 && clear_value.is_none()
        && let Some((texture, range)) = pool_place(host, &d, host::legal_for(d.flags, host::texture_state(state)))
    {
        host::stage(host::STAGE_HOST_TEXTURE, started);
        PLACED_COUNTS[PC_POOLED].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        HOST_TEXTURES.count(&texture, true);
        return Some((texture, Some(range)));
    }
    if pool_only {
        return None;
    }
    let mut texture: Option<ID3D12Resource> = None;
    PLACED_COUNTS[PC_COMMITTED].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: the description has the host layout (checked size, same
    // fields); the out pointer is ours.
    let made = unsafe {
        host.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            (&raw const d).cast(),
            D3D12_RESOURCE_STATES(
                host::legal_for(d.flags, host::texture_state(state)) as i32,
            ),
            // The same layout: format, then color or depth and stencil.
            clear_value.as_ref().map(|v| std::ptr::from_ref(v).cast()),
            &mut texture,
        )
    };
    host::stage(host::STAGE_HOST_TEXTURE, started);
    host::note_creation("committed texture", d.width * u64::from(d.height), started);
    if let Err(e) = made {
        xnw_core::syscalls::creation_failed(xnw_core::syscalls::Made::Texture, &e, &format_args!("committed host texture {d:?}"));
    }
    if let Some(t) = &texture {
        HOST_TEXTURES.count(t, true);
        host::name(t, || {
            format!("texture {}x{} f{} flags {:#x}", d.width, d.height, d.format, d.flags)
        });
    }
    texture.map(|t| (t, None))
}

// The texture heap pool. Sampled-only host textures are placed
// resources in 64 MiB default heaps (non-target textures only), placed at
// 4 KiB alignment when the host allows (small textures) and 64 KiB
// otherwise; first fit over the heaps, a new heap when none fits, up to
// `XNW_TEX_POOL_MB` (default 4096; 0 off), then committed resources. A
// range given back waits [`POOL_QUARANTINE`] presented frames (the host
// texture itself is released 4 frames after its last use, `host::retire`)
// before another texture is placed there, so no two live resources alias.
// Textures over half a heap stay committed.

const POOL_HEAP: u64 = 64 << 20;
const POOL_QUARANTINE: u64 = 8;

struct PoolHeap {
    heap: windows::Win32::Graphics::Direct3D12::ID3D12Heap,
    /// Free ranges: offset -> bytes.
    free: std::collections::BTreeMap<u64, u64>,
}

struct Pool {
    heaps: Vec<PoolHeap>,
    /// Ranges given back: (presented frame, heap, offset, bytes).
    quarantine: Vec<(u64, u32, u64, u64)>,
}

// SAFETY: D3D12 heaps are free-threaded.
unsafe impl Send for Pool {}

static POOL: std::sync::Mutex<Pool> = std::sync::Mutex::new(Pool { heaps: Vec::new(), quarantine: Vec::new() });

/// A host texture's range of a pool heap, given back when dropped.
pub(crate) struct PoolRange {
    heap: u32,
    offset: u64,
    size: u64,
}

impl Drop for PoolRange {
    fn drop(&mut self) {
        let frame = super::queue::presented();
        POOL.lock().unwrap_or_else(|e| e.into_inner()).quarantine.push((frame, self.heap, self.offset, self.size));
    }
}

fn pool_budget() -> u64 {
    static MB: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| std::env::var("XNW_TEX_POOL_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(4096));
    (*MB << 20).min(tex_budget())
}

/// Host texture bytes allowed (all host textures, pooled and
/// committed): `XNW_TEX_BUDGET_MB`, default 60 % of the local video memory
/// budget DXGI gives at the first ask (4 GiB without an adapter).
pub(crate) fn tex_budget() -> u64 {
    static BUDGET: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| {
        if let Some(mb) = std::env::var("XNW_TEX_BUDGET_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()) {
            return mb << 20;
        }
        use windows::Win32::Graphics::Dxgi::DXGI_MEMORY_SEGMENT_GROUP_LOCAL;
        let mut info = windows::Win32::Graphics::Dxgi::DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: a plain query into a local.
        let asked = adapter().is_some_and(|a| unsafe { a.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }.is_ok());
        let budget = if asked && info.Budget > 0 { info.Budget / 10 * 6 } else { 4 << 30 };
        xnw_core::info!("d3d12x: host texture budget {} MiB (XNW_TEX_BUDGET_MB)", budget >> 20);
        budget
    });
    *BUDGET
}

/// Whether `need` more bytes fit the card: the process's local video
/// memory in use, plus `need`, under nine tenths of the budget DXGI gives
/// it now. Ported from victory-fork (fxceless/victory-fork,
/// crates/victory-gpu/src/resource.rs `room`, 4e17478, GPL-3.0-or-later),
/// where it gates host copies of textures the title fills past a 2 GiB
/// floor; here it gates sampled host copies, which then lose top mips (an
/// origin copy) instead. Asked of DXGI at most every 2 ms (a texture made
/// in between is counted on top). True without an adapter.
pub(crate) fn room(need: u64) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    use windows::Win32::Graphics::Dxgi::DXGI_MEMORY_SEGMENT_GROUP_LOCAL;
    static ASKED: std::sync::Mutex<(Option<std::time::Instant>, u64, u64)> = std::sync::Mutex::new((None, 0, 0));
    static SINCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let (used, budget) = {
        let mut a = ASKED.lock().unwrap_or_else(|e| e.into_inner());
        if a.0.is_none_or(|t| t.elapsed() >= std::time::Duration::from_millis(2)) {
            let Some(adapter) = adapter() else { return true };
            let mut info = windows::Win32::Graphics::Dxgi::DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            // SAFETY: a plain query into a local.
            if unsafe { adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }.is_err() || info.Budget == 0 {
                return true;
            }
            *a = (Some(std::time::Instant::now()), info.CurrentUsage, info.Budget);
            SINCE.store(0, Relaxed);
        }
        (a.1 + SINCE.load(Relaxed), a.2)
    };
    let fits = used + need <= budget / 10 * 9;
    if fits {
        SINCE.fetch_add(need, Relaxed);
    } else {
        PLACED_COUNTS[PC_NO_ROOM].fetch_add(1, Relaxed);
    }
    fits
}

/// Whether `need` more bytes of CPU-visible memory fit 9/10 of the
/// non-local budget DXGI gives the process (asked each time: rare calls).
pub(crate) fn room_nonlocal(need: u64) -> bool {
    use windows::Win32::Graphics::Dxgi::DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL;
    let Some(a) = adapter() else { return true };
    let mut info = windows::Win32::Graphics::Dxgi::DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    // SAFETY: a plain query into a local.
    if unsafe { a.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut info) }.is_err() || info.Budget == 0 {
        return true;
    }
    info.CurrentUsage + need <= info.Budget / 10 * 9
}

/// "video memory local U of B MiB, non-local U of B MiB" now.
pub(crate) fn video_memory_text() -> String {
    use windows::Win32::Graphics::Dxgi::{DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL};
    let one = |g| {
        let a = adapter()?;
        let mut info = windows::Win32::Graphics::Dxgi::DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: a plain query into a local.
        unsafe { a.QueryVideoMemoryInfo(0, g, &mut info) }.ok()?;
        Some(format!("{} of {} MiB", info.CurrentUsage >> 20, info.Budget >> 20))
    };
    format!("video memory local {}, non-local {}", one(DXGI_MEMORY_SEGMENT_GROUP_LOCAL).unwrap_or_default(), one(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL).unwrap_or_default())
}

/// Bytes of every host texture alive now.
pub(crate) fn host_texture_bytes() -> u64 {
    HOST_TEXTURES.bytes.load(std::sync::atomic::Ordering::Relaxed)
}

/// The video memory a host copy of `desc` without `dropped` top mips
/// takes (as `host_texture` would describe it).
pub(crate) fn host_bytes(desc: &ResourceDesc, dropped: u32) -> u64 {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC;
    let Some(host) = host::host() else { return 0 };
    let mut d = *desc;
    d.layout = 0;
    d.alignment = 0;
    d.flags &= 0xf;
    d.format = pc_format(d.format);
    if dropped > 0 {
        d.width = (d.width >> dropped).max(1);
        d.height = (d.height >> dropped).max(1);
        d.mip_levels = (mip_count(desc) - dropped) as u16;
    }
    // SAFETY: a plain query; the description has the host layout.
    let info = unsafe { host.device.GetResourceAllocationInfo(0, &[(&raw const d).cast::<D3D12_RESOURCE_DESC>().read()]) };
    if info.SizeInBytes == u64::MAX { 0 } else { info.SizeInBytes }
}

/// The video memory a live texture takes.
pub(crate) fn texture_bytes(t: &windows::Win32::Graphics::Direct3D12::ID3D12Resource) -> u64 {
    allocation_bytes(std::iter::once(t.clone()))
}

/// Adds a free range, merged with its neighbours.
fn pool_free(free: &mut std::collections::BTreeMap<u64, u64>, mut offset: u64, mut size: u64) {
    if let Some((&at, &len)) = free.range(..offset).next_back()
        && at + len == offset
    {
        free.remove(&at);
        offset = at;
        size += len;
    }
    if let Some(len) = free.remove(&(offset + size)) {
        size += len;
    }
    free.insert(offset, size);
}

/// Takes `size` bytes at `align` from `free`; the offset.
fn pool_take(free: &mut std::collections::BTreeMap<u64, u64>, size: u64, align: u64) -> Option<u64> {
    let (at, len, start) = free.iter().find_map(|(&at, &len)| {
        let start = at.next_multiple_of(align);
        (start + size <= at + len).then_some((at, len, start))
    })?;
    free.remove(&at);
    if start > at {
        free.insert(at, start - at);
    }
    if start + size < at + len {
        free.insert(start + size, at + len - start - size);
    }
    Some(start)
}

/// A range of `size` bytes at `align` in a pool heap (a new heap if none
/// has one and the budget allows).
fn pool_alloc(host: &host::Host, size: u64, align: u64) -> Option<(windows::Win32::Graphics::Direct3D12::ID3D12Heap, PoolRange)> {
    use windows::Win32::Graphics::Direct3D12::*;
    if size > POOL_HEAP / 2 {
        return None;
    }
    let frame = super::queue::presented();
    let mut guard = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let pool = &mut *guard;
    let (ready, waiting): (Vec<_>, Vec<_>) = pool.quarantine.drain(..).partition(|&(f, ..)| f + POOL_QUARANTINE <= frame);
    pool.quarantine = waiting;
    for (_, heap, offset, size) in ready {
        pool_free(&mut pool.heaps[heap as usize].free, offset, size);
    }
    for (i, h) in pool.heaps.iter_mut().enumerate() {
        if let Some(offset) = pool_take(&mut h.free, size, align) {
            return Some((h.heap.clone(), PoolRange { heap: i as u32, offset, size }));
        }
    }
    if (pool.heaps.len() as u64 + 1) * POOL_HEAP > pool_budget() {
        return None;
    }
    // The heap is made outside the pool's lock (a creation may sit in
    // the kernel; every pooled texture would wait behind it).
    let made = pool.heaps.len();
    drop(guard);
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: POOL_HEAP,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        Alignment: 64 << 10,
        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
    };
    let mut heap: Option<ID3D12Heap> = None;
    let creating = std::time::Instant::now();
    // SAFETY: plain heap creation.
    let created = unsafe { host.device.CreateHeap(&desc, &mut heap) };
    host::note_creation("texture pool heap", POOL_HEAP, creating);
    if let Err(e) = created {
        xnw_core::warn!("d3d12x: texture pool heap not made: {e}");
        return None;
    }
    let heap = heap?;
    host::name(&heap, || format!("texture pool heap {made}"));
    let mut guard = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let pool = &mut *guard;
    let mut free = std::collections::BTreeMap::new();
    free.insert(0, POOL_HEAP);
    let offset = pool_take(&mut free, size, align)?;
    pool.heaps.push(PoolHeap { heap: heap.clone(), free });
    Some((heap, PoolRange { heap: pool.heaps.len() as u32 - 1, offset, size }))
}

/// `d` (a host description) placed in the pool in `state`.
fn pool_place(host: &host::Host, d: &ResourceDesc, state: u32) -> Option<(windows::Win32::Graphics::Direct3D12::ID3D12Resource, PoolRange)> {
    use windows::Win32::Graphics::Direct3D12::*;
    if pool_budget() == 0 {
        return None;
    }
    let mut d = *d;
    // Small textures may take 4 KiB alignment; the host says which.
    d.alignment = 4096;
    // SAFETY: plain queries with the host layout of the description.
    let mut info = unsafe { host.device.GetResourceAllocationInfo(0, &[(&raw const d).cast::<D3D12_RESOURCE_DESC>().read()]) };
    if info.SizeInBytes == u64::MAX || info.Alignment != 4096 {
        d.alignment = 0;
        // SAFETY: as above.
        info = unsafe { host.device.GetResourceAllocationInfo(0, &[(&raw const d).cast::<D3D12_RESOURCE_DESC>().read()]) };
    }
    if info.SizeInBytes == u64::MAX {
        return None;
    }
    let (heap, range) = pool_alloc(host, info.SizeInBytes, info.Alignment.max(4096))?;
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: a range of our heap that no live resource uses, sized and
    // aligned as the host asked; the out pointer is ours.
    let creating = std::time::Instant::now();
    let made = unsafe { host.device.CreatePlacedResource(&heap, range.offset, (&raw const d).cast(), D3D12_RESOURCE_STATES(state as i32), None, &mut texture) };
    host::note_creation("pooled texture", info.SizeInBytes, creating);
    if let Err(e) = made {
        xnw_core::syscalls::creation_failed(xnw_core::syscalls::Made::Pooled, &e, &format_args!("pooled host texture {d:?}"));
        return None;
    }
    let texture = texture?;
    host::name(&texture, || format!("pooled texture {}x{} f{}", d.width, d.height, d.format));
    Some((texture, range))
}

/// A texture's console memory and description; `None` for buffers.
pub(crate) fn texture_memory(this: This) -> Option<(*const u8, ResourceDesc)> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    (r.desc.dimension != BUFFER).then(|| (r.data().cast_const(), r.desc))
}

/// A buffer's host memory and size; `None` for textures.
pub(crate) fn buffer_bytes(this: This) -> Option<(*mut u8, u64)> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    (r.desc.dimension == BUFFER).then(|| (r.data(), r.desc.width))
}

impl Resource {
    fn data(&self) -> *mut u8 {
        match self.backing {
            Backing::Committed(ref memory) => memory.ptr.as_ptr(),
            // SAFETY: placement checked the range lies inside the heap.
            Backing::Placed { heap, offset } => unsafe {
                (*heap.cast::<Heap>())
                    .memory
                    .ptr
                    .as_ptr()
                    .add(offset as usize)
            },
            Backing::Address(at) => at,
        }
    }
}

fn resource<'a>(this: This) -> &'a Resource {
    // SAFETY: called from resource methods only.
    unsafe { &*this.cast::<Resource>() }
}

/// Logs each new shape of render target or depth buffer (texture with
/// flag 1 or 2), so a run shows the sizes the title renders at.
fn log_target(desc: &ResourceDesc, heap_flags: u32, how: &str) {
    const RENDER_TARGET: u32 = 1;
    const DEPTH_STENCIL: u32 = 2;
    if desc.dimension == BUFFER || desc.flags & (RENDER_TARGET | DEPTH_STENCIL) == 0 {
        return;
    }
    static SEEN: std::sync::Mutex<Vec<(u64, u32, u16, u32, u32)>> = std::sync::Mutex::new(Vec::new());
    let key = (desc.width, desc.height, desc.depth_or_array_size, desc.format, desc.flags);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.len() >= 256 || seen.contains(&key) {
        return;
    }
    seen.push(key);
    let kind = if desc.flags & DEPTH_STENCIL != 0 { "depth" } else { "render target" };
    xnw_core::info!(
        "d3d12x: {kind} {}x{}x{} format {} mips {} flags {:#x} heap flags {heap_flags:#x} ({how})",
        desc.width,
        desc.height,
        desc.depth_or_array_size,
        desc.format,
        desc.mip_levels,
        desc.flags
    );
}

// The creation methods' own arguments, passed on.
#[allow(clippy::too_many_arguments)]
fn hand_out_resource(
    desc: ResourceDesc,
    heap: HeapProperties,
    heap_flags: u32,
    backing: Backing,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    let how = match backing {
        Backing::Committed(_) => "committed",
        Backing::Placed { .. } => "placed",
        Backing::Address(_) => "at address",
    };
    log_target(&desc, heap_flags, how);
    if desc.dimension != BUFFER {
        TITLE_TEXTURES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    // The clear value is the caller's only during the call.
    // SAFETY: the title passes a whole clear value (5 words) or null.
    let made_with = (state, (!clear.is_null()).then(|| unsafe { clear.cast::<[u32; 5]>().read_unaligned() }));
    let resource = Box::new(Resource {
        object: Object::new(&RESOURCE_FUNCTIONS),
        desc,
        heap,
        heap_flags,
        backing,
        host: std::sync::OnceLock::new(),
        full: std::sync::OnceLock::new(),
        made_with,
        released: false.into(),
    });
    if desc.dimension == BUFFER {
        let at = std::ptr::from_ref::<Resource>(&resource) as usize;
        super::shadow::register(at, resource.data() as u64, desc.width);
    }
    LIVE.lock().unwrap_or_else(|e| e.into_inner()).insert(std::ptr::from_ref::<Resource>(&resource) as usize);
    object::hand_out(RESOURCE, resource, iid, &RESOURCE_IIDS, out)
}

/// Title resource objects not yet released (for `XNW_GPU_CHECK`; locked only at
/// creation and release; a driver fault under `CreateShaderResourceView`
/// in a test run led to it). Views name resources by address and the title
/// releases a resource on any thread, also while a submission makes host
/// views of slots that name it. The object's memory is kept
/// ([`QUARANTINE`]) after its release, so a view taken before it can still
/// read the count and see 0: holding is one atomic, no lock (a global lock
/// per view cost plan making 10 to 100x).
static LIVE: std::sync::Mutex<std::collections::BTreeSet<usize>> = std::sync::Mutex::new(std::collections::BTreeSet::new());

/// Released resource objects (submission count and time at release),
/// freed once two submissions and 100 ms have passed: no view lookup lasts
/// that long (they run inside one plan, record or descriptor write).
static QUARANTINE: std::sync::Mutex<std::collections::VecDeque<(u64, std::time::Instant, usize)>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());
static SUBMIT_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// References whose release would free the object, let go at the next
/// submission (releasing a resource takes descriptor locks a holder may hold).
static LAST_HELD: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

/// Counters (memory line): views of released resources made null
/// instead, last references put off to the next submission.
pub(crate) static HELD_COUNTS: [std::sync::atomic::AtomicU64; 9] = [const { std::sync::atomic::AtomicU64::new(0) }; 9];
pub(crate) const HC_STALE_NULLED: usize = 0;
const HC_PUT_OFF: usize = 1;
/// Heap slots of released views made null; free targets not taken back
/// because something else still holds them (a slot, a plan, the retire list).
pub(crate) const HC_SLOTS_NULLED: usize = 2;
const HC_POOL_HELD: usize = 3;
/// Bindless line, taken per line: holds and their time (one in 64
/// timed, scaled), microseconds nulling slots of released views.
pub(crate) const HC_HOLDS: usize = 4;
pub(crate) const HC_HOLD_US: usize = 5;
pub(crate) const HC_SWEEP_US: usize = 6;
/// Memory line: slots in place over a released shared window made
/// null; stale slots `XNW_GPU_CHECK=1` made null at its check.
pub(crate) const HC_WINDOW_NULLED: usize = 7;
pub(crate) const HC_CHECK_NULLED: usize = 8;

/// The bindless line's part: holds and their time, release sweeps.
pub(crate) fn held_text() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let holds = HELD_COUNTS[HC_HOLDS].swap(0, Relaxed);
    let hold_ns = HELD_COUNTS[HC_HOLD_US].swap(0, Relaxed);
    let sweep = HELD_COUNTS[HC_SWEEP_US].swap(0, Relaxed);
    format!("{holds} view resources held (~{} us), {sweep} us nulling slots of released views", hold_ns * 64 / 1000)
}

/// The COM reference count of `object` (one taken and given back).
pub(crate) fn com_count<T: windows::core::Interface>(object: &T) -> u32 {
    let raw = object.as_raw();
    // SAFETY: every COM interface starts with IUnknown's table; an
    // AddRef/Release pair on a live object, Release answers the count left.
    unsafe {
        let table = *raw.cast::<*const windows::core::IUnknown_Vtbl>();
        ((*table).AddRef)(raw);
        ((*table).Release)(raw)
    }
}

/// A title resource held alive (see [`LIVE`]).
pub struct LiveHold(This);
impl Drop for LiveHold {
    fn drop(&mut self) {
        if !object::release_unless_last(self.0) {
            HELD_COUNTS[HC_PUT_OFF].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            LAST_HELD.lock().unwrap_or_else(|e| e.into_inner()).push(self.0 as usize);
        }
    }
}

/// `this` held, if it is a title resource not released (`None` for a
/// released one: its views are stale). One atomic; the memory of a
/// released object is kept long enough ([`QUARANTINE`]).
pub fn hold_live(this: This) -> Option<LiveHold> {
    if this.is_null() {
        return None;
    }
    let n = HELD_COUNTS[HC_HOLDS].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let at = n.is_multiple_of(64).then(std::time::Instant::now);
    let held = object::try_add_ref(this).then_some(LiveHold(this));
    if let Some(at) = at {
        HELD_COUNTS[HC_HOLD_US].fetch_add(at.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    }
    held
}

/// Whether `this` is a title resource not released (`XNW_GPU_CHECK`).
pub(crate) fn is_live(this: usize) -> bool {
    LIVE.lock().unwrap_or_else(|e| e.into_inner()).contains(&this)
}

/// Lets go of the references [`LiveHold`] put off, and frees the
/// released objects old enough; at each submission, before its locks.
pub(crate) fn release_held() {
    let seq = SUBMIT_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    let last = std::mem::take(&mut *LAST_HELD.lock().unwrap_or_else(|e| e.into_inner()));
    for this in last {
        resource_release(this as This);
    }
    free_quarantined(|at, when| at + 2 <= seq && when.elapsed() >= std::time::Duration::from_millis(100));
}

/// Frees the quarantined objects `due` says are old enough, oldest first.
fn free_quarantined(due: impl Fn(u64, std::time::Instant) -> bool) {
    let mut gone = Vec::new();
    {
        let mut q = QUARANTINE.lock().unwrap_or_else(|e| e.into_inner());
        while let Some(&(at, when, this)) = q.front()
            && due(at, when)
        {
            q.pop_front();
            gone.push(this);
        }
    }
    for this in gone {
        // SAFETY: released (count 0) and quarantined since: the object is ours.
        unsafe { object::free(this as *mut Resource) };
    }
}

/// Tests: frees every quarantined object now.
#[cfg(test)]
pub(crate) fn free_quarantined_now() {
    free_quarantined(|_, _| true);
}

impl Resource {
    /// What a release does at once (the memory goes later): the views
    /// naming it go stale, its buffer range is let go.
    fn let_go(&self) {
        if self.released.swap(true, std::sync::atomic::Ordering::AcqRel) {
            return;
        }
        let me = std::ptr::from_ref::<Resource>(self) as usize;
        LIVE.lock().unwrap_or_else(|e| e.into_inner()).remove(&me);
        if self.desc.dimension != BUFFER {
            TITLE_TEXTURES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        super::texture::forget(me);
        super::descriptor::forget(me);
        if self.desc.dimension == BUFFER {
            super::shadow::unregister(me, self.data() as u64);
        }
    }
}

/// Texture resources made and released so far (the title's and placed),
/// for the memory line: whether the title lets go of what it streams.
static TITLE_TEXTURES: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Drop for Resource {
    fn drop(&mut self) {
        // Released already, unless it was never handed out.
        self.let_go();
    }
}

pub(crate) extern "system" fn create_committed(
    _: This,
    heap: *const HeapProperties,
    heap_flags: u32,
    desc: *const ResourceDesc,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes both descriptions (null is tolerated).
    let (Some(&heap), Some(&desc)) = (unsafe { (heap.as_ref(), desc.as_ref()) }) else {
        return E_INVALIDARG;
    };
    let Some(info) = footprint(&desc) else {
        xnw_core::warn!("d3d12x: no size for resource {desc:?}");
        return E_INVALIDARG;
    };
    let memory = match Memory::for_resource(info, heap.kind) {
        Ok(memory) => memory,
        Err(hr) => return hr,
    };
    hand_out_resource(
        desc,
        heap,
        heap_flags,
        Backing::Committed(memory),
        state,
        clear,
        iid,
        out,
    )
}

/// Device method 0x3f8, inferred from the title's calls
/// (title+0x193e1b4: its back buffers, heap flag "allow display"): the
/// committed resource creation of the enhanced-barrier interfaces,
/// `(heap, heap flags, desc1, initial layout, clear value, protected
/// session, castable format count, castable formats, iid, out)`. The host
/// copy starts in the common state, as layouts are not tracked.
#[allow(clippy::too_many_arguments)] // the method's own arguments
pub(crate) extern "system" fn create_committed3(
    this: This,
    heap: *const HeapProperties,
    heap_flags: u32,
    desc: *const ResourceDesc,
    _layout: u32,
    clear: *const c_void,
    _protected: This,
    _castable_count: u32,
    _castable: *const u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    const COMMON: u32 = 0;
    create_committed(this, heap, heap_flags, desc, COMMON, clear, iid, out)
}

/// Registered page pools (64 KiB pages the title maps into tiled
/// resources by page index): first address and page count; a handle is an
/// index plus one, a removed pool leaves `None`.
static PAGE_POOLS: std::sync::Mutex<Vec<Option<(u64, u32)>>> = std::sync::Mutex::new(Vec::new());

fn page_pools() -> std::sync::MutexGuard<'static, Vec<Option<(u64, u32)>>> {
    PAGE_POOLS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Device method 0x250: `(address, page count, handle out)`.
pub(crate) extern "system" fn register_page_pool(
    _: This,
    address: u64,
    pages: u32,
    out: *mut usize,
) -> HRESULT {
    if out.is_null() || pages == 0 || !address.is_multiple_of(KIB_64) {
        return E_INVALIDARG;
    }
    let mut pools = page_pools();
    pools.push(Some((address, pages)));
    // SAFETY: checked non-null; the title passes room for a handle.
    unsafe { out.write(pools.len()) };
    S_OK
}

/// Device method 0x258.
pub(crate) extern "system" fn unregister_page_pool(_: This, handle: usize) -> HRESULT {
    match page_pools().get_mut(handle.wrapping_sub(1)) {
        Some(pool @ Some(_)) => {
            *pool = None;
            S_OK
        }
        _ => E_INVALIDARG,
    }
}

/// Device method 0x428, one of the `ID3D12Device9` to `ID3D12Device11`
/// additions. From the title's call: a resource at a virtual
/// address the title mapped itself, `(address, desc1, initial barrier
/// layout, clear value, iid, out)`, as `CreatePlacedResourceX` with
/// enhanced barriers. GPU addresses are CPU addresses, so the resource is
/// a view of that memory. `D3D12_RESOURCE_DESC1` starts like the plain
/// description; its sampler feedback region is not needed.
pub(crate) extern "system" fn create_at_address(
    _: This,
    address: u64,
    desc: *const ResourceDesc,
    _layout: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if address == 0 {
        return E_INVALIDARG;
    }
    // The title's calls only (placed textures of ours come in through
    // `place_at_address`; counted here they were a test run's 674 "other" a
    // frame).
    note_placement_caller();
    place_at_address(address, desc, clear, iid, out)
}

/// [`create_at_address`]'s resource, for the title and for placed
/// textures of ours ([`texture_at`]).
fn place_at_address(address: u64, desc: ResourceDesc, clear: *const c_void, iid: *const Iid, out: *mut This) -> HRESULT {
    const COMMON: u32 = 0;
    const DEFAULT: HeapProperties = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 1,
        visible_nodes: 1,
    };
    hand_out_resource(
        desc,
        DEFAULT,
        0,
        Backing::Address(address as *mut u8),
        COMMON,
        clear,
        iid,
        out,
    )
}

/// Textures the title placed in its own memory and views without a
/// resource object (device 0x230, streamed and interface textures), by
/// address and description: a resource at that address and where its
/// footprint ends, made the first time a draw reads one and evicted when
/// [`super::watch::title_memory_remapped`] unmaps or remaps the range, or
/// least recently used first when video memory passes its limit
/// ([`evict_over_budget`]). The third field is the presented frame it was
/// last reached in.
type PlacedTextureMap = std::collections::BTreeMap<(u64, [u32; 14]), (usize, u64, u64)>;
static PLACED_TEXTURES: std::sync::Mutex<PlacedTextureMap> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// Incremented whenever [`evict_placed_textures`] removes an entry so
/// bindless heaps re-check placed texture slots.
static PLACED_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) fn placed_epoch() -> u64 {
    PLACED_EPOCH.load(std::sync::atomic::Ordering::Acquire)
}

/// A reference to a placed texture: eviction does not free it while held.
pub struct HeldTexture(This);
impl HeldTexture {
    pub(crate) fn as_this(&self) -> This {
        self.0
    }
}
impl Drop for HeldTexture {
    fn drop(&mut self) {
        resource_release(self.0);
    }
}

/// A reference to `texture` if it is a placed texture alive now.
pub(crate) fn hold_placed(texture: This) -> Option<HeldTexture> {
    let mut map = super::counters::timed(super::counters::LOCK_PLACED, &PLACED_TEXTURES);
    let frame = super::queue::presented();
    map.values_mut().find(|(r, ..)| *r == texture as usize).map(|(_, _, used)| {
        *used = frame;
        object::add_ref(texture);
        HeldTexture(texture)
    })
}

/// The placed textures alive now (resource objects).
pub(crate) fn placed_live() -> std::collections::HashSet<usize> {
    PLACED_TEXTURES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .values()
        .map(|&(r, ..)| r)
        .collect()
}

/// A resource description (as words) without the garbage in its padding
/// (`d[1]` after the dimension, `d[13]` at the end), so equal descriptions
/// key the same placed texture (VIctory, e7b2182).
pub(crate) fn canonical_resource_desc(mut d: [u32; 14]) -> [u32; 14] {
    d[1] = 0;
    d[13] = 0;
    d
}

/// The resource for a texture placed at `address` with the resource
/// description `desc` (as words), made on first use. Returns a reference
/// taken under `PLACED_TEXTURES` so eviction cannot free it out from under
/// the caller.
pub(crate) fn texture_at(address: u64, desc: &[u32; 14]) -> Option<HeldTexture> {
    if address == 0 {
        return None;
    }
    let desc = &canonical_resource_desc(*desc);
    let frame = super::queue::presented();
    let mut map = placed_lock();
    if let Some((r, _, used)) = map.get_mut(&(address, *desc)) {
        *used = frame;
        let texture = *r as This;
        object::add_ref(texture);
        host::TEXTURE_EVENTS[host::EV_HIT].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Some(HeldTexture(texture));
    }
    // SAFETY: 56 bytes in the description's layout.
    let d = unsafe { desc.as_ptr().cast::<ResourceDesc>().read_unaligned() };
    if d.dimension == BUFFER {
        return None;
    }
    let size = footprint(&d).map_or(1, |info| info.size.max(1));
    let end = address.saturating_add(size);
    let started = std::time::Instant::now();
    let retired = {
        let mut f = PLACED_FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if f.frame != frame {
            *f = PlacedFrame { frame, released: 0, starved: false, lru_done: false };
        }
        let room = if f.starved { 0 } else { PLACED_RELEASES_PER_FRAME - f.released };
        let held = PLACED.bytes.load(std::sync::atomic::Ordering::Relaxed);
        // The over-the-cap cut (all cold textures collected and sorted)
        // once a frame, not for every texture made.
        let lru_due = !f.lru_done;
        let (retired, starved) = placed_to_retire(&map, address, end, frame, placed_limit(), held, room, lru_due);
        f.lru_done = true;
        f.released += retired.len();
        f.starved |= starved;
        retired
    };
    PLACED_COUNTS[PC_RETIRE_US].fetch_add(started.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    note_miss(address, desc);
    let retired: Vec<Retired> = retired
        .into_iter()
        .filter_map(|k| map.remove(&k).map(|(r, e, _)| (k, r, e - k.0, if k.0 < end && e > address { WHY_OVERLAP } else { WHY_CAP })))
        .collect();
    // The resource and its host copy are made without the lock (it
    // was held across creation); another thread that
    // made the same one meanwhile wins and ours goes.
    drop(map);
    release_placed(retired);
    // No per-frame cap on making placed textures (an earlier version had one): a slot
    // that found none was not looked at again while its table stood, so
    // the texture stayed black (intermittent: only when the cap bit, under
    // load; the loading-screen artwork). Pooled creation is cheap.
    let mut out = std::ptr::null();
    let creating = std::time::Instant::now();
    let made = place_at_address(address, d, std::ptr::null(), &RESOURCE_IIDS[0], &mut out);
    let made = made == S_OK && resource(out).host().is_some();
    PLACED_COUNTS[PC_CREATE_US].fetch_add(creating.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    if !made {
        if !out.is_null() {
            resource_release(out);
        }
        host::stage(host::STAGE_PLACED, started);
        return None;
    }
    let mut map = placed_lock();
    if let Some((r, _, used)) = map.get_mut(&(address, *desc)) {
        *used = frame;
        let theirs = *r as This;
        object::add_ref(theirs);
        drop(map);
        resource_release(out);
        PLACED_COUNTS[PC_RACES].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        host::stage(host::STAGE_PLACED, started);
        return Some(HeldTexture(theirs));
    }
    // One reference stays in `PLACED_TEXTURES`; the caller gets a second.
    object::add_ref(out);
    map.insert((address, *desc), (out as usize, end, frame));
    drop(map);
    LARGEST_PLACED.fetch_max(size, std::sync::atomic::Ordering::Relaxed);
    PLACED.made(size, true);
    let events = &host::TEXTURE_EVENTS;
    events[host::EV_MADE].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if let Some(why) = RELEASED_KEYS.lock().unwrap_or_else(|e| e.into_inner()).remove(&(address, *desc)) {
        events[host::EV_REMADE].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        PLACED_COUNTS[PC_REMADE + why as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    host::stage(host::STAGE_PLACED, started);
    Some(HeldTexture(out))
}

/// `PLACED_TEXTURES`, with the time waited for it counted.
fn placed_lock() -> std::sync::MutexGuard<'static, PlacedTextureMap> {
    let at = std::time::Instant::now();
    let map = super::counters::timed(super::counters::LOCK_PLACED, &PLACED_TEXTURES);
    PLACED_COUNTS[PC_LOCK_WAIT_US].fetch_add(at.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    map
}

/// Why a placed texture was released: an overlapping placement, the cap,
/// a remap of its memory.
const WHY_OVERLAP: u8 = 0;
const WHY_CAP: u8 = 1;
const WHY_REMAP: u8 = 2;

/// Counters (frame statistics, `placed_text`): microseconds waited for
/// `PLACED_TEXTURES`; host textures made committed and placed in the pool;
/// placed textures made twice at once (one dropped); placed textures made
/// again by why they had gone (overlap, cap, remap).
pub(crate) static PLACED_COUNTS: [std::sync::atomic::AtomicU64; 30] = [const { std::sync::atomic::AtomicU64::new(0) }; 30];
const PC_LOCK_WAIT_US: usize = 0;
const PC_COMMITTED: usize = 1;
const PC_POOLED: usize = 2;
const PC_RACES: usize = 3;
const PC_REMADE: usize = 4;
/// Mips dropped to fit the budget or the pool (one per step); sampled
/// textures that still had to be committed; placed textures not made this
/// frame (creation cap).
const PC_BUDGET_DROPS: usize = 7;
const PC_COMMITTED_FALLBACK: usize = 8;
const PC_DEFERRED: usize = 9;
/// Reduced copies made whole for a GPU write; title copies naming a mip
/// a host copy lacks (skipped).
const PC_REMADE_FULL: usize = 10;
const PC_COPIES_SKIPPED: usize = 11;
/// Microseconds choosing what to release before a placed texture is
/// made; misses by kind (both new, description made elsewhere before, the
/// address held another description, the same key made again).
const PC_RETIRE_US: usize = 12;
const PC_MISS_NEW: usize = 13;
const PC_MISS_MOVED: usize = 14;
const PC_MISS_REDESCRIBED: usize = 15;
const PC_MISS_SAME_KEY: usize = 16;
/// Microseconds in making the resource and its host copy.
const PC_CREATE_US: usize = 17;
/// Target host textures taken back from the free pool, put in it, and
/// let go from it unused; title placements (device 0x428) by caller.
const PC_TARGET_REUSED: usize = 18;
const PC_TARGET_FREED: usize = 19;
const PC_TARGET_DROPPED: usize = 20;
const PC_AT_TRANSIENT: usize = 21;
const PC_AT_STREAMED: usize = 22;
const PC_AT_GENERIC: usize = 23;
const PC_AT_OTHER: usize = 24;
/// Host copies of textures by content identity (origin, the
/// streamed pages behind them): taken from the origin cache, made for an
/// origin, made with no origin (contents not known to repeat).
const PC_ORIGIN_SHARED: usize = 25;
const PC_ORIGIN_MADE: usize = 26;
const PC_ORIGIN_NONE: usize = 27;
/// Misses of a description made elsewhere with none of it released since:
/// a new texture of a common description, not a move.
const PC_MISS_COMMON: usize = 28;
/// Host copies asked for when the card had no room (`room`).
const PC_NO_ROOM: usize = 29;
/// The title's callers of its CreateAtAddress wrapper (0x14193DE10),
/// as image offsets of their return addresses, and the counter of each.
/// From static reading: the render graph's
/// transient textures (Update 0x142473560 -> 0x14131A270, released and
/// placed again every frame), streamed textures, generic creation.
const PLACEMENT_CALLERS: [(usize, usize); 3] = [
    (0x131_a28d, PC_AT_TRANSIENT),
    (0x32_5146, PC_AT_STREAMED),
    (0x34_c001, PC_AT_GENERIC),
];

/// Counts a title placement by which known caller is on the stack (the
/// first few frames above create_at_address).
#[inline(never)]
fn note_placement_caller() {
    let base = object::title_base();
    let mut frames = [std::ptr::null_mut(); 6];
    // SAFETY: room for six frames; skips this function and its caller.
    let n = unsafe { xnw_core::win::RtlCaptureStackBackTrace(2, frames.len() as u32, frames.as_mut_ptr(), std::ptr::null_mut()) } as usize;
    let which = frames[..n]
        .iter()
        .filter_map(|&f| (f as usize).checked_sub(base).filter(|_| base != 0))
        .find_map(|rva| PLACEMENT_CALLERS.iter().find(|&&(at, _)| at == rva).map(|&(_, c)| c))
        .unwrap_or(PC_AT_OTHER);
    PLACED_COUNTS[which].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Host textures of render, depth and unordered-access targets the
/// title let go, kept for a target of the same description and clear value
/// (the render graph places its transients again every frame: ~400 made a
/// frame). A target is rendered or written before
/// it is read, so one taken back needs no contents: nothing is hashed or
/// uploaded for it (targets never are, 	exture::title_written). Taken
/// back only [TARGET_REUSE_AFTER] presented frames after the last list
/// holding it let go (the age host::retire waits before releasing one);
/// let go to host::retire after [TARGET_KEEP_FRAMES] unused.
type TargetKey = ([u32; 14], Option<[u32; 5]>);
type FreeTargets = std::collections::BTreeMap<TargetKey, Vec<(u64, windows::Win32::Graphics::Direct3D12::ID3D12Resource, u32)>>;
static FREE_TARGETS: std::sync::Mutex<(u64, usize, FreeTargets)> = std::sync::Mutex::new((0, 0, std::collections::BTreeMap::new()));
const TARGET_REUSE_AFTER: u64 = 4;
const TARGET_KEEP_FRAMES: u64 = 16;
/// Free targets kept at most.
const TARGET_FREE_MAX: usize = 4096;

/// The free-pool key of a target texture's host copy; None for a texture
/// that takes its contents from title memory or a pack.
fn target_key(desc: &ResourceDesc, clear: *const c_void) -> Option<TargetKey> {
    const TARGET_DEPTH_UNORDERED: u32 = 0x7;
    if desc.dimension == BUFFER || desc.flags & TARGET_DEPTH_UNORDERED == 0 {
        return None;
    }
    // SAFETY: 56 bytes in the description's layout; the title passes a whole
    // clear value or null.
    let words = unsafe { std::ptr::from_ref(desc).cast::<[u32; 14]>().read_unaligned() };
    let clear = (!clear.is_null()).then(|| unsafe { clear.cast::<[u32; 5]>().read_unaligned() });
    Some((canonical_resource_desc(words), clear))
}

/// A free target host texture for key and the state it was left in.
fn take_free_target(key: &TargetKey) -> Option<(windows::Win32::Graphics::Direct3D12::ID3D12Resource, u32)> {
    let now = super::queue::presented();
    let mut free = FREE_TARGETS.lock().unwrap_or_else(|e| e.into_inner());
    let list = free.2.get_mut(key)?;
    let (texture, state) = take_unheld(list, now)?;
    if list.is_empty() {
        free.2.remove(key);
    }
    free.1 -= 1;
    PLACED_COUNTS[PC_TARGET_REUSED].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some((texture, state))
}

/// Of a key's free targets, one old enough and held by nothing else:
/// a heap slot made from a released view, a plan or the retire list still
/// holding one would read it as another target's (device removed).
fn take_unheld(list: &mut Vec<(u64, windows::Win32::Graphics::Direct3D12::ID3D12Resource, u32)>, now: u64) -> Option<(windows::Win32::Graphics::Direct3D12::ID3D12Resource, u32)> {
    let mut held = 0;
    let at = list.iter().position(|(f, t, _)| {
        if now.saturating_sub(*f) < TARGET_REUSE_AFTER {
            return false;
        }
        // The list's own reference is the one expected.
        let alone = com_count(t) == 1;
        held += u64::from(!alone);
        alone
    });
    HELD_COUNTS[HC_POOL_HELD].fetch_add(held, std::sync::atomic::Ordering::Relaxed);
    let (_, texture, state) = list.swap_remove(at?);
    Some((texture, state))
}

/// `XNW_GPU_CHECK`: whether `texture` waits in the target free pool.
pub(crate) fn in_free_pool(texture: &windows::Win32::Graphics::Direct3D12::ID3D12Resource) -> bool {
    use windows::core::Interface;
    let free = FREE_TARGETS.lock().unwrap_or_else(|e| e.into_inner());
    free.2.values().flatten().any(|(_, t, _)| t.as_raw() == texture.as_raw())
}

/// Keeps a target host texture nobody holds for the next of its kind; once
/// a frame, those kept too long go to host::retire.
fn free_target(key: TargetKey, texture: windows::Win32::Graphics::Direct3D12::ID3D12Resource, state: u32) {
    let now = super::queue::presented();
    let mut gone = Vec::new();
    {
        let mut free = FREE_TARGETS.lock().unwrap_or_else(|e| e.into_inner());
        if free.0 != now {
            free.0 = now;
            let mut n = 0;
            free.2.retain(|_, list| {
                list.retain(|(f, t, _)| {
                    let keep = now.saturating_sub(*f) < TARGET_KEEP_FRAMES;
                    if !keep {
                        gone.push(t.clone());
                    }
                    keep
                });
                n += list.len();
                !list.is_empty()
            });
            free.1 = n;
        }
        if free.1 < TARGET_FREE_MAX {
            free.1 += 1;
            free.2.entry(key).or_default().push((now, texture, state));
            PLACED_COUNTS[PC_TARGET_FREED].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            gone.push(texture);
        }
    }
    PLACED_COUNTS[PC_TARGET_DROPPED].fetch_add(gone.len() as u64, std::sync::atomic::Ordering::Relaxed);
    for t in gone {
        host::retire(t);
    }
}

/// Free target textures kept now.
fn free_targets() -> usize {
    FREE_TARGETS.lock().unwrap_or_else(|e| e.into_inner()).1
}

/// The placed-texture part of the frame statistics, over `n` frames (per frame),
/// zeroing the counts.
pub(crate) fn placed_text(n: u64) -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let c: Vec<u64> = PLACED_COUNTS.iter().map(|c| c.swap(0, Relaxed)).collect();
    let n = n.max(1);
    let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    format!(
        "host textures: {} MiB of {} MiB budget; made per frame: {} committed ({} sampled, pool or budget full), {} in the pool ({} heaps, {} MiB, {} MiB free), {} mip steps dropped to fit, {} placed deferred (creation cap), {} reduced copies made whole for a GPU write, {} copies skipped (mip not on the host); placed: lock wait {} us per frame, {} made twice, made again by why gone: overlap {}, cap {}, remap {}; choosing releases {} us, making {} us per frame; misses per frame: {} new description, {} new of a description seen elsewhere, {} moved (one of its description released), {} address held another description, {} same key again; title placements per frame: {} transient (0x14131A28D), {} streamed (0x140325146), {} generic (0x14034C001), {} other; target textures per frame: {} taken back, {} freed, {} let go unused, {} free now; host copies by content per frame: {} shared by origin, {} made for an origin, {} with no origin; budget room per frame: {} looked ({} us), {} answered by the frame's failure; card room refused {} per frame (mips dropped instead)",
        host_texture_bytes() >> 20,
        tex_budget() >> 20,
        c[PC_COMMITTED] / n,
        c[PC_COMMITTED_FALLBACK] / n,
        c[PC_POOLED] / n,
        pool.heaps.len(),
        (pool.heaps.len() as u64 * POOL_HEAP) >> 20,
        pool.heaps.iter().map(|h| h.free.values().sum::<u64>()).sum::<u64>() >> 20,
        c[PC_BUDGET_DROPS] / n,
        c[PC_DEFERRED] / n,
        c[PC_REMADE_FULL],
        c[PC_COPIES_SKIPPED],
        c[PC_LOCK_WAIT_US] / n,
        c[PC_RACES],
        c[PC_REMADE + WHY_OVERLAP as usize] / n,
        c[PC_REMADE + WHY_CAP as usize] / n,
        c[PC_REMADE + WHY_REMAP as usize] / n,
        c[PC_RETIRE_US] / n,
        c[PC_CREATE_US] / n,
        c[PC_MISS_NEW] / n,
        c[PC_MISS_COMMON] / n,
        c[PC_MISS_MOVED] / n,
        c[PC_MISS_REDESCRIBED] / n,
        c[PC_MISS_SAME_KEY] / n,
        c[PC_AT_TRANSIENT] / n,
        c[PC_AT_STREAMED] / n,
        c[PC_AT_GENERIC] / n,
        c[PC_AT_OTHER] / n,
        c[PC_TARGET_REUSED] / n,
        c[PC_TARGET_FREED] / n,
        c[PC_TARGET_DROPPED] / n,
        free_targets(),
        c[PC_ORIGIN_SHARED] / n,
        c[PC_ORIGIN_MADE] / n,
        c[PC_ORIGIN_NONE] / n,
        super::texture::ROOM_COUNTS[0].swap(0, Relaxed) / n,
        super::texture::ROOM_COUNTS[2].swap(0, Relaxed) / n,
        super::texture::ROOM_COUNTS[1].swap(0, Relaxed) / n,
        c[PC_NO_ROOM] / n
    )
}

/// Placed textures kept at most: a ceiling far above a frame's working set;
/// the byte cap ([`placed_limit`]) is what binds. VIctory (360f382)
/// caps the count at 384 and cuts to 256, below one world frame's working
/// set: 254 made, 248 of them made again, 254 released per frame.
const MAX_PLACED: usize = 8192;
/// A placed texture used in the last this many presented frames is never
/// released for the cap or an overlap: a frame's working set stays whole
/// even when it is over the cap (which then lets more be held).
const PLACED_HOT_FRAMES: u64 = 8;
/// Cold placed textures released for the cap and overlaps per presented
/// frame at most (remaps are not bounded: their memory is gone); the rest
/// waits for later frames.
const PLACED_RELEASES_PER_FRAME: usize = 32;

/// What [`texture_at`] released in the current frame, and whether a look
/// found nothing cold to release (then it does not look again until the next
/// frame: each look sorts the map).
struct PlacedFrame {
    frame: u64,
    released: usize,
    starved: bool,
    /// The over-the-cap cut was looked for in this frame.
    lru_done: bool,
}
static PLACED_FRAME: std::sync::Mutex<PlacedFrame> = std::sync::Mutex::new(PlacedFrame { frame: u64::MAX, released: 0, starved: false, lru_done: false });

/// The largest footprint of a placed texture made so far: how far below an
/// address one can reach it.
static LARGEST_PLACED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// What a placed texture made had in common with earlier ones (why the
/// 400 a frame miss): the description last made at its address, and the
/// descriptions made anywhere lately.
struct Seen {
    at: std::collections::BTreeMap<u64, [u32; 14]>,
    descs: std::collections::BTreeSet<[u32; 14]>,
    /// Placed textures released and not yet made again, by description:
    /// a miss of one of these is a texture that moved.
    released: std::collections::BTreeMap<[u32; 14], u32>,
}
static SEEN: std::sync::Mutex<Seen> = std::sync::Mutex::new(Seen {
    at: std::collections::BTreeMap::new(),
    descs: std::collections::BTreeSet::new(),
    released: std::collections::BTreeMap::new(),
});

/// Counts why `(address, desc)` was not in the map: its address held another
/// description, its description was made at another address, or both new.
fn note_miss(address: u64, desc: &[u32; 14]) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut s = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if s.at.len() > 1 << 16 {
        s.at.clear();
    }
    if s.descs.len() > 1 << 16 {
        s.descs.clear();
    }
    if s.released.len() > 1 << 16 {
        s.released.clear();
    }
    // "seen elsewhere" alone was 97 % of a test run's misses while only 10
    // placed textures a frame were released:
    // most were new textures of a common description, not moves.
    let moved = match s.released.get_mut(desc) {
        Some(n) => {
            *n -= 1;
            if *n == 0 {
                s.released.remove(desc);
            }
            true
        }
        None => false,
    };
    let which = match s.at.insert(address, *desc) {
        Some(old) if old != *desc => PC_MISS_REDESCRIBED,
        Some(_) => PC_MISS_SAME_KEY,
        None if moved => PC_MISS_MOVED,
        None if s.descs.contains(desc) => PC_MISS_COMMON,
        None => PC_MISS_NEW,
    };
    s.descs.insert(*desc);
    PLACED_COUNTS[which].fetch_add(1, Relaxed);
}

/// The placed textures to retire before one at `[address, end)` is made in
/// `frame`, with `held` footprint bytes in the map and room for `room`
/// releases: cold ones it overlaps (the title placed a new texture over their
/// memory, another runtime's idea) and, past `limit`, the least recently used cold
/// ones down to three quarters of it (VIctory caps the count; we cap
/// bytes). Cold: not used in the last [`PLACED_HOT_FRAMES`] frames; two
/// descriptions over the same memory used close together are aliases, not a
/// replacement. Also whether the map is over `limit` with nothing cold left.
fn placed_to_retire(
    map: &PlacedTextureMap,
    address: u64,
    end: u64,
    frame: u64,
    limit: PlacedLimit,
    held: u64,
    room: usize,
    lru_due: bool,
) -> (Vec<(u64, [u32; 14])>, bool) {
    let cold = |used: u64| used + PLACED_HOT_FRAMES <= frame;
    let bytes = |k: &(u64, [u32; 14]), e: &(usize, u64, u64)| e.1 - k.0;
    let mut out: Vec<(u64, [u32; 14])> = Vec::new();
    let mut left = map.len() + 1;
    let mut held = held + (end - address);
    // Only the textures that can reach `address` (none spans more than
    // the largest footprint made): the scan started at the map's start, all
    // 12,000 textures for each of 400 made a frame.
    // (0 before any is made: the whole map, as tests build it by hand.)
    let largest = LARGEST_PLACED.load(std::sync::atomic::Ordering::Relaxed);
    let reach = if largest == 0 { 0 } else { address.saturating_sub(largest) };
    for (k, e) in map.range((reach, [0; 14])..(end, [0; 14])) {
        if out.len() >= room {
            break;
        }
        if e.1 > address && k.0 < end && cold(e.2) {
            out.push(*k);
            (left, held) = (left - 1, held.saturating_sub(bytes(k, e)));
        }
    }
    let over = |left: usize, held: u64| left > limit.count || held > limit.bytes;
    if out.len() >= room || !over(left, held) || !lru_due {
        return (out, false);
    }
    let mut cold: Vec<(u64, (u64, [u32; 14]), u64)> = map
        .iter()
        .filter(|&(k, &(_, _, used))| cold(used) && !out.contains(k))
        .map(|(k, e)| (e.2, *k, bytes(k, e)))
        .collect();
    let starved = cold.is_empty();
    cold.sort_unstable_by_key(|&(used, ..)| used);
    let (count, cut) = (limit.count / 4 * 3, limit.bytes / 4 * 3);
    for (_, k, b) in cold {
        if out.len() >= room || left <= count && held <= cut {
            break;
        }
        out.push(k);
        (left, held) = (left - 1, held.saturating_sub(b));
    }
    (out, starved)
}

/// When placed textures are cut back: past `count` textures or `bytes` of
/// footprint, down to three quarters of both.
#[derive(Clone, Copy)]
struct PlacedLimit {
    count: usize,
    bytes: u64,
}

/// The placed-texture cap: `XNW_PLACED_MB` MiB of console footprint (512
/// by default) and [`MAX_PLACED`] textures.
fn placed_limit() -> PlacedLimit {
    static LIMIT: std::sync::LazyLock<PlacedLimit> = std::sync::LazyLock::new(|| {
        let set = std::env::var("XNW_PLACED_MB").ok().and_then(|v| v.trim().parse::<u64>().ok());
        if let Some(mb) = set {
            xnw_core::info!("d3d12x: XNW_PLACED_MB: cold placed textures are released above {mb} MiB");
        }
        PlacedLimit { count: MAX_PLACED, bytes: set.unwrap_or(512) << 20 }
    });
    *LIMIT
}

/// A placed texture taken out of `PLACED_TEXTURES`: key, object,
/// footprint bytes, why (`WHY_*`).
type Retired = ((u64, [u32; 14]), usize, u64, u8);

/// Keys of placed textures released, to count those made again (the frame
/// statistics); forgotten in bulk past 16,384.
static RELEASED_KEYS: std::sync::Mutex<std::collections::BTreeMap<(u64, [u32; 14]), u8>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// Releases placed textures taken out of `PLACED_TEXTURES`, after its lock
/// is let go: lists and holders keep theirs alive, the host copy goes
/// through [`host::retire`].
fn release_placed(retired: Vec<Retired>) {
    use std::sync::atomic::Ordering::Relaxed;
    if retired.is_empty() {
        return;
    }
    PLACED_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Release);
    PLACED_RETIRED.fetch_add(retired.len() as u64, Relaxed);
    host::TEXTURE_EVENTS[host::EV_RELEASED].fetch_add(retired.len() as u64, Relaxed);
    {
        let mut keys = RELEASED_KEYS.lock().unwrap_or_else(|e| e.into_inner());
        if keys.len() > 16_384 {
            keys.clear();
        }
        keys.extend(retired.iter().map(|&(k, _, _, why)| (k, why)));
    }
    {
        let mut s = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        for ((_, desc), ..) in &retired {
            *s.released.entry(*desc).or_default() += 1;
        }
    }
    for (_, res, bytes, _) in retired {
        PLACED.made(bytes, false);
        resource_release(res as This);
    }
}

/// Placed textures released (overlap, cap, remap) so far (memory line).
static PLACED_RETIRED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Releases placed textures whose backing range overlaps `[start, end)`.
pub(crate) fn evict_placed_textures(start: u64, end: u64) {
    if start >= end {
        return;
    }
    let mut evicted = Vec::new();
    {
        let mut map = super::counters::timed(super::counters::LOCK_PLACED, &PLACED_TEXTURES);
        // Only textures that can reach `start` (none spans more than the
        // largest made), not a walk of the whole map at every remapping.
        let largest = LARGEST_PLACED.load(std::sync::atomic::Ordering::Relaxed);
        let reach = if largest == 0 { 0 } else { start.saturating_sub(largest) };
        let hit: Vec<(u64, [u32; 14])> =
            map.range((reach, [0; 14])..(end, [0; 14])).filter(|&(k, e)| k.0 < end && e.1 > start).map(|(k, _)| *k).collect();
        for k in hit {
            if let Some((res, tex_end, _)) = map.remove(&k) {
                evicted.push((k, res, tex_end - k.0, WHY_REMAP));
            }
        }
    }
    release_placed(evicted);
}

/// Whether a placed texture is made for `(address, desc)` (bindless
/// slots wait for the texture thread rather than make it).
pub(crate) fn has_placed_texture(address: u64, desc: &[u32; 14]) -> bool {
    PLACED_TEXTURES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .contains_key(&(address, canonical_resource_desc(*desc)))
}

pub(crate) extern "system" fn create_placed(
    _: This,
    heap: This,
    offset: u64,
    desc: *const ResourceDesc,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if heap.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title passes one of our heaps.
    let h = unsafe { &*heap.cast::<Heap>() };
    let fits = footprint(&desc).is_some_and(|info| {
        offset.is_multiple_of(info.alignment)
            && offset
                .checked_add(info.size)
                .is_some_and(|end| end <= h.desc.size)
    });
    if !fits {
        xnw_core::warn!("d3d12x: placed resource {desc:?} at {offset:#x} does not fit a heap of {:#x}", h.desc.size);
        return E_INVALIDARG;
    }
    object::add_ref(heap);
    let backing = Backing::Placed { heap, offset };
    hand_out_resource(
        desc,
        h.desc.properties,
        h.desc.flags,
        backing,
        state,
        clear,
        iid,
        out,
    )
}

extern "system" fn resource_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        RESOURCE,
        iid,
        object::is(iid, &RESOURCE_IIDS).then_some(this),
        out,
    )
}
/// The last release lets the object go at once and frees its memory
/// later ([`QUARANTINE`]).
extern "system" fn resource_release(this: This) -> u32 {
    match object::count_down(this) {
        Some(0) => {
            object::drop_private(this);
            resource(this).let_go();
            let seq = SUBMIT_SEQ.load(std::sync::atomic::Ordering::Relaxed);
            QUARANTINE.lock().unwrap_or_else(|e| e.into_inner()).push_back((seq, std::time::Instant::now(), this as usize));
            0
        }
        Some(n) => n,
        None => 0,
    }
}
/// Memory is always mapped (subresources are not told apart yet); a null
/// `out` only asks for residency.
extern "system" fn map(this: This, _sub: u32, _read: *const c_void, out: *mut *mut u8) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a pointer.
        unsafe { out.write(resource(this).data()) };
    }
    S_OK
}
extern "system" fn unmap(_: This, _sub: u32, _written: *const c_void) {}
/// Returns its struct through a hidden pointer after `this`.
extern "system" fn resource_desc(this: This, out: *mut ResourceDesc) -> *mut ResourceDesc {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(resource(this).desc) };
    }
    out
}

/// Device method 0xd0, a struct returned through a hidden first argument:
/// a standard heap type as a custom heap. Memory is unified and
/// cache-coherent: pool L0, default heaps not CPU-visible, others write-back.
pub(crate) extern "system" fn custom_heap_properties(a: *mut c_void, b: *mut c_void, nodes: u32, kind: u32) -> *mut HeapProperties {
    let out = return_slot::<HeapProperties>(a, b);
    const CUSTOM: u32 = 4;
    const NOT_AVAILABLE: u32 = 1;
    const WRITE_BACK: u32 = 3;
    const L0: u32 = 1;
    let nodes = nodes.max(1);
    let cpu_pages = match kind {
        1 => NOT_AVAILABLE,
        2 | 3 => WRITE_BACK,
        _ => 0,
    };
    let properties = HeapProperties { kind: CUSTOM, cpu_pages, pool: if cpu_pages == 0 { 0 } else { L0 }, creation_nodes: nodes, visible_nodes: nodes };
    // SAFETY: the title passes room for the result (null is tolerated).
    if let Some(out) = unsafe { out.as_mut() } {
        *out = properties;
    }
    out
}

extern "system" fn gpu_address(this: This) -> u64 {
    resource(this).data() as u64
}
extern "system" fn heap_properties(
    this: This,
    properties: *mut HeapProperties,
    flags: *mut u32,
) -> HRESULT {
    let r = resource(this);
    // SAFETY: either pointer may be null; otherwise the title passes room.
    unsafe {
        if let Some(p) = properties.as_mut() {
            *p = r.heap;
        }
        if let Some(f) = flags.as_mut() {
            *f = r.heap_flags;
        }
    }
    S_OK
}

/// IUnknown 3, object 4, device child 1, heap 1.
pub(crate) const HEAP_SLOTS: &[Slot] = &[
    (0, heap_query as *const (), Done, "resource.rs:1330"),
    (2, heap_release as *const (), Done, "resource.rs:1331"),
    (7, object::parent_device as *const (), Done, "resource.rs:1332"),
    (8, heap_desc as *const (), Done, "resource.rs:1333"),
];
pub(crate) const HEAP_LEN: usize = 24;
static HEAP_FUNCTIONS: Table<HEAP_LEN> = object::table::<HEAP, HEAP_LEN>(true, HEAP_SLOTS);

/// IUnknown 3, object 4, device child 1, resource 10 (7 as on PC, then 3
/// console ones).
pub(crate) const RESOURCE_SLOTS: &[Slot] = &[
    (0, resource_query as *const (), Done, "resource.rs:1341"),
    (2, resource_release as *const (), Done, "resource.rs:1342"),
    (7, object::parent_device as *const (), Done, "resource.rs:1343"),
    (8, map as *const (), Done, "resource.rs:1344"),
    (9, unmap as *const (), Done, "resource.rs:1345"),
    (10, resource_desc as *const (), Done, "resource.rs:1346"),
    (11, gpu_address as *const (), Done, "resource.rs:1347"),
    (14, heap_properties as *const (), Done, "resource.rs:1348"),
];
pub(crate) const RESOURCE_LEN: usize = 32;
static RESOURCE_FUNCTIONS: Table<RESOURCE_LEN> = object::table::<RESOURCE, RESOURCE_LEN>(true, RESOURCE_SLOTS);

/// What `PresentX` needs of a plane's resource: its description, heap
/// flags and memory (mip 0 first, rows packed). `None` unless `this` is one
/// of our resources.
pub(crate) fn displayed(this: This) -> Option<(ResourceDesc, u32, *const u8)> {
    if this.is_null() {
        return None;
    }
    // SAFETY: the title passes an object pointer; its first word is the
    // method table, which tells whether it is a resource of ours.
    let table = unsafe { *this.cast::<*const *const ()>() };
    if table != RESOURCE_FUNCTIONS.0.as_ptr() {
        return None;
    }
    let r = resource(this);
    Some((r.desc, r.heap_flags, r.data()))
}

/// Host textures alive and their bytes on the card, for the memory line.
struct Census {
    count: std::sync::atomic::AtomicU64,
    bytes: std::sync::atomic::AtomicU64,
}

static HOST_TEXTURES: Census = Census { count: std::sync::atomic::AtomicU64::new(0), bytes: std::sync::atomic::AtomicU64::new(0) };

impl Census {
    fn count(&self, texture: &windows::Win32::Graphics::Direct3D12::ID3D12Resource, made: bool) {
        use std::sync::atomic::Ordering::Relaxed;
        let Some(host) = host::host() else {
            return;
        };
        // SAFETY: plain queries on live objects.
        let bytes = unsafe { host.device.GetResourceAllocationInfo(0, &[texture.GetDesc()]) }.SizeInBytes;
        if made {
            self.count.fetch_add(1, Relaxed);
            self.bytes.fetch_add(bytes, Relaxed);
        } else {
            self.count.fetch_sub(1, Relaxed);
            self.bytes.fetch_sub(bytes, Relaxed);
        }
    }
}

/// Placed textures alive (console footprint bytes), and how many were
/// evicted for video memory, for the memory line. Atomics, not the map's
/// lock: the line is also written from the crash handler.
struct Placed {
    count: std::sync::atomic::AtomicU64,
    bytes: std::sync::atomic::AtomicU64,
}

static PLACED: Placed = Placed { count: std::sync::atomic::AtomicU64::new(0), bytes: std::sync::atomic::AtomicU64::new(0) };

impl Placed {
    fn made(&self, bytes: u64, made: bool) {
        use std::sync::atomic::Ordering::Relaxed;
        if made {
            self.count.fetch_add(1, Relaxed);
            self.bytes.fetch_add(bytes, Relaxed);
        } else {
            self.count.fetch_sub(1, Relaxed);
            self.bytes.fetch_sub(bytes, Relaxed);
        }
    }
}

/// Host textures whose contents come from title memory (no render target,
/// depth or unordered access), the ones [`evict_over_budget`] may evict, by
/// host resource pointer. An `Arc` taken from it is dropped only after the
/// lock is let go (the last reference's drop may want it).
type Evictable = std::collections::HashMap<usize, std::sync::Weak<HostTexture>>;
static EVICTABLE: std::sync::Mutex<Option<Evictable>> = std::sync::Mutex::new(None);

/// Video memory residency, for the memory line: textures evicted now and
/// their bytes, evictions and returns so far, the peak local use (MiB).
static RESIDENCY: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];
const EVICTED_NOW: usize = 0;
const EVICTED_BYTES: usize = 1;
const EVICTIONS: usize = 2;
const RETURNS: usize = 3;
const PEAK_MIB: usize = 4;

/// The local video memory limit in bytes: `XNW_VRAM_LIMIT_MB`, else 85 %
/// of the budget DXGI gives the process now (it moves with what other
/// processes use).
fn vram_limit(budget: u64) -> u64 {
    static SET: std::sync::LazyLock<Option<u64>> = std::sync::LazyLock::new(|| {
        let mb = std::env::var("XNW_VRAM_LIMIT_MB").ok()?.trim().parse::<u64>().ok()?;
        xnw_core::info!("d3d12x: XNW_VRAM_LIMIT_MB: host textures are evicted above {mb} MiB of video memory");
        Some(mb << 20)
    });
    SET.unwrap_or(budget / 100 * 85)
}

/// At a submission, before its lists run: the host textures they use are
/// marked used in this frame, and any evicted one is made resident again.
pub fn resident<'a>(copies: impl Iterator<Item = &'a std::sync::Arc<HostTexture>>) {
    use std::sync::atomic::Ordering::Relaxed;
    let frame = super::queue::presented();
    let mut back = Vec::new();
    for copy in copies {
        copy.used.store(frame, Relaxed);
        if copy.evicted.swap(false, Relaxed) {
            back.push(copy.texture.clone());
        }
    }
    if back.is_empty() {
        return;
    }
    let Some(host) = host::host() else { return };
    let bytes = allocation_bytes(back.iter().cloned());
    let back: Vec<_> = back.iter().map(|t| windows::core::Interface::cast(t).ok()).collect();
    set_priority(host, &back, windows::Win32::Graphics::Direct3D12::D3D12_RESIDENCY_PRIORITY_NORMAL);
    RESIDENCY[EVICTED_NOW].fetch_sub(back.len() as u64, Relaxed);
    RESIDENCY[EVICTED_BYTES].fetch_sub(bytes, Relaxed);
    RESIDENCY[RETURNS].fetch_add(back.len() as u64, Relaxed);
}

/// At a submission, for host resources a bindless heap's slots view (not
/// transitioned per list): as [`resident`].
pub(crate) fn resident_views<'a>(views: impl Iterator<Item = &'a windows::Win32::Graphics::Direct3D12::ID3D12Resource>) {
    let copies: Vec<_> = {
        let all = EVICTABLE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(all) = all.as_ref() else { return };
        views.filter_map(|v| all.get(&(windows::core::Interface::as_raw(v) as usize))?.upgrade()).collect()
    };
    resident(copies.iter());
}

/// Sets the residency priority of `objects`; false if the device cannot.
fn set_priority(host: &host::Host, objects: &[Option<windows::Win32::Graphics::Direct3D12::ID3D12Pageable>], priority: windows::Win32::Graphics::Direct3D12::D3D12_RESIDENCY_PRIORITY) -> bool {
    let Ok(device) = windows::core::Interface::cast::<windows::Win32::Graphics::Direct3D12::ID3D12Device1>(&host.device) else {
        return false;
    };
    let priorities = vec![priority; objects.len()];
    // SAFETY: live resources, one priority each.
    match unsafe { device.SetResidencyPriority(objects.len() as u32, objects.as_ptr(), priorities.as_ptr()) } {
        Ok(()) => true,
        Err(e) => {
            xnw_core::warn!("d3d12x: residency priority of {} host textures not set: {e}", objects.len());
            false
        }
    }
}

/// The video memory `textures` take.
fn allocation_bytes(textures: impl Iterator<Item = windows::Win32::Graphics::Direct3D12::ID3D12Resource>) -> u64 {
    let Some(host) = host::host() else { return 0 };
    // SAFETY: plain queries on live objects.
    textures.map(|t| unsafe { host.device.GetResourceAllocationInfo(0, &[t.GetDesc()]) }.SizeInBytes).sum()
}

/// Whether local video memory use is over its limit now (the preload
/// stops there).
pub(crate) fn vram_over() -> bool {
    use windows::Win32::Graphics::Dxgi::DXGI_MEMORY_SEGMENT_GROUP_LOCAL;
    let Some(a) = adapter() else { return false };
    let mut info = windows::Win32::Graphics::Dxgi::DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    // SAFETY: a plain query into a local.
    if unsafe { a.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }.is_err() {
        return false;
    }
    info.CurrentUsage > vram_limit(info.Budget)
}

/// At every `PresentX`: when the host adapter's local video memory use is
/// over its limit ([`vram_limit`]), evicts host textures taken from title
/// memory, least recently used first, that no list used in this frame or
/// the one before, until the use is 5 % of the budget under the limit.
/// "Evicted" is the lowest residency priority, not `ID3D12Device::Evict`:
/// the system moves those out of video memory first and the GPU may still
/// read them (run 0002: with `Evict`, a texture used through a path that
/// does not make it resident again hung the device at 107 s). A list that
/// uses one again sets it back to normal ([`resident`]). After evicting,
/// waits 8 frames for the use to fall before looking again.
fn evict_over_budget(frame: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    use windows::Win32::Graphics::Dxgi::DXGI_MEMORY_SEGMENT_GROUP_LOCAL;
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let (Some(a), Some(host)) = (adapter(), host::host()) else { return };
    let mut info = windows::Win32::Graphics::Dxgi::DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    // SAFETY: a plain query into a local.
    if unsafe { a.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }.is_err() {
        return;
    }
    RESIDENCY[PEAK_MIB].fetch_max(info.CurrentUsage >> 20, Relaxed);
    let limit = vram_limit(info.Budget);
    if info.CurrentUsage <= limit || frame < NEXT.load(Relaxed) {
        return;
    }
    let want = info.CurrentUsage - limit + info.Budget / 20;
    let (mut candidates, mut live) = (Vec::new(), Vec::new());
    {
        let mut all = EVICTABLE.lock().unwrap_or_else(|e| e.into_inner());
        all.get_or_insert_with(Default::default).retain(|_, w| match w.upgrade() {
            Some(copy) => {
                let used = copy.used.load(Relaxed);
                if used + 1 < frame && !copy.evicted.load(Relaxed) {
                    candidates.push((used, copy));
                } else {
                    live.push(copy);
                }
                true
            }
            None => false,
        });
    }
    drop(live);
    candidates.sort_by_key(|&(used, _)| used);
    let (mut evict, mut bytes) = (Vec::new(), 0u64);
    for (_, copy) in candidates {
        if bytes >= want {
            break;
        }
        bytes += allocation_bytes(std::iter::once(copy.texture.clone()));
        copy.evicted.store(true, Relaxed);
        evict.push(windows::core::Interface::cast(&copy.texture).ok());
    }
    if evict.is_empty() {
        return;
    }
    if !set_priority(host, &evict, windows::Win32::Graphics::Direct3D12::D3D12_RESIDENCY_PRIORITY_MINIMUM) {
        return;
    }
    RESIDENCY[EVICTED_NOW].fetch_add(evict.len() as u64, Relaxed);
    RESIDENCY[EVICTED_BYTES].fetch_add(bytes, Relaxed);
    RESIDENCY[EVICTIONS].fetch_add(evict.len() as u64, Relaxed);
    NEXT.store(frame + 8, Relaxed);
    xnw_core::info!(
        "d3d12x: video memory {} of {} MiB (limit {}): demoted {} host textures ({} MiB) not used since frame {}",
        info.CurrentUsage >> 20,
        info.Budget >> 20,
        limit >> 20,
        evict.len(),
        bytes >> 20,
        frame - 1
    );
}

/// Writes the memory line now, once per `why` kind (process exit, a
/// reported exception).
pub(crate) fn log_memory(why: &str) {
    xnw_core::info!("{} ({why})", memory_line(super::queue::presented()));
}

/// Called at every `PresentX` (`frame.rs`): every 256 frames, one line of
/// memory use: the process's commit and working set, the host adapter's
/// local (on the card) and non-local video memory with their budgets, and
/// our host textures and host copies of title memory.
pub(crate) fn frame_presented(frame: u64) {
    evict_over_budget(frame);
    super::texture::sweep(frame);
    if frame == 0 || !frame.is_multiple_of(256) {
        return;
    }
    xnw_core::info!("{}", memory_line(frame));
}

fn memory_line(frame: u64) -> String {
    use std::sync::atomic::Ordering::Relaxed;
    const MIB: u64 = 1 << 20;
    let mut counters = xnw_core::win::ProcessMemoryCounters { cb: size_of::<xnw_core::win::ProcessMemoryCounters>() as u32, ..Default::default() };
    // SAFETY: a counters block of the size passed.
    unsafe { xnw_core::win::K32GetProcessMemoryInfo(xnw_core::win::GetCurrentProcess(), &mut counters, counters.cb) };
    let video = |group| {
        let a = adapter()?;
        let mut info = windows::Win32::Graphics::Dxgi::DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: a plain query into a local.
        unsafe { a.QueryVideoMemoryInfo(0, group, &mut info) }.ok()?;
        Some(format!("{} of {} MiB", info.CurrentUsage / MIB, info.Budget / MIB))
    };
    use windows::Win32::Graphics::Dxgi::{DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL};
    let unknown = || "unknown".to_string();
    format!(
        "memory at frame {frame}: commit {} MiB, working set {} MiB; video memory local {}, non-local {}; host textures {} ({} MiB), placed {} ({} MiB, {} released), texture objects made {} released {}, staging {} MiB, demoted for video memory {} ({} MiB; {} demotions, {} returns, peak local {} MiB); title memory copies {} MiB; views of released resources made null {}, last references put off {}, heap slots of released views made null {}, free targets held elsewhere (not taken back) {}, slots over released shared windows made null {}, stale slots nulled by XNW_GPU_CHECK {}; host creations failed so far: {}",
        counters.pagefile_usage as u64 / MIB,
        counters.working_set_size as u64 / MIB,
        video(DXGI_MEMORY_SEGMENT_GROUP_LOCAL).unwrap_or_else(unknown),
        video(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL).unwrap_or_else(unknown),
        HOST_TEXTURES.count.load(Relaxed),
        HOST_TEXTURES.bytes.load(Relaxed) / MIB,
        PLACED.count.load(Relaxed),
        PLACED.bytes.load(Relaxed) / MIB,
        PLACED_RETIRED.load(Relaxed),
        TITLE_TEXTURES[0].load(Relaxed),
        TITLE_TEXTURES[1].load(Relaxed),
        super::texture::staging_bytes() / MIB,
        RESIDENCY[EVICTED_NOW].load(Relaxed),
        RESIDENCY[EVICTED_BYTES].load(Relaxed) / MIB,
        RESIDENCY[EVICTIONS].load(Relaxed),
        RESIDENCY[RETURNS].load(Relaxed),
        RESIDENCY[PEAK_MIB].load(Relaxed),
        super::shadow::held() / MIB,
        HELD_COUNTS[HC_STALE_NULLED].load(Relaxed),
        HELD_COUNTS[HC_PUT_OFF].load(Relaxed),
        HELD_COUNTS[HC_SLOTS_NULLED].load(Relaxed),
        HELD_COUNTS[HC_POOL_HELD].load(Relaxed),
        HELD_COUNTS[HC_WINDOW_NULLED].load(Relaxed),
        HELD_COUNTS[HC_CHECK_NULLED].load(Relaxed),
        xnw_core::syscalls::failed_text()
    )
}

/// The host device's adapter as DXGI 1.4 (video memory queries).
fn adapter() -> Option<&'static windows::Win32::Graphics::Dxgi::IDXGIAdapter3> {
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter3, IDXGIFactory4};
    struct Shared(IDXGIAdapter3);
    // SAFETY: DXGI adapters are free-threaded.
    unsafe impl Send for Shared {}
    // SAFETY: as above.
    unsafe impl Sync for Shared {}
    static ADAPTER: std::sync::OnceLock<Option<Shared>> = std::sync::OnceLock::new();
    ADAPTER
        .get_or_init(|| {
            let device = &host::host()?.device;
            // SAFETY: plain COM calls on a live device and a new factory.
            unsafe {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().ok()?;
                factory.EnumAdapterByLuid(device.GetAdapterLuid()).ok().map(Shared)
            }
        })
        .as_ref()
        .map(|a| &a.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The slot at `offset` of one of our objects (tests know the signatures).
    fn slot<F: Copy>(object: This, offset: usize) -> F {
        // SAFETY: every caller here names the slot's signature.
        unsafe { super::super::object::slot(object, offset) }
    }

    type Release = extern "system" fn(This) -> u32;
    type Map = extern "system" fn(This, u32, *const c_void, *mut *mut u8) -> HRESULT;
    type Address = extern "system" fn(This) -> u64;

    const PROPERTIES: HeapProperties = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 0,
        visible_nodes: 0,
    };

    fn buffer(width: u64) -> ResourceDesc {
        ResourceDesc {
            dimension: BUFFER,
            alignment: 0,
            width,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 0,
            sample_count: 1,
            sample_quality: 0,
            layout: 1,
            flags: 0,
        }
    }

    fn texture(width: u64, height: u32, format: u32, mips: u16) -> ResourceDesc {
        ResourceDesc {
            dimension: 3,
            width,
            height,
            format,
            mip_levels: mips,
            layout: 0,
            ..buffer(0)
        }
    }

    fn heap(size: u64) -> This {
        let desc = HeapDesc {
            size,
            properties: PROPERTIES,
            alignment: 0,
            flags: 0,
        };
        let mut heap = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &desc, &HEAP_IIDS[0], &mut heap),
            S_OK
        );
        heap
    }

    /// The title compresses on the GPU, copying R32G32B32A32
    /// blocks into each mip of a BC3 texture. A host copy reduced to its
    /// three coarsest mips is made whole before the first copy, every copy
    /// names a subresource the host copy has, and the list closes (run with
    /// `XNW_GPU_DEBUG=layer` for the validation layer's word on it).
    #[test]
    fn copies_into_a_reduced_texture_record_valid_subresources() {
        let Some(_) = host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        let make = |desc: &ResourceDesc| {
            let mut out = null;
            assert_eq!(create_committed(null, &PROPERTIES, 0, desc, 0, std::ptr::null(), &RESOURCE_IIDS[0], &mut out), S_OK);
            out
        };
        let bc3 = make(&texture(1024, 1024, 77, 11));
        // Reduced as the budget or a pack would leave a `.ttpf` copy: 4x4, 3 mips.
        let r = resource(bc3);
        let reduced = host_copy_made(&r.desc, 0, std::ptr::null(), None, 8).unwrap();
        assert!(r.host.set(Some(reduced)).is_ok());
        assert_eq!(host_subresource(bc3, 9), Some(1));
        let mut allocator = null;
        assert_eq!(super::super::list::create_allocator(null, 0, &super::super::list::ALLOCATOR_IIDS[0], &mut allocator), S_OK);
        let mut list = null;
        assert_eq!(super::super::list::create_list(null, 0, 0, allocator, null, &super::super::list::LIST_IIDS[1], &mut list), S_OK);
        #[repr(C)]
        struct Location {
            resource: This,
            kind: u32,
            index: u64,
            rest: [u32; 6],
        }
        type Copy = extern "system" fn(This, *const Location, u32, u32, u32, *const Location, *const [u32; 6]);
        let skipped = PLACED_COUNTS[PC_COPIES_SKIPPED].load(std::sync::atomic::Ordering::Relaxed);
        let mut sources = Vec::new();
        for m in 0..11u32 {
            let blocks = (1024u32 >> m).max(4) / 4;
            let source = make(&texture(u64::from(blocks), blocks, 2, 1));
            let to = Location { resource: bc3, kind: 0, index: u64::from(m), rest: [0; 6] };
            let from = Location { resource: source, kind: 0, index: 0, rest: [0; 6] };
            let region = [0, 0, 0, blocks, blocks, 1];
            slot::<Copy>(list, 0x80)(list, &to, 0, 0, 0, &from, &region);
            sources.push(source);
        }
        assert_eq!(slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list), S_OK);
        // SAFETY: a live texture.
        assert_eq!(unsafe { host_resource(bc3).unwrap().GetDesc() }.MipLevels, 11, "made whole for the copies");
        assert_eq!(PLACED_COUNTS[PC_COPIES_SKIPPED].load(std::sync::atomic::Ordering::Relaxed), skipped, "no copy skipped");
        for t in sources.into_iter().chain([bc3, list, allocator]) {
            slot::<Release>(t, 0x10)(t);
        }
    }

    /// Title subresources map to the host copy's past dropped mips;
    /// a dropped mip has none.
    #[test]
    fn subresources_map_past_dropped_mips() {
        let d = ResourceDesc { dimension: 3, width: 1024, height: 1024, depth_or_array_size: 2, mip_levels: 11, format: 77, sample_count: 1, ..Default::default() };
        assert_eq!(map_subresource(&d, 0, 13), Some(13));
        assert_eq!(map_subresource(&d, 8, 7), None, "mip 7 dropped");
        assert_eq!(map_subresource(&d, 8, 8), Some(0));
        assert_eq!(map_subresource(&d, 8, 10), Some(2));
        assert_eq!(map_subresource(&d, 8, 11 + 9), Some(3 + 1), "slice 1 mip 9");
    }

    /// Pool ranges are taken first fit at their alignment and merge
    /// back with their neighbours.
    #[test]
    fn pool_ranges_take_and_merge() {
        let mut free = std::collections::BTreeMap::new();
        free.insert(0, 0x10_0000);
        let a = pool_take(&mut free, 0x1000, 0x1000).unwrap();
        let b = pool_take(&mut free, 0x2_0000, 0x1_0000).unwrap();
        let c = pool_take(&mut free, 0x1000, 0x1000).unwrap();
        assert_eq!((a, b, c), (0, 0x1_0000, 0x1000));
        assert!(pool_take(&mut free, 0x10_0000, 0x1000).is_none());
        for (at, size) in [(a, 0x1000), (c, 0x1000), (b, 0x2_0000)] {
            pool_free(&mut free, at, size);
        }
        assert_eq!(free.into_iter().collect::<Vec<_>>(), vec![(0, 0x10_0000)]);
    }

    /// A sampled-only host texture is placed in the pool; a target is
    /// committed.
    #[test]
    fn sampled_host_textures_are_pooled() {
        if host::host().is_none() {
            return; // no host GPU here
        }
        let sampled = ResourceDesc { dimension: 3, width: 64, height: 64, depth_or_array_size: 1, mip_levels: 1, format: 28, sample_count: 1, ..Default::default() };
        let (_, range) = host_texture(&sampled, 0, std::ptr::null(), 0, false).unwrap();
        assert!(range.is_some());
        let target = ResourceDesc { flags: 1, ..sampled };
        let (_, range) = host_texture(&target, 0, std::ptr::null(), 0, false).unwrap();
        assert!(range.is_none());
    }

    #[test]
    fn heaps_are_zeroed_aligned_memory() {
        let heap = heap(1 << 20);
        // SAFETY: a heap we just made.
        let memory = unsafe { &(*heap.cast::<Heap>()).memory };
        assert_eq!(memory.ptr.as_ptr() as u64 % KIB_64, 0);
        let mut back = std::mem::MaybeUninit::<HeapDesc>::uninit();
        slot::<extern "system" fn(This, *mut HeapDesc)>(heap, 0x40)(heap, back.as_mut_ptr());
        // SAFETY: written above.
        assert_eq!(unsafe { back.assume_init() }.size, 1 << 20);
        assert_eq!(slot::<Release>(heap, 0x10)(heap), 0);

        let bad = HeapDesc {
            size: 1 << 20,
            properties: PROPERTIES,
            alignment: 3,
            flags: 0,
        };
        let mut out = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &bad, &HEAP_IIDS[0], &mut out),
            E_INVALIDARG
        );
    }

    #[test]
    fn device_setup_heap_properties_and_resources_at_addresses() {
        let device = super::super::device::device();
        // GetCustomHeapProperties (0xd0): the struct comes back through the
        // hidden first argument, which is also returned.
        type Custom =
            extern "system" fn(*mut HeapProperties, This, u32, u32) -> *mut HeapProperties;
        let mut p = PROPERTIES;
        let back = slot::<Custom>(device, 0xd0)(&mut p, device, 1, 2);
        assert_eq!(back, &raw mut p);
        assert_eq!(
            (
                p.kind,
                p.cpu_pages,
                p.pool,
                p.creation_nodes,
                p.visible_nodes
            ),
            (4, 3, 1, 1, 1),
            "upload: custom, write-back, L0"
        );
        slot::<Custom>(device, 0xd0)(&mut p, device, 1, 1);
        assert_eq!(p.cpu_pages, 1, "default: not CPU-visible");

        // 0x428: a buffer over memory the title owns, as the title asks.
        type At = extern "system" fn(
            This,
            u64,
            *const ResourceDesc,
            u32,
            *const c_void,
            *const Iid,
            *mut This,
        ) -> HRESULT;
        let memory = vec![0u8; 0x1_0000];
        let address = memory.as_ptr() as u64;
        let mut r = std::ptr::null();
        let create = slot::<At>(device, 0x428);
        let desc = buffer(0x1_0000);
        let null = std::ptr::null();
        assert_eq!(
            create(
                device,
                address,
                &desc,
                u32::MAX,
                null,
                &RESOURCE_IIDS[0],
                &mut r
            ),
            0
        );
        assert_eq!(slot::<Address>(r, 0x58)(r), address);
        assert_eq!(slot::<Release>(r, 0x10)(r), 0);
        // The title's memory is still there after the release.
        assert_eq!(memory[0xffff], 0);
        assert_ne!(
            create(device, 0, &desc, u32::MAX, null, &RESOURCE_IIDS[0], &mut r),
            0
        );
    }

    #[test]
    fn sizes_cover_every_mip() {
        let size = |d: &ResourceDesc| footprint(d).map(|i| i.size);
        assert_eq!(size(&buffer(256)), Some(KIB_64));
        // Textures take their console layout and its alignment (4 KiB
        // standard swizzle when the description leaves it to the library):
        // 512x512 BC7 with 10 mips is 0x56000 bytes, 4 KiB aligned, as on
        // the console.
        let layout = |d: &ResourceDesc| texture_layout(d).unwrap().size;
        let rgba = texture(256, 256, 28, 0);
        assert_eq!(size(&rgba), Some(layout(&rgba).next_multiple_of(4096)));
        let bc7 = footprint(&texture(512, 512, 98, 10));
        let console = AllocationInfo {
            size: 0x56000,
            alignment: 0x1000,
        };
        assert_eq!(bc7, Some(console));
        assert!(layout(&rgba) >= 4 * (256 * 256 * 4) / 3, "every mip");
        // BC1 is 8 bytes per 4x4 block: 1024x1024, one mip = 512 KiB.
        assert_eq!(size(&texture(1024, 1024, 71, 1)), Some(512 << 10));
        assert_eq!(size(&texture(16, 16, 66, 1)), None, "1-bit format");
        // Linear layout (1) and an explicit mode (64 KiB display, 0x10a).
        let linear = ResourceDesc {
            layout: 1,
            ..texture(100, 60, 28, 1)
        };
        assert_eq!(
            texture_layout(&linear).unwrap().planes[0]
                .layout
                .desc
                .swizzle
                .0,
            0
        );
        let display = ResourceDesc {
            layout: 0x10a,
            ..texture(100, 60, 28, 1)
        };
        assert_eq!(
            texture_layout(&display).unwrap().planes[0]
                .layout
                .desc
                .swizzle
                .0,
            10
        );

        let mut info = AllocationInfo {
            size: 0,
            alignment: 0,
        };
        let descs = [buffer(100), texture(256, 256, 28, 0)];
        allocation_info((&raw mut info).cast(), std::ptr::null_mut(), 0, 2, descs.as_ptr());
        assert_eq!(
            info,
            AllocationInfo {
                size: KIB_64 + size(&descs[1]).unwrap(),
                alignment: KIB_64
            }
        );
        allocation_info((&raw mut info).cast(), std::ptr::null_mut(), 0, 1, &texture(1, 1, 999, 1));
        assert_eq!(info.size, u64::MAX);
    }

    #[test]
    fn upload_footprints_follow_the_pc_rules() {
        let null = std::ptr::null();
        let mut layouts = [NO_FOOTPRINT; 2];
        let (mut rows, mut sizes, mut total) = ([0; 2], [0; 2], 0);
        let mut footprints = |desc: &ResourceDesc, first, count| {
            let (l, r, s) = (layouts.as_mut_ptr(), rows.as_mut_ptr(), sizes.as_mut_ptr());
            copyable_footprints(null, desc, first, count, 0, l, r, s, &mut total);
            (layouts, rows, sizes, total)
        };
        // 100x60 RGBA8, 2 mips: 400-byte rows pitched to 512.
        let (l, r, s, t) = footprints(&texture(100, 60, 28, 2), 0, 2);
        assert_eq!((l[0].row_pitch, r[0], s[0]), (512, 60, 400));
        // Mip 1 (50x30, 200-byte rows) starts after 59 pitched rows and one
        // packed row, rounded up to 512.
        assert_eq!(l[1].offset, (512 * 59 + 400u64).next_multiple_of(512));
        assert_eq!((l[1].width, l[1].height, r[1]), (50, 30, 30));
        assert_eq!(t, l[1].offset + 256 * 29 + 200);
        // BC1: rows of 4x4 blocks, 8 bytes each.
        let (_, r, s, _) = footprints(&texture(64, 64, 71, 1), 0, 1);
        assert_eq!((r[0], s[0]), (16, 128));
        let (_, _, _, t) = footprints(&texture(64, 64, 71, 1), 1, 1);
        assert_eq!(t, u64::MAX, "no second subresource");
    }

    /// A 4x4 texture filled from a title buffer through our list and queue,
    /// then read back from the host GPU.
    #[test]
    fn buffer_data_reaches_the_host_texture() {
        use windows::Win32::Graphics::Direct3D12::*;
        let Some(host) = host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        let iid = &RESOURCE_IIDS[0];
        const COPY_DEST: u32 = 0x400;
        let (mut image, mut source) = (null, null);
        let rgba = texture(4, 4, 28, 1);
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &rgba,
                COPY_DEST,
                null.cast(),
                iid,
                &mut image
            ),
            S_OK
        );
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &buffer(1024),
                0,
                null.cast(),
                iid,
                &mut source
            ),
            S_OK
        );
        let (data, _) = buffer_bytes(source).unwrap();
        // SAFETY: the buffer holds 1024 bytes.
        let bytes = unsafe { std::slice::from_raw_parts_mut(data, 1024) };
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }

        let queue_iid = guid(
            0x0ec870a6,
            0x5d7e,
            0x4c22,
            [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
        );
        let allocator_iid = guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        let (mut queue, mut allocator, mut list) = (null, null, null);
        let queue_desc = [0u32; 4];
        assert_eq!(
            super::super::queue::create(null, queue_desc.as_ptr().cast(), &queue_iid, &mut queue),
            S_OK
        );
        assert_eq!(
            super::super::list::create_allocator(null, 0, &allocator_iid, &mut allocator),
            S_OK
        );
        assert_eq!(
            super::super::list::create_list(null, 0, 0, allocator, null, &list_iid, &mut list),
            S_OK
        );

        #[repr(C)]
        struct Location {
            resource: This,
            kind: u32,
            footprint: PlacedFootprint,
        }
        let rows = PlacedFootprint {
            offset: 0,
            format: 28,
            width: 4,
            height: 4,
            depth: 1,
            row_pitch: 256,
        };
        let to = Location {
            resource: image,
            kind: 0,
            footprint: PlacedFootprint { offset: 0, ..rows },
        };
        let from = Location {
            resource: source,
            kind: 1,
            footprint: rows,
        };
        type Copy =
            extern "system" fn(This, *const Location, u32, u32, u32, *const Location, *const u8);
        slot::<Copy>(list, 0x80)(list, &to, 0, 0, 0, &from, null.cast());
        slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list);
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);

        // Read the host texture back into a readback buffer.
        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 1024,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();
        let host_image = host_resource(image).unwrap();
        let back = host.new_list().unwrap();
        let (_, before) = host_transition(image, 0x800).unwrap();
        // The upload list holds the texture until it is reset.
        assert!(std::sync::Arc::strong_count(&host_copy(image).unwrap()) > 2);
        use windows::core::Interface;
        let out_to = Location {
            resource: readback.as_raw().cast_const().cast(),
            kind: 1,
            footprint: rows,
        };
        let out_from = Location {
            resource: host_image.as_raw().cast_const().cast(),
            kind: 0,
            footprint: to.footprint,
        };
        // SAFETY: an open list, live resources, locations in the host layout.
        unsafe {
            back.list.ResourceBarrier(&[host::transition(
                host_image,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            back.list.CopyTextureRegion(
                (&raw const out_to).cast(),
                0,
                0,
                0,
                (&raw const out_from).cast(),
                None,
            );
        }
        back.close();
        host.execute(&[&back]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy finished (execute waits); the buffer is 1024 bytes.
        let read = unsafe {
            readback
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: 1024,
                    }),
                    Some(&mut mapped),
                )
                .unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), 1024)
        };
        for row in 0..4 {
            let at = row * 256;
            assert_eq!(read[at..at + 16], bytes[at..at + 16], "row {row}");
        }
    }

    #[test]
    fn committed_and_placed_resources_map_at_their_gpu_address() {
        let null = std::ptr::null();
        let mut committed = null;
        let hr = create_committed(
            null,
            &PROPERTIES,
            0,
            &buffer(256),
            0,
            null.cast(),
            &RESOURCE_IIDS[0],
            &mut committed,
        );
        assert_eq!(hr, S_OK);
        let mut data = std::ptr::null_mut();
        assert_eq!(
            slot::<Map>(committed, 0x40)(committed, 0, null.cast(), &mut data),
            S_OK
        );
        assert_eq!(slot::<Address>(committed, 0x58)(committed), data as u64);
        // SAFETY: the buffer is at least 256 zeroed bytes.
        assert_eq!(unsafe { *data.add(255) }, 0);
        assert_eq!(slot::<Release>(committed, 0x10)(committed), 0);

        let heap = heap(4 * KIB_64);
        let place = |offset, desc: &ResourceDesc, out: &mut This| {
            create_placed(
                null,
                heap,
                offset,
                desc,
                0,
                null.cast(),
                &RESOURCE_IIDS[0],
                out,
            )
        };
        let mut placed = null;
        assert_eq!(place(KIB_64, &buffer(KIB_64), &mut placed), S_OK);
        // SAFETY: a heap we made.
        let base = unsafe { (*heap.cast::<Heap>()).memory.ptr.as_ptr() } as u64;
        assert_eq!(slot::<Address>(placed, 0x58)(placed), base + KIB_64);
        let mut other = null;
        assert_eq!(
            place(4 * KIB_64, &buffer(1), &mut other),
            E_INVALIDARG,
            "past the end"
        );
        assert_eq!(
            place(100, &buffer(1), &mut other),
            E_INVALIDARG,
            "misaligned"
        );
        // The resource keeps the heap alive.
        assert_eq!(slot::<Release>(heap, 0x10)(heap), 1);
        assert_eq!(slot::<Release>(placed, 0x10)(placed), 0);
    }

    /// Slots 25, 26 and 108 return a struct through a hidden argument in
    /// either order. A return buffer on the caller's stack that still holds
    /// a stale pointer to the device's own table is still the buffer: the
    /// choice goes by address, never by contents, so the device is never
    /// written over (the failure another runtime and TuxBox report).
    #[test]
    fn the_return_buffer_is_never_the_device() {
        let device = super::super::device::device().cast::<c_void>().cast_mut();
        // SAFETY: the static device object starts with its table pointer.
        let table = unsafe { device.cast::<usize>().read() };
        let mut buffer = [table, table, 0, 0];
        let b = buffer.as_mut_ptr().cast::<c_void>();
        assert_eq!(return_slot::<AllocationInfo>(device, b), b.cast());
        assert_eq!(return_slot::<AllocationInfo>(b, device), b.cast());
        let desc = ResourceDesc { dimension: BUFFER, width: 0x1_0000, height: 1, depth_or_array_size: 1, mip_levels: 1, sample_count: 1, ..Default::default() };
        for (x, y) in [(device, b), (b, device)] {
            let out = allocation_info(x, y, 0, 1, &desc);
            assert_eq!(out.cast::<c_void>(), b);
            // SAFETY: the buffer holds an AllocationInfo now.
            assert_eq!(unsafe { out.read() }, AllocationInfo { size: 0x1_0000, alignment: KIB_64 });
            // SAFETY: as above.
            assert_eq!(unsafe { device.cast::<usize>().read() }, table);
        }
        let out = custom_heap_properties(b, device, 1, 1);
        assert_eq!(out.cast::<c_void>(), b);
    }

    fn placed_map(entries: &[(u64, u64, u64)]) -> PlacedTextureMap {
        entries.iter().map(|&(at, len, used)| ((at, [at as u32; 14]), (at as usize, at + len, used))).collect()
    }

    fn held(map: &PlacedTextureMap) -> u64 {
        map.iter().map(|(k, e)| e.1 - k.0).sum()
    }

    #[test]
    fn placed_overlap_retires_the_old_texture_only_when_cold() {
        let map = placed_map(&[(0x1000, 0x1000, 3), (0x3000, 0x1000, 9), (0x8000, 0x100, 1)]);
        let limit = PlacedLimit { count: 100, bytes: 1 << 30 };
        // A new placement over [0x1800, 0x3800) in frame 12: the first (last
        // used in frame 3) is replaced, the second was used 3 frames ago (an
        // alias), the third is elsewhere.
        let got = placed_to_retire(&map, 0x1800, 0x3800, 12, limit, held(&map), 32, true);
        assert_eq!(got, (vec![(0x1000, [0x1000; 14])], false));
        // In frame 10 the first is still hot.
        assert!(placed_to_retire(&map, 0x1800, 0x3800, 10, limit, held(&map), 32, true).0.is_empty());
    }

    #[test]
    fn placed_cap_retires_cold_least_recently_used_within_the_frames_room() {
        let map = placed_map(&[(0x1000, 0x10, 4), (0x2000, 0x10, 2), (0x3000, 0x10, 9), (0x4000, 0x10, 3)]);
        // Count: 5 with the new one past 4, cut to 3; frame 9's is hot at 16.
        let limit = PlacedLimit { count: 4, bytes: 1 << 30 };
        let got = placed_to_retire(&map, 0x9000, 0x9010, 16, limit, held(&map), 32, true);
        assert_eq!(got.0, vec![(0x2000, [0x2000; 14]), (0x4000, [0x4000; 14])]);
        // Bytes: 0x50 with the new one past 0x40, cut to 0x30.
        let limit = PlacedLimit { count: 100, bytes: 0x40 };
        let got = placed_to_retire(&map, 0x9000, 0x9010, 16, limit, held(&map), 32, true);
        assert_eq!(got.0, vec![(0x2000, [0x2000; 14]), (0x4000, [0x4000; 14])]);
        // Room for one release this frame: the oldest only.
        let got = placed_to_retire(&map, 0x9000, 0x9010, 16, limit, held(&map), 1, true);
        assert_eq!(got.0, vec![(0x2000, [0x2000; 14])]);
        // Under both: nothing.
        let limit = PlacedLimit { count: 100, bytes: 1 << 30 };
        assert!(placed_to_retire(&map, 0x9000, 0x9010, 16, limit, held(&map), 32, true).0.is_empty());
    }

    /// A working set larger than the cap, used every frame, is never released
    /// (an earlier cap of 384 released and remade 250 textures a frame).
    #[test]
    fn placed_cap_never_thrashes_a_hot_working_set() {
        let limit = PlacedLimit { count: 4, bytes: 1 << 30 };
        let mut map = PlacedTextureMap::new();
        let mut made = 0;
        for frame in 10..40u64 {
            for i in 0..8u64 {
                let at = 0x10_0000 * (i + 1);
                let key = (at, [at as u32; 14]);
                if let Some(e) = map.get_mut(&key) {
                    e.2 = frame;
                    continue;
                }
                let (retired, starved) = placed_to_retire(&map, at, at + 0x10, frame, limit, held(&map), 32, true);
                assert!(retired.is_empty(), "frame {frame}: a hot texture released");
                assert_eq!(starved, frame > 10 || i >= 4);
                map.insert(key, (at as usize, at + 0x10, frame));
                made += 1;
            }
        }
        assert_eq!(made, 8, "each texture made once");
    }

    /// A texture's host copy is made at its first use, and the title's
    /// release lets it go (VIctory).
    #[test]
    fn host_copy_made_on_first_use_and_released_with_the_texture() {
        if host::host().is_none() {
            return;
        }
        let desc = texture(64, 64, 28, 1);
        let mut out = std::ptr::null();
        assert_eq!(create_committed(std::ptr::null(), &PROPERTIES, 0, &desc, 0, std::ptr::null(), &RESOURCE_IIDS[0], &mut out), S_OK);
        assert!(resource(out).host.get().is_none(), "no host copy at creation");
        let copy = std::sync::Arc::downgrade(&host_copy(out).expect("a host copy at first use"));
        assert!(copy.upgrade().is_some());
        resource_release(out);
        free_quarantined_now();
        assert!(copy.upgrade().is_none(), "the release lets the host copy go");
    }

    /// A free target is taken back only once old enough and held by
    /// nothing else (a heap slot made from a released view holds it).
    #[test]
    fn free_targets_named_elsewhere_are_not_taken_back() {
        use windows::Win32::Graphics::Direct3D12::*;
        let Some(host) = host::host() else { return };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 16,
            Height: 16,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let mut slot: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe { host.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_RENDER_TARGET, None, &mut slot) }.unwrap();
        let slot = slot.unwrap();
        let mut list = vec![(0, slot.clone(), 4u32)];
        assert!(take_unheld(&mut list, 2).is_none(), "too young");
        assert!(take_unheld(&mut list, 10).is_none(), "a slot still holds it");
        assert_eq!(list.len(), 1);
        drop(slot);
        let (_, state) = take_unheld(&mut list, 10).expect("held by the pool alone");
        assert_eq!((state, list.len()), (4, 0));
    }

    /// A view's resource held while its host view is made outlives the
    /// title's release; its last reference goes at the next submission, and
    /// a freed resource is held no more.
    #[test]
    fn views_hold_their_resource_past_the_titles_release() {
        let mut out = std::ptr::null();
        assert_eq!(create_committed(std::ptr::null(), &PROPERTIES, 0, &buffer(4096), 0, std::ptr::null(), &RESOURCE_IIDS[0], &mut out), S_OK);
        let held = hold_live(out).expect("a live resource is held");
        resource_release(out);
        assert!(is_live(out as usize), "the hold keeps it");
        drop(held);
        assert!(is_live(out as usize), "the last reference is put off");
        release_held();
        assert!(!is_live(out as usize), "freed at the next submission");
        assert!(hold_live(out).is_none(), "a released resource is not held");
        free_quarantined_now();
    }

    /// A target's host texture the title let go waits in the free pool
    /// and is not taken back within the frames a submitted list may still
    /// use it; sampled textures never go there.
    #[test]
    fn released_target_waits_in_the_free_pool() {
        if host::host().is_none() {
            return;
        }
        assert!(target_key(&texture(64, 64, 28, 1), std::ptr::null()).is_none(), "sampled textures have no key");
        let mut desc = texture(72, 40, 28, 1);
        desc.flags = 1; // render target
        let key = target_key(&desc, std::ptr::null()).expect("a target has a key");
        let pooled = || FREE_TARGETS.lock().unwrap().2.get(&key).map_or(0, Vec::len);
        let mut out = std::ptr::null();
        assert_eq!(create_committed(std::ptr::null(), &PROPERTIES, 0, &desc, 0, std::ptr::null(), &RESOURCE_IIDS[0], &mut out), S_OK);
        let first = host_resource(out).expect("a host copy").clone();
        resource_release(out);
        free_quarantined_now();
        assert_eq!(pooled(), 1, "the released target is kept");
        assert_eq!(create_committed(std::ptr::null(), &PROPERTIES, 0, &desc, 0, std::ptr::null(), &RESOURCE_IIDS[0], &mut out), S_OK);
        let second = host_resource(out).expect("a host copy").clone();
        assert_ne!(first, second, "not taken back in the same frame");
        resource_release(out);
        free_quarantined_now();
        assert_eq!(pooled(), 2);
        let mut free = FREE_TARGETS.lock().unwrap();
        if let Some(v) = free.2.remove(&key) {
            free.1 -= v.len();
        }
    }
}
