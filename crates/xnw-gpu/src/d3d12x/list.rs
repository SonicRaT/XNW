//! Command allocators and command lists.
//!
//! The title's inlined code writes packets into a list only while its put
//! pointer is below a limit, and calls the list's method otherwise. Our
//! lists keep the pointer and every limit at zero, so every command reaches
//! the table as a call.
//!
//! Each list records onto a host command list as its methods are called
//! (VIctory's decision 0007): copies, clears, resolves, barriers and
//! fixed-function state are recorded and run when the list is submitted;
//! buffer copies and bottom-of-pipe writes are done on the CPU at
//! submission, in order. Draw state lives in `draw.rs`, where draws and
//! dispatches bind and record.
//!
//! Ported from VIctory-base (VIctory main d2a4bae + 26 PRs, branch
//! `xnw/base` @ 5d0516e, GPL-3.0-or-later), `crates/victory-gpu/src/
//! list.rs` 1-1441 whole (PR touching it: #681), tests 1396-1441.
//! Changed: our slot tables with roles (`object::table`, for the
//! inventory); `ExecuteIndirect` as the base; no pipeline binding
//! notice; counts of the
//! work recorded onto host lists for the frame log; `tracing` is
//! `xnw_core::{info,warn}`.
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]

use super::uikeep::Command;
use std::mem::offset_of;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::{D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_STATES};
use windows::core::Interface;

use super::host::HostList;
use super::object::{self, ALLOCATOR, Done, E_INVALIDARG, HRESULT, Iid, LIST, Object, S_OK, Slot, Table, This, guid};
use super::queue::Written;
use super::resource::PAGEABLE_IIDS;

pub(crate) const ALLOCATOR_IIDS: [Iid; 4] = [
    guid(0x6102dee4, 0xaf59, 0x4b09, [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24]),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

/// The list interfaces up to `ID3D12GraphicsCommandList6`: those with PC's
/// ids, the console's own List5 and List6 (the title asks for List6), then
/// the device child's.
pub(crate) const LIST_IIDS: [Iid; 10] = [
    guid(0x7116d91c, 0xe7e4, 0x47ce, [0xb8, 0xc6, 0xec, 0x81, 0x68, 0xf4, 0x37, 0xe5]),
    guid(0x5b160d0f, 0xac1b, 0x4185, [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55]),
    guid(0x553103fb, 0x1fe7, 0x4557, [0xbb, 0x38, 0x94, 0x6d, 0x7d, 0x0e, 0x7c, 0xa7]),
    guid(0x38c3e585, 0xff17, 0x412c, [0x91, 0x50, 0x4f, 0xc6, 0xf9, 0xd7, 0x2a, 0x28]),
    guid(0x6fda83a7, 0xb84c, 0x4e38, [0x9a, 0xc8, 0xc7, 0xbd, 0x22, 0x01, 0x6b, 0x3d]),
    guid(0x8754318e, 0xd3a9, 0x4541, [0x98, 0xcf, 0x64, 0x5b, 0x50, 0xdc, 0x48, 0x74]),
    guid(0x55050859, 0x4024, 0x474c, [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74]),
    guid(0xc3827890, 0x4024, 0x474c, [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74]),
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Allocator {
    object: Object,
    _kind: u32,
}

/// The layout the title's inlined code reads.
#[repr(C)]
struct List {
    object: Object,
    kind: u32,
    /// The put pointer, then the plain, draw, API and unbiased API limits:
    /// all zero, so every inline check fails.
    put: [u64; 5],
    reserved: [u32; 64],
    /// State the inline code keeps for building packets; unused by us.
    graphics_root_headers: [u32; 128],
    compute_root_headers: [u32; 128],
    pipeline_shadow: [u32; 8],
}

/// Ours, after the console layout and its pad (see [`pad`]): memory handed
/// out by [`allocate_upload`], kept until the list is reset; the host list
/// the calls record into, and the draw state the title set.
struct Ours {
    uploads: std::sync::Mutex<Vec<Upload>>,
    host: Option<HostList>,
    state: std::sync::Mutex<super::draw::State>,
}

/// Newer titles' lists are larger: the title's state flush (title+0x32786)
/// copies a pipeline's 32 bytes to +0x544..+0x564, tests +0x548, and reads
/// a pointer past +0x560 (a canary there was read and dereferenced),
/// so its layout runs past 0x560. Its size is unknown: zeroed bytes after
/// the console layout keep our fields clear of whatever the title keeps
/// there. `XNW_LIST_PAD=<bytes>` (default 1024, rounded up to 16).
fn pad() -> usize {
    static PAD: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        let pad = std::env::var("XNW_LIST_PAD").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(1024).next_multiple_of(16);
        if pad != 1024 {
            xnw_core::info!("d3d12x: command list pad {pad} bytes after +0x560 (XNW_LIST_PAD)");
        }
        pad
    });
    *PAD
}

/// Where [`Ours`] starts in a list object, and the object's layout.
fn ours_at() -> usize {
    (size_of::<List>() + pad()).next_multiple_of(align_of::<Ours>())
}
fn list_layout() -> std::alloc::Layout {
    std::alloc::Layout::from_size_align(ours_at() + size_of::<Ours>(), align_of::<List>().max(align_of::<Ours>())).expect("a list layout")
}

/// Our part of list `this`.
fn ours<'a>(this: This) -> &'a Ours {
    // SAFETY: the title passes one of our lists, laid out by `new_list`.
    unsafe { &*this.cast::<u8>().add(ours_at()).cast::<Ours>() }
}

/// Upload memory handed to the title: heap memory, or a guarded block.
enum Upload {
    Heap(#[allow(dead_code)] Vec<u8>),
    Guarded(usize),
}
impl Drop for Upload {
    fn drop(&mut self) {
        if let Upload::Guarded(at) = *self {
            super::guard::free(at as *mut u8);
        }
    }
}

/// List `this`'s pad (zero at creation, as the title expects) at
/// each submission and at release (`XNW_OBJECT_GUARD`): the non-zero bytes
/// the title left there, by range from the object's start, logged once per
/// distinct range with 64 bytes around the first and the number of ranges
/// seen; counted per frame ([`pad_written`]).
pub(crate) fn guard_check(this: This) {
    if super::guard::mode() == 0 || this.is_null() {
        return;
    }
    // The slack between the object's end and its guard page.
    super::guard::slack_check(this.cast());
    let head = size_of::<List>();
    let Some((first, last)) = pad_range(this) else {
        return;
    };
    // SAFETY: the pad bytes of one of our lists.
    let pad = unsafe { std::slice::from_raw_parts(this.cast::<u8>().add(head), pad()) };
    PAD_WRITTEN.fetch_add(1, Ordering::Relaxed);
    static SEEN: std::sync::Mutex<std::collections::BTreeMap<(usize, usize), u64>> = std::sync::Mutex::new(std::collections::BTreeMap::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let n = seen.entry((first, last)).or_insert(0);
    *n += 1;
    if *n == 1 {
        let from = first.saturating_sub(16).max(head) & !15;
        let to = (from + 64).min(head + pad.len());
        // SAFETY: inside the pad.
        let bytes = unsafe { std::slice::from_raw_parts(this.cast::<u8>().add(from), to - from) };
        let hex: Vec<String> = bytes.chunks(8).map(|c| c.iter().map(|b| format!("{b:02x}")).collect::<String>()).collect();
        xnw_core::warn!(
            "OBJECT GUARD: list {:#x}: the title left non-zero bytes in the pad after +0x560: first +{first:#x}, last +{last:#x} (pad ends at +{:#x}); bytes from +{from:#x}: {}; {} distinct ranges so far",
            this as usize,
            head + pad.len(),
            hex.join(" "),
            seen.len()
        );
    }
}

/// The first and last non-zero byte of list `this`'s pad, as offsets
/// from the object's start.
fn pad_range(this: This) -> Option<(usize, usize)> {
    let head = size_of::<List>();
    // SAFETY: the pad bytes of one of our lists.
    let pad = unsafe { std::slice::from_raw_parts(this.cast::<u8>().add(head), pad()) };
    Some((head + pad.iter().position(|&b| b != 0)?, head + pad.iter().rposition(|&b| b != 0)?))
}

/// List checks that found pad bytes written (taken per frame line).
static PAD_WRITTEN: AtomicU64 = AtomicU64::new(0);
pub(crate) fn pad_written() -> u64 {
    PAD_WRITTEN.swap(0, Ordering::Relaxed)
}
const _: () = assert!(offset_of!(List, put) == 0x18);
const _: () = assert!(offset_of!(List, pipeline_shadow) + 32 == 0x560);

/// Runs `f` with the host list and draw state of a title list, if there
/// is a host GPU.
fn with_state(this: This, f: impl FnOnce(&HostList, &mut super::draw::State)) {
    let Some(list) = host_list(this) else { return };
    let state = &ours(this).state;
    f(list, &mut state.lock().unwrap_or_else(|e| e.into_inner()));
}

/// With `XNW_GPU=null...,ui=keep` (uikeep.rs): whether a draw-like
/// command recorded now is kept; a kept one marks its list for execution.
/// Always true with the switch off or before the null point.
fn kept(list: &HostList, c: super::uikeep::Command) -> bool {
    let k = super::uikeep::command(c);
    if k && super::uikeep::on() {
        list.kept.store(true, Ordering::Relaxed);
    }
    k
}

/// [`kept`] for a plain draw; the instrument sees its target (uikeep.rs).
fn kept_draw(list: &HostList, s: &super::draw::State) -> Option<super::uikeep::Drawing> {
    let k = kept(list, Command::Draw);
    s.ui_seen(k);
    k.then(super::uikeep::Drawing::begin)
}

/// The host list behind a title list, if there is a host GPU.
pub(crate) fn host_list<'a>(this: This) -> Option<&'a HostList> {
    if this.is_null() {
        return None;
    }
    ours(this).host.as_ref()
}

/// Device method 0x48.
pub(crate) extern "system" fn create_allocator(_: This, kind: u32, iid: *const Iid, out: *mut This) -> HRESULT {
    let allocator = Box::new(Allocator { object: Object::new(&ALLOCATOR_FUNCTIONS), _kind: kind });
    object::hand_out(ALLOCATOR, allocator, iid, &ALLOCATOR_IIDS, out)
}

/// Device method 0x60. Lists start open, as in D3D12.
pub(crate) extern "system" fn create_list(_: This, _nodes: u32, kind: u32, allocator: This, _pipeline: This, iid: *const Iid, out: *mut This) -> HRESULT {
    new_list(kind, allocator, iid, out)
}

/// The console's list description: type, flags, node mask, barrier batch size.
#[repr(C)]
#[derive(Debug)]
pub struct ListDesc {
    pub kind: u32,
    pub flags: u32,
    pub nodes: u32,
    pub barrier_batch: u32,
}

/// Device method 0x248 `CreateCommandListX`.
pub(crate) extern "system" fn create_list_x(_: This, desc: *const ListDesc, allocator: This, _pipeline: This, iid: *const Iid, out: *mut This) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    new_list(desc.kind, allocator, iid, out)
}

fn new_list(kind: u32, allocator: This, iid: *const Iid, out: *mut This) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    let functions = if super::device::newer() { &NEWER_FUNCTIONS } else { &FUNCTIONS };
    if !object::is(iid, &LIST_IIDS) {
        return object::answer(LIST, iid, None, out);
    }
    if out.is_null() {
        return object::S_FALSE;
    }
    let layout = list_layout();
    // Against a guard page under XNW_OBJECT_GUARD, else the heap;
    // zeroed either way (the pad stays zero, as the title expects).
    let at = super::guard::alloc("list", layout.size(), layout.align()).unwrap_or_else(|| {
        // SAFETY: a non-zero size.
        unsafe { std::alloc::alloc_zeroed(layout) }
    });
    if at.is_null() {
        return object::E_OUTOFMEMORY;
    }
    // SAFETY: fresh memory of the list's layout.
    unsafe {
        at.write_bytes(0, layout.size());
        at.cast::<List>().write(List {
            object: Object::new(functions),
            kind,
            put: [0; 5],
            reserved: [0; 64],
            graphics_root_headers: [0; 128],
            compute_root_headers: [0; 128],
            pipeline_shadow: [0; 8],
        });
        at.add(ours_at()).cast::<Ours>().write(Ours {
            uploads: Default::default(),
            host: super::host::host().and_then(|h| match h.new_list() {
                Ok(list) => Some(list),
                Err(e) => {
                    xnw_core::warn!("d3d12x: no host command list: {e}");
                    None
                }
            }),
            state: Default::default(),
        });
    }
    object::answer(LIST, iid, Some(at.cast_const().cast()), out)
}

extern "system" fn allocator_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(ALLOCATOR, iid, object::is(iid, &ALLOCATOR_IIDS).then_some(this), out)
}
extern "system" fn allocator_release(this: This) -> u32 {
    object::release_boxed::<Allocator>(this)
}
/// Each list keeps its own host allocator, reset with the list.
extern "system" fn allocator_reset(_: This) -> HRESULT {
    S_OK
}

extern "system" fn list_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(LIST, iid, object::is(iid, &LIST_IIDS).then_some(this), out)
}
extern "system" fn list_release(this: This) -> u32 {
    guard_check(this);
    match object::count_down(this) {
        Some(0) => {
            // PASS1: a submission of it sent without a wait has run before its memory goes.
            if let Some(host) = &ours(this).host {
                host.settle();
            }
            object::drop_private(this);
            let at = this.cast_mut().cast::<u8>();
            // SAFETY: the last reference: our part is dropped, then the
            // memory goes back where `new_list` took it.
            unsafe {
                std::ptr::drop_in_place(at.add(ours_at()).cast::<Ours>());
                if !super::guard::free(at) {
                    std::alloc::dealloc(at, list_layout());
                }
            }
            0
        }
        Some(n) => n,
        None => 0,
    }
}
extern "system" fn list_type(this: This) -> u32 {
    // SAFETY: `this` is one of our lists.
    unsafe { (*this.cast::<List>()).kind }
}
extern "system" fn close(this: This) -> HRESULT {
    with_state(this, |list, s| s.snapshot_targets(list));
    if let Some(host) = host_list(this) {
        host.close();
    }
    S_OK
}
extern "system" fn reset(this: This, allocator: This, pipeline: This) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    // Every submission of the list has run (submissions are waited for;
    // PASS1: one sent without a wait is waited for here, before its upload
    // memory and allocator are reused).
    if let Some(host) = host_list(this) {
        host.settle();
    }
    let uploads = &ours(this).uploads;
    uploads.lock().unwrap_or_else(|e| e.into_inner()).clear();
    super::pipeline::bind(this, pipeline);
    with_state(this, |host, state| {
        host.reset();
        state.reset(pipeline);
    });
    S_OK
}

// ---- What is recorded onto host lists, for the frame log -------------

/// Commands recorded onto host lists (they run when the list is
/// submitted), by kind.
pub(crate) static RECORDED: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
pub(crate) const RECORDED_NAMES: [&str; 6] = ["target clears", "depth clears", "texture copies", "resolves", "buffer copies", "barriers"];
const CLEAR: usize = 0;
const CLEAR_DEPTH: usize = 1;
const COPY_TEXTURE: usize = 2;
const RESOLVE: usize = 3;
const COPY_BUFFER: usize = 4;
const BARRIER: usize = 5;

fn recorded(kind: usize) {
    RECORDED[kind].fetch_add(1, Ordering::Relaxed);
}

/// Copies, clears and resolves still dropped: what they name has no host
/// copy yet (sampled textures, read-backs) or no view record.
static OTHER: AtomicU64 = AtomicU64::new(0);

fn not_recorded(what: &str) {
    OTHER.fetch_add(1, Ordering::Relaxed);
    static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.iter().any(|s| s == what) {
        seen.push(what.to_owned());
        xnw_core::info!("d3d12x: a list command not recorded (first time): {what}");
    }
}

/// A resource barrier as titles pass it: type, flags, then a resource and
/// either a subresource and two states (transition) or a second resource
/// (aliasing). The same layout as the host's.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Barrier {
    kind: u32,
    flags: u32,
    first: This,
    second: u64,
    last: u64,
}

/// The barrier on host resources, or `None` if it names a resource the host
/// GPU does not have (buffers so far).
fn host_barrier(list: &HostList, b: &Barrier) -> Option<Barrier> {
    let host = |r: This| super::resource::host_resource(r).map(|h| h.as_raw().cast_const().cast());
    match b.kind {
        0 => {
            // A split barrier's first half: its second half transitions.
            const BEGIN_ONLY: u32 = 1;
            if b.flags & BEGIN_ONLY != 0 {
                return None;
            }
            // shortcut: whole resources, from the state the list tracked
            // (`HostList::transition`); subresources are not told apart.
            let wanted = super::host::texture_state(b.last as u32);
            let (texture, before) = list.transition(b.first, wanted)?;
            Some(Barrier {
                kind: 0,
                flags: 0,
                first: texture.as_raw().cast_const().cast(),
                second: 0xffff_ffff | (u64::from(before) << 32),
                last: u64::from(super::host::legal_state(texture, wanted)),
            })
        }
        1 => Some(Barrier {
            first: if b.first.is_null() { b.first } else { host(b.first)? },
            second: if b.second == 0 { 0 } else { host(b.second as This)? as u64 },
            ..*b
        }),
        2 => Some(Barrier { first: if b.first.is_null() { b.first } else { host(b.first)? }, ..*b }),
        _ => None,
    }
}

/// An unordered access barrier on every resource: earlier shader writes
/// finish before later work reads or writes.
const UAV_BARRIER: Barrier = Barrier { kind: 2, flags: 0, first: std::ptr::null(), second: 0, last: 0 };

/// Returns the console's barrier "actions" (none).
extern "system" fn barrier(this: This, count: u32, barriers: *const Barrier) -> u32 {
    let _call = super::counters::call(super::counters::BARRIER);
    if let Some(list) = host_list(this)
        && !barriers.is_null()
    {
        // SAFETY: the title passes `count` barriers.
        let barriers = unsafe { std::slice::from_raw_parts(barriers, count as usize) };
        let mut host: Vec<Barrier> = barriers.iter().filter_map(|b| host_barrier(list, b)).collect();
        list.crumbs().note(|| {
            let each: Vec<String> = barriers
                .iter()
                .take(6)
                .map(|b| match b.kind {
                    0 => format!("transition {:#x} sub {} {:#x}->{:#x}", b.first as usize, b.second as u32, (b.second >> 32) as u32, b.last as u32),
                    1 => format!("aliasing {:#x}->{:#x}", b.first as usize, b.second),
                    _ => format!("UAV {:#x}", b.first as usize),
                })
                .collect();
            format!("title barriers ({} given, {} to the host{}): {}", barriers.len(), host.len(), if barriers.len() > 6 { ", first 6" } else { "" }, each.join(", "))
        });
        // Buffers have host copies, not resources the barrier can name:
        // a barrier on one waits for every shader write before it.
        if barriers.iter().any(|b| b.kind != 1 && !b.first.is_null() && super::resource::buffer_bytes(b.first).is_some()) {
            host.push(UAV_BARRIER);
        }
        if !host.is_empty() {
            recorded(BARRIER);
            const _: () = assert!(size_of::<Barrier>() == size_of::<D3D12_RESOURCE_BARRIER>());
            // SAFETY: the same layout; borrowed resource pointers, never
            // released through this slice.
            let host = unsafe { std::slice::from_raw_parts(host.as_ptr().cast::<D3D12_RESOURCE_BARRIER>(), host.len()) };
            // SAFETY: an open list and live resources.
            unsafe { list.list.ResourceBarrier(host) };
        }
    }
    0
}

/// Moves a title texture's host copy to `state` in the list, if needed.
pub(crate) fn ensure_state(list: &HostList, resource: This, state: u32) {
    if let Some((texture, before)) = list.transition(resource, state) {
        let state = super::host::legal_state(texture, state);
        list.crumbs().note(|| format!("transition host texture of {:#x}: {before:#x}->{state:#x}", resource as usize));
        let barrier = super::host::transition(texture, D3D12_RESOURCE_STATES(before as i32), D3D12_RESOURCE_STATES(state as i32));
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
}

const COPY_DEST: u32 = 0x400;
const COPY_SOURCE: u32 = 0x800;

/// `len` bytes at `offset` of a title buffer, if they are inside it.
fn buffer_range(buffer: This, offset: u64, len: u64) -> Option<*mut u8> {
    let (data, size) = super::resource::buffer_bytes(buffer)?;
    let end = offset.checked_add(len)?;
    if end > size {
        xnw_core::warn!("d3d12x: copy outside its buffer ({offset:#x} + {len:#x} > {size:#x})");
        return None;
    }
    // SAFETY: inside the buffer, checked above.
    Some(unsafe { data.add(offset as usize) })
}

/// Buffer to buffer: both are host memory, copied at submission.
extern "system" fn copy_buffer_region(this: This, dst: This, dst_offset: u64, src: This, src_offset: u64, len: u64) {
    let _call = super::counters::call(super::counters::COPY_BUFFER);
    let Some(list) = host_list(this) else { return };
    if let Some(to) = buffer_range(dst, dst_offset, len)
        && let Some(from) = buffer_range(src, src_offset, len)
    {
        recorded(COPY_BUFFER);
        list.defer_copy(to, from, len as usize);
    } else {
        not_recorded("buffer copy outside a buffer of ours");
    }
}

extern "system" fn copy_resource(this: This, dst: This, src: This) {
    let _call = super::counters::call(super::counters::COPY_RESOURCE);
    let Some(list) = host_list(this) else {
        return;
    };
    whole_for_write(list, dst);
    super::uikeep::copy_seen(super::uikeep::keep_now(), dst as usize, src as usize);
    if super::resource::host_copy(src).is_some_and(|c| c.dropped > 0) {
        // A source without its top mips does not match the title's.
        super::resource::skipped_copy();
        not_recorded("resource copy from a texture whose host copy lacks mips");
        return;
    }
    if let (Some(to), Some(from)) = (super::resource::host_resource(dst), super::resource::host_resource(src)) {
        if !kept(list, Command::Copy) {
            return;
        }
        // A CPU-written source reaches its host copy first (VIctory 0064c08).
        if super::texture::title_written(from) {
            super::texture::upload(list, src, from);
        }
        ensure_state(list, dst, COPY_DEST);
        ensure_state(list, src, COPY_SOURCE);
        super::texture::host_written(dst);
        recorded(COPY_TEXTURE);
        // SAFETY: an open list and live textures.
        let _crumb = super::crumbs::begin(list, || "copy texture".to_string());
        unsafe { list.list.CopyResource(to, from) };
    } else if let (Some((to, a)), Some((from, b))) = (super::resource::buffer_bytes(dst), super::resource::buffer_bytes(src)) {
        recorded(COPY_BUFFER);
        list.defer_copy(to, from, a.min(b) as usize);
    } else {
        not_recorded("resource copy between textures without host copies");
    }
}

/// Before the title writes `texture` on the GPU, its host copy holds
/// the whole chain; a reduced one replaced now gets what it held uploaded
/// first (the upload runs before this list's commands).
fn whole_for_write(list: &HostList, texture: This) {
    if super::resource::full_shape(texture)
        && let Some(host) = super::resource::host_resource(texture)
        && super::texture::title_written(host)
    {
        super::texture::upload(list, texture, host);
    }
}

extern "system" fn copy_resource_x(this: This, dst: This, src: This, _flags: u32) {
    copy_resource(this, dst, src);
}

/// A copy location as titles pass it, laid out as the host's: a resource,
/// then a subresource index (type 0) or a placed footprint (type 1).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CopyLocation {
    resource: This,
    kind: u32,
    footprint: super::resource::PlacedFootprint,
}

/// Texture to texture on the GPU; buffer to texture through an upload
/// buffer filled at submission. Texture to buffer (read back) is not done
/// yet.
extern "system" fn copy_texture_region(this: This, dst: *const CopyLocation, x: u32, y: u32, z: u32, src: *const CopyLocation, source_box: *const [u32; 6]) {
    let _call = super::counters::call(super::counters::COPY_TEXTURE);
    use windows::Win32::Graphics::Direct3D12::{D3D12_BOX, D3D12_TEXTURE_COPY_LOCATION};
    const FOOTPRINT: u32 = 1;
    // SAFETY: the title passes both locations (null is tolerated).
    let (Some(list), Some(&dst), Some(&src)) = (host_list(this), unsafe { dst.as_ref() }, unsafe { src.as_ref() }) else {
        return;
    };
    let k = kept(list, Command::Copy);
    super::uikeep::copy_seen(k, dst.resource as usize, src.resource as usize);
    if !k {
        return;
    }
    whole_for_write(list, dst.resource);
    let Some(to) = super::resource::host_resource(dst.resource) else {
        not_recorded(if super::resource::buffer_bytes(dst.resource).is_some() {
            "texture to buffer copy (read back)"
        } else {
            "copy into a texture without a host copy (sampled textures take their own path)"
        });
        return;
    };
    let mut from = src;
    let _upload; // keeps the source alive while this call records it
    if src.kind == FOOTPRINT {
        let (Some(host), Some(bytes)) = (super::host::host(), src.footprint.bytes()) else {
            return;
        };
        let Some(data) = buffer_range(src.resource, src.footprint.offset, bytes) else {
            return;
        };
        let Some((buffer, staging)) = list.upload(host, bytes) else {
            return;
        };
        // The list owns this mapped upload buffer: the host-upload path
        // (VIctory af22e5d; there for Wine, which can report its address as
        // MEM_FREE; here it also leaves the copied-range bookkeeping alone).
        list.defer_fill(staging, data, bytes as usize);
        from.resource = buffer.as_raw().cast_const().cast();
        from.footprint.offset = 0;
        _upload = buffer;
    } else {
        let Some(texture) = super::resource::host_resource(src.resource) else {
            not_recorded("texture copy from a texture without a host copy");
            return;
        };
        // A CPU-written source reaches its host copy first (VIctory 4f53d6f).
        if super::texture::title_written(texture) {
            super::texture::upload(list, src.resource, texture);
        }
        ensure_state(list, src.resource, COPY_SOURCE);
        from.resource = texture.as_raw().cast_const().cast();
        // The title's subresource, as the host copy numbers it.
        let Some(index) = super::resource::host_subresource(src.resource, src.footprint.offset as u32) else {
            super::resource::skipped_copy();
            not_recorded("texture copy from a mip the host copy does not have");
            return;
        };
        from.footprint.offset = u64::from(index);
    }
    let mut to_location = dst;
    if dst.kind != FOOTPRINT {
        let Some(index) = super::resource::host_subresource(dst.resource, dst.footprint.offset as u32) else {
            super::resource::skipped_copy();
            not_recorded("texture copy into a mip the host copy does not have");
            return;
        };
        to_location.footprint.offset = u64::from(index);
    }
    ensure_state(list, dst.resource, COPY_DEST);
    super::texture::host_written(dst.resource);
    recorded(COPY_TEXTURE);
    to_location.resource = to.as_raw().cast_const().cast();
    let host_box = (!source_box.is_null()).then(|| source_box.cast::<D3D12_BOX>());
    const _: () = assert!(size_of::<CopyLocation>() == size_of::<D3D12_TEXTURE_COPY_LOCATION>());
    // SAFETY: an open list; both locations in the host's layout, naming
    // live resources; the box is the title's (same layout).
    let _crumb = super::crumbs::begin(list, || format!("copy texture region to {x},{y},{z}"));
    unsafe { list.list.CopyTextureRegion((&raw const to_location).cast(), x, y, z, (&raw const from).cast(), host_box) };
}

extern "system" fn copy_texture_region_x(this: This, dst: *const CopyLocation, x: u32, y: u32, z: u32, src: *const CopyLocation, source_box: *const [u32; 6], _flags: u32) {
    copy_texture_region(this, dst, x, y, z, src, source_box);
}

extern "system" fn resolve(this: This, dst: This, dst_sub: u32, src: This, src_sub: u32, format: u32) {
    let _call = super::counters::call(super::counters::COPY_RESOLVE);
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Direct3D12::{D3D12_RESOLVE_MODE_AVERAGE, ID3D12GraphicsCommandList1};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
    let Some(list) = host_list(this) else { return };
    if !kept(list, Command::Copy) {
        return;
    }
    whole_for_write(list, dst);
    let (Some(to), Some(from)) = (super::resource::host_resource(dst), super::resource::host_resource(src)) else {
        not_recorded("resolve between textures without host copies");
        return;
    };
    // Both subresources as the host copies number them.
    let (Some(dst_sub), Some(src_sub)) = (super::resource::host_subresource(dst, dst_sub), super::resource::host_subresource(src, src_sub)) else {
        super::resource::skipped_copy();
        not_recorded("resolve of a mip a host copy does not have");
        return;
    };
    const RESOLVE_DEST: u32 = 0x1000;
    const RESOLVE_SOURCE: u32 = 0x2000;
    ensure_state(list, dst, RESOLVE_DEST);
    ensure_state(list, src, RESOLVE_SOURCE);
    super::texture::host_written(dst);
    recorded(RESOLVE);
    // SAFETY: an open list and live textures; the region is inside both.
    unsafe {
        let (a, b) = (to.GetDesc(), from.GetDesc());
        let format = DXGI_FORMAT(format as i32);
        if (a.Width, a.Height) == (b.Width, b.Height) {
            let _crumb = super::crumbs::begin(list, || format!("resolve {}x{}", a.Width, a.Height));
            list.list.ResolveSubresource(to, dst_sub, from, src_sub, format);
        } else if let Ok(list1) = list.list.cast::<ID3D12GraphicsCommandList1>() {
            // The console clips a larger source to the destination.
            let rect = RECT { left: 0, top: 0, right: a.Width.min(b.Width) as i32, bottom: a.Height.min(b.Height) as i32 };
            list1.ResolveSubresourceRegion(to, dst_sub, 0, 0, from, src_sub, Some(&rect), format, D3D12_RESOLVE_MODE_AVERAGE);
        }
    }
}

#[repr(C)]
pub(crate) struct Rect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

extern "system" fn clear_rtv(this: This, handle: u64, color: *const [f32; 4], count: u32, rects: *const Rect) {
    let _call = super::counters::call(super::counters::CLEAR_TARGET);
    let Some(list) = host_list(this) else { return };
    let mut k = kept(list, Command::Clear);
    if super::uikeep::on()
        && let Some(record) = super::descriptor::rtv_record(handle)
    {
        if !k {
            match super::uikeep::clear_admit(record.resource as usize) {
                1 => {
                    list.kept.store(true, Ordering::Relaxed);
                    k = true;
                }
                // A whole clear of a UI-written texture waits for a kept
                // draw into it in this frame (`clear_held`).
                // SAFETY: the title passes four floats (null is tolerated).
                2 if count == 0 => {
                    if let Some(color) = unsafe { color.as_ref() } {
                        super::uikeep::hold_clear(record.resource as usize, handle, *color);
                    }
                }
                _ => {}
            }
        }
        super::uikeep::clear_seen(k, record.resource as usize);
    }
    if !k {
        return;
    }
    // SAFETY: the title passes four floats (null is tolerated).
    let Some(color) = (unsafe { color.as_ref() }) else {
        return;
    };
    let rects = (count > 0 && !rects.is_null()).then(|| {
        // SAFETY: the title passes `count` rectangles, laid out as the host's.
        unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
    });
    clear_target(list, handle, color, rects);
}

/// KEEP1: the clear held for the target of the kept draw being recorded
/// (uikeep::held_clear) is done now, on the draw's list, before the draw.
pub(crate) fn clear_held(list: &HostList) {
    if let Some((target, handle, color)) = super::uikeep::held_clear() {
        super::uikeep::clear_seen(true, target);
        clear_target(list, handle, &color, None);
    }
}

/// Records a target clear on `list`.
fn clear_target(list: &HostList, handle: u64, color: &[f32; 4], rects: Option<&[windows::Win32::Foundation::RECT]>) {
    let Some(record) = super::descriptor::rtv_record(handle) else {
        not_recorded("target clear of a view not recorded");
        return;
    };
    let (Some(texture), Some(host)) = (super::resource::host_resource(record.resource), super::host::host()) else {
        not_recorded("target clear of a texture without a host copy");
        return;
    };
    const RENDER_TARGET: u32 = 0x4;
    let rtv = host.rtv(texture, record.desc().as_ref());
    ensure_state(list, record.resource, RENDER_TARGET);
    super::texture::host_written(record.resource);
    recorded(CLEAR);
    first_clear(handle, record.resource, color);
    // SAFETY: an open list, a fresh view of a live texture.
    let _crumb = super::crumbs::begin(list, || "clear render target".to_string());
    unsafe { list.list.ClearRenderTargetView(rtv, color, rects) };
}

/// Logs the first target clear: what it clears to.
fn first_clear(handle: u64, resource: This, color: &[f32; 4]) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| xnw_core::info!("d3d12x: first target clear recorded: view {handle:#x}, resource {resource:?}, colour {color:?}"));
}

/// `ClearDepthStencilView(handle, flags, depth, stencil, count, rects)`;
/// flags: 1 depth, 2 stencil.
extern "system" fn clear_dsv(this: This, handle: u64, flags: u32, depth: f32, stencil: u8, count: u32, rects: *const Rect) {
    let _call = super::counters::call(super::counters::CLEAR_DEPTH);
    use windows::Win32::Graphics::Direct3D12::D3D12_CLEAR_FLAGS;
    let Some(list) = host_list(this) else { return };
    if !kept(list, Command::Clear) {
        return;
    }
    let Some(record) = super::descriptor::dsv_record(handle) else {
        not_recorded("depth clear of a view not recorded");
        return;
    };
    let (Some(texture), Some(host)) = (super::resource::host_resource(record.resource), super::host::host()) else {
        not_recorded("depth clear of a texture without a host copy");
        return;
    };
    let rects = (count > 0 && !rects.is_null()).then(|| {
        // SAFETY: the title passes `count` rectangles, laid out as the host's.
        unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
    });
    const DEPTH_WRITE: u32 = 0x10;
    ensure_state(list, record.resource, DEPTH_WRITE);
    let dsv = host.dsv(texture, record.depth_desc().as_ref());
    super::texture::host_written(record.resource);
    recorded(CLEAR_DEPTH);
    // SAFETY: an open list, a fresh view of a live texture.
    let _crumb = super::crumbs::begin(list, || "clear depth".to_string());
    unsafe { list.list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAGS(flags as i32 & 3), depth, stencil, rects) };
}

/// `CopyMemoryToMemoryX(destination, source, size, flags)`: GPU addresses
/// are title memory, copied at submission like buffer copies (if mapped).
extern "system" fn copy_memory(this: This, dst: u64, src: u64, len: u64, _flags: u32) {
    let _call = super::counters::call(super::counters::COPY_MEMORY);
    if let Some(list) = host_list(this)
        && dst != 0
        && src != 0
    {
        let len = len as usize;
        recorded(COPY_BUFFER);
        // No VirtualQuery of both ranges first (two queries a copy,
        // milliseconds each in the title's big pools: most of a skipped
        // frame's cost in a test run); the kernel checks both ends as it copies.
        list.defer(move || super::host::copy_mapped(dst as *mut u8, src as *const u8, len, true));
    }
}

/// `DriverHintX(feature, value)`: no hints for the host.
extern "system" fn driver_hint(_: This, _feature: u64, _value: u64) -> u64 {
    0
}

/// `DispatchX(x, y, z, flags)`: a dispatch with console flags, which the
/// host has no use for (the ray-tracing builder dispatches this way).
extern "system" fn dispatch_x(this: This, x: u32, y: u32, z: u32, _flags: u32) {
    state::dispatch(this, x, y, z);
}

/// `BeginQuery`/`EndQuery(heap, type, index)`: queries are answered when
/// resolved.
extern "system" fn query_mark(_: This, _heap: This, _kind: u32, _index: u32) {}

/// `ResolveQueryData(heap, type, first, count, buffer, offset)`: writes
/// each query's result into the buffer once the list has run. Timestamps
/// (type 2) are the GPU clock then; occlusion counts (0) say a million
/// samples passed and binary occlusion (1) visible, so nothing is culled
/// by a query.
/// shortcut: no query runs on the host GPU; results are plausible values,
/// not measured ones.
extern "system" fn resolve_queries(this: This, _heap: This, kind: u32, _first: u32, count: u32, buffer: This, offset: u64) {
    const OCCLUSION: u32 = 0;
    const BINARY: u32 = 1;
    const TIMESTAMP: u32 = 2;
    const PIPELINE_STATISTICS: u32 = 3;
    let Some((data, size)) = super::resource::buffer_bytes(buffer) else {
        return;
    };
    let (what, bytes) = match kind {
        OCCLUSION => (Written::Value(1 << 20), 8),
        BINARY => (Written::Value(1), 8),
        TIMESTAMP => (Written::GpuTimestamp, 8),
        PIPELINE_STATISTICS => (Written::Value(0), 88),
        _ => (Written::Value(0), 16),
    };
    if offset + u64::from(count) * bytes > size {
        return;
    }
    for i in 0..u64::from(count) {
        // Wider results are written as 8-byte pieces.
        for piece in 0..bytes / 8 {
            // SAFETY: inside the buffer, checked above.
            let at = unsafe { data.add((offset + i * bytes + piece * 8) as usize) };
            let what = if piece == 0 { what } else { Written::Value(0) };
            defer_write(this, at as u64, what, 8);
        }
    }
}

extern "system" fn set_pipeline(this: This, pipeline: This) {
    let _call = super::counters::call(super::counters::PIPELINE);
    super::pipeline::bind(this, pipeline);
    with_state(this, |_, state| state.set_pipeline(pipeline));
}

/// Pipeline syncs: waits for earlier work and cache flushes, at the top or
/// bottom of the pipe `(flags, control, address, size)`. D3D12 does not
/// order work inside a list without barriers (the console's sync is how the
/// title orders its dependent passes: meshlet culling's chain, a test run), so
/// each is a global UAV barrier: earlier shader writes finish and are
/// visible before later work.
extern "system" fn sync(this: This, _flags: u32, _control: u32, _address: u64, _size: u64) {
    if let Some(list) = host_list(this) {
        recorded(BARRIER);
        list.crumbs().note(|| "title pipeline sync (global UAV barrier)".into());
        super::counters::UAV_ADDED[super::counters::UAV_TITLE_SYNCS].fetch_add(1, Ordering::Relaxed);
        // SAFETY: the same layout; no resource named.
        let barrier = unsafe { std::mem::transmute::<Barrier, D3D12_RESOURCE_BARRIER>(UAV_BARRIER) };
        // SAFETY: an open list.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
}

/// Reset with console flags `(allocator, pipeline, flags)`; the flags only
/// tune the console's command memory.
extern "system" fn reset_x(this: This, allocator: This, pipeline: This, _flags: u32) -> HRESULT {
    reset(this, allocator, pipeline)
}

/// Newer lists' 0x4a0 `(size, alignment) -> pointer`: memory the list
/// owns until it is reset, which the CPU writes and the GPU reads at the
/// same address; null on failure. The title's BVH builder puts its
/// instance descriptions there.
extern "system" fn allocate_upload(this: This, size: u64, alignment: u64) -> *mut u8 {
    let align = usize::try_from(alignment).unwrap_or(0).max(16);
    let Some(len) = usize::try_from(size).ok().and_then(|s| s.checked_add(align)) else {
        return std::ptr::null_mut();
    };
    if this.is_null() || !align.is_power_of_two() {
        return std::ptr::null_mut();
    }
    let uploads = &ours(this).uploads;
    // A guarded block, its end flush against a no-access page.
    if let Some(at) = super::guard::alloc("upload memory", size as usize, align) {
        // SAFETY: fresh memory of ours, `size` bytes.
        unsafe { std::ptr::write_bytes(at, 0, size as usize) };
        uploads.lock().unwrap_or_else(|e| e.into_inner()).push(Upload::Guarded(at as usize));
        return at;
    }
    let mut block = vec![0u8; len];
    let start = block.as_mut_ptr();
    // SAFETY: the offset is below `align`, so `size` bytes fit after it.
    let at = unsafe { start.add(start.align_offset(align)) };
    // Moving the vector keeps its heap block where it is.
    uploads.lock().unwrap_or_else(|e| e.into_inner()).push(Upload::Heap(block));
    at
}

/// Draw state the host has no use for (root signatures: ours come with
/// the pipeline; depth bounds, depth order and shading rate hints): any
/// arguments, nothing returned.
extern "system" fn ignored_state(_: This) {}

/// The title's draw state, kept for its draws (`draw.rs`) or set on the
/// host list at once. Without a host GPU these do nothing.
mod state {
    use super::super::draw::{self, IndexView, VertexView};
    use super::{This, host_list, with_state};
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT;

    /// `count` items at `items`, or none.
    fn items<'a, T>(items: *const T, count: u32) -> &'a [T] {
        if items.is_null() || count == 0 {
            return &[];
        }
        // SAFETY: the title passes `count` items.
        unsafe { std::slice::from_raw_parts(items, count as usize) }
    }

    pub(super) extern "system" fn topology(this: This, topology: u32) {
        with_state(this, |list, s| s.set_topology(list, topology));
    }
    pub(super) extern "system" fn viewports(this: This, count: u32, viewports: *const D3D12_VIEWPORT) {
        with_state(this, |list, s| s.set_viewports(list, items(viewports, count)));
    }
    pub(super) extern "system" fn scissors(this: This, count: u32, rects: *const RECT) {
        with_state(this, |list, s| s.set_scissors(list, items(rects, count)));
    }
    pub(super) extern "system" fn blend_factor(this: This, factor: *const [f32; 4]) {
        if let Some(list) = host_list(this) {
            // SAFETY: the title passes four floats or null.
            draw::blend_factor(list, unsafe { factor.as_ref() });
        }
    }
    pub(super) extern "system" fn stencil_ref(this: This, reference: u32) {
        if let Some(list) = host_list(this) {
            draw::stencil_ref(list, reference);
        }
    }
    pub(super) extern "system" fn heaps(this: This, count: u32, heaps: *const This) {
        let _call = super::super::counters::call(super::super::counters::HEAPS);
        with_state(this, |_, s| s.set_heaps(items(heaps, count)));
    }
    pub(super) extern "system" fn compute_table(this: This, slot: u32, handle: u64) {
        let _call = super::super::counters::call(super::super::counters::TABLE);
        with_state(this, |_, s| s.set_table(true, slot, handle));
    }
    pub(super) extern "system" fn graphics_table(this: This, slot: u32, handle: u64) {
        let _call = super::super::counters::call(super::super::counters::TABLE);
        with_state(this, |_, s| s.set_table(false, slot, handle));
    }
    pub(super) extern "system" fn compute_constant(this: This, slot: u32, value: u32, offset: u32) {
        with_state(this, |_, s| s.set_constants(true, slot, &[value], offset));
    }
    pub(super) extern "system" fn graphics_constant(this: This, slot: u32, value: u32, offset: u32) {
        with_state(this, |_, s| s.set_constants(false, slot, &[value], offset));
    }
    pub(super) extern "system" fn compute_constants(this: This, slot: u32, count: u32, values: *const u32, offset: u32) {
        with_state(this, |_, s| s.set_constants(true, slot, items(values, count), offset));
    }
    pub(super) extern "system" fn graphics_constants(this: This, slot: u32, count: u32, values: *const u32, offset: u32) {
        with_state(this, |_, s| s.set_constants(false, slot, items(values, count), offset));
    }
    /// Root constant buffer, shader resource and unordered access views.
    pub(super) extern "system" fn compute_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(true, slot, address));
    }
    pub(super) extern "system" fn graphics_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(false, slot, address));
    }
    pub(super) extern "system" fn index_buffer(this: This, view: *const IndexView) {
        // SAFETY: the title passes a view or null.
        with_state(this, |_, s| s.set_index_buffer(unsafe { view.as_ref() }));
    }
    pub(super) extern "system" fn vertex_buffers(this: This, start: u32, count: u32, views: *const VertexView) {
        with_state(this, |_, s| {
            if views.is_null() {
                s.clear_vertex_buffers(start, count);
            } else {
                s.set_vertex_buffers(start, items(views, count));
            }
        });
    }
    /// `(count, handles, one range, depth)`: with one range, the handles
    /// follow the first at the render target view size (128 bytes).
    pub(super) extern "system" fn targets(this: This, count: u32, handles: *const u64, one_range: i32, depth: *const u64) {
        let _call = super::super::counters::call(super::super::counters::TARGETS);
        let handles: Vec<u64> = if one_range != 0 && !handles.is_null() {
            // SAFETY: the title passes the first handle.
            let first = unsafe { *handles };
            (0..u64::from(count)).map(|i| first + 128 * i).collect()
        } else {
            items(handles, count).to_vec()
        };
        // SAFETY: the title passes a handle or null.
        let depth = unsafe { depth.as_ref() }.copied().unwrap_or(0);
        with_state(this, |list, s| s.set_targets(list, &handles, depth));
    }
    pub(super) extern "system" fn draw(this: This, vertices: u32, instances: u32, first: u32, first_instance: u32) {
        super::count_draw();
        let _call = super::super::counters::call(super::super::counters::DRAW);
        super::super::pipeline::used(this, "draw");
        with_state(this, |list, s| {
            if let Some(_drawing) = super::kept_draw(list, s) {
                s.draw(list, vertices, instances, first, first_instance)
            } else {
                // A test run: a draw ui=keep drops still feeds the recorder (our pass reads it).
                s.tap(xnw_formats::framepack::op::DRAW, vertices, [instances, first, 0, first_instance]);
            }
        });
    }
    pub(super) extern "system" fn dispatch(this: This, x: u32, y: u32, z: u32) {
        super::count_dispatch();
        super::super::pipeline::used(this, "dispatch");
        with_state(this, |list, s| {
            s.tap_dispatch(x, y, z);
            if super::kept(list, super::super::uikeep::Command::Dispatch) {
                s.dispatch(list, x, y, z);
                super::super::uikeep::done();
            }
        });
    }
    /// `DispatchMesh(x, y, z)` (the PC method of List6); the title's mesh
    /// programs take `x` groups only.
    pub(super) extern "system" fn dispatch_mesh(this: This, x: u32, y: u32, z: u32) {
        super::count_draw();
        let _call = super::super::counters::call(super::super::counters::DRAW_MESH);
        super::super::pipeline::note_mesh_draw();
        if y * z != 1 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| xnw_core::warn!("d3d12x: a mesh dispatch with y or z groups ({x}, {y}, {z})"));
        }
        with_state(this, |list, s| {
            if super::kept(list, super::super::uikeep::Command::Mesh) {
                s.dispatch_mesh(list, x, false)
            } else {
                // A test run: no ped drawn under ui=keep (PreSkinD sinks 0): the dropped mesh draw never reached the tap.
                s.tap(xnw_formats::framepack::op::MESH, x, [0; 4]);
            }
        });
    }
    /// `DispatchMeshX(x, flags, indirect arguments, payload)`; indirect
    /// arguments stay on the GPU (`State::dispatch_mesh_x_indirect`).
    pub(super) extern "system" fn dispatch_mesh_x(this: This, x: u32, _flags: u64, indirect: u64, _payload: u64) {
        super::count_draw();
        let _call = super::super::counters::call(super::super::counters::DRAW_MESH);
        super::super::pipeline::note_mesh_draw();
        if indirect != 0 {
            with_state(this, |list, s| {
                if super::kept(list, super::super::uikeep::Command::Mesh) {
                    s.dispatch_mesh_x_indirect(list, indirect)
                } else {
                    s.tap(xnw_formats::framepack::op::MESH_X, 0, [0; 4]);
                }
            });
            return;
        }
        with_state(this, |list, s| {
            if super::kept(list, super::super::uikeep::Command::Mesh) {
                s.dispatch_mesh(list, x, false)
            } else {
                // A test run: no ped drawn under ui=keep (PreSkinD sinks 0): the dropped mesh draw never reached the tap.
                s.tap(xnw_formats::framepack::op::MESH, x, [0; 4]);
            }
        });
    }
    pub(super) extern "system" fn draw_indexed(this: This, indices: u32, instances: u32, first: u32, base: i32, first_instance: u32) {
        super::count_draw();
        let _call = super::super::counters::call(super::super::counters::DRAW_INDEXED);
        super::super::pipeline::used(this, "draw");
        with_state(this, |list, s| {
            if let Some(_drawing) = super::kept_draw(list, s) {
                s.draw_indexed(list, indices, instances, first, base, first_instance)
            } else {
                s.tap(xnw_formats::framepack::op::INDEXED, indices, [instances, first, base as u32, first_instance]);
            }
        });
    }
}

/// `ExecuteIndirect(signature, max count, arguments, offset, count buffer,
/// offset)`. The title's signatures (title+0x2479466, 2026-11-24)
/// hold one argument each: draw (stride 16), indexed draw (20), dispatch
/// (12), mesh dispatch (12), and the console's single-pass dispatch, type
/// 105 (stride 12: the three group counts; its flags live in the
/// signature, the title passes none), run as a plain dispatch. They run
/// through the host's indirect execution (`State::execute_indirect`).
/// shortcut: signatures that change root arguments are dropped (logged
/// once per signature).
///
/// Base list.rs 1135-1196, `tracing` -> `xnw_core`.
extern "system" fn execute_indirect(this: This, signature: This, max: u32, arguments: This, offset: u64, count: This, count_offset: u64) {
    count_draw();
    let _call = super::counters::call(super::counters::DRAW_INDIRECT);
    let Some((stride, args)) = super::signature::layout(signature) else {
        return;
    };
    let address = super::resource::buffer_bytes(arguments).map(|(p, _)| p as u64 + offset);
    let counter = super::resource::buffer_bytes(count).map(|(p, _)| p as u64 + count_offset);
    let mut ran = false;
    if let ([[kind, ..]], Some(address)) = (args, address)
        && (count.is_null() || counter.is_some())
    {
        const DISPATCH: u32 = 2;
        const DISPATCH_SINGLE: u32 = 105;
        let kind = if *kind == DISPATCH_SINGLE { DISPATCH } else { *kind };
        with_state(this, |list, s| {
            // The arguments as the title gave them (bytes read at report time).
            let bytes = match kind { 0 => 16, 1 => 20, _ => 12 };
            list.crumbs().note_args(
                || format!("execute indirect: kind {kind}, stride {stride}, max {max}, title arguments at {address:#x} (resource {:#x} + {offset:#x}), count {}", arguments as usize, counter.map_or("none".into(), |c| format!("at {c:#x} (resource {:#x} + {count_offset:#x})", count as usize))),
                address,
                bytes,
            );
            // A dropped one counts as run (it is not "dropped" for want of support).
            ran = if kept(list, Command::Indirect) {
                s.execute_indirect(list, (kind, stride, max), address, counter)
            } else {
                // A test run: an indirect ui=keep drops still feeds the recorder, as execute_indirect does first.
                let op = match kind {
                    10 => xnw_formats::framepack::op::INDIRECT_MESH,
                    2 => xnw_formats::framepack::op::INDIRECT_DISPATCH,
                    _ => xnw_formats::framepack::op::INDIRECT,
                };
                let args = if kind == 2 { [address as u32, (address >> 32) as u32, stride, 0] } else { [0; 4] };
                s.tap(op, max, args);
                true
            };
        });
    }
    if ran {
        return;
    }
    INDIRECT.fetch_add(1, Ordering::Relaxed);
    static SEEN: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&(signature as usize)) {
        seen.push(signature as usize);
        xnw_core::info!(
            "d3d12x: indirect execution dropped (first time for this signature): stride {stride}, arguments {args:?}, max {max}, address {:?}, counted {}",
            address.map(|a| format!("{a:#x}")),
            !count.is_null()
        );
    }
}

/// Indirect executions dropped since the last call.
pub(crate) fn take_indirect() -> u64 {
    INDIRECT.swap(0, Ordering::Relaxed)
}
static INDIRECT: AtomicU64 = AtomicU64::new(0);

static DRAWS: AtomicU64 = AtomicU64::new(0);
static DISPATCHES: AtomicU64 = AtomicU64::new(0);
fn count_draw() {
    DRAWS.fetch_add(1, Ordering::Relaxed);
}
fn count_dispatch() {
    let _call = super::counters::call(super::counters::DISPATCH);
    DISPATCHES.fetch_add(1, Ordering::Relaxed);
}

/// What the frame line in `queue.rs` reports: draws (any kind) and
/// dispatches the title made, and copies, clears and resolves not recorded.
pub(crate) struct Counts {
    pub draws: u64,
    pub dispatches: u64,
    pub other: u64,
}

/// Commands since the last call.
pub(crate) fn take_counts() -> Counts {
    Counts { draws: DRAWS.swap(0, Ordering::Relaxed), dispatches: DISPATCHES.swap(0, Ordering::Relaxed), other: OTHER.swap(0, Ordering::Relaxed) }
}


/// VIctory: the 32-bit (slot 71) and 64-bit (slot 72) memory
/// waits `(address, comparison, reference, flags)`. The comparison is a
/// D3D12_COMPARISON_FUNC (7, greater or equal, at the retail call sites
/// VIctory found). A list that waits on a value another list writes
/// must not run first: submissions run one at a time, so the wait is kept
/// and met before its list is submitted (`queue::execute`); a no-op lets
/// the consumer read before its producer writes (VIctory's
/// reproduction: the result read 0 instead of the payload).
extern "system" fn wait32(this: This, address: u64, comparison: u32, reference: u64) {
    if let Some(list) = host_list(this) {
        list.defer_wait(address, comparison, reference & 0xffff_ffff, false);
    }
}
extern "system" fn wait64(this: This, address: u64, comparison: u32, reference: u64) {
    if let Some(list) = host_list(this) {
        list.defer_wait(address, comparison, reference, true);
    }
}

/// Whether `value` passes `comparison` (a D3D12_COMPARISON_FUNC) against
/// `reference`.
pub(crate) fn compare(comparison: u32, value: u64, reference: u64) -> bool {
    match comparison {
        1 => false,
        2 => value < reference,
        3 => value == reference,
        4 => value <= reference,
        5 => value > reference,
        6 => value != reference,
        7 => value >= reference,
        _ => true,
    }
}

/// The enhanced barriers `(group count, groups)`: buffer, texture and
/// global barriers with layouts instead of states. Every host use of a
/// title texture (clears, copies) moves its host copy to the state it needs
/// first, so only their ordering matters: one barrier on every shader
/// write (the title's buffer barriers order compute passes).
extern "system" fn barrier_groups(this: This, count: u32, _groups: *const u8) {
    let _call = super::counters::call(super::counters::BARRIER);
    if count > 0
        && let Some(list) = host_list(this)
    {
        recorded(BARRIER);
        list.crumbs().note(|| format!("title barrier groups ({count}; one global UAV barrier)"));
        // SAFETY: the same layout; no resource named.
        let barrier = unsafe { std::mem::transmute::<Barrier, D3D12_RESOURCE_BARRIER>(UAV_BARRIER) };
        // SAFETY: an open list.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
}

/// The bottom-of-pipe value writes `(address, value, flush, flags)`, 32 and
/// 64 bits: done when the list has run, as the GPU would. Flag 1 writes the
/// GPU timestamp instead (64-bit form), flag 2 the system timestamp.
extern "system" fn write32(this: This, address: u64, value: u32, _flush: u32, _flags: u32) {
    defer_write(this, address, Written::Value(u64::from(value)), 4);
}
extern "system" fn write64(this: This, address: u64, value: u64, _flush: u32, flags: u32) {
    const GPU_TIMESTAMP: u32 = 1;
    const SYSTEM_TIMESTAMP: u32 = 2;
    let what = match flags {
        GPU_TIMESTAMP => Written::GpuTimestamp,
        SYSTEM_TIMESTAMP => Written::SystemTimestamp,
        _ => Written::Value(value),
    };
    defer_write(this, address, what, 8);
}

fn defer_write(this: This, address: u64, what: Written, bytes: usize) {
    if address == 0 {
        return;
    }
    if let Some(list) = host_list(this) {
        list.defer_write(address as *mut u8, what, bytes);
        super::queue::writer_note(address, super::queue::Writer::Recorded);
        return;
    }
    // Without a host GPU nothing runs later: write now.
    what.write(address as *mut u8, bytes);
}

/// IUnknown 3, object 4, device child 1, allocator 2.
pub(crate) const ALLOCATOR_SLOTS: &[Slot] = &[
    (0, allocator_query as *const (), Done, "list.rs:693"),
    (2, allocator_release as *const (), Done, "list.rs:694"),
    (7, object::parent_device as *const (), Done, "list.rs:695"),
    (8, allocator_reset as *const (), Done, "list.rs:696"),
    (9, object::ignore as *const (), Done, "list.rs:697"),
];
pub(crate) const ALLOCATOR_LEN: usize = 24;
static ALLOCATOR_FUNCTIONS: Table<ALLOCATOR_LEN> = object::table::<ALLOCATOR, ALLOCATOR_LEN>(true, ALLOCATOR_SLOTS);

/// The list methods by their slot in older titles' lists (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, list 1, graphics list 96
/// (59 as on PC, then console ones), then versions 1-6. Sources are base
/// `list.rs` lines of its method table (1280-1344). Mesh dispatches run
/// the compute fallback.
const METHODS: [Slot; 61] = [
    (0, list_query as *const (), Done, "list.rs:1280"),
    (2, list_release as *const (), Done, "list.rs:1281"),
    (7, object::parent_device as *const (), Done, "list.rs:1282"),
    (8, list_type as *const (), Done, "list.rs:1283"),
    (9, close as *const (), Done, "list.rs:1284"),
    (10, reset as *const (), Done, "list.rs:1285"),
    (12, state::draw as *const (), Done, "list.rs:1286 instanced draw"),
    (13, state::draw_indexed as *const (), Done, "list.rs:1287 indexed draw"),
    (14, state::dispatch as *const (), Done, "list.rs:1288 dispatch"),
    (15, copy_buffer_region as *const (), Done, "list.rs:1289 copy buffer region"),
    (16, copy_texture_region as *const (), Done, "list.rs:1290 copy texture region"),
    (17, copy_resource as *const (), Done, "list.rs:1291 copy resource"),
    (19, resolve as *const (), Done, "list.rs:1292 resolve"),
    (20, state::topology as *const (), Done, "list.rs:1293 topology"),
    (21, state::viewports as *const (), Done, "list.rs:1294 viewports"),
    (22, state::scissors as *const (), Done, "list.rs:1295 scissors"),
    (23, state::blend_factor as *const (), Done, "list.rs:1296 blend factor"),
    (24, state::stencil_ref as *const (), Done, "list.rs:1297 stencil ref"),
    (25, set_pipeline as *const (), Done, "list.rs:1298 pipeline"),
    (26, barrier as *const (), Done, "list.rs:1299 barrier"),
    (32, state::heaps as *const (), Done, "list.rs:1300 descriptor heaps"),
    (33, ignored_state as *const (), Done, "list.rs:1301 compute root signature"),
    (34, ignored_state as *const (), Done, "list.rs:1302 graphics root signature"),
    (35, state::compute_table as *const (), Done, "list.rs:1303 compute table"),
    (36, state::graphics_table as *const (), Done, "list.rs:1304 graphics table"),
    (37, state::compute_constant as *const (), Done, "list.rs:1305 compute constant"),
    (38, state::graphics_constant as *const (), Done, "list.rs:1306 graphics constant"),
    (39, state::compute_constants as *const (), Done, "list.rs:1307 compute constants"),
    (40, state::graphics_constants as *const (), Done, "list.rs:1308 graphics constants"),
    (41, state::compute_view as *const (), Done, "list.rs:1309 compute CBV"),
    (42, state::graphics_view as *const (), Done, "list.rs:1310 graphics CBV"),
    (43, state::compute_view as *const (), Done, "list.rs:1311 compute SRV"),
    (44, state::graphics_view as *const (), Done, "list.rs:1312 graphics SRV"),
    (45, state::compute_view as *const (), Done, "list.rs:1313 compute UAV"),
    (46, state::graphics_view as *const (), Done, "list.rs:1314 graphics UAV"),
    (47, state::index_buffer as *const (), Done, "list.rs:1315 index buffer"),
    (48, state::vertex_buffers as *const (), Done, "list.rs:1316 vertex buffers"),
    (50, state::targets as *const (), Done, "list.rs:1317 render targets"),
    (51, clear_dsv as *const (), Done, "list.rs:1318 clear depth"),
    (52, clear_rtv as *const (), Done, "list.rs:1319 clear render target"),
    (59, execute_indirect as *const (), Done, "list.rs:1320 execute indirect"),
    (60, copy_memory as *const (), Done, "list.rs:1321 copy memory"),
    (68, ignored_state as *const (), Done, "list.rs:1322 depth bounds"),
    (69, write32 as *const (), Done, "list.rs:1323"),
    (70, write64 as *const (), Done, "list.rs:1324"),
    (71, wait32 as *const (), Done, "list.rs:1325 32-bit memory wait"),
    (72, wait64 as *const (), Done, "list.rs:1326 64-bit memory wait"),
    (73, ignored_state as *const (), Done, "list.rs:1327 depth order"),
    (83, copy_texture_region_x as *const (), Done, "list.rs:1328 copy texture region X"),
    (84, copy_resource_x as *const (), Done, "list.rs:1329 copy resource X"),
    (124, ignored_state as *const (), Done, "list.rs:1330 shading rate"),
    (127, state::dispatch_mesh as *const (), Done, "list.rs:1331 mesh dispatch (fallback)"),
    (28, query_mark as *const (), Done, "list.rs:1335 begin query"),
    (29, query_mark as *const (), Done, "list.rs:1336 end query"),
    (30, resolve_queries as *const (), Done, "list.rs:1337 resolve queries"),
    (78, dispatch_x as *const (), Done, "list.rs:1338 dispatch X"),
    (98, state::dispatch_mesh_x as *const (), Done, "list.rs:1339 mesh dispatch X (fallback)"),
    (99, driver_hint as *const (), Done, "list.rs:1340"),
    (129, sync as *const (), Done, "list.rs:1342"),
    (130, sync as *const (), Done, "list.rs:1343"),
    (137, reset_x as *const (), Done, "list.rs:1344"),
];

/// Newer titles' lists have two more methods in the console part of the
/// base list, after the clears and before ExecuteIndirect, which moved
/// from 0x1d8 to 0x1e8: every slot from 0x1d8 on is two later (inferred
/// from the title's call shapes: ExecuteIndirect 0x1e8, bottom-of-pipe
/// writes 0x238/0x240, waits 0x248/0x250, ResetX 0x458).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 59 { slot + 2 } else { slot }
}

/// Methods only newer titles' lists have. 0x490: the enhanced barrier
/// `(group count, groups)`; 0x470 `(u64)`, called from the title's state
/// flush with its result unused (purpose unknown); 0x4a0 upload memory.
const NEWER_ONLY: [Slot; 3] = [
    (142, ignored_state as *const (), Done, "list.rs:1365 unknown (u64)"),
    (146, barrier_groups as *const (), Done, "list.rs:1366 barrier groups"),
    (148, allocate_upload as *const (), Done, "list.rs:1367"),
];

const fn layout(newer: bool) -> [Slot; 64] {
    let mut m = [(0, std::ptr::null(), Done, ""); 64];
    let mut i = 0;
    while i < METHODS.len() {
        m[i] = METHODS[i];
        if newer {
            m[i].0 = newer_slot(m[i].0);
        }
        i += 1;
    }
    // Old lists: the newer-only slots stay naming stubs (slot 0 again).
    let mut j = 0;
    while j < NEWER_ONLY.len() {
        m[i + j] = if newer { NEWER_ONLY[j] } else { METHODS[0] };
        j += 1;
    }
    m
}

pub(crate) const SLOTS: [Slot; 64] = layout(false);
pub(crate) const NEWER_SLOTS: [Slot; 64] = layout(true);
pub(crate) const LEN: usize = 160;
static FUNCTIONS: Table<LEN> = object::table::<LIST, LEN>(true, &SLOTS);
static NEWER_FUNCTIONS: Table<LEN> = object::table::<LIST, LEN>(true, &NEWER_SLOTS);

#[cfg(test)]
mod tests {
    use super::*;

    /// Slot `offset` of one of our objects as `F`.
    fn slot<F: Copy>(object: This, offset: usize) -> F {
        // SAFETY: the tests call our own objects with their slots' signatures.
        unsafe { crate::d3d12x::object::slot(object, offset) }
    }

    /// The comparisons of the list memory waits (D3D12_COMPARISON_FUNC).
    #[test]
    fn memory_wait_comparisons() {
        assert!(compare(7, 2, 1) && compare(7, 1, 1) && !compare(7, 0, 1), "greater or equal");
        assert!(compare(3, 5, 5) && !compare(3, 4, 5));
        assert!(compare(2, 1, 2) && !compare(2, 2, 2));
        assert!(!compare(1, 0, 0) && compare(8, 0, 1));
    }

    #[test]
    fn every_inline_command_falls_back_to_the_table() {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator), S_OK);
        let mut list = null;
        let hr = create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list);
        assert_eq!(hr, S_OK);
        // What the title's inline draw checks: put < draw limit.
        // SAFETY: the list is 0x560 bytes.
        let (put, draw_limit, api_limit) = unsafe {
            let at = |o: usize| list.cast::<u8>().add(o).cast::<u64>().read();
            (at(0x18), at(0x28), at(0x30))
        };
        assert!(put >= draw_limit && put >= api_limit);
        // The console's ID3D12GraphicsCommandList6, as the title asks.
        let mut same = null;
        assert_eq!(list_query(list, &LIST_IIDS[7], &mut same), S_OK);
        assert_eq!(same, list);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x10)(list), 1);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x40)(list), 0);
        assert_eq!(slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list), S_OK);
        // List-owned upload memory: aligned, writable, gone at reset.
        // The pad after +0x560 starts zero; bytes the title leaves
        // there are found by range, and our part after it is untouched.
        assert_eq!(pad_range(list), None, "the pad starts zero");
        // SAFETY: inside the list's pad.
        unsafe {
            list.cast_mut().cast::<u8>().add(0x570).write(1);
            list.cast_mut().cast::<u8>().add(0x5a3).write(2);
        }
        assert_eq!(pad_range(list), Some((0x570, 0x5a3)));
        assert!(ours_at() >= 0x560 + 1024, "our part after the whole pad");
        // SAFETY: as above.
        unsafe { list.cast_mut().cast::<u8>().add(0x570).write_bytes(0, 0x40) };
        let p = allocate_upload(list, 40, 0x100);
        assert!(!p.is_null() && (p as usize).is_multiple_of(0x100));
        // SAFETY: 40 bytes were handed out.
        unsafe { p.write_bytes(0xab, 40) };
        assert!(allocate_upload(list, 8, 0x30).is_null(), "not a power of two");
        assert_eq!(reset(list, allocator, null), S_OK);
        assert!(ours(list).uploads.lock().unwrap().is_empty());
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }
}
