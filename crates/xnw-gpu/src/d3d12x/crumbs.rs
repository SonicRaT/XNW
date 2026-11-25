//! Our own GPU breadcrumbs (`XNW_GPU_CRUMBS=1`), as DRED's are made
//! (`XNW_GPU_DEBUG=dred` is not used: it took the machine down). Each op we
//! record on a host list (draw, dispatch, indirect execution, copy, clear,
//! resolve) is bracketed by two `WriteBufferImmediate` of its 32-bit id into
//! the list's slot pair of a small buffer: `MARKER_IN` where the op starts,
//! `MARKER_OUT` once it is done. The buffer is placed on a heap opened over
//! memory of ours (`OpenExistingHeapFromAddress`, as `shared.rs` does for
//! title memory; `cargo x gpu-check --shared-memory` showed GPU writes land
//! in process memory on this RX 5700 XT), so after a removal or hang the CPU
//! reads it as plain memory. A CPU ring names each op id; on a failure
//! [`report`] logs, per list of the last submissions, the last op started
//! and the last finished: one started and not finished is the op the GPU
//! was in.
//!
//! Preludes (uploads, state moves) and the presenter's list have slots
//! of their own; every list writes a "list begin" op first and a "list end"
//! op last, and the CPU zeroes the slots of a submission's lists before it
//! runs, so 0/0 means the list never ran on the GPU. Slot 0 is the
//! submission pair, written by two tiny lists of ours around the
//! submission's lists: [submission started, submission finished].
//!
//! `XNW_GPU_SYNC=1`: every submission is already waited for; this adds a
//! 2 s watchdog line naming the submission that is slow (host.rs).
//!
//! Cost unset: one cached flag test per op. Set: two immediate writes per
//! op on the GPU, one lock and one string per op on the CPU.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::core::Interface;

/// Lists with a slot pair of their own (ids taken modulo).
const SLOTS: u32 = 4096;
/// Op descriptions kept (ids taken modulo).
const OPS: usize = 1 << 16;
/// Submissions remembered for the report.
const SUBMISSIONS: usize = 8;

/// `XNW_GPU_CRUMBS=1` (or `XNW_GPU_STEP`, which needs them).
pub(crate) fn on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        let on = std::env::var("XNW_GPU_CRUMBS").is_ok_and(|v| v.trim() == "1") || step_spec().is_some();
        if on {
            xnw_core::info!("d3d12x: GPU breadcrumbs on (XNW_GPU_CRUMBS=1)");
        }
        on
    });
    *ON
}

/// `XNW_GPU_SYNC=1`.
pub(crate) fn sync() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_GPU_SYNC").is_ok_and(|v| v.trim() == "1"));
    *ON
}

/// The breadcrumb buffer: our memory, a heap over it, a buffer on the heap.
struct Buffer {
    memory: *mut u32,
    gpu: u64,
    _heap: ID3D12Heap,
    _buffer: ID3D12Resource,
}
// SAFETY: D3D12 objects are free-threaded; the memory is ours for the run.
unsafe impl Send for Buffer {}
// SAFETY: as above.
unsafe impl Sync for Buffer {}

fn buffer() -> Option<&'static Buffer> {
    static BUFFER: OnceLock<Option<Buffer>> = OnceLock::new();
    BUFFER
        .get_or_init(|| {
            let device = &super::host::host()?.device;
            let made = make_buffer(device);
            match &made {
                Ok(b) => xnw_core::info!("d3d12x: breadcrumbs at {:#x} (GPU {:#x}), {SLOTS} list slots", b.memory as usize, b.gpu),
                Err(e) => xnw_core::warn!("d3d12x: no breadcrumb buffer: {e}"),
            }
            made.ok()
        })
        .as_ref()
}

fn make_buffer(device: &ID3D12Device) -> windows::core::Result<Buffer> {
    let (memory, gpu, heap, buffer) = process_buffer(device, 64 << 10, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER)?;
    Ok(Buffer { memory: memory.cast(), gpu, _heap: heap, _buffer: buffer })
}

/// A buffer placed over `bytes` of our own memory (a heap opened over a
/// VirtualAlloc of ours), which the CPU reads after a removal or hang:
/// its memory, GPU address, heap and buffer. Also the loop counts'.
pub(crate) fn process_buffer(device: &ID3D12Device, bytes: usize, flags: D3D12_RESOURCE_FLAGS) -> windows::core::Result<(*mut u8, u64, ID3D12Heap, ID3D12Resource)> {
    let bytes = bytes.next_multiple_of(64 << 10);
    // SAFETY: a fresh allocation of ours, kept for the run.
    let memory = unsafe { xnw_core::win::VirtualAlloc(std::ptr::null(), bytes, xnw_core::win::MEM_COMMIT | xnw_core::win::MEM_RESERVE, xnw_core::win::PAGE_READWRITE) };
    if memory.is_null() {
        return Err(windows::core::Error::empty());
    }
    let d3: ID3D12Device3 = device.cast()?;
    // SAFETY: committed memory of ours at its allocation base.
    let heap: ID3D12Heap = unsafe { d3.OpenExistingHeapFromAddress(memory.cast_const())? };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: the description outlives the call.
    unsafe { device.CreatePlacedResource(&heap, 0, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)? };
    let buffer = buffer.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: a plain query.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    Ok((memory.cast(), gpu, heap, buffer))
}

/// A host list's breadcrumb state: its slot pair, its `List2` interface,
/// the pipeline it bound last.
pub(crate) struct ListCrumbs {
    slot: u32,
    /// What the list recorded since its begin, in order (crumbs on):
    /// ops (their id) and the records between them (id 0).
    ring: std::sync::Arc<Mutex<Ring>>,
    list2: OnceLock<Option<ID3D12GraphicsCommandList2>>,
    pipeline: Mutex<String>,
}

impl ListCrumbs {
    /// Slots from 1 (slot 0 is the submission pair).
    pub(crate) fn new() -> ListCrumbs {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let slot = 1 + NEXT.fetch_add(1, Ordering::Relaxed) % (SLOTS - 1);
        let ring = std::sync::Arc::new(Mutex::new(Ring::default()));
        RINGS.lock().unwrap_or_else(|e| e.into_inner()).insert(slot, std::sync::Arc::downgrade(&ring));
        ListCrumbs { slot, ring, list2: OnceLock::new(), pipeline: Mutex::new(String::new()) }
    }

    /// A record between ops (crumbs on): a barrier, a transition, a
    /// pipeline bind, a root view, an indirect execution's arguments...
    pub(crate) fn note(&self, what: impl FnOnce() -> String) {
        if on() {
            self.ring.lock().unwrap_or_else(|e| e.into_inner()).push(Record { op: 0, text: what(), args: None });
        }
    }

    /// An indirect execution's argument bytes (`len` at title address
    /// `at`), shown at report time.
    pub(crate) fn note_args(&self, what: String, at: u64, len: usize) {
        if on() {
            self.ring.lock().unwrap_or_else(|e| e.into_inner()).push(Record { op: 0, text: what, args: Some((at, len)) });
        }
    }

    fn op(&self, id: u32) {
        self.ring.lock().unwrap_or_else(|e| e.into_inner()).push(Record { op: id, text: String::new(), args: None });
    }

    /// A marker op on `list` (begin, end, or an op of ours with no guard).
    pub(crate) fn mark(&self, list: &ID3D12GraphicsCommandList, what: &str) {
        if !on() {
            return;
        }
        if what.ends_with("begin") {
            self.ring.lock().unwrap_or_else(|e| e.into_inner()).clear();
        }
        let id = describe(format!("{what} (pipeline {})", self.pipeline.lock().unwrap_or_else(|e| e.into_inner())));
        self.op(id);
        write(self, list, id, 0, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN);
        write(self, list, id, 1, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT);
    }

    /// An op of ours on `list` (a prelude's uploads): in now, out on drop.
    pub(crate) fn begin<'a>(&'a self, list: &'a ID3D12GraphicsCommandList, what: impl FnOnce() -> String) -> Option<RawOp<'a>> {
        if !on() {
            return None;
        }
        let id = describe(what());
        self.op(id);
        write(self, list, id, 0, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN);
        Some(RawOp { crumbs: self, list, id })
    }

    /// Notes the pipeline bound now (crumbs on only).
    pub(crate) fn pipeline(&self, name: &str) {
        if on() {
            let mut p = self.pipeline.lock().unwrap_or_else(|e| e.into_inner());
            if *p != name {
                name.clone_into(&mut p);
            }
        }
    }

    pub(crate) fn slot(&self) -> u32 {
        self.slot
    }
}

/// One recorded thing of a list: an op (`op` its id) or a record
/// between ops (`op` 0, `text`), with the argument range of an indirect
/// execution to show at report time.
struct Record {
    op: u32,
    text: String,
    args: Option<(u64, usize)>,
}

/// A list's records since its begin (bounded: the oldest go).
#[derive(Default)]
struct Ring(std::collections::VecDeque<Record>);
const RING: usize = 16_384;
impl Ring {
    fn push(&mut self, r: Record) {
        if self.0.len() == RING {
            self.0.pop_front();
        }
        self.0.push_back(r);
    }
    fn clear(&mut self) {
        self.0.clear();
    }
}

/// The rings by slot (the newest list on a slot).
static RINGS: Mutex<std::collections::BTreeMap<u32, std::sync::Weak<Mutex<Ring>>>> = Mutex::new(std::collections::BTreeMap::new());

/// For a list that did not reach its end: the next `n` things it
/// recorded after op `after` (0: from its start), as report lines.
fn following(slot: u32, after: u32, n: usize) -> Vec<String> {
    let Some(ring) = RINGS.lock().unwrap_or_else(|e| e.into_inner()).get(&slot).and_then(std::sync::Weak::upgrade) else {
        return vec!["(no record of this list)".into()];
    };
    let ring = ring.lock().unwrap_or_else(|e| e.into_inner());
    let from = if after == 0 { 0 } else {
        match ring.0.iter().position(|r| r.op == after) {
            Some(i) => i + 1,
            None => return vec![format!("(op {after} is not in this list's records)")],
        }
    };
    ring.0.iter().skip(from).take(n).map(record_text).collect()
}

fn record_text(r: &Record) -> String {
    let mut s = if r.op != 0 { described(r.op) } else { r.text.clone() };
    if let Some((at, len)) = r.args {
        let mut title = vec![0u8; len];
        let read = super::host::read_into(at as *const u8, &mut title);
        let host = super::shadow::find(at, len as u64).map(|sh| sh.bytes_at(at, len));
        let hex = |b: &[u8]| b.chunks(4).map(|w| format!("{:08x}", u32::from_le_bytes(w.try_into().unwrap_or([0; 4])))).collect::<Vec<_>>().join(" ");
        s.push_str(&format!(
            "; argument bytes as the CPU sees them now (after the fact): title {}, host copy {}",
            if read { hex(&title) } else { "unreadable".into() },
            host.map_or("none".into(), |h| hex(&h))
        ));
    }
    s
}

/// Op ids handed out (0 is none) and their descriptions by id modulo [`OPS`].
static NEXT_OP: AtomicU32 = AtomicU32::new(1);
static DESCRIBED: Mutex<Vec<(u32, String)>> = Mutex::new(Vec::new());

/// The last submissions: number, fence value, and their lists' slots (main
/// list, prelude if it had one).
type Submitted = (u64, u64, Vec<(u32, Option<u32>)>);
static SUBMITTED: Mutex<std::collections::VecDeque<Submitted>> = Mutex::new(std::collections::VecDeque::new());
static SUBMISSION: AtomicU64 = AtomicU64::new(0);

/// The presenter's state for the report: 0 idle, 1 recording or
/// running its blit (a host submission), 2 inside the swap chain's Present;
/// and the frame.
static PRESENTING: AtomicU64 = AtomicU64::new(0);

/// The presenter is at `state` (see [`PRESENTING`]) for `frame`.
/// When it entered `Present` (log clock, microseconds).
pub(crate) fn presenting(state: u64, frame: u64) {
    if on() {
        if state == 2 {
            PRESENT_AT.store(xnw_core::uptime().as_micros() as u64, Ordering::Relaxed);
        }
        PRESENTING.store(state | frame << 8, Ordering::Relaxed);
    }
}
static PRESENT_AT: AtomicU64 = AtomicU64::new(0);

/// The presenter's blit list (its slot) and the frame it last blitted.
static PRESENTER_BLIT: AtomicU64 = AtomicU64::new(u64::MAX);

/// The presenter's blit list `slot` was submitted for `frame`.
pub(crate) fn presenter_blit(slot: u32, frame: u64) {
    if on() {
        PRESENTER_BLIT.store(u64::from(slot) | frame << 16, Ordering::Relaxed);
    }
}

/// The presenter part of the failure header.
fn presenter_text() -> String {
    match PRESENTING.load(Ordering::Relaxed) {
        0 => "idle".to_string(),
        v if v & 0xff == 1 => format!("in its blit for frame {}", v >> 8),
        v => {
            let at = PRESENT_AT.load(Ordering::Relaxed);
            let now = xnw_core::uptime().as_micros() as u64;
            format!("INSIDE Present for frame {}, entered at {:.3} s, {} ms ago", v >> 8, at as f64 / 1e6, now.saturating_sub(at) / 1000)
        }
    }
}

/// The two tiny lists of ours around each submission.
struct Bracket {
    allocator: ID3D12CommandAllocator,
    lists: [ID3D12GraphicsCommandList2; 2],
}
// SAFETY: used under the submission lock only.
unsafe impl Send for Bracket {}
static BRACKET: Mutex<Option<Bracket>> = Mutex::new(None);

/// The lists that write submission `n` into slot 0 (started, before
/// the submission's lists; finished, after them); `None` with crumbs off.
pub(crate) fn bracket(device: &ID3D12Device, n: u64) -> Option<[ID3D12CommandList; 2]> {
    if !on() {
        return None;
    }
    let b = buffer()?;
    let mut guard = BRACKET.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        // SAFETY: plain object creation.
        let made = unsafe {
            (|| -> windows::core::Result<Bracket> {
                let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let a: ID3D12GraphicsCommandList2 = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                a.Close()?;
                let z: ID3D12GraphicsCommandList2 = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                z.Close()?;
                Ok(Bracket { allocator, lists: [a, z] })
            })()
        };
        *guard = Some(made.ok()?);
    }
    let br = guard.as_ref()?;
    let value = n as u32;
    // SAFETY: the previous submission was waited for; our lists and buffer.
    unsafe {
        br.allocator.Reset().ok()?;
        for (which, l) in br.lists.iter().enumerate() {
            l.Reset(&br.allocator, None).ok()?;
            let param = D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: b.gpu + which as u64 * 4, Value: value };
            let mode = if which == 0 { D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN } else { D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT };
            l.WriteBufferImmediate(1, &param, Some(&mode));
            l.Close().ok()?;
        }
    }
    Some([br.lists[0].cast().ok()?, br.lists[1].cast().ok()?])
}

/// An op being recorded: its `MARKER_OUT` goes on the list when dropped
/// (`id` 0: no crumb), then a global UAV barrier if `barrier` or
/// stepping.
pub(crate) struct Op<'a> {
    list: &'a super::host::HostList,
    id: u32,
    barrier: bool,
}

impl Drop for Op<'_> {
    fn drop(&mut self) {
        if self.id != 0 {
            write(self.list.crumbs(), &self.list.list, self.id, 1, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT);
        }
        if self.barrier {
            self.list.crumbs().note(|| "global UAV barrier (ours, XNW_UAV_BARRIER)".into());
            uav_barrier(&self.list.list);
        } else {
            drain(&self.list.list);
        }
    }
}

/// `XNW_UAV_BARRIER`: 0 `off`, 1 `dispatch` (the default: a global UAV
/// barrier after every compute op we record and at each list's start), 2
/// `all` (after every op, as stepping does).
pub(crate) fn uav_mode() -> u8 {
    static MODE: std::sync::LazyLock<u8> = std::sync::LazyLock::new(|| match std::env::var("XNW_UAV_BARRIER").as_deref().map(str::trim) {
        Ok("off") => 0,
        Ok("all") => 2,
        Ok("dispatch") | Err(_) => 1,
        Ok(other) => {
            xnw_core::warn!("XNW_UAV_BARRIER={other}: not off, dispatch or all; dispatch");
            1
        }
    });
    *MODE
}

pub(crate) fn uav_mode_name() -> &'static str {
    ["off", "dispatch", "all"][uav_mode() as usize]
}

/// A global UAV barrier on `list`, counted.
fn uav_barrier(list: &ID3D12GraphicsCommandList) {
    super::counters::UAV_ADDED[super::counters::UAV_AFTER_OPS].fetch_add(1, Ordering::Relaxed);
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }) },
    };
    // SAFETY: an open list; a barrier on no resource (all UAV work).
    unsafe { list.ResourceBarrier(std::slice::from_ref(&barrier)) };
}

/// At a list's start (reset): work of the list before it in the same
/// submission is not ordered with this one's without a barrier.
pub(crate) fn list_start(list: &ID3D12GraphicsCommandList) {
    if uav_mode() != 0 {
        uav_barrier(list);
    }
}

/// An op of ours on a list that is not a host list's main one.
pub(crate) struct RawOp<'a> {
    crumbs: &'a ListCrumbs,
    list: &'a ID3D12GraphicsCommandList,
    id: u32,
}

impl Drop for RawOp<'_> {
    fn drop(&mut self) {
        write(self.crumbs, self.list, self.id, 1, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT);
        drain(self.list);
    }
}

/// Keeps `text` under a new op id.
fn describe(text: String) -> u32 {
    let id = NEXT_OP.fetch_add(1, Ordering::Relaxed).max(1);
    let mut d = DESCRIBED.lock().unwrap_or_else(|e| e.into_inner());
    if d.is_empty() {
        d.resize(OPS, (0, String::new()));
    }
    d[id as usize % OPS] = (id, text);
    id
}

/// Before an op recorded on `list`: its `MARKER_IN`, and its description
/// (`what`, with the list's pipeline) kept; `None` with crumbs off and no
/// barrier due after it.
pub(crate) fn begin(list: &super::host::HostList, what: impl FnOnce() -> String) -> Option<Op<'_>> {
    begin_kind(list, false, what)
}

/// [`begin`] for a compute op (dispatch, compute indirect execution, our
/// compute passes): with `XNW_UAV_BARRIER=dispatch` a barrier follows it.
pub(crate) fn begin_compute(list: &super::host::HostList, what: impl FnOnce() -> String) -> Option<Op<'_>> {
    begin_kind(list, true, what)
}

pub(crate) fn begin_kind(list: &super::host::HostList, compute: bool, what: impl FnOnce() -> String) -> Option<Op<'_>> {
    let barrier = match uav_mode() {
        2 => true,
        1 => compute,
        _ => false,
    };
    if !on() {
        return barrier.then_some(Op { list, id: 0, barrier });
    }
    let id = describe(format!("{} (pipeline {})", what(), list.crumbs().pipeline.lock().unwrap_or_else(|e| e.into_inner())));
    list.crumbs().op(id);
    write(list.crumbs(), &list.list, id, 0, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN);
    Some(Op { list, id, barrier })
}

/// Writes `id` into word `which` (0 started, 1 finished) of `c`'s slot, on
/// `list` (always the same list for one `c`).
fn write(c: &ListCrumbs, list: &ID3D12GraphicsCommandList, id: u32, which: u64, mode: D3D12_WRITEBUFFERIMMEDIATE_MODE) {
    let Some(b) = buffer() else { return };
    let Some(l2) = c.list2.get_or_init(|| list.cast().ok()) else { return };
    let param = D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: b.gpu + (u64::from(c.slot) * 2 + which) * 4, Value: id };
    // SAFETY: an open list; the destination lies in our buffer.
    unsafe { l2.WriteBufferImmediate(1, &param, Some(&mode)) };
}

/// At a submission (crumbs on), before it runs: its lists' slots (main,
/// prelude) zeroed and remembered with the fence value it signals; the
/// submission's number.
pub(crate) fn submitted(fence: u64, slots: Vec<(u32, Option<u32>)>) -> u64 {
    let n = SUBMISSION.fetch_add(1, Ordering::Relaxed) + 1;
    if let Some(b) = buffer() {
        for &(main, prelude) in &slots {
            for slot in std::iter::once(main).chain(prelude) {
                // SAFETY: our memory; the GPU writes these words only once
                // the submission runs.
                unsafe { b.memory.add(slot as usize * 2).cast::<u64>().write_volatile(0) };
            }
        }
    }
    let mut s = SUBMITTED.lock().unwrap_or_else(|e| e.into_inner());
    s.push_back((n, fence, slots));
    if s.len() > SUBMISSIONS {
        s.pop_front();
    }
    n
}

/// The description of op `id`, if still kept.
fn described(id: u32) -> String {
    if id == 0 {
        return "none".into();
    }
    let d = DESCRIBED.lock().unwrap_or_else(|e| e.into_inner());
    match d.get(id as usize % OPS) {
        Some((i, text)) if *i == id => format!("op {id} ({text})"),
        _ => format!("op {id} (no longer described)"),
    }
}

/// The words of slot `slot`: last op started, last finished.
fn read(slot: u32) -> Option<(u32, u32)> {
    let b = buffer()?;
    // SAFETY: our memory; the GPU writes it, so it is read volatile.
    unsafe {
        let at = b.memory.add(slot as usize * 2);
        Some((at.read_volatile(), at.add(1).read_volatile()))
    }
}

/// Whether our buffer's memory is still committed and readable.
fn readable(b: &Buffer) -> bool {
    // SAFETY: plain data; zero is a valid value.
    let mut info: xnw_core::win::MemoryBasicInformation = unsafe { std::mem::zeroed() };
    // SAFETY: a query about our own memory into a local.
    let n = unsafe { xnw_core::win::VirtualQueryEx(xnw_core::win::GetCurrentProcess(), b.memory.cast_const().cast(), &mut info, size_of::<xnw_core::win::MemoryBasicInformation>()) };
    n != 0 && info.state == xnw_core::win::MEM_COMMIT && info.protect & 0xee != 0
}

/// The breadcrumbs of the last submissions' lists, one line each (newest
/// submission last), with raw slot words; the submission pair; the queue
/// (`completed` fence value, the last `submitted`); the presenter.
///
/// How to read it: a list `(IN AN OP)` was running that op. All lists
/// "finished list end" and the submission pair started = finished: the GPU
/// finished our work (the hang is in what carries no crumbs: the swap
/// chain's Present, or outside our queue). A list with raw 0/0 never ran
/// (the GPU stopped before it: look at the list before). Markers lost (the
/// writes did not survive) shows as every slot of the last submission 0/0
/// with the submission pair also 0 or older than the fence says ran, or
/// the buffer reported unreadable.
pub(crate) fn report(completed: u64, submitted_value: u64) {
    if !on() {
        return;
    }
    let Some(b) = buffer() else { return };
    let ok = readable(b);
    let pair = read(0);
    xnw_core::warn!(
        "crumbs: buffer {} at {:#x}; submission pair: started {}, finished {}; fence completed {completed} of {submitted_value} signalled; presenter {}",
        if ok { "readable" } else { "NOT READABLE" },
        b.memory as usize,
        pair.map_or(0, |p| p.0),
        pair.map_or(0, |p| p.1),
        presenter_text()
    );
    if !ok {
        return;
    }
    // The presenter's own list (submitted before the title's last
    // submissions, so not among them).
    let blit = PRESENTER_BLIT.load(Ordering::Relaxed);
    if blit != u64::MAX
        && let Some((started, finished)) = read((blit & 0xffff) as u32)
    {
        xnw_core::warn!(
            "crumbs: presenter blit list for frame {}: started {}, finished {} [raw {started}/{finished}]",
            blit >> 16,
            described(started),
            described(finished)
        );
    }
    let submitted: Vec<Submitted> = SUBMITTED.lock().unwrap_or_else(|e| e.into_inner()).iter().cloned().collect();
    for (n, fence, slots) in submitted {
        let state = if fence <= completed { "done" } else if fence == completed + 1 { "IN FLIGHT" } else { "queued" };
        xnw_core::warn!("crumbs: submission {n} (fence {fence}, {state}): {} lists, {} with a prelude", slots.len(), slots.iter().filter(|s| s.1.is_some()).count());
        for (k, (main, prelude)) in slots.into_iter().enumerate() {
            for (what, slot) in std::iter::once(("list", main)).chain(prelude.map(|p| ("prelude of list", p))) {
                let Some((started, finished)) = read(slot) else { return };
                let open = started != 0 && started != finished;
                let finished_text = described(finished);
                xnw_core::warn!(
                    "crumbs: {what} {k} of submission {n}{}: started {}, finished {finished_text} [raw {started}/{finished}{}]",
                    if open { " (IN AN OP)" } else { "" },
                    described(started),
                    if started == 0 && finished == 0 { ", never ran" } else { "" }
                );
                // A list that ran but did not reach its end: what it
                // recorded next (the GPU is in or before the first of these).
                let ended = finished_text.contains("list end (") || finished_text.contains("prelude end (");
                if finished != 0 && !ended {
                    for (i, line) in following(slot, finished, 8).into_iter().enumerate() {
                        xnw_core::warn!("crumbs:   next {} after op {finished} in {what} {k}: {line}", i + 1);
                    }
                }
            }
        }
    }
}

/// `XNW_GPU_SYNC=1`: the first and last op of a submission's lists as read
/// now (for the slow-submission line).
pub(crate) fn submission_text(slots: &[(u32, Option<u32>)]) -> String {
    let words: Vec<(u32, u32)> = slots.iter().filter_map(|&(s, _)| read(s)).collect();
    match (words.first(), words.last()) {
        (Some(&(first, _)), Some(&(last, done))) => format!("first list at {}, last list at {} (finished {})", described(first), described(last), described(done)),
        _ => "no breadcrumbs".into(),
    }
}

// ---------------------------------------------------------------------
// `XNW_GPU_STEP=<trigger>`: one op at a time on the GPU, each timed.
//
// Host lists are recorded straight into D3D12 command lists (draw state is
// set on the list as the title sets it), so they cannot be cut into one
// list per op after recording without replaying the title's state. Instead,
// while stepping is armed, every op that carries a crumb is followed on its
// list by a global UAV barrier (a drain: the next op starts only once this
// one and everything before it are done), and the submission's wait polls
// the crumb slots from the CPU: the op between its MARKER_IN and MARKER_OUT
// is the one running, timed from the poll that first saw it start to the
// one that saw it finish. Ops shorter than a poll (microseconds) are not
// timed. Triggers: `at+<seconds>`, `frame+<n>` (presented frame), `world`
// (the world-start detector of capture.rs), `card` (the first presented
// frame after the first mesh draw: the mission card is the first thing the
// title draws with mesh shaders; `at+<s>` is the reliable form). It lasts
// `XNW_GPU_STEP_FRAMES` presented frames (default 3).

/// The trigger: mode (0 at+, 1 frame+, 2 world, 3 card) and its number.
fn step_spec() -> Option<(u8, u64)> {
    static SPEC: std::sync::LazyLock<Option<(u8, u64)>> = std::sync::LazyLock::new(|| {
        let v = std::env::var("XNW_GPU_STEP").ok()?;
        let v = v.trim();
        let spec = if let Some(k) = v.strip_prefix("at+") {
            (0, k.parse().ok()?)
        } else if let Some(k) = v.strip_prefix("frame+") {
            (1, k.parse().ok()?)
        } else if v == "world" {
            (2, 0)
        } else if v == "card" {
            (3, 0)
        } else {
            xnw_core::warn!("XNW_GPU_STEP={v}: not at+<s>, frame+<n>, world or card; stepping off");
            return None;
        };
        xnw_core::info!("step: armed for XNW_GPU_STEP={v}, {} frames (XNW_GPU_STEP_FRAMES)", step_frames());
        Some(spec)
    });
    *SPEC
}

fn step_frames() -> u64 {
    static N: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| std::env::var("XNW_GPU_STEP_FRAMES").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(3).max(1));
    *N
}

/// The presented frame stepping started at (`u64::MAX`: not yet), done.
static STEP_FROM: AtomicU64 = AtomicU64::new(u64::MAX);
static STEP_DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[cfg(test)]
static STEP_FORCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether ops are stepped now (checked as each op is recorded and at each
/// submission). Starts at the trigger, ends `XNW_GPU_STEP_FRAMES` later.
pub(crate) fn stepping() -> bool {
    #[cfg(test)]
    if STEP_FORCED.load(Ordering::Relaxed) {
        return true;
    }
    let Some((mode, n)) = step_spec() else { return false };
    if STEP_DONE.load(Ordering::Relaxed) {
        return false;
    }
    let now = super::queue::presented();
    let mut from = STEP_FROM.load(Ordering::Relaxed);
    if from == u64::MAX {
        let go = match mode {
            0 => xnw_core::uptime().as_secs() >= n,
            1 => now >= n,
            2 => super::capture::world_start().is_some_and(|w| now >= w),
            _ => crate::mesh::MESH_DRAWS.load(Ordering::Relaxed) > 0,
        };
        if !go {
            return false;
        }
        if STEP_FROM.compare_exchange(u64::MAX, now, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            xnw_core::warn!("step: started at presented frame {now}: one op at a time, each timed");
        }
        from = STEP_FROM.load(Ordering::Relaxed);
    }
    if now >= from + step_frames() {
        if !STEP_DONE.swap(true, Ordering::AcqRel) {
            let max = STEP_MAX_US.load(Ordering::Relaxed) as f64 / 1000.0;
            xnw_core::warn!("step: done after {} frames, no op over {max:.1} ms", step_frames());
            top_pipelines();
        }
        return false;
    }
    true
}

/// While stepping: a drain after an op (a global UAV barrier).
fn drain(list: &ID3D12GraphicsCommandList) {
    if !stepping() {
        return;
    }
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }) },
    };
    // SAFETY: an open list; a barrier on no resource (all UAV work).
    unsafe { list.ResourceBarrier(std::slice::from_ref(&barrier)) };
}

/// The longest op stepped (microseconds); per pipeline: total, longest
/// (microseconds), ops timed.
static STEP_MAX_US: AtomicU64 = AtomicU64::new(0);
static BY_PIPELINE: Mutex<Vec<(String, u64, u64, u64)>> = Mutex::new(Vec::new());

/// The op's threshold for a line (`XNW_GPU_STEP_MS`, default 100).
fn slow_ms() -> u64 {
    static MS: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| std::env::var("XNW_GPU_STEP_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(100));
    *MS
}

/// Op `id` took `us`: counted, and logged past the threshold.
fn stepped(id: u32, us: u64) {
    STEP_MAX_US.fetch_max(us, Ordering::Relaxed);
    let text = described(id);
    let pipeline = text.rsplit_once("(pipeline ").map_or("", |(_, p)| p.trim_end_matches(')').trim_end_matches(')')).to_string();
    {
        let mut by = BY_PIPELINE.lock().unwrap_or_else(|e| e.into_inner());
        match by.iter_mut().find(|e| e.0 == pipeline) {
            Some(e) => (e.1, e.2, e.3) = (e.1 + us, e.2.max(us), e.3 + 1),
            None => by.push((pipeline, us, us, 1)),
        }
    }
    if us >= slow_ms() * 1000 {
        xnw_core::warn!("step: {text} took {:.1} ms", us as f64 / 1000.0);
    }
}

/// The 20 pipelines with the most GPU time stepped so far.
pub(crate) fn top_pipelines() {
    let mut by = BY_PIPELINE.lock().unwrap_or_else(|e| e.into_inner()).clone();
    if by.is_empty() {
        return;
    }
    by.sort_by(|a, b| b.1.cmp(&a.1));
    let rows: Vec<String> = by.iter().take(20).map(|(p, sum, max, n)| format!("{} {:.1} ms (max {:.1}, {n} ops)", if p.is_empty() { "(none)" } else { p }, *sum as f64 / 1000.0, *max as f64 / 1000.0)).collect();
    xnw_core::warn!("step: top pipelines by GPU time: {}", rows.join("; "));
}

/// While stepping, the submission's wait: polls `done` and the crumb
/// slots of its lists, timing each op seen between its markers; an op
/// still running after 1,500 ms is logged at once (the log is written
/// unbuffered), and the wait goes on.
pub(crate) fn watch(done: impl Fn() -> bool, slots: &[(u32, Option<u32>)]) {
    static LAST_TOP: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let all: Vec<u32> = slots.iter().flat_map(|&(m, p)| std::iter::once(m).chain(p)).collect();
    // Per slot: the op seen running, since when, whether reported.
    let mut running: Vec<(u32, std::time::Instant, bool)> = vec![(0, std::time::Instant::now(), false); all.len()];
    loop {
        let finished = done();
        let now = std::time::Instant::now();
        for (k, &slot) in all.iter().enumerate() {
            let Some((started, ended)) = read(slot) else { continue };
            let r = &mut running[k];
            if r.0 != 0 && (ended == r.0 || started != r.0 || finished) {
                stepped(r.0, now.duration_since(r.1).as_micros() as u64);
                r.0 = 0;
            }
            if started != 0 && started != ended && r.0 != started && !finished {
                *r = (started, now, false);
            }
            if r.0 != 0 && !r.2 && now.duration_since(r.1) >= std::time::Duration::from_millis(1500) {
                r.2 = true;
                xnw_core::warn!("step: {} STILL RUNNING after 1500 ms", described(r.0));
            }
        }
        if finished {
            break;
        }
        std::thread::yield_now();
    }
    let mut last = LAST_TOP.lock().unwrap_or_else(|e| e.into_inner());
    if last.is_none_or(|t| t.elapsed() >= std::time::Duration::from_secs(5)) {
        *last = Some(std::time::Instant::now());
        drop(last);
        top_pipelines();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On the real device, a list with three ops leaves its in and out
    /// markers in process memory, read without a map.
    #[test]
    fn three_ops_leave_their_markers_in_our_memory() {
        let Some(host) = super::super::host::host() else { return };
        // Crumbs are off in tests: the writes are made directly.
        let Some(b) = buffer() else { panic!("a breadcrumb buffer on this device") };
        let list = host.new_list().unwrap();
        let slot = list.crumbs().slot();
        for id in [101, 102, 103] {
            write(list.crumbs(), &list.list, id, 0, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN);
            if id != 103 {
                write(list.crumbs(), &list.list, id, 1, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT);
            }
        }
        list.close();
        host.execute(&[&list]);
        assert_eq!(read(slot), Some((103, 102)), "op 103 started, 102 the last finished");
        let _ = b;
    }

    /// Ops of a list stepped run one at a time in order: each op's
    /// markers and a drain, the watch sees them finish in order, the final
    /// target contents are those of the last op (as unstepped), and an op
    /// over the threshold is counted as the longest.
    #[test]
    fn stepped_ops_run_in_order_and_are_timed() {
        let Some(host) = super::super::host::host() else { return };
        buffer().expect("a breadcrumb buffer on this device");
        STEP_FORCED.store(true, Ordering::Relaxed);
        let list = host.new_list().unwrap();
        // Five copies into one buffer, each a whole 64 MiB: each op slow
        // enough to be seen by the poll; the last one's source wins.
        const BYTES: u64 = 64 << 20;
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: BYTES,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let make = || {
            let mut r: Option<ID3D12Resource> = None;
            // SAFETY: plain creation.
            unsafe { host.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r) }.unwrap();
            r.unwrap()
        };
        let (dst, src) = (make(), make());
        let mut ids = Vec::new();
        for _ in 0..5 {
            let op = begin_forced(&list, "test copy 64 MiB");
            ids.push(op.id);
            let _ = op.barrier;
            // SAFETY: an open list, live buffers.
            unsafe { list.list.CopyBufferRegion(&dst, 0, &src, 0, BYTES) };
        }
        list.close();
        STEP_MAX_US.store(0, Ordering::Relaxed);
        // The watch runs from the host's wait while stepping.
        host.execute(&[&list]);
        STEP_FORCED.store(false, Ordering::Relaxed);
        let slot = list.crumbs().slot();
        assert_eq!(read(slot), Some((ids[4], ids[4])), "the fifth op last, finished");
        let by = BY_PIPELINE.lock().unwrap().clone();
        let timed: u64 = by.iter().filter(|e| e.0 == "test").map(|e| e.3).sum();
        assert!(timed >= 1, "stepped ops were timed ({by:?})");
        assert!(STEP_MAX_US.load(Ordering::Relaxed) > 0, "the longest op is known");
    }

    /// What a global UAV barrier costs on this device: 2,000 tiny
    /// driver compute passes (unordered-access clears of 256 bytes) in one
    /// list, without and with a barrier after each (printed; the barrier
    /// list must complete).
    #[test]
    fn global_uav_barrier_cost() {
        let Some(host) = super::super::host::host() else { return };
        let device = &host.device;
        const N: usize = 2000;
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 256,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation, views into our heaps, recording on our lists.
        unsafe {
            device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, None, &mut buffer).unwrap();
            let buffer = buffer.unwrap();
            let heap = |flags| device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 1, Flags: flags, NodeMask: 0 }).unwrap();
            let (visible, cpu) = (heap(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE), heap(D3D12_DESCRIPTOR_HEAP_FLAG_NONE));
            let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: D3D12_BUFFER_UAV { FirstElement: 0, NumElements: 64, ..Default::default() } },
            };
            device.CreateUnorderedAccessView(&buffer, None, Some(&uav), visible.GetCPUDescriptorHandleForHeapStart());
            device.CreateUnorderedAccessView(&buffer, None, Some(&uav), cpu.GetCPUDescriptorHandleForHeapStart());
            let mut ms = [0.0; 2];
            for (k, with) in [false, true].into_iter().enumerate() {
                let list = host.new_list().unwrap();
                list.list.SetDescriptorHeaps(&[Some(visible.clone())]);
                for i in 0..N {
                    list.list.ClearUnorderedAccessViewUint(visible.GetGPUDescriptorHandleForHeapStart(), cpu.GetCPUDescriptorHandleForHeapStart(), &buffer, &[i as u32; 4], &[]);
                    if with {
                        uav_barrier(&list.list);
                    }
                }
                list.close();
                let at = std::time::Instant::now();
                host.execute(&[&list]);
                ms[k] = at.elapsed().as_secs_f64() * 1000.0;
            }
            println!("barriers: {N} tiny compute passes: {:.2} ms without barriers, {:.2} ms with a global UAV barrier after each ({:.2} us per barrier)", ms[0], ms[1], (ms[1] - ms[0]) * 1000.0 / N as f64);
        }
    }

    /// A list that stopped after an op: the report's lines name what
    /// it recorded next, ops and the records between them, with the
    /// argument bytes of an indirect execution.
    #[test]
    fn the_records_after_the_last_finished_op_are_named() {
        let c = ListCrumbs::new();
        let args = [3u32, 1, 1];
        let push = |op: u32, text: &str, a: Option<(u64, usize)>| c.ring.lock().unwrap().push(Record { op, text: text.into(), args: a });
        let first = describe("dispatch 16x1x1 (pipeline compute f56f14cc)".into());
        push(first, "", None);
        push(0, "global UAV barrier (ours, XNW_UAV_BARRIER)", None);
        push(0, "pipeline bind compute 1234", None);
        push(0, "execute indirect: dispatch, title arguments", Some((args.as_ptr() as u64, 12)));
        let next = describe("execute indirect kind 2 up to 1 (pipeline compute 1234)".into());
        push(next, "", None);
        let lines = following(c.slot(), first, 8);
        assert_eq!(lines.len(), 4, "{lines:?}");
        assert!(lines[0].contains("UAV barrier") && lines[1].contains("pipeline bind"));
        assert!(lines[2].contains("title 00000003 00000001 00000001"), "{}", lines[2]);
        assert!(lines[3].contains("execute indirect kind 2"));
        assert_eq!(following(c.slot(), 0, 2).len(), 2, "from the start when nothing finished");
    }

    /// The header names how long the presenter has been inside Present.
    #[test]
    fn the_header_says_since_when_present_blocks() {
        PRESENT_AT.store(xnw_core::uptime().as_micros() as u64, Ordering::Relaxed);
        std::thread::sleep(std::time::Duration::from_millis(20));
        PRESENTING.store(2 | 1304 << 8, Ordering::Relaxed);
        let text = presenter_text();
        assert!(text.starts_with("INSIDE Present for frame 1304, entered at "), "{text}");
        let ms: u64 = text.rsplit_once(", ").unwrap().1.trim_end_matches(" ms ago").parse().unwrap();
        assert!(ms >= 20, "{text}");
        PRESENTING.store(1 | 7 << 8, Ordering::Relaxed);
        assert_eq!(presenter_text(), "in its blit for frame 7");
        PRESENTING.store(0, Ordering::Relaxed);
    }

    /// The test's ops: `begin` with crumbs forced (they are off in tests).
    fn begin_forced<'a>(list: &'a super::super::host::HostList, what: &str) -> Op<'a> {
        let id = describe(format!("{what} (pipeline test)"));
        write(list.crumbs(), &list.list, id, 0, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN);
        Op { list, id, barrier: false }
    }

    /// The watch's bookkeeping, without a device: an op seen started and
    /// then finished is timed and over the threshold is the longest.
    #[test]
    fn a_slow_op_is_counted_over_the_threshold() {
        let id = describe("slow op (pipeline slow-test)".into());
        stepped(id, (slow_ms() + 1) * 1000);
        assert!(STEP_MAX_US.load(Ordering::Relaxed) > slow_ms() * 1000);
        assert!(BY_PIPELINE.lock().unwrap().iter().any(|e| e.0 == "slow-test" && e.3 >= 1));
    }
}
