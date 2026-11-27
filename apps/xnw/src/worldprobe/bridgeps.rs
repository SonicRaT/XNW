//! The live source of the bridged rows (`XNW_WORLD_RENDER=...,bridgeps=on|
//! <hash>+<hash>`, default off): at the list sink of a draw whose pixel
//! program is one of the named ones, its binding inputs are read from title
//! memory where the frame pack finds them (`pack.rs`: root constants, root
//! views, table views, the draw words at the frame table's slot 86 + root
//! 4 dword 1 and the views they name in the bound heap, the sampler heap)
//! and given to this thread's last hooked draw, the list row the sink
//! belongs to (as the pre-skin's own record). `xnw_world::bridged::take`
//! does the reading over [`Title`]; our pass draws the rows
//! (`xnw_world::bridged::record`), the replay the same over a frame pack.
//!
//! Bounds: only the table slots the two programs name are read (each view
//! at most `bridged::VIEW_MAX`, once per frame by address and length: the
//! frame table's views are read once for all rows); root views (vertex
//! streams) are kept across frames by address, length and the hash of
//! their first 256 bytes; the sampler heap once per frame per heap object.
//! Every read is fault-safe (`fault_site` "bridged rows"); a draw that
//! cannot be read is counted, never a fault. A draw of a pipeline whose
//! pipelines are not made yet (they are made on a thread of their own the
//! first time the pipeline is seen) is counted and stays ours.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use xnw_core::framewall;

use xnw_formats::framepack::view;
use xnw_gpu::d3d12x::tap::{self, TapDraw, TapRoot, TapView};
use xnw_world::bridge::{self, SamplerDesc};
use xnw_world::bridged::{self, Key, Source, SrcRoot, SrcView, VIEW_MAX, WORDS, WORDS_DWORD, WORDS_ROOT, WORDS_SLOT};

/// The named pixel programs (read once; empty: off).
pub fn wanted() -> &'static [u32] {
    static W: std::sync::LazyLock<Vec<u32>> = std::sync::LazyLock::new(|| bridge::parse_switch(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default()));
    &W
}

pub fn on() -> bool {
    !wanted().is_empty()
}

/// SW1 live `bridgeps` start: on when programs are named.
pub(crate) fn live_start() -> u32 {
    u32::from(on())
}

/// SW1 live `bridgefam` start: a bit per named program (bit k: `wanted()[k]`;
/// the programs past bit 30, e.g. `livery=on`'s, share bit 30).
pub(crate) fn fam_start() -> u32 {
    (1u64 << wanted().len().min(31)) as u32 - 1
}

/// SW1: whether the live switches give a row of `key` its inputs (else it
/// draws our stand-in); one load each, at the sink.
fn live_ok(key: Key) -> bool {
    use xnw_core::live;
    if !live::on(live::BRIDGEPS, live_start) {
        return false;
    }
    let fam = live::value(live::BRIDGEFAM, fam_start);
    fam == fam_start() || wanted().iter().position(|&w| w == (key.1 >> 96) as u32).is_some_and(|k| fam >> k.min(30) & 1 != 0)
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The (mesh, pixel) container pair of a pipeline whose pixel program is
/// named (None: not one), from its stages.
pub fn key_of(stages: &[tap::TapStage], wanted: &[u32]) -> Option<Key> {
    let ps = stages.iter().find(|s| s.name == "pixel" && s.container)?;
    let ms = stages.iter().find(|s| s.name == "mesh" && s.container)?;
    wanted.contains(&((ps.hash >> 96) as u32)).then_some((ms.hash, ps.hash))
}

thread_local! {
    /// Pipeline object -> its pair when its pixel program is named.
    static PIPES: std::cell::RefCell<HashMap<usize, Option<Key>>> = std::cell::RefCell::new(HashMap::new());
}

fn pipe_key(pipeline: usize) -> Option<Key> {
    PIPES.with(|m| {
        let mut m = m.borrow_mut();
        if let Some(&k) = m.get(&pipeline) {
            return k;
        }
        if m.len() > 4096 {
            m.clear();
        }
        let stages = tap::stages(pipeline);
        let k = key_of(&stages, wanted());
        if let Some(key) = k
            && !bridged::cache().asked(key)
            && let Some(host) = xnw_gpu::d3d12x::host::host_device()
        {
            // The pipelines are made on a thread of their own; until then the rows stay ours.
            let bytes = |name: &str| stages.iter().position(|s| s.name == name).and_then(|i| tap::stage_bytes(pipeline, i));
            if let (Some(msc), Some(psc)) = (bytes("mesh"), bytes("pixel")) {
                let title = tap::targets(pipeline).map(|t| t.0).unwrap_or_default();
                bridged::cache().ask(key, msc, psc, &title, &host.device, &host.caps, false);
            }
        }
        m.insert(pipeline, k);
        k
    })
}

/// The draw tap's filter: a named pixel program (asked once per pipeline and thread).
pub(super) fn draw_want(pipeline: usize) -> bool {
    pipe_key(pipeline).is_some()
}

/// Sinks since the last frame line: read and given to their draw, pipelines
/// not made yet, no hook draw to give it to.
static SINK_N: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// This frame's counters (read and reset): (not ready, no hook draw).
pub fn take_counts() -> (u32, u32) {
    SINK_N[0].swap(0, Ordering::Relaxed);
    (SINK_N[1].swap(0, Ordering::Relaxed) as u32, SINK_N[2].swap(0, Ordering::Relaxed) as u32)
}

/// The draw tap (title thread, at the list sink, after the draw's hook).
pub(super) fn on_draw_sink(d: &TapDraw) {
    let Some(key) = pipe_key(d.pipeline).filter(|&k| live_ok(k)) else { return };
    let Some(built) = bridged::cache().get(key) else {
        SINK_N[1].fetch_add(1, Ordering::Relaxed);
        return;
    };
    let _site = super::fault_site(super::SITE_BRIDGE);
    // The hooked draw's vertex stream and meshlet buffer: the root view is
    // read over the stream, the group count from the meshlet buffer.
    let Some((vb_at, vb_len, mesh_at)) = super::last_draw_streams() else {
        SINK_N[2].fetch_add(1, Ordering::Relaxed);
        return;
    };
    let mut o = bridged::take(&Title { d, stream: (vb_at, vb_len), frame: super::frame_key() }, key, &built.used);
    o.set_groups(mesh_at);
    if super::attach_bridged(Box::new(o)) {
        SINK_N[0].fetch_add(1, Ordering::Relaxed);
    } else {
        SINK_N[2].fetch_add(1, Ordering::Relaxed);
    }
}

fn readable(addr: u64, len: u64) -> bool {
    addr != 0 && len != 0 && (0x1_0000..0x8000_0000_0000).contains(&addr) && addr.checked_add(len).is_some_and(|e| e <= 0x8000_0000_0000)
}

fn read(addr: u64, len: u64) -> Option<Vec<u8>> {
    if !readable(addr, len) {
        return None;
    }
    let _site = super::fault_site_or_kept(super::SITE_BRIDGE);
    super::dcopy(addr, len as usize)
}

/// [`read`] into `dst` (PERF4: small reads on the stack, no allocation).
fn read_into(addr: u64, dst: &mut [u8]) -> bool {
    if !readable(addr, dst.len() as u64) {
        return false;
    }
    let _site = super::fault_site_or_kept(super::SITE_BRIDGE);
    super::dcopy_into(addr, dst)
}

/// Bytes read from title memory this frame (the frame line), and the per
/// frame view cache: (frame, (address, length) -> bytes). PERF4: an entry
/// is made before its read, which fills it once: the title threads' sinks
/// that miss it together wait for that one copy (each made its own, up to
/// `VIEW_MAX` each, while the read ran outside the lock).
static READ_BYTES: AtomicU64 = AtomicU64::new(0);
type Shared = Arc<OnceLock<Option<Arc<Vec<u8>>>>>;
type Views = (u64, HashMap<(u64, u64), Shared>);
static VIEWS: Mutex<Option<Views>> = Mutex::new(None);
/// PERF4: the largest frame-table view read since the last line: bytes << 20 | slot.
static LARGEST: AtomicU64 = AtomicU64::new(0);
/// Root views kept across frames: (address, length) -> (hash of the first
/// 256 bytes, bytes, frame last used).
type Kept = HashMap<(u64, u64), (u128, Arc<Vec<u8>>, u64)>;
static KEPT: Mutex<Option<Kept>> = Mutex::new(None);
/// Frames a kept root view stays unused before it goes.
const KEEP_FRAMES: u64 = 600;
type Samplers = (u64, HashMap<usize, Option<Arc<Vec<(u32, SamplerDesc)>>>>);
static SAMPLERS: Mutex<Option<Samplers>> = Mutex::new(None);

/// Bytes since the last line: [frame tables (once a frame), constant
/// buffers (each frame), kept views read anew, kept views not read again].
static READ_SPLIT: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Bytes read since the last call, and their split ([`READ_SPLIT`]).
pub fn take_read_bytes() -> (u64, [u64; 4]) {
    (READ_BYTES.swap(0, Ordering::Relaxed), std::array::from_fn(|k| READ_SPLIT[k].swap(0, Ordering::Relaxed)))
}

/// The reads line (the live list build every 60 frames, `xnw burst` every
/// frame): `rows` bridged rows, the bytes read since the last line.
pub fn reads_line(frame: u64, rows: usize) -> String {
    let (read, split) = take_read_bytes();
    let largest = LARGEST.swap(0, Ordering::Relaxed);
    format!(
        "world render: frame {frame}: bridgeps reads: {rows} rows, {} KiB read from title memory since the last line (frame tables {} KiB, constant buffers {} KiB, kept views {} KiB new; kept views not read again {} KiB); largest frame-table view {} KiB (slot {})",
        read >> 10,
        split[0] >> 10,
        split[1] >> 10,
        split[2] >> 10,
        split[3] >> 10,
        largest >> 30,
        largest & 0xF_FFFF
    ) + &big_line()
}

/// PERF5: the reads line's tail: the three largest frame-table views, the
/// pooled buffers, the views filled at the drain (since the last line).
fn big_line() -> String {
    let top = std::mem::take(&mut *lock(&TOP));
    let views: Vec<String> = top.iter().filter(|&&v| v != 0).map(|v| format!("{} KiB (slot {})", v >> 30, v & 0xF_FFFF)).collect();
    let (n, mib) = pool_size();
    let p: [u64; 3] = std::array::from_fn(|k| POOL_N[k].swap(0, Ordering::Relaxed));
    let d: [u64; 2] = std::array::from_fn(|k| DRAIN_N[k].swap(0, Ordering::Relaxed));
    format!(
        "; largest three {}; big views (>= 1 MiB): {n} buffers pooled ({mib} MiB), taken again {}, made {}, made past the pool {}; filled at the drain {} (faulted {}, bigcopy={})",
        if views.is_empty() { "none".to_string() } else { views.join(", ") },
        p[0],
        p[1],
        p[2],
        d[0],
        d[1],
        if xnw_core::live::on(xnw_core::live::BIGCOPY, drain_start) { "drain" } else { "worker" }
    )
}

/// A sample of a view's bytes that tells whether it changed: the first
/// and the last 256 bytes and eight 64-byte pieces between (`None`: not readable).
/// PERF4: one stack buffer, no allocation per piece (ten a view before).
fn sample(addr: u64, len: u64) -> Option<u128> {
    let mut buf = [0u8; 1024];
    if len <= 1024 {
        let b = &mut buf[..len as usize];
        return read_into(addr, b).then(|| xnw_core::hash128(&[b]));
    }
    let (head, rest) = buf.split_at_mut(256);
    let (tail, mids) = rest.split_at_mut(256);
    let ok = read_into(addr, head) && read_into(addr + len - 256, tail) && mids.chunks_exact_mut(64).zip(1u64..).all(|(c, k)| read_into(addr + (len - 64) * k / 9 / 4 * 4, c));
    ok.then(|| xnw_core::hash128(&[&buf]))
}

#[cfg(test)]
fn frame_read(frame: u64, addr: u64, len: u64) -> Option<Arc<Vec<u8>>> {
    frame_read_slot(frame, addr, len, 0)
}

/// A frame-table view (root 0, table slot `slot`): read once a frame for
/// every row and thread. PERF5: a thread's own views of the frame first
/// ([`MINE`], no lock); a view of at least [`BIG`] bytes is copied into a
/// pooled buffer ([`big_read`]) or, with `bigcopy=drain`, only its head
/// now and the rest at the drain cut ([`fill_drained`]).
fn frame_read_slot(frame: u64, addr: u64, len: u64, slot: u32) -> Option<Arc<Vec<u8>>> {
    if let Some(b) = mine(frame, |m| m.1.get(&(addr, len)).cloned()) {
        return Some(b);
    }
    let cell = {
        let mut g = lock(&VIEWS);
        let v = g.get_or_insert_with(|| (frame, HashMap::new()));
        if v.0 != frame {
            *v = (frame, HashMap::new());
        }
        v.1.entry((addr, len)).or_default().clone()
    };
    // Read outside the lock; a sink that finds the entry being read waits for it.
    let t = std::time::Instant::now();
    let mut copied = false;
    let got = cell
        .get_or_init(|| {
            copied = true;
            // Drained: not slot 20, whose bytes the sink cuts a row's region from (CLOTH2, `bridged::cut_region`).
            let b = if len >= BIG { big_read(frame, addr, len, slot != bridged::REGION_SLOT && drain_on())? } else { Arc::new(read(addr, len)?) };
            READ_BYTES.fetch_add(len, Ordering::Relaxed);
            READ_SPLIT[0].fetch_add(len, Ordering::Relaxed);
            LARGEST.fetch_max(len << 20 | u64::from(slot.min(0xF_FFFF)), Ordering::Relaxed);
            top_view(len << 20 | u64::from(slot.min(0xF_FFFF)));
            Some(b)
        })
        .clone();
    framewall::add(if copied { framewall::SINK_BIG_COPY } else { framewall::SINK_BIG_WAIT }, t.elapsed().as_nanos() as u64);
    match &got {
        Some(b) => mine(frame, |m| {
            m.1.insert((addr, len), b.clone());
        }),
        None => {
            // Not readable now: the next sink tries again (as before PERF4).
            let mut g = lock(&VIEWS);
            if let Some(v) = g.as_mut().filter(|v| v.0 == frame)
                && v.1.get(&(addr, len)).is_some_and(|c| Arc::ptr_eq(c, &cell))
            {
                v.1.remove(&(addr, len));
            }
        }
    }
    got
}

/// PERF5: frame-table views of at least this many bytes are pooled.
const BIG: u64 = 1 << 20;
/// PERF5: pooled buffers at most (a buffer past them is made and freed as before).
const POOL_MOST: usize = 48;
/// Frames a free pooled buffer stays untaken before it goes.
const POOL_FRAMES: u64 = 600;

/// PERF5: a pooled buffer: its capacity class (a multiple of [`BIG`]),
/// the frame it was last taken, the bytes.
struct Pooled {
    class: usize,
    last: u64,
    buf: Arc<Vec<u8>>,
}

/// PERF5: the big frame-table views' buffers, reused across frames. A
/// buffer is taken again only when nothing else holds it (`Arc::get_mut`:
/// no row, no kept input of `bridge::Kept`, which names inputs by their
/// bytes' address and holds them while it does, no thread's cache), so its
/// address never names other bytes to anyone who saw it. Each is made
/// zeroed once (first touch then) and keeps its pages.
static POOL: Mutex<Vec<Pooled>> = Mutex::new(Vec::new());
/// Since the last line: buffers taken again, made new, made outside the pool (full).
static POOL_N: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// A buffer of at least `len` bytes that no one else holds (`len` set, zeros or older bytes).
fn pool_take(frame: u64, len: usize) -> Arc<Vec<u8>> {
    let class = len.next_multiple_of(BIG as usize);
    let mut g = lock(&POOL);
    if let Some(i) = g.iter_mut().position(|p| p.class == class && Arc::get_mut(&mut p.buf).is_some()) {
        let mut p = g.swap_remove(i);
        drop(g);
        POOL_N[0].fetch_add(1, Ordering::Relaxed);
        let v = Arc::get_mut(&mut p.buf).expect("a pooled buffer no one else holds");
        // SAFETY: within the capacity, every byte written once (made zeroed).
        unsafe { v.set_len(len) };
        return p.buf;
    }
    // Free buffers long untaken go (those of a view no longer read).
    g.retain_mut(|p| Arc::get_mut(&mut p.buf).is_none() || frame.saturating_sub(p.last) <= POOL_FRAMES);
    let room = g.len() < POOL_MOST;
    drop(g);
    POOL_N[if room { 1 } else { 2 }].fetch_add(1, Ordering::Relaxed);
    let mut v = vec![0u8; class];
    v.truncate(len);
    Arc::new(v)
}

/// Gives a taken buffer back to the pool (it is taken again once its last holder lets it go).
fn pool_put(frame: u64, buf: &Arc<Vec<u8>>) {
    let class = buf.capacity();
    if !class.is_multiple_of(BIG as usize) {
        return;
    }
    let mut g = lock(&POOL);
    if g.len() < POOL_MOST {
        g.push(Pooled { class, last: frame, buf: buf.clone() });
    }
}

/// (buffers pooled, their MiB).
fn pool_size() -> (usize, usize) {
    let g = lock(&POOL);
    (g.len(), g.iter().map(|p| p.class).sum::<usize>() >> 20)
}

/// PERF5 `bigcopy=drain|worker` (default worker; live `bigcopy` 1 drain,
/// 0 worker): with drain a big frame-table view's sink copies its first
/// [`HEAD`] bytes alone (a row reads its first word, `Owned::firsts`;
/// slot 20, which the sink reads whole for CLOTH2's regions, is copied
/// at once) and
/// the render thread copies the rest at the drain cut ([`fill_drained`]),
/// before the frame's capture leaves: the workers neither copy nor wait.
/// A burst's served memory is copied at once.
pub(crate) fn drain_start() -> u32 {
    static V: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
        let v = std::env::var("XNW_WORLD_RENDER").unwrap_or_default();
        u32::from(v.split(',').map(str::trim).filter_map(|p| p.strip_prefix("bigcopy=")).last() == Some("drain"))
    });
    *V
}

fn drain_on() -> bool {
    xnw_core::live::on(xnw_core::live::BIGCOPY, drain_start) && !super::burst::serving()
}

/// The bytes a drained view's sink copies at once.
const HEAD: usize = 4096;

/// A drained view: its frame, title address and length, the buffer's
/// bytes, and the buffer, held until they are filled.
struct Drained {
    frame: u64,
    addr: u64,
    len: usize,
    at: *mut u8,
    _hold: Arc<Vec<u8>>,
}
// SAFETY: the pointer is the held buffer's own bytes (`Vec::as_mut_ptr`),
// written once, by [`fill_drained`], before any holder reads past the head.
unsafe impl Send for Drained {}
static DRAINED: Mutex<Vec<Drained>> = Mutex::new(Vec::new());
/// Since the last line: views filled at the drain, of them faulted (zeros).
static DRAIN_N: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A big view's bytes: a pooled buffer, filled now or (`drain`, `bigcopy=drain`) its head now.
fn big_read(frame: u64, addr: u64, len: u64, drain: bool) -> Option<Arc<Vec<u8>>> {
    if !readable(addr, len) {
        return None;
    }
    let mut b = pool_take(frame, len as usize);
    let v = Arc::get_mut(&mut b).expect("a taken buffer is ours alone");
    let ok = if drain {
        let head = HEAD.min(v.len());
        read_into(addr, &mut v[..head])
    } else {
        let _site = super::fault_site_or_kept(super::SITE_BRIDGE);
        super::dcopy_vec(addr, len as usize, v)
    };
    if ok && drain {
        let at = v.as_mut_ptr();
        lock(&DRAINED).push(Drained { frame, addr, len: len as usize, at, _hold: b.clone() });
    }
    pool_put(frame, &b);
    ok.then_some(b)
}

/// PERF5 `bigcopy=drain`: the render thread at the drain cut (every
/// recording thread is done with the frames `take` passes): copies the
/// rest of their drained views, before their captures leave. A view that
/// faults is zeros past its head (counted).
pub(crate) fn fill_drained(take: &dyn Fn(u64) -> bool) {
    let due: Vec<Drained> = {
        let mut g = lock(&DRAINED);
        if g.is_empty() {
            return;
        }
        let (due, rest) = std::mem::take(&mut *g).into_iter().partition(|d| take(d.frame));
        *g = rest;
        // Views of frames never sent (the cut moved on) go after a while.
        if let Some(last) = due.iter().map(|d: &Drained| d.frame).max() {
            g.retain(|d| d.frame + 8 >= last);
        }
        due
    };
    let t = std::time::Instant::now();
    let _site = super::fault_site_or_kept(super::SITE_BRIDGE);
    for d in &due {
        let (addr, len, at) = (d.addr, d.len, d.at);
        let head = HEAD.min(len);
        // SAFETY: the held buffer's bytes past the head, which no holder reads
        // before its capture leaves (after this); the title's range was readable at the sink.
        let ok = unsafe { super::fcopy(at.add(head), (addr + head as u64) as *const u8, len - head) };
        if !ok {
            // SAFETY: as above.
            unsafe { std::ptr::write_bytes(at.add(head), 0, len - head) };
            DRAIN_N[1].fetch_add(1, Ordering::Relaxed);
        }
        DRAIN_N[0].fetch_add(1, Ordering::Relaxed);
    }
    framewall::add(framewall::DRAIN_BIG_COPY, t.elapsed().as_nanos() as u64);
}

/// PERF5: constant buffers (and other small views read each frame) of at
/// most this many bytes come from this thread's [`CBS`].
const CB_MOST: usize = 4096;
/// Buffers a ring of [`CBS`] holds at most.
const CB_RING: usize = 1024;

thread_local! {
    /// PERF5: this thread's small buffers by capacity (256 B to 4 KiB, a
    /// ring each, oldest first): the oldest is taken again once no row
    /// holds it any more (`Arc::get_mut`), else a new one is made; no
    /// allocation per constant buffer once warm.
    static CBS: std::cell::RefCell<[std::collections::VecDeque<Arc<Vec<u8>>>; 5]> = std::cell::RefCell::new(Default::default());
}

/// A small view's bytes read now, in a buffer of [`CBS`] (larger: as [`read`]).
fn cb_read(addr: u64, len: u64) -> Option<Arc<Vec<u8>>> {
    if len as usize > CB_MOST {
        return read(addr, len).map(Arc::new);
    }
    if !readable(addr, len) {
        return None;
    }
    let class = (len as usize).max(256).next_power_of_two();
    CBS.with(|r| {
        let mut r = r.borrow_mut();
        let ring = &mut r[class.trailing_zeros() as usize - 8];
        let mut b = match ring.front_mut().map(|f| Arc::get_mut(f).is_some()) {
            Some(true) => ring.pop_front().expect("the front"),
            _ => Arc::new(Vec::with_capacity(class)),
        };
        let ok = {
            let _site = super::fault_site_or_kept(super::SITE_BRIDGE);
            super::dcopy_vec(addr, len as usize, Arc::get_mut(&mut b).expect("a ring buffer no one else holds"))
        };
        ring.push_back(b.clone());
        if ring.len() > CB_RING {
            ring.pop_front();
        }
        ok.then_some(b)
    })
}

/// The three largest frame-table views since the last line (bytes << 20 | slot).
static TOP: Mutex<[u64; 3]> = Mutex::new([0; 3]);

fn top_view(v: u64) {
    let mut g = lock(&TOP);
    if g.contains(&v) || v <= g[2] {
        return;
    }
    g[2] = v;
    g.sort_unstable_by(|a, b| b.cmp(a));
}

/// PERF5: this thread's frame-table views and sampler heaps of one frame:
/// (frame, views by (address, length), samplers by heap).
type Mine = (u64, HashMap<(u64, u64), Arc<Vec<u8>>>, HashMap<usize, Option<Arc<Vec<(u32, SamplerDesc)>>>>);

thread_local! {
    /// PERF5: a row after this thread's first of the frame takes the
    /// frame's views and samplers here, without a lock.
    static MINE: std::cell::RefCell<Mine> = std::cell::RefCell::new((u64::MAX, HashMap::new(), HashMap::new()));
}

/// PERF5: this thread's cache, emptied when `frame` is a new frame.
fn mine<R>(frame: u64, f: impl FnOnce(&mut Mine) -> R) -> R {
    MINE.with(|m| {
        let mut m = m.borrow_mut();
        if m.0 != frame {
            m.0 = frame;
            m.1.clear();
            m.2.clear();
        }
        f(&mut m)
    })
}

/// A view kept across frames: the same bytes (the same `Arc`, which our
/// pass keeps on the device) while its sample ([`sample`]) is unchanged.
fn kept_read(frame: u64, addr: u64, len: u64) -> Option<Arc<Vec<u8>>> {
    let h = sample(addr, len)?;
    {
        let mut g = lock(&KEPT);
        let m = g.get_or_insert_with(HashMap::new);
        if let Some(e) = m.get_mut(&(addr, len)).filter(|e| e.0 == h) {
            if e.2 != frame {
                READ_SPLIT[3].fetch_add(len, Ordering::Relaxed);
            }
            e.2 = frame;
            return Some(e.1.clone());
        }
    }
    let b = Arc::new(read(addr, len)?);
    READ_BYTES.fetch_add(len, Ordering::Relaxed);
    READ_SPLIT[2].fetch_add(len, Ordering::Relaxed);
    let mut g = lock(&KEPT);
    let m = g.get_or_insert_with(HashMap::new);
    if m.len() > 4096 || frame.is_multiple_of(60) {
        m.retain(|_, e| frame.saturating_sub(e.2) <= KEEP_FRAMES);
    }
    m.insert((addr, len), (h, b.clone(), frame));
    Some(b)
}

fn src_view(v: &TapView) -> SrcView {
    // A placed view's words: its resource description (14), then its view
    // description, whose second word is the view's dimension.
    SrcView { kind: v.kind, addr: v.addr, bytes: v.bytes, format: v.format, width: v.width, dim: bridged::title_dim(v.kind, &v.words) }
}

/// A draw at its list sink as the bridged rows' source.
pub struct Title<'a> {
    pub d: &'a TapDraw<'a>,
    /// The hooked draw's vertex stream (address, bytes).
    pub stream: (u64, u64),
    pub frame: u64,
}

impl Source for Title<'_> {
    fn roots(&self) -> Vec<SrcRoot> {
        let _t = framewall::scope(framewall::SINK_ROOTS);
        self.d
            .roots
            .iter()
            .map(|r| match r {
                TapRoot::None => SrcRoot::None,
                TapRoot::Constants(c) => SrcRoot::Constants(c.to_vec()),
                TapRoot::Table { handle, .. } => SrcRoot::Table(*handle),
                TapRoot::View(a) => SrcRoot::View(*a),
            })
            .collect()
    }

    fn table(&self, root: u16, slot: u32) -> Option<SrcView> {
        match self.d.roots.get(usize::from(root)) {
            Some(TapRoot::Table { handle, slots, .. }) if slot < *slots => Some(tap::view(handle + tap::SLOT * u64::from(slot))).filter(|v| v.kind != view::NONE).map(|v| src_view(&v)),
            _ => None,
        }
    }

    /// The frame table's views (root 0) once a frame for every row;
    /// a draw's constant buffers each frame; its other views kept across
    /// frames while their sample holds ([`kept_read`]).
    fn bytes(&self, root: u16, slot: u32, v: &SrcView) -> Option<Arc<Vec<u8>>> {
        let len = v.bytes.min(VIEW_MAX);
        if root == 0 {
            let _t = framewall::scope(framewall::SINK_FVIEWS);
            return frame_read_slot(self.frame, v.addr, len, slot);
        }
        if v.kind == view::CONSTANT || (len as usize) < bridge::KEEP_MIN {
            let _t = framewall::scope(framewall::SINK_CB);
            let b = cb_read(v.addr, len)?;
            READ_BYTES.fetch_add(len, Ordering::Relaxed);
            READ_SPLIT[1].fetch_add(len, Ordering::Relaxed);
            return Some(b);
        }
        let _t = framewall::scope(framewall::SINK_KEPT);
        kept_read(self.frame, v.addr, len)
    }

    fn root_view(&self, _: u16, addr: u64) -> Option<Arc<Vec<u8>>> {
        let _t = framewall::scope(framewall::SINK_KEPT);
        let (s_at, s_len) = self.stream;
        let stream = |x: u64| (s_at != 0 && x >= s_at && x < s_at + s_len).then(|| s_at + s_len - x);
        let (at, full, _, how) = super::pack::root_view_at(addr, tap::buffer_left, stream);
        if how == super::pack::ROOT_NONE {
            return None;
        }
        kept_read(self.frame, at, full.min(VIEW_MAX))
    }

    fn words(&self) -> Option<[u32; WORDS]> {
        let _t = framewall::scope(framewall::SINK_WORDS);
        let v = self.table(0, WORDS_SLOT).filter(|v| matches!(v.kind, view::BUFFER | view::PLACED_BUFFER | view::CONSTANT))?;
        let off = match self.d.roots.get(WORDS_ROOT) {
            Some(TapRoot::Constants(c)) => u64::from(*c.get(WORDS_DWORD)?),
            _ => return None,
        };
        if off + 4 * WORDS as u64 > v.bytes {
            return None;
        }
        let mut b = [0u8; 4 * WORDS];
        read_into(v.addr + off, &mut b).then_some(())?;
        Some(std::array::from_fn(|k| u32::from_le_bytes([b[4 * k], b[4 * k + 1], b[4 * k + 2], b[4 * k + 3]])))
    }

    fn title_words(&self, at: u64, len: usize) -> Option<Vec<u8>> {
        let _t = framewall::scope(framewall::SINK_WORDS);
        let v = self.table(0, WORDS_SLOT).filter(|v| matches!(v.kind, view::BUFFER | view::PLACED_BUFFER | view::CONSTANT))?;
        if at + len as u64 > v.bytes {
            return None;
        }
        let mut b = vec![0u8; len];
        read_into(v.addr + at, &mut b).then_some(())?;
        READ_BYTES.fetch_add(len as u64, Ordering::Relaxed);
        Some(b)
    }

    fn heap(&self, index: u32) -> Option<(u64, SrcView)> {
        let _t = framewall::scope(framewall::SINK_WORDS);
        let (_, h) = tap::heap_slot(self.d.view_heap, index)?;
        // The slot's record first (fault-safe): `tap::view` reads it in place
        // (PERF4: probed on the stack, no allocation).
        let mut probe = [0u8; tap::SLOT as usize];
        read_into(h, &mut probe).then_some(())?;
        Some((h, src_view(&tap::view(h))))
    }

    fn samplers(&self) -> Option<Arc<Vec<(u32, SamplerDesc)>>> {
        let _t = framewall::scope(framewall::SINK_SAMPLERS);
        let heap = self.d.sampler_heap;
        if let Some(s) = mine(self.frame, |m| m.2.get(&heap).cloned()) {
            return s;
        }
        {
            let mut g = lock(&SAMPLERS);
            let v = g.get_or_insert_with(|| (self.frame, HashMap::new()));
            if v.0 != self.frame {
                *v = (self.frame, HashMap::new());
            }
            if let Some(s) = v.1.get(&heap) {
                let s = s.clone();
                mine(self.frame, |m| m.2.insert(heap, s.clone()));
                return s;
            }
        }
        let got = tap::sampler_heap(heap).and_then(|(start, count)| {
            let n = count.min(xnw_formats::framepack::SAMPLER_MAX).min(bridge::SAMPLER_HEAP_MAX);
            let b = read(start, tap::SAMPLER_SLOT * u64::from(n))?;
            let entries: Vec<(u32, SamplerDesc)> = b
                .chunks_exact(tap::SAMPLER_SLOT as usize)
                .enumerate()
                .filter_map(|(i, c)| {
                    let raw: [u32; 4] = std::array::from_fn(|k| u32::from_le_bytes([c[4 * k], c[4 * k + 1], c[4 * k + 2], c[4 * k + 3]]));
                    let (form, desc) = tap::sampler(raw)?;
                    Some((i as u32, bridge::sampler_desc(form, &desc)?))
                })
                .collect();
            Some(Arc::new(entries))
        });
        let mut g = lock(&SAMPLERS);
        if let Some(v) = g.as_mut().filter(|v| v.0 == self.frame) {
            v.1.insert(heap, got.clone());
        }
        drop(g);
        mine(self.frame, |m| m.2.insert(heap, got.clone()));
        got
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(name: &'static str, hash: u128) -> tap::TapStage {
        tap::TapStage { name, container: true, hash, size: 0, prefix: [0; 24] }
    }

    #[test]
    fn named_pixel_program_keys_the_pipeline() {
        let ps = 0x3ed6_8103_u128 << 96 | 7;
        let ms = 0x07d0_a626_u128 << 96;
        assert_eq!(key_of(&[stage("mesh", ms), stage("pixel", ps)], &[0x3ed6_8103]), Some((ms, ps)));
        assert_eq!(key_of(&[stage("mesh", ms), stage("pixel", ps)], &[0x3811_0e02]), None);
        // No mesh program (a vertex pipeline): not bridged.
        assert_eq!(key_of(&[stage("vertex", ms), stage("pixel", ps)], &[0x3ed6_8103]), None);
    }

    /// The frame-read tests share the view cache (a new frame empties it).
    static SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    fn frame_reads_are_shared_within_a_frame() {
        let _one = lock(&SERIAL);
        let buf = vec![7u8; 4096];
        let a = buf.as_ptr() as u64;
        let x = frame_read(1_000_001, a, 4096).expect("read");
        let y = frame_read(1_000_001, a, 4096).expect("read again");
        assert!(Arc::ptr_eq(&x, &y));
        // A new frame reads afresh.
        let z = frame_read(1_000_002, a, 4096).expect("next frame");
        assert!(!Arc::ptr_eq(&x, &z));
    }

    /// PERF4: sinks on several threads missing the same view together get
    /// one copy (each made its own before); an unreadable view is not kept.
    #[test]
    fn concurrent_frame_reads_share_one_copy() {
        let _one = lock(&SERIAL);
        let buf = vec![9u8; 1 << 20];
        let a = buf.as_ptr() as u64;
        for frame in 3_000_001..3_000_021u64 {
            let got: Vec<Arc<Vec<u8>>> = std::thread::scope(|sc| {
                let hs: Vec<_> = (0..5).map(|_| sc.spawn(move || frame_read_slot(frame, a, 1 << 20, 3).expect("read"))).collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            assert!(got.iter().all(|g| Arc::ptr_eq(g, &got[0]) && g[1000] == 9), "frame {frame}: one copy for all threads");
        }
        assert!(frame_read_slot(3_000_100, 0x10, 64, 0).is_none());
        let g = lock(&VIEWS);
        assert!(g.as_ref().is_some_and(|v| v.0 != 3_000_100 || !v.1.contains_key(&(0x10, 64))), "a failed read leaves no entry");
    }

    /// PERF4: the stack sample tells a change in its pieces and sees none elsewhere.
    #[test]
    fn the_sample_sees_its_pieces() {
        let mut buf = vec![5u8; 4096];
        let a = buf.as_ptr() as u64;
        let h = sample(a, 4096).expect("sample");
        assert_eq!(sample(a, 4096), Some(h));
        buf[300] = 6; // between the head and the first middle piece
        assert_eq!(sample(a, 4096), Some(h));
        buf[4095] = 6; // the tail
        assert_ne!(sample(a, 4096), Some(h));
        let small = sample(a, 100).expect("small");
        assert_eq!(sample(a, 100), Some(small));
        assert_eq!(sample(0x10, 4096), None);
    }

    #[test]
    fn kept_views_are_the_same_bytes_until_their_sample_changes() {
        let mut buf = vec![3u8; 8192];
        let a = buf.as_ptr() as u64;
        let x = kept_read(2_000_001, a, 8192).expect("read");
        let y = kept_read(2_000_002, a, 8192).expect("next frame, same bytes");
        assert!(Arc::ptr_eq(&x, &y), "kept across frames");
        // A change the sample sees (the last 256 bytes) reads the view anew.
        buf[8000] = 4;
        let z = kept_read(2_000_003, a, 8192).expect("changed");
        assert!(!Arc::ptr_eq(&x, &z) && z[8000] == 4);
    }
    /// PERF5: a big view's buffer is taken again only once no one holds it,
    /// and always holds this frame's bytes.
    #[test]
    fn big_views_reuse_only_buffers_no_one_holds() {
        let _one = lock(&SERIAL);
        let mut buf = vec![1u8; 2 << 20];
        let a = buf.as_ptr() as u64;
        let x = frame_read_slot(4_000_001, a, 2 << 20, 5).expect("read");
        buf[(2 << 20) - 1] = 2;
        // `x` is held (a row, or `bridge::Kept`): the next frame's copy is another buffer.
        let y = frame_read_slot(4_000_002, a, 2 << 20, 5).expect("next frame");
        assert!(!std::ptr::eq(x.as_ptr(), y.as_ptr()) && x[(2 << 20) - 1] == 1 && y[(2 << 20) - 1] == 2);
        let (px, py) = (x.as_ptr(), y.as_ptr());
        drop((x, y));
        buf[0] = 3;
        // Let go of (this thread's cache lets go of the last frame's at the next frame): taken again, the new bytes.
        let z = frame_read_slot(4_000_003, a, 2 << 20, 5).expect("third frame");
        assert!(std::ptr::eq(z.as_ptr(), px) || std::ptr::eq(z.as_ptr(), py), "a free buffer is taken again");
        assert!(z.len() == 2 << 20 && z[0] == 3 && z[1] == 1 && z[(2 << 20) - 1] == 2);
        // A shorter view of the same class takes it too, at its own length.
        drop(z);
        let w = frame_read_slot(4_000_004, a + 4096, (2 << 20) - 4096, 5).expect("shorter");
        assert!(w.len() == (2 << 20) - 4096 && w[0] == 1 && w[w.len() - 1] == 2);
    }

    /// PERF5 `bigcopy=drain`: the sink copies the head; the drain cut the rest.
    #[test]
    fn drained_views_are_filled_at_the_cut() {
        let _one = lock(&SERIAL);
        let mut buf = vec![4u8; 3 << 20];
        let a = buf.as_ptr() as u64;
        let x = big_read(5_000_001, a, 3 << 20, true).expect("head");
        assert!(x.len() == 3 << 20 && x[0] == 4 && x[HEAD - 1] == 4);
        buf[HEAD] = 6;
        buf[(3 << 20) - 1] = 7;
        // Another frame's cut leaves it.
        fill_drained(&|f| f == 5_000_002);
        fill_drained(&|f| f <= 5_000_001);
        assert!(x[HEAD] == 6 && x[(3 << 20) - 1] == 7 && x[HEAD + 1] == 4, "the rest as it is at the cut");
        assert!(lock(&DRAINED).iter().all(|d| d.frame != 5_000_001), "filled once");
    }

    /// PERF5: a constant buffer's bytes in this thread's ring, reused once let go.
    #[test]
    fn small_reads_reuse_their_ring() {
        let mut buf = vec![8u8; 600];
        let a = buf.as_ptr() as u64;
        let x = cb_read(a, 600).expect("read");
        let px = x.as_ptr();
        assert!(x.len() == 600 && x[599] == 8);
        drop(x);
        // Fill the 1 KiB ring's front with the let-go buffer: it is taken again with the new bytes.
        CBS.with(|r| {
            let mut r = r.borrow_mut();
            let ring = &mut r[2];
            while ring.front().is_some_and(|f| !std::ptr::eq(f.as_ptr(), px)) {
                ring.pop_front();
            }
        });
        buf[0] = 9;
        let y = cb_read(a, 600).expect("again");
        assert!(std::ptr::eq(y.as_ptr(), px) && y[0] == 9 && y.len() == 600);
        // Held: the next read is another buffer.
        let z = cb_read(a, 700).expect("held");
        assert!(!std::ptr::eq(z.as_ptr(), px) && z.len() == 700 && z[0] == 9);
        assert!(cb_read(0x10, 64).is_none());
    }
}
