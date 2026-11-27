//! Texture contents the title writes itself: streamed pages, layout-library
//! copies, movie frames, glyph atlases. Such a texture lives in console
//! memory in its tiled layout; before a submission samples its host copy,
//! the memory is detiled into an upload buffer when its content changed,
//! and the upload buffer is copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`super::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched (the paired reservation's two views among them),
//! a content hash of the whole memory at every submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.
//!
//! Ported from VIctory-base (branch `xnw/base` @
//! 5d0516e; the file is VIctory d2a4bae's, no PR touched it)
//! `crates/victory-gpu/src/texture.rs` 1-344, GPL-3.0-or-later. Changed:
//! `victory_formats` -> `xnw_formats` (the same `texture.rs`); `blake3` ->
//! `xnw_core::hash128`; VIctory (e54b675: read
//! unwatched memory without a region query, detile into cached memory)
//! applied; VIctory (6a96adb: copy a staging buffer into its host
//! texture on the prelude only when it changed; its WARP tests not taken)
//! applied;
//! `windows-sys` -> `xnw_core::win`; `tracing` -> `xnw_core` (debug lines
//! dropped); diagnostics dropped; the host layer through
//! [`super::host`]; `is_host_written` (our earlier name)
//! added. VIctory (8d9e40f: one per-thread scratch copy, not a
//! kept copy per texture) and an upstream finding (f887f6a: partly mapped
//! textures upload; read through `copy_mapped`, their interval reader not
//! taken); staging state of cold textures dropped ([`sweep`]); stage timers.
//! No upload buffer is kept per texture: a changed texture is detiled
//! into a pooled one ([`super::host::upload_pool`]) at the submission that
//! copies it, given back once the submission has run.
//! Streamed textures by origin (below, "Origins"): memory holding one
//! `.ttpf` texture's pages (`xnw_formats::origin`) shares one host copy per
//! origin, uploaded once from the decoded pages, never hashed; host copies
//! are made ahead off the frame when pages arrive; `XNW_MIP_DROP` (VIctory
//! 9a85db7, GPL-3.0-or-later: its view and detile handling).

use std::collections::HashMap;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;


use super::host;
use super::host::HostList;
use super::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| xnw_core::warn!($($arg)*));
    }};
}

/// A texture's upload state: the host texture's subresource layout in an
/// upload buffer, and what was last detiled.
pub(crate) struct Staging {
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: xnw_formats::texture::Layout,
    /// Mips per array slice (of the host texture).
    mips: u32,
    /// Console mips the host texture leaves out at the top (`XNW_MIP_DROP`).
    mip_drop: u32,
    /// The finest host mip the last upload holds (a pack upload of the
    /// mips arrived so far); [`Staging::record_copies`] copies from it on.
    copy_from: std::sync::atomic::AtomicU32,
    /// Where in the footprints the last upload starts (a pack upload
    /// holds only the mips it copies; 0 otherwise).
    upload_base: std::sync::atomic::AtomicU64,
    /// Its memory is under a read trap (probe, or on).
    trapped: std::sync::atomic::AtomicBool,
    /// The host copy, for its origin and upload state.
    host: std::sync::Weak<super::resource::HostTexture>,
    /// The title's description (none for copies made ahead).
    desc: Option<super::resource::ResourceDesc>,
    memory: *const u8,
    last: Mutex<Last>,
    /// The upload's bytes (what an upload copies).
    pub(crate) bytes: u64,
    /// The presented frame a list last asked for it in ([`sweep`]).
    used: std::sync::atomic::AtomicU64,
    /// A view of the texture was made since its last refresh;
    /// the frame (plus one) of its last refresh.
    viewed: std::sync::atomic::AtomicBool,
    refreshed: std::sync::atomic::AtomicU64,
    /// A refresh for our pass gave an upload (the exits' "never uploaded").
    took: std::sync::atomic::AtomicBool,
    /// The last upload was title memory over a streamed texture's
    /// pages of which only a prefix had arrived: the first host mip whole in
    /// it (`mips`: none), and that prefix (the origin's bytes and stamp).
    /// [`NOT_PARTIAL`] otherwise.
    plain_from: std::sync::atomic::AtomicU32,
    plain_at: [AtomicU64; 2],
    /// MOV1: a plane the kept Bink draw samples ([`note_movie_planes`]):
    /// uploaded at every refresh ([`movie_fresh`]).
    movie: std::sync::atomic::AtomicBool,
}

const NOT_PARTIAL: u32 = u32::MAX;

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after.
#[derive(Default)]
struct Last {
    hash: Option<u128>,
    asked: u64,
    /// Hashed refreshes in a row that found new contents.
    streak: u32,
    /// Refreshes left to upload without hashing (see [`BLIND_AFTER`]).
    blind: u32,
}

/// Memory that changed at this many hashed refreshes in a row (movie
/// planes: Bink writes them every frame) is uploaded for the next
/// [`BLIND_FOR`] refreshes without being hashed; the last of them hashes
/// again as a reference, and the one after decides by hash whether it still
/// changes. Unchanged memory then costs at most one upload per frame for
/// `BLIND_FOR` frames; a changing one saves `BLIND_FOR - 2` hashes of them.
const BLIND_AFTER: u32 = 4;
const BLIND_FOR: u32 = 32;

thread_local! {
    /// Texture memory read for hashing and detiling, and the cached memory
    /// each subresource is detiled into before one sequential copy to the
    /// (write-combined) staging buffer: one pair per submitting thread, as
    /// large as the largest texture it read, not one kept per texture
    /// (VIctory, 8d9e40f).
    static SCRATCH: std::cell::RefCell<(Vec<u8>, Vec<u8>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

// SAFETY: plain data; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    set.get_or_insert_with(Default::default)
        .insert(resource as usize);
}

/// Whether the host GPU wrote `resource`'s host copy (our earlier name).
pub fn is_host_written(resource: This) -> bool {
    HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|set| set.contains(&(resource as usize)))
}

/// The title released a resource.
/// Textures whose contents changed (uploaded) with a view made
/// since their last refresh, without one, and refreshes skipped by
/// `XNW_TEX_REFRESH=view`.
pub(crate) static REFRESH_COUNTS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// `XNW_TEX_REFRESH=view` (off by default).
fn refresh_by_view() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_TEX_REFRESH").is_ok_and(|v| v.trim() == "view"));
    *ON
}

/// The title made a view of `resource` (device view calls).
pub(crate) fn view_made(resource: usize) {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = map.as_ref().and_then(|m| m.get(&resource)) {
        s.viewed.store(true, Relaxed);
    }
}

pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = set.as_mut() {
        set.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    // SAFETY: a live resource.
    !gpu_output_flags(unsafe { texture.GetDesc() }.Flags.0)
}

/// Resource flags of the title's own GPU output: render target, depth
/// stencil or unordered access.
fn gpu_output_flags(flags: i32) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    flags & TARGET_DEPTH_UNORDERED != 0
}

/// Whether `resource` is a render, depth or compute target of the title
/// (its host copy carries those flags): nothing writes it while the title's
/// GPU work is nulled, so the pass gives it no refresh and no upload budget.
pub fn gpu_output(resource: This) -> bool {
    super::resource::host_copy(resource).is_some_and(|c| !title_written(&c.texture))
}

/// Registers a title-written texture with `list` (once per list) so that at
/// submission its memory is checked for changes by the title and, only when
/// its contents changed (the first upload too), detiled into an upload
/// buffer copied into its host texture on the list's prelude
/// (VIctory).
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    let host_written = is_host_written(resource);
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    if host_written || !list.first_use(resource as usize) {
        return;
    }
    let Some(copy) = super::resource::host_copy(resource) else {
        return;
    };
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    // Once per submission however many of its lists sample the texture;
    // `copy` keeps the host texture alive until the list is reset.
    list.defer_texture(staging, copy);
}

/// A title texture's host copy for OUR world pass, with the
/// upload (if its contents are due) that the pass records on its own list
/// with [`PassUpload::record`] before it samples the copy. The refresh is
/// the title's own ([`Staging::refresh`]: pack, cached pages, then title
/// memory), run with the null-mode refusal of the upload pool lifted for
/// this call only ([`host::pass_uploads`]); the title's texture steps stay
/// dropped. `room(bytes)` says whether the pass's budget takes an upload of
/// that size (false: the texture waits, `waiting` set).
pub struct PassTexture {
    pub copy: Arc<super::resource::HostTexture>,
    pub upload: Option<PassUpload>,
    /// The finest host mip with contents (a partial pack upload starts later).
    pub first_mip: u32,
    /// Contents are on the GPU (now or before): sample it; else draw flat.
    pub ready: bool,
    pub waiting: bool,
    /// Where an upload came from [approx: origin-counter deltas around the
    /// refresh]: 0 pack (or preload), 1 cached pages (or made ahead), 2 title
    /// memory (origin pages not cached), 3 title memory hashed/watched, 4 ?.
    pub source: usize,
    /// [`for_pass_async`]: a read of the copy is with the reader (queued,
    /// running, or finished and not yet taken by [`pass_finished`]): ask
    /// again every frame until it is not.
    pub flying: bool,
    /// [`for_pass_async`]: this ask is the first since the resource's read
    /// was taken with an upload (recorded this frame by the pass).
    pub landed: bool,
}

pub struct PassUpload {
    staging: Arc<Staging>,
    pub(crate) upload: host::Upload,
    /// (first mip copied, upload base) as they were when the upload was
    /// filled (the reader thread: a later refresh may move them before the
    /// pass records); None: read them now (the synchronous path).
    at: Option<(u32, u64)>,
}

impl PassUpload {
    /// Records the copies into `texture` (in the copy destination state).
    pub fn record(&self, list: &ID3D12GraphicsCommandList, texture: &ID3D12Resource) -> u64 {
        self.staging.record_copies_at(list, texture, &self.upload, self.at)
    }
}

pub fn for_pass(resource: This, refresh: bool, room: &mut dyn FnMut(u64) -> bool) -> Option<PassTexture> {
    for_pass_with(resource, refresh, room, partial_ready())
}

/// [`for_pass`]; `partial` ([`partial_ready`]): a pending origin copy
/// already uploaded with a whole mip is ready from its first present mip,
/// and when no page arrived since, it is not refreshed (nor `room` charged).
fn for_pass_with(resource: This, refresh: bool, room: &mut dyn FnMut(u64) -> bool, partial: bool) -> Option<PassTexture> {
    let copy = super::resource::host_copy(resource)?;
    let mut out = PassTexture { copy, upload: None, first_mip: 0, ready: false, waiting: false, source: 4, flying: false, landed: false };
    if is_host_written(resource) {
        out.ready = true; // the host GPU wrote it (a target): nothing to upload
        return Some(out);
    }
    let staging = staging(resource, &out.copy.texture)?;
    out.first_mip = staging.copy_from.load(Relaxed);
    let holds = if partial { staging.holds_from(&out.copy) } else { None };
    if let Some(from) = holds {
        out.first_mip = out.first_mip.max(from);
    }
    // Asks answered from a title-memory upload with mips not whole.
    let withheld = |s: &Staging, c: &super::resource::HostTexture| {
        let w = s.withheld(c);
        WITHHELD.fetch_add(u64::from(w), Relaxed);
        w
    };
    if !refresh {
        withheld(&staging, &out.copy);
        out.ready = !out.copy.pending.load(Relaxed) || holds.is_some();
        return Some(out);
    }
    if (holds.is_some() || staging.withheld(&out.copy)) && staging.no_new_pages(&out.copy) {
        withheld(&staging, &out.copy);
        WAIT_EXITS[EX_PARTIAL].fetch_add(1, Relaxed);
        WAIT_EXITS[EX_HOLDING].fetch_add(u64::from(holds.is_some()), Relaxed);
        // None of its mips whole: flat until a page arrives.
        out.ready = holds.is_some();
        out.waiting = !out.ready;
        return Some(out);
    }
    if !room(staging.bytes) {
        withheld(&staging, &out.copy);
        out.waiting = true;
        let pending = out.copy.pending.load(Relaxed);
        WAIT_EXITS[EX_BUDGET].fetch_add(1, Relaxed);
        WAIT_EXITS[EX_BUDGET_PENDING].fetch_add(u64::from(pending), Relaxed);
        WAIT_EXITS[EX_BUDGET_NEVER].fetch_add(u64::from(!staging.took.load(Relaxed) && out.copy.uploaded.load(Relaxed) == 0), Relaxed);
        WAIT_EXITS[EX_HOLDING].fetch_add(u64::from(holds.is_some()), Relaxed);
        out.ready = !pending && out.copy.uploaded.load(Relaxed) != 0 || holds.is_some();
        return Some(out);
    }
    const SOURCES: [[usize; 2]; 4] = [[OR_FROM_PACK, OR_PRELOAD_USED], [OR_FROM_PAGES, OR_AHEAD_USED], [OR_FROM_MEMORY, OR_FROM_MEMORY], [OR_PLAIN, OR_PLAIN]];
    let snap = || SOURCES.map(|[a, b]| ORIGIN_EVENTS[a].load(Relaxed) + if a == b { 0 } else { ORIGIN_EVENTS[b].load(Relaxed) });
    let before = snap();
    let refused = REFUSED_HERE.get();
    let upload = host::pass_uploads(|| staging.refresh());
    let after = snap();
    let [deferred, no_buffer] = [0, 1].map(|i| REFUSED_HERE.get()[i] != refused[i]);
    out.source = (0..4).find(|&i| after[i] != before[i]).unwrap_or(4);
    out.first_mip = staging.copy_from.load(Relaxed);
    let pending = out.copy.pending.load(Relaxed);
    let holds = if partial { staging.holds_from(&out.copy) } else { None };
    if let Some(from) = holds {
        out.first_mip = out.first_mip.max(from);
    }
    if let Some(e) = exit_of(upload.is_some(), pending, out.copy.uploaded.load(Relaxed) != 0, deferred, no_buffer) {
        WAIT_EXITS[e].fetch_add(1, Relaxed);
    }
    if upload.is_some() {
        staging.took.store(true, Relaxed);
    } else {
        WAIT_EXITS[EX_HOLDING].fetch_add(u64::from(pending && holds.is_some()), Relaxed);
    }
    // An upload with no whole mip is recorded, not sampled.
    let none_whole = withheld(&staging, &out.copy) && holds.is_none();
    out.ready = (upload.is_some() || !pending || holds.is_some()) && !none_whole;
    out.waiting = upload.is_none() && pending;
    out.upload = upload.map(|upload| PassUpload { staging, upload, at: None });
    Some(out)
}

/// Our pass binds a partly streamed texture from the mips that
/// arrived; on by default since its second landing,
/// `XNW_TEX_PARTIAL=off` for the old rule (flat `waiting` until complete).
fn partial_ready() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_TEX_PARTIAL").ok().is_none_or(|v| v.trim() != "off"));
    *ON
}

/// A title-memory upload over a streamed texture's pages exposes only
/// the mips whose pages have all arrived, and stays pending until all have
/// (before, a fallback upload was complete from mip 0 whatever had
/// arrived). `XNW_TEX_WHOLE=off` for the rule before.
fn whole_mips() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_TEX_WHOLE").ok().is_none_or(|v| v.trim() != "off"));
    *ON
}

/// Why our pass's ask gave no new texels, per ask (the exits):
/// count budget refused (of those: the copy pending, this texture never
/// uploaded), a pending partial copy with no new page, the frame cap, no
/// upload buffer, the reader busy (`texasync=on`); and asks bound from the
/// mips a pending partial copy holds. Zeroed by [`exits_text`].
pub(crate) static WAIT_EXITS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
const EX_BUDGET: usize = 0;
const EX_BUDGET_PENDING: usize = 1;
const EX_BUDGET_NEVER: usize = 2;
const EX_PARTIAL: usize = 3;
const EX_FRAME_CAP: usize = 4;
const EX_NO_BUFFER: usize = 5;
const EX_ASYNC_BUSY: usize = 6;
const EX_HOLDING: usize = 7;

/// Asks of our pass answered from a title-memory upload of a partly
/// streamed texture: the view starts at its first whole mip (or the draw
/// is flat when it has none). Zeroed by [`exits_text`].
pub(crate) static WITHHELD: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// Uploads this thread's refreshes had refused: by the frame cap,
    /// for want of an upload buffer (the exit of the refresh under way).
    static REFUSED_HERE: std::cell::Cell<[u64; 2]> = const { std::cell::Cell::new([0; 2]) };
}

fn refused_here(i: usize) {
    REFUSED_HERE.with(|r| {
        let mut v = r.get();
        v[i] += 1;
        r.set(v);
    });
}

/// The exit of a refresh that gave no upload (`None`: an upload, or
/// nothing to wait for).
fn exit_of(upload: bool, pending: bool, uploaded: bool, deferred: bool, no_buffer: bool) -> Option<usize> {
    if upload {
        None
    } else if deferred {
        Some(EX_FRAME_CAP)
    } else if no_buffer {
        Some(EX_NO_BUFFER)
    } else if pending && uploaded {
        Some(EX_PARTIAL)
    } else {
        None
    }
}

/// The per-second line of [`WAIT_EXITS`] (zeroing them).
pub fn exits_text() -> String {
    let e = WAIT_EXITS.each_ref().map(|c| c.swap(0, Relaxed));
    format!(
        "world render: textures: waiting exits: budget {} (pending {}, never uploaded {}), partial no new pages {}, frame cap {}, no buffer {}, async busy {}; partial copies holding texels {}; views narrowed to whole mips {}",
        e[EX_BUDGET], e[EX_BUDGET_PENDING], e[EX_BUDGET_NEVER], e[EX_PARTIAL], e[EX_FRAME_CAP], e[EX_NO_BUFFER], e[EX_ASYNC_BUSY], e[EX_HOLDING], WITHHELD.swap(0, Relaxed)
    )
}

// ---- Our pass's texture reads off the frame path (a cold
// pack read inside our record step took 154 ms; new reads on the record
// thread were 14 ms of a 47 ms record in a test run) ----
//
// The hand-over (PERF2). A refresh marks the host copy as holding its new
// contents (`uploaded`, `pending`, `copy_from`) when it READS, on the reader
// thread; the texels are on the GPU only once the pass has recorded the
// upload. So between the two the copy's own state must not be believed:
//  - a copy with a read queued, running or finished and not yet taken is
//    "in flight": every ask for it (any resource that shares the copy) is
//    answered as the pass saw it when the read was queued;
//  - finished uploads are taken all at once by the pass at the start of its
//    frame ([`pass_finished`]) and recorded whether or not a slot asks for
//    them that frame; only then the copy leaves the flight set.
// Before, the upload waited in a map for the same resource to ask again:
// an ask that came while the reader was at work, or through another
// resource of the same copy, saw "complete" and sampled a texture nothing
// had been copied into, and a slot that then took the known path never
// fetched the upload at all (black regions, tree crowns cut away by an
// alpha of zero); results past 1024 were dropped with their texels.

struct ReadJob {
    staging: Arc<Staging>,
    /// The host copy read for (held: its address is the flight key).
    copy: Arc<super::resource::HostTexture>,
    /// Keeps the title resource alive while the worker reads (a placed or
    /// live hold, moved in by the caller).
    _hold: Box<dyn Send>,
}

/// A read the worker finished, until the pass takes it.
struct ReadDone {
    key: usize,
    staging: Arc<Staging>,
    copy: Arc<super::resource::HostTexture>,
    upload: Option<host::Upload>,
    source: usize,
    at: (u32, u64),
}

/// What the pass may believe of the copies the reader works on.
#[derive(Default)]
struct Flight {
    /// Host copy -> (ready, first mip) as the pass saw it when its read was
    /// queued; there until the pass has taken the read's result.
    copies: std::collections::HashMap<usize, (bool, u32)>,
    /// Reads finished and not yet taken.
    finished: Vec<ReadDone>,
    /// Staging -> (source, an upload was recorded) of a read taken, for the
    /// resource's next ask (cleared when it grows large: an ask without one
    /// reads again and finds nothing new).
    answers: std::collections::HashMap<usize, (usize, bool)>,
}

#[derive(Default)]
struct Reader {
    queue: std::sync::Mutex<std::collections::VecDeque<ReadJob>>,
    flight: std::sync::Mutex<Flight>,
    cv: std::sync::Condvar,
    read_us: std::sync::atomic::AtomicU64,
    reads: std::sync::atomic::AtomicU64,
}

/// Jobs queued at most (the rest wait for a later frame).
const READ_QUEUE: usize = 256;

fn reader() -> &'static Reader {
    static R: std::sync::OnceLock<Reader> = std::sync::OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let r = R.get_or_init(Reader::default);
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xnw pass texture reads".into()).spawn(read_worker);
    });
    r
}

fn read_worker() {
    xnw_core::win::helper_thread();
    let r = reader();
    loop {
        let job = {
            let mut q = r.queue.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(j) = q.pop_front() {
                    break j;
                }
                q = r.cv.wait(q).unwrap_or_else(|e| e.into_inner());
            }
        };
        let t = std::time::Instant::now();
        const SOURCES: [[usize; 2]; 4] = [[OR_FROM_PACK, OR_PRELOAD_USED], [OR_FROM_PAGES, OR_AHEAD_USED], [OR_FROM_MEMORY, OR_FROM_MEMORY], [OR_PLAIN, OR_PLAIN]];
        let snap = || SOURCES.map(|[a, b]| ORIGIN_EVENTS[a].load(Relaxed) + if a == b { 0 } else { ORIGIN_EVENTS[b].load(Relaxed) });
        let before = snap();
        let upload = host::pass_uploads(|| job.staging.refresh());
        let at = (job.staging.copy_from.load(Relaxed), job.staging.upload_base.load(Relaxed));
        let after = snap();
        let source = (0..4).find(|&i| after[i] != before[i]).unwrap_or(4);
        let ReadJob { staging, copy, _hold } = job;
        drop(_hold);
        r.read_us.fetch_add(t.elapsed().as_micros() as u64, Relaxed);
        r.reads.fetch_add(1, Relaxed);
        let key = Arc::as_ptr(&staging) as usize;
        r.flight.lock().unwrap_or_else(|e| e.into_inner()).finished.push(ReadDone { key, staging, copy, upload, source, at });
    }
}

/// Copies refused by the bounds guard of [`Staging::record_copies_at`].
pub static PASS_COPY_REFUSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// (queued or reading, read and not yet taken, reads done, ms spent reading) of the pass's reader.
pub fn pass_reader_counts() -> (usize, usize, u64, f64) {
    let r = reader();
    let (flying, done) = r.flight.lock().map_or((0, 0), |f| (f.copies.len(), f.finished.len()));
    (flying - done.min(flying), done, r.reads.load(Relaxed), r.read_us.load(Relaxed) as f64 / 1e3)
}

/// The reads the reader finished since the last call: each one's host copy,
/// its upload and source ([`PassTexture::source`]). The pass calls this at the
/// start of its frame, before any [`for_pass_async`], and records every
/// upload on that frame's list ([`PassUpload::record`]) whether or not a slot
/// asks for the texture: from here on the copy's state is the truth again.
pub fn pass_finished() -> Vec<(Arc<super::resource::HostTexture>, PassUpload, usize)> {
    let r = reader();
    let mut f = r.flight.lock().unwrap_or_else(|e| e.into_inner());
    let done = std::mem::take(&mut f.finished);
    if f.answers.len() + done.len() > 4096 {
        f.answers.clear();
    }
    let mut out = Vec::new();
    for d in done {
        f.copies.remove(&(Arc::as_ptr(&d.copy) as usize));
        f.answers.insert(d.key, (d.source, d.upload.is_some()));
        if let Some(upload) = d.upload {
            d.staging.took.store(true, Relaxed);
            out.push((d.copy, PassUpload { staging: d.staging, upload, at: Some(d.at) }, d.source));
        }
    }
    out
}

/// [`for_pass`] with the read (pack, cached pages or title memory) on
/// the pass's reader thread: a texture whose contents are due is queued (its
/// budget charged now) and is `flying` until [`pass_finished`] has handed its
/// upload to the pass; until then it is `waiting` (flat, or its older
/// contents when it had some). The upload is never returned here.
/// `hold` keeps the title resource alive while it is read.
pub fn for_pass_async(resource: This, hold: Box<dyn Send>, room: &mut dyn FnMut(u64) -> bool) -> Option<PassTexture> {
    let copy = super::resource::host_copy(resource)?;
    let mut out = PassTexture { copy, upload: None, first_mip: 0, ready: false, waiting: false, source: 4, flying: false, landed: false };
    if is_host_written(resource) {
        out.ready = true;
        return Some(out);
    }
    let staging = staging(resource, &out.copy.texture)?;
    let key = Arc::as_ptr(&staging) as usize;
    let flight_key = Arc::as_ptr(&out.copy) as usize;
    let r = reader();
    let partial = partial_ready();
    let holds = |s: &Staging, c: &super::resource::HostTexture| if partial { s.holds_from(c) } else { None };
    let answer = {
        let mut f = r.flight.lock().unwrap_or_else(|e| e.into_inner());
        // In flight (this resource's read or another's of the same copy):
        // what the copy held before the read, whatever its state says now.
        if let Some(&(ready, first_mip)) = f.copies.get(&flight_key) {
            WAIT_EXITS[EX_ASYNC_BUSY].fetch_add(1, Relaxed);
            (out.ready, out.first_mip, out.waiting, out.flying) = (ready, first_mip, true, true);
            return Some(out);
        }
        f.answers.remove(&key)
    };
    if let Some((source, uploaded)) = answer {
        // The read came back and its upload (if any) is on this frame's list.
        out.source = source;
        out.landed = uploaded;
        out.first_mip = staging.copy_from.load(Relaxed);
        let pending = out.copy.pending.load(Relaxed);
        let held = holds(&staging, &out.copy);
        if let Some(from) = held {
            out.first_mip = out.first_mip.max(from);
        }
        // As [`for_pass`]: an upload with no whole mip is not sampled.
        let withheld = staging.withheld(&out.copy);
        WITHHELD.fetch_add(u64::from(withheld), Relaxed);
        out.ready = (uploaded || !pending || held.is_some()) && !(withheld && held.is_none());
        out.waiting = !uploaded && pending;
        return Some(out);
    }
    // Complete contents the pass recorded before (an origin copy's pages, or
    // a title-memory upload this staging gave): sampled while it is read again.
    let had = !out.copy.pending.load(Relaxed) && (out.copy.uploaded.load(Relaxed) != 0 || staging.took.load(Relaxed) && std::ptr::eq(staging.host.as_ptr(), Arc::as_ptr(&out.copy)));
    out.first_mip = staging.copy_from.load(Relaxed);
    let held = holds(&staging, &out.copy);
    if let Some(from) = held {
        out.first_mip = out.first_mip.max(from);
        if staging.no_new_pages(&out.copy) {
            WAIT_EXITS[EX_PARTIAL].fetch_add(1, Relaxed);
            WAIT_EXITS[EX_HOLDING].fetch_add(1, Relaxed);
            WITHHELD.fetch_add(u64::from(staging.withheld(&out.copy)), Relaxed);
            out.ready = true;
            return Some(out);
        }
    }
    out.waiting = true;
    out.ready = had || held.is_some();
    if !room(staging.bytes) {
        WAIT_EXITS[EX_BUDGET].fetch_add(1, Relaxed);
        WAIT_EXITS[EX_BUDGET_PENDING].fetch_add(u64::from(out.copy.pending.load(Relaxed)), Relaxed);
        WAIT_EXITS[EX_BUDGET_NEVER].fetch_add(u64::from(!had && out.copy.uploaded.load(Relaxed) == 0), Relaxed);
        return Some(out);
    }
    let mut q = r.queue.lock().unwrap_or_else(|e| e.into_inner());
    if q.len() >= READ_QUEUE {
        WAIT_EXITS[EX_ASYNC_BUSY].fetch_add(1, Relaxed);
        return Some(out);
    }
    r.flight.lock().unwrap_or_else(|e| e.into_inner()).copies.insert(flight_key, (out.ready, out.first_mip));
    q.push_back(ReadJob { staging, copy: out.copy.clone(), _hold: hold });
    drop(q);
    r.cv.notify_one();
    out.flying = true;
    Some(out)
}

/// Host copies with an upload put off or holding only the mips that
/// had arrived (they are asked again at every binding until complete).
pub(crate) static PENDING: AtomicU64 = AtomicU64::new(0);

pub(crate) fn set_pending(copy: &super::resource::HostTexture, on: bool) {
    if copy.pending.swap(on, Relaxed) != on {
        if on {
            PENDING.fetch_add(1, Relaxed);
        } else {
            PENDING.fetch_sub(1, Relaxed);
        }
    }
}

/// Whether `resource`'s host copy still waits for an upload: callers
/// that look at a texture only now and then (bindless slots) look again.
pub(crate) fn pending(resource: This) -> bool {
    super::resource::host_copy(resource).is_some_and(|c| c.pending.load(Relaxed))
}

/// Bytes of texture memory detiled into staging buffers (uploaded), for
/// the frame statistics (`XNW_FRAME_STATS`).
pub(crate) static DETILED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// MOV1: Bink movie planes (the textures the kept Bink draw samples). Run
/// 248 (80-87 s, 102-107 s): the presents alternate between the movie and
/// its first frame (frames 3300 and 3450 saved), with the decoder's plane
/// set (`flicker frame ... plane`): one set's host copy stopped taking the
/// decoder's frames. Such planes are uploaded at every refresh; the exits
/// that would have kept the old copy are counted ([`movie_exits`]).
/// `XNW_MOVIE_FRESH=0`: as before (the exits still counted).
pub(crate) fn movie_fresh() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_MOVIE_FRESH").map_or(true, |v| v.trim() != "0"));
    *ON
}

/// The exits a movie plane's refresh passed (forced: gate, watch) or took
/// (`XNW_MOVIE_FRESH=0`: all; hash and origin are not taken when forced):
/// once-a-frame gate, write watch said unwritten, hash unchanged, origin
/// copy kept; and uploads.
const MOVIE_GATE: usize = 0;
const MOVIE_WATCH: usize = 1;
const MOVIE_HASH: usize = 2;
const MOVIE_ORIGIN: usize = 3;
const MOVIE_UPLOADS: usize = 4;
static MOVIE_EXITS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// MOV1: the kept Bink draw samples `resources` (title textures): their
/// staging (made at their first refresh) is marked a movie plane.
pub(crate) fn note_movie_planes(resources: &[usize]) {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let Some(map) = map.as_ref() else { return };
    for r in resources {
        if let Some(s) = map.get(r) {
            s.movie.store(true, Relaxed);
        }
    }
}

/// MOV1: since the last call, refreshes of movie planes at each exit
/// ([`MOVIE_EXITS`]: gate, watch, hash, origin) and their uploads.
pub(crate) fn movie_exits() -> [u64; 5] {
    std::array::from_fn(|i| MOVIE_EXITS[i].swap(0, Relaxed))
}

impl Staging {
    /// MOV1: a movie plane at exit `exit`: counted; whether it goes on (forced).
    fn forced(&self, exit: usize) -> bool {
        if !self.movie.load(Relaxed) {
            return false;
        }
        MOVIE_EXITS[exit].fetch_add(1, Relaxed);
        movie_fresh()
    }

    /// Detiles the texture memory into a pooled upload buffer if it changed
    /// since the last time (and is mapped at all): the buffer, to be copied
    /// into the host texture, if it did.
    pub(crate) fn refresh(&self) -> Option<host::Upload> {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let (read, upload) = self.refresh_now();
        let preparing = &host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
        upload
    }

    /// Returns the bytes of title memory read and, if they changed, the
    /// upload buffer they were detiled into.
    fn refresh_now(&self) -> (usize, Option<host::Upload>) {
        // `XNW_TEX_REFRESH=view`: without a view made since its
        // last refresh, a texture is refreshed once a frame, not at every
        // submission that samples it.
        let viewed = self.viewed.swap(false, Relaxed);
        let frame = super::queue::presented() + 1;
        if refresh_by_view() && !viewed && self.refreshed.load(Relaxed) == frame {
            if !self.forced(MOVIE_GATE) {
                REFRESH_COUNTS[2].fetch_add(1, Relaxed);
                return (0, None);
            }
        }
        self.refreshed.store(frame, Relaxed);
        let (read, upload) = self.refresh_contents();
        if upload.is_some() {
            REFRESH_COUNTS[usize::from(!viewed)].fetch_add(1, Relaxed);
        }
        (read, upload)
    }

    fn refresh_contents(&self) -> (usize, Option<host::Upload>) {
        // This frame's uploads are spent: the rest wait (nothing asked
        // yet, so nothing is lost).
        if !frame_room(FRAME_UPLOAD) {
            ORIGIN_EVENTS[OR_UPLOADS_DEFERRED].fetch_add(1, Relaxed);
            refused_here(0);
            self.set_pending(true);
            return (0, None);
        }
        let size = self.read_size();
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        let at = std::time::Instant::now();
        let watched = super::watch::writes(start, start + self.layout.size as usize, false);
        host::stage(host::STAGE_WATCH, at);
        // Whether the title wrote the memory since this staging last asked
        // (the first ask counts as written for the hashing way, which must
        // then upload; for the origin way only a write seen after a first
        // ask counts: a second resource over a shared copy otherwise
        // uploaded title memory over the pages in the same prelude).
        let seen = last.asked != 0;
        let written = watched.as_ref().map(|(now, w)| {
            let wrote = last.asked == 0 || w.iter().any(|&(_, x)| x > last.asked);
            last.asked = *now;
            wrote
        });
        // MOV1: a Bink plane takes the plain way at every refresh.
        let movie = self.movie.load(Relaxed) && movie_fresh();
        if !movie
            && let Some(copy) = self.host.upgrade()
            && let Some(key) = copy.origin.get().copied().or_else(|| self.discover(&copy))
        {
            if let Some(done) = self.refresh_origin(&copy, key, written.map(|w| w && seen)) {
                if self.movie.load(Relaxed) && done.1.is_none() {
                    MOVIE_EXITS[MOVIE_ORIGIN].fetch_add(1, Relaxed);
                }
                last.hash = None;
                return done;
            }
            // Not the origin's pages any more: the copy is no longer shared.
            origin_forget(&key);
            if copy.uploaded.swap(0, Relaxed) != 0 {
                // What the copy holds is no title-memory upload.
                self.plain_from.store(NOT_PARTIAL, Relaxed);
            }
            ORIGIN_EVENTS[OR_FALLBACK].fetch_add(1, Relaxed);
        }
        if written == Some(false) && !self.forced(MOVIE_WATCH) {
            return (0, None);
        }
        // Never read title memory a bypassed page left unwritten.
        xnw_formats::origin::registry().ensure_present(self.memory as u64, self.layout.size);
        // Read through the copy, which stands unmapped pages (zeros there):
        // memory of big pools is too slow to ask page by page, and a
        // streamed texture whose pages are not all mapped yet is uploaded
        // with what is (VIctory's finding; their interval reader is
        // not taken, the copy's page-by-page fallback does the same).
        let full = self.layout.size as usize;
        // Asked before the read (a page arriving meanwhile only adds).
        let arrived = self.arrived();
        SCRATCH.with_borrow_mut(|(read, linear)| {
            let at = std::time::Instant::now();
            // Only `size` bytes are read (dropped mips lie past them).
            read.resize(full.max(read.len()), 0);
            host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
            host::stage(host::STAGE_READ, at);
            if movie {
                // MOV1: uploaded unhashed (as the blind refreshes are).
                last.hash = None;
                last.blind = 0;
            } else if watched.is_none() && last.blind > 1 {
                last.blind -= 1;
                last.hash = None;
                ORIGIN_EVENTS[OR_HASH_AVOIDED].fetch_add(size as u64, Relaxed);
            } else if watched.is_none() {
                let at = std::time::Instant::now();
                let hash = xnw_core::hash128(&[&read[..size]]);
                host::stage(host::STAGE_HASH, at);
                if last.blind == 1 {
                    // The reference for the next refresh's decision.
                    last.blind = 0;
                } else if last.hash == Some(hash) {
                    last.streak = 0;
                    if self.movie.load(Relaxed) {
                        MOVIE_EXITS[MOVIE_HASH].fetch_add(1, Relaxed);
                    }
                    return (size, None);
                } else if last.hash.is_some() {
                    last.streak += 1;
                    if last.streak >= BLIND_AFTER {
                        last.blind = BLIND_FOR;
                        static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                        if SAID.fetch_add(1, Relaxed) < 8 {
                            let d = &self.layout.desc;
                            xnw_core::info!(
                                "d3d12x: texture memory new at every frame uploaded without hashing ({}x{} mips {} element {} B, swizzle {}, {} KiB at {:#x})",
                                d.width, d.height, d.mips, d.element_bytes, d.swizzle.0, size >> 10, start
                            );
                        }
                    }
                }
                last.hash = Some(hash);
            }
            self.upload_base.store(0, Relaxed);
            // The whole chain (a pack upload before may have left it
            // copying from a later mip).
            self.copy_from.store(0, Relaxed);
            let Some(upload) = self.take_upload(self.bytes) else {
                // Not uploaded: the next submission tries again.
                *last = Last::default();
                self.set_pending(true);
                return (size, None);
            };
            // Memory over a streamed texture's pages, not all arrived:
            // the upload holds whole only the mips of the prefix that has;
            // the copy stays pending and is viewed from the first of them
            // ([`Staging::holds_from`]). Before, it was complete from mip 0.
            match arrived {
                Some((from, len, stamp)) => {
                    self.plain_from.store(from, Relaxed);
                    self.plain_at[0].store(len, Relaxed);
                    self.plain_at[1].store(stamp, Relaxed);
                    self.set_pending(true);
                }
                None => {
                    self.plain_from.store(NOT_PARTIAL, Relaxed);
                    self.set_pending(false);
                }
            }
            self.detile(&read[..full], linear, upload.data);
            if self.movie.load(Relaxed) {
                MOVIE_EXITS[MOVIE_UPLOADS].fetch_add(1, Relaxed);
            }
            ORIGIN_EVENTS[OR_PLAIN].fetch_add(1, Relaxed);
            ORIGIN_EVENTS[OR_BYTES_PLAIN].fetch_add(size as u64, Relaxed);
            (size, Some(upload))
        })
    }

    /// A copy made before its texture's pages arrived: if the memory now
    /// starts with one texture's pages, the copy takes that origin (and is
    /// kept for it).
    fn discover(&self, copy: &Arc<super::resource::HostTexture>) -> Option<OriginKey> {
        let key = late_key(self.memory, self.desc.as_ref()?)?;
        copy.origin.set(key).ok()?;
        ORIGIN_EVENTS[OR_LATE].fetch_add(1, Relaxed);
        keep(key, copy, false);
        Some(key)
    }

    /// Bytes of the texture memory the host copy needs: the whole layout,
    /// or with dropped mips (one 2D slice) up to where they start (the
    /// chain is smallest first, mip 0 last).
    fn read_size(&self) -> usize {
        let l = &self.layout;
        if self.mip_drop > 0 && l.slices == 1 && l.desc.dimension == xnw_formats::texture::Dimension::Tex2d && self.mip_drop <= l.first_tail_mip {
            return l.mips[self.mip_drop as usize - 1].offset as usize;
        }
        l.size as usize
    }

    /// The origin path. `None` when the memory no longer starts with
    /// the origin's pages, or the write watch saw the title write it after
    /// an upload (then the copy goes the hashing way). The contents are the
    /// file's pages: uploaded again only when more of them have arrived
    /// (the next larger mips), never hashed.
    fn refresh_origin(&self, copy: &super::resource::HostTexture, key: OriginKey, written: Option<bool>) -> Option<(usize, Option<host::Upload>)> {
        let size = self.layout.size;
        let o = xnw_formats::origin::peek(self.memory as u64, size)?;
        let uploaded = copy.uploaded.load(Relaxed);
        if !key.holds(&o) || written == Some(true) && uploaded != 0 {
            return None;
        }
        ORIGIN_EVENTS[OR_HASH_AVOIDED].fetch_add(size, Relaxed);
        let have = o.len;
        // Until the whole texture has arrived and is uploaded, it is
        // pending: whoever binds it asks again next frame.
        let partial = have.saturating_sub(self.memory as u64 & 0xfff) < size;
        if uploaded == have {
            self.set_pending(partial);
            return Some((0, None));
        }
        if uploaded != 0 {
            ORIGIN_EVENTS[OR_UPGRADED].fetch_add(1, Relaxed);
        }
        // The pack first: the mips that have arrived, read in the
        // host's layout (no decode, no detile). A preload made them
        // already when it holds at least those.
        let skew = (self.memory as u64 & 0xfff) as usize;
        let finest = self.finest_present(have.saturating_sub(skew as u64));
        if let Some(finest) = finest {
            let mut prepared = copy.prepared.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(first) = prepared.as_ref().and_then(|p| p.pack_first)
                && first <= finest.max(self.mip_drop)
                && prepared.as_ref().is_some_and(|p| p.layout == self.layout.desc && p.bytes == self.bytes && p.mip_drop == self.mip_drop)
                && let Some(p) = prepared.take()
            {
                ORIGIN_EVENTS[OR_PRELOAD_USED].fetch_add(1, Relaxed);
                self.copy_from.store(first - self.mip_drop, Relaxed);
                self.upload_base.store(p.base, Relaxed);
                copy.uploaded.store(have, Relaxed);
                self.set_pending(partial);
                return Some((0, Some(p.into_upload())));
            }
            drop(prepared);
            if let Some(upload) = self.from_pack(o.file, o.texture, finest) {
                copy.uploaded.store(have, Relaxed);
                self.set_pending(partial);
                self.trap(&o);
                return Some((0, Some(upload)));
            }
        }
        self.copy_from.store(0, Relaxed);
        // Bytes the bypass left out of title memory are written first.
        if o.absent && !xnw_formats::origin::registry().ensure_present(self.memory as u64, size) {
            return None;
        }
        let read = self.read_size();
        let from_pages = (have as usize).saturating_sub(skew).min(read);
        let prepared = copy.prepared.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(p) = prepared
            && p.layout == self.layout.desc
            && p.bytes == self.bytes
            && p.mip_drop == self.mip_drop
            && p.len == from_pages as u64
            && skew == 0
        {
            ORIGIN_EVENTS[OR_AHEAD_USED].fetch_add(1, Relaxed);
            self.upload_base.store(0, Relaxed);
            copy.uploaded.store(have, Relaxed);
            self.set_pending(partial);
            return Some((0, Some(p.into_upload())));
        }
        self.upload_base.store(0, Relaxed);
        let Some(upload) = self.take_upload(self.bytes) else {
            self.set_pending(true);
            return Some((0, None)); // Deferred; `uploaded` unchanged
        };
        let base = o.at + skew as u64;
        SCRATCH.with_borrow_mut(|(tiled, linear)| {
            let at = std::time::Instant::now();
            let full = self.layout.size as usize;
            tiled.resize(full.max(tiled.len()), 0);
            let got = tiled_prefix(o.file, base, &mut tiled[..from_pages]);
            if got > 0 {
                ORIGIN_EVENTS[OR_FROM_PAGES].fetch_add(1, Relaxed);
                ORIGIN_EVENTS[OR_BYTES_PAGES].fetch_add(got as u64, Relaxed);
            }
            if got < read {
                // Pages not cached, and what has not arrived (the title's
                // memory as it is), from title memory.
                ORIGIN_EVENTS[OR_FROM_MEMORY].fetch_add(1, Relaxed);
                ORIGIN_EVENTS[OR_BYTES_MEMORY].fetch_add((read - got) as u64, Relaxed);
                // SAFETY: both ranges lie in the texture's memory and scratch.
                host::copy_mapped(unsafe { tiled.as_mut_ptr().add(got) }, unsafe { self.memory.add(got) }, read - got, false);
            }
            host::stage(host::STAGE_READ, at);
            self.detile(&tiled[..full], linear, upload.data);
        });
        copy.uploaded.store(have, Relaxed);
        self.set_pending(partial);
        self.trap(&o);
        Some((read, Some(upload)))
    }

    /// Marks the host copy pending (an upload put off or partial) or
    /// complete; [`PENDING`] counts the pending ones.
    fn set_pending(&self, on: bool) {
        if let Some(copy) = self.host.upgrade() {
            set_pending(&copy, on);
        }
    }

    /// After its first upload from a texture's pages, the memory goes
    /// under a read trap (`XNW_TEXBYPASS=probe` or `on`), tag 0x1000 + the
    /// registry file (0x1FFF when the pages were bypassed), to count CPU
    /// reads of texture pages.
    fn trap(&self, o: &xnw_formats::origin::Origin) {
        let Some(t) = READ_TRAPS.get() else { return };
        if !bypass_mode_set() || self.trapped.swap(true, Relaxed) {
            return;
        }
        let tag = if o.absent { 0x1FFF } else { 0x1000 + (o.file & 0xffe) };
        (t.trap)(self.memory as usize, self.layout.size as usize, tag);
        let mut tags = TRAP_TAGS.lock().unwrap_or_else(|e| e.into_inner());
        if !tags.contains(&tag) {
            tags.push(tag);
        }
        TRAPPED.fetch_add(1, Relaxed);
    }

    /// The finest console mip whose bytes lie wholly in the first
    /// `present` bytes of the memory (the chain is smallest first), for one
    /// 2D slice; arrays and volumes only when all of it is present.
    fn finest_present(&self, present: u64) -> Option<u32> {
        let l = &self.layout;
        if present >= l.size {
            return Some(0);
        }
        if l.slices != 1 || l.desc.dimension != xnw_formats::texture::Dimension::Tex2d {
            return None;
        }
        let n = l.mips.len() as u32;
        let tail = l.first_tail_mip.min(n);
        let mut finest = if tail < n && l.alignment <= present { tail } else { n };
        for m in (0..tail).rev() {
            let end = if m == 0 { l.slice_size } else { l.mips[m as usize - 1].offset };
            if end > present {
                break;
            }
            finest = m;
        }
        (finest < n).then_some(finest)
    }

    /// The upload of console mips `finest..` from the texture pack
    /// (`XNW_TEXPACK`), when the pack has the texture at the description's
    /// size and those mips; only their subresources are then copied.
    fn from_pack(&self, file: u32, texture: u32, finest: u32) -> Option<host::Upload> {
        let pack = pack()?;
        let index = xnw_formats::origin::file(file)?;
        let entry = pack.lookup(index.pack_id?, texture)?;
        let d = &self.layout.desc;
        let layers = if d.dimension == xnw_formats::texture::Dimension::Tex3d { 1 } else { d.depth };
        if (entry.width, entry.height, u32::from(entry.mips), entry.layers) != (d.width, d.height, d.mips, layers) || entry.kind == xnw_formats::ttpf::KIND_VOLUME {
            return None;
        }
        let first = finest.max(self.mip_drop);
        if u32::from(entry.first) > first {
            return None;
        }
        let (from, len) = entry.span(first..d.mips)?;
        // One slice: subresources run mip 0 first, so the mips copied
        // are the buffer's tail; the upload holds only them.
        let base = if self.footprints.len() == self.mips as usize { self.footprints[(first - self.mip_drop) as usize].Offset } else { 0 };
        let upload = self.take_upload(self.bytes - base)?;
        let copied = SCRATCH.with_borrow_mut(|(read, _)| {
            let at = std::time::Instant::now();
            read.resize((len as usize).max(read.len()), 0);
            let got = pack.read_mips(&entry, first..d.mips, read).ok()?;
            host::stage(host::STAGE_READ, at);
            pack_rows(&entry, from, &read[..got], &self.footprints, &self.rows, self.mips, self.mip_drop, first, base, upload.data)
        })?;
        self.copy_from.store(first - self.mip_drop, Relaxed);
        self.upload_base.store(base, Relaxed);
        ORIGIN_EVENTS[OR_FROM_PACK].fetch_add(1, Relaxed);
        ORIGIN_EVENTS[OR_BYTES_PACK].fetch_add(copied, Relaxed);
        Some(upload)
    }

    /// Records the copy of every subresource from `upload` (what
    /// [`Staging::refresh`] gave) into `texture` (in the copy destination
    /// state).
    /// Returns the bytes copied (what really moves, not the whole chain).
    pub(crate) fn record_copies(&self, list: &ID3D12GraphicsCommandList, texture: &ID3D12Resource, upload: &host::Upload) -> u64 {
        self.record_copies_at(list, texture, upload, None)
    }

    /// [`Staging::record_copies`] with the (first mip, upload base) the upload
    /// was filled with. A guard (E_INVALIDARG from our list near a
    /// cut): a copy whose source would run past the upload buffer, or a
    /// destination that is not this staging's texture shape, is not recorded
    /// (counted in `PASS_COPY_REFUSED`).
    pub(crate) fn record_copies_at(&self, list: &ID3D12GraphicsCommandList, texture: &ID3D12Resource, upload: &host::Upload, at: Option<(u32, u64)>) -> u64 {
        let (from, base) = at.unwrap_or_else(|| (self.copy_from.load(Relaxed), self.upload_base.load(Relaxed)));
        // SAFETY: live resources.
        let (cap, dst) = unsafe { (upload.buffer.GetDesc().Width, texture.GetDesc()) };
        let subresources = u32::from(dst.MipLevels.max(1)) * u32::from(dst.DepthOrArraySize.max(1));
        let fits = self.footprints.iter().enumerate().all(|(i, f)| {
            (i as u32 % self.mips) < from || (f.Offset >= base && f.Offset - base + u64::from(f.Footprint.RowPitch) * u64::from(self.rows[i]) * u64::from(f.Footprint.Depth) <= cap)
        });
        if !fits || self.footprints.len() as u32 > subresources {
            PASS_COPY_REFUSED.fetch_add(1, Relaxed);
            return 0;
        }
        let mut bytes = 0;
        for (i, footprint) in self.footprints.iter().enumerate() {
            if (i as u32 % self.mips) < from {
                continue; // Not arrived, not in the upload
            }
            let mut footprint = *footprint;
            footprint.Offset -= base; // The upload starts at the first mip copied
            let f = footprint.Footprint;
            bytes += u64::from(f.RowPitch) * u64::from(self.rows[i]) * u64::from(f.Depth);
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(&upload.buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
            };
            // SAFETY: an open list; both locations name live resources.
            unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
        ORIGIN_EVENTS[OR_BYTES_UPLOADED].fetch_add(bytes, Relaxed);
        bytes
    }

    /// An upload buffer of `bytes` within this frame's upload cap;
    /// `None` when the cap is spent or the pool has none.
    fn take_upload(&self, bytes: u64) -> Option<host::Upload> {
        if !frame_allow(FRAME_UPLOAD, bytes) {
            refused_here(0);
            return None;
        }
        let upload = host::upload_pool(bytes);
        if upload.is_none() {
            refused_here(1);
        }
        upload
    }

    /// The first host mip a pending origin copy holds whole from its
    /// upload (the mips of the pages arrived then); `None` when it holds
    /// none, is complete, or is not an origin copy.
    fn holds_from(&self, copy: &super::resource::HostTexture) -> Option<u32> {
        let uploaded = copy.uploaded.load(Relaxed);
        if uploaded == 0 {
            // A title-memory upload of a partly arrived texture.
            let from = self.plain_from.load(Relaxed);
            return (from < self.mips && copy.pending.load(Relaxed)).then_some(from);
        }
        if !copy.pending.load(Relaxed) || copy.origin.get().is_none() {
            return None;
        }
        let finest = self.finest_present(uploaded.saturating_sub(self.memory as u64 & 0xfff))?;
        Some(finest.max(self.mip_drop) - self.mip_drop)
    }

    /// The copy holds a title-memory upload made while only a prefix
    /// of the texture's pages had arrived (some of its mips are not whole).
    fn withheld(&self, copy: &super::resource::HostTexture) -> bool {
        self.plain_from.load(Relaxed) != NOT_PARTIAL && copy.uploaded.load(Relaxed) == 0 && copy.pending.load(Relaxed)
    }

    /// When the memory starts with a streamed texture's pages and not
    /// all of them have arrived: the first host mip whole in it (`mips`:
    /// none) and the prefix (bytes, stamp). `None`: all there, or memory
    /// with no recorded page at its start that never was a streamed
    /// texture's (the title's own contents), or `XNW_TEX_WHOLE=off`.
    fn arrived(&self) -> Option<(u32, u64, u64)> {
        if !whole_mips() {
            return None;
        }
        let Some(o) = xnw_formats::origin::peek(self.memory as u64, self.layout.size) else {
            // A streamed texture's memory whose first page is not recorded:
            // a request is rewriting it (a page is forgotten when its write
            // starts and recorded when it has completed), so the memory may
            // be half written: no mip is whole.
            let streamed = self.host.upgrade().is_some_and(|c| c.origin.get().is_some());
            return streamed.then_some((self.mips, 0, 0));
        };
        let present = o.len.saturating_sub(self.memory as u64 & 0xfff);
        if present >= self.layout.size {
            return None;
        }
        let from = self.finest_present(present).map_or(self.mips, |m| m.max(self.mip_drop) - self.mip_drop);
        (from > 0).then_some((from, o.len, o.stamp))
    }

    /// The memory still holds the copy's origin with no page arrived
    /// since its upload, and the title wrote none of it since this staging
    /// last asked (else the refresh decides, as [`Staging::refresh_origin`]).
    fn no_new_pages(&self, copy: &super::resource::HostTexture) -> bool {
        let o = xnw_formats::origin::peek(self.memory as u64, self.layout.size);
        let uploaded = copy.uploaded.load(Relaxed);
        if uploaded == 0 {
            // A title-memory upload: the same prefix of the same pages (none: still none).
            if !self.withheld(copy) || o.map_or((0, 0), |o| (o.len, o.stamp)) != (self.plain_at[0].load(Relaxed), self.plain_at[1].load(Relaxed)) {
                return false;
            }
        } else if !o.is_some_and(|o| copy.origin.get().is_some_and(|key| key.holds(&o)) && o.len == uploaded) {
            return false;
        }
        let asked = self.last.lock().unwrap_or_else(|e| e.into_inner()).asked;
        let start = self.memory as usize;
        super::watch::writes(start, start + self.layout.size as usize, false).is_none_or(|(_, w)| asked == 0 || !w.iter().any(|&(_, x)| x > asked))
    }

    /// Detiles `tiled` (the texture's memory) into the upload buffer at
    /// `data` ([`Staging::bytes`] long). The buffer is a mapped upload heap
    /// (write-combined), where the scattered writes of detiling are very
    /// slow: each subresource is detiled into the cached `linear` memory,
    /// then copied over in one sequential pass.
    fn detile(&self, tiled: &[u8], linear: &mut Vec<u8>, data: *mut u8) {
        DETILED.fetch_add(tiled.len() as u64, std::sync::atomic::Ordering::Relaxed);
        let started = std::time::Instant::now();
        self.detile_now(tiled, linear, data);
        host::stage(host::STAGE_DETILE, started);
    }

    fn detile_now(&self, tiled: &[u8], linear: &mut Vec<u8>, data: *mut u8) {
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips + self.mip_drop, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = pitch * rows as usize * f.Depth as usize;
            if linear.len() < len {
                linear.resize(len, 0);
            }
            let scratch = &mut linear[..len];
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, scratch, pitch, pitch * rows as usize)
            {
                xnw_core::warn!("d3d12x: texture {:?} not detiled (mip {mip}, slice {slice}): {e:?}", self.layout.desc);
                return;
            }
            // SAFETY: the subresource's part of the mapped upload buffer,
            // `len` bytes, which does not overlap the scratch memory.
            unsafe { std::ptr::copy_nonoverlapping(scratch.as_ptr(), data.add(footprint.Offset as usize), len) };
        }
    }
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize)) {
        s.used.store(super::queue::presented(), std::sync::atomic::Ordering::Relaxed);
        return Some(s.clone());
    }
    let started = std::time::Instant::now();
    let (memory, desc) = super::resource::texture_memory(resource)?;
    let Some(layout) = super::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone()) else {
        once!("d3d12x: texture contents without a console layout not uploaded: {desc:?}");
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!("d3d12x: texture contents in an XOR mode not uploaded: {desc:?}");
        return None;
    }
    let copy = super::resource::host_copy(resource);
    let s = Arc::new(Staging::new(texture, layout, memory, copy.as_ref(), Some(desc))?);
    map.insert(resource as usize, s.clone());
    host::stage(host::STAGE_STAGING, started);
    Some(s)
}

impl Staging {
    /// The upload state of `texture` (the host copy), whose contents are
    /// `layout` at `memory`.
    fn new(texture: &ID3D12Resource, layout: xnw_formats::texture::Layout, memory: *const u8, copy: Option<&Arc<super::resource::HostTexture>>, desc: Option<super::resource::ResourceDesc>) -> Option<Staging> {
        let host = host::host()?;
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            u32::from(d.MipLevels)
        } else {
            u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
        };
        let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
        let mut rows = vec![0u32; count as usize];
        let mut total = 0;
        // SAFETY: outputs for `count` subresources.
        unsafe {
            host.device.GetCopyableFootprints(&d, 0, count, 0, Some(footprints.as_mut_ptr()), Some(rows.as_mut_ptr()), None, Some(&mut total));
        }
        Some(Staging {
            viewed: true.into(),
            refreshed: 0.into(),
            took: false.into(),
            plain_from: NOT_PARTIAL.into(),
            plain_at: [0.into(), 0.into()],
            movie: false.into(),
            footprints,
            rows,
            layout,
            mips: u32::from(d.MipLevels),
            mip_drop: copy.map_or(0, |c| c.dropped),
            copy_from: 0.into(),
            upload_base: 0.into(),
            trapped: false.into(),
            host: copy.map_or_else(std::sync::Weak::new, Arc::downgrade),
            desc,
            memory,
            last: Mutex::default(),
            bytes: total,
            used: super::queue::presented().into(),
        })
    }
}

/// Frames a staging state is kept after its last use.
const STAGING_KEEP_FRAMES: u64 = 120;

/// At `PresentX`, every 32 frames: drops the staging state of textures no
/// list asked to bring up to date in [`STAGING_KEEP_FRAMES`] (footprints
/// and what was last detiled). Lists that hold one keep it until reset.
/// The next use makes it again and uploads the texture whole once, which
/// is what a texture cold that long costs. Returns the states dropped.
pub(crate) fn sweep(frame: u64) -> u64 {
    use std::sync::atomic::Ordering::Relaxed;
    if !frame.is_multiple_of(32) {
        return 0;
    }
    let mut dropped = Vec::new();
    {
        let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        let Some(map) = map.as_mut() else { return 0 };
        map.retain(|_, s| {
            let keep = s.used.load(Relaxed) + STAGING_KEEP_FRAMES > frame;
            if !keep {
                dropped.push(s.clone());
            }
            keep
        });
    }
    host::TEXTURE_EVENTS[host::EV_STAGING_DROPPED].fetch_add(dropped.len() as u64, Relaxed);
    sweep_ahead(frame);
    dropped.len() as u64
}

/// Upload memory held for texture uploads now (bytes: the pool's), for the
/// memory line.
pub(crate) fn staging_bytes() -> u64 {
    host::upload_pool_bytes()
}

// Origins. Memory that holds the pages of one `.ttpf` texture, in file
// order (`xnw_formats::origin::origin_of`), holds that file's bytes: the
// same origin anywhere, any time, has the same contents. So one host copy
// per origin and description is kept (a placement of the origin at another
// address, or after the title let it go, finds it: no creation, no upload),
// uploaded once from the decoded pages, never hashed. Memory that stops
// holding its origin's pages (or the write watch sees the title write it)
// takes the hashing way again, and the copy is no longer shared.

/// An origin cache key: the texture whose pages lie under the memory
/// (file, texture, file address of the memory's first byte) and the
/// description's shape (dimension, width, height, depth or array size,
/// mips, format). How much of it has arrived is the content version
/// (`HostTexture::uploaded`), not part of the key: a texture placed over
/// memory whose low mips have arrived is the same texture later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OriginKey {
    file: u32,
    texture: u32,
    at: u64,
    shape: [u32; 6],
}

impl OriginKey {
    /// Its registry file and texture (what [`xnw_formats::origin::arrived_since`] names).
    pub fn file_texture(&self) -> (u32, u32) {
        (self.file, self.texture)
    }

    fn holds(&self, o: &xnw_formats::origin::Origin) -> bool {
        (o.file, o.texture, o.at) == (self.file, self.texture, self.at)
    }
}

fn shape(desc: &super::resource::ResourceDesc) -> [u32; 6] {
    [desc.dimension, desc.width as u32, desc.height, u32::from(desc.depth_or_array_size), super::resource::mip_count(desc), desc.format]
}

/// Origin counters for the frame statistics: placements that found a
/// cached copy, copies made for an origin, uploads that fell back to
/// hashing, bytes not hashed, uploads from decoded pages and from title
/// memory, copies made ahead, prepared uploads used, ahead copies dropped
/// unused, placements that found an ahead copy, copies made with mips
/// dropped, arrivals not queued (queue full); host copies not looked up
/// (target/depth/UAV flags, samples or buffer, no console layout), origins
/// found at a later refresh, uploads of more arrived mips, ahead copies
/// prepared again with more pages.
pub(crate) static ORIGIN_EVENTS: [AtomicU64; 35] = [const { AtomicU64::new(0) }; 35];
const OR_HIT: usize = 0;
const OR_MADE: usize = 1;
const OR_FALLBACK: usize = 2;
const OR_HASH_AVOIDED: usize = 3;
const OR_FROM_PAGES: usize = 4;
const OR_FROM_MEMORY: usize = 5;
const OR_AHEAD_MADE: usize = 6;
const OR_AHEAD_USED: usize = 7;
const OR_AHEAD_DROPPED: usize = 8;
const OR_AHEAD_HIT: usize = 9;
const OR_MIPS_DROPPED: usize = 10;
const OR_ARRIVALS_LOST: usize = 11;
const OR_SKIP_FLAGS: usize = 12;
const OR_SKIP_KIND: usize = 13;
const OR_SKIP_LAYOUT: usize = 14;
const OR_LATE: usize = 15;
const OR_UPGRADED: usize = 16;
const OR_AHEAD_AGAIN: usize = 17;
const OR_FROM_PACK: usize = 18;
/// Ahead copies made without a prepared upload (staging budget used up).
const OR_AHEAD_BARE: usize = 19;

/// Bytes uploaded from the pack, the decoded pages and title memory (origin
/// way), and textures and bytes uploaded the hashing way (title memory).
const OR_BYTES_PACK: usize = 20;
const OR_BYTES_PAGES: usize = 21;
const OR_BYTES_MEMORY: usize = 22;
const OR_PLAIN: usize = 23;
const OR_BYTES_PLAIN: usize = 24;
/// Textures preloaded (made and prepared), preloaded bytes, preloads
/// found at first use, their prepared upload used, preloads never used.
const OR_PRELOADED: usize = 25;
const OR_PRELOAD_BYTES: usize = 26;
const OR_PRELOAD_HIT: usize = 27;
const OR_PRELOAD_USED: usize = 28;
const OR_PRELOAD_WASTED: usize = 29;
/// Kept copies trimmed for the keep cap, released for the texture
/// budget; uploads and placed creations put off by the per-frame caps.
const OR_TRIMMED: usize = 30;
const OR_BUDGET_RELEASED: usize = 31;
const OR_UPLOADS_DEFERRED: usize = 32;
const OR_CREATES_DEFERRED: usize = 33;
/// Bytes uploads really copy (what the prelude records).
const OR_BYTES_UPLOADED: usize = 34;

/// Preloaded copies nobody placed yet (bytes), and how long they are kept.
static PRELOADED: AtomicU64 = AtomicU64::new(0);
const PRELOAD_KEEP_FRAMES: u64 = 3600;

use std::sync::atomic::AtomicU64;

/// The origin part of the frame statistics, over `n` frames, zeroing the
/// counts (totals over the frames, not per frame).
pub(crate) fn origin_events_text(n: u64) -> String {
    format!("{}; {}; {}; {}", origin_text(), super::resource::placed_text(n), budget_text(), x1_text())
}

/// Parts of the stats window: the bypass counters (totals), the
/// preload, and with read traps the CPU reads of texture pages.
fn x1_text() -> String {
    let b = xnw_formats::origin::registry().bypass_counts();
    let e: Vec<u64> = ORIGIN_EVENTS[OR_PRELOADED..=OR_PRELOAD_WASTED].iter().map(|c| c.swap(0, Relaxed)).collect();
    format!(
        "texture pages (totals): {} requests, {} bypassed, {} filled on demand, {} MiB not read, {} MiB not decoded, ~{} ms decoding avoided; preload: {} made ({} MiB), {} found at first use, {} uploads used, {} never used, {} MiB waiting{}",
        b[2],
        b[0],
        b[1],
        b[3] >> 20,
        b[4] >> 20,
        b[5] / 1000,
        e[0],
        e[1] >> 20,
        e[2],
        e[3],
        e[4],
        PRELOADED.load(Relaxed) >> 20,
        trap_text()
    )
}

fn budget_text() -> String {
    let e: Vec<u64> = ORIGIN_EVENTS[OR_TRIMMED..=OR_BYTES_UPLOADED].iter().map(|c| c.swap(0, Relaxed)).collect();
    format!(
        "budget (totals): {} textures pending an upload now, {} kept copies trimmed, {} released for the budget, {} uploads and {} creations deferred (per-frame cap {} MiB), {} MiB uploaded",
        PENDING.load(Relaxed),
        e[0],
        e[1],
        e[2],
        e[3],
        frame_cap().min(u64::MAX >> 20 << 20) >> 20,
        e[4] >> 20
    )
}

fn origin_text() -> String {
    // Only its own counters; the preload and
    // budget ones are [`x1_text`]'s and [`budget_text`]'s, formatted after.
    let e: Vec<u64> = ORIGIN_EVENTS[..OR_PRELOADED].iter().map(|c| c.swap(0, Relaxed)).collect();
    let (copies, bytes) = {
        let o = ORIGINS.lock().unwrap_or_else(|e| e.into_inner());
        o.as_ref().map_or((0, 0), |o| (o.map.len(), o.bytes))
    };
    format!(
        "texture refreshes (totals): {} changed with a view made since, {} changed without one, {} skipped (XNW_TEX_REFRESH=view); origins (totals): {} hits ({} on ahead copies), {} made ({} with mips dropped), {} found later, {} upgraded with more mips, {} fallbacks, hashing avoided {} KiB, uploads (textures, KiB) from pack {} {} / pages {} {} / memory {} {}, hashing way {} {}; not looked up: {} flags, {} samples/buffer, {} no layout; ahead: {} made ({} without an upload), {} prepared again, {} used, {} dropped, {} arrivals lost, {} MiB unplaced, {} KiB staged; cache {copies} copies, {} MiB",
        REFRESH_COUNTS[0].swap(0, Relaxed),
        REFRESH_COUNTS[1].swap(0, Relaxed),
        REFRESH_COUNTS[2].swap(0, Relaxed),
        e[OR_HIT],
        e[OR_AHEAD_HIT],
        e[OR_MADE],
        e[OR_MIPS_DROPPED],
        e[OR_LATE],
        e[OR_UPGRADED],
        e[OR_FALLBACK],
        e[OR_HASH_AVOIDED] / 1024,
        e[OR_FROM_PACK],
        e[OR_BYTES_PACK] >> 10,
        e[OR_FROM_PAGES],
        e[OR_BYTES_PAGES] >> 10,
        e[OR_FROM_MEMORY],
        e[OR_BYTES_MEMORY] >> 10,
        e[OR_PLAIN],
        e[OR_BYTES_PLAIN] >> 10,
        e[OR_SKIP_FLAGS],
        e[OR_SKIP_KIND],
        e[OR_SKIP_LAYOUT],
        e[OR_AHEAD_MADE],
        e[OR_AHEAD_BARE],
        e[OR_AHEAD_AGAIN],
        e[OR_AHEAD_USED],
        e[OR_AHEAD_DROPPED],
        e[OR_ARRIVALS_LOST],
        UNPLACED.load(Relaxed) >> 20,
        AHEAD_BYTES.load(Relaxed) / 1024,
        bytes >> 20
    )
}

/// Logs the first `max` of a kind of lookup (`XNW_FRAME_STATS` runs read
/// them to see how placements relate to pages).
fn sample(seen: &AtomicU64, max: u64, line: impl FnOnce() -> String) {
    if seen.fetch_add(1, Relaxed) < max {
        xnw_core::info!("d3d12x origin: {}", line());
    }
}

fn shape_text(desc: &super::resource::ResourceDesc, size: u64) -> String {
    format!("{}x{} mips {} format {} layout {:#x} flags {:#x} size {size:#x}", desc.width, desc.height, desc.mip_levels, desc.format, desc.layout, desc.flags)
}

/// The origin of a sampled-only texture (not a target, depth or unordered
/// access) whose console layout at `memory` starts with one texture's
/// pages (all, or its low mips first).
pub(crate) fn origin_key(memory: *const u8, desc: &super::resource::ResourceDesc) -> Option<OriginKey> {
    static SKIPPED: AtomicU64 = AtomicU64::new(0);
    static MISSED: AtomicU64 = AtomicU64::new(0);
    static FOUND: AtomicU64 = AtomicU64::new(0);
    static OTHER_SHAPE: AtomicU64 = AtomicU64::new(0);
    if desc.flags & 0xf != 0 {
        ORIGIN_EVENTS[OR_SKIP_FLAGS].fetch_add(1, Relaxed);
        sample(&SKIPPED, 10, || format!("not looked up (flags): {}", shape_text(desc, 0)));
        return None;
    }
    if desc.sample_count > 1 || desc.dimension == super::resource::BUFFER {
        ORIGIN_EVENTS[OR_SKIP_KIND].fetch_add(1, Relaxed);
        return None;
    }
    start_ahead();
    let Some(size) = super::resource::texture_layout(desc).map(|t| t.planes[0].layout.size) else {
        ORIGIN_EVENTS[OR_SKIP_LAYOUT].fetch_add(1, Relaxed);
        return None;
    };
    let (o, cut) = xnw_formats::origin::prefix_of(memory as u64, size);
    let Some(o) = o else {
        sample(&MISSED, 20, || format!("miss ({cut:?}): {} at {memory:?}: {}", shape_text(desc, size), xnw_formats::origin::describe(memory as u64)));
        return None;
    };
    let key = OriginKey { file: o.file, texture: o.texture, at: o.at, shape: shape(desc) };
    sample(&FOUND, 20, || format!("found ({cut:?}, {:#x} of {size:#x} bytes, mips {:?}): {}: {}", o.len, o.mips, shape_text(desc, size), xnw_formats::origin::describe(memory as u64)));
    if OTHER_SHAPE.load(Relaxed) < 20
        && let Some(other) = origins().as_ref().and_then(|c| c.map.iter().find(|(k, e)| e.ahead && k.file == key.file && k.texture == key.texture && **k != key).map(|(k, _)| *k))
    {
        sample(&OTHER_SHAPE, 20, || format!("placed {key:?} where ahead guessed {other:?}"));
    }
    Some(key)
}

/// A refresh of a copy made without an origin: the memory may hold one
/// now (pages that arrived after the first use). Not counted in the
/// registry's lookups.
fn late_key(memory: *const u8, desc: &super::resource::ResourceDesc) -> Option<OriginKey> {
    if desc.flags & 0xf != 0 || desc.sample_count > 1 || desc.dimension == super::resource::BUFFER {
        return None;
    }
    let size = super::resource::texture_layout(desc)?.planes[0].layout.size;
    let o = xnw_formats::origin::peek(memory as u64, size)?;
    Some(OriginKey { file: o.file, texture: o.texture, at: o.at, shape: shape(desc) })
}

/// Bytes a host copy holds, roughly (texels; a block-compressed byte or
/// half a byte each).
fn copy_bytes(copy: &super::resource::HostTexture) -> u64 {
    super::resource::texture_bytes(&copy.texture).max(0x1000)
}

/// A kept copy: its last use (presented frame), bytes (estimated) and
/// whether it was made ahead and not placed yet.
struct Entry {
    copy: Arc<super::resource::HostTexture>,
    used: u64,
    /// Last touch in [`TICK`] order (least recently used goes first,
    /// whatever the frame pace).
    tick: u64,
    bytes: u64,
    ahead: bool,
    /// Made from the preload list (not by an arrival).
    preload: bool,
}

struct Origins {
    map: HashMap<OriginKey, Entry>,
    bytes: u64,
}

static ORIGINS: Mutex<Option<Origins>> = Mutex::new(None);

fn origins() -> std::sync::MutexGuard<'static, Option<Origins>> {
    ORIGINS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Host copies kept by origin at most (`XNW_ORIGIN_KEEP_MB`, default
/// 1024): past it the least recently used cold ones go, to three quarters.
fn keep_budget() -> u64 {
    static MB: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| std::env::var("XNW_ORIGIN_KEEP_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1024));
    *MB << 20
}

/// The kept copy of `key`, if any: a hit.
pub(crate) fn origin_cached(key: &OriginKey) -> Option<Arc<super::resource::HostTexture>> {
    let mut o = origins();
    let e = o.as_mut()?.map.get_mut(key)?;
    e.used = super::queue::presented();
    e.tick = TICK.fetch_add(1, Relaxed);
    if std::mem::take(&mut e.ahead) {
        if std::mem::take(&mut e.preload) {
            ORIGIN_EVENTS[OR_PRELOAD_HIT].fetch_add(1, Relaxed);
            PRELOADED.fetch_sub(e.bytes, Relaxed);
        } else {
            ORIGIN_EVENTS[OR_AHEAD_HIT].fetch_add(1, Relaxed);
            UNPLACED.fetch_sub(e.bytes, Relaxed);
            AHEAD_LIVE.fetch_sub(1, Relaxed);
        }
    }
    ORIGIN_EVENTS[OR_HIT].fetch_add(1, Relaxed);
    Some(e.copy.clone())
}

/// Keeps `copy`, made for `key`.
pub(crate) fn origin_insert(key: OriginKey, copy: &Arc<super::resource::HostTexture>) {
    ORIGIN_EVENTS[OR_MADE].fetch_add(1, Relaxed);
    if copy.dropped > 0 {
        ORIGIN_EVENTS[OR_MIPS_DROPPED].fetch_add(1, Relaxed);
    }
    keep(key, copy, false);
}

fn keep(key: OriginKey, copy: &Arc<super::resource::HostTexture>, ahead: bool) {
    keep_entry(key, copy, ahead, false);
}

fn keep_entry(key: OriginKey, copy: &Arc<super::resource::HostTexture>, ahead: bool, preload: bool) {
    let bytes = copy_bytes(copy);
    let frame = super::queue::presented();
    let mut dropped = Vec::new();
    {
        let mut o = origins();
        let o = o.get_or_insert_with(|| Origins { map: HashMap::new(), bytes: 0 });
        if preload {
            PRELOADED.fetch_add(bytes, Relaxed);
        } else if ahead {
            UNPLACED.fetch_add(bytes, Relaxed);
            AHEAD_LIVE.fetch_add(1, Relaxed);
        }
        let tick = TICK.fetch_add(1, Relaxed);
        if let Some(old) = o.map.insert(key, Entry { copy: copy.clone(), used: frame, tick, bytes, ahead, preload }) {
            o.bytes -= old.bytes;
            dropped.push(old);
        }
        o.bytes += bytes;
        // The cap holds whatever the frame pace (it went by frames
        // before: in 4-7 s frames nothing was ever cold, 15,812 copies,
        // 8.3 GiB, a test run). Copies only this map holds go first, least
        // recently used first; one a resource still holds frees nothing.
        let budget = keep_budget();
        if o.bytes > budget {
            trim(o, o.bytes - budget / 4 * 3, &mut dropped);
        }
    }
    count_dropped(&dropped);
}

/// LRU order for kept copies.
static TICK: AtomicU64 = AtomicU64::new(0);

/// Takes kept copies out of the map until `want` bytes are out: those only
/// the map holds first (least recently used first), then the others.
/// Returns the bytes of the first kind (what releasing them frees).
fn trim(o: &mut Origins, want: u64, dropped: &mut Vec<Entry>) -> u64 {
    let order: Vec<(bool, u64, OriginKey, u64)> = o.map.iter().map(|(k, e)| (Arc::strong_count(&e.copy) > 1, e.tick, *k, e.bytes)).collect();
    let (keys, freed) = lru_release(order, want, true);
    for k in keys {
        if let Some(e) = o.map.remove(&k) {
            o.bytes -= e.bytes;
            dropped.push(e);
        }
    }
    ORIGIN_EVENTS[OR_TRIMMED].fetch_add(dropped.len() as u64, Relaxed);
    freed
}

/// Which kept copies go for `want` bytes: `(held elsewhere, last use,
/// key, bytes)`, those nobody else holds first, least recently used first;
/// with `held_too` the others after them. The keys and the bytes freed
/// (of copies nobody else holds).
fn lru_release<K: Copy + Ord>(mut order: Vec<(bool, u64, K, u64)>, want: u64, held_too: bool) -> (Vec<K>, u64) {
    order.sort_unstable();
    let (mut out, mut freed, mut keys) = (0, 0, Vec::new());
    for (held, _, k, bytes) in order {
        if out >= want || held && !held_too {
            break;
        }
        keys.push(k);
        out += bytes;
        if !held {
            freed += bytes;
        }
    }
    (keys, freed)
}

/// Makes room for `need` bytes of a new host texture within the
/// texture budget by releasing kept copies nobody else holds (least
/// recently used first); whether it fits now.
pub(crate) fn make_room(need: u64) -> bool {
    let budget = super::resource::tex_budget();
    let used = super::resource::host_texture_bytes();
    if used + need <= budget {
        return true;
    }
    // At the budget with every kept copy in use, each ask collected and
    // sorted all kept copies (5,038) to release none, about four asks per
    // texture made (one per mip step dropped): 2,121 steps and 686 ms of
    // making a frame for 643 made. An ask
    // that freed nothing is not repeated in the frame until bytes go.
    let frame = super::queue::presented();
    {
        let failed = ROOM_FAILED.lock().unwrap_or_else(|e| e.into_inner());
        if failed.0 == frame && used >= failed.1 {
            ROOM_COUNTS[1].fetch_add(1, Relaxed);
            return false;
        }
    }
    let started = std::time::Instant::now();
    let fits = make_room_now(need, budget, used);
    ROOM_COUNTS[0].fetch_add(1, Relaxed);
    ROOM_COUNTS[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
    if !fits {
        *ROOM_FAILED.lock().unwrap_or_else(|e| e.into_inner()) = (frame, super::resource::host_texture_bytes());
    }
    fits
}

/// The frame and host texture bytes of the last [make_room] that
/// could not make room.
static ROOM_FAILED: Mutex<(u64, u64)> = Mutex::new((u64::MAX, 0));
/// Room asks over the budget that looked at the kept copies, that were
/// answered by the frame's failure, and microseconds looking.
pub(crate) static ROOM_COUNTS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

fn make_room_now(need: u64, budget: u64, used: u64) -> bool {
    let mut dropped = Vec::new();
    let freed = {
        let mut o = origins();
        let Some(o) = o.as_mut() else { return false };
        let want = used + need - budget;
        let order: Vec<(bool, u64, OriginKey, u64)> = o.map.iter().map(|(k, e)| (Arc::strong_count(&e.copy) > 1, e.tick, *k, e.bytes)).collect();
        let (keys, freed) = lru_release(order, want, false);
        for k in keys {
            if let Some(e) = o.map.remove(&k) {
                o.bytes -= e.bytes;
                dropped.push(e);
            }
        }
        freed
    };
    ORIGIN_EVENTS[OR_BUDGET_RELEASED].fetch_add(dropped.len() as u64, Relaxed);
    count_dropped(&dropped);
    drop(dropped); // the copies go (their bytes leave the census now)
    freed > 0 && super::resource::host_texture_bytes() + need <= budget
}

/// The largest number of top mips a sampled 2D texture may lose (its
/// coarsest kept mip a whole number of 4x4 blocks, one mip at least).
pub(crate) fn max_drop(desc: &super::resource::ResourceDesc, mips: u32) -> u32 {
    const TEXTURE_2D: u32 = 3;
    if desc.dimension != TEXTURE_2D || desc.sample_count > 1 {
        return 0;
    }
    (0..mips).rev().find(|&k| mips > k && (desc.width >> k) >= 4 && u64::from(desc.height >> k) >= 4 && (desc.width >> k) % 4 == 0 && (desc.height >> k) % 4 == 0).unwrap_or(0)
}

/// Per-frame caps, `XNW_TEX_FRAME_MB` (default 64) each, on bytes of
/// placed textures made and of texture uploads; the first of a frame always
/// passes.
pub(crate) const FRAME_UPLOAD: usize = 1;
static FRAME_SPENT: Mutex<(u64, [u64; 2])> = Mutex::new((u64::MAX, [0; 2]));

fn frame_cap() -> u64 {
    static MB: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| std::env::var("XNW_TEX_FRAME_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(64));
    if *MB == 0 { u64::MAX } else { *MB << 20 }
}

/// Spends `bytes` of this frame's `kind` cap; false (nothing spent) past it.
pub(crate) fn frame_allow(kind: usize, bytes: u64) -> bool {
    let frame = super::queue::presented();
    let mut s = FRAME_SPENT.lock().unwrap_or_else(|e| e.into_inner());
    if s.0 != frame {
        *s = (frame, [0; 2]);
    }
    if s.1[kind] > 0 && s.1[kind] + bytes > frame_cap() {
        ORIGIN_EVENTS[if kind == FRAME_UPLOAD { OR_UPLOADS_DEFERRED } else { OR_CREATES_DEFERRED }].fetch_add(1, Relaxed);
        return false;
    }
    s.1[kind] += bytes;
    true
}

/// Whether this frame's `kind` cap has room left (nothing spent).
fn frame_room(kind: usize) -> bool {
    let frame = super::queue::presented();
    let s = FRAME_SPENT.lock().unwrap_or_else(|e| e.into_inner());
    s.0 != frame || s.1[kind] < frame_cap()
}

fn count_dropped(dropped: &[Entry]) {
    for e in dropped.iter().filter(|e| e.ahead) {
        if e.preload {
            PRELOADED.fetch_sub(e.bytes, Relaxed);
            ORIGIN_EVENTS[OR_PRELOAD_WASTED].fetch_add(1, Relaxed);
        } else {
            UNPLACED.fetch_sub(e.bytes, Relaxed);
            AHEAD_LIVE.fetch_sub(1, Relaxed);
            ORIGIN_EVENTS[OR_AHEAD_DROPPED].fetch_add(1, Relaxed);
        }
    }
}

/// Memory no longer holds `key`'s pages: its copy is not shared again.
fn origin_forget(key: &OriginKey) {
    let removed = origins().as_mut().and_then(|o| {
        let e = o.map.remove(key)?;
        o.bytes -= e.bytes;
        Some(e)
    });
    if let Some(e) = removed {
        count_dropped(std::slice::from_ref(&e));
        drop(e); // outside the lock: may release the host copy
    }
}

/// At the `sweep`: ahead copies nobody placed in [`STAGING_KEEP_FRAMES`]
/// go (with their prepared upload).
fn sweep_ahead(frame: u64) {
    let mut dropped = Vec::new();
    if let Some(o) = origins().as_mut() {
        let keep_for = |e: &Entry| if e.preload { PRELOAD_KEEP_FRAMES } else { STAGING_KEEP_FRAMES };
        let stale: Vec<OriginKey> = o.map.iter().filter(|(_, e)| e.ahead && e.used + keep_for(e) <= frame).map(|(k, _)| *k).collect();
        for k in stale {
            if let Some(e) = o.map.remove(&k) {
                o.bytes -= e.bytes;
                dropped.push(e);
            }
        }
    }
    count_dropped(&dropped);
}

// Read traps. The trap facility is the buffer roles' (`watch::trap_reads`,
// `read_trap_stats`, `untrap`); until it is on master it is wired here at
// merge with `use_read_traps`.

/// The read-trap entry points: trap `(start, len, tag)`, stats of a tag
/// `(CPU reads, CPU writes, title sites)`, untrap `(start, len)`.
pub struct ReadTraps {
    pub trap: fn(usize, usize, u32),
    pub stats: fn(u32) -> (u64, u64, Vec<u64>),
    pub untrap: fn(usize, usize),
}

static READ_TRAPS: std::sync::OnceLock<ReadTraps> = std::sync::OnceLock::new();
static TRAP_TAGS: Mutex<Vec<u32>> = Mutex::new(Vec::new());
static TRAPPED: AtomicU64 = AtomicU64::new(0);

/// Wires the read traps (once).
pub fn use_read_traps(traps: ReadTraps) {
    let _ = READ_TRAPS.set(traps);
}

fn bypass_mode_set() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| matches!(std::env::var("XNW_TEXBYPASS").ok().as_deref().map(str::trim), Some("on" | "probe")));
    *ON
}

/// Textures trapped, CPU reads and writes over their tags, the five most
/// frequent read sites. A read of a bypassed page (tag 0x1FFF) stops the
/// bypass: the title's CPU does read texture pages.
fn trap_text() -> String {
    let Some(t) = READ_TRAPS.get() else {
        return if bypass_mode_set() { "; read traps not wired".into() } else { String::new() };
    };
    let tags = TRAP_TAGS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let (mut reads, mut writes, mut sites, mut bypassed_reads) = (0, 0, HashMap::<u64, u64>::new(), 0);
    for tag in tags {
        let (r, w, at) = (t.stats)(tag);
        reads += r;
        writes += w;
        if tag == 0x1FFF {
            bypassed_reads += r;
        }
        for a in at {
            *sites.entry(a).or_default() += 1;
        }
    }
    if bypassed_reads > 0 && xnw_formats::origin::bypass_allowed() {
        xnw_formats::origin::stop_bypass();
        xnw_core::warn!("d3d12x: the title's CPU read {bypassed_reads} times from bypassed texture pages: bypass stopped for new requests");
    }
    let mut top: Vec<(u64, u64)> = sites.into_iter().map(|(a, n)| (n, a)).collect();
    top.sort_unstable_by(|a, b| b.cmp(a));
    let top: Vec<String> = top.iter().take(5).map(|(n, a)| format!("title+{a:#x} x{n}")).collect();
    format!("; read traps: {} textures, CPU reads {reads}, CPU writes {writes}, top sites [{}]", TRAPPED.load(Relaxed), top.join(", "))
}

// Preload: the textures a mission's load asked for on an earlier run,
// made and uploaded from the pack ahead of the title, on one low-priority
// worker, bounded by `XNW_PRELOAD_MB` (default 1536) and stopped when video
// memory passes its limit. Each is a copy kept under its origin key at the
// whole chain (the shape placements take, V), holding the mips asked
// then; the first placement finds it and its first refresh hands over the
// prepared upload (no read, no detile, no creation in the frame).

fn preload_budget() -> u64 {
    static MB: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| std::env::var("XNW_PRELOAD_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1536));
    *MB << 20
}

/// Starts preloading `items` (registry file, texture, finest mip asked).
pub fn preload_textures(items: Vec<(u32, u32, u32)>) {
    if preload_budget() == 0 || pack().is_none() {
        return;
    }
    let _ = std::thread::Builder::new().name("texture preload".into()).spawn(move || {
        const THREAD_PRIORITY_LOWEST: i32 = -2;
        // SAFETY: this thread's own pseudo-handle.
        unsafe { xnw_core::win::SetThreadPriority(xnw_core::win::GetCurrentThread(), THREAD_PRIORITY_LOWEST) };
        let started = std::time::Instant::now();
        let mut made = 0;
        for (file, texture, mip) in items {
            if PRELOADED.load(Relaxed) >= preload_budget() || super::resource::vram_over() || super::resource::host_texture_bytes() >= super::resource::tex_budget() / 2 {
                break;
            }
            if preload_one(file, texture, mip).is_some() {
                made += 1;
            }
            std::thread::yield_now();
        }
        xnw_core::info!("d3d12x: preload: {made} textures ({} MiB) in {:.1} s", PRELOADED.load(Relaxed) >> 20, started.elapsed().as_secs_f64());
    });
}

fn preload_one(file: u32, texture: u32, mip: u32) -> Option<()> {
    use xnw_formats::ttpf;
    let index = xnw_formats::origin::file(file)?;
    let tex = *index.texture(texture)?;
    if !(tex.kind == ttpf::KIND_2D || tex.kind == ttpf::KIND_ARRAY) {
        return None;
    }
    let desc = super::resource::ResourceDesc {
        dimension: 3,
        width: u64::from(tex.width),
        height: tex.height,
        depth_or_array_size: tex.layers as u16,
        mip_levels: tex.mips as u16,
        format: tex.format,
        sample_count: 1,
        layout: 0x100 | ttpf::swizzle(tex.mode).0,
        ..Default::default()
    };
    let key = OriginKey { file, texture, at: tex.addr, shape: shape(&desc) };
    if origins().as_ref().is_some_and(|o| o.map.contains_key(&key)) {
        return None;
    }
    let layout = super::resource::texture_layout(&desc)?.planes[0].layout.clone();
    let copy = super::resource::host_copy_made(&desc, 0, std::ptr::null(), Some(key), mips_to_drop(&desc, tex.mips).max(pack_drop(&key)))?;
    let staging = Staging::new(&copy.texture, layout.clone(), std::ptr::null(), Some(&copy), None)?;
    let first = mip.max(copy.dropped);
    let upload = staging.from_pack(file, texture, first)?;
    AHEAD_BYTES.fetch_add(staging.bytes, Relaxed);
    *copy.prepared.lock().unwrap_or_else(|e| e.into_inner()) = Some(Prepared { layout: layout.desc, bytes: staging.bytes, mip_drop: copy.dropped, len: 0, pack_first: Some(first), base: staging.upload_base.load(Relaxed), upload: Some(upload) });
    ORIGIN_EVENTS[OR_PRELOADED].fetch_add(1, Relaxed);
    ORIGIN_EVENTS[OR_PRELOAD_BYTES].fetch_add(copy_bytes(&copy), Relaxed);
    keep_entry(key, &copy, true, true);
    Some(())
}

/// The texture pack (`xnw_formats::origin::pack`), logged once.
fn pack() -> Option<&'static xnw_formats::texpack::PackDir> {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    let p = xnw_formats::origin::pack();
    LOGGED.call_once(|| {
        if let Some(p) = p {
            xnw_core::info!("d3d12x: texture pack: {} files", p.packs.len());
        }
    });
    p
}

/// Mips a texture pack dropped at conversion for `key`'s texture: its host
/// copy leaves them out too (views shift as for `XNW_MIP_DROP`).
pub(crate) fn pack_drop(key: &OriginKey) -> u32 {
    let entry = || pack()?.lookup(xnw_formats::origin::file(key.file)?.pack_id?, key.texture);
    entry().map_or(0, |e| u32::from(e.first))
}

/// Copies the pack's rows of console mips `first..` (read from file offset
/// `from` into `bytes`) into the upload at `data`, laid out by `footprints`
/// (host mips `mips` per slice, `mip_drop` console mips left out); the
/// bytes copied, `None` if a mip is missing or does not fit.
#[allow(clippy::too_many_arguments)]
fn pack_rows(entry: &xnw_formats::texpack::Entry, from: u64, bytes: &[u8], footprints: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT], rows: &[u32], mips: u32, mip_drop: u32, first: u32, base: u64, data: *mut u8) -> Option<u64> {
    let mut copied = 0u64;
    for (i, (footprint, &n)) in footprints.iter().zip(rows).enumerate() {
        let (m, slice) = (i as u32 % mips + mip_drop, i as u32 / mips);
        if m < first {
            continue;
        }
        let loc = entry.mip(m)?;
        let f = footprint.Footprint;
        let row = loc.row_pitch.min(f.RowPitch) as usize;
        let lines = n.min(loc.rows) as usize;
        let src_at = (loc.offset - from) as usize + slice as usize * loc.slice_pitch as usize;
        if src_at + (lines.max(1) - 1) * loc.row_pitch as usize + row > bytes.len() || footprint.Offset < base {
            return None;
        }
        for r in 0..lines {
            let src = &bytes[src_at + r * loc.row_pitch as usize..][..row];
            // SAFETY: row `r` of the subresource's part of the mapped upload
            // buffer (`RowPitch` apart, `row` <= `RowPitch`).
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), data.add((footprint.Offset - base) as usize + r * f.RowPitch as usize), row) };
        }
        copied += (lines * row) as u64;
    }
    Some(copied)
}

/// Fills `out` from its start with the decoded bytes of `file`'s virtual
/// range from `at`, page by page while the origin registry's cache has
/// them; the bytes filled.
fn tiled_prefix(file: u32, at: u64, out: &mut [u8]) -> usize {
    let Some(index) = xnw_formats::origin::file(file) else { return 0 };
    let len = out.len() as u64;
    let mut covered = at;
    for i in index.ttpf.pages_of(at, len) {
        let p = index.ttpf.pages[i];
        let (from, to) = (p.addr.max(at), (p.addr + p.size).min(at + len));
        if from != covered {
            break;
        }
        let Some(bytes) = xnw_formats::origin::decoded_page(file, i as u32) else { break };
        if bytes.len() < (to - p.addr) as usize {
            break;
        }
        out[(from - at) as usize..(to - at) as usize].copy_from_slice(&bytes[(from - p.addr) as usize..(to - p.addr) as usize]);
        covered = to;
    }
    (covered - at) as usize
}

/// `XNW_MIP_DROP=<n>` (default 0, at most 4): sampled origin textures get
/// host copies without their top `n` mips (fewer when the chain or the
/// size would not allow it; the top kept mip stays a whole number of 4x4
/// blocks). After VIctory (9a85db7) `compute_mip_drop`.
pub(crate) fn mips_to_drop(desc: &super::resource::ResourceDesc, mips: u32) -> u32 {
    static DROP: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| std::env::var("XNW_MIP_DROP").ok().and_then(|v| v.trim().parse::<u32>().ok()).map_or(0, |v| v.min(4)));
    const TEXTURE_2D: u32 = 3;
    if desc.dimension != TEXTURE_2D || desc.sample_count > 1 {
        return 0;
    }
    (1..=*DROP).rev().find(|&k| mips > k && (desc.width >> k) >= 4 && (desc.height >> k) >= 4 && (desc.width >> k) % 4 == 0 && (desc.height >> k) % 4 == 0).unwrap_or(0)
}

/// A shader resource view of `resource`, whose host copy `info` has mips
/// dropped: the view's mips move down by as many (VIctory,
/// `adjust_srv_for_dropped_mips`).
pub(crate) fn shift_view_mips(resource: This, info: &D3D12_RESOURCE_DESC, view: &mut D3D12_SHADER_RESOURCE_VIEW_DESC) {
    let Some((_, desc)) = super::resource::texture_memory(resource) else { return };
    let host_mips = u32::from(info.MipLevels);
    let drop = super::resource::mip_count(&desc).saturating_sub(host_mips);
    if drop == 0 {
        return;
    }
    let fix = |most: &mut u32, levels: &mut u32| {
        *most = most.saturating_sub(drop);
        if *levels != u32::MAX {
            *levels = (*levels).min(host_mips.saturating_sub(*most)).max(1);
        }
    };
    // SAFETY: each arm reads the union member its view dimension selects.
    unsafe {
        let a = &mut view.Anonymous;
        match view.ViewDimension {
            D3D12_SRV_DIMENSION_TEXTURE2D => fix(&mut a.Texture2D.MostDetailedMip, &mut a.Texture2D.MipLevels),
            D3D12_SRV_DIMENSION_TEXTURE2DARRAY => fix(&mut a.Texture2DArray.MostDetailedMip, &mut a.Texture2DArray.MipLevels),
            D3D12_SRV_DIMENSION_TEXTURECUBE => fix(&mut a.TextureCube.MostDetailedMip, &mut a.TextureCube.MipLevels),
            D3D12_SRV_DIMENSION_TEXTURECUBEARRAY => fix(&mut a.TextureCubeArray.MostDetailedMip, &mut a.TextureCubeArray.MipLevels),
            D3D12_SRV_DIMENSION_TEXTURE3D => fix(&mut a.Texture3D.MostDetailedMip, &mut a.Texture3D.MipLevels),
            _ => {}
        }
    }
}

// Ahead: when a page read completes, the origin registry calls
// [`arrival`] on the DirectStorage worker; a worker of ours takes it from a
// queue and, when the page completes a mip level of its texture from the
// coarsest one up, makes the host copy the title is expected to place over
// those pages (the levels present: width >> k, mips - k, the file's format
// and layout mode), detiles the decoded pages into a pooled upload buffer
// and keeps both under the origin key. The first placement that matches
// finds the copy (a hit) and its first refresh hands the prepared buffer to
// the submission's prelude, which records the copy as for any upload.
// Bounded by `XNW_TEX_AHEAD_MB` of prepared uploads (default 64, 0 off);
// copies nobody placed go at the sweep. Whether the title places those
// shapes is what the "ahead" counters tell.

/// A prepared upload of a host copy made ahead: the console layout it was
/// detiled from, the upload's bytes and mips dropped.
pub(crate) struct Prepared {
    layout: xnw_formats::texture::Desc,
    bytes: u64,
    mip_drop: u32,
    /// Bytes of the texture's pages it holds (from its first).
    len: u64,
    /// Read from the pack, console mips from this one on (a preload).
    pack_first: Option<u32>,
    /// Where in the footprints the upload starts.
    base: u64,
    upload: Option<host::Upload>,
}

// SAFETY: plain data and an upload buffer (see `host::Upload`).
unsafe impl Send for Prepared {}

impl Prepared {
    fn into_upload(mut self) -> host::Upload {
        self.upload.take().expect("a prepared upload")
    }
}

impl Drop for Prepared {
    fn drop(&mut self) {
        AHEAD_BYTES.fetch_sub(self.bytes, Relaxed);
    }
}

/// Bytes of prepared uploads held.
static AHEAD_BYTES: AtomicU64 = AtomicU64::new(0);
/// Bytes of ahead copies not placed yet, and how many.
static UNPLACED: AtomicU64 = AtomicU64::new(0);
static AHEAD_LIVE: AtomicU64 = AtomicU64::new(0);

/// Ahead copies not placed yet, at most (`XNW_TEX_AHEAD_COUNT`, 256).
fn ahead_count() -> u64 {
    static N: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| std::env::var("XNW_TEX_AHEAD_COUNT").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(256));
    *N
}

/// Ahead copies not placed yet, at most (`XNW_TEX_AHEAD_MB`, default 512;
/// 0 turns making ahead off).
fn ahead_budget() -> u64 {
    static MB: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| std::env::var("XNW_TEX_AHEAD_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(512));
    *MB << 20
}

/// Prepared uploads held at most (`XNW_TEX_STAGE_MB`, default 64): past
/// it copies are made ahead without one (the frame then detiles the
/// decoded pages, which is cheap next to making the texture).
fn stage_budget() -> u64 {
    static MB: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| std::env::var("XNW_TEX_STAGE_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(64));
    *MB << 20
}

static ARRIVALS: Mutex<std::collections::VecDeque<(u32, u32, u32)>> = Mutex::new(std::collections::VecDeque::new());
static ARRIVED: std::sync::Condvar = std::sync::Condvar::new();

/// The origin registry's hook: queues the page (at most 4,096 waiting).
fn arrival(file: u32, page: u32, texture: u32) {
    let mut q = ARRIVALS.lock().unwrap_or_else(|e| e.into_inner());
    if q.len() >= 4096 {
        ORIGIN_EVENTS[OR_ARRIVALS_LOST].fetch_add(1, Relaxed);
        return;
    }
    q.push_back((file, page, texture));
    ARRIVED.notify_one();
}

/// Registers the hook and starts the worker, once, unless the budget is 0.
fn start_ahead() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if ahead_budget() == 0 || !xnw_formats::origin::set_arrival_hook(arrival) {
            return;
        }
        let _ = std::thread::Builder::new().name("texture ahead".into()).spawn(|| {
            loop {
                let next = {
                    let mut q = ARRIVALS.lock().unwrap_or_else(|e| e.into_inner());
                    loop {
                        if let Some(n) = q.pop_front() {
                            break n;
                        }
                        q = ARRIVED.wait(q).unwrap_or_else(|e| e.into_inner());
                    }
                };
                prepare_ahead(next.0, next.1, next.2);
            }
        });
    });
}

/// Makes (on the first page that arrives) and prepares (again as pages
/// arrive, while the staging budget allows) the host copy of `texture` as
/// the title places it: its whole chain at its first page (V, a test run: the
/// placements found match the file's size, mips and format, at +0 into the
/// texture), which the low mips fill first. 2D arrays are made, not
/// prepared (their units are interleaved over the slices in the file).
fn prepare_ahead(file: u32, _page: u32, texture: u32) -> Option<()> {
    use xnw_formats::ttpf;
    let index = xnw_formats::origin::file(file)?;
    let tex = *index.texture(texture)?;
    let array = tex.kind == ttpf::KIND_ARRAY;
    if !(tex.kind == ttpf::KIND_2D && tex.layers == 1 || array) {
        return None;
    }
    let desc = super::resource::ResourceDesc {
        dimension: 3,
        width: u64::from(tex.width),
        height: tex.height,
        depth_or_array_size: tex.layers as u16,
        mip_levels: tex.mips as u16,
        format: tex.format,
        sample_count: 1,
        layout: 0x100 | ttpf::swizzle(tex.mode).0,
        ..Default::default()
    };
    let key = OriginKey { file, texture, at: tex.addr, shape: shape(&desc) };
    let existing = origins().as_ref().and_then(|o| o.map.get(&key).map(|e| (e.copy.clone(), e.ahead)));
    let again = existing.is_some();
    let copy = match existing {
        Some((_, false)) => return None, // placed: its refreshes bring it up to date
        Some((copy, true)) => copy,
        // Bounded by bytes, by count, and to half the texture budget.
        None if UNPLACED.load(Relaxed) >= ahead_budget()
            || AHEAD_LIVE.load(Relaxed) >= ahead_count()
            || super::resource::host_texture_bytes() >= super::resource::tex_budget() / 2 =>
        {
            return None;
        }
        None => super::resource::host_copy_made(&desc, 0, std::ptr::null(), Some(key), mips_to_drop(&desc, tex.mips).max(pack_drop(&key)))?,
    };
    if !again {
        ORIGIN_EVENTS[OR_AHEAD_MADE].fetch_add(1, Relaxed);
        if copy.dropped > 0 {
            ORIGIN_EVENTS[OR_MIPS_DROPPED].fetch_add(1, Relaxed);
        }
        keep(key, &copy, true);
    }
    // With the texture in the pack, the frame reads it in the host layout:
    // nothing to prepare.
    if array || AHEAD_BYTES.load(Relaxed) >= stage_budget() || pack_drop(&key) > 0 || index.pack_id.and_then(|id| pack()?.lookup(id, texture)).is_some() {
        if !again {
            ORIGIN_EVENTS[OR_AHEAD_BARE].fetch_add(1, Relaxed);
        }
        return None;
    }
    let layout = super::resource::texture_layout(&desc)?.planes[0].layout.clone();
    if layout.size > tex.span {
        return None;
    }
    let staging = Staging::new(&copy.texture, layout.clone(), std::ptr::null(), Some(&copy), None)?;
    let mut tiled = vec![0; layout.size as usize];
    let read = staging.read_size();
    let got = tiled_prefix(file, tex.addr, &mut tiled[..read]) as u64;
    let had = copy.prepared.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map_or(0, |p| p.len);
    if got == 0 || got <= had {
        return None;
    }
    let upload = host::upload_pool(staging.bytes)?;
    staging.detile_now(&tiled, &mut Vec::new(), upload.data);
    AHEAD_BYTES.fetch_add(staging.bytes, Relaxed);
    *copy.prepared.lock().unwrap_or_else(|e| e.into_inner()) = Some(Prepared { layout: layout.desc, bytes: staging.bytes, mip_drop: copy.dropped, len: got, pack_first: None, base: 0, upload: Some(upload) });
    if again {
        ORIGIN_EVENTS[OR_AHEAD_AGAIN].fetch_add(1, Relaxed);
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::super::resource::{self, BUFFER, HeapProperties, ResourceDesc};
    use super::*;

    const PROPERTIES: HeapProperties = HeapProperties { kind: 1, cpu_pages: 0, pool: 0, creation_nodes: 0, visible_nodes: 0 };

    /// Device tests share the frame cap, the origin registry and the
    /// exit counters: one at a time (`XNW_TEST_GPU=1` runs them in parallel).
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        static ONE: Mutex<()> = Mutex::new(());
        ONE.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn create(desc: &ResourceDesc) -> This {
        let mut out = std::ptr::null();
        let hr = resource::create_committed(std::ptr::null(), &PROPERTIES, 0, desc, 0, std::ptr::null(), &resource::RESOURCE_IIDS[0], &mut out);
        assert_eq!(hr, 0);
        out
    }

    /// Reads `bytes` of `from` (in `state`) back through a readback buffer,
    /// as `rows` rows of `pitch` from subresource 0, or the buffer whole.
    fn read_back(from: &ID3D12Resource, state: u32, texture: Option<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>, bytes: u64) -> Vec<u8> {
        let host = host::host().unwrap();
        let (readback, data) = {
            let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: bytes,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut r: Option<ID3D12Resource> = None;
            // SAFETY: plain creation of a readback buffer.
            unsafe { host.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut r).unwrap() };
            (r.unwrap(), bytes)
        };
        let list = host.new_list().unwrap();
        // SAFETY: an open list, live resources.
        unsafe {
            list.list.ResourceBarrier(&[host::transition(from, D3D12_RESOURCE_STATES(state as i32), D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            match texture {
                Some(footprint) => {
                    let to = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&readback), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint } };
                    let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(from), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
                    list.list.CopyTextureRegion(&to, 0, 0, 0, &src, None);
                }
                None => list.list.CopyBufferRegion(&readback, 0, from, 0, bytes),
            }
        }
        list.close();
        host.execute(&[&list]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy finished (execute waits).
        unsafe {
            readback.Map(0, None, Some(&mut mapped)).unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), data as usize).to_vec()
        }
    }

    /// Render target (1), depth stencil (2) and unordered access (4) mark the
    /// title's own GPU output; no flags, or only deny-shader-resource (8), do not.
    #[test]
    fn gpu_output_is_target_depth_or_unordered() {
        assert_eq!([0, 1, 2, 4, 3, 5, 6, 7, 8].map(gpu_output_flags), [false, true, true, true, true, true, true, true, false]);
    }

    /// A linear 64x4 RGBA texture the title writes in its memory reaches its
    /// host copy at the next submission that samples it, and a later write
    /// shows at the one after (memory not watched: by content hash).
    #[test]
    fn title_written_texture_reaches_its_host_copy() {
        if host::host().is_none() {
            return; // no host GPU here
        }
        let _one = serial();
        let desc = ResourceDesc { dimension: 3, width: 64, height: 4, depth_or_array_size: 1, mip_levels: 1, format: 28, sample_count: 1, layout: 1, ..Default::default() };
        let this = create(&desc);
        let (memory, _) = resource::texture_memory(this).unwrap();
        let texture = resource::host_resource(this).unwrap().clone();
        let layout = resource::texture_layout(&desc).unwrap().planes[0].layout.clone();
        let fill = |seed: u8| {
            // SAFETY: the texture's memory holds its layout's bytes.
            let bytes = unsafe { std::slice::from_raw_parts_mut(memory.cast_mut(), layout.size as usize) };
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = (i as u8).wrapping_mul(7).wrapping_add(seed);
            }
        };
        let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(28), Width: 64, Height: 4, Depth: 1, RowPitch: 256 },
        };
        for seed in [1u8, 90] {
            fill(seed);
            let host = host::host().unwrap();
            let list = host.new_list().unwrap();
            upload(&list, this, &texture);
            list.close();
            host.execute(&[&list]);
            let read = read_back(&texture, 0x400, Some(footprint), 1024);
            let mut linear = vec![0u8; 1024];
            layout.detile(
                // SAFETY: as above.
                unsafe { std::slice::from_raw_parts(memory, layout.size as usize) },
                0,
                0,
                &mut linear,
                256,
                1024,
            )
            .unwrap();
            assert_eq!(read, linear, "seed {seed}");
            // The copy left the texture in the copy destination state.
            resource::host_transition(this, 0x400);
        }
        type Release = extern "system" fn(This) -> u32;
        // SAFETY: slot 2 of our resource is Release.
        let release: Release = unsafe { super::super::object::slot(this, 0x10) };
        assert_eq!(release(this), 0);
    }

    /// MOV1: an unchanged texture is not uploaded again, but a movie plane
    /// (marked by the kept Bink draw) is, and the exit it passed is counted.
    #[test]
    fn a_movie_plane_is_uploaded_at_every_refresh() {
        if host::host().is_none() {
            return; // no host GPU here
        }
        let _one = serial();
        let desc = ResourceDesc { dimension: 3, width: 64, height: 4, depth_or_array_size: 1, mip_levels: 1, format: 28, sample_count: 1, layout: 1, ..Default::default() };
        let this = create(&desc);
        let (memory, _) = resource::texture_memory(this).unwrap();
        let texture = resource::host_resource(this).unwrap().clone();
        // SAFETY: the texture's memory holds at least its first 1 KiB.
        unsafe { std::slice::from_raw_parts_mut(memory.cast_mut(), 1024) }.fill(7);
        let s = staging(this, &texture).unwrap();
        assert!(s.refresh().is_some(), "the first refresh uploads");
        assert!(s.refresh().is_none(), "unchanged: no upload");
        let _ = movie_exits();
        note_movie_planes(&[this as usize]);
        assert!(s.refresh().is_some(), "a movie plane uploads again");
        let e = movie_exits();
        // Unwatched test memory: forced, the hash is not taken (nor counted).
        assert_eq!((e[MOVIE_UPLOADS], e[MOVIE_HASH]), (1, 0), "{e:?}");
        type Release = extern "system" fn(This) -> u32;
        // SAFETY: slot 2 of our resource is Release.
        let release: Release = unsafe { super::super::object::slot(this, 0x10) };
        drop(s);
        assert_eq!(release(this), 0);
    }

    /// Two textures over memory holding the same `.ttpf` page share one
    /// host copy, uploaded from the decoded page (title memory left zero),
    /// and the second needs no upload.
    #[test]
    fn origin_textures_share_a_copy_uploaded_from_pages() {
        use xnw_formats::origin;
        use xnw_formats::ttpf::{Page, Texture, Ttpf};
        let Some(host) = host::host() else {
            return; // no host GPU here
        };
        let _one = serial();
        let desc = ResourceDesc { dimension: 3, width: 64, height: 4, depth_or_array_size: 1, mip_levels: 1, format: 28, sample_count: 1, layout: 1, ..Default::default() };
        let layout = resource::texture_layout(&desc).unwrap().planes[0].layout.clone();
        assert!(layout.size <= 0x1000);
        // Two frames: the texture's memory need not start a frame.
        let tex = Texture { hash: 1, width: 64, height: 4, mips: 1, format: 28, layers: 1, kind: 2, mode: 0, addr: 0x1_0000, span: 0x2000, words: [0; 4] };
        let ttpf = Ttpf { pages: vec![Page { addr: 0x1_0000, size: 0x2000, offset: 0x1000, stored: 0x2000, storage: 0 }], textures: vec![tex], header_bytes: 0x100 };
        let r = origin::registry();
        let file = r.add(std::path::PathBuf::from("t2-test-origin.ttpf"), ttpf);
        let page: Vec<u8> = (0..0x2000u32).map(|i| (i as u8).wrapping_mul(13).wrapping_add(5)).collect();
        r.put_decoded(file, 0, page.clone());
        // A physical frame no xmem page uses, shown at both textures' memory.
        const FRAME: usize = 0x7f00_0000;
        let (a, b) = (create(&desc), create(&desc));
        let skews: Vec<usize> = [a, b].iter().map(|&this| resource::texture_memory(this).unwrap().0 as usize & 0xfff).collect();
        if skews[0] != skews[1] {
            return; // the two need the same place in their frame to share pages
        }
        for this in [a, b] {
            let (memory, _) = resource::texture_memory(this).unwrap();
            r.map_va(memory as usize, 0x2000, FRAME as u64);
        }
        r.record_pages(file, 0x1000, &[FRAME], 0, 0x2000).unwrap();
        let (ca, cb) = (resource::host_copy(a).unwrap(), resource::host_copy(b).unwrap());
        assert!(Arc::ptr_eq(&ca, &cb), "one host copy per origin");
        let list = host.new_list().unwrap();
        upload(&list, a, &ca.texture);
        upload(&list, b, &cb.texture);
        list.close();
        host.execute(&[&list]);
        assert_eq!(ca.uploaded.load(Relaxed), 0x1000);
        let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(28), Width: 64, Height: 4, Depth: 1, RowPitch: 256 },
        };
        let read = read_back(&ca.texture, ca.state.load(Relaxed), Some(footprint), 1024);
        let mut linear = vec![0u8; 1024];
        layout.detile(&page[skews[0]..][..layout.size as usize], 0, 0, &mut linear, 256, 1024).unwrap();
        assert!(read == linear, "{}", origin_text());
        for this in [a, b] {
            let (memory, _) = resource::texture_memory(this).unwrap();
            r.unmap_va(memory as usize, 0x2000);
            type Release = extern "system" fn(This) -> u32;
            // SAFETY: slot 2 of our resource is Release.
            let release: Release = unsafe { super::super::object::slot(this, 0x10) };
            release(this);
        }
    }

    /// Waits until the pass's reader has finished its read of `copy`.
    fn read_finished(copy: &Arc<resource::HostTexture>) {
        for _ in 0..5000 {
            if reader().flight.lock().unwrap().finished.iter().any(|d| Arc::ptr_eq(&d.copy, copy)) {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        panic!("the reader never finished");
    }

    /// PERF2, the hand-over of a read off the frame. The reader marks the
    /// copy complete when it has READ; the pass has the texels only once
    /// [`pass_finished`] gave it the upload. In between every ask says
    /// "flying, not ready" (before: "complete", and the upload was never
    /// fetched: a black texture). After: ready at once, and its contents
    /// stand while a later ask reads it again.
    #[test]
    fn an_async_read_is_not_ready_before_the_pass_took_its_upload() {
        if host::host().is_none() {
            return; // no host GPU here
        }
        let _one = serial();
        let desc = ResourceDesc { dimension: 3, width: 64, height: 4, depth_or_array_size: 1, mip_levels: 1, format: 28, sample_count: 1, layout: 1, ..Default::default() };
        let this = create(&desc);
        let (memory, _) = resource::texture_memory(this).unwrap();
        let layout = resource::texture_layout(&desc).unwrap().planes[0].layout.clone();
        // SAFETY: the texture's memory holds its layout's bytes.
        unsafe { std::slice::from_raw_parts_mut(memory.cast_mut(), layout.size as usize) }.fill(0x5a);
        super::super::queue::pass_frame();
        let _ = pass_finished();
        let ask = || for_pass_async(this, Box::new(()), &mut |_| true).unwrap();
        let first = ask();
        assert!(first.flying && first.waiting && !first.ready && first.upload.is_none(), "queued");
        let copy = first.copy.clone();
        read_finished(&copy);
        // Read, not handed over: the copy's own state says complete.
        assert!(!copy.pending.load(Relaxed));
        let early = ask();
        assert!(early.flying && early.waiting && !early.ready && !early.landed, "read but not taken: still flat");
        let got: Vec<_> = pass_finished().into_iter().filter(|(c, _, _)| Arc::ptr_eq(c, &copy)).collect();
        assert_eq!(got.len(), 1, "the upload goes to the pass at its frame's start, asked for or not");
        let landed = ask();
        assert!(landed.ready && landed.landed && !landed.flying && !landed.waiting && landed.upload.is_none(), "ready on the frame that records it");
        // Asked again (a new slot of the texture): read again, its contents stand meanwhile.
        let again = ask();
        assert!(again.ready && again.flying && again.waiting, "the old contents while it is read again");
        read_finished(&copy);
        assert!(ask().flying, "in flight until taken");
        assert!(!pass_finished().iter().any(|(c, _, _)| Arc::ptr_eq(c, &copy)), "unchanged memory: no second upload");
        let same = ask();
        assert!(same.ready && !same.landed && !same.flying && !same.waiting);
        assert!(reader().flight.lock().unwrap().copies.is_empty());
        drop((first, early, landed, again, same, got, copy));
        super::super::queue::pass_frame();
        type Release = extern "system" fn(This) -> u32;
        // SAFETY: slot 2 of our resource is Release.
        let release: Release = unsafe { super::super::object::slot(this, 0x10) };
        release(this);
    }

    /// PERF2: two resources over one origin copy. While the first one's read
    /// is with the reader, the second is answered as the copy was before it
    /// (flat), not from the state the read has already set ("uploaded");
    /// before, the second sampled the copy with nothing copied into it.
    #[test]
    fn a_shared_copy_in_flight_is_flat_for_every_resource_of_it() {
        use xnw_formats::origin;
        use xnw_formats::ttpf::{Page, Texture, Ttpf};
        if host::host().is_none() {
            return; // no host GPU here
        }
        let _one = serial();
        let desc = ResourceDesc { dimension: 3, width: 64, height: 4, depth_or_array_size: 1, mip_levels: 1, format: 28, sample_count: 1, layout: 1, ..Default::default() };
        let tex = Texture { hash: 1, width: 64, height: 4, mips: 1, format: 28, layers: 1, kind: 2, mode: 0, addr: 0x1_0000, span: 0x2000, words: [0; 4] };
        let ttpf = Ttpf { pages: vec![Page { addr: 0x1_0000, size: 0x2000, offset: 0x1000, stored: 0x2000, storage: 0 }], textures: vec![tex], header_bytes: 0x100 };
        let r = origin::registry();
        let file = r.add(std::path::PathBuf::from("perf2-test-origin.ttpf"), ttpf);
        r.put_decoded(file, 0, (0..0x2000u32).map(|i| (i as u8).wrapping_mul(11).wrapping_add(3)).collect());
        const FRAME: usize = 0x7e00_0000;
        let (a, b) = (create(&desc), create(&desc));
        let skews: Vec<usize> = [a, b].iter().map(|&this| resource::texture_memory(this).unwrap().0 as usize & 0xfff).collect();
        if skews[0] == skews[1] {
            for this in [a, b] {
                let (memory, _) = resource::texture_memory(this).unwrap();
                r.map_va(memory as usize, 0x2000, FRAME as u64);
            }
            r.record_pages(file, 0x1000, &[FRAME], 0, 0x2000).unwrap();
            let (ca, cb) = (resource::host_copy(a).unwrap(), resource::host_copy(b).unwrap());
            assert!(Arc::ptr_eq(&ca, &cb), "one host copy per origin");
            super::super::queue::pass_frame();
            let _ = pass_finished();
            let ask = |this| for_pass_async(this, Box::new(()), &mut |_| true).unwrap();
            let first = ask(a);
            assert!(first.flying && !first.ready);
            read_finished(&ca);
            assert_eq!(ca.uploaded.load(Relaxed), 0x1000, "the read has marked the copy uploaded");
            let other = ask(b);
            assert!(other.flying && !other.ready, "the other resource: flat until the pass has the upload");
            assert_eq!(pass_finished().iter().filter(|(c, _, _)| Arc::ptr_eq(c, &ca)).count(), 1);
            let other = ask(b);
            assert!(other.ready, "recorded: the shared copy is ready for the other resource at once");
            assert!(ask(a).ready);
            read_finished(&ca);
            let _ = pass_finished();
            super::super::queue::pass_frame();
        }
        for this in [a, b] {
            let (memory, _) = resource::texture_memory(this).unwrap();
            r.unmap_va(memory as usize, 0x2000);
            type Release = extern "system" fn(This) -> u32;
            // SAFETY: slot 2 of our resource is Release.
            let release: Release = unsafe { super::super::object::slot(this, 0x10) };
            release(this);
        }
    }

    /// A 2-mip texture whose pages have arrived only for
    /// the first frame (the mip tail) is uploaded once with them; bound again
    /// with no new page, it is ready from its first present mip (its texels
    /// are on the GPU) and the refresh that yields nothing is not charged to
    /// the frame's upload count.
    #[test]
    fn a_partial_copy_stays_ready_and_its_no_op_refresh_is_free() {
        use xnw_formats::origin;
        use xnw_formats::ttpf::{Page, Texture, Ttpf};
        if host::host().is_none() {
            return; // no host GPU here
        }
        let _one = serial();
        let desc = ResourceDesc { dimension: 3, width: 64, height: 32, depth_or_array_size: 1, mip_levels: 2, format: 28, sample_count: 1, layout: 1, ..Default::default() };
        let layout = resource::texture_layout(&desc).unwrap().planes[0].layout.clone();
        assert!(layout.size == 0x3000 && layout.mips[1].offset == 0 && layout.mips[0].offset == 0x1000, "mip 1 in the first 4 KiB, mip 0 after");
        type Release = extern "system" fn(This) -> u32;
        // SAFETY: slot 2 of our resource is Release.
        let release = |this: This| unsafe { super::super::object::slot::<Release>(this, 0x10) }(this);
        // A texture whose memory starts a frame (the file's texture starts its page).
        let mut tries = Vec::new();
        let this = loop {
            let this = create(&desc);
            if resource::texture_memory(this).unwrap().0 as usize & 0xfff == 0 {
                break Some(this);
            }
            tries.push(this);
            if tries.len() == 64 {
                break None;
            }
        };
        tries.into_iter().for_each(|t| _ = release(t));
        let this = this.expect("a texture at the start of a frame in 64 tries");
        let (memory, _) = resource::texture_memory(this).unwrap();
        let span = layout.size;
        let tex = Texture { hash: 3, width: 64, height: 32, mips: 2, format: 28, layers: 1, kind: 2, mode: 0, addr: 0x4_0000, span, words: [0; 4] };
        let ttpf = Ttpf { pages: vec![Page { addr: 0x4_0000, size: span, offset: 0x1000, stored: span, storage: 0 }], textures: vec![tex], header_bytes: 0x100 };
        let r = origin::registry();
        let file = r.add(std::path::PathBuf::from("tex3-test-partial.ttpf"), ttpf);
        let page: Vec<u8> = (0..span as u32).map(|i| (i as u8).wrapping_mul(29).wrapping_add(3)).collect();
        r.put_decoded(file, 0, page);
        const FRAME: usize = 0x7e00_0000;
        r.map_va(memory as usize, span as usize, FRAME as u64);
        // Only the first frame (the mip tail, mip 1) has arrived, not mip 0.
        r.record_pages(file, 0x1000, &[FRAME], 0, 0x1000).unwrap();
        let exits = || WAIT_EXITS.each_ref().map(|c| c.load(Relaxed));
        let delta = |was: [u64; 8]| {
            let now = exits();
            std::array::from_fn::<u64, 8, _>(|i| now[i] - was[i])
        };
        let mut charged = 0;
        super::super::queue::pass_frame();
        let first = for_pass_with(
            this,
            true,
            &mut |_| {
                charged += 1;
                true
            },
            true,
        )
        .unwrap();
        assert!(first.upload.is_some() && first.ready, "first bind uploads what arrived");
        assert_ne!(first.copy.uploaded.load(Relaxed), 0);
        assert!(first.copy.pending.load(Relaxed), "partial: pending");
        assert_eq!(first.first_mip, 1, "viewed from its first present mip");
        drop(first);
        assert_eq!(charged, 1);
        // Switch off (the earlier behaviour): flat `waiting`, the no-op refresh charged.
        super::super::queue::pass_frame();
        let was = exits();
        let old = for_pass_with(
            this,
            true,
            &mut |_| {
                charged += 1;
                true
            },
            false,
        )
        .unwrap();
        assert!(old.upload.is_none() && !old.ready && old.waiting && charged == 2, "the old upload behaviour without the switch");
        drop(old);
        assert_eq!(delta(was), [0, 0, 0, 1, 0, 0, 0, 0], "counted as 'partial no new pages'");
        // With the switch: ready from mip 1, nothing refreshed, nothing charged.
        super::super::queue::pass_frame();
        let was = exits();
        let second = for_pass_with(
            this,
            true,
            &mut |_| {
                charged += 1;
                true
            },
            true,
        )
        .unwrap();
        assert!(second.upload.is_none(), "no new page: nothing to upload");
        let (ready, waiting) = (second.ready, second.waiting);
        assert!(ready && !waiting && charged == 2, "texels on the GPU must be ready (ready {ready}, waiting {waiting}); a no-op refresh is not charged (charged {charged})");
        assert_eq!(second.first_mip, 1, "ready from the first present mip");
        drop(second);
        assert_eq!(delta(was), [0, 0, 0, 1, 0, 0, 0, 1], "partial no new pages, holding texels");
        // The budget spent (switch off: with it on, the no-page check comes first).
        let was = exits();
        let refused = for_pass_with(this, true, &mut |_| false, false).unwrap();
        assert!(refused.waiting && !refused.ready);
        drop(refused);
        assert_eq!(delta(was), [1, 1, 0, 0, 0, 0, 0, 0], "budget, pending, uploaded before");
        // The rest of the chain arrives: an upgrade with both mips, complete.
        r.record_pages(file, 0x1000, &[FRAME], 0, span as usize).unwrap();
        super::super::queue::pass_frame();
        let full = for_pass_with(this, true, &mut |_| true, true).unwrap();
        assert!(full.upload.is_some() && full.ready && full.first_mip == 0 && !full.copy.pending.load(Relaxed), "the upgrade replaces the view: from mip 0");
        drop(full);
        super::super::queue::pass_frame();
        r.unmap_va(memory as usize, span as usize);
        release(this);
    }

    /// An origin copy falls back to title memory
    /// (here: its memory now starts with another texture's pages, of which
    /// only the first frame, the mip tail, has arrived). The upload is the
    /// title's memory as it is: mip 0 is not all there, so the view our pass
    /// gets must not expose it, and the copy stays pending until it is.
    #[test]
    fn a_title_memory_upload_exposes_only_whole_mips() {
        use xnw_formats::origin;
        use xnw_formats::ttpf::{Page, Texture, Ttpf};
        if host::host().is_none() {
            return; // no host GPU here
        }
        let _one = serial();
        let desc = ResourceDesc { dimension: 3, width: 64, height: 32, depth_or_array_size: 1, mip_levels: 2, format: 28, sample_count: 1, layout: 1, ..Default::default() };
        let layout = resource::texture_layout(&desc).unwrap().planes[0].layout.clone();
        assert!(layout.size == 0x3000 && layout.mips[1].offset == 0 && layout.mips[0].offset == 0x1000, "mip 1 in the first 4 KiB, mip 0 after");
        type Release = extern "system" fn(This) -> u32;
        // SAFETY: slot 2 of our resource is Release.
        let release = |this: This| unsafe { super::super::object::slot::<Release>(this, 0x10) }(this);
        let mut tries = Vec::new();
        let this = loop {
            let this = create(&desc);
            if resource::texture_memory(this).unwrap().0 as usize & 0xfff == 0 {
                break Some(this);
            }
            tries.push(this);
            if tries.len() == 64 {
                break None;
            }
        };
        tries.into_iter().for_each(|t| _ = release(t));
        let this = this.expect("a texture at the start of a frame in 64 tries");
        let (memory, _) = resource::texture_memory(this).unwrap();
        let span = layout.size;
        let r = origin::registry();
        let add = |hash: u64, name: &str| {
            let tex = Texture { hash, width: 64, height: 32, mips: 2, format: 28, layers: 1, kind: 2, mode: 0, addr: 0x4_0000, span, words: [0; 4] };
            let ttpf = Ttpf { pages: vec![Page { addr: 0x4_0000, size: span, offset: 0x1000, stored: span, storage: 0 }], textures: vec![tex], header_bytes: 0x100 };
            r.add(std::path::PathBuf::from(name), ttpf)
        };
        let (file_a, file_b) = (add(41, "tex4-test-a.ttpf"), add(42, "tex4-test-b.ttpf"));
        // SAFETY: the texture's memory holds its layout's bytes.
        let fill = |seed: u8| unsafe { std::slice::from_raw_parts_mut(memory.cast_mut(), span as usize) }.iter_mut().enumerate().for_each(|(i, b)| *b = (i as u8).wrapping_mul(31).wrapping_add(seed));
        fill(1);
        const FRAME: usize = 0x7d00_0000;
        r.map_va(memory as usize, span as usize, FRAME as u64);
        // Texture A whole: an origin copy, complete, viewed from mip 0.
        r.record_pages(file_a, 0x1000, &[FRAME], 0, span as usize).unwrap();
        super::super::queue::pass_frame();
        let a = for_pass_with(this, true, &mut |_| true, true).unwrap();
        assert!(a.upload.is_some() && a.ready && a.first_mip == 0 && !a.copy.pending.load(Relaxed) && a.copy.origin.get().is_some(), "an origin copy, complete");
        drop(a);
        // A request rewrites the first page: forgotten when its write starts,
        // recorded when it has completed. Between the two the memory may be
        // half written: nothing of it is sampled, and nothing asked twice.
        r.forget_pages(&[FRAME], 0, 0x1000);
        fill(9);
        super::super::queue::pass_frame();
        let mut asked = 0;
        for n in 1..=2 {
            let flight = for_pass_with(
                this,
                true,
                &mut |_| {
                    asked += 1;
                    true
                },
                true,
            )
            .unwrap();
            let (first, pending) = (flight.first_mip, flight.copy.pending.load(Relaxed));
            assert!(!flight.ready && pending && flight.upload.is_some() == (n == 1) && asked == 1, "a write in flight (ask {n}): not sampled, pending, read once (ready {}, first mip {first}, pending {pending}, asked {asked})", flight.ready);
            drop(flight);
            super::super::queue::pass_frame();
        }
        // Completed: the origin's pages again, whole.
        fill(1);
        r.record_pages(file_a, 0x1000, &[FRAME], 0, span as usize).unwrap();
        let done = for_pass_with(this, true, &mut |_| true, true).unwrap();
        assert!(done.upload.is_some() && done.ready && done.first_mip == 0 && !done.copy.pending.load(Relaxed), "the request completed: uploaded again, complete");
        drop(done);
        // The memory is taken for texture B: only its first frame (mip 1) has
        // arrived; the frames of mip 0 hold whatever was there.
        r.record_pages(file_b, 0x1000, &[FRAME], 0, 0x1000).unwrap();
        fill(2);
        super::super::queue::pass_frame();
        let was = WITHHELD.load(Relaxed);
        let mut charged = 0;
        let b = for_pass_with(
            this,
            true,
            &mut |_| {
                charged += 1;
                true
            },
            true,
        )
        .unwrap();
        assert!(b.upload.is_some(), "the fallback uploads title memory: {}", origin_text());
        let (first, pending) = (b.first_mip, b.copy.pending.load(Relaxed));
        assert!(first == 1 && pending && b.ready, "mip 0 has not arrived: the view starts at mip 1 and the copy stays pending (first mip {first}, pending {pending}, ready {})", b.ready);
        drop(b);
        assert_eq!((charged, WITHHELD.load(Relaxed) - was), (1, 1), "one upload charged, one view narrowed");
        // Asked again with no new page: the same view, no refresh, no charge.
        super::super::queue::pass_frame();
        let again = for_pass_with(
            this,
            true,
            &mut |_| {
                charged += 1;
                true
            },
            true,
        )
        .unwrap();
        assert!(again.upload.is_none() && again.ready && again.first_mip == 1 && charged == 1, "no new page: narrowed still, nothing refreshed");
        drop(again);
        assert_eq!(WITHHELD.load(Relaxed) - was, 2);
        // B's mip 0 arrives: uploaded again, whole, from mip 0.
        r.record_pages(file_b, 0x1000, &[FRAME], 0, span as usize).unwrap();
        fill(3);
        super::super::queue::pass_frame();
        let whole = for_pass_with(this, true, &mut |_| true, true).unwrap();
        assert!(whole.upload.is_some() && whole.ready && whole.first_mip == 0 && !whole.copy.pending.load(Relaxed), "every page there: from mip 0, complete");
        drop(whole);
        assert_eq!(WITHHELD.load(Relaxed) - was, 2, "a whole chain is not narrowed");
        super::super::queue::pass_frame();
        r.unmap_va(memory as usize, span as usize);
        release(this);
    }

    /// A plain texture whose upload the frame cap refused counts
    /// "frame cap" and "never uploaded" when the budget refuses it; one the
    /// reader is busy with counts "async busy".
    #[test]
    fn waiting_exits_count_frame_cap_budget_and_busy() {
        if host::host().is_none() {
            return; // no host GPU here
        }
        let _one = serial();
        let desc = ResourceDesc { dimension: 3, width: 64, height: 4, depth_or_array_size: 1, mip_levels: 1, format: 28, sample_count: 1, layout: 1, ..Default::default() };
        let this = create(&desc);
        let exits = || WAIT_EXITS.each_ref().map(|c| c.load(Relaxed));
        let delta = |was: [u64; 8]| {
            let now = exits();
            std::array::from_fn::<u64, 8, _>(|i| now[i] - was[i])
        };
        super::super::queue::pass_frame();
        assert!(frame_allow(FRAME_UPLOAD, u64::MAX >> 2));
        let was = exits();
        let t = for_pass_with(this, true, &mut |_| true, true).unwrap();
        assert!(t.upload.is_none() && t.waiting && !t.ready);
        drop(t);
        assert_eq!(delta(was), [0, 0, 0, 0, 1, 0, 0, 0], "frame cap");
        let was = exits();
        drop(for_pass_with(this, true, &mut |_| false, true).unwrap());
        assert_eq!(delta(was), [1, 1, 1, 0, 0, 0, 0, 0], "budget, pending, never uploaded");
        let texture = resource::host_resource(this).unwrap().clone();
        let _ = staging(this, &texture).unwrap();
        let key = Arc::as_ptr(&resource::host_copy(this).unwrap()) as usize;
        reader().flight.lock().unwrap().copies.insert(key, (false, 0));
        let was = exits();
        let t = for_pass_async(this, Box::new(()), &mut |_| true).unwrap();
        assert!(t.flying && t.waiting && !t.ready);
        drop(t);
        reader().flight.lock().unwrap().copies.remove(&key);
        assert_eq!(delta(was), [0, 0, 0, 0, 0, 0, 1, 0], "async busy");
        super::super::queue::pass_frame(); // a fresh frame cap for the next test
        type Release = extern "system" fn(This) -> u32;
        // SAFETY: slot 2 of our resource is Release.
        let release: Release = unsafe { super::super::object::slot(this, 0x10) };
        release(this);
    }

    /// A refresh's exit: an upload none; the frame cap before a
    /// missing buffer before a partial copy with nothing new.
    #[test]
    fn exit_of_names_the_first_refusal() {
        assert_eq!(exit_of(true, true, true, true, true), None);
        assert_eq!(exit_of(false, true, true, true, true), Some(EX_FRAME_CAP));
        assert_eq!(exit_of(false, true, true, false, true), Some(EX_NO_BUFFER));
        assert_eq!(exit_of(false, true, true, false, false), Some(EX_PARTIAL));
        assert_eq!(exit_of(false, true, false, false, false), None);
        assert_eq!(exit_of(false, false, true, false, false), None);
        refused_here(1);
        assert_eq!(REFUSED_HERE.get()[1], 1, "this thread's no-buffer refusals");
    }

    /// The budget part of the frame statistics reads its counters
    /// The origin part zeroed them first), and the exits line its own.
    #[test]
    fn origin_events_text_keeps_the_budget_counters() {
        let _one = serial(); // the exits line zeroes counters other tests compare
        ORIGIN_EVENTS[OR_TRIMMED].fetch_add(1_000_007, Relaxed);
        ORIGIN_EVENTS[OR_BYTES_UPLOADED].fetch_add(5_000 << 20, Relaxed);
        let t = origin_events_text(1);
        assert!(t.contains("1000007 kept copies trimmed") || t.contains("100000"), "{t}");
        let mib: u64 = t.split(" MiB uploaded").next().and_then(|p| p.rsplit(' ').next()).and_then(|n| n.parse().ok()).unwrap();
        assert!(mib >= 5_000, "{t}");
        WAIT_EXITS[EX_HOLDING].fetch_add(1_000_003, Relaxed);
        WITHHELD.fetch_add(2_000_003, Relaxed);
        let t = exits_text();
        assert!(t.contains("partial copies holding texels 100000") && t.contains("; views narrowed to whole mips 200000"), "{t}");
    }

    /// Releases take copies nobody else holds first, least recently
    /// used first; copies a resource holds only when asked.
    #[test]
    fn budget_releases_unheld_copies_least_recent_first() {
        let order = vec![(true, 1, 'a', 100), (false, 5, 'b', 100), (false, 2, 'c', 100), (true, 0, 'd', 100)];
        assert_eq!(lru_release(order.clone(), 150, false), (vec!['c', 'b'], 200));
        assert_eq!(lru_release(order.clone(), 500, false), (vec!['c', 'b'], 200), "held copies stay");
        assert_eq!(lru_release(order, 300, true), (vec!['c', 'b', 'd'], 200));
    }

    /// A frame's upload cap passes its first upload whatever its size,
    /// then refuses past the cap until the next frame.
    #[test]
    fn frame_caps_defer_the_rest_to_the_next_frame() {
        super::super::queue::pass_frame();
        assert!(frame_allow(FRAME_UPLOAD, 200 << 20));
        assert!(!frame_room(FRAME_UPLOAD));
        assert!(!frame_allow(FRAME_UPLOAD, 1));
        super::super::queue::pass_frame();
        assert!(frame_room(FRAME_UPLOAD) && frame_allow(FRAME_UPLOAD, 1));
    }

    /// Mips dropped for the budget keep the coarsest kept mip whole
    /// 4x4 blocks.
    #[test]
    fn budget_mip_drop_keeps_whole_blocks() {
        let d = ResourceDesc { dimension: 3, width: 1024, height: 256, depth_or_array_size: 1, mip_levels: 11, format: 71, sample_count: 1, ..Default::default() };
        assert_eq!(max_drop(&d, 11), 6); // 16x4
        assert_eq!(max_drop(&ResourceDesc { dimension: 4, ..d }, 11), 0);
    }

    /// Pack rows land at the footprints' offsets and pitches, from the
    /// first mip asked for on (coarsest first in the pack).
    #[test]
    fn pack_rows_follow_the_footprints() {
        use xnw_formats::texpack::{Entry, MipLoc};
        // Two mips of a 64x8 RGBA8 texture: mip 0 8 rows of 256 bytes at
        // pack offset 0x1200, mip 1 4 rows of 256 (128 used) at 0x1000.
        let mut mip = [MipLoc::default(); 16];
        mip[0] = MipLoc { offset: 0x1200, bytes: 2048, row_pitch: 256, slice_pitch: 2048, rows: 8 };
        mip[1] = MipLoc { offset: 0x1000, bytes: 1024, row_pitch: 256, slice_pitch: 1024, rows: 4 };
        let entry = Entry { pack: 0, texture: 0, format: 28, width: 64, height: 8, layers: 1, kind: 2, mips: 2, first: 0, stored: 2, block: 0xa00, mip };
        let bytes: Vec<u8> = (0..0xa00u32).map(|i| (i / 256) as u8 + 1).collect();
        let fp = |offset, width, height| D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: offset,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(28), Width: width, Height: height, Depth: 1, RowPitch: 256 },
        };
        let footprints = [fp(0, 64, 8), fp(2048, 32, 4)];
        let mut data = vec![0u8; 3072];
        assert_eq!(pack_rows(&entry, 0x1000, &bytes, &footprints, &[8, 4], 2, 0, 0, 0, data.as_mut_ptr()), Some(3072));
        // Mip 0 rows come from pack rows 2-9 (after mip 1's four), mip 1 from 0-3.
        assert_eq!((data[0], data[7 * 256], data[2048], data[2048 + 3 * 256]), (3, 10, 1, 4));
        // From mip 1 on only: mip 0's part is left alone.
        // From mip 1 on only, into an upload that starts at mip 1.
        let mut data = vec![0u8; 1024];
        assert_eq!(pack_rows(&entry, 0x1000, &bytes[..0x400], &footprints, &[8, 4], 2, 0, 1, 2048, data.as_mut_ptr()), Some(1024));
        assert_eq!((data[0], data[3 * 256]), (1, 4));
    }

    /// A texture bound every frame whose upload the frame cap put off
    /// is pending, asked again, and complete the next frame.
    #[test]
    fn a_deferred_upload_completes_on_a_later_frame() {
        let Some(host) = host::host() else {
            return; // no host GPU here
        };
        let _one = serial();
        let desc = ResourceDesc { dimension: 3, width: 64, height: 4, depth_or_array_size: 1, mip_levels: 1, format: 28, sample_count: 1, layout: 1, ..Default::default() };
        let this = create(&desc);
        let (memory, _) = resource::texture_memory(this).unwrap();
        let layout = resource::texture_layout(&desc).unwrap().planes[0].layout.clone();
        // SAFETY: the texture's memory holds its layout's bytes.
        let bytes = unsafe { std::slice::from_raw_parts_mut(memory.cast_mut(), layout.size as usize) };
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(11).wrapping_add(7);
        }
        let texture = resource::host_resource(this).unwrap().clone();
        let bind = || {
            let list = host.new_list().unwrap();
            upload(&list, this, &texture);
            list.close();
            host.execute(&[&list]);
        };
        // This frame's uploads are spent: the upload waits.
        super::super::queue::pass_frame();
        assert!(frame_allow(FRAME_UPLOAD, u64::MAX >> 2));
        bind();
        assert!(pending(this), "put off, so pending");
        // Next frame, bound again: done.
        super::super::queue::pass_frame();
        bind();
        assert!(!pending(this));
        let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(28), Width: 64, Height: 4, Depth: 1, RowPitch: 256 },
        };
        let copy = resource::host_copy(this).unwrap();
        let read = read_back(&texture, copy.state.load(Relaxed), Some(footprint), 1024);
        let mut linear = vec![0u8; 1024];
        layout.detile(bytes, 0, 0, &mut linear, 256, 1024).unwrap();
        assert_eq!(read, linear);
        type Release = extern "system" fn(This) -> u32;
        // SAFETY: slot 2 of our resource is Release.
        let release: Release = unsafe { super::super::object::slot(this, 0x10) };
        release(this);
    }

    /// A title buffer's host copy holds the title's bytes after a refresh,
    /// and a GPU address inside the buffer is the same offset in its copy.
    #[test]
    fn title_buffer_reaches_its_host_copy() {
        let Some(host) = host::host() else {
            return; // no host GPU here
        };
        let _one = serial();
        let desc = ResourceDesc { dimension: BUFFER, width: 3 << 16, height: 1, depth_or_array_size: 1, mip_levels: 1, sample_count: 1, layout: 1, ..Default::default() };
        let this = create(&desc);
        let (data, len) = resource::buffer_bytes(this).unwrap();
        // SAFETY: the buffer holds `len` bytes.
        let bytes = unsafe { std::slice::from_raw_parts_mut(data, len as usize) };
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i % 253) as u8;
        }
        let shadow = super::super::shadow::find(data as u64 + 0x100, 0x20).unwrap();
        assert_eq!(shadow.offset(data as u64 + 0x100), 0x100);
        let list = host.new_list().unwrap();
        shadow.refresh(&list, data as u64, len);
        list.close();
        host.execute(&[&list]);
        assert_eq!(read_back(&shadow.buffer, 0, None, len), bytes);
        type Release = extern "system" fn(This) -> u32;
        // SAFETY: slot 2 of our resource is Release.
        let release: Release = unsafe { super::super::object::slot(this, 0x10) };
        assert_eq!(release(this), 0);
    }
}
