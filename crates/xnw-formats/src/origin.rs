//! Texture origin registry: which `.ttpf` page lies in which title memory.
//!
//! Every streamed texture page comes from a `.ttpf` file by one DirectStorage
//! request per page, read whole. So a completed
//! request (file, offset) names the texture and the mips its destination now
//! holds. The HLE (DirectStorage, xmem, the kernel read paths) feeds this
//! registry; the GPU layer asks it, per placement, whether a range of title
//! memory is one texture's pages and how new they are, and uses
//! {file, texture, range} as its cache key and the stamp as the content
//! version instead of hashing the bytes. It lives here, not in `xnw-hle`,
//! because `xnw-gpu` cannot see `xnw-hle` (hle depends on gpu).
//!
//! Two interval maps in 4 KiB frames, under one `RwLock` (readers: the GPU's
//! per-frame lookups; writers: DirectStorage workers and xmem, never across
//! I/O):
//! - `phys`: physical frame (xmem's page numbers count 4 KiB frames) ->
//!   which file page bytes it holds, with a stamp from one counter, raised on
//!   every write; a frame written by anything else is dropped.
//! - `va`: title address -> physical frame, from xmem's views (maps and
//!   unmaps); a remap changes only this map, the frames keep their content.
//!
//! What it cannot see: the title's own CPU or GPU writes into a texture's
//! pages. The stamp versions DirectStorage content only; the GPU layer's
//! write watch stays the guard for the rest.
//!
//! Decoded pages (the bytes a request wrote, console layout) are kept in a
//! bounded first-in first-out cache (`XNW_ORIGIN_CACHE_MB`, default 256) so
//! an upload can skip the copy out of title memory.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use crate::ttpf::{self, KIND_2D, KIND_ARRAY, Texture, Ttpf};

const FRAME: u64 = 0x1000;
const BIG_PAGE: u64 = 0x1_0000;
/// No texture holds the page (it lies before the first texture).
pub const NO_TEXTURE: u32 = u32::MAX;

/// What `[addr, addr + len)` of title memory holds: pages of one texture
/// of one file, laid out as in the file's virtual space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Origin {
    /// Registry file id ([`file`]), index of the texture in its
    /// `Ttpf::textures`.
    pub file: u32,
    pub texture: u32,
    /// The file's virtual address of the range's first byte (rounded down to
    /// 4 KiB) and the bytes covered (whole 4 KiB frames).
    pub at: u64,
    pub len: u64,
    /// Mips the pages hold: finest (lowest number) and coarsest, inclusive.
    /// Per whole page, so a range that cuts a page may claim one more.
    pub mips: (u8, u8),
    /// File pages (index in `Ttpf::pages`), first and last.
    pub pages: (u32, u32),
    /// The newest write among them: changes whenever any of them is
    /// rewritten. With `file`, `texture` and `at`, a content version.
    pub stamp: u64,
    /// Some of the pages were bypassed (`XNW_TEXBYPASS=on`): recorded
    /// but never written to title memory. Their bytes come from the pack,
    /// or [`Registry::ensure_present`] writes them first.
    pub absent: bool,
}

/// One `.ttpf` index, parsed once per host file.
pub struct FileIndex {
    pub id: u32,
    pub path: PathBuf,
    pub ttpf: Ttpf,
    /// The texture pack's id of the file (`texpack::file_id`: stable across
    /// runs and paths, unlike `id`).
    pub pack_id: Option<u64>,
    /// Per page, the texture it belongs to (or [`NO_TEXTURE`]).
    page_texture: Vec<u32>,
    /// Per texture, its units: (span offset, end, finest mip, coarsest mip);
    /// computed on first use.
    units: Mutex<HashMap<u32, Arc<[Unit]>>>,
}

#[derive(Debug, Clone, Copy)]
struct Unit {
    from: u64,
    to: u64,
    mips: (u8, u8),
}

/// A request resolved to a file page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolved {
    pub page: u32,
    /// The file's virtual address the request's first byte lands at.
    pub at: u64,
    pub texture: u32,
}

impl FileIndex {
    pub fn new(id: u32, path: PathBuf, ttpf: Ttpf) -> FileIndex {
        let mut order: Vec<(u64, u32)> = ttpf.textures.iter().enumerate().map(|(i, t)| (t.addr, i as u32)).collect();
        order.sort_unstable();
        let page_texture = ttpf
            .pages
            .iter()
            .map(|p| order.partition_point(|&(a, _)| a <= p.addr).checked_sub(1).map_or(NO_TEXTURE, |k| order[k].1))
            .collect();
        let pack_id = crate::texpack::file_id(&path);
        FileIndex { id, path, ttpf, pack_id, page_texture, units: Mutex::new(HashMap::new()) }
    }

    /// The page a request at file `offset` reads: a compressed page is read
    /// whole from its start, a stored one from anywhere in it.
    pub fn resolve(&self, offset: u64) -> Option<Resolved> {
        let pages = &self.ttpf.pages;
        let i = pages.partition_point(|p| p.offset <= offset).checked_sub(1)?;
        let p = &pages[i];
        let at = if offset == p.offset {
            p.addr
        } else if p.storage == 0 && offset < p.offset + p.stored {
            p.addr + (offset - p.offset)
        } else {
            return None;
        };
        Some(Resolved { page: i as u32, at, texture: self.page_texture[i] })
    }

    pub fn texture(&self, i: u32) -> Option<&Texture> {
        self.ttpf.textures.get(i as usize)
    }

    /// Mips held by the file's virtual range `[from, to)` of texture `t`.
    pub fn mips_in(&self, t: u32, from: u64, to: u64) -> (u8, u8) {
        let Some(tex) = self.texture(t) else { return (0, 0) };
        let units = {
            let cached = self.units.lock().unwrap_or_else(|e| e.into_inner()).get(&t).cloned();
            match cached {
                Some(u) => u,
                None => {
                    let u: Arc<[Unit]> = units_of(tex).into();
                    self.units.lock().unwrap_or_else(|e| e.into_inner()).insert(t, u.clone());
                    u
                }
            }
        };
        let (from, to) = (from.saturating_sub(tex.addr), to.saturating_sub(tex.addr));
        let mut out: Option<(u8, u8)> = None;
        for u in units.iter().filter(|u| u.from < to && from < u.to) {
            out = Some(out.map_or(u.mips, |(a, b)| (a.min(u.mips.0), b.max(u.mips.1))));
        }
        out.unwrap_or((0, coarsest(tex)))
    }
}

fn coarsest(t: &Texture) -> u8 {
    t.mips.saturating_sub(1).min(255) as u8
}

/// A texture's units in its span: 2D and 2D arrays per mip (the tail block
/// as one unit; array units interleaved over the slices); cube maps,
/// volumes and layouts we cannot place as one unit of every mip.
fn units_of(t: &Texture) -> Vec<Unit> {
    let whole = vec![Unit { from: 0, to: t.span, mips: (0, coarsest(t)) }];
    if !matches!(t.kind, KIND_2D | KIND_ARRAY) {
        return whole;
    }
    let Ok(p) = ttpf::place(t) else { return whole };
    let l = &p.chain;
    let s = u64::from(p.slices);
    let n = l.mips.len() as u32;
    let tail = l.first_tail_mip;
    let mut v = Vec::new();
    let mut next = tail; // mip of the next non-tail unit, counting down
    for (i, (off, size)) in ttpf::units(l).into_iter().enumerate() {
        let mips = if i == 0 && tail < n {
            (tail as u8, (n - 1) as u8)
        } else {
            next -= 1;
            (next as u8, next as u8)
        };
        v.push(Unit { from: off * s, to: (off + size) * s, mips });
    }
    if v.is_empty() { whole } else { v }
}

/// An interval map over u64 keys: disjoint runs `[start, end)` with a value
/// that shifts with its start (`shift(v, d)` is the value `d` keys in).
pub struct IntervalMap<V: Copy> {
    runs: BTreeMap<u64, (u64, V)>,
    shift: fn(V, u64) -> V,
}

impl<V: Copy> IntervalMap<V> {
    pub fn new(shift: fn(V, u64) -> V) -> Self {
        IntervalMap { runs: BTreeMap::new(), shift }
    }

    pub fn len(&self) -> usize {
        self.runs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// The run holding `key`: its start, end and value at its start.
    pub fn at(&self, key: u64) -> Option<(u64, u64, V)> {
        self.runs.range(..=key).next_back().filter(|(_, (end, _))| *end > key).map(|(&s, &(e, v))| (s, e, v))
    }

    /// Removes `[start, end)`, cutting runs that cross its edges.
    pub fn remove(&mut self, start: u64, end: u64) {
        if start >= end {
            return;
        }
        if let Some((s, e, v)) = self.at(start)
            && s < start
        {
            self.runs.insert(s, (start, v));
            if e > end {
                self.runs.insert(end, (e, (self.shift)(v, end - s)));
                return;
            }
        }
        while let Some((&s, &(e, v))) = self.runs.range(start..end).next() {
            self.runs.remove(&s);
            if e > end {
                self.runs.insert(end, (e, (self.shift)(v, end - s)));
            }
        }
    }

    /// Sets `[start, end)` to `v`, over whatever was there.
    pub fn insert(&mut self, start: u64, end: u64, v: V) {
        if start >= end {
            return;
        }
        self.remove(start, end);
        self.runs.insert(start, (end, v));
    }
}

/// Where a file page's bytes lie in physical frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Held {
    file: u32,
    page: u32,
    texture: u32,
    /// File virtual address at the run's start.
    at: u64,
    mips: (u8, u8),
    stamp: u64,
    /// Recorded without the bytes being written (bypass).
    absent: bool,
}

fn shift_held(h: Held, frames: u64) -> Held {
    Held { at: h.at + frames * FRAME, ..h }
}

fn shift_frame(f: u64, frames: u64) -> u64 {
    f + frames
}

struct Maps {
    phys: IntervalMap<Held>,
    va: IntervalMap<u64>,
}

/// Counters (`XNW_TRACE=perf` logs them): lookups, hits, decoded-cache
/// hits and misses.
#[derive(Debug, Default, Clone, Copy)]
pub struct Stats {
    pub lookups: u64,
    pub hits: u64,
    pub recorded: u64,
    pub phys_runs: u64,
    pub va_runs: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_bytes: u64,
    pub reasons: [u64; 10],
}

/// The decoded-page cache: first in, first out, under a byte budget.
struct Cache {
    pages: HashMap<(u32, u32), Arc<Vec<u8>>>,
    order: VecDeque<(u32, u32)>,
    bytes: u64,
    budget: u64,
}

pub struct Registry {
    maps: RwLock<Maps>,
    stamp: AtomicU64,
    files: RwLock<Vec<Arc<FileIndex>>>,
    cache: Mutex<Cache>,
    counts: [AtomicU64; 5], // lookups, hits, recorded, cache hits, cache misses
    /// [`Registry::prefix_of`] outcomes, by [`REASONS`].
    reasons: [AtomicU64; 10],
    /// Runs ever recorded absent, and the bypass counters.
    absent_runs: AtomicU64,
    counts_x1: [AtomicU64; 6],
}

/// Names of [`Registry::bypass_counts`].
pub const BYPASS_COUNTS: [&str; 6] = ["bypassed", "filled on demand", "page requests", "bytes not read", "bytes not decoded", "decode us avoided (est.)"];

/// Bypassing allowed (cleared when a read trap shows the title's CPU
/// reading a bypassed page).
static BYPASS_ALLOWED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn bypass_allowed() -> bool {
    BYPASS_ALLOWED.load(Ordering::Relaxed)
}

pub fn stop_bypass() {
    BYPASS_ALLOWED.store(false, Ordering::Relaxed);
}

/// Where a walk over title memory stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cut {
    Whole = 0,
    /// No physical frames at an address.
    Unmapped = 1,
    /// A frame no texture page read was recorded into (not arrived, or
    /// written by something else since).
    NotArrived = 2,
    /// The next frame holds another texture's page.
    OtherTexture = 3,
    /// The next frame holds this texture's page, but not the next bytes.
    OutOfOrder = 4,
}

const MISS_BASE: usize = 4; // + Cut::Unmapped / NotArrived: 5, 6
const MISS_NO_TEXTURE: usize = 7;
const SUB_PAGE: usize = 8;
/// Names of [`Stats::reasons`].
pub const REASONS: [&str; 10] = [
    "whole",
    "prefix, then unmapped",
    "prefix, then not arrived",
    "prefix, then another texture",
    "prefix, then out of order",
    "miss: unmapped",
    "miss: not arrived",
    "miss: no texture",
    "start inside a page",
    "",
];

impl Registry {
    pub fn new(cache_budget: u64) -> Registry {
        Registry {
            maps: RwLock::new(Maps { phys: IntervalMap::new(shift_held), va: IntervalMap::new(shift_frame) }),
            stamp: AtomicU64::new(0),
            files: RwLock::new(Vec::new()),
            cache: Mutex::new(Cache { pages: HashMap::new(), order: VecDeque::new(), bytes: 0, budget: cache_budget }),
            counts: Default::default(),
            reasons: Default::default(),
            absent_runs: AtomicU64::new(0),
            counts_x1: Default::default(),
        }
    }

    fn maps(&self) -> std::sync::RwLockReadGuard<'_, Maps> {
        self.maps.read().unwrap_or_else(|e| e.into_inner())
    }

    fn maps_mut(&self) -> std::sync::RwLockWriteGuard<'_, Maps> {
        self.maps.write().unwrap_or_else(|e| e.into_inner())
    }

    /// The id of the index of `path`, parsing it on first use (outside any
    /// lock). `None` if it does not parse.
    pub fn open(&self, path: &Path) -> Result<u32, String> {
        if let Some(f) = self.files.read().unwrap_or_else(|e| e.into_inner()).iter().find(|f| f.path == path) {
            return Ok(f.id);
        }
        let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let len = f.metadata().map_err(|e| e.to_string())?.len();
        let ttpf = Ttpf::read(&mut f, len)?;
        Ok(self.add(path.to_owned(), ttpf))
    }

    /// Adds a parsed index (or finds the one already added for `path`).
    pub fn add(&self, path: PathBuf, ttpf: Ttpf) -> u32 {
        let mut files = self.files.write().unwrap_or_else(|e| e.into_inner());
        if let Some(f) = files.iter().find(|f| f.path == path) {
            return f.id;
        }
        let id = files.len() as u32;
        files.push(Arc::new(FileIndex::new(id, path, ttpf)));
        id
    }

    pub fn file(&self, id: u32) -> Option<Arc<FileIndex>> {
        self.files.read().unwrap_or_else(|e| e.into_inner()).get(id as usize).cloned()
    }

    /// A completed request of `file` at `offset` wrote `len` bytes to
    /// physical pages (64 KiB page numbers in 4 KiB frames), from
    /// `page_offset` into the first. Returns what it resolved to.
    pub fn record_pages(&self, file: u32, offset: u64, pages: &[usize], page_offset: usize, len: usize) -> Option<Resolved> {
        self.record(file, offset, pages, page_offset, len, false)
    }

    /// As [`Registry::record_pages`] for a read that was not done (the
    /// page's bytes are not in title memory; the pack has them).
    pub fn record_absent(&self, file: u32, offset: u64, pages: &[usize], page_offset: usize, len: usize) -> Option<Resolved> {
        self.record(file, offset, pages, page_offset, len, true)
    }

    fn record(&self, file: u32, offset: u64, pages: &[usize], page_offset: usize, len: usize, absent: bool) -> Option<Resolved> {
        let index = self.file(file)?;
        let Some(r) = index.resolve(offset) else {
            self.forget_pages(pages, page_offset, len);
            return None;
        };
        let (first, end) = (page_offset as u64, (page_offset + len) as u64);
        if first % FRAME != 0 || r.at % FRAME != 0 {
            self.forget_pages(pages, page_offset, len);
            return None;
        }
        // Whole frames only: a short last frame is not claimed.
        let end = end / FRAME * FRAME;
        let mips = index.mips_in(r.texture, r.at, r.at + (end - first));
        let mut maps = self.maps_mut();
        let stamp = self.stamp.fetch_add(1, Ordering::Relaxed) + 1;
        let mut at = first;
        while at < end {
            let big = (at / BIG_PAGE) as usize;
            let Some(&number) = pages.get(big) else { break };
            let to = ((big as u64 + 1) * BIG_PAGE).min(end);
            let frame = number as u64 + (at % BIG_PAGE) / FRAME;
            let held = Held { file, page: r.page, texture: r.texture, at: r.at + (at - first), mips, stamp, absent };
            maps.phys.insert(frame, frame + (to - at) / FRAME, held);
            at = to;
        }
        drop(maps);
        self.note_absent(absent);
        self.counts[2].fetch_add(1, Ordering::Relaxed);
        if let Some(hook) = ARRIVAL.get() {
            hook(file, r.page, r.texture);
        }
        Some(r)
    }

    /// Anything else wrote these physical pages: what they held is gone.
    pub fn forget_pages(&self, pages: &[usize], page_offset: usize, len: usize) {
        let (mut at, end) = (page_offset as u64, (page_offset + len) as u64);
        let mut maps = self.maps_mut();
        while at < end {
            let big = (at / BIG_PAGE) as usize;
            let Some(&number) = pages.get(big) else { break };
            let to = ((big as u64 + 1) * BIG_PAGE).min(end);
            let frame = number as u64 + (at % BIG_PAGE) / FRAME;
            maps.phys.remove(frame, number as u64 + (to - big as u64 * BIG_PAGE).div_ceil(FRAME));
            at = to;
        }
    }

    /// Physical pages freed (64 KiB page numbers).
    pub fn forget_frames(&self, pages: &[usize]) {
        let mut maps = self.maps_mut();
        for &p in pages {
            maps.phys.remove(p as u64, p as u64 + BIG_PAGE / FRAME);
        }
    }

    /// A write into title memory `[addr, addr + len)` (kernel reads,
    /// DirectStorage address destinations): the frames behind it lose what
    /// they held.
    pub fn forget_va(&self, addr: usize, len: usize) {
        if len == 0 {
            return;
        }
        let (lo, hi) = ((addr as u64) / FRAME, (addr as u64 + len as u64).div_ceil(FRAME));
        // Most kernel reads land in memory with no frames we track: look
        // first under the read lock.
        let tracked = |m: &Maps| m.va.at(lo).is_some() || m.va.runs.range(lo..hi).next().is_some();
        if !tracked(&self.maps()) {
            return;
        }
        let mut maps = self.maps_mut();
        let mut at = lo;
        while at < hi {
            let Some((s, e, f)) = maps.va.at(at) else {
                // Skip to the next mapped run, if any inside.
                match maps.va.runs.range(at..hi).next() {
                    Some((&s, _)) => {
                        at = s;
                        continue;
                    }
                    None => break,
                }
            };
            let to = e.min(hi);
            let frame = f + (at - s);
            maps.phys.remove(frame, frame + (to - at));
            at = to;
        }
    }

    /// xmem mapped `[addr, addr + len)` to consecutive frames from `frame`.
    pub fn map_va(&self, addr: usize, len: usize, frame: u64) {
        let lo = addr as u64 / FRAME;
        self.maps_mut().va.insert(lo, lo + (len as u64).div_ceil(FRAME), frame);
    }

    /// xmem unmapped `[addr, addr + len)`.
    pub fn unmap_va(&self, addr: usize, len: usize) {
        let lo = addr as u64 / FRAME;
        self.maps_mut().va.remove(lo, lo + (len as u64).div_ceil(FRAME));
    }

    /// Whether the whole of `[addr, addr + len)` holds pages of one texture,
    /// in file order, and if so which and how new. No allocation.
    pub fn origin_of(&self, addr: u64, len: u64) -> Option<Origin> {
        match self.prefix_of(addr, len) {
            (Some(o), Cut::Whole) => Some(o),
            _ => None,
        }
    }

    /// The longest start of `[addr, addr + len)` that holds pages of one
    /// texture in file order (`Origin::len` its bytes, whole frames from
    /// the frame of `addr`), and where the walk stopped: a texture placed
    /// over memory whose low mips (the start of its chain) have arrived and
    /// whose larger mips have not is a prefix. Counted by reason. No
    /// allocation.
    pub fn prefix_of(&self, addr: u64, len: u64) -> (Option<Origin>, Cut) {
        self.counts[0].fetch_add(1, Ordering::Relaxed);
        if !addr.is_multiple_of(FRAME) {
            self.reasons[SUB_PAGE].fetch_add(1, Ordering::Relaxed);
        }
        let (o, cut) = self.walk(addr, len);
        let o = o.filter(|o| o.texture != NO_TEXTURE && o.len > 0);
        let reason = match (o.is_some(), cut) {
            (_, Cut::Whole) if o.is_some() => 0,
            (true, c) => c as usize,
            (false, Cut::Whole | Cut::OtherTexture | Cut::OutOfOrder) => MISS_NO_TEXTURE,
            (false, c) => MISS_BASE + c as usize,
        };
        self.reasons[reason].fetch_add(1, Ordering::Relaxed);
        if o.is_some() && cut == Cut::Whole {
            self.counts[1].fetch_add(1, Ordering::Relaxed);
        }
        (o, if o.is_some() { cut } else if cut == Cut::Whole { Cut::NotArrived } else { cut })
    }

    /// [`Registry::prefix_of`] without counting (refreshes of known
    /// origins, retries).
    pub fn peek(&self, addr: u64, len: u64) -> Option<Origin> {
        self.walk(addr, len).0.filter(|o| o.texture != NO_TEXTURE && o.len > 0)
    }

    fn walk(&self, addr: u64, len: u64) -> (Option<Origin>, Cut) {
        if len == 0 {
            return (None, Cut::Unmapped);
        }
        let (lo, hi) = (addr / FRAME, (addr + len).div_ceil(FRAME));
        let maps = self.maps();
        let mut out: Option<Origin> = None;
        let mut at = lo;
        while at < hi {
            let Some((s, e, f)) = maps.va.at(at) else { return (out, Cut::Unmapped) };
            let to = e.min(hi);
            let (mut p, pend) = (f + (at - s), f + (to - s));
            while p < pend {
                let Some((ps, pe, h)) = maps.phys.at(p) else { return (out, Cut::NotArrived) };
                let file_at = h.at + (p - ps) * FRAME;
                let step = pe.min(pend) - p;
                match &mut out {
                    None => {
                        out = Some(Origin { file: h.file, texture: h.texture, at: file_at, len: 0, mips: h.mips, pages: (h.page, h.page), stamp: h.stamp, absent: h.absent });
                    }
                    Some(o) => {
                        if o.file != h.file || o.texture != h.texture {
                            return (out, Cut::OtherTexture);
                        }
                        if file_at != o.at + o.len {
                            return (out, Cut::OutOfOrder);
                        }
                        o.mips = (o.mips.0.min(h.mips.0), o.mips.1.max(h.mips.1));
                        o.pages = (o.pages.0.min(h.page), o.pages.1.max(h.page));
                        o.stamp = o.stamp.max(h.stamp);
                        o.absent |= h.absent;
                    }
                }
                if let Some(o) = &mut out {
                    o.len += step * FRAME;
                }
                p += step;
            }
            at = to;
        }
        (out, Cut::Whole)
    }

    /// Before title memory `[addr, addr + len)` is read for its bytes
    /// (a fallback upload, a capture): every bypassed page under it is
    /// written now by the fill hook (the HLE reads and decodes it as the
    /// request would have). False if one could not be.
    pub fn ensure_present(&self, addr: u64, len: u64) -> bool {
        if self.absent_runs.load(Ordering::Relaxed) == 0 || len == 0 {
            return true;
        }
        let mut wanted: Vec<(u32, u32)> = Vec::new();
        {
            let maps = self.maps();
            let (mut at, hi) = (addr / FRAME, (addr + len).div_ceil(FRAME));
            while at < hi {
                let Some((s, e, f)) = maps.va.at(at) else {
                    at += 1;
                    continue;
                };
                let to = e.min(hi);
                let (mut p, pend) = (f + (at - s), f + (to - s));
                while p < pend {
                    match maps.phys.at(p) {
                        Some((_, pe, h)) => {
                            if h.absent && !wanted.contains(&(h.file, h.page)) {
                                wanted.push((h.file, h.page));
                            }
                            p = pe.min(pend);
                        }
                        None => p += 1,
                    }
                }
                at = to;
            }
        }
        let Some(fill) = FILL.get() else { return wanted.is_empty() };
        wanted.into_iter().all(|(file, page)| {
            self.counts_x1[1].fetch_add(1, Ordering::Relaxed);
            fill(file, page)
        })
    }

    /// Counts runs recorded absent (a cheap "is any bypass active").
    fn note_absent(&self, absent: bool) {
        if absent {
            self.absent_runs.fetch_add(1, Ordering::Relaxed);
            self.counts_x1[0].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The bypass counters ([`BYPASS_COUNTS`]).
    pub fn bypass_counts(&self) -> [u64; 6] {
        std::array::from_fn(|i| self.counts_x1[i].load(Ordering::Relaxed))
    }

    /// A `.ttpf` page request completed, bypassed or not: its stored
    /// bytes (read from the file), decoded bytes, and the decoding time a
    /// bypass saves (estimated, microseconds).
    pub fn count_request(&self, bypassed: bool, stored: u64, decoded: u64, decode_us: u64) {
        self.counts_x1[2].fetch_add(1, Ordering::Relaxed);
        if bypassed {
            self.counts_x1[3].fetch_add(stored, Ordering::Relaxed);
            self.counts_x1[4].fetch_add(decoded, Ordering::Relaxed);
            self.counts_x1[5].fetch_add(decode_us, Ordering::Relaxed);
        }
    }

    /// Whether the frames of `pages` from `page_offset` still hold
    /// `file`'s page `page` recorded absent.
    pub fn held_absent(&self, file: u32, page: u32, pages: &[usize], page_offset: usize) -> bool {
        let Some(&number) = pages.get(page_offset / BIG_PAGE as usize) else { return false };
        let frame = number as u64 + (page_offset as u64 % BIG_PAGE) / FRAME;
        self.maps().phys.at(frame).is_some_and(|(_, _, h)| h.absent && h.file == file && h.page == page)
    }

    /// For the log: what the registry knows about the frame of `addr` (the
    /// address run, the frame's file page and its texture).
    pub fn describe(&self, addr: u64) -> String {
        let maps = self.maps();
        let at = addr / FRAME;
        let Some((s, e, f)) = maps.va.at(at) else { return format!("{addr:#x}: no physical frames mapped") };
        let frame = f + (at - s);
        let Some((ps, pe, h)) = maps.phys.at(frame) else {
            return format!("{addr:#x}: frame {frame:#x} (run of {} frames) holds no recorded page", e - s);
        };
        let file_at = h.at + (frame - ps) * FRAME;
        drop(maps);
        let tex = self.file(h.file).and_then(|f| f.texture(h.texture).copied());
        let page = self.file(h.file).and_then(|f| f.ttpf.pages.get(h.page as usize).copied());
        format!(
            "{addr:#x}: frame {frame:#x} = file {} page {} ({:?}) at file {file_at:#x} (run of {} frames), texture {} {}",
            h.file,
            h.page,
            page.map(|p| (p.addr, p.size)),
            pe - ps,
            h.texture,
            tex.map_or_else(String::new, |t| format!(
                "{}x{} mips {} format {} kind {} mode {:#x} at {:#x} span {:#x} (+{:#x} into it)",
                t.width,
                t.height,
                t.mips,
                t.format,
                t.kind,
                t.mode,
                t.addr,
                t.span,
                file_at.wrapping_sub(t.addr)
            ))
        )
    }

    /// Keeps the decoded bytes of a whole file page.
    pub fn put_decoded(&self, file: u32, page: u32, bytes: Vec<u8>) {
        let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        let n = bytes.len() as u64;
        if n > c.budget {
            return;
        }
        if let Some(old) = c.pages.insert((file, page), Arc::new(bytes)) {
            c.bytes -= old.len() as u64;
        } else {
            c.order.push_back((file, page));
        }
        c.bytes += n;
        while c.bytes > c.budget {
            let Some(k) = c.order.pop_front() else { break };
            if let Some(old) = c.pages.remove(&k) {
                c.bytes -= old.len() as u64;
            }
        }
    }

    /// The decoded bytes of file page `page` (its whole virtual size, console
    /// layout), if still cached. No allocation.
    pub fn decoded_page(&self, file: u32, page: u32) -> Option<Arc<Vec<u8>>> {
        let hit = self.cache.lock().unwrap_or_else(|e| e.into_inner()).pages.get(&(file, page)).cloned();
        self.counts[if hit.is_some() { 3 } else { 4 }].fetch_add(1, Ordering::Relaxed);
        hit
    }

    pub fn stats(&self) -> Stats {
        let [lookups, hits, recorded, cache_hits, cache_misses] = std::array::from_fn(|i| self.counts[i].load(Ordering::Relaxed));
        let maps = self.maps();
        Stats {
            lookups,
            hits,
            recorded,
            phys_runs: maps.phys.len() as u64,
            va_runs: maps.va.len() as u64,
            cache_hits,
            cache_misses,
            cache_bytes: self.cache.lock().unwrap_or_else(|e| e.into_inner()).bytes,
            reasons: std::array::from_fn(|i| self.reasons[i].load(Ordering::Relaxed)),
        }
    }
}

/// Called after each recorded page read, `(file, page, texture)`, on the
/// DirectStorage worker that completed it, no lock held: the GPU layer
/// queues it for off-frame preparation (the hook lives here so that
/// `xnw-gpu` need not see `xnw-hle`). Set once.
static ARRIVAL: OnceLock<fn(u32, u32, u32)> = OnceLock::new();

/// Writes a bypassed page into title memory now, `(file, page)`; true
/// if it did (or it is no longer held). Set by the HLE (DirectStorage).
static FILL: OnceLock<fn(u32, u32) -> bool> = OnceLock::new();

/// Sets the fill hook; false if one is set already.
pub fn set_fill_hook(hook: fn(u32, u32) -> bool) -> bool {
    FILL.set(hook).is_ok()
}

/// The texture pack (`XNW_TEXPACK=<dir>`, default `out\\texpack`; `0`
/// off), opened once for the process; `None` if it holds no pack.
pub fn pack() -> Option<&'static crate::texpack::PackDir> {
    static PACK: OnceLock<Option<crate::texpack::PackDir>> = OnceLock::new();
    PACK.get_or_init(|| {
        let dir = std::env::var("XNW_TEXPACK").unwrap_or_else(|_| "out\\texpack".into());
        if dir == "0" {
            return None;
        }
        crate::texpack::open_dir(Path::new(&dir)).ok().filter(|p| !p.packs.is_empty())
    })
    .as_ref()
}

/// Sets the arrival hook; false if one is set already.
pub fn set_arrival_hook(hook: fn(u32, u32, u32)) -> bool {
    ARRIVAL.set(hook).is_ok()
}

/// The process's registry; the decoded-page budget from
/// `XNW_ORIGIN_CACHE_MB` (default 256, 0 turns the cache off).
pub fn registry() -> &'static Registry {
    static R: OnceLock<Registry> = OnceLock::new();
    R.get_or_init(|| {
        let mb = std::env::var("XNW_ORIGIN_CACHE_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(256);
        Registry::new(mb << 20)
    })
}

/// [`Registry::origin_of`] on the process's registry: the GPU layer's call.
pub fn origin_of(addr: u64, len: u64) -> Option<Origin> {
    registry().origin_of(addr, len)
}

/// [`Registry::prefix_of`] on the process's registry.
pub fn prefix_of(addr: u64, len: u64) -> (Option<Origin>, Cut) {
    registry().prefix_of(addr, len)
}

/// [`Registry::peek`] on the process's registry.
pub fn peek(addr: u64, len: u64) -> Option<Origin> {
    registry().peek(addr, len)
}

/// [`Registry::describe`] on the process's registry.
pub fn describe(addr: u64) -> String {
    registry().describe(addr)
}

/// [`Registry::decoded_page`] on the process's registry.
pub fn decoded_page(file: u32, page: u32) -> Option<Arc<Vec<u8>>> {
    registry().decoded_page(file, page)
}

/// [`Registry::file`] on the process's registry: the texture's
/// description (`file(o.file)?.texture(o.texture)`).
pub fn file(id: u32) -> Option<Arc<FileIndex>> {
    registry().file(id)
}

/// Whether a path names a texture page file.
pub fn is_ttpf(path: &Path) -> bool {
    path.extension().is_some_and(|e| e.eq_ignore_ascii_case("ttpf"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttpf::Page;

    fn shift(v: u64, d: u64) -> u64 {
        v + d
    }

    fn runs(m: &IntervalMap<u64>) -> Vec<(u64, u64, u64)> {
        m.runs.iter().map(|(&s, &(e, v))| (s, e, v)).collect()
    }

    #[test]
    fn interval_insert_overwrite_remove() {
        let mut m = IntervalMap::new(shift);
        m.insert(10, 20, 100);
        m.insert(30, 40, 300);
        assert_eq!(m.at(15), Some((10, 20, 100)));
        assert_eq!(m.at(20), None);
        // Overwrite across both: the edges are cut and keep their values.
        m.insert(15, 35, 900);
        assert_eq!(runs(&m), vec![(10, 15, 100), (15, 35, 900), (35, 40, 305)]);
        // A hole inside one run splits it.
        m.remove(20, 22);
        assert_eq!(runs(&m), vec![(10, 15, 100), (15, 20, 900), (22, 35, 907), (35, 40, 305)]);
        m.remove(0, 100);
        assert!(m.is_empty());
    }

    /// One file: texture 0 at 0x0 (pages 0x0 + 0x1000 + 0x2000), texture
    /// 1 at 0x10000 (one 64 KiB page).
    fn registry_with_file() -> (Registry, u32) {
        let page = |addr, size, offset| Page { addr, size, offset, stored: 0x10, storage: 8 };
        let tex = |addr, span| Texture { hash: addr, width: 4, height: 4, mips: 1, format: 71, layers: 1, kind: 6, mode: 0, addr, span, words: [0; 4] };
        let ttpf = Ttpf {
            pages: vec![page(0, 0x1000, 0x100), page(0x1000, 0x1000, 0x200), page(0x2000, 0xe000, 0x300), page(0x10000, 0x10000, 0x400)],
            textures: vec![tex(0, 0x10000), tex(0x10000, 0x10000)],
            header_bytes: 0x100,
        };
        let r = Registry::new(1 << 20);
        let id = r.add(PathBuf::from("x.ttpf"), ttpf);
        (r, id)
    }

    #[test]
    fn origin_follows_requests_and_maps() {
        let (r, f) = registry_with_file();
        // Physical 64 KiB pages 0x100000 and 0x100010 mapped at 0x5_0000.
        r.map_va(0x5_0000, 0x2_0000, 0x10_0000);
        // Texture 0's three pages land in page 0x100000 in order.
        assert_eq!(r.record_pages(f, 0x100, &[0x10_0000], 0, 0x1000).map(|x| x.page), Some(0));
        assert!(r.origin_of(0x5_0000, 0x2000).is_none(), "partial cover is None");
        r.record_pages(f, 0x200, &[0x10_0000], 0x1000, 0x1000);
        r.record_pages(f, 0x300, &[0x10_0000], 0x2000, 0xe000);
        let o = r.origin_of(0x5_0000, 0x1_0000).unwrap();
        assert_eq!((o.file, o.texture, o.at, o.len, o.pages, o.stamp), (f, 0, 0, 0x1_0000, (0, 2), 3));
        // Texture 1 next to it: a range over both is not one texture.
        r.record_pages(f, 0x400, &[0x10_0010], 0, 0x1_0000);
        assert_eq!(r.origin_of(0x6_0000, 0x1_0000).map(|o| o.texture), Some(1));
        assert!(r.origin_of(0x5_0000, 0x2_0000).is_none());
        // Rewriting a page raises the stamp.
        r.record_pages(f, 0x200, &[0x10_0000], 0x1000, 0x1000);
        assert_eq!(r.origin_of(0x5_0000, 0x1_0000).unwrap().stamp, 5);
        // A kernel read over one frame drops it.
        r.forget_va(0x5_1000, 10);
        assert!(r.origin_of(0x5_0000, 0x1_0000).is_none());
        assert!(r.origin_of(0x5_2000, 0xe000).is_some());
        // Remap: the frames keep their content at the new address.
        r.unmap_va(0x6_0000, 0x1_0000);
        assert!(r.origin_of(0x6_0000, 0x1_0000).is_none());
        r.map_va(0x9_0000, 0x1_0000, 0x10_0010);
        assert_eq!(r.origin_of(0x9_0000, 0x1_0000).map(|o| o.texture), Some(1));
        // Freed physical pages lose it.
        r.forget_frames(&[0x10_0010]);
        assert!(r.origin_of(0x9_0000, 0x1_0000).is_none());
        // Another write into physical pages drops what was there.
        r.forget_pages(&[0x10_0000], 0x2000, 0x1000);
        assert!(r.origin_of(0x5_2000, 0xe000).is_none());
        assert!(r.origin_of(0x5_3000, 0xd000).is_some());
        // An offset that is no page start resolves to nothing.
        assert!(r.record_pages(f, 0x108, &[0x10_0000], 0, 0x1000).is_none());
    }

    /// A bypassed page is recorded absent; reading its memory for its
    /// bytes asks the fill hook once per page.
    #[test]
    fn absent_pages_are_filled_on_demand() {
        static FILLED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        fn fill(_: u32, page: u32) -> bool {
            FILLED.fetch_add(1 + page, Ordering::Relaxed);
            true
        }
        let (r, f) = registry_with_file();
        r.map_va(0x5_0000, 0x1_0000, 0x10_0000);
        assert!(r.ensure_present(0x5_0000, 0x1_0000), "nothing absent yet");
        r.record_absent(f, 0x100, &[0x10_0000], 0, 0x1000).unwrap();
        r.record_pages(f, 0x200, &[0x10_0000], 0x1000, 0x1000).unwrap();
        let o = r.origin_of(0x5_0000, 0x2000).unwrap();
        assert!(o.absent);
        assert!(!r.origin_of(0x5_1000, 0x1000).unwrap().absent);
        assert!(r.held_absent(f, 0, &[0x10_0000], 0));
        assert!(!r.held_absent(f, 1, &[0x10_0000], 0x1000));
        set_fill_hook(fill);
        assert!(r.ensure_present(0x5_0000, 0x2000));
        assert_eq!(FILLED.load(Ordering::Relaxed), 1, "page 0 only");
        assert_eq!(r.bypass_counts()[..2], [1, 1]);
    }

    #[test]
    fn decoded_cache_is_bounded() {
        let r = Registry::new(0x3000);
        for p in 0..4 {
            r.put_decoded(0, p, vec![p as u8; 0x1000]);
        }
        assert!(r.decoded_page(0, 0).is_none());
        assert_eq!(r.decoded_page(0, 3).unwrap()[0], 3);
        assert_eq!(r.stats().cache_bytes, 0x3000);
    }
}
