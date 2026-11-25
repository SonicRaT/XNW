//! Host copies of title buffers. Title buffers are title memory, which the
//! host GPU cannot read, so a buffer a draw uses gets a persistent host
//! buffer of its size on first use (system memory shaders may also write),
//! and the ranges each submission reads (vertex, index and constant data,
//! buffer views) are brought up to date from title memory before it runs:
//! the 64 KiB pages the title wrote since they were last copied
//! ([`super::watch`]: write watching, or read-only guards on physical-page
//! views), all of it where neither works. A title GPU address inside the
//! buffer is the same offset in its copy. Ranges shaders write go back to
//! title memory after the submission ([`Shadow::gpu_writes`]). Addresses in
//! the GPU view of the paired reservation work as any other: the watch
//! asks their CPU view.
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.
//!
//! Ported from VIctory-base (branch `xnw/base` @
//! 5d0516e = VIctory d2a4bae + an upstream change) `crates/victory-gpu/src/shadow.rs`
//! 1-512, GPL-3.0-or-later. Changed: `tracing` -> `xnw_core` (debug lines
//! dropped), the diagnostics counters dropped, the host layer through
//! [`super::host`]; bytes held counted for the
//! memory line.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use super::host;
use super::host::HostList;
use super::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// The copy's bytes before the submission ran, by title page (64 KiB,
    /// cut to the buffer), for the pages its shaders may write: what the GPU
    /// wrote differs.
    before: Mutex<BTreeMap<usize, Vec<u8>>>,
    /// What the buffer was seen doing (`XNW_ROLES`).
    role: Option<super::roles::Role>,
    /// Roles on: what shaders were declared to write and has not gone back.
    owed: Mutex<Owed>,
    /// Roles on: title pages shaders were ever declared to write through
    /// this copy; views over them are read through it too (sticky).
    gpu_pages: Mutex<std::collections::BTreeSet<usize>>,
}

/// Roles on, per title page of a copy: the byte ranges (offsets in the
/// copy) shaders were declared to write, while their submission is being
/// prepared or run (`inflight`) and after it ran (`done`); and, for an in-
/// flight page the CPU touched (its trap went), the title's bytes then
/// (`baseline`). Pages are trapped from the declaration on, so a CPU write
/// cannot slip in unseen: a done page still trapped was not touched since,
/// and its declared bytes are the GPU's; a touched page is merged at the end
/// of the submission: bytes the CPU changed since its baseline stay the
/// CPU's (whole ranges put back over the title's own fence records,
/// title+0x4e9b54a read null + 0xc), the rest of the declared bytes the GPU's.
#[derive(Default)]
struct Owed {
    inflight: BTreeMap<usize, Vec<(usize, usize)>>,
    done: BTreeMap<usize, Vec<(usize, usize)>>,
    /// Lock mode: per done page, the write-watch asking noted when the GPU
    /// write was declared: a CPU write journalled after it wins.
    done_at: BTreeMap<usize, u64>,
    baseline: BTreeMap<usize, Vec<u8>>,
}

/// Adds `[x, y)` to a page's ranges unless one covers it.
fn add_range(list: &mut Vec<(usize, usize)>, (x, y): (usize, usize)) {
    if !list.iter().any(|&(p, q)| p <= x && y <= q) {
        list.push((x, y));
    }
}

/// Lock mode: GPU-written pages the CPU wrote again before the title locked
/// them (the CPU's bytes won), for the statistics.
pub(crate) static CPU_WON: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// In-flight pages the CPU touched (a baseline kept), for the statistics.
pub(crate) static BASELINES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Bytes put back on demand (roles on), for the statistics.
pub(crate) static DEFERRED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Drop for Shadow {
    fn drop(&mut self) {
        // SAFETY: a live buffer.
        HELD.fetch_sub(unsafe { self.buffer.GetDesc() }.Width, Ordering::Relaxed);
    }
}

/// Snapshot pages kept for reuse (already touched, so no page faults):
/// fresh snapshots cost ~2.3 million page faults per world frame.
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());
/// Snapshot pages kept at most (256 MiB). 64 MiB left a submission of
/// the world bench (VIctory) taking fresh pages every time:
/// median 2,028 -> 1,420 ms a frame there.
const POOL_PAGES: usize = 4096;

/// An empty page-sized buffer, from the pool if it has one.
fn pooled() -> Vec<u8> {
    let mut v = POOL.lock().unwrap_or_else(|e| e.into_inner()).pop().unwrap_or_else(|| Vec::with_capacity(PAGE));
    v.clear();
    v
}

/// Returns snapshot pages to the pool.
fn give_back(pages: impl Iterator<Item = Vec<u8>>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    for v in pages {
        if pool.len() < POOL_PAGES {
            pool.push(v);
        }
    }
}

/// Snapshot bytes taken per frame statistics window, and pages.
pub(crate) static SNAPSHOT: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// The copy's bytes at title `address` (tests; failure reports).
    pub(crate) fn bytes_at(&self, address: u64, len: usize) -> Vec<u8> {
        // SAFETY: inside the copy (the caller's range lies in the buffer).
        unsafe { std::slice::from_raw_parts(self.data.add((address - self.start) as usize), len) }.to_vec()
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// `len` bytes of the copy at title address `address` (tests: what the
    /// GPU wrote there).
    #[cfg(test)]
    pub(crate) fn copy_bytes(&self, address: u64, len: usize) -> Vec<u8> {
        // SAFETY: inside the mapping (callers stay within the buffer).
        unsafe { std::slice::from_raw_parts(self.data.add((address - self.start) as usize), len) }.to_vec()
    }

    /// The title memory the copy holds: its first address and length
    /// (VIctory fe078c6, GPL-3.0-or-later; for the address map).
    pub(crate) fn span(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        // SAFETY: a live buffer.
        HELD.fetch_add(unsafe { buffer.GetDesc() }.Width, Ordering::Relaxed);
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            before: Mutex::default(),
            role: super::roles::new_role(start, len),
            owed: Mutex::default(),
            gpu_pages: Mutex::default(),
        }
    }

    /// What the buffer was seen doing, when roles are on or probed.
    pub(crate) fn role(&self) -> Option<&super::roles::Role> {
        self.role.as_ref()
    }

    /// Roles on: a trapped block of the buffer is touched (the title, or our
    /// own read of title memory), and is writable for this call. A page of
    /// it whose submission has not finished keeps the title's bytes as its
    /// baseline. Every done range of the buffer goes back now, the whole
    /// buffer at once (one stop, not one per page: another emulator's per-fault
    /// read-backs cost 7 ms a frame): those
    /// pages stayed trapped since their declaration, so the title did not
    /// write them and the declared bytes in the copy are the GPU's.
    /// shortcut: a title write between the refresh that fed the shaders and
    /// the declaration (within one submission's preparation) is not seen.
    pub(crate) fn copy_back_deferred(&self, block_start: usize) {
        let start = self.start as usize;
        let end = start + self.len as usize;
        let page = (block_start + super::watch::alias_of(start, end)) / PAGE;
        // The done ranges first, then baselines: a page done from an earlier
        // submission and in flight again keeps, as its baseline, the title's
        // bytes with the earlier results in (its trap goes with the write).
        // shortcut: a CPU write between that write-back and the baseline is
        // taken for the title's bytes from before.
        let (done, mut needs) = {
            let mut owed = self.owed.lock().unwrap_or_else(|e| e.into_inner());
            let done = std::mem::take(&mut owed.done);
            let needs: Vec<usize> = std::iter::once(page)
                .chain(done.keys().copied())
                .filter(|p| owed.inflight.contains_key(p) && !owed.baseline.contains_key(p))
                .collect();
            (done, needs)
        };
        needs.sort_unstable();
        needs.dedup();
        let mut written = 0u64;
        for (page, ranges) in done {
            let (a, b) = self.page_in_copy(page);
            // Our own write: the block's trap goes first (a guard stays; a
            // write through it opens it in the handler).
            super::watch::untrap(start + a, b - a);
            for (x, y) in ranges {
                // SAFETY: inside the mapping and the title buffer, which is
                // accessible now.
                unsafe { std::ptr::copy_nonoverlapping(self.data.add(x), (start + x) as *mut u8, y - x) };
                written += (y - x) as u64;
            }
        }
        DEFERRED.fetch_add(written, Ordering::Relaxed);
        let mut owed = self.owed.lock().unwrap_or_else(|e| e.into_inner());
        for page in needs {
            if owed.baseline.contains_key(&page) {
                continue;
            }
            let (a, b) = self.page_in_copy(page);
            // SAFETY: the title page: the faulting block (writable for this
            // call) or one untrapped above.
            let bytes = unsafe { std::slice::from_raw_parts((start + a) as *const u8, b - a) }.to_vec();
            owed.baseline.insert(page, bytes);
            BASELINES.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Roles on: an in-flight page the CPU touched, after its submission
    /// ran: per declared byte, the CPU's if it changed the byte since the
    /// baseline, else the GPU's (the copy's). The copy takes the result too.
    /// Returns the bytes written to title memory.
    fn merge_page(&self, page: usize, ranges: &[(usize, usize)], base: &[u8]) -> usize {
        let start = self.start as usize;
        let (a, _) = self.page_in_copy(page);
        let mut written = 0;
        for &(x, y) in ranges {
            let mut title = vec![0u8; y - x];
            if !host::read_into((start + x) as *const u8, &mut title) {
                continue;
            }
            // SAFETY: inside the mapping; only the submitting thread writes it.
            let copy = unsafe { std::slice::from_raw_parts_mut(self.data.add(x), y - x) };
            let mut changed = false;
            for ((t, h), b) in title.iter_mut().zip(copy.iter_mut()).zip(&base[x - a..y - a]) {
                if *t == *b {
                    if *t != *h {
                        *t = *h;
                        changed = true;
                    }
                } else {
                    *h = *t;
                }
            }
            if changed {
                super::watch::writing(start + x, y - x);
                host::write_mapped((start + x) as *mut u8, title.as_ptr(), y - x);
                written += y - x;
            }
        }
        written
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// each title page of it once per copy and submission however many
    /// ranges declare it (the world declares overlapping views of its
    /// 174 MB heap dozens of times), in reused memory.
    ///
    /// The snapshot was one span from the lowest declared byte to the
    /// highest, taken again whole each time a range widened it: views
    /// spread over the big heap copied most of it several times per
    /// submission (bindless write ranges 410 ms and gpu-write steps 248 ms a
    /// frame). Pages nobody declared are
    /// not taken.
    pub(crate) fn writing(&self, address: usize, len: usize) {
        if no_copy_back() {
            return;
        }
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address + len));
        let (from, to) = self.inside(address, len);
        if from >= to {
            return;
        }
        let start = self.start as usize;
        if let Some(role) = &self.role {
            role.gpu_wrote();
            // Roles on: no snapshot. After the GPU the copy is the truth
            // for the range, which goes back when touched; other views of
            // its pages read this copy from now on.
            // Lock mode: no snapshot and no trap: the declared
            // ranges wait in the copy until the title locks the buffer for
            // the CPU ([`buffer_locked`]); other views read this copy.
            if super::roles::mode() == super::roles::Mode::Lock {
                let pages = (start + from) / PAGE..=(start + to - 1) / PAGE;
                self.gpu_pages.lock().unwrap_or_else(|e| e.into_inner()).extend(pages.clone());
                let asked = super::watch::asked();
                let mut owed = self.owed.lock().unwrap_or_else(|e| e.into_inner());
                for page in pages {
                    let (a, b) = self.page_in_copy(page);
                    add_range(owed.done.entry(page).or_default(), (a.max(from), b.min(to)));
                    owed.done_at.insert(page, asked);
                }
                return;
            }
            if role.deferred() {
                let pages = (start + from) / PAGE..=(start + to - 1) / PAGE;
                self.gpu_pages.lock().unwrap_or_else(|e| e.into_inner()).extend(pages.clone());
                // In flight from now on, and trapped now (before the
                // submission runs and returns to the title): a CPU touch from
                // here on is seen. A page touched already in this flight
                // keeps its baseline and is not trapped again.
                let mut owed = self.owed.lock().unwrap_or_else(|e| e.into_inner());
                let mut trap = Vec::new();
                for page in pages {
                    let (a, b) = self.page_in_copy(page);
                    add_range(owed.inflight.entry(page).or_default(), (a.max(from), b.min(to)));
                    if !owed.baseline.contains_key(&page) {
                        trap.push((a.max(from), b.min(to)));
                    }
                }
                drop(owed);
                for (x, y) in trap {
                    super::watch::trap_reads(start + x, y - x, role.tag);
                }
                return;
            }
        }
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let (mut bytes, mut pages) = (0, 0);
        for page in (start + from) / PAGE..=(start + to - 1) / PAGE {
            if before.contains_key(&page) {
                continue;
            }
            let (a, b) = self.page_in_copy(page);
            let mut v = pooled();
            // SAFETY: inside the mapping.
            v.extend_from_slice(unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) });
            before.insert(page, v);
            (bytes, pages) = (bytes + (b - a), pages + 1);
        }
        SNAPSHOT[0].fetch_add(bytes as u64, Ordering::Relaxed);
        SNAPSHOT[1].fetch_add(pages, Ordering::Relaxed);
        // The snapshot counter: bytes taken now.
        super::counters::snapshot(bytes);
    }

    /// Title page `page` cut to the buffer, as offsets in the copy.
    fn page_in_copy(&self, page: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        ((page * PAGE).max(start) - start, ((page + 1) * PAGE).min(end) - start)
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        if no_copy_back() {
            return 0;
        }
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        let deferred = self.role.as_ref().is_some_and(|r| r.deferred());
        // Our own reads of the range below must not fault on a trap (nor
        // run the roles' hook under the snapshot's lock). Roles on: the
        // range stays trapped (that is what keeps it exact).
        if from < to && !deferred {
            super::watch::reading(start + from, to - from);
        }
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        if from < to && deferred {
            // Roles on: the submission ran. Pages still trapped since
            // their declaration are done (they go back when touched); pages
            // the CPU touched meanwhile are merged now: per declared byte,
            // the CPU's if it changed the byte since its baseline, else the
            // GPU's.
            let mut merge = Vec::new();
            {
                let mut owed = self.owed.lock().unwrap_or_else(|e| e.into_inner());
                for page in (start + from) / PAGE..=(start + to - 1) / PAGE {
                    let Some(ranges) = owed.inflight.remove(&page) else { continue };
                    match owed.baseline.remove(&page) {
                        Some(base) => merge.push((page, ranges, base)),
                        None => {
                            let done = owed.done.entry(page).or_default();
                            for r in ranges {
                                add_range(done, r);
                            }
                        }
                    }
                }
            }
            for (page, ranges, base) in merge {
                written += self.merge_page(page, &ranges, &base);
            }
        } else if from < to {
            // By 64 KiB piece (a title page): the copy compared with its
            // bytes before the submission, title memory read only where the
            // GPU wrote.
            //
            // Each snapshot page is compared once, whole, at the first
            // range that reaches it, and by 4 KiB chunk: title memory is read
            // and merged only for chunks the GPU changed. Pages were compared
            // per declared range (dozens of overlapping views, parts of a page
            // again and again) and merged byte by byte for a whole 64 KiB
            // piece: 1.7 s a frame to write back 1.7 MiB in a test run.
            // The whole page is right: the GPU writes nothing a
            // declared range does not cover, and nothing else changes the copy
            // between the snapshot and now.
            // (Only the declared ranges of a page, merged: the rest of it may
            // have been refreshed from title memory after the snapshot.)
            const CHUNK: usize = 4 << 10;
            let mut declared: Vec<(usize, usize)> = self
                .pending
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|&(a, e)| self.inside(a, e - a))
                .filter(|&(a, e)| a < e)
                .collect();
            declared.sort_unstable();
            let mut title = vec![0u8; CHUNK];
            let mut synced = Vec::new();
            for page in (start + from) / PAGE..=(start + to - 1) / PAGE {
                // Not declared, or already compared at this submission.
                let Some(taken) = snapshot.remove(&page) else { continue };
                let (a, b) = self.page_in_copy(page);
                let mut any = false;
                // The declared parts of the page, merged, cut into chunks.
                let mut parts: Vec<(usize, usize)> = Vec::new();
                for &(x, y) in &declared {
                    let (x, y) = (x.max(a), y.min(b));
                    if x >= y {
                        continue;
                    }
                    match parts.last_mut() {
                        Some((_, e)) if x <= *e => *e = (*e).max(y),
                        _ => parts.push((x, y)),
                    }
                }
                let chunks = parts.into_iter().flat_map(|(x, y)| {
                    let mut out = Vec::new();
                    let mut at = x;
                    while at < y {
                        let n = (CHUNK - (start + at) % CHUNK).min(y - at);
                        out.push((at, n));
                        at += n;
                    }
                    out
                });
                for (at, n) in chunks {
                    // SAFETY: inside the mapping; only the submitting thread
                    // writes it.
                    let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                    let old = &taken[at - a..at - a + n];
                    let title = &mut title[..n];
                    if host != old && host::read_into((start + at) as *const u8, title) {
                        // Title memory with the GPU's bytes put in; the copy
                        // takes the title's bytes in turn.
                        let mut changed = false;
                        for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
                            if *h != *o && *t != *h {
                                *t = *h;
                                changed = true;
                            }
                            *h = *t;
                        }
                        if changed {
                            super::watch::writing(start + at, n);
                            host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                            written += n;
                        }
                        any = true;
                    }
                }
                if any {
                    synced.push(page);
                }
                give_back(std::iter::once(taken));
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if first < last
                && !synced.is_empty()
                && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
                && let Some((now, _)) = super::watch::writes(first * PAGE, last * PAGE, true)
            {
                let base = start / PAGE;
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                    copied[page - base] = now;
                }
            }
        }
        // Roles probed: copied back as before, then trapped to see
        // whether the title ever reads what the GPU wrote.
        if from < to
            && let Some(role) = self.role.as_ref().filter(|_| super::roles::mode() == super::roles::Mode::Probe)
        {
            super::watch::trap_reads(start + from, to - from, role.tag);
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|&r| r != (address, address + len));
        if pending.is_empty() && !snapshot.is_empty() {
            give_back(std::mem::take(&mut *snapshot).into_values());
        }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        // Lock mode: GPU results not taken by the title yet are newer in
        // the copy than in title memory: not refreshed over — unless
        // the CPU wrote the page after the GPU write was declared: then the
        // CPU's bytes win and the page is refreshed as any other (in a
        // test run, pages the GPU wrote once were never refreshed again, so the
        // title's later CPU writes there, pool memory reused, never reached
        // the copy, and the GPU hung on them at world entry).
        if super::roles::mode() == super::roles::Mode::Lock {
            let mut owed = self.owed.lock().unwrap_or_else(|e| e.into_inner());
            let pages: Vec<usize> = owed.done.range(from / PAGE..=(to - 1) / PAGE).map(|(&p, _)| p).collect();
            if !pages.is_empty() {
                // Memory the watch cannot answer for: the CPU may have
                // written it; it wins.
                let written: BTreeMap<usize, u64> = super::watch::writes(from, to, true).map(|(_, w)| w.into_iter().collect()).unwrap_or_else(|| pages.iter().map(|&p| (p, u64::MAX)).collect());
                for page in pages {
                    let declared = owed.done_at.get(&page).copied().unwrap_or(0);
                    if written.get(&page).is_some_and(|&w| w > declared) {
                        owed.done.remove(&page);
                        owed.done_at.remove(&page);
                        CPU_WON.fetch_add(1, Ordering::Relaxed);
                    } else if let Some(ranges) = owed.done.get(&page) {
                        busy.extend(ranges.iter().map(|&(x, y)| (start + x, start + y)).filter(|&(a, e)| a < to && e > from));
                    }
                }
            }
        }
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        // `XNW_SYNC=pools`: the title's page pool and frame arenas are
        // copied whole and plainly: what the CPU sends the GPU each frame
        // lives there, written before the submission that reads it.
        if crate::shared::sync_pools() && crate::shared::in_pools(from, to - from) {
            {
                // SAFETY: mapped title memory (xmem's pool or arena region)
                // into the copy; `[from, to)` lies in the buffer.
                unsafe { std::ptr::copy_nonoverlapping(from as *const u8, self.data.add(from - self.start as usize), to - from) };
                crate::shared::SYNC_POOLS[0].fetch_add(1, Ordering::Relaxed);
                crate::shared::SYNC_POOLS[1].fetch_add((to - from) as u64, Ordering::Relaxed);
                return to - from;
            }
        }
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = super::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                if let Some(role) = &self.role {
                    role.cpu_wrote(copied.div_ceil(PAGE) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.copy(from, to);
        to - from
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        host::copy_mapped(at, from as *const u8, to - from, false);
        to - from
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Bytes of host copies alive (system memory; a list that holds one keeps
/// it), for the memory line: added when made, taken off when dropped.
static HELD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Bytes of host copies of title memory alive now.
pub(crate) fn held() -> u64 {
    HELD.load(Ordering::Relaxed)
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Bumped whenever the set of copies changes, and whenever the set of
/// title buffers does (the address map is made again only then; the
/// buffers without a copy matter to it only with `XNW_SHARED=2`).
static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BUFFER_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// See [`GENERATION`]: copies, buffers.
pub(crate) fn generation() -> (u64, u64) {
    (GENERATION.load(Ordering::Relaxed), BUFFER_GENERATION.load(Ordering::Relaxed))
}

fn changed() {
    GENERATION.fetch_add(1, Ordering::Relaxed);
}

/// Every title buffer (start, length, its copy if made) and every copy of
/// title memory outside buffers, for the address map. VIctory
/// (fe078c6) `all`, which lists copies only; XNW: buffers without a
/// copy too, which the map reads in place when they resolve.
pub(crate) fn all() -> Vec<(u64, u64, Option<Arc<Shadow>>)> {
    let mut out: Vec<(u64, u64, Option<Arc<Shadow>>)> = buffers().iter().map(|(&(start, _), e)| (start, e.end - start, e.shadow.clone())).collect();
    out.extend(LOOSE.lock().unwrap_or_else(|e| e.into_inner()).iter().map(|(&(start, len), s)| (start, len, Some(s.clone()))));
    out
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    BUFFER_GENERATION.fetch_add(1, Ordering::Relaxed);
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    BUFFER_GENERATION.fetch_add(1, Ordering::Relaxed);
    if buffers().remove(&(start, resource)).is_some_and(|e| e.shadow.is_some()) {
        changed();
    }
}

/// Lock mode: the title locked `[address, address + len)` of a buffer
/// for the CPU (its lock function, `title+0x524dab0`: every CPU read
/// of a GPU-written buffer goes through it; detoured by the loader) from
/// `site`. The GPU's declared writes there go back to title memory now:
/// the copy's bytes for exactly the declared ranges in the locked range.
/// The title locks a version of the buffer older than the one the GPU
/// writes, so it does not write these ranges meanwhile. Returns the bytes
/// written. The address may be at either view of the paired reservation.
pub(crate) fn buffer_locked(address: usize, len: usize, site: usize) -> usize {
    if super::roles::mode() != super::roles::Mode::Lock || len == 0 {
        return 0;
    }
    // The title reads through the CPU view; copies may be keyed at the GPU
    // view (the paired reservation): both.
    let mut spans = vec![(address, address + len)];
    if let Some(gpu) = super::watch::gpu_alias(address) {
        spans.push((gpu, gpu + len));
    }
    let mut copies: Vec<Arc<Shadow>> = Vec::new();
    for &(a, e) in &spans {
        let map = buffers();
        copies.extend(
            map.range(..(e as u64, 0))
                .rev()
                .take(64)
                .filter(|&(&(s, _), en)| (s as usize) < e && a < en.end as usize)
                .filter_map(|(_, en)| en.shadow.clone()),
        );
        drop(map);
        copies.extend(LOOSE.lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|&(&(s, n), _)| (s as usize) < e && a < (s + n) as usize).map(|(_, s)| s.clone()));
    }
    let mut written = 0;
    for shadow in copies {
        let start = shadow.start as usize;
        for &(a, e) in &spans {
            let (lo, hi) = (a.max(start), e.min(start + shadow.len as usize));
            if lo >= hi {
                continue;
            }
            let ranges: Vec<(usize, usize)> = {
                let mut owed = shadow.owed.lock().unwrap_or_else(|e| e.into_inner());
                let mut out = Vec::new();
                for page in lo / PAGE..=(hi - 1) / PAGE {
                    let Some(list) = owed.done.get_mut(&page) else { continue };
                    // The parts inside the lock go back; what is left waits.
                    let mut keep = Vec::new();
                    for &(x, y) in list.iter() {
                        let (p, q) = ((start + x).max(lo), (start + y).min(hi));
                        if p < q {
                            out.push((p, q));
                            if start + x < p {
                                keep.push((x, p - start));
                            }
                            if q < start + y {
                                keep.push((q - start, y));
                            }
                        } else {
                            keep.push((x, y));
                        }
                    }
                    if keep.is_empty() {
                        owed.done.remove(&page);
                        owed.done_at.remove(&page);
                    } else {
                        *list = keep;
                    }
                }
                out
            };
            for (p, q) in ranges {
                super::watch::writing(p, q - p);
                // SAFETY: inside the copy's mapping; the title range is the
                // title's memory, written through the kernel.
                host::write_mapped(p as *mut u8, unsafe { shadow.data.add(p - start) }, q - p);
                written += q - p;
            }
        }
    }
    let mut locks = LOCKS.lock().unwrap_or_else(|e| e.into_inner());
    let s = locks.entry(site).or_default();
    s.0 += 1;
    s.1 += written as u64;
    written
}

/// Lock mode: per caller of the title's lock function, calls and bytes
/// written back (taken by the statistics).
static LOCKS: Mutex<BTreeMap<usize, (u64, u64)>> = Mutex::new(BTreeMap::new());

/// The lock calls' part of the statistics over `n` frames, zeroing them.
pub(crate) fn locks_text(n: u64) -> String {
    let all = std::mem::take(&mut *LOCKS.lock().unwrap_or_else(|e| e.into_inner()));
    let (calls, bytes) = all.values().fold((0, 0), |(c, b), &(x, y)| (c + x, b + y));
    let sites: Vec<String> = all.iter().map(|(&site, &(c, b))| format!("title+{:#x} {}/{} B", site.wrapping_sub(super::object::title_base()), c / n.max(1), b / n.max(1))).collect();
    format!("lock calls {}/frame, {} bytes/frame written back, GPU-written pages the CPU wrote again {}/frame; by caller: {}", calls / n.max(1), bytes / n.max(1), CPU_WON.swap(0, Ordering::Relaxed) / n.max(1), if sites.is_empty() { "none".into() } else { sites.join(", ") })
}

/// With GPU writes going back on demand (roles on): whether a host copy the
/// GPU writes (through a view not in place) overlaps `[address, address +
/// len)`: then the range is read through that copy too, not in place
/// (an in-place read of what the GPU wrote to the copy read stale
/// title memory). The copies that exist; none is made here.
pub(crate) fn gpu_written_copy(address: u64, len: u64) -> bool {
    if !super::roles::one_way() {
        return false;
    }
    let end = address.saturating_add(len.max(1));
    // By page, not by buffer: the world's 174 MB heap holds most buffers.
    let (first, last) = (address as usize / PAGE, (end as usize).div_ceil(PAGE));
    let written = |s: &Arc<Shadow>| s.gpu_pages.lock().unwrap_or_else(|e| e.into_inner()).range(first..last).next().is_some();
    let overlaps = |start: u64, stop: u64| start < end && address < stop;
    if buffers()
        .range(..(end, 0))
        .rev()
        .take(64)
        .any(|(&(start, _), e)| overlaps(start, e.end) && e.shadow.as_ref().is_some_and(written))
    {
        return true;
    }
    LOOSE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .any(|(&(start, size), s)| overlaps(start, start + size) && written(s))
}

/// Whether a title buffer holds `[address, address + len)` (its memory is
/// the title's, without asking the kernel).
pub(crate) fn in_buffer(address: u64, len: u64) -> bool {
    let Some(end) = address.checked_add(len.max(1)) else {
        return false;
    };
    buffers()
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .any(|(_, e)| e.end >= end)
}

/// Bytes from `address` to the end of the innermost title buffer holding
/// it, without making a copy.
pub(crate) fn buffer_left(address: u64) -> Option<u64> {
    buffers().range(..=(address, usize::MAX)).rev().take(64).find(|(_, e)| e.end > address).map(|(_, e)| e.end - address)
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if let Some(s) = &entry.shadow {
        return Some(s.clone());
    }
    let len = entry.end - key.0;
    // Made outside the buffers' lock (a title thread held it
    // while CreateCommittedResource sat in the kernel for tens of seconds,
    // and every draw thread queued behind it). Another thread may make the
    // same copy meanwhile; the first published wins.
    drop(map);
    // Rounded up so that views ending in the buffer's last 256 bytes fit.
    // Shaders write some buffers (unordered access), so not upload memory.
    let shadow = make_copy(key.0, len, len.next_multiple_of(1 << 16), "title buffer")?;
    let mut map = buffers();
    let entry = map.get_mut(&key)?;
    if let Some(s) = &entry.shadow {
        return Some(s.clone());
    }
    entry.shadow = Some(shadow.clone());
    drop(map);
    super::roles::register(&shadow);
    // The address map rebuilds when a copy appears.
    changed();
    if len >= 16 << 20 {
        xnw_core::info!("d3d12x: large buffer {:#x}+{len:#x} copied to the host", key.0);
    }
    Some(shadow)
}

/// A new host copy of title memory `[start, start + len)` in a buffer of
/// `bytes`, made with no lock of ours held; `None` in a null-GPU frame
/// Nothing will read it) or when the host refuses.
fn make_copy(start: u64, len: u64, bytes: u64, what: &str) -> Option<Arc<Shadow>> {
    if super::uikeep::refuse() {
        NULL_REFUSED.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let (buffer, data) = host::host()?.shared_buffer(bytes)?;
    host::name(&buffer, || format!("{what} {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    Some(Arc::new(Shadow::new(buffer, data, gpu, start, len)))
}

/// Host copies not made because the GPU runs nothing (null mode).
pub(crate) static NULL_REFUSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

/// The runtime remapped or unmapped `[at, at + len)`: copies of windows
/// over it are let go (lists that hold one keep it). VIctory
/// (360f382) `evict_loose`.
pub(crate) fn evict_loose(at: usize, len: usize) {
    let (at, len) = (at as u64, len as u64);
    let mut dropped = Vec::new();
    {
        let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
        map.retain(|&(start, size), s| {
            let keep = !(start < at.saturating_add(len) && at < start.saturating_add(size));
            if !keep {
                dropped.push(s.clone());
            }
            keep
        });
    }
    if !dropped.is_empty() {
        changed();
    }
    drop(dropped);
}

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    // VIctory (360f382): at 128 copies, those no list holds go,
    // down to 96.
    const MAX_LOOSE: usize = 128;
    const TARGET_LOOSE: usize = 96;
    if map.len() >= MAX_LOOSE {
        let keys: Vec<(u64, u64)> = map.iter().filter(|(_, s)| Arc::strong_count(s) <= 1).map(|(&k, _)| k).collect();
        let n = map.len().saturating_sub(TARGET_LOOSE);
        for k in keys.into_iter().take(n) {
            map.remove(&k);
        }
        changed();
    }
    // Made outside LOOSE's lock (gpu_written_copy takes it per view).
    drop(map);
    let shadow = make_copy(start, len, len, "title memory")?;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = map.get(&(start, len)) {
        return Some(s.clone());
    }
    map.insert((start, len), shadow.clone());
    drop(map);
    super::roles::register(&shadow);
    changed();
    Some(shadow)
}

/// The copy a buffer view of `len` bytes at `address`, whose elements are
/// `element` bytes, reads: the innermost title buffer's ([`find`]), unless
/// the view does not start on an element of that copy. A host view counts
/// its first element from its buffer's start; a console view starts at any
/// address, and the title places structured views at ring positions inside
/// its big GPU heap (the meshlet culling's constants: refused, their slots
/// read zeros, every meshlet culled). Such a view gets a copy of its own
/// bytes, starting at its address.
///
/// Ported from VIctory (main 86a5ed6, `crates/victory-gpu/src/shadow.rs`
/// `for_view`, `exact`, GPL-3.0-or-later); the counters are always on here.
pub(crate) fn for_view(address: u64, len: u64, element: u64) -> Option<Arc<Shadow>> {
    let shadow = find(address, len)?;
    if element == 0 || shadow.offset(address).is_multiple_of(element) {
        return Some(shadow);
    }
    OFF_ELEMENT[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    exact(address, len)
}

/// Views off their copy's elements: asked for, copies made (the
/// bindless line).
pub(crate) static OFF_ELEMENT: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Copies of single views' bytes ([`for_view`]), by (start, length), with
/// the use that last asked for each.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(u64, BTreeMap<(u64, u64), (Arc<Shadow>, u64)>)> = Mutex::new((0, BTreeMap::new()));

/// Copies [`EXACT`] keeps: ring positions change every frame, so the least
/// recently asked for go (lists and heap slots using one keep it alive).
const EXACT_KEPT: usize = 1024;

fn exact(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (tick, map) = &mut *guard;
    *tick += 1;
    if let Some((shadow, used)) = map.get_mut(&(address, len)) {
        *used = *tick;
        return Some(shadow.clone());
    }
    if map.len() >= EXACT_KEPT
        && let Some(&oldest) = map.iter().min_by_key(|(_, (_, used))| *used).map(|(k, _)| k)
    {
        map.remove(&oldest);
    }
    // Made outside EXACT's lock.
    let now = *tick;
    drop(guard);
    let shadow = make_copy(address, len, len.max(1).next_multiple_of(256), "title view")?;
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (_, map) = &mut *guard;
    if let Some((s, _)) = map.get(&(address, len)) {
        return Some(s.clone());
    }
    map.insert((address, len), (shadow.clone(), now));
    OFF_ELEMENT[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
#[allow(dead_code)] // the DRED report (base host.rs 717), dropped with DRED
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

/// VIctory's world-frame copy benchmark over our copies.
#[cfg(test)]
mod world_bench;

#[cfg(test)]
mod tests {
    use super::*;
    use xnw_core::win;

    /// A pagefile view of one page: title memory that can be trapped. Kept
    /// for the run (the write-watch journal keeps its pages).
    fn title_page() -> usize {
        // SAFETY: plain section creation and mapping.
        let view = unsafe {
            let section = win::CreateFileMappingW(win::INVALID_HANDLE_VALUE, std::ptr::null(), win::PAGE_READWRITE, 0, PAGE as u32, std::ptr::null());
            assert!(!section.is_null());
            let v = win::MapViewOfFile(section, win::FILE_MAP_WRITE, 0, 0, PAGE) as usize;
            win::CloseHandle(section);
            v
        };
        assert_ne!(view, 0);
        view
    }

    /// Lock mode: a GPU write to a copy stays there, untrapped;
    /// a refresh does not overwrite it with older title bytes; the title's
    /// lock of the range puts it into title memory; a lock elsewhere does
    /// nothing.
    #[test]
    fn lock_mode_writes_back_what_the_title_locks() {
        use super::super::roles::{Mode, TEST_MODE};
        if host::host().is_none() {
            return;
        }
        TEST_MODE.with(|t| t.set(Some(Mode::Lock)));
        let view = title_page();
        // SAFETY: our view.
        unsafe { (view as *mut u8).write_bytes(1, 512) };
        let shadow = find(view as u64, 512).expect("a window copy");
        shadow.update(view, 512);
        shadow.writing(view, 256);
        let at = shadow.offset(view as u64) as usize;
        // SAFETY: inside the copy's mapping: the GPU's write.
        unsafe { shadow.data.add(at).write_bytes(7, 256) };
        shadow.copy_back(view, 256);
        // SAFETY: our view, never trapped in this mode.
        assert_eq!(unsafe { *(view as *const u8) }, 1, "not written back at submission");
        // A refresh of the whole buffer leaves the GPU's bytes in the copy.
        shadow.update(view, 512);
        // SAFETY: inside the copy's mapping.
        assert_eq!(unsafe { *shadow.data.add(at) }, 7);
        assert_eq!(buffer_locked(view + 0x1000, 64, 0), 0, "a lock elsewhere");
        assert_eq!(buffer_locked(view, 128, 0), 128, "the locked part of the declared range");
        assert_eq!(buffer_locked(view, 512, 0), 128, "the rest, once");
        // SAFETY: our view.
        let title = unsafe { std::slice::from_raw_parts(view as *const u8, 512) };
        assert!(title[..256].iter().all(|&b| b == 7) && title[256..].iter().all(|&b| b == 1));
        TEST_MODE.with(|t| t.set(None));
    }

    /// A test run's hang: in lock mode a page the GPU wrote is not refreshed
    /// over until the title locks it, unless the CPU writes it afterwards:
    /// then the CPU's bytes reach the copy.
    #[test]
    fn lock_mode_cpu_writes_after_a_gpu_write_win() {
        use super::super::roles::{Mode, TEST_MODE};
        if host::host().is_none() {
            return;
        }
        TEST_MODE.with(|t| t.set(Some(Mode::Lock)));
        let view = title_page();
        // SAFETY: our view.
        unsafe { (view as *mut u8).write_bytes(1, 256) };
        let shadow = find(view as u64, 256).expect("a window copy");
        shadow.update(view, 256);
        shadow.writing(view, 256);
        let at = shadow.offset(view as u64) as usize;
        // SAFETY: inside the copy's mapping: the GPU's write.
        unsafe { shadow.data.add(at).write_bytes(7, 256) };
        shadow.copy_back(view, 256);
        shadow.update(view, 256);
        // SAFETY: inside the copy's mapping.
        assert_eq!(unsafe { *shadow.data.add(at) }, 7, "unwritten by the CPU: the GPU's bytes stay");
        // The title fills the memory again on the CPU (a guard opens).
        // SAFETY: our view.
        unsafe { (view as *mut u8).write_bytes(9, 256) };
        shadow.update(view, 256);
        // SAFETY: inside the copy's mapping.
        assert_eq!(unsafe { *shadow.data.add(at) }, 9, "the CPU's later write wins");
        assert_eq!(buffer_locked(view, 256, 0), 0, "nothing owed any more");
        TEST_MODE.with(|t| t.set(None));
    }

    /// A test run's title crash (its own records overwritten by a whole
    /// declared range put back): with roles on, a range declared for the GPU
    /// is trapped at once; a CPU write while the submission is in flight
    /// keeps a baseline, and after the GPU each declared byte is the CPU's if
    /// it changed it, else the GPU's.
    #[test]
    fn cpu_writes_during_a_gpu_write_survive_the_merge() {
        if host::host().is_none() {
            return;
        }
        super::super::roles::TEST_MODE.with(|t| t.set(Some(super::super::roles::Mode::On)));
        let view = title_page();
        // SAFETY: our view.
        unsafe { (view as *mut u8).write_bytes(1, 256) };
        let shadow = find(view as u64, 256).expect("a window copy");
        shadow.update(view, 256);
        shadow.writing(view, 256);
        // The title writes a record in the range while the GPU runs: the
        // trap keeps its bytes from before.
        // SAFETY: our view; the handler lets the write through.
        unsafe { std::ptr::write_volatile((view + 10) as *mut u8, 0xAA) };
        // The GPU writes the whole range in the copy.
        let at = shadow.offset(view as u64) as usize;
        // SAFETY: inside the copy's mapping.
        unsafe { shadow.data.add(at).write_bytes(7, 256) };
        shadow.copy_back(view, 256);
        // SAFETY: our view.
        let title = unsafe { std::slice::from_raw_parts(view as *const u8, 256) };
        assert_eq!(title[10], 0xAA, "the CPU's byte stays");
        assert!(title.iter().enumerate().all(|(i, &b)| i == 10 || b == 7), "the GPU's bytes elsewhere");
        super::super::roles::TEST_MODE.with(|t| t.set(None));
    }

    /// A test run's GPU hang: with roles on, a GPU write to a host copy is
    /// owed (trapped, not copied back); a later in-place read of the same
    /// memory must see the GPU's bytes. Here: the range is kept on its copy
    /// for other views (one copy), and the settle an in-place binding does
    /// (`watch::reading`) puts the owed bytes into title memory first.
    #[test]
    fn owed_gpu_writes_reach_title_memory_before_an_in_place_read() {
        if host::host().is_none() {
            return;
        }
        super::super::roles::TEST_MODE.with(|t| t.set(Some(super::super::roles::Mode::On)));
        // SAFETY: plain section creation and mapping: title memory that can
        // be trapped.
        let view = unsafe {
            let section = win::CreateFileMappingW(win::INVALID_HANDLE_VALUE, std::ptr::null(), win::PAGE_READWRITE, 0, PAGE as u32, std::ptr::null());
            assert!(!section.is_null());
            let v = win::MapViewOfFile(section, win::FILE_MAP_WRITE, 0, 0, PAGE) as usize;
            win::CloseHandle(section);
            v
        };
        assert_ne!(view, 0);
        // SAFETY: our view.
        unsafe { (view as *mut u8).write_bytes(1, 256) };
        let shadow = find(view as u64, 256).expect("a window copy");
        shadow.update(view, 256);
        // A submission's shaders write the range through the copy.
        shadow.writing(view, 256);
        let at = shadow.offset(view as u64) as usize;
        // SAFETY: inside the copy's mapping.
        unsafe { shadow.data.add(at).write_bytes(7, 256) };
        shadow.copy_back(view, 256);
        assert!(gpu_written_copy(view as u64 + 16, 16), "other views of the range read the copy");
        assert!(!gpu_written_copy(view as u64 + 2 * PAGE as u64, 16), "not other pages");
        // What an in-place binding of the range does before the GPU reads it.
        super::super::watch::reading(view + 16, 16);
        // SAFETY: our view, accessible again.
        assert_eq!(unsafe { std::slice::from_raw_parts(view as *const u8, 256) }, &[7u8; 256][..]);
        super::super::roles::TEST_MODE.with(|t| t.set(None));
        // The view stays mapped for the run: the write-watch journal keeps
        // its pages, and a later test given the address would find them
        // written.
    }
}

/// `XNW_COPYBACK=off`: what the GPU writes into a host copy is never taken
/// back to title memory and no snapshot is kept for it (an experiment: the
/// title reads GPU results one or two frames late through its own lock
/// call and tolerates stale data; another runtime runs
/// this way too). The price is that every such readback sees old bytes.
fn no_copy_back() -> bool {
    static OFF: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_COPYBACK").is_ok_and(|v| v == "off"));
    *OFF
}
